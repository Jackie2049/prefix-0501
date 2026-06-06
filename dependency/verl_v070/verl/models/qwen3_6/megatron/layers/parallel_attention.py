# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.6 full attention layer with QK normalization, partial RoPE, and fused output gate.

Key differences from our initial design and Qwen2:
- q_proj output = num_heads * head_dim * 2 (fused query + output gate via chunk)
- q_norm and k_norm (per-head RMSNorm before attention, matching Qwen3_5 reference)
- RoPE applied AFTER q_norm/k_norm (not before)
- Partial RoPE: only apply RoPE to the first rope_dim (64/256) dimensions
- Output gate: attn_output * sigmoid(gate) where gate = chunk(q_proj, 2)[1]
  (NOT separate gate_proj(hidden_states) * sigmoid)
- No bias on projections (matching reference, config.attention_bias=False)
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa: F401

import torch
from flash_attn.layers.rotary import apply_rotary_emb
from megatron.core import ModelParallelConfig, tensor_parallel
from megatron.core import parallel_state as mpu
from torch import nn

from verl.utils.megatron import tensor_parallel as tp_utils


class Qwen3_6RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb_rmpad_flash(q, k, cos, sin, cu_seqlens, max_seqlen):
    q_embed = apply_rotary_emb(
        q, cos, sin, interleaved=False, inplace=False, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )
    k_embed = apply_rotary_emb(
        k, cos, sin, interleaved=False, inplace=False, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )
    return q_embed, k_embed


class RMSNormPerHead(nn.Module):
    """Per-head RMSNorm for Qwen3.5's QK normalization.

    Unlike the full RMSNorm (which operates on hidden_size),
    this operates on head_dim for each attention head independently.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        input_dtype = x.dtype
        output = self._norm(x.float()).to(input_dtype)
        return output * self.weight


class ParallelQwen3_6Attention(nn.Module):
    """Qwen3.6 full attention matching official Qwen3_5 reference.

    Key architectural changes:
    - q_proj outputs num_heads * head_dim * 2 (fused query + gate)
    - After chunk, query_states and gate are separated
    - q_norm and k_norm applied per-head BEFORE RoPE and attention
    - Output gate: attn_output * sigmoid(gate) from q_proj chunk
    - No bias on projections (config.attention_bias=False)
    """

    def __init__(self, config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.scaling = self.head_dim ** -0.5

        # Partial RoPE dimension
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.rope_dim = int(self.head_dim * self.partial_rotary_factor)

        # Output gate (from chunk of q_proj, not separate gate_proj)
        self.attn_output_gate = getattr(config, "attn_output_gate", False)

        # Has QK normalization (Qwen3.5 uses per-head RMSNorm)
        self.use_qk_norm = True  # Qwen3.5 always has q_norm and k_norm

        tp_size = mpu.get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        assert self.num_key_value_heads % tp_size == 0

        self.num_heads_per_tp = self.num_heads // tp_size
        self.num_key_value_heads_per_tp = self.num_key_value_heads // tp_size

        # q_proj outputs num_heads * head_dim * 2 (fused query + gate)
        # Per TP shard: num_heads_per_tp * head_dim * 2
        self.q_output_size_per_tp = self.num_heads_per_tp * self.head_dim  # without gate
        self.q_gate_output_size_per_tp = self.num_heads_per_tp * self.head_dim  # gate portion

        config_head_dim = getattr(config, "head_dim", None)
        if config_head_dim is None and (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and "
                f"`num_heads`: {self.num_heads})."
            )

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        # Separate q/k/v projections (no bias, matching reference)
        # q_proj: hidden_size → num_heads * head_dim * 2 (fused query + gate)
        self.q_proj = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim * 2,  # 12288 (6144 query + 6144 gate)
            bias=False,  # No bias (config.attention_bias=False)
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )
        self.k_proj = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_key_value_heads * self.head_dim,  # 1024 (GQA: 4 KV heads)
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )
        self.v_proj = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_key_value_heads * self.head_dim,  # 1024
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        # Per-head RMSNorm for query and key (QK normalization)
        self.q_norm = RMSNormPerHead(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormPerHead(self.head_dim, eps=config.rms_norm_eps)

        # Output projection
        self.o_proj = tensor_parallel.RowParallelLinear(
            input_size=self.num_heads * self.head_dim,  # 6144
            output_size=self.hidden_size,  # 5120
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            **row_kwargs,
        )

        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = Qwen3_6RotaryEmbedding(
            self.rope_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _apply_partial_rope(self, query_states, key_states, cos, sin, position_ids):
        """Apply partial RoPE: only rotate the first rope_dim dims."""
        if self.rope_dim == self.head_dim:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            q_rot = query_states[..., :self.rope_dim]
            q_pass = query_states[..., self.rope_dim:]
            k_rot = key_states[..., :self.rope_dim]
            k_pass = key_states[..., self.rope_dim:]

            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids)
            query_states = torch.cat([q_rot, q_pass], dim=-1)
            key_states = torch.cat([k_rot, k_pass], dim=-1)
        return query_states, key_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # q_proj: hidden → num_heads * head_dim * 2 (per shard: num_heads_per_tp * head_dim * 2)
        q_full = self.q_proj(hidden_states)[0]  # (bsz, q_len, num_heads_per_tp * head_dim * 2)

        # Chunk into query and gate (matching reference)
        # q_full: (bsz, q_len, num_heads_per_tp * head_dim * 2)
        # After reshape to per-head with head_dim*2, chunk on last dim
        hidden_shape = (bsz, q_len, self.num_heads_per_tp, self.head_dim * 2)
        query_states, gate = torch.chunk(
            q_full.view(*hidden_shape), 2, dim=-1
        )
        # query_states: (bsz, q_len, num_heads_per_tp, head_dim)
        # gate: (bsz, q_len, num_heads_per_tp, head_dim)

        # Reshape gate for later use (flatten heads)
        gate = gate.reshape(bsz, q_len, self.num_heads_per_tp * self.head_dim)

        # k_proj and v_proj (per shard)
        key_states = self.k_proj(hidden_states)[0].view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim)
        value_states = self.v_proj(hidden_states)[0].view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim)

        # QK normalization (per-head RMSNorm BEFORE RoPE)
        query_states = self.q_norm(query_states)  # (bsz, q_len, heads_per_tp, head_dim)
        key_states = self.k_norm(key_states)

        # Transpose for RoPE and attention: (bsz, heads, seq, dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply partial RoPE AFTER q_norm/k_norm
        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self._apply_partial_rope(query_states, key_states, cos, sin, position_ids)

        # GQA: repeat KV heads to match query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Standard attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape attention output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.q_output_size_per_tp)

        # Apply output gate BEFORE o_proj (matching reference)
        # gate: (bsz, q_len, num_heads_per_tp * head_dim) — per-shard, matches attn_output shape
        if self.attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate)

        # Output projection (RowParallelLinear, AllReduce across TP)
        output = self.o_proj(attn_output)[0]

        return output


class ParallelQwen3_6AttentionRmPad(ParallelQwen3_6Attention):
    """Qwen3.6 full attention with remove-padding (flash_attn_varlen_func)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen_in_batch: int = None,
    ):
        total_nnz, _, _ = hidden_states.size()

        if self.megatron_config.sequence_parallel:
            total_nnz = total_nnz * mpu.get_tensor_model_parallel_world_size()

        # q_proj: hidden → num_heads * head_dim * 2 (per shard)
        q_full = self.q_proj(hidden_states)[0]

        # Chunk into query and gate
        hidden_shape = (total_nnz, self.num_heads_per_tp, self.head_dim * 2)
        query_states, gate = torch.chunk(
            q_full.view(*hidden_shape), 2, dim=-1
        )
        # gate: (total_nnz, num_heads_per_tp, head_dim) → flatten for later
        gate = gate.reshape(total_nnz, self.num_heads_per_tp * self.head_dim)

        key_states = self.k_proj(hidden_states)[0]
        value_states = self.v_proj(hidden_states)[0]

        if self.megatron_config.sequence_parallel:
            sequence_parallel_pad = total_nnz - cu_seqlens[-1]
            total_nnz = cu_seqlens[-1]
            query_states = query_states[:total_nnz]
            key_states = key_states[:total_nnz]
            value_states = value_states[:total_nnz]
            gate = gate[:total_nnz]

        query_states = query_states.view(total_nnz, self.num_heads_per_tp, self.head_dim)
        key_states = key_states.view(total_nnz, self.num_key_value_heads_per_tp, self.head_dim)
        value_states = value_states.view(total_nnz, self.num_key_value_heads_per_tp, self.head_dim)

        # QK normalization (per-head RMSNorm)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Partial RoPE using flash_attn's apply_rotary_emb (AFTER q_norm/k_norm)
        cos, sin = self.rotary_emb(value_states, seq_len=sequence_length)
        cos, sin = cos[:, : cos.shape[1] // 2], sin[:, : sin.shape[1] // 2]

        if self.rope_dim == self.head_dim:
            query_states, key_states = apply_rotary_pos_emb_rmpad_flash(
                query_states, key_states, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch
            )
        else:
            q_rot = query_states[:, :, :self.rope_dim]
            q_pass = query_states[:, :, self.rope_dim:]
            k_rot = key_states[:, :, :self.rope_dim]
            k_pass = key_states[:, :, self.rope_dim:]

            q_rot, k_rot = apply_rotary_pos_emb_rmpad_flash(
                q_rot, k_rot, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch
            )
            query_states = torch.cat([q_rot, q_pass], dim=-1)
            key_states = torch.cat([k_rot, k_pass], dim=-1)

        dropout_rate = 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen_in_batch,
            max_seqlen_k=max_seqlen_in_batch,
            dropout_p=dropout_rate,
            softmax_scale=self.scaling,
            causal=True,
        )

        attn_output_unpad = attn_output_unpad.to(input_dtype)
        attn_output_unpad = attn_output_unpad.reshape(total_nnz, self.num_heads_per_tp * self.head_dim)

        # Apply output gate BEFORE o_proj
        if self.attn_output_gate:
            attn_output_unpad = attn_output_unpad * torch.sigmoid(gate)

        # Reshape for o_proj
        attn_output_unpad = attn_output_unpad.reshape(total_nnz, 1, self.q_output_size_per_tp).contiguous()

        if self.megatron_config.sequence_parallel:
            attn_output_unpad = F.pad(attn_output_unpad, pad=(0, 0, 0, 0, 0, sequence_parallel_pad))

        attn_output_unpad = self.o_proj(attn_output_unpad)[0]

        return attn_output_unpad