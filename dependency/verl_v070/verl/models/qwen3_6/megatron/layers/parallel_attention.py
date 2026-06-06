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

"""Qwen3.6 full attention layer with partial RoPE and output gate.

Key differences from Qwen2:
- Partial RoPE: only apply RoPE to the first `rope_dim` dimensions (64/256)
- Output gate: output = o_proj(attn_output) * sigmoid(gate_proj(hidden_states))
"""

import math
from typing import Optional

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
from transformers import AutoConfig

from verl.models.qwen3_6.megatron.layers.parallel_linear import QKVParallelLinear
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


class ParallelQwen3_6Attention(nn.Module):
    """Qwen3.6 full attention with partial RoPE and output gate."""

    def __init__(self, config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Use explicit head_dim from config if available (Qwen3.6 uses head_dim=256, not hidden_size/num_heads)
        self.head_dim = getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)

        # Partial RoPE dimension
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.rope_dim = int(self.head_dim * self.partial_rotary_factor)

        # Output gate
        self.attn_output_gate = getattr(config, "attn_output_gate", False)

        tp_size = mpu.get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        assert self.num_key_value_heads % tp_size == 0

        self.num_heads_per_tp = self.num_heads // tp_size
        self.num_key_value_heads_per_tp = self.num_key_value_heads // tp_size
        # With explicit head_dim, Q output size != hidden_size per TP shard
        self.q_output_size_per_tp = self.num_heads_per_tp * self.head_dim

        config_head_dim = getattr(config, "head_dim", None)
        if config_head_dim is None and (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and "
                f"`num_heads`: {self.num_heads})."
            )

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        if megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            assert row_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        self.qkv_proj = QKVParallelLinear(
            input_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        self.q_size = self.num_heads_per_tp * self.head_dim
        self.k_size = self.num_key_value_heads_per_tp * self.head_dim
        self.v_size = self.num_key_value_heads_per_tp * self.head_dim

        self.o_proj = tensor_parallel.RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            **row_kwargs,
        )

        # Output gate projection (ColumnParallelLinear, hidden -> hidden)
        # Must gather_output since o_proj (RowParallelLinear) produces full output
        if self.attn_output_gate:
            self.gate_proj = tensor_parallel.ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.hidden_size,
                bias=False,
                gather_output=True,
                skip_bias_add=False,
                **column_kwargs,
            )

        self._init_rope()

    def _init_rope(self):
        # RoPE only applied to the first rope_dim dimensions
        self.rotary_emb = Qwen3_6RotaryEmbedding(
            self.rope_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _apply_partial_rope(self, query_states, key_states, cos, sin, position_ids):
        """Apply partial RoPE: only rotate the first rope_dim dims, pass through the rest."""
        if self.rope_dim == self.head_dim:
            # Full RoPE (same as Qwen2)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            # Partial RoPE
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
        qkv = self.qkv_proj(hidden_states)[0]
        query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads_per_tp, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self._apply_partial_rope(query_states, key_states, cos, sin, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.q_output_size_per_tp)
        attn_output = self.o_proj(attn_output)[0]

        # Apply output gate
        if self.attn_output_gate:
            gate = torch.sigmoid(self.gate_proj(hidden_states)[0])
            attn_output = attn_output * gate

        return attn_output


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

        # Save hidden_states for gate_proj before QKV projection
        residual_hidden = hidden_states if self.attn_output_gate else None

        qkv = self.qkv_proj(hidden_states)[0]
        query_states, key_states, value_states = qkv.split(
            [self.q_size, self.k_size, self.v_size], dim=-1
        )

        if self.megatron_config.sequence_parallel:
            sequence_parallel_pad = total_nnz - cu_seqlens[-1]
            total_nnz = cu_seqlens[-1]
            query_states = query_states[:total_nnz]
            key_states = key_states[:total_nnz]
            value_states = value_states[:total_nnz]

        query_states = query_states.view(total_nnz, self.num_heads_per_tp, self.head_dim)
        key_states = key_states.view(total_nnz, self.num_key_value_heads_per_tp, self.head_dim)
        value_states = value_states.view(total_nnz, self.num_key_value_heads_per_tp, self.head_dim)

        # Partial RoPE using flash_attn's apply_rotary_emb
        cos, sin = self.rotary_emb(value_states, seq_len=sequence_length)
        cos, sin = cos[:, : cos.shape[1] // 2], sin[:, : sin.shape[1] // 2]

        if self.rope_dim == self.head_dim:
            # Full RoPE
            query_states, key_states = apply_rotary_pos_emb_rmpad_flash(
                query_states, key_states, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch
            )
        else:
            # Partial RoPE: rotate only first rope_dim dims
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
            softmax_scale=None,
            causal=True,
        )

        attn_output_unpad = attn_output_unpad.to(input_dtype)
        attn_output_unpad = attn_output_unpad.reshape(total_nnz, 1, self.q_output_size_per_tp).contiguous()

        if self.megatron_config.sequence_parallel:
            attn_output_unpad = F.pad(attn_output_unpad, pad=(0, 0, 0, 0, 0, sequence_parallel_pad))

        attn_output_unpad = self.o_proj(attn_output_unpad)[0]

        # Apply output gate
        if self.attn_output_gate:
            if self.megatron_config.sequence_parallel:
                gate_hidden = residual_hidden[:total_nnz] if residual_hidden is not None else hidden_states[:total_nnz]
            else:
                gate_hidden = residual_hidden if residual_hidden is not None else hidden_states
            gate_hidden = gate_hidden.reshape(total_nnz, 1, self.hidden_size)
            # gate_proj needs SP padding too
            if self.megatron_config.sequence_parallel and residual_hidden is not None:
                gate_hidden = residual_hidden  # Use original padded hidden for gate_proj
            gate = torch.sigmoid(self.gate_proj(gate_hidden)[0])
            # Trim gate to match attn_output_unpad shape
            gate = gate[:attn_output_unpad.shape[0]]
            attn_output_unpad = attn_output_unpad * gate

        return attn_output_unpad
