# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Qwen3.6 GatedDeltaNet linear attention layer.

Implements linear attention via cumsum-based recurrent state:
  kv = einsum('thd,the->thde', k, v)       # outer product
  state_update = beta * kv                   # gate the update
  trajectory = cumsum(state_update, dim=0)   # cumulative state
  y = einsum('thd,thde->the', q, trajectory) # query the state

Also includes partial RoPE and output gate, same as full attention.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, einsum
from megatron.core import ModelParallelConfig, tensor_parallel
from megatron.core import parallel_state as mpu
from torch import nn
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa: F401

from verl.utils.megatron import tensor_parallel as tp_utils
from .parallel_attention import Qwen3_6RotaryEmbedding, apply_rotary_pos_emb_rmpad_flash


class ParallelQwen3_6GatedDeltaNet(nn.Module):
    """GatedDeltaNet linear attention for Qwen3.6.

    Uses separate q, k, v projections (not fused QKV), plus beta and decay gates.
    The core computation uses cumsum for online linear attention.
    """

    def __init__(self, config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Use explicit head_dim from config if available (Qwen3.6 uses head_dim=256)
        self.head_dim = getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        # DeltaNet doesn't use standard GQA — it has separate k/v head counts
        # The relationship between q (24,256) and k (16,128)/v (48,128) is handled
        # through the cumsum computation, not GQA-style head repetition
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)

        # Partial RoPE
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.rope_dim = int(self.head_dim * self.partial_rotary_factor)

        # Output gate
        self.attn_output_gate = getattr(config, "attn_output_gate", False)

        # Linear attention-specific dimensions (different from full attention)
        self.linear_num_key_heads = getattr(config, "linear_num_key_heads", self.num_key_value_heads)
        self.linear_key_head_dim = getattr(config, "linear_key_head_dim", self.head_dim)
        self.linear_num_value_heads = getattr(config, "linear_num_value_heads", self.num_key_value_heads)
        self.linear_value_head_dim = getattr(config, "linear_value_head_dim", self.head_dim)

        tp_size = mpu.get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        assert self.linear_num_key_heads % tp_size == 0
        assert self.linear_num_value_heads % tp_size == 0

        self.num_heads_per_tp = self.num_heads // tp_size
        self.linear_num_key_heads_per_tp = self.linear_num_key_heads // tp_size
        self.linear_num_value_heads_per_tp = self.linear_num_value_heads // tp_size
        self.q_output_size_per_tp = self.num_heads_per_tp * self.head_dim

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        # Separate projections for q, k, v (not fused QKV like standard attention)
        # Note: k/v use linear attention dimensions (different from full attention)
        self.q_proj = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )
        self.k_proj = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.linear_num_key_heads * self.linear_key_head_dim,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )
        self.v_proj = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.linear_num_value_heads * self.linear_value_head_dim,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        # Beta gate: controls how much new information to write
        self.beta_proj = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_heads,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        # Decay gate: controls state decay rate
        self.decay_proj = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_heads,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        # Output projection
        self.o_proj = tensor_parallel.RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            **row_kwargs,
        )

        # Output gate projection (must gather_output to match o_proj full output)
        if self.attn_output_gate:
            self.gate_proj = tensor_parallel.ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.hidden_size,
                bias=False,
                gather_output=True,
                skip_bias_add=False,
                **column_kwargs,
            )

        # RoPE embedding (same as full attention, applied to rope_dim)
        self.rotary_emb = Qwen3_6RotaryEmbedding(
            self.rope_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _apply_partial_rope_rmpad(self, q, k, cos, sin, cu_seqlens, max_seqlen):
        """Apply partial RoPE in RmPad mode."""
        cos_half, sin_half = cos[:, :cos.shape[1] // 2], sin[:, :sin.shape[1] // 2]

        if self.rope_dim == self.head_dim:
            q, k = apply_rotary_pos_emb_rmpad_flash(q, k, cos_half, sin_half, cu_seqlens, max_seqlen)
        else:
            q_rot, q_pass = q[:, :, :self.rope_dim], q[:, :, self.rope_dim:]
            k_rot, k_pass = k[:, :, :self.rope_dim], k[:, :, self.rope_dim:]
            q_rot, k_rot = apply_rotary_pos_emb_rmpad_flash(q_rot, k_rot, cos_half, sin_half, cu_seqlens, max_seqlen)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Standard (padded) forward — for non-RmPad path."""
        bsz, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)[0].view(bsz, seq_len, self.num_heads_per_tp, self.head_dim)
        k = self.k_proj(hidden_states)[0].view(bsz, seq_len, self.linear_num_key_heads_per_tp, self.linear_key_head_dim)
        v = self.v_proj(hidden_states)[0].view(bsz, seq_len, self.linear_num_value_heads_per_tp, self.linear_value_head_dim)

        beta = torch.sigmoid(self.beta_proj(hidden_states)[0]).unsqueeze(-1).unsqueeze(-1)  # (bsz, seq, heads_tp, 1, 1)
        decay = torch.sigmoid(self.decay_proj(hidden_states)[0]).unsqueeze(-1).unsqueeze(-1)

        # Apply partial RoPE
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        # Standard apply_rotary_pos_emb needs [bsz, heads, seq, dim]
        q = q.transpose(1, 2)  # (bsz, heads_tp, seq, head_dim)
        k = k.transpose(1, 2)
        if self.rope_dim == self.head_dim:
            from .parallel_attention import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        else:
            from .parallel_attention import rotate_half
            cos_emb = cos[position_ids].unsqueeze(1)
            sin_emb = sin[position_ids].unsqueeze(1)
            q_rot, q_pass = q[..., :self.rope_dim], q[..., self.rope_dim:]
            k_rot, k_pass = k[..., :self.rope_dim], k[..., self.rope_dim:]
            q_rot = (q_rot * cos_emb) + (rotate_half(q_rot) * sin_emb)
            k_rot = (k_rot * cos_emb) + (rotate_half(k_rot) * sin_emb)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        q = q.transpose(1, 2)  # back to (bsz, seq, heads, dim)
        k = k.transpose(1, 2)

        # Linear attention via cumsum
        # k: (bsz, seq, linear_key_heads, key_dim), v: (bsz, seq, linear_value_heads, value_dim)
        # k: (bsz, seq, heads, dim), v: (bsz, seq, heads, dim)
        # kv outer product: (bsz, seq, heads, dim, dim)
        kv = torch.einsum('bshd,bthe->bshde', k, v)
        update = beta * kv

        # Cumsum over sequence dimension
        trajectory = torch.cumsum(update, dim=1)

        # Query the state
        y = torch.einsum('bshd,bshde->bshe', q, trajectory)

        # Reshape and output projection
        y = y.reshape(bsz, seq_len, self.q_output_size_per_tp)
        output = self.o_proj(y)[0]

        if self.attn_output_gate:
            gate = torch.sigmoid(self.gate_proj(hidden_states)[0])
            output = output * gate

        return output


class ParallelQwen3_6GatedDeltaNetRmPad(ParallelQwen3_6GatedDeltaNet):
    """GatedDeltaNet with remove-padding for flash attention compatibility.

    TODO: The einsum computation for mismatched k/v head counts needs
    proper implementation based on Qwen3.5 reference. For now, using
    a simplified pass-through to test full attention layers.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids=None,
        sequence_length=None,
        indices=None,
        cu_seqlens=None,
        max_seqlen_in_batch=None,
    ):
        # Simplified forward: pass through with o_proj matching dimensions
        # Reshape hidden_states from (N, 1, hidden_size) to (N, 1, q_output_size_per_tp)
        # by padding/expanding, then apply o_proj
        # For now, just return hidden_states unchanged (identity)
        return hidden_states


def _packed_cumsum(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Per-sequence cumsum for packed (THD) format.

    x: (total_nnz, heads, dim, dim2)
    cu_seqlens: (batch_size + 1,)

    Returns cumsum with per-sequence accumulation (resets at boundaries).
    """
    if cu_seqlens is None or len(cu_seqlens) <= 1:
        return torch.cumsum(x, dim=0)

    # Use segment cumsum: full cumsum then subtract sequence starts
    full_cumsum = torch.cumsum(x, dim=0)

    # Build index that maps each token to its sequence start offset
    # For sequence i (starting at cu_seqlens[i]), the offset is cumsum up to cu_seqlens[i]-1
    offsets = []
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i].item()
        if start > 0:
            offsets.append(full_cumsum[start - 1:start].clone())
        else:
            offsets.append(torch.zeros_like(x[:1]))
    if not offsets:
        return full_cumsum

    # Subtract per-sequence offset from full cumsum
    # Build a "start_value" tensor aligned to each token
    start_values = torch.zeros_like(x)
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        if start > 0:
            start_values[start:end] = full_cumsum[start - 1:start]

    return full_cumsum - start_values
