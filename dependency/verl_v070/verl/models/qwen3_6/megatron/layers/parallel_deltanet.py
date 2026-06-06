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

Implements the gated delta rule for linear attention, matching the official
Qwen3_5GatedDeltaNet reference implementation from HuggingFace transformers.

Key architectural differences from our initial cumsum-based design:
- Uses causal conv1d (depthwise) for position encoding, NOT RoPE
- Query uses head_k_dim (128), NOT head_dim (256), then GQA-expanded to v_heads
- Decay uses A_log parameter + dt_bias + in_proj_a, NOT simple sigmoid(decay_proj)
- Beta uses in_proj_b (per v_head), NOT per attention_head
- Output uses RMSNormGated(core_attn_out, z) with SiLU gate, NOT sigmoid(gate_proj)
- out_proj maps value_dim → hidden_size, NOT num_heads*head_dim → hidden_size
- All projections have NO bias (matching reference)
"""

from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core import ModelParallelConfig, tensor_parallel
from megatron.core import parallel_state as mpu
from torch import nn
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa: F401

from verl.utils.megatron import tensor_parallel as tp_utils


# ============================================================================
# Helper functions (adapted from HuggingFace Qwen3_5 reference)
# ============================================================================


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization, matching flash-linear-attention's l2norm."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure PyTorch fallback for chunk_gated_delta_rule.

    Adapted from HuggingFace Qwen3_5 reference torch_chunk_gated_delta_rule.
    Operates on (batch, heads, seq, dim) format.

    Args:
        query: (batch, num_v_heads, seq, head_k_dim)
        key:   (batch, num_v_heads, seq, head_k_dim)
        value: (batch, num_v_heads, seq, head_v_dim)
        g:     (batch, num_v_heads, seq) — decay factor (log space)
        beta:  (batch, num_v_heads, seq) — beta gate

    Returns:
        core_attn_out: (batch, seq, num_v_heads, head_v_dim)
        last_recurrent_state: (batch, num_v_heads, head_k_dim, head_v_dim) or None
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # Transpose to (batch, heads, seq, dim) and cast to float32 for precision
    query, key, value, beta, g = [
        x.contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    # Pad sequence length to multiple of chunk_size
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)  # (batch, heads, total_seq, v_dim)
    k_beta = key * beta.unsqueeze(-1)    # (batch, heads, total_seq, k_dim)

    # Reshape to chunks
    query = query.reshape(batch_size, num_heads, -1, chunk_size, k_head_dim)
    key = key.reshape(batch_size, num_heads, -1, chunk_size, k_head_dim)
    value = value.reshape(batch_size, num_heads, -1, chunk_size, v_head_dim)
    k_beta = k_beta.reshape(batch_size, num_heads, -1, chunk_size, k_head_dim)
    v_beta = v_beta.reshape(batch_size, num_heads, -1, chunk_size, v_head_dim)
    g = g.reshape(batch_size, num_heads, -1, chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # Chunk decay
    g_cumsum = g.cumsum(dim=-1)
    decay_mask = ((g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).tril().exp().float()).tril()

    # Intra-chunk computation
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    # Inter-chunk recurrent state
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    mask_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    num_chunks = total_sequence_length // chunk_size
    for i in range(num_chunks):
        q_i = query[:, :, i]      # (batch, heads, chunk, k_dim)
        k_i = key[:, :, i]        # (batch, heads, chunk, k_dim)
        v_i = value[:, :, i]      # (batch, heads, chunk, v_dim)

        # Cross-chunk attention
        attn_cross = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_cross @ v_new

        # Update recurrent state
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # Transpose back to (batch, seq, heads, dim)
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class RMSNormGated(nn.Module):
    """Gated RMSNorm: weight * RMSNorm(x) * silu(gate).

    This is the output gating mechanism used in Qwen3_5GatedDeltaNet.
    Unlike the full attention's gate (sigmoid(gate_proj(x)) * o_proj(attn)),
    DeltaNet uses silu(z) * RMSNorm(core_attn_out) with per-head normalization.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32)).to(input_dtype)
        return hidden_states.to(input_dtype)


# ============================================================================
# Main DeltaNet module (Megatron TP-compatible)
# ============================================================================


class ParallelQwen3_6GatedDeltaNet(nn.Module):
    """GatedDeltaNet linear attention for Qwen3.6, matching official reference.

    Architecture differences from the initial cumsum-based design:
    - No RoPE (uses causal conv1d for position encoding)
    - Fused QKV projection → causal conv1d + SiLU → split into q, k, v
    - q and k use head_k_dim (128), v uses head_v_dim (128)
    - GQA expansion: repeat_interleave q/k to match v_heads (48/16=3x)
    - Decay: g = -exp(A_log) * softplus(a + dt_bias)
    - Beta: sigmoid(in_proj_b), per v_head
    - Output: RMSNormGated(core_attn_out, z) with SiLU gate → out_proj
    """

    def __init__(self, config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.hidden_size = config.hidden_size  # 5120
        self.layer_idx = getattr(config, "_layer_idx", 0)

        # DeltaNet dimensions (different from full attention!)
        self.num_v_heads = getattr(config, "linear_num_value_heads", config.num_key_value_heads)  # 48
        self.num_k_heads = getattr(config, "linear_num_key_heads", config.num_key_value_heads)   # 16
        self.head_k_dim = getattr(config, "linear_key_head_dim", 128)  # 128
        self.head_v_dim = getattr(config, "linear_value_head_dim", 128) # 128
        self.key_dim = self.num_k_heads * self.head_k_dim    # 2048
        self.value_dim = self.num_v_heads * self.head_v_dim  # 6144
        self.conv_kernel_size = getattr(config, "linear_conv_kernel_dim", 4)  # 4

        # GQA expansion factor (v_heads / k_heads)
        self.num_v_per_k = self.num_v_heads // self.num_k_heads  # 3

        # TP sharding
        tp_size = mpu.get_tensor_model_parallel_world_size()
        assert self.num_k_heads % tp_size == 0, \
            f"num_k_heads({self.num_k_heads}) must be divisible by tp_size({tp_size})"
        assert self.num_v_heads % tp_size == 0, \
            f"num_v_heads({self.num_v_heads}) must be divisible by tp_size({tp_size})"

        self.num_k_heads_per_tp = self.num_k_heads // tp_size   # 4
        self.num_v_heads_per_tp = self.num_v_heads // tp_size   # 12
        self.key_dim_per_tp = self.num_k_heads_per_tp * self.head_k_dim     # 512
        self.value_dim_per_tp = self.num_v_heads_per_tp * self.head_v_dim   # 1536
        self.conv_dim_per_tp = self.key_dim_per_tp * 2 + self.value_dim_per_tp  # 2560

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        # Separate Q, K, V projections (all NO bias, matching reference)
        # Note: Q maps to key_dim (not num_heads * head_dim like full attention)
        self.in_proj_q = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.key_dim,  # 2048 (num_k_heads * head_k_dim)
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )
        self.in_proj_k = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.key_dim,  # 2048
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )
        self.in_proj_v = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.value_dim,  # 6144 (num_v_heads * head_v_dim)
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        # Beta gate: per v_head, controls how much new info to write
        self.in_proj_b = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_v_heads,  # 48
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        # Decay/A gate: per v_head, controls state decay rate
        # g = -exp(A_log) * softplus(a + dt_bias)
        self.in_proj_a = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_v_heads,  # 48
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        # Output gate z: per v_head * head_v_dim, for RMSNormGated
        self.in_proj_z = tensor_parallel.ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.value_dim,  # 6144
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        # Causal conv1d (depthwise, per TP shard)
        # Applied on fused QKV after projections, before splitting
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_per_tp,
            out_channels=self.conv_dim_per_tp,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim_per_tp,  # depthwise
            padding=self.conv_kernel_size - 1,  # causal padding
            bias=False,
        )

        # Per-head decay parameters (sharded across TP)
        A = torch.empty(self.num_v_heads_per_tp).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads_per_tp))

        # RMSNormGated: operates on head_v_dim (128), replicated across TP
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        # Output projection: value_dim → hidden_size (RowParallelLinear, AllReduce)
        self.out_proj = tensor_parallel.RowParallelLinear(
            input_size=self.value_dim,  # 6144 (num_v_heads * head_v_dim)
            output_size=self.hidden_size,  # 5120
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            **row_kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Standard (padded) forward for DeltaNet.

        No RoPE is applied — position encoding is handled by causal conv1d.
        """
        bsz, seq_len, _ = hidden_states.size()

        # Save hidden_states for in_proj_z (output gate)
        residual_hidden = hidden_states

        # Separate Q, K, V projections (per TP shard)
        q = self.in_proj_q(hidden_states)[0]  # (bsz, seq, key_dim_per_tp)
        k = self.in_proj_k(hidden_states)[0]  # (bsz, seq, key_dim_per_tp)
        v = self.in_proj_v(hidden_states)[0]  # (bsz, seq, value_dim_per_tp)

        # Concatenate for causal conv1d (depthwise)
        mixed_qkv = torch.cat([q, k, v], dim=-1)  # (bsz, seq, conv_dim_per_tp)
        mixed_qkv = mixed_qkv.transpose(1, 2)      # (bsz, conv_dim_per_tp, seq) for Conv1d

        # Apply causal conv1d + SiLU activation (PyTorch fallback)
        mixed_qkv = self.conv1d(mixed_qkv)
        mixed_qkv = mixed_qkv[:, :, :seq_len]      # trim causal padding
        mixed_qkv = F.silu(mixed_qkv)

        mixed_qkv = mixed_qkv.transpose(1, 2)      # back to (bsz, seq, conv_dim_per_tp)

        # Split back into query, key, value
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim_per_tp, self.key_dim_per_tp, self.value_dim_per_tp],
            dim=-1,
        )

        # Reshape to per-head format
        # Note: query and key use head_k_dim, value uses head_v_dim
        query = query.reshape(bsz, seq_len, self.num_k_heads_per_tp, self.head_k_dim)
        key = key.reshape(bsz, seq_len, self.num_k_heads_per_tp, self.head_k_dim)
        value = value.reshape(bsz, seq_len, self.num_v_heads_per_tp, self.head_v_dim)

        # Compute beta (write gate) and g (decay factor)
        b = self.in_proj_b(residual_hidden)[0]  # (bsz, seq, num_v_heads_per_tp)
        a = self.in_proj_a(residual_hidden)[0]  # (bsz, seq, num_v_heads_per_tp)

        beta = b.sigmoid()  # (bsz, seq, num_v_heads_per_tp)

        # g = -exp(A_log) * softplus(a + dt_bias)
        # Cast to float32 for numerical stability (A_log can produce -inf in fp16)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        g = g.to(hidden_states.dtype)  # (bsz, seq, num_v_heads_per_tp)

        # GQA expansion: expand query and key to match v_heads
        if self.num_v_per_k > 1:
            # (bsz, seq, num_k_heads_per_tp, head_k_dim) → (bsz, seq, num_v_heads_per_tp, head_k_dim)
            query = query.repeat_interleave(self.num_v_per_k, dim=2)
            key = key.repeat_interleave(self.num_v_per_k, dim=2)

        # Compute output gate z (for RMSNormGated)
        z = self.in_proj_z(residual_hidden)[0]  # (bsz, seq, value_dim_per_tp)
        z = z.reshape(bsz, seq_len, self.num_v_heads_per_tp, self.head_v_dim)

        # Transpose to (bsz, num_v_heads_per_tp, seq, dim) for chunk_gated_delta_rule
        query = query.transpose(1, 2)  # (bsz, v_heads_per_tp, seq, head_k_dim)
        key = key.transpose(1, 2)      # (bsz, v_heads_per_tp, seq, head_k_dim)
        value = value.transpose(1, 2)  # (bsz, v_heads_per_tp, seq, head_v_dim)
        beta = beta.transpose(1, 2)    # (bsz, v_heads_per_tp, seq)
        g = g.transpose(1, 2)          # (bsz, v_heads_per_tp, seq)

        # Core computation: chunk_gated_delta_rule (pure PyTorch fallback)
        core_attn_out, _ = torch_chunk_gated_delta_rule(
            query, key, value, g, beta,
            chunk_size=64,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        # core_attn_out: (bsz, seq, num_v_heads_per_tp, head_v_dim)

        # RMSNormGated: normalize + SiLU gate
        core_attn_out = self.norm(core_attn_out, z)

        # Reshape for output projection
        core_attn_out = core_attn_out.reshape(bsz, seq_len, self.value_dim_per_tp)
        output = self.out_proj(core_attn_out)[0]

        return output


class ParallelQwen3_6GatedDeltaNetRmPad(ParallelQwen3_6GatedDeltaNet):
    """GatedDeltaNet with remove-padding for flash attention compatibility.

    For DeltaNet, we need to unpack the packed format to (batch, seq, hidden)
    for the conv1d and core computation, then pack back for downstream layers.
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
        total_nnz, _, _ = hidden_states.size()

        if self.megatron_config.sequence_parallel:
            total_nnz = total_nnz * mpu.get_tensor_model_parallel_world_size()

        # Determine batch size from cu_seqlens
        batch_size = len(cu_seqlens) - 1 if cu_seqlens is not None else 1

        # Unpack: (total_nnz, 1, hidden_size) → (batch_size, seq_len, hidden_size)
        hidden_states_3d = hidden_states.reshape(total_nnz, self.hidden_size)
        hidden_states_3d = pad_input(hidden_states_3d, indices, batch_size, sequence_length)

        # Run the padded forward
        output_3d = ParallelQwen3_6GatedDeltaNet.forward(
            self,
            hidden_states=hidden_states_3d,
            attention_mask=None,
            position_ids=position_ids,
        )
        # output_3d: (batch_size, seq_len, hidden_size)

        # Pack back: (batch_size, seq_len, hidden_size) → (total_nnz, 1, hidden_size)
        output_flat = output_3d.reshape(batch_size * sequence_length, self.hidden_size)
        output_unpad, _ = unpad_input(output_3d, None)  # Use indices to unpad

        # Actually, we need to use indices to unpad properly
        output_flat = output_3d.reshape(batch_size * sequence_length, self.hidden_size)
        # Select only the non-padding tokens using indices
        output_selected = output_flat[indices]

        # Handle SP padding
        if self.megatron_config.sequence_parallel:
            sp_pad = total_nnz - (cu_seqlens[-1] if cu_seqlens is not None else total_nnz)
            output_selected = F.pad(
                output_selected.reshape(-1, 1, self.hidden_size),
                pad=(0, 0, 0, 0, 0, sp_pad),
            )
        else:
            output_selected = output_selected.reshape(-1, 1, self.hidden_size)

        return output_selected