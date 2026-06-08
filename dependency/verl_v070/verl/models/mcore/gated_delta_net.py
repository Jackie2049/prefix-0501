# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""GatedDeltaNet linear attention layer for Qwen3.6 HybridAttention.

This module inherits from Megatron's SelfAttention and replaces the
standard attention computation with cumsum-based linear attention
used by GatedDeltaNet.

It keeps the same submodules interface (linear_qkv, linear_proj, etc.)
so it can be dropped into existing TransformerLayer specs.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple, Union

from megatron.core import tensor_parallel
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


class GatedDeltaNetAttention(SelfAttention):
    """GatedDeltaNet linear attention layer.

    Inherits from SelfAttention to maintain compatibility with Megatron's
    TransformerLayer, but replaces the attention computation with cumsum-based
    linear attention.

    Extra parameters compared to SelfAttention:
    - beta_proj: gate for how much new state to write
    - decay_proj: gate for state decay rate
    - gate_proj: output gate (attn_output_gate)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.causal,
        cp_comm_type: str = None,
        partial_rotary_factor: float = 1.0,
        attn_output_gate: bool = False,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
        )

        self.partial_rotary_factor = partial_rotary_factor
        self.attn_output_gate = attn_output_gate

        # Store dimensions
        self.rope_dim = int(self.hidden_size_per_attention_head * partial_rotary_factor)

        tp_size = self.config.tensor_model_parallel_size
        self.num_heads_per_tp = self.config.num_attention_heads // tp_size
        self.num_kv_heads_per_tp = self.config.num_query_groups // tp_size
        self.kv_groups = self.num_heads_per_tp // self.num_kv_heads_per_tp

        # Beta and decay projections: hidden_size -> num_heads_per_tp
        from megatron.core.tensor_parallel import ColumnParallelLinear
        self.beta_proj = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.num_attention_heads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        )
        self.decay_proj = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.num_attention_heads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        )

        # Output gate
        if self.attn_output_gate:
            from megatron.core.tensor_parallel import ColumnParallelLinear as CPL
            self.gate_proj = CPL(
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=True,  # Must gather to match linear_proj output shape
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context=None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params=None,
        sequence_len_offset: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """GatedDeltaNet forward with optional prefix-sharing hook.

        Returns (output, bias) like SelfAttention.
        """
        # hidden_states: [sq, b, h]

        # QKV projection using inherited linear_qkv
        # Use parent's get_query_key_value_tensors for correct GQA interleaved split
        query, key, value = self.get_query_key_value_tensors(hidden_states, None)
        sq, b, _ = hidden_states.size()

        # Note: QK layernorm is already applied by get_query_key_value_tensors()

        # --- Prefix-sharing: detect context early (needed for RoPE correction) ---
        ps_slot_id = None
        ps_ctx = None
        ps_is_suffix_pass = False
        try:
            from prefix_sharing.integrations.context import current_prefix_sharing_context
            from prefix_sharing.core.model_spec import AttentionLayerType
            from prefix_sharing.core.prefix_store import PrefixActivationSlotId, PREFIX_STATE_TYPE_DELTANET_STATE
            from prefix_sharing.integrations.megatron_runtime import _read_parallel_rank_info, _normalize_layer_number
            ps_ctx = current_prefix_sharing_context()
        except ImportError:
            ps_ctx = None

        if ps_ctx is not None and ps_ctx.prefix_sharing_plan.has_sharing:
            ps_layer_id = _normalize_layer_number(self)
            ps_model_spec = ps_ctx.model_spec
            if ps_layer_id >= 0 and ps_model_spec is not None and \
                    ps_model_spec.layer_type(ps_layer_id) == AttentionLayerType.LINEAR_ATTENTION:
                _, ps_tp_rank, _ = _read_parallel_rank_info()
                ps_slot_id = PrefixActivationSlotId(
                    forward_id=ps_ctx.prefix_sharing_plan.forward_id,
                    micro_batch_id=ps_ctx.prefix_sharing_plan.micro_batch_id,
                    layer_id=ps_layer_id,
                    sample_idx_in_batch=0,
                    prefix_state_type=PREFIX_STATE_TYPE_DELTANET_STATE,
                    tp_rank=ps_tp_rank,
                )
                ps_is_suffix_pass = ps_ctx.deltanet_store.contains(ps_slot_id)

        # Apply RoPE - same pattern as SelfAttention.forward
        # For self attention, duplicate rotary_pos_emb if not already a tuple
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # Prefix-sharing: correct RoPE positions for suffix pass
        # In suffix pass, rotary_pos_emb covers positions 0..suffix_len-1,
        # but suffix tokens need absolute positions PREFIX_LEN..total_len-1.
        # We extend and slice to the correct absolute positions.
        if ps_is_suffix_pass and rotary_pos_emb is not None:
            from prefix_sharing.integrations.megatron_runtime import _extend_rope_pos_emb
            q_pos_emb, k_pos_emb = rotary_pos_emb
            stored_state = ps_ctx.deltanet_store.load(ps_slot_id)
            prefix_len_actual = stored_state.prefix_len
            suffix_len = sq
            total_len_needed = prefix_len_actual + suffix_len
            q_pos_emb_ext = _extend_rope_pos_emb(q_pos_emb, total_len_needed)
            k_pos_emb_ext = _extend_rope_pos_emb(k_pos_emb, total_len_needed)
            suffix_positions = torch.arange(prefix_len_actual, total_len_needed, device=query.device)
            q_pos_emb_suffix = q_pos_emb_ext.index_select(0, suffix_positions)
            k_pos_emb_suffix = k_pos_emb_ext.index_select(0, suffix_positions)
            rotary_pos_emb = (q_pos_emb_suffix, k_pos_emb_suffix)

        if rotary_pos_emb is not None:
            from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config)
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config)
        elif rotary_pos_cos is not None and rotary_pos_sin is not None:
            from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb_with_cos_sin
            key = apply_rotary_pos_emb_with_cos_sin(key, rotary_pos_cos, rotary_pos_sin)
            query = apply_rotary_pos_emb_with_cos_sin(query, rotary_pos_cos, rotary_pos_sin)

        # Beta and decay
        beta = torch.sigmoid(self.beta_proj(hidden_states)[0])  # [sq, b, heads_per_tp]
        decay = torch.sigmoid(self.decay_proj(hidden_states)[0])  # [sq, b, heads_per_tp]
        beta = beta.unsqueeze(-1)  # [sq, b, heads_per_tp, 1]
        decay = decay.unsqueeze(-1)

        # GQA: expand k, v to match q heads
        if self.kv_groups > 1:
            key = key.repeat_interleave(self.kv_groups, dim=2)
            value = value.repeat_interleave(self.kv_groups, dim=2)


        # Chunked cumsum attention (vectorized + memory-efficient)
        # Uses vectorized cumsum in chunks to avoid Python for-loop overhead
        # and avoid O(sq * b * h * d^2) full-sequence tensor
        d = self.hidden_size_per_attention_head
        h = self.num_heads_per_tp
        bh = b * h

        # Reshape: [sq, b, heads, head_dim] -> [sq, b*heads, head_dim]
        q_flat = query.reshape(sq, bh, d)
        k_flat = key.reshape(sq, bh, d)
        v_flat = value.reshape(sq, bh, d)
        beta_flat = beta.reshape(sq, bh, 1)
        decay_flat = decay.reshape(sq, bh, 1)

        # Initialize carry state
        carry = torch.zeros(bh, d, d, dtype=query.dtype, device=query.device)

        # Suffix pass: inject stored prefix carry state
        if ps_is_suffix_pass:
            stored_state = ps_ctx.deltanet_store.load(ps_slot_id)
            prefix_carry = stored_state.recurrent_state  # (prefix_bh, d, d)
            # Expand from prefix batch (1*h) to current batch (N*h)
            if prefix_carry.shape[0] != bh:
                prefix_carry = prefix_carry.repeat(bh // prefix_carry.shape[0], 1, 1)
            carry = prefix_carry.contiguous()

        CHUNK_SIZE = 8
        y = torch.zeros(sq, bh, d, dtype=query.dtype, device=query.device)

        num_chunks = (sq + CHUNK_SIZE - 1) // CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, sq)
            cs = end - start  # chunk size

            q_c = q_flat[start:end]
            k_c = k_flat[start:end]
            v_c = v_flat[start:end]
            beta_c = beta_flat[start:end]
            decay_c = decay_flat[start:end]

            # kv outer products: [cs, b*h, d, d]
            kv = torch.einsum("thd,the->thde", k_c, v_c)
            # beta-weighted update: [cs, b*h, d, d]
            update = beta_c.unsqueeze(-1) * kv

            # Cumulative decay product: [cs, b*h, 1]
            log_decay = torch.log(decay_c.clamp(min=1e-7))
            decay_prod = torch.exp(torch.cumsum(log_decay, dim=0))

            # Normalized update: update[i] / decay_prod[i]
            inv_decay = 1.0 / decay_prod.clamp(min=1e-7)
            norm_update = update * inv_decay.unsqueeze(-1)  # [cs, b*h, d, d]
            norm_cumsum = torch.cumsum(norm_update, dim=0)  # [cs, b*h, d, d]

            # State trajectory: state[t] = decay_prod[t] * (carry + norm_cumsum[t])
            carry_exp = carry.unsqueeze(0)  # [1, b*h, d, d]
            state_traj = decay_prod.unsqueeze(-1) * (carry_exp + norm_cumsum)  # [cs, b*h, d, d]

            # Query output: y[start:end] = q_c @ state_traj
            y[start:end] = torch.einsum("thd,thde->the", q_c, state_traj)

            # Carry forward: last state in chunk
            carry = state_traj[-1]  # [b*h, d, d]

        # Reshape back: [sq, b*heads, head_dim] -> [sq, b, heads*head_dim]
        y = y.reshape(sq, b, h * d)

        # --- Prefix-sharing: store carry state for prefix pass ---
        if ps_slot_id is not None and not ps_is_suffix_pass:
            ps_ctx.deltanet_store.store(
                ps_slot_id,
                recurrent_state=carry.contiguous(),
                prefix_len=sq,
            )


        # Output projection
        output, bias = self.linear_proj(y)

        # Output gate
        if self.attn_output_gate:
            gate = torch.sigmoid(self.gate_proj(hidden_states)[0])
            output = output * gate

        return output, bias

    def _apply_rotary_emb(self, query, key, q_pos_emb, k_pos_emb):
        """Apply partial RoPE: only rotate the first rope_dim dimensions."""
        from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

        if self.partial_rotary_factor < 1.0:
            rope_dim = self.rope_dim
            q_rot, q_pass = query[..., :rope_dim], query[..., rope_dim:]
            k_rot, k_pass = key[..., :rope_dim], key[..., rope_dim:]

            q_rot = apply_rotary_pos_emb(q_rot, q_pos_emb, config=self.config)
            k_rot = apply_rotary_pos_emb(k_rot, k_pos_emb, config=self.config)

            query = torch.cat([q_rot, q_pass], dim=-1)
            key = torch.cat([k_rot, k_pass], dim=-1)
        else:
            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config)

        return query, key

    def _apply_fused_rotary_emb(self, query, key, cos, sin):
        """Apply partial RoPE with fused cos/sin."""
        from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb_with_cos_sin

        if self.partial_rotary_factor < 1.0:
            rope_dim = self.rope_dim
            q_rot, q_pass = query[..., :rope_dim], query[..., rope_dim:]
            k_rot, k_pass = key[..., :rope_dim], key[..., rope_dim:]

            q_rot = apply_rotary_pos_emb_with_cos_sin(q_rot, cos, sin, config=self.config)
            k_rot = apply_rotary_pos_emb_with_cos_sin(k_rot, cos, sin, config=self.config)

            query = torch.cat([q_rot, q_pass], dim=-1)
            key = torch.cat([k_rot, k_pass], dim=-1)
        else:
            query = apply_rotary_pos_emb_with_cos_sin(query, cos, sin, config=self.config)
            key = apply_rotary_pos_emb_with_cos_sin(key, cos, sin, config=self.config)

        return query, key
