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
from megatron.core.process_groups_config import ProcessGroupCollection


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
        pg_collection: ProcessGroupCollection = None,
        partial_rotary_factor: float = 1.0,
        attn_output_gate: bool = False,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
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
                gather_output=False,
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
        # Check prefix-sharing context first
        from prefix_sharing.integrations.megatron_runtime import maybe_run_prefix_sharing_deltanet
        result = maybe_run_prefix_sharing_deltanet(
            attention_module=self,
            state_update=None,  # Will be computed inside
            packed_seq_params=packed_seq_params,
        )
        if result is not None:
            return result, None

        # Standard path: compute GatedDeltaNet linear attention
        # hidden_states: [sq, b, h]

        # QKV projection using inherited linear_qkv
        # Use parent's get_query_key_value_tensors for correct GQA interleaved split
        query, key, value = self.get_query_key_value_tensors(hidden_states, None, True)
        sq, b, _ = hidden_states.size()

        # Note: QK layernorm is already applied by get_query_key_value_tensors()

        # Apply RoPE using parent class method
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query, key = self._apply_rotary_emb(query, key, q_pos_emb, k_pos_emb)
        elif rotary_pos_cos is not None and rotary_pos_sin is not None:
            query, key = self._apply_fused_rotary_emb(query, key, rotary_pos_cos, rotary_pos_sin)

        # Beta and decay
        beta = torch.sigmoid(self.beta_proj(hidden_states)[0])  # [sq, b, heads_per_tp]
        decay = torch.sigmoid(self.decay_proj(hidden_states)[0])  # [sq, b, heads_per_tp]
        beta = beta.unsqueeze(-1)  # [sq, b, heads_per_tp, 1]
        decay = decay.unsqueeze(-1)

        # GQA: expand k, v to match q heads
        if self.kv_groups > 1:
            key = key.repeat_interleave(self.kv_groups, dim=2)
            value = value.repeat_interleave(self.kv_groups, dim=2)

        # GatedDeltaNet cumsum attention
        # k, v, query: [sq, b, heads, head_dim]
        # Reshape to [sq * b, heads, head_dim] for cumsum
        q_flat = query.reshape(sq * b, self.num_heads_per_tp, self.hidden_size_per_attention_head)
        k_flat = key.reshape(sq * b, self.num_heads_per_tp, self.hidden_size_per_attention_head)
        v_flat = value.reshape(sq * b, self.num_heads_per_tp, self.hidden_size_per_attention_head)
        beta_flat = beta.reshape(sq * b, self.num_heads_per_tp, 1)

        # kv outer product: [t, h, d, d]
        kv = torch.einsum('thd,the->thde', k_flat, v_flat)
        update = beta_flat.unsqueeze(-1) * kv  # [t, h, d, d]

        # Cumsum over sequence
        trajectory = torch.cumsum(update, dim=0)

        # Query the state
        y = torch.einsum('thd,thde->the', q_flat, trajectory)

        # Reshape back: [sq, b, hidden_per_tp]
        y = y.reshape(sq, b, self.num_heads_per_tp * self.hidden_size_per_attention_head)

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
