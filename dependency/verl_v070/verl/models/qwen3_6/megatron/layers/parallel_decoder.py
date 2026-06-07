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

"""Qwen3.6 hybrid decoder layer.

Routes between full attention (ParallelQwen3_6AttentionRmPad) and
GatedDeltaNet linear attention (ParallelQwen3_6GatedDeltaNetRmPad)
based on layer_idx % full_attention_interval.
"""

from typing import Optional

import torch
from megatron.core import ModelParallelConfig
from torch import nn

from verl.utils.megatron_utils import TransformerConfig, convert_config

from .parallel_attention import ParallelQwen3_6Attention, ParallelQwen3_6AttentionRmPad
from .parallel_deltanet import ParallelQwen3_6GatedDeltaNet, ParallelQwen3_6GatedDeltaNetRmPad
from .parallel_mlp import ParallelQwen2MLP
from .parallel_rmsnorm import ParallelQwen2RMSNorm


def _is_full_attention_layer(config, layer_idx: int) -> bool:
    """Check if a layer should use full attention vs GatedDeltaNet.

    Uses the config's layer_types list (if available) for accurate routing.
    Falls back to positional heuristic: full attention at layers where
    (layer_idx + 1) % full_attention_interval == 0, meaning the last layer
    in each interval group gets full attention (e.g., layers 3,7,11,...).
    """
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and layer_idx < len(layer_types):
        return layer_types[layer_idx] == "full_attention"
    full_attention_interval = getattr(config, "full_attention_interval", 1)
    return (layer_idx + 1) % full_attention_interval == 0


class ParallelQwen3_6DecoderLayer(nn.Module):
    """Standard (padded) decoder layer with hybrid attention routing."""

    def __init__(self, config, megatron_config: ModelParallelConfig, layer_idx: int):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        if _is_full_attention_layer(config, layer_idx):
            self.self_attn = ParallelQwen3_6Attention(config=config, megatron_config=megatron_config)
        else:
            self.self_attn = ParallelQwen3_6GatedDeltaNet(config=config, megatron_config=megatron_config)
        self.self_attn.layer_idx = layer_idx

        self.mlp = ParallelQwen2MLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelQwen2RMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelQwen2RMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ParallelQwen3_6DecoderLayerRmPad(nn.Module):
    """Remove-padding decoder layer with hybrid attention routing."""

    def __init__(self, config, megatron_config: ModelParallelConfig, layer_idx: int):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        if _is_full_attention_layer(config, layer_idx):
            self.self_attn = ParallelQwen3_6AttentionRmPad(config=config, megatron_config=megatron_config)
        else:
            self.self_attn = ParallelQwen3_6GatedDeltaNetRmPad(config=config, megatron_config=megatron_config)
        self.self_attn.layer_idx = layer_idx

        self.mlp = ParallelQwen2MLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelQwen2RMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelQwen2RMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
