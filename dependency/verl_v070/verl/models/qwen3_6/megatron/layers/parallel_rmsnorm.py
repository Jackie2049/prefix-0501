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

import importlib
import numbers

import torch
from megatron.core import ModelParallelConfig
from torch import nn
from transformers import Qwen2Config

from verl.utils.megatron import sequence_parallel as sp_utils

try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine
    # Test if fused_rms_norm_affine actually works (apex CUDA kernel may not be compiled)
    _HAS_APEX = True
except ImportError:
    _HAS_APEX = False

# Further check: apex may be importable but CUDA kernel not available
if _HAS_APEX:
    try:
        import apex.normalization.fused_layer_norm as _fln
        # Check if the CUDA module is available
        importlib.import_module("fused_layer_norm_cuda")
    except (ImportError, ModuleNotFoundError):
        _HAS_APEX = False


class ParallelQwen2RMSNorm(nn.Module):
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        if isinstance(config.hidden_size, numbers.Integral):
            normalized_shape = (config.hidden_size,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.variance_epsilon = config.rms_norm_eps

        if megatron_config.sequence_parallel:
            sp_utils.mark_parameter_as_sequence_parallel(self.weight)

    def forward(self, hidden_states):
        if _HAS_APEX:
            return fused_rms_norm_affine(
                input=hidden_states,
                weight=self.weight,
                normalized_shape=self.normalized_shape,
                eps=self.variance_epsilon,
                memory_efficient=True,
            )
        else:
            # Pure PyTorch fallback: RMSNorm = x * rsqrt(mean(x^2) + eps) * weight
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states.to(torch.float32) * torch.rsqrt(variance + self.variance_epsilon)
            return (self.weight * hidden_states).to(input_dtype)
