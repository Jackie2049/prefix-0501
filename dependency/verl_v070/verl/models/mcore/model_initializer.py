# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# use mcore transformer config to initialize the model
import inspect
from abc import ABC, abstractmethod

import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from .config_converter import PretrainedConfig, TransformerConfig


def _add_output_gate_to_self_attention(attn_module):
    """Add output gate (gate_proj) to a SelfAttention module.

    Qwen3.6 HybridAttention uses attn_output_gate=True for ALL layers,
    including full attention (SelfAttention) layers. Megatron's SelfAttention
    doesn't have built-in output gating, so we add gate_proj and register a
    forward hook to apply the gate after linear_proj.

    The forward hook works for both normal forward and PS hook paths, since
    it fires after SelfAttention.forward() returns regardless of whether
    the PS hook intercepted.
    """
    from megatron.core.tensor_parallel import ColumnParallelLinear

    attn_module.attn_output_gate = True
    attn_module.gate_proj = ColumnParallelLinear(
        attn_module.config.hidden_size,
        attn_module.config.hidden_size,
        config=attn_module.config,
        init_method=attn_module.config.init_method,
        gather_output=True,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )

    def _output_gate_hook(module, input, output):
        """Apply output gate after linear_proj.

        Config says output_gate_type="swish" but the real Qwen3.6 HuggingFace
        implementation uses torch.sigmoid(gate) (modeling_qwen3_next.py line 413).
        The gate is embedded in a doubled q_proj (2*head_dim), not a separate
        gate_proj — our Megatron model adds a separate gate_proj for convenience.
        """
        if isinstance(output, tuple) and len(output) >= 2:
            hidden_states = input[0]
            gate = torch.sigmoid(module.gate_proj(hidden_states)[0])
            gated_output = output[0] * gate
            return (gated_output, output[1])
        return output

    attn_module.register_forward_hook(_output_gate_hook)


class BaseModelInitializer(ABC):
    """Base class for model initializers."""

    def __init__(self, tfconfig: TransformerConfig, hf_config: PretrainedConfig):
        self.tfconfig = tfconfig
        self.hf_config = hf_config
        self.has_vp_stage = inspect.signature(get_gpt_decoder_block_spec).parameters.get("vp_stage", None) is not None

    @abstractmethod
    def get_transformer_layer_spec(self, vp_stage=None):
        """Get the transformer layer specification.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_layer_specs.py"""
        pass

    def get_rope_scaling_args(self) -> dict:
        """Get rope scaling args."""
        rope_scaling_args = {}
        if "rope_scaling" in self.hf_config:
            if self.hf_config.rope_scaling is not None:
                # assert self.hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
                rope_scaling_args["seq_len_interpolation_factor"] = self.hf_config.rope_scaling["factor"]
        return rope_scaling_args

    def initialize(
        self,
        pre_process: bool = True,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
        value: bool = False,
        **extra_kwargs,
    ) -> GPTModel:
        """Initialize a GPT model with the given configuration.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py

        Args:
            pre_process (bool): include embedding layer.
            post_process (bool): including an output layer.
            share_embeddings_and_output_weights (bool): input embeddings and output logit weights are shared.
            value (bool): add an extra linear layer for classification or regression.

        Returns:
            GPTModel: An initialized GPT model instance
        """
        rotary_percent = extra_kwargs.pop('rotary_percent', 1.0)
        vp_stage = extra_kwargs.get("vp_stage", None)
        transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
        rope_scaling_args = self.get_rope_scaling_args()
        mtp_block_spec = extra_kwargs.get("mtp_block_spec", None)
        model = GPTModel(
            config=self.tfconfig,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
            rotary_percent=rotary_percent,
            **rope_scaling_args,
            mtp_block_spec=mtp_block_spec,
            **({} if not self.has_vp_stage else {"vp_stage": vp_stage}),
        )

        if post_process and value:
            from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

            model.output_layer = LinearForLastLayer(
                input_size=self.tfconfig.hidden_size, output_size=1, config=self.tfconfig
            )

        return model


class DenseModel(BaseModelInitializer):
    """Initializer for dense models like Llama and Qwen2."""

    def get_transformer_layer_spec(self, vp_stage=None):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        return get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)


class Qwen2MoEModel(BaseModelInitializer):
    """Initializer for Qwen2 MoE models."""

    def get_transformer_layer_spec(self, vp_stage=None):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)

        # Patch layer spec for shared experts
        for i in range(len(transformer_layer_spec.layer_specs)):
            transformer_layer_spec.layer_specs[i].submodules.mlp.submodules.shared_experts.params["gate"] = True

        return transformer_layer_spec

    def initialize(self, **kwargs):
        # Qwen default freeze_moe_router: true
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class MixtralModel(BaseModelInitializer):
    """Initializer for Mixtral models."""

    def get_transformer_layer_spec(self, vp_stage=None):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        return transformer_layer_spec

    def initialize(self, **kwargs):
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", False)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class Qwen3MoEModel(BaseModelInitializer):
    """Initializer for Qwen3 MoE models."""

    def get_transformer_layer_spec(self, vp_stage=None):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        return transformer_layer_spec

    def initialize(self, **kwargs):
        # Qwen default freeze_moe_router: true
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class DeepseekV3Model(BaseModelInitializer):
    """Initializer for DeepseekV3 models."""

    def get_transformer_layer_spec(self, vp_stage=None):
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        return transformer_layer_spec

    def get_rope_scaling_args(self) -> dict:
        """Get rope scaling args."""
        rope_scaling_args = {}
        return rope_scaling_args

    def initialize(
        self,
        **kwargs,
    ):
        vp_stage = kwargs.get("vp_stage", None)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            self.tfconfig.moe_router_load_balancing_type = "none"
        # MTP
        if self.tfconfig.mtp_num_layers is not None and self.tfconfig.mtp_num_layers > 0:
            transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
            mtp_block_spec = get_gpt_mtp_block_spec(
                self.tfconfig, transformer_layer_spec, use_transformer_engine=True, vp_stage=vp_stage
            )
            kwargs["mtp_block_spec"] = mtp_block_spec

        model = super().initialize(**kwargs)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                if hasattr(layer.mlp, "router"):
                    layer.mlp.router.weight.requires_grad = False
        return model


class Qwen25VLModel(BaseModelInitializer):
    """Initializer for Qwen2.5 VL models."""

    def get_transformer_layer_spec(self, vp_stage=None):
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        return transformer_layer_spec

    def initialize(
        self,
        pre_process=None,
        post_process=None,
        share_embeddings_and_output_weights=False,
        value=False,
        **extra_kwargs,
    ):
        tfconfig = self.tfconfig
        hf_config = self.hf_config
        # Qwen2_5_VLForConditionalGeneration
        from copy import deepcopy

        transformer_layer_spec = self.get_transformer_layer_spec()

        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
        from megatron.core.models.gpt.moe_module_specs import MLPSubmodules
        from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec

        from .qwen2_5_vl import Qwen2_5VLModel, get_vision_model_config, get_vision_projection_config

        vision_transformer_config = get_vision_model_config(deepcopy(tfconfig))
        vision_transformer_config.pipeline_model_parallel_size = 1
        vision_transformer_config.first_pipeline_num_layers = None

        vision_projection_config = get_vision_projection_config(
            deepcopy(tfconfig),
            vision_transformer_config.hidden_size,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        vision_projection_layer_spec = MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        )
        vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

        qwen25_vl_model = Qwen2_5VLModel(
            language_transformer_config=tfconfig,
            language_transformer_layer_spec=transformer_layer_spec,
            language_vocab_size=hf_config.vocab_size,
            language_max_sequence_length=hf_config.max_position_embeddings,
            vision_transformer_config=vision_transformer_config,
            vision_transformer_layer_spec=vision_transformer_layer_spec,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_layer_spec,
            vision_projection_type="mlp",
            language_rotary_base=hf_config.rope_theta,
            pre_process=pre_process,
            post_process=post_process,
            add_decoder=True,
            add_encoder=True,
            parallel_output=True,
            language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

        if post_process and value:
            from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

            qwen25_vl_model.language_model.output_layer = LinearForLastLayer(
                input_size=tfconfig.hidden_size, output_size=1, config=tfconfig
            )

        return qwen25_vl_model


class Qwen3_6HybridModel(BaseModelInitializer):
    """Initializer for Qwen3.6 HybridAttention models.

    Qwen3.6 uses interleaved full attention and GatedDeltaNet linear attention:
    - Layers where layer_idx % full_attention_interval == 0 → standard SelfAttention
    - All other layers → GatedDeltaNet linear attention

    This initializer creates a standard GPTModel, then replaces the linear
    attention layers with GatedDeltaNetAttention modules.
    """

    def get_transformer_layer_spec(self, vp_stage=None):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        # Check if Transformer Engine is available at runtime
        try:
            from megatron.core.extensions.transformer_engine import TENorm  # noqa: F401
            use_te = True
        except ImportError:
            use_te = False

        # Qwen3.6 uses RMSNorm; must ensure LNImpl is WrappedTorchNorm BEFORE
        # calling get_gpt_decoder_block_spec, because it captures the global LNImpl
        # for block-level final_layernorm at line 340 BEFORE calling
        # get_gpt_layer_local_spec which later changes LNImpl. FusedLayerNorm
        # (the default LNImpl when apex is partially importable) does NOT support
        # RMSNorm, causing an AssertionError at TransformerBlock._build_layers().
        if not use_te and self.tfconfig.normalization == "RMSNorm":
            import megatron.core.models.gpt.gpt_layer_specs as specs_module
            from megatron.core.transformer.torch_norm import WrappedTorchNorm
            specs_module.LNImpl = WrappedTorchNorm

        return get_gpt_decoder_block_spec(
            self.tfconfig, use_transformer_engine=use_te,
            normalization=self.tfconfig.normalization, **extra_kwargs)

    def initialize(self, **kwargs):
        # Pass rotary_percent=partial_rotary_factor for partial RoPE (Qwen3.6)
        partial_rotary_factor = getattr(self.hf_config, "partial_rotary_factor", 1.0)
        kwargs.setdefault('rotary_percent', partial_rotary_factor)

        model = super().initialize(**kwargs)

        # Read HybridAttention config from hf_config
        full_attention_interval = getattr(self.hf_config, "full_attention_interval", 1)
        attn_output_gate = getattr(self.hf_config, "attn_output_gate", False)

        if full_attention_interval <= 1:
            # No hybrid attention, all layers are full attention
            return model

        # Replace linear attention layers with GatedDeltaNetAttention
        # Real Qwen3.6 pattern: self_attn at layer_idx % interval == interval - 1
        # (e.g., interval=4 → self_attn at L3, L7, L11, L15)
        # All other layers use GatedDeltaNet linear attention
        from .gated_delta_net import GatedDeltaNetAttention

        # Get the SelfAttentionSubmodules from the block spec
        vp_stage = kwargs.get("vp_stage", None)
        block_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
        sa_modspec = block_spec.layer_specs[0].submodules.self_attention
        sa_submodules = sa_modspec.submodules

        for i, layer in enumerate(model.decoder.layers):
            if i % full_attention_interval != full_attention_interval - 1:
                # This is a linear attention layer — replace SelfAttention
                old_attn = layer.self_attention

                # Create GatedDeltaNetAttention with correct submodules
                new_attn = GatedDeltaNetAttention(
                    config=self.tfconfig,
                    submodules=sa_submodules,
                    layer_number=old_attn.layer_number,
                    attn_mask_type=old_attn.attn_mask_type,
                    partial_rotary_factor=partial_rotary_factor,
                    attn_output_gate=attn_output_gate,
                )
                new_attn.to(next(old_attn.parameters()).device)

                # Copy weights from old SelfAttention to new GatedDeltaNet
                # linear_qkv and linear_proj are shared
                new_attn.linear_qkv = old_attn.linear_qkv
                new_attn.linear_proj = old_attn.linear_proj
                if hasattr(old_attn, 'q_layernorm') and old_attn.q_layernorm is not None:
                    new_attn.q_layernorm = old_attn.q_layernorm
                if hasattr(old_attn, 'k_layernorm') and old_attn.k_layernorm is not None:
                    new_attn.k_layernorm = old_attn.k_layernorm

                # Replace the attention module
                layer.self_attention = new_attn
            else:
                # This is a full attention layer — add output gate if needed
                if attn_output_gate:
                    _add_output_gate_to_self_attention(layer.self_attention)

        return model
