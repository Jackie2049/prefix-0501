"""Model-specific configuration for prefix-sharing.

Encapsulates model architecture details that affect how prefix-sharing operates:
layer types (full attention vs linear attention), head configurations, RoPE
parameters, and gated output settings.

Usage:
    from prefix_sharing.core.model_spec import ModelSpec, QWEN3_6_27B

    spec = ModelSpec.from_hf_config(hf_config)
    # Or use a preset:
    spec = QWEN3_6_27B
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class AttentionLayerType(Enum):
    FULL_ATTENTION = "full_attention"
    LINEAR_ATTENTION = "linear_attention"


@dataclass(frozen=True)
class ModelSpec:
    """Model architecture specification for prefix-sharing.

    Captures the information needed by prefix-sharing backends and
    integrations to correctly handle different layer types, head
    configurations, and RoPE settings.
    """

    num_hidden_layers: int
    num_attention_heads: int  # query heads
    num_key_value_heads: int  # KV heads (GQA)
    head_dim: int
    full_attention_interval: int = 0  # 0 = all layers are full attention
    attn_output_gate: bool = False
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int = 32768

    def layer_type(self, layer_id: int) -> AttentionLayerType:
        """Determine the attention type for a given layer.

        When full_attention_interval > 0, the last layer in each interval
        group uses full attention (e.g., layers 3,7,11,... for interval=4).
        All other layers use linear attention.
        When full_attention_interval == 0, all layers use full attention.
        """
        if self.full_attention_interval <= 0:
            return AttentionLayerType.FULL_ATTENTION
        # Qwen3.6 pattern: full attention at layers where (layer_id + 1) % interval == 0
        # i.e., the last layer in each group of `interval` layers
        if (layer_id + 1) % self.full_attention_interval == 0:
            return AttentionLayerType.FULL_ATTENTION
        return AttentionLayerType.LINEAR_ATTENTION

    def full_attention_layer_ids(self) -> list[int]:
        """Return indices of all full attention layers."""
        return [
            i for i in range(self.num_hidden_layers)
            if self.layer_type(i) == AttentionLayerType.FULL_ATTENTION
        ]

    def linear_attention_layer_ids(self) -> list[int]:
        """Return indices of all linear attention layers."""
        return [
            i for i in range(self.num_hidden_layers)
            if self.layer_type(i) == AttentionLayerType.LINEAR_ATTENTION
        ]

    @property
    def num_full_attention_layers(self) -> int:
        return len(self.full_attention_layer_ids())

    @property
    def num_linear_attention_layers(self) -> int:
        return len(self.linear_attention_layer_ids())

    @property
    def gqa_group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf_config(cls, config: Any) -> ModelSpec:
        """Build a ModelSpec from a HuggingFace model config object or dict."""
        def _get(cfg: Any, name: str, default: Any = None) -> Any:
            if isinstance(cfg, Mapping):
                return cfg.get(name, default)
            return getattr(cfg, name, default)

        return cls(
            num_hidden_layers=_get(config, "num_hidden_layers", 1),
            num_attention_heads=_get(config, "num_attention_heads", 1),
            num_key_value_heads=_get(config, "num_key_value_heads",
                                     _get(config, "num_attention_heads", 1)),
            head_dim=_get(config, "head_dim",
                          _get(config, "hidden_size", 4096)
                          // _get(config, "num_attention_heads", 32)),
            full_attention_interval=_get(config, "full_attention_interval", 0),
            attn_output_gate=_get(config, "attn_output_gate", False),
            partial_rotary_factor=_get(config, "partial_rotary_factor", 1.0),
            max_position_embeddings=_get(config, "max_position_embeddings", 32768),
        )


# ---------------------------------------------------------------------------
# Preset model specifications
# ---------------------------------------------------------------------------

QWEN3_6_27B = ModelSpec(
    num_hidden_layers=64,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    full_attention_interval=4,
    attn_output_gate=True,
    partial_rotary_factor=0.25,
    max_position_embeddings=131072,
)
"""Qwen3.6-27B HybridAttention: 16 full attention (GQA 24:4) + 48 linear attention layers."""
