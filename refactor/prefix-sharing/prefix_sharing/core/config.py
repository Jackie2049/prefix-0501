"""Configuration and hard phase-1 constraint validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class PrefixSharingConfigError(ValueError):
    """Raised when prefix sharing is enabled under unsupported constraints."""


def _read_config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class PrefixSharingConfig:
    """User-facing phase-1 configuration.

    The defaults are deliberately conservative: enabling this feature without
    explicitly opting in should do nothing, and unsupported Megatron/verl paths
    should fail loudly instead of silently changing training semantics.
    """

    enabled: bool = False
    detector: str = "trie"
    backend: str = "torch_ref"
    min_prefix_len: int = 1  # Prefixes shorter than this won't be cached (too short = not worth it)
    min_group_size: int = 2  # Groups smaller than this won't share (need 2+ samples to share)
    boundary_strategy: str = "restore_last_prefix_token"

    supported_pp_size: int = 1  # Pipeline parallel size supported in phase 1 (1 = no PP)
    supported_cp_size: int = 1  # Context parallel size supported in phase 1 (1 = no CP)
    supported_rope_fusion: bool = False  # RoPE fusion kernel support (False = must disable)
    supported_fused_qkv_rope: bool = False  # Fused QKV+RoPE kernel support (False = must disable)

    validate_precision: bool = False
    integrate_mode: str = "verl_megatron_actor"
    model_type: str = "text_only_causal_lm"  # Model type identifier; phase 1 only supports text-only causal LM

    def validate(self, model_config: Any | None = None, integrate_mode: str | None = None) -> None:
        """Validate phase-1 constraints against a model/config object.

        Args:
            model_config: Mapping or object with Megatron-like attributes.
            integrate_mode: Optional integration mode name.
        """

        if not self.enabled:
            return
        if self.detector != "trie":
            raise PrefixSharingConfigError("phase 1 supports only detector='trie'")
        if self.boundary_strategy != "restore_last_prefix_token":
            raise PrefixSharingConfigError(
                "phase 1 currently implements only "
                "boundary_strategy='restore_last_prefix_token'; future strategies may include "
                "'boundary_token' and 'strict_suffix'"
            )
        if self.min_prefix_len < 1:
            raise PrefixSharingConfigError("min_prefix_len must be >= 1")
        if self.min_group_size < 2:
            raise PrefixSharingConfigError("min_group_size must be >= 2")

        active_mode = integrate_mode or self.integrate_mode
        if active_mode != "verl_megatron_actor":
            raise PrefixSharingConfigError(
                "phase 1 supports only integrate_mode='verl_megatron_actor'"
            )

        pp_size = _read_config_value(
            model_config,
            "pipeline_model_parallel_size",
            _read_config_value(model_config, "pipeline_parallel_size", self.supported_pp_size),
        )
        cp_size = _read_config_value(
            model_config,
            "context_parallel_size",
            self.supported_cp_size,
        )
        rope_fusion = _read_config_value(model_config, "apply_rope_fusion", False)
        fused_qkv_rope = _read_config_value(model_config, "fused_single_qkv_rope", False)
        model_type = _read_config_value(model_config, "model_type", "text_only_causal_lm")

        if pp_size != self.supported_pp_size:
            raise PrefixSharingConfigError(
                f"[Config Error] pipeline_model_parallel_size={pp_size} 不支持当前阶段。"
                f"Phase 1 仅支持 pipeline_model_parallel_size=1 (无流水线并行)，"
                f"请修改配置将 PP 大小设为 1，或禁用 prefix sharing。"
            )
        if cp_size != self.supported_cp_size:
            raise PrefixSharingConfigError(
                f"[Config Error] context_parallel_size={cp_size} 不支持当前阶段。"
                f"Phase 1 仅支持 context_parallel_size=1 (无上下文并行)，"
                f"请修改配置将 CP 大小设为 1，或禁用 prefix sharing。"
            )
        if not self.supported_rope_fusion and rope_fusion:
            raise PrefixSharingConfigError(
                f"[Config Error] apply_rope_fusion=True 不支持当前阶段。"
                f"Phase 1 要求关闭 rope fusion (apply_rope_fusion=False)，"
                f"请修改配置或禁用 prefix sharing。"
            )
        if not self.supported_fused_qkv_rope and fused_qkv_rope:
            raise PrefixSharingConfigError(
                f"[Config Error] fused_single_qkv_rope=True 不支持当前阶段。"
                f"Phase 1 要求关闭 fused QKV rope (fused_single_qkv_rope=False)，"
                f"请修改配置或禁用 prefix sharing。"
            )
        if self.model_type == "text_only_causal_lm" and model_type != "text_only_causal_lm":
            raise PrefixSharingConfigError(
                f"[Config Error] 当前模型类型 '{model_type}' 不支持当前阶段。"
                f"Phase 1 仅支持 model_type='text_only_causal_lm' (纯文本因果语言模型)，"
                f"请使用支持的模型类型或禁用 prefix sharing。"
            )
