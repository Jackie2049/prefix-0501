from dataclasses import dataclass

import pytest

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError


@dataclass
class ModelConfig:
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    apply_rope_fusion: bool = False
    fused_single_qkv_rope: bool = False
    model_type: str = "text_only_causal_lm"


def test_disabled_config_does_not_validate_model_constraints():
    config = PrefixSharingConfig(enable_prefix_sharing=False)
    config.validate(ModelConfig(pipeline_model_parallel_size=8))


def test_enabled_config_accepts_phase_one_constraints():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate(ModelConfig())


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("pipeline_model_parallel_size", 2, "pipeline_model_parallel_size=2"),
        ("context_parallel_size", 2, "context_parallel_size=2"),
        ("apply_rope_fusion", True, "apply_rope_fusion=True"),
        ("fused_single_qkv_rope", True, "fused_single_qkv_rope=True"),
        ("model_type", "vlm", "model_type"),
    ],
)
def test_enabled_config_rejects_unsupported_phase_one_constraints(field, value, message):
    model_config = ModelConfig()
    setattr(model_config, field, value)
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    with pytest.raises(PrefixSharingConfigError, match=message):
        config.validate(model_config)


def test_enabled_config_rejects_non_phase_one_modes():
    with pytest.raises(PrefixSharingConfigError, match="detector"):
        PrefixSharingConfig(enable_prefix_sharing=True, detector="prompt").validate(ModelConfig())
    with pytest.raises(PrefixSharingConfigError, match="boundary_strategy"):
        PrefixSharingConfig(enable_prefix_sharing=True, boundary_strategy="restore_last_prefix_token").validate(ModelConfig())
    with pytest.raises(PrefixSharingConfigError, match="integrate_mode"):
        PrefixSharingConfig(enable_prefix_sharing=True).validate(ModelConfig(), integrate_mode="verl_fsdp")
