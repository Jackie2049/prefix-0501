from dataclasses import dataclass
from typing import Optional

import pytest

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError


@dataclass
class ModelConfig:
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    num_layers_per_virtual_pipeline_stage: Optional[int] = None
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    context_parallel_size: int = 1
    context_parallel_algo: str = "kvallgather_cp_algo"
    apply_rope_fusion: bool = False
    fused_single_qkv_rope: bool = False
    model_type: str = "text_only_causal_lm"


def test_disabled_config_does_not_validate_model_constraints():
    config = PrefixSharingConfig(enable_prefix_sharing=False)
    config.validate(ModelConfig(pipeline_model_parallel_size=8))


def test_enabled_config_accepts_phase_one_constraints():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate(ModelConfig())


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_enabled_config_accepts_kvallgather_context_parallel_probe(cp_size):
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate(
        ModelConfig(
            context_parallel_size=cp_size,
            context_parallel_algo="kvallgather_cp_algo",
        )
    )


@pytest.mark.parametrize("pp_size", [1, 2, 4, 8])
def test_enabled_config_accepts_physical_pipeline_parallel_sizes(pp_size):
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate(ModelConfig(pipeline_model_parallel_size=pp_size))


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_enabled_config_accepts_megatron_sequence_parallel_with_common_tp_sizes(tp_size):
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate(ModelConfig(tensor_model_parallel_size=tp_size, sequence_parallel=True))


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "y"])
def test_env_var_can_enable_prefix_sharing(monkeypatch, value):
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", value)

    config = PrefixSharingConfig.from_raw(None)

    assert config.enable_prefix_sharing is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "n", ""])
def test_env_var_false_values_do_not_enable_prefix_sharing(monkeypatch, value):
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", value)

    config = PrefixSharingConfig.from_raw(None)

    assert config.enable_prefix_sharing is False


def test_env_var_rejects_invalid_prefix_sharing_value(monkeypatch):
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", "maybe")

    with pytest.raises(PrefixSharingConfigError, match="ENABLE_PREFIX_SHARING"):
        PrefixSharingConfig.from_raw(None)


@pytest.mark.parametrize(
    "field,value,message",
    [
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


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("pipeline_model_parallel_size", 0, "pipeline_model_parallel_size"),
        ("virtual_pipeline_model_parallel_size", 2, "virtual_pipeline_model_parallel_size"),
        ("num_layers_per_virtual_pipeline_stage", 8, "num_layers_per_virtual_pipeline_stage"),
    ],
)
def test_enabled_config_rejects_unsupported_pipeline_parallel_variants(field, value, message):
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


@pytest.mark.parametrize(
    "cp_algo",
    ["megatron_cp_algo", "ulysses_cp_algo", "hybrid_cp_algo", "adaptive_cp_algo"],
)
def test_enabled_config_rejects_non_kvallgather_context_parallel_algos(cp_algo):
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    with pytest.raises(PrefixSharingConfigError, match="context_parallel_algo"):
        config.validate(
            ModelConfig(
                context_parallel_size=2,
                context_parallel_algo=cp_algo,
            )
        )


def test_enabled_config_rejects_context_parallel_without_explicit_algo():
    model_config = {"context_parallel_size": 2}
    config = PrefixSharingConfig(enable_prefix_sharing=True)

    with pytest.raises(PrefixSharingConfigError, match="context_parallel_algo"):
        config.validate(model_config)


def test_engine_validation_accepts_kvallgather_context_parallel_probe():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate_for_engine(
        use_remove_padding=True,
        context_parallel_size=2,
        context_parallel_algo="kvallgather_cp_algo",
        dynamic_context_parallel=False,
    )


def test_engine_validation_rejects_dynamic_context_parallel():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    with pytest.raises(PrefixSharingConfigError, match="dynamic_context_parallel"):
        config.validate_for_engine(
            use_remove_padding=True,
            context_parallel_size=2,
            context_parallel_algo="kvallgather_cp_algo",
            dynamic_context_parallel=True,
        )


def test_engine_validation_rejects_context_parallel_without_explicit_algo():
    config = PrefixSharingConfig(enable_prefix_sharing=True)

    with pytest.raises(PrefixSharingConfigError, match="context_parallel_algo"):
        config.validate_for_engine(
            use_remove_padding=True,
            context_parallel_size=2,
            context_parallel_algo=None,
            dynamic_context_parallel=False,
        )
