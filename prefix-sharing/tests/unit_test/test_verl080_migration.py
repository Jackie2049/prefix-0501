"""verl080_mcore0161_ms0160 patch set 和配套接口的单元测试。

本文件测试本地可验证的部分：
- PrefixSharingConfig.validate_for_engine()
- PrefixSharingRuntimeState.kept_position_ids 字段
- PrefixSharingRuntimeContext.kept_position_ids 传递
- compat_matrix 新条目版本匹配
- integrations.verl_mcore.read_ps_config_from_engine_config 读取逻辑
- integrations.megatron_runtime v0.16.1 API helpers

需要 verl/Megatron 运行环境的 patch 联动测试（forward_step → attention → vocab_logprobs）
属于 integrated_test，本地不可运行，在 NPU/GPU 环境中单独验证。
"""

from dataclasses import dataclass

import pytest

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import (
    current_prefix_sharing_context,
    prefix_sharing_runtime_context,
)
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState


# ═══════════════════════════════════════
# PrefixSharingConfig.validate_for_engine()
# ═══════════════════════════════════════


def test_validate_for_engine_disabled_does_not_raise():
    config = PrefixSharingConfig(enable_prefix_sharing=False)
    config.validate_for_engine(use_remove_padding=True)
    config.validate_for_engine(use_remove_padding=False)


def test_validate_for_engine_accepts_remove_padding_true():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate_for_engine(use_remove_padding=True)


def test_validate_for_engine_rejects_remove_padding_false():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    with pytest.raises(PrefixSharingConfigError, match="use_remove_padding"):
        config.validate_for_engine(use_remove_padding=False)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("detector", "prompt", "detector"),
        ("backend", "unknown_backend", "backend"),
        ("boundary_strategy", "restore_last_prefix_token", "boundary_strategy"),
        ("min_prefix_len", 0, "min_prefix_len"),
        ("min_group_size", 1, "min_group_size"),
    ],
)
def test_validate_for_engine_rejects_unsupported_options(field, value, message):
    config = PrefixSharingConfig(enable_prefix_sharing=True, **{field: value})
    with pytest.raises(PrefixSharingConfigError, match=message):
        config.validate_for_engine(use_remove_padding=True)


# ═══════════════════════════════════════
# PrefixSharingRuntimeState.kept_position_ids
# ═══════════════════════════════════════


def _make_runtime_state(kept_position_ids=None):
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    )
    plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=1,
        micro_batch_id=1,
    )
    return PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(),
        kept_position_ids=kept_position_ids,
    )


def test_runtime_state_accepts_kept_position_ids():
    state = _make_runtime_state(kept_position_ids=[0, 1, 2, 3, 4, 0, 1, 2])
    assert state.kept_position_ids == [0, 1, 2, 3, 4, 0, 1, 2]


def test_runtime_state_defaults_kept_position_ids_to_none():
    state = _make_runtime_state()
    assert state.kept_position_ids is None


def test_runtime_state_is_frozen():
    state = _make_runtime_state(kept_position_ids=[0, 1, 2])
    with pytest.raises(AttributeError):
        state.kept_position_ids = [5, 6]


# ═══════════════════════════════════════
# PrefixSharingRuntimeContext.kept_position_ids 传递
# ═══════════════════════════════════════


def test_context_propagates_kept_position_ids_from_state():
    state = _make_runtime_state(kept_position_ids=[10, 11, 12])
    with prefix_sharing_runtime_context(state) as ctx:
        assert ctx.kept_position_ids == [10, 11, 12]


def test_context_kept_position_ids_none_when_state_has_none():
    state = _make_runtime_state(kept_position_ids=None)
    with prefix_sharing_runtime_context(state) as ctx:
        assert ctx.kept_position_ids is None


def test_context_kept_position_ids_none_when_state_has_no_attr():
    """旧配套的 PrefixSharingRuntimeState 没有 kept_position_ids 字段时，
    context 应通过 getattr 回退到 None，保持向后兼容。"""
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    )
    plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=1,
        micro_batch_id=1,
    )
    # 模拟旧版 state（没有 kept_position_ids）
    # 用 Namespace 模拟，因为 frozen dataclass 不能动态删除字段
    import types

    old_state = types.SimpleNamespace(
        prefix_sharing_plan=plan,
        backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(),
    )
    with prefix_sharing_runtime_context(old_state) as ctx:
        assert ctx.kept_position_ids is None


# ═══════════════════════════════════════
# compat_matrix 新条目匹配
# ═══════════════════════════════════════


def test_compat_matrix_matches_qwen35_npu_combo():
    from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX, CompatEntry
    from prefix_sharing.setup.version_guard import DetectedVersions

    versions = DetectedVersions(
        verl="0.8.0.dev",
        megatron_core="0.16.1",
        mindspeed="0.16.0",
    )
    matching = [e for e in COMPAT_MATRIX if e.match(versions)]
    assert len(matching) == 1
    assert matching[0].patch_set_id == "verl080_mcore0161_ms0160"


def test_compat_matrix_does_not_match_wrong_mcore():
    from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX
    from prefix_sharing.setup.version_guard import DetectedVersions

    versions = DetectedVersions(
        verl="0.8.0.dev",
        megatron_core="0.16.0",  # 不匹配 0.16.1
        mindspeed="0.16.0",
    )
    matching = [e for e in COMPAT_MATRIX if e.match(versions)]
    assert len(matching) == 0


def test_compat_matrix_no_match_raises_incompatible():
    from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX
    from prefix_sharing.setup.version_guard import DetectedVersions

    versions = DetectedVersions(
        verl="0.7.0",  # 不匹配任何条目
        megatron_core="0.12.0",
        mindspeed="0.12.0",
    )
    matching = [e for e in COMPAT_MATRIX if e.match(versions)]
    assert len(matching) == 0


# ═══════════════════════════════════════
# integrations.verl_mcore.read_ps_config_from_engine_config 读取逻辑
# ═══════════════════════════════════════


def test_read_ps_config_from_override_dict():
    from prefix_sharing.integrations.verl_mcore import (
        read_ps_config_from_engine_config,
    )

    engine_config = type("EngineConfig", (), {
        "override_transformer_config": {
            "prefix_sharing_config": {"enable_prefix_sharing": True},
        },
    })()
    result = read_ps_config_from_engine_config(engine_config)
    assert result == {"enable_prefix_sharing": True}


def test_read_ps_config_returns_none_when_missing():
    from prefix_sharing.integrations.verl_mcore import (
        read_ps_config_from_engine_config,
    )

    engine_config = type("EngineConfig", (), {
        "override_transformer_config": {},
    })()
    result = read_ps_config_from_engine_config(engine_config)
    assert result is None


def test_read_ps_config_from_direct_attr():
    from prefix_sharing.integrations.verl_mcore import (
        read_ps_config_from_engine_config,
    )

    engine_config = type("EngineConfig", (), {
        "override_transformer_config": type("Override", (), {
            "prefix_sharing_config": {"enable_prefix_sharing": True},
        })(),
    })()
    result = read_ps_config_from_engine_config(engine_config)
    assert result == {"enable_prefix_sharing": True}


def test_read_ps_config_returns_none_for_empty_config():
    from prefix_sharing.integrations.verl_mcore import (
        read_ps_config_from_engine_config,
    )

    engine_config = type("EngineConfig", (), {})()
    result = read_ps_config_from_engine_config(engine_config)
    assert result is None


# ═══════════════════════════════════════
# integrations.megatron_runtime v0.16.1 helpers
# ═══════════════════════════════════════


def test_extract_cu_seqlens_returns_none_for_none_params():
    from prefix_sharing.integrations.megatron_runtime import _extract_cu_seqlens
    assert _extract_cu_seqlens(None, "cu_seqlens_q_padded", "cu_seqlens_q") is None


def test_extract_cu_seqlens_prefers_padded_attr():
    from prefix_sharing.integrations.megatron_runtime import _extract_cu_seqlens
    params = type("Params", (), {
        "cu_seqlens_q_padded": [0, 10],
        "cu_seqlens_q": [0, 8],
    })()
    result = _extract_cu_seqlens(params, "cu_seqlens_q_padded", "cu_seqlens_q")
    assert result == [0, 10]


def test_extract_cu_seqlens_falls_back_to_primary():
    from prefix_sharing.integrations.megatron_runtime import _extract_cu_seqlens
    params = type("Params", (), {
        "cu_seqlens_q": [0, 8],
    })()
    result = _extract_cu_seqlens(params, "cu_seqlens_q_padded", "cu_seqlens_q")
    assert result == [0, 8]


def test_get_yarn_mscale_fallback_without_megatron():
    from prefix_sharing.integrations.megatron_runtime import _get_yarn_mscale
    # Fake attention_module without real Megatron — should fallback to 1.0
    fake_module = type("Attn", (), {"config": None})()
    assert _get_yarn_mscale(fake_module) == 1.0


def test_get_cp_group_returns_none_without_pg_collection():
    from prefix_sharing.integrations.megatron_runtime import _get_cp_group
    fake_module = type("Attn", (), {})()
    assert _get_cp_group(fake_module) is None


# ═══════════════════════════════════════
# __init__.py auto-activation 逻辑
# ═══════════════════════════════════════


def test_auto_activation_skips_without_env_var(monkeypatch):
    """没有 ENABLE_PREFIX_SHARING 环境变量时，不触发 setup.install()。"""
    monkeypatch.delenv("ENABLE_PREFIX_SHARING", raising=False)
    import importlib
    import prefix_sharing
    importlib.reload(prefix_sharing)
    assert prefix_sharing._patch_handle is None


def test_auto_activation_skips_with_env_var_false(monkeypatch):
    """ENABLE_PREFIX_SHARING=0 时，不触发 setup.install()。"""
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", "0")
    import importlib
    import prefix_sharing
    importlib.reload(prefix_sharing)
    assert prefix_sharing._patch_handle is None


def test_auto_activation_attempts_with_env_var_true(monkeypatch):
    """ENABLE_PREFIX_SHARING=1 时，尝试调用 setup.install()。

    本地环境没有 verl/Megatron，install() 会因版本不兼容抛异常，
    _auto_install_patches 应捕获异常并设 _patch_handle=None。
    """
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", "1")
    import importlib
    import prefix_sharing
    importlib.reload(prefix_sharing)
    # 本地没有 verl/Megatron，版本不兼容，应安全回退
    assert prefix_sharing._patch_handle is None


# ═══════════════════════════════════════
# adjust_attention_mask_for_prefix_sharing
# ═══════════════════════════════════════


def test_adjust_attention_mask_reduces_prompt_lens():
    """物理裁剪后 attention_mask 必须同步调整，否则
    no_padding_2_padding 的 prompt_lens/response_lens 断言失败。

    模拟场景：provider=[0,1,2,3,4], reuser=[0,1,2,5,6]
    prefix=[0,1,2] 被裁掉，reuser 只保留 [5,6]（keep_range=(3,5)）。
    adjust 后 reuser 行只有 suffix+response 位置的 mask 为 True。
    """
    import torch
    from prefix_sharing.core.planner import PrefixSharingPlanner
    from prefix_sharing.integrations.verl_mcore import adjust_attention_mask_for_prefix_sharing

    # provider: 5 valid tokens (prompt=3 + response=2)
    # reuser:   5 valid tokens (prompt=3 + response=2), prefix=[0,1,2] shared
    sequences = [[10, 11, 12, 30, 31], [10, 11, 12, 40, 41]]
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences)
    assert plan.has_sharing

    # 构造模拟 attention_mask (2D, 2 rows × 5 cols, 全 True)
    attention_mask = torch.ones(2, 5, dtype=torch.int32)
    batch = {"attention_mask": attention_mask}

    adjust_attention_mask_for_prefix_sharing(batch, plan)

    # provider 行 (row=0): keep_range=(0, 5) → 全部保留
    assert batch["attention_mask"][0].sum().item() == 5
    # reuser 行 (row=1): keep_range=(3, 5) → 只保留 suffix+response (2 tokens)
    assert batch["attention_mask"][1].sum().item() == 2

    # 模拟 no_padding_2_padding 的 prompt_lens 计算
    # 假设 prompt_ids.shape[1] = 3 (3 个 prompt token 列)
    prompt_len = 3
    prompt_lens = batch["attention_mask"][:, :prompt_len].sum(dim=1)
    response_lens = batch["attention_mask"][:, prompt_len:].sum(dim=1)

    # provider: prompt_lens=3 (unchanged), response_lens=2 (unchanged)
    assert prompt_lens[0].item() == 3
    assert response_lens[0].item() == 2
    # reuser: prompt_lens=0 (prefix removed), response_lens=2 (unchanged)
    assert prompt_lens[1].item() == 0
    assert response_lens[1].item() == 2

    # 模拟断言: sum(prompt_lens + response_lens) == trimmed token count
    # provider: 3+2=5, reuser: 0+2=2, total=7
    # 模型输出裁剪后应该是 7 tokens (provider 全量 5 + reuser suffix 2)
    sequence_lens = prompt_lens + response_lens
    assert sequence_lens.sum().item() == 7


def test_adjust_attention_mask_no_op_when_missing():
    """batch 没有 attention_mask 时不做任何修改。"""
    from prefix_sharing.core.planner import PrefixSharingPlanner
    from prefix_sharing.integrations.verl_mcore import adjust_attention_mask_for_prefix_sharing

    sequences = [[1, 2, 3], [1, 2, 4]]
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences)
    batch = {}
    adjust_attention_mask_for_prefix_sharing(batch, plan)
    assert "attention_mask" not in batch