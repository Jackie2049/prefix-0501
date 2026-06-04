import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.integrations.megatron_attention import (
    IntegrationUnavailable,
    MegatronAttentionIntegration,
)
from prefix_sharing.integrations.patch_manager import PatchManager
from prefix_sharing.integrations.verl_mcore import (
    VerlMCoreBatchAdapter,
    VerlMCoreIntegration,
    prefix_sharing_enabled,
)


class Target:
    def method(self):
        return "original"


def test_patch_manager_installs_and_disables_patch():
    target = Target()
    manager = PatchManager()

    def replacement(instance):
        return "patched"

    manager.patch_attr(Target, "method", replacement)
    handle = manager.handle()
    assert target.method() == "patched"
    assert handle.active

    handle.disable()
    assert target.method() == "original"
    assert not handle.active


def test_patch_manager_prevents_double_install():
    """Double-installing the same patch on the same attr raises RuntimeError."""
    manager1 = PatchManager()
    manager1.patch_attr(Target, "method", lambda instance: "patched1")
    handle1 = manager1.handle()

    try:
        manager2 = PatchManager()
        with pytest.raises(RuntimeError, match="already patched"):
            manager2.patch_attr(Target, "method", lambda instance: "patched2")
    finally:
        handle1.disable()


def test_patch_manager_allows_reinstall_after_disable():
    """After disabling, the same attr can be patched again."""
    manager1 = PatchManager()
    manager1.patch_attr(Target, "method", lambda instance: "patched1")
    handle1 = manager1.handle()
    handle1.disable()

    manager2 = PatchManager()
    manager2.patch_attr(Target, "method", lambda instance: "patched2")
    handle2 = manager2.handle()
    assert Target().method() == "patched2"
    handle2.disable()


def test_patch_manager_context_manager_restores_original():
    target = Target()
    manager = PatchManager()
    manager.patch_attr(Target, "method", lambda instance: "patched")

    with manager.handle():
        assert target.method() == "patched"
    assert target.method() == "original"


def test_megatron_integration_reports_missing_dependency_cleanly():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = MegatronAttentionIntegration(config=config, backend=TorchReferenceBackend())
    with pytest.raises(IntegrationUnavailable, match="Megatron"):
        integration.install(model_config={})


def test_verl_integration_reports_missing_dependency_cleanly():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = VerlMCoreIntegration(config=config)
    with pytest.raises(IntegrationUnavailable, match="verl"):
        integration.install(model_config={})


def test_verl_mcore_batch_adapter_uses_mapping_for_preprocess_and_restore():
    adapter = VerlMCoreBatchAdapter(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    )
    prefix_sharing_batch = adapter.prepare_micro_batch(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22], [9, 9]],
        labels=[
            ["p0", "p1", "p2", "a", "b"],
            ["p0", "p1", "p2", "x", "y", "z"],
            ["u", "v"],
        ],
        loss_masks=[
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1],
        ],
        forward_id=7,
        micro_batch_id=8,
    )

    assert [(s.reuse_idx_in_batch, s.provider_idx_in_batch, s.prefix_len) for s in prefix_sharing_batch.prefix_sharing_plan.reuse_specs] == [
        (1, 0, 3),
    ]
    assert prefix_sharing_batch.input_ids.rows == [[1, 2, 3, 10, 11], [20, 21, 22], [9, 9]]
    assert prefix_sharing_batch.labels is not None
    assert prefix_sharing_batch.labels.rows == [["p0", "p1", "p2", "a", "b"], ["x", "y", "z"], ["u", "v"]]
    assert prefix_sharing_batch.loss_masks is not None
    assert prefix_sharing_batch.loss_masks.rows == [[1, 1, 1, 1, 1], [1, 1, 1], [1, 1]]

    with adapter.prefix_sharing_runtime_context(prefix_sharing_batch) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.prefix_sharing_plan is prefix_sharing_batch.prefix_sharing_plan
    assert current_prefix_sharing_context() is None

    restored = adapter.restore_logprobs(
        [[-1.0, -1.1, -1.2, -1.3, -1.4], [-2.1, -2.2], [-9.0, -9.1]],
        [0.0, -2.0, 0.0],
        prefix_sharing_batch.prefix_sharing_plan,
    )
    assert restored == [
        [-1.0, -1.1, -1.2, -1.3, -1.4],
        [-2.0, -2.1, -2.2],
        [-9.0, -9.1],
    ]


def test_prefix_sharing_config_from_raw_accepts_nested_config():
    config = PrefixSharingConfig.from_raw(
        {
            "enable_prefix_sharing": True,
            "min_prefix_len": 4,
            "min_group_size": 3,
            "boundary_strategy": "prefix_last_restore",
        }
    )

    assert config.enable_prefix_sharing is True
    assert config.min_prefix_len == 4
    assert config.min_group_size == 3


def test_prefix_sharing_config_from_raw_rejects_legacy_enabled_key():
    with pytest.raises(TypeError, match="enabled"):
        PrefixSharingConfig.from_raw({"enabled": True})


def test_prefix_sharing_config_from_raw_accepts_env_enable(monkeypatch):
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", "1")

    config = PrefixSharingConfig.from_raw(None)

    assert config.enable_prefix_sharing is True


def test_prefix_sharing_enabled_propagates_install_failure(monkeypatch):
    class FakeIntegration:
        def __init__(self, config, backend=None):
            pass

        def install(self, model_config=None):
            raise RuntimeError("install failed")

    monkeypatch.setattr("prefix_sharing.integrations.verl_mcore.VerlMCoreIntegration", FakeIntegration)
    with pytest.raises(RuntimeError, match="install failed"):
        with prefix_sharing_enabled(PrefixSharingConfig(enable_prefix_sharing=True)):
            pass


def test_megatron_patch_installs_forward_wrapper(monkeypatch):
    """Verify that install() actually replaces SelfAttention.forward."""
    # Create a fake megatron.core.transformer.attention module
    fake_attention = ModuleType("megatron.core.transformer.attention")

    original_forward_calls = []

    class FakeSelfAttention:
        def forward(self, hidden_states, **kwargs):
            original_forward_calls.append(True)
            return "original_output"

    FakeSelfAttention.forward = FakeSelfAttention.forward
    fake_attention.SelfAttention = FakeSelfAttention
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.attention", fake_attention)
    # Also need the parent modules
    for parent in ["megatron", "megatron.core", "megatron.core.transformer"]:
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, ModuleType(parent))

    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = MegatronAttentionIntegration(config=config, backend=TorchReferenceBackend())
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
    )
    handle = integration.install(model_config=model_config)

    try:
        # Without a prefix-sharing context, the patched forward should delegate
        # to the original (which appends to original_forward_calls)
        instance = FakeSelfAttention()
        result = instance.forward(hidden_states="fake", attention_mask=None)
        assert result == "original_output"
        assert len(original_forward_calls) == 1
    finally:
        handle.disable()

    # After disabling, the original forward is restored
    original_forward_calls.clear()
    instance2 = FakeSelfAttention()
    result2 = instance2.forward(hidden_states="fake", attention_mask=None)
    assert result2 == "original_output"
    assert len(original_forward_calls) == 1

    # Verify handle is no longer active
    assert not handle.active


def test_megatron_patch_falls_through_for_cross_attention(monkeypatch):
    """Verify that key_value_states (cross-attention) falls through to original."""
    fake_attention = ModuleType("megatron.core.transformer.attention")

    original_forward_calls = []

    class FakeSelfAttention:
        def forward(self, hidden_states, **kwargs):
            original_forward_calls.append(kwargs.get("key_value_states"))
            return "original_output"

    FakeSelfAttention.forward = FakeSelfAttention.forward
    fake_attention.SelfAttention = FakeSelfAttention
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.attention", fake_attention)
    for parent in ["megatron", "megatron.core", "megatron.core.transformer"]:
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, ModuleType(parent))

    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = MegatronAttentionIntegration(config=config, backend=TorchReferenceBackend())
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
    )
    handle = integration.install(model_config=model_config)

    try:
        # Set up a prefix-sharing context so the cross-attention guard is tested
        planner = PrefixSharingPlanner(
            PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
        )
        plan = planner.plan([[1, 2, 3, 10], [1, 2, 3, 20]])

        from prefix_sharing.backends.packed_layout import PackedBatchLayout
        from prefix_sharing.integrations.context import prefix_sharing_runtime_context
        from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState

        state = PrefixSharingRuntimeState(
            prefix_sharing_plan=plan,
            backend=TorchReferenceBackend(),
            packed_batch_layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q),
        )

        with prefix_sharing_runtime_context(state):
            instance = FakeSelfAttention()
            # key_value_states triggers cross-attention fallback
            result = instance.forward(
                hidden_states="fake", attention_mask=None,
                key_value_states="cross_kv",
            )
            assert result == "original_output"
            assert len(original_forward_calls) == 1
            assert original_forward_calls[0] == "cross_kv"

        # Also test inference_params fallback (without context now)
        original_forward_calls.clear()
        instance2 = FakeSelfAttention()
        result2 = instance2.forward(
            hidden_states="fake", attention_mask=None,
            inference_params="inf_params",
        )
        assert result2 == "original_output"
        assert len(original_forward_calls) == 1
    finally:
        handle.disable()


def test_megatron_patch_uses_get_query_key_value_tensors(monkeypatch):
    """Verify the patched forward calls get_query_key_value_tensors for QKV split."""
    pytest.importorskip("torch")
    import torch

    from prefix_sharing.backends.packed_layout import PackedBatchLayout
    from prefix_sharing.core.planner import PrefixSharingPlanner
    from prefix_sharing.integrations.context import prefix_sharing_runtime_context
    from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState

    # Create a fake megatron module with get_query_key_value_tensors
    fake_attention = ModuleType("megatron.core.transformer.attention")

    qkv_calls = []

    class FakeSelfAttention:
        def __init__(self):
            self.config = SimpleNamespace(
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
            )

        def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
            qkv_calls.append(hidden_states)
            # Return [total_tokens, num_heads, head_dim] tensors
            n_tokens = hidden_states.shape[0]
            q = torch.randn(n_tokens, 4, 8)
            k = torch.randn(n_tokens, 2, 8)
            v = torch.randn(n_tokens, 2, 8)
            return q, k, v

        def linear_proj(self, x):
            return torch.randn(x.shape[0], 4 * 8)

    class FakeOriginal:
        """Track calls to the original forward."""
        calls = 0

        @classmethod
        def __call__(cls, *args, **kwargs):
            cls.calls += 1
            return "original"

    fake_attention.SelfAttention = FakeSelfAttention
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.attention", fake_attention)
    for parent in ["megatron", "megatron.core", "megatron.core.transformer"]:
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, ModuleType(parent))

    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = MegatronAttentionIntegration(config=config, backend=TorchReferenceBackend())
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
    )
    handle = integration.install(model_config=model_config)

    try:
        # Create a prefix-sharing context
        planner = PrefixSharingPlanner(
            PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
        )
        plan = planner.plan([[1, 2, 3, 10], [1, 2, 3, 20]])
        state = PrefixSharingRuntimeState(
            prefix_sharing_plan=plan,
            backend=TorchReferenceBackend(),
            packed_batch_layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q),
        )

        instance = FakeSelfAttention()
        # Fake packed_seq_params with qkv_format='thd' is needed
        # But without a full Megatron environment, the hook will hit errors.
        # We just verify get_query_key_value_tensors was called.
        with prefix_sharing_runtime_context(state) as ctx:
            assert ctx is not None
            # The patched forward will call get_query_key_value_tensors
            # then fail on Megatron-specific RoPE calls - that's expected.
            # What matters is qkv_calls was populated.
            fake_input = torch.randn(5, 32)
            try:
                instance.forward(hidden_states=fake_input, attention_mask=None,
                                 packed_seq_params=None, rotary_pos_emb=None)
            except (RuntimeError, AttributeError, TypeError):
                pass  # Expected - missing Megatron runtime deps

        # Verify get_query_key_value_tensors was called
        assert len(qkv_calls) >= 1, "get_query_key_value_tensors should have been called"
    finally:
        handle.disable()
