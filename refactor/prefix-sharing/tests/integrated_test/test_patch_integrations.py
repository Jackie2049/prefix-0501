import pytest

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
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


def test_patch_manager_context_manager_restores_original():
    target = Target()
    manager = PatchManager()
    manager.patch_attr(Target, "method", lambda instance: "patched")

    with manager.handle():
        assert target.method() == "patched"
    assert target.method() == "original"


def test_megatron_integration_reports_missing_dependency_cleanly():
    config = PrefixSharingConfig(enabled=True)
    integration = MegatronAttentionIntegration(config=config, backend=TorchReferenceBackend())
    with pytest.raises(IntegrationUnavailable, match="Megatron"):
        integration.install(model_config={})


def test_verl_integration_reports_missing_dependency_cleanly():
    config = PrefixSharingConfig(enabled=True)
    integration = VerlMCoreIntegration(config=config)
    with pytest.raises(IntegrationUnavailable, match="verl"):
        integration.install(model_config={})


def test_verl_mcore_batch_adapter_uses_mapping_for_preprocess_and_restore():
    adapter = VerlMCoreBatchAdapter(
        PrefixSharingConfig(enabled=True, min_prefix_len=3, min_group_size=2)
    )
    prepared = adapter.prepare_micro_batch(
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

    assert [(s.reuse_idx_in_batch, s.provider_idx_in_batch, s.prefix_len) for s in prepared.meta.reuse_specs] == [
        (1, 0, 3),
    ]
    assert prepared.input_ids.rows == [[1, 2, 3, 10, 11], [20, 21, 22], [9, 9]]
    assert prepared.labels is not None
    assert prepared.labels.rows == [["p0", "p1", "p2", "a", "b"], ["x", "y", "z"], ["u", "v"]]
    assert prepared.loss_masks is not None
    assert prepared.loss_masks.rows == [[1, 1, 1, 1, 1], [1, 1, 1], [1, 1]]

    with adapter.prepared_context(prepared) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.meta is prepared.meta
    assert current_prefix_sharing_context() is None

    restored = adapter.restore_logprobs(
        [[-1.0, -1.1, -1.2, -1.3, -1.4], [-2.1, -2.2], [-9.0, -9.1]],
        [0.0, -2.0, 0.0],
        prepared.meta,
    )
    assert restored == [
        [-1.0, -1.1, -1.2, -1.3, -1.4],
        [-2.0, -2.1, -2.2],
        [-9.0, -9.1],
    ]


def test_prefix_sharing_enabled_propagates_install_failure(monkeypatch):
    class FakeIntegration:
        def __init__(self, config, backend=None):
            pass

        def install(self, model_config=None):
            raise RuntimeError("install failed")

    monkeypatch.setattr("prefix_sharing.integrations.verl_mcore.VerlMCoreIntegration", FakeIntegration)
    with pytest.raises(RuntimeError, match="install failed"):
        with prefix_sharing_enabled(PrefixSharingConfig(enabled=True)):
            pass
