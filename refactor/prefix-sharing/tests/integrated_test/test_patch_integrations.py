import pytest

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.megatron_attention import (
    IntegrationUnavailable,
    MegatronAttentionIntegration,
)
from prefix_sharing.integrations.patch_manager import PatchManager
from prefix_sharing.integrations.verl_mcore import VerlMCoreIntegration


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
