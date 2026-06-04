"""Unit tests for the backend factory."""

from __future__ import annotations

import pytest

from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig


def test_factory_torch_ref() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="torch_ref")
    backend = get_backend_instance(config)
    assert isinstance(backend, TorchReferenceBackend)
    assert backend.capabilities.name == "torch_ref"


def test_factory_flash_atten_gpu() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_gpu")
    backend = get_backend_instance(config)
    assert isinstance(backend, GpuFlashAttentionBackend)
    assert backend.capabilities.name == "flash_atten_gpu"
    assert backend.capabilities.supports_flash_attention


def test_factory_unknown_backend() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="unknown")
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend_instance(config)


def test_resolve_backend_explicit() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="torch_ref")
    explicit = GpuFlashAttentionBackend()
    resolved = get_backend_instance(config, explicit)
    assert resolved is explicit


def test_resolve_backend_from_config() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_gpu")
    resolved = get_backend_instance(config)
    assert isinstance(resolved, GpuFlashAttentionBackend)


def test_config_validates_backends() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_gpu")
    config.validate()

    bad_config = PrefixSharingConfig(enable_prefix_sharing=True, backend="not_real")
    with pytest.raises(Exception):
        bad_config.validate()


def test_factory_falls_back_to_torch_ref_when_flash_attn_not_importable(monkeypatch):
    """When flash_atten_gpu module import fails, factory falls back to torch_ref."""
    # Simulate the flash_atten_gpu module being unloadable by replacing it
    import sys
    from types import ModuleType

    # Create a module that raises on attribute access to simulate broken import
    broken_mod = ModuleType("prefix_sharing.backends.flash_atten_gpu")

    def broken_getattr(name):
        raise ImportError("flash_attn not available")

    broken_mod.__getattr__ = broken_getattr
    monkeypatch.setitem(sys.modules, "prefix_sharing.backends.flash_atten_gpu", broken_mod)

    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_gpu")
    backend = get_backend_instance(config)
    assert isinstance(backend, TorchReferenceBackend)
