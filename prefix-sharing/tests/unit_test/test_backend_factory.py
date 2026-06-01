"""Unit tests for the backend factory."""

from __future__ import annotations

import pytest

from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.gpu_flash_attn import GpuFlashAttentionBackend
from prefix_sharing.backends.npu_flash_attn import NpuFlashAttentionBackend
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig


def test_factory_torch_ref() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="torch_ref")
    backend = get_backend_instance(config)
    assert isinstance(backend, TorchReferenceBackend)
    assert backend.capabilities.name == "torch_ref"


def test_factory_gpu_flash_attn() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="gpu_flash_attn")
    backend = get_backend_instance(config)
    assert isinstance(backend, GpuFlashAttentionBackend)
    assert backend.capabilities.name == "gpu_flash_attn"
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
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="gpu_flash_attn")
    resolved = get_backend_instance(config)
    assert isinstance(resolved, GpuFlashAttentionBackend)


def test_config_validates_backends() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="gpu_flash_attn")
    config.validate()

    bad_config = PrefixSharingConfig(enable_prefix_sharing=True, backend="not_real")
    with pytest.raises(Exception):
        bad_config.validate()
