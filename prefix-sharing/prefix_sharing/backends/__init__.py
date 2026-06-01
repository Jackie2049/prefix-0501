"""Attention backend adapters."""

from prefix_sharing.backends.base import BackendCapabilities, PrefixAttentionBackend
from prefix_sharing.backends.factory import get_backend_instance, resolve_backend
from prefix_sharing.backends.flash_common import FlashAttentionMixin
from prefix_sharing.backends.gpu_flash_attn import GpuFlashAttentionBackend
from prefix_sharing.backends.npu_flash_attn import NpuFlashAttentionBackend
from prefix_sharing.backends.torch_ref import TorchReferenceBackend

__all__ = [
    "BackendCapabilities",
    "FlashAttentionMixin",
    "get_backend_instance",
    "GpuFlashAttentionBackend",
    "NpuFlashAttentionBackend",
    "PrefixAttentionBackend",
    "resolve_backend",
    "TorchReferenceBackend",
]
