"""Backend factory: instantiate a concrete backend from config string."""

from __future__ import annotations

from typing import Any

from prefix_sharing.backends.base import PrefixAttentionBackend
from prefix_sharing.core.config import PrefixSharingConfig


def get_backend_instance(
    config: PrefixSharingConfig, backend: Any | None = None
) -> PrefixAttentionBackend:
    """Return an explicit ``backend`` if given, otherwise build from ``config``.

    Supported values for ``config.backend``:
    * ``"torch_ref"``      -> :class:`~prefix_sharing.backends.torch_ref.TorchReferenceBackend`
    * ``"gpu_flash_attn"`` -> :class:`~prefix_sharing.backends.gpu_flash_attn.GpuFlashAttentionBackend`
    * ``"npu_flash_attn"`` -> :class:`~prefix_sharing.backends.npu_flash_attn.NpuFlashAttentionBackend`
    """
    if backend is not None:
        return backend
    if config.backend == "torch_ref":
        from prefix_sharing.backends.torch_ref import TorchReferenceBackend

        return TorchReferenceBackend()
    if config.backend == "gpu_flash_attn":
        from prefix_sharing.backends.gpu_flash_attn import GpuFlashAttentionBackend

        return GpuFlashAttentionBackend()
    if config.backend == "npu_flash_attn":
        from prefix_sharing.backends.npu_flash_attn import NpuFlashAttentionBackend

        return NpuFlashAttentionBackend()
    raise ValueError(
        f"Unknown backend '{config.backend}'. "
        f"Supported: torch_ref, gpu_flash_attn, npu_flash_attn"
    )
