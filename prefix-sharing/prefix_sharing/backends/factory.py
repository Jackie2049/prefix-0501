"""Backend factory: instantiate a concrete backend from config string."""

from __future__ import annotations

import logging
from typing import Any

from prefix_sharing.backends.base import PrefixAttentionBackend
from prefix_sharing.core.config import PrefixSharingConfig

logger = logging.getLogger(__name__)


def get_backend_instance(
    config: PrefixSharingConfig, backend: Any | None = None
) -> PrefixAttentionBackend:
    """Return an explicit ``backend`` if given, otherwise build from ``config``.

    Supported values for ``config.backend``:
    * ``"torch_ref"``      -> :class:`~prefix_sharing.backends.torch_ref.TorchReferenceBackend`
    * ``"flash_atten_gpu"`` -> :class:`~prefix_sharing.backends.flash_atten_gpu.GpuFlashAttentionBackend`
    * ``"flash_atten_npu"`` -> :class:`~prefix_sharing.backends.flash_atten_npu.NpuFlashAttentionBackend`

    If the requested backend cannot be imported (e.g. flash-attn not installed),
    falls back to ``torch_ref`` with a warning.
    """
    if backend is not None:
        return backend

    if config.backend == "torch_ref":
        from prefix_sharing.backends.torch_ref import TorchReferenceBackend
        return TorchReferenceBackend()

    if config.backend == "flash_atten_gpu":
        try:
            from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
            return GpuFlashAttentionBackend()
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "flash_atten_gpu backend unavailable (%s), falling back to torch_ref. "
                "Install flash-attn for optimal performance.",
                exc,
            )
            from prefix_sharing.backends.torch_ref import TorchReferenceBackend
            return TorchReferenceBackend()

    if config.backend == "flash_atten_npu":
        try:
            from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
            return NpuFlashAttentionBackend()
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "flash_atten_npu backend unavailable (%s), falling back to torch_ref.",
                exc,
            )
            from prefix_sharing.backends.torch_ref import TorchReferenceBackend
            return TorchReferenceBackend()

    raise ValueError(
        f"Unknown backend '{config.backend}'. "
        f"Supported: torch_ref, flash_atten_gpu, flash_atten_npu"
    )
