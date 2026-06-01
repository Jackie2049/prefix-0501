"""CANN/NPU Flash Attention backend for prefix sharing.

This backend targets Huawei Ascend NPU via ``torch_npu``.  The exact
flash-attention API varies across CANN versions, so the implementation is
structured as:

1. Try the native ``torch_npu`` flash-attention path (if available).
2. Fall back to ``TorchReferenceBackend.attention()`` with a loud warning
   so that training can still proceed on NPU while the kernel integration
   is being tuned.

Operators such as ``torch_npu.npu_fusion_attention`` (from MindSpeed) or
``torch_npu.flash_attention`` may be wired in here once the target CANN
version is confirmed.
"""

from __future__ import annotations

import logging
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.flash_common import FlashAttentionMixin, FlashBackendValidationError
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan

_logger = logging.getLogger(__name__)


def _have_torch_npu() -> bool:
    try:
        import torch_npu  # type: ignore[import-untyped]

        _ = torch_npu.npu  # ensure the module is functional
        return True
    except Exception:
        return False


def _npu_flash_attn_varlen(
    q: Any,
    k: Any,
    v: Any,
    cu_seqlens_q: Any,
    cu_seqlens_kv: Any,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> Any:
    """Attempt to run NPU flash attention.

    Tries (in order):
    1. ``torch_npu.flash_attention`` (newer CANN versions).
    2. ``torch_npu.npu_fusion_attention`` (MindSpeed-style wrapper).

    If none of the above is available, raises ``FlashBackendValidationError``
    so that the caller can fall back to the reference path.
    """
    # ------------------------------------------------------------------
    # TODO: adapt the exact API once the target CANN / torch_npu version is
    # pinned.  The signatures below are based on publicly available
    # MindSpeed/ CANN documentation and may need adjustment.
    # ------------------------------------------------------------------

    # Option 1: newer torch_npu unified API
    try:
        import torch_npu  # type: ignore[import-untyped]

        if hasattr(torch_npu, "flash_attention"):
            out = torch_npu.flash_attention(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                causal=True,
            )
            return out
    except Exception:
        pass

    # Option 2: MindSpeed / CANN fused attention
    try:
        import torch_npu  # type: ignore[import-untyped]

        if hasattr(torch_npu, "npu_fusion_attention"):
            # npu_fusion_attention expects (B, N, S, D) by default.
            # For varlen we may need to call a different variant or reshape.
            # The code below is a placeholder and should be replaced with the
            # actual varlen signature once verified.
            raise FlashBackendValidationError(
                "npu_fusion_attention varlen path is not yet integrated. "
                "Please update npu_flash_attn.py with the correct CANN API."
            )
    except FlashBackendValidationError:
        raise
    except Exception:
        pass

    raise FlashBackendValidationError(
        "No NPU flash-attention kernel available. "
        "Checked torch_npu.flash_attention and torch_npu.npu_fusion_attention."
    )


class NpuFlashAttentionBackend(FlashAttentionMixin):
    """Ascend NPU Flash Attention backend.

    ``apply_rope`` and ``build_kv`` are delegated to
    :class:`TorchReferenceBackend`.  ``attention()`` attempts to call an NPU
    flash-attention kernel and falls back to the reference PyTorch path with
    a warning when the kernel is unavailable or fails.
    """

    capabilities = BackendCapabilities(
        name="npu_flash_attn",
        supports_cpu=False,
        supports_cuda=False,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_flash_attention=True,
    )

    def __init__(self, *, strict: bool = False) -> None:
        self._torch_ref = TorchReferenceBackend()
        self._strict = strict
        self._npu_available = _have_torch_npu()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)
        if not self._npu_available:
            if self._strict:
                raise RuntimeError(
                    "NpuFlashAttentionBackend strict mode requires torch_npu, "
                    "but it is not installed or functional."
                )
            _logger.warning(
                "torch_npu is not available; NpuFlashAttentionBackend will "
                "fall back to TorchReferenceBackend.attention() on NPU."
            )

    # ------------------------------------------------------------------
    # RoPE & KV build: reuse the reference implementation
    # ------------------------------------------------------------------
    def apply_rope(
        self,
        query: Any,
        key: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        return self._torch_ref.apply_rope(query, key, prefix_sharing_plan, **kwargs)

    def build_kv(
        self,
        key: Any,
        value: Any,
        store: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        layer_id: int,
        tp_rank: int = 0,
    ) -> tuple[Any, Any]:
        return self._torch_ref.build_kv(
            key, value, store, prefix_sharing_plan, layer_id=layer_id, tp_rank=tp_rank
        )

    # ------------------------------------------------------------------
    # Attention: try NPU kernel, fall back to reference
    # ------------------------------------------------------------------
    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        if not self._npu_available:
            _logger.warning(
                "NPU flash attention kernel unavailable, falling back to "
                "TorchReferenceBackend.attention()"
            )
            return self._torch_ref.attention(query, key, value, prefix_sharing_plan, **kwargs)

        q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = (
            self._prepare_flash_inputs(
                query, key, value, prefix_sharing_plan, attention_mask=kwargs.get("attention_mask")
            )
        )

        try:
            out = _npu_flash_attn_varlen(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
            )
            return out
        except FlashBackendValidationError:
            if self._strict:
                raise
            _logger.warning(
                "NPU flash attention kernel failed, falling back to "
                "TorchReferenceBackend.attention().",
                exc_info=True,
            )
            return self._torch_ref.attention(query, key, value, prefix_sharing_plan, **kwargs)
