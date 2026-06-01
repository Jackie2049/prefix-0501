"""Flash Attention common abstraction layer.

This module provides shared input-normalization logic for GPU and NPU Flash
Attention backends.  It converts the packed THD tensors produced by the
Megatron hook into the shape/dtype/device layout expected by Flash Attention
kernels, while keeping hardware-specific dispatch in the concrete backends.
"""

from __future__ import annotations

from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


class FlashBackendValidationError(RuntimeError):
    """Raised when a flash-attention backend cannot run with the given plan."""


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Flash Attention backends require PyTorch") from exc
    return torch


class FlashAttentionMixin:
    """Shared helpers for GPU / NPU Flash Attention backends.

    Concrete subclasses must still define ``capabilities``, ``validate()``,
    ``apply_rope()``, ``build_kv()`` and ``attention()``; this mixin only
    normalises the tensor shapes for the ``attention()`` step.
    """

    capabilities: BackendCapabilities

    # ------------------------------------------------------------------
    # Shared validation
    # ------------------------------------------------------------------
    def _validate_plan_for_flash(self, prefix_sharing_plan: PrefixSharingPlan) -> None:
        """Common checks before calling any Flash Attention kernel."""
        if prefix_sharing_plan.cu_seqlens_q is None or prefix_sharing_plan.cu_seqlens_kv is None:
            raise FlashBackendValidationError(
                "Flash Attention requires cu_seqlens_q and cu_seqlens_kv; "
                "got None. Make sure PrefixSharingPlan was built with packed THD metadata."
            )
        if len(prefix_sharing_plan.cu_seqlens_q) != prefix_sharing_plan.batch_size + 1:
            raise FlashBackendValidationError(
                f"cu_seqlens_q length ({len(prefix_sharing_plan.cu_seqlens_q)}) must be "
                f"batch_size + 1 ({prefix_sharing_plan.batch_size + 1})"
            )
        if len(prefix_sharing_plan.cu_seqlens_kv) != prefix_sharing_plan.batch_size + 1:
            raise FlashBackendValidationError(
                f"cu_seqlens_kv length ({len(prefix_sharing_plan.cu_seqlens_kv)}) must be "
                f"batch_size + 1 ({prefix_sharing_plan.batch_size + 1})"
            )

    # ------------------------------------------------------------------
    # Tensor normalisation helpers
    # ------------------------------------------------------------------
    def _ensure_3d_thd(self, tensor: Any, name: str) -> Any:
        """Ensure packed tensor is (total_tokens, num_heads, head_dim).

        The Megatron hook squeezes the dummy batch dimension before handing
        tensors to the backend, so we expect 3-D inputs.  If 2-D is ever
        received we raise loudily rather than guessing.
        """
        torch = _torch()
        if tensor.dim() == 3:
            return tensor
        if tensor.dim() == 2:
            raise FlashBackendValidationError(
                f"{name} has 2 dims {tuple(tensor.shape)}; Flash Attention expects "
                "(total_tokens, num_heads, head_dim)."
            )
        raise FlashBackendValidationError(
            f"{name} has unexpected rank {tensor.dim()} with shape {tuple(tensor.shape)}"
        )

    def _build_cu_seqlens_tensor(self, lengths: list[int], device: Any, dtype: Any = None) -> Any:
        """Convert a Python list of cumulative lengths to a CUDA/NPU tensor."""
        torch = _torch()
        t = torch.tensor(lengths, device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t

    def _prepare_flash_inputs(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        attention_mask: Any | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, int, int]:
        """Normalise inputs for ``flash_attn_varlen_func``.

        Returns
        -------
        q, k, v : (total_tokens, num_heads, head_dim)
        cu_seqlens_q : (batch_size + 1,)
        cu_seqlens_kv : (batch_size + 1,)
        max_seqlen_q : int
        max_seqlen_kv : int
        """
        torch = _torch()
        self._validate_plan_for_flash(prefix_sharing_plan)

        q = self._ensure_3d_thd(query, "query")
        k = self._ensure_3d_thd(key, "key")
        v = self._ensure_3d_thd(value, "value")

        device = q.device

        # cu_seqlens must be int32 on the same device for Flash Attention
        cu_seqlens_q = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_q, device=device, dtype=torch.int32
        )
        cu_seqlens_kv = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_kv, device=device, dtype=torch.int32
        )

        max_seqlen_q = prefix_sharing_plan.max_seqlen_q
        max_seqlen_kv = prefix_sharing_plan.max_seqlen_kv

        # Flash Attention ignores dense attention_mask; if one is passed we
        # simply drop it because cu_seqlens already encodes the per-sample
        # boundaries.  This is consistent with Megatron's THD path.
        _ = attention_mask

        return q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
