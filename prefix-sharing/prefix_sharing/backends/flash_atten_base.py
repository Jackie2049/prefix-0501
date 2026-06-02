"""Flash Attention common abstraction for prefix-sharing backends.

GPU (Transformer Engine) and NPU (MindSpeed ``npu_fusion_attention``) share
the same input-normalisation logic: convert the packed THD tensors emitted by
the Megatron hook into the shape/dtype/device layout their kernels expect, and
validate that the plan carries the required cumulative-seqlens metadata.

Both concrete backends consume the boolean block-causal mask produced by
:func:`prefix_sharing.backends.block_causal_mask.build_block_causal_mask`.
"""

from __future__ import annotations

from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
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
    normalises tensor shapes for the ``attention()`` step.
    """

    capabilities: BackendCapabilities

    def _validate_plan_for_flash(self, prefix_sharing_plan: PrefixSharingPlan) -> None:
        plan = prefix_sharing_plan
        if plan.cu_seqlens_q is None or plan.cu_seqlens_kv is None:
            raise FlashBackendValidationError(
                "Flash Attention requires cu_seqlens_q and cu_seqlens_kv; "
                "got None. Make sure PrefixSharingPlan was built with packed THD metadata."
            )
        if len(plan.cu_seqlens_q) != plan.batch_size + 1:
            raise FlashBackendValidationError(
                f"cu_seqlens_q length ({len(plan.cu_seqlens_q)}) must be "
                f"batch_size + 1 ({plan.batch_size + 1})"
            )
        if len(plan.cu_seqlens_kv) != plan.batch_size + 1:
            raise FlashBackendValidationError(
                f"cu_seqlens_kv length ({len(plan.cu_seqlens_kv)}) must be "
                f"batch_size + 1 ({plan.batch_size + 1})"
            )

    def _ensure_3d_thd(self, tensor: Any, name: str) -> Any:
        """Ensure packed tensor is ``(total_tokens, num_heads, head_dim)``."""
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

    def _build_cu_seqlens_tensor(
        self, lengths: list[int], device: Any, dtype: Any = None
    ) -> Any:
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
    ) -> tuple[Any, Any, Any, Any, Any, int, int]:
        """Normalise inputs and return THD tensors + cumulative seqlens."""
        torch = _torch()
        self._validate_plan_for_flash(prefix_sharing_plan)

        q = self._ensure_3d_thd(query, "query")
        k = self._ensure_3d_thd(key, "key")
        v = self._ensure_3d_thd(value, "value")

        device = q.device
        cu_seqlens_q = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_q, device=device, dtype=torch.int32
        )
        cu_seqlens_kv = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_kv, device=device, dtype=torch.int32
        )
        return (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            prefix_sharing_plan.max_seqlen_q,
            prefix_sharing_plan.max_seqlen_kv,
        )
