"""CANN/NPU Flash Attention backend for prefix sharing.

Uses MindSpeed's ``npu_fusion_attention`` fused kernel. Unlike the upstream
``flash_attn`` package, the NPU op accepts an explicit ``atten_mask`` (a
BoolTensor where True = masked) and ``actual_seq_qlen`` / ``actual_seq_kvlen``
Python lists for the THD layout.

.. important::

   In TND mode the kernel requires ``atten_mask`` of shape
   ``(max_seqlen_q, max_seqlen_kv)`` and applies the **same** mask to every
   sample.  This means NPU FA can express standard causal masking (all
   providers), but cannot express per-sample-varying prefix patterns when
   reusers are present.  For plans that contain any reuser, this backend
   **falls back** to :class:`TorchReferenceBackend` for the attention step.
   ``apply_rope`` and ``build_kv`` are always delegated to the reference
   backend regardless.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.flash_atten_base import (
    FlashAttentionMixin,
    FlashBackendValidationError,
)
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan

def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("NpuFlashAttentionBackend requires PyTorch") from exc
    return torch

@lru_cache(maxsize=None)
def _import_npu_fusion_attention() -> Any:
    try:
        from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "NpuFlashAttentionBackend requires MindSpeed (mindspeed.ops.fusion_attention_v2). "
            "Install MindSpeed matching your CANN version."
        ) from exc
    return npu_fusion_attention

def _build_causal_bool_mask(max_q: int, max_kv: int, device: Any) -> Any:
    """Build a ``(max_q, max_kv)`` causal boolean mask for the NPU kernel.

    The returned tensor uses ``True = masked (invisible)``, matching
    ``npu_fusion_attention``'s ``atten_mask`` convention.  The mask is
    lower-triangular: ``Q[i]`` can attend to ``KV[0..i]``.
    """
    torch = _torch()
    eye = torch.ones(max_q, max_kv, dtype=torch.bool, device=device)
    tri = torch.tril(eye, diagonal=0)
    return ~tri

class NpuFlashAttentionBackend(FlashAttentionMixin):
    """Ascend NPU backend via ``npu_fusion_attention`` (TND layout).

    Provider-only plans are executed on the NPU fused kernel.  Plans that
    contain any reuser fall back to :class:`TorchReferenceBackend` because
    the kernel's per-sample-shared ``atten_mask`` cannot express varying
    prefix-visibility patterns.
    """

    capabilities = BackendCapabilities(
        name="flash_atten_npu",
        supports_cpu=False,
        supports_cuda=False,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_gated_attention=False,
        supports_deltanet_state_reuse=False,
    )

    def __init__(self) -> None:
        self._torch_ref = TorchReferenceBackend()

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)
        _import_npu_fusion_attention()

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
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
    ) -> tuple[Any, Any]:
        return self._torch_ref.build_kv(
            key,
            value,
            store,
            prefix_sharing_plan,
            packed_batch_layout=packed_batch_layout,
            layer_id=layer_id,
            tp_rank=tp_rank,
        )

    # ------------------------------------------------------------------
    # Attention dispatch
    # ------------------------------------------------------------------
    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        q, k, v, _cu_q, _cu_kv, _max_q, _max_kv, pad_layout = self._prepare_flash_inputs(
            query, key, value, prefix_sharing_plan,
            attention_mask=kwargs.get("attention_mask"),
            packed_batch_layout=packed_batch_layout,
        )

        has_reuser = any(
            prefix_sharing_plan.is_reuser(i)
            for i in range(prefix_sharing_plan.batch_size)
        )

        if has_reuser:
            # NPU kernel's TND atten_mask is (max_q, max_kv) shared across
            # all samples.  It cannot express per-sample-varying prefix
            # patterns — fall back to the reference implementation.
            out = self._attention_via_torch_ref(q, k, v, prefix_sharing_plan, **kwargs)
        else:
            out = self._attention_via_npu(q, k, v, prefix_sharing_plan, **kwargs)

        if pad_layout is not None:
            out = self._repad_output(out, pad_layout)

        return out

    # ------------------------------------------------------------------
    # NPU fast path (provider-only)
    # ------------------------------------------------------------------
    def _attention_via_npu(
        self,
        q: Any,
        k: Any,
        v: Any,
        plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        """Call npu_fusion_attention with a causal atten_mask.

        Only valid when *plan* contains no reusers, because
        ``atten_mask`` of shape ``(max_q, max_kv)`` is shared across all
        samples and a simple causal pattern is only correct for providers.
        """
        npu_fusion_attention = _import_npu_fusion_attention()

        # atten_mask shape must be (max_seqlen_q, max_seqlen_kv), NOT
        # (total_q, total_kv).  The kernel broadcasts this single mask
        # to every sample.
        atten_mask = _build_causal_bool_mask(
            plan.max_seqlen_q,
            plan.max_seqlen_kv,
            device=q.device,
        )

        num_heads = q.shape[1]
        # Per-sample lengths (NOT cumulative).
        actual_seq_qlen = list(plan.kept_lengths_q)
        actual_seq_kvlen = list(plan.expanded_lengths_kv)

        head_dim = q.shape[-1]
        scale = kwargs.get("softmax_scale") or (1.0 / math.sqrt(head_dim))
        dropout_p = kwargs.get("dropout_p", 0.0)
        keep_prob = kwargs.get("keep_prob", 1.0 - dropout_p)

        try:
            result = npu_fusion_attention(
                q,
                k,
                v,
                num_heads,
                "TND",
                atten_mask=atten_mask,
                scale=scale,
                keep_prob=keep_prob,
                sparse_mode=0,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )
        except Exception as exc:
            raise FlashBackendValidationError(
                f"npu_fusion_attention failed: q={tuple(q.shape)}, k={tuple(k.shape)}, "
                f"max_q={plan.max_seqlen_q}, max_kv={plan.max_seqlen_kv}"
            ) from exc

        # The op returns a tuple of 7 elements; [0] is the attention output.
        if isinstance(result, (tuple, list)):
            return result[0]
        return result

    # ------------------------------------------------------------------
    # Torch fallback (reuser plans)
    # ------------------------------------------------------------------
    def _attention_via_torch_ref(
        self,
        q: Any,
        k: Any,
        v: Any,
        plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        """Fall back to :class:`TorchReferenceBackend` for plans with reusers.

        The NPU kernel cannot express per-sample-varying prefix masks, so
        we delegate to the pure-PyTorch reference implementation which
        correctly handles absolute-position causal masking.
        """
        return self._torch_ref.attention(q, k, v, plan, **kwargs)
