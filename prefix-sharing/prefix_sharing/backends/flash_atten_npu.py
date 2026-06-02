"""CANN/NPU Flash Attention backend for prefix sharing.

Uses MindSpeed's ``npu_fusion_attention`` fused kernel. Unlike the upstream
``flash_attn`` package, the NPU op accepts an explicit ``atten_mask`` (a
BoolTensor where True = masked) and ``actual_seq_qlen`` / ``actual_seq_kvlen``
Python lists for the THD layout — exactly what we need to express the
absolute-position block-causal mask required by reusers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.block_causal_mask import build_block_causal_mask
from prefix_sharing.backends.flash_atten_base import (
    FlashAttentionMixin,
    FlashBackendValidationError,
)
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


@lru_cache(maxsize=None)
def _import_npu_fusion_attention() -> Any:
    try:
        from mindspeed.ops import npu_fusion_attention
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "NpuFlashAttentionBackend requires MindSpeed (mindspeed.ops). "
            "Install MindSpeed matching your CANN version."
        ) from exc
    return npu_fusion_attention


class NpuFlashAttentionBackend(FlashAttentionMixin):
    """Ascend NPU backend via ``npu_fusion_attention`` (TND layout)."""

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

    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        q, k, v, _cu_q, _cu_kv, _max_q, _max_kv = self._prepare_flash_inputs(
            query, key, value, prefix_sharing_plan
        )

        npu_fusion_attention = _import_npu_fusion_attention()

        # ``atten_mask`` is a BoolTensor with True = masked, matching our
        # build_block_causal_mask convention directly.
        atten_mask = build_block_causal_mask(prefix_sharing_plan, device=q.device)

        num_heads = q.shape[1]
        # ``actual_seq_qlen`` / ``actual_seq_kvlen`` are Python lists of per-sample
        # lengths (NOT cumulative).
        actual_seq_qlen = list(prefix_sharing_plan.cu_seqlens_q[1:])
        actual_seq_kvlen = list(prefix_sharing_plan.cu_seqlens_kv[1:])

        import math
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
                f"npu_fusion_attention failed: q={tuple(q.shape)}, k={tuple(k.shape)}"
            ) from exc

        # The op returns a tuple of 7 elements; [0] is the attention output.
        if isinstance(result, (tuple, list)):
            return result[0]
        return result
