"""GPU Flash Attention backend for prefix sharing.

This backend fixes the mask-correctness issue that arises when reusers are
present. ``flash_attn_varlen_func(causal=True)`` cannot be used for any plan
that contains a reuser, because its causal mask is segment-relative: Q[i] of a
reuser's suffix would only see KV[0..i] of the prefix+suffix block, leaving
suffix KV columns entirely masked. See ``docs/pr6-fusion-plan.html`` §3 for
the full analysis.

Resolution
----------
* If the plan has **no reuser**, we may safely call ``flash_attn_varlen_func``
  with ``causal=True`` (segment-relative and absolute-relative coincide).
* If the plan has **any reuser**, we route through Transformer Engine's
  ``DotProductAttention`` with an explicit ``core_attention_bias`` constructed
  from the boolean block-causal mask. TE accepts a custom bias and therefore
  preserves the absolute-position mask semantics that prefix sharing requires.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.block_causal_mask import (
    build_block_causal_mask,
    mask_to_te_bias,
)
from prefix_sharing.backends.flash_atten_base import (
    FlashAttentionMixin,
    FlashBackendValidationError,
)
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


@lru_cache(maxsize=None)
def _import_flash_attn_varlen() -> Any:
    try:
        from flash_attn import flash_attn_varlen_func
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GpuFlashAttentionBackend requires the flash-attn package. "
            "Install flash-attention first."
        ) from exc
    return flash_attn_varlen_func


@lru_cache(maxsize=None)
def _import_te_dot_product_attention() -> Any:
    try:
        from transformer_engine.pytorch.attention import DotProductAttention
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GpuFlashAttentionBackend requires Transformer Engine for reuser plans. "
            "Install transformer-engine; or pick a plan with no reusers; or fall "
            "back to backend='torch_ref'."
        ) from exc
    return DotProductAttention


class GpuFlashAttentionBackend(FlashAttentionMixin):
    """CUDA/GPU backend: TE for reuser plans, flash-attn for provider-only plans."""

    capabilities = BackendCapabilities(
        name="flash_atten_gpu",
        supports_cpu=False,
        supports_cuda=True,
        supports_cann=False,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_gated_attention=False,
        supports_deltanet_state_reuse=False,
    )

    def __init__(self) -> None:
        self._torch_ref = TorchReferenceBackend()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)

    # ------------------------------------------------------------------
    # RoPE & KV build: delegate to reference
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
        **kwargs: Any,
    ) -> Any:
        has_reuser = any(
            prefix_sharing_plan.is_reuser(i)
            for i in range(prefix_sharing_plan.batch_size)
        )
        if has_reuser:
            return self._attention_via_te(query, key, value, prefix_sharing_plan, **kwargs)
        return self._attention_via_flash_attn(query, key, value, prefix_sharing_plan, **kwargs)

    def _attention_via_flash_attn(
        self,
        query: Any,
        key: Any,
        value: Any,
        plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        """Provider-only / no-sharing fast path.

        FA's segment-relative causal mask coincides with absolute-position causal
        when every sample is its own provider, so ``causal=True`` is correct.
        """
        q, k, v, cu_q, cu_kv, max_q, max_kv = self._prepare_flash_inputs(
            query, key, value, plan
        )
        flash_attn_varlen_func = _import_flash_attn_varlen()
        try:
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_kv,
                max_q,
                max_kv,
                dropout_p=kwargs.get("dropout_p", 0.0),
                softmax_scale=kwargs.get("softmax_scale"),
                causal=kwargs.get("causal", True),
                window_size=kwargs.get("window_size", (-1, -1)),
                softcap=kwargs.get("softcap", 0.0),
                alibi_slopes=kwargs.get("alibi_slopes"),
                deterministic=kwargs.get("deterministic", False),
            )
        except Exception as exc:
            raise FlashBackendValidationError(
                f"flash_attn_varlen_func failed on device={q.device}, "
                f"q_shape={tuple(q.shape)}, k_shape={tuple(k.shape)}, "
                f"cu_seqlens_q={cu_q.tolist()}, cu_seqlens_kv={cu_kv.tolist()}"
            ) from exc

    def _attention_via_te(
        self,
        query: Any,
        key: Any,
        value: Any,
        plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        """Reuser-capable path via Transformer Engine.

        TE's ``DotProductAttention`` accepts ``core_attention_bias`` which lets
        us inject an explicit (Q, KV) bias built from the block-causal mask.
        TE's THD path requires ``attention_mask_type='padding_causal'`` and a
        dummy batch dimension.
        """
        torch = _torch()
        q, k, v, cu_q, cu_kv, max_q, max_kv = self._prepare_flash_inputs(
            query, key, value, plan
        )

        bool_mask = build_block_causal_mask(plan, device=q.device)
        bias = mask_to_te_bias(bool_mask, dtype=q.dtype)
        # TE expects (batch=1, total, heads, dim) for THD.
        q4 = q.unsqueeze(0)
        k4 = k.unsqueeze(0)
        v4 = v.unsqueeze(0)

        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[-1]

        DotProductAttention = _import_te_dot_product_attention()
        # Build a fresh DPA instance. The kernel is stateless w.r.t. inputs; the
        # constructor is cheap enough that this does not dominate the step.
        dpa = DotProductAttention(
            num_heads=num_heads,
            attention_type="self",
            num_gqa_groups=num_kv_heads if num_kv_heads != num_heads else None,
            sm_margin_only=False,
            tp_size=1,
            get_rng_state_tracker=None,
            tp_group=None,
            sequence_parallel=False,
            attention_mask_type="padding_causal",
            attention_dropout=kwargs.get("dropout_p", 0.0),
            qkv_format="thd",
            core_attention_bias_type="post_scale_bias",
            window_size=kwargs.get("window_size", (1, 1)),
        )

        softmax_scale = kwargs.get("softmax_scale")
        if softmax_scale is None:
            import math
            softmax_scale = 1.0 / math.sqrt(head_dim)

        try:
            out = dpa(
                q4,
                k4,
                v4,
                core_attention_bias=bias,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_kv,
                max_seqlen_q=max_q,
                max_seqlen_kv=max_kv,
                attention_metric=None,
                checkpoint_core_attention=False,
                core_attention_bias_type="post_scale_bias",
                fast_zero_fill=False,
                num_logits_to_keep=None,
                sequence_parallel=False,
                attention_scale=softmax_scale,
            )
        except Exception as exc:
            raise FlashBackendValidationError(
                f"TE DotProductAttention failed on device={q.device}, "
                f"q_shape={tuple(q.shape)}, k_shape={tuple(k.shape)}"
            ) from exc
        return out.squeeze(0)
