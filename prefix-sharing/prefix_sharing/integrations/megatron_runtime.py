"""Runtime hook used by the minimal Megatron attention patch."""

from __future__ import annotations

from typing import Any

import torch

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.integrations.utils import ensure_global_packed_token_lengths


def maybe_run_prefix_sharing_attention(
    attention_module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    rotary_pos_emb: Any,
    packed_seq_params: Any,
) -> tuple[Any, Any] | None:
    """Run prefix-sharing attention when a runtime context is active.

    Returns ``None`` for the normal Megatron path. When active, this function
    owns RoPE, KV expansion, causal masking, and output projection.
    """

    import logging
    prefix_log = logging.getLogger(__file__)
    prefix_log.warning("\n\n\nsuccess come into def maybe_run_prefix_sharing_attention\n\n\n")

    ctx = current_prefix_sharing_context()
    if ctx is None:
        prefix_log.warning("\n\n\nctx is None\n\n\n")
        return None
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        raise RuntimeError("prefix sharing phase 1 requires packed_seq_params.qkv_format='thd'")
    if rotary_pos_emb is None:
        raise RuntimeError("prefix sharing phase 1 requires rotary_pos_emb")
    if ctx.packed_batch_layout.packed_position_ids is None:
        raise RuntimeError("prefix sharing context is missing packed_position_ids")

    packed_batch_layout = ctx.packed_batch_layout
    ensure_global_packed_token_lengths(
        {
            "query_length": query.shape[0],
            "key_length": key.shape[0],
            "value_length": value.shape[0],
        },
        total_padded_length=packed_batch_layout.total_padded_length,
        context="attention hook",
    )

    # v0.16.1: extract cu_seqlens, mscale, cp_group from runtime objects.
    # Returns None/defaults for v070 (mcore <= 0.15.x) — backward compatible.
    cu_seqlens_q = _extract_cu_seqlens(packed_seq_params, "cu_seqlens_q_padded", "cu_seqlens_q")
    cu_seqlens_kv = _extract_cu_seqlens(packed_seq_params, "cu_seqlens_kv_padded", "cu_seqlens_kv")
    mscale = _get_yarn_mscale(attention_module)
    cp_group = _get_cp_group(attention_module)

    # mcore 0.16.1: rotary_pos_emb 是单 tensor（Q/K 共用）
    # mcore <= 0.15.x: rotary_pos_emb 是 (q_pos_emb, k_pos_emb) tuple
    if isinstance(rotary_pos_emb, (tuple, list)) and len(rotary_pos_emb) == 2:
        q_pos_emb, k_pos_emb = rotary_pos_emb
    else:
        # 单 tensor 格式，Q/K 共用同一个频率嵌入
        q_pos_emb = rotary_pos_emb
        k_pos_emb = rotary_pos_emb
    query, key = _apply_positioned_rope(
        attention_module,
        query,
        key,
        q_pos_emb,
        k_pos_emb,
        packed_batch_layout.packed_position_ids,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        mscale=mscale,
        cp_group=cp_group,
    )

    backend = ctx.backend or TorchReferenceBackend()
    parallel_info = ctx.parallel_info
    layer_id = int(getattr(attention_module, "layer_number", 0) or 0)

    prefix_log.warning("\n\n\ntry to build kv\n\n\n")
    prefix_log.warning(
        "[PS][attention][global_rank=%s tp_rank=%s/tp_size=%s pp_rank=%s/pp_size=%s layer=%s] "
        "enter prefix-sharing path: "
        "sequence_parallel=%s query_token_length=%s total_padded_length=%s "
        "query_shape=%s, key_shape=%s, value_shape=%s, valid_lengths=%s, "
        "padded_lengths=%s, cu_seqlens=%s",
        parallel_info.global_rank,
        parallel_info.tp_rank,
        parallel_info.tp_size,
        parallel_info.pp_rank,
        parallel_info.pp_size,
        layer_id,
        getattr(getattr(attention_module, "config", None), "sequence_parallel", None),
        query.shape[0],
        packed_batch_layout.total_padded_length,
        tuple(query.shape),
        tuple(key.shape),
        tuple(value.shape),
        packed_batch_layout.valid_lengths,
        packed_batch_layout.padded_lengths,
        packed_batch_layout.cu_seqlens,
    )
    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        ctx.store,
        ctx.prefix_sharing_plan,
        packed_batch_layout=packed_batch_layout,
        layer_id=layer_id,
        tp_rank=parallel_info.tp_rank,
    )
    prefix_log.warning(
        "[PS][attention][global_rank=%s tp_rank=%s/tp_size=%s pp_rank=%s/pp_size=%s layer=%s] "
        "built expanded kv: "
        "expanded_key_shape=%s, expanded_value_shape=%s",
        parallel_info.global_rank,
        parallel_info.tp_rank,
        parallel_info.tp_size,
        parallel_info.pp_rank,
        parallel_info.pp_size,
        layer_id,
        tuple(expanded_key.shape),
        tuple(expanded_value.shape),
    )
    core_attn_out = backend.attention(
        query,
        expanded_key,
        expanded_value,
        ctx.prefix_sharing_plan,
        packed_batch_layout=packed_batch_layout,
        attention_mask=attention_mask,
    )
    core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
    return attention_module.linear_proj(core_attn_out)


def _apply_positioned_rope(
    attention_module: Any,
    query: Any,
    key: Any,
    q_pos_emb: Any,
    k_pos_emb: Any,
    packed_position_ids: Any,
    *,
    cu_seqlens_q: Any | None = None,
    cu_seqlens_kv: Any | None = None,
    mscale: float | None = None,
    cp_group: Any | None = None,
) -> tuple[Any, Any]:
    """Apply RoPE using packed_position_ids, with optional v0.16.1 API params.

    v070 (mcore <= 0.15.x): cu_seqlens=None, no mscale/cp_group.
    v0.16.1+ (mcore 0.16.1): cu_seqlens from packed_seq_params, mscale for
    yarn models, cp_group for context parallel.

    Backward compatible: mscale and cp_group are only passed to
    apply_rotary_pos_emb when they differ from defaults, so v0.15.x
    (which doesn't have these kwargs) won't get a TypeError.
    """
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    positions = packed_position_ids.to(device=query.device, dtype=torch.long)
    max_needed = positions.max().item() + 1

    # Extend q_pos_emb / k_pos_emb when they are shorter than the largest
    # position_id needed by packed_position_ids.  THD-mode generated pos_emb
    # only covers positions 0 .. max_seqlen_q-1, which is too small because
    # prefix-sharing preserves the original position_ids (e.g. suffix starts
    # at position 75).
    #
    # RoPE is linear: freqs[p] = p * inv_freq.  The step (inv_freq) can be
    # recovered from the pos_emb itself as pos_emb[1] - pos_emb[0].
    if q_pos_emb is not None and max_needed > q_pos_emb.shape[0]:
        dim_half = q_pos_emb.shape[-1] // 2
        step = q_pos_emb[1:2, :, :, :dim_half] - q_pos_emb[0:1, :, :, :dim_half]
        extra_positions = torch.arange(
            q_pos_emb.shape[0], max_needed,
            device=q_pos_emb.device, dtype=q_pos_emb.dtype,
        )
        extra_angles = extra_positions[:, None, None, None] * step
        extra_emb = torch.cat([extra_angles, extra_angles], dim=-1)
        q_pos_emb = torch.cat([q_pos_emb, extra_emb], dim=0)
    if k_pos_emb is not None and max_needed > k_pos_emb.shape[0]:
        dim_half = k_pos_emb.shape[-1] // 2
        step = k_pos_emb[1:2, :, :, :dim_half] - k_pos_emb[0:1, :, :, :dim_half]
        extra_positions = torch.arange(
            k_pos_emb.shape[0], max_needed,
            device=k_pos_emb.device, dtype=k_pos_emb.dtype,
        )
        extra_angles = extra_positions[:, None, None, None] * step
        extra_emb = torch.cat([extra_angles, extra_angles], dim=-1)
        k_pos_emb = torch.cat([k_pos_emb, extra_emb], dim=0)

    # Build kwargs for apply_rotary_pos_emb.
    # Only include version-specific params when they're provided,
    # to maintain backward compat with v070 (mcore <= 0.15.x).
    def _rope_kwargs(cu_seqlens: Any | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"config": attention_module.config, "cu_seqlens": cu_seqlens}
        if mscale is not None and mscale != 1.0:
            kwargs["mscale"] = mscale
        if cp_group is not None:
            kwargs["cp_group"] = cp_group
        return kwargs

    if q_pos_emb is not None:
        q_freqs = q_pos_emb.index_select(0, positions)
        query = apply_rotary_pos_emb(
            query.unsqueeze(1),
            q_freqs,
            **_rope_kwargs(cu_seqlens_q),
        ).squeeze(1)
    if k_pos_emb is not None:
        k_freqs = k_pos_emb.index_select(0, positions)
        key = apply_rotary_pos_emb(
            key.unsqueeze(1),
            k_freqs,
            **_rope_kwargs(cu_seqlens_kv),
        ).squeeze(1)
    return query, key


# ═══════════════════════════════════════
# v0.16.1 API helpers (backward compatible with v070)
# ═══════════════════════════════════════

def _extract_cu_seqlens(packed_seq_params: Any, primary_attr: str, fallback_attr: str) -> Any | None:
    """Extract cu_seqlens from packed_seq_params, preferring padded version.

    Returns None for v070 (mcore <= 0.15.x) where these attributes don't exist.
    """
    if packed_seq_params is None:
        return None
    val = getattr(packed_seq_params, primary_attr, None)
    if val is None:
        val = getattr(packed_seq_params, fallback_attr, None)
    return val


def _get_yarn_mscale(attention_module: Any) -> float:
    """Get yarn mscale from attention module config (v0.16.1+).

    Returns 1.0 for v070 (mcore <= 0.15.x) where this function doesn't exist.
    """
    try:
        from megatron.core.transformer.attention import _yarn_get_concentration_factor_from_config
        return float(_yarn_get_concentration_factor_from_config(attention_module.config))
    except (ImportError, AttributeError):
        return 1.0


def _get_cp_group(attention_module: Any) -> Any | None:
    """Get context parallel group from attention module (v0.16.1+).

    Returns None for v070 (mcore <= 0.15.x) where pg_collection doesn't exist.
    """
    pg_collection = getattr(attention_module, "pg_collection", None)
    if pg_collection is None:
        return None
    return getattr(pg_collection, "cp", None)
