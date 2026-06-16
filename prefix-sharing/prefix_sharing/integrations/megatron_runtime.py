"""Runtime hook used by the minimal Megatron attention patch."""

from __future__ import annotations

from typing import Any

import torch

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.utils import ensure_global_packed_token_lengths

import logging

logger = logging.getLogger(__name__)


def prefix_attention(
    attention_module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    rotary_pos_emb: Any,
    packed_seq_params: Any,
) -> tuple[Any, Any] | None:
    """Run prefix-sharing attention when a runtime context is active."""

    ctx = current_prefix_sharing_context()
    if ctx is None:
        return None

    batch_runtime_layout = ctx.batch_runtime_layout
    if batch_runtime_layout.layout_kind == "thd":
        if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
            raise RuntimeError("prefix sharing THD path requires packed_seq_params.qkv_format='thd'")
        ensure_global_packed_token_lengths(
            {
                "query_length": query.shape[0],
                "key_length": key.shape[0],
                "value_length": value.shape[0],
            },
            total_padded_length=batch_runtime_layout.total_padded_length,
            context="attention hook",
        )
    elif batch_runtime_layout.layout_kind == "bshd":
        if packed_seq_params is not None:
            raise RuntimeError("prefix sharing BSHD path requires packed_seq_params=None")
        _ensure_bshd_attention_tensor_shapes(query, key, value, batch_runtime_layout)
    else:
        raise ValueError(f"unsupported batch runtime layout: {batch_runtime_layout.layout_kind}")

    if rotary_pos_emb is None:
        raise RuntimeError("prefix sharing requires rotary_pos_emb")
    if batch_runtime_layout.position_ids is None:
        raise RuntimeError("prefix sharing context is missing layout position_ids")

    cu_seqlens_q = _extract_cu_seqlens(packed_seq_params, "cu_seqlens_q_padded", "cu_seqlens_q")
    cu_seqlens_kv = _extract_cu_seqlens(packed_seq_params, "cu_seqlens_kv_padded", "cu_seqlens_kv")
    mscale = _get_yarn_mscale(attention_module)
    cp_group = _get_cp_group(attention_module)
    q_pos_emb, k_pos_emb = _unpack_rotary_pos_emb(rotary_pos_emb)

    query, key = _apply_positioned_rope(
        attention_module,
        query,
        key,
        q_pos_emb,
        k_pos_emb,
        batch_runtime_layout,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        mscale=mscale,
        cp_group=cp_group,
    )

    parallel_info = ctx.parallel_info
    layer_id = int(getattr(attention_module, "layer_number", 0) or 0)
    seq_parallel = getattr(getattr(attention_module, "config", None), "sequence_parallel", None)
    logger.warning(
        f"[PS][attention][global_rank={parallel_info.global_rank} tp_rank={parallel_info.tp_rank}/tp_size={parallel_info.tp_size} sequence_parallel={seq_parallel} "
        f"pp_rank={parallel_info.pp_rank}/pp_size={parallel_info.pp_size} layer={layer_id}] enter prefix-sharing path: "
        f"layout_kind={batch_runtime_layout.layout_kind} query_shape={tuple(query.shape)} key_shape={tuple(key.shape)} value_shape={tuple(value.shape)} valid_lengths={batch_runtime_layout.valid_lengths}"
    )

    attention_backend = ctx.attention_backend or TorchReferenceBackend()
    expanded_key, expanded_value = attention_backend.build_kv(
        key,
        value,
        ctx.store,
        ctx.prefix_sharing_plan,
        batch_runtime_layout=batch_runtime_layout,
        layer_id=layer_id,
        tp_rank=parallel_info.tp_rank,
    )
    logger.warning(
        f"[PS][attention][global_rank={parallel_info.global_rank} tp_rank={parallel_info.tp_rank}/tp_size={parallel_info.tp_size} sequence_parallel={seq_parallel} "
        f"pp_rank={parallel_info.pp_rank}/pp_size={parallel_info.pp_size} layer={layer_id}] built expanded kv: "
        f"expanded_key_shape={_shape_for_log(expanded_key)} expanded_value_shape={_shape_for_log(expanded_value)}"
    )

    core_attn_out = attention_backend.attention(
        query,
        expanded_key,
        expanded_value,
        ctx.prefix_sharing_plan,
        batch_runtime_layout=batch_runtime_layout,
        attention_mask=attention_mask,
    )
    if batch_runtime_layout.layout_kind == "thd":
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
    elif batch_runtime_layout.layout_kind == "bshd":
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:2], -1)
    return attention_module.linear_proj(core_attn_out)


def _shape_for_log(value: Any) -> Any:
    if isinstance(value, list):
        return [tuple(item.shape) for item in value]
    return tuple(value.shape)


def _ensure_bshd_attention_tensor_shapes(query: Any, key: Any, value: Any, layout: Any) -> None:
    for name, tensor in {"query": query, "key": key, "value": value}.items():
        if tensor.dim() < 3:
            raise RuntimeError(f"prefix sharing BSHD path expects at least 3D {name}, got shape={tuple(tensor.shape)}")
        if tensor.shape[0] == layout.batch_size and tensor.shape[1] >= layout.max_seqlen:
            continue
        if tensor.shape[0] >= max(layout.valid_lengths, default=0) and tensor.shape[1] == layout.batch_size:
            continue
        if tensor.shape[0] == layout.total_valid_length:
            continue
        raise RuntimeError(
            "prefix sharing BSHD path cannot align tensor with batch layout: "
            f"name={name}, shape={tuple(tensor.shape)}, batch_size={layout.batch_size}, "
            f"max_seqlen={layout.max_seqlen}, total_valid={layout.total_valid_length}"
        )


def _apply_positioned_rope(
    attention_module: Any,
    query: Any,
    key: Any,
    q_pos_emb: Any,
    k_pos_emb: Any,
    batch_runtime_layout: Any,
    *,
    cu_seqlens_q: Any | None = None,
    cu_seqlens_kv: Any | None = None,
    mscale: float | None = None,
    cp_group: Any | None = None,
) -> tuple[Any, Any]:
    """Apply RoPE by selecting exact position ids from the runtime layout."""

    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    q_positions = _positions_for_tensor(batch_runtime_layout, query, device=query.device)
    k_positions = _positions_for_tensor(batch_runtime_layout, key, device=key.device)
    max_needed = max(int(q_positions.max().item()), int(k_positions.max().item())) + 1

    query, q_restore_shape, q_squeeze_batch_dim = _flatten_rope_tensor(query, q_positions)
    key, k_restore_shape, k_squeeze_batch_dim = _flatten_rope_tensor(key, k_positions)

    if q_pos_emb is not None and max_needed > q_pos_emb.shape[0]:
        q_pos_emb = _extend_rope_positions(q_pos_emb, max_needed)
    if k_pos_emb is not None and max_needed > k_pos_emb.shape[0]:
        k_pos_emb = _extend_rope_positions(k_pos_emb, max_needed)

    def _rope_kwargs(_unused_cu_seqlens: Any | None) -> dict[str, Any]:
        # We pre-select frequencies by index, so passing real cu_seqlens would
        # trigger Megatron's THD split path again. That is unnecessary and has
        # been observed to fail on NPU cat kernels.
        kwargs: dict[str, Any] = {"config": attention_module.config, "cu_seqlens": None}
        if mscale is not None and mscale != 1.0:
            kwargs["mscale"] = mscale
        return kwargs

    if q_pos_emb is not None:
        q_freqs = q_pos_emb.index_select(0, q_positions)
        query = apply_rotary_pos_emb(
            query.unsqueeze(1) if q_squeeze_batch_dim else query,
            q_freqs,
            **_rope_kwargs(cu_seqlens_q),
        )
        if q_squeeze_batch_dim:
            query = query.squeeze(1)
        query = query.reshape(q_restore_shape)
    if k_pos_emb is not None:
        k_freqs = k_pos_emb.index_select(0, k_positions)
        key = apply_rotary_pos_emb(
            key.unsqueeze(1) if k_squeeze_batch_dim else key,
            k_freqs,
            **_rope_kwargs(cu_seqlens_kv),
        )
        if k_squeeze_batch_dim:
            key = key.squeeze(1)
        key = key.reshape(k_restore_shape)
    return query, key


def _extend_rope_positions(pos_emb: Any, max_needed: int) -> Any:
    dim_half = pos_emb.shape[-1] // 2
    step = pos_emb[1:2, :, :, :dim_half] - pos_emb[0:1, :, :, :dim_half]
    extra_positions = torch.arange(
        pos_emb.shape[0], max_needed,
        device=pos_emb.device,
        dtype=pos_emb.dtype,
    )
    extra_angles = extra_positions[:, None, None, None] * step
    extra_emb = torch.cat([extra_angles, extra_angles], dim=-1)
    return torch.cat([pos_emb, extra_emb], dim=0)


def _positions_for_tensor(batch_runtime_layout: Any, tensor: Any, *, device: Any) -> Any:
    position_ids = batch_runtime_layout.position_ids.to(device=device, dtype=torch.long)
    if batch_runtime_layout.layout_kind == "thd":
        positions = position_ids.reshape(-1)
    elif batch_runtime_layout.layout_kind == "bshd":
        valid_positions = batch_runtime_layout.valid_position_ids(device=device).to(dtype=torch.long)
        kept_padded_positions = batch_runtime_layout.kept_padded_position_ids(device=device).to(dtype=torch.long)
        full_positions = position_ids.reshape(-1)
        if (
            tensor.dim() >= 4
            and tensor.shape[0] >= kept_padded_positions.shape[0]
            and tensor.shape[1] == kept_padded_positions.shape[1]
        ):
            positions = batch_runtime_layout.kept_padded_position_ids(
                device=device,
                padded_length=int(tensor.shape[0]),
            ).to(dtype=torch.long).reshape(-1)
        elif tensor.dim() >= 4 and tensor.shape[0] * tensor.shape[1] == full_positions.numel():
            positions = full_positions
        elif tensor.shape[0] == valid_positions.numel():
            positions = valid_positions
        elif tensor.shape[0] == full_positions.numel():
            positions = full_positions
        else:
            raise RuntimeError(
                "prefix sharing BSHD RoPE cannot align tensor with position_ids: "
                f"tensor_shape={tuple(tensor.shape)}, full_positions={full_positions.numel()}, "
                f"valid_positions={valid_positions.numel()}, "
                f"kept_padded_positions={kept_padded_positions.numel()}"
            )
    else:
        raise ValueError(f"unsupported batch runtime layout: {batch_runtime_layout.layout_kind}")
    return positions.to(device=device, dtype=torch.long).reshape(-1)


def _flatten_rope_tensor(tensor: Any, positions: Any) -> tuple[Any, Any, bool]:
    restore_shape = tensor.shape
    position_count = int(positions.numel())
    if tensor.dim() == 3:
        if tensor.shape[0] != position_count:
            raise RuntimeError(
                "prefix sharing RoPE position count does not match 3D tensor: "
                f"tensor_shape={tuple(tensor.shape)}, positions={position_count}"
            )
        return tensor, restore_shape, True
    if tensor.dim() == 4:
        if tensor.shape[0] == position_count:
            return tensor, restore_shape, False
        if tensor.shape[0] * tensor.shape[1] == position_count:
            return tensor.reshape(position_count, 1, *tensor.shape[2:]), restore_shape, False
        raise RuntimeError(
            "prefix sharing RoPE position count does not match 4D tensor: "
            f"tensor_shape={tuple(tensor.shape)}, positions={position_count}"
        )
    raise RuntimeError(f"prefix sharing RoPE expects 3D or 4D Q/K tensor, got shape={tuple(tensor.shape)}")


# ═══════════════════════════════════════
# v0.16.1 API helpers (backward compatible with v070)
# ═══════════════════════════════════════


def _unpack_rotary_pos_emb(rotary_pos_emb: Any) -> tuple[Any, Any]:
    """解包 rotary_pos_emb，兼容 mcore 版本差异。

    mcore 0.12.1 ~ 0.15.x: (q_pos_emb, k_pos_emb) tuple
    mcore 0.16.1+:             单 tensor（Q/K 共用）
    """
    if isinstance(rotary_pos_emb, (tuple, list)) and len(rotary_pos_emb) == 2:
        return rotary_pos_emb[0], rotary_pos_emb[1]
    return rotary_pos_emb, rotary_pos_emb


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
