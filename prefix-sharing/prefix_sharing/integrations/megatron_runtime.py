"""Runtime hook used by the minimal Megatron attention patch."""

from __future__ import annotations

from typing import Any

import torch

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.integrations.context import current_prefix_sharing_context


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
    batch_runtime_layout = ctx.batch_runtime_layout
    if batch_runtime_layout.layout_kind == "thd" and (
        packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd"
    ):
        raise RuntimeError("prefix sharing THD path requires packed_seq_params.qkv_format='thd'")
    if batch_runtime_layout.layout_kind == "bshd" and packed_seq_params is not None:
        raise RuntimeError("prefix sharing BSHD path requires packed_seq_params=None")
    if rotary_pos_emb is None:
        raise RuntimeError("prefix sharing requires rotary_pos_emb")
    if batch_runtime_layout.position_ids is None:
        raise RuntimeError("prefix sharing context is missing layout position_ids")

    q_pos_emb, k_pos_emb = rotary_pos_emb
    query, key = _apply_positioned_rope(
        attention_module,
        query,
        key,
        q_pos_emb,
        k_pos_emb,
        batch_runtime_layout,
    )

    backend = ctx.backend or TorchReferenceBackend()
    parallel_info = ctx.parallel_info
    layer_id = int(getattr(attention_module, "layer_number", 0) or 0)

    prefix_log.warning("\n\n\ntry to build kv\n\n\n")
    prefix_log.warning(
        "[PS][attention][global_rank=%s tp_rank=%s/tp_size=%s pp_rank=%s/pp_size=%s layer=%s] "
        "enter prefix-sharing path: "
        "layout_kind=%s, query_shape=%s, key_shape=%s, value_shape=%s, valid_lengths=%s",
        parallel_info.global_rank,
        parallel_info.tp_rank,
        parallel_info.tp_size,
        parallel_info.pp_rank,
        parallel_info.pp_size,
        layer_id,
        batch_runtime_layout.layout_kind,
        tuple(query.shape),
        tuple(key.shape),
        tuple(value.shape),
        batch_runtime_layout.valid_lengths,
    )
    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        ctx.store,
        ctx.prefix_sharing_plan,
        batch_runtime_layout=batch_runtime_layout,
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
        _shape_for_log(expanded_key),
        _shape_for_log(expanded_value),
    )
    core_attn_out = backend.attention(
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
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), core_attn_out.size(1), -1)
    else:
        raise ValueError(f"unsupported batch runtime layout: {batch_runtime_layout.layout_kind}")
    return attention_module.linear_proj(core_attn_out)


def _shape_for_log(value: Any) -> Any:
    if isinstance(value, list):
        return [tuple(item.shape) for item in value]
    return tuple(value.shape)


def _apply_positioned_rope(
    attention_module: Any,
    query: Any,
    key: Any,
    q_pos_emb: Any,
    k_pos_emb: Any,
    batch_runtime_layout: Any,
) -> tuple[Any, Any]:
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    q_positions = _positions_for_tensor(batch_runtime_layout, query, device=query.device)
    k_positions = _positions_for_tensor(batch_runtime_layout, key, device=key.device)
    max_needed = max(q_positions.max().item(), k_positions.max().item()) + 1

    query, q_restore_shape, q_squeeze_batch_dim = _flatten_rope_tensor(query, q_positions)
    key, k_restore_shape, k_squeeze_batch_dim = _flatten_rope_tensor(key, k_positions)

    # Extend q_pos_emb / k_pos_emb when they are shorter than the largest
    # position_id needed by the runtime layout.  THD-mode generated pos_emb
    # only covers positions 0 .. max_seqlen_q-1, which is too small because
    # prefix-sharing preserves the original position_ids (e.g. suffix starts
    # at position 75).
    #
    # RoPE is linear: freqs[p] = p * inv_freq.  The step (inv_freq) can be
    # recovered from the pos_emb itself as pos_emb[1] - pos_emb[0].
    if q_pos_emb is not None and max_needed > q_pos_emb.shape[0]:
        dim_half = q_pos_emb.shape[-1] // 2
        step = q_pos_emb[1:2, :, :, :dim_half] - q_pos_emb[0:1, :, :, :dim_half]
        n_extra = max_needed - q_pos_emb.shape[0]
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
        n_extra = max_needed - k_pos_emb.shape[0]
        extra_positions = torch.arange(
            k_pos_emb.shape[0], max_needed,
            device=k_pos_emb.device, dtype=k_pos_emb.dtype,
        )
        extra_angles = extra_positions[:, None, None, None] * step
        extra_emb = torch.cat([extra_angles, extra_angles], dim=-1)
        k_pos_emb = torch.cat([k_pos_emb, extra_emb], dim=0)

    if q_pos_emb is not None:
        q_freqs = q_pos_emb.index_select(0, q_positions)
        query = apply_rotary_pos_emb(
            query.unsqueeze(1) if q_squeeze_batch_dim else query,
            q_freqs,
            config=attention_module.config,
            cu_seqlens=None,
        )
        if q_squeeze_batch_dim:
            query = query.squeeze(1)
        query = query.reshape(q_restore_shape)
    if k_pos_emb is not None:
        k_freqs = k_pos_emb.index_select(0, k_positions)
        key = apply_rotary_pos_emb(
            key.unsqueeze(1) if k_squeeze_batch_dim else key,
            k_freqs,
            config=attention_module.config,
            cu_seqlens=None,
        )
        if k_squeeze_batch_dim:
            key = key.squeeze(1)
        key = key.reshape(k_restore_shape)
    return query, key


def _positions_for_tensor(batch_runtime_layout: Any, tensor: Any, *, device: Any) -> Any:
    position_ids = batch_runtime_layout.position_ids.to(device=device, dtype=torch.long)
    if batch_runtime_layout.layout_kind == "thd":
        positions = position_ids.reshape(-1)
    elif batch_runtime_layout.layout_kind == "bshd":
        valid_positions = batch_runtime_layout.valid_position_ids(device=device).to(dtype=torch.long)
        full_positions = position_ids.reshape(-1)
        if tensor.dim() >= 4 and tensor.shape[0] * tensor.shape[1] == full_positions.numel():
            positions = full_positions
        elif tensor.shape[0] == valid_positions.numel():
            positions = valid_positions
        elif tensor.shape[0] == full_positions.numel():
            positions = full_positions
        else:
            raise RuntimeError(
                "prefix sharing BSHD RoPE cannot align tensor with position_ids: "
                f"tensor_shape={tuple(tensor.shape)}, full_positions={full_positions.numel()}, "
                f"valid_positions={valid_positions.numel()}"
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
