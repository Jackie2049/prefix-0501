"""Runtime hook used by the minimal Megatron attention patch."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.model_spec import AttentionLayerType
from prefix_sharing.integrations.context import current_prefix_sharing_context


def maybe_run_prefix_sharing_attention(
    attention_module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    rotary_pos_emb: Any,
    packed_seq_params: Any,
    rotary_pos_cos: Any | None = None,
    rotary_pos_sin: Any | None = None,
    mscale: Any = None,
    **kwargs,
) -> tuple[Any, Any] | None:
    """Run prefix-sharing attention when a runtime context is active.

    Returns ``None`` for the normal Megatron path. When active, this function
    owns RoPE, KV expansion, causal masking, and output projection.
    """

    import logging
    prefix_log = logging.getLogger(__name__)

    ctx = current_prefix_sharing_context()
    if ctx is None:
        return None
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        raise RuntimeError("prefix sharing phase 1 requires packed_seq_params.qkv_format='thd'")
    if rotary_pos_emb is None and (rotary_pos_cos is None or rotary_pos_sin is None):
        raise RuntimeError("prefix sharing phase 1 requires rotary_pos_emb or rotary_pos_cos/sin")
    if ctx.packed_batch_layout.packed_position_ids is None:
        raise RuntimeError("prefix sharing context is missing packed_position_ids")

    packed_batch_layout = ctx.packed_batch_layout

    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        query, key = _apply_positioned_rope(
            attention_module,
            query,
            key,
            q_pos_emb,
            k_pos_emb,
            packed_batch_layout.packed_position_ids,
        )
    else:
        query, key = _apply_fused_rope(
            attention_module,
            query,
            key,
            rotary_pos_cos,
            rotary_pos_sin,
            packed_batch_layout.packed_position_ids,
        )
    backend = ctx.backend or TorchReferenceBackend()
    global_rank, tp_rank, tp_size = _read_parallel_rank_info()
    # Megatron layer_number is 1-indexed; convert to 0-indexed for ModelSpec
    layer_id = _normalize_layer_number(attention_module)

    # Determine layer type for HybridAttention models (e.g., Qwen3.6-27B)
    if layer_id < 0:
        return None  # Could not determine layer_number, skip PS
    model_spec = ctx.model_spec
    if model_spec is not None:
        layer_type = model_spec.layer_type(layer_id)
        if layer_type == AttentionLayerType.LINEAR_ATTENTION:
            # Linear attention layers use recurrent state (GatedDeltaNet),
            # not standard KV-based attention. Skip KV injection for now;
            # DeltaNet state reuse is handled by a separate integration point.
            return None

    prefix_log.debug(
        "[PS][attention][rank=%s tp=%s/%s layer=%s] query=%s key=%s value=%s",
        global_rank, tp_rank, tp_size, layer_id,
        tuple(query.shape), tuple(key.shape), tuple(value.shape),
    )
    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        ctx.store,
        ctx.prefix_sharing_plan,
        packed_batch_layout=packed_batch_layout,
        layer_id=layer_id,
        tp_rank=tp_rank,
    )
    prefix_log.debug(
        "[PS][attention] built expanded kv: key=%s value=%s",
        tuple(expanded_key.shape), tuple(expanded_value.shape),
    )
    core_attn_out = backend.attention(
        query,
        expanded_key,
        expanded_value,
        ctx.prefix_sharing_plan,
        packed_batch_layout=packed_batch_layout,
        attention_mask=attention_mask,
    )
    # Merge heads: (total_tokens, num_heads, head_dim) -> (total_tokens, hidden)
    core_attn_out = core_attn_out.reshape(core_attn_out.size(0), -1)
    return attention_module.linear_proj(core_attn_out)


def _apply_positioned_rope(
    attention_module: Any,
    query: Any,
    key: Any,
    q_pos_emb: Any,
    k_pos_emb: Any,
    packed_position_ids: Any,
) -> tuple[Any, Any]:
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
        q_freqs = q_pos_emb.index_select(0, positions)
        query = apply_rotary_pos_emb(
            query.unsqueeze(1),
            q_freqs,
            config=attention_module.config,
            cu_seqlens=None,
        ).squeeze(1)
    if k_pos_emb is not None:
        k_freqs = k_pos_emb.index_select(0, positions)
        key = apply_rotary_pos_emb(
            key.unsqueeze(1),
            k_freqs,
            config=attention_module.config,
            cu_seqlens=None,
        ).squeeze(1)
    return query, key


def _apply_fused_rope(
    attention_module: Any,
    query: Any,
    key: Any,
    rotary_pos_cos: Any,
    rotary_pos_sin: Any,
    packed_position_ids: Any,
) -> tuple[Any, Any]:
    """Apply RoPE using fused cos/sin embeddings with position indexing.

    This handles the Megatron fused RoPE path where ``rotary_pos_cos`` and
    ``rotary_pos_sin`` are provided instead of ``rotary_pos_emb``.  We
    combine cos and sin into a single pos_emb tensor and use the same
    ``apply_rotary_pos_emb`` helper.
    """
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    positions = packed_position_ids.to(device=query.device, dtype=torch.long)
    max_needed = positions.max().item() + 1

    # Combine cos and sin into a single pos_emb tensor: [seq_len, 1, 1, head_dim]
    # and apply the same extension logic as _apply_positioned_rope.
    pos_emb = torch.cat([rotary_pos_cos, rotary_pos_sin], dim=-1)  # [seq, 1, 1, 2*half]

    if max_needed > pos_emb.shape[0]:
        dim_half = pos_emb.shape[-1] // 2
        step = pos_emb[1:2, :, :, :dim_half] - pos_emb[0:1, :, :, :dim_half]
        extra_positions = torch.arange(
            pos_emb.shape[0], max_needed,
            device=pos_emb.device, dtype=pos_emb.dtype,
        )
        extra_angles = extra_positions[:, None, None, None] * step
        extra_emb = torch.cat([extra_angles, extra_angles], dim=-1)
        pos_emb = torch.cat([pos_emb, extra_emb], dim=0)

    freqs = pos_emb.index_select(0, positions)
    query = apply_rotary_pos_emb(
        query.unsqueeze(1),
        freqs,
        config=attention_module.config,
        cu_seqlens=None,
    ).squeeze(1)
    key = apply_rotary_pos_emb(
        key.unsqueeze(1),
        freqs,
        config=attention_module.config,
        cu_seqlens=None,
    ).squeeze(1)
    return query, key


def _normalize_layer_number(attention_module: Any) -> int:
    """Convert Megatron 1-indexed layer_number to 0-indexed for ModelSpec.

    Megatron uses 1-based layer_number (layer 0 has layer_number=1), but
    ModelSpec.layer_type() expects 0-based layer IDs. This helper safely
    converts the value and guards against invalid inputs.

    Returns -1 if the layer_number cannot be determined, which will cause
    ``model_spec.layer_type()`` to treat the layer as unknown (skipping PS).
    """
    raw = getattr(attention_module, "layer_number", None)
    if raw is None:
        return -1
    layer_number = int(raw)
    if layer_number < 1:
        return -1
    return layer_number - 1


def _read_parallel_rank_info() -> tuple[int | str, int, int]:
    global_rank: int | str = "unknown"
    tp_rank = 0
    tp_size = 1

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            global_rank = int(dist.get_rank())
    except Exception:
        pass

    try:
        from megatron.core import parallel_state

        tp_size = int(parallel_state.get_tensor_model_parallel_world_size())
        if hasattr(parallel_state, "get_tensor_model_parallel_rank"):
            tp_rank = int(parallel_state.get_tensor_model_parallel_rank())
    except Exception:
        pass

    return global_rank, tp_rank, tp_size


def maybe_run_prefix_sharing_deltanet(
    attention_module: Any,
    state_update: Any,
    packed_seq_params: Any,
) -> Any | None:
    """Build prefix-expanded GatedDeltaNet recurrent states for linear attention layers.

    This hook is called from linear attention layers (e.g., Qwen3.6 HybridAttention)
    where GatedDeltaNet replaces standard KV-based attention. The state update tensor
    is the recurrent update from the current forward pass; prefix sharing reuses the
    provider's accumulated state trajectory so reusers can start from the prefix
    boundary without recomputing the full prefix.

    Returns the expanded state output tensor, or ``None`` when prefix sharing is not
    active (caller should use the normal Megatron path).
    """
    ctx = current_prefix_sharing_context()
    if ctx is None:
        return None
    if not ctx.prefix_sharing_plan.has_sharing:
        return None
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        logger.debug("[PS][deltanet] skipping: no THD packed_seq_params")
        return None

    backend = ctx.backend or TorchReferenceBackend()
    _, tp_rank, _ = _read_parallel_rank_info()
    # Megatron layer_number is 1-indexed; convert to 0-indexed for ModelSpec
    layer_id = _normalize_layer_number(attention_module)

    # Only process linear attention layers
    if layer_id < 0:
        logger.debug("[PS][deltanet] skipping: could not determine layer_number")
        return None
    model_spec = ctx.model_spec
    if model_spec is not None:
        layer_type = model_spec.layer_type(layer_id)
        if layer_type != AttentionLayerType.LINEAR_ATTENTION:
            return None

    packed_batch_layout = ctx.packed_batch_layout

    expanded_state = backend.build_deltanet_states(
        state_update,
        ctx.deltanet_store,
        ctx.prefix_sharing_plan,
        packed_batch_layout=packed_batch_layout,
        layer_id=layer_id,
        tp_rank=tp_rank,
    )
    return expanded_state
