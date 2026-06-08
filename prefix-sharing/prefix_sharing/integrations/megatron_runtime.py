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

    # Dispatch based on data format:
    # THD mode: packed_seq_params with qkv_format='thd' → 3D packed data
    # BSHD mode: packed_seq_params is None → 4D (sq, b, h, hn) data
    is_thd = packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd"
    is_bshd = packed_seq_params is None

    if is_thd:
        # --- THD path (existing) ---
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
        layer_id = _normalize_layer_number(attention_module)

        if layer_id < 0:
            return None
        model_spec = ctx.model_spec
        if model_spec is not None:
            layer_type = model_spec.layer_type(layer_id)
            if layer_type == AttentionLayerType.LINEAR_ATTENTION:
                return None

        prefix_log.debug(
            "[PS][attention][rank=%s tp=%s/%s layer=%s] THD query=%s key=%s value=%s",
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

    elif is_bshd:
        # --- BSHD path (new) ---
        return _run_bshd_prefix_sharing_attention(
            attention_module, query, key, value,
            rotary_pos_emb, ctx,
        )

    else:
        # packed_seq_params exists but qkv_format is not 'thd' — unsupported
        prefix_log.debug("prefix hook: unsupported packed_seq_params format, falling through")
        return None


def _run_bshd_prefix_sharing_attention(
    attention_module: Any,
    query: Any,
    key: Any,
    value: Any,
    rotary_pos_emb: Any,
    ctx: Any,
) -> tuple[Any, Any] | None:
    """BSHD-mode prefix-sharing attention for Megatron SelfAttention.

    Data format: (sq, b, h, hn) — standard Megatron BSHD format.

    Two-pass approach:
    - Prefix pass: store K/V (after RoPE), run flash_attn on prefix tokens
    - Suffix pass: load stored prefix K/V, expand to batch size,
      concatenate with suffix K/V, run flash_attn_varlen_func with
      causal=True (handles "prefill with prefix" natively).

    flash_attn_varlen_func with causal=True and different Q/K lengths
    correctly handles the causal mask: suffix Q[i] at absolute position
    prefix_len+i can attend to all prefix K tokens (positions 0..prefix_len-1)
    AND causal suffix K tokens (positions prefix_len..prefix_len+i).
    """
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from prefix_sharing.core.prefix_store import PrefixActivationSlotId, PREFIX_STATE_TYPE_ATTENTION_KV

    layer_id = _normalize_layer_number(attention_module)
    if layer_id < 0:
        return None
    model_spec = ctx.model_spec
    if model_spec is not None:
        layer_type = model_spec.layer_type(layer_id)
        if layer_type == AttentionLayerType.LINEAR_ATTENTION:
            return None

    # Apply RoPE at correct absolute positions.
    # The PS hook intercepts BEFORE SelfAttention's RoPE, so we own RoPE.
    # For prefix pass: positions 0..PREFIX_LEN-1 (rotary_pos_emb is correct)
    # For suffix pass: positions PREFIX_LEN..total_len-1 (need offset correction)
    if rotary_pos_emb is None:
        logger.debug("[PS][bshd] skipping: no rotary_pos_emb for RoPE")
        return None

    # Build slot_id early (needed for contains check and both prefix/suffix passes)
    _, tp_rank, _ = _read_parallel_rank_info()
    slot_id = PrefixActivationSlotId(
        forward_id=ctx.prefix_sharing_plan.forward_id,
        micro_batch_id=ctx.prefix_sharing_plan.micro_batch_id,
        layer_id=layer_id,
        sample_idx_in_batch=0,
        prefix_state_type=PREFIX_STATE_TYPE_ATTENTION_KV,
        tp_rank=tp_rank,
    )

    # Check whether we're in suffix pass (prefix KV already stored) or prefix pass
    is_suffix_pass = ctx.store.contains(slot_id)

    if is_suffix_pass:
        # --- Suffix pass: apply RoPE at absolute positions PREFIX_LEN..total_len-1 ---
        # rotary_pos_emb covers positions 0..suffix_len-1 (from suffix forward's rotary_seq_len)
        # We need positions PREFIX_LEN..total_len-1. Extend and slice.
        q_pos_emb, k_pos_emb = rotary_pos_emb
        prefix_len_actual = ctx.store.load(slot_id).key_tensor.shape[0]  # stored prefix seq_len
        suffix_len = query.shape[0]
        total_len_needed = prefix_len_actual + suffix_len

        q_pos_emb_extended = _extend_rope_pos_emb(q_pos_emb, total_len_needed)
        k_pos_emb_extended = _extend_rope_pos_emb(k_pos_emb, total_len_needed)

        # Select the suffix range: positions PREFIX_LEN..total_len-1
        suffix_positions = torch.arange(prefix_len_actual, total_len_needed, device=query.device)
        q_pos_emb_suffix = q_pos_emb_extended.index_select(0, suffix_positions)
        k_pos_emb_suffix = k_pos_emb_extended.index_select(0, suffix_positions)

        query = apply_rotary_pos_emb(query, q_pos_emb_suffix, config=attention_module.config)
        key = apply_rotary_pos_emb(key, k_pos_emb_suffix, config=attention_module.config)
    else:
        # --- Prefix pass: apply RoPE at positions 0..PREFIX_LEN-1 (standard path) ---
        q_pos_emb, k_pos_emb = rotary_pos_emb
        query = apply_rotary_pos_emb(query, q_pos_emb, config=attention_module.config)
        key = apply_rotary_pos_emb(key, k_pos_emb, config=attention_module.config)

    
    num_heads = attention_module.num_attention_heads_per_partition
    kv_heads = attention_module.num_query_groups_per_partition
    head_dim = attention_module.hidden_size_per_attention_head
    num_kv_groups = num_heads // kv_heads
    b = query.shape[1]

    if ctx.store.contains(slot_id):
        # === Suffix pass: load prefix KV, expand, concatenate ===
        stored_kv = ctx.store.load(slot_id)
        prefix_key = stored_kv.key_tensor    # (P, 1_or_b, kv_heads, hn) or (P, kv_heads, hn)
        prefix_value = stored_kv.value_tensor

        # Normalize to 4D BSHD format
        if prefix_key.dim() == 3:
            prefix_key = prefix_key.unsqueeze(1)
            prefix_value = prefix_value.unsqueeze(1)

        prefix_len = prefix_key.shape[0]
        suffix_len = query.shape[0]

        # Expand prefix KV from batch=1 to match current batch size
        prefix_key_expanded = prefix_key.expand(prefix_len, b, -1, -1).contiguous()
        prefix_value_expanded = prefix_value.expand(prefix_len, b, -1, -1).contiguous()

        # Concatenate prefix + suffix KV along seq dim
        expanded_key = torch.cat([prefix_key_expanded, key], dim=0)   # (P+S, b, kv_heads, hn)
        expanded_value = torch.cat([prefix_value_expanded, value], dim=0)

        # GQA expand KV heads
        if num_kv_groups > 1:
            expanded_key = expanded_key.repeat_interleave(num_kv_groups, dim=2)
            expanded_value = expanded_value.repeat_interleave(num_kv_groups, dim=2)

        # Convert BSHD → THD for flash_attn_varlen_func
        query_flat = query.permute(1, 0, 2, 3).reshape(b * suffix_len, num_heads, head_dim).contiguous()
        expanded_key_flat = expanded_key.permute(1, 0, 2, 3).reshape(
            b * (prefix_len + suffix_len), num_heads, head_dim
        ).contiguous()
        expanded_value_flat = expanded_value.permute(1, 0, 2, 3).reshape(
            b * (prefix_len + suffix_len), num_heads, head_dim
        ).contiguous()

        cu_seqlens_q = torch.tensor(
            [0] + [suffix_len] * b, device=query.device, dtype=torch.int32
        ).cumsum(0).to(torch.int32)
        cu_seqlens_k = torch.tensor(
            [0] + [prefix_len + suffix_len] * b, device=query.device, dtype=torch.int32
        ).cumsum(0).to(torch.int32)

        input_dtype = query_flat.dtype
        if input_dtype == torch.float32:
            query_flat = query_flat.to(torch.float16)
            expanded_key_flat = expanded_key_flat.to(torch.float16)
            expanded_value_flat = expanded_value_flat.to(torch.float16)

        from flash_attn import flash_attn_varlen_func
        attn_output = flash_attn_varlen_func(
            query_flat, expanded_key_flat, expanded_value_flat,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=suffix_len, max_seqlen_k=prefix_len + suffix_len,
            dropout_p=0.0, causal=True,
        )
        attn_output = attn_output.to(input_dtype)

        # Reshape: (b*S, h, hn) → (S, b, h*hn) for linear_proj
        attn_output = attn_output.reshape(b, suffix_len, num_heads * head_dim)
        attn_output = attn_output.permute(1, 0, 2).contiguous()

        logger.debug(
            "[PS][bshd-attn-inject][layer=%s] Suffix pass: b=%d, suffix=%d, prefix=%d",
            layer_id, b, suffix_len, prefix_len,
        )
        return attention_module.linear_proj(attn_output)

    else:
        # === Prefix pass: store KV, run flash_attn ===
        sq = query.shape[0]

        # Store K/V in BSHD format (sq, b, kv_heads, hn)
        ctx.store.store(
            slot_id,
            key_tensor=key.contiguous(),
            value_tensor=value.contiguous(),
            prefix_len=sq,
        )

        logger.debug(
            "[PS][bshd-attn-store][layer=%s] Prefix pass: stored KV key=%s value=%s",
            layer_id, tuple(key.shape), tuple(value.shape),
        )

        # GQA expand KV heads for flash_attn
        if num_kv_groups > 1:
            key_expanded = key.repeat_interleave(num_kv_groups, dim=2)
            value_expanded = value.repeat_interleave(num_kv_groups, dim=2)
        else:
            key_expanded = key
            value_expanded = value

        # Convert BSHD → THD
        query_flat = query.permute(1, 0, 2, 3).reshape(b * sq, num_heads, head_dim).contiguous()
        key_flat = key_expanded.permute(1, 0, 2, 3).reshape(b * sq, num_heads, head_dim).contiguous()
        value_flat = value_expanded.permute(1, 0, 2, 3).reshape(b * sq, num_heads, head_dim).contiguous()

        cu_seqlens = torch.tensor([0] + [sq] * b, device=query.device, dtype=torch.int32).cumsum(0).to(torch.int32)

        input_dtype = query_flat.dtype
        if input_dtype == torch.float32:
            query_flat = query_flat.to(torch.float16)
            key_flat = key_flat.to(torch.float16)
            value_flat = value_flat.to(torch.float16)

        from flash_attn import flash_attn_varlen_func
        attn_output = flash_attn_varlen_func(
            query_flat, key_flat, value_flat,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=sq, max_seqlen_k=sq,
            dropout_p=0.0, causal=True,
        )
        attn_output = attn_output.to(input_dtype)

        attn_output = attn_output.reshape(b, sq, num_heads * head_dim)
        attn_output = attn_output.permute(1, 0, 2).contiguous()

        return attention_module.linear_proj(attn_output)


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
        if q_pos_emb.shape[0] < 2:
            # Can't recover RoPE frequency step from a single position.
            # Fall through: apply_rotary_pos_emb will use the available
            # positions and might truncate — rare edge case.
            pass
        else:
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
        if k_pos_emb.shape[0] < 2:
            pass
        else:
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


def _extend_rope_pos_emb(pos_emb: torch.Tensor, max_needed: int) -> torch.Tensor:
    """Extend a RotaryEmbedding pos_emb tensor to cover more positions.

    RoPE angles are linear in position: freqs[p] = p * inv_freq.
    The step (inv_freq) is recovered from pos_emb[1] - pos_emb[0].

    Args:
        pos_emb: [seq_len, 1, 1, head_dim] RoPE embedding tensor
        max_needed: total number of positions needed

    Returns:
        Extended pos_emb of shape [max_needed, 1, 1, head_dim]
    """
    if max_needed <= pos_emb.shape[0]:
        return pos_emb

    if pos_emb.shape[0] < 2:
        # Can't recover step from a single entry — rare edge case
        return pos_emb

    dim_half = pos_emb.shape[-1] // 2
    # The pos_emb format is interleaved: first half is cos-like, second half is sin-like
    # Step recovery must use the same half for both cos and sin components
    step = pos_emb[1:2, :, :, :dim_half] - pos_emb[0:1, :, :, :dim_half]
    extra_positions = torch.arange(
        pos_emb.shape[0], max_needed,
        device=pos_emb.device, dtype=pos_emb.dtype,
    )
    extra_angles = extra_positions[:, None, None, None] * step
    # Mirror the step for the second half (sin component)
    extra_emb = torch.cat([extra_angles, extra_angles], dim=-1)
    return torch.cat([pos_emb, extra_emb], dim=0)


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
    # THD-only path: require packed_seq_params with qkv_format='thd'
    # BSHD path: handled by GatedDeltaNetAttention.forward directly (carry state injection)
    if packed_seq_params is None:
        logger.debug("[PS][deltanet] skipping THD hook: no packed_seq_params (BSHD mode uses in-class hook)")
        return None
    if getattr(packed_seq_params, "qkv_format", None) != "thd":
        logger.debug("[PS][deltanet] skipping: packed_seq_params.qkv_format is not 'thd'")
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
