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

    q_pos_emb, k_pos_emb = rotary_pos_emb
    query, key = _apply_positioned_rope(
        attention_module,
        query,
        key,
        q_pos_emb,
        k_pos_emb,
        packed_batch_layout.packed_position_ids,
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
        stats=ctx.stats,
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
    output = attention_module.linear_proj(core_attn_out)  # (tensor, bias) tuple

    # --- diag dump (ON attention) ---
    try:
        from prefix_sharing.tools.diagnostic_dump import dump_attn_on
        dump_attn_on(output[0], packed_seq_params, ctx.prefix_sharing_plan,
                     attention_module.layer_number,
                     attention_module.config.num_layers)
    except Exception as e:
        prefix_log.warning(f"last-attn dump (ON) failed: {e}")
    # ---

    return output


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
