"""Runtime hook used by the minimal Megatron attention patch."""

from __future__ import annotations

from typing import Any

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
    *,
    mscale: float = 1.0,
) -> tuple[Any, Any] | None:
    """Run prefix-sharing attention when a runtime context is active.

    Returns ``None`` for the normal Megatron path. When active, this function
    owns RoPE, KV expansion, causal masking, and output projection.
    """

    ctx = current_prefix_sharing_context()
    if ctx is None:
        return None
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        raise RuntimeError("prefix sharing phase 1 requires packed_seq_params.qkv_format='thd'")
    if rotary_pos_emb is None:
        raise RuntimeError("prefix sharing phase 1 requires rotary_pos_emb")
    if ctx.kept_position_ids is None:
        raise RuntimeError("prefix sharing context is missing kept_position_ids")

    q_pos_emb, k_pos_emb = rotary_pos_emb
    query, key = _apply_positioned_rope(
        attention_module,
        query,
        key,
        q_pos_emb,
        k_pos_emb,
        ctx.kept_position_ids,
        mscale=mscale,
    )

    backend = ctx.backend or TorchReferenceBackend()
    tp_rank = _tensor_parallel_rank()
    layer_id = int(getattr(attention_module, "layer_number", 0) or 0)
    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        ctx.store,
        ctx.meta,
        layer_id=layer_id,
        tp_rank=tp_rank,
    )
    core_attn_out = backend.attention(
        query,
        expanded_key,
        expanded_value,
        ctx.meta,
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
    kept_position_ids: Any,
    *,
    mscale: float,
) -> tuple[Any, Any]:
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    positions = kept_position_ids.to(device=query.device, dtype="long")
    if q_pos_emb is not None:
        q_freqs = q_pos_emb.index_select(0, positions)
        query = apply_rotary_pos_emb(
            query.unsqueeze(1),
            q_freqs,
            config=attention_module.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=attention_module.pg_collection.cp,
        ).squeeze(1)
    if k_pos_emb is not None:
        k_freqs = k_pos_emb.index_select(0, positions)
        key = apply_rotary_pos_emb(
            key.unsqueeze(1),
            k_freqs,
            config=attention_module.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=attention_module.pg_collection.cp,
        ).squeeze(1)
    return query, key


def _tensor_parallel_rank() -> int:
    try:
        from megatron.core import parallel_state

        return int(parallel_state.get_tensor_model_parallel_rank())
    except Exception:
        return 0
