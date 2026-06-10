"""Block-causal mask construction for prefix-sharing attention.

The mask produced here is the single source of truth for *which KV positions
each Q position is allowed to attend to* under prefix sharing. It encodes
**absolute-position** semantics:

* Provider / standalone row: standard lower-triangular causal mask.
* Reuser row: prefix columns are all-visible, suffix columns are lower-triangular.

This is the mask that ``flash_attn_varlen_func(causal=True)`` cannot express,
because FA's ``causal=True`` uses *segment-relative* positions: for a reuser
whose Q is the suffix (absolute positions P..P+L-1) and whose KV is
prefix+suffix (positions 0..P+L-1), FA would only let Q[i] see KV[0..i] —
that is, the suffix KV columns (positions P..P+L-1) end up *masked*,
which is mathematically wrong.

Backends that respect an explicit ``attention_bias`` / ``atten_mask`` argument
(NPU ``npu_fusion_attention``, CUDA ``transformer_engine.DotProductAttention``)
all consume the boolean mask produced here, possibly after a dtype conversion.
"""

from __future__ import annotations

from typing import Any

from prefix_sharing.core.planner import PrefixSharingPlan

def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("block_causal_mask requires PyTorch") from exc
    return torch

def build_block_causal_mask(
    prefix_sharing_plan: PrefixSharingPlan,
    device: Any,
) -> Any:
    """Construct a packed (total_q, total_kv) boolean mask.

    The returned tensor uses ``True = masked (invisible)`` / ``False = visible``,
    matching the convention of ``npu_fusion_attention``'s ``atten_mask`` argument.
    TE / flash-attn wrappers convert this to a float bias separately.

    The mask is *block-diagonal*: each sample's Q rows are non-zero only in
    that sample's KV columns; cross-sample attention is always masked.
    Within a sample:
        * provider: standard causal (lower-triangular in absolute positions).
        * reuser  : prefix columns all-visible + suffix columns lower-triangular.
    """
    torch = _torch()
    plan = prefix_sharing_plan
    total_q = plan.cu_seqlens_q[-1]
    total_kv = plan.cu_seqlens_kv[-1]
    mask = torch.ones(total_q, total_kv, dtype=torch.bool, device=device)

    for batch_index in range(plan.batch_size):
        q_lo, q_hi = plan.q_range_for_batch(batch_index)
        kv_lo, kv_hi = plan.kv_range_for_batch(batch_index)
        q_len = q_hi - q_lo
        kv_len = kv_hi - kv_lo
        if q_len == 0 or kv_len == 0:
            continue

        if plan.is_reuser(batch_index):
            prefix_len = plan.prefix_lens[batch_index]
            # Prefix columns: all visible for every Q row in this sample.
            if prefix_len > 0:
                mask[q_lo:q_hi, kv_lo:kv_lo + prefix_len] = False
            # Suffix columns: causal in absolute positions.
            # Q absolute position = prefix_len + i (i in 0..q_len-1)
            # KV suffix absolute position = prefix_len + j (j in 0..kv_len-prefix_len-1)
            suffix_len_kv = kv_len - prefix_len
            eye = torch.ones(q_len, suffix_len_kv, dtype=torch.bool, device=device)
            tri = torch.tril(eye, diagonal=0)
            mask[q_lo:q_hi, kv_lo + prefix_len:kv_hi] = ~tri
        else:
            eye = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
            tri = torch.tril(eye, diagonal=0)
            mask[q_lo:q_hi, kv_lo:kv_hi] = ~tri
    return mask

def mask_to_te_bias(mask: Any, dtype: Any) -> Any:
    """Convert boolean mask to a float attention bias for Transformer Engine.

    TE's ``DotProductAttention(core_attention_bias=...)`` expects a float tensor
    where ``0.0`` means *visible* and ``-inf`` means *masked*. Input mask uses
    ``True = masked``.
    """
    torch = _torch()
    neg_inf = torch.tensor(float("-inf"), dtype=dtype, device=mask.device)
    zero = torch.tensor(0.0, dtype=dtype, device=mask.device)
    return torch.where(mask, neg_inf, zero)
