"""Pure PyTorch reference backend.

This module imports torch lazily so the package can be developed and tested in
CPU environments where PyTorch is not installed. Tests that exercise this module
skip automatically when torch is unavailable.
"""

from __future__ import annotations

import math
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import PrefixKVSlotId, PrefixKVStore


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("TorchReferenceBackend requires PyTorch") from exc
    return torch


class TorchReferenceBackend:
    capabilities = BackendCapabilities(
        name="torch_ref",
        supports_cpu=True,
        supports_cuda=True,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
    )

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)

    def apply_rope(
        self,
        query: Any,
        key: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        rope_fn: Any | None = None,
        **_: Any,
    ) -> tuple[Any, Any]:
        if rope_fn is None:
            return query, key
        return rope_fn(query, key, prefix_sharing_plan.q_position_offsets, prefix_sharing_plan.kv_position_offsets)

    def build_kv(
        self,
        key: Any,
        value: Any,
        store: PrefixKVStore,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
        stats: PrefixSharingStats | None = None,
    ) -> tuple[Any, Any]:
        torch = _torch()
        layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        # Input K/V still follow the framework's padded packed layout; only
        # valid tokens may enter the store or expanded KV.
        key_rows = _split_packed(key, layout.padded_lengths)
        value_rows = _split_packed(value, layout.padded_lengths)
        expanded_keys = []
        expanded_values = []
        store_count = 0
        reuse_count = 0
        reuse_hit_count = 0
        reuse_miss_count = 0
        stored_tokens = 0
        reused_prefix_tokens = 0
        # This loop relies on the current online detector invariant that a provider
        # appears before every reuser that loads from it. All rows' QKV tensors have
        # already been produced in parallel by this point; the ordering here only
        # controls KV assembly before attention. Do not reorder or parallelize this
        # loop unless provider dependencies are handled explicitly, e.g. by a
        # topology-aware build phase.
        for batch_index, (key_row, value_row) in enumerate(zip(key_rows, value_rows)):
            valid_length = layout.valid_lengths[batch_index]
            valid_key_row = key_row[:valid_length]
            valid_value_row = value_row[:valid_length]
            if not prefix_sharing_plan.is_reuser(batch_index):
                slot_id = PrefixKVSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    tp_rank,
                )
                # Publish this row's KV so later reusers in this micro-batch can load it.
                store.store(
                    slot_id,
                    key_tensor=valid_key_row,
                    value_tensor=valid_value_row,
                    prefix_len=valid_key_row.shape[0],
                    overwrite=True,
                )
                store_count += 1
                stored_tokens += int(valid_key_row.shape[0])
                expanded_keys.append(valid_key_row)
                expanded_values.append(valid_value_row)
            else:
                provider = prefix_sharing_plan.provider_index[batch_index]
                provider_slot_id = PrefixKVSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    provider,
                    tp_rank,
                )
                # Load the already-published provider KV before building this reuser's expanded KV.
                reuse_count += 1
                try:
                    entry = store.load(provider_slot_id)
                except KeyError:
                    reuse_miss_count += 1
                    if stats is not None:
                        stats.record_attention_kv_build(
                            layer_id=layer_id,
                            store_count=store_count,
                            reuse_count=reuse_count,
                            reuse_hit_count=reuse_hit_count,
                            reuse_miss_count=reuse_miss_count,
                            stored_tokens=stored_tokens,
                            reused_prefix_tokens=reused_prefix_tokens,
                            expanded_kv_tokens=sum(int(row.shape[0]) for row in expanded_keys),
                            valid_q_tokens=layout.total_valid_length,
                            padded_q_tokens=layout.total_padded_length,
                        )
                    raise
                reuse_hit_count += 1
                prefix_len = prefix_sharing_plan.prefix_lens[batch_index]
                reused_prefix_tokens += int(prefix_len)
                expanded_key = torch.cat([entry.key_tensor[:prefix_len], valid_key_row], dim=0)
                expanded_value = torch.cat([entry.value_tensor[:prefix_len], valid_value_row], dim=0)
                own_slot_id = PrefixKVSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    tp_rank,
                )
                # Publish the expanded reuser KV because a later row may reuse this longer prefix.
                store.store(
                    own_slot_id,
                    key_tensor=expanded_key,
                    value_tensor=expanded_value,
                    prefix_len=expanded_key.shape[0],
                    overwrite=True,
                )
                store_count += 1
                stored_tokens += int(expanded_key.shape[0])
                expanded_keys.append(expanded_key)
                expanded_values.append(expanded_value)
        if stats is not None:
            stats.record_attention_kv_build(
                layer_id=layer_id,
                store_count=store_count,
                reuse_count=reuse_count,
                reuse_hit_count=reuse_hit_count,
                reuse_miss_count=reuse_miss_count,
                stored_tokens=stored_tokens,
                reused_prefix_tokens=reused_prefix_tokens,
                expanded_kv_tokens=sum(int(row.shape[0]) for row in expanded_keys),
                valid_q_tokens=layout.total_valid_length,
                padded_q_tokens=layout.total_padded_length,
            )
        return torch.cat(expanded_keys, dim=0), torch.cat(expanded_values, dim=0)

    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        **_: Any,
    ) -> Any:
        torch = _torch()
        layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        # Query keeps the framework packed shape, so split by padded lengths.
        # K/V were already depadded and prefix-expanded by build_kv().
        query_rows = _split_packed(query, layout.padded_lengths)
        key_rows = _split_packed(key, prefix_sharing_plan.expanded_lengths_kv)
        value_rows = _split_packed(value, prefix_sharing_plan.expanded_lengths_kv)
        outputs = []
        for batch_index, (q_row, k_row, v_row) in enumerate(zip(query_rows, key_rows, value_rows)):
            valid_length = layout.valid_lengths[batch_index]
            # Padding query slots are layout-only; they must not participate in attention.
            q_valid = q_row[:valid_length]
            prefix_len = prefix_sharing_plan.q_position_offsets[batch_index]
            if valid_length == 0:
                outputs.append(torch.zeros_like(q_row))
                continue
            mask = _causal_q_kv_mask(
                q_len=q_valid.shape[0],
                kv_len=k_row.shape[0],
                q_start=prefix_len,
                device=q_valid.device,
            )
            valid_output = _attention_row(q_valid, k_row, v_row, mask)
            if valid_length == q_row.shape[0]:
                outputs.append(valid_output)
                continue
            padded_output = torch.zeros_like(q_row)
            padded_output[:valid_length] = valid_output
            outputs.append(padded_output)
        return torch.cat(outputs, dim=0)


def _split_packed(tensor: Any, lengths: list[int]) -> list[Any]:
    torch = _torch()
    if not lengths:
        return []
    if sum(lengths) != tensor.shape[0]:
        raise ValueError("packed tensor first dimension does not match lengths")
    return list(torch.split(tensor, lengths, dim=0))


def _causal_q_kv_mask(q_len: int, kv_len: int, q_start: int, device: Any) -> Any:
    torch = _torch()
    q_positions = torch.arange(q_start, q_start + q_len, device=device).unsqueeze(1)
    kv_positions = torch.arange(0, kv_len, device=device).unsqueeze(0)
    return kv_positions <= q_positions


def _attention_row(q_row: Any, k_row: Any, v_row: Any, mask: Any) -> Any:
    torch = _torch()
    scale = math.sqrt(q_row.shape[-1])
    if q_row.dim() == 2:
        scores = q_row @ k_row.transpose(-1, -2) / scale
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        return probs @ v_row
    if q_row.dim() != 3:
        raise ValueError("TorchReferenceBackend attention expects packed rows with 2 or 3 dims")

    q_heads = q_row.shape[1]
    kv_heads = k_row.shape[1]
    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError("query heads must be a multiple of kv heads for grouped-query attention")
        repeat = q_heads // kv_heads
        k_row = k_row.repeat_interleave(repeat, dim=1)
        v_row = v_row.repeat_interleave(repeat, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q_row, k_row) / scale
    scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v_row)
