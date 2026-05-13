"""Pure PyTorch reference backend.

This module imports torch lazily so the package can be developed and tested in
CPU environments where PyTorch is not installed. Tests that exercise this module
skip automatically when torch is unavailable.
"""

from __future__ import annotations

import math
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.core.cache import PrefixKVCache, PrefixKVCacheKey
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.metadata import PrefixSharingBatchMeta


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
        meta: PrefixSharingBatchMeta,
        *,
        rope_fn: Any | None = None,
        **_: Any,
    ) -> tuple[Any, Any]:
        if rope_fn is None:
            return query, key
        return rope_fn(query, key, meta.q_position_offsets, meta.kv_position_offsets)

    def build_kv(
        self,
        key: Any,
        value: Any,
        cache: PrefixKVCache,
        meta: PrefixSharingBatchMeta,
        *,
        layer_id: int,
        tp_rank: int = 0,
    ) -> tuple[Any, Any]:
        torch = _torch()
        key_rows = _split_packed(key, meta.kept_lengths_q)
        value_rows = _split_packed(value, meta.kept_lengths_q)
        expanded_keys = []
        expanded_values = []
        for batch_index, (key_row, value_row) in enumerate(zip(key_rows, value_rows)):
            if not meta.is_reuser(batch_index):
                cache_key = PrefixKVCacheKey(
                    meta.forward_id,
                    meta.micro_batch_id,
                    layer_id,
                    batch_index,
                    tp_rank,
                )
                cache.store(
                    cache_key,
                    key=key_row,
                    value=value_row,
                    prefix_len=key_row.shape[0],
                    overwrite=True,
                )
                expanded_keys.append(key_row)
                expanded_values.append(value_row)
            else:
                provider = meta.provider_index[batch_index]
                provider_cache_key = PrefixKVCacheKey(
                    meta.forward_id,
                    meta.micro_batch_id,
                    layer_id,
                    provider,
                    tp_rank,
                )
                entry = cache.load(provider_cache_key)
                prefix_len = meta.prefix_lens[batch_index]
                expanded_key = torch.cat([entry.key[:prefix_len], key_row], dim=0)
                expanded_value = torch.cat([entry.value[:prefix_len], value_row], dim=0)
                own_cache_key = PrefixKVCacheKey(
                    meta.forward_id,
                    meta.micro_batch_id,
                    layer_id,
                    batch_index,
                    tp_rank,
                )
                cache.store(
                    own_cache_key,
                    key=expanded_key,
                    value=expanded_value,
                    prefix_len=expanded_key.shape[0],
                    overwrite=True,
                )
                expanded_keys.append(expanded_key)
                expanded_values.append(expanded_value)
        return torch.cat(expanded_keys, dim=0), torch.cat(expanded_values, dim=0)

    def attention(self, query: Any, key: Any, value: Any, meta: PrefixSharingBatchMeta, **_: Any) -> Any:
        torch = _torch()
        query_rows = _split_packed(query, meta.kept_lengths_q)
        key_rows = _split_packed(key, meta.expanded_lengths_kv)
        value_rows = _split_packed(value, meta.expanded_lengths_kv)
        outputs = []
        for batch_index, (q_row, k_row, v_row) in enumerate(zip(query_rows, key_rows, value_rows)):
            prefix_len = meta.q_position_offsets[batch_index]
            mask = _causal_q_kv_mask(
                q_len=q_row.shape[0],
                kv_len=k_row.shape[0],
                q_start=prefix_len,
                device=q_row.device,
            )
            outputs.append(_attention_row(q_row, k_row, v_row, mask))
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
