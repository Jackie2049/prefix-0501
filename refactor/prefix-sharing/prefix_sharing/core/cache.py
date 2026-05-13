"""Prefix K/V cache for one forward/backward lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefixKVCacheKey:
    forward_id: int
    micro_batch_id: int
    layer_id: int
    provider_batch_index: int
    tp_rank: int = 0


@dataclass(frozen=True)
class PrefixKVEntry:
    key: Any
    value: Any
    prefix_len: int


class PrefixKVCache:
    """Lifecycle-bound cache that never detaches tensors."""

    def __init__(self) -> None:
        self._entries: dict[PrefixKVCacheKey, PrefixKVEntry] = {}
        self._closed = False

    def store(
        self,
        cache_key: PrefixKVCacheKey,
        *,
        key: Any,
        value: Any,
        prefix_len: int,
        overwrite: bool = False,
    ) -> None:
        self._ensure_open()
        if prefix_len < 0:
            raise ValueError("prefix_len must be >= 0")
        if cache_key in self._entries and not overwrite:
            raise KeyError(f"prefix KV already exists for {cache_key}")
        self._entries[cache_key] = PrefixKVEntry(key=key, value=value, prefix_len=prefix_len)

    def load(self, cache_key: PrefixKVCacheKey) -> PrefixKVEntry:
        self._ensure_open()
        try:
            return self._entries[cache_key]
        except KeyError as exc:
            raise KeyError(f"missing prefix KV for {cache_key}") from exc

    def contains(self, cache_key: PrefixKVCacheKey) -> bool:
        return cache_key in self._entries

    def clear(self) -> None:
        self._entries.clear()

    def close(self) -> None:
        self.clear()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def size(self) -> int:
        return len(self._entries)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PrefixKVCache is closed")
