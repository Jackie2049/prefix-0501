"""Prefix K/V cache for one forward/backward lifecycle.

This module manages **logical** key/value tensors produced during prefix sharing
within a single forward/backward pass. It sits downstream of planning
(:mod:`prefix_sharing.core.planner`) and is consumed by attention backends
(e.g. :mod:`prefix_sharing.backends.torch_ref`) and integration context
(:mod:`prefix_sharing.integrations.context`). The cache does not decide reuse
relations or prefix lengths; it only stores and retrieves K/V by a composite key.

Core Responsibilities:
    1. **Isolate K/V by scope**: index entries with ``forward_id``,
       ``micro_batch_id``, ``layer_id``, ``sample_idx_in_batch``, and
       ``tp_rank`` so micro-batches, layers, and tensor-parallel ranks do not
       collide.
    2. **Store full logical K/V**: keep provider or expanded reuser tensors
       intact; reusers slice by their own ``prefix_len`` when loading a provider
       entry (``group_id`` is intentionally absent from the cache key).
    3. **Lifecycle management**: ``clear`` / ``close`` bound the cache to one
       runtime context; after ``close``, ``store`` and ``load`` raise.

Key Concepts:
    - Provider entry: K/V written when a row is not a reuser, keyed by its batch
      index.
    - Reuser entry: K/V built by concatenating a provider prefix slice with the
      row's suffix, then stored under the reuser's batch index for transitive
      reuse in deeper layers.
    - ``prefix_len`` on :class:`CachedPrefixKV`: metadata for how much of the
      stored tensor counts as prefix when slicing; backends use it with per-row
      ``prefix_lens`` from batch metadata.

Key Components:
    - :class:`PrefixKVSlotId`: Immutable identifier for one cache slot.
    - :class:`CachedPrefixKV`: ``key_tensor``, ``value_tensor``, and ``prefix_len`` for one slot.
    - :class:`PrefixKVCache`: Dict-backed store with ``store``, ``load``,
      ``contains``, ``clear``, and ``close``.

Design Principles:
    - **Never detach**: cached tensors must retain the autograd graph so gradients
      flow correctly through shared prefix K/V.
    - **No overwrite by default**: duplicate keys raise unless ``overwrite=True``
      (backends set this when updating expanded reuser entries).
    - **Framework-agnostic**: ``key`` / ``value`` are ``Any``; no PyTorch import
      in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefixKVSlotId:
    """Unique slot id for a cached logical K/V tensor within one runtime context."""

    forward_id: int
    micro_batch_id: int
    layer_id: int
    sample_idx_in_batch: int
    tp_rank: int = 0


@dataclass(frozen=True)
class CachedPrefixKV:
    """Cached logical prefix K/V tensors plus the prefix slice length."""

    key_tensor: Any
    value_tensor: Any
    prefix_len: int


class PrefixKVCache:
    """Lifecycle-bound cache that never detaches tensors."""

    def __init__(self) -> None:
        self._entries: dict[PrefixKVSlotId, CachedPrefixKV] = {}
        self._closed = False

    def store(
        self,
        slot_id: PrefixKVSlotId,
        *,
        key_tensor: Any,
        value_tensor: Any,
        prefix_len: int,
        overwrite: bool = False,
    ) -> None:
        self._ensure_open()
        if prefix_len < 0:
            raise ValueError("prefix_len must be >= 0")
        if slot_id in self._entries and not overwrite:
            raise KeyError(f"prefix KV already exists for {slot_id}")
        self._entries[slot_id] = CachedPrefixKV(
            key_tensor=key_tensor,
            value_tensor=value_tensor,
            prefix_len=prefix_len,
        )

    def load(self, slot_id: PrefixKVSlotId) -> CachedPrefixKV:
        self._ensure_open()
        try:
            return self._entries[slot_id]
        except KeyError as exc:
            raise KeyError(f"missing prefix KV for {slot_id}") from exc

    def contains(self, slot_id: PrefixKVSlotId) -> bool:
        return slot_id in self._entries

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
