"""Trim per-row batches after prefix-sharing planning.

This module sits **after** :mod:`prefix_sharing.core.planner` and consumes
:class:`~prefix_sharing.core.metadata.PrefixSharingBatchMeta`. The planner encodes
*which* token spans each row keeps on the Q path and how packed lengths line up;
this module applies those decisions to concrete per-row data: it slices
``input_ids``, ``labels``, and ``loss_masks`` into trimmed rows, and builds a
packed ``flattened`` view with ``cu_seqlens`` for variable-length layouts.

Core Responsibilities:
    1. **Trim rows by metadata ranges**: for each batch index, take the contiguous
       span prescribed by ``input_keep_ranges``, ``label_keep_ranges``, or
       ``loss_mask_keep_ranges`` and validate bounds against the source row.
    2. **Emit packed batch views**: return :class:`TrimmedBatch` with per-row
       lists, a single concatenation in batch order, and cumulative lengths
       ``[0, len_0, len_0+len_1, ...]`` suitable for packed execution conventions.

Key Concepts:
    - **Keep ranges**: half-open ``(start, end)`` slices applied row-wise; inputs,
      labels, and masks may use different ranges when label/mask semantics differ
      from raw tokens.

Key Components:
    - :class:`TrimmedBatch`: Immutable container for trimmed ``rows``,
      ``flattened``, and ``cu_seqlens``.
    - :func:`trim_batch`, :func:`trim_inputs`, :func:`trim_labels`,
      :func:`trim_loss_masks`: Row slicing helpers; the ``trim_*`` entry points
      wire the correct meta field into :func:`trim_batch`.

Design Principles:
    - **Meta-driven only**: no layout heuristics here; all spans come from
      ``PrefixSharingBatchMeta`` produced by the planner.
    - **Sequence-generic**: type parameter ``T`` supports token ids, mask weights,
      or other per-position scalars.
    - **Backend-agnostic**: returns Python lists; integration stacks map the same
      indexing to tensors without changing metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from prefix_sharing.core.metadata import PrefixSharingBatchMeta


T = TypeVar("T")


@dataclass(frozen=True)
class TrimmedBatch(Generic[T]):
    """A batch after applying ``input_keep_ranges`` from metadata."""

    rows: list[list[T]]
    flattened: list[T]
    cu_seqlens: list[int]


def _slice_row(row: Sequence[T], keep_range: tuple[int, int]) -> list[T]:
    start, end = keep_range
    if start < 0 or end < start or end > len(row):
        raise ValueError(f"invalid keep range {keep_range} for row length {len(row)}")
    return list(row[start:end])


def trim_batch(rows: Sequence[Sequence[T]], ranges: Sequence[tuple[int, int]]) -> TrimmedBatch[T]:
    if len(rows) != len(ranges):
        raise ValueError("rows and ranges must have same length")
    trimmed_rows: list[list[T]] = []
    flattened: list[T] = []
    cu_seqlens = [0]
    total = 0
    for row, keep_range in zip(rows, ranges):
        trimmed = _slice_row(row, keep_range)
        trimmed_rows.append(trimmed)
        flattened.extend(trimmed)
        total += len(trimmed)
        cu_seqlens.append(total)
    return TrimmedBatch(rows=trimmed_rows, flattened=flattened, cu_seqlens=cu_seqlens)


def trim_inputs(input_ids: Sequence[Sequence[T]], meta: PrefixSharingBatchMeta) -> TrimmedBatch[T]:
    return trim_batch(input_ids, meta.input_keep_ranges)


def trim_labels(labels: Sequence[Sequence[T]], meta: PrefixSharingBatchMeta) -> TrimmedBatch[T]:
    return trim_batch(labels, meta.label_keep_ranges)


def trim_loss_masks(loss_masks: Sequence[Sequence[T]], meta: PrefixSharingBatchMeta) -> TrimmedBatch[T]:
    return trim_batch(loss_masks, meta.loss_mask_keep_ranges)
