"""Batch materialization after planning: slice sequences and align logprobs.

This module sits **after** :mod:`prefix_sharing.core.planner` and consumes
:class:`~prefix_sharing.core.metadata.PrefixSharingBatchMeta`. The planner encodes
*which* token spans each row keeps on the Q path and how packed lengths line up;
this module applies those decisions to concrete per-row data: it slices
``input_ids``, ``labels``, and ``loss_masks`` into trimmed rows, builds a packed
``flattened`` view with ``cu_seqlens`` for variable-length layouts, and—when
``prefix_last_restore`` is non-empty—reassembles per-row logprobs so reuse rows
include the provider's **prefix-last** prediction for the first suffix token.

Core Responsibilities:
    1. **Trim rows by metadata ranges**: for each batch index, take the contiguous
       span prescribed by ``input_keep_ranges``, ``label_keep_ranges``, or
       ``loss_mask_keep_ranges`` and validate bounds against the source row.
    2. **Emit packed batch views**: return :class:`TrimmedBatch` with per-row
       lists, a single concatenation in batch order, and cumulative lengths
       ``[0, len_0, len_0+len_1, ...]`` suitable for packed execution conventions.
    3. **Prefix-last restore for RL / logprob pipelines**: insert the provider
       prefix-last logprob at the slot indicated by each
       :class:`~prefix_sharing.core.planner.PrefixLastRestoreSpec`, and gather
       those provider values for tests or tensor code that mirrors the same
       indexing contract.

Key Concepts:
    - **Keep ranges**: half-open ``(start, end)`` slices applied row-wise; inputs,
      labels, and masks may use different ranges when label/mask semantics differ
      from raw tokens.
    - **Reuse logprob gap**: on reuser rows the first kept Q position predicts the
      *second* suffix token; the missing logprob for the *first* suffix token
      comes from the provider row at ``prefix_len - 1`` (encoded in restore specs).

Key Components:
    - :class:`TrimmedBatch`: Immutable container for trimmed ``rows``,
      ``flattened``, and ``cu_seqlens``.
    - :func:`trim_batch`, :func:`trim_inputs`, :func:`trim_labels`,
      :func:`trim_loss_masks`: Row slicing helpers; the ``trim_*`` entry points
      wire the correct meta field into :func:`trim_batch`.
    - :func:`restore_prefix_last_logprobs`: Mutates per-row copies to insert
      provider prefix-last logprobs at ``output_slot`` for each restore spec.
    - :func:`build_provider_prefix_last_values`: Tensor-agnostic gather of
      provider prefix-last scalars indexed by ``reuse_idx_in_batch`` (CPU tests
      and reference for device-side gathers).

Design Principles:
    - **Meta-driven only**: no layout heuristics here; all spans come from
      ``PrefixSharingBatchMeta`` produced by the planner.
    - **Sequence-generic**: type parameter ``T`` supports token ids, mask weights,
      or other per-position scalars; logprob helpers are float-specific where needed.
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


def restore_prefix_last_logprobs(
    suffix_logprobs: Sequence[Sequence[float]],
    provider_prefix_last_logprobs: Sequence[float],
    meta: PrefixSharingBatchMeta,
) -> list[list[float]]:
    """Assemble response logprobs with Prefix-Last Restore semantics.

    ``suffix_logprobs`` contains logprobs produced by kept query positions. For
    provider samples it is already complete. For reuse samples, slot 0 predicts
    the second suffix token, so this function prepends the provider prefix-last
    logprob that predicts the first suffix token.
    """

    if len(suffix_logprobs) != meta.batch_size:
        raise ValueError("suffix_logprobs length must equal batch size")
    if len(provider_prefix_last_logprobs) != meta.batch_size:
        raise ValueError("provider_prefix_last_logprobs length must equal batch size")

    restored = [list(row) for row in suffix_logprobs]
    for spec in meta.prefix_last_restore:
        restored_value = provider_prefix_last_logprobs[spec.reuse_idx_in_batch]
        row = restored[spec.reuse_idx_in_batch]
        if spec.output_slot < 0 or spec.output_slot > len(row):
            raise ValueError("restore output_slot out of range")
        row.insert(spec.output_slot, restored_value)
    return restored


def build_provider_prefix_last_values(
    provider_values_by_batch: Sequence[Sequence[T]],
    meta: PrefixSharingBatchMeta,
) -> list[T | None]:
    """Gather provider prefix-last values for every reuse batch index.

    This helper is tensor-agnostic and is used by CPU tests. Integration code can
    perform the same mapping on tensors without changing the metadata contract.
    """

    values: list[T | None] = [None] * meta.batch_size
    for spec in meta.prefix_last_restore:
        provider_row = provider_values_by_batch[spec.provider_idx_in_batch]
        values[spec.reuse_idx_in_batch] = provider_row[spec.provider_prefix_last_pos]
    return values
