"""Prefix-sharing planning: turn detection output into batch metadata.

This module sits between **prefix detection** (:mod:`prefix_sharing.core.prefix_detector`)
and **downstream execution** (mapping, attention backends, RoPE). It does not
discover shared prefixes; it consumes :class:`~prefix_sharing.core.prefix_detector.PrefixDetectionResult`
and materializes per-row layout: what to keep on the Q path, cumulative sequence
lengths for packed tensors, position offsets, and optional prefix-last restore
hints for RL / logprob alignment.

Core Responsibilities:
    1. **Per-row layout after detection**: for each batch index, decide how that
       row participates in packed execution—whether the Q path uses the full
       sequence or only the post-prefix suffix, how long the row occupies on
       the KV path, and which contiguous token spans to keep for inputs, labels,
       and loss masks. Inputs are only ``provider_index``, ``prefix_lens``, and
       per-row original length; the result is encoded as lengths, cumulative
       seqlens, and keep ranges inside ``PrefixSharingBatchMeta``.
    2. Set ``q_position_offsets`` so reuser rows preserve correct absolute positions
       when the Q path only packs the suffix after a shared prefix.
    3. Emit :class:`PrefixLastRestoreSpec` entries when a reuser has both a
       non-zero prefix and a non-empty suffix.

Key Concepts:
    - Reuser (planner view): ``provider_index[i] != i`` and ``prefix_lens[i] > 0``.
      The Q path keeps ``input_ids[i][prefix_len:original_len)``; KV expanded
      length stays the full original length for backend conventions.
    - Provider or standalone row: full sequence on the Q path; offsets zero.

Key Components:
    - :class:`PrefixSharingPlanner`: Optional embedded :class:`~prefix_sharing.core.prefix_detector.TriePrefixDetector`;
      :meth:`~PrefixSharingPlanner.plan` runs detect-then-plan;
      :meth:`~PrefixSharingPlanner.plan_from_detection` accepts precomputed detection
      (tests or alternate detectors).
    - :func:`_cumsum`: Computes cumulative lengths for ``cu_seqlens_q`` and ``cu_seqlens_kv``.

Design Principles:
    - Single layout source: all ranges and lengths derive from detector fields
      plus original lengths; planner does not reinterpret group structure.
    - Detector-agnostic planning: only ``PrefixDetectionResult`` + ``input_ids``
      lengths matter in :meth:`~PrefixSharingPlanner.plan_from_detection`.
    - Correlation hooks: optional ``forward_id`` and ``micro_batch_id`` for logging,
      cache keys, and pipeline tracing (auto-assigned when omitted).

Example:
    >>> from prefix_sharing.core.config import PrefixSharingConfig
    >>> planner = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=3))
    >>> meta = planner.plan([[1, 2, 3, 4], [1, 2, 3, 5]], forward_id=1, micro_batch_id=1)
    >>> meta.has_sharing
    True
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.prefix_detector import PrefixDetectionResult, TriePrefixDetector


@dataclass(frozen=True)
class PrefixLastRestoreSpec:
    """Plan for one reuse row's prefix-last prediction slot.

    Under prefix sharing the reuser's query is trimmed to the suffix while prefix
    KV is still supplied (e.g. via cache injection). The forward pass therefore
    never materializes a query/output at the prefix-last index, yet the first
    suffix token is still predicted in the *next-token* sense from that position:
    its logprob must come from logits at ``prefix_len - 1``, not from the first
    suffix timestep in the packed sequence. This spec records where to read that
    value from the provider row and where to insert it when assembling per-row
    response logprobs or logits (keeping autograd tied to the shared prefix).
    """

    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    provider_prefix_last_pos: int
    reuse_first_suffix_label_pos: int
    output_slot: int
    group_id: int


from prefix_sharing.core.metadata import PrefixSharingBatchMeta  # noqa: E402


_forward_ids = itertools.count(1)


def _cumsum(lengths: Sequence[int]) -> list[int]:
    values = [0]
    total = 0
    for length in lengths:
        total += int(length)
        values.append(total)
    return values


@dataclass
class PrefixSharingPlanner:
    config: PrefixSharingConfig
    detector: TriePrefixDetector | None = None
    _micro_batch_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.detector is None:
            self.detector = TriePrefixDetector(
                min_prefix_len=self.config.min_prefix_len,
                min_group_size=self.config.min_group_size,
            )

    def plan(
        self,
        input_ids: Sequence[Sequence[int]],
        *,
        forward_id: int | None = None,
        micro_batch_id: int | None = None,
    ) -> PrefixSharingBatchMeta:
        detection = self.detector.detect(input_ids)
        return self.plan_from_detection(
            input_ids,
            detection,
            forward_id=forward_id,
            micro_batch_id=micro_batch_id,
        )

    def plan_from_detection(
        self,
        input_ids: Sequence[Sequence[int]],
        detection: PrefixDetectionResult,
        *,
        forward_id: int | None = None,
        micro_batch_id: int | None = None,
    ) -> PrefixSharingBatchMeta:
        if len(input_ids) != detection.batch_size:
            raise ValueError("input_ids batch size does not match detection result")
        if forward_id is None:
            forward_id = next(_forward_ids)
        if micro_batch_id is None:
            self._micro_batch_counter += 1
            micro_batch_id = self._micro_batch_counter

        batch_size = len(input_ids)
        original_lengths = [len(seq) for seq in input_ids]
        group_ids = list(detection.group_ids)
        is_provider = list(detection.is_provider)
        provider_index = list(detection.provider_index)
        prefix_lens = list(detection.prefix_lens)
        reuse_specs = list(detection.reuse_specs)

        suffix_lens: list[int] = []
        kept_lengths_q: list[int] = []
        expanded_lengths_kv: list[int] = []
        q_position_offsets: list[int] = []
        kv_position_offsets: list[int] = []
        input_keep_ranges: list[tuple[int, int]] = []
        label_keep_ranges: list[tuple[int, int]] = []
        loss_mask_keep_ranges: list[tuple[int, int]] = []
        restore_specs: list[PrefixLastRestoreSpec] = []

        for index, original_len in enumerate(original_lengths):
            prefix_len = prefix_lens[index]
            if prefix_len > original_len:
                raise ValueError(f"prefix_len exceeds sequence length for batch index {index}")
            is_reuser = provider_index[index] != index and prefix_len > 0
            suffix_len = original_len - prefix_len if is_reuser else original_len
            suffix_lens.append(suffix_len)
            expanded_lengths_kv.append(original_len)

            if is_reuser:
                keep_start = prefix_len
                keep_end = original_len
                kept_len = suffix_len
                q_offset = prefix_len
                if prefix_len > 0 and suffix_len > 0:
                    restore_specs.append(
                        PrefixLastRestoreSpec(
                            reuse_idx_in_batch=index,
                            provider_idx_in_batch=provider_index[index],
                            provider_prefix_last_pos=prefix_len - 1,
                            reuse_first_suffix_label_pos=prefix_len,
                            output_slot=0,
                            group_id=group_ids[index],
                        )
                    )
            else:
                keep_start = 0
                keep_end = original_len
                kept_len = original_len
                q_offset = 0

            kept_lengths_q.append(kept_len)
            q_position_offsets.append(q_offset)
            kv_position_offsets.append(0)
            keep_range = (keep_start, keep_end)
            input_keep_ranges.append(keep_range)
            label_keep_ranges.append(keep_range)
            loss_mask_keep_ranges.append(keep_range)

        cu_seqlens_q = _cumsum(kept_lengths_q)
        cu_seqlens_kv = _cumsum(expanded_lengths_kv)

        return PrefixSharingBatchMeta(
            forward_id=forward_id,
            micro_batch_id=micro_batch_id,
            batch_size=batch_size,
            original_lengths=original_lengths,
            reuse_specs=reuse_specs,
            group_ids=group_ids,
            is_provider=is_provider,
            provider_index=provider_index,
            prefix_lens=prefix_lens,
            suffix_lens=suffix_lens,
            kept_lengths_q=kept_lengths_q,
            expanded_lengths_kv=expanded_lengths_kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max(kept_lengths_q, default=0),
            max_seqlen_kv=max(expanded_lengths_kv, default=0),
            q_position_offsets=q_position_offsets,
            kv_position_offsets=kv_position_offsets,
            input_keep_ranges=input_keep_ranges,
            label_keep_ranges=label_keep_ranges,
            loss_mask_keep_ranges=loss_mask_keep_ranges,
            prefix_last_restore=restore_specs,
        )
