"""Turn detection results into executable prefix-sharing metadata."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.prefix_detector import PrefixDetectionResult, TriePrefixDetector
from prefix_sharing.core.metadata import PrefixLastRestoreSpec, PrefixSharingBatchMeta


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
                            reuse_batch_index=index,
                            provider_batch_index=provider_index[index],
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
