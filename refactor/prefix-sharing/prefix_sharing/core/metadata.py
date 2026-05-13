"""Metadata objects that describe one prefix-sharing micro-batch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from prefix_sharing.core.prefix_detector import PrefixReuseSpec

if TYPE_CHECKING:
    from prefix_sharing.core.planner import PrefixLastRestoreSpec


Range = tuple[int, int]


@dataclass(frozen=True)
class PrefixSharingBatchMeta:
    """Complete framework-independent plan for a single micro-batch."""

    forward_id: int
    micro_batch_id: int
    batch_size: int
    original_lengths: list[int]

    reuse_specs: list[PrefixReuseSpec]
    group_ids: list[int]
    is_provider: list[bool]
    provider_index: list[int]
    prefix_lens: list[int]
    suffix_lens: list[int]

    kept_lengths_q: list[int]
    expanded_lengths_kv: list[int]
    cu_seqlens_q: list[int]
    cu_seqlens_kv: list[int]
    max_seqlen_q: int
    max_seqlen_kv: int

    q_position_offsets: list[int]
    kv_position_offsets: list[int]

    input_keep_ranges: list[Range]
    label_keep_ranges: list[Range]
    loss_mask_keep_ranges: list[Range]

    prefix_last_restore: list[PrefixLastRestoreSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected = self.batch_size
        fields: Sequence[tuple[str, list[object]]] = (
            ("original_lengths", self.original_lengths),
            ("group_ids", self.group_ids),
            ("is_provider", self.is_provider),
            ("provider_index", self.provider_index),
            ("prefix_lens", self.prefix_lens),
            ("suffix_lens", self.suffix_lens),
            ("kept_lengths_q", self.kept_lengths_q),
            ("expanded_lengths_kv", self.expanded_lengths_kv),
            ("q_position_offsets", self.q_position_offsets),
            ("kv_position_offsets", self.kv_position_offsets),
            ("input_keep_ranges", self.input_keep_ranges),
            ("label_keep_ranges", self.label_keep_ranges),
            ("loss_mask_keep_ranges", self.loss_mask_keep_ranges),
        )
        for name, value in fields:
            if len(value) != expected:
                raise ValueError(f"{name} length must equal batch_size={expected}")
        if len(self.cu_seqlens_q) != expected + 1:
            raise ValueError("cu_seqlens_q length must equal batch_size + 1")
        if len(self.cu_seqlens_kv) != expected + 1:
            raise ValueError("cu_seqlens_kv length must equal batch_size + 1")

    @property
    def has_sharing(self) -> bool:
        return bool(self.reuse_specs)

    def is_reuser(self, idx_in_batch: int) -> bool:
        return self.provider_index[idx_in_batch] != idx_in_batch and self.prefix_lens[idx_in_batch] > 0

    def q_range_for_batch(self, idx_in_batch: int) -> Range:
        return self.cu_seqlens_q[idx_in_batch], self.cu_seqlens_q[idx_in_batch + 1]

    def kv_range_for_batch(self, idx_in_batch: int) -> Range:
        return self.cu_seqlens_kv[idx_in_batch], self.cu_seqlens_kv[idx_in_batch + 1]

    def restore_for_reuse(self, idx_in_batch: int) -> PrefixLastRestoreSpec | None:
        for spec in self.prefix_last_restore:
            if spec.reuse_idx_in_batch == idx_in_batch:
                return spec
        return None
