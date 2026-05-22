"""Prefix-group aware DP partitioning.

This module is intentionally framework-light. It works on plain token lists and
group ids so verl integration can stay as a thin adapter around DataProto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


TokenSequence = Sequence[int]


@dataclass(frozen=True)
class PrefixGroup:
    group_id: str
    sample_indices: tuple[int, ...]
    original_tokens: int
    estimated_compute_tokens: int
    reusable_prefix_tokens: int


@dataclass(frozen=True)
class PrefixGroupPartition:
    dp_rank_to_indices: tuple[tuple[int, ...], ...]
    dp_rank_to_group_ids: tuple[tuple[str, ...], ...]
    group_workloads: dict[str, int]
    group_sizes: dict[str, int] = field(default_factory=dict)
    fallback_reason: str | None = None
    original_tokens: int = 0
    compute_tokens: int = 0
    reusable_prefix_tokens: int = 0

    @property
    def is_fallback(self) -> bool:
        return self.fallback_reason is not None


class _TrieNode:
    __slots__ = ("children",)

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}


class _IncrementalPrefixTrie:
    """Tracks prefixes already materialized inside one scheduling group."""

    def __init__(self) -> None:
        self._root = _TrieNode()

    def count_new_tokens_and_insert(self, tokens: Sequence[int]) -> int:
        node = self._root
        matched_prefix_len = 0

        for token in tokens:
            child = node.children.get(token)
            if child is None:
                break
            node = child
            matched_prefix_len += 1

        for token in tokens[matched_prefix_len:]:
            child = _TrieNode()
            node.children[token] = child
            node = child

        return len(tokens) - matched_prefix_len


def estimate_incremental_prefix_compute_tokens(token_rows: Sequence[TokenSequence]) -> int:
    """Estimate compute tokens with rank-local incremental prefix reuse.

    The estimator models the policy used for DP scheduling only: samples in the
    same prefix group are handled in their existing order, and each later sample
    may reuse the longest prefix already seen in that group.
    """

    trie = _IncrementalPrefixTrie()
    compute_tokens = 0
    for row in token_rows:
        tokens = [int(token) for token in row]
        compute_tokens += trie.count_new_tokens_and_insert(tokens)
    return compute_tokens


def estimate_group_workloads(
    input_ids: Any,
    attention_mask: Any,
    group_ids: Sequence[Any],
) -> dict[str, PrefixGroup]:
    """Estimate prefix-aware workload per group id."""

    token_rows = _valid_token_rows(input_ids, attention_mask)
    if len(token_rows) != len(group_ids):
        raise ValueError("group_ids length must match input rows")

    group_to_indices: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        group_to_indices.setdefault(str(group_id), []).append(index)

    groups: dict[str, PrefixGroup] = {}
    for group_id, indices in group_to_indices.items():
        group_tokens = [token_rows[index] for index in indices]
        original_tokens = sum(len(tokens) for tokens in group_tokens)
        compute_tokens = estimate_incremental_prefix_compute_tokens(group_tokens)
        groups[group_id] = PrefixGroup(
            group_id=group_id,
            sample_indices=tuple(indices),
            original_tokens=original_tokens,
            estimated_compute_tokens=compute_tokens,
            reusable_prefix_tokens=original_tokens - compute_tokens,
        )
    return groups


def partition_prefix_groups(
    input_ids: Any,
    attention_mask: Any,
    group_ids: Sequence[Any],
    dp_size: int,
    *,
    require_equal_size: bool = True,
) -> PrefixGroupPartition:
    """Partition samples by prefix group while balancing prefix-aware workload."""

    empty = tuple(() for _ in range(max(dp_size, 0)))
    if dp_size <= 1:
        return PrefixGroupPartition(empty, empty, {}, fallback_reason="dp_size_le_1")

    batch_size = _row_count(input_ids)
    if len(group_ids) != batch_size:
        return PrefixGroupPartition(empty, empty, {}, fallback_reason="group_ids_length_mismatch")
    if require_equal_size and batch_size % dp_size != 0:
        return PrefixGroupPartition(empty, empty, {}, fallback_reason="uneven_batch_size_for_dp")

    try:
        groups = estimate_group_workloads(input_ids, attention_mask, group_ids)
    except (TypeError, ValueError):
        return PrefixGroupPartition(empty, empty, {}, fallback_reason="prefix_workload_unavailable")

    if len(groups) < dp_size:
        return PrefixGroupPartition(empty, empty, {}, fallback_reason="group_count_lt_dp_size")

    target_size = batch_size // dp_size if require_equal_size else None
    rank_groups: list[list[PrefixGroup]] = [[] for _ in range(dp_size)]
    rank_sizes = [0] * dp_size
    rank_workloads = [0] * dp_size

    sorted_groups = sorted(
        groups.values(),
        key=lambda group: (-group.estimated_compute_tokens, group.group_id),
    )
    for group in sorted_groups:
        group_size = len(group.sample_indices)
        if target_size is not None and group_size > target_size:
            return PrefixGroupPartition(empty, empty, {}, fallback_reason="group_too_large_for_equal_dp_partition")

        candidates = [
            rank
            for rank in range(dp_size)
            if target_size is None or rank_sizes[rank] + group_size <= target_size
        ]
        if not candidates:
            return PrefixGroupPartition(empty, empty, {}, fallback_reason="cannot_fit_groups_equal_size")

        best_rank = min(candidates, key=lambda rank: (rank_workloads[rank], rank_sizes[rank], rank))
        rank_groups[best_rank].append(group)
        rank_sizes[best_rank] += group_size
        rank_workloads[best_rank] += group.estimated_compute_tokens

    if target_size is not None and any(size != target_size for size in rank_sizes):
        return PrefixGroupPartition(empty, empty, {}, fallback_reason="cannot_fit_groups_equal_size")

    dp_rank_to_group_ids: list[tuple[str, ...]] = []
    dp_rank_to_indices: list[tuple[int, ...]] = []
    for groups_for_rank in rank_groups:
        ordered_groups = sorted(
            groups_for_rank,
            key=lambda group: (group.estimated_compute_tokens, group.group_id),
        )
        dp_rank_to_group_ids.append(tuple(group.group_id for group in ordered_groups))
        dp_rank_to_indices.append(
            tuple(index for group in ordered_groups for index in group.sample_indices)
        )

    original_tokens = sum(group.original_tokens for group in groups.values())
    compute_tokens = sum(group.estimated_compute_tokens for group in groups.values())
    return PrefixGroupPartition(
        dp_rank_to_indices=tuple(dp_rank_to_indices),
        dp_rank_to_group_ids=tuple(dp_rank_to_group_ids),
        group_workloads={group_id: group.estimated_compute_tokens for group_id, group in groups.items()},
        group_sizes={group_id: len(group.sample_indices) for group_id, group in groups.items()},
        original_tokens=original_tokens,
        compute_tokens=compute_tokens,
        reusable_prefix_tokens=original_tokens - compute_tokens,
    )


def _valid_token_rows(input_ids: Any, attention_mask: Any) -> list[list[int]]:
    rows = _to_nested_list(input_ids)
    masks = _to_nested_list(attention_mask)
    if len(rows) != len(masks):
        raise ValueError("input_ids and attention_mask row counts must match")

    valid_rows: list[list[int]] = []
    for row, mask in zip(rows, masks):
        if len(row) != len(mask):
            raise ValueError("input_ids and attention_mask row lengths must match")
        valid_rows.append([int(token) for token, keep in zip(row, mask) if bool(keep)])
    return valid_rows


def _row_count(input_ids: Any) -> int:
    return len(_to_nested_list(input_ids))


def _to_nested_list(value: Any) -> list[list[Any]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [list(row) for row in value]
