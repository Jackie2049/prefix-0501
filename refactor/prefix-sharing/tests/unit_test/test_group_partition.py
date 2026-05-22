from prefix_sharing.core.group_partition import (
    PrefixGroupPartition,
    estimate_incremental_prefix_compute_tokens,
    estimate_group_workloads,
    partition_prefix_groups,
)


def test_estimate_incremental_prefix_compute_tokens_counts_rank_local_prefix_reuse():
    token_lists = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 6],
        [7, 8],
    ]

    assert estimate_incremental_prefix_compute_tokens(token_lists) == 8


def test_estimate_incremental_prefix_compute_tokens_handles_duplicates_and_shorter_rows():
    token_lists = [
        [1, 2, 3, 4],
        [1, 2],
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [],
    ]

    assert estimate_incremental_prefix_compute_tokens(token_lists) == 5


def test_estimate_group_workloads_uses_attention_mask_and_prefix_reuse():
    input_ids = [
        [1, 2, 3, 4, 0],
        [1, 2, 3, 5, 0],
        [9, 9, 0, 0, 0],
    ]
    attention_mask = [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0],
    ]
    group_ids = ["a", "a", "b"]

    groups = estimate_group_workloads(input_ids, attention_mask, group_ids)

    assert groups["a"].original_tokens == 8
    assert groups["a"].estimated_compute_tokens == 5
    assert groups["a"].reusable_prefix_tokens == 3
    assert groups["a"].sample_indices == (0, 1)
    assert groups["b"].estimated_compute_tokens == 2


def test_partition_prefix_groups_keeps_same_uid_on_one_dp_rank():
    input_ids = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [7, 8, 0, 0],
        [7, 9, 0, 0],
    ]
    attention_mask = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]
    group_ids = ["a", "a", "b", "b"]

    partition = partition_prefix_groups(input_ids, attention_mask, group_ids, dp_size=2)

    assert isinstance(partition, PrefixGroupPartition)
    assert partition.fallback_reason is None
    assert sorted(sum((list(indices) for indices in partition.dp_rank_to_indices), [])) == [0, 1, 2, 3]
    for indices in partition.dp_rank_to_indices:
        rank_group_ids = {group_ids[idx] for idx in indices}
        assert rank_group_ids in ({"a"}, {"b"})


def test_partition_prefix_groups_balances_by_prefix_aware_compute_tokens():
    input_ids = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [10, 11],
        [30, 31],
    ]
    attention_mask = [[1] * len(seq) for seq in input_ids]
    group_ids = ["a", "a", "b", "c"]

    partition = partition_prefix_groups(input_ids, attention_mask, group_ids, dp_size=2)

    assert partition.group_workloads == {"a": 5, "b": 2, "c": 2}
    rank_workloads = [
        sum(partition.group_workloads[group_id] for group_id in group_ids_)
        for group_ids_ in partition.dp_rank_to_group_ids
    ]
    assert sorted(rank_workloads) == [4, 5]


def test_partition_prefix_groups_falls_back_when_group_key_missing_or_invalid():
    partition = partition_prefix_groups(
        input_ids=[[1, 2], [1, 3]],
        attention_mask=[[1, 1], [1, 1]],
        group_ids=["a"],
        dp_size=2,
    )

    assert partition.fallback_reason == "group_ids_length_mismatch"
    assert partition.dp_rank_to_indices == ((), ())


def test_partition_prefix_groups_falls_back_when_equal_size_cannot_be_satisfied():
    partition = partition_prefix_groups(
        input_ids=[[1, 2], [1, 3], [9, 9]],
        attention_mask=[[1, 1], [1, 1], [1, 1]],
        group_ids=["a", "a", "b"],
        dp_size=2,
    )

    assert partition.fallback_reason == "uneven_batch_size_for_dp"
