from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.core.planner import PrefixSharingPlanner


def test_prefix_sharing_stats_from_plan_records_expected_reuse_summary():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]],
        forward_id=10,
        micro_batch_id=20,
    )
    layout = PackedBatchLayout(
        valid_lengths=[5, 2],
        padded_lengths=[6, 2],
        cu_seqlens=[0, 6, 8],
        max_seqlen=6,
    )

    stats = PrefixSharingStats.from_plan(prefix_sharing_plan, layout)

    assert stats.forward_id == 10
    assert stats.micro_batch_id == 20
    assert stats.batch_size == 2
    assert stats.original_tokens == 10
    assert stats.kept_valid_tokens == 7
    assert stats.kept_padded_tokens == 8
    assert stats.reused_valid_tokens == 3
    assert stats.reused_valid_token_ratio == 0.3
    assert stats.provider_count == 1
    assert stats.reuser_count == 1
    assert stats.sharing_group_count == 1
    assert stats.expected_reused_counts_per_layer == 1
    assert stats.expected_reused_prefix_tokens_per_layer == 3
    assert stats.expected_restore_count == 1
    assert stats.actual_restore_count == 0


def test_prefix_sharing_stats_records_layer_runtime_and_expected_match():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]],
        forward_id=10,
        micro_batch_id=20,
    )
    stats = PrefixSharingStats.from_plan(
        prefix_sharing_plan,
        PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q),
    )

    stats.record_attention_kv_build(
        layer_id=7,
        store_count=2,
        reuse_count=1,
        reuse_hit_count=1,
        reuse_miss_count=0,
        stored_tokens=10,
        reused_prefix_tokens=3,
        expanded_kv_tokens=10,
        valid_q_tokens=7,
        padded_q_tokens=7,
    )
    stats.record_restore(1)

    layer = stats.layers[7]
    assert layer.store_count == 2
    assert layer.reuse_count == 1
    assert layer.reuse_hit_count == 1
    assert layer.reuse_miss_count == 0
    assert layer.stored_tokens == 10
    assert layer.reused_prefix_tokens == 3
    assert layer.expanded_kv_tokens == 10
    assert layer.valid_q_tokens == 7
    assert layer.padded_q_tokens == 7
    assert stats.actual_restore_count == 1
    assert stats.layer_matches_expected(7)
