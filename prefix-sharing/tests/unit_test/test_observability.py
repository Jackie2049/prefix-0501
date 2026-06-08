"""Tests for prefix-sharing observability stats."""

from prefix_sharing.core.observability import (
    PrefixSharingLayerStats,
    PrefixSharingStats,
    current_prefix_sharing_stats,
    _stats_var,
)
from prefix_sharing.core.planner import PrefixSharingPlan, PrefixSharingPlanner
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.backends.packed_layout import PackedBatchLayout


def _make_plan_and_layout():
    """Build a simple plan with sharing for testing."""
    config = PrefixSharingConfig(
        enable_prefix_sharing=True,
        min_group_size=2,
    )
    # Two sequences sharing prefix tokens [1, 2, 3]
    sequences = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9]]
    planner = PrefixSharingPlanner(config)
    plan = planner.plan(sequences)
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    return plan, layout


def test_prefix_sharing_stats_from_plan_records_expected_reuse_summary():
    plan, layout = _make_plan_and_layout()
    stats = PrefixSharingStats.from_plan(plan, layout)

    assert stats.forward_id == plan.forward_id
    assert stats.micro_batch_id == plan.micro_batch_id
    assert stats.batch_size == 2
    assert stats.original_tokens == 12  # 6 + 6
    assert stats.kept_valid_tokens == sum(plan.kept_lengths_q)
    assert stats.reused_valid_tokens == stats.original_tokens - stats.kept_valid_tokens
    assert stats.reused_valid_token_ratio > 0
    assert stats.provider_count == 1  # "ABCDEF" is the provider
    assert stats.reuser_count == 1    # "ABCXYZ" is the reuser
    assert stats.sharing_group_count >= 1
    assert stats.expected_reused_counts_per_layer == 1  # one reuser
    assert stats.expected_reused_prefix_tokens_per_layer == 3  # "ABC" = 3 tokens
    assert stats.actual_restore_count == 0


def test_prefix_sharing_stats_records_layer_runtime_and_expected_match():
    plan, layout = _make_plan_and_layout()
    stats = PrefixSharingStats.from_plan(plan, layout)

    # Simulate a successful KV build for layer 0
    stats.record_attention_kv_build(
        layer_id=0,
        store_count=2,          # provider stored + reuser expanded
        reuse_count=1,          # reuser tried to load
        reuse_hit_count=1,      # reuser succeeded
        reuse_miss_count=0,     # no misses
        stored_tokens=9,        # provider 6 + reuser expanded 3
        reused_prefix_tokens=3, # "ABC" = 3 tokens reused
        expanded_kv_tokens=9,   # total expanded KV
        valid_q_tokens=stats.kept_valid_tokens,
        padded_q_tokens=stats.kept_padded_tokens,
    )

    # Simulate a restore
    stats.record_restore(1)

    # Check expected-vs-actual match
    assert stats.layer_matches_expected(0)
    assert stats.actual_restore_count == 1
    assert stats.layers[0].store_count == 2
    assert stats.layers[0].reuse_count == 1
    assert stats.layers[0].reuse_hit_count == 1
    assert stats.layers[0].reuse_miss_count == 0
    assert stats.layers[0].reused_prefix_tokens == 3


def test_prefix_sharing_stats_detects_mismatch():
    plan, layout = _make_plan_and_layout()
    stats = PrefixSharingStats.from_plan(plan, layout)

    # Simulate a KV build with a miss
    stats.record_attention_kv_build(
        layer_id=0,
        store_count=1,
        reuse_count=1,
        reuse_hit_count=0,
        reuse_miss_count=1,     # missed!
        stored_tokens=6,
        reused_prefix_tokens=0,  # no reuse happened
        expanded_kv_tokens=6,
        valid_q_tokens=stats.kept_valid_tokens,
        padded_q_tokens=stats.kept_padded_tokens,
    )

    # Should not match expected
    assert not stats.layer_matches_expected(0)


def test_current_prefix_sharing_stats_context_var():
    """Stats ContextVar should be None outside a context and populated inside."""
    assert current_prefix_sharing_stats() is None

    plan, layout = _make_plan_and_layout()
    stats = PrefixSharingStats.from_plan(plan, layout)

    token = _stats_var.set(stats)
    assert current_prefix_sharing_stats() is stats
    _stats_var.reset(token)

    assert current_prefix_sharing_stats() is None