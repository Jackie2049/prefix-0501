from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context, prefix_sharing_context
from prefix_sharing.integrations.parallel_env import ParallelEnv


def _meta():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=3))
    return planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=10,
        micro_batch_id=20,
    )


def test_prefix_sharing_context_sets_and_clears_current_context():
    meta = _meta()
    assert current_prefix_sharing_context() is None
    with prefix_sharing_context(meta) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.meta is meta
        assert not ctx.store.closed
    assert current_prefix_sharing_context() is None
    assert ctx.store.closed


def test_prefix_sharing_context_records_dp_stats():
    meta = _meta()
    parallel_env = ParallelEnv(dp_rank=1, dp_world_size=2)

    with prefix_sharing_context(meta, parallel_env=parallel_env) as ctx:
        assert ctx.parallel_env is parallel_env
        assert ctx.stats is not None
        assert ctx.stats.trace_key == "dp1/tp0/pp0/cp0/fw10/mb20"
        assert ctx.stats.dp_rank == 1
        assert ctx.stats.dp_world_size == 2
        assert ctx.stats.reuse_count == 1
        assert ctx.stats.saved_tokens_q == 3


def test_prefix_sharing_context_isolates_store_between_micro_batches():
    first_meta = _meta()
    second_meta = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=2)).plan(
        [[7, 8, 10], [7, 8, 20]],
        forward_id=11,
        micro_batch_id=21,
    )

    with prefix_sharing_context(first_meta) as first_ctx:
        first_store = first_ctx.store
        assert not first_store.closed
    with prefix_sharing_context(second_meta) as second_ctx:
        assert second_ctx.store is not first_store
        assert first_store.closed
        assert not second_ctx.store.closed
    assert second_ctx.store.closed
