from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context, prefix_sharing_context


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
