from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context, prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState


def _prefix_sharing_runtime_state():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=10,
        micro_batch_id=20,
    )
    return PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q),
    )


def test_prefix_sharing_runtime_context_sets_and_clears_current_context():
    prefix_sharing_runtime_state = _prefix_sharing_runtime_state()
    assert current_prefix_sharing_context() is None
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.prefix_sharing_plan is prefix_sharing_runtime_state.prefix_sharing_plan
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 2
        assert ctx.prefix_last_restore_indices[0].reuse_1d_pos == 5
        assert not ctx.store.closed
    assert current_prefix_sharing_context() is None
    assert ctx.store.closed


def test_prefix_sharing_runtime_context_uses_padded_layout_for_restore_indices():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]],
        forward_id=10,
        micro_batch_id=20,
    )
    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        backend=None,
        packed_batch_layout=PackedBatchLayout(
            valid_lengths=[5, 2],
            padded_lengths=[6, 2],
            cu_seqlens=[0, 6, 8],
            max_seqlen=6,
        ),
    )

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 2
        assert ctx.prefix_last_restore_indices[0].reuse_1d_pos == 6
