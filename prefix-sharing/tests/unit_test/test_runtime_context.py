import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.batch_layout import BshdBatchLayout, BshdTokenIndex, ThdBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context, prefix_sharing_runtime_context
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
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
        batch_runtime_layout=ThdBatchLayout.construct_from_valid_lengths(prefix_sharing_plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(pp_rank=1, pp_size=2, is_pipeline_first_stage=False),
    )


def test_prefix_sharing_runtime_context_sets_and_clears_current_context():
    prefix_sharing_runtime_state = _prefix_sharing_runtime_state()
    assert current_prefix_sharing_context() is None
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.prefix_sharing_plan is prefix_sharing_runtime_state.prefix_sharing_plan
        assert ctx.parallel_info is prefix_sharing_runtime_state.parallel_info
        assert ctx.parallel_info.pp_rank == 1
        assert ctx.parallel_info.pp_size == 2
        assert ctx.prefix_last_restore_indices[0].provider_token_index == 2
        assert ctx.prefix_last_restore_indices[0].reuse_token_index == 5
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
        batch_runtime_layout=ThdBatchLayout(
            valid_lengths=[5, 2],
            padded_lengths=[6, 2],
            cu_seqlens=[0, 6, 8],
            max_seqlen=6,
        ),
        parallel_info=MegatronParallelInfo(),
    )

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        assert ctx.prefix_last_restore_indices[0].provider_token_index == 2
        assert ctx.prefix_last_restore_indices[0].reuse_token_index == 6


def test_prefix_sharing_runtime_context_uses_bshd_layout_for_restore_indices():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]],
        forward_id=10,
        micro_batch_id=20,
    )
    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        backend=None,
        batch_runtime_layout=BshdBatchLayout.from_valid_token_mask(
            torch.tensor(
                [
                    [True, True, True, True, True, False],
                    [False, False, False, True, True, False],
                ]
            )
        ),
        parallel_info=MegatronParallelInfo(),
    )

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        assert ctx.prefix_last_restore_indices[0].provider_token_index == BshdTokenIndex(seq_idx_in_batch=0, token_idx_in_seq=2)
        assert ctx.prefix_last_restore_indices[0].reuse_token_index == BshdTokenIndex(seq_idx_in_batch=1, token_idx_in_seq=3)
