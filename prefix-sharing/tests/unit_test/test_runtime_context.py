import pytest

from prefix_sharing.backends.packed_layout import PackedBatchLayout
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
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(pp_rank=1, pp_size=2, is_pipeline_first_stage=False),
    )


def test_prefix_sharing_runtime_context_sets_and_clears_current_context():
    prefix_sharing_runtime_state = _prefix_sharing_runtime_state()
    assert current_prefix_sharing_context() is None
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.prefix_sharing_plan is prefix_sharing_runtime_state.prefix_sharing_plan
        # 3 restore specs: 2 interior (provider_predict_pos=0,1) + 1 prefix-last (pos=2)
        assert len(ctx.prefix_last_restore_indices) == 3
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 0  # interior pos1
        assert ctx.prefix_last_restore_indices[2].provider_1d_pos == 2  # prefix-last
        assert ctx.prefix_last_restore_indices[0].reuse_1d_pos == -1  # sentinel: no slot in reuser packed region
        assert ctx.parallel_info is prefix_sharing_runtime_state.parallel_info
        assert ctx.parallel_info.pp_rank == 1
        assert ctx.parallel_info.pp_size == 2
        assert ctx.stats.original_tokens == 11
        assert ctx.stats.kept_valid_tokens == 8
        assert ctx.stats.expected_reused_counts_per_layer == 1
        assert ctx.stats.expected_reused_prefix_tokens_per_layer == 3
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
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout(
            valid_lengths=[5, 2],
            padded_lengths=[6, 2],
            cu_seqlens=[0, 6, 8],
            max_seqlen=6,
        ),
        parallel_info=MegatronParallelInfo(),
    )

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        # 3 restore specs: 2 interior + 1 prefix-last
        assert len(ctx.prefix_last_restore_indices) == 3
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 0  # interior pos1
        assert ctx.prefix_last_restore_indices[2].provider_1d_pos == 2  # prefix-last
        assert ctx.prefix_last_restore_indices[0].reuse_1d_pos == -1  # sentinel: no slot in reuser packed region
        assert ctx.stats.kept_padded_tokens == 8


def _chain_reuse_runtime_state():
    """Chain-reuse: row0=provider, row1=reuse(0,prefix=3), row2=reuse(1,prefix=3).

    Row2's prefix-last predict position (prefix_len-1 = 2) falls on row1's
    keep_start-1 (row1 keep_start=3).  Under v080 physical trimming row1's
    packed region has no slot for position 2, so the prefix-last logits must
    be fetched from the chain ancestor row0 whose packed region strictly
    contains position 2 (keep_start=0 <= 2 < keep_end=8).

    The chain is forced via a hand-built PrefixDetectionResult (the trie
    detector naturally routes row2 to row0 because it records the *first*
    inserter as provider at each depth; we need the transitive edge
    row2->row1 specifically to exercise the keep_start-1 miss path).
    """
    from prefix_sharing.core.prefix_detector import (
        PrefixDetectionResult,
        PrefixReuseSpec,
    )

    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2))
    sequences = [
        [100, 101, 102, 103, 104, 105, 106, 107],   # row0 provider, len 8
        [100, 101, 102, 108, 109, 110],              # row1 reuse(0) prefix=3, len 6
        [100, 101, 102, 111, 112, 113],              # row2 reuse(1) prefix=3, len 6
    ]
    detection = PrefixDetectionResult(
        batch_size=3,
        reuse_specs=(
            PrefixReuseSpec(reuse_idx_in_batch=1, provider_idx_in_batch=0, prefix_len=3),
            PrefixReuseSpec(reuse_idx_in_batch=2, provider_idx_in_batch=1, prefix_len=3),
        ),
        prefix_groups=(),
        group_ids=[0, 0, 1],
        provider_index=[0, 0, 1],
        prefix_lens=[0, 3, 3],
        is_provider=[True, False, False],
    )
    prefix_sharing_plan = planner.plan_from_detection(
        sequences, detection, forward_id=10, micro_batch_id=20,
    )
    # Sanity: the chain is row2 -> row1 -> row0 with prefix_len 3 each.
    assert prefix_sharing_plan.provider_index[1] == 0
    assert prefix_sharing_plan.provider_index[2] == 1
    assert prefix_sharing_plan.prefix_lens[1] == 3
    assert prefix_sharing_plan.prefix_lens[2] == 3
    return PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(),
    )


def test_prefix_last_in_chain_reuse_resolves_to_ancestor_with_packed_slot():
    """Regression: prefix-last in chain-reuse must fetch logits from the
    chain ancestor whose packed region strictly contains the predict position,
    not the intermediate reuser whose keep_start-1 has no packed slot.

    Before the fix _build_prefix_last_restore_indices left provider_1d_pos=-1
    (sentinel) for such specs, causing vocab_logprobs save to skip them and
    the downstream restore to KeyError on the saved-logits lookup.
    """
    runtime_state = _chain_reuse_runtime_state()
    with prefix_sharing_runtime_context(runtime_state) as ctx:
        # Collect prefix-last (non-interior) indices for row2 (reuse_idx=2).
        row2_plast = [
            idx for idx in ctx.prefix_last_restore_indices
            if idx.reuse_idx_in_batch == 2 and idx.restore_type != "restore_prefix_interior"
        ]
        assert len(row2_plast) == 1, "row2 should have exactly one prefix-last restore"
        spec = row2_plast[0]

        # target_2d_pos = prefix_len-1 = 2 (prefix-last predict position).
        assert spec.target_2d_pos == 2
        # The fix: provider_1d_pos must NOT be the -1 sentinel. It must point
        # into row0's packed region (row0 has 8 tokens, position 2 → packed
        # index 2 since row0 starts at offset 0 in the packed tensor).
        assert spec.provider_1d_pos != -1, (
            "prefix-last in chain-reuse must resolve provider_1d_pos to a "
            "chain ancestor with a real packed slot, not the -1 sentinel"
        )
        assert spec.provider_1d_pos == 2, (
            f"expected packed index 2 (row0 offset 0 + pos 2), got {spec.provider_1d_pos}"
        )
