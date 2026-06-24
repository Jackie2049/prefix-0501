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


def test_chain_reuse_resolves_position_to_provider_with_matching_label():
    """Chain-reuse must route a deeper reuser to the provider whose
    *continuation* matches, not the root.

    Trie detection produces a natural chain here:
        seq0: [1, 2, 3, 4, 5]   provider (root)
        seq1: [1, 2, 3, 7, 8]   reuse(seq0), shared prefix [1,2,3] (len 3)
        seq2: [1, 2, 3, 7, 9]   reuse(seq1), shared prefix [1,2,3,7] (len 4)

    The trie routes seq2 to seq1 (not seq0): at position 3 both seq1 and seq2
    hold token 7, whereas seq0 holds 4.  So seq2's restore for position 2
    (token "3", which predicts token 7) must reference seq1 — whose label at
    that position is 7 — rather than seq0, whose label there is 4.

    The keep_start-1 extension in ``_resolve_provider_for_position`` keeps the
    boundary position on the direct provider seq1 (seq1 keeps [3, 5), so
    keep_start-1 = 2 covers position 2).  Positions shared identically with
    the root (0, 1) walk up to seq0.  The prefix-last (position 3) lands on a
    real packed slot of seq1 because seq1's kept range [3, 5) contains it.
    """
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2))
    plan = planner.plan(
        [[1, 2, 3, 4, 5], [1, 2, 3, 7, 8], [1, 2, 3, 7, 9]],
        forward_id=10,
        micro_batch_id=20,
    )
    # Chain: seq2 -> seq1 -> seq0; seq2 shares the longer prefix [1,2,3,7].
    assert plan.provider_index == [0, 0, 1]
    assert plan.prefix_lens == [0, 3, 4]

    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(),
    )
    with prefix_sharing_runtime_context(runtime_state) as ctx:
        seq2 = {
            idx.target_2d_pos: idx
            for idx in ctx.prefix_last_restore_indices
            if idx.reuse_idx_in_batch == 2
        }

        # Positions 0, 1 (tokens 1, 2): labels 2, 3 are identical across all
        # rows, so they resolve up the chain to the root seq0.
        assert seq2[0].provider_idx_in_batch == 0
        assert seq2[0].label_value == 2
        assert seq2[1].provider_idx_in_batch == 0
        assert seq2[1].label_value == 3

        # Position 2 (token "3"): seq2/seq1 predict 7 here, seq0 predicts 4.
        # Must resolve to the direct provider seq1 (matching label 7), and
        # stay there via the keep_start-1 extension — NOT walk up to seq0.
        assert seq2[2].provider_idx_in_batch == 1, (
            "seq2 position 2 (token 3 -> label 7) must resolve to seq1 "
            "(matching continuation 7), not seq0 (continuation 4)"
        )
        assert seq2[2].label_value == 7

        # Prefix-last (position 3, predicts token 9): seq1 keeps [3, 5) which
        # strictly contains position 3, so it resolves to a real packed slot
        # on seq1 (not the -1 sentinel).
        plast = seq2[3]
        assert plast.is_shared_prefix_interior is False
        assert plast.provider_idx_in_batch == 1
        assert plast.label_value == 9
        assert plast.provider_1d_pos != -1, (
            "prefix-last must resolve to a real packed slot on the direct "
            "provider seq1, not the -1 sentinel"
        )
