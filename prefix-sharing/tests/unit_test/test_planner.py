from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


def test_planner_builds_phase_one_prefix_sharing_plan_and_restore_specs():
    input_ids = [
        [1, 2, 3, 4, 5, 10, 11],
        [1, 2, 3, 20, 21, 22],
        [1, 2, 3, 4, 5, 30, 31],
        [9, 9, 9],
    ]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(input_ids, forward_id=7, micro_batch_id=3)

    assert prefix_sharing_plan.forward_id == 7
    assert prefix_sharing_plan.micro_batch_id == 3
    assert prefix_sharing_plan.original_lengths == [7, 6, 7, 3]
    assert [(s.reuse_idx_in_batch, s.provider_idx_in_batch, s.prefix_len) for s in prefix_sharing_plan.reuse_specs] == [
        (1, 0, 3),
        (2, 0, 5),
    ]
    assert prefix_sharing_plan.is_provider == [True, False, False, True]
    assert prefix_sharing_plan.provider_index == [0, 0, 0, 3]
    assert prefix_sharing_plan.prefix_lens == [0, 3, 5, 0]
    assert prefix_sharing_plan.suffix_lens == [7, 3, 2, 3]
    assert prefix_sharing_plan.input_keep_ranges == [(0, 7), (3, 6), (5, 7), (0, 3)]
    assert prefix_sharing_plan.kept_lengths_q == [7, 3, 2, 3]
    assert prefix_sharing_plan.expanded_lengths_kv == [7, 6, 7, 3]
    # cu_seqlens_q: cumsum of kept_lengths_q (reuser Q packs suffix only) -> total 15.
    # cu_seqlens_kv: cumsum of original_lengths (full logical KV: injected prefix + suffix) -> 23.
    assert prefix_sharing_plan.cu_seqlens_q == [0, 7, 10, 12, 15]
    assert prefix_sharing_plan.cu_seqlens_kv == [0, 7, 13, 20, 23]
    # q_position_offsets: logical position of the first packed Q token (reuser == prefix_len).
    # kv_position_offsets: KV always numbered from logical position 0.
    assert prefix_sharing_plan.q_position_offsets == [0, 3, 5, 0]
    assert prefix_sharing_plan.kv_position_offsets == [0, 0, 0, 0]

    # prefix_last_restore: reuser Q skips prefix-last; first-suffix logprob is taken from
    # provider output at prefix_len-1 and prepended to the reuser result.
    assert len(prefix_sharing_plan.prefix_last_restore) == 2
    first, second = prefix_sharing_plan.prefix_last_restore
    # Sample 1: provider row 0 at pos 2 (token `3`) predicts first suffix `20` (original pos 3).
    assert first.reuse_idx_in_batch == 1
    assert first.provider_idx_in_batch == 0
    assert first.provider_prefix_last_pos == 2  # prefix_len - 1
    assert first.reuse_first_suffix_label_pos == 3  # prefix_len, index of first suffix token
    assert first.output_slot == 0  # prepend to suffix logprob sequence
    assert first.target_2d_pos == 2  # label position in 2D output
    assert first.label_value == 20  # input_ids[1][3], the first suffix token
    # Sample 2: provider pos 4 (token `5`) predicts `30` (original pos 5).
    assert second.reuse_idx_in_batch == 2
    assert second.provider_idx_in_batch == 0
    assert second.provider_prefix_last_pos == 4
    assert second.reuse_first_suffix_label_pos == 5
    assert second.output_slot == 0
    assert second.target_2d_pos == 4
    assert second.label_value == 30


def test_planner_no_shared_prefix_keeps_original_shapes():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2))
    prefix_sharing_plan = planner.plan([[1, 2], [1, 3], [4, 5]], forward_id=1, micro_batch_id=1)

    assert not prefix_sharing_plan.has_sharing
    assert prefix_sharing_plan.group_ids == [-1, -1, -1]
    assert prefix_sharing_plan.input_keep_ranges == [(0, 2), (0, 2), (0, 2)]
    assert prefix_sharing_plan.reuse_specs == []
    assert prefix_sharing_plan.prefix_last_restore == []


def test_planner_generates_interior_response_restore_specs():
    # seq1 (provider): [1,2,3 | A,B,C]   prompt=[1,2,3], response=[A,B,C]
    # seq2 (reuser):   [1,2,3 | A,D,E]   prompt=[1,2,3], response=[A,D,E]
    # Shared prefix: [1,2,3,A] (len 4). A is a response token in both,
    # so it needs interior-prefix logprob restore.
    input_ids = [
        [1, 2, 3, 4, 5, 6],     # 1,2,3,A,B,C
        [1, 2, 3, 4, 7, 8],     # 1,2,3,A,D,E
    ]
    prompt_lens = [3, 3]
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    )
    plan = planner.plan(input_ids, prompt_lens=prompt_lens, forward_id=1, micro_batch_id=1)

    # Two restore specs expected:
    #   1. interior-response for token A (pos 3, output_slot=0)
    #   2. prefix-last for first suffix token D (pos 4, output_slot=1)
    assert len(plan.prefix_last_restore) == 2

    interior_spec = plan.prefix_last_restore[0]
    assert interior_spec.is_interior_response
    assert interior_spec.reuse_idx_in_batch == 1
    assert interior_spec.provider_idx_in_batch == 0
    assert interior_spec.provider_prefix_last_pos == 2  # logits[2]
    assert interior_spec.reuse_first_suffix_label_pos == 3  # label pos 3 = A
    assert interior_spec.output_slot == 0
    assert interior_spec.target_2d_pos == 2  # label position interior_pos-1
    assert interior_spec.label_value == 4  # input_ids[1][3] = token A

    prefix_last_spec = plan.prefix_last_restore[1]
    assert not prefix_last_spec.is_interior_response
    assert prefix_last_spec.reuse_idx_in_batch == 1
    assert prefix_last_spec.provider_idx_in_batch == 0
    assert prefix_last_spec.provider_prefix_last_pos == 3  # logits[3] = prefix-last
    assert prefix_last_spec.reuse_first_suffix_label_pos == 4  # label pos 4 = D
    assert prefix_last_spec.output_slot == 1  # after interior slot
    assert prefix_last_spec.target_2d_pos == 3  # prefix_len-1 label position
    assert prefix_last_spec.label_value == 7  # input_ids[1][4] = first suffix token D

    # Trimming still at prefix_len
    assert plan.input_keep_ranges == [(0, 6), (4, 6)]
    assert plan.kept_lengths_q == [6, 2]


def test_planner_prompt_lens_none_preserves_backward_compat():
    """Without prompt_lens, interior-response restore is skipped."""
    input_ids = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
    ]
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    )
    plan = planner.plan(input_ids, forward_id=1, micro_batch_id=1)

    # Only one spec (prefix-last), no interior response tokens
    assert len(plan.prefix_last_restore) == 1
    spec = plan.prefix_last_restore[0]
    assert not spec.is_interior_response
    assert spec.output_slot == 0
