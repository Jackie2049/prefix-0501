from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


def test_planner_builds_phase_one_metadata_and_restore_specs():
    input_ids = [
        [1, 2, 3, 4, 5, 10, 11],
        [1, 2, 3, 20, 21, 22],
        [1, 2, 3, 4, 5, 30, 31],
        [9, 9, 9],
    ]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=3))
    meta = planner.plan(input_ids, forward_id=7, micro_batch_id=3)

    assert meta.forward_id == 7
    assert meta.micro_batch_id == 3
    assert meta.original_lengths == [7, 6, 7, 3]
    assert [(s.reuse_idx_in_batch, s.provider_idx_in_batch, s.prefix_len) for s in meta.reuse_specs] == [
        (1, 0, 3),
        (2, 0, 5),
    ]
    assert meta.is_provider == [True, False, False, True]
    assert meta.provider_index == [0, 0, 0, 3]
    assert meta.prefix_lens == [0, 3, 5, 0]
    assert meta.suffix_lens == [7, 3, 2, 3]
    assert meta.input_keep_ranges == [(0, 7), (3, 6), (5, 7), (0, 3)]
    assert meta.kept_lengths_q == [7, 3, 2, 3]
    assert meta.expanded_lengths_kv == [7, 6, 7, 3]
    # cu_seqlens_q: cumsum of kept_lengths_q (reuser Q packs suffix only) -> total 15.
    # cu_seqlens_kv: cumsum of original_lengths (full logical KV: injected prefix + suffix) -> 23.
    assert meta.cu_seqlens_q == [0, 7, 10, 12, 15]
    assert meta.cu_seqlens_kv == [0, 7, 13, 20, 23]
    # q_position_offsets: logical position of the first packed Q token (reuser == prefix_len).
    # kv_position_offsets: KV always numbered from logical position 0.
    assert meta.q_position_offsets == [0, 3, 5, 0]
    assert meta.kv_position_offsets == [0, 0, 0, 0]

    # prefix_last_restore: reuser Q skips prefix-last; first-suffix logprob is taken from
    # provider output at prefix_len-1 and prepended to the reuser result.
    assert len(meta.prefix_last_restore) == 2
    first, second = meta.prefix_last_restore
    # Sample 1: provider row 0 at pos 2 (token `3`) predicts first suffix `20` (original pos 3).
    assert first.reuse_idx_in_batch == 1
    assert first.provider_idx_in_batch == 0
    assert first.provider_prefix_last_pos == 2  # prefix_len - 1
    assert first.reuse_first_suffix_label_pos == 3  # prefix_len, index of first suffix token
    assert first.output_slot == 0  # prepend to suffix logprob sequence
    # Sample 2: provider pos 4 (token `5`) predicts `30` (original pos 5).
    assert second.reuse_idx_in_batch == 2
    assert second.provider_idx_in_batch == 0
    assert second.provider_prefix_last_pos == 4
    assert second.reuse_first_suffix_label_pos == 5
    assert second.output_slot == 0


def test_planner_no_shared_prefix_keeps_original_shapes():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=2))
    meta = planner.plan([[1, 2], [1, 3], [4, 5]], forward_id=1, micro_batch_id=1)

    assert not meta.has_sharing
    assert meta.group_ids == [-1, -1, -1]
    assert meta.input_keep_ranges == [(0, 2), (0, 2), (0, 2)]
    assert meta.reuse_specs == []
    assert meta.prefix_last_restore == []
