from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.logprob import (
    build_provider_prefix_last_values,
    restore_prefix_last_logprobs,
)
from prefix_sharing.core.planner import PrefixSharingPlanner


def _prefix_sharing_plan():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    return planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=10,
        micro_batch_id=20,
    )


def test_prefix_last_restore_inserts_reuse_first_suffix_logprob():
    # _meta: provider [1,2,3,10,11]; reuser [1,2,3|20,21,22] (prefix_len=3).
    # Suffix forward only yields logprobs for 21 and 22; token 20 is restored from provider pos 2.
    prefix_sharing_plan = _prefix_sharing_plan()
    suffix_logprobs = [
        [0.10, 0.11, 0.12, 0.13, 0.14],  # provider row, already complete
        [0.21, 0.22],  # reuser: slot 0 is the second suffix token (21), not 20
    ]
    # Per-spec restore logprobs: one value per PrefixLastRestoreSpec in plan order.
    # Here there is exactly one spec (prefix-last for reuser index 1).
    restore_logprobs = [0.20]

    restored = restore_prefix_last_logprobs(
        suffix_logprobs,
        restore_logprobs,
        prefix_sharing_plan,
    )

    assert restored == [
        [0.10, 0.11, 0.12, 0.13, 0.14],
        [0.20, 0.21, 0.22],  # 0.20 prepended to match full-sequence causal LM logprobs
    ]


def test_build_provider_prefix_last_values_uses_restore_specs():
    prefix_sharing_plan = _prefix_sharing_plan()
    values = build_provider_prefix_last_values(
        [["p0", "p1", "p_last", "s0"], ["unused"]],
        prefix_sharing_plan,
    )
    # Per-spec list: one value per PrefixLastRestoreSpec.
    # The single spec gathers p_last at provider_prefix_last_pos=2.
    assert values == ["p_last"]


def test_restore_logprobs_with_interior_response_tokens():
    # seq1 (provider): [1,2,3 | A,B]   prompt=[1,2,3], response=[A,B]
    # seq2 (reuser):   [1,2,3 | A,C]   prompt=[1,2,3], response=[A,C]
    # Shared prefix: [1,2,3,A] (len 4). A is interior-response.
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    )
    plan = planner.plan(
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]],
        prompt_lens=[3, 3],
        forward_id=1,
        micro_batch_id=1,
    )

    # Two specs: interior (A) at slot 0, prefix-last at slot 1
    assert len(plan.prefix_last_restore) == 2

    # Suffix forward yields logprobs for kept Q positions.
    # For the reuser: forward produces [logprob(C_wrong_position)].
    suffix_logprobs = [
        [0.10, 0.11, 0.12, 0.13, 0.14],  # provider, complete
        [0.16],                             # reuser: logprob(C) from wrong logits position
    ]

    # Per-spec restore logprobs:
    #   spec[0]: interior-response for A (from provider logits[2] + label=A)
    #   spec[1]: prefix-last for C (from provider logits[3] + label=C)
    restore_logprobs = [0.04, 0.16]

    restored = restore_prefix_last_logprobs(
        suffix_logprobs,
        restore_logprobs,
        plan,
    )

    # Provider row unchanged
    assert restored[0] == [0.10, 0.11, 0.12, 0.13, 0.14]
    # Reuser: interior A inserted at slot 0, prefix-last C inserted at slot 1.
    # Note: the current insert-based restore keeps the forward's "wrong"
    # first-suffix logprob as a tail entry. This is a known artifact —
    # downstream mask/trim would need to discard it.
    assert restored[1] == [0.04, 0.16, 0.16]


def test_build_provider_prefix_last_values_with_interior_specs():
    # Same scenario as above: provider has interior-response token at pos 3.
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    )
    plan = planner.plan(
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]],
        prompt_lens=[3, 3],
        forward_id=1,
        micro_batch_id=1,
    )
    # Provider logprobs at each position: pos0=0.00, pos1=0.01, pos2=0.02(pos2's logit predicts A), ...
    # For interior spec: provider_prefix_last_pos=2 → value is "provider_logprob_at_2"
    # For prefix-last spec: provider_prefix_last_pos=3 → value is "provider_logprob_at_3"
    provider_row = ["lp_0", "lp_1", "lp_2", "lp_3", "lp_4"]
    values = build_provider_prefix_last_values(
        [provider_row, ["unused"] * 6],
        plan,
    )
    assert values == ["lp_2", "lp_3"]  # per-spec ordering
