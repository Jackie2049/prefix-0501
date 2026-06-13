from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.batch_trim import trim_inputs
from prefix_sharing.core.logprob import restore_prefix_last_logprobs
from prefix_sharing.core.planner import PrefixSharingPlanner


def test_phase_one_core_system_flow_without_framework_dependencies():
    input_ids = [
        [101, 102, 103, 201, 202],
        [101, 102, 103, 301, 302, 303],
        [901, 902],
    ]
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    )
    prefix_sharing_plan = planner.plan(input_ids, forward_id=123, micro_batch_id=456)
    trimmed = trim_inputs(input_ids, prefix_sharing_plan)

    assert prefix_sharing_plan.has_sharing
    assert trimmed.rows == [
        [101, 102, 103, 201, 202],
        [301, 302, 303],
        [901, 902],
    ]

    # Simulate logprob assembly after a framework computes:
    # - provider full logprobs
    # - reuse suffix output logprobs
    # - reuse first suffix logprob using provider prefix-last logits and reuse label
    suffix_logprobs = [
        [-1.0, -1.1, -1.2, -1.3, -1.4],
        [-3.1, -3.2],
        [-9.1, -9.2],
    ]
    # Per-spec restore logprobs: one value per PrefixLastRestoreSpec.
    # prefix_len=3 → interior positions 1,2 (2 specs) + prefix-last (1 spec) = 3 specs.
    restore_logprobs = [-0.1, -0.2, -3.0]

    restored = restore_prefix_last_logprobs(
        suffix_logprobs,
        restore_logprobs,
        prefix_sharing_plan,
    )

    assert restored[0] == [-1.0, -1.1, -1.2, -1.3, -1.4]
    assert restored[1] == [-0.1, -0.2, -3.0, -3.1, -3.2]
    assert restored[2] == [-9.1, -9.2]
