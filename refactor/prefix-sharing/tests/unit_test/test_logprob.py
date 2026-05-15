from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.logprob import (
    build_provider_prefix_last_values,
    restore_prefix_last_logprobs,
)
from prefix_sharing.core.planner import PrefixSharingPlanner


def _meta():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=3))
    return planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=10,
        micro_batch_id=20,
    )


def test_prefix_last_restore_inserts_reuse_first_suffix_logprob():
    meta = _meta()
    suffix_logprobs = [
        [0.10, 0.11, 0.12, 0.13, 0.14],
        [0.21, 0.22],
    ]
    first_suffix_logprobs_by_batch = [0.0, 0.20]

    restored = restore_prefix_last_logprobs(
        suffix_logprobs,
        first_suffix_logprobs_by_batch,
        meta,
    )

    assert restored == [
        [0.10, 0.11, 0.12, 0.13, 0.14],
        [0.20, 0.21, 0.22],
    ]


def test_build_provider_prefix_last_values_uses_restore_specs():
    meta = _meta()
    values = build_provider_prefix_last_values(
        [["p0", "p1", "p_last", "s0"], ["unused"]],
        meta,
    )
    assert values == [None, "p_last"]
