from prefix_sharing.core.batch_trim import (
    trim_inputs,
    trim_labels,
)
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


def _meta():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=3))
    return planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=10,
        micro_batch_id=20,
    )


def test_trim_inputs_and_labels_follow_metadata_ranges():
    meta = _meta()
    inputs = [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]]
    labels = [["p1", "p2", "s0", "s1", "eos"], ["p1", "p2", "s0", "s1", "s2", "eos"]]

    trimmed_inputs = trim_inputs(inputs, meta)
    trimmed_labels = trim_labels(labels, meta)

    assert trimmed_inputs.rows == [[1, 2, 3, 10, 11], [20, 21, 22]]
    assert trimmed_inputs.flattened == [1, 2, 3, 10, 11, 20, 21, 22]
    assert trimmed_inputs.cu_seqlens == meta.cu_seqlens_q
    assert trimmed_labels.rows == [["p1", "p2", "s0", "s1", "eos"], ["s1", "s2", "eos"]]
