from prefix_sharing.core.batch_trim import (
    trim_inputs,
    trim_labels,
    trim_loss_masks,
)
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


def _prefix_sharing_plan():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    return planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=10,
        micro_batch_id=20,
    )


def test_trim_inputs_and_labels_follow_metadata_ranges():
    prefix_sharing_plan = _prefix_sharing_plan()
    inputs = [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]]
    labels = [["p1", "p2", "s0", "s1", "eos"], ["p1", "p2", "s0", "s1", "s2", "eos"]]

    trimmed_inputs = trim_inputs(inputs, prefix_sharing_plan)
    trimmed_labels = trim_labels(labels, prefix_sharing_plan)

    assert trimmed_inputs.rows == [[1, 2, 3, 10, 11], [20, 21, 22]]
    assert trimmed_inputs.flattened == [1, 2, 3, 10, 11, 20, 21, 22]
    assert trimmed_inputs.cu_seqlens == prefix_sharing_plan.cu_seqlens_q
    assert trimmed_labels.rows == [["p1", "p2", "s0", "s1", "eos"], ["s1", "s2", "eos"]]


def test_trim_loss_masks_follow_metadata_ranges():
    prefix_sharing_plan = _prefix_sharing_plan()
    # loss_masks use the same keep_ranges as labels
    loss_masks = [[1.0, 1.0, 0.5, 0.5, 0.5], [1.0, 1.0, 0.5, 0.5, 0.5, 0.5]]

    trimmed = trim_loss_masks(loss_masks, prefix_sharing_plan)

    assert trimmed.rows == [[1.0, 1.0, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    assert trimmed.flattened == [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert trimmed.cu_seqlens == prefix_sharing_plan.cu_seqlens_q
