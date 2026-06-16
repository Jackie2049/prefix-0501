from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.batch_trim import trim_inputs
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
