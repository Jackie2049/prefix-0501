"""Framework patch integrations."""

from prefix_sharing.integrations.context import (
    PrefixSharingStats,
    PrefixSharingRuntimeContext,
    build_prefix_sharing_stats,
    current_prefix_sharing_context,
    prefix_sharing_context,
)
from prefix_sharing.integrations.parallel_env import ParallelEnv, current_parallel_env
from prefix_sharing.integrations.patch_manager import PatchHandle, PatchManager
from prefix_sharing.integrations.verl_mcore import (
    DensePrefixLastRestoreSpec,
    MegatronActorPreparedMicroBatch,
    VerlMCoreBatchAdapter,
    VerlMCoreIntegration,
    VerlMCorePreparedBatch,
    enable_prefix_sharing,
    megatron_actor_prefix_sharing_context,
    prefix_sharing_enabled,
    prepare_megatron_actor_micro_batch,
    restore_megatron_actor_log_probs,
)
from prefix_sharing.integrations.verl_dp_balance import (
    prefix_sharing_dp_balance_enabled,
    prefix_sharing_dp_balance_group_key,
    reorder_dataproto_for_prefix_group_dp_balance,
)

__all__ = [
    "PatchHandle",
    "PatchManager",
    "ParallelEnv",
    "PrefixSharingStats",
    "PrefixSharingRuntimeContext",
    "DensePrefixLastRestoreSpec",
    "MegatronActorPreparedMicroBatch",
    "VerlMCoreBatchAdapter",
    "VerlMCoreIntegration",
    "VerlMCorePreparedBatch",
    "build_prefix_sharing_stats",
    "current_prefix_sharing_context",
    "current_parallel_env",
    "enable_prefix_sharing",
    "megatron_actor_prefix_sharing_context",
    "prefix_sharing_context",
    "prefix_sharing_dp_balance_enabled",
    "prefix_sharing_dp_balance_group_key",
    "prefix_sharing_enabled",
    "prepare_megatron_actor_micro_batch",
    "reorder_dataproto_for_prefix_group_dp_balance",
    "restore_megatron_actor_log_probs",
]
