"""Framework patch integrations."""

from prefix_sharing.integrations.context import (
    PrefixSharingRuntimeContext,
    current_prefix_sharing_context,
    prefix_sharing_context,
)
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

__all__ = [
    "PatchHandle",
    "PatchManager",
    "PrefixSharingRuntimeContext",
    "DensePrefixLastRestoreSpec",
    "MegatronActorPreparedMicroBatch",
    "VerlMCoreBatchAdapter",
    "VerlMCoreIntegration",
    "VerlMCorePreparedBatch",
    "current_prefix_sharing_context",
    "enable_prefix_sharing",
    "megatron_actor_prefix_sharing_context",
    "prefix_sharing_context",
    "prefix_sharing_enabled",
    "prepare_megatron_actor_micro_batch",
    "restore_megatron_actor_log_probs",
]
