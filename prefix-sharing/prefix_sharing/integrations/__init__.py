"""Framework patch integrations."""

from prefix_sharing.integrations.context import (
    PrefixSharingRuntimeContext,
    current_prefix_sharing_context,
    prefix_sharing_runtime_context,
)
from prefix_sharing.integrations.patch_manager import PatchHandle, PatchManager
from prefix_sharing.integrations.verl_mcore import (
    PackedPackedPrefixLastRestoreSlot,
    PrefixSharingRuntimeState,
    VerlMCoreBatchAdapter,
    VerlMCoreIntegration,
    VerlMCorePrefixSharingBatch,
    enable_prefix_sharing,
    prefix_sharing_enabled,
    build_prefix_sharing_micro_batch,
    restore_megatron_actor_log_probs,
)

__all__ = [
    "PatchHandle",
    "PatchManager",
    "PrefixSharingRuntimeContext",
    "PackedPackedPrefixLastRestoreSlot",
    "PrefixSharingRuntimeState",
    "VerlMCoreBatchAdapter",
    "VerlMCoreIntegration",
    "VerlMCorePrefixSharingBatch",
    "current_prefix_sharing_context",
    "enable_prefix_sharing",
    "prefix_sharing_runtime_context",
    "prefix_sharing_enabled",
    "build_prefix_sharing_micro_batch",
    "restore_megatron_actor_log_probs",
]
