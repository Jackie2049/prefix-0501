"""Framework patch integrations."""

from prefix_sharing.integrations.context import (
    PrefixSharingRuntimeContext,
    current_prefix_sharing_context,
    prefix_sharing_context,
)
from prefix_sharing.integrations.patch_manager import PatchHandle, PatchManager
from prefix_sharing.integrations.verl_mcore import (
    VerlMCoreBatchAdapter,
    VerlMCoreIntegration,
    VerlMCorePreparedBatch,
    enable_prefix_sharing,
    prefix_sharing_enabled,
)

__all__ = [
    "PatchHandle",
    "PatchManager",
    "PrefixSharingRuntimeContext",
    "VerlMCoreBatchAdapter",
    "VerlMCoreIntegration",
    "VerlMCorePreparedBatch",
    "current_prefix_sharing_context",
    "enable_prefix_sharing",
    "prefix_sharing_context",
    "prefix_sharing_enabled",
]
