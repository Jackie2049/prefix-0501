"""Framework patch integrations."""

from prefix_sharing.integrations.context import (
    PackedPrefixLastRestoreIndex,
    PrefixSharingRuntimeContext,
    current_prefix_sharing_context,
    prefix_sharing_runtime_context,
)
from prefix_sharing.integrations.patch_manager import PatchHandle, PatchManager
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.integrations.verl_mcore import (
    PrefixSharingRuntimeState,
    VerlMCoreIntegration,
    enable_prefix_sharing,
    prefix_sharing_enabled,
    build_prefix_sharing_micro_batch,
    restore_reuser_prefix_columns_2d,
)

__all__ = [
    "PatchHandle",
    "PatchManager",
    "PackedBatchLayout",
    "PackedPrefixLastRestoreIndex",
    "PrefixSharingRuntimeContext",
    "PrefixSharingRuntimeState",
    "VerlMCoreIntegration",
    "current_prefix_sharing_context",
    "enable_prefix_sharing",
    "prefix_sharing_runtime_context",
    "prefix_sharing_enabled",
    "build_prefix_sharing_micro_batch",
    "restore_reuser_prefix_columns_2d",
]
