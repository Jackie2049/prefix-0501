"""Framework patch integrations."""

from prefix_sharing.integrations.context import (
    PackedPrefixRestoreIndex,
    PrefixSharingRuntimeContext,
    current_prefix_sharing_context,
    prefix_sharing_runtime_context,
)
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo, get_megatron_parallel_info
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.integrations.verl_mcore import (
    PrefixSharingRuntimeState,
    build_prefix_sharing_micro_batch_verl070,
    build_prefix_sharing_micro_batch_verl080,
    restore_reuser_prefix_columns_2d,
    read_ps_config_from_engine_config,
)
from prefix_sharing.integrations.megatron_runtime import (
    prefix_attention,
)

__all__ = [
    "PackedBatchLayout",
    "PackedPrefixRestoreIndex",
    "MegatronParallelInfo",
    "PrefixSharingRuntimeContext",
    "PrefixSharingRuntimeState",
    "current_prefix_sharing_context",
    "prefix_sharing_runtime_context",
    "build_prefix_sharing_micro_batch_verl070",
    "build_prefix_sharing_micro_batch_verl080",
    "restore_reuser_prefix_columns_2d",
    "read_ps_config_from_engine_config",
    "prefix_attention",
    "get_megatron_parallel_info",
]
