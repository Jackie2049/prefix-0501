"""Framework-independent prefix sharing semantics."""

from prefix_sharing.core.batch_trim import TrimmedBatch, trim_batch, trim_inputs, trim_labels, trim_loss_masks
from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.observability import PrefixSharingLayerStats, PrefixSharingStats
from prefix_sharing.core.prefix_detector import PrefixDetectionResult, PrefixReuseSpec, TriePrefixDetector
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PREFIX_STATE_TYPE_DELTANET_STATE,
    PrefixActivationSlotId,
    PrefixActivationStore,
    PrefixAttentionStore,
    PrefixDeltanetStore,
    StoredAttentionKV,
    StoredDeltanetState,
)
from prefix_sharing.core.planner import PrefixRestoreSpec, PrefixSharingPlan, PrefixSharingPlanner

__all__ = [
    "PrefixDetectionResult",
    "PrefixReuseSpec",
    "PREFIX_STATE_TYPE_ATTENTION_KV",
    "PREFIX_STATE_TYPE_DELTANET_STATE",
    "PrefixActivationSlotId",
    "PrefixActivationStore",
    "PrefixAttentionStore",
    "PrefixDeltanetStore",
    "PrefixSharingLayerStats",
    "PrefixSharingPlan",
    "PrefixSharingStats",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixRestoreSpec",
    "PrefixSharingPlanner",
    "StoredAttentionKV",
    "StoredDeltanetState",
    "TrimmedBatch",
    "TriePrefixDetector",
    "trim_batch",
    "trim_inputs",
    "trim_labels",
    "trim_loss_masks",
]
