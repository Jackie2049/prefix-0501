"""Framework-independent prefix sharing semantics."""

from prefix_sharing.core.batch_trim import TrimmedBatch, trim_batch, trim_inputs, trim_labels, trim_loss_masks
from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.observability import (
    PrefixSharingLayerStats,
    PrefixSharingStats,
    current_prefix_sharing_stats,
)
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
from prefix_sharing.core.logprob import (
    build_provider_prefix_last_values,
    compute_token_logprobs_from_logits,
    gather_provider_prefix_last_logits,
    restore_prefix_last_logprobs,
    restore_prefix_last_logprobs_tensor,
)
from prefix_sharing.core.planner import PrefixLastRestoreSpec, PrefixSharingPlan, PrefixSharingPlanner

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
    "PrefixSharingStats",
    "current_prefix_sharing_stats",
    "PrefixSharingPlan",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixLastRestoreSpec",
    "PrefixSharingPlanner",
    "StoredAttentionKV",
    "StoredDeltanetState",
    "TrimmedBatch",
    "TriePrefixDetector",
    "build_provider_prefix_last_values",
    "compute_token_logprobs_from_logits",
    "gather_provider_prefix_last_logits",
    "restore_prefix_last_logprobs",
    "restore_prefix_last_logprobs_tensor",
    "trim_batch",
    "trim_inputs",
    "trim_labels",
    "trim_loss_masks",
]
