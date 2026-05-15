"""Framework-independent prefix sharing semantics."""

from prefix_sharing.core.cache import PrefixKVCache, PrefixKVSlotId
from prefix_sharing.core.batch_trim import TrimmedBatch, trim_batch, trim_inputs, trim_labels, trim_loss_masks
from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.prefix_detector import PrefixDetectionResult, PrefixReuseSpec, TriePrefixDetector
from prefix_sharing.core.logprob import (
    build_provider_prefix_last_values,
    compute_token_logprobs_from_logits,
    gather_provider_prefix_last_logits,
    restore_prefix_last_logprobs,
    restore_prefix_last_logprobs_tensor,
)
from prefix_sharing.core.metadata import PrefixSharingBatchMeta
from prefix_sharing.core.planner import PrefixLastRestoreSpec, PrefixSharingPlanner

__all__ = [
    "PrefixDetectionResult",
    "PrefixReuseSpec",
    "PrefixKVCache",
    "PrefixKVSlotId",
    "PrefixSharingBatchMeta",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixLastRestoreSpec",
    "PrefixSharingPlanner",
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
