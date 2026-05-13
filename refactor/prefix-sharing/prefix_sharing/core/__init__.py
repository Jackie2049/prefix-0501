"""Framework-independent prefix sharing semantics."""

from prefix_sharing.core.cache import PrefixKVCache, PrefixKVCacheKey
from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.prefix_detector import PrefixDetectionResult, PrefixReuseSpec, TriePrefixDetector
from prefix_sharing.core.logprob import (
    compute_token_logprobs_from_logits,
    gather_provider_prefix_last_logits,
    restore_prefix_last_logprobs_tensor,
)
from prefix_sharing.core.metadata import PrefixSharingBatchMeta
from prefix_sharing.core.planner import PrefixLastRestoreSpec, PrefixSharingPlanner

__all__ = [
    "PrefixDetectionResult",
    "PrefixReuseSpec",
    "PrefixKVCache",
    "PrefixKVCacheKey",
    "PrefixSharingBatchMeta",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixLastRestoreSpec",
    "PrefixSharingPlanner",
    "TriePrefixDetector",
    "compute_token_logprobs_from_logits",
    "gather_provider_prefix_last_logits",
    "restore_prefix_last_logprobs_tensor",
]
