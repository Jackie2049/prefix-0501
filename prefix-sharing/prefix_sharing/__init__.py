"""Prefix sharing phase-1 package.

The public API intentionally starts from framework-independent core pieces.
Integrations install patches around verl/Megatron, but the semantics live here.
"""

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.prefix_detector import PrefixReuseSpec, TriePrefixDetector
from prefix_sharing.core.metadata import PrefixSharingBatchMeta
from prefix_sharing.core.planner import PrefixLastRestoreSpec, PrefixSharingPlanner
from prefix_sharing.integrations.verl_mcore import enable_prefix_sharing, prefix_sharing_enabled

__all__ = [
    "PrefixSharingBatchMeta",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixLastRestoreSpec",
    "PrefixReuseSpec",
    "PrefixSharingPlanner",
    "TriePrefixDetector",
    "enable_prefix_sharing",
    "prefix_sharing_enabled",
]
