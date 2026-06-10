"""Prefix sharing phase-1 package.

The public API intentionally starts from framework-independent core pieces.
Integrations install patches around verl/Megatron, but the semantics live here.

Monkey-patch activation:
    When ENABLE_PREFIX_SHARING is set (1/true/yes/on), importing this package
    automatically calls setup.install() which detects the verl/Megatron/MindSpeed
    versions and applies the matching patch set via monkey patch injection.
    No modification to verl or Megatron source code is required.

    If the environment variable is not set, or the version combo is not in the
    compat matrix, no patches are applied and prefix_sharing behaves as a
    pure library (core + backends).
"""

import os
import logging

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.prefix_detector import PrefixReuseSpec, TriePrefixDetector
from prefix_sharing.core.planner import PrefixLastRestoreSpec, PrefixSharingPlan, PrefixSharingPlanner
from prefix_sharing.integrations.verl_mcore import enable_prefix_sharing, prefix_sharing_enabled

logger = logging.getLogger(__name__)

__all__ = [
    "PrefixSharingPlan",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixLastRestoreSpec",
    "PrefixReuseSpec",
    "PrefixSharingPlanner",
    "TriePrefixDetector",
    "enable_prefix_sharing",
    "prefix_sharing_enabled",
]

# ── Monkey-patch auto-activation ──
# When ENABLE_PREFIX_SHARING is set, automatically detect versions and
# inject patches. This is the primary entry point for verl080+ environments
# where prefix-sharing is injected via monkey patch instead of invasive edits.
#
# For verl_v070 environments, the invasive import in megatron_actor.py
# still works (enable_prefix_sharing / prefix_sharing_enabled), but the
# setup module is not invoked — the old integration code handles everything.
#
# The setup.install() call is safe even when verl/Megatron are not present:
# if the detected versions match no compat matrix entry, it raises
# IncompatibleEnvironment which we catch and log as a warning (no patches
# applied, training proceeds normally without prefix-sharing).
_patch_handle = None  # module-level reference for introspection / rollback


def _auto_install_patches() -> None:
    """Auto-activate monkey patches when ENABLE_PREFIX_SHARING is enabled.

    Only runs once per process. If the environment combo is not in the
    compat matrix, logs a warning and proceeds without patches.
    """
    global _patch_handle

    env_val = os.getenv("ENABLE_PREFIX_SHARING", "").strip().lower()
    if env_val not in {"1", "true", "yes", "on", "y"}:
        return

    if _patch_handle is not None:
        logger.info("[PS] Patches already installed, skipping.")
        return

    try:
        from prefix_sharing.setup import install
        _patch_handle = install()
        logger.info("[PS] Auto-activation succeeded: %s", _patch_handle.describe())
    except Exception as exc:
        # IncompatibleEnvironment or import errors — log and continue
        # Training proceeds normally without prefix-sharing patches.
        logger.warning(
            "[PS] Auto-activation skipped: %s. "
            "Training will proceed without prefix-sharing patches.",
            exc,
        )


_auto_install_patches()
