"""Prefix sharing phase-1 package.

The public API intentionally starts from framework-independent core pieces.
Integrations install patches around verl/Megatron, but the semantics live here.
"""

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.model_spec import ModelSpec, QWEN3_6_27B
from prefix_sharing.core.prefix_detector import PrefixReuseSpec, TriePrefixDetector
from prefix_sharing.core.planner import PrefixLastRestoreSpec, PrefixSharingPlan, PrefixSharingPlanner
from prefix_sharing.integrations.verl_mcore import enable_prefix_sharing, prefix_sharing_enabled

__all__ = [
    "PrefixSharingPlan",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixLastRestoreSpec",
    "PrefixReuseSpec",
    "PrefixSharingPlanner",
    "TriePrefixDetector",
    "ModelSpec",
    "QWEN3_6_27B",
    "enable_prefix_sharing",
    "prefix_sharing_enabled",
    "diagnose",
]


def diagnose(config: "PrefixSharingConfig | None" = None) -> "dict[str, object]":
    """Run prefix-sharing diagnostics and return a status dict.

    Useful for verifying setup before training. Example::

        import prefix_sharing
        status = prefix_sharing.diagnose()
        for k, v in status.items():
            print(f"  {k}: {v}")

    Returns a dict with keys like ``torch_available``, ``cuda_available``,
    ``flash_attn_available``, ``backend``, ``megatron_available``, etc.
    """
    info: dict[str, object] = {}

    # PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["torch_available"] = False

    # flash-attn
    try:
        import flash_attn
        info["flash_attn_version"] = flash_attn.__version__
        info["flash_attn_available"] = True
    except ImportError:
        info["flash_attn_available"] = False

    # Megatron
    try:
        import megatron.core  # noqa: F401
        info["megatron_available"] = True
    except ImportError:
        info["megatron_available"] = False

    # verl
    try:
        import verl  # noqa: F401
        info["verl_available"] = True
    except ImportError:
        info["verl_available"] = False

    # Backend resolution
    if config is not None:
        info["config_backend"] = config.backend
        info["config_enabled"] = config.enable_prefix_sharing
        if config.enable_prefix_sharing:
            try:
                from prefix_sharing.backends.factory import get_backend_instance
                backend = get_backend_instance(config)
                info["resolved_backend"] = backend.capabilities.name
            except Exception as exc:
                info["resolved_backend"] = f"ERROR: {exc}"

    return info
