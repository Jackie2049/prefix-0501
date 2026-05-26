"""RoPE integration guards for phase 1."""

from __future__ import annotations

from typing import Any

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError


def validate_rope_config(config: PrefixSharingConfig, model_config: Any | None = None) -> None:
    try:
        config.validate(model_config=model_config, integrate_mode="verl_megatron_actor")
    except PrefixSharingConfigError:
        raise
