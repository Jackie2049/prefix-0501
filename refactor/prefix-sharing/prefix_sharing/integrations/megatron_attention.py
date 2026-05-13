"""Megatron attention integration skeleton for phase 1."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.patch_manager import PatchHandle, PatchManager


class IntegrationUnavailable(RuntimeError):
    pass


@dataclass
class MegatronAttentionIntegration:
    config: PrefixSharingConfig
    backend: Any

    def install(self, model_config: Any | None = None) -> PatchHandle:
        self.config.validate(model_config=model_config, integrate_mode="verl_megatron_actor")
        attention_mod = self._import_attention_module()
        self_attention_cls = getattr(attention_mod, "SelfAttention", None)
        if self_attention_cls is None:
            raise IntegrationUnavailable("Megatron SelfAttention class was not found")

        original_forward = getattr(self_attention_cls, "forward", None)
        if original_forward is None:
            raise IntegrationUnavailable("Megatron SelfAttention.forward was not found")

        def patched_forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            from prefix_sharing.integrations.context import current_prefix_sharing_context

            ctx = current_prefix_sharing_context()
            if ctx is None:
                return original_forward(instance, *args, **kwargs)
            raise NotImplementedError(
                "Phase-1 Megatron attention patch entry is installed, but real QKV "
                "rewiring requires Megatron runtime dependencies and is covered by "
                "optional integration tests."
            )

        manager = PatchManager()
        try:
            manager.patch_attr(self_attention_cls, "forward", patched_forward)
            return manager.handle()
        except Exception:
            manager.rollback()
            raise

    @staticmethod
    def _import_attention_module() -> Any:
        try:
            return importlib.import_module("megatron.core.transformer.attention")
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("Megatron is not importable in this environment") from exc
