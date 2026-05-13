"""verl Megatron actor integration skeleton."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.megatron_attention import IntegrationUnavailable, MegatronAttentionIntegration
from prefix_sharing.integrations.patch_manager import PatchHandle


@dataclass
class VerlMCoreIntegration:
    config: PrefixSharingConfig
    backend: Any | None = None

    def install(self, model_config: Any | None = None) -> PatchHandle:
        self.config.validate(model_config=model_config, integrate_mode="verl_megatron_actor")
        self._ensure_verl_importable()
        backend = self.backend or TorchReferenceBackend()
        return MegatronAttentionIntegration(config=self.config, backend=backend).install(
            model_config=model_config
        )

    @staticmethod
    def _ensure_verl_importable() -> None:
        try:
            importlib.import_module("verl")
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("verl is not importable in this environment") from exc
