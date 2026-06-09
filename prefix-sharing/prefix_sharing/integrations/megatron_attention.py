"""Megatron attention integration entrypoint for phase 1.

Installs a monkey-patch on ``megatron.core.transformer.attention.SelfAttention.forward``
that intercepts each attention call and routes it through the prefix-sharing path
when a runtime context is active.

The patch wraps the original ``SelfAttention.forward``.  When no prefix-sharing
context is set (the common case), the wrapper delegates straight through with
zero overhead.  When a context is active, the wrapper:

1. Performs QKV projection using the module's own ``linear_qkv`` layer.
2. Reshapes into THD format.
3. Calls ``maybe_run_prefix_sharing_attention`` which owns RoPE, KV expansion,
   causal attention, and output projection.
4. Returns the result, skipping the rest of the original forward.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.patch_manager import PatchHandle, PatchManager


class IntegrationUnavailable(RuntimeError):
    pass


def _make_patched_forward(original_forward: Any) -> Any:
    """Create a wrapped ``SelfAttention.forward`` that tries prefix sharing first."""

    def patched_forward(
        self_attention_module: Any,
        hidden_states: Any,
        attention_mask: Any,
        key_value_states: Any | None = None,
        inference_params: Any | None = None,
        rotary_pos_emb: Any | None = None,
        rotary_pos_cos: Any | None = None,
        rotary_pos_sin: Any | None = None,
        attention_bias: Any | None = None,
        packed_seq_params: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        from prefix_sharing.integrations.context import current_prefix_sharing_context

        # Fast path: no prefix-sharing context → original Megatron path
        ctx = current_prefix_sharing_context()
        if ctx is None:
            return original_forward(
                self_attention_module,
                hidden_states,
                attention_mask=attention_mask,
                key_value_states=key_value_states,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )

        # Cross-attention (key_value_states provided) or inference mode:
        # fall through to original forward, prefix sharing only applies to
        # self-attention in training.
        if key_value_states is not None or inference_params is not None:
            return original_forward(
                self_attention_module,
                hidden_states,
                attention_mask=attention_mask,
                key_value_states=key_value_states,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )

        # Prefix-sharing path: do QKV projection ourselves, then hand off
        # to the runtime hook which handles RoPE, KV expansion, and attention.
        import logging
        import torch

        _ps_log = logging.getLogger(__name__)

        # For remove-padding (THD) mode, use Megatron's own QKV splitting
        # which correctly handles GQA interleaving (grouped-query layout).
        if hasattr(self_attention_module, "tp_group") and hasattr(
            self_attention_module, "tp_comm_overlap"
        ):
            # TP overlap mode - use the original forward instead
            return original_forward(
                self_attention_module,
                hidden_states,
                attention_mask=attention_mask,
                key_value_states=key_value_states,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )

        # Use Megatron's own get_query_key_value_tensors to correctly handle
        # GQA interleaving and TP head partitioning. This avoids replicating
        # the complex QKV split logic here.
        query, key, value = self_attention_module.get_query_key_value_tensors(
            hidden_states,
        )

        # For self attention, duplicate rotary_pos_emb if not already a tuple
        # (same pattern as SelfAttention.forward). The BSHD PS path expects
        # (q_pos_emb, k_pos_emb) tuple format.
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # In THD (remove-padding) mode, the tensor shape is [total_tokens, heads, head_dim]
        # but Megatron's get_query_key_value_tensors may output [sq, b, h, hn].
        # We need to flatten to [total_tokens, heads, head_dim] for our backends.
        if query.dim() == 4:
            # [sq, b, h, hn] -> [sq * b, h, hn] — THD mode has b=1, so just squeeze
            query = query.reshape(-1, query.shape[-2], query.shape[-1])
            key = key.reshape(-1, key.shape[-2], key.shape[-1])
            value = value.reshape(-1, value.shape[-2], value.shape[-1])
        elif query.dim() != 3:
            raise RuntimeError(
                f"prefix sharing expects QKV tensors with 3 or 4 dims, "
                f"got query.dim()={query.dim()}, shape={tuple(query.shape)}"
            )

        from prefix_sharing.integrations.megatron_runtime import (
            maybe_run_prefix_sharing_attention,
        )

        try:
            result = maybe_run_prefix_sharing_attention(
                attention_module=self_attention_module,
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
            )
            if result is not None:
                return result
        except Exception as exc:
            _ps_log.warning(
                "[PS] prefix-sharing attention failed at layer %s, "
                "falling back to original forward: %s",
                getattr(self_attention_module, "layer_number", "?"),
                exc,
            )

        # Hook returned None (e.g., linear attention layer) → fall through
        return original_forward(
            self_attention_module,
            hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )

    return patched_forward


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

        patched = _make_patched_forward(original_forward)
        mgr = PatchManager()
        mgr.patch_attr(self_attention_cls, "forward", patched)
        return mgr.handle()

    @staticmethod
    def _import_attention_module() -> Any:
        try:
            return importlib.import_module("megatron.core.transformer.attention")
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("Megatron is not importable in this environment") from exc
