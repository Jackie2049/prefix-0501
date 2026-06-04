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
        sequence_id: Any | None = None,
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
                sequence_id=sequence_id,
                **kwargs,
            )

        # Prefix-sharing path: do QKV projection ourselves, then hand off
        # to the runtime hook which handles RoPE, KV expansion, and attention.
        import torch

        # THD packed mode: QKV projection
        mixed_layer = self_attention_module.linear_qkv(hidden_states)

        # For remove-padding (THD) mode, Megatron stores projection config
        # on the module. The mixed qkv tensor shape is [total_tokens, heads * 3 * head_dim]
        # or [total_tokens, q_tokens + 2 * kv_tokens * head_dim] for GQA.
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
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )

        # Split QKV
        config = self_attention_module.config
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_query_groups
        hidden_size_per_head = config.kv_channels
        q_sz = num_q_heads * hidden_size_per_head
        kv_sz = num_kv_heads * hidden_size_per_head

        # Handle TP-split QKV
        tp_size = 1
        try:
            from megatron.core import parallel_state

            tp_size = parallel_state.get_tensor_model_parallel_world_size()
        except Exception:
            pass

        # In TP mode, heads are already split per rank
        tp_q_sz = q_sz // tp_size
        tp_kv_sz = kv_sz // tp_size

        query, key, value = torch.split(
            mixed_layer, [tp_q_sz, tp_kv_sz, tp_kv_sz], dim=-1
        )

        # Reshape to THD: [total_tokens, heads, head_dim]
        query = query.view(query.shape[0], num_q_heads // tp_size, hidden_size_per_head)
        key = key.view(key.shape[0], num_kv_heads // tp_size, hidden_size_per_head)
        value = value.view(value.shape[0], num_kv_heads // tp_size, hidden_size_per_head)

        from prefix_sharing.integrations.megatron_runtime import (
            maybe_run_prefix_sharing_attention,
        )

        result = maybe_run_prefix_sharing_attention(
            attention_module=self_attention_module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        if result is not None:
            return result

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
            sequence_id=sequence_id,
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
