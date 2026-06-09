"""Patch A: Attention.forward — Megatron Core v0.16.x

无 prefix-sharing context → 调用原始 forward
有 context → 自行 QKV + THD squeeze + runtime_adapters RoPE
             + backends build_kv/attention + self.linear_proj
"""

from __future__ import annotations

import importlib
from typing import Any


def make_attention_patch(original_forward: Any) -> Any:
    """创建 Attention.forward 的 patch wrapper。"""

    attn_mod = importlib.import_module("megatron.core.transformer.attention")
    _yarn_fn = getattr(
        attn_mod,
        "_yarn_get_concentration_factor_from_config",
        lambda config: 1.0,
    )

    def patched_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        from prefix_sharing.integrations.context import current_prefix_sharing_context

        ctx = current_prefix_sharing_context()
        if ctx is not None:
            # ── prefix-sharing path ──
            # phase 1: training, THD, no fusion, no output gate
            query, key, value = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                split_qkv=True,
                output_gate=False,
            )
            if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
                query = query.squeeze(1)
                key = key.squeeze(1)
                value = value.squeeze(1)

            # RoPE (使用 setup/runtime_adapters 适配 v0.16.0)
            from prefix_sharing.setup.runtime_adapters import (
                apply_positioned_rope_v016,
            )

            query, key = apply_positioned_rope_v016(
                self,
                query,
                key,
                rotary_pos_emb[0] if rotary_pos_emb else None,
                rotary_pos_emb[1] if rotary_pos_emb else None,
                ctx.kept_position_ids,
                packed_seq_params,
                mscale=_yarn_fn(self.config),
            )

            # KV expansion + attention (使用 backends)
            from prefix_sharing.backends.torch_ref import TorchReferenceBackend

            backend = ctx.backend or TorchReferenceBackend()
            tp_rank = _tensor_parallel_rank()
            layer_id = int(getattr(self, "layer_number", 0) or 0)

            expanded_key, expanded_value = backend.build_kv(
                key,
                value,
                ctx.store,
                ctx.prefix_sharing_plan,
                layer_id=layer_id,
                tp_rank=tp_rank,
            )
            core_attn_out = backend.attention(
                query,
                expanded_key,
                expanded_value,
                ctx.prefix_sharing_plan,
                attention_mask=attention_mask,
            )
            core_attn_out = core_attn_out.reshape(
                core_attn_out.size(0), 1, -1
            )
            output, bias = self.linear_proj(core_attn_out)
            return output, bias

        # ── normal path: 调用原始 forward ──
        return original_forward(
            self,
            hidden_states,
            attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )

    return patched_forward


def _tensor_parallel_rank() -> int:
    try:
        from megatron.core import parallel_state
        return int(parallel_state.get_tensor_model_parallel_rank())
    except Exception:
        return 0