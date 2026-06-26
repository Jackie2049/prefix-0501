"""Patch: Attention.forward — thin wrapper

无 context → 调用原始 forward
有 context → QKV + THD squeeze → delegate to integrations.prefix_attention

业务逻辑（RoPE、KV expansion、attention 计算）全部由 integrations 层处理，
本 patch 只负责 QKV 提取（attention module 交互）和 THD squeeze（格式适配）。
"""

from __future__ import annotations

from typing import Any


def patch_megatron_attention(original_forward: Any) -> Any:
    """创建 Attention.forward 的 patch wrapper。"""

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
        if ctx is None:
            # ── normal path: 调用原始 forward ──
            _result = original_forward(
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
            # ##### [PS-diag] OFF attn_outputs + rope_freqs_off + rope_emb dump #####
            import os as _os
            if _os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                from prefix_sharing.tools.diagnostic_dump import (
                    dump_attn_off, dump_rope_freqs_off, dump_rope_emb_layer,
                )
                from prefix_sharing.integrations.megatron_runtime import _unpack_rotary_pos_emb
                from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
                _attn_out = _result[0] if isinstance(_result, tuple) else _result
                _bs = (
                    len(packed_seq_params.cu_seqlens_q_padded) - 1
                    if (packed_seq_params is not None
                        and hasattr(packed_seq_params, "cu_seqlens_q_padded"))
                    else 0
                )
                dump_attn_off(_attn_out, packed_seq_params,
                              self.layer_number, _bs, self.config.num_layers)
                if rotary_pos_emb is not None:
                    _q_pos_emb, _k_pos_emb = _unpack_rotary_pos_emb(rotary_pos_emb)
                    dump_rope_freqs_off(_q_pos_emb, self.layer_number, self.config.num_layers)
                    # OFF post-RoPE Q/K dump: 提取 QKV + 手动 apply RoPE 后 dump，供与 ON 对比
                    # OFF 路径 position IDs 是标准的 0..seg_len-1（每 segment 内连续）
                    if (packed_seq_params is not None
                            and packed_seq_params.qkv_format == "thd"
                            and hasattr(packed_seq_params, "cu_seqlens_q_padded")):
                        _off_q, _off_k, _off_v = self.get_query_key_value_tensors(
                            hidden_states, key_value_states,
                            split_qkv=True, output_gate=False,
                        )
                        _off_q = _off_q.squeeze(1)
                        _off_k = _off_k.squeeze(1)
                        _cu = packed_seq_params.cu_seqlens_q_padded
                        _pos_list = [torch.arange(_cu[i+1] - _cu[i], device=_off_q.device)
                                     for i in range(len(_cu) - 1)]
                        _off_positions = torch.cat(_pos_list, dim=0).long()
                        _off_q_freqs = _q_pos_emb.index_select(0, _off_positions)
                        _off_k_freqs = (_k_pos_emb or _q_pos_emb).index_select(0, _off_positions)
                        _off_q_rope = apply_rotary_pos_emb(
                            _off_q.unsqueeze(1), _off_q_freqs,
                            config=self.config, cu_seqlens=None,
                        ).squeeze(1)
                        _off_k_rope = apply_rotary_pos_emb(
                            _off_k.unsqueeze(1), _off_k_freqs,
                            config=self.config, cu_seqlens=None,
                        ).squeeze(1)
                        dump_rope_emb_layer(
                            self.layer_number, _off_q_rope, _off_k_rope,
                            self.config.num_layers,
                        )
            # ##### [PS-diag] OFF attn_outputs + rope_freqs_off + rope_emb dump end #####
            return _result

        # ── prefix-sharing path ──
        # phase 1: training, THD, no fusion, no output gate

        # QKV extraction — attention module interaction, not business logic
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

        # delegate to verified integrations code
        from prefix_sharing.integrations.megatron_runtime import (
            prefix_attention,
        )

        result = prefix_attention(
            self,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            packed_seq_params,
        )
        if result is not None:
            return result

        # fallback: should not reach here if context is active,
        # but return original forward as safety net
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