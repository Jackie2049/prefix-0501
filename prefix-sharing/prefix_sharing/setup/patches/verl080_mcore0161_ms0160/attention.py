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
            # post-RoPE Q/K / full_kv / preqk 由 Megatron attention.py 侵入式 dump 写入；
            # patch 层只负责 attn_outputs + rope_freqs（侵入式未覆盖的）。
            import os as _os
            _diag_on = _os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None
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
            # ##### [PS-diag] OFF attn_outputs + rope_freqs dump #####
            if _diag_on:
                import torch  # 仅用于构造 positions（freqs 切 per-token + Q/K debug）
                from prefix_sharing.tools.diagnostic_dump import (
                    dump_attn_off, dump_rope_freqs,
                )
                # rope_postqk / preqk / full_kv 已由 Megatron 侵入式 dump 覆盖
                from prefix_sharing.integrations.megatron_runtime import _unpack_rotary_pos_emb
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
                    # OFF 标准 positions（每 segment 内 0..seg-1）：切 per-token freqs + Q/K debug
                    _off_positions = None
                    if (packed_seq_params is not None
                            and hasattr(packed_seq_params, "cu_seqlens_q_padded")):
                        _cu = packed_seq_params.cu_seqlens_q_padded
                        # device 必须显式到 _cu.device(GPU)：arange 默认 CPU，否则后面
                        # _q_pos_emb.index_select(0, _off_positions) 会 device 不匹配崩 forward。
                        _off_positions = torch.cat([
                            torch.arange(int(_cu[i + 1] - _cu[i]), device=_cu.device)
                            for i in range(len(_cu) - 1)
                        ]).long()
                    # rope_freqs：存 per-token 角度（与 ON 同款），统一 rope_freqs.pt
                    if _off_positions is not None:
                        dump_rope_freqs(
                            _q_pos_emb.index_select(0, _off_positions),
                            self.layer_number, self.config.num_layers,
                        )
                    # rope_postqk + full_kv 已由 megatron attention.py 侵入式 dump
                    # （rotary block 之后），不在 patch 层重复——避免 hook 二次 flush
                    # 覆盖侵入式已写好的完整 24 层文件。
            # ##### [PS-diag] OFF attn_outputs + rope_freqs dump end #####
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

        # ##### [PS-diag] ON pre-RoPE Q/K/V 统一 dump（get_qkv 之后、RoPE 之前）#####
        # 全部在此点 dump（squeeze 后、_apply_positioned_rope / build_kv 之前），
        # 与 OFF baseline（hook 在 get_qkv 输出处截）同口径，集中对比，避免分散。
        import os as _os
        if _os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump_verl080 import (
                dump_rope_preqk_verl080, dump_build_kv_input_v_on,
            )
            try:
                dump_rope_preqk_verl080(self.layer_number, query, key,
                                        self.config.num_layers)
            except Exception as _e:
                print(f"rope_preqk (pre-RoPE Q/K) dump failed: {_e}", flush=True)
            try:
                dump_build_kv_input_v_on(self.layer_number, value,
                                         self.config.num_layers)
            except Exception as _e:
                print(f"build_kv_input_v (pre-RoPE V) dump failed: {_e}", flush=True)
            # [PS-diag] dump hidden_states for input-level comparison
            try:
                from prefix_sharing.tools.diagnostic_dump_verl080 import dump_hidden_states_on
                dump_hidden_states_on(self.layer_number, hidden_states,
                                      self.config.num_layers)
            except Exception as _e:
                print(f"hidden_states dump failed: {_e}", flush=True)
        # ##### [PS-diag] ON pre-RoPE Q/K/V dump end #####

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