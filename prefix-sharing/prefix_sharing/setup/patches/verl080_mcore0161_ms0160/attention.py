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
            # diag: hook 截获 original_forward 内部 apply_rotary_pos_emb 的真实 post-RoPE
            # Q/K（mcore Attention.forward THD prefill 每层调两次：先 Q 后 K）。post-RoPE Q/K
            # 是 forward 内部中间变量，唯一能拿到真实张量的办法就是 hook 那个模块级 rotary 函数。
            import os as _os
            from contextlib import nullcontext
            _diag_on = _os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None
            if _diag_on and rotary_pos_emb is not None:
                from prefix_sharing.tools.diagnostic_dump_verl080 import capture_rope_qk
                _rope_cm = capture_rope_qk(self)
            else:
                _rope_cm = nullcontext()
            with _rope_cm as _rope_caps:
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
            # ##### [PS-diag] OFF attn_outputs + rope_freqs + rope_postqk/preqk dump #####
            if _diag_on:
                import torch  # 仅用于构造 positions（freqs 切 per-token + Q/K debug）
                from prefix_sharing.tools.diagnostic_dump import (
                    dump_attn_off, dump_rope_freqs,
                )
                from prefix_sharing.tools.diagnostic_dump_verl080 import (
                    dump_rope_postqk_verl080, dump_rope_preqk_verl080, dump_full_kv_off,
                )
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
                    # OFF post-RoPE Q/K：直接用 hook 截获的真实张量
                    # （original_forward 内部 apply_rotary_pos_emb 的返回值），
                    # captures[0]=Q, captures[1]=K。不再 get_query_key_value_tensors + 重算 RoPE。
                    # _rope_caps = (qk_captures, v_captures)，均 in-context 截获
                    if _rope_caps is not None:
                        _qk_caps, _v_caps = _rope_caps
                    else:
                        _qk_caps, _v_caps = None, None
                    if _qk_caps is not None and len(_qk_caps) >= 2:
                        # 每个捕获是 {"pre": 旋转前, "post": 旋转后}
                        _q_cap, _k_cap = _qk_caps[0], _qk_caps[1]
                        # post-RoPE（旋转后）
                        dump_rope_postqk_verl080(
                            self.layer_number, _q_cap["post"], _k_cap["post"],
                            self.config.num_layers, positions=_off_positions,
                        )
                        # pre-RoPE（旋转前，纯 QKV 投影）
                        dump_rope_preqk_verl080(
                            self.layer_number, _q_cap["pre"], _k_cap["pre"],
                            self.config.num_layers,
                        )
                        # full KV（post-RoPE K + V，均 in-context 截获，不再事后 re-call）
                        _off_v = _v_caps[-1] if _v_caps else None
                        if _off_v is not None and _off_v.dim() > 2:
                            _off_v = _off_v.squeeze(1)
                        if _off_v is not None:
                            dump_full_kv_off(
                                self.layer_number, _k_cap["post"], _off_v,
                                self.config.num_layers,
                            )
                        else:
                            print(f"[PS-diag] OFF full_kv L{self.layer_number}: V 未截获; skip",
                                  flush=True)
                    else:
                        print(
                            f"[PS-diag] OFF rope_postqk L{self.layer_number}: "
                            f"expected 2 captures (Q,K), got "
                            f"{len(_qk_caps) if _qk_caps is not None else 'None'}; skip",
                            flush=True,
                        )
            # ##### [PS-diag] OFF attn_outputs + rope_freqs + rope_postqk/preqk dump end #####
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