"""Patch: vocab_parallel_log_probs_from_logits — thin wrapper.

无 context → 直接调用原始函数
有 context → 调用原始 + 保存 provider prefix-last 的 vocab 维 logits（含 autograd 图）

业务逻辑（restore 重组）由 forward_step 出口的
``restore_via_2d_unfold_verl080`` 完成，本 patch 只负责在 packed 1D logits
仍在、context 激活时，把 prefix-last 重算所需的 provider logits 保存到
``ctx.prefix_last_logits_saved``，供后续 restore 使用。

保存条件：仅非 interior（prefix-last）的 index。interior 走 2D 复制路径，
不需要 logits。
"""

from __future__ import annotations

from typing import Any


def patch_megatron_vocab(original_fn: Any) -> Any:
    """创建 vocab_parallel_log_probs_from_logits 的 patch wrapper。"""

    def patched_fn(logits, labels):
        # ##### [PS-diag] dump logits（ON/OFF 都 dump，必须在 original_fn 之前） #####
        # logits 形态 [N, V//tp]（或 [N,1,V//tp]），cmp_diag.cmp_logits_packed 会
        # reshape 成 token-major [N,V] 再用 cu_seqlens+prefix_lens 对齐。
        import os as _os
        _diag_on = _os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None
        if _diag_on:
            from prefix_sharing.tools.diagnostic_dump_verl080 import dump_logits_verl080
            dump_logits_verl080(logits)
        # ##### [PS-diag] dump logits end #####

        from prefix_sharing.integrations.context import current_prefix_sharing_context

        ctx = current_prefix_sharing_context()

        # ── 保存 provider prefix-last logits（必须在 original_fn 之前！） ──
        # original_fn = -vocab_parallel_cross_entropy，其 forward 对 logits 做 in-place：
        #   (1) logits -= logits_max        (megatron cross_entropy.py:45)
        #   (2) torch.exp(logits, out=...)  (cross_entropy.py:64-65)
        # 调用后 logits 已变成 exp(L-max) 废值。若在 original_fn 之后 clone，存的是废值，
        # restore 侧重算 logp(exp(L-max), label) ≠ logp(L, label)，logp 会完全错。
        # 必须在 original_fn 之前 clone 原始 logits（dump 同理）。
        if ctx is not None and ctx.prefix_restore_indices:
            # logits 形态可能是 [N, V//tp] 或 [N, 1, V//tp]，统一 view 成 2D。
            # N = 裁剪后 packed 1D 总长度（provider 行完整含 prefix-last token）。
            logits_2d = logits.view(-1, logits.size(-1))

            # ##### [PS-diag] 验证 packed 坐标对齐（logits N 是 valid 还是 padded） #####
            if _diag_on:
                _layout = ctx.packed_batch_layout
                print(
                    f"[PS-diag][packed-align] logits_N={logits_2d.shape[0]} "
                    f"total_padded={_layout.total_padded_length} "
                    f"total_valid={_layout.total_valid_length} "
                    f"has_padding={_layout.has_padding}",
                    flush=True,
                )
                for _idx in ctx.prefix_restore_indices:
                    if _idx.restore_type == "restore_prefix_interior":
                        continue
                    print(
                        f"[PS-diag][packed-align] reuser={_idx.reuse_idx_in_batch} "
                        f"provider={_idx.provider_idx_in_batch} "
                        f"provider_1d_pos={_idx.provider_1d_pos} "
                        f"target_2d_pos={_idx.target_2d_pos}",
                        flush=True,
                    )
            # ##### [PS-diag] 验证 packed 坐标对齐 end #####

            for index in ctx.prefix_restore_indices:
                # interior 走 2D 复制路径，不需要 logits；只保存 prefix-last。
                if index.restore_type == "restore_prefix_interior":
                    continue
                pos = index.provider_1d_pos
                key = (index.reuse_idx_in_batch, index.target_2d_pos)
                if pos < 0:
                    # 不应再发生：_build_prefix_last_restore_indices 已对 prefix-last
                    # 二次 strict 解析到 packed 真含 target_pos 的祖先。若到这里说明
                    # 解析逻辑有遗漏，直接 raise 暴露，避免下游 restore 静默 KeyError。
                    raise RuntimeError(
                        f"[vocab_logprobs] prefix-last spec got provider_1d_pos<0 "
                        f"after strict resolve; key={key} provider_1d_pos={pos}. "
                        f"_build_prefix_last_restore_indices 解析逻辑可能有遗漏。"
                    )
                # clone 保留 autograd 图（restore 重算 logp 要走反向传播，禁止 detach）。
                saved = logits_2d[pos:pos + 1, :].clone()  # [1, V//tp]
                if saved.shape[0] == 0:
                    raise RuntimeError(
                        f"[vocab_logprobs] empty logits slice: key={key} "
                        f"pos={pos} N={logits_2d.shape[0]} — strict resolve 返回的 "
                        f"packed 位置越界"
                    )
                # key 约定：(reuser_row, target_2d_pos)，与 restore_reuser_prefix_columns_2d
                # 的 saved_key = (reuser_row, valid_col) 对齐。
                ctx.prefix_last_logits_saved[key] = saved

        # 调原始函数（此后 logits 被 in-place 改成 exp(L-max)，但 dump/save 已完成）
        log_probs = original_fn(logits, labels)
        return log_probs

    return patched_fn