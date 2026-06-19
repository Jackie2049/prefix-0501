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
        log_probs = original_fn(logits, labels)

        from prefix_sharing.integrations.context import current_prefix_sharing_context

        ctx = current_prefix_sharing_context()
        if ctx is None or not ctx.prefix_last_restore_indices:
            return log_probs

        # logits 形态可能是 [N, V//tp] 或 [N, 1, V//tp]，统一 view 成 2D。
        # N = 裁剪后 packed 1D 总长度（provider 行完整含 prefix-last token）。
        logits_2d = logits.view(-1, logits.size(-1))

        for index in ctx.prefix_last_restore_indices:
            # interior 走 2D 复制路径，不需要 logits；只保存 prefix-last。
            if index.is_shared_prefix_interior:
                continue
            pos = index.provider_1d_pos
            if pos < 0:
                # chain-reuse 中间 reuser 的 keep_start-1 哨兵，无 packed slot。
                continue
            # clone 保留 autograd 图（restore 重算 logp 要走反向传播，禁止 detach）。
            saved = logits_2d[pos:pos + 1, :].clone()  # [1, V//tp]
            # key 约定：(reuser_row, target_2d_pos)，与 restore_reuser_prefix_columns_2d
            # 第 335 行 saved_key = (reuser_row, valid_col) 对齐。
            ctx.prefix_last_logits_saved[
                (index.reuse_idx_in_batch, index.target_2d_pos)
            ] = saved

        return log_probs

    return patched_fn