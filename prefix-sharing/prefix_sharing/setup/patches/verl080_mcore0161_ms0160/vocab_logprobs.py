"""Patch: vocab_parallel_log_probs_from_logits — thin wrapper

无 context → 直接调用原始函数
有 context → 调用原始 + integrations.restore_suffix_first_log_probs_from_prefix

业务逻辑（PP stage 检查、global packed token length 验证、logprob 还原）
全部由 integrations 层处理，本 patch 只负责编排调用顺序。
"""

from __future__ import annotations

from typing import Any


def patch_megatron_vocab(original_fn: Any) -> Any:
    """创建 vocab_parallel_log_probs_from_logits 的 patch wrapper。"""

    def patched_fn(logits, labels):
        log_probs = original_fn(logits, labels)

        from prefix_sharing.integrations.context import (
            current_prefix_sharing_context,
        )

        ctx = current_prefix_sharing_context()
        if ctx is not None and ctx.prefix_last_restore_indices:
            from prefix_sharing.integrations.verl_mcore import (
                restore_suffix_first_log_probs_from_prefix,
            )

            log_probs = restore_suffix_first_log_probs_from_prefix(
                logits, labels, log_probs, original_fn,
            )

        return log_probs

    return patched_fn