"""Patch B: vocab_parallel_log_probs_from_logits — verl 0.8.x

无 context → 直接调用原始函数
有 context → 调用原始 + runtime_adapters.restore_logprobs
"""

from __future__ import annotations

from typing import Any


def make_vocab_logprobs_patch(original_fn: Any) -> Any:
    """创建 vocab_parallel_log_probs_from_logits 的 patch wrapper。"""

    def patched_fn(logits, labels):
        log_probs = original_fn(logits, labels)

        from prefix_sharing.integrations.context import (
            current_prefix_sharing_context,
        )

        ctx = current_prefix_sharing_context()
        if ctx is not None and ctx.prefix_last_restore_indices:
            from prefix_sharing.setup.runtime_adapters import restore_logprobs

            log_probs = restore_logprobs(
                logits,
                labels,
                log_probs,
                original_fn,
                ctx.prefix_last_restore_indices,
            )
        return log_probs

    return patched_fn