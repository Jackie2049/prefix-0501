"""Patch: MegatronEngineWithLMHead.forward_step — verl 0.8.0 engine 架构

thin wrapper：消费 batch → 读 config → 构建状态 → 设 context → 喂回原始 forward_step。

所有业务逻辑（config 读取、batch 构建、layout 计算）由 integrations 层处理，
本 patch 只负责编排调用顺序和设置 runtime context。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def patch_verl_forward_step(original_forward_step: Any) -> Any:
    """创建 MegatronEngineWithLMHead.forward_step 的 patch wrapper。"""

    def patched_forward_step(
        self,
        batch_iter,
        model,
        logits_processor_func,
        postprocess_micro_batch_func,
    ):
        # ── 清理跨 micro-batch 残留（每个 forward_step 开头执行） ──
        from prefix_sharing.integrations.verl_mcore import clear_trimmed_valid_lengths
        clear_trimmed_valid_lengths()

        # ── 消费 micro-batch ──
        batch = next(batch_iter)

        # ── prefix-sharing: 读 config → 构建状态 ──
        ps_state = None

        from prefix_sharing.integrations.verl_mcore import (
            read_ps_config_from_engine_config,
            build_prefix_sharing_micro_batch_verl080,
        )
        from prefix_sharing.core.config import PrefixSharingConfig

        ps_config_raw = read_ps_config_from_engine_config(self.engine_config)
        ps_config = PrefixSharingConfig.from_raw(ps_config_raw)

        if ps_config.enable_prefix_sharing:
            # batch.to(device) 使 tensor 在目标设备上，
            # 原始 forward_step 会再次 batch.to(device)（幂等）
            from verl.utils.megatron_utils import get_device_id
            batch = batch.to(get_device_id())

            # 返回 (trimmed_batch, state) — 解包 tuple
            batch, ps_state = build_prefix_sharing_micro_batch_verl080(self, batch, ps_config)

        # ── 构造修改后的 iterator 喂回原始 forward_step ──
        modified_iter = iter([batch])

        # ── runtime context ──
        from prefix_sharing.integrations.context import prefix_sharing_runtime_context
        from contextlib import nullcontext

        context_manager = (
            prefix_sharing_runtime_context(ps_state)
            if ps_state is not None
            else nullcontext()
        )

        with context_manager:
            return original_forward_step(
                self,
                modified_iter,
                model,
                logits_processor_func,
                postprocess_micro_batch_func,
            )

    return patched_forward_step