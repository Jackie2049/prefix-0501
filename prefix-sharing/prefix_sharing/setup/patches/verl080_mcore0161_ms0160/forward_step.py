"""Patch: MegatronEngineWithLMHead.forward_step — verl 0.8.0 engine 架构

核心编排 patch：消费 batch → prefix-sharing reorg → 构建 runtime context →
传回原始 forward_step。

设计要点：
1. 不重复原始 forward_step 中的逻辑（dynamic CP、batch.to 等），只做
   prefix-sharing 特有的微批次重组
2. 消费 batch_iter 中的 batch，做 prefix-sharing 处理后，构造新的
   单元素 iterator 喂回原始 forward_step
3. 原始 forward_step 内部的 logits_processor 闭包会调用
   vocab_parallel_log_probs_from_logits（由 vocab_logprobs patch 拦截做 restore）
4. 原始 forward_step 内部的 model forward 会经过 Attention.forward
   （由 attention patch 拦截做 KV expansion）

三者联动：forward_step（本 patch）设 context → attention patch 读 context →
vocab_logprobs patch 读 context。不需要修改原始 logits_processor 闭包。
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
        # ── 消费 micro-batch ──
        batch = next(batch_iter)

        # ── prefix-sharing: micro-batch重组 ──
        ps_state = None
        ps_config_raw = _read_ps_config(self.engine_config)

        # 与 v070 行为一致：先看 prefix_sharing_config，
        # 如果没设置再看 ENABLE_PREFIX_SHARING 环境变量。
        # PrefixSharingConfig.from_raw() 实现了这个分层逻辑。
        from prefix_sharing.core.config import PrefixSharingConfig

        ps_config = PrefixSharingConfig.from_raw(ps_config_raw)

        if ps_config.enable_prefix_sharing:
            # 必须在 batch.to(device) 之后做 prefix-sharing 处理，
            # 因为 extract_sequences 需要 tensor 操作。
            # 但原始 forward_step 会再次 batch.to(device)，
            # TensorDict.to() 对已在目标设备上的 batch 是幂等的。
            from verl.utils.megatron_utils import get_device_id

            batch = batch.to(get_device_id())

            ps_state = _build_ps_state(self, batch, ps_config)

            if ps_state is not None:
                # fused kernels guard
                from verl.utils import tensordict_utils as tu

                use_fused = tu.get_non_tensor_data(
                    batch, "use_fused_kernels", default=False
                )
                if use_fused:
                    raise RuntimeError(
                        "prefix sharing phase 1 requires fused kernels disabled"
                    )

                # dynamic CP guard
                if getattr(self.engine_config, "dynamic_context_parallel", False):
                    raise RuntimeError(
                        "prefix sharing phase 1 does not support dynamic context parallel"
                    )

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


def _read_ps_config(engine_config: Any) -> dict | None:
    """从 engine_config 读取 prefix_sharing_config，不改 config 源码。

    verl engine_config 中 prefix_sharing_config 的位置：
    1. engine_config.override_transformer_config 是 dict → 从 dict 中取
    2. engine_config.override_transformer_config 是对象 → 从对象属性取
    3. 回退 → engine_config.prefix_sharing_config（直接挂在 engine_config 上）
    """
    override = getattr(engine_config, "override_transformer_config", None)
    if override is not None:
        if isinstance(override, dict):
            return override.get("prefix_sharing_config")
        return getattr(override, "prefix_sharing_config", None)
    return getattr(engine_config, "prefix_sharing_config", None)


def _build_ps_state(
    engine_self: Any,
    batch: Any,
    ps_config: PrefixSharingConfig,
) -> Any | None:
    """从 batch 中提取序列 → plan → trim → 构建 PrefixSharingRuntimeState。

    返回 None 表示当前 micro-batch 没有共享前缀，不做处理。

    ps_config 参数已经是 PrefixSharingConfig 对象（由调用方
    通过 PrefixSharingConfig.from_raw() 解析完成），不需要再次 from_raw。
    """
    from prefix_sharing.core.planner import PrefixSharingPlanner
    from prefix_sharing.setup.runtime_adapters import (
        extract_sequences_from_batch,
        trim_batch,
        compute_packed_cu_seqlens,
        collect_kept_position_ids,
    )
    from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
    from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
    from prefix_sharing.backends.torch_ref import TorchReferenceBackend

    # ── 阶段 1: 配置校验 ──
    config = ps_config
    use_remove_padding = getattr(
        engine_self.engine_config, "use_remove_padding", True
    )
    # PATH 2: use_remove_padding=False → validate_for_engine 内部报错
    # PATH 3: multi-modal → validate_for_engine 后续 PR 扩展
    config.validate_for_engine(use_remove_padding=use_remove_padding)

    # ── 阶段 2: 从 batch 提取序列 ──
    # PATH 4: 数据格式不匹配 → runtime_adapters 内部处理
    sequences, fmt = extract_sequences_from_batch(batch)

    # ── 阶段 3: 前缀共享规划 ──
    plan = PrefixSharingPlanner(config).plan(sequences)

    # ── PATH 5: 当前 micro-batch 无共享前缀 ──
    if not plan.has_sharing:
        logger.info("[PS][prepare] PATH 5: no sharing detected")
        return None

    # ── 阶段 4: 修剪 batch（去掉 reuser 的前缀 token）──
    batch = trim_batch(batch, plan, fmt)

    # ── 阶段 5: 构建运行时布局 ──
    cu_seqlens = compute_packed_cu_seqlens(batch, plan, fmt)
    kept_position_ids = collect_kept_position_ids(batch, plan, fmt)

    from prefix_sharing.backends.packed_layout import PackedBatchLayout

    packed_layout = PackedBatchLayout.from_cu_seqlens(
        cu_seqlens=cu_seqlens,
        plan=plan,
    )
    parallel_info = MegatronParallelInfo.from_runtime()

    state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        backend=TorchReferenceBackend(),
        packed_batch_layout=packed_layout,
        parallel_info=parallel_info,
        kept_position_ids=kept_position_ids,
    )

    # ── PATH 6: 检测到共享前缀，构建成功 ──
    logger.info(
        "[PS][prepare] PATH 6: sharing detected, "
        "plan=%s, layout=%s",
        plan,
        packed_layout,
    )
    return state