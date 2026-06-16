"""Patch C: MegatronEngineWithLMHead.forward_step — verl 0.8.x

消费 batch → prefix-sharing reorg → 构建 context → 传回原始 forward_step
原始方法内部调用 vocab_parallel (Patch B) 和 model forward → Attention (Patch A)
三者联动，不需要修改 logits_processor 闭包
"""

from __future__ import annotations

from typing import Any


def make_forward_step_patch(original_forward_step: Any) -> Any:
    """创建 MegatronEngineWithLMHead.forward_step 的 patch wrapper。"""

    def patched_forward_step(
        self,
        batch_iter,
        model,
        logits_processor_func,
        postprocess_micro_batch_func,
    ):
        batch = next(batch_iter)

        # dynamic CP (phase 1 disallows, but still pass through)
        if getattr(self.engine_config, "dynamic_context_parallel", False):
            from verl.utils.megatron_utils import dynamic_cp_split_batch
            from megatron.core import parallel_state as mpu

            batch = dynamic_cp_split_batch(
                batch=batch,
                engine_config=self.engine_config,
                dp_size=mpu.get_data_parallel_world_size(),
                dp_rank=mpu.get_data_parallel_rank(),
            )

        from verl.utils.megatron_utils import get_device_id
        batch = batch.to(get_device_id())

        # ── prefix-sharing: micro-batch reorg ──
        ps_state = None
        ps_config = _read_ps_config(self.engine_config)
        if ps_config and ps_config.get("enable_prefix_sharing", False):
            from prefix_sharing.core.config import PrefixSharingConfig
            from prefix_sharing.core.planner import PrefixSharingPlanner
            from prefix_sharing.setup.runtime_adapters import (
                extract_sequences_from_batch,
                trim_batch,
                compute_packed_cu_seqlens,
                collect_kept_position_ids,
            )
            from prefix_sharing.integrations.context import (
                PrefixSharingRuntimeState,
            )
            from prefix_sharing.backends.torch_ref import TorchReferenceBackend

            config = PrefixSharingConfig.from_raw(ps_config)
            config.validate(
                model_config=self.tf_config,
                integrate_mode="verl_megatron_actor",
            )

            sequences, fmt = extract_sequences_from_batch(batch)
            plan = PrefixSharingPlanner(config).plan(sequences)

            if plan.has_sharing:
                batch = trim_batch(batch, plan, fmt)
                cu_seqlens = compute_packed_cu_seqlens(batch, plan, fmt)
                kept_position_ids = collect_kept_position_ids(batch, plan, fmt)

                ps_state = PrefixSharingRuntimeState(
                    prefix_sharing_plan=plan,
                    backend=TorchReferenceBackend(),
                    kept_position_ids=kept_position_ids,
                    packed_cu_seqlens=cu_seqlens,
                )

        # ── fused kernels guard ──
        if ps_state is not None:
            from verl.utils import tensordict_utils as tu

            use_fused = tu.get_non_tensor_data(
                batch, "use_fused_kernels", default=False
            )
            if use_fused:
                raise RuntimeError(
                    "prefix sharing phase 1 requires fused kernels disabled"
                )

        # ── runtime context ──
        from prefix_sharing.integrations.context import (
            prefix_sharing_runtime_context,
        )
        from contextlib import nullcontext

        ctx_mgr = (
            prefix_sharing_runtime_context(ps_state)
            if ps_state is not None
            else nullcontext()
        )

        # ── feed modified batch to original forward_step ──
        modified_iter = iter([batch])
        with ctx_mgr:
            return original_forward_step(
                self,
                modified_iter,
                model,
                logits_processor_func,
                postprocess_micro_batch_func,
            )

    return patched_forward_step


def _read_ps_config(engine_config: Any) -> dict | None:
    """从 engine_config 读取 prefix_sharing_config，不改 config 源码。"""
    override = getattr(engine_config, "override_transformer_config", {})
    if isinstance(override, dict):
        return override.get("prefix_sharing_config")
    return getattr(engine_config, "prefix_sharing_config", None)