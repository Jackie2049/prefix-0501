"""Patch: MegatronEngineWithLMHead.forward_step — verl 0.8.0 engine 架构

thin wrapper：消费 batch → 读 config → 构建状态 → 设 context → 喂回原始 forward_step。

所有业务逻辑（config 读取、batch 构建、layout 计算）由 integrations 层处理，
本 patch 只负责编排调用顺序和设置 runtime context。
"""

from __future__ import annotations

from typing import Any


def _ps_forward_step_probe(event: str, **fields: Any) -> None:
    """Emit a compact rank-aware probe for distributed hang diagnosis."""

    try:
        from prefix_sharing.integrations.parallel_info import get_megatron_parallel_info

        parallel_info = get_megatron_parallel_info()
        rank_prefix = (
            f"global_rank={parallel_info.global_rank} "
            f"tp_rank={parallel_info.tp_rank}/tp_size={parallel_info.tp_size} "
            f"pp_rank={parallel_info.pp_rank}/pp_size={parallel_info.pp_size} "
            f"cp_rank={parallel_info.cp_rank}/cp_size={parallel_info.cp_size}"
        )
    except Exception as exc:
        rank_prefix = f"parallel_info_unavailable={type(exc).__name__}:{exc}"

    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {field_text}" if field_text else ""
    print(f"[PS][forward_step][{rank_prefix}] {event}{suffix}", flush=True)


def _describe_batch(batch: Any) -> str:
    try:
        keys = list(batch.keys())
    except Exception:
        keys = []

    pieces = [f"type={type(batch).__name__}", f"keys={keys[:8]}"]
    for key in ("input_ids", "attention_mask", "position_ids", "labels"):
        try:
            value = batch[key]
        except Exception:
            continue
        shape = getattr(value, "shape", None)
        is_nested = getattr(value, "is_nested", None)
        pieces.append(f"{key}_shape={tuple(shape) if shape is not None else None}")
        if is_nested is not None:
            pieces.append(f"{key}_is_nested={is_nested}")
    return ",".join(pieces)


def patch_verl_forward_step(original_forward_step: Any) -> Any:
    """创建 MegatronEngineWithLMHead.forward_step 的 patch wrapper。"""

    def patched_forward_step(
        self,
        batch_iter,
        model,
        logits_processor_func,
        postprocess_micro_batch_func,
    ):
        # ── 获取原始 micro-batch ──
        # batch_iter 来自外层 engine 的 forward_step 调用方。
        # 消费 batch，由 build_prefix_sharing_micro_batch_verl080 进行物理裁剪
        # 返回 trimmed_batch（物理裁剪后的 micro-batch）和 ps_state。
        _ps_forward_step_probe("enter")
        _ps_forward_step_probe("before_next_batch")
        original_batch = next(batch_iter)
        _ps_forward_step_probe("after_next_batch", batch=_describe_batch(original_batch))
        batch_for_forward = original_batch

        # ── 读取配置 ──
        _ps_forward_step_probe("before_read_config")
        from prefix_sharing.integrations.verl_mcore import read_ps_config_from_engine_config
        from prefix_sharing.core.config import PrefixSharingConfig
        ps_config_raw = read_ps_config_from_engine_config(self.engine_config)
        ps_config = PrefixSharingConfig.from_raw(ps_config_raw)
        _ps_forward_step_probe(
            "after_read_config",
            enable_prefix_sharing=ps_config.enable_prefix_sharing,
            backend=ps_config.backend,
        )

        ps_state = None
        if ps_config.enable_prefix_sharing:
            # batch.to(device) 使 tensor 在目标设备上，
            # 原始 forward_step 会再次 batch.to(device)（幂等）
            from verl.utils.megatron_utils import get_device_id
            device_id = get_device_id()
            _ps_forward_step_probe("before_batch_to_device", device_id=device_id)
            batch_on_device = original_batch.to(device_id)
            _ps_forward_step_probe("after_batch_to_device")

            # batch裁剪
            _ps_forward_step_probe("before_prepare_micro_batch")
            from prefix_sharing.integrations.verl_mcore import build_prefix_sharing_micro_batch_verl080
            batch_for_forward, ps_state = build_prefix_sharing_micro_batch_verl080(
                self, batch_on_device, ps_config,
            )
            _ps_forward_step_probe(
                "after_prepare_micro_batch",
                has_runtime_state=ps_state is not None,
                batch=_describe_batch(batch_for_forward),
                layout=(
                    None
                    if ps_state is None
                    else (
                        f"valid={ps_state.packed_batch_layout.valid_lengths},"
                        f"padded={ps_state.packed_batch_layout.padded_lengths},"
                        f"cu={ps_state.packed_batch_layout.cu_seqlens}"
                    )
                ),
            )
        else:
            _ps_forward_step_probe("skip_prepare_prefix_sharing_disabled")

        # ── 构造修改后的 iterator 喂回原始 forward_step ──
        modified_iter = iter([batch_for_forward])

        # ── runtime context ──
        from prefix_sharing.integrations.context import prefix_sharing_runtime_context
        from contextlib import nullcontext

        context_manager = (
            prefix_sharing_runtime_context(ps_state)
            if ps_state is not None
            else nullcontext()
        )

        _ps_forward_step_probe("before_original_forward_step", has_runtime_state=ps_state is not None)
        with context_manager:
            output = original_forward_step(
                self,
                modified_iter,
                model,
                logits_processor_func,
                postprocess_micro_batch_func,
            )
            # v080 restore：在 context 仍激活时重组 reuser prefix 区段。
            # forward_step 返回 (output_dict, partial(postprocess_func))，
            # 解包处理 output_dict 再重包。restore_via_2d_unfold_verl080 内部
            # 会检查 context / restore_indices，无 restore 需求时 early return。
            if ps_state is not None:
                from prefix_sharing.integrations.verl_mcore import restore_via_2d_unfold_verl080
                from prefix_sharing.integrations.context import current_prefix_sharing_context
                from verl.utils.megatron.tensor_parallel import (
                    vocab_parallel_entropy,
                    vocab_parallel_log_probs_from_logits,
                )
                output_dict, postprocess_fn = output
                output_dict = restore_via_2d_unfold_verl080(
                    output_dict,
                    vocab_parallel_log_probs_from_logits,
                    vocab_parallel_entropy,
                )
                # 释放 vocab 维 logits（占用大，只在 context 生命周期内持有，
                # restore 已消费完毕）。clear 职责在此，不在包装函数内。
                ctx = current_prefix_sharing_context()
                if ctx is not None:
                    ctx.prefix_last_logits_saved.clear()
                output = (output_dict, postprocess_fn)
        _ps_forward_step_probe("after_original_forward_step")
        return output

    return patched_forward_step
