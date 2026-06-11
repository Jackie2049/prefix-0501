"""verl Megatron actor integration helpers.

This module covers both v070 and v080 (verl 0.8.0 engine) paths:

* v070: ``build_prefix_sharing_micro_batch`` and ``restore_suffix_first_log_probs_from_prefix``
  handle the invasive integration via ``megatron_actor.py``.
* v080: ``build_prefix_sharing_micro_batch_verl080`` and ``read_ps_config_from_engine_config``
  handle the monkey-patch integration via ``setup/patches/``.

Both paths share the same core logic (plan -> trim -> layout -> state).

``VerlMCoreBatchAdapter`` is framework-light and testable locally. It turns a
verl-style micro-batch payload into prefix-sharing metadata plus trimmed
inputs/labels/masks, and it assembles restored logprobs after forward.
``VerlMCoreIntegration`` installs the Megatron attention patch. The real
Megatron QKV rewiring still requires the framework runtime and remains guarded
by optional integration tests.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence, TypeVar

from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.batch_trim import (
    TrimmedBatch,
    trim_inputs,
    trim_labels,
    trim_loss_masks,
)
from prefix_sharing.core.logprob import restore_prefix_last_logprobs
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.integrations.context import prefix_sharing_runtime_context as _prefix_sharing_runtime_context
from prefix_sharing.integrations.megatron_attention import IntegrationUnavailable, MegatronAttentionIntegration
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.parallel_info import get_megatron_parallel_info
from prefix_sharing.integrations.patch_manager import PatchHandle
from prefix_sharing.integrations.utils import ensure_global_packed_token_lengths

import logging
logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass(frozen=True)
class PrefixSharingRuntimeState:
    prefix_sharing_plan: PrefixSharingPlan
    backend: Any
    packed_batch_layout: PackedBatchLayout
    parallel_info: MegatronParallelInfo
    kept_position_ids: Any | None = None


@dataclass(frozen=True)
class VerlMCorePrefixSharingBatch:
    """Framework-independent materialization of one verl actor micro-batch."""

    prefix_sharing_plan: PrefixSharingPlan
    input_ids: TrimmedBatch[int]
    labels: TrimmedBatch[Any] | None = None
    loss_masks: TrimmedBatch[Any] | None = None


@dataclass
class VerlMCoreBatchAdapter:
    """Prepare and restore verl Megatron actor micro-batches.

    The adapter is intentionally tensor-agnostic for Phase 1 local tests. A real
    verl integration can map the returned ``TrimmedBatch.flattened`` and
    ``prefix_sharing_plan.cu_seqlens_q`` fields to torch tensors without changing core
    semantics.
    """

    config: PrefixSharingConfig
    planner: PrefixSharingPlanner | None = None

    def __post_init__(self) -> None:
        if self.planner is None:
            self.planner = PrefixSharingPlanner(self.config)

    def prepare_micro_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        *,
        labels: Sequence[Sequence[T]] | None = None,
        loss_masks: Sequence[Sequence[T]] | None = None,
        forward_id: int | None = None,
        micro_batch_id: int | None = None,
    ) -> VerlMCorePrefixSharingBatch:
        """Plan prefix sharing and trim a micro-batch."""

        assert self.planner is not None
        prefix_sharing_plan = self.planner.plan(
            input_ids,
            forward_id=forward_id,
            micro_batch_id=micro_batch_id,
        )
        trimmed_inputs = trim_inputs(input_ids, prefix_sharing_plan)
        trimmed_labels = trim_labels(labels, prefix_sharing_plan) if labels is not None else None
        trimmed_loss_masks = trim_loss_masks(loss_masks, prefix_sharing_plan) if loss_masks is not None else None
        return VerlMCorePrefixSharingBatch(
            prefix_sharing_plan=prefix_sharing_plan,
            input_ids=trimmed_inputs,
            labels=trimmed_labels,
            loss_masks=trimmed_loss_masks,
        )

    def prefix_sharing_runtime_context(
        self,
        prefix_sharing_batch: VerlMCorePrefixSharingBatch,
    ) -> Iterator[Any]:
        """Open the runtime context consumed by patched attention."""

        runtime_state = PrefixSharingRuntimeState(
            prefix_sharing_plan=prefix_sharing_batch.prefix_sharing_plan,
            backend=get_backend_instance(self.config),
            packed_batch_layout=PackedBatchLayout.from_valid_lengths(
                prefix_sharing_batch.prefix_sharing_plan.kept_lengths_q
            ),
            parallel_info=get_megatron_parallel_info(),
        )
        return _prefix_sharing_runtime_context(runtime_state)

    def restore_logprobs(
        self,
        suffix_logprobs: Sequence[Sequence[float]],
        provider_prefix_last_logprobs: Sequence[float],
        prefix_sharing_plan: PrefixSharingPlan,
    ) -> list[list[float]]:
        """Assemble per-row logprobs with Prefix-Last Restore."""

        return restore_prefix_last_logprobs(
            suffix_logprobs,
            provider_prefix_last_logprobs,
            prefix_sharing_plan,
        )


@dataclass
class VerlMCoreIntegration:
    config: PrefixSharingConfig
    backend: Any | None = None
    batch_adapter: VerlMCoreBatchAdapter = field(init=False)

    def __post_init__(self) -> None:
        self.batch_adapter = VerlMCoreBatchAdapter(self.config)

    def install(self, model_config: Any | None = None) -> PatchHandle:
        self.config.validate(model_config=model_config, integrate_mode="verl_megatron_actor")
        self._ensure_verl_importable()
        backend = get_backend_instance(self.config, self.backend)
        return MegatronAttentionIntegration(config=self.config, backend=backend).install(
            model_config=model_config
        )

    @staticmethod
    def _ensure_verl_importable() -> None:
        try:
            importlib.import_module("verl")
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("verl is not importable in this environment") from exc


def enable_prefix_sharing(
    config: PrefixSharingConfig,
    *,
    model_config: Any | None = None,
    backend: Any | None = None,
) -> PatchHandle:
    """Install Phase 1 prefix-sharing patches for the verl + Megatron path."""

    return VerlMCoreIntegration(config=config, backend=backend).install(model_config=model_config)


@contextmanager
def prefix_sharing_enabled(
    config: PrefixSharingConfig,
    *,
    model_config: Any | None = None,
    backend: Any | None = None,
) -> Iterator[PatchHandle]:
    """Context manager wrapper around :func:`enable_prefix_sharing`."""

    handle = enable_prefix_sharing(config, model_config=model_config, backend=backend)
    try:
        yield handle
    finally:
        handle.disable()


def build_prefix_sharing_micro_batch(
    batch: Any,
    actor_config: Any,
    model_config: Any,
    *,
    backend: Any | None = None,
) -> tuple[Any, PrefixSharingRuntimeState | None]:
    """Trim one verl Megatron actor micro-batch in-place for prefix sharing.

    The framework-facing contract is intentionally small: dependency/verl calls
    this once after obtaining the micro-batch and then opens
    :func:`prefix_sharing_runtime_context` around its existing forward
    call. Unsupported or disabled cases return ``(batch, None)``.
    """

    batch_size = len(batch["input_ids"]) if "input_ids" in batch else None
    logger.warning(f"[PS][prepare] ENTER: batch_size={batch_size}, batch_keys={list(batch.keys())}")

    config = PrefixSharingConfig.from_raw(
        _read_actor_value(actor_config, "prefix_sharing_config", None)
    )

    # --- Path 1: prefix sharing disabled by config ---
    if not config.enable_prefix_sharing:
        logger.warning(f"[PS][prepare] PATH 1: prefix sharing disabled (config.enable_prefix_sharing=False), returning (batch, None)")
        return batch, None

    logger.warning(f"[PS][prepare] config.enable_prefix_sharing=True, validating config...")

    config.validate(model_config=model_config, integrate_mode="verl_megatron_actor")
    logger.warning(f"[PS][prepare] config.validate() returned OK")

    # --- Path 2: missing use_remove_padding ---
    logger.warning(f"[PS][prepare] checking megatron.use_remove_padding...")
    if not _read_actor_bool(actor_config, "megatron.use_remove_padding", False):
        logger.warning(f"[PS][prepare] PATH 2: megatron.use_remove_padding=False, raising RuntimeError")
        raise RuntimeError("prefix sharing phase 1 requires verl megatron.use_remove_padding=True")

    # --- Path 3: multi_modal check ---
    logger.warning(f"[PS][prepare] use_remove_padding=True, about to batch.get(multi_modal_inputs)...")
    multi_modal_inputs = batch.get("multi_modal_inputs")
    if multi_modal_inputs is not None:
        # tensorclass 无法遍历（触发 CUDA 同步），改用底层 td 检查字段数
        import inspect
        is_tensorclass = hasattr(multi_modal_inputs, 'batch_size')
        logger.warning(f"[PS][prepare] multi_modal_inputs type: tensorclass={is_tensorclass}, type={type(multi_modal_inputs).__name__}")
        if is_tensorclass:
            _td = getattr(multi_modal_inputs, 'td', None) or getattr(multi_modal_inputs, '_tensordict', None)
            _keys = list(_td.keys()) if _td is not None else []
            has_mm = len(_keys) > 0
            logger.warning(f"[PS][prepare] tensorclass td keys={_keys}, has_mm={has_mm}")
        else:
            has_mm = any(mmi is not None and len(mmi.keys()) > 0 for mmi in multi_modal_inputs)
        if has_mm:
            logger.warning(f"[PS][prepare] PATH 3: multi_modal_inputs has content, raising RuntimeError")
            raise RuntimeError("prefix sharing phase 1 supports only text-only actor micro-batches")
        logger.warning(f"[PS][prepare] multi_modal check PASSED (no real multi-modal content)")

    # --- Read tensors ---
    attention_mask = batch["attention_mask"].to(bool)
    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]
    logger.warning(f"[PS][prepare] tensor shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, position_ids={position_ids.shape}")

    # --- Path 4: wrong tensor dims ---
    if attention_mask.dim() != 2 or input_ids.dim() != 2 or position_ids.dim() != 2:
        logger.warning(f"[PS][prepare] PATH 4: non-2D tensors detected, raising RuntimeError")
        raise RuntimeError("prefix sharing phase 1 expects 2D input_ids/attention_mask/position_ids")

    # --- Planning ---
    valid_indices = [attention_mask[row].nonzero(as_tuple=False).flatten() for row in range(input_ids.shape[0])]
    sequences = [input_ids[row, indices].detach().cpu().tolist() for row, indices in enumerate(valid_indices)]
    seq_lens = [len(s) for s in sequences]
    logger.warning(f"[PS][prepare] sequences: num_seq={len(sequences)}, seq_lens={seq_lens}")

    prefix_sharing_plan = PrefixSharingPlanner(config).plan(sequences)
    logger.warning(
        f"[PS][prepare] prefix_sharing_plan result: has_sharing={prefix_sharing_plan.has_sharing}, "
        f"keep_ranges={prefix_sharing_plan.input_keep_ranges}, "
        f"prefix_last_restore={prefix_sharing_plan.prefix_last_restore}"
    )

    # --- Path 5: no sharing found ---
    if not prefix_sharing_plan.has_sharing:
        logger.warning(f"[PS][prepare] PATH 5: no sharing detected, returning (batch, None)")
        return batch, None

    # --- Path 6: sharing found, trim the original micro-batch ---
    logger.warning(f"[PS][prepare] PATH 6: sharing detected, preparing trimmed batch...")
    trimmed_micro_batch = _clone_batch(batch)
    new_attention_mask = attention_mask.clone()
    new_attention_mask[:] = False
    new_input_ids = input_ids.clone()
    new_position_ids = position_ids.clone()
    kept_position_rows = []

    for row, indices in enumerate(valid_indices):
        keep_start, keep_end = prefix_sharing_plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        new_attention_mask[row, kept_indices] = True
        kept_position_rows.append(position_ids[row, kept_indices])

    trimmed_micro_batch["input_ids"] = new_input_ids
    trimmed_micro_batch["attention_mask"] = new_attention_mask
    trimmed_micro_batch["position_ids"] = new_position_ids

    parallel_info = get_megatron_parallel_info()
    align_size = (
        parallel_info.tp_size * parallel_info.cp_size * 2
        if parallel_info.cp_size > 1
        else parallel_info.tp_size
    )
    packed_batch_layout = PackedBatchLayout.from_kept_position_rows(
        kept_position_rows,
        align_size=int(align_size),
    )
    logger.warning(
        "[PS][prepare][global_rank=%s tp_rank=%s/tp_size=%s cp_rank=%s/cp_size=%s "
        "pp_rank=%s/pp_size=%s is_pp_first=%s is_pp_last=%s] packed_batch_layout: "
        "valid_lengths=%s, padded_lengths=%s, cu_seqlens=%s, max_seqlen=%s, "
        "total_valid=%s, total_padded=%s",
        parallel_info.global_rank,
        parallel_info.tp_rank,
        parallel_info.tp_size,
        parallel_info.cp_rank,
        parallel_info.cp_size,
        parallel_info.pp_rank,
        parallel_info.pp_size,
        parallel_info.is_pipeline_first_stage,
        parallel_info.is_pipeline_last_stage,
        packed_batch_layout.valid_lengths,
        packed_batch_layout.padded_lengths,
        packed_batch_layout.cu_seqlens,
        packed_batch_layout.max_seqlen,
        packed_batch_layout.total_valid_length,
        packed_batch_layout.total_padded_length,
    )
    prefix_sharing_runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        backend=get_backend_instance(config, backend),
        packed_batch_layout=packed_batch_layout,
        parallel_info=parallel_info,
    )
    logger.warning(
        "[PS][prepare] PATH 6 DONE: returning (trimmed_micro_batch, "
        f"prefix_sharing_runtime_state) with keep_ranges={prefix_sharing_plan.input_keep_ranges}"
    )
    return trimmed_micro_batch, prefix_sharing_runtime_state


def restore_suffix_first_log_probs_from_prefix(
    logits: Any,
    labels: Any,
    log_probs: Any,
    vocab_parallel_log_probs_fn: Any,
) -> Any:
    """Restore reuser suffix-first logprob from provider prefix-last logits."""

    ctx = current_prefix_sharing_context()
    if ctx is None or not ctx.prefix_last_restore_indices:
        return log_probs
    parallel_info = ctx.parallel_info
    if not parallel_info.is_pipeline_last_stage:
        logger.warning(
            "[PS][restore][global_rank=%s pp_rank=%s/pp_size=%s is_pp_last=%s] "
            "skip prefix-last restore on non-last PP stage: restore_indices=%s "
            "logits_token_length=%s log_probs_token_length=%s total_padded_length=%s",
            parallel_info.global_rank,
            parallel_info.pp_rank,
            parallel_info.pp_size,
            parallel_info.is_pipeline_last_stage,
            len(ctx.prefix_last_restore_indices),
            logits.shape[1],
            log_probs.shape[1],
            ctx.packed_batch_layout.total_padded_length,
        )
        return log_probs
    ensure_global_packed_token_lengths(
        {
            "logits_token_length": logits.shape[1],
            "log_probs_token_length": log_probs.shape[1],
        },
        total_padded_length=ctx.packed_batch_layout.total_padded_length,
        context="prefix-last restore",
    )
    logger.warning(
        "[PS][restore][global_rank=%s pp_rank=%s/pp_size=%s is_pp_last=%s] "
        "running prefix-last restore: restore_indices=%s "
        "logits_token_length=%s log_probs_token_length=%s total_padded_length=%s",
        parallel_info.global_rank,
        parallel_info.pp_rank,
        parallel_info.pp_size,
        parallel_info.is_pipeline_last_stage,
        len(ctx.prefix_last_restore_indices),
        logits.shape[1],
        log_probs.shape[1],
        ctx.packed_batch_layout.total_padded_length,
    )
    restored = log_probs.clone()
    for index in ctx.prefix_last_restore_indices:
        provider_logits = logits[
            0:1,
            index.provider_1d_pos : index.provider_1d_pos + 1,
            :,
        ].clone()
        reuse_label = labels[
            0:1,
            index.reuse_1d_pos : index.reuse_1d_pos + 1,
        ]
        restored_value = vocab_parallel_log_probs_fn(provider_logits, reuse_label)
        restored[0, index.reuse_1d_pos] = restored_value.reshape(())
    return restored


def _clone_batch(batch: Any) -> Any:
    if hasattr(batch, "clone"):
        return batch.clone()
    if hasattr(batch, "copy"):
        return batch.copy()
    return dict(batch)


def _read_actor_bool(config: Any, dotted_name: str, default: bool) -> bool:
    value = _read_actor_value(config, dotted_name, default)
    return bool(value)


def _read_actor_value(config: Any, dotted_name: str, default: Any) -> Any:
    current = config
    for part in dotted_name.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(part, default)
        else:
            getter = getattr(current, "get", None)
            if callable(getter):
                current = getter(part, default)
            else:
                current = getattr(current, part, default)
    return current


# ═══════════════════════════════════════════════════════════════
# verl 0.8.0 engine 架构适配
# ═══════════════════════════════════════════════════════════════


def read_ps_config_from_engine_config(engine_config: Any) -> Any | None:
    """从 verl080 engine_config 读取 prefix_sharing_config。

    verl080 engine_config 中 prefix_sharing_config 的位置：
    1. engine_config.override_transformer_config 是 dict -> 从 dict 中取
    2. engine_config.override_transformer_config 是对象 -> 从对象属性取
    3. 回退 -> engine_config.prefix_sharing_config（直接挂在 engine_config 上）
    """
    override = getattr(engine_config, "override_transformer_config", None)
    if override is not None:
        if isinstance(override, dict):
            return override.get("prefix_sharing_config")
        return getattr(override, "prefix_sharing_config", None)
    return getattr(engine_config, "prefix_sharing_config", None)


def build_prefix_sharing_micro_batch_verl080(
    engine_self: Any,
    batch: Any,
    ps_config: PrefixSharingConfig,
) -> tuple[Any, PrefixSharingRuntimeState | None]:
    """verl 0.8.0 engine 架构下的 prefix-sharing micro-batch 构建。

    与 v070 的 build_prefix_sharing_micro_batch 共用核心流程（PATH 1-6），
    差异仅在 config 来源和 batch 格式适配。

    参数 ps_config 已由调用方通过 PrefixSharingConfig.from_raw() 解析完成，
    不需要再次 from_raw。

    核心原则：以 v070 验证过的 2D + attention_mask 路径为主，
    NestedTensor 路径仅在 GPU + use_remove_padding=True 时作为可选优化。
    NPU 不支持 torch.nested，所有 NPU 场景都走 2D 路径。
    """
    # ── PATH 1: prefix sharing disabled ──
    if not ps_config.enable_prefix_sharing:
        logger.info("[PS][prepare] PATH 1: prefix sharing disabled")
        return batch, None

    # ── 阶段 1: 配置校验 ──
    use_remove_padding = getattr(engine_self.engine_config, "use_remove_padding", True)
    ps_config.validate_for_engine(use_remove_padding=use_remove_padding)

    # ── 阶段 2: fused kernels guard ──
    try:
        from verl.utils import tensordict_utils as tu
        use_fused = tu.get_non_tensor_data(batch, "use_fused_kernels", default=False)
    except Exception:
        use_fused = False
    if use_fused:
        raise RuntimeError("prefix sharing phase 1 requires fused kernels disabled")

    # ── 阶段 2.5: dynamic CP guard ──
    if getattr(engine_self.engine_config, "dynamic_context_parallel", False):
        raise RuntimeError("prefix sharing phase 1 does not support dynamic context parallel")

    # ── 阶段 3: 从 batch 提取 2D tensors ──
    # NestedTensor -> 转换为 2D + attention_mask -> 复用 v070 逻辑
    # Plain 2D tensor -> 直接使用（与 v070 完全一致）
    attention_mask, input_ids, position_ids = _extract_2d_tensors_from_batch(batch)

    # ── PATH 4: wrong tensor dims ──
    if attention_mask.dim() != 2 or input_ids.dim() != 2 or position_ids.dim() != 2:
        logger.info("[PS][prepare] PATH 4: non-2D tensors detected")
        return batch, None

    # ── 阶段 4: 提取序列（与 v070 完全一致）──
    attention_mask_bool = attention_mask.to(bool)
    valid_indices = [
        attention_mask_bool[row].nonzero(as_tuple=False).flatten()
        for row in range(input_ids.shape[0])
    ]
    sequences = [
        input_ids[row, indices].detach().cpu().tolist()
        for row, indices in enumerate(valid_indices)
    ]

    # ── 阶段 5: 前缀共享规划（与 v070 完全一致）──
    plan = PrefixSharingPlanner(ps_config).plan(sequences)

    # ── PATH 5: no sharing found ──
    if not plan.has_sharing:
        logger.info("[PS][prepare] PATH 5: no sharing detected")
        return batch, None

    # ── 阶段 6: trim + layout + state（与 v070 PATH 6 完全一致）──
    trimmed_batch = _clone_batch(batch)
    new_attention_mask = attention_mask_bool.clone()
    new_attention_mask[:] = False
    kept_position_rows = []

    for row, indices in enumerate(valid_indices):
        keep_start, keep_end = plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        new_attention_mask[row, kept_indices] = True
        kept_position_rows.append(position_ids[row, kept_indices])

    trimmed_batch["attention_mask"] = new_attention_mask

    parallel_info = get_megatron_parallel_info()
    align_size = (
        parallel_info.tp_size * parallel_info.cp_size * 2
        if parallel_info.cp_size > 1
        else parallel_info.tp_size
    )
    packed_layout = PackedBatchLayout.from_kept_position_rows(
        kept_position_rows,
        align_size=int(align_size),
    )

    state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        backend=get_backend_instance(ps_config),
        packed_batch_layout=packed_layout,
        parallel_info=parallel_info,
    )

    logger.info(
        "[PS][prepare] PATH 6: sharing detected, plan=%s, layout=%s",
        plan,
        packed_layout,
    )
    return trimmed_batch, state


# ═══════════════════════════════════════
# Batch 格式适配（verl080 TensorDict/NestedTensor）
# ═══════════════════════════════════════

def _extract_2d_tensors_from_batch(
    batch: Any,
) -> tuple[Any, Any, Any]:
    """从 verl batch 中提取 attention_mask, input_ids, position_ids 为 2D tensors。

    - Plain 2D tensor (v070 / NPU): 直接返回
    - NestedTensor (GPU THD): 转换为 2D + attention_mask (padding 无效位置)

    返回 (attention_mask, input_ids, position_ids)，均为 2D tensor。
    """
    import torch

    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]

    # NestedTensor (jagged layout) -> 转换为 2D + attention_mask
    if _is_nested_tensor(input_ids):
        input_ids_2d, attention_mask_2d = _nested_to_2d(input_ids)
        position_ids_2d = (
            _nested_to_2d_position(position_ids)
            if _is_nested_tensor(position_ids)
            else position_ids
        )
        return attention_mask_2d, input_ids_2d, position_ids_2d

    # Plain 2D tensor (v070 path)
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        # THD format without explicit attention_mask -> all positions valid
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=input_ids.device
        )
    return attention_mask, input_ids, position_ids


def _is_nested_tensor(tensor: Any) -> bool:
    """安全检测 NestedTensor，避免在 NPU 上引用 torch.nested 模块。

    NPU 不支持 torch.nested，直接 isinstance(tensor, torch.nested.NestedTensor)
    会崩溃。使用 duck-typing: 有 offsets() 和 values() 方法即为 NestedTensor。
    """
    return (
        hasattr(tensor, "offsets")
        and callable(tensor.offsets)
        and hasattr(tensor, "values")
        and callable(tensor.values)
    )


def _nested_to_2d(nested_tensor: Any) -> tuple[Any, Any]:
    """将 NestedTensor (jagged layout) 转换为 2D tensor + attention_mask。

    Padding 短序列到 max_seqlen，attention_mask 标记有效位置。
    """
    import torch

    offsets = nested_tensor.offsets()
    values = nested_tensor.values()
    lengths = offsets.diff().tolist()
    max_seqlen = int(max(lengths))
    batch_size = len(lengths)

    padded = torch.zeros(
        batch_size, max_seqlen,
        dtype=values.dtype, device=values.device,
    )
    mask = torch.zeros(
        batch_size, max_seqlen,
        dtype=torch.bool, device=values.device,
    )
    for i in range(batch_size):
        seq_len = int(lengths[i])
        padded[i, :seq_len] = values[offsets[i]:offsets[i + 1]]
        mask[i, :seq_len] = True

    return padded, mask


def _nested_to_2d_position(nested_tensor: Any) -> Any:
    """将 position_ids NestedTensor 转换为 2D tensor（padding 用 0）。"""
    import torch

    offsets = nested_tensor.offsets()
    values = nested_tensor.values()
    lengths = offsets.diff().tolist()
    max_seqlen = int(max(lengths))
    batch_size = len(lengths)

    padded = torch.zeros(
        batch_size, max_seqlen,
        dtype=values.dtype, device=values.device,
    )
    for i in range(batch_size):
        seq_len = int(lengths[i])
        padded[i, :seq_len] = values[offsets[i]:offsets[i + 1]]

    return padded
