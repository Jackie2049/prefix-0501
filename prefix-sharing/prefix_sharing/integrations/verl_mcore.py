"""verl Megatron actor integration helpers.

This module covers both v070 and v080 (verl 0.8.0 engine) paths:

* v070: ``build_prefix_sharing_micro_batch_verl070`` and ``restore_suffix_first_log_probs_from_prefix``
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
from prefix_sharing.backends.batch_layout import BshdBatchLayout
from prefix_sharing.backends.batch_layout import BshdTokenIndex
from prefix_sharing.backends.batch_layout import ThdBatchLayout
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
from prefix_sharing.utils import ensure_global_packed_token_lengths, pad_to_multiple

import logging
logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass(frozen=True)
class PrefixSharingRuntimeState:
    prefix_sharing_plan: PrefixSharingPlan
    attention_backend: Any
    batch_runtime_layout: Any
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
            attention_backend=get_backend_instance(self.config),
            batch_runtime_layout=ThdBatchLayout.construct_from_valid_lengths(
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


def build_prefix_sharing_micro_batch_verl070(
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
    batch_runtime_layout = ThdBatchLayout.construct_from_kept_position_ids(
        kept_position_rows,
        align_size=int(align_size),
    )
    logger.warning(
        "[PS][prepare][global_rank=%s tp_rank=%s/tp_size=%s cp_rank=%s/cp_size=%s "
        "pp_rank=%s/pp_size=%s is_pp_first=%s is_pp_last=%s] thd_batch_layout: "
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
        batch_runtime_layout.valid_lengths,
        batch_runtime_layout.padded_lengths,
        batch_runtime_layout.cu_seqlens,
        batch_runtime_layout.max_seqlen,
        batch_runtime_layout.total_valid_length,
        batch_runtime_layout.total_padded_length,
    )
    prefix_sharing_runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        attention_backend=get_backend_instance(config, backend),
        batch_runtime_layout=batch_runtime_layout,
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
            "skip prefix-last restore on non-last PP stage: restore_indices=%s",
            parallel_info.global_rank,
            parallel_info.pp_rank,
            parallel_info.pp_size,
            parallel_info.is_pipeline_last_stage,
            len(ctx.prefix_last_restore_indices),
        )
        return log_probs
    logger.warning(
        "[PS][restore][global_rank=%s pp_rank=%s/pp_size=%s is_pp_last=%s] "
        "running prefix-last restore: restore_indices=%s",
        parallel_info.global_rank,
        parallel_info.pp_rank,
        parallel_info.pp_size,
        parallel_info.is_pipeline_last_stage,
        len(ctx.prefix_last_restore_indices),
    )
    restored = log_probs.clone()
    layout = ctx.batch_runtime_layout
    if layout.layout_kind == "thd":
        ensure_global_packed_token_lengths(
            {
                "logits_token_length": logits.shape[1],
                "log_probs_token_length": log_probs.shape[1],
            },
            total_padded_length=layout.total_padded_length,
            context="prefix-last restore",
        )
    for index in ctx.prefix_last_restore_indices:
        if layout.layout_kind == "thd":
            provider_pos = int(index.provider_token_index)
            reuse_pos = int(index.reuse_token_index)
            provider_logits = logits[0:1, provider_pos : provider_pos + 1, :].clone()
            reuse_label = labels[0:1, reuse_pos : reuse_pos + 1]
            restored_value = vocab_parallel_log_probs_fn(provider_logits, reuse_label)
            restored[0, reuse_pos] = restored_value.reshape(())
            continue
        if layout.layout_kind == "bshd":
            provider_pos = index.provider_token_index
            reuse_pos = index.reuse_token_index
            if not isinstance(provider_pos, BshdTokenIndex) or not isinstance(reuse_pos, BshdTokenIndex):
                raise TypeError("BSHD prefix-last restore requires BshdTokenIndex values")
            provider_logits = _take_bshd_restore_token(logits, layout, provider_pos, keep_vocab_dim=True).clone()
            reuse_label = _take_bshd_restore_token(labels, layout, reuse_pos, keep_vocab_dim=False)
            _ensure_non_empty_bshd_restore_token(
                provider_logits=provider_logits,
                reuse_label=reuse_label,
                logits=logits,
                labels=labels,
                log_probs=log_probs,
                layout=layout,
                provider_pos=provider_pos,
                reuse_pos=reuse_pos,
            )
            restored_value = vocab_parallel_log_probs_fn(provider_logits, reuse_label)
            _write_bshd_restore_token(restored, layout, reuse_pos, restored_value.reshape(()))
            continue
        raise ValueError(f"unsupported batch runtime layout: {layout.layout_kind}")
    return restored


def _take_bshd_restore_token(tensor: Any, layout: BshdBatchLayout, token_index: BshdTokenIndex, *, keep_vocab_dim: bool) -> Any:
    valid_offset = _bshd_valid_offset(layout, token_index)
    if _is_compact_bshd_tensor(tensor, layout):
        compact_pos = sum(layout.valid_lengths[: token_index.seq_idx_in_batch]) + valid_offset
        if keep_vocab_dim:
            return tensor[compact_pos : compact_pos + 1].unsqueeze(0)
        return tensor[compact_pos : compact_pos + 1].unsqueeze(0)
    if _is_padded_sbh_tensor(tensor, layout):
        if keep_vocab_dim:
            return tensor[valid_offset : valid_offset + 1, token_index.seq_idx_in_batch : token_index.seq_idx_in_batch + 1, :]
        return tensor[valid_offset : valid_offset + 1, token_index.seq_idx_in_batch : token_index.seq_idx_in_batch + 1]
    if _is_kept_padded_bsh_tensor(tensor, layout):
        if keep_vocab_dim:
            return tensor[token_index.seq_idx_in_batch : token_index.seq_idx_in_batch + 1, valid_offset : valid_offset + 1, :]
        return tensor[token_index.seq_idx_in_batch : token_index.seq_idx_in_batch + 1, valid_offset : valid_offset + 1]
    if _is_flattened_kept_padded_bshd_tensor(tensor, layout):
        flat_pos = valid_offset * layout.batch_size + token_index.seq_idx_in_batch
        return tensor[:, flat_pos : flat_pos + 1, :] if keep_vocab_dim else tensor[:, flat_pos : flat_pos + 1]
    if _is_flattened_full_bshd_tensor(tensor, layout):
        flat_pos = token_index.seq_idx_in_batch * layout.max_seqlen + token_index.token_idx_in_seq
        return tensor[:, flat_pos : flat_pos + 1, :] if keep_vocab_dim else tensor[:, flat_pos : flat_pos + 1]
    if keep_vocab_dim:
        return tensor[token_index.seq_idx_in_batch : token_index.seq_idx_in_batch + 1, token_index.token_idx_in_seq : token_index.token_idx_in_seq + 1, :]
    return tensor[token_index.seq_idx_in_batch : token_index.seq_idx_in_batch + 1, token_index.token_idx_in_seq : token_index.token_idx_in_seq + 1]


def _write_bshd_restore_token(tensor: Any, layout: BshdBatchLayout, token_index: BshdTokenIndex, value: Any) -> None:
    valid_offset = _bshd_valid_offset(layout, token_index)
    if _is_compact_bshd_tensor(tensor, layout):
        compact_pos = sum(layout.valid_lengths[: token_index.seq_idx_in_batch]) + valid_offset
        tensor[compact_pos] = value
        return
    if _is_padded_sbh_tensor(tensor, layout):
        tensor[valid_offset, token_index.seq_idx_in_batch] = value
        return
    if _is_kept_padded_bsh_tensor(tensor, layout):
        tensor[token_index.seq_idx_in_batch, valid_offset] = value
        return
    if _is_flattened_kept_padded_bshd_tensor(tensor, layout):
        flat_pos = valid_offset * layout.batch_size + token_index.seq_idx_in_batch
        tensor[:, flat_pos] = value
        return
    if _is_flattened_full_bshd_tensor(tensor, layout):
        flat_pos = token_index.seq_idx_in_batch * layout.max_seqlen + token_index.token_idx_in_seq
        tensor[:, flat_pos] = value
        return
    tensor[token_index.seq_idx_in_batch, token_index.token_idx_in_seq] = value


def _ensure_non_empty_bshd_restore_token(
    *,
    provider_logits: Any,
    reuse_label: Any,
    logits: Any,
    labels: Any,
    log_probs: Any,
    layout: BshdBatchLayout,
    provider_pos: BshdTokenIndex,
    reuse_pos: BshdTokenIndex,
) -> None:
    if provider_logits.numel() > 0 and reuse_label.numel() > 0:
        return
    raise RuntimeError(
        "prefix sharing BSHD restore selected an empty token: "
        f"logits_shape={tuple(logits.shape)}, labels_shape={tuple(labels.shape)}, "
        f"log_probs_shape={tuple(log_probs.shape)}, provider_pos={provider_pos}, "
        f"reuse_pos={reuse_pos}, provider_valid_offset={_bshd_valid_offset(layout, provider_pos)}, "
        f"reuse_valid_offset={_bshd_valid_offset(layout, reuse_pos)}, "
        f"valid_lengths={layout.valid_lengths}, max_seqlen={layout.max_seqlen}"
    )


def _bshd_valid_offset(layout: BshdBatchLayout, token_index: BshdTokenIndex) -> int:
    row_mask = layout.valid_token_mask[token_index.seq_idx_in_batch]
    preceding = row_mask[: token_index.token_idx_in_seq].sum()
    if not bool(row_mask[token_index.token_idx_in_seq]):
        raise IndexError("BshdTokenIndex points to a non-valid token")
    return int(preceding.detach().cpu().item())


def _is_compact_bshd_tensor(tensor: Any, layout: BshdBatchLayout) -> bool:
    return tensor.dim() >= 1 and int(tensor.shape[0]) == layout.total_valid_length


def _is_padded_sbh_tensor(tensor: Any, layout: BshdBatchLayout) -> bool:
    max_valid_length = max(layout.valid_lengths, default=0)
    return (
        tensor.dim() >= 2
        and int(tensor.shape[0]) >= max_valid_length
        and int(tensor.shape[1]) == layout.batch_size
    )


def _is_kept_padded_bsh_tensor(tensor: Any, layout: BshdBatchLayout) -> bool:
    max_valid_length = max(layout.valid_lengths, default=0)
    return (
        tensor.dim() >= 2
        and int(tensor.shape[0]) == layout.batch_size
        and int(tensor.shape[1]) >= max_valid_length
        and max_valid_length != layout.max_seqlen
        and int(tensor.shape[1]) != layout.max_seqlen
    )


def _is_flattened_kept_padded_bshd_tensor(tensor: Any, layout: BshdBatchLayout) -> bool:
    return (
        tensor.dim() >= 2
        and int(tensor.shape[0]) == 1
        and int(tensor.shape[1]) == max(layout.valid_lengths, default=0) * layout.batch_size
    )


def _is_flattened_full_bshd_tensor(tensor: Any, layout: BshdBatchLayout) -> bool:
    return tensor.dim() >= 2 and int(tensor.shape[0]) == 1 and int(tensor.shape[1]) == layout.batch_size * layout.max_seqlen


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

    与 v070 的 build_prefix_sharing_micro_batch_verl070 共用核心流程（PATH 1-6），
    差异仅在 config 来源和 batch 格式适配。

    参数 ps_config 已由调用方通过 PrefixSharingConfig.from_raw() 解析完成，
    不需要再次 from_raw。

    核心原则：以 v070 验证过的 2D + attention_mask 路径为主。
    v080 BSHD 原生路径也可能接收 NestedTensor；此时先物理裁剪
    NestedTensor，再由 verl 的 preprocess_bshd_engine pad 成 dense BSHD。
    """
    # ── PATH 1: prefix sharing disabled ──
    if not ps_config.enable_prefix_sharing:
        logger.info("[PS][prepare] PATH 1: prefix sharing disabled")
        return batch, None

    # ── 阶段 1: 配置校验 ──
    use_remove_padding = getattr(engine_self.engine_config, "use_remove_padding", True)
    ps_config.validate_for_engine(use_remove_padding=use_remove_padding)

    # ── 阶段 2: 拒绝不支持的特性 ──
    try:
        from verl.utils import tensordict_utils as tu
        use_fused = tu.get_non_tensor_data(batch, "use_fused_kernels", default=False)
    except Exception:
        use_fused = False
    if use_fused:
        raise RuntimeError("prefix sharing phase 1 requires fused kernels disabled")
    if getattr(engine_self.engine_config, "dynamic_context_parallel", False):
        raise RuntimeError("prefix sharing phase 1 does not support dynamic context parallel")

    # ── 阶段 3: 从 batch 提取序列 ──
    # NestedTensor → 从 offsets/values 提取
    # Plain 2D → 从 attention_mask.nonzero() 提取
    # 同时保留 attention_mask_bool，供阶段 6 的 _collect_kept_position_rows 使用。
    input_ids = batch["input_ids"]
    is_nested_tensor = _is_nested_tensor(input_ids)
    attention_mask_bool_for_layout = None

    if is_nested_tensor:
        sequences = _extract_seq_from_nested_tensor(input_ids)
    else:
        # plain 2D tensor（需要 attention_mask）
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            logger.info("[PS][prepare] PATH 4: plain 2D batch without attention_mask")
            return batch, None
        attention_mask_bool = attention_mask.to(bool)
        attention_mask_bool_for_layout = attention_mask_bool  # 供阶段 6 使用
        valid_indices = [
            attention_mask_bool[row].nonzero(as_tuple=False).flatten()
            for row in range(input_ids.shape[0])
        ]
        sequences = [
            input_ids[row, indices].detach().cpu().tolist()
            for row, indices in enumerate(valid_indices)
        ]

    # ── 阶段 4: 前缀共享规划 ──
    plan = PrefixSharingPlanner(ps_config).plan(sequences)
    if not plan.has_sharing:
        logger.info("[PS][prepare] no prefix sharing detected")
        return batch, None

    # ── 阶段 5: 根据 verl layout 裁剪 batch ──
    # use_remove_padding=True: THD preprocess reads input_ids directly, so the
    # removed prefix tokens must be physically absent.
    # use_remove_padding=False: BSHD keeps dense tensors and uses attention_mask
    # to expose only kept tokens.
    if use_remove_padding:
        if is_nested_tensor:
            trimmed_batch = _trim_nested_batch(batch, plan)
        else:
            trimmed_batch = _trim_plain_batch_thd(batch, plan)
    else:
        if is_nested_tensor:
            trimmed_batch = _trim_nested_batch(batch, plan)
        else:
            trimmed_batch = _trim_plain_batch_bshd(batch, plan, attention_mask_bool_for_layout)

    # ── 阶段 6: 构建 layout ──
    parallel_info = get_megatron_parallel_info()
    if use_remove_padding:
        kept_position_rows = _collect_kept_position_rows(
            trimmed_batch, plan, is_nested_tensor,
            attention_mask_bool=attention_mask_bool_for_layout,
        )
        align_size = (
            parallel_info.tp_size * parallel_info.cp_size * 2
            if parallel_info.cp_size > 1
            else parallel_info.tp_size
        )
        batch_runtime_layout = ThdBatchLayout.construct_from_kept_position_ids(
            kept_position_rows,
            align_size=int(align_size),
        )
    else:
        if is_nested_tensor:
            batch_runtime_layout = _build_bshd_layout_from_nested_position_ids(
                trimmed_batch["position_ids"],
                parallel_info,
            )
        else:
            batch_runtime_layout = BshdBatchLayout.from_valid_token_mask(
                trimmed_batch["attention_mask"].to(bool),
                position_ids=trimmed_batch["position_ids"],
            )

    # ── 阶段 7: 构建 state ──
    state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=get_backend_instance(ps_config),
        batch_runtime_layout=batch_runtime_layout,
        parallel_info=parallel_info,
    )

    logger.info(
        f"[PS][prepare] PATH 6: sharing detected, plan={plan}, layout={batch_runtime_layout}"
    )

    return trimmed_batch, state


# ═══════════════════════════════════════
# Batch 物理裁剪（verl080 THD 路径必须改数据本身，不能只改 mask）
# ═══════════════════════════════════════

def _trim_nested_batch(batch: Any, plan: PrefixSharingPlan) -> Any:
    """物理裁剪 NestedTensor batch（verl080 GPU THD 路径）。

    preprocess_thd_engine 从 NestedTensor offsets/values 直接创建 packed 数据，
    不看 attention_mask。因此必须物理裁剪 input_ids/position_ids，
    去掉 reuser 的 prefix tokens，只保留 provider + reuser 的 kept 区段。
    """
    import torch

    trimmed_batch = _clone_batch(batch)

    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]

    # 裁剪 input_ids NestedTensor
    trimmed_ids_seqs = _slice_nested_sequences(input_ids, plan)
    new_input_ids = torch.nested.nested_tensor(trimmed_ids_seqs, layout=torch.jagged)
    trimmed_batch["input_ids"] = new_input_ids

    # 裁剪 position_ids NestedTensor
    if _is_nested_tensor(position_ids):
        trimmed_pos_seqs = _slice_nested_sequences(position_ids, plan)
        new_position_ids = torch.nested.nested_tensor(trimmed_pos_seqs, layout=torch.jagged)
    else:
        # position_ids 是 2D tensor → 需要用 attention_mask 的
        # valid_indices 切片（keep_range 是序列偏移，不是列索引）
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask_bool = attention_mask.to(bool)
        else:
            import torch
            # NestedTensor batch 无 explicit attention_mask → 所有位置都 valid
            # 这意味着 position_ids 每行的有效位置从列 0 开始，
            # valid_indices 等于 range(seq_len)，keep_range 可以直接当列索引用。
            # 但仍然走 nonzero 路径保持一致性。
            attention_mask_bool = torch.ones(
                position_ids.shape[0], position_ids.shape[1],
                dtype=torch.bool, device=position_ids.device,
            )
        trimmed_pos_seqs = _slice_2d_position_rows(
            position_ids, plan, attention_mask_bool,
        )
        new_position_ids = torch.nested.nested_tensor(trimmed_pos_seqs, layout=torch.jagged)
    trimmed_batch["position_ids"] = new_position_ids

    # loss_mask 也需要裁剪（如果存在）
    loss_mask = batch.get("loss_mask")
    if loss_mask is not None and _is_nested_tensor(loss_mask):
        trimmed_loss_seqs = _slice_nested_sequences(loss_mask, plan)
        trimmed_batch["loss_mask"] = torch.nested.nested_tensor(
            trimmed_loss_seqs, layout=torch.jagged
        )

    return trimmed_batch


def _trim_plain_batch_thd(batch: Any, plan: PrefixSharingPlan) -> Any:
    """物理裁剪 plain 2D tensor batch（verl080 NPU THD 路径）。

    v080 THD 路径即使 input_ids 是 2D tensor，preprocess_thd_engine
    也直接从 input_ids 数据创建 packed（不看 attention_mask）。
    因此 2D 路径同样需要物理裁剪 input_ids/position_ids，
    去掉 reuser 的 prefix tokens。

    裁剪方式：将被移除的 prefix 位置用 padding 填充（0 值），
    attention_mask 标记为 False，这样 Megatron 处理时只看 True 的位置。
    同时将裁剪后的数据转为 NestedTensor 格式，确保
    preprocess_thd_engine 正确处理。
    """
    import torch

    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]
    attention_mask = batch.get("attention_mask")

    if attention_mask is not None:
        attention_mask_bool = attention_mask.to(bool)
    else:
        # 无 explicit attention_mask → 所有位置都 valid
        attention_mask_bool = torch.ones(
            input_ids.shape[0], input_ids.shape[1],
            dtype=torch.bool, device=input_ids.device,
        )

    # 按 keep_ranges 从每个 row 中提取 kept 区段
    kept_id_rows = []
    kept_pos_rows = []
    kept_mask_rows = []

    for row in range(input_ids.shape[0]):
        indices = attention_mask_bool[row].nonzero(as_tuple=False).flatten()
        keep_start, keep_end = plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        kept_id_rows.append(input_ids[row, kept_indices])
        kept_pos_rows.append(position_ids[row, kept_indices])
        # loss_mask 的 kept 区段（如果存在）
        kept_mask_rows.append(torch.ones(
            kept_indices.shape[0], dtype=torch.bool, device=input_ids.device,
        ))

    # 用裁剪后的序列构建 NestedTensor（jagged layout）
    # 这样 preprocess_thd_engine 会从 offsets 正确计算 cu_seqlens
    trimmed_batch = _clone_batch(batch)
    trimmed_batch["input_ids"] = torch.nested.nested_tensor(kept_id_rows, layout=torch.jagged)
    trimmed_batch["position_ids"] = torch.nested.nested_tensor(kept_pos_rows, layout=torch.jagged)

    # loss_mask
    loss_mask = batch.get("loss_mask")
    if loss_mask is not None:
        kept_loss_rows = []
        for row in range(loss_mask.shape[0]):
            indices = attention_mask_bool[row].nonzero(as_tuple=False).flatten()
            keep_start, keep_end = plan.input_keep_ranges[row]
            kept_indices = indices[keep_start:keep_end]
            kept_loss_rows.append(loss_mask[row, kept_indices])
        trimmed_batch["loss_mask"] = torch.nested.nested_tensor(
            kept_loss_rows, layout=torch.jagged
        )

    return trimmed_batch


def _trim_plain_batch_bshd(batch: Any, plan: PrefixSharingPlan, attention_mask_bool: Any) -> Any:
    """Mask-trim a dense BSHD batch without changing tensor shapes."""

    trimmed_batch = _clone_batch(batch)
    new_attention_mask = attention_mask_bool.clone()
    new_attention_mask[:] = False
    valid_indices = [
        attention_mask_bool[row].nonzero(as_tuple=False).flatten()
        for row in range(attention_mask_bool.shape[0])
    ]
    for row, indices in enumerate(valid_indices):
        keep_start, keep_end = plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        new_attention_mask[row, kept_indices] = True
    trimmed_batch["attention_mask"] = new_attention_mask
    return trimmed_batch


def _build_bshd_layout_from_nested_position_ids(
    position_ids: Any,
    parallel_info: MegatronParallelInfo,
) -> BshdBatchLayout:
    """Build BSHD runtime layout matching verl's preprocess_bshd_engine output.

    v080 BSHD can receive jagged NestedTensor input. The framework later pads
    it to dense [batch, max_seqlen] where max_seqlen is aligned to TP (CP is
    still outside the supported prefix-sharing scope). The layout must describe
    that post-preprocess dense coordinate system, not the jagged input object.
    """
    import torch

    if parallel_info.cp_size != 1:
        raise RuntimeError("prefix sharing BSHD NestedTensor path currently supports CP=1 only")
    offsets = position_ids.offsets()
    values = position_ids.values()
    lengths = [int(length) for length in offsets.diff().detach().cpu().tolist()]
    batch_size = len(lengths)
    max_valid_length = max(lengths, default=0)
    align_size = max(int(parallel_info.tp_size), 1)
    max_seqlen = pad_to_multiple(max_valid_length, align_size)

    valid_token_mask = torch.zeros(
        batch_size,
        max_seqlen,
        dtype=torch.bool,
        device=values.device,
    )
    dense_position_ids = torch.zeros(
        batch_size,
        max_seqlen,
        dtype=values.dtype,
        device=values.device,
    )
    for row, valid_length in enumerate(lengths):
        if valid_length == 0:
            continue
        start = offsets[row]
        end = offsets[row + 1]
        valid_token_mask[row, :valid_length] = True
        dense_position_ids[row, :valid_length] = values[start:end]

    return BshdBatchLayout.from_valid_token_mask(
        valid_token_mask,
        position_ids=dense_position_ids,
    )


def _slice_nested_sequences(nested_tensor: Any, plan: PrefixSharingPlan) -> list[Any]:
    """从 NestedTensor 中按 keep_ranges 切片每个序列。

    Provider 序列：保留全部 tokens。
    Reuser 序列：只保留 keep_range 区段的 tokens。
    """
    offsets = nested_tensor.offsets()
    values = nested_tensor.values()

    sliced = []
    for i in range(len(plan.input_keep_ranges)):
        seq_values = values[offsets[i]:offsets[i + 1]]
        keep_start, keep_end = plan.input_keep_ranges[i]
        sliced.append(seq_values[keep_start:keep_end])

    return sliced


def _slice_2d_position_rows(
    position_ids: Any,
    plan: PrefixSharingPlan,
    attention_mask_bool: Any,
) -> list[Any]:
    """从 2D position_ids 中按 keep_ranges 提取每个 row 的 kept 区段。

    keep_range 指的是序列中第几个有效 token（在去掉 padding 后的偏移量），
    不是 position_ids 张量的列索引。必须先通过 attention_mask 找到有效列索引，
    再取子范围：kept_indices = valid_indices[keep_start:keep_end]。
    """
    kept_rows = []
    for row in range(position_ids.shape[0]):
        indices = attention_mask_bool[row].nonzero(as_tuple=False).flatten()
        keep_start, keep_end = plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        kept_rows.append(position_ids[row, kept_indices])
    return kept_rows


def _collect_kept_position_rows(
    trimmed_batch: Any,
    plan: PrefixSharingPlan,
    is_nested_tensor: bool,
    attention_mask_bool: Any | None = None,
) -> list[Any]:
    """从裁剪后的 batch 中收集各序列的 kept position_ids（per-row 1D tensors）。

    用于 ThdBatchLayout.construct_from_kept_position_ids 构建 layout。

    当 position_ids 是 2D tensor 时，需要 attention_mask_bool 来定位有效列索引
    （keep_range 是序列偏移，不是列索引）。当 position_ids 是 NestedTensor 时，
    offsets/values 已包含裁剪后的正确数据，无需 attention_mask。
    """
    position_ids = trimmed_batch["position_ids"]

    if is_nested_tensor or _is_nested_tensor(position_ids):
        offsets = position_ids.offsets()
        values = position_ids.values()
        return [values[offsets[i]:offsets[i + 1]] for i in range(len(plan.input_keep_ranges))]

    # 2D tensor — 用 attention_mask 的 valid_indices 切片
    if attention_mask_bool is None:
        raise ValueError(
            "attention_mask_bool is required when position_ids is 2D tensor; "
            "keep_range is a sequence offset, not a column index"
        )
    rows = []
    for i in range(len(plan.input_keep_ranges)):
        indices = attention_mask_bool[i].nonzero(as_tuple=False).flatten()
        keep_start, keep_end = plan.input_keep_ranges[i]
        kept_indices = indices[keep_start:keep_end]
        rows.append(position_ids[i, kept_indices])
    return rows


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


def _extract_seq_from_nested_tensor(nested_tensor: Any) -> list[list[int]]:
    """从 NestedTensor (jagged layout) 中提取每个序列的 token ID 列表。"""
    offsets = nested_tensor.offsets()
    values = nested_tensor.values()
    return [
        values[offsets[i]:offsets[i + 1]].detach().cpu().tolist()
        for i in range(offsets.diff().shape[0])
    ]
