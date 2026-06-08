"""verl Megatron actor integration helpers.

This module has two layers:

* ``VerlMCoreBatchAdapter`` is framework-light and testable locally. It turns a
  verl-style micro-batch payload into prefix-sharing metadata plus trimmed
  inputs/labels/masks, and it assembles restored logprobs after forward.
* ``VerlMCoreIntegration`` installs the Megatron attention patch. The real
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
from prefix_sharing.core.model_spec import ModelSpec
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.integrations.context import prefix_sharing_runtime_context as _prefix_sharing_runtime_context
from prefix_sharing.integrations.megatron_attention import IntegrationUnavailable, MegatronAttentionIntegration
from prefix_sharing.integrations.patch_manager import PatchHandle
from prefix_sharing.integrations.verl_attention import VerlQwen3_6Integration

import torch
import logging
logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass(frozen=True)
class PrefixSharingRuntimeState:
    prefix_sharing_plan: PrefixSharingPlan
    backend: Any
    packed_batch_layout: PackedBatchLayout
    model_spec: ModelSpec | None = None
    # Two-pass PS: prefix tokens for the provider sequence
    prefix_input_ids: Any | None = None
    prefix_attention_mask: Any | None = None
    prefix_position_ids: Any | None = None


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
        *,
        model_spec: ModelSpec | None = None,
    ) -> Iterator[Any]:
        """Open the runtime context consumed by patched attention."""

        runtime_state = PrefixSharingRuntimeState(
            prefix_sharing_plan=prefix_sharing_batch.prefix_sharing_plan,
            backend=get_backend_instance(self.config),
            packed_batch_layout=PackedBatchLayout.from_valid_lengths(
                prefix_sharing_batch.prefix_sharing_plan.kept_lengths_q
            ),
            model_spec=model_spec,
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
    model_spec: ModelSpec | None = None,
) -> tuple[Any, PrefixSharingRuntimeState | None]:
    """Trim one verl Megatron actor micro-batch in-place for prefix sharing.

    The framework-facing contract is intentionally small: dependency/verl calls
    this once after obtaining the micro-batch and then opens
    :func:`prefix_sharing_runtime_context` around its existing forward
    call. Unsupported or disabled cases return ``(batch, None)``.
    """

    batch_size = len(batch["input_ids"]) if "input_ids" in batch else None
    logger.debug(f"[PS][prepare] ENTER: batch_size={batch_size}, batch_keys={list(batch.keys())}")

    config = PrefixSharingConfig.from_raw(
        _read_actor_value(actor_config, "prefix_sharing_config", None)
    )

    # --- Path 1: prefix sharing disabled by config ---
    if not config.enable_prefix_sharing:
        logger.debug(f"[PS][prepare] PATH 1: prefix sharing disabled (config.enable_prefix_sharing=False), returning (batch, None)")
        return batch, None

    logger.debug(f"[PS][prepare] config.enable_prefix_sharing=True, validating config...")

    config.validate(model_config=model_config, integrate_mode="verl_megatron_actor")
    logger.debug(f"[PS][prepare] config.validate() returned OK")

    # --- Path 2: data format validation ---
    # PS now supports both BSHD (use_remove_padding=False) and THD (use_remove_padding=True)
    logger.debug("[PS][prepare] checking data format...")
    use_remove_padding = _read_actor_bool(actor_config, "megatron.use_remove_padding", False)
    logger.debug(f"[PS][prepare] use_remove_padding={use_remove_padding}")

    # --- Path 3: multi_modal check ---
    logger.debug(f"[PS][prepare] use_remove_padding=True, about to batch.get(multi_modal_inputs)...")
    multi_modal_inputs = batch.get("multi_modal_inputs")
    if multi_modal_inputs is not None:
        # tensorclass 无法遍历（触发 CUDA 同步），改用底层 td 检查字段数
        is_tensorclass = hasattr(multi_modal_inputs, 'batch_size')
        logger.debug(f"[PS][prepare] multi_modal_inputs type: tensorclass={is_tensorclass}, type={type(multi_modal_inputs).__name__}")
        if is_tensorclass:
            _td = getattr(multi_modal_inputs, 'td', None) or getattr(multi_modal_inputs, '_tensordict', None)
            _keys = list(_td.keys()) if _td is not None else []
            has_mm = len(_keys) > 0
            logger.debug(f"[PS][prepare] tensorclass td keys={_keys}, has_mm={has_mm}")
        elif hasattr(multi_modal_inputs, '__iter__') and not isinstance(multi_modal_inputs, (str, bytes)):
            has_mm = any(mmi is not None and len(mmi.keys()) > 0 for mmi in multi_modal_inputs)
        else:
            has_mm = False
            logger.debug(f"[PS][prepare] multi_modal_inputs non-iterable: {type(multi_modal_inputs).__name__}")
        if has_mm:
            logger.debug(f"[PS][prepare] PATH 3: multi_modal_inputs has content, raising RuntimeError")
            raise RuntimeError("prefix sharing phase 1 supports only text-only actor micro-batches")
        logger.debug(f"[PS][prepare] multi_modal check PASSED (no real multi-modal content)")

    # --- Read tensors ---
    attention_mask = batch["attention_mask"].to(bool)
    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]
    logger.debug(f"[PS][prepare] tensor shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, position_ids={position_ids.shape}")

    # --- Path 4: wrong tensor dims ---
    if attention_mask.dim() != 2 or input_ids.dim() != 2 or position_ids.dim() != 2:
        logger.debug(f"[PS][prepare] PATH 4: non-2D tensors detected, raising RuntimeError")
        raise RuntimeError("prefix sharing phase 1 expects 2D input_ids/attention_mask/position_ids")

    # --- Planning ---
    valid_indices = [attention_mask[row].nonzero(as_tuple=False).flatten() for row in range(input_ids.shape[0])]
    sequences = [input_ids[row, indices].detach().cpu().tolist() for row, indices in enumerate(valid_indices)]
    seq_lens = [len(s) for s in sequences]
    logger.debug(f"[PS][prepare] sequences: num_seq={len(sequences)}, seq_lens={seq_lens}")

    prefix_sharing_plan = PrefixSharingPlanner(config).plan(sequences)
    logger.debug(
        f"[PS][prepare] prefix_sharing_plan result: has_sharing={prefix_sharing_plan.has_sharing}, "
        f"keep_ranges={prefix_sharing_plan.input_keep_ranges}, "
        f"prefix_last_restore={prefix_sharing_plan.prefix_last_restore}"
    )
    if prefix_sharing_plan.has_sharing:
        logger.info("[PS][prepare] %s", prefix_sharing_plan.summary())

    # --- Path 5: no sharing found ---
    if not prefix_sharing_plan.has_sharing:
        logger.debug(f"[PS][prepare] PATH 5: no sharing detected, returning (batch, None)")
        return batch, None

    # --- Path 6: sharing found, trim the original micro-batch ---
    logger.debug(f"[PS][prepare] PATH 6: sharing detected, preparing two-pass trimmed batch...")

    # Extract provider's prefix tokens for the prefix pass (two-pass PS)
    # Find the provider index (the sequence that provides prefix KV/DeltaNet state)
    provider_idx = None
    for i, is_prov in enumerate(prefix_sharing_plan.is_provider):
        if is_prov:
            provider_idx = i
            break
    if provider_idx is None:
        logger.warning("[PS][prepare] No provider found in prefix sharing plan, falling back to single-pass")
        return batch, None

    # IMPORTANT: provider's own prefix_lens[provider_idx] is 0 (providers don't reuse).
    # The actual shared prefix length comes from the reuse_specs — the maximum prefix_len
    # among all reusers that reference this provider.
    provider_prefix_len = 0
    for spec in prefix_sharing_plan.reuse_specs:
        if spec.provider_idx_in_batch == provider_idx:
            provider_prefix_len = max(provider_prefix_len, spec.prefix_len)
    if provider_prefix_len == 0:
        logger.warning("[PS][prepare] No reuse specs with prefix_len>0 for provider %s, falling back", provider_idx)
        return batch, None

    provider_indices = valid_indices[provider_idx]

    # Provider's prefix tokens: positions 0..prefix_len-1 (before suffix)
    prefix_token_indices = provider_indices[:provider_prefix_len]
    prefix_input_ids = input_ids[provider_idx, prefix_token_indices].unsqueeze(0)  # (1, prefix_len)
    prefix_attention_mask = torch.ones(1, provider_prefix_len, dtype=bool, device=input_ids.device)
    prefix_position_ids = position_ids[provider_idx, prefix_token_indices].unsqueeze(0)  # (1, prefix_len)

    logger.debug(
        f"[PS][prepare] Two-pass prefix tokens: provider_idx={provider_idx}, "
        f"prefix_len={provider_prefix_len}, prefix_input_ids shape={prefix_input_ids.shape}"
    )

    # Trim all sequences to suffix-only (including provider)
    # This is the key change for two-pass: provider also becomes suffix-only,
    # so DeltaNet state injection works for ALL sequences in the suffix pass.
    trimmed_micro_batch = _clone_batch(batch)
    new_attention_mask = attention_mask.clone()
    new_attention_mask[:] = False
    new_input_ids = input_ids.clone()
    new_position_ids = position_ids.clone()
    kept_position_rows = []

    for row, indices in enumerate(valid_indices):
        # For two-pass, all sequences keep only suffix tokens
        # Provider: suffix starts at prefix_len
        # Reuser: suffix starts at their keep_start
        if row == provider_idx:
            # Provider keeps suffix portion (after prefix_len)
            suffix_start = provider_prefix_len
            suffix_end = len(indices)
            kept_indices = indices[suffix_start:suffix_end]
        else:
            # Reuser keeps their suffix portion (as planned)
            keep_start, keep_end = prefix_sharing_plan.input_keep_ranges[row]
            kept_indices = indices[keep_start:keep_end]
        new_attention_mask[row, kept_indices] = True
        kept_position_rows.append(position_ids[row, kept_indices])

    trimmed_micro_batch["input_ids"] = new_input_ids
    trimmed_micro_batch["attention_mask"] = new_attention_mask
    trimmed_micro_batch["position_ids"] = new_position_ids

    global_rank, tp_rank, tp_size, cp_rank, cp_size = _read_megatron_parallel_state()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    packed_batch_layout = PackedBatchLayout.from_kept_position_rows(
        kept_position_rows,
        align_size=int(align_size),
    )
    logger.debug(
        "[PS][prepare][global_rank=%s tp_rank=%s/tp_size=%s cp_rank=%s/cp_size=%s] packed_batch_layout: "
        "valid_lengths=%s, padded_lengths=%s, cu_seqlens=%s, max_seqlen=%s, "
        "total_valid=%s, total_padded=%s",
        global_rank,
        tp_rank,
        tp_size,
        cp_rank,
        cp_size,
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
        model_spec=model_spec or ModelSpec.from_hf_config(model_config),
        prefix_input_ids=prefix_input_ids,
        prefix_attention_mask=prefix_attention_mask,
        prefix_position_ids=prefix_position_ids,
    )
    logger.debug(
        "[PS][prepare] PATH 6 DONE: returning (trimmed_micro_batch, "
        f"prefix_sharing_runtime_state) with keep_ranges={prefix_sharing_plan.input_keep_ranges}"
    )
    return trimmed_micro_batch, prefix_sharing_runtime_state


def _read_megatron_parallel_state() -> tuple[int | str, int, int, int, int]:
    global_rank: int | str = "unknown"
    tp_rank = 0
    tp_size = 1
    cp_rank = 0
    cp_size = 1

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            global_rank = int(dist.get_rank())
    except Exception:
        pass

    try:
        from megatron.core import parallel_state as mpu

        tp_size = int(mpu.get_tensor_model_parallel_world_size())
        if hasattr(mpu, "get_tensor_model_parallel_rank"):
            tp_rank = int(mpu.get_tensor_model_parallel_rank())
        if hasattr(mpu, "get_context_parallel_world_size"):
            cp_size = int(mpu.get_context_parallel_world_size())
        if hasattr(mpu, "get_context_parallel_rank"):
            cp_rank = int(mpu.get_context_parallel_rank())
    except (ImportError, RuntimeError, AssertionError, AttributeError):
        pass

    return global_rank, tp_rank, tp_size, cp_rank, cp_size


def restore_suffix_first_log_probs_from_prefix(
    logits: Any,
    labels: Any,
    log_probs: Any,
    vocab_parallel_log_probs_fn: Any,
) -> Any:
    """Restore reuser suffix-first logprob from provider prefix-last logits.

    Called from the THD (remove-padding) path where ``logits`` and ``labels``
    have batch dimension 1 (all sequences packed into a single flat dim-1).
    The ``provider_1d_pos`` / ``reuse_1d_pos`` offsets are absolute positions
    within that packed sequence.
    """

    ctx = current_prefix_sharing_context()
    if ctx is None or not ctx.prefix_last_restore_indices:
        return log_probs
    if logits is None or log_probs is None:
        return log_probs
    restored = log_probs.clone()
    n_restored = 0
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
        n_restored += 1
    logger.debug("[PS][restore] restored %d suffix-first logprobs", n_restored)
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
