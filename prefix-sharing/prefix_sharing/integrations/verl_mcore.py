"""verl Megatron actor integration helpers.

This module has two layers:

* ``build_prefix_sharing_micro_batch`` / ``restore_reuser_prefix_columns_2d``
  are the production entry points consumed by the verl Megatron actor. They
  prepare a trimmed micro-batch and restore reuser prefix columns in 2D space
  after forward.
* ``VerlMCoreIntegration`` installs the Megatron attention patch. The real
  Megatron QKV rewiring still requires the framework runtime and remains guarded
  by optional integration tests.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.integrations.megatron_attention import IntegrationUnavailable, MegatronAttentionIntegration
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.parallel_info import get_megatron_parallel_info
from prefix_sharing.integrations.patch_manager import PatchHandle
from prefix_sharing.integrations.utils import ensure_global_packed_token_lengths

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrefixSharingRuntimeState:
    prefix_sharing_plan: PrefixSharingPlan
    backend: Any
    packed_batch_layout: PackedBatchLayout
    parallel_info: MegatronParallelInfo
    valid_indices: list | None = None
    """Per-row tensor positions of valid (non-padding) tokens in the
    original 2D tensors.  Used to map planner's valid-space target_2d_pos
    to tensor-space columns (needed when sequences have left padding)."""


@dataclass
class VerlMCoreIntegration:
    config: PrefixSharingConfig
    backend: Any | None = None

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
    prompt_lens: Sequence[int] | None = None,
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

    prefix_sharing_plan = PrefixSharingPlanner(config).plan(sequences, prompt_lens=prompt_lens)
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
        valid_indices=valid_indices,
    )
    logger.warning(
        "[PS][prepare] PATH 6 DONE: returning (trimmed_micro_batch, "
        f"prefix_sharing_runtime_state) with keep_ranges={prefix_sharing_plan.input_keep_ranges}"
    )
    return trimmed_micro_batch, prefix_sharing_runtime_state


def restore_reuser_prefix_columns_2d(
    output: dict[str, Any],
    label_2d: Any,
    vocab_parallel_log_probs_fn: Any,
    vocab_parallel_entropy_fn: Any = None,
) -> dict[str, Any]:
    """Restore reuser prefix columns in 2D space after postprocess_packed_seqs.

    All prefix-column restoration happens purely in 2D [B, L] space,
    consolidating the previous three-phase approach (packed compute →
    cache → 2D inject) into a single post-forward step.

    For each :class:`PackedPrefixLastRestoreIndex` in the runtime context:

    - **Interior response** (shared-prefix token): logprob and entropy
      are identical between provider and reuser because the label is the
      same shared token and the logits are the same (same KV).  Directly
      copy from the provider's 2D row.

    - **Prefix-last token**: entropy is still the same (same logits),
      so copy from provider's 2D row.  Logprob depends on the label
      which differs (reuser's first suffix token ≠ provider's), so
      recompute from saved provider packed logits + reuser's 2D label.

    Must be called while ``prefix_sharing_runtime_context`` is still
    active (i.e. before the context manager exits), and after
    ``postprocess_packed_seqs`` has produced the 2D output dict.

    Args:
        output: Output dict from forward, with ``log_probs`` [B, L] and
            optionally ``entropy`` [B, L] in 2D space.
        label_2d: Original 2D label [B, L] (from ``forward_step``, before
            packed preprocessing).  In verl convention,
            ``label[p] = token at p+1``.
        vocab_parallel_log_probs_fn: Function to compute logprob from
            packed logits [1, 1, V//tp] and label [1, 1] → scalar.
            Typically :func:`verl.utils.megatron.tensor_parallel.vocab_parallel_log_probs_from_logits`.
        vocab_parallel_entropy_fn: Optional function to compute entropy
            from packed logits [1, V//tp] → scalar.

    Returns:
        ``output`` with ``log_probs`` and ``entropy`` mutated in-place.
    """

    ctx = current_prefix_sharing_context()
    if ctx is None or not ctx.prefix_last_restore_indices:
        return output

    log_probs = output.get("log_probs")
    if log_probs is None:
        return output
    entropy = output.get("entropy")
    # Map planner's valid-space target_2d_pos (0-based within valid content)
    # to tensor-space 2D columns.  postprocess_packed_seqs places tokens at
    # their ORIGINAL attention_mask positions, so a valid-space position p
    # maps to valid_indices[row][p] in the [B, L] output.
    valid_indices = ctx.valid_indices

    def _map_2d_col(row: int, valid_pos: int) -> int:
        if valid_indices is not None:
            vi = valid_indices[row]
            if vi is not None and 0 <= valid_pos < len(vi):
                return int(vi[valid_pos].item())
        return valid_pos

    non_interior_count = 0
    for index in ctx.prefix_last_restore_indices:
        reuser_row = index.reuse_idx_in_batch
        provider_row = index.provider_idx_in_batch
        valid_col = index.target_2d_pos
        # Map valid-space col to per-row tensor-space columns:
        # postprocess_packed_seqs places tokens at original
        # attention_mask positions, which may differ between rows.
        provider_col = _map_2d_col(provider_row, valid_col)
        reuser_col = _map_2d_col(reuser_row, valid_col)

        if index.is_interior_response:
            # Interior: token is in shared prefix → logprob and entropy
            # are identical to the provider's (same label, same logits).
            log_probs[reuser_row, reuser_col] = log_probs[provider_row, provider_col]
            if entropy is not None:
                entropy[reuser_row, reuser_col] = entropy[provider_row, provider_col]
        else:
            # Prefix-last: entropy is the same (same logits), so copy
            # from provider's 2D row.  Logprob differs because reuser's
            # first suffix token ≠ provider's.
            non_interior_count += 1
            if entropy is not None:
                entropy[reuser_row, reuser_col] = entropy[provider_row, provider_col]

            # Recompute logprob from saved provider packed logits
            # with reuser's own label.
            # vocab_parallel_log_probs_from_logits expects:
            #   logits: [N, V//tp]   labels: [N]
            saved_key = (reuser_row, valid_col)
            provider_logits = ctx.prefix_last_logits_saved[saved_key]  # [1, V//tp]
            reuser_label = label_2d[reuser_row:reuser_row + 1, reuser_col:reuser_col + 1].view(1)  # [1]
            log_probs[reuser_row, reuser_col] = vocab_parallel_log_probs_fn(
                provider_logits,  # [1, V//tp]
                reuser_label,    # [1]
            ).reshape(())

    # === DIAGNOSTIC: sample after restore ===
    if ctx.prefix_last_restore_indices:
        _diag_entries = ctx.prefix_last_restore_indices
        _diag_interior = [e for e in _diag_entries if e.is_interior_response]
        if _diag_interior:
            _sample = _diag_interior[0]
            _s_prov_col = _map_2d_col(_sample.provider_idx_in_batch, _sample.target_2d_pos)
            _s_reu_col = _map_2d_col(_sample.reuse_idx_in_batch, _sample.target_2d_pos)
            logger.warning(
                f"[RESTORE_DIAG] sample after restore: "
                f"reuser_row={_sample.reuse_idx_in_batch} valid_col={_sample.target_2d_pos} "
                f"reuser_tensor_col={_s_reu_col} provider_tensor_col={_s_prov_col} "
                f"log_probs[reuser]={log_probs[_sample.reuse_idx_in_batch, _s_reu_col].item():.6f} "
                f"log_probs[provider]={log_probs[_sample.provider_idx_in_batch, _s_prov_col].item():.6f}"
            )
        _diag_plast = [e for e in _diag_entries if not e.is_interior_response]
        if _diag_plast:
            _sample = _diag_plast[0]
            _s_prov_col = _map_2d_col(_sample.provider_idx_in_batch, _sample.target_2d_pos)
            _s_reu_col = _map_2d_col(_sample.reuse_idx_in_batch, _sample.target_2d_pos)
            logger.warning(
                f"[RESTORE_DIAG] sample after restore (prefix-last): "
                f"reuser_row={_sample.reuse_idx_in_batch} valid_col={_sample.target_2d_pos} "
                f"reuser_tensor_col={_s_reu_col} provider_tensor_col={_s_prov_col} "
                f"log_probs[reuser]={log_probs[_sample.reuse_idx_in_batch, _s_reu_col].item():.6f} "
                f"log_probs[provider]={log_probs[_sample.provider_idx_in_batch, _s_prov_col].item():.6f}"
            )
    # === END DIAGNOSTIC ===

    if ctx.stats is not None:
        ctx.stats.record_restore(len(ctx.prefix_last_restore_indices))
    return output


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
