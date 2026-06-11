"""Runtime context passed from framework preprocess into patched attention."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import logging
from typing import Any, Iterator

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import PrefixKVStore


_current_context: ContextVar["PrefixSharingRuntimeContext | None"] = ContextVar(
    "prefix_sharing_context",
    default=None,
)
logger = logging.getLogger(__name__)


@dataclass
class PackedPrefixLastRestoreIndex:
    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    provider_1d_pos: int
    reuse_1d_pos: int
    is_interior_response: bool = False
    output_slot: int = 0
    target_2d_pos: int = -1
    """Absolute 2D position in output where restored logprob is written."""
    label_value: int = -1
    """Actual token ID used as label (needed when label isn't in trimmed packed region)."""


@dataclass
class PrefixSharingRuntimeContext:
    prefix_sharing_plan: PrefixSharingPlan
    packed_batch_layout: PackedBatchLayout
    store: PrefixKVStore
    backend: Any | None = None
    prefix_last_restore_indices: list[PackedPrefixLastRestoreIndex] = field(default_factory=list)
    logprob_restore_cache: dict[tuple[int, int], Any] = field(default_factory=dict)
    """Cache for restored logprob scalars keyed by (batch_idx, target_2d_pos).

    Populated during packed 1D logits_processor stage with non-detached tensors
    and drained to 2D output after postprocess_packed_seqs.
    Holds both interior-response and prefix-last restore entries.
    """
    entropy_restore_cache: dict[tuple[int, int], Any] = field(default_factory=dict)
    """Cache for restored entropy scalars keyed by (batch_idx, target_2d_pos).

    Same pattern as logprob_restore_cache: populated during packed 1D
    logits_processor stage and drained to 2D output after
    postprocess_packed_seqs.
    """
    stats: PrefixSharingStats | None = None


def current_prefix_sharing_context() -> PrefixSharingRuntimeContext | None:
    return _current_context.get()


def _build_prefix_last_restore_indices(
    prefix_sharing_plan: PrefixSharingPlan,
    packed_batch_layout: PackedBatchLayout,
) -> list[PackedPrefixLastRestoreIndex]:
    indices = []
    for spec in prefix_sharing_plan.prefix_last_restore:
        provider_idx = spec.provider_idx_in_batch
        reuse_idx = spec.reuse_idx_in_batch
        provider_offset = (
            spec.provider_prefix_last_pos - prefix_sharing_plan.input_keep_ranges[provider_idx][0]
        )
        # compute provider_1d_pos: packed index for the provider's logits position
        provider_1d = packed_batch_layout.packed_index(provider_idx, provider_offset)

        if spec.is_interior_response:
            # Interior: label is in shared prefix, available in provider's
            # packed labels at provider_1d_pos (label[p] = token at p+1,
            # so label[interior_pos-1] = token at interior_pos).
            reuse_1d = -1  # sentinel: no slot in reuser packed region
        else:
            # Prefix-last: label (reuser's first suffix token) is NOT in
            # the reuser's trimmed packed labels. We use spec.label_value
            # directly instead. reuse_1d is a sentinel.
            reuse_1d = -1

        indices.append(
            PackedPrefixLastRestoreIndex(
                reuse_idx_in_batch=reuse_idx,
                provider_idx_in_batch=provider_idx,
                provider_1d_pos=provider_1d,
                reuse_1d_pos=reuse_1d,
                is_interior_response=spec.is_interior_response,
                output_slot=spec.output_slot,
                target_2d_pos=spec.target_2d_pos,
                label_value=spec.label_value,
            )
        )
    return indices


@contextmanager
def prefix_sharing_runtime_context(
    prefix_sharing_runtime_state: Any | None,
) -> Iterator[PrefixSharingRuntimeContext | None]:
    if prefix_sharing_runtime_state is None:
        yield None
        return

    ctx = PrefixSharingRuntimeContext(
        prefix_sharing_plan=prefix_sharing_runtime_state.prefix_sharing_plan,
        packed_batch_layout=prefix_sharing_runtime_state.packed_batch_layout,
        store=PrefixKVStore(),
        backend=prefix_sharing_runtime_state.backend,
        prefix_last_restore_indices=_build_prefix_last_restore_indices(
            prefix_sharing_runtime_state.prefix_sharing_plan,
            prefix_sharing_runtime_state.packed_batch_layout,
        ),
        stats=PrefixSharingStats.from_plan(
            prefix_sharing_runtime_state.prefix_sharing_plan,
            prefix_sharing_runtime_state.packed_batch_layout,
        ),
    )
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)
        _log_prefix_sharing_audit(ctx)
        ctx.store.close()


def _log_prefix_sharing_audit(ctx: PrefixSharingRuntimeContext) -> None:
    stats = ctx.stats
    if stats is None:
        return
    global_rank, tp_rank, tp_size = _read_parallel_rank_info()
    logger.warning(
        "[PS][audit][global_rank=%s tp_rank=%s/tp_size=%s] summary: "
        "forward_id=%s micro_batch_id=%s batch_size=%s "
        "original_tokens=%s kept_valid_tokens=%s kept_padded_tokens=%s "
        "reused_valid_tokens=%s reused_valid_token_ratio=%.4f "
        "provider_count=%s reuser_count=%s sharing_group_count=%s "
        "expected_reused_counts_per_layer=%s expected_reused_prefix_tokens_per_layer=%s "
        "expected_restore_count=%s actual_restore_count=%s",
        global_rank,
        tp_rank,
        tp_size,
        stats.forward_id,
        stats.micro_batch_id,
        stats.batch_size,
        stats.original_tokens,
        stats.kept_valid_tokens,
        stats.kept_padded_tokens,
        stats.reused_valid_tokens,
        stats.reused_valid_token_ratio,
        stats.provider_count,
        stats.reuser_count,
        stats.sharing_group_count,
        stats.expected_reused_counts_per_layer,
        stats.expected_reused_prefix_tokens_per_layer,
        stats.expected_restore_count,
        stats.actual_restore_count,
    )
    if not stats.layers and stats.expected_reused_counts_per_layer > 0:
        logger.warning(
            "[PS][audit][global_rank=%s tp_rank=%s/tp_size=%s] runtime_missing: "
            "expected_reused_counts_per_layer=%s expected_reused_prefix_tokens_per_layer=%s",
            global_rank,
            tp_rank,
            tp_size,
            stats.expected_reused_counts_per_layer,
            stats.expected_reused_prefix_tokens_per_layer,
        )
    for layer_id in sorted(stats.layers):
        layer = stats.layers[layer_id]
        logger.warning(
            "[PS][audit][global_rank=%s tp_rank=%s/tp_size=%s layer=%s] runtime: "
            "store_count=%s reuse_count=%s reuse_hit_count=%s reuse_miss_count=%s "
            "stored_tokens=%s reused_prefix_tokens=%s expanded_kv_tokens=%s "
            "valid_q_tokens=%s padded_q_tokens=%s matches_expected=%s",
            global_rank,
            tp_rank,
            tp_size,
            layer_id,
            layer.store_count,
            layer.reuse_count,
            layer.reuse_hit_count,
            layer.reuse_miss_count,
            layer.stored_tokens,
            layer.reused_prefix_tokens,
            layer.expanded_kv_tokens,
            layer.valid_q_tokens,
            layer.padded_q_tokens,
            stats.layer_matches_expected(layer_id),
        )


def _read_parallel_rank_info() -> tuple[int | str, int, int]:
    global_rank: int | str = "unknown"
    tp_rank = 0
    tp_size = 1

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
    except (ImportError, RuntimeError, AssertionError, AttributeError):
        pass

    return global_rank, tp_rank, tp_size
