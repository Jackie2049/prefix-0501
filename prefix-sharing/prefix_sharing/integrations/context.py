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
    prefix_last_logits_saved: dict[tuple[int, int], Any] = field(default_factory=dict)
    """Saved provider packed logits for prefix-last logprob recompute in 2D space.

    Keyed by (reuse_idx_in_batch, target_2d_pos). Each value is [1, V//tp]
    cloned from packed logits after temperature scaling, before any
    in-place modification by entropy/logprob computation.

    Only populated for non-interior (prefix-last) restore specs.
    Interior specs use direct 2D copy instead.
    """
    valid_indices: list | None = None
    """Per-row tensor positions of valid tokens in the original 2D tensors.
    Used to map planner's valid-space target_2d_pos to tensor-space columns."""
    stats: PrefixSharingStats | None = None


def current_prefix_sharing_context() -> PrefixSharingRuntimeContext | None:
    return _current_context.get()


def _resolve_provider_for_position(
    plan: PrefixSharingPlan,
    provider_idx: int,
    target_pos: int,
) -> int:
    """Walk up the provider chain to find the nearest ancestor whose
    packed layout contains ``target_pos`` (absolute original position).

    In chain-reuse scenarios (row 0 → row 1 → row 2), intermediate
    providers are reusers with truncated packed layouts.  The root
    provider may also be shorter than the reuser's prefix — in that
    case intermediate providers contribute the extended range.

    The range is extended left by one position (keep_start - 1) for
    reusers: that position holds the logprob for the token at
    keep_start which is still inside the shared prefix.  Its value was
    computed via prefix-last recompute with the correct shared label
    and is safe to copy from.
    """
    while True:
        keep_start, keep_end = plan.input_keep_ranges[provider_idx]
        if keep_start - 1 <= target_pos < keep_end:
            return provider_idx
        if plan.is_reuser(provider_idx):
            provider_idx = plan.provider_index[provider_idx]
        else:
            # Can't walk further; return current (should not happen
            # with valid detection, but guard)
            return provider_idx


def _build_prefix_last_restore_indices(
    prefix_sharing_plan: PrefixSharingPlan,
    packed_batch_layout: PackedBatchLayout,
) -> list[PackedPrefixLastRestoreIndex]:
    indices = []
    for spec in prefix_sharing_plan.prefix_last_restore:
        reuse_idx = spec.reuse_idx_in_batch
        # Resolve through chain reuse to the nearest provider whose
        # packed layout contains provider_prefix_last_pos.
        target_pos = spec.provider_prefix_last_pos
        resolved_provider = _resolve_provider_for_position(
            prefix_sharing_plan, spec.provider_idx_in_batch, target_pos
        )

        # Interior: provider and reuser share the same prefix tokens,
        # so the logprob at target_pos is identical — just copy.
        # 2D restore uses provider_idx_in_batch + valid_indices mapping,
        # no packed-index lookup needed.  provider_1d_pos is only used
        # for prefix-last entries to fetch saved provider logits.
        # When keep_start-1 resolves to an intermediate reuser, offset
        # can be negative — skip packed_index and use sentinel -1.
        provider_offset = (
            target_pos - prefix_sharing_plan.input_keep_ranges[resolved_provider][0]
        )
        if provider_offset >= 0:
            provider_1d = packed_batch_layout.packed_index(resolved_provider, provider_offset)
        else:
            provider_1d = -1  # keep_start-1 of intermediate provider, no packed slot

        reuse_1d = -1  # sentinel: no slot in reuser packed region

        indices.append(
            PackedPrefixLastRestoreIndex(
                reuse_idx_in_batch=reuse_idx,
                provider_idx_in_batch=resolved_provider,
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
        valid_indices=prefix_sharing_runtime_state.valid_indices,
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
