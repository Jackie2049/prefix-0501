"""Runtime context passed from framework preprocess into patched attention."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.observability import (
    PrefixSharingStats,
    _stats_var,
)
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo

logger = logging.getLogger(__file__)

_current_context: ContextVar["PrefixSharingRuntimeContext | None"] = ContextVar(
    "prefix_sharing_context",
    default=None,
)


@dataclass
class PackedPrefixLastRestoreIndex:
    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    provider_1d_pos: int
    reuse_1d_pos: int


@dataclass(init=False)
class PrefixSharingRuntimeContext:
    prefix_sharing_plan: PrefixSharingPlan
    packed_batch_layout: PackedBatchLayout
    parallel_info: MegatronParallelInfo
    store: PrefixAttentionStore
    backend: Any | None = None
    stats: PrefixSharingStats | None = None
    prefix_last_restore_indices: list[PackedPrefixLastRestoreIndex] = field(default_factory=list)

    def __init__(self, runtime_state: Any, store: PrefixAttentionStore) -> None:
        self.prefix_sharing_plan = runtime_state.prefix_sharing_plan
        self.packed_batch_layout = runtime_state.packed_batch_layout
        self.parallel_info = runtime_state.parallel_info
        self.store = store
        self.backend = runtime_state.backend
        self.stats = PrefixSharingStats.from_plan(
            runtime_state.prefix_sharing_plan,
            runtime_state.packed_batch_layout,
        )
        self.prefix_last_restore_indices = _build_prefix_last_restore_indices(
            runtime_state.prefix_sharing_plan,
            runtime_state.packed_batch_layout,
        )


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
        reuse_offset = (
            spec.reuse_first_suffix_label_pos - prefix_sharing_plan.input_keep_ranges[reuse_idx][0]
        )
        indices.append(
            PackedPrefixLastRestoreIndex(
                reuse_idx_in_batch=reuse_idx,
                provider_idx_in_batch=provider_idx,
                provider_1d_pos=packed_batch_layout.packed_index(provider_idx, provider_offset),
                reuse_1d_pos=packed_batch_layout.packed_index(reuse_idx, reuse_offset),
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

    store = PrefixAttentionStore()
    ctx = PrefixSharingRuntimeContext(prefix_sharing_runtime_state, store)
    token = _current_context.set(ctx)
    stats_token = _stats_var.set(ctx.stats)
    try:
        yield ctx
    finally:
        _current_context.reset(token)
        _stats_var.reset(stats_token)
        _log_prefix_sharing_audit(ctx)
        ctx.store.close()


def _log_prefix_sharing_audit(ctx: PrefixSharingRuntimeContext) -> None:
    """Log full stats summary and per-layer mismatch detection at context exit."""
    if ctx.stats is None:
        return
    stats = ctx.stats
    parallel_info = ctx.parallel_info

    mismatch_layers = []
    for layer_id in sorted(stats.layers):
        if not stats.layer_matches_expected(layer_id):
            mismatch_layers.append(layer_id)

    logger.warning(
        "[PS][audit][global_rank=%s tp_rank=%s/tp_size=%s "
        "pp_rank=%s/pp_size=%s] "
        "forward_id=%s micro_batch_id=%s batch_size=%s "
        "original_tokens=%s kept_valid_tokens=%s kept_padded_tokens=%s "
        "reused_valid_tokens=%s reused_valid_token_ratio=%.4f "
        "provider_count=%s reuser_count=%s sharing_group_count=%s "
        "expected_reused_per_layer=%s expected_reused_prefix_per_layer=%s "
        "expected_restore=%s actual_restore=%s "
        "layers_recorded=%s mismatch_layers=%s",
        parallel_info.global_rank,
        parallel_info.tp_rank,
        parallel_info.tp_size,
        parallel_info.pp_rank,
        parallel_info.pp_size,
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
        len(stats.layers),
        mismatch_layers,
    )

    # Log per-layer detail
    for layer_id in sorted(stats.layers):
        layer = stats.layers[layer_id]
        logger.warning(
            "[PS][audit][layer=%s] store_count=%s reuse_count=%s "
            "reuse_hit=%s reuse_miss=%s stored_tokens=%s "
            "reused_prefix_tokens=%s expanded_kv_tokens=%s "
            "valid_q_tokens=%s padded_q_tokens=%s "
            "matches_expected=%s",
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
