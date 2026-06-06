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
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo


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


@dataclass(init=False)
class PrefixSharingRuntimeContext:
    prefix_sharing_plan: PrefixSharingPlan
    packed_batch_layout: PackedBatchLayout
    parallel_info: MegatronParallelInfo
    store: PrefixAttentionStore
    backend: Any | None = None
    prefix_last_restore_indices: list[PackedPrefixLastRestoreIndex] = field(default_factory=list)
    stats: PrefixSharingStats

    def __init__(self, runtime_state: Any, store: PrefixAttentionStore) -> None:
        self.prefix_sharing_plan = runtime_state.prefix_sharing_plan
        self.packed_batch_layout = runtime_state.packed_batch_layout
        self.parallel_info = runtime_state.parallel_info
        self.store = store
        self.backend = runtime_state.backend
        self.prefix_last_restore_indices = _build_prefix_last_restore_indices(
            runtime_state.prefix_sharing_plan,
            runtime_state.packed_batch_layout,
        )
        self.stats = PrefixSharingStats.from_plan(
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
    try:
        yield ctx
    finally:
        _current_context.reset(token)
        _log_prefix_sharing_audit(ctx)
        ctx.store.close()


def _log_prefix_sharing_audit(ctx: PrefixSharingRuntimeContext) -> None:
    stats = ctx.stats
    parallel_info = ctx.parallel_info
    logger.warning(
        "[PS][audit][global_rank=%s tp_rank=%s/tp_size=%s pp_rank=%s/pp_size=%s] "
        "summary: forward_id=%s micro_batch_id=%s batch_size=%s "
        "original_tokens=%s kept_valid_tokens=%s kept_padded_tokens=%s "
        "saved_valid_tokens=%s saved_valid_token_ratio=%.4f "
        "provider_count=%s reuser_count=%s sharing_group_count=%s "
        "expected_loads_per_layer=%s expected_loaded_prefix_tokens_per_layer=%s "
        "expected_restore_count=%s actual_restore_count=%s",
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
        stats.saved_valid_tokens,
        stats.saved_valid_token_ratio,
        stats.provider_count,
        stats.reuser_count,
        stats.sharing_group_count,
        stats.expected_attention_load_count_per_layer,
        stats.expected_loaded_prefix_tokens_per_layer,
        stats.expected_restore_count,
        stats.actual_restore_count,
    )
    for layer_id in sorted(stats.layers):
        layer = stats.layers[layer_id]
        logger.warning(
            "[PS][audit][global_rank=%s tp_rank=%s/tp_size=%s pp_rank=%s/pp_size=%s layer=%s] "
            "runtime: store_count=%s load_count=%s hit_count=%s miss_count=%s "
            "stored_tokens=%s loaded_prefix_tokens=%s expanded_kv_tokens=%s "
            "valid_q_tokens=%s padded_q_tokens=%s matches_expected=%s",
            parallel_info.global_rank,
            parallel_info.tp_rank,
            parallel_info.tp_size,
            parallel_info.pp_rank,
            parallel_info.pp_size,
            layer_id,
            layer.store_count,
            layer.load_count,
            layer.hit_count,
            layer.miss_count,
            layer.stored_tokens,
            layer.loaded_prefix_tokens,
            layer.expanded_kv_tokens,
            layer.valid_q_tokens,
            layer.padded_q_tokens,
            stats.layer_matches_expected(layer_id),
        )
