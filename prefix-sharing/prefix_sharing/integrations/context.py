"""Runtime context passed from framework preprocess into patched attention."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from prefix_sharing.backends.batch_layout import BatchRuntimeLayout
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo


_current_context: ContextVar["PrefixSharingRuntimeContext | None"] = ContextVar(
    "prefix_sharing_context",
    default=None,
)


@dataclass
class PrefixLastRestoreIndex:
    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    provider_token_index: Any
    reuse_token_index: Any


@dataclass(init=False)
class PrefixSharingRuntimeContext:
    prefix_sharing_plan: PrefixSharingPlan
    batch_runtime_layout: BatchRuntimeLayout
    parallel_info: MegatronParallelInfo
    store: PrefixAttentionStore
    backend: Any | None = None
    prefix_last_restore_indices: list[PrefixLastRestoreIndex] = field(default_factory=list)

    def __init__(self, runtime_state: Any, store: PrefixAttentionStore) -> None:
        self.prefix_sharing_plan = runtime_state.prefix_sharing_plan
        self.batch_runtime_layout = runtime_state.batch_runtime_layout
        self.parallel_info = runtime_state.parallel_info
        self.store = store
        self.backend = runtime_state.backend
        self.prefix_last_restore_indices = _build_prefix_last_restore_indices(
            runtime_state.prefix_sharing_plan,
            runtime_state.batch_runtime_layout,
        )


def current_prefix_sharing_context() -> PrefixSharingRuntimeContext | None:
    return _current_context.get()


def _build_prefix_last_restore_indices(
    prefix_sharing_plan: PrefixSharingPlan,
    batch_runtime_layout: BatchRuntimeLayout,
) -> list[PrefixLastRestoreIndex]:
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
            PrefixLastRestoreIndex(
                reuse_idx_in_batch=reuse_idx,
                provider_idx_in_batch=provider_idx,
                provider_token_index=batch_runtime_layout.token_index(provider_idx, provider_offset),
                reuse_token_index=batch_runtime_layout.token_index(reuse_idx, reuse_offset),
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
        ctx.store.close()
