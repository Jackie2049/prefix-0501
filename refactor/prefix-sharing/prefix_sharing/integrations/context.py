"""Runtime context passed from framework preprocess into patched attention."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from prefix_sharing.core.metadata import PrefixSharingBatchMeta
from prefix_sharing.core.prefix_store import PrefixKVStore
from prefix_sharing.integrations.parallel_env import ParallelEnv


_current_context: ContextVar["PrefixSharingRuntimeContext | None"] = ContextVar(
    "prefix_sharing_context",
    default=None,
)


@dataclass
class PrefixSharingRuntimeContext:
    meta: PrefixSharingBatchMeta
    store: PrefixKVStore
    backend: Any | None = None
    kept_position_ids: Any | None = None
    restore_positions: list[Any] = field(default_factory=list)
    parallel_env: ParallelEnv = field(default_factory=ParallelEnv)
    stats: "PrefixSharingStats | None" = None


@dataclass(frozen=True)
class PrefixSharingStats:
    trace_key: str
    dp_rank: int
    dp_world_size: int
    batch_size: int
    reuse_count: int
    provider_count: int
    saved_tokens_q: int
    expanded_tokens_kv: int
    fallback_reason: str | None = None


def current_prefix_sharing_context() -> PrefixSharingRuntimeContext | None:
    return _current_context.get()


@contextmanager
def prefix_sharing_context(
    meta: PrefixSharingBatchMeta,
    *,
    backend: Any | None = None,
    kept_position_ids: Any | None = None,
    restore_positions: list[Any] | None = None,
    parallel_env: ParallelEnv | None = None,
    stats: PrefixSharingStats | None = None,
) -> Iterator[PrefixSharingRuntimeContext]:
    active_parallel_env = parallel_env or ParallelEnv()
    ctx = PrefixSharingRuntimeContext(
        meta=meta,
        store=PrefixKVStore(),
        backend=backend,
        kept_position_ids=kept_position_ids,
        restore_positions=list(restore_positions or []),
        parallel_env=active_parallel_env,
        stats=stats or build_prefix_sharing_stats(meta, active_parallel_env),
    )
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)
        ctx.store.close()


def optional_prefix_sharing_context(
    prepared: Any | None,
) -> Iterator[PrefixSharingRuntimeContext | None]:
    if prepared is None:
        return nullcontext(None)
    return prefix_sharing_context(
        prepared.meta,
        backend=getattr(prepared, "backend", None),
        kept_position_ids=getattr(prepared, "kept_position_ids", None),
        restore_positions=getattr(prepared, "restore_positions", None),
        parallel_env=getattr(prepared, "parallel_env", None),
    )


def build_prefix_sharing_stats(
    meta: PrefixSharingBatchMeta,
    parallel_env: ParallelEnv,
    *,
    fallback_reason: str | None = None,
) -> PrefixSharingStats:
    trace_key = f"{parallel_env.trace_prefix()}/fw{meta.forward_id}/mb{meta.micro_batch_id}"
    return PrefixSharingStats(
        trace_key=trace_key,
        dp_rank=parallel_env.dp_rank,
        dp_world_size=parallel_env.dp_world_size,
        batch_size=meta.batch_size,
        reuse_count=len(meta.reuse_specs),
        provider_count=sum(1 for is_provider in meta.is_provider if is_provider),
        saved_tokens_q=sum(meta.original_lengths) - sum(meta.kept_lengths_q),
        expanded_tokens_kv=sum(meta.expanded_lengths_kv),
        fallback_reason=fallback_reason,
    )
