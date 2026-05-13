"""Runtime context passed from framework preprocess into patched attention."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from prefix_sharing.core.cache import PrefixKVCache
from prefix_sharing.core.metadata import PrefixSharingBatchMeta


_current_context: ContextVar["PrefixSharingRuntimeContext | None"] = ContextVar(
    "prefix_sharing_context",
    default=None,
)


@dataclass
class PrefixSharingRuntimeContext:
    meta: PrefixSharingBatchMeta
    cache: PrefixKVCache
    backend: Any | None = None
    kept_position_ids: Any | None = None
    restore_positions: list[Any] = field(default_factory=list)


def current_prefix_sharing_context() -> PrefixSharingRuntimeContext | None:
    return _current_context.get()


@contextmanager
def prefix_sharing_context(
    meta: PrefixSharingBatchMeta,
    *,
    backend: Any | None = None,
    kept_position_ids: Any | None = None,
    restore_positions: list[Any] | None = None,
) -> Iterator[PrefixSharingRuntimeContext]:
    ctx = PrefixSharingRuntimeContext(
        meta=meta,
        cache=PrefixKVCache(),
        backend=backend,
        kept_position_ids=kept_position_ids,
        restore_positions=list(restore_positions or []),
    )
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)
        ctx.cache.close()


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
    )
