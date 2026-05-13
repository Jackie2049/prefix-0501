"""Runtime context passed from framework preprocess into patched attention."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

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


def current_prefix_sharing_context() -> PrefixSharingRuntimeContext | None:
    return _current_context.get()


@contextmanager
def prefix_sharing_context(meta: PrefixSharingBatchMeta) -> Iterator[PrefixSharingRuntimeContext]:
    ctx = PrefixSharingRuntimeContext(meta=meta, cache=PrefixKVCache())
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)
        ctx.cache.close()
