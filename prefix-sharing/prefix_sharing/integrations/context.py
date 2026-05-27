"""Runtime context passed from framework preprocess into patched attention."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from prefix_sharing.core.metadata import PrefixSharingPlan
from prefix_sharing.core.prefix_store import PrefixKVStore


_current_context: ContextVar["PrefixSharingRuntimeContext | None"] = ContextVar(
    "prefix_sharing_context",
    default=None,
)


@dataclass
class PrefixSharingRuntimeContext:
    prefix_sharing_plan: PrefixSharingPlan
    store: PrefixKVStore
    backend: Any | None = None
    kept_position_ids: Any | None = None
    prefix_last_restore_slots: list[Any] = field(default_factory=list)


def current_prefix_sharing_context() -> PrefixSharingRuntimeContext | None:
    return _current_context.get()


@contextmanager
def prefix_sharing_context(
    prefix_sharing_plan: PrefixSharingPlan,
    *,
    backend: Any | None = None,
    kept_position_ids: Any | None = None,
    prefix_last_restore_slots: list[Any] | None = None,
) -> Iterator[PrefixSharingRuntimeContext]:
    ctx = PrefixSharingRuntimeContext(
        prefix_sharing_plan=prefix_sharing_plan,
        store=PrefixKVStore(),
        backend=backend,
        kept_position_ids=kept_position_ids,
        prefix_last_restore_slots=list(prefix_last_restore_slots or []),
    )
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)
        ctx.store.close()


def optional_prefix_sharing_context(
    prefix_sharing_runtime_state: Any | None,
) -> Iterator[PrefixSharingRuntimeContext | None]:
    if prefix_sharing_runtime_state is None:
        return nullcontext(None)
    return prefix_sharing_context(
        prefix_sharing_runtime_state.prefix_sharing_plan,
        backend=getattr(prefix_sharing_runtime_state, "backend", None),
        kept_position_ids=getattr(prefix_sharing_runtime_state, "kept_position_ids", None),
        prefix_last_restore_slots=getattr(prefix_sharing_runtime_state, "prefix_last_restore_slots", None),
    )
