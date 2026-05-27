"""Runtime context passed from framework preprocess into patched attention."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from prefix_sharing.core.planner import PrefixSharingPlan
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
def prefix_sharing_runtime_context(
    prefix_sharing_runtime_state: Any | None,
) -> Iterator[PrefixSharingRuntimeContext | None]:
    if prefix_sharing_runtime_state is None:
        yield None
        return

    ctx = PrefixSharingRuntimeContext(
        prefix_sharing_plan=prefix_sharing_runtime_state.prefix_sharing_plan,
        store=PrefixKVStore(),
        backend=prefix_sharing_runtime_state.backend,
        kept_position_ids=prefix_sharing_runtime_state.kept_position_ids,
        prefix_last_restore_slots=list(prefix_sharing_runtime_state.prefix_last_restore_slots),
    )
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)
        ctx.store.close()
