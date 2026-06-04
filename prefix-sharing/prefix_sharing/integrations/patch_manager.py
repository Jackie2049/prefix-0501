"""Small monkey-patch manager with rollback and idempotent handles."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable


@dataclass(frozen=True)
class _PatchRecord:
    target: Any
    attr_name: str
    original: Any
    replacement: Any


class PatchHandle:
    def __init__(self, records: list[_PatchRecord]) -> None:
        self._records = records
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def disable(self) -> None:
        if not self._active:
            return
        for record in reversed(self._records):
            setattr(record.target, record.attr_name, record.original)
            key = (id(record.target), record.attr_name)
            if PatchManager._active_patches.get(key) is record:
                del PatchManager._active_patches[key]
        self._active = False

    def __enter__(self) -> "PatchHandle":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disable()


class PatchManager:
    """Install attribute patches and roll back if any step fails."""

    # Class-level registry keyed by (id(target), attr_name) to prevent
    # double-install of the same patch without disabling the previous one.
    _active_patches: dict[tuple[int, str], _PatchRecord] = {}

    def __init__(self) -> None:
        self._records: list[_PatchRecord] = []

    def patch_attr(
        self,
        target: Any,
        attr_name: str,
        replacement: Any,
        *,
        signature_check: Callable[[Any], None] | None = None,
    ) -> None:
        if not hasattr(target, attr_name):
            raise AttributeError(f"{target!r} has no attribute {attr_name!r}")
        original = getattr(target, attr_name)
        if original is replacement:
            return

        key = (id(target), attr_name)
        existing = self._active_patches.get(key)
        if existing is not None and existing.replacement is replacement:
            # Same replacement already active — no-op
            return
        if existing is not None:
            raise RuntimeError(
                f"{target!r}.{attr_name} is already patched. Disable the previous "
                f"patch before installing a new one."
            )

        if signature_check is not None:
            signature_check(original)
        record = _PatchRecord(
            target=target,
            attr_name=attr_name,
            original=original,
            replacement=replacement,
        )
        setattr(target, attr_name, replacement)
        self._records.append(record)
        self._active_patches[key] = record

    def handle(self) -> PatchHandle:
        records = list(self._records)
        self._records.clear()
        return PatchHandle(records)

    def rollback(self) -> None:
        handle = self.handle()
        handle.disable()
