"""带日志的 monkey-patch manager — setup 专用。

功能与 integrations/patch_manager.py 相同，但增加：
- patch_attr 时 INFO 日志
- disable 时 INFO 日志（逐条打印恢复详情）
- describe() 方法返回人类可读 patch 清单

此文件独立于 integrations/patch_manager.py，不影响原有代码。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)


def _target_name(target: Any) -> str:
    """最佳的人类可读名称。"""
    if hasattr(target, "__module__") and hasattr(target, "__qualname__"):
        return f"{target.__module__}.{target.__qualname__}"
    if hasattr(target, "__name__"):
        return target.__name__
    return repr(target)


@dataclass(frozen=True)
class PatchRecord:
    """一个已安装 patch 的记录，用于回滚和 describe。"""

    target: Any
    attr_name: str
    original: Any
    replacement: Any

    def describe(self) -> str:
        """一行人类可读描述：目标.属性: 原始 → 替换。"""
        orig = getattr(self.original, "__qualname__", repr(self.original))
        new = getattr(self.replacement, "__qualname__", repr(self.replacement))
        return f"{_target_name(self.target)}.{self.attr_name}: {orig} → {new}"


class PatchHandle:
    """patch 生命周期 handle：describe() 查看、disable() 回滚。"""

    def __init__(self, records: list[PatchRecord]) -> None:
        self._records = records
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def disable(self) -> None:
        """回滚所有 patch，逐条打印恢复日志。"""
        if not self._active:
            return
        for record in reversed(self._records):
            setattr(record.target, record.attr_name, record.original)
            logger.info(
                "[PS] Restored %s.%s → %s",
                _target_name(record.target),
                record.attr_name,
                getattr(record.original, "__qualname__", "original"),
            )
        self._active = False
        logger.info("[PS] All %d patches reverted.", len(self._records))

    def describe(self) -> str:
        """返回所有 patch 的人类可读清单，含 ACTIVE/INACTIVE 状态。"""
        status = "ACTIVE" if self._active else "INACTIVE (rolled back)"
        lines = [f"PatchHandle ({status}, {len(self._records)} patches):"]
        for i, r in enumerate(self._records, 1):
            lines.append(f"  {i}. {r.describe()}")
        return "\n".join(lines)

    def __enter__(self) -> "PatchHandle":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disable()


class LoggedPatchManager:
    """安装属性 patch 并写日志，与 PatchManager 功能相同。"""

    def __init__(self) -> None:
        self._records: list[PatchRecord] = []

    def patch_attr(self, target: Any, attr_name: str, replacement: Any) -> None:
        """替换 target.attr_name 并记录 + 写日志。"""
        if not hasattr(target, attr_name):
            raise AttributeError(
                f"{_target_name(target)} has no attribute {attr_name!r}"
            )
        original = getattr(target, attr_name)
        if original is replacement:
            return  # idempotent
        setattr(target, attr_name, replacement)
        self._records.append(
            PatchRecord(
                target=target,
                attr_name=attr_name,
                original=original,
                replacement=replacement,
            )
        )
        logger.info(
            "[PS] Patched %s.%s: %s → %s",
            _target_name(target),
            attr_name,
            getattr(original, "__qualname__", "original"),
            getattr(replacement, "__qualname__", "replacement"),
        )

    def handle(self) -> PatchHandle:
        records = list(self._records)
        self._records.clear()
        return PatchHandle(records)

    def rollback(self) -> None:
        self.handle().disable()