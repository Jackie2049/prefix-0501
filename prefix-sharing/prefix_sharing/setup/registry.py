"""Patch 注册与调度：PatchSpec 注册 + LoggedPatchManager 安装 + import hook。"""

from __future__ import annotations

import builtins
import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable

from prefix_sharing.setup.logged_patch import LoggedPatchManager, PatchHandle, PatchRecord

logger = logging.getLogger(__name__)


@dataclass
class PatchSpec:
    """一个待安装的 patch 规格。"""

    module_name: str          # 目标模块全限定名
    target_getter: Callable   # (module) → (target_obj, attr_name)
    patch_factory: Callable   # (original) → patched
    description: str = ""     # 人类可读描述


class PatchRegistry:
    """全局 patch 注册表，install_all() 时一次性应用所有已注册的 patch。"""

    _specs: list[PatchSpec] = []

    @classmethod
    def register(cls, spec: PatchSpec) -> None:
        cls._specs.append(spec)

    @classmethod
    def install_all(cls) -> PatchHandle:
        """对已加载模块立即 patch；对未加载模块注册 import hook。"""
        # 共享记录列表：即时 patch 和 import hook 共用，
        # 使 PatchHandle 始终拥有完整的 patch 记录。
        shared_records: list[PatchRecord] = []
        mgr = LoggedPatchManager(shared_records)
        pending: list[PatchSpec] = []

        for spec in cls._specs:
            module = sys.modules.get(spec.module_name)
            if module is not None:
                target_obj, attr_name = spec.target_getter(module)
                original = getattr(target_obj, attr_name)
                patched = spec.patch_factory(original)
                mgr.patch_attr(target_obj, attr_name, patched)
            else:
                pending.append(spec)

        handle = PatchHandle(shared_records, specs=list(cls._specs))

        if pending:
            _activate_import_hook(pending, shared_records)

        return handle


_original_import = None


def _activate_import_hook(
    pending_specs: list[PatchSpec],
    shared_records: list[PatchRecord],
) -> None:
    """对未加载的目标模块，临时拦截 __import__，加载时自动 patch。

    所有目标模块加载完毕后立即恢复原始 __import__。
    patch 记录追加到 shared_records，使 PatchHandle 始终完整。
    """
    global _original_import

    if _original_import is not None:
        return  # hook 已激活

    lookup = {spec.module_name: spec for spec in pending_specs}
    _original_import = builtins.__import__

    def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        global _original_import
        module = _original_import(name, globals, locals, fromlist, level)
        if name in lookup:
            spec = lookup.pop(name)
            # __import__ 在 fromlist 为空时返回顶层包而非子模块，
            # 必须从 sys.modules 取实际加载的模块对象。
            actual_module = sys.modules[name]
            target_obj, attr_name = spec.target_getter(actual_module)
            original = getattr(target_obj, attr_name)
            patched = spec.patch_factory(original)
            setattr(target_obj, attr_name, patched)
            # 记录追加到共享列表，使 handle.describe() / inspect_patch() 可见
            shared_records.append(
                PatchRecord(
                    target=target_obj,
                    attr_name=attr_name,
                    original=original,
                    replacement=patched,
                )
            )
            logger.info(
                "[PS] Auto-patched %s on import of %s",
                spec.description, name,
            )
            if not lookup:
                builtins.__import__ = _original_import
                _original_import = None
                logger.info(
                    "[PS] All import hooks resolved, __import__ restored"
                )
        return module

    builtins.__import__ = hooked_import
    logger.info(
        "[PS] Import hook activated for %d modules: %s",
        len(lookup), list(lookup.keys()),
    )