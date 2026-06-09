"""setup — 版本门卫 + 条件化运行时 patch 注入。

使用：
    import prefix_sharing
    handle = prefix_sharing.setup.install()
    print(handle.describe())
    handle.disable()
"""

from __future__ import annotations

import importlib
import logging

from prefix_sharing.setup.version_guard import detect_versions, DetectedVersions
from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX, CompatEntry
from prefix_sharing.setup.registry import PatchSpec, PatchRegistry
from prefix_sharing.setup.logged_patch import PatchHandle

logger = logging.getLogger(__name__)


class IncompatibleEnvironment(RuntimeError):
    """版本组合不在兼容矩阵中。"""


def check() -> DetectedVersions:
    """仅探测版本并校验兼容性，不安装 patch。

    Returns: 探测到的版本信息
    Raises: IncompatibleEnvironment — 版本组合不兼容
    """
    versions = detect_versions()
    entry = _find_compat_entry(versions)
    if entry is None:
        raise IncompatibleEnvironment(
            f"不兼容的版本组合: verl={versions.verl}, "
            f"megatron_core={versions.megatron_core}, "
            f"megatron_bridge={versions.megatron_bridge}, "
            f"mindspeed={versions.mindspeed}。\n"
            + _format_compat_matrix()
        )
    logger.info(
        "[PS] Version check: verl=%s, megatron_core=%s, "
        "megatron_bridge=%s, mindspeed=%s → compatible (patch_set=%s)",
        versions.verl, versions.megatron_core,
        versions.megatron_bridge, versions.mindspeed,
        entry.patch_set_id,
    )
    return versions


def install() -> PatchHandle:
    """一键安装：版本探测 → 矩阵匹配 → 注册 patch → 应用 → 返回 handle。

    Returns: PatchHandle — 可调用 describe() 查看详情、disable() 回滚
    Raises: IncompatibleEnvironment — 版本组合不兼容
    """
    versions = check()
    entry = _find_compat_entry(versions)
    patch_set = _load_patch_set(entry.patch_set_id)

    for spec in patch_set:
        PatchRegistry.register(spec)

    handle = PatchRegistry.install_all()

    logger.info(
        "[PS] install() complete. %d patches active. patch_set=%s",
        len(patch_set), entry.patch_set_id,
    )
    return handle


def _find_compat_entry(versions: DetectedVersions) -> CompatEntry | None:
    for entry in COMPAT_MATRIX:
        if entry.match(versions):
            return entry
    return None


def _load_patch_set(patch_set_id: str) -> list[PatchSpec]:
    mod = importlib.import_module(
        f"prefix_sharing.setup.patches.{patch_set_id}"
    )
    return mod.PATCH_SET


def _format_compat_matrix() -> str:
    lines = ["支持的组合："]
    for e in COMPAT_MATRIX:
        parts = []
        if e.verl is not None:
            parts.append(f"verl={e.verl}")
        parts.append(f"megatron-core={e.megatron_core}")
        if e.megatron_bridge is not None:
            parts.append(f"megatron-bridge={e.megatron_bridge}")
        if e.mindspeed is not None:
            parts.append(f"mindspeed={e.mindspeed}")
        lines.append(f"  组合{e.patch_set_id}: " + " + ".join(parts))
    return "\n".join(lines)