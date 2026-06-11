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
import sys

from prefix_sharing.setup.version_guard import detect_versions, DetectedVersions
from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX, CompatEntry
from prefix_sharing.setup.registry import PatchSpec, PatchRegistry
from prefix_sharing.setup.logged_patch import PatchHandle

logger = logging.getLogger(__name__)


def _ensure_logging_visible() -> None:
    """确保 prefix_sharing 的 INFO 日志对用户可见。

    verl 等训练框架的 root logger 通常只输出 WARNING 及以上级别，
    但 prefix-sharing 的 patch 安装、运行状态信息（INFO 级别）
    对使用者至关重要。此函数为 prefix_sharing 日志命名空间配置
    专用的 StreamHandler(stderr) + INFO 级别 + propagate=False，
    确保 [PS] 消息始终可见，不受 root logger 配置影响。
    用户已自行配置 handler 时，不干预。
    """
    ns_logger = logging.getLogger("prefix_sharing")
    if ns_logger.handlers:
        return  # 用户已自行配置，不干预
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    ns_logger.addHandler(handler)
    ns_logger.setLevel(logging.INFO)
    ns_logger.propagate = False


class IncompatibleEnvironment(RuntimeError):
    """版本组合不在兼容矩阵中。"""


def check() -> DetectedVersions:
    """仅探测版本并校验兼容性，不安装 patch。

    Returns: 探测到的版本信息
    Raises: IncompatibleEnvironment — 版本组合不兼容
    """
    _ensure_logging_visible()
    versions = detect_versions()
    entry = _find_compat_entry(versions)
    if entry is None:
        raise IncompatibleEnvironment(
            f"不兼容的版本组合: verl={versions.verl}, "
            f"megatron_core={versions.megatron_core}, "
            f"mindspeed={versions.mindspeed}。\n"
            + _format_compat_matrix()
        )
    logger.info(
        "[PS] Version check: verl=%s, megatron_core=%s, "
        "mindspeed=%s → compatible (patch_set=%s)",
        versions.verl, versions.megatron_core,
        versions.mindspeed,
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
        if e.mindspeed is not None:
            parts.append(f"mindspeed={e.mindspeed}")
        lines.append(f"  组合{e.patch_set_id}: " + " + ".join(parts))
    return "\n".join(lines)