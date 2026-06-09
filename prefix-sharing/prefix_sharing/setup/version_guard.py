"""运行时版本探测：从已加载/可导入的模块中提取版本号。"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import sys

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedVersions:
    verl: str | None
    megatron_core: str | None
    megatron_bridge: str | None
    mindspeed: str | None


def detect_versions() -> DetectedVersions:
    """探测当前环境中 verl、Megatron Core、Megatron Bridge、MindSpeed 的版本。

    探测顺序：sys.modules → importlib.import_module → importlib.metadata

    返回 None 表示该库不在当前环境中。
    """
    verl_ver = _detect("verl", "__version__")
    mcore_ver = _detect("megatron.core", "__version__")
    mbridge_ver = _detect("megatron.bridge", "__version__")
    ms_ver = _detect_mindspeed()

    logger.info(
        "[PS] Detected: verl=%s, megatron_core=%s, "
        "megatron_bridge=%s, mindspeed=%s",
        verl_ver, mcore_ver, mbridge_ver, ms_ver,
    )
    return DetectedVersions(
        verl=verl_ver,
        megatron_core=mcore_ver,
        megatron_bridge=mbridge_ver,
        mindspeed=ms_ver,
    )


def _detect(module_name: str, version_attr: str) -> str | None:
    """从已加载或可导入模块中读取版本属性。"""
    if module_name in sys.modules:
        return getattr(sys.modules[module_name], version_attr, None)
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, version_attr, None)
    except ModuleNotFoundError:
        return None


def _detect_mindspeed() -> str | None:
    """MindSpeed 不暴露 __version__，从 importlib.metadata 读。"""
    if "mindspeed" in sys.modules:
        return _metadata_version("mindspeed")
    try:
        importlib.import_module("mindspeed")
        return _metadata_version("mindspeed")
    except ModuleNotFoundError:
        return None


def _metadata_version(package: str) -> str | None:
    """从 importlib.metadata 读取包版本号。"""
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return None