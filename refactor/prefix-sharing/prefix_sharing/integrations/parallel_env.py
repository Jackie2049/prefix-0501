"""Parallel runtime environment helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping


def _read_config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _safe_call(fn: Callable[[], Any] | None, default: int) -> int:
    if fn is None:
        return default
    try:
        return int(fn())
    except Exception:
        return default


@dataclass(frozen=True)
class ParallelEnv:
    """Rank-local parallel layout observed by prefix-sharing runtime."""

    dp_rank: int = 0
    dp_world_size: int = 1
    tp_rank: int = 0
    tp_world_size: int = 1
    pp_rank: int = 0
    pp_world_size: int = 1
    virtual_pp_rank: int | None = None
    cp_rank: int = 0
    cp_world_size: int = 1
    ep_rank: int = 0
    ep_world_size: int = 1
    sequence_parallel: bool = False

    @property
    def is_data_parallel(self) -> bool:
        return self.dp_world_size > 1

    def trace_prefix(self) -> str:
        return f"dp{self.dp_rank}/tp{self.tp_rank}/pp{self.pp_rank}/cp{self.cp_rank}"


def current_parallel_env(model_config: Any | None = None, parallel_state: Any | None = None) -> ParallelEnv:
    """Read Megatron parallel state when available, otherwise return single-rank defaults."""

    if parallel_state is None:
        try:
            parallel_state = importlib.import_module("megatron.core.parallel_state")
        except ModuleNotFoundError:
            parallel_state = None

    return ParallelEnv(
        dp_rank=_safe_call(getattr(parallel_state, "get_data_parallel_rank", None), 0),
        dp_world_size=_safe_call(getattr(parallel_state, "get_data_parallel_world_size", None), 1),
        tp_rank=_safe_call(getattr(parallel_state, "get_tensor_model_parallel_rank", None), 0),
        tp_world_size=_safe_call(getattr(parallel_state, "get_tensor_model_parallel_world_size", None), 1),
        pp_rank=_safe_call(getattr(parallel_state, "get_pipeline_model_parallel_rank", None), 0),
        pp_world_size=_safe_call(getattr(parallel_state, "get_pipeline_model_parallel_world_size", None), 1),
        virtual_pp_rank=_read_virtual_pp_rank(parallel_state),
        cp_rank=_safe_call(getattr(parallel_state, "get_context_parallel_rank", None), 0),
        cp_world_size=_safe_call(getattr(parallel_state, "get_context_parallel_world_size", None), 1),
        ep_rank=_safe_call(getattr(parallel_state, "get_expert_model_parallel_rank", None), 0),
        ep_world_size=_safe_call(getattr(parallel_state, "get_expert_model_parallel_world_size", None), 1),
        sequence_parallel=bool(_read_config_value(model_config, "sequence_parallel", False)),
    )


def _read_virtual_pp_rank(parallel_state: Any | None) -> int | None:
    if parallel_state is None:
        return None
    fn = getattr(parallel_state, "get_virtual_pipeline_model_parallel_rank", None)
    if fn is None:
        return None
    try:
        value = fn()
    except Exception:
        return None
    return None if value is None else int(value)
