"""Megatron parallel topology snapshot used by integration helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MegatronParallelInfo:
    """Rank information defined by Megatron parallel_state."""

    global_rank: int | str = "unknown"
    tp_rank: int = 0
    tp_size: int = 1
    cp_rank: int = 0
    cp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    is_pipeline_first_stage: bool = True
    is_pipeline_last_stage: bool = True
    virtual_pp_size: int | None = None


def get_megatron_parallel_info() -> MegatronParallelInfo:
    """Read the current Megatron TP/CP/PP topology, falling back to single-rank defaults."""

    global_rank: int | str = "unknown"
    tp_rank = 0
    tp_size = 1
    cp_rank = 0
    cp_size = 1
    pp_rank = 0
    pp_size = 1
    virtual_pp_size = None

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            global_rank = int(dist.get_rank())
    except Exception:
        pass

    try:
        from megatron.core import parallel_state as mpu

        tp_size = int(mpu.get_tensor_model_parallel_world_size())
        if hasattr(mpu, "get_tensor_model_parallel_rank"):
            tp_rank = int(mpu.get_tensor_model_parallel_rank())
        if hasattr(mpu, "get_context_parallel_world_size"):
            cp_size = int(mpu.get_context_parallel_world_size())
        if hasattr(mpu, "get_context_parallel_rank"):
            cp_rank = int(mpu.get_context_parallel_rank())
        if hasattr(mpu, "get_pipeline_model_parallel_world_size"):
            pp_size = int(mpu.get_pipeline_model_parallel_world_size())
        if hasattr(mpu, "get_pipeline_model_parallel_rank"):
            pp_rank = int(mpu.get_pipeline_model_parallel_rank())
        if hasattr(mpu, "get_virtual_pipeline_model_parallel_world_size"):
            raw_virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            virtual_pp_size = None if raw_virtual_pp_size is None else int(raw_virtual_pp_size)
        is_pipeline_first_stage = _read_pipeline_stage_flag(
            mpu,
            "is_pipeline_first_stage",
            fallback=pp_rank == 0,
        )
        is_pipeline_last_stage = _read_pipeline_stage_flag(
            mpu,
            "is_pipeline_last_stage",
            fallback=pp_rank == pp_size - 1,
        )
    except (ImportError, RuntimeError, AssertionError, AttributeError):
        is_pipeline_first_stage = True
        is_pipeline_last_stage = True

    return MegatronParallelInfo(
        global_rank=global_rank,
        tp_rank=tp_rank,
        tp_size=tp_size,
        cp_rank=cp_rank,
        cp_size=cp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        is_pipeline_first_stage=is_pipeline_first_stage,
        is_pipeline_last_stage=is_pipeline_last_stage,
        virtual_pp_size=virtual_pp_size,
    )


def _read_pipeline_stage_flag(mpu: object, name: str, *, fallback: bool) -> bool:
    predicate = getattr(mpu, name, None)
    if predicate is None:
        return fallback
    try:
        return bool(predicate(ignore_virtual=True))
    except TypeError:
        return bool(predicate())
