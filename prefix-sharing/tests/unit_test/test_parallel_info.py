import sys
from types import ModuleType

from prefix_sharing.integrations.parallel_info import MegatronParallelInfo, get_megatron_parallel_info


def test_get_megatron_parallel_info_defaults_without_megatron(monkeypatch):
    monkeypatch.delitem(sys.modules, "megatron", raising=False)
    monkeypatch.delitem(sys.modules, "megatron.core", raising=False)
    monkeypatch.delitem(sys.modules, "megatron.core.parallel_state", raising=False)

    parallel_info = get_megatron_parallel_info()

    assert isinstance(parallel_info, MegatronParallelInfo)
    assert parallel_info.tp_rank == 0
    assert parallel_info.tp_size == 1
    assert parallel_info.cp_rank == 0
    assert parallel_info.cp_size == 1
    assert parallel_info.pp_rank == 0
    assert parallel_info.pp_size == 1
    assert parallel_info.is_pipeline_first_stage is True
    assert parallel_info.is_pipeline_last_stage is True
    assert parallel_info.virtual_pp_size is None


def test_get_megatron_parallel_info_reads_megatron_parallel_state(monkeypatch):
    parallel_state = ModuleType("megatron.core.parallel_state")
    parallel_state.get_tensor_model_parallel_world_size = lambda: 4
    parallel_state.get_tensor_model_parallel_rank = lambda: 2
    parallel_state.get_context_parallel_world_size = lambda: 1
    parallel_state.get_context_parallel_rank = lambda: 0
    parallel_state.get_pipeline_model_parallel_world_size = lambda: 3
    parallel_state.get_pipeline_model_parallel_rank = lambda: 1
    parallel_state.is_pipeline_first_stage = lambda ignore_virtual=True: False
    parallel_state.is_pipeline_last_stage = lambda ignore_virtual=True: False
    parallel_state.get_virtual_pipeline_model_parallel_world_size = lambda: None
    core = ModuleType("megatron.core")
    core.parallel_state = parallel_state
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)

    parallel_info = get_megatron_parallel_info()

    assert parallel_info.tp_rank == 2
    assert parallel_info.tp_size == 4
    assert parallel_info.cp_rank == 0
    assert parallel_info.cp_size == 1
    assert parallel_info.pp_rank == 1
    assert parallel_info.pp_size == 3
    assert parallel_info.is_pipeline_first_stage is False
    assert parallel_info.is_pipeline_last_stage is False
    assert parallel_info.virtual_pp_size is None
