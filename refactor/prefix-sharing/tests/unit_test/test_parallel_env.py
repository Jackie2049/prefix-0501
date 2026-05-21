from types import SimpleNamespace

from prefix_sharing.integrations.parallel_env import ParallelEnv, current_parallel_env


def test_parallel_env_defaults_to_single_rank_without_megatron():
    env = current_parallel_env(parallel_state=None)

    assert env == ParallelEnv()
    assert not env.is_data_parallel
    assert env.trace_prefix() == "dp0/tp0/pp0/cp0"


def test_parallel_env_reads_mock_megatron_parallel_state():
    parallel_state = SimpleNamespace(
        get_data_parallel_rank=lambda: 1,
        get_data_parallel_world_size=lambda: 4,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_pipeline_model_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_world_size=lambda: 1,
        get_virtual_pipeline_model_parallel_rank=lambda: None,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_world_size=lambda: 1,
        get_expert_model_parallel_rank=lambda: 0,
        get_expert_model_parallel_world_size=lambda: 1,
    )

    env = current_parallel_env(
        model_config={"sequence_parallel": True},
        parallel_state=parallel_state,
    )

    assert env.dp_rank == 1
    assert env.dp_world_size == 4
    assert env.is_data_parallel
    assert env.sequence_parallel is True
    assert env.trace_prefix() == "dp1/tp0/pp0/cp0"


def test_parallel_env_uses_defaults_when_parallel_state_is_not_initialized():
    parallel_state = SimpleNamespace(
        get_data_parallel_rank=lambda: (_ for _ in ()).throw(RuntimeError("not initialized")),
        get_data_parallel_world_size=lambda: (_ for _ in ()).throw(RuntimeError("not initialized")),
    )

    env = current_parallel_env(parallel_state=parallel_state)

    assert env.dp_rank == 0
    assert env.dp_world_size == 1
