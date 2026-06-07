import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.integrations.context import current_prefix_sharing_context, prefix_sharing_runtime_context
from prefix_sharing.integrations.megatron_runtime import maybe_run_prefix_sharing_attention
from prefix_sharing.integrations.verl_mcore import (
    build_prefix_sharing_micro_batch,
    restore_suffix_first_log_probs_from_prefix,
)


def _install_megatron_parallel_state(
    monkeypatch,
    *,
    tp_size=1,
    tp_rank=0,
    pp_size=1,
    pp_rank=0,
    cp_size=1,
    cp_rank=0,
    virtual_pp_size=None,
):
    parallel_state = ModuleType("megatron.core.parallel_state")
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size
    parallel_state.get_tensor_model_parallel_rank = lambda: tp_rank
    parallel_state.get_context_parallel_world_size = lambda: cp_size
    parallel_state.get_context_parallel_rank = lambda: cp_rank
    parallel_state.get_pipeline_model_parallel_world_size = lambda: pp_size
    parallel_state.get_pipeline_model_parallel_rank = lambda: pp_rank
    parallel_state.is_pipeline_first_stage = lambda ignore_virtual=True: pp_rank == 0
    parallel_state.is_pipeline_last_stage = lambda ignore_virtual=True: pp_rank == pp_size - 1
    parallel_state.get_virtual_pipeline_model_parallel_world_size = lambda: virtual_pp_size
    core = ModuleType("megatron.core")
    core.parallel_state = parallel_state
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)


def test_build_prefix_sharing_micro_batch_trims_reuser_mask_and_context_positions():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )

    trimmed_micro_batch, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    assert prefix_sharing_runtime_state is not None
    assert prefix_sharing_runtime_state.prefix_sharing_plan.has_sharing
    assert trimmed_micro_batch["attention_mask"].tolist() == [
        [True, True, True, True, True],
        [False, False, False, True, True],
    ]
    layout = prefix_sharing_runtime_state.packed_batch_layout
    assert layout.valid_lengths == [5, 2]
    assert layout.padded_lengths == [5, 2]
    assert layout.cu_seqlens == [0, 5, 7]
    assert layout.packed_position_ids.tolist() == [0, 1, 2, 3, 4, 3, 4]

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 2
        assert ctx.prefix_last_restore_indices[0].reuse_1d_pos == 5
    assert current_prefix_sharing_context() is None


@pytest.mark.parametrize(
    ("tp_size", "expected_padded_lengths", "expected_cu_seqlens", "expected_positions", "expected_mask"),
    [
        (
            2,
            [6, 2],
            [0, 6, 8],
            [0, 1, 2, 3, 4, 0, 3, 4],
            [True, True, True, True, True, False, True, True],
        ),
        (
            4,
            [8, 4],
            [0, 8, 12],
            [0, 1, 2, 3, 4, 0, 0, 0, 3, 4, 0, 0],
            [True, True, True, True, True, False, False, False, True, True, False, False],
        ),
        (
            8,
            [8, 8],
            [0, 8, 16],
            [0, 1, 2, 3, 4, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0],
            [True, True, True, True, True, False, False, False, True, True, False, False, False, False, False, False],
        ),
    ],
)
def test_build_prefix_sharing_micro_batch_builds_common_tp_padded_layouts(
    monkeypatch,
    tp_size,
    expected_padded_lengths,
    expected_cu_seqlens,
    expected_positions,
    expected_mask,
):
    _install_megatron_parallel_state(monkeypatch, tp_size=tp_size)

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=False,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )

    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    layout = prefix_sharing_runtime_state.packed_batch_layout
    assert layout.valid_lengths == [5, 2]
    assert layout.padded_lengths == expected_padded_lengths
    assert layout.cu_seqlens == expected_cu_seqlens
    assert layout.packed_position_ids.tolist() == expected_positions
    assert layout.valid_token_mask.tolist() == expected_mask
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 2
        assert ctx.prefix_last_restore_indices[0].reuse_1d_pos == expected_cu_seqlens[1]


@pytest.mark.parametrize(
    ("tp_size", "expected_padded_lengths", "expected_cu_seqlens"),
    [
        (2, [6, 2], [0, 6, 8]),
        (4, [8, 4], [0, 8, 12]),
        (8, [8, 8], [0, 8, 16]),
    ],
)
def test_build_prefix_sharing_micro_batch_keeps_global_layout_with_sequence_parallel(
    monkeypatch,
    tp_size,
    expected_padded_lengths,
    expected_cu_seqlens,
):
    _install_megatron_parallel_state(monkeypatch, tp_size=tp_size)

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True, "sequence_parallel": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=True,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )

    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    layout = prefix_sharing_runtime_state.packed_batch_layout
    assert layout.valid_lengths == [5, 2]
    assert layout.padded_lengths == expected_padded_lengths
    assert layout.cu_seqlens == expected_cu_seqlens
    assert layout.total_padded_length == expected_cu_seqlens[-1]
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 2
        assert ctx.prefix_last_restore_indices[0].reuse_1d_pos == expected_cu_seqlens[1]


@pytest.mark.parametrize("pp_size", [2, 4, 8])
def test_build_prefix_sharing_micro_batch_records_physical_pipeline_parallel_info(monkeypatch, pp_size):
    pp_rank = pp_size - 1
    _install_megatron_parallel_state(monkeypatch, pp_size=pp_size, pp_rank=pp_rank)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=None,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )

    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    parallel_info = prefix_sharing_runtime_state.parallel_info
    assert parallel_info.pp_size == pp_size
    assert parallel_info.pp_rank == pp_rank
    assert parallel_info.is_pipeline_first_stage is False
    assert parallel_info.is_pipeline_last_stage is True
    assert prefix_sharing_runtime_state.packed_batch_layout.cu_seqlens == [0, 5, 7]


@pytest.mark.parametrize(
    ("tp_size", "pp_size", "expected_cu_seqlens"),
    [
        (2, 2, [0, 6, 8]),
        (2, 4, [0, 6, 8]),
    ],
)
def test_build_prefix_sharing_micro_batch_combines_tp_padding_with_physical_pp(
    monkeypatch,
    tp_size,
    pp_size,
    expected_cu_seqlens,
):
    _install_megatron_parallel_state(monkeypatch, tp_size=tp_size, pp_size=pp_size, pp_rank=1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=None,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=False,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )

    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    assert prefix_sharing_runtime_state.parallel_info.tp_size == tp_size
    assert prefix_sharing_runtime_state.parallel_info.pp_size == pp_size
    assert prefix_sharing_runtime_state.packed_batch_layout.cu_seqlens == expected_cu_seqlens
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert ctx.prefix_last_restore_indices[0].reuse_1d_pos == expected_cu_seqlens[1]


def test_restore_suffix_first_log_probs_from_prefix_keeps_provider_autograd_path():
    # THD compact format: [1, total_kept_tokens, V]
    # row0 (provider) all 5 tokens, row1 (reuser) suffix 2 tokens (positions 3,4)
    # total = 5 + 2 = 7, align_size=1 so no padding
    logits = torch.randn(1, 7, 8, requires_grad=True)
    labels = torch.tensor([[0, 0, 0, 10, 11, 3, 4]])
    log_probs = torch.zeros(1, 7, requires_grad=True)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )
    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    def gather_fn(provider_logits, reuse_label):
        return torch.gather(
            torch.log_softmax(provider_logits, dim=-1),
            dim=-1,
            index=reuse_label.unsqueeze(-1),
        ).squeeze(-1)

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        restored = restore_suffix_first_log_probs_from_prefix(logits, labels, log_probs, gather_fn)
        assert ctx.stats.actual_restore_count == ctx.stats.expected_restore_count == 1

    restored[0, 5].backward()
    assert logits.grad is not None
    assert logits.grad[0, 2].abs().sum() > 0


def test_restore_suffix_first_log_probs_from_prefix_noops_on_non_last_pp_stage(monkeypatch):
    _install_megatron_parallel_state(monkeypatch, pp_size=2, pp_rank=0)
    logits = torch.randn(1, 7, 8, requires_grad=True)
    labels = torch.tensor([[0, 0, 0, 10, 11, 3, 4]])
    log_probs = torch.zeros(1, 7, requires_grad=True)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )
    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    def gather_fn(provider_logits, reuse_label):
        raise AssertionError("non-last PP stage must not run prefix-last restore")

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state):
        restored = restore_suffix_first_log_probs_from_prefix(logits, labels, log_probs, gather_fn)

    assert restored is log_probs


def test_attention_hook_rejects_sp_local_shard_token_length():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=True,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )
    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)
    attention_module = SimpleNamespace(config=SimpleNamespace(sequence_parallel=True), layer_number=1)
    packed_seq_params = SimpleNamespace(qkv_format="thd")
    query = torch.randn(6, 1, 2)
    key = torch.randn(6, 1, 2)
    value = torch.randn(6, 1, 2)

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state):
        with pytest.raises(RuntimeError, match="SP-local shard"):
            maybe_run_prefix_sharing_attention(
                attention_module,
                query,
                key,
                value,
                attention_mask=None,
                rotary_pos_emb=(object(), object()),
                packed_seq_params=packed_seq_params,
            )


def test_restore_suffix_first_log_probs_rejects_sp_local_shard_token_length():
    logits = torch.randn(1, 6, 8, requires_grad=True)
    labels = torch.tensor([[0, 0, 0, 10, 11, 3, 4]])
    log_probs = torch.zeros(1, 7, requires_grad=True)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=True,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )
    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    def gather_fn(provider_logits, reuse_label):
        raise AssertionError("restore should reject local shard token lengths before gather")

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state):
        with pytest.raises(RuntimeError, match="SP-local shard"):
            restore_suffix_first_log_probs_from_prefix(logits, labels, log_probs, gather_fn)
