import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.integrations.context import current_prefix_sharing_context, prefix_sharing_runtime_context
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
    layout = prefix_sharing_runtime_state.batch_runtime_layout
    assert layout.layout_kind == "thd"
    assert layout.valid_lengths == [5, 2]
    assert layout.padded_lengths == [5, 2]
    assert layout.cu_seqlens == [0, 5, 7]
    assert layout.position_ids.tolist() == [0, 1, 2, 3, 4, 3, 4]

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.prefix_last_restore_indices[0].provider_token_index == 2
        assert ctx.prefix_last_restore_indices[0].reuse_token_index == 5
    assert current_prefix_sharing_context() is None


def test_build_prefix_sharing_micro_batch_builds_bshd_layout_without_remove_padding():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": False},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )

    trimmed_micro_batch, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    assert prefix_sharing_runtime_state is not None
    assert trimmed_micro_batch["attention_mask"].tolist() == [
        [True, True, True, True, True],
        [False, False, False, True, True],
    ]
    layout = prefix_sharing_runtime_state.batch_runtime_layout
    assert layout.layout_kind == "bshd"
    assert layout.valid_lengths == [5, 2]
    assert layout.max_seqlen == 5
    assert layout.valid_token_mask.tolist() == trimmed_micro_batch["attention_mask"].tolist()
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        provider_index = ctx.prefix_last_restore_indices[0].provider_token_index
        reuse_index = ctx.prefix_last_restore_indices[0].reuse_token_index
        assert (provider_index.row, provider_index.seq_pos) == (0, 2)
        assert (reuse_index.row, reuse_index.seq_pos) == (1, 3)


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
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )

    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    layout = prefix_sharing_runtime_state.batch_runtime_layout
    assert layout.layout_kind == "thd"
    assert layout.valid_lengths == [5, 2]
    assert layout.padded_lengths == expected_padded_lengths
    assert layout.cu_seqlens == expected_cu_seqlens
    assert layout.position_ids.tolist() == expected_positions
    assert layout.valid_token_mask.tolist() == expected_mask
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert ctx.prefix_last_restore_indices[0].provider_token_index == 2
        assert ctx.prefix_last_restore_indices[0].reuse_token_index == expected_cu_seqlens[1]


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
    assert prefix_sharing_runtime_state.batch_runtime_layout.cu_seqlens == [0, 5, 7]


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
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )

    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    assert prefix_sharing_runtime_state.parallel_info.tp_size == tp_size
    assert prefix_sharing_runtime_state.parallel_info.pp_size == pp_size
    assert prefix_sharing_runtime_state.batch_runtime_layout.cu_seqlens == expected_cu_seqlens
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert ctx.prefix_last_restore_indices[0].reuse_token_index == expected_cu_seqlens[1]


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

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state):
        restored = restore_suffix_first_log_probs_from_prefix(logits, labels, log_probs, gather_fn)

    restored[0, 5].backward()
    assert logits.grad is not None
    assert logits.grad[0, 2].abs().sum() > 0


def test_restore_suffix_first_log_probs_from_prefix_supports_bshd_dense_indices():
    logits = torch.randn(2, 5, 8, requires_grad=True)
    labels = torch.tensor([[0, 0, 0, 10, 11], [0, 0, 0, 3, 4]])
    log_probs = torch.zeros(2, 5, requires_grad=True)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": False},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
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

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state):
        restored = restore_suffix_first_log_probs_from_prefix(logits, labels, log_probs, gather_fn)

    restored[1, 3].backward()
    assert logits.grad is not None
    assert logits.grad[0, 2].abs().sum() > 0


def test_restore_suffix_first_log_probs_from_prefix_supports_bshd_kept_padded_sbh_indices():
    logits = torch.randn(5, 2, 8, requires_grad=True)
    labels = torch.tensor(
        [
            [0, 3],
            [0, 4],
            [0, 0],
            [10, 0],
            [11, 0],
        ]
    )
    log_probs = torch.zeros(5, 2, requires_grad=True)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": False},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )
    _, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(batch, actor_config, model_config)

    def gather_fn(provider_logits, reuse_label):
        assert provider_logits.shape == (1, 1, 8)
        assert reuse_label.shape == (1, 1)
        return torch.gather(
            torch.log_softmax(provider_logits, dim=-1),
            dim=-1,
            index=reuse_label.unsqueeze(-1),
        ).squeeze(-1)

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state):
        restored = restore_suffix_first_log_probs_from_prefix(logits, labels, log_probs, gather_fn)

    restored[0, 1].backward()
    assert logits.grad is not None
    assert logits.grad[2, 0].abs().sum() > 0


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
