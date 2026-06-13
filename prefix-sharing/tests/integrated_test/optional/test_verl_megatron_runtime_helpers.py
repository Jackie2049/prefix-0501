import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.integrations.context import current_prefix_sharing_context, prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_mcore import (
    build_prefix_sharing_micro_batch,
    restore_reuser_prefix_columns_2d,
)


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
    layout = prefix_sharing_runtime_state.packed_batch_layout
    assert layout.valid_lengths == [5, 2]
    assert layout.padded_lengths == [5, 2]
    assert layout.cu_seqlens == [0, 5, 7]
    assert layout.packed_position_ids.tolist() == [0, 1, 2, 3, 4, 3, 4]

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 2
        assert ctx.prefix_last_restore_indices[0].reuse_1d_pos == -1  # sentinel
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
    parallel_state = ModuleType("megatron.core.parallel_state")
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size
    parallel_state.get_context_parallel_world_size = lambda: 1
    core = ModuleType("megatron.core")
    core.parallel_state = parallel_state
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)

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

    layout = prefix_sharing_runtime_state.packed_batch_layout
    assert layout.valid_lengths == [5, 2]
    assert layout.padded_lengths == expected_padded_lengths
    assert layout.cu_seqlens == expected_cu_seqlens
    assert layout.packed_position_ids.tolist() == expected_positions
    assert layout.valid_token_mask.tolist() == expected_mask
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 2
        assert ctx.prefix_last_restore_indices[0].reuse_1d_pos == -1  # sentinel


def test_restore_reuser_prefix_columns_2d_prefix_last_keeps_autograd():
    # prefix-last-only case (no interior response).
    # input: [[1,2,3,10,11], [1,2,3,20,21]] → prefix_len=3, prompt_len=3
    # Only prefix-last spec: provider_prefix_last_pos=2, target_2d_pos=2.
    # Reuser's first suffix token at target_2d_pos=2 differs from provider's,
    # so logprob must be recomputed from saved provider logits.
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

    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert len(ctx.prefix_last_restore_indices) == 1
        index = ctx.prefix_last_restore_indices[0]
        assert not index.is_interior_response

        # Simulate 2D postprocess: output dict with [B, L] log_probs.
        log_probs_2d = torch.zeros(2, 5)
        output = {"log_probs": log_probs_2d}

        # 2D label: label[p] = token at p+1 (verl convention).
        label_2d = torch.tensor([[0, 0, 0, 10, 11], [0, 0, 0, 20, 21]])

        # Saved provider packed logits for prefix-last recompute.
        saved_logits = torch.randn(1, 32, requires_grad=True)
        ctx.prefix_last_logits_saved[(index.reuse_idx_in_batch, index.target_2d_pos)] = saved_logits

        output = restore_reuser_prefix_columns_2d(output, label_2d, gather_fn)
        assert ctx.stats.actual_restore_count == ctx.stats.expected_restore_count == 1

        restored_val = output["log_probs"][index.reuse_idx_in_batch, index.target_2d_pos]

    # Gradient must flow through saved_logits.
    restored_val.backward()
    assert saved_logits.grad is not None
    assert saved_logits.grad.abs().sum() > 0
