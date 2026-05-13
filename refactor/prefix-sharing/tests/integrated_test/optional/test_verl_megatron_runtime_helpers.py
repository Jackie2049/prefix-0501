from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.integrations.verl_mcore import (
    megatron_actor_prefix_sharing_context,
    prepare_megatron_actor_micro_batch,
    restore_megatron_actor_log_probs,
)


def test_prepare_megatron_actor_micro_batch_trims_reuser_mask_and_context_positions():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing": {"enabled": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )

    prepared_batch, prepared = prepare_megatron_actor_micro_batch(batch, actor_config, model_config)

    assert prepared is not None
    assert prepared.meta.has_sharing
    assert prepared_batch["attention_mask"].tolist() == [
        [True, True, True, True, True],
        [False, False, False, True, True],
    ]
    assert prepared.kept_position_ids.tolist() == [0, 1, 2, 3, 4, 3, 4]

    with megatron_actor_prefix_sharing_context(prepared) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.restore_positions[0].provider_dense_pos == 2
        assert ctx.restore_positions[0].reuse_dense_pos == 3
    assert current_prefix_sharing_context() is None


def test_restore_megatron_actor_log_probs_keeps_provider_autograd_path():
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
        "prefix_sharing": {"enabled": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )
    _, prepared = prepare_megatron_actor_micro_batch(batch, actor_config, model_config)

    def gather_fn(provider_logits, reuse_label):
        return torch.gather(
            torch.log_softmax(provider_logits, dim=-1),
            dim=-1,
            index=reuse_label.unsqueeze(-1),
        ).squeeze(-1)

    with megatron_actor_prefix_sharing_context(prepared):
        restored = restore_megatron_actor_log_probs(logits, labels, log_probs, gather_fn)

    restored[1, 3].backward()
    assert logits.grad is not None
    assert logits.grad[0, 2].abs().sum() > 0
