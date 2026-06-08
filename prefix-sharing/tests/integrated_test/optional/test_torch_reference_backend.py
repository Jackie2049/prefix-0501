import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.batch_layout import BshdBatchLayout, ThdBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.logprob import (
    compute_token_logprobs_from_logits,
    gather_provider_prefix_last_logits,
    restore_prefix_last_logprobs_tensor,
)
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PREFIX_STATE_TYPE_DELTANET_STATE,
    PrefixActivationSlotId,
    PrefixAttentionStore,
    PrefixDeltanetStore,
)


def test_torch_reference_backend_supports_q_len_different_from_kv_len():
    prefix_sharing_plan = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)).plan(
        [[1, 2, 10], [1, 2, 20, 21]],
        forward_id=1,
        micro_batch_id=1,
    )
    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()
    query = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 4)
    key = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 4)
    value = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 4)

    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        store,
        prefix_sharing_plan,
        layer_id=0,
    )
    output = backend.attention(query, expanded_key, expanded_value, prefix_sharing_plan)

    assert output.shape == query.shape
    assert expanded_key.shape[0] == sum(prefix_sharing_plan.expanded_lengths_kv)


def test_torch_reference_backend_supports_thd_grouped_query_attention():
    prefix_sharing_plan = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)).plan(
        [[1, 2, 10], [1, 2, 20, 21]],
        forward_id=1,
        micro_batch_id=1,
    )
    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()
    query = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 4, 8)
    key = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 2, 8)
    value = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 2, 8)

    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        store,
        prefix_sharing_plan,
        layer_id=0,
    )
    output = backend.attention(query, expanded_key, expanded_value, prefix_sharing_plan)

    assert output.shape == query.shape
    assert expanded_key.shape[1] == 2


def test_torch_reference_backend_caches_expanded_reuser_for_later_reuse():
    prefix_sharing_plan = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)).plan(
        [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        forward_id=2,
        micro_batch_id=1,
    )
    assert [(s.reuse_idx_in_batch, s.provider_idx_in_batch, s.prefix_len) for s in prefix_sharing_plan.reuse_specs] == [
        (1, 0, 3),
        (2, 1, 4),
    ]

    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()
    key = torch.arange(sum(prefix_sharing_plan.kept_lengths_q) * 2, dtype=torch.float32).reshape(-1, 2)
    value = key + 100

    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        store,
        prefix_sharing_plan,
        layer_id=0,
    )

    reuser_slot_id = PrefixActivationSlotId(
        prefix_sharing_plan.forward_id,
        prefix_sharing_plan.micro_batch_id,
        0,
        1,
        PREFIX_STATE_TYPE_ATTENTION_KV,
        0,
    )
    assert store.load(reuser_slot_id).key_tensor.shape[0] == 4
    assert expanded_key.shape[0] == sum(prefix_sharing_plan.expanded_lengths_kv)
    assert expanded_value.shape[0] == sum(prefix_sharing_plan.expanded_lengths_kv)


@pytest.mark.parametrize(
    ("tp_size", "padded_lengths", "cu_seqlens", "padding_indices"),
    [
        (2, [6, 2], [0, 6, 8], [5]),
        (4, [8, 4], [0, 8, 12], [5, 6, 7, 10, 11]),
        (8, [8, 8], [0, 8, 16], [5, 6, 7, 10, 11, 12, 13, 14, 15]),
    ],
)
def test_torch_reference_backend_uses_padded_layout_without_storing_padding_kv(
    tp_size,
    padded_lengths,
    cu_seqlens,
    padding_indices,
):
    prefix_sharing_plan = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)).plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]],
        forward_id=3,
        micro_batch_id=1,
    )
    layout = ThdBatchLayout(
        valid_lengths=[5, 2],
        padded_lengths=padded_lengths,
        cu_seqlens=cu_seqlens,
        max_seqlen=max(padded_lengths),
    )
    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()
    query = torch.randn(layout.total_padded_length, 2)
    key = torch.arange(layout.total_padded_length * 2, dtype=torch.float32).reshape(-1, 2)
    value = key + 100

    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        store,
        prefix_sharing_plan,
        batch_runtime_layout=layout,
        layer_id=0,
    )
    output = backend.attention(
        query,
        expanded_key,
        expanded_value,
        prefix_sharing_plan,
        batch_runtime_layout=layout,
    )

    provider_slot_id = PrefixActivationSlotId(
        prefix_sharing_plan.forward_id,
        prefix_sharing_plan.micro_batch_id,
        0,
        0,
        PREFIX_STATE_TYPE_ATTENTION_KV,
        0,
    )
    reuser_slot_id = PrefixActivationSlotId(
        prefix_sharing_plan.forward_id,
        prefix_sharing_plan.micro_batch_id,
        0,
        1,
        PREFIX_STATE_TYPE_ATTENTION_KV,
        0,
    )
    assert store.load(provider_slot_id).key_tensor.shape[0] == 5
    assert store.load(reuser_slot_id).key_tensor.shape[0] == 5
    assert expanded_key.shape[0] == sum(prefix_sharing_plan.expanded_lengths_kv)
    assert expanded_value.shape[0] == sum(prefix_sharing_plan.expanded_lengths_kv)
    assert output.shape == query.shape
    for padding_index in padding_indices:
        assert output[padding_index].abs().sum().item() == 0


def test_torch_reference_backend_bshd_attention_matches_thd_valid_tokens():
    prefix_sharing_plan = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)).plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]],
        forward_id=30,
        micro_batch_id=1,
    )
    bshd_layout = BshdBatchLayout.from_valid_token_mask(
        torch.tensor(
            [
                [True, True, True, True, True, False],
                [False, False, False, True, True, False],
            ]
        )
    )
    thd_layout = ThdBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
    backend = TorchReferenceBackend()
    query_bshd = torch.randn(2, 6, 3)
    key_bshd = torch.randn(2, 6, 3)
    value_bshd = torch.randn(2, 6, 3)
    query_thd = torch.cat([bshd_layout.valid_row(query_bshd, 0), bshd_layout.valid_row(query_bshd, 1)], dim=0)
    key_thd = torch.cat([bshd_layout.valid_row(key_bshd, 0), bshd_layout.valid_row(key_bshd, 1)], dim=0)
    value_thd = torch.cat([bshd_layout.valid_row(value_bshd, 0), bshd_layout.valid_row(value_bshd, 1)], dim=0)

    bshd_store = PrefixAttentionStore()
    bshd_key, bshd_value = backend.build_kv(
        key_bshd,
        value_bshd,
        bshd_store,
        prefix_sharing_plan,
        batch_runtime_layout=bshd_layout,
        layer_id=0,
    )
    bshd_output = backend.attention(
        query_bshd,
        bshd_key,
        bshd_value,
        prefix_sharing_plan,
        batch_runtime_layout=bshd_layout,
    )

    thd_store = PrefixAttentionStore()
    thd_key, thd_value = backend.build_kv(
        key_thd,
        value_thd,
        thd_store,
        prefix_sharing_plan,
        batch_runtime_layout=thd_layout,
        layer_id=0,
    )
    thd_output = backend.attention(
        query_thd,
        thd_key,
        thd_value,
        prefix_sharing_plan,
        batch_runtime_layout=thd_layout,
    )

    assert bshd_output.shape == query_bshd.shape
    assert torch.allclose(bshd_layout.valid_row(bshd_output, 0), thd_output[:5])
    assert torch.allclose(bshd_layout.valid_row(bshd_output, 1), thd_output[5:])
    assert bshd_output[0, 5].abs().sum().item() == 0
    assert bshd_output[1, :3].abs().sum().item() == 0
    assert bshd_output[1, 5].abs().sum().item() == 0


def test_torch_reference_backend_gated_attention_keeps_gate_on_current_tokens():
    prefix_sharing_plan = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)).plan(
        [[1, 2, 10], [1, 2, 20]],
        forward_id=4,
        micro_batch_id=1,
    )
    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()
    query = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 4)
    key = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 4)
    value = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 4)
    gate = torch.randn_like(query)

    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        store,
        prefix_sharing_plan,
        layer_id=0,
    )
    gated_output = backend.gated_attention(query, expanded_key, expanded_value, gate, prefix_sharing_plan)

    attention_output = backend.attention(query, expanded_key, expanded_value, prefix_sharing_plan)
    assert torch.allclose(gated_output, attention_output * torch.sigmoid(gate))


@pytest.mark.parametrize("tp_rank", [0, 1, 7])
def test_torch_reference_backend_builds_prefix_deltanet_states_with_autograd(tp_rank):
    prefix_sharing_plan = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)).plan(
        [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        forward_id=5,
        micro_batch_id=1,
    )
    backend = TorchReferenceBackend()
    store = PrefixDeltanetStore()
    state_update = torch.arange(
        sum(prefix_sharing_plan.kept_lengths_q) * 3,
        dtype=torch.float32,
    ).reshape(-1, 3)
    state_update.requires_grad_()

    output = backend.build_deltanet_states(
        state_update,
        store,
        prefix_sharing_plan,
        layer_id=2,
        tp_rank=tp_rank,
    )

    provider_slot_id = PrefixActivationSlotId(
        prefix_sharing_plan.forward_id,
        prefix_sharing_plan.micro_batch_id,
        2,
        0,
        PREFIX_STATE_TYPE_DELTANET_STATE,
        tp_rank,
    )
    first_reuser_slot_id = PrefixActivationSlotId(
        prefix_sharing_plan.forward_id,
        prefix_sharing_plan.micro_batch_id,
        2,
        1,
        PREFIX_STATE_TYPE_DELTANET_STATE,
        tp_rank,
    )
    other_tp_slot_id = PrefixActivationSlotId(
        prefix_sharing_plan.forward_id,
        prefix_sharing_plan.micro_batch_id,
        2,
        0,
        PREFIX_STATE_TYPE_DELTANET_STATE,
        tp_rank + 100,
    )
    assert store.load(provider_slot_id).recurrent_state.shape[0] == 3
    assert store.load(first_reuser_slot_id).recurrent_state.shape[0] == 4
    assert not store.contains(other_tp_slot_id)

    provider_prefix_state = state_update[:3].cumsum(dim=0)[-1]
    first_reuser_update = state_update[3]
    second_reuser_update = state_update[4]
    assert torch.allclose(output[3], provider_prefix_state + first_reuser_update)
    assert torch.allclose(output[4], provider_prefix_state + first_reuser_update + second_reuser_update)

    output.sum().backward()
    assert state_update.grad is not None
    assert state_update.grad[:3].abs().sum() > 0


@pytest.mark.parametrize(
    ("tp_size", "padded_lengths", "cu_seqlens"),
    [
        (2, [4, 2], [0, 4, 6]),
        (4, [4, 4], [0, 4, 8]),
        (8, [8, 8], [0, 8, 16]),
    ],
)
def test_torch_reference_backend_deltanet_states_ignore_padding_slots(tp_size, padded_lengths, cu_seqlens):
    prefix_sharing_plan = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)).plan(
        [[1, 2, 10], [1, 2, 20]],
        forward_id=6,
        micro_batch_id=1,
    )
    layout = ThdBatchLayout(
        valid_lengths=[3, 1],
        padded_lengths=padded_lengths,
        cu_seqlens=cu_seqlens,
        max_seqlen=max(padded_lengths),
    )
    backend = TorchReferenceBackend()
    store = PrefixDeltanetStore()
    state_update = torch.ones(layout.total_padded_length, 2, requires_grad=True)

    output = backend.build_deltanet_states(
        state_update,
        store,
        prefix_sharing_plan,
        batch_runtime_layout=layout,
        layer_id=0,
        tp_rank=tp_size,
    )

    assert output.shape == state_update.shape
    assert layout.valid_row(output, 0).abs().sum() > 0
    assert layout.valid_row(output, 1).abs().sum() > 0
    for batch_index, padded_length in enumerate(layout.padded_lengths):
        valid_length = layout.valid_lengths[batch_index]
        if valid_length < padded_length:
            row_start = layout.row_start(batch_index)
            assert output[row_start + valid_length : row_start + padded_length].abs().sum().item() == 0


def test_prefix_last_restore_tensor_keeps_autograd_path():
    prefix_sharing_plan = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)).plan(
        [[1, 2, 10], [1, 2, 20, 21]],
        forward_id=1,
        micro_batch_id=1,
    )
    logits = torch.randn(2, 4, 8, requires_grad=True)
    restored_logits = gather_provider_prefix_last_logits(logits, prefix_sharing_plan)
    labels = torch.tensor([0, 3])
    first_suffix_logprobs = compute_token_logprobs_from_logits(restored_logits, labels)
    suffix_logprobs = torch.randn(2, 4, requires_grad=True)

    restored = restore_prefix_last_logprobs_tensor(suffix_logprobs, first_suffix_logprobs, prefix_sharing_plan)
    loss = restored[1, 0] + restored[1, 1]
    loss.backward()

    assert logits.grad is not None
    assert suffix_logprobs.grad is not None
    assert logits.grad[0, 1].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available locally")
def test_torch_reference_backend_cuda_optional_smoke():
    prefix_sharing_plan = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)).plan(
        [[1, 2, 10], [1, 2, 20]],
        forward_id=1,
        micro_batch_id=1,
    )
    backend = TorchReferenceBackend()
    query = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 4, device="cuda")
    key = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 4, device="cuda")
    value = torch.randn(sum(prefix_sharing_plan.kept_lengths_q), 4, device="cuda")
    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        PrefixAttentionStore(),
        prefix_sharing_plan,
        layer_id=0,
    )
    assert backend.attention(query, expanded_key, expanded_value, prefix_sharing_plan).is_cuda
