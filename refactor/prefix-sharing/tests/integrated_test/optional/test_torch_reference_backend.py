import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.cache import PrefixKVCache, PrefixKVCacheKey
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.logprob import (
    compute_token_logprobs_from_logits,
    gather_provider_prefix_last_logits,
    restore_prefix_last_logprobs_tensor,
)
from prefix_sharing.core.planner import PrefixSharingPlanner


def test_torch_reference_backend_supports_q_len_different_from_kv_len():
    meta = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=2)).plan(
        [[1, 2, 10], [1, 2, 20, 21]],
        forward_id=1,
        micro_batch_id=1,
    )
    backend = TorchReferenceBackend()
    cache = PrefixKVCache()
    query = torch.randn(sum(meta.kept_lengths_q), 4)
    key = torch.randn(sum(meta.kept_lengths_q), 4)
    value = torch.randn(sum(meta.kept_lengths_q), 4)

    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        cache,
        meta,
        layer_id=0,
    )
    output = backend.attention(query, expanded_key, expanded_value, meta)

    assert output.shape == query.shape
    assert expanded_key.shape[0] == sum(meta.expanded_lengths_kv)


def test_torch_reference_backend_caches_expanded_reuser_for_later_reuse():
    meta = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=2)).plan(
        [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        forward_id=2,
        micro_batch_id=1,
    )
    assert [(s.reuse_batch_index, s.provider_batch_index, s.prefix_len) for s in meta.reuse_specs] == [
        (1, 0, 3),
        (2, 1, 4),
    ]

    backend = TorchReferenceBackend()
    cache = PrefixKVCache()
    key = torch.arange(sum(meta.kept_lengths_q) * 2, dtype=torch.float32).reshape(-1, 2)
    value = key + 100

    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        cache,
        meta,
        layer_id=0,
    )

    reuser_cache_key = PrefixKVCacheKey(meta.forward_id, meta.micro_batch_id, 0, 1, 0)
    assert cache.load(reuser_cache_key).key.shape[0] == 4
    assert expanded_key.shape[0] == sum(meta.expanded_lengths_kv)
    assert expanded_value.shape[0] == sum(meta.expanded_lengths_kv)


def test_prefix_last_restore_tensor_keeps_autograd_path():
    meta = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=2)).plan(
        [[1, 2, 10], [1, 2, 20, 21]],
        forward_id=1,
        micro_batch_id=1,
    )
    logits = torch.randn(2, 4, 8, requires_grad=True)
    restored_logits = gather_provider_prefix_last_logits(logits, meta)
    labels = torch.tensor([0, 3])
    first_suffix_logprobs = compute_token_logprobs_from_logits(restored_logits, labels)
    suffix_logprobs = torch.randn(2, 4, requires_grad=True)

    restored = restore_prefix_last_logprobs_tensor(suffix_logprobs, first_suffix_logprobs, meta)
    loss = restored[1, 0] + restored[1, 1]
    loss.backward()

    assert logits.grad is not None
    assert suffix_logprobs.grad is not None
    assert logits.grad[0, 1].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available locally")
def test_torch_reference_backend_cuda_optional_smoke():
    meta = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=2)).plan(
        [[1, 2, 10], [1, 2, 20]],
        forward_id=1,
        micro_batch_id=1,
    )
    backend = TorchReferenceBackend()
    query = torch.randn(sum(meta.kept_lengths_q), 4, device="cuda")
    key = torch.randn(sum(meta.kept_lengths_q), 4, device="cuda")
    value = torch.randn(sum(meta.kept_lengths_q), 4, device="cuda")
    expanded_key, expanded_value = backend.build_kv(
        key,
        value,
        PrefixKVCache(),
        meta,
        layer_id=0,
    )
    assert backend.attention(query, expanded_key, expanded_value, meta).is_cuda
