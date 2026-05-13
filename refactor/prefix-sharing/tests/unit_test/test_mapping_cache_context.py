import pytest

from prefix_sharing.core.cache import PrefixKVCache, PrefixKVCacheKey
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.mapping import (
    build_provider_prefix_last_values,
    restore_prefix_last_logprobs,
    trim_inputs,
    trim_labels,
)
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context, prefix_sharing_context


def _meta():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enabled=True, min_prefix_len=3))
    return planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=10,
        micro_batch_id=20,
    )


def test_trim_inputs_and_labels_follow_metadata_ranges():
    meta = _meta()
    inputs = [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]]
    labels = [["p1", "p2", "s0", "s1", "eos"], ["p1", "p2", "s0", "s1", "s2", "eos"]]

    trimmed_inputs = trim_inputs(inputs, meta)
    trimmed_labels = trim_labels(labels, meta)

    assert trimmed_inputs.rows == [[1, 2, 3, 10, 11], [20, 21, 22]]
    assert trimmed_inputs.flattened == [1, 2, 3, 10, 11, 20, 21, 22]
    assert trimmed_inputs.cu_seqlens == meta.cu_seqlens_q
    assert trimmed_labels.rows == [["p1", "p2", "s0", "s1", "eos"], ["s1", "s2", "eos"]]


def test_prefix_last_restore_inserts_reuse_first_suffix_logprob():
    meta = _meta()
    suffix_logprobs = [
        [0.10, 0.11, 0.12, 0.13, 0.14],
        [0.21, 0.22],
    ]
    first_suffix_logprobs_by_batch = [0.0, 0.20]

    restored = restore_prefix_last_logprobs(
        suffix_logprobs,
        first_suffix_logprobs_by_batch,
        meta,
    )

    assert restored == [
        [0.10, 0.11, 0.12, 0.13, 0.14],
        [0.20, 0.21, 0.22],
    ]


def test_build_provider_prefix_last_values_uses_restore_specs():
    meta = _meta()
    values = build_provider_prefix_last_values(
        [["p0", "p1", "p_last", "s0"], ["unused"]],
        meta,
    )
    assert values == [None, "p_last"]


def test_prefix_kv_cache_lifecycle_and_isolation():
    cache = PrefixKVCache()
    key = PrefixKVCacheKey(1, 2, 3, 4, 5)
    cache.store(key, key="k", value="v", prefix_len=7)
    entry = cache.load(key)
    assert entry.key == "k"
    assert entry.value == "v"
    assert entry.prefix_len == 7

    with pytest.raises(KeyError):
        cache.store(key, key="k2", value="v2", prefix_len=7)

    other_micro_batch = PrefixKVCacheKey(1, 99, 3, 4, 5)
    assert not cache.contains(other_micro_batch)
    cache.close()
    assert cache.closed
    with pytest.raises(RuntimeError):
        cache.load(key)


def test_prefix_sharing_context_sets_and_clears_current_context():
    meta = _meta()
    assert current_prefix_sharing_context() is None
    with prefix_sharing_context(meta) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.meta is meta
        assert not ctx.cache.closed
    assert current_prefix_sharing_context() is None
    assert ctx.cache.closed
