import pytest

from prefix_sharing.core.prefix_store import PrefixKVSlotId, PrefixKVStore


def test_prefix_kv_store_lifecycle_and_isolation():
    store = PrefixKVStore()
    key = PrefixKVSlotId(1, 2, 3, 4, 5)
    store.store(key, key_tensor="k", value_tensor="v", prefix_len=7)
    entry = store.load(key)
    assert entry.key_tensor == "k"
    assert entry.value_tensor == "v"
    assert entry.prefix_len == 7

    with pytest.raises(KeyError):
        store.store(key, key_tensor="k2", value_tensor="v2", prefix_len=7)

    other_micro_batch = PrefixKVSlotId(1, 99, 3, 4, 5)
    assert not store.contains(other_micro_batch)
    store.close()
    assert store.closed
    with pytest.raises(RuntimeError):
        store.load(key)
