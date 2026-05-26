import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.core.prefix_store import PrefixKVSlotId, PrefixKVStore


def test_prefix_kv_store_lifecycle_and_isolation():
    store = PrefixKVStore()
    slot_id = PrefixKVSlotId(1, 2, 3, 4, 5)
    key_tensor = torch.randn(7, 2, 4, requires_grad=True)
    value_tensor = torch.randn(7, 2, 4, requires_grad=True)

    store.store(slot_id, key_tensor=key_tensor, value_tensor=value_tensor, prefix_len=7)
    entry = store.load(slot_id)

    assert entry.key_tensor is key_tensor
    assert entry.value_tensor is value_tensor
    assert entry.prefix_len == 7
    assert entry.key_tensor.requires_grad
    assert entry.value_tensor.requires_grad

    with pytest.raises(KeyError):
        store.store(
            slot_id,
            key_tensor=torch.zeros_like(key_tensor),
            value_tensor=torch.zeros_like(value_tensor),
            prefix_len=7,
        )

    other_micro_batch = PrefixKVSlotId(1, 99, 3, 4, 5)
    assert not store.contains(other_micro_batch)
    store.close()
    assert store.closed
    with pytest.raises(RuntimeError):
        store.load(slot_id)
