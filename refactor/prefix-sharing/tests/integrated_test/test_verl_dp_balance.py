from types import SimpleNamespace
from pathlib import Path

from prefix_sharing.integrations.verl_dp_balance import (
    prefix_sharing_dp_balance_enabled,
    reorder_dataproto_for_prefix_group_dp_balance,
)


class FakeDataProto:
    def __init__(self):
        self.batch = {
            "input_ids": [
                [1, 2, 3, 4],
                [1, 2, 3, 5],
                [10, 11],
                [30, 31],
            ],
            "attention_mask": [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1],
                [1, 1],
            ],
            "sample": ["a0", "a1", "b0", "c0"],
        }
        self.non_tensor_batch = {
            "uid": ["a", "a", "b", "c"],
        }
        self.reorder_indices = None

    def reorder(self, indices):
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        self.reorder_indices = list(indices)
        for key, value in self.batch.items():
            self.batch[key] = [value[index] for index in indices]
        for key, value in self.non_tensor_batch.items():
            self.non_tensor_batch[key] = [value[index] for index in indices]


def test_reorder_dataproto_for_prefix_group_dp_balance_keeps_uid_locality():
    data = FakeDataProto()
    original_samples = list(data.batch["sample"])

    metrics = reorder_dataproto_for_prefix_group_dp_balance(data, dp_size=2)

    assert metrics["prefix_dp_balance/enabled"] == 1
    assert metrics["prefix_dp_balance/fallback_reason"] == "none"
    assert data.reorder_indices is not None
    first_rank_uids = set(data.non_tensor_batch["uid"][:2])
    second_rank_uids = set(data.non_tensor_batch["uid"][2:])
    assert first_rank_uids in ({"a"}, {"b", "c"})
    assert second_rank_uids in ({"a"}, {"b", "c"})
    assert data.batch["sample"] == [original_samples[index] for index in data.reorder_indices]


def test_reorder_dataproto_for_prefix_group_dp_balance_fallback_does_not_reorder():
    data = FakeDataProto()
    data.non_tensor_batch.pop("uid")

    metrics = reorder_dataproto_for_prefix_group_dp_balance(data, dp_size=2)

    assert metrics["prefix_dp_balance/enabled"] == 0
    assert metrics["prefix_dp_balance/fallback_reason"] == "missing_group_key"
    assert data.reorder_indices is None


def test_prefix_sharing_dp_balance_enabled_reads_nested_actor_config():
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(
                prefix_sharing={
                    "dp_balance": {
                        "enabled": True,
                    }
                }
            )
        )
    )

    assert prefix_sharing_dp_balance_enabled(config) is True


def test_verl_ray_trainer_has_thin_prefix_dp_balance_hook():
    repo_root = Path(__file__).resolve().parents[4]
    ray_trainer = repo_root / "dependency" / "verl_v070" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    source = ray_trainer.read_text()

    assert "reorder_dataproto_for_prefix_group_dp_balance" in source
    assert "prefix_sharing_dp_balance_enabled(self.config)" in source
    assert "prefix_dp_balance/enabled" in source
