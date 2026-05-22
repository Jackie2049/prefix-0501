"""Thin verl-facing adapter for prefix-group DP balancing."""

from __future__ import annotations

from typing import Any

from prefix_sharing.core.group_partition import PrefixGroupPartition, partition_prefix_groups


def prefix_sharing_dp_balance_enabled(config: Any) -> bool:
    """Return whether prefix-group DP balancing is enabled in a verl config."""

    candidates = [
        _read_path(config, ("actor_rollout_ref", "actor", "prefix_sharing", "dp_balance", "enabled")),
        _read_path(config, ("prefix_sharing", "dp_balance", "enabled")),
    ]
    return any(bool(value) for value in candidates if value is not None)


def prefix_sharing_dp_balance_group_key(config: Any, default: str = "uid") -> str:
    value = _read_path(config, ("actor_rollout_ref", "actor", "prefix_sharing", "dp_balance", "group_key"))
    if value is None:
        value = _read_path(config, ("prefix_sharing", "dp_balance", "group_key"))
    return str(value or default)


def reorder_dataproto_for_prefix_group_dp_balance(
    batch: Any,
    dp_size: int,
    *,
    group_key: str = "uid",
    metric_prefix: str = "prefix_dp_balance",
) -> dict[str, Any]:
    """Reorder a DataProto-like object by prefix group for DP dispatch."""

    if group_key not in getattr(batch, "non_tensor_batch", {}):
        return _fallback_metrics(dp_size, "missing_group_key", metric_prefix)

    tensor_batch = getattr(batch, "batch", {})
    if "input_ids" not in tensor_batch or "attention_mask" not in tensor_batch:
        return _fallback_metrics(dp_size, "missing_token_fields", metric_prefix)

    partition = partition_prefix_groups(
        tensor_batch["input_ids"],
        tensor_batch["attention_mask"],
        batch.non_tensor_batch[group_key],
        dp_size,
    )
    if not partition.is_fallback:
        batch.reorder(_reorder_arg([index for indices in partition.dp_rank_to_indices for index in indices]))
    return _metrics(partition, metric_prefix)


def _empty_fallback_partition(dp_size: int, reason: str) -> PrefixGroupPartition:
    empty = tuple(() for _ in range(max(dp_size, 0)))
    return PrefixGroupPartition(
        dp_rank_to_indices=empty,
        dp_rank_to_group_ids=empty,
        group_workloads={},
        fallback_reason=reason,
    )


def _fallback_metrics(dp_size: int, reason: str, metric_prefix: str) -> dict[str, Any]:
    return _metrics(_empty_fallback_partition(dp_size, reason), metric_prefix)


def _metrics(partition: PrefixGroupPartition, prefix: str) -> dict[str, Any]:
    rank_workloads = [
        sum(partition.group_workloads[group_id] for group_id in group_ids)
        for group_ids in partition.dp_rank_to_group_ids
    ]
    non_empty_workloads = [workload for workload in rank_workloads if workload > 0]
    min_workload = min(non_empty_workloads) if non_empty_workloads else 0
    max_workload = max(non_empty_workloads) if non_empty_workloads else 0
    group_sizes = list(partition.group_sizes.values())
    return {
        f"{prefix}/enabled": 0 if partition.is_fallback else 1,
        f"{prefix}/fallback_reason": partition.fallback_reason or "none",
        f"{prefix}/group_count": len(partition.group_workloads),
        f"{prefix}/group_min_size": min(group_sizes) if group_sizes else 0,
        f"{prefix}/group_max_size": max(group_sizes) if group_sizes else 0,
        f"{prefix}/dp_min_workload": min_workload,
        f"{prefix}/dp_max_workload": max_workload,
        f"{prefix}/dp_imbalance_ratio": (max_workload / min_workload) if min_workload else 0,
        f"{prefix}/original_tokens": partition.original_tokens,
        f"{prefix}/compute_tokens": partition.compute_tokens,
        f"{prefix}/reusable_prefix_tokens": partition.reusable_prefix_tokens,
    }


def _read_path(root: Any, path: tuple[str, ...]) -> Any:
    value = root
    for key in path:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
    return value


def _reorder_arg(indices: list[int]) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        return indices
    return torch.tensor(indices)
