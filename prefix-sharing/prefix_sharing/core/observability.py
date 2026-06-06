"""Lightweight observability for prefix-sharing diagnosis.

The goal is not to build a full metrics backend here. This module keeps the
minimal counters needed to answer three operational questions:

1. Did the planner find enough reusable prefix tokens?
2. Did runtime KV assembly actually load those prefixes?
3. Do the expected and actual reuse numbers match?
"""

from __future__ import annotations

from dataclasses import dataclass, field

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.planner import PrefixSharingPlan


@dataclass
class PrefixSharingLayerStats:
    layer_id: int
    store_count: int = 0
    load_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    stored_tokens: int = 0
    loaded_prefix_tokens: int = 0
    expanded_kv_tokens: int = 0
    valid_q_tokens: int = 0
    padded_q_tokens: int = 0


@dataclass
class PrefixSharingStats:
    """Per-runtime-context stats for one prefix-sharing micro-batch."""

    forward_id: int
    micro_batch_id: int
    batch_size: int
    original_tokens: int
    kept_valid_tokens: int
    kept_padded_tokens: int
    saved_valid_tokens: int
    saved_valid_token_ratio: float
    provider_count: int
    reuser_count: int
    sharing_group_count: int
    expected_attention_load_count_per_layer: int
    expected_loaded_prefix_tokens_per_layer: int
    expected_restore_count: int
    actual_restore_count: int = 0
    layers: dict[int, PrefixSharingLayerStats] = field(default_factory=dict)

    @classmethod
    def from_plan(
        cls,
        prefix_sharing_plan: PrefixSharingPlan,
        packed_batch_layout: PackedBatchLayout,
    ) -> "PrefixSharingStats":
        original_tokens = sum(prefix_sharing_plan.original_lengths)
        kept_valid_tokens = sum(prefix_sharing_plan.kept_lengths_q)
        saved_valid_tokens = original_tokens - kept_valid_tokens
        reuser_indices = [
            index
            for index in range(prefix_sharing_plan.batch_size)
            if prefix_sharing_plan.is_reuser(index)
        ]
        sharing_group_ids = {spec.group_id for spec in prefix_sharing_plan.reuse_specs}
        return cls(
            forward_id=prefix_sharing_plan.forward_id,
            micro_batch_id=prefix_sharing_plan.micro_batch_id,
            batch_size=prefix_sharing_plan.batch_size,
            original_tokens=original_tokens,
            kept_valid_tokens=kept_valid_tokens,
            kept_padded_tokens=packed_batch_layout.total_padded_length,
            saved_valid_tokens=saved_valid_tokens,
            saved_valid_token_ratio=(
                saved_valid_tokens / original_tokens if original_tokens > 0 else 0.0
            ),
            provider_count=sum(prefix_sharing_plan.is_provider),
            reuser_count=len(reuser_indices),
            sharing_group_count=len(sharing_group_ids),
            expected_attention_load_count_per_layer=len(reuser_indices),
            expected_loaded_prefix_tokens_per_layer=sum(
                prefix_sharing_plan.prefix_lens[index] for index in reuser_indices
            ),
            expected_restore_count=len(prefix_sharing_plan.prefix_last_restore),
        )

    def layer(self, layer_id: int) -> PrefixSharingLayerStats:
        if layer_id not in self.layers:
            self.layers[layer_id] = PrefixSharingLayerStats(layer_id=layer_id)
        return self.layers[layer_id]

    def record_attention_kv_build(
        self,
        *,
        layer_id: int,
        store_count: int,
        load_count: int,
        hit_count: int,
        miss_count: int,
        stored_tokens: int,
        loaded_prefix_tokens: int,
        expanded_kv_tokens: int,
        valid_q_tokens: int,
        padded_q_tokens: int,
    ) -> None:
        layer = self.layer(layer_id)
        layer.store_count += store_count
        layer.load_count += load_count
        layer.hit_count += hit_count
        layer.miss_count += miss_count
        layer.stored_tokens += stored_tokens
        layer.loaded_prefix_tokens += loaded_prefix_tokens
        layer.expanded_kv_tokens += expanded_kv_tokens
        layer.valid_q_tokens += valid_q_tokens
        layer.padded_q_tokens += padded_q_tokens

    def record_restore(self, count: int) -> None:
        self.actual_restore_count += count

    def layer_matches_expected(self, layer_id: int) -> bool:
        layer = self.layers.get(layer_id)
        if layer is None:
            return False
        return (
            layer.load_count == self.expected_attention_load_count_per_layer
            and layer.loaded_prefix_tokens == self.expected_loaded_prefix_tokens_per_layer
            and layer.miss_count == 0
        )
