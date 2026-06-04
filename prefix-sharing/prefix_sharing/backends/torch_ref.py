"""Pure PyTorch reference backend.

This module imports torch lazily so the package can be developed and tested in
CPU environments where PyTorch is not installed. Tests that exercise this module
skip automatically when torch is unavailable.
"""

from __future__ import annotations

import math
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PREFIX_STATE_TYPE_DELTANET_STATE,
    PrefixActivationSlotId,
    PrefixAttentionStore,
    PrefixDeltanetStore,
)


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("TorchReferenceBackend requires PyTorch") from exc
    return torch


class TorchReferenceBackend:
    capabilities = BackendCapabilities(
        name="torch_ref",
        supports_cpu=True,
        supports_cuda=True,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_gated_attention=True,
        supports_deltanet_state_reuse=True,
    )

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)

    def apply_rope(
        self,
        query: Any,
        key: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        rope_fn: Any | None = None,
        **_: Any,
    ) -> tuple[Any, Any]:
        if rope_fn is None:
            return query, key
        return rope_fn(query, key, prefix_sharing_plan.q_position_offsets, prefix_sharing_plan.kv_position_offsets)

    def build_kv(
        self,
        key: Any,
        value: Any,
        store: PrefixAttentionStore,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
    ) -> tuple[Any, Any]:
        torch = _torch()
        layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        key_rows = _split_packed(key, layout.padded_lengths)
        value_rows = _split_packed(value, layout.padded_lengths)

        # Try optimized fast path for the common RL pattern:
        # batch_index 0 is provider, all others are reusers from the same provider
        # with the same prefix length.
        fast_result = _try_build_kv_fast_path(
            key_rows, value_rows, store, prefix_sharing_plan, layout,
            layer_id=layer_id, tp_rank=tp_rank,
        )
        if fast_result is not None:
            return fast_result

        # General path with cascading support
        expanded_keys = []
        expanded_values = []
        for batch_index, (key_row, value_row) in enumerate(zip(key_rows, value_rows)):
            valid_length = layout.valid_lengths[batch_index]
            valid_key_row = key_row[:valid_length]
            valid_value_row = value_row[:valid_length]
            if not prefix_sharing_plan.is_reuser(batch_index):
                slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    PREFIX_STATE_TYPE_ATTENTION_KV,
                    tp_rank,
                )
                store.store(
                    slot_id,
                    key_tensor=valid_key_row,
                    value_tensor=valid_value_row,
                    prefix_len=valid_key_row.shape[0],
                    overwrite=True,
                )
                expanded_keys.append(valid_key_row)
                expanded_values.append(valid_value_row)
            else:
                provider = prefix_sharing_plan.provider_index[batch_index]
                provider_slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    provider,
                    PREFIX_STATE_TYPE_ATTENTION_KV,
                    tp_rank,
                )
                entry = store.load(provider_slot_id)
                prefix_len = prefix_sharing_plan.prefix_lens[batch_index]
                expanded_key = torch.cat([entry.key_tensor[:prefix_len], valid_key_row], dim=0)
                expanded_value = torch.cat([entry.value_tensor[:prefix_len], valid_value_row], dim=0)
                own_slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    PREFIX_STATE_TYPE_ATTENTION_KV,
                    tp_rank,
                )
                store.store(
                    own_slot_id,
                    key_tensor=expanded_key,
                    value_tensor=expanded_value,
                    prefix_len=expanded_key.shape[0],
                    overwrite=True,
                )
                expanded_keys.append(expanded_key)
                expanded_values.append(expanded_value)
        return torch.cat(expanded_keys, dim=0), torch.cat(expanded_values, dim=0)

    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        **_: Any,
    ) -> Any:
        torch = _torch()
        layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        # Query keeps the framework packed shape, so split by padded lengths.
        # K/V were already depadded and prefix-expanded by build_kv().
        query_rows = _split_packed(query, layout.padded_lengths)
        key_rows = _split_packed(key, prefix_sharing_plan.expanded_lengths_kv)
        value_rows = _split_packed(value, prefix_sharing_plan.expanded_lengths_kv)
        outputs = []
        for batch_index, (q_row, k_row, v_row) in enumerate(zip(query_rows, key_rows, value_rows)):
            valid_length = layout.valid_lengths[batch_index]
            # Padding query slots are layout-only; they must not participate in attention.
            q_valid = q_row[:valid_length]
            prefix_len = prefix_sharing_plan.q_position_offsets[batch_index]
            if valid_length == 0:
                outputs.append(torch.zeros_like(q_row))
                continue
            mask = _causal_q_kv_mask(
                q_len=q_valid.shape[0],
                kv_len=k_row.shape[0],
                q_start=prefix_len,
                device=q_valid.device,
            )
            valid_output = _attention_row(q_valid, k_row, v_row, mask)
            if valid_length == q_row.shape[0]:
                outputs.append(valid_output)
                continue
            padded_output = torch.zeros_like(q_row)
            padded_output[:valid_length] = valid_output
            outputs.append(padded_output)
        return torch.cat(outputs, dim=0)

    def gated_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        gate: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply Qwen3.5-style output gate after prefix-expanded attention.

        The gate is derived from the current kept hidden states in the model, so
        prefix sharing must not cache it. This reference helper keeps that
        invariant explicit for future HybridAttention integrations.
        """

        torch = _torch()
        attention_output = self.attention(
            query,
            key,
            value,
            prefix_sharing_plan,
            packed_batch_layout=packed_batch_layout,
            **kwargs,
        )
        if attention_output.shape != gate.shape:
            raise ValueError("gate shape must match attention output shape")
        return attention_output * torch.sigmoid(gate)

    def build_deltanet_states(
        self,
        state_update: Any,
        store: PrefixDeltanetStore,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
    ) -> Any:
        """Build prefix-expanded Qwen3.5 GatedDeltaNet recurrent trajectories.

        This reference uses a cumulative recurrent trajectory to verify the
        critical prefix boundary and autograd semantics; real integrations
        should map the same store entry to the engine's recurrent/conv cache
        params.
        """

        torch = _torch()
        layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        update_rows = _split_packed(state_update, layout.padded_lengths)
        outputs = []
        for batch_index, update_row in enumerate(update_rows):
            valid_length = layout.valid_lengths[batch_index]
            valid_update_row = update_row[:valid_length]
            if not prefix_sharing_plan.is_reuser(batch_index):
                state_trajectory = torch.cumsum(valid_update_row, dim=0)
                slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    PREFIX_STATE_TYPE_DELTANET_STATE,
                    tp_rank,
                )
                # Publish provider state so later reusers can start from the
                # exact prefix boundary instead of recomputing the prefix.
                store.store(
                    slot_id,
                    recurrent_state=state_trajectory,
                    prefix_len=state_trajectory.shape[0],
                    overwrite=True,
                )
                outputs.append(_pad_like_row(state_trajectory, update_row))
                continue

            provider = prefix_sharing_plan.provider_index[batch_index]
            provider_slot_id = PrefixActivationSlotId(
                prefix_sharing_plan.forward_id,
                prefix_sharing_plan.micro_batch_id,
                layer_id,
                provider,
                PREFIX_STATE_TYPE_DELTANET_STATE,
                tp_rank,
            )
            # The provider trajectory is indexed at prefix_len - 1 to obtain the
            # reusable state after the shared prefix has been consumed.
            entry = store.load(provider_slot_id)
            prefix_len = prefix_sharing_plan.prefix_lens[batch_index]
            if prefix_len <= 0:
                initial_state = torch.zeros_like(valid_update_row[:1]).squeeze(0)
                provider_prefix_trajectory = valid_update_row[:0]
            else:
                if prefix_len > entry.recurrent_state.shape[0]:
                    raise ValueError("prefix_len exceeds stored provider activation length")
                initial_state = entry.recurrent_state[prefix_len - 1]
                provider_prefix_trajectory = entry.recurrent_state[:prefix_len]
            suffix_trajectory = initial_state + torch.cumsum(valid_update_row, dim=0)
            own_state_trajectory = torch.cat([provider_prefix_trajectory, suffix_trajectory], dim=0)
            own_slot_id = PrefixActivationSlotId(
                prefix_sharing_plan.forward_id,
                prefix_sharing_plan.micro_batch_id,
                layer_id,
                batch_index,
                PREFIX_STATE_TYPE_DELTANET_STATE,
                tp_rank,
            )
            # Publish the expanded reuser trajectory for transitive reuse by a later row.
            store.store(
                own_slot_id,
                recurrent_state=own_state_trajectory,
                prefix_len=own_state_trajectory.shape[0],
                overwrite=True,
            )
            outputs.append(_pad_like_row(suffix_trajectory, update_row))
        return torch.cat(outputs, dim=0)


def _try_build_kv_fast_path(
    key_rows: list[Any],
    value_rows: list[Any],
    store: PrefixAttentionStore,
    plan: PrefixSharingPlan,
    layout: PackedBatchLayout,
    *,
    layer_id: int,
    tp_rank: int = 0,
) -> tuple[Any, Any] | None:
    """Vectorised build_kv for the common RL pattern: one provider, N reusers.

    The pattern: batch_index 0 is a provider; all other rows are reusers that
    reference batch_index 0 as their provider *with the same prefix_len*.  When
    this holds we can store the provider KV once and expand every reuser in a
    single batched ``torch.cat`` instead of looping one-by-one in Python.

    Returns ``None`` when the pattern does not match, so the caller can fall
    through to the general (cascading / mixed-group) path.
    """
    torch = _torch()
    batch_size = len(key_rows)
    if batch_size < 2:
        return None

    # batch 0 must be a provider (provider_index[0] == 0)
    if plan.provider_index[0] != 0:
        return None

    provider_prefix_len = plan.prefix_lens[0]

    # All other rows must be reusers from provider 0 with the same prefix_len
    for i in range(1, batch_size):
        if plan.provider_index[i] != 0:
            return None
        if plan.prefix_lens[i] != provider_prefix_len:
            return None
        if provider_prefix_len <= 0:
            return None

    # --- Pattern matched: execute fast path ---

    # 1. Store provider KV
    provider_slot = PrefixActivationSlotId(
        plan.forward_id, plan.micro_batch_id, layer_id, 0,
        PREFIX_STATE_TYPE_ATTENTION_KV, tp_rank,
    )
    provider_valid_len = layout.valid_lengths[0]
    provider_key = key_rows[0][:provider_valid_len]
    provider_value = value_rows[0][:provider_valid_len]
    store.store(
        provider_slot,
        key_tensor=provider_key,
        value_tensor=provider_value,
        prefix_len=provider_key.shape[0],
        overwrite=True,
    )

    # 2. Provider output: its own valid KV
    expanded_keys = [provider_key]
    expanded_values = [provider_value]

    # 3. All reusers share the same prefix slice from the provider
    prefix_key = provider_key[:provider_prefix_len]     # (P, H, D)
    prefix_value = provider_value[:provider_prefix_len]  # (P, H, D)

    # 4. Expand all reusers at once
    for i in range(1, batch_size):
        valid_len = layout.valid_lengths[i]
        suffix_key = key_rows[i][:valid_len]
        suffix_value = value_rows[i][:valid_len]

        expanded_key = torch.cat([prefix_key, suffix_key], dim=0)
        expanded_value = torch.cat([prefix_value, suffix_value], dim=0)

        reuser_slot = PrefixActivationSlotId(
            plan.forward_id, plan.micro_batch_id, layer_id, i,
            PREFIX_STATE_TYPE_ATTENTION_KV, tp_rank,
        )
        store.store(
            reuser_slot,
            key_tensor=expanded_key,
            value_tensor=expanded_value,
            prefix_len=expanded_key.shape[0],
            overwrite=True,
        )
        expanded_keys.append(expanded_key)
        expanded_values.append(expanded_value)

    return torch.cat(expanded_keys, dim=0), torch.cat(expanded_values, dim=0)


def _split_packed(tensor: Any, lengths: list[int]) -> list[Any]:
    torch = _torch()
    if not lengths:
        return []
    if sum(lengths) != tensor.shape[0]:
        raise ValueError("packed tensor first dimension does not match lengths")
    return list(torch.split(tensor, lengths, dim=0))


def _causal_q_kv_mask(q_len: int, kv_len: int, q_start: int, device: Any) -> Any:
    torch = _torch()
    q_positions = torch.arange(q_start, q_start + q_len, device=device).unsqueeze(1)
    kv_positions = torch.arange(0, kv_len, device=device).unsqueeze(0)
    return kv_positions <= q_positions


def _attention_row(q_row: Any, k_row: Any, v_row: Any, mask: Any) -> Any:
    """Compute attention for a single row with causal masking.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` when available
    (PyTorch >= 2.0) which dispatches to FlashAttention-2 or memory-efficient
    attention on CUDA. Falls back to manual implementation on CPU or when the
    SDPA path is not applicable (e.g., 2D inputs).
    """
    torch = _torch()
    scale = math.sqrt(q_row.shape[-1])
    if q_row.dim() == 2:
        scores = q_row @ k_row.transpose(-1, -2) / scale
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        return probs @ v_row
    if q_row.dim() != 3:
        raise ValueError("TorchReferenceBackend attention expects packed rows with 2 or 3 dims")

    q_heads = q_row.shape[1]
    kv_heads = k_row.shape[1]
    if q_heads != kv_heads:
        if q_heads > kv_heads:
            # GQA: repeat KV heads to match query heads
            if q_heads % kv_heads != 0:
                raise ValueError("query heads must be a multiple of kv heads for grouped-query attention")
            repeat = q_heads // kv_heads
            k_row = k_row.repeat_interleave(repeat, dim=1)
            v_row = v_row.repeat_interleave(repeat, dim=1)
        elif kv_heads > q_heads:
            # TP partition with replicated KV: repeat Q to match KV heads
            # Each Q head attends to all its corresponding KV heads
            if kv_heads % q_heads != 0:
                raise ValueError("kv heads must be a multiple of q heads for TP partitioning")
            repeat = kv_heads // q_heads
            q_row = q_row.repeat_interleave(repeat, dim=1)

    # Use SDPA on CUDA for significant speedup (dispatches to FA2/mem-efficient)
    if q_row.is_cuda and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        # SDPA expects (B, H, S, D) format
        q_4d = q_row.unsqueeze(0).transpose(1, 2)  # (1, H, Q, D)
        k_4d = k_row.unsqueeze(0).transpose(1, 2)  # (1, H, KV, D)
        v_4d = v_row.unsqueeze(0).transpose(1, 2)  # (1, H, KV, D)
        # Convert boolean mask to additive attention mask: 0.0 for attend, -inf for mask
        attn_mask = torch.zeros_like(mask, dtype=q_row.dtype)
        attn_mask = attn_mask.masked_fill(~mask, torch.finfo(q_row.dtype).min)
        # Broadcast mask to (1, H, Q, KV)
        attn_mask = attn_mask.unsqueeze(0)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_4d, k_4d, v_4d,
            attn_mask=attn_mask,
            scale=1.0 / scale,
        )
        return out.squeeze(0).transpose(0, 1)  # (Q, H, D)

    # Fallback: manual einsum (slower but works everywhere)
    scores = torch.einsum("qhd,khd->hqk", q_row, k_row) / scale
    scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v_row)


def _pad_like_row(valid_row: Any, packed_row: Any) -> Any:
    if valid_row.shape[0] == packed_row.shape[0]:
        return valid_row
    torch = _torch()
    padded_row = torch.zeros_like(packed_row)
    padded_row[: valid_row.shape[0]] = valid_row
    return padded_row
