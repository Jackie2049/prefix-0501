"""Pure PyTorch reference backend."""

from __future__ import annotations

import math
import torch
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.batch_layout import BatchRuntimeLayout, BshdBatchLayout, ThdBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PREFIX_STATE_TYPE_DELTANET_STATE,
    PrefixActivationSlotId,
    PrefixAttentionStore,
    PrefixDeltanetStore,
)


class TorchReferenceBackend:
    """Pure-PyTorch reference backend for correctness verification.

    Produces the same prefix-expanded KV and attention outputs as a real
    integration, but using explicit Python loops and standard torch ops —
    no fused kernels, no Flash Attention.  Useful for debugging and as a
    ground truth for GPU/NPU backend comparisons.
    """

    # Declared capabilities: this backend runs everywhere (CPU, CUDA, CANN)
    # and supports all prefix-sharing features.
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
        batch_runtime_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
    ) -> tuple[Any, Any]:
        """Build prefix-expanded key/value tensors for attention.

        Provider rows keep their own suffix KV; reuser rows prepend the
        provider's prefix KV before their suffix, producing longer expanded
        KV sequences.  Each row publishes its (possibly expanded) KV to the
        store so that later reusers in the same micro-batch can load it.
        """

        # -- Resolve layout; fall back to THD if caller did not supply one --
        layout = batch_runtime_layout or ThdBatchLayout.construct_from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        is_bshd = layout.layout_kind == "bshd"
        is_thd = layout.layout_kind == "thd"

        # -- Slice per-row padded KV views for the assembly loop below --
        key_rows = [layout.padded_tokens(key, seq_idx_in_batch) for seq_idx_in_batch in range(layout.batch_size)]
        value_rows = [layout.padded_tokens(value, seq_idx_in_batch) for seq_idx_in_batch in range(layout.batch_size)]
        expanded_keys = []
        expanded_values = []

        # -- Sequential KV assembly loop --
        # This loop relies on the current online detector invariant that a provider
        # appears before every reuser that loads from it. All rows' QKV tensors have
        # already been produced in parallel by this point; the ordering here only
        # controls KV assembly before attention. Do not reorder or parallelize this
        # loop unless provider dependencies are handled explicitly, e.g. by a
        # topology-aware build phase.
        for batch_index, (key_row, value_row) in enumerate(zip(key_rows, value_rows)):
            valid_length = layout.valid_lengths[batch_index]
            valid_key_row = layout.valid_tokens(key, batch_index)
            valid_value_row = layout.valid_tokens(value, batch_index)

            # -- Provider path: publish own suffix KV, no expansion needed --
            if not prefix_sharing_plan.is_reuser(batch_index):
                slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    PREFIX_STATE_TYPE_ATTENTION_KV,
                    tp_rank,
                )
                # Publish this row's KV so later reusers in this micro-batch can load it.
                store.store(
                    slot_id,
                    key_tensor=valid_key_row,
                    value_tensor=valid_value_row,
                    prefix_len=valid_key_row.shape[0],
                    overwrite=True,
                )
                expanded_keys.append(valid_key_row)
                expanded_values.append(valid_value_row)

            # -- Reuser path: load provider prefix KV, prepend to own suffix --
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
                # Load the already-published provider KV before building this reuser's expanded KV.
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
                # Publish the expanded reuser KV because a later row may reuse this longer prefix.
                store.store(
                    own_slot_id,
                    key_tensor=expanded_key,
                    value_tensor=expanded_value,
                    prefix_len=expanded_key.shape[0],
                    overwrite=True,
                )
                expanded_keys.append(expanded_key)
                expanded_values.append(expanded_value)

        # -- Return: BSHD keeps per-row lists; THD concatenates into packed 1-D tensors --
        if is_bshd:
            return expanded_keys, expanded_values
        if is_thd:
            return torch.cat(expanded_keys, dim=0), torch.cat(expanded_values, dim=0)
        raise ValueError(f"unsupported batch runtime layout: {layout.layout_kind}")

    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        batch_runtime_layout: Any | None = None,
        **_: Any,
    ) -> Any:
        """Run per-row causal attention over prefix-expanded KV.

        Each row computes scaled dot-product attention between its valid query
        tokens and the full expanded KV for that row, masked so that query
        positions can only attend to positions ≤ their absolute offset in
        the original sequence.  Padding slots are excluded from attention.
        """

        # -- Resolve layout --
        layout = batch_runtime_layout or ThdBatchLayout.construct_from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        is_bshd = layout.layout_kind == "bshd"
        is_thd = layout.layout_kind == "thd"

        # -- Slice per-row padded query views --
        query_rows = [layout.padded_tokens(query, seq_idx_in_batch) for seq_idx_in_batch in range(layout.batch_size)]

        # -- Prepare KV rows and output tensor according to layout kind --
        if is_bshd:
            key_rows = key
            value_rows = value
            dense_output = torch.zeros_like(query)
        elif is_thd:
            key_rows = _split_packed(key, prefix_sharing_plan.expanded_lengths_kv)
            value_rows = _split_packed(value, prefix_sharing_plan.expanded_lengths_kv)
            dense_output = None
        else:
            raise ValueError(f"unsupported batch runtime layout: {layout.layout_kind}")

        # -- Per-row attention loop --
        outputs = []
        for batch_index, (q_row, k_row, v_row) in enumerate(zip(query_rows, key_rows, value_rows)):
            valid_length = layout.valid_lengths[batch_index]

            # Extract only the valid (non-padding) query tokens for this row.
            q_valid = layout.valid_tokens(query, batch_index)
            prefix_len = prefix_sharing_plan.q_position_offsets[batch_index]

            # -- Skip empty rows: BSHD just ignores them; THD emits a zero placeholder --
            if valid_length == 0:
                if is_bshd:
                    continue
                if is_thd:
                    outputs.append(torch.zeros_like(q_row))
                    continue
                raise ValueError(f"unsupported batch runtime layout: {layout.layout_kind}")

            # -- Build causal mask: each query position can attend to KV positions
            #    at or before its absolute offset in the original sequence. --
            mask = _causal_q_kv_mask(
                q_len=q_valid.shape[0],
                kv_len=k_row.shape[0],
                q_start=prefix_len,
                device=q_valid.device,
            )
            valid_output = _attention_row(q_valid, k_row, v_row, mask)

            # -- BSHD: write valid output directly into the dense output tensor --
            if is_bshd:
                layout.write_valid_tokens(dense_output, batch_index, valid_output)
                continue

            # -- THD: collect per-row outputs; pad if the row has padding slots --
            if not is_thd:
                raise ValueError(f"unsupported batch runtime layout: {layout.layout_kind}")
            if valid_length == q_row.shape[0]:
                outputs.append(valid_output)
                continue
            padded_output = torch.zeros_like(q_row)
            padded_output[:valid_length] = valid_output
            outputs.append(padded_output)

        # -- Return: BSHD yields the dense tensor; THD concatenates into packed 1-D --
        if is_bshd:
            return dense_output
        if is_thd:
            return torch.cat(outputs, dim=0)
        raise ValueError(f"unsupported batch runtime layout: {layout.layout_kind}")

    def gated_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        gate: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        batch_runtime_layout: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply Qwen3.5-style output gate after prefix-expanded attention.

        The gate is derived from the current kept hidden states in the model, so
        prefix sharing must not cache it. This reference helper keeps that
        invariant explicit for future HybridAttention integrations.
        """
        attention_output = self.attention(
            query,
            key,
            value,
            prefix_sharing_plan,
            batch_runtime_layout=batch_runtime_layout,
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
        batch_runtime_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
    ) -> Any:
        """Build prefix-expanded Qwen3.5 GatedDeltaNet recurrent trajectories.

        This reference uses a cumulative recurrent trajectory to verify the
        critical prefix boundary and autograd semantics; real integrations
        should map the same store entry to the engine's recurrent/conv cache
        params.
        """

        layout = batch_runtime_layout or ThdBatchLayout.construct_from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        is_bshd = layout.layout_kind == "bshd"
        is_thd = layout.layout_kind == "thd"

        # -- Slice per-row padded state-update views --
        update_rows = [layout.padded_tokens(state_update, seq_idx_in_batch) for seq_idx_in_batch in range(layout.batch_size)]

        # -- Allocate output tensor according to layout kind --
        if is_bshd:
            dense_output = torch.zeros_like(state_update)
        elif is_thd:
            dense_output = None
        else:
            raise ValueError(f"unsupported batch runtime layout: {layout.layout_kind}")
        outputs = []

        # -- Sequential state assembly loop (same ordering invariant as build_kv) --
        for batch_index, update_row in enumerate(update_rows):
            valid_length = layout.valid_lengths[batch_index]
            valid_update_row = layout.valid_tokens(state_update, batch_index)

            # -- Provider path: cumsum the state update, publish as a trajectory --
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
                if is_bshd:
                    layout.write_valid_tokens(dense_output, batch_index, state_trajectory)
                elif is_thd:
                    outputs.append(_pad_like_tokens(state_trajectory, update_row))
                else:
                    raise ValueError(f"unsupported batch runtime layout: {layout.layout_kind}")
                continue

            # -- Reuser path: load provider prefix trajectory, then cumsum own suffix --
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
            if is_bshd:
                layout.write_valid_tokens(dense_output, batch_index, suffix_trajectory)
            elif is_thd:
                outputs.append(_pad_like_tokens(suffix_trajectory, update_row))
            else:
                raise ValueError(f"unsupported batch runtime layout: {layout.layout_kind}")

        # -- Return: BSHD yields the dense tensor; THD concatenates into packed 1-D --
        if is_bshd:
            return dense_output
        if is_thd:
            return torch.cat(outputs, dim=0)
        raise ValueError(f"unsupported batch runtime layout: {layout.layout_kind}")


def _split_packed(tensor: Any, lengths: list[int]) -> list[Any]:
    """Split a 1-D packed tensor into per-row chunks along dim-0."""
    if not lengths:
        return []
    if sum(lengths) != tensor.shape[0]:
        raise ValueError("packed tensor first dimension does not match lengths")
    return list(torch.split(tensor, lengths, dim=0))


def _causal_q_kv_mask(q_len: int, kv_len: int, q_start: int, device: Any) -> Any:
    """Build a 2-D causal mask where each query position can attend to KV at or before its absolute offset."""
    q_positions = torch.arange(q_start, q_start + q_len, device=device).unsqueeze(1)
    kv_positions = torch.arange(0, kv_len, device=device).unsqueeze(0)
    return kv_positions <= q_positions


def _attention_row(q_row: Any, k_row: Any, v_row: Any, mask: Any) -> Any:
    """Scaled dot-product attention for a single row, supporting MHA (2-D) and GQA (3-D)."""
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
        if q_heads % kv_heads != 0:
            raise ValueError("query heads must be a multiple of kv heads for grouped-query attention")
        repeat = q_heads // kv_heads
        k_row = k_row.repeat_interleave(repeat, dim=1)
        v_row = v_row.repeat_interleave(repeat, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q_row, k_row) / scale
    scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v_row)


def _pad_like_tokens(valid_tokens: Any, packed_tokens: Any) -> Any:
    """Zero-pad a valid-length tensor to match the shape of a padded row."""
    if valid_tokens.shape[0] == packed_tokens.shape[0]:
        return valid_tokens
    padded_tokens = torch.zeros_like(packed_tokens)
    padded_tokens[: valid_tokens.shape[0]] = valid_tokens
    return padded_tokens
