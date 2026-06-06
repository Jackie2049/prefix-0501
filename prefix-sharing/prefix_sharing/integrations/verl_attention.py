"""Prefix-sharing integration for verl's Qwen3.6 model classes.

When verl uses its own model classes (ParallelQwen3_6AttentionRmPad,
ParallelQwen3_6GatedDeltaNetRmPad), this module provides prefix-sharing
hooks that intercept the attention forward pass.

For full attention layers (16 out of 64 in Qwen3.6-27B):
- Intercept after q/k/v projection and before flash_attn
- Apply q_norm/k_norm (per-head RMSNorm) and partial RoPE
- Expand KV with prefix-sharing data
- Run flash_attn_varlen_func with expanded KV
- Apply output gate (attn_output * sigmoid(gate))
- Run o_proj (RowParallelLinear)

For DeltaNet layers (48 out of 64) — two-pass state injection:
- Prefix pass: process provider's prefix tokens with output_final_state=True,
  extract conv1d overlap context from the last (kernel_size-1) layernorm
  outputs, and store recurrent state + conv overlap in deltanet_store.
- Suffix pass: load stored state, inject as initial_state + conv_overlap_hidden
  into the DeltaNet forward for state injection.
- This achieves precision alignment (cos_sim > 0.999) as validated in the
  two-pass E2E test (scripts/run_ps_e2e_twopass_v2.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.megatron_attention import IntegrationUnavailable
from prefix_sharing.integrations.megatron_runtime import _read_parallel_rank_info
from prefix_sharing.integrations.patch_manager import PatchManager

import logging
logger = logging.getLogger(__name__)


def _make_verl_attention_patch(original_forward: Any) -> Any:
    """Create a patched forward for verl's ParallelQwen3_6AttentionRmPad.

    Two-pass prefix-sharing for full attention layers.

    Works in RmPad (3D flat) format like the original forward, using
    apply_rotary_emb with cu_seqlens for RoPE.

    **Prefix pass** (store has no KV for this layer):
      - Compute QKV + QK norm + partial RoPE (in 3D flat format)
      - Store prefix KV (after GQA expand) in 4D format for easy suffix expansion
      - Run flash_attn_varlen_func on prefix tokens (normal RmPad path)
      - Apply output gate + o_proj

    **Suffix pass** (store has KV for this layer):
      - Compute QKV + QK norm (in 3D flat format)
      - Generate cos/sin for positions 0..prefix_len+suffix_len, slice for suffix
      - Apply partial RoPE to suffix tokens (in 3D flat format with cu_seqlens)
      - Load stored prefix KV, expand to N sequences, concat with suffix KV
      - Build custom cu_seqlens for flash_attn_varlen_func (Q=suffix, KV=prefix+suffix)
      - Run flash_attn_varlen_func
      - Apply output gate + o_proj
    """

    def patched_forward(
        self_attention_module,
        hidden_states,
        position_ids=None,
        sequence_length=None,
        indices=None,
        cu_seqlens=None,
        max_seqlen_in_batch=None,
        **kwargs,
    ):
        from prefix_sharing.integrations.context import current_prefix_sharing_context
        from prefix_sharing.core.model_spec import AttentionLayerType

        ctx = current_prefix_sharing_context()
        if ctx is None or not ctx.prefix_sharing_plan.has_sharing:
            return original_forward(
                self_attention_module,
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
                **kwargs,
            )

        # Check if this layer should use prefix-sharing
        layer_id = int(getattr(self_attention_module, "layer_idx", 0) or 0)
        model_spec = ctx.model_spec
        if model_spec is not None and model_spec.layer_type(layer_id) != AttentionLayerType.FULL_ATTENTION:
            # Not a full attention layer, skip prefix-sharing
            return original_forward(
                self_attention_module,
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
                **kwargs,
            )

        # === Prefix-sharing path for full attention (two-pass) ===
        import torch
        from flash_attn import flash_attn_varlen_func
        from flash_attn.layers.rotary import apply_rotary_emb
        from flash_attn.bert_padding import pad_input as _pad_input
        from prefix_sharing.integrations.megatron_runtime import _read_parallel_rank_info
        from prefix_sharing.core.prefix_store import (
            PrefixActivationSlotId,
            PREFIX_STATE_TYPE_ATTENTION_KV,
        )

        attn_module = self_attention_module
        _, tp_rank, _ = _read_parallel_rank_info()

        total_nnz, _, _ = hidden_states.size()

        if attn_module.megatron_config.sequence_parallel:
            from megatron.core import parallel_state as mpu
            tp_size = mpu.get_tensor_model_parallel_world_size()
            total_nnz = total_nnz * tp_size

        batch_size = len(cu_seqlens) - 1 if cu_seqlens is not None else 1

        # QKV projections (same as original RmPad forward)
        q_full = attn_module.q_proj(hidden_states)[0]
        key_raw = attn_module.k_proj(hidden_states)[0]
        value_raw = attn_module.v_proj(hidden_states)[0]

        if attn_module.megatron_config.sequence_parallel:
            sp_pad = total_nnz - cu_seqlens[-1]
            total_nnz = cu_seqlens[-1]
            q_full = q_full[:total_nnz]
            key_raw = key_raw[:total_nnz]
            value_raw = value_raw[:total_nnz]

        # Chunk q_proj into query and gate (same as original)
        q_shape = (total_nnz, attn_module.num_heads_per_tp, attn_module.head_dim * 2)
        query_states, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)
        gate = gate.reshape(total_nnz, attn_module.num_heads_per_tp * attn_module.head_dim)

        query_states = query_states.view(total_nnz, attn_module.num_heads_per_tp, attn_module.head_dim)
        key_states = key_raw.view(total_nnz, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)
        value_states = value_raw.view(total_nnz, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)

        # QK normalization (same as original)
        query_states = attn_module.q_norm(query_states)
        key_states = attn_module.k_norm(key_states)

        # Build slot ID for this layer's attention KV
        slot_id = PrefixActivationSlotId(
            forward_id=ctx.prefix_sharing_plan.forward_id,
            micro_batch_id=ctx.prefix_sharing_plan.micro_batch_id,
            layer_id=layer_id,
            sample_idx_in_batch=0,
            prefix_state_type=PREFIX_STATE_TYPE_ATTENTION_KV,
            tp_rank=tp_rank,
        )

        if ctx.store.contains(slot_id):
            # === Suffix pass: load stored prefix KV and expand ===
            stored_kv = ctx.store.load(slot_id)
            prefix_key_4d = stored_kv.key_tensor   # (1, prefix_len, num_heads_per_tp, head_dim) after GQA expand
            prefix_value_4d = stored_kv.value_tensor  # (1, prefix_len, num_heads_per_tp, head_dim) after GQA expand
            prefix_len = prefix_key_4d.shape[1]
            suffix_len = sequence_length  # In suffix pass, all tokens are suffix

            # Generate cos/sin for positions 0..prefix_len+suffix_len-1, slice for suffix
            # Use a dummy tensor just for device/dtype, then slice for suffix positions
            cos_full, sin_full = attn_module.rotary_emb(
                value_states, seq_len=prefix_len + suffix_len
            )
            # Halve dimensions for partial RoPE
            cos_full = cos_full[:, :cos_full.shape[1] // 2]
            sin_full = sin_full[:, :sin_full.shape[1] // 2]
            # Slice for suffix positions only
            cos_suffix = cos_full[prefix_len:prefix_len + suffix_len]
            sin_suffix = sin_full[prefix_len:prefix_len + suffix_len]

            # Apply partial RoPE in 3D flat format (RmPad mode with cu_seqlens)
            if attn_module.rope_dim == attn_module.head_dim:
                query_states = apply_rotary_emb(
                    query_states, cos_suffix, sin_suffix,
                    interleaved=False, inplace=False,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
                )
                key_states = apply_rotary_emb(
                    key_states, cos_suffix, sin_suffix,
                    interleaved=False, inplace=False,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
                )
            else:
                # Partial RoPE: only apply to the first rope_dim dimensions
                q_rot = query_states[:, :, :attn_module.rope_dim]
                q_pass = query_states[:, :, attn_module.rope_dim:]
                k_rot = key_states[:, :, :attn_module.rope_dim]
                k_pass = key_states[:, :, attn_module.rope_dim:]
                q_rot = apply_rotary_emb(
                    q_rot, cos_suffix, sin_suffix,
                    interleaved=False, inplace=False,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
                )
                k_rot = apply_rotary_emb(
                    k_rot, cos_suffix, sin_suffix,
                    interleaved=False, inplace=False,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
                )
                query_states = torch.cat([q_rot, q_pass], dim=-1)
                key_states = torch.cat([k_rot, k_pass], dim=-1)

            # GQA expand for suffix KV (in 3D flat format)
            num_key_value_groups = attn_module.num_key_value_groups
            if num_key_value_groups > 1:
                key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
                value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

            # KV expansion: need to interleave prefix KV with suffix KV per sequence
            # prefix_key_4d is (1, prefix_len, num_heads_per_tp, head_dim) — already GQA expanded
            # We need flat format: each sequence gets prefix_len + suffix_len KV tokens
            # Approach: pad suffix KV to 4D, concat with expanded prefix KV, reshape to flat
            # First, pad suffix KV from flat to (N, suffix_len, num_heads_per_tp, head_dim)
            key_4d_suffix = _pad_input(key_states, indices, batch_size, sequence_length)
            value_4d_suffix = _pad_input(value_states, indices, batch_size, sequence_length)

            # Expand prefix KV from (1, prefix_len, ...) to (N, prefix_len, ...)
            prefix_key_expanded = prefix_key_4d.expand(batch_size, -1, -1, -1).contiguous()
            prefix_value_expanded = prefix_value_4d.expand(batch_size, -1, -1, -1).contiguous()

            # Concat prefix + suffix KV along sequence dimension
            expanded_key_4d = torch.cat([prefix_key_expanded, key_4d_suffix], dim=1)
            expanded_value_4d = torch.cat([prefix_value_expanded, value_4d_suffix], dim=1)

            # Reshape to flat for flash_attn_varlen_func
            N = batch_size
            total_kv = N * (prefix_len + suffix_len)
            expanded_key_flat = expanded_key_4d.reshape(total_kv, attn_module.num_heads_per_tp, attn_module.head_dim)
            expanded_value_flat = expanded_value_4d.reshape(total_kv, attn_module.num_heads_per_tp, attn_module.head_dim)

            # Build cu_seqlens for flash_attn_varlen_func
            # Q: each sequence has suffix_len tokens
            # K/V: each sequence has prefix_len + suffix_len tokens
            cu_seqlens_q = cu_seqlens  # Same as original — suffix tokens only
            cu_seqlens_k = torch.tensor(
                [0] + [prefix_len + suffix_len] * N,
                device=hidden_states.device, dtype=torch.int32,
            ).cumsum(0)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                query_states = query_states.to(torch.float16)
                expanded_key_flat = expanded_key_flat.to(torch.float16)
                expanded_value_flat = expanded_value_flat.to(torch.float16)

            # flash_attn with expanded KV
            attn_output = flash_attn_varlen_func(
                query_states, expanded_key_flat, expanded_value_flat,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch, max_seqlen_k=prefix_len + suffix_len,
                dropout_p=0.0, softmax_scale=attn_module.scaling, causal=True,
            )
            attn_output = attn_output.to(input_dtype)

            # Output is in flat format (total_nnz, num_heads, head_dim) —
            # same as original RmPad forward, no _unpad_input needed
            attn_output = attn_output.reshape(total_nnz, attn_module.q_output_size_per_tp)

            # Apply output gate
            if attn_module.attn_output_gate:
                gate = gate[:total_nnz]
                attn_output = attn_output * torch.sigmoid(gate)

            # Reshape for o_proj (same as original)
            attn_output = attn_output.reshape(total_nnz, 1, attn_module.q_output_size_per_tp).contiguous()

            if attn_module.megatron_config.sequence_parallel:
                import torch.nn.functional as F
                attn_output = F.pad(attn_output, pad=(0, 0, 0, 0, 0, sp_pad))

            attn_output = attn_module.o_proj(attn_output)[0]

            logger.debug(
                "[PS][verl-attn-inject][layer=%s] Suffix pass: N=%d, suffix=%d, prefix=%d",
                layer_id, N, suffix_len, prefix_len,
            )

            return attn_output

        else:
            # === Prefix pass: compute KV, store it, run flash_attn ===
            # Same as original RmPad forward, but store KV before flash_attn

            # Generate cos/sin for prefix positions (0..prefix_len-1)
            cos, sin = attn_module.rotary_emb(value_states, seq_len=sequence_length)
            cos, sin = cos[:, :cos.shape[1] // 2], sin[:, :sin.shape[1] // 2]

            # Apply partial RoPE in 3D flat format (RmPad mode)
            if attn_module.rope_dim == attn_module.head_dim:
                query_states = apply_rotary_emb(
                    query_states, cos, sin,
                    interleaved=False, inplace=False,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
                )
                key_states = apply_rotary_emb(
                    key_states, cos, sin,
                    interleaved=False, inplace=False,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
                )
            else:
                q_rot = query_states[:, :, :attn_module.rope_dim]
                q_pass = query_states[:, :, attn_module.rope_dim:]
                k_rot = key_states[:, :, :attn_module.rope_dim]
                k_pass = key_states[:, :, attn_module.rope_dim:]
                q_rot = apply_rotary_emb(
                    q_rot, cos, sin,
                    interleaved=False, inplace=False,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
                )
                k_rot = apply_rotary_emb(
                    k_rot, cos, sin,
                    interleaved=False, inplace=False,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
                )
                query_states = torch.cat([q_rot, q_pass], dim=-1)
                key_states = torch.cat([k_rot, k_pass], dim=-1)

            # GQA expand in 3D flat format (same as original)
            num_key_value_groups = attn_module.num_key_value_groups
            if num_key_value_groups > 1:
                key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
                value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

            # Store prefix KV in 4D format for easy expand in suffix pass
            # Convert flat to (1, prefix_len, num_heads_per_tp, head_dim)
            # Prefix pass has only 1 batch (provider), so this is straightforward
            key_4d = _pad_input(key_states, indices, batch_size, sequence_length)
            value_4d = _pad_input(value_states, indices, batch_size, sequence_length)

            # For prefix pass, batch_size might be 1 (only provider's prefix tokens)
            # Store the entire batch's KV — in prefix pass with batch=1, shape is (1, prefix_len, ...)
            ctx.store.store(
                slot_id,
                key_tensor=key_4d.contiguous(),
                value_tensor=value_4d.contiguous(),
                prefix_len=sequence_length,
            )

            logger.debug(
                "[PS][verl-attn-store][layer=%s] Prefix pass: stored KV key=%s value=%s",
                layer_id, tuple(key_4d.shape), tuple(value_4d.shape),
            )

            # Convert back to flat for flash_attn (undo the _pad_input)
            key_flat = key_4d.reshape(total_nnz, attn_module.num_heads_per_tp, attn_module.head_dim)
            value_flat = value_4d.reshape(total_nnz, attn_module.num_heads_per_tp, attn_module.head_dim)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                query_states = query_states.to(torch.float16)
                key_flat = key_flat.to(torch.float16)
                value_flat = value_flat.to(torch.float16)

            # flash_attn on prefix tokens (same cu_seqlens as original)
            attn_output = flash_attn_varlen_func(
                query_states, key_flat, value_flat,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen_in_batch, max_seqlen_k=max_seqlen_in_batch,
                dropout_p=0.0, softmax_scale=attn_module.scaling, causal=True,
            )
            attn_output = attn_output.to(input_dtype)

            # Output in flat format — same as original RmPad forward
            attn_output = attn_output.reshape(total_nnz, attn_module.q_output_size_per_tp)

            # Apply output gate
            if attn_module.attn_output_gate:
                gate = gate[:total_nnz]
                attn_output = attn_output * torch.sigmoid(gate)

            # Reshape for o_proj (same as original)
            attn_output = attn_output.reshape(total_nnz, 1, attn_module.q_output_size_per_tp).contiguous()

            if attn_module.megatron_config.sequence_parallel:
                import torch.nn.functional as F
                attn_output = F.pad(attn_output, pad=(0, 0, 0, 0, 0, sp_pad))

            attn_output = attn_module.o_proj(attn_output)[0]

            return attn_output

    return patched_forward


def _make_verl_deltanet_patch(original_forward: Any) -> Any:
    """Create a patched forward for verl's ParallelQwen3_6GatedDeltaNetRmPad.

    Supports two-pass prefix-sharing for DeltaNet (GatedDeltaNet) layers.

    **Prefix pass** (deltanet_store has no state for this layer):
      - Run DeltaNet forward with output_final_state=True to capture the
        accumulated recurrent state after processing prefix tokens.
      - Extract conv1d overlap context: the last (conv_kernel_size-1) tokens
        of the layernorm output (hidden_states input to DeltaNet). These
        provide causal context for the conv1d at the prefix/suffix boundary.
      - Store recurrent_state + conv_state in deltanet_store for the suffix
        pass to inject.

    **Suffix pass** (deltanet_store has state for this layer):
      - Load stored recurrent_state and conv_state (conv1d overlap).
      - Pass them as initial_state and conv_overlap_hidden to the DeltaNet
        forward, enabling state injection so suffix tokens start from the
        prefix-accumulated recurrent state instead of zero initial state.
      - All sequences share the same prefix state (expanded by DeltaNet's
        padded forward when conv_overlap_hidden has bsz=1).

    This two-pass approach achieves cos_sim > 0.999 precision alignment
    as validated in scripts/run_ps_e2e_twopass_v2.py.
    """

    def patched_forward(
        deltanet_module,
        hidden_states,
        position_ids=None,
        sequence_length=None,
        indices=None,
        cu_seqlens=None,
        max_seqlen_in_batch=None,
        **kwargs,
    ):
        from prefix_sharing.integrations.context import current_prefix_sharing_context
        from prefix_sharing.core.model_spec import AttentionLayerType
        from prefix_sharing.core.prefix_store import (
            PrefixActivationSlotId, PREFIX_STATE_TYPE_DELTANET_STATE,
        )
        from prefix_sharing.integrations.megatron_runtime import _read_parallel_rank_info

        ctx = current_prefix_sharing_context()
        if ctx is None or not ctx.prefix_sharing_plan.has_sharing:
            return original_forward(
                deltanet_module,
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
                **kwargs,
            )

        # Check if this is a DeltaNet (linear attention) layer
        layer_id = int(getattr(deltanet_module, "layer_idx", 0) or 0)
        model_spec = ctx.model_spec
        if model_spec is not None and model_spec.layer_type(layer_id) == AttentionLayerType.FULL_ATTENTION:
            # Full attention layer — skip, handled by attention patch
            return original_forward(
                deltanet_module, hidden_states, position_ids=position_ids,
                sequence_length=sequence_length, indices=indices,
                cu_seqlens=cu_seqlens, max_seqlen_in_batch=max_seqlen_in_batch,
                **kwargs,
            )

        # Build slot ID for this layer's DeltaNet state
        _, tp_rank, _ = _read_parallel_rank_info()
        slot_id = PrefixActivationSlotId(
            forward_id=ctx.prefix_sharing_plan.forward_id,
            micro_batch_id=ctx.prefix_sharing_plan.micro_batch_id,
            layer_id=layer_id,
            sample_idx_in_batch=0,  # provider's state (shared by all reusers)
            prefix_state_type=PREFIX_STATE_TYPE_DELTANET_STATE,
            tp_rank=tp_rank,
        )

        if ctx.deltanet_store.contains(slot_id):
            # === Suffix pass: inject stored DeltaNet state ===
            stored_state = ctx.deltanet_store.load(slot_id)
            initial_state = stored_state.recurrent_state
            conv_overlap_hidden = stored_state.conv_state

            logger.debug(
                "[PS][deltanet-inject][layer=%s] Injecting state: "
                "initial_state=%s, conv_overlap=%s",
                layer_id,
                tuple(initial_state.shape) if initial_state is not None else None,
                tuple(conv_overlap_hidden.shape) if conv_overlap_hidden is not None else None,
            )

            return original_forward(
                deltanet_module,
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
                initial_state=initial_state,
                output_final_state=False,
                conv_overlap_hidden=conv_overlap_hidden,
                **kwargs,
            )
        else:
            # === Prefix pass: capture and store DeltaNet state ===
            import torch
            from flash_attn.bert_padding import pad_input as _pad_input

            # Run DeltaNet forward with output_final_state=True to capture
            # the accumulated recurrent state after processing prefix tokens.
            result = original_forward(
                deltanet_module,
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
                initial_state=None,
                output_final_state=True,
                **kwargs,
            )

            # original_forward returns (output, final_state) when output_final_state=True
            output, final_state = result

            if final_state is not None:
                # Extract conv1d overlap context from hidden_states.
                # hidden_states is the layernorm output (input to DeltaNet).
                # The last (conv_kernel_size-1) tokens provide causal context
                # for the conv1d at the prefix/suffix boundary.
                conv_overlap = deltanet_module.conv_kernel_size - 1  # 3
                hidden_size = deltanet_module.hidden_size

                # hidden_states is in RmPad format: (total_nnz, 1, hidden_size)
                total_nnz = hidden_states.shape[0]

                if deltanet_module.megatron_config.sequence_parallel:
                    from megatron.core import parallel_state as mpu
                    total_nnz = total_nnz * mpu.get_tensor_model_parallel_world_size()

                # Unpack to padded format to extract last tokens
                hidden_flat = hidden_states.reshape(total_nnz, hidden_size)

                if indices is not None and cu_seqlens is not None:
                    batch_size = len(cu_seqlens) - 1
                    hidden_padded = _pad_input(hidden_flat, indices, batch_size, sequence_length)
                else:
                    # Already padded: (bsz, seq_len, hidden_size) with dim-1=1 removed
                    hidden_padded = hidden_flat.unsqueeze(0) if hidden_flat.dim() == 2 else hidden_flat

                # Extract last conv_overlap tokens as conv1d context.
                # Shape: (1, conv_overlap, hidden_size) — provider's overlap only,
                # will be expanded to all sequences by DeltaNet padded forward.
                conv_overlap_hidden = hidden_padded[:1, -conv_overlap:, :].contiguous()

                # Determine prefix_len from the plan or from actual input length.
                # In prefix pass, we're processing only the provider's prefix tokens,
                # so the total sequence length equals the prefix length.
                prefix_len = sequence_length if sequence_length is not None else 0

                # Store in deltanet_store for suffix pass injection
                ctx.deltanet_store.store(
                    slot_id,
                    recurrent_state=final_state,
                    prefix_len=prefix_len,
                    conv_state=conv_overlap_hidden,
                )

                logger.debug(
                    "[PS][deltanet-store][layer=%s] Stored state: "
                    "recurrent_state=%s, conv_overlap=%s, prefix_len=%s",
                    layer_id,
                    tuple(final_state.shape),
                    tuple(conv_overlap_hidden.shape),
                    prefix_len,
                )

            # Return only the output (not the tuple)
            return output

    return patched_forward


@dataclass
class VerlQwen3_6Integration:
    """Prefix-sharing integration for verl's Qwen3.6 model."""

    config: PrefixSharingConfig
    backend: Any = None

    def install(self, model_config: Any | None = None):
        """Patch verl's Qwen3.6 attention classes."""
        self.config.validate(model_config=model_config, integrate_mode="verl_megatron_actor")
        self._ensure_verl_importable()

        mgr = PatchManager()

        try:
            from verl.models.qwen3_6.megatron.layers.parallel_attention import (
                ParallelQwen3_6AttentionRmPad,
            )
            original_attn_forward = ParallelQwen3_6AttentionRmPad.forward
            patched_attn = _make_verl_attention_patch(original_attn_forward)
            mgr.patch_attr(ParallelQwen3_6AttentionRmPad, "forward", patched_attn)
            logger.info("Patched ParallelQwen3_6AttentionRmPad.forward for prefix-sharing")
        except ImportError:
            logger.warning("Could not import ParallelQwen3_6AttentionRmPad, skipping")

        try:
            from verl.models.qwen3_6.megatron.layers.parallel_deltanet import (
                ParallelQwen3_6GatedDeltaNetRmPad,
            )
            original_deltanet_forward = ParallelQwen3_6GatedDeltaNetRmPad.forward
            patched_deltanet = _make_verl_deltanet_patch(original_deltanet_forward)
            mgr.patch_attr(ParallelQwen3_6GatedDeltaNetRmPad, "forward", patched_deltanet)
            logger.info("Patched ParallelQwen3_6GatedDeltaNetRmPad.forward for two-pass PS state injection")
        except ImportError:
            logger.warning("Could not import ParallelQwen3_6GatedDeltaNetRmPad, skipping")

        return mgr.handle()

    @staticmethod
    def _ensure_verl_importable() -> None:
        try:
            import verl  # noqa: F401
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("verl is not importable in this environment") from exc