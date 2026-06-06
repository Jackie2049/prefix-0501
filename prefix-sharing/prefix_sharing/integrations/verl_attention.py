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

    The patch intercepts after QKV projection, applies QK normalization
    and partial RoPE, expands KV with prefix-sharing data, then runs
    flash_attn_varlen_func with expanded KV.
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

        # === Prefix-sharing path for full attention ===
        import torch
        from flash_attn import flash_attn_varlen_func
        from prefix_sharing.backends.torch_ref import TorchReferenceBackend
        from prefix_sharing.backends.packed_layout import PackedBatchLayout

        attn_module = self_attention_module
        total_nnz, _, _ = hidden_states.size()

        if attn_module.megatron_config.sequence_parallel:
            from megatron.core import parallel_state as mpu
            tp_size = mpu.get_tensor_model_parallel_world_size()
            total_nnz = total_nnz * tp_size

        # QKV projections (same as original forward)
        q_full = attn_module.q_proj(hidden_states)[0]
        key_states = attn_module.k_proj(hidden_states)[0]
        value_states = attn_module.v_proj(hidden_states)[0]

        if attn_module.megatron_config.sequence_parallel:
            sp_pad = total_nnz - cu_seqlens[-1]
            total_nnz = cu_seqlens[-1]
            q_full = q_full[:total_nnz]
            key_states = key_states[:total_nnz]
            value_states = value_states[:total_nnz]

        # Chunk q_proj into query and gate
        hidden_shape = (total_nnz, attn_module.num_heads_per_tp, attn_module.head_dim * 2)
        query_states, gate = torch.chunk(q_full.view(*hidden_shape), 2, dim=-1)
        gate = gate.reshape(total_nnz, attn_module.num_heads_per_tp * attn_module.head_dim)

        # Reshape
        query_states = query_states.view(total_nnz, attn_module.num_heads_per_tp, attn_module.head_dim)
        key_states = key_states.view(total_nnz, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)
        value_states = value_states.view(total_nnz, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)

        # QK normalization (per-head RMSNorm BEFORE RoPE)
        query_states = attn_module.q_norm(query_states)
        key_states = attn_module.k_norm(key_states)

        # Partial RoPE AFTER q_norm/k_norm
        cos, sin = attn_module.rotary_emb(value_states, seq_len=sequence_length)
        cos, sin = cos[:, :cos.shape[1] // 2], sin[:, :sin.shape[1] // 2]

        if attn_module.rope_dim == attn_module.head_dim:
            from flash_attn.layers.rotary import apply_rotary_emb
            query_states = apply_rotary_emb(
                query_states, cos, sin, interleaved=False, inplace=False,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
            )
            key_states = apply_rotary_emb(
                key_states, cos, sin, interleaved=False, inplace=False,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
            )
        else:
            # Partial RoPE: only rotate first rope_dim dims
            q_rot = query_states[:, :, :attn_module.rope_dim]
            q_pass = query_states[:, :, attn_module.rope_dim:]
            k_rot = key_states[:, :, :attn_module.rope_dim]
            k_pass = key_states[:, :, attn_module.rope_dim:]

            from flash_attn.layers.rotary import apply_rotary_emb
            q_rot = apply_rotary_emb(
                q_rot, cos, sin, interleaved=False, inplace=False,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
            )
            k_rot = apply_rotary_emb(
                k_rot, cos, sin, interleaved=False, inplace=False,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,
            )
            query_states = torch.cat([q_rot, q_pass], dim=-1)
            key_states = torch.cat([k_rot, k_pass], dim=-1)

        # GQA: repeat KV heads to match query heads
        num_key_value_groups = attn_module.num_key_value_groups
        if num_key_value_groups > 1:
            key_states = key_states.unsqueeze(2).expand(
                -1, -1, num_key_value_groups, -1
            ).reshape(total_nnz, attn_module.num_heads_per_tp, attn_module.head_dim)
            value_states = value_states.unsqueeze(2).expand(
                -1, -1, num_key_value_groups, -1
            ).reshape(total_nnz, attn_module.num_heads_per_tp, attn_module.head_dim)

        # === Prefix-sharing KV expansion ===
        backend = ctx.backend or TorchReferenceBackend()
        _, tp_rank, tp_size = _read_parallel_rank_info()

        # Compute actual per-sequence lengths from cu_seqlens (not from packed_batch_layout)
        # The model's unpad_input produces cu_seqlens that reflect the actual non-zero tokens.
        # Each sequence's length = cu_seqlens[i+1] - cu_seqlens[i]
        actual_lengths = [cu_seqlens[i+1].item() - cu_seqlens[i].item()
                          for i in range(len(cu_seqlens) - 1)]

        # Build a PackedBatchLayout from actual_lengths for build_kv splitting
        kv_layout = PackedBatchLayout.from_valid_lengths(actual_lengths)

        expanded_key, expanded_value = backend.build_kv(
            key_states,
            value_states,
            ctx.store,
            ctx.prefix_sharing_plan,
            packed_batch_layout=kv_layout,
            layer_id=layer_id,
            tp_rank=tp_rank,
        )

        # Build expanded cu_seqlens for flash_attn (must be int32)
        # cumsum() can upcast int32 to int64, so force int32 after cumsum
        expanded_cu_seqlens = torch.tensor(
            [0] + list(ctx.prefix_sharing_plan.expanded_lengths_kv),
            device=cu_seqlens.device,
            dtype=torch.int32,
        ).cumsum(0).to(torch.int32)
        max_seqlen_expanded = max(ctx.prefix_sharing_plan.expanded_lengths_kv)

        # Cast if needed (fp32 → fp16 for flash_attn)
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            expanded_key = expanded_key.to(torch.float16)
            expanded_value = expanded_value.to(torch.float16)

        # Run flash_attn with expanded KV
        attn_output = flash_attn_varlen_func(
            query_states,
            expanded_key,
            expanded_value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=expanded_cu_seqlens,
            max_seqlen_q=max_seqlen_in_batch,
            max_seqlen_k=max_seqlen_expanded,
            dropout_p=0.0,
            softmax_scale=attn_module.scaling,
            causal=True,
        )

        attn_output = attn_output.to(input_dtype)
        attn_output = attn_output.reshape(total_nnz, attn_module.q_output_size_per_tp)

        # Apply output gate BEFORE o_proj
        if attn_module.attn_output_gate:
            gate = gate[:total_nnz]
            attn_output = attn_output * torch.sigmoid(gate)

        # Reshape for o_proj and handle SP padding
        attn_output = attn_output.reshape(total_nnz, 1, attn_module.q_output_size_per_tp).contiguous()

        if attn_module.megatron_config.sequence_parallel:
            import torch.nn.functional as F
            attn_output = F.pad(attn_output, pad=(0, 0, 0, 0, 0, sp_pad))

        attn_output = attn_module.o_proj(attn_output)[0]

        logger.debug(
            "[PS][verl-attn][layer=%s] query=%s expanded_kv=%s output=%s",
            layer_id, tuple(query_states.shape),
            tuple(expanded_key.shape), tuple(attn_output.shape),
        )

        return attn_output

    return patched_forward


def _make_verl_deltanet_patch(original_forward: Any) -> Any:
    """Create a patched forward for verl's ParallelQwen3_6GatedDeltaNetRmPad.

    Supports two-pass prefix-sharing for DeltaNet (GatedDeltaNet) layers:

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