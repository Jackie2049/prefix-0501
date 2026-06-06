"""Prefix-sharing integration for verl's Qwen3.6 model classes.

When verl uses its own model classes (ParallelQwen3_6AttentionRmPad,
ParallelQwen3_6GatedDeltaNetRmPad), this module provides prefix-sharing
hooks that intercept the attention forward pass to expand KV with
prefix-sharing data.

For full attention layers (16 out of 64 in Qwen3.6-27B):
- Intercept after q/k/v projection and before flash_attn
- Apply q_norm/k_norm (per-head RMSNorm) and partial RoPE
- Expand KV with prefix-sharing data
- Run flash_attn_varlen_func with expanded KV
- Apply output gate (attn_output * sigmoid(gate))
- Run o_proj (RowParallelLinear)

For DeltaNet layers (48 out of 64):
- These use recurrent state (GatedDeltaNet), not KV-based attention
- Prefix-sharing for DeltaNet would require state injection, which
  is more complex. For now, we skip PS for DeltaNet layers.
- DeltaNet layers don't have traditional KV cache, so the main
  prefix-sharing benefit is from the 16 full attention layers.
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
        cos, sin = cos[:, :cos.shape[1] // 2], sin[:, sin.shape[1] // 2]

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
                -1, -1, num_key_value_groups, -1, -1
            ).reshape(total_nnz, attn_module.num_heads_per_tp, attn_module.head_dim)
            value_states = value_states.unsqueeze(2).expand(
                -1, -1, num_key_value_groups, -1, -1
            ).reshape(total_nnz, attn_module.num_heads_per_tp, attn_module.head_dim)

        # === Prefix-sharing KV expansion ===
        backend = ctx.backend or TorchReferenceBackend()
        _, tp_rank, tp_size = _read_parallel_rank_info()

        packed_batch_layout = ctx.packed_batch_layout
        expanded_key, expanded_value = backend.build_kv(
            key_states,
            value_states,
            ctx.store,
            ctx.prefix_sharing_plan,
            packed_batch_layout=packed_batch_layout,
            layer_id=layer_id,
            tp_rank=tp_rank,
        )

        # Build expanded cu_seqlens for flash_attn
        expanded_cu_seqlens = torch.tensor(
            [0] + list(ctx.prefix_sharing_plan.expanded_lengths_kv),
            device=cu_seqlens.device,
            dtype=cu_seqlens.dtype,
        ).cumsum(0)
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

    DeltaNet layers use recurrent state (GatedDeltaNet), not KV-based attention.
    Prefix-sharing for DeltaNet would require injecting accumulated state from
    the prefix computation into the suffix computation.

    For now, we pass through to the original forward when prefix-sharing is
    active for DeltaNet layers, since the main benefit of prefix-sharing comes
    from the 16 full attention layers which DO have KV cache.
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
        # DeltaNet prefix-sharing is not implemented yet
        # Just run the original forward
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
            logger.info("Patched ParallelQwen3_6GatedDeltaNetRmPad.forward (passthrough)")
        except ImportError:
            logger.warning("Could not import ParallelQwen3_6GatedDeltaNetRmPad, skipping")

        return mgr.handle()

    @staticmethod
    def _ensure_verl_importable() -> None:
        try:
            import verl  # noqa: F401
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("verl is not importable in this environment") from exc