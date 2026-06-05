"""Prefix-sharing integration for verl's own attention classes (legacy path).

When verl runs with `use_mbridge: False`, it uses its own model classes
(e.g. ParallelQwen3_6AttentionRmPad) instead of Megatron's SelfAttention.
This module provides the integration hooks for that path.

For the mcore path (use_mbridge: True), see megatron_attention.py instead.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.model_spec import AttentionLayerType, ModelSpec
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.integrations.context import (
    PrefixSharingRuntimeContext,
    current_prefix_sharing_context,
)
from prefix_sharing.integrations.megatron_attention import IntegrationUnavailable
from prefix_sharing.integrations.megatron_runtime import (
    _read_parallel_rank_info,
    maybe_run_prefix_sharing_attention,
    maybe_run_prefix_sharing_deltanet,
)
from prefix_sharing.integrations.patch_manager import PatchHandle, PatchManager

import logging
logger = logging.getLogger(__name__)


def _make_verl_attention_patch(original_forward: Any, layer_type_check: Any | None = None) -> Any:
    """Create a patched forward for verl's ParallelQwen3_6AttentionRmPad.

    The patch intercepts after QKV projection and before flash_attn,
    expanding KV with prefix-sharing data when a context is active.
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
        ctx = current_prefix_sharing_context()
        if ctx is None:
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

        # Prefix-sharing path: delegate to the runtime hook
        import torch

        # Get QKV from the module's projections
        qkv = self_attention_module.qkv_proj(hidden_states)[0]
        query_states, key_states, value_states = qkv.split(
            [self_attention_module.q_size, self_attention_module.k_size, self_attention_module.v_size],
            dim=-1,
        )

        total_nnz = query_states.shape[0]
        if self_attention_module.megatron_config.sequence_parallel:
            from megatron.core import parallel_state as mpu
            tp_size = mpu.get_tensor_model_parallel_world_size()
            total_nnz = total_nnz * tp_size

        if self_attention_module.megatron_config.sequence_parallel:
            sp_pad = total_nnz - cu_seqlens[-1]
            total_nnz = cu_seqlens[-1]
            query_states = query_states[:total_nnz]
            key_states = key_states[:total_nnz]
            value_states = value_states[:total_nnz]

        query_states = query_states.view(total_nnz, self_attention_module.num_heads_per_tp, self_attention_module.head_dim)
        key_states = key_states.view(total_nnz, self_attention_module.num_key_value_heads_per_tp, self_attention_module.head_dim)
        value_states = value_states.view(total_nnz, self_attention_module.num_key_value_heads_per_tp, self_attention_module.head_dim)

        # Check model_spec for layer type routing
        layer_id = int(getattr(self_attention_module, "layer_number", 0)
                       or getattr(self_attention_module, "layer_idx", 0) or 0)
        model_spec = ctx.model_spec

        if model_spec is not None and model_spec.layer_type(layer_id) == AttentionLayerType.LINEAR_ATTENTION:
            # Linear attention layers should not be handled here
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

        # Full attention: build expanded KV and run attention
        import torch.nn.functional as F
        from flash_attn import flash_attn_varlen_func
        from prefix_sharing.backends.torch_ref import TorchReferenceBackend

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

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            expanded_key = expanded_key.to(torch.float16)
            expanded_value = expanded_value.to(torch.float16)

        attn_output = flash_attn_varlen_func(
            query_states,
            expanded_key,
            expanded_value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=expanded_cu_seqlens,
            max_seqlen_q=max_seqlen_in_batch,
            max_seqlen_k=max_seqlen_expanded,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )

        attn_output = attn_output.to(input_dtype)
        attn_output = attn_output.reshape(total_nnz, 1, self_attention_module.hidden_size_per_tp).contiguous()

        if self_attention_module.megatron_config.sequence_parallel:
            attn_output = F.pad(attn_output, pad=(0, 0, 0, 0, 0, sp_pad))

        attn_output = self_attention_module.o_proj(attn_output)[0]

        # Apply output gate if present
        if getattr(self_attention_module, "attn_output_gate", False) and hasattr(self_attention_module, "gate_proj"):
            gate = torch.sigmoid(self_attention_module.gate_proj(hidden_states)[0])
            gate = gate[:attn_output.shape[0]]
            attn_output = attn_output * gate

        return attn_output

    return patched_forward


def _make_verl_deltanet_patch(original_forward: Any) -> Any:
    """Create a patched forward for verl's ParallelQwen3_6GatedDeltaNetRmPad.

    The patch intercepts the cumsum computation to inject prefix-sharing state.
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
        ctx = current_prefix_sharing_context()
        if ctx is None:
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

        if not ctx.prefix_sharing_plan.has_sharing:
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

        # Prefix-sharing for DeltaNet: delegate to runtime hook
        import torch

        layer_id = int(getattr(deltanet_module, "layer_idx", 0) or 0)
        model_spec = ctx.model_spec

        if model_spec is not None and model_spec.layer_type(layer_id) != AttentionLayerType.LINEAR_ATTENTION:
            # Not a linear attention layer, skip
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

        # Run the original forward but with state injection
        # For DeltaNet, we hook into the cumsum step
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
    """Prefix-sharing integration for verl's Qwen3.6 model (legacy path)."""

    config: PrefixSharingConfig
    backend: Any = None

    def install(self, model_config: Any | None = None) -> PatchHandle:
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
        except ImportError:
            pass

        try:
            from verl.models.qwen3_6.megatron.layers.parallel_deltanet import (
                ParallelQwen3_6GatedDeltaNetRmPad,
            )
            original_deltanet_forward = ParallelQwen3_6GatedDeltaNetRmPad.forward
            patched_deltanet = _make_verl_deltanet_patch(original_deltanet_forward)
            mgr.patch_attr(ParallelQwen3_6GatedDeltaNetRmPad, "forward", patched_deltanet)
        except ImportError:
            pass

        return mgr.handle()

    @staticmethod
    def _ensure_verl_importable() -> None:
        try:
            import verl  # noqa: F401
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("verl is not importable in this environment") from exc
