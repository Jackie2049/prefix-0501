"""Flash Attention common abstraction layer.

This module provides shared input-normalization logic for GPU and NPU Flash
Attention backends.  It converts the packed THD tensors produced by the
Megatron hook into the shape/dtype/device layout expected by Flash Attention
kernels, while keeping hardware-specific dispatch in the concrete backends.
"""

from __future__ import annotations

from typing import Any

import torch

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.batch_layout import BshdBatchLayout, ThdBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


class FlashBackendValidationError(RuntimeError):
    """Raised when a flash-attention backend cannot run with the given plan."""


class FlashAttentionMixin:
    """Shared helpers for GPU / NPU Flash Attention backends.

    Concrete subclasses must still define ``capabilities``, ``validate()``,
    ``apply_rope()``, ``build_kv()`` and ``attention()``; this mixin only
    normalises the tensor shapes for the ``attention()`` step.
    """

    capabilities: BackendCapabilities

    # ------------------------------------------------------------------
    # Shared validation
    # ------------------------------------------------------------------
    def _validate_plan_for_flash(self, prefix_sharing_plan: PrefixSharingPlan) -> None:
        """Common checks before calling any Flash Attention kernel."""
        if prefix_sharing_plan.cu_seqlens_q is None or prefix_sharing_plan.cu_seqlens_kv is None:
            raise FlashBackendValidationError(
                "Flash Attention requires cu_seqlens_q and cu_seqlens_kv; "
                "got None. Make sure PrefixSharingPlan was built with packed THD metadata."
            )
        if len(prefix_sharing_plan.cu_seqlens_q) != prefix_sharing_plan.batch_size + 1:
            raise FlashBackendValidationError(
                f"cu_seqlens_q length ({len(prefix_sharing_plan.cu_seqlens_q)}) must be "
                f"batch_size + 1 ({prefix_sharing_plan.batch_size + 1})"
            )
        if len(prefix_sharing_plan.cu_seqlens_kv) != prefix_sharing_plan.batch_size + 1:
            raise FlashBackendValidationError(
                f"cu_seqlens_kv length ({len(prefix_sharing_plan.cu_seqlens_kv)}) must be "
                f"batch_size + 1 ({prefix_sharing_plan.batch_size + 1})"
            )

    # ------------------------------------------------------------------
    # Tensor normalisation helpers
    # ------------------------------------------------------------------
    def _ensure_3d_thd(self, tensor: Any, name: str) -> Any:
        """Ensure packed tensor is (total_tokens, num_heads, head_dim).

        The Megatron hook squeezes the dummy batch dimension before handing
        tensors to the backend, so we expect 3-D inputs.  If 2-D is ever
        received we raise loudily rather than guessing.
        """
        if tensor.dim() == 3:
            return tensor
        if tensor.dim() == 2:
            raise FlashBackendValidationError(
                f"{name} has 2 dims {tuple(tensor.shape)}; Flash Attention expects "
                "(total_tokens, num_heads, head_dim)."
            )
        raise FlashBackendValidationError(
            f"{name} has unexpected rank {tensor.dim()} with shape {tuple(tensor.shape)}"
        )

    def _build_cu_seqlens_tensor(self, lengths: list[int], device: Any, dtype: Any = None) -> Any:
        """Convert a Python list of cumulative lengths to a CUDA/NPU tensor."""
        t = torch.tensor(lengths, device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t

    def _strip_tp_padding(
        self, q: Any, batch_runtime_layout: Any | None
    ) -> tuple[Any, ThdBatchLayout | None]:
        """Strip TP padding from Q and validate the tensor shape.

        Returns ``(q_unpadded, layout)`` where *layout* is the THD runtime
        layout when padding was stripped, or ``None`` when no stripping was
        needed.  The caller must pass *layout* to :meth:`_repad_output`.
        """
        if getattr(batch_runtime_layout, "layout_kind", "thd") != "thd":
            return q, None
        if not hasattr(batch_runtime_layout, "unpad") or not hasattr(batch_runtime_layout, "repad"):
            return q, None
        if not batch_runtime_layout.has_padding:
            return q, None

        # Defensive: ensure the padded Q tensor shape matches the layout.
        expected_len = batch_runtime_layout.total_padded_length
        if q.shape[0] != expected_len:
            raise FlashBackendValidationError(
                f"Query total_tokens={q.shape[0]} does not match "
                f"batch_runtime_layout total_padded_length={expected_len} "
                f"(padded_lengths={batch_runtime_layout.padded_lengths})"
            )

        q_unpadded = batch_runtime_layout.unpad(q)
        return q_unpadded, batch_runtime_layout

    def _prepare_flash_inputs(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        attention_mask: Any | None = None,
        batch_runtime_layout: Any | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, int, int, ThdBatchLayout | None]:
        """Normalise inputs for ``flash_attn_varlen_func``.

        When *batch_runtime_layout* is THD and contains TP-induced padding
        (``has_padding`` is True, i.e. TP > 1), the padding
        tokens are **stripped from Q** before the FA call so that the
        kernel uses the semantic (unpadded) ``cu_seqlens`` where the
        causal alignment is correct.  The padding is re-applied to the
        output by the caller via :meth:`_repad_output`.

        K/V tensors are already de-padded from :meth:`build_kv` and
        follow ``prefix_sharing_plan.expanded_lengths_kv``.

        Returns
        -------
        q, k, v : (total_tokens, num_heads, head_dim)  — Q is de-padded when layout has padding
        cu_seqlens_q : (batch_size + 1,)
        cu_seqlens_kv : (batch_size + 1,)
        max_seqlen_q : int
        max_seqlen_kv : int
        pad_layout : ThdBatchLayout | None
            When not None the caller must re-pad the FA output via
            :meth:`_repad_output`.
        """
        self._validate_plan_for_flash(prefix_sharing_plan)

        q = self._ensure_3d_thd(query, "query")
        k = self._ensure_3d_thd(key, "key")
        v = self._ensure_3d_thd(value, "value")

        # ---- Detect & strip TP padding ----
        q, pad_layout = self._strip_tp_padding(q, batch_runtime_layout)

        device = q.device

        # ---- Q cu_seqlens & max_seqlen ----
        # After de-padding, Q follows the plan's semantic kept_lengths_q.
        cu_seqlens_q = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_q, device=device, dtype=torch.int32
        )
        max_seqlen_q = prefix_sharing_plan.max_seqlen_q

        # ---- KV cu_seqlens & max_seqlen ----
        # K/V have already been de-padded and prefix-expanded by build_kv();
        # they follow the plan's semantic expanded_lengths_kv without any TP
        # alignment padding.
        cu_seqlens_kv = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_kv, device=device, dtype=torch.int32
        )
        max_seqlen_kv = prefix_sharing_plan.max_seqlen_kv

        # Flash Attention ignores dense attention_mask; if one is passed we
        # simply drop it because cu_seqlens already encodes the per-sample
        # boundaries.  This is consistent with Megatron's THD path.
        _ = attention_mask

        return q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, pad_layout

    def _prepare_bshd_varlen_flash_inputs(
        self,
        query: Any,
        key_rows: Any,
        value_rows: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        batch_runtime_layout: BshdBatchLayout,
    ) -> tuple[Any, Any, Any, Any, Any, int, int]:
        """Convert BSHD runtime tensors to valid-only THD tensors for varlen FA."""
        self._validate_plan_for_flash(prefix_sharing_plan)
        if getattr(batch_runtime_layout, "layout_kind", None) != "bshd":
            raise FlashBackendValidationError("expected BshdBatchLayout for BSHD varlen flash inputs")
        if not isinstance(key_rows, list) or not isinstance(value_rows, list):
            raise FlashBackendValidationError("BSHD Flash Attention expects per-row key/value lists")
        if len(key_rows) != batch_runtime_layout.batch_size or len(value_rows) != batch_runtime_layout.batch_size:
            raise FlashBackendValidationError(
                "BSHD key/value row count must match batch_runtime_layout.batch_size"
            )

        q_rows = [
            batch_runtime_layout.valid_tokens(query, seq_idx_in_batch)
            for seq_idx_in_batch in range(batch_runtime_layout.batch_size)
        ]
        self._validate_bshd_row_lengths(
            q_rows,
            batch_runtime_layout.valid_lengths,
            "query",
        )
        self._validate_bshd_row_lengths(
            key_rows,
            prefix_sharing_plan.expanded_lengths_kv,
            "key",
        )
        self._validate_bshd_row_lengths(
            value_rows,
            prefix_sharing_plan.expanded_lengths_kv,
            "value",
        )

        q = torch.cat(q_rows, dim=0) if q_rows else query.new_empty((0, *query.shape[-2:]))
        k = torch.cat(key_rows, dim=0) if key_rows else query.new_empty((0, *query.shape[-2:]))
        v = torch.cat(value_rows, dim=0) if value_rows else query.new_empty((0, *query.shape[-2:]))

        q = self._ensure_3d_thd(q, "query")
        k = self._ensure_3d_thd(k, "key")
        v = self._ensure_3d_thd(v, "value")

        cu_seqlens_q = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_q,
            device=q.device,
            dtype=torch.int32,
        )
        cu_seqlens_kv = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_kv,
            device=q.device,
            dtype=torch.int32,
        )
        return (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            prefix_sharing_plan.max_seqlen_q,
            prefix_sharing_plan.max_seqlen_kv,
        )

    def _scatter_bshd_varlen_output(
        self,
        output: Any,
        query: Any,
        batch_runtime_layout: BshdBatchLayout,
    ) -> Any:
        """Scatter valid-only THD output back to the BSHD runtime tensor shape."""
        output = self._ensure_3d_thd(output, "output")
        if output.shape[0] != batch_runtime_layout.total_valid_length:
            raise FlashBackendValidationError(
                f"BSHD flash output length={output.shape[0]} does not match "
                f"total_valid_length={batch_runtime_layout.total_valid_length}"
            )
        dense_output = torch.zeros_like(query)
        offset = 0
        for seq_idx_in_batch, valid_length in enumerate(batch_runtime_layout.valid_lengths):
            next_offset = offset + valid_length
            if valid_length > 0:
                batch_runtime_layout.write_valid_tokens(
                    dense_output,
                    seq_idx_in_batch,
                    output[offset:next_offset],
                )
            offset = next_offset
        return dense_output

    def _validate_bshd_row_lengths(self, rows: list[Any], lengths: list[int], name: str) -> None:
        """Validate per-row BSHD tensors before concatenating or padding them."""
        if len(rows) != len(lengths):
            raise FlashBackendValidationError(f"{name} row count does not match lengths")
        for row_index, (row, expected_length) in enumerate(zip(rows, lengths)):
            if row.shape[0] != expected_length:
                raise FlashBackendValidationError(
                    f"{name} row {row_index} length={row.shape[0]} does not match expected={expected_length}"
                )
            if row.dim() != 3:
                raise FlashBackendValidationError(
                    f"{name} row {row_index} must be 3D (tokens, heads, dim), got shape={tuple(row.shape)}"
                )

    def _repad_output(
        self,
        output: Any,
        pad_layout: ThdBatchLayout,
    ) -> Any:
        """Re-apply TP padding to the FA output so it matches the original Q shape.

        Delegates to :meth:`ThdBatchLayout.repad`.
        """
        return pad_layout.repad(output)
