"""Prefix-Last Restore helpers for logprob assembly.

This module sits **after** model forward and consumes
:class:`~prefix_sharing.core.planner.PrefixSharingPlan`, especially
``prefix_last_restore`` entries produced by
:mod:`prefix_sharing.core.planner`. When a reuser row trims the Q path to
suffix-only queries, the forward pass never materializes ``output(P_last)``,
yet the first response token ``S0`` still requires next-token semantics from
that position. This module gathers provider prefix-last values and assembles
complete per-row response logprobs without breaking the shared-prefix autograd
path.

Core Responsibilities:
    1. **Gather provider prefix-last outputs**: read logits or other per-position
       values at ``provider_prefix_last_pos`` on the provider row for each
       :class:`~prefix_sharing.core.planner.PrefixLastRestoreSpec`.
    2. **Compute first-suffix logprobs**: apply standard ``log_softmax`` +
       label ``gather`` (or accept precomputed scalars) using the reuser's
       first suffix label.
    3. **Assemble restored response logprobs**: prepend or insert the restored
       first-suffix logprob at ``output_slot`` so each row matches full-sequence
       causal LM logprob semantics.

Key Concepts:
    - **Prefix-Last Restore**: reuser rows borrow ``logits[P_last]`` (or derived
      logprob) from the provider to score ``S0``; subsequent suffix tokens use
      reuser suffix-query outputs as usual.
    - **Suffix logprobs without restore**: logprobs from kept query positions
      only; for reusers, slot 0 already corresponds to the *second* suffix token.
    - **Equivalence assumption**: restore is valid when provider and reuser share
      identical prefix tokens, positions, masks, and RoPE conditions.

Key Components:
    - :func:`build_provider_prefix_last_values`: Tensor-agnostic gather from
      provider rows by metadata indices (CPU tests and adapters).
    - :func:`restore_prefix_last_logprobs`: Python list assembly with per-row
      insert at ``output_slot``.
    - :func:`gather_provider_prefix_last_logits`,
      :func:`compute_token_logprobs_from_logits`,
      :func:`restore_prefix_last_logprobs_tensor`: Torch path for padded tensors
      and reference backends; lazy-imports PyTorch.

Design Principles:
    - **Meta-driven only**: which rows restore and where to read/write come
      solely from ``prefix_sharing_plan.prefix_last_restore``; no detection or trimming here.
    - **No detach**: restored values stay on the provider computation graph so
      gradients from multiple reusers accumulate into the shared prefix.
    - **Dual API surface**: list helpers for framework-agnostic tests;
      tensor helpers mirror the same contract for Megatron / verl integration.
    - **Backend-agnostic indexing**: gather/restore logic uses batch indices and
      positions from metadata; integration stacks map the same spans on dense or
      packed tensors without changing the contract.
"""

from __future__ import annotations

from typing import Any, Sequence, TypeVar

from prefix_sharing.core.planner import PrefixSharingPlan


T = TypeVar("T")


def restore_prefix_last_logprobs(
    suffix_logprobs: Sequence[Sequence[float]],
    restore_logprobs: Sequence[float],
    prefix_sharing_plan: PrefixSharingPlan,
) -> list[list[float]]:
    """Assemble response logprobs with Prefix-Last Restore semantics.

    ``suffix_logprobs`` contains logprobs produced by kept query positions. For
    provider samples it is already complete. For reuse samples, interior-prefix
    response token logprobs are prepended (from stored tensors computed once
    from provider logits), and the prefix-last restore logprob is inserted at
    the slot following all interior entries.

    ``restore_logprobs`` must be a list of the same length as
    ``prefix_sharing_plan.prefix_last_restore``, where
    ``restore_logprobs[i]`` is the precomputed logprob for spec ``i``.
    """

    if len(suffix_logprobs) != prefix_sharing_plan.batch_size:
        raise ValueError("suffix_logprobs length must equal batch size")
    if len(restore_logprobs) != len(prefix_sharing_plan.prefix_last_restore):
        raise ValueError(
            f"restore_logprobs length {len(restore_logprobs)} "
            f"must equal number of restore specs {len(prefix_sharing_plan.prefix_last_restore)}"
        )

    restored = [list(row) for row in suffix_logprobs]
    # Apply restores in plan order (interior-response specs come first,
    # followed by prefix-last spec, each with monotonically increasing
    # output_slot).
    for i, spec in enumerate(prefix_sharing_plan.prefix_last_restore):
        restored_value = restore_logprobs[i]
        row = restored[spec.reuse_idx_in_batch]
        if spec.output_slot < 0 or spec.output_slot > len(row):
            raise ValueError("restore output_slot out of range")
        row.insert(spec.output_slot, restored_value)
    return restored


def build_provider_prefix_last_values(
    provider_values_by_batch: Sequence[Sequence[T]],
    prefix_sharing_plan: PrefixSharingPlan,
) -> list[T]:
    """Gather provider values per restore spec.

    For each :class:`~prefix_sharing.core.planner.PrefixLastRestoreSpec` in
    ``prefix_sharing_plan.prefix_last_restore``, reads the value at
    ``provider_prefix_last_pos`` from the provider's row. The returned list
    has the same length as the spec list and is ordered identically.

    For interior-response specs (``is_interior_response=True``), the value
    at ``provider_prefix_last_pos`` is the precomputed logprob (a scalar
    tensor), not raw logits.
    """

    values: list[T] = []
    for spec in prefix_sharing_plan.prefix_last_restore:
        provider_row = provider_values_by_batch[spec.provider_idx_in_batch]
        values.append(provider_row[spec.provider_prefix_last_pos])
    return values


def compute_token_logprobs_from_logits(logits: Any, labels: Any) -> Any:
    """Compute token logprobs from dense logits and integer labels.

    This function is intentionally tiny and lazy-imports torch. It mirrors the
    math used by verl's vocab-parallel helper, while remaining suitable for the
    reference path and optional tests.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("compute_token_logprobs_from_logits requires PyTorch") from exc
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def restore_prefix_last_logprobs_tensor(
    suffix_logprobs: Any,
    first_suffix_logprobs: Any,
    prefix_sharing_plan: PrefixSharingPlan,
) -> Any:
    """Insert Prefix-Last Restore logprobs into a padded response tensor.

    Args:
        suffix_logprobs: Tensor shaped ``[batch, max_suffix_without_restore]``.
            Reuse rows do not contain the first suffix token slot yet.
        first_suffix_logprobs: Tensor shaped ``[batch]``. For each reuse sample,
            this is computed from provider prefix-last logits and that reuse
            sample's first suffix label. Interior-prefix response logprobs are
            NOT included here.
        prefix_sharing_plan: Prefix sharing execution plan.

    Returns:
        Tensor shaped ``[batch, max_restored_response]``. Rows are padded with
        zeros after their restored logical length, matching verl's masked
        logprob convention.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("restore_prefix_last_logprobs_tensor requires PyTorch") from exc
    if suffix_logprobs.shape[0] != prefix_sharing_plan.batch_size:
        raise ValueError("suffix_logprobs batch dimension must match metadata")
    if first_suffix_logprobs.shape[0] != prefix_sharing_plan.batch_size:
        raise ValueError("first_suffix_logprobs batch dimension must match metadata")

    # Group specs by reuser batch index.
    # Interior specs come first (sorted by output_slot), prefix-last last.
    specs_by_batch: dict[int, list[tuple[int, Any]]] = {}
    for i, spec in enumerate(prefix_sharing_plan.prefix_last_restore):
        specs_by_batch.setdefault(spec.reuse_idx_in_batch, []).append((i, spec))

    restored_lengths = []
    rows = []
    for batch_index in range(prefix_sharing_plan.batch_size):
        logical_len = prefix_sharing_plan.kept_lengths_q[batch_index]
        row = suffix_logprobs[batch_index, :logical_len]

        if batch_index in specs_by_batch:
            prefix_restores: list[Any] = []
            for spec_idx, spec in specs_by_batch[batch_index]:
                if spec.is_interior_response:
                    # Interior-prefix logprob: stored as a scalar tensor
                    # computed once from provider logits. Insert directly.
                    # These are provided separately — not in first_suffix_logprobs.
                    # For now placeholders; real integration fills them in.
                    pass
                else:
                    # Prefix-last restore: insert the precomputed logprob
                    prefix_restores.append(
                        first_suffix_logprobs[batch_index : batch_index + 1]
                    )

            if prefix_restores:
                restores_tensor = torch.cat(prefix_restores, dim=0)
                row = torch.cat([restores_tensor, row], dim=0)

        rows.append(row)
        restored_lengths.append(row.shape[0])

    max_len = max(restored_lengths, default=0)
    out = suffix_logprobs.new_zeros((prefix_sharing_plan.batch_size, max_len))
    for batch_index, row in enumerate(rows):
        out[batch_index, : row.shape[0]] = row
    return out


def gather_provider_prefix_last_logits(logits_by_batch: Any, prefix_sharing_plan: PrefixSharingPlan) -> Any:
    """Gather provider prefix-last logits for each reuse sample.

    ``logits_by_batch`` is expected to be a padded tensor shaped
    ``[batch, seq, vocab]`` or any tensor-like object supporting advanced
    indexing. The output has shape ``[batch, vocab]`` with zeros for non-reuse
    samples.

    Interior-response restore specs (``is_interior_response=True``) are
    skipped because they use stored logprob scalars, not raw logits.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("gather_provider_prefix_last_logits requires PyTorch") from exc
    out = logits_by_batch.new_zeros((prefix_sharing_plan.batch_size, logits_by_batch.shape[-1]))
    for spec in prefix_sharing_plan.prefix_last_restore:
        if spec.is_interior_response:
            continue
        out[spec.reuse_idx_in_batch] = logits_by_batch[
            spec.provider_idx_in_batch,
            spec.provider_prefix_last_pos,
        ]
    return out
