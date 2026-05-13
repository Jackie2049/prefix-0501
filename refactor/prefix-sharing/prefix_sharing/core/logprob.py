"""Prefix-Last Restore helpers for logprob assembly."""

from __future__ import annotations

from typing import Any

from prefix_sharing.core.metadata import PrefixSharingBatchMeta


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
    meta: PrefixSharingBatchMeta,
) -> Any:
    """Insert Prefix-Last Restore logprobs into a padded response tensor.

    Args:
        suffix_logprobs: Tensor shaped ``[batch, max_suffix_without_restore]``.
            Reuse rows do not contain the first suffix token slot yet.
        first_suffix_logprobs: Tensor shaped ``[batch]``. For each reuse sample,
            this is computed from provider prefix-last logits and that reuse
            sample's first suffix label.
        meta: Prefix-sharing metadata.

    Returns:
        Tensor shaped ``[batch, max_restored_response]``. Rows are padded with
        zeros after their restored logical length, matching verl's masked
        logprob convention.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("restore_prefix_last_logprobs_tensor requires PyTorch") from exc
    if suffix_logprobs.shape[0] != meta.batch_size:
        raise ValueError("suffix_logprobs batch dimension must match metadata")
    if first_suffix_logprobs.shape[0] != meta.batch_size:
        raise ValueError("first_suffix_logprobs batch dimension must match metadata")

    restored_lengths = []
    rows = []
    restore_by_batch = {spec.reuse_idx_in_batch: spec for spec in meta.prefix_last_restore}
    for batch_index in range(meta.batch_size):
        logical_len = meta.kept_lengths_q[batch_index]
        row = suffix_logprobs[batch_index, :logical_len]
        if batch_index in restore_by_batch:
            row = torch.cat([first_suffix_logprobs[batch_index : batch_index + 1], row], dim=0)
        rows.append(row)
        restored_lengths.append(row.shape[0])

    max_len = max(restored_lengths, default=0)
    out = suffix_logprobs.new_zeros((meta.batch_size, max_len))
    for batch_index, row in enumerate(rows):
        out[batch_index, : row.shape[0]] = row
    return out


def gather_provider_prefix_last_logits(logits_by_batch: Any, meta: PrefixSharingBatchMeta) -> Any:
    """Gather provider prefix-last logits for each reuse sample.

    ``logits_by_batch`` is expected to be a padded tensor shaped
    ``[batch, seq, vocab]`` or any tensor-like object supporting advanced
    indexing. The output has shape ``[batch, vocab]`` with zeros for non-reuse
    samples.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("gather_provider_prefix_last_logits requires PyTorch") from exc
    out = logits_by_batch.new_zeros((meta.batch_size, logits_by_batch.shape[-1]))
    for spec in meta.prefix_last_restore:
        out[spec.reuse_idx_in_batch] = logits_by_batch[
            spec.provider_idx_in_batch,
            spec.provider_prefix_last_pos,
        ]
    return out
