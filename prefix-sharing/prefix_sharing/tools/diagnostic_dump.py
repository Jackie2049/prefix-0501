"""Unified diagnostic dump for ON/OFF precision comparison.

Single env var ``PREFIX_SHARING_DIAG_DUMP`` controls the dump directory.
Supports two dump modes:

1. Packed format (attention output / logits):
   - ``attn_output.pt``  — last-layer attention output  [N, 1, hidden] or [N, hidden]
   - ``logits.pt``       — model output logits            [N, vocab//tp]
   - Metadata: ``cu_seqlens_q.pt`` (attn) / ``cu_seqlens_q_logits.pt`` (logits),
     ``prefix_lens.pt`` (shared)

2. 2D format (logprobs / entropy):
   - ``logprobs_{tag}.pt``  — 2D log probabilities  [B, L]
   - ``entropy_{tag}.pt``   — 2D entropy             [B, L]
   - Metadata: ``label.pt``, ``label_mask.pt``

Usage:
    export PREFIX_SHARING_DIAG_DUMP=/path/to/dump_on   # ON  run
    export PREFIX_SHARING_DIAG_DUMP=/path/to/dump_off  # OFF run
    # Then compare with cmp_diag.py
"""

from __future__ import annotations

import os
from typing import Any

import torch

# ── Internal state ──────────────────────────────────────────────
_DUMP_DIR: str | None = None
_META_SAVED: set[str] = set()  # set of saved metadata keys (per-run dedup per key)


def _get_dump_dir() -> str | None:
    """Read env var once, create directory on first call (cached)."""
    global _DUMP_DIR
    if _DUMP_DIR is not None:
        return _DUMP_DIR
    path = os.environ.get("PREFIX_SHARING_DIAG_DUMP")
    if path:
        _DUMP_DIR = path
        os.makedirs(path, exist_ok=True)
    return _DUMP_DIR


def _rank0_only() -> bool:
    """Returns True on rank 0 (or if distributed is not initialized)."""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass
    return True


# ── Generic helpers ─────────────────────────────────────────────

def _save_tensor(name: str, tensor: torch.Tensor, dump_dir: str) -> None:
    """Save a tensor to dump_dir/name.pt, rank-0-only with try/except."""
    if not _rank0_only():
        return
    import logging
    _log = logging.getLogger(__name__)
    try:
        path = os.path.join(dump_dir, name)
        torch.save(tensor.detach().cpu().clone(), path)
        _log.warning("%s saved (%s)", name, tensor.shape)
    except Exception as e:
        _log.warning("%s save failed: %s", name, e)


def _save_meta(packed_seq_params: Any,
               prefix_lens_list: list[int],
               dump_dir: str,
               meta_key: str = "attn") -> None:
    """Save cu_seqlens + prefix_lens (dedup per meta_key across one run).

    meta_key determines the filename suffix:
      - "attn"  → cu_seqlens_q_attn.pt
      - "logits" → cu_seqlens_q_logits.pt
    prefix_lens.pt is shared (same across dump types within a run).
    """
    global _META_SAVED
    cu_key = f"cu_{meta_key}"
    pl_key = "pl"
    if not _rank0_only():
        return
    import logging
    _log = logging.getLogger(__name__)
    # cu_seqlens (per dump-type, separate files)
    if cu_key not in _META_SAVED:
        try:
            cu = packed_seq_params.cu_seqlens_q_padded.detach().cpu().clone()
            cu_fname = f"cu_seqlens_q_{meta_key}.pt" if meta_key != "attn" else "cu_seqlens_q.pt"
            torch.save(cu, os.path.join(dump_dir, cu_fname))
            _log.warning("%s saved (%s)", cu_fname, cu.shape)
            _META_SAVED.add(cu_key)
        except Exception as e:
            _log.warning("cu_seqlens (%s) save failed: %s", meta_key, e)
    # prefix_lens (shared, once per run)
    if pl_key not in _META_SAVED:
        try:
            pl = torch.tensor(prefix_lens_list, dtype=torch.int32)
            torch.save(pl, os.path.join(dump_dir, "prefix_lens.pt"))
            _log.warning("prefix_lens.pt saved (%s)", pl.shape)
            _META_SAVED.add(pl_key)
        except Exception as e:
            _log.warning("prefix_lens.pt save failed: %s", e)


# ── Packed format: attention output ────────────────────────────

def dump_attn_on(
    output_tensor: torch.Tensor,
    packed_seq_params: Any,
    prefix_sharing_plan: Any,
    layer_number: int,
    num_layers: int,
) -> None:
    """Called from ON-mode attention hook (last layer only)."""
    if layer_number != num_layers:
        return
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("attn_output.pt", output_tensor, dump_dir)
    _save_meta(packed_seq_params,
               list(prefix_sharing_plan.prefix_lens), dump_dir,
               meta_key="attn")


def dump_attn_off(
    output_tensor: torch.Tensor,
    packed_seq_params: Any,
    layer_number: int,
    batch_size: int,
    num_layers: int,
) -> None:
    """Called from OFF-mode attention hook (last layer only)."""
    if layer_number != num_layers:
        return
    if packed_seq_params is None \
       or not hasattr(packed_seq_params, 'cu_seqlens_q_padded'):
        return
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("attn_output.pt", output_tensor, dump_dir)
    _save_meta(packed_seq_params, [0] * batch_size, dump_dir,
               meta_key="attn")


# ── Packed format: logits ──────────────────────────────────────

def dump_logits(
    logits: torch.Tensor,
    packed_seq_params: Any,
    prefix_lens_list: list[int],
) -> None:
    """Dump model output logits [N, vocab//tp]."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("logits.pt", logits, dump_dir)
    _save_meta(packed_seq_params, prefix_lens_list, dump_dir,
               meta_key="logits")


# ── 2D format: logprobs / entropy / label ──────────────────────

def dump_logprobs_legacy(
    log_probs: torch.Tensor,
    input_ids: torch.Tensor,
) -> None:
    """Dump legacy logprobs.pt + input_ids.pt (per-rank)."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    import logging
    _log = logging.getLogger(__name__)
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:
        rank = 0
    try:
        _save_tensor(f"logprobs_rank{rank}.pt",
                     log_probs.clone(), dump_dir)
        _save_tensor(f"input_ids_rank{rank}.pt",
                     input_ids.cpu().clone(), dump_dir)
    except Exception as e:
        _log.warning("legacy logprob dump failed: %s", e)


def dump_logprobs_2d(
    log_probs: torch.Tensor,
    tag: str,
) -> None:
    """Dump 2D logprobs_{tag}.pt."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"logprobs_{tag}.pt", log_probs.detach().clone(), dump_dir)


def dump_entropy_2d(
    entropy: torch.Tensor,
    tag: str,
) -> None:
    """Dump 2D entropy_{tag}.pt."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"entropy_{tag}.pt", entropy.detach().clone(), dump_dir)


def dump_label(
    label: torch.Tensor,
) -> None:
    """Dump label.pt (shared across on/off, tag-agnostic)."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("label.pt", label.detach().cpu().clone(), dump_dir)


def dump_label_mask(
    label_mask: torch.Tensor,
) -> None:
    """Dump label_mask.pt (canonical structural mask, pre-packing)."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("label_mask.pt", label_mask.cpu().clone(), dump_dir)
