"""Unified diagnostic dump for ON/OFF precision comparison.

Single env var ``PREFIX_SHARING_DIAG_DUMP`` controls the dump directory.
Supports two dump modes:

1. Packed format:
   - ``attn_outputs.pt``    — per-layer attention output dict {layer_idx: tensor}
   - ``logits.pt``          — model output logits [B, N, vocab//tp] (vocab=last dim)
   - Metadata: ``cu_seqlens_q.pt`` / ``cu_seqlens_q_logits.pt``, ``prefix_lens.pt``

2. Positional encoding:
   - ``position_ids.pt``    — packed position ids [N]
   - ``rope_emb.pt``        — per-layer RoPE encoding dict
                              {layer_idx: {"query": rotated_q, "key": rotated_k}}

3. 2D format:
   - ``logprobs_{tag}.pt``  — 2D log probabilities [B, L]
   - ``entropy_{tag}.pt``   — 2D entropy [B, L]
   - Metadata: ``label.pt``, ``label_mask.pt``

Usage:
    export PREFIX_SHARING_DIAG_DUMP=/path/to/dump_on   # ON  run
    export PREFIX_SHARING_DIAG_DUMP=/path/to/dump_off  # OFF run
    # Then compare with cmp_diag.py
"""

from __future__ import annotations

import os
import logging
from typing import Any

import torch

_log = logging.getLogger(__name__)

# ── Internal state ──────────────────────────────────────────────
_DUMP_DIR: str | None = None
_META_SAVED: set[str] = set()       # saved metadata keys (dedup per key)
_ATTN_BUFFER: dict[int, torch.Tensor] | None = None  # {layer_idx: tensor}
_ROPE_BUFFER: dict[int, dict] | None = None           # {layer_idx: {"query": q, "key": k}}
_ROPE_FREQS_ON_BUFFER: dict[int, torch.Tensor] | None = None   # {layer_idx: q_freqs per-token}
_ROPE_FREQS_OFF_BUFFER: dict[int, torch.Tensor] | None = None  # {layer_idx: q_pos_emb table}


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


# ── Parallel-topology-aware dumping (TP / SP / PP) ──────────────
#
# Each dumped tensor has a *rank-scope* describing how (if at all) it is split
# across ranks.  The scope decides WHO dumps and under WHAT filename:
#
#   "global"    — identical on every rank that holds it → dumped once by the
#                 representative rank (rank-0 under TP/DP; PP will need
#                 stage-aware routing, left as a per-tensor concern then).
#                 Plain filename, e.g. ``logprobs_old.pt``.
#   "tp_vocab"  — vocab-sharded by TP → EVERY tp rank dumps its own shard as
#                 ``<stem>_tp{r}.pt`` (no cross-rank comm on the dump path;
#                 cmp reassembles offline).  tp_size==1 degrades to the plain
#                 ``logits.pt`` so single-card dumps are unchanged.
#
# Future scopes (reserved, wired when SP/PP land):
#   "tp_seq"    — sequence-sharded by SP (reduce-scatter) → ``_tp{r}``.
#   "pp_stage"  — layer-partitioned by PP → ``_pp{r}``.
#
# The static name→scope map is written to ``parallel_info.json`` (manifest) so
# cmp_diag knows how to gather each tensor without guessing.

_TENSOR_SCOPES: dict[str, str] = {
    "logits": "tp_vocab",   # packed logits [N, V//tp] — vocab-sharded under TP
}

_PARALLEL_INFO_CACHE: Any = None
_MANIFEST_WRITTEN: set[str] = set()


def _cached_parallel_info() -> Any:
    """Read MegatronParallelInfo once and cache (tp/pp/cp ranks + sizes).

    Returns None when distributed/Megatron isn't initialized (single-rank),
    which the callers treat as the single-card case.
    """
    global _PARALLEL_INFO_CACHE
    if _PARALLEL_INFO_CACHE is not None:
        return _PARALLEL_INFO_CACHE
    try:
        from prefix_sharing.integrations.parallel_info import (
            get_megatron_parallel_info,
        )
        _PARALLEL_INFO_CACHE = get_megatron_parallel_info()
    except Exception:
        _PARALLEL_INFO_CACHE = None
    return _PARALLEL_INFO_CACHE


def _with_suffix(name: str, suffix: str) -> str:
    """Insert a rank suffix before the extension: logits.pt → logits_tp0.pt."""
    if not suffix:
        return name
    stem, sep, ext = name.rpartition(".")
    return f"{stem}{suffix}{sep}{ext}" if sep else f"{name}{suffix}"


def _ensure_manifest(dump_dir: str, pi: Any) -> None:
    """Write parallel_info.json once from rank-0: topology + scope map.

    Content is global/identical, so a single writer avoids filesystem races.
    Read by cmp_diag to gather per-rank files (e.g. concat logits_tp{0..tp-1}.pt
    back to full vocab) and to validate ON/OFF topology match.
    """
    if dump_dir in _MANIFEST_WRITTEN:
        return
    _MANIFEST_WRITTEN.add(dump_dir)
    if not _rank0_only():
        return
    import json
    manifest = {
        "tp_size": getattr(pi, "tp_size", 1) if pi else 1,
        "pp_size": getattr(pi, "pp_size", 1) if pi else 1,
        "cp_size": getattr(pi, "cp_size", 1) if pi else 1,
        "global_rank_of_dumper": getattr(pi, "global_rank", 0) if pi else 0,
        "scopes": dict(_TENSOR_SCOPES),
    }
    try:
        with open(os.path.join(dump_dir, "parallel_info.json"),
                  "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        _log.warning("parallel_info.json saved (tp=%d pp=%d)",
                     manifest["tp_size"], manifest["pp_size"])
    except Exception as e:
        _log.warning("parallel_info.json save failed: %s", e)


# ── Generic helpers ─────────────────────────────────────────────

def _save_tensor(name: str, tensor: torch.Tensor, dump_dir: str,
                 scope: str = "global") -> bool:
    """Save a tensor to ``dump_dir/<name>``, scope-aware. Returns True on success.

    See the ``_TENSOR_SCOPES`` block above for the scope semantics.  ``scope``
    defaults to ``"global"`` (rank-0-only, plain filename) so existing callers
    are unchanged; per-rank tensors opt in via ``scope="tp_vocab"`` (etc.).
    """
    pi = _cached_parallel_info()
    _ensure_manifest(dump_dir, pi)

    if scope == "tp_vocab" and pi is not None and pi.tp_size > 1:
        # Every tp rank dumps its own vocab shard; no rank-0 gate, no comm.
        fname = _with_suffix(name, f"_tp{pi.tp_rank}")
    else:
        # global scope, OR tp_vocab with tp_size==1 (single shard == full):
        # rank-0 dumps once under the plain name (single-card compatible).
        if scope == "tp_vocab":
            scope = "global"  # tp==1 → behaves as global for logging
        if not _rank0_only():
            return False
        fname = name
    try:
        path = os.path.join(dump_dir, fname)
        torch.save(tensor.detach().cpu().clone(), path)
        _log.warning("%s saved (%s, scope=%s)", fname, tensor.shape, scope)
        return True
    except Exception as e:
        _log.warning("%s save failed: %s", fname, e)
        return False


def _save_meta(packed_seq_params: Any,
               prefix_lens_list: list[int],
               dump_dir: str,
               meta_key: str = "attn") -> None:
    """Save cu_seqlens + prefix_lens (dedup per meta_key across one run).

    meta_key determines the filename suffix:
      - "attn"  → cu_seqlens_q.pt (or cu_seqlens_q_attn.pt)
      - "logits" → cu_seqlens_q_logits.pt
    prefix_lens.pt is shared (same across dump types within a run).
    """
    global _META_SAVED
    cu_key = f"cu_{meta_key}"
    pl_key = "pl"
    if not _rank0_only():
        return
    try:
        if cu_key not in _META_SAVED:
            cu = packed_seq_params.cu_seqlens_q_padded.detach().cpu().clone()
            cu_fname = "cu_seqlens_q.pt" if meta_key == "attn" else f"cu_seqlens_q_{meta_key}.pt"
            torch.save(cu, os.path.join(dump_dir, cu_fname))
            _log.warning("%s saved (%s)", cu_fname, cu.shape)
            _META_SAVED.add(cu_key)
    except Exception as e:
        _log.warning("cu_seqlens (%s) save failed: %s", meta_key, e)

    try:
        if pl_key not in _META_SAVED:
            pl = torch.tensor(prefix_lens_list, dtype=torch.int32)
            torch.save(pl, os.path.join(dump_dir, "prefix_lens.pt"))
            _log.warning("prefix_lens.pt saved (%s)", pl.shape)
            _META_SAVED.add(pl_key)
    except Exception as e:
        _log.warning("prefix_lens.pt save failed: %s", e)


# ── Per-layer buffer helpers ────────────────────────────────────

def _add_to_attn_buffer(layer_number: int, tensor: torch.Tensor) -> None:
    """Accumulate one layer's attention output into the global buffer."""
    global _ATTN_BUFFER
    if _ATTN_BUFFER is None:
        _ATTN_BUFFER = {}
    _ATTN_BUFFER[layer_number] = tensor.detach().cpu().clone()


def _flush_attn_buffer(dump_dir: str) -> None:
    """Write accumulated attn_outputs dict to disk and clear buffer."""
    global _ATTN_BUFFER
    if _ATTN_BUFFER is None:
        return
    if not _rank0_only():
        _ATTN_BUFFER = None
        return
    try:
        # Move every tensor to CPU (dict has no .detach()/.cpu()/.clone())
        moved = {k: v.detach().cpu().clone() for k, v in _ATTN_BUFFER.items()}
        torch.save(moved, os.path.join(dump_dir, "attn_outputs.pt"))
        _log.warning("attn_outputs.pt saved (%d layers)", len(moved))
        _ATTN_BUFFER = None
    except Exception as e:
        _log.warning("attn_outputs.pt save failed: %s", e)


def _add_to_rope_buffer(layer_number: int, rotated_query: torch.Tensor,
                        rotated_key: torch.Tensor) -> None:
    """Accumulate one layer's RoPE encoding into the global buffer."""
    global _ROPE_BUFFER
    if _ROPE_BUFFER is None:
        _ROPE_BUFFER = {}
    _ROPE_BUFFER[layer_number] = {
        "query": rotated_query.detach().cpu().clone(),
        "key": rotated_key.detach().cpu().clone(),
    }


def _flush_rope_buffer(dump_dir: str) -> None:
    """Write accumulated rope_emb dict to disk and clear buffer."""
    global _ROPE_BUFFER
    if _ROPE_BUFFER is None:
        return
    if not _rank0_only():
        _ROPE_BUFFER = None
        return
    try:
        torch.save(_ROPE_BUFFER, os.path.join(dump_dir, "rope_emb.pt"))
        _log.warning("rope_emb.pt saved (%d layers)", len(_ROPE_BUFFER))
        _ROPE_BUFFER = None
    except Exception as e:
        _log.warning("rope_emb.pt save failed: %s", e)


# ── RoPE angle dump (pre-apply, per-layer) ──────────────────────

def dump_rope_freqs_on(q_freqs: torch.Tensor, layer_number: int,
                       num_layers: int) -> None:
    """Accumulate ON-mode per-token RoPE angles. Auto-flush on last layer.

    ``q_freqs`` is the result of ``q_pos_emb.index_select(0, packed_position_ids)``
    — shape [T_on, 1, 1, D], each token's actual rotation angles **before
    cos/sin**.  Stored as ``rope_freqs_on.pt`` (dict {layer_idx: tensor}).
    """
    global _ROPE_FREQS_ON_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _ROPE_FREQS_ON_BUFFER is None:
        _ROPE_FREQS_ON_BUFFER = {}
    _ROPE_FREQS_ON_BUFFER[layer_number] = q_freqs.detach().cpu().clone()
    if layer_number == num_layers:
        if _rank0_only():
            try:
                torch.save(_ROPE_FREQS_ON_BUFFER,
                           os.path.join(dump_dir, "rope_freqs_on.pt"))
                _log.warning("rope_freqs_on.pt saved (%d layers)",
                             len(_ROPE_FREQS_ON_BUFFER))
            except Exception as e:
                _log.warning("rope_freqs_on.pt save failed: %s", e)
        _ROPE_FREQS_ON_BUFFER = None


def dump_rope_freqs_off(q_pos_emb: torch.Tensor, layer_number: int,
                        num_layers: int) -> None:
    """Accumulate OFF-mode raw RoPE angle table. Auto-flush on last layer.

    ``q_pos_emb`` is the raw angle table (before per-token slicing) —
    shape [L0, 1, 1, D], where ``freqs[p] = p * inv_freq``.
    Stored as ``rope_freqs_off.pt`` (dict {layer_idx: tensor}).
    """
    global _ROPE_FREQS_OFF_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _ROPE_FREQS_OFF_BUFFER is None:
        _ROPE_FREQS_OFF_BUFFER = {}
    _ROPE_FREQS_OFF_BUFFER[layer_number] = q_pos_emb.detach().cpu().clone()
    if layer_number == num_layers:
        if _rank0_only():
            try:
                torch.save(_ROPE_FREQS_OFF_BUFFER,
                           os.path.join(dump_dir, "rope_freqs_off.pt"))
                _log.warning("rope_freqs_off.pt saved (%d layers)",
                             len(_ROPE_FREQS_OFF_BUFFER))
            except Exception as e:
                _log.warning("rope_freqs_off.pt save failed: %s", e)
        _ROPE_FREQS_OFF_BUFFER = None


# ── Position IDs dump ───────────────────────────────────────────

def dump_position_ids(position_ids: torch.Tensor) -> None:
    """Dump packed position_ids [N].

    Call once per forward pass from the integration layer after position_ids
    are available in packed format.
    """
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("position_ids.pt", position_ids, dump_dir)


# ── RoPE encoding dump (per layer) ──────────────────────────────

def dump_rope_emb_layer(layer_number: int, rotated_query: torch.Tensor,
                        rotated_key: torch.Tensor, num_layers: int) -> None:
    """Accumulate one layer's post-RoPE query/key. Auto-flush on last layer.

    Call after RoPE is applied in each attention layer, for both ON and OFF modes.
    """
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _add_to_rope_buffer(layer_number, rotated_query, rotated_key)
    if layer_number == num_layers:
        _flush_rope_buffer(dump_dir)


# ── Packed format: attention output (per layer) ─────────────────

def dump_attn_on(
    output_tensor: torch.Tensor,
    packed_seq_params: Any,
    prefix_sharing_plan: Any,
    layer_number: int,
    num_layers: int,
) -> None:
    """Called from ON-mode attention hook (every layer).

    Accumulates each layer's attention output and saves
    attn_outputs.pt (dict {layer_idx: tensor}) on the last layer.
    """
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return

    # Accumulate per-layer output
    _add_to_attn_buffer(layer_number, output_tensor)

    # Save metadata on first layer
    if layer_number == 1:
        _save_meta(packed_seq_params, list(prefix_sharing_plan.prefix_lens),
                   dump_dir, meta_key="attn")

    # Flush on last layer
    if layer_number == num_layers:
        _flush_attn_buffer(dump_dir)


def dump_attn_off(
    output_tensor: torch.Tensor,
    packed_seq_params: Any,
    layer_number: int,
    batch_size: int,
    num_layers: int,
) -> None:
    """Called from OFF-mode attention hook (every layer).

    Accumulates each layer's attention output.  Metadata uses all-zero prefix_lens.
    """
    if packed_seq_params is None \
       or not hasattr(packed_seq_params, 'cu_seqlens_q_padded'):
        return
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return

    # Accumulate per-layer output
    _add_to_attn_buffer(layer_number, output_tensor)

    # Save metadata on first layer
    if layer_number == 1:
        _save_meta(packed_seq_params, [0] * batch_size, dump_dir, meta_key="attn")

    # Flush on last layer
    if layer_number == num_layers:
        _flush_attn_buffer(dump_dir)


# ── Packed format: logits ──────────────────────────────────────

def dump_logits(
    logits: torch.Tensor,
    packed_seq_params: Any,
    prefix_lens_list: list[int],
) -> None:
    """Dump model output logits [B, N, vocab//tp] — vocab is always the last dim."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("logits.pt", logits, dump_dir)
    _save_meta(packed_seq_params, prefix_lens_list, dump_dir, meta_key="logits")


# ── 2D format: logprobs / entropy / label ──────────────────────

def dump_logprobs_legacy(
    log_probs: torch.Tensor,
    input_ids: torch.Tensor,
) -> None:
    """Dump legacy logprobs.pt + input_ids.pt (per-rank)."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:
        rank = 0
    try:
        _save_tensor(f"logprobs_rank{rank}.pt", log_probs.clone(), dump_dir)
        _save_tensor(f"input_ids_rank{rank}.pt", input_ids.cpu().clone(), dump_dir)
    except Exception as e:
        _log.warning("legacy logprob dump failed: %s", e)


def dump_logprobs_2d(log_probs: torch.Tensor, tag: str) -> None:
    """Dump 2D logprobs_{tag}.pt."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"logprobs_{tag}.pt", log_probs.detach().clone(), dump_dir)


def dump_entropy_2d(entropy: torch.Tensor, tag: str) -> None:
    """Dump 2D entropy_{tag}.pt."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"entropy_{tag}.pt", entropy.detach().clone(), dump_dir)


def dump_label(label: torch.Tensor) -> None:
    """Dump label.pt (shared across on/off, tag-agnostic)."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("label.pt", label.detach().cpu().clone(), dump_dir)


def dump_label_mask(label_mask: torch.Tensor) -> None:
    """Dump label_mask.pt (canonical structural mask, pre-packing)."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("label_mask.pt", label_mask.cpu().clone(), dump_dir)


# ── 2D attention_mask dump (per forward pass) ───────────────────

def dump_attention_mask_2d(attention_mask: torch.Tensor) -> None:
    """Dump 2D [B, L_max] attention_mask.pt (single tensor per forward pass).

    Call once per forward pass from ``megatron_actor`` right after reading
    ``batch["attention_mask"]``, BEFORE preprocess_packed_seqs unpacks it.

    Semantics:
      - ON mode: mask has already been rewritten by
        ``verl_mcore._trim_micro_batch`` (see verl_mcore.py:200) to True only
        at suffix columns per row. Shape [B, L_max].
      - OFF mode: original per-row valid-token mask (True at all valid token
        columns). Shape [B, L_max].

    Both ON and OFF share the same [B, L_max] shape, and ON mask is a strict
    subset of OFF mask per row (suffix columns ⊆ valid columns). cmp_diag
    uses this property to align packed attn_outputs between ON and OFF.
    """
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("attention_mask.pt", attention_mask, dump_dir)
