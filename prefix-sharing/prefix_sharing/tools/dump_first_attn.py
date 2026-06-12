"""Dump last-layer attention output for ON/OFF comparison.

Error accumulation will amplify systematic bugs by the last layer,
while precision noise stays bounded.  Comparing the last layer
therefore gives a stronger signal than the first layer.

Usage:
    export PREFIX_SHARING_FIRST_ATTN_DUMP=/path/to/dump_dir
    # Run ON and OFF forward_only, then compare with cmp_first_attn.py
"""

from __future__ import annotations

import os
from typing import Any

import torch

_FIRST_ATTN_DUMP_DIR: str | None = None


def _get_dump_dir() -> str | None:
    global _FIRST_ATTN_DUMP_DIR
    if _FIRST_ATTN_DUMP_DIR is not None:
        return _FIRST_ATTN_DUMP_DIR
    path = os.environ.get("PREFIX_SHARING_FIRST_ATTN_DUMP")
    if path:
        _FIRST_ATTN_DUMP_DIR = path
        os.makedirs(path, exist_ok=True)
    return _FIRST_ATTN_DUMP_DIR


def _dump(
    output_tensor: torch.Tensor,
    packed_seq_params: Any,
    prefix_lens_list: list[int],
) -> None:
    """Save attention output + metadata for the last layer.

    Only saves on rank 0 (no distributed check needed for single-GPU).
    """
    import logging
    _log = logging.getLogger(__name__)

    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return

    try:
        torch.save(output_tensor.detach().cpu().clone(),
                   os.path.join(dump_dir, "first_attn_output.pt"))
        _log.warning("first_attn_output.pt saved (%s)", output_tensor.shape)
    except Exception as e:
        _log.warning("first_attn_output.pt save failed: %s", e)

    try:
        cu = packed_seq_params.cu_seqlens_q_padded.detach().cpu().clone()
        torch.save(cu, os.path.join(dump_dir, "cu_seqlens_q.pt"))
        _log.warning("cu_seqlens_q.pt saved (%s)", cu.shape)
    except Exception as e:
        _log.warning("cu_seqlens_q.pt save failed: %s", e)

    try:
        pl = torch.tensor(prefix_lens_list, dtype=torch.int32)
        torch.save(pl, os.path.join(dump_dir, "prefix_lens.pt"))
        _log.warning("prefix_lens.pt saved (%s)", pl.shape)
    except Exception as e:
        _log.warning("prefix_lens.pt save failed: %s", e)


def dump_on(
    output_tensor: torch.Tensor,
    packed_seq_params: Any,
    prefix_sharing_plan: Any,
    layer_number: int,
    num_layers: int,
) -> None:
    """Called from maybe_run_prefix_sharing_attention (ON mode).

    Dumps last-layer (layer_number==num_layers) attention output.
    Megatron layer_number is 1-indexed.
    """
    if layer_number != num_layers:
        return
    _dump(output_tensor, packed_seq_params,
          list(prefix_sharing_plan.prefix_lens))


def dump_off(
    output_tensor: torch.Tensor,
    packed_seq_params: Any,
    layer_number: int,
    batch_size: int,
    num_layers: int,
) -> None:
    """Called from attention.py forward normal path (OFF mode).

    Dumps last-layer (layer_number==num_layers) attention output.
    """
    if layer_number != num_layers:
        return
    if packed_seq_params is None \
       or not hasattr(packed_seq_params, 'cu_seqlens_q_padded'):
        return
    _dump(output_tensor, packed_seq_params, [0] * batch_size)
