"""Runtime batch layouts consumed by backend implementations.

The layout describes the tensor coordinate system seen by model backends. It is
not prefix-sharing semantic metadata; integrations construct it from framework
inputs, and backends consume it to read/write valid token rows safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Protocol, Sequence

import torch


@dataclass(frozen=True)
class BshdTokenIndex:
    """A (seq_idx_in_batch, token_idx_in_seq) coordinate in a BSHD tensor."""

    seq_idx_in_batch: int  # index of the sequence within the batch
    token_idx_in_seq: int  # absolute position of the token within the padded sequence dimension


class BatchRuntimeLayout(Protocol):
    """Minimal interface shared by THD and BSHD layouts."""

    layout_kind: str  # "thd" or "bshd"
    valid_lengths: list[int]  # per-sequence count of non-padding tokens
    max_seqlen: int  # longest padded sequence length
    valid_token_mask: Any | None  # boolean mask marking valid positions
    position_ids: Any | None  # per-token position ids (optional)

    @property
    def batch_size(self) -> int:
        """Number of sequences in the batch."""
        ...

    @property
    def total_valid_length(self) -> int:
        """Sum of valid token counts across all sequences."""
        ...

    def token_index(self, seq_idx_in_batch: int, valid_offset: int) -> Any:
        """Convert a (seq_idx_in_batch, valid_offset) pair to a flat or structured index."""
        ...

    def valid_tokens(self, tensor: Any, seq_idx_in_batch: int) -> Any:
        """Extract only the valid (non-padding) tokens from sequence *seq_idx_in_batch*."""
        ...

    def padded_tokens(self, tensor: Any, seq_idx_in_batch: int) -> Any:
        """Extract the full padded sequence (valid tokens + alignment padding)."""
        ...

    def write_valid_tokens(self, output: Any, seq_idx_in_batch: int, valid_output: Any) -> None:
        """Write *valid_output* into the valid positions of *output* at *seq_idx_in_batch*."""
        ...


@dataclass(frozen=True)
class ThdBatchLayout:
    """Packed (THD) layout: sequences are concatenated along dim-0 with cu_seqlens offsets."""

    valid_lengths: list[int]  # per-sequence count of non-padding tokens
    padded_lengths: list[int]  # per-sequence total length including alignment padding
    cu_seqlens: list[int]  # cumulative sum of padded_lengths; length = batch_size + 1
    max_seqlen: int  # longest padded length across all sequences
    position_ids: Any | None = None  # 1-D packed position ids for all tokens (optional)
    valid_token_mask: Any | None = None  # 1-D packed boolean mask for valid vs padding (optional)

    layout_kind: str = "thd"

    def __post_init__(self) -> None:
        batch_size = len(self.valid_lengths)
        if len(self.padded_lengths) != batch_size:
            raise ValueError("padded_lengths must match valid_lengths")
        if len(self.cu_seqlens) != batch_size + 1:
            raise ValueError("cu_seqlens length must equal batch_size + 1")
        if self.cu_seqlens[0] != 0:
            raise ValueError("cu_seqlens must start with 0")
        if any(length < 0 for length in self.valid_lengths):
            raise ValueError("valid_lengths must be non-negative")
        if any(length < 0 for length in self.padded_lengths):
            raise ValueError("padded_lengths must be non-negative")
        for valid_length, padded_length in zip(self.valid_lengths, self.padded_lengths):
            if padded_length < valid_length:
                raise ValueError("padded_lengths cannot be smaller than valid_lengths")
        if self.cu_seqlens != _cumsum(self.padded_lengths):
            raise ValueError("cu_seqlens must be the cumulative padded lengths")

    @classmethod
    def construct_from_valid_lengths(cls, valid_lengths: Sequence[int]) -> "ThdBatchLayout":
        """Build a layout with no padding (padded_lengths == valid_lengths)."""
        lengths = [int(length) for length in valid_lengths]
        return cls(
            valid_lengths=lengths,
            padded_lengths=list(lengths),
            cu_seqlens=_cumsum(lengths),
            max_seqlen=max(lengths, default=0),
            position_ids=None,
            valid_token_mask=None,
        )

    @classmethod
    def construct_from_kept_position_ids(
        cls,
        kept_position_ids: Sequence[Any],
        *,
        align_size: int,
    ) -> "ThdBatchLayout":
        """Build a layout with alignment padding and packed position_ids / valid_token_mask."""
        if align_size < 1:
            raise ValueError("align_size must be >= 1")
        if not kept_position_ids:
            return cls.construct_from_valid_lengths([])

        device = kept_position_ids[0].device
        dtype = kept_position_ids[0].dtype

        valid_lengths = [int(seq_positions.shape[0]) for seq_positions in kept_position_ids]
        padded_lengths = [_pad_to_multiple(length, align_size) for length in valid_lengths]
        total_padded = sum(padded_lengths)

        # Pre-allocate full-size tensors; fill via slice assignment instead of per-row cat.
        packed_positions = torch.zeros(total_padded, dtype=dtype, device=device)
        valid_mask = torch.zeros(total_padded, dtype=torch.bool, device=device)

        offset = 0
        for seq_positions, valid_length, padded_length in zip(kept_position_ids, valid_lengths, padded_lengths):
            packed_positions[offset : offset + valid_length] = seq_positions.to(device)
            valid_mask[offset : offset + valid_length] = True
            offset += padded_length

        return cls(
            valid_lengths=valid_lengths,
            padded_lengths=padded_lengths,
            cu_seqlens=_cumsum(padded_lengths),
            max_seqlen=max(padded_lengths, default=0),
            position_ids=packed_positions,
            valid_token_mask=valid_mask,
        )

    @property
    def batch_size(self) -> int:
        """Number of sequences in the batch."""
        return len(self.valid_lengths)

    @property
    def total_padded_length(self) -> int:
        """Total length of the packed 1-D tensor (sum of padded_lengths)."""
        return self.cu_seqlens[-1]

    @property
    def total_valid_length(self) -> int:
        """Sum of valid token counts across all sequences."""
        return sum(self.valid_lengths)

    def seq_start(self, seq_idx_in_batch: int) -> int:
        """Flat offset where sequence *seq_idx_in_batch* begins in the packed tensor."""
        return self.cu_seqlens[seq_idx_in_batch]

    def token_index(self, seq_idx_in_batch: int, valid_offset: int) -> int:
        """Flat index in the packed tensor for (seq_idx_in_batch, valid_offset)."""
        if valid_offset < 0 or valid_offset >= self.valid_lengths[seq_idx_in_batch]:
            raise IndexError("valid_offset is outside the valid token range")
        return self.seq_start(seq_idx_in_batch) + valid_offset

    def valid_tokens(self, tensor: Any, seq_idx_in_batch: int) -> Any:
        """Extract only the valid (non-padding) tokens from sequence *seq_idx_in_batch* of a packed 1-D tensor."""
        start = self.seq_start(seq_idx_in_batch)
        end = start + self.valid_lengths[seq_idx_in_batch]
        return tensor[start:end]

    def padded_tokens(self, tensor: Any, seq_idx_in_batch: int) -> Any:
        """Extract the full padded sequence (valid tokens + alignment padding) from a packed 1-D tensor."""
        start = self.seq_start(seq_idx_in_batch)
        end = start + self.padded_lengths[seq_idx_in_batch]
        return tensor[start:end]

    def write_valid_tokens(self, output: Any, seq_idx_in_batch: int, valid_output: Any) -> None:
        """Write *valid_output* into the valid positions of *output* at sequence *seq_idx_in_batch*."""
        start = self.seq_start(seq_idx_in_batch)
        output[start : start + self.valid_lengths[seq_idx_in_batch]] = valid_output


@dataclass(frozen=True)
class BshdBatchLayout:
    """Dense (BSHD) layout: tensors stay in [batch, seq, head, dim] shape, masked by valid_token_mask."""

    valid_lengths: list[int]  # per-sequence count of non-padding tokens
    max_seqlen: int  # padded sequence dimension size
    valid_token_mask: Any  # 2-D bool tensor [batch, seq]; True = valid token
    position_ids: Any | None = None  # 2-D position ids [batch, seq] (optional)

    layout_kind: str = "bshd"

    def __post_init__(self) -> None:
        if any(length < 0 for length in self.valid_lengths):
            raise ValueError("valid_lengths must be non-negative")
        if self.valid_token_mask.dim() != 2:
            raise ValueError("BshdBatchLayout valid_token_mask must be 2D")
        if self.valid_token_mask.shape[0] != len(self.valid_lengths):
            raise ValueError("valid_token_mask batch dimension must match valid_lengths")
        if self.valid_token_mask.shape[1] != self.max_seqlen:
            raise ValueError("valid_token_mask sequence dimension must match max_seqlen")
        mask_lengths = [int(value) for value in self.valid_token_mask.sum(dim=1).detach().cpu().tolist()]
        if mask_lengths != self.valid_lengths:
            raise ValueError("valid_lengths must match valid_token_mask sequence sums")
        if self.position_ids is not None and tuple(self.position_ids.shape[:2]) != tuple(self.valid_token_mask.shape):
            raise ValueError("position_ids leading dimensions must match valid_token_mask")

    @classmethod
    def from_valid_token_mask(
        cls,
        valid_token_mask: Any,
        *,
        position_ids: Any | None = None,
    ) -> "BshdBatchLayout":
        """Build a layout from a 2-D boolean mask and optional position ids."""
        mask = valid_token_mask.to(bool)
        valid_lengths = [int(value) for value in mask.sum(dim=1).detach().cpu().tolist()]
        return cls(
            valid_lengths=valid_lengths,
            max_seqlen=int(mask.shape[1]),
            valid_token_mask=mask,
            position_ids=position_ids,
        )

    @property
    def batch_size(self) -> int:
        """Number of sequences in the batch."""
        return len(self.valid_lengths)

    @property
    def total_valid_length(self) -> int:
        """Sum of valid token counts across all sequences."""
        return sum(self.valid_lengths)

    def token_index(self, seq_idx_in_batch: int, valid_offset: int) -> BshdTokenIndex:
        """Convert (seq_idx_in_batch, valid_offset) to a (seq_idx_in_batch, token_idx_in_seq) coordinate in the dense tensor."""
        if valid_offset < 0 or valid_offset >= self.valid_lengths[seq_idx_in_batch]:
            raise IndexError("valid_offset is outside the valid token range")
        valid_positions = torch.nonzero(self.valid_token_mask[seq_idx_in_batch], as_tuple=False).flatten()
        return BshdTokenIndex(seq_idx_in_batch=seq_idx_in_batch, token_idx_in_seq=int(valid_positions[valid_offset].item()))

    def valid_tokens(self, tensor: Any, seq_idx_in_batch: int) -> Any:
        """Extract only the valid (non-padding) tokens from sequence *seq_idx_in_batch*; supports compact, SBH, and dense tensors."""
        if self._is_compact_valid_tensor(tensor):
            start = sum(self.valid_lengths[:seq_idx_in_batch])
            end = start + self.valid_lengths[seq_idx_in_batch]
            return tensor[start:end]
        if self._is_padded_sbh_tensor(tensor):
            return tensor[: self.valid_lengths[seq_idx_in_batch], seq_idx_in_batch]
        return tensor[seq_idx_in_batch, self.valid_token_mask[seq_idx_in_batch]]

    def padded_tokens(self, tensor: Any, seq_idx_in_batch: int) -> Any:
        """Extract the full padded sequence (valid tokens + alignment padding); supports compact, SBH, and dense tensors."""
        if self._is_compact_valid_tensor(tensor):
            start = sum(self.valid_lengths[:seq_idx_in_batch])
            end = start + self.valid_lengths[seq_idx_in_batch]
            return tensor[start:end]
        if self._is_padded_sbh_tensor(tensor):
            return tensor[:, seq_idx_in_batch]
        return tensor[seq_idx_in_batch]

    def write_valid_tokens(self, output: Any, seq_idx_in_batch: int, valid_output: Any) -> None:
        """Write *valid_output* into the valid positions of *output* at sequence *seq_idx_in_batch*."""
        if self._is_compact_valid_tensor(output):
            start = sum(self.valid_lengths[:seq_idx_in_batch])
            output[start : start + self.valid_lengths[seq_idx_in_batch]] = valid_output
            return
        if self._is_padded_sbh_tensor(output):
            output[: self.valid_lengths[seq_idx_in_batch], seq_idx_in_batch] = valid_output
            return
        output[seq_idx_in_batch, self.valid_token_mask[seq_idx_in_batch]] = valid_output

    def valid_position_ids(self, *, device: Any | None = None) -> Any:
        """1-D position ids for all valid tokens (masked from the 2-D position_ids)."""
        if self.position_ids is None:
            raise RuntimeError("BshdBatchLayout is missing position_ids")
        mask = self.valid_token_mask
        position_ids = self.position_ids
        if device is not None:
            mask = mask.to(device=device)
            position_ids = position_ids.to(device=device)
        return position_ids[mask]

    def kept_padded_position_ids(self, *, device: Any | None = None, padded_length: int | None = None) -> Any:
        """2-D padded position ids [padded_length, batch_size] for kept tokens, zero-filled for padding slots."""
        if self.position_ids is None:
            raise RuntimeError("BshdBatchLayout is missing position_ids")
        mask = self.valid_token_mask
        position_ids = self.position_ids
        if device is not None:
            mask = mask.to(device=device)
            position_ids = position_ids.to(device=device)
        max_valid_length = max(self.valid_lengths, default=0)
        target_length = max_valid_length if padded_length is None else int(padded_length)
        if target_length < max_valid_length:
            raise ValueError("padded_length cannot be smaller than max valid length")
        rows = []
        for seq_idx_in_batch, valid_length in enumerate(self.valid_lengths):
            valid_positions = position_ids[seq_idx_in_batch, mask[seq_idx_in_batch]]
            pad_length = target_length - valid_length
            if pad_length > 0:
                padding = torch.zeros(pad_length, dtype=valid_positions.dtype, device=valid_positions.device)
                valid_positions = torch.cat([valid_positions, padding], dim=0)
            rows.append(valid_positions)
        if not rows:
            return torch.empty(0, 0, dtype=position_ids.dtype, device=position_ids.device)
        return torch.stack(rows, dim=1)

    def _is_compact_valid_tensor(self, tensor: Any) -> bool:
        return tensor.dim() >= 2 and int(tensor.shape[0]) == self.total_valid_length

    def _is_padded_sbh_tensor(self, tensor: Any) -> bool:
        max_valid_length = max(self.valid_lengths, default=0)
        return (
            tensor.dim() >= 3
            and int(tensor.shape[0]) >= max_valid_length
            and int(tensor.shape[1]) == self.batch_size
        )


def _pad_to_multiple(length: int, align_size: int) -> int:
    return int(length + (align_size - length % align_size) % align_size)


def _cumsum(lengths: Sequence[int]) -> list[int]:
    """Cumulative sum with a leading 0; list length = n + 1."""
    values = [0]
    values.extend(accumulate(int(l) for l in lengths))
    return values




