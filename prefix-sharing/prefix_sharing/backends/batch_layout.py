"""Runtime batch layouts consumed by backend implementations.

The layout describes the tensor coordinate system seen by model backends. It is
not prefix-sharing semantic metadata; integrations construct it from framework
inputs, and backends consume it to read/write valid token rows safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class BshdTokenIndex:
    row: int
    seq_pos: int


class BatchRuntimeLayout(Protocol):
    layout_kind: str
    valid_lengths: list[int]
    max_seqlen: int
    valid_token_mask: Any | None
    position_ids: Any | None

    @property
    def batch_size(self) -> int:
        ...

    @property
    def total_valid_length(self) -> int:
        ...

    def token_index(self, row: int, valid_offset: int) -> Any:
        ...

    def valid_row(self, tensor: Any, row: int) -> Any:
        ...

    def padded_row(self, tensor: Any, row: int) -> Any:
        ...

    def scatter_valid_row(self, output: Any, row: int, valid_output: Any) -> None:
        ...


@dataclass(frozen=True)
class ThdBatchLayout:
    valid_lengths: list[int]
    padded_lengths: list[int]
    cu_seqlens: list[int]
    max_seqlen: int
    position_ids: Any | None = None
    valid_token_mask: Any | None = None

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
        if align_size < 1:
            raise ValueError("align_size must be >= 1")
        if not kept_position_ids:
            return cls.construct_from_valid_lengths([])

        first = kept_position_ids[0]
        torch = _torch()
        valid_lengths = [int(row.shape[0]) for row in kept_position_ids]
        padded_lengths = [_pad_to_multiple(length, align_size) for length in valid_lengths]
        packed_position_rows = []
        valid_mask_rows = []
        for row, valid_length, padded_length in zip(kept_position_ids, valid_lengths, padded_lengths):
            row = row.to(first.device)
            pad_length = padded_length - valid_length
            valid_mask_rows.append(
                torch.cat(
                    [
                        torch.ones(valid_length, dtype=torch.bool, device=first.device),
                        torch.zeros(pad_length, dtype=torch.bool, device=first.device),
                    ],
                    dim=0,
                )
            )
            if pad_length == 0:
                packed_position_rows.append(row)
                continue
            padding = torch.zeros(pad_length, dtype=row.dtype, device=first.device)
            packed_position_rows.append(torch.cat([row, padding], dim=0))

        return cls(
            valid_lengths=valid_lengths,
            padded_lengths=padded_lengths,
            cu_seqlens=_cumsum(padded_lengths),
            max_seqlen=max(padded_lengths, default=0),
            position_ids=torch.cat(packed_position_rows, dim=0),
            valid_token_mask=torch.cat(valid_mask_rows, dim=0),
        )

    @property
    def batch_size(self) -> int:
        return len(self.valid_lengths)

    @property
    def total_padded_length(self) -> int:
        return self.cu_seqlens[-1]

    @property
    def total_valid_length(self) -> int:
        return sum(self.valid_lengths)

    def row_start(self, row: int) -> int:
        return self.cu_seqlens[row]

    def token_index(self, row: int, valid_offset: int) -> int:
        if valid_offset < 0 or valid_offset >= self.valid_lengths[row]:
            raise IndexError("valid_offset is outside the row valid token range")
        return self.row_start(row) + valid_offset

    def valid_row(self, tensor: Any, row: int) -> Any:
        start = self.row_start(row)
        end = start + self.valid_lengths[row]
        return tensor[start:end]

    def padded_row(self, tensor: Any, row: int) -> Any:
        start = self.row_start(row)
        end = start + self.padded_lengths[row]
        return tensor[start:end]

    def scatter_valid_row(self, output: Any, row: int, valid_output: Any) -> None:
        start = self.row_start(row)
        output[start : start + self.valid_lengths[row]] = valid_output


@dataclass(frozen=True)
class BshdBatchLayout:
    valid_lengths: list[int]
    max_seqlen: int
    valid_token_mask: Any
    position_ids: Any | None = None

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
            raise ValueError("valid_lengths must match valid_token_mask row sums")
        if self.position_ids is not None and tuple(self.position_ids.shape[:2]) != tuple(self.valid_token_mask.shape):
            raise ValueError("position_ids leading dimensions must match valid_token_mask")

    @classmethod
    def from_valid_token_mask(
        cls,
        valid_token_mask: Any,
        *,
        position_ids: Any | None = None,
    ) -> "BshdBatchLayout":
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
        return len(self.valid_lengths)

    @property
    def total_valid_length(self) -> int:
        return sum(self.valid_lengths)

    def token_index(self, row: int, valid_offset: int) -> BshdTokenIndex:
        if valid_offset < 0 or valid_offset >= self.valid_lengths[row]:
            raise IndexError("valid_offset is outside the row valid token range")
        torch = _torch()
        valid_positions = torch.nonzero(self.valid_token_mask[row], as_tuple=False).flatten()
        return BshdTokenIndex(row=row, seq_pos=int(valid_positions[valid_offset].item()))

    def valid_row(self, tensor: Any, row: int) -> Any:
        if self._is_compact_valid_tensor(tensor):
            start = sum(self.valid_lengths[:row])
            end = start + self.valid_lengths[row]
            return tensor[start:end]
        if self._is_kept_padded_sbh_tensor(tensor):
            return tensor[: self.valid_lengths[row], row]
        return tensor[row, self.valid_token_mask[row]]

    def padded_row(self, tensor: Any, row: int) -> Any:
        if self._is_compact_valid_tensor(tensor):
            start = sum(self.valid_lengths[:row])
            end = start + self.valid_lengths[row]
            return tensor[start:end]
        if self._is_kept_padded_sbh_tensor(tensor):
            return tensor[:, row]
        return tensor[row]

    def scatter_valid_row(self, output: Any, row: int, valid_output: Any) -> None:
        if self._is_compact_valid_tensor(output):
            start = sum(self.valid_lengths[:row])
            output[start : start + self.valid_lengths[row]] = valid_output
            return
        if self._is_kept_padded_sbh_tensor(output):
            output[: self.valid_lengths[row], row] = valid_output
            return
        output[row, self.valid_token_mask[row]] = valid_output

    def valid_position_ids(self, *, device: Any | None = None) -> Any:
        if self.position_ids is None:
            raise RuntimeError("BshdBatchLayout is missing position_ids")
        mask = self.valid_token_mask
        position_ids = self.position_ids
        if device is not None:
            mask = mask.to(device=device)
            position_ids = position_ids.to(device=device)
        return position_ids[mask]

    def kept_padded_position_ids(self, *, device: Any | None = None, padded_length: int | None = None) -> Any:
        if self.position_ids is None:
            raise RuntimeError("BshdBatchLayout is missing position_ids")
        torch = _torch()
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
        for row, valid_length in enumerate(self.valid_lengths):
            valid_positions = position_ids[row, mask[row]]
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

    def _is_kept_padded_sbh_tensor(self, tensor: Any) -> bool:
        max_valid_length = max(self.valid_lengths, default=0)
        return (
            tensor.dim() >= 3
            and int(tensor.shape[0]) >= max_valid_length
            and int(tensor.shape[1]) == self.batch_size
        )


def _pad_to_multiple(length: int, align_size: int) -> int:
    return int(length + (align_size - length % align_size) % align_size)


def _cumsum(lengths: Sequence[int]) -> list[int]:
    values = [0]
    running = 0
    for length in lengths:
        running += int(length)
        values.append(running)
    return values


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("BatchRuntimeLayout tensor construction requires PyTorch") from exc
    return torch
