"""Shared helpers for the prefix_sharing package."""

from __future__ import annotations

from collections.abc import Mapping


def pad_to_multiple(length: int, align_size: int) -> int:
    """Pad ``length`` up to the next multiple of ``align_size``.

    A no-op when ``length`` is already aligned (e.g. align_size=8, length=16).
    Used to align packed sequence lengths to TP/CP boundaries.
    """
    if align_size <= 1:
        return int(length)
    return int(length + (align_size - length % align_size) % align_size)


def ensure_global_packed_token_lengths(
    lengths: Mapping[str, int],
    *,
    total_padded_length: int,
    context: str,
) -> None:
    """Require runtime tensors to use global packed token coordinates."""

    if all(length == total_padded_length for length in lengths.values()):
        return

    rendered_lengths = ", ".join(f"{name}={length}" for name, length in lengths.items())
    raise RuntimeError(
        f"prefix sharing {context} currently requires inputs to use global packed "
        "token coordinates; SP-local shard inputs are not supported. "
        f"{rendered_lengths}, total_padded_length={total_padded_length}"
    )
