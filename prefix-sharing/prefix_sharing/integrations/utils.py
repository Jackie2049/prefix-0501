"""Shared helpers for framework integration code."""

from __future__ import annotations

from collections.abc import Mapping


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
