"""Shared utilities used across the prefix-sharing package."""

from __future__ import annotations

from itertools import accumulate
from typing import Any, Mapping, Sequence


def cumsum(lengths: Sequence[int]) -> list[int]:
    """Cumulative sum with a leading 0; list length = n + 1."""
    values = [0]
    values.extend(accumulate(int(l) for l in lengths))
    return values


def read_config_value(config: Any, dotted_key: str, default: Any = None) -> Any:
    """Walk a dotted path through nested configs, returning *default* on any miss.

    Handles ``Mapping`` (dotted), objects with a ``.get()`` method, and plain
    ``getattr`` fallback.
    """
    current = config
    for part in dotted_key.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(part, default)
        elif callable(getattr(current, "get", None)):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current
