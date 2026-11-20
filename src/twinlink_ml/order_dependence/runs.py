"""Chunks of a request and the comparison of two runs of the same units."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["TIMING_FIELDS", "chunks_of", "compare_records"]

TIMING_FIELDS = ("decide_s", "reset_s")
"""Fields of a record that hold wall-clock times (never equal between two runs)."""


def chunks_of(ids: Sequence[int], workers: int, layout: str) -> list[list[int]]:
    """Units per task of a request.

    Layout ``a`` as the drivers of the findings: ``ceil(n / workers)`` units per task, in order.
    Layout ``b``: reversed order and ``ceil(n / (workers + 1))`` units per task, so that every
    unit meets other units and another history of the worker process.
    """
    if layout not in ("a", "b"):
        raise ValueError(f"unknown layout {layout!r}")
    ordered = list(ids) if layout == "a" else list(ids)[::-1]
    parts = workers if layout == "a" else workers + 1
    size = max(1, math.ceil(len(ordered) / parts))
    return [ordered[s : s + size] for s in range(0, len(ordered), size)]


def compare_records(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Units whose records are equal in every field but the timings, and the largest differences.

    The records are compared as they are stored (Python equality of the values, i.e. bit
    equality of every float); ``first`` and ``second`` list the same units in the same order.
    """
    equal = 0
    differing: list[int] = []
    largest: dict[str, float] = {}
    for a, b in zip(first, second, strict=True):
        if a["unit"] != b["unit"]:
            raise ValueError("the two runs list different units")
        ka = {k: v for k, v in a.items() if k not in TIMING_FIELDS}
        kb = {k: v for k, v in b.items() if k not in TIMING_FIELDS}
        if ka == kb:
            equal += 1
            continue
        differing.append(int(a["unit"]))
        for field in ("energy_kwh", "cost_e", "max_violation"):
            if isinstance(a.get(field), float) and isinstance(b.get(field), float):
                largest[field] = max(largest.get(field, 0.0), abs(a[field] - b[field]))
    return {
        "units": len(first),
        "equal_units": equal,
        "differing_units": differing,
        "largest_abs_difference": largest,
    }
