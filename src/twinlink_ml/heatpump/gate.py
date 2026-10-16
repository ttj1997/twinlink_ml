r"""Gate L1/L2/L3 of Stage B (heat pump plus storage) and the statistics around it.

The definitions are the ones PRE-REGISTERED in the docstring of ``scripts/gate_heat_pump.py``
(committed before any computation on the plant data); this module only implements them.

A CELL is (run, capacity ``C`` in {0.5, 1, 2, 5 MWh}). With the rule gap
``D = (J_rule - J_foresight) / J_none`` (best of the three rules) and the storage gain
``G = (J_pump - J_foresight) / J_none``:

* L1 (RL / MPC set-up justified): some realistic cell has ``D >= 2`` percentage points;
* L2 (storage with the pump pays, a rule suffices): no such cell, but some has ``G >= 5 %``;
* L3 (no relevant potential): neither.

Only REALISTIC runs (all placeholders inside their bands) with ``E_direct >= 100 kWh/d`` count.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DIRECT_FLOOR_KWH_PER_DAY",
    "GATE_GAP",
    "GATE_STORAGE_GAIN",
    "Cell",
    "cells_of",
    "foresight_map",
    "gate_verdict",
    "smallest_amplitude",
    "subset_verdicts",
]

GATE_GAP = 0.02
GATE_STORAGE_GAIN = 0.05
DIRECT_FLOOR_KWH_PER_DAY = 100.0
_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class Cell:
    """One (run, capacity) cell; ``meta`` holds the scenario axes of the run."""

    key: str
    meta: Mapping[str, Any]
    realistic: bool
    capacity_kwh: float
    gap: float
    gain: float
    weekly_value: float | None


def cells_of(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Cell], dict[str, int]]:
    """All cells of the Stage-B records (realistic or not; the floor applied).

    Each record needs ``key``, ``meta``, ``realistic``, ``direct_kwh_per_day`` and the lists
    ``capacities_kwh``, ``gap_fraction``, ``storage_gain_fraction`` and optionally
    ``weekly_value_fraction``.
    """
    cells: list[Cell] = []
    counts = {"n_runs": len(records), "n_unrealistic": 0, "n_excluded_floor": 0}
    for rec in records:
        if not rec["realistic"]:
            counts["n_unrealistic"] += 1
        if rec["direct_kwh_per_day"] < DIRECT_FLOOR_KWH_PER_DAY:
            counts["n_excluded_floor"] += 1
            continue
        weekly = rec.get("weekly_value_fraction")
        for i, cap in enumerate(rec["capacities_kwh"]):
            cells.append(
                Cell(
                    key=str(rec["key"]),
                    meta=rec["meta"],
                    realistic=bool(rec["realistic"]),
                    capacity_kwh=float(cap),
                    gap=float(rec["gap_fraction"][i]),
                    gain=float(rec["storage_gain_fraction"][i]),
                    weekly_value=None if weekly is None else float(weekly[i]),
                )
            )
    return cells, counts


def gate_verdict(cells: Sequence[Cell], counts: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Letter reading over the REALISTIC cells and the statistics that must accompany it."""
    real = [c for c in cells if c.realistic]
    gap_hit = [c for c in real if c.gap >= GATE_GAP - _EPS]
    gain_hit = [c for c in real if c.gain >= GATE_STORAGE_GAIN - _EPS]
    reading = "L1" if gap_hit else ("L2" if gain_hit else "L3")
    widest = max(real, key=lambda c: c.gap, default=None)
    best_gain = max(real, key=lambda c: c.gain, default=None)
    runs = {c.key for c in real}
    out: dict[str, Any] = {
        "reading": reading,
        "n_cells_realistic": len(real),
        "n_runs_realistic": len(runs),
        "n_cells_gap_at_least_2_points": len(gap_hit),
        "share_cells_gap_at_least_2_points": len(gap_hit) / len(real) if real else None,
        "n_runs_gap_at_least_2_points": len({c.key for c in gap_hit}),
        "n_cells_gain_at_least_5_percent": len(gain_hit),
        "share_cells_gain_at_least_5_percent": len(gain_hit) / len(real) if real else None,
        "max_gap": None if widest is None else widest.gap,
        "max_gap_key": None if widest is None else widest.key,
        "max_gap_capacity_kwh": None if widest is None else widest.capacity_kwh,
        "max_gap_meta": None if widest is None else dict(widest.meta),
        "max_gain": None if best_gain is None else best_gain.gain,
        "max_gain_key": None if best_gain is None else best_gain.key,
        "max_gain_capacity_kwh": None if best_gain is None else best_gain.capacity_kwh,
        "max_gain_meta": None if best_gain is None else dict(best_gain.meta),
    }
    weekly = [c.weekly_value for c in real if c.weekly_value is not None]
    out["max_weekly_over_daily_foresight"] = max(weekly) if weekly else None
    if counts:
        out.update(counts)
    return out


def subset_verdicts(
    cells: Sequence[Cell], groupings: Mapping[str, Callable[[Cell], object]]
) -> dict[str, dict[str, Any]]:
    """Readings per value of each grouping (criterion, horizon, site, day, ...)."""
    out: dict[str, dict[str, Any]] = {}
    for name, key in groupings.items():
        for value in sorted({str(key(c)) for c in cells}):
            subset = [c for c in cells if str(key(c)) == value]
            v = gate_verdict(subset)
            out[f"{name}={value}"] = {
                k: v[k]
                for k in (
                    "reading",
                    "n_cells_realistic",
                    "n_cells_gap_at_least_2_points",
                    "n_cells_gain_at_least_5_percent",
                    "max_gap",
                    "max_gain",
                )
            }
    return out


def foresight_map(
    cells: Sequence[Cell], kind_of: Callable[[Cell], str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Max ``D`` and max ``G`` over runs per (kind, amplitude, capacity), ALL runs of the grid.

    The realism flag is not applied here (amplitudes outside the band are part of the map and
    are marked by the caller); the floor is.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for kind in sorted({kind_of(c) for c in cells}):
        sub = [c for c in cells if kind_of(c) == kind]
        amps = sorted({float(c.meta["amplitude"]) for c in sub})
        caps = sorted({c.capacity_kwh for c in sub})
        table: dict[str, dict[str, float]] = {}
        for a in amps:
            row: dict[str, float] = {}
            for cap in caps:
                sel = [c for c in sub if float(c.meta["amplitude"]) == a and c.capacity_kwh == cap]
                row[f"max_gap|{cap:g}"] = float(max(c.gap for c in sel)) if sel else float("nan")
                row[f"max_gain|{cap:g}"] = float(max(c.gain for c in sel)) if sel else float("nan")
                row[f"median_gap|{cap:g}"] = (
                    float(np.median([c.gap for c in sel])) if sel else float("nan")
                )
            table[f"{a:g}"] = row
        out[kind] = table
    return out


def smallest_amplitude(
    cells: Sequence[Cell], kind_of: Callable[[Cell], str], threshold: float = GATE_GAP
) -> dict[str, dict[str, float | None]]:
    """Smallest amplitude of the grid at which some run reaches ``D >= threshold``, per capacity."""
    out: dict[str, dict[str, float | None]] = {}
    for kind in sorted({kind_of(c) for c in cells}):
        sub = [c for c in cells if kind_of(c) == kind]
        row: dict[str, float | None] = {}
        for cap in sorted({c.capacity_kwh for c in sub}):
            hits = [
                float(c.meta["amplitude"])
                for c in sub
                if c.capacity_kwh == cap and c.gap >= threshold - _EPS
            ]
            row[f"{cap:g}"] = min(hits) if hits else None
        out[kind] = row
    return out
