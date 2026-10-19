r"""Gate L1/L2/L3 of storage test 1 and the summary statistics around it.

The definitions are the ones PRE-REGISTERED in the docstring of
``scripts/gate_storage_timeslice.py`` (committed before any computation); this module only
implements them.

A CELL is (scenario, capacity ``C`` with ``0 < C <= 10 MWh``). With the saving
``G = (E_direct - E_ideal) / E_direct`` and the rule gap ``D = (E_rule - E_ideal) / E_direct``:

* L1 (RL / MPC set-up justified): some cell has ``G >= 5 %`` and ``D >= 2`` percentage points;
* L2 (storage pays, a rule suffices): some cell has ``G >= 5 %`` but no cell has both;
* L3 (no relevant potential): no cell has ``G >= 5 %``.

Only REALISTIC scenarios (all placeholders inside their bands) with ``E_direct >= 100 kWh/d`` count.
The reading is a statement about the best case over the placeholder bands.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import BAND_CAPACITY_MAX_KWH, Scenario, is_realistic
from .evaluate import ScenarioResult

__all__ = [
    "DIRECT_FLOOR_KWH_PER_DAY",
    "GATE_GAP",
    "GATE_SAVING",
    "Cell",
    "capacity_at_fraction_of_maximum",
    "gate_cells",
    "gate_verdict",
    "rank_sensitivity",
    "subset_verdicts",
]

GATE_SAVING = 0.05
GATE_GAP = 0.02
DIRECT_FLOOR_KWH_PER_DAY = 100.0
_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class Cell:
    """One (scenario, capacity) cell of the gate."""

    scenario: Scenario
    label: str
    capacity_kwh: float
    direct_kwh_per_day: float
    saving_fraction: float
    saving_kwh_per_day: float
    gap_fraction: float


def gate_cells(
    records: Sequence[tuple[Scenario, ScenarioResult]],
    realistic: Callable[[Scenario], bool] = is_realistic,
) -> tuple[list[Cell], dict[str, int]]:
    """Cells of all realistic scenarios above the denominator floor.

    :param realistic: which scenarios count; default :func:`is_realistic` (the pre-registered
        bands). A re-check that moves one placeholder outside its band on purpose passes its own
        predicate and says so.
    :returns: ``(cells, counts)`` with the number of scenarios, of unrealistic ones and of those
        excluded by the floor.
    """
    cells: list[Cell] = []
    counts = {"n_scenarios": len(records), "n_unrealistic": 0, "n_excluded_floor": 0}
    for scenario, result in records:
        if not realistic(scenario):
            counts["n_unrealistic"] += 1
            continue
        if result.direct_kwh_per_day < DIRECT_FLOOR_KWH_PER_DAY:
            counts["n_excluded_floor"] += 1
            continue
        for row in result.rows:
            if row.capacity_kwh <= 0.0 or row.capacity_kwh > BAND_CAPACITY_MAX_KWH:
                continue
            cells.append(
                Cell(
                    scenario=scenario,
                    label=result.label,
                    capacity_kwh=row.capacity_kwh,
                    direct_kwh_per_day=result.direct_kwh_per_day,
                    saving_fraction=result.saving_fraction(row),
                    saving_kwh_per_day=result.direct_kwh_per_day - row.ideal_kwh_per_day,
                    gap_fraction=result.gap_fraction(row),
                )
            )
    return cells, counts


def gate_verdict(cells: Sequence[Cell], counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Letter reading and the statistics that must accompany it."""
    hit = [c for c in cells if c.saving_fraction >= GATE_SAVING - _EPS]
    both = [c for c in hit if c.gap_fraction >= GATE_GAP - _EPS]
    reading = "L1" if both else ("L2" if hit else "L3")
    best = max(cells, key=lambda c: c.saving_fraction, default=None)
    best_kwh = max(cells, key=lambda c: c.saving_kwh_per_day, default=None)
    widest = max(cells, key=lambda c: c.gap_fraction, default=None)
    scenarios_hit = {c.label for c in hit}
    scenarios_all = {c.label for c in cells}
    out: dict[str, Any] = {
        "reading": reading,
        "n_cells": len(cells),
        "n_cells_saving_at_least_5_percent": len(hit),
        "n_cells_saving_and_gap": len(both),
        "n_scenarios_in_gate": len(scenarios_all),
        "n_scenarios_with_a_cell_saving_at_least_5_percent": len(scenarios_hit),
        "share_scenarios_with_saving_at_least_5_percent": (
            len(scenarios_hit) / len(scenarios_all) if scenarios_all else None
        ),
        "max_saving_fraction": None if best is None else best.saving_fraction,
        "max_saving_fraction_label": None if best is None else best.label,
        "max_saving_fraction_capacity_kwh": None if best is None else best.capacity_kwh,
        "max_saving_kwh_per_day": None if best_kwh is None else best_kwh.saving_kwh_per_day,
        "max_saving_kwh_per_day_label": None if best_kwh is None else best_kwh.label,
        "max_saving_kwh_per_day_capacity_kwh": None if best_kwh is None else best_kwh.capacity_kwh,
        "max_gap_fraction": None if widest is None else widest.gap_fraction,
        "max_gap_fraction_label": None if widest is None else widest.label,
    }
    if counts:
        out.update(counts)
    return out


def subset_verdicts(
    cells: Sequence[Cell], groupings: dict[str, Callable[[Scenario], object]]
) -> dict[str, dict[str, Any]]:
    """Readings per value of each grouping (site, horizon, day, ...)."""
    out: dict[str, dict[str, Any]] = {}
    for name, key in groupings.items():
        values = sorted({str(key(c.scenario)) for c in cells})
        for value in values:
            subset = [c for c in cells if str(key(c.scenario)) == value]
            v = gate_verdict(subset)
            out[f"{name}={value}"] = {
                k: v[k]
                for k in (
                    "reading",
                    "n_cells",
                    "n_cells_saving_at_least_5_percent",
                    "n_cells_saving_and_gap",
                    "max_saving_fraction",
                    "max_gap_fraction",
                )
            }
    return out


def capacity_at_fraction_of_maximum(result: ScenarioResult, fraction: float = 0.9) -> float | None:
    """Smallest capacity of the grid that reaches ``fraction`` of the maximum saving.

    Over the whole capacity grid (including 20 MWh); ``None`` if the maximum saving is zero.
    """
    savings = np.array([result.direct_kwh_per_day - r.ideal_kwh_per_day for r in result.rows])
    top = float(np.max(savings))
    if top <= 1e-6 * max(1.0, result.direct_kwh_per_day):
        return None
    for row, s in zip(result.rows, savings, strict=True):
        if s >= fraction * top - 1e-12:
            return row.capacity_kwh
    return None


def rank_sensitivity(
    base: ScenarioResult, variants: dict[str, ScenarioResult], capacity_kwh: float
) -> list[dict[str, Any]]:
    """Variants ranked by the absolute change of the saving at ``capacity_kwh`` (kWh/d)."""

    def at(result: ScenarioResult) -> tuple[float, float]:
        row = next(r for r in result.rows if r.capacity_kwh == capacity_kwh)
        return result.direct_kwh_per_day - row.ideal_kwh_per_day, result.saving_fraction(row)

    base_kwh, base_fraction = at(base)
    rows: list[dict[str, Any]] = []
    for name, res in variants.items():
        kwh, fraction = at(res)
        rows.append(
            {
                "variant": name,
                "saving_kwh_per_day": kwh,
                "delta_saving_kwh_per_day": kwh - base_kwh,
                "saving_fraction": fraction,
                "delta_saving_percentage_points": 100.0 * (fraction - base_fraction),
                "direct_kwh_per_day": res.direct_kwh_per_day,
            }
        )
    rows.sort(key=lambda r: -abs(r["delta_saving_kwh_per_day"]))
    return rows
