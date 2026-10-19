"""Tests of the Stage-B gate of the heat-pump study (pre-registered L1/L2/L3)."""

from __future__ import annotations

from typing import Any

import pytest

from twinlink_ml.heatpump.gate import (
    cells_of,
    foresight_map,
    gate_verdict,
    smallest_amplitude,
    subset_verdicts,
)


def _rec(
    key: str,
    gaps: list[float],
    gains: list[float],
    *,
    realistic: bool = True,
    amplitude: float = 0.4,
    kind: str = "cost",
    direct: float = 1000.0,
) -> dict[str, Any]:
    return {
        "key": key,
        "meta": {"amplitude": amplitude, "kind": kind},
        "realistic": realistic,
        "direct_kwh_per_day": direct,
        "capacities_kwh": [500.0, 2000.0][: len(gaps)],
        "gap_fraction": gaps,
        "storage_gain_fraction": gains,
    }


def test_readings() -> None:
    cells, counts = cells_of([_rec("a", [0.0, 0.019], [0.01, 0.06])])
    assert gate_verdict(cells, counts)["reading"] == "L2"
    cells, _ = cells_of([_rec("a", [0.0, 0.02], [0.01, 0.01])])
    assert gate_verdict(cells)["reading"] == "L1"
    cells, _ = cells_of([_rec("a", [0.0, 0.01], [0.01, 0.04])])
    assert gate_verdict(cells)["reading"] == "L3"


def test_unrealistic_and_floor_do_not_count() -> None:
    cells, counts = cells_of(
        [
            _rec("u", [0.5, 0.5], [0.5, 0.5], realistic=False),
            _rec("f", [0.5, 0.5], [0.5, 0.5], direct=50.0),
            _rec("ok", [0.0, 0.0], [0.0, 0.0]),
        ]
    )
    v = gate_verdict(cells, counts)
    assert v["reading"] == "L3"
    assert v["n_unrealistic"] == 1
    assert v["n_excluded_floor"] == 1
    assert v["n_cells_realistic"] == 2


def test_maps_and_subsets() -> None:
    recs = [
        _rec("a", [0.0, 0.01], [0.0, 0.02], amplitude=0.2),
        _rec("b", [0.01, 0.03], [0.02, 0.06], amplitude=0.6),
        _rec("c", [0.05, 0.08], [0.1, 0.1], amplitude=0.8, realistic=False),
    ]
    cells, _ = cells_of(recs)
    m = foresight_map(cells, lambda c: str(c.meta["kind"]))
    assert m["cost"]["0.8"]["max_gap|2000"] == pytest.approx(0.08)
    s = smallest_amplitude(cells, lambda c: str(c.meta["kind"]))
    assert s["cost"]["2000"] == pytest.approx(0.6)
    assert s["cost"]["500"] == pytest.approx(0.8)
    sub = subset_verdicts(cells, {"amp": lambda c: c.meta["amplitude"]})
    assert sub["amp=0.6"]["reading"] == "L1"
    assert sub["amp=0.8"]["reading"] == "L3"  # unrealistic cells only
