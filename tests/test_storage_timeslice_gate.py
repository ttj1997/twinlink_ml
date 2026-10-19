"""Tests of the gate logic (L1/L2/L3), the sensitivity ranking and the study grids."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from twinlink_ml.storage_timeslice import CapacityRow, Scenario, ScenarioResult, is_realistic
from twinlink_ml.storage_timeslice.gate import (
    DIRECT_FLOOR_KWH_PER_DAY,
    capacity_at_fraction_of_maximum,
    gate_cells,
    gate_verdict,
    rank_sensitivity,
    subset_verdicts,
)

CAPS = (0.0, 500.0, 2000.0, 5000.0, 10000.0, 20000.0)


def _result(
    direct: float, ideal: list[float], rule: list[float], label: str = "s"
) -> ScenarioResult:
    rows = tuple(
        CapacityRow(c, i, 50.0, r, 50.0, True) for c, i, r in zip(CAPS, ideal, rule, strict=True)
    )
    return ScenarioResult(label, "fp", (), (), direct, rows)


def _scenario(**kw: object) -> Scenario:
    return replace(Scenario("A+C", "product_1", "day", "shift"), **kw)  # type: ignore[arg-type]


def test_l3_when_no_cell_reaches_five_percent() -> None:
    res = _result(1000.0, [1000, 980, 960, 955, 951, 950], [1000, 980, 960, 955, 951, 950])
    cells, counts = gate_cells([(_scenario(), res)])
    v = gate_verdict(cells, counts)
    assert v["reading"] == "L3"
    assert v["n_cells"] == 4  # 0 and 20 MWh are not cells
    assert v["max_saving_fraction"] == pytest.approx(0.049)


def test_l2_when_saving_but_rule_is_close() -> None:
    res = _result(1000.0, [1000, 900, 800, 700, 650, 600], [1000, 905, 810, 710, 660, 610])
    v = gate_verdict(gate_cells([(_scenario(), res)])[0])
    assert v["reading"] == "L2"
    assert v["n_cells_saving_at_least_5_percent"] == 4
    assert v["n_cells_saving_and_gap"] == 0


def test_l1_needs_saving_and_gap_in_the_same_cell() -> None:
    # gap of 3 points exists only where the saving is below 5 %: still L2
    res = _result(1000.0, [1000, 990, 960, 940, 940, 940], [1000, 1020, 1000, 940, 940, 940])
    # cells: C=500 (G 1 %, D 3 %), 2000 (G 4 %, D 4 %), 5000 (G 6 %, D 0), 10000 (G 6 %, D 0)
    assert gate_verdict(gate_cells([(_scenario(), res)])[0])["reading"] == "L2"
    both = _result(1000.0, [1000, 990, 940, 900, 900, 900], [1000, 990, 980, 940, 900, 900])
    assert gate_verdict(gate_cells([(_scenario(), both)])[0])["reading"] == "L1"


def test_floor_and_unrealistic_scenarios_are_excluded() -> None:
    big = _result(1000.0, [1000, 800, 700, 600, 500, 400], [1000, 800, 700, 600, 500, 400])
    small = _result(
        DIRECT_FLOOR_KWH_PER_DAY - 1.0, [99, 50, 10, 0, 0, 0], [99, 50, 10, 0, 0, 0], "small"
    )
    out_of_band = _scenario(hall_kw=500.0)
    cells, counts = gate_cells([(_scenario(), small), (out_of_band, big)])
    assert cells == []
    assert counts["n_excluded_floor"] == 1 and counts["n_unrealistic"] == 1
    assert gate_verdict(cells, counts)["reading"] == "L3"


def test_realism_predicate_can_be_replaced() -> None:
    big = _result(1000.0, [1000, 800, 700, 600, 500, 400], [1000, 800, 700, 600, 500, 400])
    off_band = _scenario().with_placeholders(solid_fraction=0.634)
    assert not is_realistic(off_band)
    assert gate_cells([(off_band, big)])[1]["n_unrealistic"] == 1
    cells, counts = gate_cells([(off_band, big)], realistic=lambda _: True)
    assert len(cells) == 4 and counts["n_unrealistic"] == 0


def test_subset_verdicts_split_by_value() -> None:
    a = _result(1000.0, [1000, 900, 800, 700, 650, 600], [1000, 900, 800, 700, 650, 600], "a")
    c = _result(1000.0, [1000, 999, 998, 997, 996, 995], [1000, 999, 998, 997, 996, 995], "c")
    cells, _ = gate_cells([(_scenario(site="A"), a), (_scenario(site="C"), c)])
    out = subset_verdicts(cells, {"site": lambda s: s.site})
    assert out["site=A"]["reading"] == "L2" and out["site=C"]["reading"] == "L3"


def test_capacity_at_ninety_percent() -> None:
    res = _result(1000.0, [1000, 700, 500, 410, 400, 400], [1000, 700, 500, 410, 400, 400])
    assert capacity_at_fraction_of_maximum(res) == 5000.0  # 590 of 600 kWh/d saved at 5 MWh
    flat = _result(1000.0, [1000] * 6, [1000] * 6)
    assert capacity_at_fraction_of_maximum(flat) is None


def test_rank_sensitivity_orders_by_absolute_change() -> None:
    base = _result(1000.0, [1000, 900, 800, 700, 650, 600], [1000, 900, 800, 700, 650, 600])
    up = _result(1000.0, [1000, 900, 800, 600, 650, 600], [1000, 900, 800, 600, 650, 600])
    down = _result(1000.0, [1000, 900, 800, 690, 650, 600], [1000, 900, 800, 690, 650, 600])
    ranking = rank_sensitivity(base, {"small": down, "big": up}, 5000.0)
    assert [r["variant"] for r in ranking] == ["big", "small"]
    assert ranking[0]["delta_saving_kwh_per_day"] == pytest.approx(100.0)
    assert ranking[0]["delta_saving_percentage_points"] == pytest.approx(10.0)


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "gate_storage_timeslice.py"
    spec = importlib.util.spec_from_file_location("gate_storage_timeslice_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grids_are_complete_unique_and_realistic() -> None:
    script = _load_script()
    grid = script.main_grid()
    assert len({s.label for s in grid}) == len(grid)
    assert all(is_realistic(s) for s in grid)
    assert all(s.mode != "weekly" or s.horizon == "week" for s in grid)
    # site A and A+C: 3 compressor levels; site C: one
    n_per_level = 2 * 4 * 5 * 2 * 2 * 2  # days x weather x (horizon, mode) x hall x water x loss
    assert len(grid) == n_per_level * (3 + 1 + 3)
    for base in script.reference_scenarios().values():
        variants = script.sensitivity_variants(base)
        assert len({s.label + s.fingerprint for s in variants.values()}) == len(variants)
        assert all(is_realistic(s) for s in variants.values())
        assert all(s != base for s in variants.values())
