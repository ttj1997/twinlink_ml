"""Tests of :mod:`twinlink_ml.case_ranking` (case-ranking study, docs/38_case_ranking.md).

The reproduction test pins, at the input point of the workbook, the total fuel of every case and
the ranking of docs/11 sections 13.3 and 13.6 in all three modes. It needs no file of ``local/``:
the inputs are the values committed in docs/11 section 13.1.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from twinlink_ml.case_ranking import ranking as rk
from twinlink_ml.case_ranking.config import SET_NAMES, load_config
from twinlink_ml.case_ranking.evaluator import (
    MODES,
    FlowOptimiser,
    FlowsheetCaseEvaluator,
    exhaust_relative_humidity,
)
from twinlink_ml.case_ranking.rule import FEATURE_NAMES, features
from twinlink_ml.case_ranking.space import (
    INPUT_NAMES,
    Box,
    OperatingPoint,
    ambient_humidity_kg_per_kg,
    ambient_relative_humidity,
    sobol_points,
)
from twinlink_ml.case_ranking.study import assemble, run_task, task_list, task_points
from twinlink_ml.case_ranking.workbook import WORKBOOK_POINT, excel_inputs
from twinlink_ml.flowsheet import Flowsheet, FlowsheetConfig
from twinlink_ml.partner_v4.inputs import ExcelDryerInputs, ExcelInputs, ExcelMaterialInputs
from twinlink_ml.partner_v4_1.inputs import ExcelInputsV41

# docs/11 section 13.6, fuel in kg/h (4 decimals), case 0 = 1.3170 in every mode.
DOCS11_FUEL = {
    "partner": [1.3170, 1.0810, 1.0863, 1.1097, 1.1275, 1.1240, 1.1425],
    "physical_unpressurised_95": [1.3170, 0.9430, 0.9539, 1.0966, 1.1333, 1.0966, 1.1333],
    "physical_pressurised": [1.3170, 0.8805, 0.8960, 1.0080, 1.0564, 1.0341, 1.0754],
}
DOCS11_RANKING = {
    "partner": "1 < 2 < 3 < 5 < 4 < 6",
    "physical_unpressurised_95": "1 < 2 < 3 = 5 < 4 = 6",
    "physical_pressurised": "1 < 2 < 3 < 5 < 4 < 6",
}


def _ranking_text(fuel: np.ndarray, exact: float = 1e-9) -> str:
    cases = list(range(1, 7))
    order = sorted(cases, key=lambda n: (fuel[n], n))
    text = str(order[0])
    for a, b in itertools.pairwise(order):
        text += (" = " if fuel[b] - fuel[a] < exact else " < ") + str(b)
    return text


@pytest.fixture(scope="module")
def workbook_results():
    return {m: FlowsheetCaseEvaluator(MODES[m]).evaluate(WORKBOOK_POINT) for m in MODES}


# ------------------------------------------------------------------ reproduction (first)
@pytest.mark.parametrize("mode", list(MODES))
def test_reproduces_docs11_fuel_and_ranking(workbook_results, mode):
    fuel = workbook_results[mode].fuel()
    assert fuel == pytest.approx(DOCS11_FUEL[mode], abs=5e-5)
    assert _ranking_text(fuel) == DOCS11_RANKING[mode]


def test_partner_gap_case_1_to_2(workbook_results):
    fuel = workbook_results["partner"].fuel()
    assert fuel[2] - fuel[1] == pytest.approx(0.0053, abs=5e-5)


def test_workbook_point_equals_hand_built_inputs():
    mat = ExcelMaterialInputs(1.006, 45000.0, 80.0, 4.185, 1.0, 1.906, 2.0, 2500.0, 4.185)
    hand = ExcelInputsV41(
        ExcelInputs(
            ExcelDryerInputs(120.0, 6.0, 300.0, 12.0, 0.4, 20.0),
            ExcelDryerInputs(100.0, 6.0, 150.0, 3.0, 0.4, 20.0),
            mat,
            20.0,
            1.0,
        ),
        (40.0, 30.0),
    )
    ours = excel_inputs(WORKBOOK_POINT)
    m = MODES["partner"]
    for inputs_a, inputs_b in ((hand, ours),):
        fa = Flowsheet(FlowsheetConfig(inputs_a.to_si(), m.switches)).evaluate(1)
        fb = Flowsheet(FlowsheetConfig(inputs_b.to_si(), m.switches)).evaluate(1)
        assert fb.fuel_corrected_kg_per_h == pytest.approx(fa.fuel_corrected_kg_per_h, abs=1e-12)
    assert WORKBOOK_POINT.humidity_kg_per_kg == pytest.approx(0.006, abs=1e-15)


# ------------------------------------------------------------------ space and config
def test_humidity_round_trip():
    for t, r in ((-10.0, 0.95), (0.0, 0.2), (40.0, 0.95)):
        y = ambient_humidity_kg_per_kg(t, r)
        assert ambient_relative_humidity(t, y) == pytest.approx(r, rel=1e-12)


def test_config_box_regions_and_sizes():
    cfg = load_config()
    assert cfg.box.contains(WORKBOOK_POINT)
    for r in cfg.regions.values():
        assert r.contains(WORKBOOK_POINT) or r is cfg.regions["plant_like"]
    plant = cfg.regions["plant_like"].interval("inlet_2_celsius")
    assert (plant.low, plant.high) == (75.0, 95.0)
    # the plant-like supply air lies inside the box for both dryers
    for name in ("inlet_1_celsius", "inlet_2_celsius"):
        i = cfg.box.interval(name)
        assert i.low < 75.0 and i.high > 95.0
    for v in cfg.points.values():
        assert set(v) == set(SET_NAMES)


def test_sobol_nested_prefixes_and_bounds():
    box = load_config().box
    a = sobol_points(box, 64, 7)
    b = sobol_points(box, 128, 7)
    np.testing.assert_array_equal(a, b[:64])
    assert (a >= box.low).all() and (a <= box.high).all()


def test_neighbourhood_is_cut_at_the_box():
    box = Box.from_mapping(dict.fromkeys(INPUT_NAMES, (0.0, 10.0)))
    centre = OperatingPoint.from_array([10.0] + [5.0] * 7)
    nb = box.neighbourhood(centre, 0.1)
    assert nb.interval(INPUT_NAMES[0]).high == 10.0
    assert nb.interval(INPUT_NAMES[0]).low == pytest.approx(9.0)
    assert nb.interval(INPUT_NAMES[1]).low == pytest.approx(4.0)


# ------------------------------------------------------------------ evaluator
def test_optimised_never_worse_than_fixed_and_pinning_is_exact():
    cfg = load_config()
    m = MODES["partner"]
    fixed = FlowsheetCaseEvaluator(m).evaluate(WORKBOOK_POINT).fuel()
    opt = FlowsheetCaseEvaluator(m, cfg.optimiser).evaluate(WORKBOOK_POINT)
    assert (opt.fuel()[1:] <= fixed[1:] + 1e-12).all()
    assert opt.fuel()[0] == fixed[0]
    lo, hi = cfg.optimiser.bounds_kg_per_h
    for c in opt.cases[1:]:
        assert lo <= c.htm_flow_kg_per_h[0] <= hi and lo <= c.htm_flow_kg_per_h[1] <= hi
    ev = FlowsheetCaseEvaluator(MODES["physical_pressurised"])
    fs = ev.flowsheet(WORKBOOK_POINT, (55.0, 25.0))
    ex = fs.exhausts()
    pinned = {i: (ex[i].temperature_celsius, ex[i].humidity_kg_per_kg) for i in (1, 2)}
    fp = ev.flowsheet(WORKBOOK_POINT, (55.0, 25.0), pinned)
    for n in range(7):
        assert fp.evaluate(n).fuel_corrected_kg_per_h == fs.evaluate(n).fuel_corrected_kg_per_h


def test_flow_optimiser_finds_interior_minimum():
    opt = FlowOptimiser(bounds_kg_per_h=(20.0, 150.0))
    x, f, nfev = opt.search(lambda b: (b[0] - 61.0) ** 2 + (b[1] - 33.0) ** 2, (40.0, 30.0))
    assert x == pytest.approx((61.0, 33.0), abs=0.2) and f < 0.05 and nfev > 26


def test_admissible_exhausts_equal_in_every_mode():
    """Below 90 % relative humidity the saturation limit does not bind: same exhausts."""
    pts = [WORKBOOK_POINT, WORKBOOK_POINT.with_(inlet_1_celsius=250.0, spray_load_1_kg_per_kg=0.08)]
    for p in pts:
        res = {m: FlowsheetCaseEvaluator(MODES[m]).evaluate(p) for m in MODES}
        assert max(res["partner"].exhaust_relative_humidity) <= 0.9
        for r in res.values():
            assert r.exhaust_celsius == pytest.approx(res["partner"].exhaust_celsius, abs=1e-9)


def test_exhaust_relative_humidity_below_zero_is_inadmissible():
    assert exhaust_relative_humidity(-1.0, 0.001) == float("inf")
    assert 0.0 < exhaust_relative_humidity(60.0, 0.02) < 1.0


# ------------------------------------------------------------------ ranking
def _fuel(rows):
    return np.array(rows, dtype=np.float64)


def test_case_one_flags_and_ranks():
    fuel = _fuel(
        [
            [2.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],  # case 1 strictly best
            [2.0, 1.0, 1.0, 1.2, 1.3, 1.4, 1.5],  # exact tie 1 = 2
            [2.0, 1.005, 1.0, 1.2, 1.3, 1.4, 1.5],  # 2 better by 0.005 < 1 % of 1.0
            [2.0, 1.05, 1.0, 1.2, 1.3, 1.4, 1.5],  # 2 better by 0.05
        ]
    )
    f = rk.case_one_flags(fuel, 1e-9, 0.01)
    assert f["strict"].tolist() == [True, False, False, False]
    assert f["co_best"].tolist() == [True, True, False, False]
    assert f["practical"].tolist() == [True, True, True, False]
    ranks = rk.competition_ranks(fuel, 1e-9)
    assert ranks[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert ranks[1].tolist() == [1, 1, 3, 4, 5, 6]
    assert rk.winner_labels(fuel, 1e-9).tolist() == ["1", "1=2", "2", "2"]
    r = rk.regret(fuel)
    assert r["kg_per_h"] == pytest.approx([0.0, 0.0, 0.005, 0.05])
    assert r["relative_to_best_saving"][3] == pytest.approx(0.05)
    assert rk.top_two_gap(fuel) == pytest.approx([0.1, 0.0, 0.005, 0.05])


def test_practical_threshold_never_below_exact():
    fuel = _fuel([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])  # no saving
    assert rk.practical_threshold(fuel, 0.01, 1e-9)[0] == 1e-9


def test_clopper_pearson():
    lo, hi = rk.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-4)
    lo, hi = rk.clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.6915, abs=1e-4)
    s = rk.share(np.array([True] * 99 + [False]))
    assert s.share == 0.99 and s.ci95[0] < 0.99 < s.ci95[1]


# ------------------------------------------------------------------ rule features
def test_features_shape_and_sign():
    x = WORKBOOK_POINT.as_array()[None, :]
    f = features(x, np.array([[137.59, 102.86]]), np.array([[0.0659, 0.0240]]))
    assert f.shape == (1, len(FEATURE_NAMES))
    d = dict(zip(FEATURE_NAMES, f[0], strict=True))
    assert d["exhaust_1_minus_2_k"] == pytest.approx(34.73)
    assert d["demand_1_kw"] > d["demand_2_kw"] > 0.0
    assert d["humidity_g_per_kg"] == pytest.approx(6.0)


# ------------------------------------------------------------------ units
def test_units_cover_every_point_once():
    cfg = load_config()
    tasks = task_list(cfg)
    for v in ("fixed", "optimised"):
        for s in SET_NAMES:
            sizes = [len(task_points(cfg, t, False)) for t in tasks if t[:3] == ("set", v, s)]
            assert sum(sizes) == cfg.points[v][s]
    assert all(len(task_points(cfg, t, True)) <= cfg.dry_run_points_per_chunk for t in tasks)
    assert len({json.dumps(list(t)) for t in tasks}) == len(tasks)


def test_run_task_and_assemble_dry():
    cfg = load_config()
    t = ("cut1d", "fixed", "inlet_1_celsius", 0)
    res = run_task(cfg, t, dry_run=True)
    assert res["point_errors"] == 0 and len(res["points"]) == cfg.dry_run_points_per_chunk
    assert res["peak_memory_mb"] > 0.0  # the dry run must report memory per worker
    arrays = assemble(cfg, {json.dumps(list(t)): res}, dry_run=True)
    fuel = arrays["fixed/cut1d_inlet_1_celsius/partner/fuel"]
    assert fuel.shape == (cfg.dry_run_points_per_chunk, 7)
    assert np.isfinite(fuel).all()
    assert arrays["fixed/full/present"].sum() == 0
