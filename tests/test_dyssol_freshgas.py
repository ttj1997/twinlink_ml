"""Surrogate of the one-line flowsheet with the fresh-gas state as an input (seven inputs).

Extension of the package ``twinlink_ml.dyssol_steady`` for ``docs/NN_dyssol_freshgas.md``: box and
design with the relative humidity of the fresh gas, pilot grid with one automatic extension,
second test set from a sweep sheet, control tabulated over the fresh-gas temperature.

All tests use laboratory-scale inputs already committed in docs/11 and a synthetic sheet built
from them; no partner value. The settings of Befund 128 must stay what they were.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from twinlink_ml.case_ranking.space import ambient_humidity_kg_per_kg
from twinlink_ml.dyssol_steady import study
from twinlink_ml.dyssol_steady.design import (
    RELATIVE_HUMIDITY,
    VARIED,
    VARIED_FRESH_GAS,
    apply_point,
    point_row,
    resolve_box,
    sobol_points,
)
from twinlink_ml.dyssol_steady.materials import (
    DEFAULT_HEAT_CAPACITY,
    DEFAULT_MOLAR_MASS,
    DyssolProperties,
)
from twinlink_ml.dyssol_steady.models import PolynomialRegression
from twinlink_ml.dyssol_steady.reference import LineInputs, evaluate
from twinlink_ml.dyssol_steady.sweep_sheet import (
    RESULT_KEYS,
    SheetLine,
    SweepSheet,
    sweep_points,
    sweep_table,
)
from twinlink_ml.flowsheet.validators import saturation_humidity_kg_per_kg

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "conf" / "dyssol_steady"
FRESHGAS = CONF / "dyssol_freshgas.yaml"

PROPS = DyssolProperties(
    cp_gas=1006.0,
    cp_vapour=1906.0,
    cp_liquid=4185.0,
    cp_solid=DEFAULT_HEAT_CAPACITY,
    heat_of_vaporization_j_per_kg=40660.5 / DEFAULT_MOLAR_MASS,
    boiling_point_k=373.15,
    gas_enthalpy_linear=(1005.0, 0.0),
    vapour_enthalpy_shomate=(
        1670.36,
        0.379262,
        0.000377093,
        -1.40685e-07,
        4.5594e06,
        -1.3926e07,
        -1.34234e07,
    ),
    gas_enthalpy_range_k=(200.0, 2773.0),
    vapour_enthalpy_range_k=(290.0, 1700.0),
)
LAB = LineInputs(
    dry_gas_kg_per_h=120.0,
    gas_humidity_g_per_kg=6.0,
    gas_celsius=20.0,
    dryer_inlet_celsius=300.0,
    spray_kg_per_h=12.0,
    solid_fraction=0.4,
    spray_celsius=20.0,
    medium_kg_per_h=40.0,
    equilibrium_moisture_g_per_kg=2.0,
    heater_efficiency_percent=80.0,
    calorific_value_kj_per_kg=45000.0,
)
ENTRY: dict[str, Any] = {
    "dry_gas_kg_per_h": [100, 140],
    "dryer_inlet_celsius": [200, 300],
    "spray_kg_per_h": {"relative": 0.25},
    "solid_fraction": {"relative": 0.25},
    "medium_kg_per_h": [0, 60],
    "gas_celsius": [-10, 30],
    "gas_relative_humidity": [0.2, 0.95],
}
FLOWS = (0.0, 20.0, 40.0)


def _script(name: str) -> ModuleType:
    """Load a driver from ``scripts/`` (not a package) as a module."""
    spec = importlib.util.spec_from_file_location(f"_under_test_{name}", ROOT / "scripts" / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- design
def test_box_with_the_fresh_gas_state_and_humidity_from_relative_humidity() -> None:
    box = resolve_box([ENTRY], [LAB], VARIED_FRESH_GAS)
    assert box.dimension == 7
    assert box.names == tuple(f"line1/{v}" for v in VARIED_FRESH_GAS)
    assert box.lower[4:] == pytest.approx((0.0, -10.0, 0.2))
    assert box.upper[4:] == pytest.approx((60.0, 30.0, 0.95))
    x = sobol_points(box, 256, seed=5)
    assert x.shape == (256, 7) and bool(box.contains(x).all())
    for row in x[:64]:
        (line,) = apply_point([LAB], row, VARIED_FRESH_GAS)
        assert line.gas_celsius == pytest.approx(row[5])
        expected = 1000.0 * ambient_humidity_kg_per_kg(row[5], row[6])
        assert line.gas_humidity_g_per_kg == pytest.approx(expected, rel=1e-14)
        # the relative humidity keeps the fresh gas below saturation; the repository's function
        # (IAPWS-IF97) is defined from 0 degC on, Buck's below
        if row[5] >= 0.0:
            saturation = 1000.0 * saturation_humidity_kg_per_kg(row[5])
            assert line.gas_humidity_g_per_kg < saturation
        assert point_row([line], VARIED_FRESH_GAS) == pytest.approx(row, rel=1e-12, abs=1e-12)
        assert line.spray_celsius == LAB.spray_celsius
    # colder fresh gas holds less water at the same relative humidity
    cold = apply_point([LAB], np.array([120, 250, 12, 0.4, 30, -10.0, 0.9]), VARIED_FRESH_GAS)[0]
    warm = apply_point([LAB], np.array([120, 250, 12, 0.4, 30, 30.0, 0.9]), VARIED_FRESH_GAS)[0]
    assert cold.gas_humidity_g_per_kg < 2.0 < 20.0 < warm.gas_humidity_g_per_kg


def test_box_rejects_unusable_humidity_entries_and_unknown_inputs() -> None:
    with pytest.raises(ValueError, match="interval"):
        resolve_box([{**ENTRY, RELATIVE_HUMIDITY: {"relative": 0.1}}], [LAB], VARIED_FRESH_GAS)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        resolve_box([{**ENTRY, RELATIVE_HUMIDITY: [0.5, 1.2]}], [LAB], VARIED_FRESH_GAS)
    with pytest.raises(ValueError, match="exactly"):
        resolve_box([ENTRY], [LAB])  # the five-input box does not take the two new entries
    with pytest.raises(ValueError, match="distinct names"):
        resolve_box([ENTRY], [LAB], (*VARIED, "gas_humidity_g_per_kg", "gas_celsius"))


def test_the_five_input_design_is_unchanged() -> None:
    entry = {k: v for k, v in ENTRY.items() if k in VARIED}
    box = resolve_box([entry], [LAB])
    assert box.names == tuple(f"line1/{v}" for v in VARIED)
    (line,) = apply_point([LAB], np.array([110.0, 250.0, 10.0, 0.35, 30.0]))
    assert line.gas_humidity_g_per_kg == LAB.gas_humidity_g_per_kg
    assert line.gas_celsius == LAB.gas_celsius
    assert point_row([line]) == pytest.approx([110.0, 250.0, 10.0, 0.35, 30.0])


# ----------------------------------------------------------------------------- settings
def test_settings_of_befund_128_keep_their_fingerprint_and_defaults() -> None:
    config = study.load_config()
    with gzip.open(
        ROOT / "artifacts" / "runs" / "dyssol_steady_v1_one_line_training.json.gz", "rt"
    ) as fh:
        header = json.load(fh)["header"]
    assert config.fingerprint == header["config_fingerprint"]
    assert config.varied == VARIED and config.box_file == "box_local.yaml"
    assert config.test_set_name == "in-box Sobol test set, independent of the training pool"
    assert config.sheet_test is None and config.prediction is None
    assert study.pilot_grid(config.pilot) == list(config.pilot["grid"])
    assert config.mlp_settings("one_line").hidden == (64, 64)
    with pytest.raises(ValueError, match="no extension"):
        study.pilot_grid(config.pilot, {"learning_rate": [0.1]})


def test_committed_settings_of_the_fresh_gas_study_are_consistent() -> None:
    config = study.load_config(FRESHGAS)
    old = study.load_config()
    assert config.varied == VARIED_FRESH_GAS and set(config.cases) == {"one_line"}
    assert config.cases["one_line"] == old.cases["one_line"]  # the source of Befund 128 / 130
    assert config.box_file == "box_freshgas_local.yaml"
    assert config.test_set_name == "in-box Sobol test set, 7 inputs"
    assert (config.pool_size, config.test_size) == (old.pool_size, old.test_size) == (1024, 256)
    assert config.budgets == old.budgets and config.seeds == old.seeds
    assert config.threshold_exhaust_k == old.threshold_exhaust_k == 1.0
    assert config.threshold_fuel_percent == old.threshold_fuel_percent == 1.0
    assert config.control_temperature_k == old.control_temperature_k
    assert config.control_relative_percent == old.control_relative_percent
    seeds = {
        config.pool_seed,
        config.test_seed,
        config.pilot["pool_seed"],
        config.pilot["test_seed"],
    }
    assert len(seeds) == 4
    assert not seeds & {
        old.pool_seed,
        old.test_seed,
        old.pilot["pool_seed"],
        old.pilot["test_seed"],
    }
    # the prediction: 36 coefficients, first step of the grid above 1.5 x 36
    coefficients = PolynomialRegression(2).features(np.zeros((1, len(config.varied)))).shape[1]
    assert coefficients == 36
    assert config.prediction == {"model": "quadratic", "n_star": 64}
    assert min(n for n in config.budgets if n > 1.5 * coefficients) == 64
    assert config.sheet_test is not None
    assert config.sheet_test["name"] == "partner sheet 2026-11-24, process 2, 11 points"
    assert config.sheet_test["gas_flow_reading"] == "dry"
    edges = config.raw["control"]["fresh_gas_bins_celsius"]
    assert edges == sorted(edges) and len(edges) >= 3
    text = (CONF / "box_freshgas_local.yaml.example").read_text(encoding="utf-8")
    assert "LOWER" in text and "relative: R" in text  # the template holds placeholders only
    assert not any(ch.isdigit() for ch in text.split("cases:")[1])


def test_pilot_grid_widens_the_grid_of_befund_128_on_three_axes() -> None:
    config, old = study.load_config(FRESHGAS), study.load_config()
    axes = config.pilot["axes"]
    grid = study.pilot_grid(config.pilot)
    assert len(grid) == 4 * 3 * 3 * 3 == len({json.dumps(s, sort_keys=True) for s in grid})
    assert grid[0] == {
        "hidden": [32, 32],
        "learning_rate": 0.0003,
        "steps": 2000,
        "weight_decay": 0.0,
    }
    assert grid[1]["weight_decay"] == 0.001 and grid[3]["steps"] == 5000  # last axis fastest
    before = {
        "learning_rate": sorted({s["learning_rate"] for s in old.pilot["grid"]}),
        "steps": sorted({s["steps"] for s in old.pilot["grid"]}),
        "width": [old.hidden[0]],
    }
    for axis, levels in before.items():
        assert min(axes[axis]) < levels[0] and max(axes[axis]) > levels[-1]
        assert set(levels) <= set(axes[axis])
    assert set(axes["weight_decay"]) == {s["weight_decay"] for s in old.pilot["grid"]}
    for setting in old.pilot["grid"]:  # every setting of Befund 128 is part of the new grid
        assert {"hidden": [64, 64], **setting} in [{**g, "hidden": g["hidden"]} for g in grid]
    for axis, rule in config.pilot["extension"].items():
        assert rule.get("lower", -1) < axes[axis][0] and rule.get("upper", 1e99) > axes[axis][-1]
    assert "lower" not in config.pilot["extension"]["weight_decay"]


def test_pilot_extension_adds_one_level_towards_the_edge_and_keeps_the_indices() -> None:
    pilot = study.load_config(FRESHGAS).pilot
    grid = study.pilot_grid(pilot)
    inside = {"hidden": [64, 64], "learning_rate": 0.001, "steps": 5000, "weight_decay": 0.001}
    none = study.pilot_extension(pilot, inside)
    assert none["extended"] == {} and none["at_edge_without_extension"] == {}
    assert study.pilot_grid(pilot, none["axes"]) == grid
    edge = {"hidden": [128, 128], "learning_rate": 0.0003, "steps": 5000, "weight_decay": 0.0}
    step = study.pilot_extension(pilot, edge)
    assert step["extended"] == {"learning_rate": "lower", "width": "upper"}
    assert step["at_edge_without_extension"] == {"weight_decay": "lower"}
    assert step["axes"]["learning_rate"] == [0.0001, 0.0003, 0.001, 0.003, 0.01]
    assert step["axes"]["width"] == [32, 64, 128, 256]
    assert step["axes"]["steps"] == pilot["axes"]["steps"]
    extended = study.pilot_grid(pilot, step["axes"])
    assert extended[: len(grid)] == grid  # an index names the same setting in both rounds
    assert len(extended) == 5 * 4 * 3 * 3
    new = extended[len(grid) :]
    assert all(s["learning_rate"] == 0.0001 or s["hidden"] == [256, 256] for s in new)
    # the extended grid is not extended again: only the edge is named
    again = study.pilot_extension({"axes": step["axes"]}, {**edge, "learning_rate": 0.0001})
    assert again["extended"] == {}
    assert again["at_edge_without_extension"] == {"learning_rate": "lower", "weight_decay": "lower"}
    with pytest.raises(ValueError, match="at least one level"):
        study.pilot_grid({"axes": {"learning_rate": [0.1]}})


# ----------------------------------------------------------------------------- sheet test set
def _lab_sheet() -> SweepSheet:
    """A synthetic sheet: two processes at laboratory scale over three medium flows."""
    lines = []
    tables = []
    for line in (
        replace(LAB, dry_gas_kg_per_h=100.0, dryer_inlet_celsius=220.0),
        replace(LAB, gas_celsius=10.0, gas_humidity_g_per_kg=5.0, dryer_inlet_celsius=280.0),
    ):
        gas = line.dry_gas_kg_per_h / 3600.0
        lines.append(
            SheetLine(
                gas_kg_per_s=gas,
                gas_water_kg_per_s=gas * line.gas_humidity_g_per_kg / 1000.0,
                gas_celsius=line.gas_celsius,
                dryer_inlet_celsius=line.dryer_inlet_celsius,
                spray_kg_per_s=line.spray_kg_per_h / 3600.0,
                spray_solid_kg_per_s=line.spray_kg_per_h * line.solid_fraction / 3600.0,
                spray_celsius=line.spray_celsius,
                heater_efficiency_percent=line.heater_efficiency_percent,
                calorific_value_kj_per_kg=line.calorific_value_kj_per_kg,
            )
        )
        outputs = [evaluate(PROPS, [replace(line, medium_kg_per_h=m)], (0,))[0] for m in FLOWS]
        tables.append({k: tuple(round(v, 1) for v in vs) for k, vs in sweep_table(outputs).items()})
    return SweepSheet(
        lines=tuple(lines),
        medium_kg_per_h=FLOWS,
        results={key: tuple(t[key] for t in tables) for key in RESULT_KEYS},
        equilibrium_moisture_g_per_kg=LAB.equilibrium_moisture_g_per_kg,
        condensation=False,
    )


def test_sweep_points_are_the_inputs_of_one_process_over_the_flows() -> None:
    sheet = _lab_sheet()
    points = sweep_points(sheet, 1, "dry")
    assert [p.medium_kg_per_h for p in points] == list(FLOWS)
    assert {p.gas_celsius for p in points} == {10.0}
    assert points[0].gas_humidity_g_per_kg == pytest.approx(5.0)
    assert points[1] == replace(points[0], medium_kg_per_h=20.0)


def test_sheet_errors_by_hand_and_against_the_sheet() -> None:
    names = study.target_names(1)
    sheet = _lab_sheet()
    outputs = [(evaluate(PROPS, [p], (0,))[0],) for p in sweep_points(sheet, 1, "dry")]
    truth = study.target_matrix(outputs)
    reported = {key: sheet.results[key][1] for key in RESULT_KEYS}
    exact = study.sheet_errors(truth, truth, names, reported)
    assert exact["exhaust_max_k"] == 0.0 and exact["fuel_max_percent"] == 0.0
    total = exact["against_sheet"]["total"]
    assert total["n"] == total["within_half"] == len(RESULT_KEYS) * len(FLOWS)
    shifted = truth.copy()
    shifted[:, names.index("line1/exhaust_celsius")] += [0.0, 0.3, -1.2]
    shifted[1, names.index("line1/fuel_kg_per_h")] *= 1.02
    errors = study.sheet_errors(truth, shifted, names, reported)
    assert errors["exhaust_abs_k"] == pytest.approx([0.0, 0.3, 1.2])
    assert errors["exhaust_max_k"] == pytest.approx(1.2)
    assert errors["fuel_rel_percent"] == pytest.approx([0.0, 2.0, 0.0])
    assert errors["fuel_max_percent"] == pytest.approx(2.0)
    per = errors["against_sheet"]["per_quantity"]
    # the exhaust is two quantities of the sheet (exhaust and medium, hot side)
    assert per["exhaust_celsius"]["within_one"] == per["medium_hot_celsius"]["within_one"] == 1
    assert per["product_celsius"]["within_half"] == len(FLOWS)
    assert "against_sheet" not in study.sheet_errors(truth, shifted, names)
    with pytest.raises(ValueError, match="one line"):
        study.sheet_errors(np.zeros((2, 10)), np.zeros((2, 10)), study.target_names(2))


def _sheet_cells(config: study.StudyConfig, exhaust: list[float], fuel: list[float]) -> list[dict]:
    return [
        {
            "model": "quadratic",
            "n": 64,
            "seed": seed,
            "metrics_sheet": {
                "exhaust_abs_k": [0.0, e],
                "fuel_rel_percent": [f, 0.0],
                "exhaust_max_k": e,
                "fuel_max_percent": f,
            },
        }
        for seed, e, f in zip(config.seeds, exhaust, fuel, strict=True)
    ]


def test_sheet_reading_uses_the_largest_error_and_the_median_over_seeds() -> None:
    config = study.load_config(FRESHGAS)
    good = study.sheet_reading(
        _sheet_cells(config, [0.2, 0.4, 0.9, 1.5, 3.0], [0.1, 0.2, 0.3, 0.4, 5.0]),
        config,
        "quadratic",
        64,
    )
    assert good["verdict"] == "B0" and good["points"] == 2
    assert good["exhaust_max_k"]["median"] == pytest.approx(0.9)
    assert good["seeds_meeting_both"] == 3
    one_number = study.sheet_reading(
        _sheet_cells(config, [0.2, 0.4, 0.9, 1.5, 3.0], [0.1, 0.2, 1.3, 1.4, 5.0]),
        config,
        "quadratic",
        64,
    )
    assert one_number["verdict"] == "B1"  # one threshold alone is not enough
    assert study.sheet_reading([], config, None, None)["verdict"] == "not applicable"
    with pytest.raises(ValueError, match="one per seed"):
        study.sheet_reading(
            _sheet_cells(config, [1.0] * 5, [1.0] * 5)[:-1], config, "quadratic", 64
        )


def test_prediction_is_confirmed_only_by_the_predicted_n_star() -> None:
    prediction = {"model": "quadratic", "n_star": 64}
    hit = {"n_star": {"quadratic": 64, "mlp": 128}, "data_need_runs": 64, "reached_by": "quadratic"}
    assert study.prediction_check(hit, prediction)["confirmed"] is True
    for found in (32, 128, None):
        miss = {"n_star": {"quadratic": found}, "data_need_runs": found, "reached_by": None}
        check = study.prediction_check(miss, prediction)
        assert check["confirmed"] is False and check["found_n_star"] == found


def test_control_by_bins_tabulates_the_largest_deviation() -> None:
    rows = study.control_by_bins(
        [-10.0, -6.0, -5.0, 4.9, 30.0],
        [1e-13, 3e-13, 2e-13, 5e-13, 7e-13],
        [1, 2, 3, 4, 5],
        [-10.0, -5.0, 5.0, 30.0],
    )
    assert [r["points"] for r in rows] == [2, 2, 1]  # the last bin includes its upper edge
    assert [r["max_temperature_k"] for r in rows] == [3e-13, 5e-13, 7e-13]
    assert [r["max_relative_percent"] for r in rows] == [2.0, 4.0, 5.0]
    assert (rows[0]["lower"], rows[-1]["upper"]) == (-10.0, 30.0)
    with pytest.raises(ValueError, match="outside"):
        study.control_by_bins([31.0], [0.0], [0.0], [-10.0, 30.0])
    with pytest.raises(ValueError, match="increase"):
        study.control_by_bins([], [], [], [0.0, 0.0])


# ----------------------------------------------------------------------------- driver
@pytest.fixture(scope="module")
def run() -> ModuleType:
    return _script("run_dyssol_steady.py")


def _context_parts(run: ModuleType, tmp_path: Path, monkeypatch, **sheet_test: Any):
    config = study.load_config(FRESHGAS)
    spec = {**config.sheet_test, "points": len(FLOWS), **sheet_test}
    config = replace(config, sheet_test=spec)
    workbook = tmp_path / "partner" / spec["delivery"] / spec["workbook"]
    workbook.parent.mkdir(parents=True)
    workbook.write_bytes(b"synthetic")
    monkeypatch.setattr(run, "LOCAL", tmp_path)
    monkeypatch.setattr(run, "load_sweep_sheet", lambda _path: _lab_sheet())
    return config


def test_sheet_set_is_expressed_as_design_rows_inside_the_box(run, tmp_path, monkeypatch) -> None:
    config = _context_parts(run, tmp_path, monkeypatch)
    box = resolve_box([ENTRY], [LAB], VARIED_FRESH_GAS)
    sheet = run.load_sheet_set(config, (LAB,), box)
    assert sheet.rows.shape == (len(FLOWS), 7) and bool(box.contains(sheet.rows).all())
    assert sheet.rows[:, 4] == pytest.approx(FLOWS)
    assert sheet.rows[:, 5] == pytest.approx(10.0)
    assert 0.2 < sheet.rows[0, 6] < 0.95
    (line,) = apply_point([LAB], sheet.rows[1], VARIED_FRESH_GAS)
    assert line.gas_humidity_g_per_kg == pytest.approx(5.0, rel=1e-12)
    assert line.dryer_inlet_celsius == 280.0
    assert set(sheet.reported) == set(RESULT_KEYS) and len(sheet.reported["fuel_kg_per_h"]) == 3
    assert sheet.gas_flow_reading == "dry" and len(sheet.workbook_sha256) == 64
    # a point outside the box, a wrong count or an input the surrogate cannot express: abort
    narrow = resolve_box([{**ENTRY, "medium_kg_per_h": [10, 60]}], [LAB], VARIED_FRESH_GAS)
    with pytest.raises(SystemExit, match="outside the box"):
        run.load_sheet_set(config, (LAB,), narrow)
    with pytest.raises(SystemExit, match="the YAML says"):
        run.load_sheet_set(
            replace(config, sheet_test={**config.sheet_test, "points": 11}), (LAB,), box
        )
    with pytest.raises(SystemExit, match="spray_celsius"):
        run.load_sheet_set(config, (replace(LAB, spray_celsius=25.0),), box)


def test_fit_cell_scores_the_sheet_without_fitting_on_it(run) -> None:
    box = resolve_box([ENTRY], [LAB], VARIED_FRESH_GAS)
    rng = np.random.default_rng(3)
    names = study.target_names(1)

    def targets(x: np.ndarray) -> np.ndarray:
        base = 100.0 + x @ np.arange(1.0, 8.0) + 2.0 * x[:, 0] * x[:, 5]
        return np.column_stack([base + k for k in range(len(names))])

    x_pool, x_test, x_sheet = (rng.uniform(-1, 1, size=(n, box.dimension)) for n in (128, 32, 3))
    payload = {
        "x_pool": x_pool,
        "y_pool": targets(x_pool),
        "x_test": x_test,
        "y_test": targets(x_test),
        "admissible": np.ones(32, dtype=bool),
        "available": np.arange(128),
        "names": names,
        "x_sheet": x_sheet,
        "y_sheet": targets(x_sheet),
        "reported": None,
    }
    task = {"id": "quadratic-n64-seed1", "model": "quadratic", "n": 64, "seed": 1, "settings": None}
    run._WORKER.clear()
    run._WORKER.update(payload)
    with_sheet = run._fit_cell(task)
    assert with_sheet["ok"] and with_sheet["metrics_sheet"]["exhaust_max_k"] < 1e-9
    # another truth at the points of the sheet changes their score and nothing else
    run._WORKER.update(y_sheet=targets(x_sheet) + 1.0)
    moved = run._fit_cell(task)
    assert moved["metrics_sheet"]["exhaust_max_k"] == pytest.approx(1.0)
    assert moved["metrics"] == with_sheet["metrics"]
    run._WORKER.update(x_sheet=None)
    assert "metrics_sheet" not in run._fit_cell(task)
    run._WORKER.clear()


def test_control_summary_is_tabulated_over_the_fresh_gas_temperature(run) -> None:
    def point(label: str, celsius: float, shift_k: float, preheated: float) -> dict[str, Any]:
        reference = {
            "exhaust_celsius": 50.0,
            "preheated_celsius": preheated,
            "fuel_kg_per_h": 2.0,
            "saturation_ratio": 0.5,
        }
        dyssol = {**reference, "exhaust_celsius": 50.0 + shift_k, "saturation_ratio": 0.6}
        return {
            "set": label,
            "x": [0, 0, 0, 0, 0, celsius, 0.5],
            "reference": [reference],
            "dyssol": [dyssol],
        }

    points = [
        point("train", -8.0, 1e-13, 5.0),
        point("train", 12.0, 4e-13, 30.0),
        point("test", 12.5, 2e-13, 16.0),
        point("sheet", 10.0, 3e-13, 10.0),
        {"set": "train", "x": [0, 0, 0, 0, 0, 0.0, 0.5], "reference": [{}]},  # no Dyssol result
    ]
    plain = run.control_summary(points)
    assert plain == {
        "points_compared": 4,
        "max_temperature_k": pytest.approx(4e-13),
        "max_relative_percent": 0.0,
        "max_by_quantity": {
            "exhaust_celsius": pytest.approx(4e-13),
            "preheated_celsius": 0.0,
            "fuel_kg_per_h": 0.0,
        },
    }
    table = {"column": 5, "edges": [-10.0, 0.0, 10.0, 20.0], "vapour_lower_k": 290.0}
    full = run.control_summary(points, table)
    assert {k: full[k] for k in plain} == plain
    assert [r["points"] for r in full["by_fresh_gas_celsius"]] == [1, 0, 3]
    assert full["by_fresh_gas_celsius"][2]["max_temperature_k"] == pytest.approx(4e-13)
    assert full["by_set"]["sheet"] == {
        "points": 1,
        "max_temperature_k": pytest.approx(3e-13),
        "max_relative_percent": 0.0,
    }
    cold = full["heater_inlet_below_vapour_correlation"]
    assert cold["points"] == 3 and cold["max_temperature_k"] == pytest.approx(3e-13)


# ----------------------------------------------------------------------------- analysis
def test_analysis_helpers_summarise_the_sheet_per_model_and_seed() -> None:
    analyze = _script("analyze_dyssol_freshgas.py")
    config = study.load_config(FRESHGAS)
    count = {"n": 4, "within_half": 1, "within_one": 2, "max_abs": 0.3, "min_signed": -0.3}
    cells = [
        {
            "model": model,
            "n": n,
            "seed": seed,
            "metrics": {"exhaust_p95_k": 0.5, "fuel_p95_percent": 0.5},
            "metrics_sheet": {
                "exhaust_abs_k": [0.1, 0.2 * seed],
                "fuel_rel_percent": [0.3, 0.1],
                "exhaust_max_k": 0.2 * seed,
                "fuel_max_percent": 0.3,
                "against_sheet": {
                    "per_quantity": {"fuel_kg_per_h": count},
                    "total": {"n": 4, "within_half": 1, "within_one": 2},
                },
            },
        }
        for model in study.MODELS
        for n in config.budgets
        for seed in config.seeds
    ]
    table = analyze.sheet_table(cells, config)
    entry = table["mlp"]["256"]
    assert entry["exhaust_max_k"] == {
        "median": pytest.approx(0.6),
        "min": pytest.approx(0.2),
        "max": pytest.approx(1.0),
    }
    assert entry["seeds_meeting_both"] == 5
    per_seed = analyze.against_sheet(cells, "mlp", 256)
    assert [row["seed"] for row in per_seed] == list(config.seeds)
    assert per_seed[0]["per_quantity"]["fuel_kg_per_h"] == {
        "n": 4,
        "within_half": 1,
        "within_one": 2,
        "max_abs": 0.3,
    }
    assert analyze.against_sheet(cells, None, None) is None
    assert analyze.with_metrics(cells, "metrics_admissible") is None
    renamed = analyze.with_metrics(cells, "metrics_sheet")
    assert renamed is not None and renamed[0]["metrics"] is cells[0]["metrics_sheet"]


def test_committed_analysis_follows_from_the_committed_training_artifact() -> None:
    runs = ROOT / "artifacts" / "runs"
    training_path = runs / "dyssol_freshgas_v1_training.json.gz"
    analysis_path = runs / "dyssol_freshgas_v1_analysis.json"
    if not (training_path.exists() and analysis_path.exists()):
        pytest.skip("artifacts of the fresh-gas study not written yet")
    config = study.load_config(FRESHGAS)
    with gzip.open(training_path, "rt", encoding="utf-8") as fh:
        training = json.load(fh)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert training["header"]["config_fingerprint"] == config.fingerprint
    assert training["mlp_settings"]["hidden"] == list(config.mlp_settings("one_line").hidden)
    assert training["test_set"]["name"] == config.test_set_name
    assert config.sheet_test is not None and config.prediction is not None
    assert training["sheet_test"]["name"] == config.sheet_test["name"]
    assert training["sheet_test"]["points_inside_box"] == config.sheet_test["points"]
    control = training["data"]["summary"]["control"]
    points = sum(s["points"] for s in training["data"]["summary"]["by_set"].values())
    assert control["points_compared"] == points == config.pool_size + config.test_size + 11
    assert control["max_temperature_k"] <= config.control_temperature_k
    assert control["max_relative_percent"] <= config.control_relative_percent
    assert sum(row["points"] for row in control["by_fresh_gas_celsius"]) == points
    cells = training["cells"]
    result = study.reading(cells, config)
    stored = analysis["reading"]
    assert (result["verdict"], result["data_need_runs"], result["reached_by"]) == (
        stored["verdict"],
        stored["data_need_runs"],
        stored["reached_by"],
    )
    assert result["n_star"] == stored["n_star"]
    assert study.prediction_check(result, config.prediction) == analysis["prediction"]
    again = study.sheet_reading(cells, config, result["reached_by"], result["data_need_runs"])
    assert again == analysis["reading_sheet"]
    # no input value and no value of the sheet: the cells hold errors and counts only
    assert all("x" not in cell and "reported" not in cell for cell in cells)


# ----------------------------------------------------------------------------- add-on: cause
def test_the_heat_recovery_of_the_units_is_a_smooth_saturation_without_a_kink() -> None:
    from twinlink_ml.dyssol_steady.reference import heat_recovery

    hot, cold = (0.030, 0.004, 360.0), (0.030, 0.0002, 283.0)
    a = hot[0] * PROPS.cp_gas + hot[1] * PROPS.cp_vapour
    c = cold[0] * PROPS.cp_gas + cold[1] * PROPS.cp_vapour
    flows = np.linspace(1e-4, 0.05, 400)  # medium from far below to far above both gas streams
    heat = np.asarray([heat_recovery(PROPS, hot, cold, m)[2] for m in flows])
    b = flows * PROPS.cp_liquid
    assert b[0] < 0.05 * min(a, c) and b[-1] > 5 * max(a, c)
    assert heat == pytest.approx((hot[2] - cold[2]) / (1 / a + 1 / b + 1 / c), rel=1e-12)
    # an ideal counter-current exchanger would give min(...) x dT and a kink at ratio 1
    kinked = np.minimum(b, min(a, c)) * (hot[2] - cold[2])
    assert np.all(heat < kinked)
    slope = np.diff(heat) / np.diff(flows)
    assert np.all(np.diff(slope) < 0)  # strictly concave everywhere: no kink
    assert np.all(slope > 0)


def test_capacity_ratio_groups_and_variant_boxes() -> None:
    from twinlink_ml.dyssol_steady.cause import (
        MEDIUM,
        RATIO_EDGES,
        capacity_ratio,
        ratio_groups,
        with_interval_of,
    )

    (out,) = evaluate(PROPS, [LAB], (0,))
    gas = LAB.dry_gas_kg_per_h / 3600.0
    hot = gas * (PROPS.cp_gas + out.dryer_outlet_humidity_g_per_kg / 1000.0 * PROPS.cp_vapour)
    cold = gas * (PROPS.cp_gas + LAB.gas_humidity_g_per_kg / 1000.0 * PROPS.cp_vapour)
    assert cold < hot  # the exhaust carries the evaporated water
    expected = LAB.medium_kg_per_h / 3600.0 * PROPS.cp_liquid / cold
    assert capacity_ratio(PROPS, LAB, out) == pytest.approx(expected, rel=1e-14)
    assert capacity_ratio(PROPS, replace(LAB, medium_kg_per_h=0.0), out) == 0.0
    groups = ratio_groups(np.array([0.0, 0.2, 0.25, 0.9, 1.0, 1.9, 2.0, 50.0]))
    assert groups["medium_below_gas"].sum() == 4 and groups["medium_above_gas"].sum() == 4
    bins = [k for k in groups if k.startswith("ratio_")]
    assert len(bins) == len(RATIO_EDGES) - 1
    assert sum(groups[k].sum() for k in bins) == 8  # every point in exactly one bin
    assert groups["ratio_0_0.25"].sum() == 2 and groups["ratio_2_inf"].sum() == 2
    wide = resolve_box([ENTRY], [LAB], VARIED_FRESH_GAS)
    narrow = resolve_box(
        [{**{k: v for k, v in ENTRY.items() if k in VARIED}, MEDIUM: {"relative": 0.25}}], [LAB]
    )
    v_a = with_interval_of(wide, narrow, MEDIUM)
    assert v_a.names == wide.names and v_a.dimension == 7
    assert (v_a.lower[4], v_a.upper[4]) == pytest.approx((30.0, 50.0))
    assert v_a.lower[:4] == wide.lower[:4] and v_a.upper[5:] == wide.upper[5:]
    v_b = with_interval_of(narrow, wide, MEDIUM)
    assert v_b.names == narrow.names and (v_b.lower[4], v_b.upper[4]) == (0.0, 60.0)
    assert len({wide.sha256(), narrow.sha256(), v_a.sha256(), v_b.sha256()}) == 4
    with pytest.raises(ValueError, match="both boxes"):
        with_interval_of(narrow, wide, "gas_celsius")


def test_cause_reading_separates_the_medium_range_from_the_additional_inputs() -> None:
    from twinlink_ml.dyssol_steady.cause import cause_reading

    def verdict(a: int | None, b: int | None, control: bool = True) -> str:
        return cause_reading({"V-a": a, "V-b": b, "V-c": None}, control)["verdict"]

    assert verdict(64, None) == "K0"
    assert verdict(None, 32) == "K1"
    assert verdict(64, 32) == "K2" and verdict(None, None) == "K2"
    assert verdict(64, None, control=False).startswith("not read")


def test_learning_table_is_the_rule_of_the_reading_for_one_model() -> None:
    config = study.load_config(FRESHGAS)
    cells = [
        {
            "model": model,
            "n": n,
            "seed": seed,
            "metrics": {
                "exhaust_p95_k": 0.5 if (model == "mlp" and n >= 128) else 2.0,
                "fuel_p95_percent": 0.5,
            },
        }
        for model in study.MODELS
        for n in config.budgets
        for seed in config.seeds
    ]
    whole = study.reading(cells, config)
    for model in study.MODELS:
        rows, n_star = study.learning_table(cells, config, model)
        assert rows == whole["table"][model] and n_star == whole["n_star"][model]
    assert study.learning_table(cells, config, "mlp")[1] == 128
    only_two = [c for c in cells if c["model"] != "linear"]
    assert study.learning_table(only_two, config, "quadratic")[1] is None
    with pytest.raises(ValueError, match="one per seed"):
        study.learning_table(only_two, config, "linear")


def test_cause_driver_tabulates_groups_and_checks_the_control(run, monkeypatch, tmp_path) -> None:
    cause = _script("run_dyssol_freshgas_cause.py")
    config = study.load_config(FRESHGAS)

    def metrics(value: float) -> dict[str, float]:
        return {
            "exhaust_p95_k": value,
            "fuel_p95_percent": 2 * value,
            "max_abs/line1/exhaust_celsius": 3 * value,
            "max_rel/total_fuel": 4 * value,
        }

    cells = [
        {
            "model": model,
            "n": n,
            "seed": seed,
            "metrics": metrics(0.1 * seed),
            "metrics_by_group": {
                "medium_below_gas": {"points": 60, **metrics(1.0 * seed)},
                "medium_above_gas": {"points": 196, **metrics(0.01 * seed)},
            },
        }
        for model in cause.MODELS
        for n in config.budgets
        for seed in config.seeds
    ]
    table = cause.group_table(cells, config)
    below = table["quadratic"]["medium_below_gas"]["512"]
    assert below == {
        "points": 60,
        "exhaust_p95_k": pytest.approx(3.0),
        "fuel_p95_percent": pytest.approx(6.0),
        "max_abs_exhaust_k": pytest.approx(9.0),
        "max_rel_fuel_percent": pytest.approx(12.0),
    }
    assert table["mlp"]["medium_above_gas"]["8"]["points"] == 196
    assert cause.shares(np.array([0.1, 0.6, 1.5, 3.0]))["medium_below_gas"] == 2
    assert set(cause.VARIANTS) == {"V-a", "V-b", "V-c"} and cause.MODELS == ("quadratic", "mlp")
    # the control compares with the committed artifacts of the main run
    with gzip.open(cause.RUNS_DIR / f"{cause.MAIN_TRAINING}.json.gz", "rt") as fh:
        training = json.load(fh)
    main_cells = [c for c in training["cells"] if c["model"] in cause.MODELS]

    class _Box:
        def __init__(self, digest: str) -> None:
            self.digest = digest

        def sha256(self) -> str:
            return self.digest

    same = cause.control_against_main(
        main_cells, {"quadratic": None, "mlp": 256}, _Box(training["header"]["box_sha256"])
    )
    assert same["reproduces"] is True
    assert same["largest_difference_of_reading_numbers"]["mlp"]["exhaust_p95_k"] == 0.0
    assert not cause.control_against_main(
        main_cells, {"quadratic": 64, "mlp": 256}, _Box(training["header"]["box_sha256"])
    )["reproduces"]
    assert not cause.control_against_main(
        main_cells, {"quadratic": None, "mlp": 256}, _Box("other")
    )["reproduces"]


def test_fit_cell_scores_named_groups_of_the_test_set(run) -> None:
    rng = np.random.default_rng(4)
    names = study.target_names(1)
    x_pool, x_test = rng.uniform(-1, 1, size=(64, 3)), rng.uniform(-1, 1, size=(20, 3))

    def targets(x: np.ndarray) -> np.ndarray:
        base = 50.0 + x @ np.array([1.0, 2.0, 3.0]) + np.abs(x[:, 0])  # a kink in input 0
        return np.column_stack([base + k for k in range(len(names))])

    left = x_test[:, 0] < 0
    run._WORKER.clear()
    run._WORKER.update(
        x_pool=x_pool,
        y_pool=targets(x_pool),
        x_test=x_test,
        y_test=targets(x_test),
        admissible=np.ones(20, dtype=bool),
        available=np.arange(64),
        names=names,
        groups={"left": left, "right": ~left, "empty": np.zeros(20, dtype=bool)},
    )
    cell = run._fit_cell(
        {"id": "linear-n64-seed1", "model": "linear", "n": 64, "seed": 1, "settings": None}
    )
    run._WORKER.clear()
    groups = cell["metrics_by_group"]
    assert set(groups) == {"left", "right"}
    assert groups["left"]["points"] + groups["right"]["points"] == 20
    largest = max(g["max_abs/line1/exhaust_celsius"] for g in groups.values())
    assert largest == pytest.approx(cell["metrics"]["max_abs/line1/exhaust_celsius"])
