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
