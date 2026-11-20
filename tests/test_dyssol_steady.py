"""Stationary Dyssol flowsheets as a data source: rebuild, topology, script, design, models.

Tests without local files use laboratory-scale inputs already committed in docs/11 and the
property values of the materials database shipped with Dyssol. Tests that need the confidential
delivery (``<local>/partner/dyssol_2026-11-19``), the Dyssol installation or the compiled units
are skipped when these are missing.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.dyssol_steady import study
from twinlink_ml.dyssol_steady.design import (
    VARIED,
    Box,
    apply_point,
    resolve_box,
    sobol_points,
    training_subset,
)
from twinlink_ml.dyssol_steady.flowsheet import (
    DyssolFlowsheet,
    DyssolInstallation,
    ReferenceFlowsheet,
    discover_topology,
    extract_outputs,
    nominal_inputs,
)
from twinlink_ml.dyssol_steady.materials import (
    DEFAULT_HEAT_CAPACITY,
    DEFAULT_MOLAR_MASS,
    DyssolProperties,
    compound_names,
    patch_materials_database,
    read_properties,
)
from twinlink_ml.dyssol_steady.models import MlpRegressor, MlpSettings, PolynomialRegression
from twinlink_ml.dyssol_steady.reference import (
    EnthalpyTable,
    LineInputs,
    dryer,
    evaluate,
    heat_recovery,
    heater,
)
from twinlink_ml.partner_v4_1.comparison import DyssolFile, DyssolStream, DyssolUnit

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", ROOT / "local"))
DELIVERY = LOCAL / "partner" / "dyssol_2026-11-19"
DYSSOL_HOME = Path(os.environ.get("DYSSOL_HOME", "C:/Program Files/Dyssol"))

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
LAB2 = replace(LAB, dry_gas_kg_per_h=100.0, dryer_inlet_celsius=150.0, spray_kg_per_h=3.0)

DMDB = """DyssolMaterialsDatabase 3

21 KEYAIR \t$ Compound's unique key
22 Air \t$ Compound's name
26 121 0.028951 \t$ Molar mass [kg/mol]
27 207 2 10 10000 1000 1e+09 1006  \t$ Heat capacity; Constant
27 208 3 200 2773 1000 1e+09 1005 0 0  \t$ Enthalpy; Linear

21 KEYLIQ \t$ Compound's unique key
22 H2O \t$ Compound's name
26 115 40660.5 \t$ Heat of vaporization  [J/mol]
27 207 2 10 10000 1000 1e+09 4185  \t$ Heat capacity; Constant

21 KEYVAP \t$ Compound's unique key
22 H2O_gas \t$ Compound's name
27 207 2 10 10000 1000 1e+09 1906  \t$ Heat capacity; Constant
27 208 8 290 1700 1000 1e+09 @LOW@
27 208 8 1700 6000 1000 1e+09 @HIGH@

21 KEYSOL \t$ Compound's unique key
22 Al2O3 \t$ Compound's name
26 121 0.10196 \t$ Molar mass [kg/mol]
27 208 3 10 10000 1000 1e+09 900 0 0  \t$ Enthalpy; Linear
"""
DMDB = DMDB.replace(
    "@LOW@", "1670.36 0.379262 0.000377093 -1.40685e-07 4.5594e+06 -1.3926e+07 -1.34234e+07"
).replace(
    "@HIGH@", "2329.37 0.478596 -8.32503e-05 5.44643e-09 -6.19342e+08 -1.51083e+07 -1.34234e+07"
)


# ----------------------------------------------------------------------------- materials
def test_patch_adds_only_the_boiling_point_and_properties_follow_the_database(tmp_path) -> None:
    source, target = tmp_path / "in.dmdb", tmp_path / "out" / "patched.dmdb"
    source.write_text(DMDB, encoding="utf-8")
    digests = patch_materials_database(source, target)
    added = set(target.read_text(encoding="utf-8").splitlines()) - set(DMDB.splitlines())
    assert len(added) == 1 and next(iter(added)).startswith("26 122 373.15")
    assert digests["source_sha256"] != digests["patched_sha256"]
    assert compound_names(target) == {
        "KEYAIR": "Air",
        "KEYLIQ": "H2O",
        "KEYVAP": "H2O_gas",
        "KEYSOL": "Al2O3",
    }
    assert read_properties(target) == PROPS
    with pytest.raises(ValueError, match="already has a boiling point"):
        patch_materials_database(target, tmp_path / "twice.dmdb")
    with pytest.raises(ValueError, match="boiling point"):
        read_properties(source)


# ----------------------------------------------------------------------------- units
def test_dryer_closes_mass_and_energy_balance() -> None:
    gas, vapour, solid, liquid = 0.03, 0.0002, 0.0013, 0.002
    t_out, vapour_out, liquid_out = dryer(PROPS, gas, vapour, 573.15, solid, liquid, 293.15, 0.002)
    assert vapour_out - vapour == pytest.approx(liquid - liquid_out)
    assert liquid_out == pytest.approx(solid * 0.002)
    t_b = PROPS.boiling_point_k
    h_in = (gas * 1006 + vapour * 1906) * (573.15 - t_b) + (solid * 1000 + liquid * 4185) * (
        293.15 - t_b
    )
    h_out = (gas * 1006 + vapour_out * 1906 + solid * 1000 + liquid_out * 4185) * (t_out - t_b) + (
        vapour_out - vapour
    ) * PROPS.heat_of_vaporization_j_per_kg
    assert h_out == pytest.approx(h_in, rel=1e-12)
    # a feed drier than the equilibrium moisture is not dried (and not wetted)
    _, same_vapour, same_liquid = dryer(PROPS, gas, vapour, 573.15, solid, 1e-7, 293.15, 0.002)
    assert same_vapour == vapour and same_liquid == 1e-7


def test_heat_recovery_satisfies_both_equations() -> None:
    hot, cold, medium = (0.03, 0.002, 400.0), (0.04, 0.0003, 290.0), 0.011
    t1, t2, heat = heat_recovery(PROPS, hot, cold, medium)
    a = hot[0] * 1006 + hot[1] * 1906
    c = cold[0] * 1006 + cold[1] * 1906
    b = medium * 4185
    assert a * (hot[2] - t2) == pytest.approx(b * (t2 - t1))
    assert c * (t1 - cold[2]) == pytest.approx(b * (t2 - t1))
    assert heat == pytest.approx(b * (t2 - t1))
    assert cold[2] < t1 < t2 < hot[2]


def test_enthalpy_table_interpolates_and_clamps_below_the_correlation_range() -> None:
    table = EnthalpyTable(PROPS)
    grid = np.linspace(173.0, 1273.0, 101)
    node, mid = float(grid[20]), float(0.5 * (grid[20] + grid[21]))
    assert table.enthalpy_j_per_kg(mid, 0.1) == pytest.approx(
        0.5 * (table.enthalpy_j_per_kg(node, 0.1) + table.enthalpy_j_per_kg(float(grid[21]), 0.1))
    )
    # dry gas: h = 1005 T exactly (linear correlation, linear interpolation)
    assert table.enthalpy_j_per_kg(333.3, 0.0) == pytest.approx(1005.0 * 333.3)
    # vapour below 290 K: both grid nodes are clamped to 290 K, the enthalpy is constant there
    assert table.enthalpy_j_per_kg(float(grid[5]), 1.0) == table.enthalpy_j_per_kg(
        float(grid[9]), 1.0
    )
    duty, fuel = heater(table, 0.03, 0.0002, 320.0, 420.0, 0.8, 45000.0)
    assert duty > 0 and fuel == pytest.approx(duty * 3600 / (0.8 * 45000.0))


def test_flowsheet_one_line_and_two_lines() -> None:
    (one,) = evaluate(PROPS, [LAB], (0,))
    assert LAB.gas_celsius < one.preheated_celsius < one.exhaust_celsius < one.dryer_outlet_celsius
    assert one.dryer_outlet_celsius < LAB.dryer_inlet_celsius
    assert one.product_kg_per_h == pytest.approx(12 * 0.4 * 1.002)
    assert not one.oversaturated
    # the heat taken from the exhaust is the heat the fresh gas gains
    gas = LAB.dry_gas_kg_per_h / 3600 * (1006 + 0.006 * 1906)
    assert one.heat_recovered_kw * 1000 == pytest.approx(
        gas * (one.preheated_celsius - LAB.gas_celsius)
    )
    crossed = evaluate(PROPS, [LAB, LAB2], (1, 0))
    own = evaluate(PROPS, [LAB, LAB2], (0, 1))
    # the dryers do not depend on the pairing, the preheating does
    assert crossed[0].dryer_outlet_celsius == own[0].dryer_outlet_celsius
    assert crossed[0].preheated_celsius != own[0].preheated_celsius
    with pytest.raises(ValueError, match="permutation"):
        evaluate(PROPS, [LAB, LAB2], (0, 0))
    wet = replace(LAB, dryer_inlet_celsius=60.0, spray_kg_per_h=20.0)
    assert evaluate(PROPS, [wet], (0,))[0].oversaturated


# ----------------------------------------------------------------------------- topology
NAMES = {"KA": "Air", "KL": "H2O", "KV": "H2O_gas", "KS": "Al2O3"}


def _fake_file(lines: list[LineInputs], preheated_by: tuple[int, ...]) -> DyssolFile:
    """A ``DyssolFile`` as the reader returns it, filled with the rebuild's results."""
    outputs = evaluate(PROPS, lines, preheated_by)
    streams: dict[str, DyssolStream] = {}
    units: dict[str, DyssolUnit] = {}

    def stream(
        name: str, mass: float, celsius: float, vapour: float = 0.0, solid: float = 0.0
    ) -> str:
        gas = 0.0 if solid else 1.0
        streams[name] = DyssolStream(
            name,
            f"key-{name}",
            3600.0,
            mass,
            celsius,
            {"Solid": solid, "Liquid": (1.0 - solid) if solid else 0.0, "Gas": gas},
            {
                "Solid": (0.0, 0.0, 0.0, 1.0),
                "Liquid": (0.0, 1.0, 0.0, 0.0),
                "Gas": (1.0 - vapour, 0.0, vapour, 0.0),
            },
        )
        return f"key-{name}"

    for i, (ln, out) in enumerate(zip(lines, outputs, strict=True), start=1):
        y = ln.gas_humidity_g_per_kg / 1000
        total = ln.dry_gas_kg_per_h * (1 + y) / 3600
        y_out = out.dryer_outlet_humidity_g_per_kg / 1000
        k_gas = stream(f"Gas in {i}", total, ln.gas_celsius, y / (1 + y))
        k_pre = stream(f"Heater in {i}", total, out.preheated_celsius, y / (1 + y))
        k_hot = stream(f"Dryer in {i}", total, ln.dryer_inlet_celsius, y / (1 + y))
        k_feed = stream(
            f"Feed {i}", ln.spray_kg_per_h / 3600, ln.spray_celsius, solid=ln.solid_fraction
        )
        wet = ln.dry_gas_kg_per_h * (1 + y_out) / 3600
        k_out = stream(f"Dryer out {i}", wet, out.dryer_outlet_celsius, y_out / (1 + y_out))
        stream(f"Gas out {i}", wet, out.exhaust_celsius, y_out / (1 + y_out))
        k_prod = stream(
            f"Product {i}", out.product_kg_per_h / 3600, out.dryer_outlet_celsius, solid=0.9
        )
        units[f"Gas inlet {i}"] = DyssolUnit(f"Gas inlet {i}", {"InletMaterial": k_gas}, {}, {})
        units[f"Suspension {i}"] = DyssolUnit(f"Suspension {i}", {"InletMaterial": k_feed}, {}, {})
        units[f"Heater {i}"] = DyssolUnit(
            f"Heater {i}",
            {"InPort": k_pre, "OutPort": k_hot},
            {
                "Efficiency": (ln.heater_efficiency_percent,),
                "Calorific Value": (ln.calorific_value_kj_per_kg,),
                "Outlet Temperature": (ln.dryer_inlet_celsius,),
            },
            {"Heat duty [kW]": out.heater_duty_kw, "Fuel consumption [kg/h]": out.fuel_kg_per_h},
        )
        units[f"Dryer {i}"] = DyssolUnit(
            f"Dryer {i}",
            {"GasIn": k_hot, "Feed": k_feed, "GasOut": k_out, "Product": k_prod},
            {"Equilibrium moisture content": (ln.equilibrium_moisture_g_per_kg,)},
            {},
        )
    for hot, out in enumerate(outputs, start=1):
        cold = preheated_by.index(hot - 1) + 1
        units[f"Recovery {hot}"] = DyssolUnit(
            f"Recovery {hot}",
            {
                "Heat recovery inlet": f"key-Dryer out {hot}",
                "Preheater inlet": f"key-Gas in {cold}",
                "Heat recovery outlet": f"key-Gas out {hot}",
                "Preheater outlet": f"key-Heater in {cold}",
            },
            {"Medium mass flow": (lines[hot - 1].medium_kg_per_h,)},
            {"Heat recovered [kW]": out.heat_recovered_kw},
        )
    return DyssolFile(("KA", "KL", "KV", "KS"), streams, units)


@pytest.mark.parametrize(
    ("lines", "pairing"), [([LAB], (0,)), ([LAB, LAB2], (1, 0)), ([LAB, LAB2], (0, 1))]
)
def test_topology_inputs_and_outputs_are_read_from_the_file(lines, pairing) -> None:
    file = _fake_file(lines, pairing)
    topology = discover_topology(file)
    assert topology.preheated_by == pairing
    assert [ln.dryer for ln in topology.lines] == [f"Dryer {i}" for i in range(1, len(lines) + 1)]
    for got, want in zip(nominal_inputs(file, topology, NAMES), lines, strict=True):
        assert asdict(got) == pytest.approx(asdict(want))
    expected = evaluate(PROPS, lines, pairing)
    for got, want in zip(extract_outputs(file, topology, NAMES), expected, strict=True):
        assert asdict(got) == pytest.approx(asdict(want))


def test_topology_rejects_a_flowsheet_outside_the_family() -> None:
    file = _fake_file([LAB, LAB2], (1, 0))
    units = dict(file.units)
    del units["Recovery 2"]
    with pytest.raises(ValueError, match="exactly one"):
        discover_topology(DyssolFile(file.compounds, file.streams, units))


def test_script_sets_feeds_and_parameters_by_the_names_of_the_file(tmp_path, monkeypatch) -> None:
    file = _fake_file([LAB, LAB2], (1, 0))
    monkeypatch.setattr(
        "twinlink_ml.dyssol_steady.flowsheet.load_dyssol_steady_state", lambda _path: file
    )
    installation = DyssolInstallation(
        tmp_path / "DyssolC.exe", tmp_path / "units", tmp_path / "m.dmdb"
    )
    flowsheet = DyssolFlowsheet(tmp_path / "source.dflw", installation, tmp_path / "work", NAMES)
    assert flowsheet.nominal[0] == pytest.approx(LAB) or asdict(
        flowsheet.nominal[0]
    ) == pytest.approx(asdict(LAB))
    point = (
        replace(LAB, dry_gas_kg_per_h=90.0, solid_fraction=0.5),
        replace(LAB2, medium_kg_per_h=33.0),
    )
    text = flowsheet.script(point, tmp_path / "work" / "r.dflw")
    rows = text.splitlines()
    assert rows[0] == "JOB" and f'MODELS_PATH "{tmp_path / "units"}"' in rows
    mass = 90.0 * 1.006 / 3600
    assert (
        f'HOLDUP_OVERALL "Gas inlet 1" "InputMaterial" MASS_FLOW 0 {mass:.17g} 3600 {mass:.17g}'
        in rows
    )
    assert 'HOLDUP_PHASES "Suspension 1" "InputMaterial" 0 0.5 0.5 0 3600 0.5 0.5 0' in rows
    assert 'HOLDUP_COMPOUNDS "Suspension 2" "InputMaterial" SOLID 0 0 0 0 1 3600 0 0 0 1' in rows
    assert 'UNIT_PARAMETER "Recovery 2" "Medium mass flow" 33' in rows
    assert 'UNIT_PARAMETER "Heater 2" "Outlet Temperature" 150' in rows
    gas = next(r for r in rows if r.startswith('HOLDUP_COMPOUNDS "Gas inlet 1"'))
    shares = [float(v) for v in gas.split("GAS ")[1].split()[1:5]]
    assert shares == pytest.approx([1 / 1.006, 0.0, 0.006 / 1.006, 0.0])
    with pytest.raises(ValueError, match="one LineInputs per line"):
        flowsheet.script(point[:1], tmp_path / "r.dflw")
    assert ReferenceFlowsheet(PROPS, (1, 0)).evaluate(point).outputs == evaluate(
        PROPS, point, (1, 0)
    )


# ----------------------------------------------------------------------------- design
def test_box_is_resolved_from_absolute_and_relative_entries() -> None:
    entry = {
        "dry_gas_kg_per_h": [100, 140],
        "dryer_inlet_celsius": [200, 300],
        "spray_kg_per_h": {"relative": 0.25},
        "solid_fraction": {"relative": 0.25},
        "medium_kg_per_h": {"relative": 0.5},
    }
    box = resolve_box([entry], [LAB])
    assert box.names == tuple(f"line1/{v}" for v in VARIED)
    assert box.lower == pytest.approx((100, 200, 9, 0.3, 20))
    assert box.upper == pytest.approx((140, 300, 15, 0.5, 60))
    assert box.sha256() == resolve_box([entry], [LAB]).sha256()
    assert box.sha256() != resolve_box([{**entry, "dry_gas_kg_per_h": [100, 141]}], [LAB]).sha256()
    with pytest.raises(ValueError, match="exactly"):
        resolve_box([{"dry_gas_kg_per_h": [1, 2]}], [LAB])
    with pytest.raises(ValueError, match="one entry per line"):
        resolve_box([entry], [LAB, LAB2])
    assert resolve_box([entry, entry], [LAB, LAB2]).dimension == 10


def test_design_points_subsets_and_application() -> None:
    box = Box(tuple(f"line1/{v}" for v in VARIED), (100, 200, 9, 0.3, 20), (140, 300, 15, 0.5, 60))
    x = sobol_points(box, 64, seed=3)
    assert x.shape == (64, 5) and bool(box.contains(x).all())
    assert np.array_equal(x, sobol_points(box, 64, seed=3))
    assert not np.array_equal(x, sobol_points(box, 64, seed=4))
    scaled = box.to_unit_symmetric(x)
    assert scaled.min() >= -1 and scaled.max() <= 1
    with pytest.raises(ValueError, match="power of two"):
        sobol_points(box, 48, seed=3)
    small, large = training_subset(64, 8, seed=1), training_subset(64, 32, seed=1)
    assert np.array_equal(small, large[:8]) and len(set(large)) == 32
    assert not np.array_equal(small, training_subset(64, 8, seed=2))
    (line,) = apply_point([LAB], x[0])
    assert [getattr(line, v) for v in VARIED] == pytest.approx(x[0])
    assert line.gas_celsius == LAB.gas_celsius
    with pytest.raises(ValueError, match="number of lines"):
        apply_point([LAB, LAB2], x[0])


# ----------------------------------------------------------------------------- models
def test_polynomial_regression_is_exact_on_its_own_class() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(40, 3))
    linear = np.column_stack([1 + x @ [1.0, -2.0, 0.5], 3 * x[:, 0]])
    quadratic = np.column_stack([linear[:, 0] + 2 * x[:, 0] * x[:, 1] - x[:, 2] ** 2])
    assert PolynomialRegression(1).fit(x, linear).predict(x) == pytest.approx(linear)
    assert PolynomialRegression(2).features(x).shape == (40, 10)
    assert PolynomialRegression(2).fit(x, quadratic).predict(x) == pytest.approx(quadratic)
    assert not np.allclose(PolynomialRegression(1).fit(x, quadratic).predict(x), quadratic)
    # fewer points than coefficients: minimum-norm solution, still interpolates
    assert PolynomialRegression(2).fit(x[:5], quadratic[:5]).predict(x[:5]) == pytest.approx(
        quadratic[:5]
    )


def test_mlp_learns_a_smooth_function_and_is_seeded() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=(128, 2))
    y = np.column_stack([300 + 20 * np.tanh(x[:, 0]) + 5 * x[:, 1], 50 * x[:, 0] * x[:, 1]])
    settings = MlpSettings(hidden=(16, 16), learning_rate=0.01, steps=600, weight_decay=0.0)
    model = MlpRegressor(settings, seed=1).fit(x, y)
    error = np.abs(model.predict(x) - y).mean(axis=0)
    assert error[0] < 1.0 and error[1] < 2.0
    again = MlpRegressor(settings, seed=1).fit(x, y)
    assert np.array_equal(model.predict(x), again.predict(x))
    assert not np.array_equal(model.predict(x), MlpRegressor(settings, seed=2).fit(x, y).predict(x))


# ----------------------------------------------------------------------------- study
def test_cell_metrics_in_physical_units() -> None:
    names = study.target_names(2)
    assert len(names) == 10 and names[0] == "line1/exhaust_celsius"
    y = np.tile(np.arange(10.0) + 10.0, (100, 1))
    pred = y.copy()
    pred[:, 0] += np.linspace(0.0, 1.0, 100)  # exhaust of line 1
    pred[:, 5] -= 2.0  # exhaust of line 2
    pred[:, 4] += 0.14  # fuel of line 1; total fuel is 14 + 19 = 33
    m = study.cell_metrics(y, pred, names)
    assert m["p95_abs/line1/exhaust_celsius"] == pytest.approx(
        np.percentile(np.linspace(0, 1, 100), 95)
    )
    assert m["exhaust_p95_k"] == pytest.approx(2.0)
    assert m["fuel_p95_percent"] == pytest.approx(100 * 0.14 / 33)
    assert m["max_abs/line2/fuel_kg_per_h"] == 0.0


def _cells(
    config: study.StudyConfig, curves: dict[str, dict[int, tuple[float, float]]]
) -> list[dict]:
    return [
        {
            "model": model,
            "n": n,
            "seed": seed,
            "metrics": {
                "exhaust_p95_k": curves[model][n][0] * (1 + 0.01 * seed),
                "fuel_p95_percent": curves[model][n][1] * (1 + 0.02 * seed * (model == "mlp")),
            },
        }
        for model in study.MODELS
        for n in config.budgets
        for seed in config.seeds
    ]


def test_reading_distinguishes_the_three_verdicts_and_needs_every_larger_n() -> None:
    config = study.load_config()
    bad, good = (5.0, 5.0), (0.5, 0.5)
    last = config.budgets[-1]

    def curve(first_good: int | None) -> dict[int, tuple[float, float]]:
        return {
            n: good if first_good is not None and n >= first_good else bad for n in config.budgets
        }

    d1 = study.reading(
        _cells(config, {"linear": curve(None), "quadratic": curve(last), "mlp": curve(64)}), config
    )
    assert (d1["verdict"], d1["data_need_runs"], d1["reached_by"]) == ("D1", 64, "mlp")
    d2 = study.reading(
        _cells(config, {"linear": curve(None), "quadratic": curve(32), "mlp": curve(32)}), config
    )
    assert (d2["verdict"], d2["data_need_runs"], d2["reached_by"]) == ("D2", 32, "quadratic")
    both = study.reading(
        _cells(config, {"linear": curve(16), "quadratic": curve(16), "mlp": curve(None)}), config
    )
    assert (both["verdict"], both["reached_by"]) == ("D2", "linear")
    d3 = study.reading(_cells(config, {m: curve(None) for m in study.MODELS}), config)
    assert (d3["verdict"], d3["data_need_runs"], d3["n_star"]["mlp"]) == ("D3", None, None)
    # meeting the thresholds at one N and missing them at a larger one does not count
    dip = curve(None)
    dip[16] = good
    assert (
        study.reading(
            _cells(config, {"linear": dip, "quadratic": curve(None), "mlp": curve(None)}), config
        )["verdict"]
        == "D3"
    )
    # one threshold alone is not enough
    half = dict.fromkeys(config.budgets, (0.5, 5.0))
    table = study.reading(_cells(config, {"linear": half, "quadratic": half, "mlp": half}), config)
    assert table["verdict"] == "D3" and table["table"]["mlp"][last]["seeds_meeting_both"] == 0
    with pytest.raises(ValueError, match="one per seed"):
        study.reading(_cells(config, {m: curve(None) for m in study.MODELS})[:-1], config)


def test_rank_tests_are_paired_and_corrected_per_family() -> None:
    config = study.load_config()
    flat = dict.fromkeys(config.budgets, (2.0, 2.0))
    better = dict.fromkeys(config.budgets, (1.0, 1.0))
    tests = study.rank_tests(
        _cells(config, {"linear": flat, "quadratic": flat, "mlp": better}), config
    )
    assert len(tests) == 2 * 2 * len(config.budgets)
    exhaust = [t for t in tests if t["family"] == "mlp_vs_linear/exhaust_p95_k"]
    assert all(t["mean_difference"] > 0 and t["resolved_holm"] for t in exhaust)
    assert {t["tests_in_family"] for t in exhaust} == {len(config.budgets)}
    # identical values in both models: zero spread of the differences, nothing is resolved
    same = study.rank_tests(_cells(config, dict.fromkeys(study.MODELS, flat)), config)
    assert all(t["degenerate"] and not t["resolved_holm"] for t in same if "exhaust" in t["family"])


def test_pilot_selection_takes_the_lowest_mean_criterion() -> None:
    config = study.load_config()
    cells = [
        {"setting": k, "metrics": {"exhaust_p95_k": e, "fuel_p95_percent": f}}
        for k, e, f in [
            (0, 2.0, 2.0),
            (0, 4.0, 2.0),
            (1, 1.0, 1.0),
            (1, 2.0, 5.0),
            (2, 2.0, 3.0),
            (2, 2.0, 1.0),
        ]
    ]
    chosen = study.pilot_selection(cells, config)
    assert chosen["criterion"] == pytest.approx({0: 2.5, 1: 2.25, 2: 2.0})
    assert chosen["selected_setting"] == 2


def test_committed_config_is_consistent() -> None:
    config = study.load_config()
    assert config.budgets[-1] <= config.pool_size and len(config.seeds) == 5
    assert set(config.selected) == set(config.cases) == {"one_line", "two_lines"}
    assert config.threshold_exhaust_k == 1.0 and config.threshold_fuel_percent == 1.0
    assert len(config.fingerprint) == 64
    text = (ROOT / "conf" / "dyssol_steady" / "box_local.yaml.example").read_text(encoding="utf-8")
    assert "LOWER" in text and "relative: R" in text  # the template holds placeholders only
    assert json.dumps(config.raw)  # plain data only


# ----------------------------------------------------------------------------- local files
def _delivery(name: str) -> Path:
    path = DELIVERY / name
    if not path.exists() or not (DYSSOL_HOME / "Materials.dmdb").exists():
        pytest.skip("delivery of 2026-11-19 or the Dyssol installation not available")
    pytest.importorskip("h5py")
    return path


@pytest.fixture(scope="module")
def patched(tmp_path_factory) -> Path:
    if not (DYSSOL_HOME / "Materials.dmdb").exists():
        pytest.skip("Dyssol installation not available")
    target = tmp_path_factory.mktemp("materials") / "patched.dmdb"
    patch_materials_database(DYSSOL_HOME / "Materials.dmdb", target)
    return target


def test_installed_database_gives_the_documented_properties(patched) -> None:
    assert read_properties(patched) == PROPS


@pytest.mark.parametrize(("name", "pairing"), [("Case 2.dflw", (1, 0)), ("Case 4.dflw", None)])
def test_rebuild_reproduces_a_stored_flowsheet(name, pairing, patched) -> None:
    from twinlink_ml.partner_v4_1.comparison import load_dyssol_steady_state

    file = load_dyssol_steady_state(_delivery(name))
    names = compound_names(patched)
    if pairing is None:  # two recoveries in series on one stream: outside the family
        with pytest.raises(ValueError):
            discover_topology(file)
        return
    topology = discover_topology(file)
    assert topology.preheated_by == pairing
    nominal = nominal_inputs(file, topology, names)
    rebuilt = evaluate(read_properties(patched), nominal, topology.preheated_by)
    for got, want in zip(rebuilt, extract_outputs(file, topology, names), strict=True):
        assert asdict(got) == pytest.approx(asdict(want), rel=1e-9, abs=1e-9)


def test_dyssol_run_at_the_stored_inputs_reproduces_the_stored_results(patched, tmp_path) -> None:
    source = _delivery("Case 2.dflw")
    units = Path(
        os.environ.get("TWINLINK_DYSSOL_UNITS", "C:/twinlink_runs/dyssol-steady-state/units")
    )
    executable = DYSSOL_HOME / "DyssolC.exe"
    if not (units / "manifest.json").exists() or not executable.exists():
        pytest.skip("compiled units or DyssolC not available")
    from twinlink_ml.partner_v4_1.comparison import load_dyssol_steady_state

    names = compound_names(patched)
    flowsheet = DyssolFlowsheet(
        source, DyssolInstallation(executable, units, patched), tmp_path, names
    )
    run = flowsheet.evaluate(flowsheet.nominal, "nominal")
    stored = extract_outputs(load_dyssol_steady_state(source), flowsheet.topology, names)
    for got, want in zip(run.outputs, stored, strict=True):
        assert asdict(got) == pytest.approx(asdict(want), rel=1e-9, abs=1e-9)
    assert run.wall_seconds > 0 and not (tmp_path / "nominal").exists()
