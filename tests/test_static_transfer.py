"""Static transfer study (docs/NN_static_transfer.md): balance, plants, designs, methods, verdicts.

Synthetic data only. One test compares the YAML with the partner workbook ``TwinLink Excel.xlsx``
(``local/partner/``, confidential, never committed; ``TWINLINK_LOCAL_DIR`` names the directory in a
worktree) and is skipped without it.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.partner_v4_1.dryer_block import (
    BlockInputs,
    BlockProperties,
    dryer_block,
    heater_power_w,
)
from twinlink_ml.static_transfer import study
from twinlink_ml.static_transfer.analysis import Family, MetricTable, compare, smallest_n_from_which
from twinlink_ml.static_transfer.balance import (
    RhDependentMoisture,
    plant_model,
    relative_humidity,
    source_model,
)
from twinlink_ml.static_transfer.config import TARGET_NAMES, load_config
from twinlink_ml.static_transfer.design import (
    admissible,
    admissible_points,
    input_offsets,
    noise_draws,
    noisy,
)
from twinlink_ml.static_transfer.surrogates import (
    BiasCorrectedBalance,
    CalibratedBalance,
    LinearRidge,
    calibrated_model,
)

CONFIG = load_config()
NOMINAL = np.asarray([CONFIG.source.nominal])


def _props(ua: float = 0.0) -> BlockProperties:
    s = CONFIG.source
    return BlockProperties(
        s.cp_gas_j_per_kg_k,
        s.cp_vapour_j_per_kg_k,
        s.cp_liquid_j_per_kg_k,
        s.cp_solid_j_per_kg_k,
        s.evaporation_enthalpy_0c_j_per_kg,
        heat_loss_ua_w_per_k=ua,
    )


def _block(x: np.ndarray, x_p: float = 0.002, ua: float = 0.0):  # type: ignore[no-untyped-def]
    t_in, load, t_amb, y_in, xs = (float(v) for v in x)
    m_g = CONFIG.source.air_kg_per_h / 3600.0
    inputs = BlockInputs(m_g, y_in / 1e3, t_in, load * m_g, xs, 20.0, x_p, t_amb)
    return inputs, dryer_block(inputs, _props(ua))


# ----------------------------------------------------------------------------- balance
def test_source_reproduces_the_exhaust_of_docs_11_section_13() -> None:
    y = source_model(CONFIG.source).predict(NOMINAL)[0]
    assert y[0] == pytest.approx(102.86, abs=0.005)
    assert y[1] == pytest.approx(23.98, abs=0.005)


def test_source_equals_the_partner_block_at_random_points() -> None:
    rng = np.random.default_rng(0)
    x = CONFIG.boxes["interpolation"].scale(rng.random((7, 5)))
    y = source_model(CONFIG.source).predict(x)
    for xi, yi in zip(x, y, strict=True):
        inputs, result = _block(xi)
        m_g = CONFIG.source.air_kg_per_h / 3600.0
        assert yi[0] == pytest.approx(result.outlet_temperature_celsius, rel=1e-14)
        assert yi[1] == pytest.approx(1e3 * result.outlet_humidity_kg_per_kg, rel=1e-14)
        power = heater_power_w(inputs, _props()) / m_g / 1e3 / CONFIG.source.heater_efficiency
        assert yi[2] == pytest.approx(power, rel=1e-14)


def test_p0_is_the_source_and_each_deviation_acts_where_declared() -> None:
    base = source_model(CONFIG.source).predict(NOMINAL)[0]
    out = {p.name: plant_model(CONFIG.source, p).predict(NOMINAL)[0] for p in CONFIG.plants}
    np.testing.assert_array_equal(out["P0"], base)
    for name, targets in CONFIG.affected_targets.items():
        changed = {t for j, t in enumerate(TARGET_NAMES) if abs(out[name][j] - base[j]) > 1e-9}
        assert changed == set(targets), name
    assert out["P-loss"][0] < base[0]  # heat loss cools
    assert out["P-air"][0] < base[0] and out["P-air"][1] < base[1]  # dilution
    assert out["P-moist"][0] > base[0] and out["P-moist"][1] < base[1]  # less evaporation
    assert out["P-eta"][2] < base[2]  # better efficiency at nominal load


def test_heat_loss_equals_the_block_with_ua() -> None:
    plant = plant_model(CONFIG.source, CONFIG.plant("P-loss"))
    _, result = _block(NOMINAL[0], ua=0.02 * 100.0)
    assert plant.predict(NOMINAL)[0, 0] == pytest.approx(
        result.outlet_temperature_celsius, rel=1e-14
    )


def test_false_air_conserves_water_and_enthalpy() -> None:
    spec = dataclasses.replace(CONFIG.plant("P0"), false_air_share=0.1)
    y, bed, _ = plant_model(CONFIG.source, spec).solve(NOMINAL)
    s = CONFIG.source
    y_in = NOMINAL[0, 3] / 1e3
    y_mix = y[0, 1] / 1e3
    assert 1.1 * y_mix == pytest.approx(bed.humidity_kg_per_kg[0] + 0.1 * y_in, rel=1e-14)

    def enthalpy(t: float, w: float) -> float:
        return s.cp_gas_j_per_kg_k * t + w * (
            s.evaporation_enthalpy_0c_j_per_kg + s.cp_vapour_j_per_kg_k * t
        )

    before = enthalpy(bed.temperature_celsius[0], bed.humidity_kg_per_kg[0]) + 0.1 * enthalpy(
        NOMINAL[0, 2], y_in
    )
    assert 1.1 * enthalpy(y[0, 0], y_mix) == pytest.approx(before, rel=1e-12)


def test_moisture_law_is_solved_and_hits_the_nominal_excess() -> None:
    plant = plant_model(CONFIG.source, CONFIG.plant("P-moist"))
    assert isinstance(plant.moisture, RhDependentMoisture)
    x = CONFIG.boxes["plant_like"].scale(np.random.default_rng(1).random((5, 5)))
    _, bed, x_p = plant.solve(np.vstack([NOMINAL, x]))
    expected = CONFIG.source.equilibrium_moisture_kg_per_kg + plant.moisture.excess(
        bed.relative_humidity
    )
    np.testing.assert_allclose(x_p, expected, rtol=0, atol=1e-12)
    assert x_p[0] == pytest.approx(0.002 + 0.06, abs=1e-12)


def test_relative_humidity_is_one_at_saturation() -> None:
    from twinlink_ml.flowsheet.validators import saturation_humidity_kg_per_kg

    y_sat = saturation_humidity_kg_per_kg(40.0)
    assert relative_humidity(np.asarray([y_sat]), np.asarray([40.0]), 101325.0)[0] == pytest.approx(
        1.0
    )


def test_load_dependent_efficiency_has_the_declared_shape() -> None:
    plant = plant_model(CONFIG.source, CONFIG.plant("P-eta"))
    eta = plant.efficiency.efficiency(np.asarray([0.4, 0.9, 1.4]))
    np.testing.assert_allclose(eta, [0.70, 0.85, 0.70], atol=1e-12)


# ----------------------------------------------------------------------------- designs
def test_admissible_points_are_nested_admissible_and_inside_the_box() -> None:
    models = study.admissibility_models(CONFIG)
    box = CONFIG.boxes["plant_like"]
    a = admissible_points(box, 20, 5, models, CONFIG.max_relative_humidity, batch=64)
    b = admissible_points(box, 40, 5, models, CONFIG.max_relative_humidity, batch=64)
    np.testing.assert_array_equal(a.x, b.x[:20])
    assert np.all(box.contains(b.x))
    assert np.all(admissible(b.x, models, CONFIG.max_relative_humidity))


def test_extrapolation_points_lie_outside_the_interpolation_box() -> None:
    models = [source_model(CONFIG.source)]
    sample = admissible_points(
        CONFIG.boxes["extrapolation_outer"],
        50,
        3,
        models,
        0.9,
        exclude=CONFIG.boxes["interpolation"],
        batch=64,
    )
    assert not np.any(CONFIG.boxes["interpolation"].contains(sample.x))


def test_input_offsets_are_zero_when_inactive_and_of_full_size_when_active() -> None:
    spec = CONFIG.input_errors
    assert input_offsets(spec, 1, False).as_dict() == dict.fromkeys(
        ("inlet_k", "spray_relative", "solid_fraction_relative", "inlet_humidity_g_per_kg"), 0.0
    )
    on = input_offsets(spec, 1, True)
    assert abs(on.inlet_k) == spec.inlet_k
    assert abs(on.spray_relative) == spec.spray_relative
    assert abs(on.solid_fraction_relative) == spec.solid_fraction_relative
    assert abs(on.inlet_humidity_g_per_kg) == spec.inlet_humidity_g_per_kg
    assert input_offsets(spec, 1, True) == on
    x = on.true_inputs(NOMINAL)
    assert x[0, 0] == pytest.approx(NOMINAL[0, 0] + on.inlet_k)
    assert x[0, 1] == pytest.approx(NOMINAL[0, 1] * (1 + on.spray_relative))


def test_noise_levels_share_one_draw() -> None:
    truth = np.tile([100.0, 20.0, 150.0], (4, 1))
    draws = noise_draws(3, 4)
    low = noisy(truth, CONFIG.noise, draws) - truth
    high = noisy(truth, CONFIG.noise.scaled(3.0), draws) - truth
    np.testing.assert_allclose(high, 3.0 * low)
    assert low[0, 2] == pytest.approx(0.02 * 150.0 * draws[0, 2])


# ----------------------------------------------------------------------------- methods
def test_calibration_recovers_parameters_on_noise_free_data() -> None:
    theta = np.asarray([0.86, 0.03, 0.04, 0.05])
    truth = calibrated_model(CONFIG.source, theta)
    x = CONFIG.boxes["interpolation"].scale(np.random.default_rng(2).random((32, 5)))
    l1 = CalibratedBalance(CONFIG.source, CONFIG.calibration_parameters, CONFIG.noise, 1e6)
    l1.fit(x, truth.predict(x))
    np.testing.assert_allclose(l1.theta, theta, rtol=1e-4, atol=1e-6)


def test_bias_correction_and_ridge() -> None:
    x = CONFIG.boxes["interpolation"].scale(np.random.default_rng(3).random((16, 5)))
    source = source_model(CONFIG.source)
    b1 = BiasCorrectedBalance(source).fit(x, source.predict(x) + np.asarray([1.0, -0.5, 2.0]))
    np.testing.assert_allclose(b1.bias, [1.0, -0.5, 2.0], atol=1e-12)
    coef = np.asarray(
        [[1.0, 0.0, 2.0], [0.5, -1.0, 0.0], [0.0, 0.0, 1.0], [2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]
    )
    z = CONFIG.boxes["pretraining"].to_unit_symmetric(x)
    ridge = LinearRidge(CONFIG.boxes["pretraining"], np.asarray([1e-10]))
    ridge.fit(x, z @ coef + 4.0)
    np.testing.assert_allclose(ridge.predict(x), z @ coef + 4.0, atol=1e-6)


def test_extend_grid_rules() -> None:
    grid = {"steps": [100.0, 300.0, 1000.0], "sp_lambda": [0.0, 1e-3, 1e-1], "lr": [1e-3, 1e-2]}
    new, counts, ext = study.extend_grid(grid, {"steps": 1000, "sp_lambda": 0.0, "lr": 1e-2}, {}, 2)
    assert ext == ["steps", "lr"]
    assert new["steps"][-1] == 3333.0 and new["lr"][-1] == pytest.approx(0.1)
    assert new["sp_lambda"] == grid["sp_lambda"]  # 0 is a natural bound
    _, _, ext2 = study.extend_grid(
        new, {"steps": 3333, "sp_lambda": 0.0, "lr": 0.1}, {"steps": 2, "lr": 2}, 2
    )
    assert ext2 == []


# ----------------------------------------------------------------------------- verdicts
def _records(values: dict[str, list[float]]) -> list[dict]:  # type: ignore[type-arg]
    recs = []
    for seed in range(5):
        fitted = {
            "2": {
                m: {"test": {"interpolation": {"rmse": [v[seed], 1.0, 1.0]}}}
                for m, v in values.items()
                if m != "L0"
            }
        }
        recs.append(
            {
                "plant": "P-x",
                "condition": "clean_low",
                "seed": seed,
                "fixed": {"L0": {"interpolation": {"rmse": [values["L0"][seed], 1.0, 1.0]}}},
                "fitted": fitted,
            }
        )
    return recs


def test_verdict_needs_holm_and_sign_agreement() -> None:
    table = MetricTable(
        _records(
            {
                "L0": [2.0, 2.1, 2.2, 2.0, 2.1],
                "L3": [1.0, 1.1, 1.0, 1.05, 1.0],  # clearly better in every seed
                "L4": [1.0, 1.0, 1.0, 1.0, 9.0],  # better in 4 seeds, one bad seed
            }
        )
    )
    c3 = compare(table, "L3", "L0", "P-x", "clean_low", 2, "interpolation", "outlet_temperature")
    c4 = compare(table, "L4", "L0", "P-x", "clean_low", 2, "interpolation", "outlet_temperature")
    fam = Family("t", [c3, c4], 0.05, 4)
    assert c3.verdict == "better" and c3.seeds_better == 5
    assert c4.verdict == "unresolved"  # mean difference negative: not significant
    assert fam.size == 2


def test_smallest_n_from_which() -> None:
    assert smallest_n_from_which([2, 4, 8], {2: False, 4: True, 8: True}) == 4
    assert smallest_n_from_which([2, 4, 8], {2: True, 4: False, 8: True}) == 8
    assert smallest_n_from_which([2, 4, 8], {2: True, 4: True, 8: False}) is None


# ----------------------------------------------------------------------------- torch
def test_unit_runs_end_to_end_with_tiny_settings() -> None:
    pytest.importorskip("torch")
    raw = {**CONFIG.raw, "selected": {
        "pilot_artifact": None,
        "L1": {"prior_scale": 1.0},
        "L2": {"learning_rate": 1e-2, "steps": 5, "weight_decay": 0.1},
        "L3": {"learning_rate": 1e-3, "steps": 5, "sp_lambda": 1e-3},
        "L4": {"learning_rate": 1e-2, "steps": 5, "weight_decay": 1e-3},
    }}  # fmt: skip
    config = dataclasses.replace(CONFIG, raw=raw, budgets=(2, 4))
    ctx = study.build_context(config, [1], test_count=16)
    net = study.fresh_net(config, 1)
    rec = study.run_unit(ctx, "P-comb", "inputerr_low", 1, net)
    assert set(rec["fitted"]) == {"2", "4"}
    assert set(rec["fitted"]["2"]) == set(study.FITTED_METHODS)
    assert set(rec["fixed"]) == set(study.FIXED_METHODS)
    assert all(math.isfinite(v) for v in rec["fitted"]["4"]["L1"]["test"]["plant_like"]["rmse"])
    assert rec["offsets"]["inlet_k"] != 0.0
    pilot = study.pilot_task(
        dataclasses.replace(
            ctx, designs={**ctx.designs, **study.build_context(config, [11, 12, 13], 16).designs}
        ),
        "L3",
        {"learning_rate": 1e-3, "steps": 3, "sp_lambda": 0.0},
        {11: net, 12: net, 13: net},
    )
    assert len(pilot["cells"]) == 6 and math.isfinite(pilot["criterion"])


def test_pretraining_lowers_the_validation_error() -> None:
    pytest.importorskip("torch")
    pre = dataclasses.replace(CONFIG.pretraining, max_epochs=30, patience=30)
    config = dataclasses.replace(CONFIG, pretraining=pre)
    net, rec = study.pretraining_task(config, 1, smoke=True)
    assert rec["best_validation_mse_scaled"] < 1.0


# ----------------------------------------------------------------------------- workbook
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", Path(__file__).resolve().parents[1] / "local"))
WORKBOOK = LOCAL / "partner" / "TwinLink Excel.xlsx"
WORKBOOK_412 = LOCAL / "partner" / "Model v4.1_2.xlsx"


def test_config_equals_the_partner_workbooks() -> None:
    if not WORKBOOK.exists() or not WORKBOOK_412.exists():
        pytest.skip("local/partner workbooks not available")
    pytest.importorskip("openpyxl")
    from twinlink_ml.partner_v4.inputs import load_excel_inputs

    ex = load_excel_inputs(WORKBOOK)
    s = CONFIG.source
    m = ex.material
    assert m.cp_gas_kj_per_kg_k * 1e3 == pytest.approx(s.cp_gas_j_per_kg_k)
    assert m.cp_vapour_kj_per_kg_k * 1e3 == pytest.approx(s.cp_vapour_j_per_kg_k)
    assert m.cp_liquid_kj_per_kg_k * 1e3 == pytest.approx(s.cp_liquid_j_per_kg_k)
    assert m.cp_solid_kj_per_kg_k * 1e3 == pytest.approx(s.cp_solid_j_per_kg_k)
    assert m.enthalpy_of_evaporation_kj_per_kg * 1e3 == pytest.approx(
        s.evaporation_enthalpy_0c_j_per_kg
    )
    assert m.equilibrium_moisture_g_per_kg / 1e3 == pytest.approx(s.equilibrium_moisture_kg_per_kg)
    assert m.heater_efficiency_percent / 100 == pytest.approx(s.heater_efficiency)
    d = ex.dryer_2
    assert d.air_mass_flow_kg_per_h == pytest.approx(s.air_kg_per_h)
    assert d.air_temperature_celsius == pytest.approx(s.nominal[0])
    assert d.spray_mass_flow_kg_per_h / d.air_mass_flow_kg_per_h == pytest.approx(s.nominal[1])
    assert ex.ambient_temperature_celsius == pytest.approx(s.nominal[2])
    assert d.air_moisture_g_per_kg == pytest.approx(s.nominal[3])
    assert d.spray_solid_fraction_kg_per_kg == pytest.approx(s.nominal[4])
    assert d.spray_temperature_celsius == pytest.approx(s.spray_temperature_celsius)
    ex2 = load_excel_inputs(WORKBOOK_412)
    sens = CONFIG.sensitivity_source
    assert ex2.material.heater_efficiency_percent / 100 == pytest.approx(sens.heater_efficiency)
    assert ex2.dryer_2.spray_temperature_celsius == pytest.approx(sens.spray_temperature_celsius)
    assert ex2.material.cp_gas_kj_per_kg_k * 1e3 == pytest.approx(sens.cp_gas_j_per_kg_k)


# ----------------------------------------------------------------------------- post hoc v2
def test_posthoc_config_and_contexts() -> None:
    from twinlink_ml.static_transfer.posthoc import (
        check_against_v1,
        load_posthoc,
        pilot_config,
        rerun_config,
    )

    spec = load_posthoc()
    assert spec.pilot_plant.name not in {p.name for p in CONFIG.plants}
    assert spec.pilot_plant != dataclasses.replace(CONFIG.pilot_plant, name=spec.pilot_plant.name)
    assert max(spec.grid["learning_rate"]) == 1e-2 and max(spec.grid["steps"]) == 5000
    assert 0.0 in spec.grid["sp_lambda"] and max(spec.grid["sp_lambda"]) > 0.0
    assert not set(spec.pilot_seeds) & (set(CONFIG.seeds) | {11, 12, 13})
    pc = pilot_config(CONFIG, spec)
    assert (
        pc.plants == CONFIG.plants
        and pc.boxes == CONFIG.boxes
        and pc.pilot_plant == spec.pilot_plant
    )
    assert list(pc.raw["pilot"]["grids"]) == ["L3"]
    rc = rerun_config(CONFIG, {"learning_rate": 1e-3, "steps": 1000, "sp_lambda": 0.0})
    for m in ("L1", "L2", "L4"):
        assert rc.selected(m) == CONFIG.selected(m)
    assert rc.selected("L3") == {"learning_rate": 1e-3, "steps": 1000.0, "sp_lambda": 0.0}
    assert CONFIG.selected("L3") != rc.selected("L3")  # the v1 config object is untouched

    def err() -> dict:  # type: ignore[type-arg]
        return {"interpolation": {"rmse": [1.0, 2.0, 3.0]}}

    unit = {
        "fixed": {"L0": err()},
        "fitted": {"2": {"L1": {"test": err()}, "L3": {"test": err()}}},
    }
    other = copy.deepcopy(unit)
    other["fitted"]["2"]["L3"]["test"]["interpolation"]["rmse"] = [9.0, 9.0, 9.0]
    assert check_against_v1(other, unit)["max_abs_difference"] == 0.0
    other["fitted"]["2"]["L1"]["test"]["interpolation"]["rmse"] = [1.0, 2.5, 3.0]
    assert check_against_v1(other, unit)["max_abs_difference"] == pytest.approx(0.5)
