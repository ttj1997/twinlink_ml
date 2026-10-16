"""Tests of the operating-point study of plant A (``twinlink_ml.plant_a_op``), synthetic data."""

from __future__ import annotations

import re
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.plant_a_op import (
    PLACEHOLDERS,
    DryerModel,
    DryerPlaceholders,
    DutyHeld,
    EnergyPlaceholders,
    FixedWaterInlet,
    PlantAModel,
    PowerLawFan,
    SearchSpace,
    consistent_solid_fraction,
    constraints_for,
    day_state,
    fan_law,
    fit_preheater_conductance,
    loss_conductance_kw_per_k,
    minimum_admissible_solid_fraction,
    optimise,
    two_point_exponent,
    within_day_exponent,
)
from twinlink_ml.plant_a_op.config import fingerprint_of
from twinlink_ml.plant_a_op.data import DayState, MeasuredDay, measured_day_from_columns
from twinlink_ml.plant_a_op.dryer import (
    _flows,
    _residual_kw,
    observed_outlet_extremes,
    relative_humidity_unclipped,
)
from twinlink_ml.plant_a_op.optimize import Constraints, coarse_grid
from twinlink_ml.plant_a_op.study import (
    Case,
    StudyData,
    build_model,
    calibration_check,
    run_case,
)
from twinlink_ml.plant_a_op.waste_heat import preheater_conductance_rate
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.thermo import relative_humidity_from_absolute_humidity

REPO = Path(__file__).resolve().parents[1]
PH = DryerPlaceholders()


def _state(**kw: float) -> DayState:
    base = {
        "label": "synthetic",
        "ambient_celsius": 5.0,
        "process_air_nm3h": 33000.0,
        "atomising_air_nm3h": 560.0,
        "supply_celsius": 84.5,
        "outlet_celsius": 60.5,
        "spray_kg_h": 930.0,
        "process_fan_kw": 117.0,
        "exhaust_fan_kw": 117.0,
        "preheater_loop_flow_m3h": 18.5,
        "preheater_water_in_celsius": 39.4,
        "preheater_duty_kw": 277.0,
    }
    base.update(kw)
    return DayState(**base)  # type: ignore[arg-type]


def _measured_day(label: str, **kw: float) -> MeasuredDay:
    s = _state(**kw)
    n = 1441
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, n)
    cols = {
        "ambient_temperature": np.full(n, s.ambient_celsius) + noise,
        "process_air_flow": np.full(n, s.process_air_nm3h) + 50.0 * noise,
        "supply_air_temperature": np.full(n, s.supply_celsius),
        "exhaust_outlet_temperature": np.full(n, s.outlet_celsius) + 0.1 * noise,
        "spray_rate": np.full(n, s.spray_kg_h) + 10.0 * noise,
        "atomising_air_flow": np.full(n, s.atomising_air_nm3h),
        "process_fan_power": np.full(n, s.process_fan_kw) * (1.0 + 0.01 * noise),
        "exhaust_fan_power": np.full(n, s.exhaust_fan_kw),
        "preheater_loop_flow": np.full(n, s.preheater_loop_flow_m3h),
        "preheater_supply_temperature": np.full(n, s.preheater_water_in_celsius),
        "preheater_return_temperature": np.full(n, s.preheater_water_in_celsius - 13.0),
    }
    return measured_day_from_columns(label, cols)


def _data() -> StudyData:
    d1 = _measured_day("p1")
    d2 = _measured_day(
        "p2",
        ambient_celsius=11.5,
        process_air_nm3h=30500.0,
        atomising_air_nm3h=820.0,
        supply_celsius=83.1,
        outlet_celsius=34.4,
        spray_kg_h=2118.0,
        process_fan_kw=112.0,
        exhaust_fan_kw=58.0,
        preheater_water_in_celsius=51.4,
    )
    return StudyData(days={"p1": d1, "p2": d2}, other={"p1": "p2", "p2": "p1"})


# -- dryer balance ----------------------------------------------------------------------------
def test_outlet_temperature_closes_the_balance() -> None:
    s = _state()
    model = DryerModel(PH, 1.5)
    t_out = model.outlet_celsius(
        s.process_air_nm3h,
        s.supply_celsius,
        atomising_air_nm3h=s.atomising_air_nm3h,
        spray_kg_h=s.spray_kg_h,
        ambient_celsius=s.ambient_celsius,
    )
    f = _flows(PH, s.process_air_nm3h, s.atomising_air_nm3h, s.spray_kg_h, s.ambient_celsius)
    residual = _residual_kw(PH, f, s.supply_celsius, t_out) - model.heat_loss_kw(t_out)
    assert abs(float(residual)) < 1e-9


def test_loss_conductance_round_trip() -> None:
    s = _state()
    t_out = float(
        DryerModel(PH, 2.0).outlet_celsius(
            s.process_air_nm3h,
            s.supply_celsius,
            atomising_air_nm3h=s.atomising_air_nm3h,
            spray_kg_h=s.spray_kg_h,
            ambient_celsius=s.ambient_celsius,
        )
    )
    assert loss_conductance_kw_per_k(PH, replace(s, outlet_celsius=t_out)) == pytest.approx(2.0)


def test_outlet_falls_with_less_air_and_rises_with_supply() -> None:
    s = _state()
    m = DryerModel(PH, 1.0)
    kw = {
        "atomising_air_nm3h": s.atomising_air_nm3h,
        "spray_kg_h": s.spray_kg_h,
        "ambient_celsius": s.ambient_celsius,
    }
    t = m.outlet_celsius(np.array([0.6, 1.0]) * s.process_air_nm3h, 84.5, **kw)
    assert t[0] < t[1]
    t2 = m.outlet_celsius(s.process_air_nm3h, np.array([80.0, 90.0]), **kw)
    assert t2[0] < t2[1]


def test_solid_fraction_bounds() -> None:
    s1, s2 = (
        _state(),
        _state(
            outlet_celsius=34.4, spray_kg_h=2118.0, supply_celsius=83.1, process_air_nm3h=30500.0
        ),
    )
    x_min = minimum_admissible_solid_fraction(PH, s1)
    assert loss_conductance_kw_per_k(replace(PH, solid_fraction=x_min), s1) == pytest.approx(
        0.0, abs=1e-9
    )
    x, ua = consistent_solid_fraction(PH, s1, s2)
    ph = replace(PH, solid_fraction=x)
    assert loss_conductance_kw_per_k(ph, s1) == pytest.approx(ua)
    assert loss_conductance_kw_per_k(ph, s2) == pytest.approx(ua)


def test_relative_humidity_matches_thermo_below_saturation() -> None:
    y = np.array([0.005, 0.02])
    t = np.array([40.0, 60.0])
    ref = relative_humidity_from_absolute_humidity(NumpyBackend(), PH.gas, y, t)
    assert np.allclose(relative_humidity_unclipped(y, t, PH), ref)
    assert float(relative_humidity_unclipped(np.array([0.2]), np.array([30.0]), PH)[0]) > 1.0


# -- fans and waste heat ----------------------------------------------------------------------
def test_fan_laws() -> None:
    assert two_point_exponent(8.0, 2.0, 1.0, 1.0) == pytest.approx(3.0)
    assert float(PowerLawFan(100.0, 3.0).power_kw(0.5)) == pytest.approx(12.5)
    assert fan_law("affinity", 0.5).exponent_exhaust == 3.0
    assert fan_law("data_fit", 0.5).exponent_exhaust == 0.5
    with pytest.raises(ValueError):
        fan_law("data_fit", 8.8)
    v = np.linspace(0.9, 1.1, 50)
    fit = within_day_exponent(2.0 * v**1.7, v)
    assert fit["slope"] == pytest.approx(1.7)


def test_preheater_fit_and_variants() -> None:
    c_w, c_air = 21.3, 12.0
    ua = fit_preheater_conductance(277.0, c_w, c_air, 39.4, 5.0)
    assert float(preheater_conductance_rate(ua, c_w, c_air)) * 34.4 == pytest.approx(277.0)
    fixed = FixedWaterInlet(ua, 18.5, 39.4)
    duties = fixed.duty_kw(np.array([0.6 * c_air, c_air]), 5.0)
    assert duties[0] < duties[1]
    held = DutyHeld(277.0, ua, 18.5, 80.0)
    assert float(held.duty_kw(np.array([0.6 * c_air]), 5.0)[0]) == pytest.approx(277.0)
    tight = DutyHeld(277.0, ua, 18.5, 30.0)
    assert float(tight.duty_kw(np.array([c_air]), 5.0)[0]) < 277.0


# -- plant model, search, decomposition --------------------------------------------------------
def _model(fan_exponent: float = 3.0, ua: float = 1.0) -> PlantAModel:
    s = _state()
    return PlantAModel(
        state=s,
        dryer=DryerModel(PH, ua),
        process_fan=PowerLawFan(s.process_fan_kw, fan_exponent),
        exhaust_fan=PowerLawFan(s.exhaust_fan_kw, fan_exponent),
        waste_heat=DutyHeld(s.preheater_duty_kw, 17.0, s.preheater_loop_flow_m3h, 80.0),
        energy=EnergyPlaceholders(),
    )


def test_baseline_booking() -> None:
    m = _model()
    b = m.baseline().at(0)
    assert b["steam_kw"] == pytest.approx(
        b["air_heating_kw"] - b["fan_heat_kw"] - b["waste_heat_kw"]
    )
    e = m.energy
    assert b["co2_kg_h"] == pytest.approx(
        b["steam_kw"] / e.boiler_efficiency * e.gas_co2_kg_per_kwh
        + (b["process_fan_kw"] + b["exhaust_fan_kw"]) * e.electricity_co2_kg_per_kwh
    )


@pytest.mark.parametrize("variant", ["a", "b", "a_and_b"])
def test_optimum_respects_constraints_and_decomposes(variant: str) -> None:
    m = _model()
    base = m.baseline()
    t_base = float(base.outlet_celsius[0])
    c = constraints_for(
        variant,  # type: ignore[arg-type]
        observed_outlet_min_celsius=t_base - 0.2,
        observed_rh_max=float(base.outlet_relative_humidity[0]) + 0.01,
        air_factor_min=0.8,
        baseline=base,
    )
    space = SearchSpace(air_factor_step=0.01, supply_step_k=0.5, refine=5)
    opt = optimise(m, c, space)
    assert opt.saving_kg_h >= 0.0
    assert opt.best["air_factor"] >= 0.8 - 1e-12
    if c.outlet_min_celsius is not None:
        assert opt.best["outlet_celsius"] >= c.outlet_min_celsius - 1e-9
    if c.rh_max is not None:
        assert opt.best["outlet_relative_humidity"] <= c.rh_max + 1e-12
    d = opt.decomposition
    assert d["total"] == pytest.approx(24.0 * opt.saving_kg_h)
    assert d["steam"] == pytest.approx(
        d["steam_from_air_heating"]
        + d["steam_from_lost_fan_heat"]
        + d["steam_from_lost_waste_heat"]
    )
    assert d["fan_electricity"] == pytest.approx(
        d["fan_electricity_process"] + d["fan_electricity_exhaust"]
    )
    assert opt.binding  # something must bind: the box or a constraint


def test_shared_grid_gives_same_optimum() -> None:
    m = _model()
    space = SearchSpace(air_factor_step=0.01, supply_step_k=0.5, refine=5)
    c = Constraints(outlet_min_celsius=55.0, rh_max=None, air_factor_min=0.7)
    assert optimise(m, c, space).best == optimise(m, c, space, coarse_grid(m, space)).best


def test_thresholds_never_exclude_the_baseline() -> None:
    m = _model()
    base = m.baseline()
    c = constraints_for(
        "a_and_b",
        observed_outlet_min_celsius=99.0,
        observed_rh_max=0.0,
        air_factor_min=0.9,
        baseline=base,
    )
    assert bool(c.admissible(base)[0])


def test_fixed_fans_without_constraints_go_to_the_box() -> None:
    """Without a proxy constraint the cheapest point has the smallest air flow (least heat)."""
    m = _model(fan_exponent=0.0)
    c = Constraints(outlet_min_celsius=None, rh_max=None, air_factor_min=0.6)
    opt = optimise(m, c, SearchSpace(air_factor_step=0.01, supply_step_k=0.5, refine=5))
    assert "air_factor_min" in opt.binding


# -- study assembly ---------------------------------------------------------------------------
def test_build_and_run_case() -> None:
    data = _data()
    case = Case(product="p1", dryer=replace(PH, solid_fraction=0.7))
    model, info = build_model(data, case)
    assert model is not None and info["ua_dryer_kw_per_k"] > 0.0
    b = model.baseline().at(0)
    s = day_state(data.days["p1"])
    assert b["outlet_celsius"] == pytest.approx(s.outlet_celsius)
    assert b["waste_heat_kw"] == pytest.approx(s.preheater_duty_kw, rel=1e-6)
    rows = run_case(
        data,
        case,
        [("a", 0.8), ("b", 0.9)],
        SearchSpace(air_factor_step=0.02, supply_step_k=1.0, refine=4),
    )
    assert len(rows) == 2 and all(r["admissible"] for r in rows)
    low = Case(product="p2", dryer=replace(PH, solid_fraction=0.3))
    assert build_model(data, low)[0] is None
    assert not run_case(data, low, [("a", 0.8)], SearchSpace())[0]["admissible"]


def test_calibration_check_in_sample_is_exact() -> None:
    data = _data()
    ph = replace(PH, solid_fraction=0.7)
    chk = calibration_check(data, ph, "p1", "p1")
    assert chk["outlet_mean_error_k"] == pytest.approx(0.0, abs=1e-9)
    ext = observed_outlet_extremes(data.days["p1"], ph)
    assert ext["outlet_min_celsius"] <= ext["outlet_mean_celsius"]
    assert ext["rh_max"] >= ext["rh_mean"]


def test_case_fingerprint_moves_with_placeholders() -> None:
    a = Case(product="p1")
    assert a.fingerprint == Case(product="p1").fingerprint
    assert a.fingerprint != Case(product="p1", dryer=replace(PH, solid_fraction=0.7)).fingerprint
    assert fingerprint_of(SearchSpace()) != fingerprint_of(SearchSpace(refine=2))


def test_placeholder_table_matches_defaults() -> None:
    d, e = DryerPlaceholders(), EnergyPlaceholders()
    for key, spec in PLACEHOLDERS.items():
        section, name = key.split(".", 1)
        obj = {"dryer": d, "energy": e}.get(section)
        if obj is not None:
            assert getattr(obj, name) == spec.value, key


# -- confidentiality --------------------------------------------------------------------------
def test_new_files_name_no_measurement_points_or_equipment_codes() -> None:
    """No tags / equipment codes in code, docs and tests of this study (roles only)."""
    out = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--cached",
            "--exclude-standard",
            "--",
            "src/twinlink_ml/plant_a_op",
            "scripts/study_plant_a_operating_point.py",
            "docs/23_plant_a_operating_point.md",
            "tests/test_plant_a_op.py",
            "conf/plant/plant_local.yaml.example",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    ).stdout.split()
    tag = re.compile(r"\b[A-Z]{2,7}-\d{5,6}\b|\b[WBV]-? ?\d{4}[A-Z]?\b|\bCV-\d")
    hits = []
    for name in out:
        path = REPO / name
        if path.is_file():
            hits += [f"{name}: {m}" for m in tag.findall(path.read_text(encoding="utf-8"))]
    assert not hits, hits
