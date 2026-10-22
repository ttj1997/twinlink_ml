"""Validators, saturation limit, medium limit and the single line of ``twinlink_ml.flowsheet``.

Synthetic cases only (no partner data): the reference values of IAPWS-IF97 are those of the
verification tables of the release (tables 35 and 36), everything else is a balance or an
ordering that must hold by construction.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from twinlink_ml.flowsheet.dryer import DryerBlock, DryerConfig, DryerInputs
from twinlink_ml.flowsheet.line import Line, LineInputs, LineMode
from twinlink_ml.flowsheet.loop import LoopInputs, StageModel, run_stage
from twinlink_ml.flowsheet.thermo import GasProps, capacity
from twinlink_ml.flowsheet.validators import (
    SaturationProps,
    adiabatic_saturation_temperature_celsius,
    check_exhaust_above_adiabatic_saturation,
    check_exhaust_saturation,
    check_medium_boiling,
    check_product_mass,
    saturation_humidity_kg_per_kg,
    water_boiling_point_celsius,
    water_saturation_pressure_pa,
)
from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import ATM_PA, SiAmbient, SiDryer, SiMaterial
from twinlink_ml.partner_v4_1.comparison import reported_decimals

MATERIAL = SiMaterial(
    cp_gas_j_per_kg_k=1006.0,
    lower_heating_value_j_per_kg=50e6,
    heater_efficiency=0.8,
    cp_liquid_j_per_kg_k=4185.0,
    cp_solid_j_per_kg_k=1000.0,
    cp_vapour_j_per_kg_k=1906.0,
    equilibrium_moisture_kg_per_kg=0.002,
    enthalpy_of_evaporation_j_per_kg=2.5e6,
)
PROPS = GasProps("synthetic", 1006.0, 1906.0)
SAT = SaturationProps.from_material(MATERIAL)
HOT_DRYER = SiDryer(100 / 3600, 0.008, 220.0, 8 / 3600, 0.4, 20.0)
"""Dry exhaust (well above saturation)."""
WET_DRYER = SiDryer(100 / 3600, 0.006, 200.0, 14 / 3600, 0.3, 20.0)
"""So much spray that the block without limit would leave the exhaust oversaturated."""


def _dryer(d: SiDryer, *, limit: bool) -> DryerBlock:
    return DryerBlock(
        DryerConfig(MATERIAL, SiAmbient(20.0, ATM_PA), PROPS, 0.0, 0.002, saturation_limit=limit)
    )


# ---------------------------------------------------------------------- water properties
@pytest.mark.parametrize(
    ("kelvin", "mpa"),
    [(300.0, 0.353658941e-2), (500.0, 0.263889776e1), (600.0, 0.123443146e2)],
)
def test_saturation_pressure_matches_if97_table_35(kelvin: float, mpa: float) -> None:
    assert water_saturation_pressure_pa(kelvin - 273.15) == pytest.approx(mpa * 1e6, rel=1e-8)


@pytest.mark.parametrize(
    ("mpa", "kelvin"),
    [(0.1, 0.372755919e3), (1.0, 0.453035632e3), (10.0, 0.584149488e3)],
)
def test_boiling_point_matches_if97_table_36(mpa: float, kelvin: float) -> None:
    assert water_boiling_point_celsius(mpa * 1e6) == pytest.approx(kelvin - 273.15, abs=1e-6)


def test_boiling_point_at_one_atmosphere_and_round_trip() -> None:
    t = water_boiling_point_celsius(ATM_PA)
    assert t == pytest.approx(99.974, abs=1e-3)
    assert water_saturation_pressure_pa(t) == pytest.approx(ATM_PA, rel=1e-6)


def test_saturation_humidity_rises_and_is_infinite_at_boiling() -> None:
    ys = [saturation_humidity_kg_per_kg(t) for t in (10.0, 30.0, 50.0, 70.0, 90.0)]
    assert ys == sorted(ys)
    assert saturation_humidity_kg_per_kg(20.0) == pytest.approx(0.0147, abs=2e-4)
    assert math.isinf(saturation_humidity_kg_per_kg(100.5))
    assert saturation_humidity_kg_per_kg(100.5, 2e5) < math.inf


def test_adiabatic_saturation_closes_its_balance_and_lies_between_dew_point_and_dry_bulb() -> None:
    t, y = 200.0, 0.006
    t_as = adiabatic_saturation_temperature_celsius(t, y, SAT)
    y_s = saturation_humidity_kg_per_kg(t_as)
    lhs = (SAT.cp_gas_j_per_kg_k + y * SAT.cp_vapour_j_per_kg_k) * (t - t_as)
    assert lhs == pytest.approx((y_s - y) * SAT.latent_heat_j_per_kg(t_as), rel=1e-8)
    assert 10.0 < t_as < t
    saturated = saturation_humidity_kg_per_kg(40.0)
    assert adiabatic_saturation_temperature_celsius(40.0, saturated, SAT) == pytest.approx(
        40.0, abs=1e-6
    )


# ---------------------------------------------------------------------- the checks
def test_exhaust_saturation_check() -> None:
    y_s = saturation_humidity_kg_per_kg(45.0)
    assert check_exhaust_saturation(45.0, 0.99 * y_s).ok
    bad = check_exhaust_saturation(45.0, 1.5 * y_s)
    assert bad.severity == "violation"
    assert bad.values["relative_saturation"] == pytest.approx(1.5)


def test_exhaust_below_adiabatic_saturation_is_a_violation() -> None:
    t_as = adiabatic_saturation_temperature_celsius(200.0, 0.006, SAT)
    assert check_exhaust_above_adiabatic_saturation(t_as + 1.0, 200.0, 0.006, SAT).ok
    bad = check_exhaust_above_adiabatic_saturation(t_as - 5.0, 200.0, 0.006, SAT)
    assert bad.severity == "violation"
    assert bad.values["margin_k"] == pytest.approx(-5.0)


def test_product_lighter_than_the_solid_feed_is_a_violation() -> None:
    spray, xs = 12 / 3600, 0.4
    assert check_product_mass(spray * xs * 1.002, spray, xs).ok
    assert check_product_mass(spray * xs, spray, xs).ok
    bad = check_product_mass(0.97 * spray * xs, spray, xs)
    assert bad.severity == "violation"
    assert bad.values["ratio"] == pytest.approx(0.97)


def test_medium_boiling_warning_states_the_saturation_pressure() -> None:
    assert check_medium_boiling(95.0).ok
    warn = check_medium_boiling(120.0)
    assert warn.severity == "warning"
    assert warn.values["required_pressure_pa"] == pytest.approx(water_saturation_pressure_pa(120.0))
    assert warn.values["required_pressure_pa"] == pytest.approx(198.67e3, rel=1e-3)
    assert check_medium_boiling(115.0, pressure_pa=2e5).ok


# ---------------------------------------------------------------------- dryer limit
def test_saturation_limit_does_nothing_for_a_dry_exhaust() -> None:
    free = _dryer(HOT_DRYER, limit=False).solve(DryerInputs(HOT_DRYER))
    limited = _dryer(HOT_DRYER, limit=True).solve(DryerInputs(HOT_DRYER))
    assert not limited.saturation_limited
    assert limited.exhaust == free.exhaust
    assert limited.product_moisture_kg_per_kg == pytest.approx(0.002)


def test_saturation_limit_saturates_the_exhaust_and_closes_the_balances() -> None:
    d = WET_DRYER
    free = _dryer(d, limit=False).solve(DryerInputs(d))
    assert not check_exhaust_saturation(
        free.exhaust.temperature_celsius, free.exhaust.humidity_kg_per_kg
    ).ok
    out = _dryer(d, limit=True).solve(DryerInputs(d))
    t, y = out.exhaust.temperature_celsius, out.exhaust.humidity_kg_per_kg
    assert out.saturation_limited
    assert y == pytest.approx(saturation_humidity_kg_per_kg(t), rel=1e-9)
    assert t > free.exhaust.temperature_celsius
    # water: spray water = evaporation + water in the product
    solid = d.spray_mass_flow_kg_per_s * d.spray_solid_fraction_kg_per_kg
    water = d.spray_mass_flow_kg_per_s - solid
    assert out.evaporation_kg_per_s + out.product_moisture_kg_per_kg * solid == pytest.approx(water)
    assert out.product_kg_per_s > free.product_kg_per_s
    # energy, in the form of the block (liquid at 0 degC as reference)
    props = _dryer(d, limit=True).properties()
    c_air = d.air_mass_flow_kg_per_s * (1006.0 + d.air_humidity_kg_per_kg * 1906.0)
    c_spray = d.spray_mass_flow_kg_per_s * props.suspension_cp(d.spray_solid_fraction_kg_per_kg)
    residual = (
        c_air * (d.air_temperature_celsius - t)
        + c_spray * (d.spray_temperature_celsius - t)
        - out.evaporation_kg_per_s * props.latent_heat_j_per_kg(t)
    )
    assert abs(residual) < 1e-6 * c_air * d.air_temperature_celsius


def test_saturation_limited_exhaust_is_the_adiabatic_saturation_temperature_for_a_warm_spray() -> (
    None
):
    """With the liquid entering at ``T_as`` (and no solid), the limited block is the saturator."""
    d = replace(WET_DRYER, spray_solid_fraction_kg_per_kg=0.0)
    t_as = adiabatic_saturation_temperature_celsius(
        d.air_temperature_celsius, d.air_humidity_kg_per_kg, SAT
    )
    d = replace(d, spray_temperature_celsius=t_as)
    out = _dryer(d, limit=True).solve(DryerInputs(d))
    assert out.saturation_limited
    assert out.exhaust.temperature_celsius == pytest.approx(t_as, abs=1e-7)
    # with a cold spray the saturated exhaust is colder than T_as: what check (a2) then reports
    cold = _dryer(WET_DRYER, limit=True).solve(DryerInputs(WET_DRYER))
    assert cold.exhaust.temperature_celsius < t_as


# ---------------------------------------------------------------------- medium limit
def _loop_inputs(flow_kg_per_h: float = 40.0) -> LoopInputs:
    exhaust = c.Stream(120 / 3600, 0.06, 150.0)
    gas = c.Stream(120 / 3600, 0.006, 20.0)
    return LoopInputs(1, 1, exhaust, gas, flow_kg_per_h / 3600, 4185.0, 250.0)


def test_medium_limit_caps_the_medium_and_none_changes_nothing() -> None:
    base = StageModel("cc", PROPS, "counter_current", "sensible", 10.0, "each")
    free = run_stage(base, _loop_inputs())
    assert free.htm_hot_celsius is not None and free.htm_hot_celsius > 100.0
    assert run_stage(replace(base, htm_max_celsius=None), _loop_inputs()) == free
    capped = run_stage(replace(base, htm_max_celsius=95.0), _loop_inputs())
    assert capped.binding == "medium_limit"
    assert capped.htm_hot_celsius == pytest.approx(95.0)
    assert 0.0 < capped.q_w < free.q_w
    loose = run_stage(replace(base, htm_max_celsius=free.htm_hot_celsius + 1.0), _loop_inputs())
    assert loose == free


# ---------------------------------------------------------------------- the line
def _line_inputs(d: SiDryer = HOT_DRYER) -> LineInputs:
    return LineInputs(d, 15.0, MATERIAL, 40 / 3600, 4185.0)


def test_partner_line_closes_every_balance() -> None:
    r = Line(LineMode.partner()).solve(_line_inputs())
    s = r.stage
    cg = capacity(_line_inputs().fresh_gas, PROPS)
    a = capacity(r.dryer.exhaust, PROPS)
    b = 40 / 3600 * 4185.0
    assert s.binding == "equal_outlet"
    assert s.q_w == pytest.approx(cg * (s.gas_out_celsius - 15.0), rel=1e-12)
    assert s.q_w == pytest.approx(b * (r.medium_hot_celsius - r.medium_cold_celsius), rel=1e-12)
    assert s.q_w == pytest.approx(a * (s.exhaust_in_celsius - s.exhaust_out_celsius), rel=1e-12)
    assert r.medium_hot_celsius == pytest.approx(s.exhaust_out_celsius)
    assert r.medium_cold_celsius == pytest.approx(s.gas_out_celsius)
    duty = cg * (HOT_DRYER.air_temperature_celsius - s.gas_out_celsius)
    assert r.burner.duty_w == pytest.approx(duty, rel=1e-12)
    assert r.fuel_kg_per_h == pytest.approx(duty / (0.8 * 50e6) * 3600, rel=1e-12)
    assert r.product_celsius == r.dryer.exhaust.temperature_celsius
    assert r.product_kg_per_h == pytest.approx(8 * 0.4 * 1.002)


def test_physical_line_recovers_more_and_the_findings_follow_the_medium() -> None:
    partner = Line(LineMode.partner()).solve(_line_inputs())
    pressurised = Line(LineMode.physical()).solve(_line_inputs())
    unpressurised = Line(LineMode.physical(htm_max_celsius=95.0)).solve(_line_inputs())
    assert pressurised.recovered_kw >= unpressurised.recovered_kw >= partner.recovered_kw
    assert pressurised.fuel_kg_per_h <= unpressurised.fuel_kg_per_h <= partner.fuel_kg_per_h
    assert unpressurised.medium_hot_celsius <= 95.0 + 1e-9
    medium = {f.check: f for f in pressurised.findings()}["medium_boiling"]
    assert (medium.severity == "warning") == (pressurised.medium_hot_celsius > 99.974)
    assert all(f.ok for f in unpressurised.findings())


def test_physical_line_with_a_wet_dryer_passes_check_a1_by_construction() -> None:
    partner = {
        f.check: f for f in Line(LineMode.partner()).solve(_line_inputs(WET_DRYER)).findings()
    }
    physical = {
        f.check: f for f in Line(LineMode.physical()).solve(_line_inputs(WET_DRYER)).findings()
    }
    assert partner["exhaust_saturation"].severity == "violation"
    assert physical["exhaust_saturation"].ok


def test_a_medium_limit_needs_the_counter_current_exchanger() -> None:
    with pytest.raises(ValueError, match="counter-current"):
        LineMode("x", "partner", False, 10.0, 95.0)


# ---------------------------------------------------------------------- reader helper
@pytest.mark.parametrize(
    ("value", "decimals"), [(4.81, 2), (136.0, 0), (0.5, 1), (136.34834132426616, None)]
)
def test_reported_decimals(value: float, decimals: int | None) -> None:
    assert reported_decimals(value) == decimals
