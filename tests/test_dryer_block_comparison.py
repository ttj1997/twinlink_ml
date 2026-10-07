"""Dryer block of the equation document against the fluidised-bed model (docs/11, section 9)."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import dryer_block_comparison as dbc
from twinlink_ml.partner_v4.corrected import dryer_balance
from twinlink_ml.partner_v4.inputs import SiDryer, SiMaterial
from twinlink_ml.partner_v4_1.dryer_block import (
    BlockInputs,
    BlockProperties,
    dryer_block,
    heater_power_w,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.greybox import GreyBoxFluidBedGranulator
from twinlink_ml.simulation.operating_point import build_variant_config
from twinlink_ml.simulation.steady_state import solve_steady_state
from twinlink_ml.simulation.thermo import dew_point_celsius

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)


@pytest.fixture(scope="module")
def config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


def _solve(config, t_in=90.0, u=1.6, t_amb=20.0, phi=0.5):
    variant = build_variant_config(
        config, t_in=t_in, u_gas=u, p_nozzle=1.0, t_amb=t_amb, phi_amb=phi
    )
    solution = solve_steady_state(
        GreyBoxFluidBedGranulator(variant), max_iterations=30, tolerance=1e-7
    )
    return variant, solution


def _partner_dryer(air_kg_h, t_in, spray_kg_h) -> BlockInputs:
    return BlockInputs(
        dry_air_mass_flow_kg_per_s=air_kg_h / 3600.0,
        air_humidity_kg_per_kg=0.006,
        air_temperature_celsius=t_in,
        spray_mass_flow_kg_per_s=spray_kg_h / 3600.0,
        spray_solid_fraction_kg_per_kg=0.4,
        spray_temperature_celsius=15.0,
        product_moisture_kg_per_kg=dbc.PARTNER_EQUILIBRIUM_MOISTURE_KG_PER_KG,
        ambient_temperature_celsius=15.0,
    )


def test_block_as_delivered_equals_the_partner_rebuild_of_equations_1_3_and_1_5():
    props = dbc.partner_properties()
    material = SiMaterial(
        cp_gas_j_per_kg_k=1200.0,
        lower_heating_value_j_per_kg=45e6,
        heater_efficiency=0.8,
        cp_liquid_j_per_kg_k=4200.0,
        cp_solid_j_per_kg_k=1000.0,
        cp_vapour_j_per_kg_k=2000.0,
        equilibrium_moisture_kg_per_kg=0.002,
        enthalpy_of_evaporation_j_per_kg=2.5e6,
    )
    for air, t_in, spray, xs in [(50, 300, 5, 0.4), (100, 150, 6, 0.4), (100, 300, 12, 0.35)]:
        inputs = dataclasses.replace(
            _partner_dryer(air, t_in, spray), spray_solid_fraction_kg_per_kg=xs
        )
        reference = dryer_balance(
            SiDryer(air / 3600.0, 0.006, float(t_in), spray / 3600.0, xs, 15.0), material
        )
        block = dryer_block(inputs, props)
        assert block.outlet_temperature_celsius == pytest.approx(
            reference.outlet_temperature_celsius, rel=1e-12
        )
        assert block.outlet_humidity_kg_per_kg == pytest.approx(
            reference.outlet_humidity_kg_per_kg, rel=1e-12
        )


def test_block_reproduces_the_documented_exhausts_of_model_v4_1():
    props = dbc.partner_properties()
    dryer_1 = dryer_block(_partner_dryer(100, 150.0, 6), props)
    dryer_2 = dryer_block(_partner_dryer(100, 300.0, 12), props)
    assert dryer_1.outlet_temperature_celsius == pytest.approx(72.28, abs=0.005)
    assert dryer_1.outlet_humidity_kg_per_kg * 1e3 == pytest.approx(41.95, abs=0.005)
    assert dryer_2.outlet_temperature_celsius == pytest.approx(134.66, abs=0.005)
    assert dryer_2.outlet_humidity_kg_per_kg * 1e3 == pytest.approx(77.90, abs=0.005)


def test_heater_power_hand_value():
    inputs = _partner_dryer(50, 300.0, 5)
    # 50/3600 kg/s * (1200 + 0.006*2000) J/(kg K) * 285 K
    expected = 50 / 3600 * 1212.0 * 285.0
    assert heater_power_w(inputs, dbc.partner_properties()) == pytest.approx(expected, rel=1e-12)


def test_bed_model_and_block_are_the_same_balance_when_the_properties_are_the_same(config):
    variant, solution = _solve(config)
    own_props = dbc.own_properties(variant)
    own_inputs = dbc.block_inputs(variant, solution)
    inputs, props = dbc.apply_steps((*dbc.STEP_ORDER, "e_latent_slope"), own_inputs, own_props)
    block = dryer_block(inputs, props)
    assert block.outlet_temperature_celsius == pytest.approx(
        solution.bed_temperature_celsius, abs=1e-3
    )
    # the block closes the mass balance exactly; the bed's reported exhaust humidity carries the
    # lag of its damped fixed-point iteration (5 steps), so it is compared to the evaporation
    assert block.evaporation_kg_per_s == pytest.approx(
        solution.diagnostics["evaporation_rate_kg_per_s"], rel=1e-5
    )
    assert abs(block.outlet_humidity_kg_per_kg - solution.outlet_humidity_kg_per_kg) < 1e-5
    assert block.heater_power_w == pytest.approx(
        solution.diagnostics["power_heater_kw"] * 1000.0 * variant.energy.heater_efficiency,
        rel=1e-9,
    )


def test_own_properties_are_read_off_the_model(config):
    props = dbc.own_properties(config)
    assert props.latent_heat_j_per_kg(0.0) == pytest.approx(2.5008e6)
    assert props.latent_heat_j_per_kg(40.0) == pytest.approx(2.5008e6 - 2358.6 * 40.0)
    assert props.heat_loss_ua_w_per_k == config.geometry.heat_loss_ua_w_per_k


def test_final_state_does_not_depend_on_the_order_of_the_steps(config):
    variant, solution = _solve(config, t_in=75.0, u=1.4, t_amb=10.0, phi=0.7)
    own_props = dbc.own_properties(variant)
    own_inputs = dbc.block_inputs(variant, solution)
    forward = dbc.apply_steps(dbc.STEP_ORDER, own_inputs, own_props)
    reverse = dbc.apply_steps(dbc.STEP_ORDER_REVERSED, own_inputs, own_props)
    assert forward == reverse
    assert dryer_block(*forward) == dryer_block(*reverse)


def test_no_step_is_the_block_as_delivered(config):
    variant, solution = _solve(config)
    own_inputs = dbc.block_inputs(variant, solution)
    inputs, props = dbc.apply_steps((), own_inputs, dbc.own_properties(variant))
    assert props == dbc.partner_properties()
    assert inputs.product_moisture_kg_per_kg == dbc.PARTNER_EQUILIBRIUM_MOISTURE_KG_PER_KG
    assert inputs.dry_air_mass_flow_kg_per_s == own_inputs.dry_air_mass_flow_kg_per_s


def test_unknown_step_is_refused(config):
    variant, solution = _solve(config)
    with pytest.raises(ValueError, match="unknown step"):
        dbc.apply_steps(
            ("nonsense",), dbc.block_inputs(variant, solution), dbc.own_properties(variant)
        )


def test_recovery_sensible_by_hand_and_no_condensation_above_the_dew_point(config):
    props = dbc.own_properties(config)
    recovery = dbc.exhaust_recovery(
        config.gas,
        props,
        dry_air_mass_flow_kg_per_s=0.025,
        outlet_humidity_kg_per_kg=0.012,
        outlet_temperature_celsius=40.0,
        cold_temperature_celsius=29.1,
    )
    # dew point of 12 g/kg is about 17 degC: nothing condenses at 29.1 degC
    assert recovery.condensate_kg_per_s == 0.0
    assert recovery.latent_w == 0.0
    assert recovery.sensible_w == pytest.approx(0.025 * (1006.0 + 0.012 * 1860.0) * 10.9)


def test_recovery_is_continuous_at_the_dew_point_and_grows_when_the_stream_is_cooled_more(config):
    props = dbc.own_properties(config)
    dew = dew_point_celsius(NumpyBackend(), config.gas, np.array(0.012))
    kwargs = {
        "dry_air_mass_flow_kg_per_s": 0.025,
        "outlet_humidity_kg_per_kg": 0.012,
        "outlet_temperature_celsius": 40.0,
    }
    above = dbc.exhaust_recovery(
        config.gas, props, cold_temperature_celsius=float(dew) + 1e-6, **kwargs
    )
    below = dbc.exhaust_recovery(
        config.gas, props, cold_temperature_celsius=float(dew) - 1e-6, **kwargs
    )
    assert above.total_w == pytest.approx(below.total_w, rel=1e-3)
    totals = [
        dbc.exhaust_recovery(config.gas, props, cold_temperature_celsius=t, **kwargs).total_w
        for t in (30.0, 20.0, 15.0, 10.0)
    ]
    assert totals == sorted(totals)


def test_recovery_of_a_stream_colder_than_t_cold_is_zero_not_negative(config):
    recovery = dbc.exhaust_recovery(
        config.gas,
        dbc.own_properties(config),
        dry_air_mass_flow_kg_per_s=0.025,
        outlet_humidity_kg_per_kg=0.005,
        outlet_temperature_celsius=20.0,
        cold_temperature_celsius=29.1,
    )
    assert recovery.total_w == 0.0


def test_block_flags_a_supersaturated_exhaust(config):
    # 100 kg/h air, 60 kg/h spray at 40 % solid into 60 degC air: the exhaust cannot hold the water
    inputs = BlockInputs(
        dry_air_mass_flow_kg_per_s=100 / 3600,
        air_humidity_kg_per_kg=0.006,
        air_temperature_celsius=60.0,
        spray_mass_flow_kg_per_s=60 / 3600,
        spray_solid_fraction_kg_per_kg=0.4,
        spray_temperature_celsius=15.0,
        product_moisture_kg_per_kg=0.002,
        ambient_temperature_celsius=15.0,
    )
    flags = dbc.block_is_normal(config.gas, dryer_block(inputs, dbc.partner_properties()))
    assert not flags["unsaturated"]
    assert not flags["normal"]
    normal = dbc.block_is_normal(
        config.gas, dryer_block(_partner_dryer(100, 300.0, 12), dbc.partner_properties())
    )
    assert normal["normal"]


def test_block_properties_defaults_are_the_mixture_rule_and_cp_v_minus_cp_l():
    props = BlockProperties(1000.0, 1800.0, 4000.0, 1000.0, 2.5e6)
    assert props.suspension_cp(0.25) == pytest.approx(0.75 * 4000.0 + 0.25 * 1000.0)
    assert props.latent_heat_slope == pytest.approx(1800.0 - 4000.0)
