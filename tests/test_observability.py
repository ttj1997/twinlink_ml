"""Tests of the observability study (docs/33_observability.md)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from twinlink_ml.headroom.config import load_headroom_config, study_process_config
from twinlink_ml.observability import (
    bed_temperature_shift,
    linear_fit,
    load_observability_config,
    operating_point_config,
    outlet_humidity_shift,
    outlet_humidity_shift_linearised,
    scale_solid_to_air_ratio,
    sensor_sigma,
    separation,
    snr,
    steady_measurements,
    step_responses,
)
from twinlink_ml.observability.config import StepSettings
from twinlink_ml.sensor_value.draws import truth_config
from twinlink_ml.simulation import MeasurementConfig, PlaceholderConfigWarning


@pytest.fixture(scope="module")
def nominal():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return study_process_config(load_headroom_config())


@pytest.fixture(scope="module")
def point(nominal):
    return operating_point_config(nominal, inlet_temperature_celsius=90.0, gas_velocity_m_per_s=0.8)


@pytest.fixture(scope="module")
def two_truths(point):
    """Same commands and disturbances, different critical moisture (slower drying)."""
    slow = truth_config(
        point, {"critical_moisture_kg_per_kg": 1.3 * point.material.critical_moisture_kg_per_kg}
    )
    return steady_measurements(point), steady_measurements(slow)


def test_analytic_outlet_humidity_shift_matches_simulation(two_truths):
    a, b = two_truths
    delta_x = b.moisture_kg_per_kg - a.moisture_kg_per_kg
    delta_t = b.bed_temperature_celsius - a.bed_temperature_celsius
    assert delta_x > 1e-3  # the slower kinetics leaves a wetter bed
    simulated = b.outlet_humidity_balance_kg_per_kg - a.outlet_humidity_balance_kg_per_kg
    assert simulated < 0.0
    # the hypothesis' expression at constant dry-air flow: right sign and order; it overstates
    # the shift because the warmer bed lowers the gas density and with it the dry-air flow
    hypothesis = outlet_humidity_shift(delta_x, a.solid_inflow_kg_per_s, a.dry_air_flow_kg_per_s)
    assert simulated == pytest.approx(hypothesis, rel=0.15)
    assert abs(hypothesis) > abs(simulated)
    # with the density term the linearisation agrees closely
    linearised = outlet_humidity_shift_linearised(delta_x, delta_t, a)
    assert simulated == pytest.approx(linearised, rel=3e-2)  # linearisation over Delta X 0.03
    # exact water balance of every steady state with its own dry-air flow (balance humidity);
    # the model's diagnostic differs from it by the Picard residual, below 1 % here
    for s in (a, b):
        exact = (
            s.inlet_humidity_kg_per_kg
            + (s.water_feed_kg_per_s - s.solid_inflow_kg_per_s * s.moisture_kg_per_kg)
            / s.dry_air_flow_kg_per_s
        )
        assert s.outlet_humidity_balance_kg_per_kg == pytest.approx(exact, rel=1e-8)
        assert s.outlet_humidity_kg_per_kg == pytest.approx(exact, rel=1e-2)


def test_analytic_bed_temperature_shift_matches_simulation(two_truths, point):
    a, b = two_truths
    delta_x = b.moisture_kg_per_kg - a.moisture_kg_per_kg
    simulated = b.bed_temperature_celsius - a.bed_temperature_celsius
    assert simulated > 0.0  # less evaporation, warmer bed
    assert simulated == pytest.approx(bed_temperature_shift(delta_x, a, point), rel=2e-2)


def test_snr_and_sensor_sigma_known_values():
    assert snr(0.008, 0.01, 8e-5) == pytest.approx(1.0)
    assert snr(-0.008, 0.01, 8e-5, redundancy=6) == pytest.approx(math.sqrt(6.0))
    m = MeasurementConfig(relative_noise_std=0.005, absolute_noise_std_temperature_k=0.2)
    assert float(sensor_sigma(40.0, m, temperature=True)) == pytest.approx(math.sqrt(0.08))
    assert float(sensor_sigma(0.02, m, temperature=False)) == pytest.approx(1e-4)
    with pytest.raises(ValueError):
        snr(1.0, 0.01, 0.0)


def test_linear_fit_and_separation():
    fit = linear_fit([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert (fit.slope, fit.intercept, fit.r2, fit.n) == pytest.approx((2.0, 1.0, 1.0, 3))
    t = np.arange(1, 6) * 5.0
    sep = separation(np.array([0, 1, 3, 1, 0.0]), np.zeros(5), np.ones(5), 2.0, 5.0, t)
    assert sep.max_ratio == 3.0 and sep.time_of_max_ratio_s == 15.0
    assert sep.time_above_s == 5.0 and sep.first_above_s == sep.last_above_s == 15.0
    none = separation(np.zeros(5), np.zeros(5), np.ones(5), 2.0, 5.0, t)
    assert none.time_above_s == 0.0 and none.first_above_s is None


def test_ratio_scaling_keeps_water_feed_and_air(point):
    base = steady_measurements(point)
    scaled, info = scale_solid_to_air_ratio(point, 0.02, iterations=6)
    assert info["achieved_ratio"] == pytest.approx(0.02, rel=1e-3)
    assert info["water_feed_kg_per_h"] == pytest.approx(base.water_feed_kg_per_s * 3600.0)
    assert scaled.actions.gas_velocity_m_per_s.nominal == point.actions.gas_velocity_m_per_s.nominal
    assert (
        scaled.actions.inlet_temperature_celsius.nominal
        == point.actions.inlet_temperature_celsius.nominal
    )
    assert scaled.material == point.material and scaled.geometry == point.geometry


def test_step_response_reference_stays_steady(point):
    s = steady_measurements(point)
    steps = (StepSettings("inlet_plus", "inlet_temperature", 1.1),)
    out = step_responses(point, s.state, steps, horizon_s=600.0)
    assert out["moisture_kg_per_kg"].shape == (2, 120)
    assert np.allclose(out["moisture_kg_per_kg"][0], s.moisture_kg_per_kg, atol=1e-8)
    assert np.allclose(out["bed_temperature_celsius"][0], s.bed_temperature_celsius, atol=1e-6)
    assert out["bed_temperature_celsius"][1, -1] > s.bed_temperature_celsius + 0.5


def test_config_loads_and_plant_ratio():
    config = load_observability_config()
    ratios = config.plant_like.ratios()
    assert ratios[0] == pytest.approx(500.0 / (30000.0 * 1.2922))
    assert ratios[1] == pytest.approx(2.0 * ratios[0])
    assert [s.name for s in config.dynamics.steps] == [
        "spray_plus",
        "spray_minus",
        "inlet_plus",
        "inlet_minus",
    ]
