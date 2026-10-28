"""Tests of the plant-near study (docs/NN_plant_scale.md). Synthetic data only, no plant data."""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings

import numpy as np
import pytest

from twinlink_ml.headroom.config import load_headroom_config, study_process_config
from twinlink_ml.observability import steady_measurements
from twinlink_ml.plant_a_op.config import DryerPlaceholders
from twinlink_ml.plant_a_op.data import DayState
from twinlink_ml.plant_a_op.dryer import loss_conductance_kw_per_k
from twinlink_ml.plant_scale import (
    Disturbance,
    Freeboard,
    LabFactory,
    PlantParameters,
    PlantPoint,
    SearchBox,
    SteadySearch,
    block_statistics,
    box_points,
    calibrate,
    day_series_from_columns,
    gap_balance,
    load_plant_scale_config,
    overall_balance,
    plant_config,
    rolling_std,
    stationary_segments,
    steady_plant,
)
from twinlink_ml.plant_scale.config import StationaritySettings
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config


@pytest.fixture(scope="module")
def study():
    return load_plant_scale_config()


@pytest.fixture(scope="module")
def base(study):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(study.process_config)


# Synthetic operating point (invented round numbers of plant-like size, not plant data).
POINT = PlantPoint(
    label="synthetic",
    ambient_celsius=10.0,
    process_air_nm3h=30000.0,
    atomising_air_nm3h=600.0,
    supply_celsius=85.0,
    spray_kg_h=1500.0,
    bed_celsius=60.0,
    exhaust_celsius=52.0,
    bed_group_a_celsius=59.0,
    bed_group_b_celsius=61.0,
    bed_sensor_min_celsius=58.5,
    bed_sensor_max_celsius=61.5,
    downstream_celsius=51.0,
)


def steady_kwargs(study):
    return {
        "normal_density_kg_per_m3": study.data.normal_density_dry_air_kg_per_m3,
        "inlet_humidity_kg_per_kg": study.placeholders.inlet_absolute_humidity_kg_per_kg,
        "feed_celsius": study.placeholders.feed_temperature_celsius,
        "iterations": study.calibration.velocity_iterations,
        "relative_tolerance": study.calibration.velocity_relative_tolerance,
    }


# ------------------------------------------------------------------------------------ config


def test_config_loads_and_is_consistent(study, base):
    assert study.calibration.candidates["C0_none"] == "none"
    assert study.headroom.lab_reference_margins_kg_per_kg == (0.04355, 0.00664)
    assert len(study.fingerprint) == 64
    # the plant process config keeps the lab line's kinetics and sensor model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        lab = load_process_config("conf/process/greybox_gab_placeholder.yaml")
    assert base.material.critical_moisture_kg_per_kg == lab.material.critical_moisture_kg_per_kg
    assert base.material.max_saturation_degree == lab.material.max_saturation_degree
    assert base.measurement == lab.measurement
    assert base.exhaust.outlet_duct_heat_loss_k == 0.0


# --------------------------------------------------------------------------------- data


def test_rolling_std_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50).cumsum()
    std, mean = rolling_std(x, 7)
    for i in (0, 13, 43):
        assert std[i] == pytest.approx(np.std(x[i : i + 7]), abs=1e-10)
        assert mean[i] == pytest.approx(np.mean(x[i : i + 7]), abs=1e-12)
    assert len(std) == 44


def _synthetic_day(n=600, seed=1):
    """Flat stretches with noise, a ramp in the bed temperature between minutes 200 and 320."""
    rng = np.random.default_rng(seed)
    bed = 60.0 + 0.05 * rng.normal(size=n)
    bed[200:320] += np.linspace(0.0, 6.0, 120)
    bed[320:] += 6.0
    columns = {
        "ambient_temperature": 10.0 + 0.01 * rng.normal(size=n),
        "process_air_flow": 30000.0 * (1.0 + 0.0005 * rng.normal(size=n)),
        "supply_air_temperature": 85.0 + 0.05 * rng.normal(size=n),
        "exhaust_outlet_temperature": bed - 8.0,
        "spray_rate": 1500.0 * (1.0 + 0.005 * rng.normal(size=n)),
        "atomising_air_flow": 600.0 + 0.1 * rng.normal(size=n),
    }
    groups = {
        "bed_temperature_group_a": np.vstack([bed - 1.0] * 3),
        "bed_temperature_group_b": np.vstack([bed + 1.0] * 3),
    }
    return day_series_from_columns("synthetic", columns, groups)


def test_stationary_segments_avoid_the_ramp_and_keep_the_gap():
    day = _synthetic_day()
    assert day["bed_temperature"] == pytest.approx(day["bed_temperature_group_a"] + 1.0)
    settings = StationaritySettings(
        window_minutes=60,
        thresholds={
            "bed_temperature_k": 0.3,
            "exhaust_outlet_temperature_k": 0.3,
            "supply_air_temperature_k": 0.3,
            "process_air_flow_relative": 0.005,
            "spray_rate_relative": 0.03,
        },
        segments_min=2,
        segments_max=4,
        minimum_gap_minutes=60,
    )
    segments = stationary_segments(day, settings)
    assert 2 <= len(segments) <= 4
    for s in segments:
        assert s.admissible
        assert s.stop <= 200 or s.start >= 320  # no window over the ramp
    starts = [s.start for s in segments]
    assert all(b - a >= 120 for a, b in itertools.pairwise(starts))


def test_stationary_segments_flag_when_criterion_not_met():
    day = _synthetic_day()
    settings = StationaritySettings(
        window_minutes=60,
        thresholds={"bed_temperature_k": 1e-6},
        segments_min=2,
        segments_max=4,
        minimum_gap_minutes=0,
    )
    segments = stationary_segments(day, settings)
    assert len(segments) == 2
    assert not any(s.admissible for s in segments)


def test_block_statistics():
    x = np.concatenate([np.full(15, 1.0), np.full(15, 3.0), [99.0]])
    stats = block_statistics(x, 15)
    assert stats["blocks"] == 2
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["relative_std"] == pytest.approx(0.5)
    assert stats["half_range"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------ balance


def test_gap_balance_candidates_close_the_gap(study):
    ph = study.placeholders
    rho = study.data.normal_density_dry_air_kg_per_m3
    out = gap_balance(POINT, ph, solid_fraction=0.65, normal_density_kg_per_m3=rho)
    assert out["gap_k"] == pytest.approx(8.0)
    # heat loss: the duct formula with the required UA returns the measured exhaust temperature
    m, gas_c = out["gas_flow_kg_per_s"], out["q_gap_kw"] * 1000.0 / (out["gas_flow_kg_per_s"] * 8.0)
    for name in ("ambient", "room"):
        t_s = out["heat_loss"][name]["surroundings_celsius"]
        ua = out["heat_loss"][name]["ua_w_per_k"]
        t_exh = t_s + (POINT.bed_celsius - t_s) * math.exp(-ua / (m * gas_c))
        assert t_exh == pytest.approx(POINT.exhaust_celsius, abs=1e-9)
        # false air: mixing (1 - r) bed gas at T_bed with r at T_s gives T_exh (per unit of gas)
        r = out["false_air"][name]["leak_share_of_gas"]
        assert (POINT.bed_celsius + r * t_s) / (1.0 + r) == pytest.approx(POINT.exhaust_celsius)
    assert out["coldest_bed_sensor_minus_exhaust_k"] == pytest.approx(6.5)
    assert out["bed_sensor_spread_k"] == pytest.approx(3.0)
    assert 0.0 < out["evaporation_share_of_water"] < 1.0


def test_overall_balance_uses_the_dryer_balance_of_docs23(study):
    ph = study.placeholders
    rho = study.data.normal_density_dry_air_kg_per_m3
    out = overall_balance(POINT, ph, solid_fraction=0.65, normal_density_kg_per_m3=rho)
    dryer = DryerPlaceholders(
        solid_fraction=0.65,
        intake_relative_humidity=0.0,
        feed_temperature_celsius=ph.feed_temperature_celsius,
        atomising_air_temperature_celsius=ph.atomising_air_temperature_celsius,
        room_temperature_celsius=ph.room_temperature_celsius,
        normal_density_kg_per_nm3=rho,
    )
    state = DayState("x", 10.0, 30000.0, 600.0, 85.0, 52.0, 1500.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert out["ua_outlet_exhaust_kw_per_k"] == loss_conductance_kw_per_k(dryer, state)
    # a warmer outlet releases less heat from the air: smaller conductance
    assert out["ua_outlet_bed_kw_per_k"] < out["ua_outlet_exhaust_kw_per_k"]


# -------------------------------------------------------------------------------- model


def test_freeboard_validation():
    with pytest.raises(ValueError):
        Freeboard("other", 1.0)
    with pytest.raises(ValueError):
        Freeboard("none", 1.0)
    with pytest.raises(ValueError):
        Freeboard("evaporation", 1.0)
    with pytest.raises(ValueError):
        Freeboard("heat_loss", -1.0)


def test_switch_off_is_bit_identical_to_the_unchanged_model(study, base):
    """Mode none: exactly the unchanged model's steady state, exhaust = bed temperature."""
    params = PlantParameters(800.0, 0.65)
    kw = steady_kwargs(study)
    s = steady_plant(base, POINT, params, **kw)
    config = plant_config(
        base,
        POINT,
        params,
        gas_velocity_m_per_s=s.gas_velocity_m_per_s,
        inlet_humidity_kg_per_kg=kw["inlet_humidity_kg_per_kg"],
    )
    direct = steady_measurements(config)
    assert s.bed.bed_temperature_celsius == direct.bed_temperature_celsius
    assert s.bed.moisture_kg_per_kg == direct.moisture_kg_per_kg
    assert s.exhaust_temperature_celsius == s.bed_temperature_celsius
    assert s.outlet_humidity_kg_per_kg == direct.outlet_humidity_kg_per_kg
    # the config differs from the YAML only in the nominal values and UA (no structural switch)
    assert config.exhaust == base.exhaust
    assert config.material == base.material
    assert config.geometry == dataclasses.replace(base.geometry, heat_loss_ua_w_per_k=800.0)
    # zero-valued switched modes reproduce mode none (bed bit for bit, exhaust to rounding)
    for mode in ("offset", "heat_loss", "evaporation"):
        other = steady_plant(base, POINT, PlantParameters(800.0, 0.65, Freeboard(mode, 0.0)), **kw)
        assert other.bed_temperature_celsius == s.bed_temperature_celsius
        assert other.exhaust_temperature_celsius == pytest.approx(
            s.exhaust_temperature_celsius, abs=1e-12
        )
    # dry-air flow matches the measured process air
    target = POINT.process_air_nm3h * kw["normal_density_kg_per_m3"] / 3600.0
    assert s.bed.dry_air_flow_kg_per_s == pytest.approx(target, rel=1e-8)


def test_switched_modes_cool_the_exhaust_consistently(study, base):
    kw = steady_kwargs(study)
    off = steady_plant(base, POINT, PlantParameters(800.0, 0.65), **kw)
    offset = steady_plant(base, POINT, PlantParameters(800.0, 0.65, Freeboard("offset", 3.0)), **kw)
    assert offset.bed_temperature_celsius == off.bed_temperature_celsius
    assert offset.exhaust_temperature_celsius == pytest.approx(off.bed_temperature_celsius - 3.0)
    loss = steady_plant(
        base, POINT, PlantParameters(800.0, 0.65, Freeboard("heat_loss", 2000.0)), **kw
    )
    assert loss.bed_temperature_celsius == off.bed_temperature_celsius
    assert loss.exhaust_temperature_celsius < off.exhaust_temperature_celsius
    evap = steady_plant(
        base, POINT, PlantParameters(800.0, 0.65, Freeboard("evaporation", 0.3)), **kw
    )
    # less water in the bed: warmer bed; the freeboard water cools the gas
    assert evap.bed_temperature_celsius > off.bed_temperature_celsius
    assert evap.exhaust_temperature_celsius < evap.bed_temperature_celsius
    # energy balance of the freeboard: gas heat = water heated and evaporated
    m = evap.bed.dry_air_flow_kg_per_s
    gas = base.gas
    c_h = (
        gas.cp_dry_air_j_per_kg_k
        + evap.bed.outlet_humidity_balance_kg_per_kg * gas.cp_vapour_j_per_kg_k
    )
    w = evap.freeboard_water_kg_per_s
    from twinlink_ml.plant_scale.model import _R0_J_PER_KG

    lhs = m * c_h * (evap.bed_temperature_celsius - evap.exhaust_temperature_celsius)
    rhs = w * (
        _R0_J_PER_KG
        + gas.cp_vapour_j_per_kg_k * evap.exhaust_temperature_celsius
        - base.material.cp_water_liquid_j_per_kg_k * kw["feed_celsius"]
    )
    assert lhs == pytest.approx(rhs, rel=1e-12)
    assert w == pytest.approx(0.3 * 1500.0 * 0.35 / 3600.0)


def test_calibration_recovers_synthetic_parameters(study, base):
    """Truth generated by the model itself (UA 600 W/K, offset 4 K); the fit recovers both."""
    kw = steady_kwargs(study)
    truth = PlantParameters(600.0, 0.65, Freeboard("offset", 4.0))
    s = steady_plant(base, POINT, truth, **kw)
    point = dataclasses.replace(
        POINT, bed_celsius=s.bed_temperature_celsius, exhaust_celsius=s.exhaust_temperature_celsius
    )
    fit = calibrate(
        base,
        [point],
        "C3_offset",
        "offset",
        solid_fraction=0.65,
        bounds=study.calibration.bounds,
        start=study.calibration.start,
        steady_kwargs=kw,
    )
    assert fit.parameters.heat_loss_ua_w_per_k == pytest.approx(600.0, rel=1e-3)
    assert fit.parameters.freeboard.value == pytest.approx(4.0, abs=1e-3)
    assert max(abs(r) for r in fit.residuals_k) < 1e-4


# ----------------------------------------------------------------------------- headroom


def test_box_points():
    corners, points = box_points(Disturbance(100.0, 0.5, 10.0), 0.1, 0.2, 3.0)
    assert len(corners) == 8 and len(points) == 9
    assert points[-1] == Disturbance(100.0, 0.5, 10.0)
    assert sorted({round(c.spray_kg_h, 9) for c in corners}) == [90.0, 110.0]
    assert sorted({round(c.ambient_celsius, 9) for c in corners}) == [7.0, 13.0]


def test_inlet_for_limit_brackets_the_moisture_limit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        lab = study_process_config(load_headroom_config())
    box = SearchBox((60.0, 120.0), (0.8, 2.5), 80.0, 5, 0.01, 1e-4)
    search = SteadySearch(LabFactory(lab), box)
    d = Disturbance(1.5, 0.30, 20.0)
    limit = 0.12
    t = search.inlet_for_limit(0.8, d, limit)
    assert t is not None
    assert search.evaluate(t, 0.8, d).moisture_kg_per_kg <= limit
    assert search.evaluate(t - 0.02, 0.8, d).moisture_kg_per_kg > limit
    assert search.inlet_for_limit(0.8, d, 1e-4) is None  # below equilibrium: never admissible


# ----------------------------------------------------------------------------- redaction


def _redaction_module():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts" / "verify"))
    spec = importlib.util.spec_from_file_location(
        "redact_plant_scale", root / "scripts" / "redact_plant_scale.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_redaction_removes_only_the_per_segment_role_values():
    module = _redaction_module()
    segment = {
        "gap_k": 7.0,
        "gas_flow_kg_per_s": 10.0,
        "heat_loss": {"ambient": {"surroundings_celsius": 5.0, "ua_w_per_k": 1.0}},
        "false_air": {"room": {"surroundings_celsius": 20.0, "leak_share_of_gas": 0.1}},
    }
    record = {
        "survey": {"gas_flow_kg_per_s": 1.0},
        "balances": {"per_segment": {"day_1": [{"0.65": segment}]}, "overall": {"x": 1}},
    }
    redacted, count = module.redact_record(record)
    kept = redacted["balances"]["per_segment"]["day_1"][0]["0.65"]
    assert count == 3
    assert kept == {
        "gap_k": 7.0,
        "heat_loss": {"ambient": {"ua_w_per_k": 1.0}},
        "false_air": {"room": {"leak_share_of_gas": 0.1}},
    }
    assert redacted["survey"] == {"gas_flow_kg_per_s": 1.0}  # outside the scope: unchanged
    assert record["balances"]["per_segment"]["day_1"][0]["0.65"] is segment  # input untouched
    assert "gas_flow_kg_per_s" in segment
