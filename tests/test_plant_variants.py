"""Plant variants (study A1): P0, P1, P2a, P2b, P3, PP, the drying-curve option, the generator."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation.experiments import DataSpec, generate_data
from twinlink_ml.evaluation.plant_data import generate_plant_data
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    ProcessConfig,
    load_process_config,
)
from twinlink_ml.simulation.plant_variants import (
    PLANT_VARIANTS,
    FoulingSpec,
    PlantVariantSimulator,
    PlantVariantSpec,
    SensorSpec,
    make_plant,
    shifted_config,
)
from twinlink_ml.simulation.steady_state import solve_steady_state

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
SAMPLE_TIME_S = 25.0


@pytest.fixture(scope="module")
def config() -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        loaded = load_process_config(CONFIG_PATH)
    return dataclasses.replace(
        loaded, numerics=dataclasses.replace(loaded.numerics, sample_time_s=SAMPLE_TIME_S)
    )


def _nominal_inputs(simulator: GreyBoxFluidBedGranulator, n_steps: int):
    actions = np.tile(simulator.nominal_action(1)[:, None, :], (1, n_steps, 1))
    disturbances = np.tile(simulator.nominal_disturbance(1)[:, None, :], (1, n_steps, 1))
    return actions, disturbances


def _quiet(config: ProcessConfig) -> ProcessConfig:
    """Sensor path without noise (lag, dead time and offset stay)."""
    return dataclasses.replace(
        config,
        measurement=dataclasses.replace(
            config.measurement, relative_noise_std=0.0, absolute_noise_std_temperature_k=0.0
        ),
    )


# ------------------------------------------------------------ drying-curve option


def test_drying_curve_exponent_enters_fingerprint_only_when_set(config: ProcessConfig) -> None:
    unset = config.fingerprint
    assert config.material.drying_curve_exponent is None
    with_exponent = dataclasses.replace(
        config, material=dataclasses.replace(config.material, drying_curve_exponent=2.0)
    )
    assert with_exponent.fingerprint != unset
    assert "drying_curve_exponent" not in config._fingerprint_payload()["material"]
    assert "drying_curve_exponent" in with_exponent._fingerprint_payload()["material"]


def test_drying_curve_exponent_must_be_positive(config: ProcessConfig) -> None:
    with pytest.raises(ValueError, match="drying_curve_exponent"):
        dataclasses.replace(config.material, drying_curve_exponent=0.0)


def test_exponent_one_is_the_linear_curve_exactly(config: ProcessConfig) -> None:
    linear = GreyBoxFluidBedGranulator(config)
    one = GreyBoxFluidBedGranulator(
        dataclasses.replace(
            config, material=dataclasses.replace(config.material, drying_curve_exponent=1.0)
        )
    )
    actions, disturbances = _nominal_inputs(linear, 60)
    a = linear.rollout(actions, disturbances, seed=3)
    b = one.rollout(actions, disturbances, seed=3)
    assert np.array_equal(a.outputs, b.outputs)


def test_exponent_reaches_the_steady_state_solver(config: ProcessConfig) -> None:
    """Every consumer of the kinetics sees the option, not only the rollout."""
    squared = dataclasses.replace(
        config, material=dataclasses.replace(config.material, drying_curve_exponent=2.0)
    )
    base = GreyBoxFluidBedGranulator(config)
    other = GreyBoxFluidBedGranulator(squared)
    solution_base = solve_steady_state(base)
    solution_other = solve_steady_state(other)
    moisture_base = solution_base.state[1] / solution_base.state[0]
    moisture_other = solution_other.state[1] / solution_other.state[0]
    assert moisture_other > moisture_base


# ------------------------------------------------------------------------- P0


def test_p0_is_bit_identical_to_the_grey_box_model(config: ProcessConfig) -> None:
    base = GreyBoxFluidBedGranulator(config)
    plant = make_plant("P0", config)
    actions, disturbances = _nominal_inputs(base, 80)
    actions = actions * (1.0 + 0.05 * np.sin(np.arange(80))[None, :, None])
    a = base.rollout(actions, disturbances, seed=11)
    b = plant.rollout(actions, disturbances, seed=11)
    for name in ("states", "outputs", "measurements", "actions"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert np.array_equal(base.remeasure(a.outputs, seed=5), plant.remeasure(a.outputs, seed=5))
    assert plant.config.fingerprint == base.config.fingerprint


def test_default_sensor_spec_is_the_base_sensor_path(config: ProcessConfig) -> None:
    base = GreyBoxFluidBedGranulator(config)
    plant = PlantVariantSimulator(config, PlantVariantSpec(name="X", sensor=SensorSpec()))
    outputs = base.rollout(*_nominal_inputs(base, 50), seed=2).outputs
    assert np.array_equal(base.remeasure(outputs, seed=9), plant.remeasure(outputs, seed=9))


# ------------------------------------------------------------------------- P1


def test_p1_is_the_shifted_config_and_keeps_the_quality_limit(config: ProcessConfig) -> None:
    plant = make_plant("P1", config)
    base_material = config.material
    material = plant.config.material
    assert material.max_saturation_degree == pytest.approx(
        base_material.max_saturation_degree * 0.75
    )
    assert material.critical_moisture_kg_per_kg == pytest.approx(
        base_material.critical_moisture_kg_per_kg * 1.20
    )
    assert plant.config.geometry.heat_loss_ua_w_per_k == pytest.approx(
        config.geometry.heat_loss_ua_w_per_k * 1.30
    )
    # kinetics only: the quality limit is the base config's, not the shifted critical moisture
    assert (
        material.quality_moisture_limit_kg_per_kg == base_material.quality_moisture_limit_kg_per_kg
    )
    assert material.quality_moisture_limit_kg_per_kg != material.critical_moisture_kg_per_kg
    plain = GreyBoxFluidBedGranulator(plant.config)
    inputs = _nominal_inputs(plain, 60)
    assert np.array_equal(
        plant.rollout(*inputs, seed=4).outputs, plain.rollout(*inputs, seed=4).outputs
    )


def test_unshifted_critical_moisture_leaves_the_limit_field_unset(config: ProcessConfig) -> None:
    shifted = shifted_config(config, PLANT_VARIANTS["P2a"])
    assert shifted.material.product_moisture_limit_kg_per_kg is None


# ------------------------------------------------------------------------ P2a


def test_p2a_changes_only_the_water_and_temperature_balances(config: ProcessConfig) -> None:
    base = GreyBoxFluidBedGranulator(config)
    plant = make_plant("P2a", config)
    state = base.initial_state(1)
    action, disturbance = base.nominal_action(1), base.nominal_disturbance(1)
    d_base, _ = base.evaluate_derivative(state, action, disturbance)
    d_plant, _ = plant.evaluate_derivative(state, action, disturbance)
    # solid mass and particle number: closed balances, untouched by the drying curve
    assert np.array_equal(d_base[..., 0], d_plant[..., 0])
    assert np.array_equal(d_base[..., 2], d_plant[..., 2])
    assert not np.array_equal(d_base[..., 1], d_plant[..., 1])
    assert not np.array_equal(d_base[..., 3], d_plant[..., 3])


def test_p2a_keeps_the_curve_pinned_at_the_ends(config: ProcessConfig) -> None:
    """f(0) = 0 and f(1) = 1: no drying at equilibrium moisture, the old maximum at eta = 1."""
    base = GreyBoxFluidBedGranulator(config)
    plant = make_plant("P2a", config)
    state = base.initial_state(1).copy()
    action, disturbance = base.nominal_action(1), base.nominal_disturbance(1)
    material = config.material
    state[..., 1] = state[..., 0] * 10.0 * material.critical_moisture_kg_per_kg  # eta = 1
    _, d_base = base.evaluate_derivative(state, action, disturbance)
    _, d_plant = plant.evaluate_derivative(state, action, disturbance)
    assert np.allclose(
        d_base["evaporation_rate_kg_per_s"], d_plant["evaporation_rate_kg_per_s"], rtol=1e-12
    )


# ------------------------------------------------------------------------ P2b


def test_p2b_leaves_the_process_untouched_and_changes_the_measurement(
    config: ProcessConfig,
) -> None:
    base = GreyBoxFluidBedGranulator(config)
    plant = make_plant("P2b", config)
    inputs = _nominal_inputs(base, 60)
    a, b = base.rollout(*inputs, seed=1), plant.rollout(*inputs, seed=1)
    assert np.array_equal(a.states, b.states)
    assert np.array_equal(a.outputs, b.outputs)
    assert not np.array_equal(a.measurements, b.measurements)
    # channels without a sensor deviation are only different in the noise draws' scale: the
    # lag defaults are those of the base model, so the noise-free paths coincide
    quiet_base = GreyBoxFluidBedGranulator(_quiet(config))
    quiet_plant = PlantVariantSimulator(_quiet(config), PLANT_VARIANTS["P2b"])
    qa = quiet_base.remeasure(a.outputs, seed=0)
    qb = quiet_plant.remeasure(a.outputs, seed=0)
    assert np.array_equal(qa[..., [0, 3]], qb[..., [0, 3]])


def test_p2b_offset_and_dead_time(config: ProcessConfig) -> None:
    plant = PlantVariantSimulator(_quiet(config), PLANT_VARIANTS["P2b"])
    n_steps = 40
    outputs = np.zeros((1, n_steps, 4))
    outputs[..., 0] = 2.0
    outputs[..., 1] = 60.0
    outputs[..., 2] = 0.03
    outputs[..., 3] = 400.0
    step_at = 10
    outputs[:, step_at:, 1] = 70.0
    outputs[:, step_at:, 2] = 0.05
    measured = plant.remeasure(outputs, seed=0)

    sensor = PLANT_VARIANTS["P2b"].sensor
    assert sensor is not None
    offset = lambda t: 1.0 + 0.08 * (t - 20.0)  # noqa: E731
    # offset between bed and exhaust, from the very first sample (held buffer, no transient)
    assert measured[0, 0, 1] == pytest.approx(60.0 - offset(60.0))
    # dead time of the temperature channel: 50 s = 2 samples, nothing moves before
    assert np.allclose(measured[0, : step_at + 2, 1], 60.0 - offset(60.0))
    assert measured[0, step_at + 2, 1] > 60.0 - offset(60.0) + 1e-6
    # dead time of the moisture channel: 75 s = 3 samples
    assert np.allclose(measured[0, : step_at + 3, 2], 0.03)
    assert measured[0, step_at + 3, 2] > 0.03 + 1e-9
    # the temperature settles on the offset reading of the new level
    assert measured[0, -1, 1] == pytest.approx(70.0 - offset(70.0), abs=1e-2)


def test_dead_time_must_be_a_multiple_of_the_sample_time(config: ProcessConfig) -> None:
    spec = PlantVariantSpec(name="X", sensor=SensorSpec(dead_times_s=(0.0, 30.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="multiple of the sample time"):
        PlantVariantSimulator(config, spec)


# ------------------------------------------------------------------------- P3


def test_p3_fouling_equals_a_config_with_the_larger_heat_loss(config: ProcessConfig) -> None:
    fouling = FoulingSpec(heat_loss_growth=0.40)
    plant = PlantVariantSimulator(config, PlantVariantSpec(name="X", fouling=fouling))
    k = 137
    plant._step_index = k
    factor = 1.0 + 0.40 * (k * SAMPLE_TIME_S) / fouling.horizon_s
    larger = dataclasses.replace(
        config,
        geometry=dataclasses.replace(
            config.geometry, heat_loss_ua_w_per_k=config.geometry.heat_loss_ua_w_per_k * factor
        ),
    )
    reference = GreyBoxFluidBedGranulator(larger)
    state = plant.initial_state(1)
    action, disturbance = plant.nominal_action(1), plant.nominal_disturbance(1)
    d_plant, _ = plant.evaluate_derivative(state, action, disturbance)
    d_reference, _ = reference.evaluate_derivative(state, action, disturbance)
    assert np.allclose(d_plant, d_reference, rtol=1e-12, atol=1e-18)


def test_p3_drift_restarts_with_every_run(config: ProcessConfig) -> None:
    plant = make_plant("P3", config)
    inputs = _nominal_inputs(plant, 120)
    first = plant.rollout(*inputs, seed=6)
    second = plant.rollout(*inputs, seed=6)
    for name in ("states", "outputs", "measurements"):
        assert np.array_equal(getattr(first, name), getattr(second, name)), name


def test_p3_fouling_grows_linearly_in_the_run(config: ProcessConfig) -> None:
    clean = PlantVariantSimulator(config, dataclasses.replace(PLANT_VARIANTS["P3"], fouling=None))
    fouled = make_plant("P3", config)
    inputs = _nominal_inputs(clean, 400)
    gap = (
        fouled.rollout(*inputs, seed=0).states[0, :, 3]
        - clean.rollout(*inputs, seed=0).states[0, :, 3]
    )
    assert abs(gap[0]) < abs(gap[100]) < abs(gap[399])  # more fouling, larger effect
    assert gap[399] < 0.0  # more heat loss cools the bed


def test_p3_sensor_offset_drifts(config: ProcessConfig) -> None:
    spec = PlantVariantSpec(
        name="X",
        sensor=SensorSpec(),
        fouling=FoulingSpec(sensor_offset_growth_k=1.5, horizon_s=1000.0),
    )
    plant = PlantVariantSimulator(_quiet(config), spec)
    outputs = np.zeros((1, 41, 4))
    outputs[..., 1] = 60.0
    outputs[..., [0, 2, 3]] = 1.0
    measured = plant.remeasure(outputs, seed=0)[0, :, 1]
    # sample k carries the offset growth 1.5 K * k * 25 s / 1000 s; the lag of the reading
    # is long settled for a constant signal, so the drift is visible directly
    assert measured[0] == pytest.approx(60.0)
    assert measured[40] == pytest.approx(60.0 - 1.5 * 40 * SAMPLE_TIME_S / 1000.0, abs=0.05)


# ------------------------------------------------------------------ all plants


def test_variants_are_deterministic_and_have_distinct_fingerprints(config: ProcessConfig) -> None:
    fingerprints = {}
    for name in PLANT_VARIANTS:
        plant = make_plant(name, config)
        fingerprints[name] = plant.variant_fingerprint
        inputs = _nominal_inputs(plant, 30)
        a = plant.rollout(*inputs, seed=8)
        b = make_plant(name, config).rollout(*inputs, seed=8)
        assert np.array_equal(a.measurements, b.measurements), name
        assert a.metadata["plant_variant"] == name
        assert a.metadata["plant_variant_fingerprint"] == fingerprints[name]
    assert len(set(fingerprints.values())) == len(PLANT_VARIANTS)
    # the process fingerprint alone does NOT separate a measurement-only plant from P0
    assert (
        make_plant("P2b", config).config.fingerprint == make_plant("P0", config).config.fingerprint
    )


def test_unknown_variant_is_an_error(config: ProcessConfig) -> None:
    with pytest.raises(ValueError, match="unknown plant variant"):
        make_plant("P9", config)


# ------------------------------------------------------------------- generator


def _small_spec() -> DataSpec:
    return DataSpec.new_study(
        run_duration_s=1_000.0, n_train_max=4, n_val=2, n_test=2, history_length=8, horizon=4
    )


def test_generate_plant_data_for_p0_is_bit_identical_to_generate_data(
    config: ProcessConfig,
) -> None:
    spec = _small_spec()
    reference = generate_data(config, spec, master_seed=21)
    data = generate_plant_data(config, PLANT_VARIANTS["P0"], spec, master_seed=21)
    for name in ("actions", "disturbances", "states", "outputs", "measurements"):
        assert np.array_equal(getattr(reference.trajectory, name), getattr(data.trajectory, name))
    assert data.split.digest == reference.split.digest
    assert data.fingerprints["process/fingerprint"] == reference.fingerprints["process/fingerprint"]
    assert data.fingerprints["plant/name"] == "P0"


def test_plants_share_the_excitation_and_differ_in_the_data(config: ProcessConfig) -> None:
    spec = _small_spec()
    p0 = generate_plant_data(config, PLANT_VARIANTS["P0"], spec, master_seed=21)
    p3 = generate_plant_data(config, PLANT_VARIANTS["P3"], spec, master_seed=21)
    assert np.array_equal(p0.trajectory.actions, p3.trajectory.actions)
    assert np.array_equal(p0.trajectory.disturbances, p3.trajectory.disturbances)
    assert not np.array_equal(p0.trajectory.outputs, p3.trajectory.outputs)
    assert p0.fingerprints["plant/fingerprint"] != p3.fingerprints["plant/fingerprint"]
    assert p3.fingerprints["excitation/fingerprint"] == p0.fingerprints["excitation/fingerprint"]


def test_plant_with_measurements_uses_the_sensor_path_of_the_plant(config: ProcessConfig) -> None:
    spec = _small_spec()
    data = generate_plant_data(config, PLANT_VARIANTS["P2b"], spec, master_seed=21)
    again = data.with_measurements(noise_seed=data.seeds.noise_seed)
    assert np.array_equal(again.trajectory.measurements, data.trajectory.measurements)
    other = data.with_measurements(noise_seed=data.seeds.noise_seed + 1)
    assert not np.array_equal(other.trajectory.measurements, data.trajectory.measurements)
    assert np.array_equal(other.trajectory.outputs, data.trajectory.outputs)
    assert other.plant == data.plant


def test_generate_plant_data_needs_exactly_one_seed_source(config: ProcessConfig) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        generate_plant_data(config, PLANT_VARIANTS["P0"], _small_spec())
