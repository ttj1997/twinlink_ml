"""Numerics, batch consistency, determinism and config round trip.

The step-size test is the most important one here: a data set from under-converged
runs looks entirely plausible, but contains systematic errors that the
sequence model dutifully learns along. The error then only shows up when switching to
Dyssol data -- as an apparent transfer problem.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from conftest import constant_excitation
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    ProcessConfig,
    Provenance,
    dump_process_config,
    load_process_config,
)
from twinlink_ml.utils import SeedBundle

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_step_size_convergence(config: ProcessConfig) -> None:
    """Doubling the substeps must hardly change the result."""
    reference = None
    results = []
    for substeps in (5, 10, 20):
        numerics = dataclasses.replace(config.numerics, substeps=substeps)
        simulator = GreyBoxFluidBedGranulator(dataclasses.replace(config, numerics=numerics))
        actions, disturbances = constant_excitation(simulator, 400)
        trajectory = simulator.rollout(actions, disturbances, seed=0)
        results.append(trajectory.outputs[0, -1, :])
    reference = results[-1]
    for value in results[:-1]:
        relative = np.abs(value - reference) / np.maximum(np.abs(reference), 1e-9)
        assert np.all(relative < 1e-3), f"Schrittweitenabhängigkeit {relative}"


def test_euler_needs_more_substeps_than_rk4(config: ProcessConfig) -> None:
    """Justification of the integrator choice, not just a claim."""

    def final_temperature(integrator: str, substeps: int) -> float:
        numerics = dataclasses.replace(config.numerics, integrator=integrator, substeps=substeps)
        simulator = GreyBoxFluidBedGranulator(dataclasses.replace(config, numerics=numerics))
        actions, disturbances = constant_excitation(simulator, 200)
        return float(simulator.rollout(actions, disturbances, seed=0).states[0, -1, 3])

    accurate = final_temperature("rk4", 50)
    error_rk4 = abs(final_temperature("rk4", 2) - accurate)
    error_euler = abs(final_temperature("euler", 2) - accurate)
    assert error_rk4 < error_euler


def test_batch_dimension_is_independent(simulator: GreyBoxFluidBedGranulator) -> None:
    """Batch-wise evaluation must be identical to the single run.

    Otherwise batch-wise generated data sets are subtly different from the individually
    generated reference runs -- an error that can hardly be localised later.
    """
    n_steps = 120
    rng = np.random.default_rng(7)
    single_actions, single_disturbances = constant_excitation(simulator, n_steps, n_batch=1)
    varied_actions = np.concatenate(
        [single_actions, single_actions * rng.uniform(0.9, 1.1, single_actions.shape)], axis=0
    )
    varied_disturbances = np.concatenate([single_disturbances, single_disturbances], axis=0)

    batched = simulator.rollout(varied_actions, varied_disturbances, seed=0)
    individual = simulator.rollout(varied_actions[:1], varied_disturbances[:1], seed=0)

    np.testing.assert_allclose(batched.outputs[0], individual.outputs[0], rtol=1e-12, atol=0.0)


def test_rollout_is_deterministic(simulator: GreyBoxFluidBedGranulator) -> None:
    actions, disturbances = constant_excitation(simulator, 60)
    first = simulator.rollout(actions, disturbances, seed=3)
    second = simulator.rollout(actions, disturbances, seed=3)
    np.testing.assert_array_equal(first.measurements, second.measurements)

    third = simulator.rollout(actions, disturbances, seed=4)
    assert not np.array_equal(first.measurements, third.measurements)
    np.testing.assert_allclose(first.outputs, third.outputs, rtol=1e-12)


def test_measurement_noise_does_not_bias_outputs(simulator: GreyBoxFluidBedGranulator) -> None:
    """The sensor model must not shift the expected value."""
    actions, disturbances = constant_excitation(simulator, 600)
    trajectory = simulator.rollout(actions, disturbances, seed=11)
    truth = trajectory.outputs[0, 300:, 1]
    measured = trajectory.measurements[0, 300:, 1]
    assert abs(float(np.mean(measured - truth))) < 0.05
    assert float(np.std(measured - truth)) > 0.05, "Rauschen fehlt vollständig"


def test_state_floor_is_inactive_at_nominal_operation(
    simulator: GreyBoxFluidBedGranulator,
) -> None:
    """The numerical lower bound must never engage in normal operation."""
    actions, disturbances = constant_excitation(simulator, 500)
    trajectory = simulator.rollout(actions, disturbances, seed=0)
    floor = simulator.config.numerics.state_floor
    assert np.all(trajectory.states[..., 0] > 1e6 * floor)
    assert np.all(trajectory.states[..., 2] > 1e6 * floor)


def test_wrong_input_shape_is_rejected(simulator: GreyBoxFluidBedGranulator) -> None:
    actions, disturbances = constant_excitation(simulator, 10)
    with pytest.raises(ValueError, match="Stellgrößen"):
        simulator.rollout(actions[..., :2], disturbances, seed=0)
    with pytest.raises(ValueError, match="Form"):
        simulator.rollout(actions[0], disturbances[0], seed=0)


def test_trajectory_select_and_concatenate(simulator: GreyBoxFluidBedGranulator) -> None:
    actions, disturbances = constant_excitation(simulator, 30, n_batch=3)
    trajectory = simulator.rollout(actions, disturbances, seed=0)
    assert trajectory.n_runs == 3
    merged = trajectory.select(0).concatenate(trajectory.select(2))
    assert merged.n_runs == 2
    assert merged.run_id == ("run_00000", "run_00002")


def test_time_constants_span_several_decades(simulator: GreyBoxFluidBedGranulator) -> None:
    """Der Befund, der die Sequenzlängenwahl bestimmt.

    Erwartet: Feuchte im Sekundenbereich, Temperatur ~Minute, Partikelgröße ~Stunden.
    Die Spreizung über mehrere Dekaden ist kein Modellartefakt, sondern die zentrale
    Eigenschaft des Prozesses -- und der Grund, warum eine einzelne "dominante
    Zeitkonstante" die Sequenzlänge nicht festlegen kann.
    """
    state = simulator.initial_state(1)
    action = simulator.backend.asarray(simulator.nominal_action(1))
    disturbance = simulator.backend.asarray(simulator.nominal_disturbance(1))
    taus = simulator.time_constants(state, action, disturbance)

    values = sorted(v for k, v in taus.items() if k.startswith("tau_"))
    assert len(values) >= 3
    assert values[0] < 60.0, "schnellste Mode sollte im Sekundenbereich liegen"
    assert values[-1] > 1800.0, "langsamste Mode sollte im Stundenbereich liegen"
    assert taus["stiffness_ratio"] > 50.0


def test_config_roundtrip(config: ProcessConfig, tmp_path: Path) -> None:
    path = tmp_path / "resolved.yaml"
    dump_process_config(config, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        reloaded = load_process_config(path)
    assert reloaded == config
    assert reloaded.fingerprint == config.fingerprint


def test_section_fingerprints_cover_all_twelve_sections(config: ProcessConfig) -> None:
    """G4b: all twelve sections from fingerprint/to_dict() are represented.

    ``exhaust`` added since the introduction of ``ExhaustConfig`` (duct/filter loss
    between bed and heat recovery network, see ``simulation.exhaust.reconstruct_exhaust_state``)
    -- previously eleven sections.
    """
    sections = config.section_fingerprints
    assert set(sections) == {
        "provenance",
        "geometry",
        "gas",
        "material",
        "energy",
        "exhaust",
        "operation",
        "actions",
        "disturbances",
        "numerics",
        "measurement",
        "backend",
    }
    assert all(isinstance(value, str) and value for value in sections.values())


def test_section_fingerprint_isolates_changed_section(config: ProcessConfig) -> None:
    """Only the changed section may change its partial fingerprint -- the purpose of G4b."""
    changed = dataclasses.replace(
        config, geometry=dataclasses.replace(config.geometry, heat_loss_ua_w_per_k=999.0)
    )
    before = config.section_fingerprints
    after = changed.section_fingerprints
    assert before["geometry"] != after["geometry"]
    for section in before:
        if section != "geometry":
            message = f"{section} haette unveraendert bleiben sollen"
            assert before[section] == after[section], message


def test_section_fingerprints_do_not_change_overall_fingerprint(config: ProcessConfig) -> None:
    """Additive: section_fingerprints must not affect fingerprint itself."""
    fingerprint_before = config.fingerprint
    _ = config.section_fingerprints
    assert config.fingerprint == fingerprint_before


def test_placeholder_config_warns(config: ProcessConfig) -> None:
    """The guarantee from 02_prozess_setup.md, anchored in the code."""
    with pytest.warns(PlaceholderConfigWarning):
        GreyBoxFluidBedGranulator(config)

    calibrated = dataclasses.replace(config, provenance=Provenance(status="calibrated"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GreyBoxFluidBedGranulator(calibrated)


def test_config_validation_rejects_inconsistent_values(config: ProcessConfig) -> None:
    with pytest.raises(ValueError, match="critical_moisture"):
        dataclasses.replace(
            config.material, critical_moisture_kg_per_kg=0.001, equilibrium_moisture_kg_per_kg=0.003
        )
    with pytest.raises(ValueError, match="substeps"):
        dataclasses.replace(config.numerics, substeps=0)


def test_simulator_construction_rejects_gab_isotherm_with_low_critical_moisture(
    config: ProcessConfig,
) -> None:
    """The gap from step 1: validate_isotherm_against_critical_moisture wired up.

    Already existed, but was not wired up anywhere -- a config with
    sorption_isotherm_kind='gab' and too low a critical_moisture_kg_per_kg must actually fail
    in GreyBoxFluidBedGranulator.__init__, not only when the check function is called
    manually.
    """
    material = dataclasses.replace(
        config.material,
        critical_moisture_kg_per_kg=0.05,  # above anchor (0.030), below GAB maximum (~0.089)
        equilibrium_moisture_kg_per_kg=0.030,
        sorption_isotherm_kind="gab",
        gab_monolayer_moisture_kg_per_kg=0.020,
        gab_c_constant_at_reference_temperature=25.0,
        gab_k_constant_at_reference_temperature=0.75,
        gab_c_enthalpy_j_per_mol=6000.0,
        gab_k_enthalpy_j_per_mol=500.0,
        gab_reference_temperature_celsius=62.0,
    )
    bad_config = dataclasses.replace(config, material=material)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        with pytest.raises(ValueError, match="critical_moisture_kg_per_kg"):
            GreyBoxFluidBedGranulator(bad_config)


def test_simulator_construction_accepts_gab_isotherm_with_lab_level_critical_moisture(
    config: ProcessConfig,
) -> None:
    """Counter-check: lab-level X_cr (0.200) constructs the simulator without error.

    The same isotherm as in the failure case above, only with a non-critical X_cr.
    """
    material = dataclasses.replace(
        config.material,
        critical_moisture_kg_per_kg=0.200,
        equilibrium_moisture_kg_per_kg=0.030,
        sorption_isotherm_kind="gab",
        gab_monolayer_moisture_kg_per_kg=0.020,
        gab_c_constant_at_reference_temperature=25.0,
        gab_k_constant_at_reference_temperature=0.75,
        gab_c_enthalpy_j_per_mol=6000.0,
        gab_k_enthalpy_j_per_mol=500.0,
        gab_reference_temperature_celsius=62.0,
    )
    ok_config = dataclasses.replace(config, material=material)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        GreyBoxFluidBedGranulator(ok_config)  # must not raise


def test_local_floor_at_nominal_never_below_envelope_reference(config: ProcessConfig) -> None:
    """Consistency test for the envelope-transformed constant (0.000203 kg/kg).

    Docs/04_stand_ap1.md Befund 34: since the k1-based variant it is no longer a
    runtime fallback, only a reference value -- the local floor value obtained from the
    state at the nominal point must always lie clearly above it.
    """
    from twinlink_ml.simulation.greybox import IDX_BED_TEMPERATURE
    from twinlink_ml.simulation.sorption import isotherm_from_material, sorption_safety_grid
    from twinlink_ml.simulation.thermo import relative_humidity_from_absolute_humidity

    material = dataclasses.replace(
        config.material,
        critical_moisture_kg_per_kg=0.200,
        equilibrium_moisture_kg_per_kg=0.030,
        sorption_isotherm_kind="gab",
        gab_monolayer_moisture_kg_per_kg=0.020,
        gab_c_constant_at_reference_temperature=25.0,
        gab_k_constant_at_reference_temperature=0.75,
        gab_c_enthalpy_j_per_mol=6000.0,
        gab_k_enthalpy_j_per_mol=500.0,
        gab_reference_temperature_celsius=62.0,
    )
    gab_config = dataclasses.replace(config, material=material)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(gab_config)

    backend = simulator.backend
    isotherm = isotherm_from_material(gab_config.material)
    phi_min, _phi_max, _t_min, t_max = sorption_safety_grid(
        backend, gab_config.gas, gab_config.actions, gab_config.disturbances
    )
    envelope_reference = float(
        backend.to_numpy(
            isotherm.equilibrium_moisture(backend, backend.asarray(phi_min), backend.asarray(t_max))
        )
    )

    state = simulator.initial_state(1)
    action = simulator.nominal_action(1)
    disturbance = simulator.nominal_disturbance(1)
    _, diagnostics = simulator.evaluate_derivative(state, action, disturbance)
    relative_humidity = relative_humidity_from_absolute_humidity(
        backend,
        gab_config.gas,
        diagnostics["outlet_humidity_kg_per_kg"],
        state[..., IDX_BED_TEMPERATURE],
    )
    local_equilibrium_moisture = float(
        backend.to_numpy(
            isotherm.equilibrium_moisture(
                backend, relative_humidity, state[..., IDX_BED_TEMPERATURE]
            )
        )[0]
    )
    assert envelope_reference < local_equilibrium_moisture
    assert local_equilibrium_moisture / envelope_reference > 10.0  # order-of-magnitude margin


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_gradient_flows_through_full_rk4_step_with_gab_isotherm(config: ProcessConfig) -> None:
    """Differentiability through a complete RK4 step (substeps=1) with GAB.

    Active GAB isotherm, Picard iteration and k1-based floor via the
    Torch backend -- torch.autograd.gradcheck plus an independent counter-check against central
    finite differences via the NumPy backend. A point at which the floor does NOT engage (X_p
    clearly above X_eq), so that an exact gradient comparison makes sense -- at a
    floor-active point a kink (maximum()) has no unique gradient
    on which FD and autograd would have to agree; for that see instead
    ``test_gradient_stays_finite_at_floor_active_point`` (only finiteness, no exact
    agreement).
    """
    import torch

    from twinlink_ml.simulation.backend import NumpyBackend, TorchBackend

    torch.set_default_dtype(torch.float64)

    material = dataclasses.replace(
        config.material,
        critical_moisture_kg_per_kg=0.200,
        equilibrium_moisture_kg_per_kg=0.030,
        sorption_isotherm_kind="gab",
        gab_monolayer_moisture_kg_per_kg=0.020,
        gab_c_constant_at_reference_temperature=25.0,
        gab_k_constant_at_reference_temperature=0.75,
        gab_c_enthalpy_j_per_mol=6000.0,
        gab_k_enthalpy_j_per_mol=500.0,
        gab_reference_temperature_celsius=62.0,
    )
    numerics = dataclasses.replace(config.numerics, substeps=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        torch_config = dataclasses.replace(
            config, material=material, numerics=numerics, backend="torch"
        )
        simulator_torch = GreyBoxFluidBedGranulator(torch_config, backend=TorchBackend())
        numpy_config = dataclasses.replace(torch_config, backend="numpy")
        simulator_numpy = GreyBoxFluidBedGranulator(numpy_config, backend=NumpyBackend())

    particle_mass = material.particle_density_kg_per_m3 * math.pi * (400e-6) ** 3 / 6.0
    solid_mass, particle_count, bed_temperature = 2.0, 2.0 / particle_mass, 70.0
    m_w0_value = solid_mass * 0.05  # X_p=0.05, clearly above the local X_eq at this T
    action_fixed = (90.0, 1.6, 1.5, 2.5)
    disturbance_fixed = (20.0, 0.50, 0.30)

    def step_torch(m_w0: torch.Tensor, spray_rate: torch.Tensor) -> torch.Tensor:
        state = torch.stack(
            [
                torch.tensor(solid_mass),
                m_w0,
                torch.tensor(particle_count),
                torch.tensor(bed_temperature),
            ]
        ).unsqueeze(0)
        action = torch.stack(
            [
                torch.tensor(action_fixed[0]),
                torch.tensor(action_fixed[1]),
                spray_rate,
                torch.tensor(action_fixed[3]),
            ]
        ).unsqueeze(0)
        disturbance = torch.tensor(disturbance_fixed).unsqueeze(0)
        result = simulator_torch.step(state, action, disturbance)
        return result.state[0, 1]  # m_w_final

    m_w0 = torch.tensor(m_w0_value, requires_grad=True)
    spray_rate0 = torch.tensor(action_fixed[2], requires_grad=True)
    assert torch.autograd.gradcheck(step_torch, (m_w0, spray_rate0), eps=1e-6, atol=1e-6, rtol=1e-4)

    def step_numpy(m_w0_value_: float) -> float:
        state = np.array([[solid_mass, m_w0_value_, particle_count, bed_temperature]])
        action = np.array([list(action_fixed)])
        disturbance = np.array([list(disturbance_fixed)])
        result = simulator_numpy.step(state, action, disturbance)
        return float(result.state[0, 1])

    h = 1e-6
    finite_difference_gradient = (step_numpy(m_w0_value + h) - step_numpy(m_w0_value - h)) / (2 * h)

    m_w0_ad = torch.tensor(m_w0_value, requires_grad=True)
    spray_ad = torch.tensor(action_fixed[2])
    m_w_final = step_torch(m_w0_ad, spray_ad)
    m_w_final.backward()
    autograd_gradient = m_w0_ad.grad.item()

    assert autograd_gradient != 0.0  # floor inactive at this point -- real gradient test
    assert autograd_gradient == pytest.approx(finite_difference_gradient, rel=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_gradient_stays_finite_at_floor_active_point(config: ProcessConfig) -> None:
    """No comparison against finite differences required, only: the gradient stays finite.

    Not to be expected at a kink (see
    test_gradient_flows_through_full_rk4_step_with_gab_isotherm) -- no NaN/Inf,
    relevant for RL rollouts that might differentiate through several floor-active
    steps. Regression point from the diversity metric
    (docs/04_stand_ap1.md, Befund 34): T_bed=T_amb=35°C, phi_amb=0.85, X_p0=0.0301 --
    the floor visibly engages there (see
    test_scattered_cold_start_below_local_equilibrium_is_floored_up).
    """
    import torch

    from twinlink_ml.simulation.backend import TorchBackend

    torch.set_default_dtype(torch.float64)
    material = dataclasses.replace(
        config.material,
        critical_moisture_kg_per_kg=0.200,
        equilibrium_moisture_kg_per_kg=0.030,
        sorption_isotherm_kind="gab",
        gab_monolayer_moisture_kg_per_kg=0.020,
        gab_c_constant_at_reference_temperature=25.0,
        gab_k_constant_at_reference_temperature=0.75,
        gab_c_enthalpy_j_per_mol=6000.0,
        gab_k_enthalpy_j_per_mol=500.0,
        gab_reference_temperature_celsius=62.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        gab_config = dataclasses.replace(config, material=material, backend="torch")
        simulator = GreyBoxFluidBedGranulator(gab_config, backend=TorchBackend())

    particle_mass = (
        material.particle_density_kg_per_m3 * math.pi * material.initial_diameter_m**3 / 6.0
    )
    solid_mass = config.geometry.bed_mass_solid_setpoint_kg
    m_w0 = torch.tensor(solid_mass * 0.0301, requires_grad=True)
    state = torch.stack(
        [
            torch.tensor(solid_mass),
            m_w0,
            torch.tensor(solid_mass / particle_mass),
            torch.tensor(35.0),
        ]
    ).unsqueeze(0)
    action = torch.tensor(simulator.nominal_action(1)[0]).unsqueeze(0)
    disturbance = torch.tensor(
        [35.0, 0.85, config.disturbances.suspension_solid_fraction.nominal]
    ).unsqueeze(0)

    result = simulator.step(state, action, disturbance)
    result.state[0, 1].backward()

    assert torch.isfinite(m_w0.grad).all()


def test_seed_bundle_is_reproducible_and_decorrelated() -> None:
    first = SeedBundle.from_master(1)
    assert first == SeedBundle.from_master(1)
    assert len({first.data_seed, first.noise_seed, first.model_seed}) == 3
    neighbour = SeedBundle.from_master(2)
    assert abs(first.data_seed - neighbour.data_seed) > 1000
