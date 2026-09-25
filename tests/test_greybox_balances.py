"""Bilanzschluss, Constraints und analytische Grenzfälle.

Die Tests hier sind die eigentliche Absicherung des Modells. Sie prüfen nicht, dass der
Code läuft, sondern dass er die Physik erfüllt, die er behauptet -- und zwar an Stellen,
wo ein Vorzeichen- oder Einheitenfehler nicht durch Augenschein an einer Trajektorie
auffällt.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from conftest import constant_excitation
from twinlink_ml.simulation import GreyBoxFluidBedGranulator, ProcessConfig
from twinlink_ml.simulation.config import nucleation_fraction_for_target_diameter
from twinlink_ml.simulation.diagnostics import mass_balance_residual
from twinlink_ml.simulation.greybox import (
    IDX_PARTICLE_COUNT,
    IDX_SOLID_MASS,
    IDX_WATER_MASS,
    BedState,
)

_SECONDS_PER_HOUR = 3600.0


def _batch_config(config: ProcessConfig, **operation_overrides: object) -> ProcessConfig:
    operation = dataclasses.replace(
        config.operation,
        **operation_overrides,  # type: ignore[arg-type]
    )
    return dataclasses.replace(config, operation=operation)


def test_solid_balance_closes_exactly_in_batch_mode(config: ProcessConfig) -> None:
    """Ohne Produktabzug muss der Feststoffzuwachs exakt dem Eintrag entsprechen.

    Exakt, nicht approximativ: die Feststoffbilanz ist linear in den Zuständen, RK4
    integriert sie fehlerfrei. Jede Abweichung wäre ein Implementierungsfehler.
    """
    simulator = GreyBoxFluidBedGranulator(_batch_config(config, mode="batch"))
    n_steps = 200
    actions, disturbances = constant_excitation(simulator, n_steps)
    trajectory = simulator.rollout(actions, disturbances, seed=0)

    spray_rate = actions[0, 0, 2] / _SECONDS_PER_HOUR
    solid_fraction = disturbances[0, 0, 2]
    duration = n_steps * simulator.sample_time_s
    expected_increase = spray_rate * solid_fraction * duration

    initial = config.geometry.bed_mass_solid_setpoint_kg
    final = trajectory.states[0, -1, IDX_SOLID_MASS]
    assert final - initial == pytest.approx(expected_increase, rel=1e-10)


def test_solid_balance_closes_within_tolerance_in_continuous_mode(
    simulator: GreyBoxFluidBedGranulator,
) -> None:
    """Feststoffbilanz inklusive Produktabzug.

    Die Betriebsart, die ``test_solid_balance_closes_exactly_in_batch_mode`` bewusst
    ausspart (dort ``mode="batch"``, kein Produktabzug, deshalb dort exakter statt
    tolerierter Schluss). Nutzt
    :func:`~twinlink_ml.simulation.diagnostics.mass_balance_residual`, die
    ``simulator``-Fixture läuft in der Default-Betriebsart ``continuous``.
    """
    n_steps = 400
    actions, disturbances = constant_excitation(simulator, n_steps)
    trajectory = simulator.rollout(actions, disturbances, seed=0)
    assert mass_balance_residual(trajectory) < 1e-3


def test_water_balance_closes_without_evaporation(config: ProcessConfig) -> None:
    """Mit abgeschalteter Trocknung sammelt sich das eingesprühte Wasser vollständig an."""
    material = dataclasses.replace(config.material, max_saturation_degree=1e-12)
    dry_config = dataclasses.replace(_batch_config(config, mode="batch"), material=material)
    simulator = GreyBoxFluidBedGranulator(dry_config)

    n_steps = 100
    actions, disturbances = constant_excitation(simulator, n_steps)
    trajectory = simulator.rollout(actions, disturbances, seed=0)

    spray_rate = actions[0, 0, 2] / _SECONDS_PER_HOUR
    water_fraction = 1.0 - disturbances[0, 0, 2]
    expected = spray_rate * water_fraction * n_steps * simulator.sample_time_s
    initial = (
        config.geometry.bed_mass_solid_setpoint_kg * config.operation.initial_moisture_kg_per_kg
    )
    increase = trajectory.states[0, -1, IDX_WATER_MASS] - initial
    assert increase == pytest.approx(expected, rel=1e-6)


def test_water_balance_closes_with_evaporation(simulator: GreyBoxFluidBedGranulator) -> None:
    """Integraler Bilanzschluss inklusive Verdampfung und Produktabzug.

    Toleranz 1e-3 relativ: hier begrenzt die Trapezintegration der Diagnosegrößen die
    Genauigkeit, nicht das Modell.
    """
    n_steps = 400
    actions, disturbances = constant_excitation(simulator, n_steps)
    trajectory = simulator.rollout(actions, disturbances, seed=0)

    dt = simulator.sample_time_s
    spray_rate = actions[0, 0, 2] / _SECONDS_PER_HOUR
    water_in = spray_rate * (1.0 - disturbances[0, 0, 2]) * n_steps * dt
    evaporated = float(np.sum(trajectory.diagnostics["evaporation_rate_kg_per_s"][0]) * dt)
    moisture = trajectory.diagnostics["particle_moisture_kg_per_kg"][0]
    product = trajectory.diagnostics["product_mass_flow_kg_per_s"][0]
    withdrawn = float(np.sum(product * moisture) * dt)

    initial = simulator.config.geometry.bed_mass_solid_setpoint_kg * (
        simulator.config.operation.initial_moisture_kg_per_kg
    )
    accumulated = trajectory.states[0, -1, IDX_WATER_MASS] - initial
    residual = water_in - evaporated - withdrawn - accumulated
    assert abs(residual) < 1e-3 * water_in


def test_enthalpy_balance_closes_at_steady_state(simulator: GreyBoxFluidBedGranulator) -> None:
    """Im Beharrungszustand muss die Enthalpiebilanz auf Null aufgehen."""
    n_steps = 2000
    actions, disturbances = constant_excitation(simulator, n_steps)
    trajectory = simulator.rollout(actions, disturbances, seed=0)

    config = simulator.config
    state = trajectory.states[0, -1]
    bed_temperature = state[3]
    diagnostics = {k: v[0, -1] for k, v in trajectory.diagnostics.items()}

    air_flow = diagnostics["dry_air_mass_flow_kg_per_s"]
    inlet_humidity = diagnostics["inlet_humidity_kg_per_kg"]
    heat_in = (
        air_flow
        * (config.gas.cp_dry_air_j_per_kg_k + inlet_humidity * config.gas.cp_vapour_j_per_kg_k)
        * (actions[0, -1, 0] - bed_temperature)
    )
    heat_suspension = (
        actions[0, -1, 2]
        / _SECONDS_PER_HOUR
        * config.material.cp_suspension_j_per_kg_k
        * (config.operation.suspension_temperature_celsius - bed_temperature)
    )
    latent = diagnostics["evaporation_rate_kg_per_s"] * (2.5008e6 - 2358.6 * bed_temperature)
    losses = config.geometry.heat_loss_ua_w_per_k * (bed_temperature - disturbances[0, -1, 0])

    residual = heat_in + heat_suspension - latent - losses
    assert abs(residual) < 1e-3 * heat_in, f"Restglied {residual:.3e} W zu groß"


def test_layering_growth_matches_analytic_rate(config: ProcessConfig) -> None:
    r"""Prüft die Identität zum Layering-Ansatz aus 02_prozess_setup.md.

    .. math:: G = \frac{2\dot V_{fest}}{A}
              = \frac{\dot m_{spray} w_{solid} d_{50}}{3 m_s}

    Ohne Nukleation und ohne Produktabzug muss die numerische Wachstumsrate des
    Durchmessers exakt diesem Ausdruck entsprechen.
    """
    material = dataclasses.replace(config.material, nucleation_solid_fraction=0.0)
    pure_layering = dataclasses.replace(_batch_config(config, mode="batch"), material=material)
    simulator = GreyBoxFluidBedGranulator(pure_layering)

    n_steps = 20
    actions, disturbances = constant_excitation(simulator, n_steps)
    trajectory = simulator.rollout(actions, disturbances, seed=0)

    diameter = trajectory.diagnostics["d50_m"][0]
    numeric_rate = (diameter[1] - diameter[0]) / simulator.sample_time_s

    solid_mass = trajectory.states[0, 0, IDX_SOLID_MASS]
    spray_solid = actions[0, 0, 2] / _SECONDS_PER_HOUR * disturbances[0, 0, 2]
    analytic_rate = spray_solid * diameter[0] / (3.0 * solid_mass)

    assert numeric_rate == pytest.approx(analytic_rate, rel=1e-3)


def test_nucleation_limited_diameter_matches_number_balance(config: ProcessConfig) -> None:
    r"""Der Beharrungsdurchmesser des keimfreien Spezialfalls folgt der Zahlenbilanz.

    .. math:: d_{50,ss} = d_{nuc}\, b_{nuc}^{-1/3}

    Damit ist ``nucleation_solid_fraction`` kein freier Fitparameter: er wird durch die
    Zielkorngröße festgelegt. Ein Modell, in dem beide unabhängig eingestellt werden,
    ist überparametrisiert und verletzt die Zahlenbilanz.
    """
    target_diameter_m = 750e-6
    fraction = nucleation_fraction_for_target_diameter(
        target_diameter_m, config.material.nucleus_diameter_m
    )
    material = dataclasses.replace(
        config.material,
        nucleation_solid_fraction=fraction,
        initial_diameter_m=target_diameter_m,
    )
    simulator = GreyBoxFluidBedGranulator(dataclasses.replace(config, material=material))
    assert material.nucleation_limited_diameter_m == pytest.approx(target_diameter_m, rel=1e-9)

    n_steps = 8000
    actions, disturbances = constant_excitation(simulator, n_steps)
    trajectory = simulator.rollout(actions, disturbances, seed=0)
    final_diameter = trajectory.diagnostics["d50_m"][0, -1]
    assert final_diameter == pytest.approx(target_diameter_m, rel=0.02)


def test_particle_moisture_stays_above_equilibrium(simulator: GreyBoxFluidBedGranulator) -> None:
    """Zweite harte Grenze: X_p >= X_eq, strukturell über eta -> 0."""
    n_steps = 1000
    actions, disturbances = constant_excitation(simulator, n_steps)
    # extrem trocken fahren: heiß, viel Luft, minimale Sprührate
    actions[..., 0] = simulator.config.actions.inlet_temperature_celsius.maximum
    actions[..., 1] = simulator.config.actions.gas_velocity_m_per_s.maximum
    actions[..., 2] = simulator.config.actions.spray_rate_kg_per_h.minimum
    trajectory = simulator.rollout(actions, disturbances, seed=0)

    moisture = trajectory.diagnostics["particle_moisture_kg_per_kg"][0]
    assert np.all(moisture >= simulator.config.material.equilibrium_moisture_kg_per_kg - 1e-9)


def _gab_config(
    config: ProcessConfig, *, critical_moisture_kg_per_kg: float = 0.200
) -> ProcessConfig:
    """GAB-Isotherme, verankert bei phi=0.5/T=62C -> 0.030 kg/kg.

    Siehe docs/04_stand_ap1.md, Befund 34 -- dieselben Parameter wie
    tests/test_sorption.py::_GAB_KWARGS.
    """
    material = dataclasses.replace(
        config.material,
        critical_moisture_kg_per_kg=critical_moisture_kg_per_kg,
        equilibrium_moisture_kg_per_kg=0.030,
        sorption_isotherm_kind="gab",
        gab_monolayer_moisture_kg_per_kg=0.020,
        gab_c_constant_at_reference_temperature=25.0,
        gab_k_constant_at_reference_temperature=0.75,
        gab_c_enthalpy_j_per_mol=6000.0,
        gab_k_enthalpy_j_per_mol=500.0,
        gab_reference_temperature_celsius=62.0,
    )
    return dataclasses.replace(config, material=material)


def test_particle_moisture_stays_above_local_equilibrium_with_gab_isotherm(
    config: ProcessConfig,
) -> None:
    """Zweite harte Grenze mit zustandsabhängiger Isotherme.

    X_p >= X_eq(phi,T) am jeweiligen Zeitschritt, nicht mehr X_p >= ein fester Skalar.
    """
    from twinlink_ml.simulation.greybox import IDX_BED_TEMPERATURE
    from twinlink_ml.simulation.sorption import isotherm_from_material
    from twinlink_ml.simulation.thermo import relative_humidity_from_absolute_humidity

    gab_config = _gab_config(config)
    simulator = GreyBoxFluidBedGranulator(gab_config)
    n_steps = 1000
    actions, disturbances = constant_excitation(simulator, n_steps)
    actions[..., 0] = simulator.config.actions.inlet_temperature_celsius.maximum
    actions[..., 1] = simulator.config.actions.gas_velocity_m_per_s.maximum
    actions[..., 2] = simulator.config.actions.spray_rate_kg_per_h.minimum
    trajectory = simulator.rollout(actions, disturbances, seed=0)

    moisture = trajectory.diagnostics["particle_moisture_kg_per_kg"][0]
    outlet_humidity = trajectory.diagnostics["outlet_humidity_kg_per_kg"][0]
    bed_temperature = trajectory.states[0, :, IDX_BED_TEMPERATURE]

    backend = simulator.backend
    isotherm = isotherm_from_material(gab_config.material)
    relative_humidity = relative_humidity_from_absolute_humidity(
        backend, gab_config.gas, outlet_humidity, bed_temperature
    )
    local_equilibrium_moisture = backend.to_numpy(
        isotherm.equilibrium_moisture(backend, relative_humidity, bed_temperature)
    )
    assert np.all(moisture >= local_equilibrium_moisture - 1e-9)


def test_scattered_cold_start_below_local_equilibrium_is_floored_up(
    config: ProcessConfig,
) -> None:
    """Regressionsfall aus der Diversitäts-Kennzahl (docs/04_stand_ap1.md, Befund 34).

    T_bed=T_amb=35C (minimale Heizdifferenz), phi_amb=0.85 (Maximum), X_p0=0.0301 (nahe
    dem Anker) -- lokal ist X_eq dort ~0.056, weit über X_p0. Der k1-basierte Floor muss
    das im allerersten internen Schritt korrekt und stetig hochklemmen.
    """
    gab_config = _gab_config(config)
    simulator = GreyBoxFluidBedGranulator(gab_config)

    initial = simulator.state_from_physical(
        solid_mass_kg=np.array([gab_config.geometry.bed_mass_solid_setpoint_kg]),
        moisture_kg_per_kg=np.array([0.0301]),
        diameter_m=np.array([gab_config.material.initial_diameter_m]),
        bed_temperature_celsius=np.array([35.0]),
    )
    action = simulator.nominal_action(1)
    disturbance = np.array(
        [[35.0, 0.85, gab_config.disturbances.suspension_solid_fraction.nominal]]
    )

    result = simulator.step(initial, action, disturbance)
    moisture_after = float(result.diagnostics["particle_moisture_kg_per_kg"][0])

    assert np.all(np.isfinite(simulator.backend.to_numpy(result.state)))
    assert moisture_after > 0.0301  # Floor hat sichtbar hochgeklemmt
    assert moisture_after < gab_config.material.critical_moisture_kg_per_kg


def test_energy_target_is_non_negative_and_monotone(simulator: GreyBoxFluidBedGranulator) -> None:
    r"""Monotonie-Constraints des Projekts als Property-Test.

    .. math:: \partial \dot E/\partial T_{in} > 0, \qquad
              \partial \dot E/\partial u_{gas} > 0

    Vor Verfügbarkeit von Hypothesis mit einem festen RNG-Seed umgesetzt; die Zusage
    ``property-based`` aus den Code-Konventionen ist damit noch nicht erfüllt, der
    Testinhalt aber schon.
    """
    rng = np.random.default_rng(42)
    n = 300
    state = simulator.initial_state(n)
    bounds = simulator.config.actions.bounds
    ranges = simulator.config.disturbances.ranges

    action = np.stack([rng.uniform(b.minimum, b.maximum, n) for b in bounds], axis=-1)
    disturbance = np.stack([rng.uniform(r.minimum, r.maximum, n) for r in ranges], axis=-1)

    base = simulator.step(state, action, disturbance).diagnostics["power_total_kw"]
    assert np.all(base > 0.0)

    for index, delta in ((0, 5.0), (1, 0.2)):
        perturbed = action.copy()
        perturbed[:, index] = np.minimum(perturbed[:, index] + delta, bounds[index].maximum)
        changed = perturbed[:, index] > action[:, index] + 1e-12
        raised = simulator.step(state, perturbed, disturbance).diagnostics["power_total_kw"]
        assert np.all(raised[changed] > base[changed]), (
            f"Monotonie in {simulator.action_names[index]} verletzt"
        )


def test_action_bounds_are_enforced(simulator: GreyBoxFluidBedGranulator) -> None:
    """Sicherheitsrelevant: der Simulator ist die letzte Instanz vor der Physik."""
    n_steps = 5
    actions, disturbances = constant_excitation(simulator, n_steps)
    actions[..., 0] = 1e4  # absurd heiße Zuluft
    actions[..., 1] = -5.0  # negative Gasgeschwindigkeit
    trajectory = simulator.rollout(actions, disturbances, seed=0)
    assert np.all(trajectory.diagnostics["power_total_kw"] > 0.0)
    assert np.all(np.isfinite(trajectory.states))
    maximum = simulator.config.actions.inlet_temperature_celsius.maximum
    assert np.all(trajectory.states[..., 3] < maximum + 1.0)


def test_bed_state_view_matches_array(simulator: GreyBoxFluidBedGranulator) -> None:
    state = simulator.initial_state(3)
    view = BedState.from_array(state)
    assert view.solid_mass_kg.shape == (3,)
    assert float(view.particle_count[0]) == pytest.approx(
        simulator.config.geometry.bed_mass_solid_setpoint_kg
        / (
            simulator.config.material.particle_density_kg_per_m3
            * math.pi
            * simulator.config.material.initial_diameter_m**3
            / 6.0
        )
    )
    assert state[0, IDX_PARTICLE_COUNT] == view.particle_count[0]
