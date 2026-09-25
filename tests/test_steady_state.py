"""Tests des exakten Beharrungszustands-Lösers.

Referenzwerte (T_bed, X_ss, d50, Verhältnis Abluftenthalpie/Leistung) sind extern
verifiziert -- siehe ``docs/04_stand_ap1.md``, Befund A/B.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    ProcessConfig,
    load_process_config,
)
from twinlink_ml.simulation.config import nucleation_fraction_for_target_diameter
from twinlink_ml.simulation.greybox import IDX_BED_TEMPERATURE, IDX_PARTICLE_COUNT, IDX_WATER_MASS
from twinlink_ml.simulation.steady_state import solve_steady_state

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GAB_CONFIG_PATH = _REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
_REGRET_SCRIPT_PATH = _REPO_ROOT / "scripts" / "study_operating_point_regret.py"
_ANCHOR_ARTIFACT_PATH = _REPO_ROOT / "artifacts" / "runs" / "bb1_bb3_operating_point_regret.json"
_KNOWN_FAILING_GRID_INDEX = 15
"""Befund-52-Gitterpunkt, an dem Loeser-DEFAULTS (100/1e-13) an einem Bracket-
Vorzeichenfehler scheitern, waehrend die SUCH-Praezision (30/1e-7) bitgleich zu Befund 52
loest -- siehe ``test_solve_steady_state_defaults_do_not_reproduce_befund_52`` unten."""


def _load_regret_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "study_operating_point_regret", _REGRET_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

# Vorfaktor der Exponentialhüllkurve für den langsamen Zustand (d50/N), siehe
# test_slow_state_approaches_transient_within_exponential_envelope.
_ENVELOPE_FACTOR = 10.0
_TRANSIENT_DURATION_S = 150_000.0


def test_matches_verified_reference_point(config: ProcessConfig) -> None:
    """T_bed, X_ss, d50 am Nominalpunkt -- extern verifiziert, siehe Modul-Docstring."""
    simulator = GreyBoxFluidBedGranulator(config)
    solution = solve_steady_state(simulator)
    assert solution.bed_temperature_celsius == pytest.approx(62.311, abs=5e-4)
    assert solution.particle_moisture_kg_per_kg == pytest.approx(0.004676, abs=1e-6)
    assert solution.diameter_m * 1e6 == pytest.approx(750.0, rel=1e-6)


def test_residual_is_below_tolerance(config: ProcessConfig) -> None:
    simulator = GreyBoxFluidBedGranulator(config)
    solution = solve_steady_state(simulator)
    assert solution.residual_inf_norm < 1e-10


def test_diameter_matches_nucleation_limited_diameter_when_seed_is_zero(
    config: ProcessConfig,
) -> None:
    """Der Test aus dem Auftrag gilt nur unter der Bedingung ``seed_mass_flow == 0``.

    Die allgemeine Formel (mit Keimzufuhr) reduziert sich auf
    ``MaterialConfig.nucleation_limited_diameter_m`` nur in diesem Spezialfall --
    siehe dessen Docstring und den Moduldocstring von ``steady_state.py``.
    """
    assert config.operation.seed_mass_flow_kg_per_h == 0.0
    simulator = GreyBoxFluidBedGranulator(config)
    solution = solve_steady_state(simulator)
    assert solution.diameter_m == pytest.approx(
        config.material.nucleation_limited_diameter_m, rel=1e-9
    )


def test_jacobian_eigenvalues_are_all_stable_at_solution(config: ProcessConfig) -> None:
    simulator = GreyBoxFluidBedGranulator(config)
    solution = solve_steady_state(simulator)
    backend = simulator.backend
    state = backend.asarray(solution.state[None, :])
    action = backend.asarray(simulator.nominal_action(1))
    disturbance = backend.asarray(simulator.nominal_disturbance(1))
    report = simulator.time_constants(state, action, disturbance)
    taus = [v for k, v in report.items() if k.startswith("tau_")]
    assert report["n_unstable_modes"] == 0.0
    assert len(taus) == 4, "alle vier Eigenwerte müssen stabil (Re < 0) sein"
    assert all(tau > 0.0 for tau in taus)


def test_bracket_sign_violation_raises_runtime_error(config: ProcessConfig) -> None:
    """Konstruiertes Gegenbeispiel: ein Bracket vollständig in der Kondensationsbranche.

    Bei Betttemperaturen nahe/unter T_amb kippt ``evaporation_rate_kg_per_s`` in die
    Kondensationsbranche (Y_in > Y_s(T_bed)); dort existiert im Feuchtebracket
    ``[X_eq, moisture_high]`` kein Vorzeichenwechsel -- die innere Bisektion muss das
    laut, nicht still, melden.
    """
    simulator = GreyBoxFluidBedGranulator(config)
    with pytest.raises(RuntimeError, match="Wasserbilanz"):
        solve_steady_state(simulator, temperature_bracket_celsius=(-30.0, -25.0))


def test_batch_mode_has_no_steady_state(config: ProcessConfig) -> None:
    batch_config = dataclasses.replace(
        config, operation=dataclasses.replace(config.operation, mode="batch")
    )
    simulator = GreyBoxFluidBedGranulator(batch_config)
    with pytest.raises(ValueError, match="Batch"):
        solve_steady_state(simulator)


def test_dew_point_is_below_bed_temperature(config: ProcessConfig) -> None:
    simulator = GreyBoxFluidBedGranulator(config)
    solution = solve_steady_state(simulator)
    assert solution.dew_point_celsius < solution.bed_temperature_celsius


def test_exhaust_enthalpy_covers_most_of_input_power(config: ProcessConfig) -> None:
    """Bezugszustand Umgebung (T_amb, Wasser flüssig) -- siehe SteadyStateSolution-Docstring.

    Verifizierter Wert am Nominalpunkt: 87.92 %.
    """
    simulator = GreyBoxFluidBedGranulator(config)
    solution = solve_steady_state(simulator)
    ratio = (solution.exhaust_sensible_kw + solution.exhaust_latent_kw) / solution.diagnostics[
        "power_total_kw"
    ]
    assert 0.80 < ratio < 0.95
    assert ratio == pytest.approx(0.8792, abs=1e-3)


def test_feasibility_flag_false_when_seed_flow_pushes_diameter_below_fluidization_window(
    config: ProcessConfig,
) -> None:
    """Eine große Keimzufuhr drückt d50 unter das Fluidisierungsfenster.

    Hier 2 kg/h gegen ~3.6 g/h Keimbildung aus dem Sprühfeststoff (siehe
    Moduldocstring) -- u_gas übersteigt dann u_t(d50) (Austrag/Elutriation).
    """
    operation = dataclasses.replace(config.operation, seed_mass_flow_kg_per_h=2.0)
    seeded_config = dataclasses.replace(config, operation=operation)
    simulator = GreyBoxFluidBedGranulator(seeded_config)
    solution = solve_steady_state(simulator)
    assert solution.diameter_m * 1e6 < 200.0
    assert not solution.feasible
    assert any("u_t" in reason for reason in solution.infeasibility_reasons)


@pytest.mark.parametrize("target_diameter_um", [750.0, 600.0, 900.0])
def test_matches_transient_integration_at_multiple_operating_points(
    config: ProcessConfig, target_diameter_um: float
) -> None:
    """Vergleich mit einer transienten Integration über 150 000 s.

    Zwei verschiedene Toleranzlogiken für zwei verschiedene Zeitskalen -- bewusst, nicht
    vereinheitlicht:

    * Wassermasse und Betttemperatur (tau bis 600 s, siehe time_constants) sind nach
      150 000 s (>= 250 tau) auf Maschinenpräzision eingeschwungen: 1e-6 relativ, wie
      ursprünglich beauftragt.
    * Partikelanzahl/d50 hängen an der langsamsten Mode (tau ~ 16000 s). Bei 150 000 s
      sind das nur ~9.4 tau; ein reiner Endwertvergleich mit 1e-6 relativ ist hier
      strukturell unerreichbar (exp(-150000/16000) ~ 8.5e-5) UND das falsche Kriterium:
      genau dieser Endwert-bei-zu-kurzer-Laufzeit-Fehlschluss war der defekte Befund
      (610.1 µm statt 750.0 µm bei duration_s=30000, siehe docs/04_stand_ap1.md, Befund
      A). Stattdessen wird geprüft, dass sich der Transient dem Löserwert **mit der
      erwarteten Abklingrate** nähert -- eine Exponentialhüllkurve auf dem relativen
      Restabstand, mit großzügigem Vorfaktor (die Mode ist nicht exakt exponentiell,
      d50 hängt über die dritte Wurzel an N). tau_slow kommt aus time_constants, nicht
      hartkodiert. Von Hand geprüft: ersetzt man den Löserwert probeweise durch den
      alten, falschen 610.1-µm-Wert, verletzt der Transient diese Hüllkurve um mehr als
      drei Größenordnungen (Restverhältnis ~0.67 gegen eine Schranke ~8e-4) -- der Test
      ist scharf genug, um genau diesen Fehler zu fangen.
    """
    fraction = nucleation_fraction_for_target_diameter(
        target_diameter_um * 1e-6, config.material.nucleus_diameter_m
    )
    material = dataclasses.replace(config.material, nucleation_solid_fraction=fraction)
    point_config = dataclasses.replace(config, material=material)
    simulator = GreyBoxFluidBedGranulator(point_config)
    solution = solve_steady_state(simulator)

    n_steps = int(_TRANSIENT_DURATION_S / simulator.sample_time_s)
    actions = np.repeat(simulator.nominal_action(1)[:, None, :], n_steps, axis=1)
    disturbances = np.repeat(simulator.nominal_disturbance(1)[:, None, :], n_steps, axis=1)
    trajectory = simulator.rollout(actions, disturbances, seed=0)

    water_transient = float(trajectory.states[0, -1, IDX_WATER_MASS])
    temperature_transient = float(trajectory.states[0, -1, IDX_BED_TEMPERATURE])
    assert water_transient == pytest.approx(solution.water_mass_kg, rel=1e-6)
    assert temperature_transient == pytest.approx(solution.bed_temperature_celsius, rel=1e-6)

    backend = simulator.backend
    state = backend.asarray(solution.state[None, :])
    action = backend.asarray(simulator.nominal_action(1))
    disturbance = backend.asarray(simulator.nominal_disturbance(1))
    tau_report = simulator.time_constants(state, action, disturbance)
    tau_slow = max(v for k, v in tau_report.items() if k.startswith("tau_"))
    envelope = _ENVELOPE_FACTOR * float(np.exp(-_TRANSIENT_DURATION_S / tau_slow))

    d50_start = trajectory.diagnostics["d50_m"][0, 0]
    d50_transient = trajectory.diagnostics["d50_m"][0, -1]
    d50_ratio = abs(d50_transient - solution.diameter_m) / abs(d50_start - solution.diameter_m)
    assert d50_ratio <= envelope

    n_start = float(trajectory.states[0, 0, IDX_PARTICLE_COUNT])
    n_transient = float(trajectory.states[0, -1, IDX_PARTICLE_COUNT])
    n_ratio = abs(n_transient - solution.particle_count) / abs(n_start - solution.particle_count)
    assert n_ratio <= envelope


def test_solve_steady_state_defaults_do_not_reproduce_befund_52() -> None:
    """WA2: die Loeser-DEFAULTS reproduzieren Befund 52 NICHT.

    Falle fuer kuenftige Aufrufer, siehe
    :func:`~twinlink_ml.simulation.steady_state.solve_steady_state`s
    ``max_iterations``/``tolerance``-Docstring. Gitterpunkt
    :data:`_KNOWN_FAILING_GRID_INDEX` scheitert mit den Loeser-Defaults (100
    Bisektionsschritte/Toleranz 1e-13) an einem Bracket-Vorzeichenfehler in der aeusseren
    Temperatur-Bisektion; mit Befund 52s eigener SUCH-Praezision
    (``DEFAULT_SEARCH_MAX_ITERATIONS``/``DEFAULT_SEARCH_TOLERANCE``, 30/1e-7) loest
    derselbe Punkt bitgleich zum dort berichteten ``energy_target``. Ein Regressionstest,
    kein Rauchtest: schlaegt fehl, falls eine kuenftige Aenderung des Loesers oder seiner
    Defaults diesen Bracket-Fehlschlag stillschweigend behebt oder verschaerft, ohne dass
    das an dieser Stelle sichtbar wuerde.
    """
    regret_module = _load_regret_module()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base_config = load_process_config(_GAB_CONFIG_PATH)
    anchor = json.loads(_ANCHOR_ARTIFACT_PATH.read_text(encoding="utf-8"))
    point = anchor["grid_results"][_KNOWN_FAILING_GRID_INDEX]

    variant_config = regret_module.build_variant_config(
        base_config,
        t_in=point["inlet_temperature_celsius"],
        u_gas=point["gas_velocity_m_per_s"],
        p_nozzle=point["nozzle_pressure_bar"],
        t_amb=point["ambient_temperature_celsius"],
        phi_amb=point["ambient_relative_humidity"],
    )
    simulator = GreyBoxFluidBedGranulator(variant_config)

    with pytest.raises(RuntimeError, match="kein Vorzeichenwechsel"):
        solve_steady_state(simulator)

    solution = solve_steady_state(
        simulator,
        max_iterations=regret_module.DEFAULT_SEARCH_MAX_ITERATIONS,
        tolerance=regret_module.DEFAULT_SEARCH_TOLERANCE,
    )
    assert solution.diagnostics["energy_target"] == pytest.approx(
        point["energy_target"], rel=1e-6
    )
