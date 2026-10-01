r"""S4: Sicherheitsprojektion von ``gas_velocity_m_per_s`` auf ``[u_mf(d50), u_terminal(d50)]``.

Config: ``conf/process/greybox_gab_placeholder.yaml`` mit ``sample_time_s=25.0`` (wie
``twinlink_ml.evaluation.generate_data``/``DataSpec`` es setzt) -- NICHT die
``config``-Fixture aus ``conftest.py`` (``greybox_placeholder.yaml``, andere Datei).

``test_gas_velocity_projection_disabled_matches_legacy_baseline`` ist der in S3e
geforderte Regressionstest: er belegt die Aequivalenz zwischen dem alten
``process/fingerprint`` ``82f65f576a2e5253`` (vor Einfuehrung von
``enforce_gas_velocity_projection``) und dem neuen ``40aae2b4526a5ed7`` (danach, Schalter
AUS) -- beide liefern bitgleiche Trajektorien fuer dieselbe Anregung. Ohne diesen Beleg
waere in einigen Monaten nicht mehr feststellbar, ob ein mit dem alten und ein mit dem
neuen Fingerprint erzeugter Datensatz vergleichbar sind.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.simulation.fluidization import (
    minimum_fluidization_velocity_m_per_s,
    terminal_velocity_m_per_s,
)
from twinlink_ml.simulation.greybox import (
    IDX_BED_TEMPERATURE,
    IDX_PARTICLE_COUNT,
    IDX_SOLID_MASS,
    GreyBoxFluidBedGranulator,
)
from twinlink_ml.simulation.thermo import absolute_humidity_kg_per_kg

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
BASELINE_PATH = (
    Path(__file__).resolve().parent / "data" / "gas_velocity_projection_regression_baseline.npz"
)
LEGACY_FINGERPRINT = "82f65f576a2e5253"
"""``process/fingerprint`` VOR Einfuehrung von ``enforce_gas_velocity_projection`` -- der
Wert, mit dem ``tests/data/gas_velocity_projection_regression_baseline.npz`` erzeugt wurde
(Code-Stand ``29fa8d8``, vor dem Schema-Zusatz)."""
NEW_FINGERPRINT_PROJECTION_OFF = "6ecedc6bd993fb3d"
"""``process/fingerprint`` AKTUELL, mit ``enforce_gas_velocity_projection=False`` --
unterschiedlich von ``LEGACY_FINGERPRINT``, weil der Fingerprint ueber die gesamte Config
hasht (jedes neue Feld aendert ihn, auch bei unveraendertem Wert). Kette bei Schalter AUS,
numerisch aequivalent: ``82f65f576a2e5253`` (vor ``enforce_gas_velocity_projection``) ->
``40aae2b4526a5ed7`` (nach Einfuehrung des Schalters) -> ``078572b3a7a0fb45`` (nach
Einfuehrung von ``ExhaustConfig``) -> ``6ecedc6bd993fb3d`` (Pseudonymisierungs-Schnitt
2026-09-25, ``provenance.open_todos`` aus dem Hash ausgeschlossen, siehe
``twinlink_ml.utils.fingerprint_history``). Die ersten drei Schritte aendern nur Schema
(neues Feld mit identitaetserhaltendem Default), keine Physik --
``test_gas_velocity_projection_disabled_matches_legacy_baseline`` belegt das jeweils gegen
dasselbe eingefrorene Trajektorien-Fixture; der letzte Schritt ist durch den
Aequivalenznachweis in ``twinlink_ml.utils.fingerprint_history`` belegt (reine
Dokumentationsaenderung, keine Physik, kein Schema)."""


def _load_config(*, projection_active: bool = False) -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    return dataclasses.replace(
        config,
        numerics=dataclasses.replace(
            config.numerics,
            sample_time_s=25.0,
            enforce_gas_velocity_projection=projection_active,
        ),
    )


def _random_actions_disturbances(
    n_batch: int, n_time: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Aktionen/Stoergroessen ueber den GESAMTEN Stellbereich, nicht ratenlimitiert.

    Bewusst aggressiver als das echte Anregungsdesign, um die Projektion (und die in S1/T1
    charakterisierte Restverletzung) mit wenigen Laeufen/Schritten zuverlaessig und schnell
    auszuloesen.
    """
    rng = np.random.default_rng(seed)
    actions = np.stack(
        [
            rng.uniform(60.0, 120.0, size=(n_batch, n_time)),
            rng.uniform(0.8, 2.5, size=(n_batch, n_time)),
            rng.uniform(0.4, 3.0, size=(n_batch, n_time)),
            rng.uniform(1.0, 4.0, size=(n_batch, n_time)),
        ],
        axis=-1,
    )
    disturbances = np.stack(
        [
            rng.uniform(5.0, 35.0, size=(n_batch, n_time)),
            rng.uniform(0.25, 0.85, size=(n_batch, n_time)),
            rng.uniform(0.20, 0.40, size=(n_batch, n_time)),
        ],
        axis=-1,
    )
    return actions, disturbances


def test_gas_velocity_projection_disabled_matches_legacy_baseline() -> None:
    """S3e-Regressionstest: Schalter AUS liefert bitgleiche Trajektorien zum alten Stand.

    ``tests/data/gas_velocity_projection_regression_baseline.npz`` wurde mit dem
    Code-Stand VOR Einfuehrung von ``enforce_gas_velocity_projection`` erzeugt
    (``process/fingerprint`` == ``LEGACY_FINGERPRINT``). Dieser Test rollt dieselbe
    Anregung mit dem AKTUELLEN Code aus (``process/fingerprint`` ==
    ``NEW_FINGERPRINT_PROJECTION_OFF``, siehe dort fuer die volle Kette
    ``82f65f576a2e5253`` -> ``40aae2b4526a5ed7`` -> ``078572b3a7a0fb45`` -- ein jeweils
    anderer Wert, weil der Fingerprint ueber die gesamte Config hasht und jedes neue Feld
    ihn auch bei unveraendertem Wert aendert) und verlangt bitgleiche Ergebnisse.
    """
    config = _load_config(projection_active=False)
    assert config.fingerprint == NEW_FINGERPRINT_PROJECTION_OFF, (
        "Fingerprint hat sich unerwartet weiterbewegt -- Baseline-Aequivalenz muss neu "
        "geprueft werden, bevor dieser Test angepasst wird"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(config)

    data = np.load(BASELINE_PATH)
    trajectory = simulator.rollout(data["actions"], data["disturbances"], seed=42)

    assert np.array_equal(trajectory.actions, data["actions"])
    assert np.array_equal(trajectory.states, data["states"])
    assert np.array_equal(trajectory.outputs, data["outputs"])
    assert np.array_equal(trajectory.measurements, data["measurements"])
    assert trajectory.projection_active is False
    assert trajectory.requested_actions is None
    assert trajectory.n_projected_steps is None


def test_project_action_clips_to_pre_step_margins() -> None:
    """Eigenschaft von ``project_action`` selbst, ohne Integration dazwischen.

    Die geclippte Geschwindigkeit liegt exakt in ``[u_mf, u_terminal]``, berechnet aus
    GENAU DEM ZUSTAND, der ``project_action`` uebergeben wurde -- unabhaengig von jeder
    Drift ueber einen Integrationsschritt (siehe die Restverletzungs-Tests unten, die
    genau diese Drift betreffen). ``u_mf``/``u_terminal`` werden hier unabhaengig ueber
    ``fluidization.py`` nachgerechnet, nicht aus der Simulator-Diagnose uebernommen.
    """
    config = _load_config(projection_active=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(config)
    backend = simulator.backend
    material, gas = config.material, config.gas

    state = np.array([[2.0, 0.1, 5.0e9, 55.0], [2.0, 0.1, 5.0e9, 55.0]])
    disturbance = np.array([[20.0, 0.5, 0.3], [20.0, 0.5, 0.3]])
    # gas_velocity: 0.001 liegt unter u_mf (~0.0029 m/s fuer diesen Zustand), 5.0 liegt
    # ueber u_terminal (~0.221 m/s) -- project_action() clippt hier ohne Box-Constraint
    # (das ist eine separate Pruefung in _clip_action, nicht Gegenstand dieses Tests)
    action = np.array([[90.0, 0.001, 1.5, 2.5], [90.0, 5.0, 1.5, 2.5]])

    projected = simulator.project_action(state, action, disturbance)

    solid_mass = state[..., IDX_SOLID_MASS]
    particle_count = state[..., IDX_PARTICLE_COUNT]
    rho_p = material.particle_density_kg_per_m3
    diameter = (6.0 * solid_mass / (np.pi * rho_p * particle_count)) ** (1.0 / 3.0)
    bed_temperature = state[..., IDX_BED_TEMPERATURE]
    humidity = absolute_humidity_kg_per_kg(backend, gas, disturbance[..., 0], disturbance[..., 1])
    u_mf = minimum_fluidization_velocity_m_per_s(
        backend, gas, material, diameter, bed_temperature, humidity
    )
    u_terminal = terminal_velocity_m_per_s(
        backend, gas, material, diameter, bed_temperature, humidity
    )

    expected_gas_velocity = np.clip(action[..., 1], u_mf, u_terminal)
    assert np.array_equal(projected[..., 1], expected_gas_velocity)
    assert bool((projected[..., 1] != action[..., 1]).all()), "muss beide Richtungen clippen"
    # kein anderer Kanal veraendert
    assert np.array_equal(projected[..., [0, 2, 3]], action[..., [0, 2, 3]])


def test_project_action_is_identity_when_switch_off() -> None:
    """Bitgleiches Verhalten bei ausgeschaltetem Schalter, isoliert auf project_action."""
    config = _load_config(projection_active=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(config)
    assert simulator.projection_active is False

    state = np.array([[2.0, 0.1, 5.0e9, 55.0]])
    disturbance = np.array([[20.0, 0.5, 0.3]])
    action = np.array([[90.0, 5.0, 1.5, 2.5]])  # 5.0 m/s -- weit ausserhalb jedes Fensters

    projected = simulator.project_action(state, action, disturbance)
    assert np.array_equal(projected, action)


def test_projection_active_but_never_engaged() -> None:
    """S4-Fall: Projektion aktiv, greift aber an keinem Punkt.

    Nominelle Konstantfahrweise (400 um Start-d50, 90 C, 1.6 m/s Gasgeschwindigkeit) --
    ``u_mf``/``u_terminal`` liegen ueber 50 Schritte bei ~0.07/~2.1 m/s (per Diagnose bei
    ausgeschaltetem Schalter unabhaengig bestaetigt), die nominelle Gasgeschwindigkeit
    bleibt damit durchgehend im Fenster.
    """
    config = _load_config(projection_active=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(config)

    n_batch, n_time = 3, 50
    actions = np.zeros((n_batch, n_time, 4))
    actions[..., 0] = 90.0
    actions[..., 1] = 1.6
    actions[..., 2] = 1.5
    actions[..., 3] = 2.5
    disturbances = np.zeros((n_batch, n_time, 3))
    disturbances[..., 0] = 20.0
    disturbances[..., 1] = 0.5
    disturbances[..., 2] = 0.3

    trajectory = simulator.rollout(actions, disturbances, seed=0)

    assert trajectory.projection_active is True
    assert trajectory.n_projected_steps is not None
    assert np.array_equal(trajectory.n_projected_steps, np.zeros(n_batch, dtype=np.int64))
    assert trajectory.requested_actions is not None
    assert np.array_equal(trajectory.requested_actions, trajectory.actions)
    assert np.array_equal(trajectory.requested_actions, actions)


def test_requested_actions_differs_exactly_where_projected() -> None:
    """``requested_actions``/``actions`` unterscheiden sich genau an den geclippten Punkten.

    Nur im ``gas_velocity_m_per_s``-Kanal, und ``n_projected_steps`` zaehlt sie korrekt.
    """
    config = _load_config(projection_active=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(config)
    gas_velocity_index = config.actions.names.index("gas_velocity_m_per_s")

    actions, disturbances = _random_actions_disturbances(8, 100, seed=11)
    trajectory = simulator.rollout(actions, disturbances, seed=0)

    assert trajectory.requested_actions is not None
    assert trajectory.n_projected_steps is not None
    diff_per_channel = trajectory.actions != trajectory.requested_actions
    # nur gas_velocity_m_per_s darf je differieren
    other_channels = [i for i in range(actions.shape[-1]) if i != gas_velocity_index]
    assert not diff_per_channel[..., other_channels].any()
    diff_any_channel = diff_per_channel.any(axis=-1)
    assert np.array_equal(diff_any_channel.sum(axis=1), trajectory.n_projected_steps)
    assert diff_any_channel.sum() > 0, "Testszenario muss tatsaechlich Punkte clippen"


def test_trajectory_select_and_concatenate_preserve_projection_fields() -> None:
    """``select``/``concatenate`` reichen die drei neuen Felder korrekt durch."""
    config = _load_config(projection_active=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(config)
    actions, disturbances = _random_actions_disturbances(4, 80, seed=13)
    trajectory = simulator.rollout(actions, disturbances, seed=0, run_id=("a", "b", "c", "d"))
    assert trajectory.n_projected_steps is not None and trajectory.requested_actions is not None

    selected = trajectory.select(2)
    assert selected.run_id == ("c",)
    assert selected.projection_active is True
    assert np.array_equal(selected.n_projected_steps, trajectory.n_projected_steps[2:3])
    assert np.array_equal(selected.requested_actions, trajectory.requested_actions[2:3])

    merged = trajectory.select(0).concatenate(trajectory.select(1))
    assert merged.run_id == ("a", "b")
    assert merged.projection_active is True
    assert np.array_equal(merged.n_projected_steps, trajectory.n_projected_steps[0:2])
    assert np.array_equal(merged.requested_actions, trajectory.requested_actions[0:2])


def test_concatenate_rejects_mismatched_projection_active() -> None:
    """Trajektorien mit und ohne Projektion duerfen nicht stillschweigend vermischt werden.

    Sonst waeren projizierte und unprojizierte Laeufe in einem Datensatz ununterscheidbar.
    """
    config_off = _load_config(projection_active=False)
    config_on = _load_config(projection_active=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        sim_off = GreyBoxFluidBedGranulator(config_off)
        sim_on = GreyBoxFluidBedGranulator(config_on)
    actions, disturbances = _random_actions_disturbances(2, 20, seed=17)

    trajectory_off = sim_off.rollout(actions, disturbances, seed=0)
    trajectory_on = sim_on.rollout(actions, disturbances, seed=0)

    with pytest.raises(ValueError, match="projection_active"):
        trajectory_off.concatenate(trajectory_on)


def test_trajectory_rejects_inconsistent_projection_fields() -> None:
    """``__post_init__`` verlangt Konsistenz zwischen den drei neuen Feldern.

    Aktiv ohne Wirkungsnachweis, oder Wirkungsnachweis ohne aktiv, sind beides
    widerspruechliche Zustaende.
    """
    base_kwargs = {
        "run_id": ("r0",),
        "time_s": np.array([25.0]),
        "actions": np.zeros((1, 1, 4)),
        "disturbances": np.zeros((1, 1, 3)),
        "states": np.zeros((1, 1, 4)),
        "outputs": np.zeros((1, 1, 4)),
        "measurements": np.zeros((1, 1, 4)),
    }
    with pytest.raises(ValueError, match="projection_active=True"):
        Trajectory(**base_kwargs, projection_active=True)
    with pytest.raises(ValueError, match="projection_active=False"):
        Trajectory(
            **base_kwargs,
            projection_active=False,
            requested_actions=np.zeros((1, 1, 4)),
            n_projected_steps=np.zeros((1,), dtype=np.int64),
        )


def test_projection_residual_violation_within_derived_bound() -> None:
    r"""Restverletzung: Betrag und Haeufigkeit unter abgeleiteten Schranken.

    Keine Verletzung der unteren Grenze erwartet. Ursache der oberen
    (S1/T1, ``scripts/probe_gas_velocity_projection.py``): die Projektion clippt
    auf ``u_terminal`` am WAHREN Zustand VOR dem Schritt (Vorgabe), die hier gepruefte
    Diagnose (``fluidization_margin_upper_m_per_s``) berechnet ``u_terminal`` aber NACH der
    Integration. ``d50`` kann innerhalb eines Abtastschritts fallen, wenn die Keimbildung
    die Partikelzahl schneller erhoeht als die Feststoffmasse waechst -- dann sinkt
    ``u_terminal`` unter den zu Schrittbeginn zulaessigen Wert, und die (am Schrittanfang
    korrekt geclippte) Geschwindigkeit liegt am Schrittende minimal darueber. Das ist eine
    strukturelle Grenze jedes Ein-Schritt-Vorgriffs unter Zero-Order-Hold, keine falsch
    platzierte Projektion (siehe ``f_root_cause`` im S1-Artefakt).

    Schranken NICHT grosszuegig gerundet, sondern knapp oberhalb des mit diesem exakten
    Szenario gemessenen Werts (deterministisch reproduziert, seed fixiert): relative
    Ueberschreitung max. 0.4184 % (Schranke 0.45 %), Anteil betroffener Punkte 5.078 %
    (Schranke 5.5 %) von 6400 Punkten. Diese Zahlen sind szenariospezifisch (aggressiv
    randomisierte, ratenunlimitierte Aktionen -- deutlich haerter als das echte
    Anregungsdesign) und NICHT mit den gepoolten S1-Werten (95-Perzentil 0.17 %, 3 Seeds x
    128 Laeufe) zu verwechseln.
    """
    config = _load_config(projection_active=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(config)
    actions, disturbances = _random_actions_disturbances(16, 400, seed=7)

    trajectory = simulator.rollout(actions, disturbances, seed=3)

    margin_upper = trajectory.diagnostics["fluidization_margin_upper_m_per_s"]
    margin_lower = trajectory.diagnostics["fluidization_margin_lower_m_per_s"]
    violation_upper = margin_upper < 0.0

    assert not (margin_lower < 0.0).any(), "u < u_mf nach Projektion -- das darf nie passieren"

    n_violating = int(violation_upper.sum())
    assert n_violating > 0, "Testszenario muss die Restverletzung tatsaechlich ausloesen"
    violation_fraction = violation_upper.mean()
    assert violation_fraction < 0.055, (
        f"Anteil betroffener Punkte gestiegen: {violation_fraction:.6%}"
    )

    u_terminal_post = trajectory.diagnostics["terminal_velocity_m_per_s"]
    exceedance_abs = np.maximum(0.0, -margin_upper)
    relative_exceedance = exceedance_abs[violation_upper] / u_terminal_post[violation_upper]
    assert relative_exceedance.max() < 0.0045, (
        f"relative Ueberschreitung gestiegen: {relative_exceedance.max():.4%}"
    )
