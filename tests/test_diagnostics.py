"""Tests für ``steady_state_report``.

Vor Schritt 3 gab es hierfür keine Tests -- der Bericht war dadurch nie gegen
konkrete Zahlenwerte abgesichert, was den 610.1-µm-Fehlbefund unentdeckt ließ (siehe
docs/04_stand_ap1.md, Befund A). Diese Datei ist neu, keine Anpassung bestehender Tests.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    ProcessConfig,
    load_process_config,
)
from twinlink_ml.simulation.diagnostics import (
    _match_single_mode,
    steady_state_report,
    time_constant_report,
)

GAB_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)


def test_steady_state_report_matches_exact_solver_not_the_old_transient_value(
    config: ProcessConfig,
) -> None:
    """Regression für Befund A: 750.0 µm, nicht die frühere 610.1 µm bei duration_s=30000."""
    simulator = GreyBoxFluidBedGranulator(config)
    report = steady_state_report(simulator)
    assert report.diameter_um == pytest.approx(750.0, rel=1e-6)
    assert abs(report.diameter_um - 610.1) > 100.0


def test_steady_state_report_carries_feasibility_and_residual(config: ProcessConfig) -> None:
    simulator = GreyBoxFluidBedGranulator(config)
    report = steady_state_report(simulator)
    assert report.feasible is True
    assert report.residual_inf_norm < 1e-10


def test_steady_state_report_format_shows_feasibility_and_residual(
    config: ProcessConfig,
) -> None:
    simulator = GreyBoxFluidBedGranulator(config)
    report = steady_state_report(simulator)
    text = report.format()
    assert "Zulässig" in text
    assert "ja" in text
    assert "Residuum" in text


def test_time_constant_report_matches_transient_within_measured_noise_floor(
    config: ProcessConfig,
) -> None:
    """Löser- und Transienten-Linearisierungspunkt vergleichen (Schritt 3b).

    Zwei verschiedene Fließkomma-Pfade -- RK4-Integration über 6000 Schritte gegen
    verschachtelte Bisektion -- landen auf demselben physikalischen Beharrungszustand
    numerisch nicht bitgleich; ein bitgleiches Kriterium war hier unerfüllbar (siehe
    docs/04_stand_ap1.md, Befund A). Gemessenes Rauschniveau der Methode liegt bei
    ~1e-10 relativ; die Toleranz 1e-8 lässt zwei Größenordnungen Luft, damit der Test
    nicht bei einem NumPy- oder Plattformwechsel flackert, würde aber jede
    strukturelle Änderung (die um Größenordnungen darüber läge) weiterhin fangen.

    Ausnahme: ``tau_3_s`` (Feststoff-Regelkreis) hat als Steigung exakt
    ``-1/mass_control_time_constant_s`` -- eine reine Config-Konstante, unabhängig vom
    numerischen Pfad. Dafür ist Bitgleichheit die schärfere, korrekt erwartbare
    Aussage, nicht nur eine Toleranz.
    """
    simulator = GreyBoxFluidBedGranulator(config)

    # Altes Verhalten reproduziert (nicht mehr Teil von time_constant_report seit
    # Schritt 3b): transiente Integration, dann Zeitkonstanten am Endzustand.
    n_steps = int(30_000.0 / simulator.sample_time_s)
    actions = np.repeat(simulator.nominal_action(1)[:, None, :], n_steps, axis=1)
    disturbances = np.repeat(simulator.nominal_disturbance(1)[:, None, :], n_steps, axis=1)
    trajectory = simulator.rollout(actions, disturbances, seed=0)
    backend = simulator.backend
    transient_state = backend.asarray(trajectory.states[:, -1, :])
    action = backend.asarray(simulator.nominal_action(1))
    disturbance = backend.asarray(simulator.nominal_disturbance(1))
    transient_report = simulator.time_constants(transient_state, action, disturbance)

    solver_report = time_constant_report(simulator)

    for key in ("tau_1_s", "tau_2_s", "tau_4_s"):
        assert solver_report[key] == pytest.approx(transient_report[key], rel=1e-8)
    assert solver_report["stiffness_ratio"] == pytest.approx(
        transient_report["stiffness_ratio"], rel=1e-8
    )
    assert solver_report["tau_3_s"] == transient_report["tau_3_s"]


def test_match_single_mode_requires_exactly_one_hit_within_tolerance() -> None:
    """N2: Wertevergleich statt Rangposition -- s. ``time_constant_report``-Docstring."""
    taus = [10.0, 100.0, 600.0, 16000.0]
    assert _match_single_mode(taus, 600.0, rtol=1e-6) == 2
    assert _match_single_mode(taus, 600.0 * (1.0 + 1e-9), rtol=1e-6) == 2
    assert _match_single_mode(taus, 601.0, rtol=1e-6) is None
    assert _match_single_mode([600.0, 600.0000001, 16000.0], 600.0, rtol=1e-4) is None


def test_time_constant_report_identifies_structural_modes_by_value(
    config: ProcessConfig,
) -> None:
    """N2: ``tau_mass_control_s``/``tau_residence_s`` treffen die Konfigurationskonstanten.

    Bitgleich mit ``operation.mass_control_time_constant_s`` -- dieselbe Begründung wie
    beim bestehenden ``tau_3_s``-Bitgleichheits-Test oben (reine Konfigurationskonstante,
    pfadunabhängig).
    """
    simulator = GreyBoxFluidBedGranulator(config)
    report = time_constant_report(simulator)

    assert report["mode_identification_note"] is None
    assert report["tau_mass_control_s"] == pytest.approx(
        config.operation.mass_control_time_constant_s, rel=1e-6
    )
    assert report["tau_residence_s"] == pytest.approx(16_000.0, rel=1e-6)

    coupled_fast = report["tau_coupled_fast_s"]
    coupled_slow = report["tau_coupled_slow_s"]
    assert coupled_fast < coupled_slow
    all_taus = sorted(v for k, v in report.items() if k.startswith("tau_") and k[4].isdigit())
    assert sorted(
        [coupled_fast, coupled_slow, report["tau_mass_control_s"], report["tau_residence_s"]]
    ) == pytest.approx(all_taus, rel=1e-8)

    dt = report["sample_time_s"]
    assert report["recommended_sequence_length_coupled_slow"] == float(
        np.ceil(3.0 * coupled_slow / dt)
    )


def test_time_constant_report_mode_identity_is_stable_across_rank_swap() -> None:
    """N2-Regression: die tau_i-Rangposition der mass_control-Mode ist nicht stabil.

    An der GAB-Config (``greybox_gab_placeholder.yaml``, Befund 47/M1/49) liegt der
    gekoppelte langsame Zweig bei 60 °C Eintrittstemperatur oberhalb von 600 s (Rang
    wandert auf ``tau_3_s``), bei 90 °C unterhalb (``tau_2_s``) -- die Namen ``tau_2_s``/
    ``tau_3_s`` sind also keine stabile Identität. Die Identifikation über den Zahlenwert
    bleibt davon unberührt und liefert an beiden Betriebspunkten
    ``tau_mass_control_s`` == ``mass_control_time_constant_s``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base_config = load_process_config(GAB_CONFIG_PATH)

    cold_bounds = dataclasses.replace(base_config.actions.inlet_temperature_celsius, nominal=60.0)
    cold_config = dataclasses.replace(
        base_config,
        actions=dataclasses.replace(base_config.actions, inlet_temperature_celsius=cold_bounds),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        cold_simulator = GreyBoxFluidBedGranulator(cold_config)
        default_simulator = GreyBoxFluidBedGranulator(base_config)
    cold_report = time_constant_report(cold_simulator)
    default_report = time_constant_report(default_simulator)

    assert cold_report["tau_coupled_slow_s"] > cold_report["tau_mass_control_s"]
    assert default_report["tau_coupled_slow_s"] < default_report["tau_mass_control_s"]

    assert cold_report["tau_mass_control_s"] == pytest.approx(
        base_config.operation.mass_control_time_constant_s, rel=1e-6
    )
    assert default_report["tau_mass_control_s"] == pytest.approx(
        base_config.operation.mass_control_time_constant_s, rel=1e-6
    )
    assert cold_report["mode_identification_note"] is None
    assert default_report["mode_identification_note"] is None


def test_time_constant_report_mode_identification_none_when_ambiguous(
    config: ProcessConfig,
) -> None:
    """N2: mehrdeutiger Treffer wird nicht geraten, sondern als ``None`` mit Notiz gemeldet."""
    simulator = GreyBoxFluidBedGranulator(config)
    report = time_constant_report(simulator, mode_match_rtol=1.0)

    assert report["tau_mass_control_s"] is None
    assert report["tau_residence_s"] is None
    assert report["tau_coupled_fast_s"] is None
    assert report["tau_coupled_slow_s"] is None
    assert report["recommended_sequence_length_coupled_slow"] is None
    assert report["mode_identification_note"] is not None
