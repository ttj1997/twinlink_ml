r"""VA2: Orakelpruefung -- die Umgebung muss Befund 52 bitgleich reproduzieren.

Wichtigster Test dieses Auftrags (siehe Auftragstext): eine Umgebung, die das bekannte
Optimum aus ``artifacts/runs/bb1_bb3_operating_point_regret.json`` nicht reproduziert, ist
fuer RL wertlos. Drei Pruefungen:

a) Orakelaktionen an allen 25 Gitterpunkten -> Belohnung exakt ``-energy_target``.
b) Fester Betriebspunkt an allen 25 Gitterpunkten -> Regret-Statistik (Median/P95/Max,
   absolut und relativ) reproduziert die dokumentierten Werte.
c) Zulaessigkeit an allen 25 Punkten fuer beide Aktionsmengen.

Massstabsfrage (Duesendruck fest vs. je Punkt frei optimiert, siehe Auftrag "Achtung
Massstab"): empirisch bitgleich (siehe ``test_fixed_nozzle_pressure_scale_is_negligible``)
-- der Duesendruck-Freiheitsgrad der Befund-52-Optima bewegt sich durchgehend in
``[1.0002, 1.0111]`` bar, unterhalb der Clamp-Schwelle ``gas.pressure_pa / 1e5 = 1.01325``
bar in ``energy.power_breakdown_kw`` (``max(pressure_ratio, 1 + 1e-12)``) -- der
Atomiser-Anteil an ``energy_target`` ist in diesem gesamten Fenster exakt konstant (~0),
unabhaengig vom genauen Duesendruck.

Loeser-Praezision: alle Aufrufe hier laufen ueber :meth:`OperatingPointBanditEnv.
evaluate_operating_point`, die intern die SUCH-Praezision aus Befund 52 verwendet
(``DEFAULT_SEARCH_MAX_ITERATIONS``/``DEFAULT_SEARCH_TOLERANCE``), nicht die Loeser-Defaults
-- siehe ``env.py``-Moduldocstring fuer die Begruendung (Bracket-Vorzeichenfehler bei 6 von
25 Punkten mit den Loeser-Defaults, empirisch geprueft).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from gymnasium import spaces

from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv
from twinlink_ml.rl.env import (
    ANCHOR_ARTIFACT_PATH,
    CONFIG_PATH,
    EXPECTED_CONFIG_FINGERPRINT,
    FIXED_NOZZLE_PRESSURE_BAR,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent

REL_TOLERANCE = 1e-6
"""Grosszuegig ueber dem beobachteten Gleitkommarauschen (~1e-9 relativ, siehe Auftrag/
Analyse) -- eng genug, echte Regressionen von Rauschen zu unterscheiden."""


@pytest.fixture(scope="module")
def anchor() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(ANCHOR_ARTIFACT_PATH).read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def env() -> OperatingPointBanditEnv:
    return OperatingPointBanditEnv()


def test_config_fingerprint_matches_anchor_artifact(anchor: dict[str, Any]) -> None:
    """Voraussetzung fuer jeden Vergleich unten: gleicher Ausgangspunkt wie Befund 52.

    ``anchor["config_fingerprint"]`` traegt den eingefrorenen Wert von vor dem
    Pseudonymisierungs-Schnitt (siehe ``EXPECTED_CONFIG_FINGERPRINT``-Docstring in
    ``env.py``) -- :func:`fingerprints_equivalent` erkennt beide als dieselbe numerische
    Konfiguration.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    assert config.fingerprint == EXPECTED_CONFIG_FINGERPRINT
    assert fingerprints_equivalent(EXPECTED_CONFIG_FINGERPRINT, anchor["config_fingerprint"])


def test_va2a_oracle_actions_reproduce_energy_target(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    for point in anchor["grid_results"]:
        reward, info = env.evaluate_operating_point(
            inlet_temperature_celsius=point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        assert not info["solve_failed"], (
            f"Gitterpunkt {point['_grid_index']}: Loeser schlaegt an der Orakelaktion fehl"
        )
        assert reward == pytest.approx(-point["energy_target"], rel=REL_TOLERANCE)
        assert info["energy_target"] == pytest.approx(point["energy_target"], rel=REL_TOLERANCE)


def test_va2c_oracle_actions_are_feasible(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    for point in anchor["grid_results"]:
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        assert info["feasible"] == point["feasible"]


def test_va2c_fixed_point_actions_are_feasible(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    fixed = anchor["fixed_point"]
    for point in fixed["per_grid_point"]:
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=fixed["inlet_temperature_celsius"],
            gas_velocity_m_per_s=fixed["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        assert info["feasible"] == point["feasible"]


def test_va2b_fixed_point_regret_matches_befund_52(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    fixed = anchor["fixed_point"]
    absolute_kw: list[float] = []
    relative_pct: list[float] = []
    for grid_point, _fixed_point_at_grid in zip(
        anchor["grid_results"], fixed["per_grid_point"], strict=True
    ):
        t_amb = grid_point["ambient_temperature_celsius"]
        phi_amb = grid_point["ambient_relative_humidity"]
        _, oracle_info = env.evaluate_operating_point(
            inlet_temperature_celsius=grid_point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=grid_point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=t_amb,
            ambient_relative_humidity=phi_amb,
        )
        _, fixed_info = env.evaluate_operating_point(
            inlet_temperature_celsius=fixed["inlet_temperature_celsius"],
            gas_velocity_m_per_s=fixed["gas_velocity_m_per_s"],
            ambient_temperature_celsius=t_amb,
            ambient_relative_humidity=phi_amb,
        )
        assert oracle_info["feasible"] and fixed_info["feasible"]
        regret = fixed_info["energy_target"] - oracle_info["energy_target"]
        absolute_kw.append(regret)
        relative_pct.append(100.0 * regret / oracle_info["energy_target"])

    absolute_arr = np.asarray(absolute_kw)
    relative_arr = np.asarray(relative_pct)
    documented = anchor["regret"]

    assert float(np.median(absolute_arr)) == pytest.approx(
        documented["absolute_kw"]["median"], rel=REL_TOLERANCE
    )
    assert float(np.percentile(absolute_arr, 95)) == pytest.approx(
        documented["absolute_kw"]["p95"], rel=REL_TOLERANCE
    )
    assert float(np.max(absolute_arr)) == pytest.approx(
        documented["absolute_kw"]["max"], rel=REL_TOLERANCE
    )
    assert float(np.median(relative_arr)) == pytest.approx(
        documented["relative_pct"]["median"], rel=REL_TOLERANCE
    )
    assert float(np.percentile(relative_arr, 95)) == pytest.approx(
        documented["relative_pct"]["p95"], rel=REL_TOLERANCE
    )
    assert float(np.max(relative_arr)) == pytest.approx(
        documented["relative_pct"]["max"], rel=REL_TOLERANCE
    )
    # Median 16.03 %, P95 22.2 %, Max 25.80 % -- Zahlen aus dem Auftrag, nur als
    # zusaetzliche, lesbare Regressionsschranke (die obigen pytest.approx-Vergleiche
    # gegen das Artefakt sind die eigentliche Pruefung).
    assert float(np.median(relative_arr)) == pytest.approx(16.03, abs=0.01)
    assert float(np.percentile(relative_arr, 95)) == pytest.approx(22.2, abs=0.1)
    assert float(np.max(relative_arr)) == pytest.approx(25.80, abs=0.01)


def test_fixed_nozzle_pressure_scale_is_negligible(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    """Auftrag, "Achtung Massstab".

    Duesendruck ist in der Umgebung fest, in den Befund-52-Optima frei -- quantifiziert die
    resultierende Abweichung. Vergleicht ``energy_target`` bei ``FIXED_NOZZLE_PRESSURE_BAR``
    (wie im Environment) gegen den jeweils PUNKTEIGENEN, frei optimierten Duesendruck --
    beide sonst am identischen Betriebspunkt. Die 0.4 % Spannweite aus Befund 52
    (``setpoint_spread.nozzle_pressure_bar``) bewegt sich vollstaendig unterhalb der
    Clamp-Schwelle in ``power_breakdown_kw`` -- die Abweichung ist deshalb nicht nur
    "innerhalb einer Toleranz", sondern numerisch nicht von Null unterscheidbar.
    """
    from twinlink_ml.rl.env import _load_regret_module
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator
    from twinlink_ml.simulation.steady_state import solve_steady_state

    regret_module = _load_regret_module()
    deltas_kw: list[float] = []
    for point in anchor["grid_results"]:
        variant_own_nozzle = regret_module.build_variant_config(
            env._base_config,
            t_in=point["inlet_temperature_celsius"],
            u_gas=point["gas_velocity_m_per_s"],
            p_nozzle=point["nozzle_pressure_bar"],
            t_amb=point["ambient_temperature_celsius"],
            phi_amb=point["ambient_relative_humidity"],
        )
        solution_own_nozzle = solve_steady_state(
            GreyBoxFluidBedGranulator(variant_own_nozzle),
            max_iterations=regret_module.DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=regret_module.DEFAULT_SEARCH_TOLERANCE,
        )
        _, fixed_nozzle_info = env.evaluate_operating_point(
            inlet_temperature_celsius=point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        deltas_kw.append(
            fixed_nozzle_info["energy_target"] - solution_own_nozzle.diagnostics["energy_target"]
        )

    max_abs_delta_kw = max(abs(delta) for delta in deltas_kw)
    # Beobachtet: ~3.3e-16 kW (Gleitkommarauschen). 1e-9 kW Schranke laesst reichlich Marge,
    # bleibt aber weit unter jeder physikalisch bedeutsamen Groessenordnung.
    assert max_abs_delta_kw < 1.0e-9
    assert env._base_config.gas.pressure_pa / 1.0e5 > FIXED_NOZZLE_PRESSURE_BAR


def test_reset_with_explicit_disturbance_is_deterministic(
    env: OperatingPointBanditEnv,
) -> None:
    obs1, info1 = env.reset(
        options={"ambient_temperature_celsius": 12.5, "ambient_relative_humidity": 0.6}
    )
    obs2, info2 = env.reset(
        options={"ambient_temperature_celsius": 12.5, "ambient_relative_humidity": 0.6}
    )
    np.testing.assert_array_equal(obs1, obs2)
    assert info1 == info2


def test_reset_without_options_samples_within_disturbance_bounds() -> None:
    env = OperatingPointBanditEnv()
    obs, _ = env.reset(seed=0)
    observation_space = env.observation_space
    assert isinstance(observation_space, spaces.Box)
    assert np.all(obs >= observation_space.low) and np.all(obs <= observation_space.high)


def test_action_space_box_matches_normalisation_convention() -> None:
    env = OperatingPointBanditEnv()
    action_space = env.action_space
    assert isinstance(action_space, spaces.Box)
    assert action_space.shape == (2,)
    np.testing.assert_array_equal(action_space.low, np.array([-1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(action_space.high, np.array([1.0, 1.0], dtype=np.float32))


def test_step_before_reset_raises() -> None:
    env = OperatingPointBanditEnv()
    with pytest.raises(RuntimeError):
        env.step(np.array([0.0, 0.0], dtype=np.float32))


def test_episode_always_terminates_after_one_step() -> None:
    env = OperatingPointBanditEnv()
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated is True
    assert truncated is False


def test_config_override_uses_custom_penalty_weight() -> None:
    default_env = OperatingPointBanditEnv()
    custom_env = OperatingPointBanditEnv(BanditEnvConfig(penalty_weight_kw=5.0))
    assert custom_env._penalty_weight_kw == 5.0
    assert default_env._penalty_weight_kw == pytest.approx(50.0)


def test_gymnasium_check_env_passes() -> None:
    gymnasium_utils = pytest.importorskip("gymnasium.utils.env_checker")
    env = OperatingPointBanditEnv()
    gymnasium_utils.check_env(env, skip_render_check=True)
