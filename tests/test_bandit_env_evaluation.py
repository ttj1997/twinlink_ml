r"""VA3/XA1: Bewertungsmetrik -- siehe ``twinlink_ml.rl.evaluation``.

Zwei Grenzfaelle validieren die Metrik selbst (nicht nur ihre Implementierung gegen sich
selbst): eine Policy, die immer den festen Betriebspunkt liefert, MUSS ueberall
``closed_regret_fraction == 0`` ergeben (und ``pooled_closed_regret_fraction == 0``); eine
Policy, die immer die Orakelaktion des jeweiligen Testpunkts liefert, MUSS ueberall ``== 1``
ergeben. Beides folgt direkt aus der Definition, ist aber der Test, der eine falsch herum
eingebaute Formel (z. B. Zaehler/Nenner vertauscht) sofort sichtbar macht.

XA1 zusaetzlich: eine einzelne unzulaessige Aktion darf ``pooled_closed_regret_fraction``
NICHT beeinflussen (sie wird aus Summe/Nenner ausgeschlossen), nur ``feasibility_rate`` und
``infeasible_violations`` -- siehe ``test_single_infeasible_action_does_not_affect_pooled_
fraction``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from twinlink_ml.rl import OperatingPointBanditEnv, evaluate_policy, load_test_points
from twinlink_ml.rl.env import ANCHOR_ARTIFACT_PATH, _normalise
from twinlink_ml.rl.evaluation import assert_disjoint_from_test_grid, is_boundary_optimum


@pytest.fixture(scope="module")
def anchor() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(ANCHOR_ARTIFACT_PATH).read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def env() -> OperatingPointBanditEnv:
    return OperatingPointBanditEnv()


def test_load_test_points_returns_all_25_befund_52_points() -> None:
    points = load_test_points()
    assert len(points) == 25


def _normalised_action(
    env: OperatingPointBanditEnv, inlet_temperature_celsius: float, gas_velocity_m_per_s: float
) -> np.ndarray:
    return np.array(
        [
            _normalise(
                inlet_temperature_celsius, env._inlet_bound.minimum, env._inlet_bound.maximum
            ),
            _normalise(gas_velocity_m_per_s, env._gas_bound.minimum, env._gas_bound.maximum),
        ],
        dtype=np.float64,
    )


def test_fixed_point_policy_closes_zero_regret_everywhere(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    fixed = anchor["fixed_point"]
    action = _normalised_action(
        env, fixed["inlet_temperature_celsius"], fixed["gas_velocity_m_per_s"]
    )

    def fixed_policy(observation: np.ndarray) -> np.ndarray:
        del observation
        return action

    result = evaluate_policy(env, fixed_policy)
    assert result.closed_fraction_median == pytest.approx(0.0, abs=1e-6)
    assert result.closed_fraction_p95 == pytest.approx(0.0, abs=1e-6)
    assert result.closed_fraction_min == pytest.approx(0.0, abs=1e-6)
    assert result.feasibility_rate == pytest.approx(1.0)
    assert result.n_solve_failed == 0
    assert result.pooled_closed_regret_fraction == pytest.approx(0.0, abs=1e-6)
    assert result.pooled_closed_regret_fraction_interior == pytest.approx(0.0, abs=1e-6)
    assert result.pooled_closed_regret_fraction_boundary == pytest.approx(0.0, abs=1e-6)
    assert result.infeasible_violations == ()


def test_oracle_policy_closes_full_regret_everywhere(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    lookup = {
        (
            round(point["ambient_temperature_celsius"], 6),
            round(point["ambient_relative_humidity"], 6),
        ): point
        for point in anchor["grid_results"]
    }

    def oracle_policy(observation: np.ndarray) -> np.ndarray:
        key = (round(float(observation[0]), 6), round(float(observation[1]), 6))
        point = lookup[key]
        return _normalised_action(
            env, point["inlet_temperature_celsius"], point["gas_velocity_m_per_s"]
        )

    result = evaluate_policy(env, oracle_policy)
    assert result.closed_fraction_median == pytest.approx(1.0, abs=1e-6)
    assert result.closed_fraction_p95 == pytest.approx(1.0, abs=1e-6)
    assert result.closed_fraction_min == pytest.approx(1.0, abs=1e-6)
    assert result.feasibility_rate == pytest.approx(1.0)
    assert result.pooled_closed_regret_fraction == pytest.approx(1.0, abs=1e-6)
    assert result.pooled_closed_regret_fraction_interior == pytest.approx(1.0, abs=1e-6)
    assert result.pooled_closed_regret_fraction_boundary == pytest.approx(1.0, abs=1e-6)
    assert result.infeasible_violations == ()


def test_oracle_optima_split_matches_known_boundary_count(
    env: OperatingPointBanditEnv,
) -> None:
    """XA3: Regressionsanker fuer :func:`is_boundary_optimum`.

    23 von 25 Befund-52-Optima haben die Gasgeschwindigkeit an der unteren Aktionsschranke
    (im warmen Regime) -- siehe Auftrag.
    """
    points = load_test_points()
    boundary_flags = [is_boundary_optimum(point, env) for point in points]
    assert sum(boundary_flags) == 23
    assert len(boundary_flags) - sum(boundary_flags) == 2


def test_single_infeasible_action_does_not_affect_pooled_fraction(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    """XA1: eine unzulaessige Aktion beeinflusst nur (a)/(c), nicht (b).

    Policy liefert an 24 von 25 Testpunkten die exakte Orakelaktion (schliesst dort
    ``== 1``), an GENAU EINEM Punkt stattdessen eine Aktion, die absichtlich die
    80-C-Bettemperaturgrenze verletzt (hohe Zulufttemperatur nahe der oberen
    Aktionsschranke, siehe WA3-Befund) -- unabhaengig vom Aktions-Kontext des jeweiligen
    Punkts, also unabhaengig davon, ob dieser Punkt ein Rand- oder Innenoptimum hat.
    """
    lookup = {
        (
            round(point["ambient_temperature_celsius"], 6),
            round(point["ambient_relative_humidity"], 6),
        ): point
        for point in anchor["grid_results"]
    }
    points = load_test_points()
    infeasible_point = points[0]
    infeasible_action = _normalised_action(env, 119.0, 2.4)
    _, infeasible_info = env.evaluate_operating_point(
        inlet_temperature_celsius=119.0,
        gas_velocity_m_per_s=2.4,
        ambient_temperature_celsius=infeasible_point.ambient_temperature_celsius,
        ambient_relative_humidity=infeasible_point.ambient_relative_humidity,
    )
    assert not infeasible_info["feasible"], "Testannahme verletzt: Aktion ist zulaessig"

    def mostly_oracle_policy(observation: np.ndarray) -> np.ndarray:
        key = (round(float(observation[0]), 6), round(float(observation[1]), 6))
        if key == (
            round(infeasible_point.ambient_temperature_celsius, 6),
            round(infeasible_point.ambient_relative_humidity, 6),
        ):
            return infeasible_action
        point = lookup[key]
        return _normalised_action(
            env, point["inlet_temperature_celsius"], point["gas_velocity_m_per_s"]
        )

    result = evaluate_policy(env, mostly_oracle_policy)
    assert result.feasibility_rate == pytest.approx(24.0 / 25.0)
    assert len(result.infeasible_violations) == 1
    assert result.infeasible_violations[0].bed_temperature_violation_k > 0.0
    # Der Kern von XA1: (b) bleibt bitgleich zu einer reinen 24-von-24-Orakel-Auswertung,
    # weil der eine unzulaessige Punkt aus Summe UND Nenner ausgeschlossen wird.
    assert result.pooled_closed_regret_fraction == pytest.approx(1.0, abs=1e-6)


def test_assert_disjoint_from_test_grid_passes_for_unrelated_points() -> None:
    training = np.array([[18.3, 0.42], [31.9, 0.77]])
    assert_disjoint_from_test_grid(training)


def test_assert_disjoint_from_test_grid_raises_on_exact_collision() -> None:
    points = load_test_points()
    colliding = np.array(
        [[points[3].ambient_temperature_celsius, points[3].ambient_relative_humidity]]
    )
    with pytest.raises(ValueError, match="Testpunkte"):
        assert_disjoint_from_test_grid(colliding)


def test_continuous_uniform_training_draws_never_collide_with_test_grid() -> None:
    """VA3: Trainingsverteilung trifft die 25 Testpunkte nicht zufaellig exakt.

    Stichprobenartige Bestaetigung der Mass-Null-Behauptung fuer die tatsaechliche
    Ziehungsroutine (``OperatingPointBanditEnv.reset()`` ohne ``options``): 500 Ziehungen
    treffen keinen der 25 diskreten Testpunkte numerisch.
    """
    env = OperatingPointBanditEnv()
    draws = []
    for seed in range(500):
        obs, _ = env.reset(seed=seed)
        draws.append(obs)
    assert_disjoint_from_test_grid(np.asarray(draws, dtype=np.float64))
