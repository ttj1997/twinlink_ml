r"""VA3/XA1: evaluation metric -- see ``twinlink_ml.rl.evaluation``.

Two limiting cases validate the metric itself (not only its implementation against
itself): a policy that always returns the fixed operating point MUST yield
``closed_regret_fraction == 0`` everywhere (and ``pooled_closed_regret_fraction == 0``); a
policy that always returns the oracle action of the respective test point MUST yield
``== 1`` everywhere. Both follow directly from the definition, but this is the test that
immediately exposes a formula built in the wrong way round (e.g. numerator/denominator
swapped).

XA1 additionally: a single infeasible action must NOT affect
``pooled_closed_regret_fraction`` (it is excluded from sum/denominator), only
``feasibility_rate`` and ``infeasible_violations`` -- see
``test_single_infeasible_action_does_not_affect_pooled_
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
    """XA3: regression anchor for :func:`is_boundary_optimum`.

    23 of 25 Befund 52 optima have the gas velocity at the lower action bound
    (in the warm regime) -- see the task.
    """
    points = load_test_points()
    boundary_flags = [is_boundary_optimum(point, env) for point in points]
    assert sum(boundary_flags) == 23
    assert len(boundary_flags) - sum(boundary_flags) == 2


def test_single_infeasible_action_does_not_affect_pooled_fraction(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    """XA1: an infeasible action affects only (a)/(c), not (b).

    The policy returns the exact oracle action at 24 of 25 test points (closes
    ``== 1`` there), and at EXACTLY ONE point instead an action that deliberately violates
    the 80 C bed temperature limit (high inlet air temperature close to the upper
    action bound, see the WA3 finding) -- independent of the action context of the
    respective point, i.e. independent of whether this point has a boundary or interior
    optimum.
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
    # The core of XA1: (b) stays bit-identical to a pure 24-of-24 oracle evaluation,
    # because the one infeasible point is excluded from sum AND denominator.
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
    """VA3: the training distribution does not hit the 25 test points exactly by chance.

    Spot-check confirmation of the measure-zero claim for the actual sampling
    routine (``OperatingPointBanditEnv.reset()`` without ``options``): 500 draws
    hit none of the 25 discrete test points numerically.
    """
    env = OperatingPointBanditEnv()
    draws = []
    for seed in range(500):
        obs, _ = env.reset(seed=seed)
        draws.append(obs)
    assert_disjoint_from_test_grid(np.asarray(draws, dtype=np.float64))
