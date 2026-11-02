r"""AB1/AB2: two paths that VA2 did NOT check.

VA2 fed in only feasible oracle actions (always physical, never through the
normalisation). Two questions remained open, triggered by the finding that the agent lies
BELOW the random level in feasibility (YA1) -- one explanation ("penalty-dominated
gradient", YA2) would have suggested an agent that avoids the penalty first as the strongest
signal, i.e. lies ABOVE random in feasibility. That does not fit together,
so: is a path broken that VA2 does not cover?

AB1 -- sign of the penalty: a known infeasible action must yield a CLEARLY
more negative reward than a known feasible action in the same context.

AB2 -- the normalised action path ``[-1, 1] -> physical``, through which an agent
actually acts (VA2 fed in physical actions directly, never through the
normalisation). Three checks: oracle actions through the normalised path -> bit-identical
to VA2; the four corners -> plausible physical extreme values, in particular ``[1, 1]`` at
maximum inlet air temperature AND maximum gas velocity (the "hot corner").

Finding (see task text/draft finding): neither check finds an error HERE --
sign and normalisation are correct. The explanation for "below random" lies elsewhere
(AB3: policy collapse onto the hot corner, see ``scripts/probe_bandit_env_policy_
inspection.py``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from twinlink_ml.rl import OperatingPointBanditEnv
from twinlink_ml.rl.env import ANCHOR_ARTIFACT_PATH, _normalise
from twinlink_ml.simulation import PlaceholderConfigWarning


@pytest.fixture(scope="module")
def anchor() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(ANCHOR_ARTIFACT_PATH).read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def env() -> OperatingPointBanditEnv:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return OperatingPointBanditEnv()


# --------------------------------------------------------------------------------- AB1


def test_infeasible_action_reward_is_much_more_negative_than_feasible(
    env: OperatingPointBanditEnv,
) -> None:
    """AB1: known infeasible against known feasible action, same context.

    Infeasible: XA3's example (inlet air temperature 108.8 °C, gas velocity 2.352 m/s,
    nominal disturbance) -- bed temperature 86.97 °C, 6.97 K above the limit. Feasible:
    the nominal action (90 °C, 1.6 m/s) in the same context.
    """
    reward_infeasible, info_infeasible = env.evaluate_operating_point(
        inlet_temperature_celsius=108.8,
        gas_velocity_m_per_s=2.352,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    reward_feasible, info_feasible = env.evaluate_operating_point(
        inlet_temperature_celsius=90.0,
        gas_velocity_m_per_s=1.6,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    assert not info_infeasible["feasible"]
    assert info_feasible["feasible"]

    # Clearly more negative, not only numerically smaller -- an order-of-magnitude check,
    # not only a sign check.
    assert reward_feasible - reward_infeasible > 10.0

    # Decomposition: reward = -(energy + penalty) -- the penalty must be positive and dominant.
    penalty_infeasible = -reward_infeasible - info_infeasible["energy_target"]
    assert penalty_infeasible > 0.0
    assert penalty_infeasible == pytest.approx(34.85, abs=0.01)
    assert info_feasible["constraint_check"]["normalized_violation"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------- AB2


def test_action_space_corners_map_to_expected_physical_extremes(
    env: OperatingPointBanditEnv,
) -> None:
    """AB2: the four corners of ``[-1, 1]^2`` must yield plausible extremes.

    ``[1, 1]`` must be the "hot corner" (maximum inlet air temperature AND maximum
    gas velocity), not some swapped combination.
    """
    env.reset(options={"ambient_temperature_celsius": 20.0, "ambient_relative_humidity": 0.5})
    _, _, _, _, info_hot = env.step(np.array([1.0, 1.0], dtype=np.float64))
    assert info_hot["action_physical_applied"]["inlet_temperature_celsius"] == pytest.approx(
        env._inlet_bound.maximum
    )
    assert info_hot["action_physical_applied"]["gas_velocity_m_per_s"] == pytest.approx(
        env._gas_bound.maximum
    )

    env.reset(options={"ambient_temperature_celsius": 20.0, "ambient_relative_humidity": 0.5})
    _, _, _, _, info_cold = env.step(np.array([-1.0, -1.0], dtype=np.float64))
    assert info_cold["action_physical_applied"]["inlet_temperature_celsius"] == pytest.approx(
        env._inlet_bound.minimum
    )
    assert info_cold["action_physical_applied"]["gas_velocity_m_per_s"] == pytest.approx(
        env._gas_bound.minimum
    )

    env.reset(options={"ambient_temperature_celsius": 20.0, "ambient_relative_humidity": 0.5})
    _, _, _, _, info_mixed_a = env.step(np.array([-1.0, 1.0], dtype=np.float64))
    assert info_mixed_a["action_physical_applied"]["inlet_temperature_celsius"] == pytest.approx(
        env._inlet_bound.minimum
    )
    assert info_mixed_a["action_physical_applied"]["gas_velocity_m_per_s"] == pytest.approx(
        env._gas_bound.maximum
    )

    env.reset(options={"ambient_temperature_celsius": 20.0, "ambient_relative_humidity": 0.5})
    _, _, _, _, info_mixed_b = env.step(np.array([1.0, -1.0], dtype=np.float64))
    assert info_mixed_b["action_physical_applied"]["inlet_temperature_celsius"] == pytest.approx(
        env._inlet_bound.maximum
    )
    assert info_mixed_b["action_physical_applied"]["gas_velocity_m_per_s"] == pytest.approx(
        env._gas_bound.minimum
    )


def test_oracle_actions_through_normalised_step_path_reproduce_va2(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    """AB2: the same 25 oracle actions as VA2, but THROUGH the normalised path.

    VA2 itself calls ``evaluate_operating_point`` with physical values directly, never
    through ``_normalise``/``_denormalise`` -- here via ``step()``, must stay bit-identical.
    """
    for point in anchor["grid_results"]:
        env.reset(
            options={
                "ambient_temperature_celsius": point["ambient_temperature_celsius"],
                "ambient_relative_humidity": point["ambient_relative_humidity"],
            }
        )
        normalised_action = np.array(
            [
                _normalise(
                    point["inlet_temperature_celsius"],
                    env._inlet_bound.minimum,
                    env._inlet_bound.maximum,
                ),
                _normalise(
                    point["gas_velocity_m_per_s"], env._gas_bound.minimum, env._gas_bound.maximum
                ),
            ],
            dtype=np.float64,
        )
        _, reward, _, _, _ = env.step(normalised_action)
        assert reward == pytest.approx(-point["energy_target"], rel=1e-6)
