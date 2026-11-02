r"""DA1/EA2: reward without penalty, bounded solver-failure value.

Constraints are still checked, just not in the reward. Reproduction mode
(``include_penalty=True``, ``solve_failure_penalty_kw=None``, both default) stays
bit-identical to VA2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gymnasium")

from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv
from twinlink_ml.rl.env import ANCHOR_ARTIFACT_PATH


@pytest.fixture(scope="module")
def anchor() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(ANCHOR_ARTIFACT_PATH).read_text(encoding="utf-8"))
    return data


def test_default_config_still_reproduces_va2_bitgleich(anchor: dict[str, Any]) -> None:
    """``include_penalty=True`` (default) stays bit-identical to VA2 -- unchanged."""
    env = OperatingPointBanditEnv()
    for point in anchor["grid_results"]:
        reward, _ = env.evaluate_operating_point(
            inlet_temperature_celsius=point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        assert reward == pytest.approx(-point["energy_target"], rel=1e-6)


def test_no_penalty_reward_equals_negative_energy_regardless_of_feasibility() -> None:
    """``include_penalty=False``: the reward is always ``-energy_target``.

    Also for infeasible actions -- feasibility is still reported.
    """
    env = OperatingPointBanditEnv(BanditEnvConfig(include_penalty=False))

    # known infeasible action (hot corner, AB3/DA0 example)
    reward, info = env.evaluate_operating_point(
        inlet_temperature_celsius=120.0,
        gas_velocity_m_per_s=2.5,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    assert not info["feasible"]
    assert reward == pytest.approx(-info["energy_target"])
    assert info["constraint_check"]["bed_temperature_violation_k"] > 0.0

    # known feasible action
    reward_feasible, info_feasible = env.evaluate_operating_point(
        inlet_temperature_celsius=90.0,
        gas_velocity_m_per_s=1.6,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    assert info_feasible["feasible"]
    assert reward_feasible == pytest.approx(-info_feasible["energy_target"])


def test_no_penalty_reward_differs_from_penalized_reward_when_infeasible() -> None:
    """For an identical infeasible action the penalty mode must be clearly more negative.

    Otherwise the switch has no effect.
    """
    env_penalized = OperatingPointBanditEnv()
    env_no_penalty = OperatingPointBanditEnv(BanditEnvConfig(include_penalty=False))

    kwargs = {
        "inlet_temperature_celsius": 120.0,
        "gas_velocity_m_per_s": 2.5,
        "ambient_temperature_celsius": 20.0,
        "ambient_relative_humidity": 0.5,
    }
    reward_penalized, _ = env_penalized.evaluate_operating_point(**kwargs)
    reward_no_penalty, _ = env_no_penalty.evaluate_operating_point(**kwargs)
    assert reward_penalized < reward_no_penalty


# Befund 79 (K1): the failure path is exercised at 60 degC / 0.8 m/s with 35 degC / phi 0.5. At
# 20 degC / phi 0.5 the solver with the fallback bracket finds a (wet, infeasible) steady state.


def test_default_solve_failure_penalty_still_uses_va2_sentinel() -> None:
    """``solve_failure_penalty_kw=None`` stays bit-identical to VA2.

    The sentinel from Befund 52 (``_SOLVE_FAILURE_PENALTY_KW`` = 1,000,000), unchanged.
    """
    env = OperatingPointBanditEnv()
    # known solver-failure point (EA1(b)/DA0(a))
    reward, info = env.evaluate_operating_point(
        inlet_temperature_celsius=60.0,
        gas_velocity_m_per_s=0.8,
        ambient_temperature_celsius=35.0,
        ambient_relative_humidity=0.5,
    )
    assert info["solve_failed"]
    assert reward == pytest.approx(-1_000_000.0)
    assert info["solve_failure_penalty_kw"] == pytest.approx(1_000_000.0)


def test_bounded_solve_failure_penalty_overrides_sentinel() -> None:
    """EA2: a set ``solve_failure_penalty_kw`` replaces the sentinel exactly."""
    env = OperatingPointBanditEnv(BanditEnvConfig(solve_failure_penalty_kw=4.5))
    reward, info = env.evaluate_operating_point(
        inlet_temperature_celsius=60.0,
        gas_velocity_m_per_s=0.8,
        ambient_temperature_celsius=35.0,
        ambient_relative_humidity=0.5,
    )
    assert info["solve_failed"]
    assert reward == pytest.approx(-4.5)
    assert info["solve_failure_penalty_kw"] == pytest.approx(4.5)


def test_fa1_bounded_solve_failure_penalty_with_penalty_mode() -> None:
    """Pins the bound used for FA2 in the DEFAULT mode "with penalty".

    From ``probe_bandit_env_solve_failure_bound.py`` (``with_penalty``, 155.57896052362634
    kW -- worst sum of energy and penalty over the solvable 13x9 grid points,
    reached at t_in=65 °C/u_gas=0.8 m/s, see Befund 62 DA0(a)). Replaces the sentinel
    bit-identically, also in the DEFAULT mode "with penalty" (``include_penalty=True``) --
    the same generic override mechanism as EA2's without-penalty bound, here pinned with
    the value actually used for FA2, so that a later change to this value shows up here
    instead of silently disappearing in the training run. ``include_penalty`` stays
    default (True) -- no second mode switch needed, the sentinel override is independent
    of ``include_penalty`` (see env.py).
    """
    env = OperatingPointBanditEnv(BanditEnvConfig(solve_failure_penalty_kw=155.57896052362634))
    reward, info = env.evaluate_operating_point(
        inlet_temperature_celsius=60.0,
        gas_velocity_m_per_s=0.8,
        ambient_temperature_celsius=35.0,
        ambient_relative_humidity=0.5,
    )
    assert info["solve_failed"]
    assert reward == pytest.approx(-155.57896052362634)
    assert info["solve_failure_penalty_kw"] == pytest.approx(155.57896052362634)
