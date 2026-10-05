"""SH1: moisture shield of the Lagrangian bandit (mechanics only, no training)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from twinlink_ml.rl import BanditEnvConfig
from twinlink_ml.rl.env import _normalise
from twinlink_ml.rl.lagrange import LagrangeBanditEnv, LagrangeConfig
from twinlink_ml.rl.shield import ShieldConfig, ShieldedLagrangeBanditEnv
from twinlink_ml.simulation import PlaceholderConfigWarning

MARGIN = 0.005961819615175404
LIMIT = 0.15
CONTEXT = (20.0, 0.5)


def _lagrange() -> LagrangeConfig:
    return LagrangeConfig(
        margin_kg_per_kg=MARGIN,
        initial_multiplier_kw=0.5,
        step_size_kw=0.2,
        update_interval_steps=250,
    )


@pytest.fixture(scope="module")
def shielded() -> ShieldedLagrangeBanditEnv:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return ShieldedLagrangeBanditEnv(
            _lagrange(),
            BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT),
            shield=ShieldConfig(xtol_celsius=0.01),
        )


@pytest.fixture(scope="module")
def unshielded() -> LagrangeBanditEnv:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return LagrangeBanditEnv(
            _lagrange(), BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT)
        )


def _normalised(env: ShieldedLagrangeBanditEnv, t_in: float, u_gas: float) -> np.ndarray:
    inlet, gas = env._inlet_bound, env._gas_bound
    return np.array(
        [
            _normalise(t_in, inlet.minimum, inlet.maximum),
            _normalise(u_gas, gas.minimum, gas.maximum),
        ]
    )


def _step(env: ShieldedLagrangeBanditEnv, t_in: float, u_gas: float) -> tuple[float, dict]:
    env.reset(
        options={
            "ambient_temperature_celsius": CONTEXT[0],
            "ambient_relative_humidity": CONTEXT[1],
        }
    )
    _, reward, terminated, truncated, info = env.step(_normalised(env, t_in, u_gas))
    assert terminated and not truncated
    return reward, info


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ShieldConfig(xtol_celsius=0.0)
    with pytest.raises(ValueError):
        ShieldConfig(max_iterations=0)


def test_wet_proposal_is_pushed_onto_the_tightened_limit(
    shielded: ShieldedLagrangeBanditEnv,
) -> None:
    reward, info = _step(shielded, 75.0, 0.9)
    assert info["shield_proposed_moisture_kg_per_kg"] > shielded.tightened_limit_kg_per_kg
    assert info["shield_active"] and not info["shield_saturated"]
    assert info["shield_executed_inlet_temperature_celsius"] > 75.0
    moisture = info["constraint_check"]["moisture_kg_per_kg"]
    # 0.01 K tolerance times a sensitivity of at most ~0.01 kg/kg/K in this region
    assert moisture == pytest.approx(shielded.tightened_limit_kg_per_kg, abs=2e-4)
    # gas velocity untouched, the info carries the executed action
    assert info["action_normalised_applied"][1] == pytest.approx(
        _normalised(shielded, 75.0, 0.9)[1]
    )
    assert info["shield_n_solves"] >= 4
    assert reward == pytest.approx(
        -(
            info["lagrange_energy_kw"]
            + info["lagrange_rest_penalty_kw"]
            + info["lagrange_multiplier_kw"] * info["lagrange_g_tilde"]
        )
    )


def test_reward_equals_the_unshielded_reward_of_the_executed_action(
    shielded: ShieldedLagrangeBanditEnv, unshielded: LagrangeBanditEnv
) -> None:
    reward, info = _step(shielded, 75.0, 0.9)
    expected, _ = unshielded.evaluate_operating_point(
        inlet_temperature_celsius=info["shield_executed_inlet_temperature_celsius"],
        gas_velocity_m_per_s=0.9,
        ambient_temperature_celsius=CONTEXT[0],
        ambient_relative_humidity=CONTEXT[1],
    )
    assert reward == expected


def test_reward_is_flat_beyond_the_boundary(shielded: ShieldedLagrangeBanditEnv) -> None:
    reward_a, info_a = _step(shielded, 72.0, 0.9)
    reward_b, info_b = _step(shielded, 78.0, 0.9)
    assert info_a["shield_active"] and info_b["shield_active"]
    assert abs(reward_a - reward_b) < 5e-3  # kW, root tolerance only


def test_dry_proposal_is_executed_unchanged(
    shielded: ShieldedLagrangeBanditEnv, unshielded: LagrangeBanditEnv
) -> None:
    reward, info = _step(shielded, 100.0, 1.6)
    assert not info["shield_active"] and info["shield_n_solves"] == 1
    assert info["shield_executed_inlet_temperature_celsius"] == pytest.approx(100.0)
    expected, _ = unshielded.evaluate_operating_point(
        inlet_temperature_celsius=100.0,
        gas_velocity_m_per_s=1.6,
        ambient_temperature_celsius=CONTEXT[0],
        ambient_relative_humidity=CONTEXT[1],
    )
    assert reward == expected


def test_unreachable_limit_saturates_at_the_upper_bound(
    shielded: ShieldedLagrangeBanditEnv,
) -> None:
    # slowest gas, coldest air: even the upper inlet bound may not be enough; if it is, the
    # precondition of the test is not met and the test says so instead of passing vacuously
    reward, info = _step(shielded, 60.0, 0.8)
    if not info["shield_saturated"]:
        pytest.skip("the tightened limit is reachable at u = 0.8 here; saturation not exercised")
    assert info["shield_executed_inlet_temperature_celsius"] == pytest.approx(
        shielded._inlet_bound.maximum
    )
    assert np.isfinite(reward)
