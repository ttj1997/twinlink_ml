"""S0.3: Lagrangian bandit environment (no training here, mechanics only)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv
from twinlink_ml.rl.lagrange import (
    LagrangeBanditEnv,
    LagrangeConfig,
    ReferenceTable,
    dual_ascent,
    failure_bound_kw,
    relabel_rewards,
)
from twinlink_ml.simulation import PlaceholderConfigWarning

MARGIN = 0.005961819615175404

NOMINAL_POINT = {
    "inlet_temperature_celsius": 90.0,
    "gas_velocity_m_per_s": 1.6,
    "ambient_temperature_celsius": 20.0,
    "ambient_relative_humidity": 0.5,
}
COLD_SLOW_POINT = {
    "inlet_temperature_celsius": 60.0,
    "gas_velocity_m_per_s": 0.8,
    "ambient_temperature_celsius": 20.0,
    "ambient_relative_humidity": 0.5,
}
BINDING_POINT = {
    "inlet_temperature_celsius": 70.0,
    "gas_velocity_m_per_s": 1.0,
    "ambient_temperature_celsius": 10.0,
    "ambient_relative_humidity": 0.5,
}


def _config(**overrides: float) -> LagrangeConfig:
    values: dict[str, float] = {
        "margin_kg_per_kg": MARGIN,
        "initial_multiplier_kw": 0.5,
        "step_size_kw": 1.0,
        "update_interval_steps": 250,
    }
    values.update(overrides)
    return LagrangeConfig(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def env() -> LagrangeBanditEnv:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return LagrangeBanditEnv(_config(), BanditEnvConfig())


@pytest.fixture(scope="module")
def plain_env() -> OperatingPointBanditEnv:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return OperatingPointBanditEnv()


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        _config(margin_kg_per_kg=-1.0)
    with pytest.raises(ValueError):
        _config(initial_multiplier_kw=-0.1)
    with pytest.raises(ValueError):
        _config(step_size_kw=0.0)
    with pytest.raises(ValueError):
        _config(update_interval_steps=0)
    with pytest.raises(ValueError):
        LagrangeConfig(0.0, 0.0, 1.0, 10, context_bins=(0, 5))


def test_dual_ascent_sign_and_floor() -> None:
    assert dual_ascent(0.5, +0.2, 1.0) == pytest.approx(0.7)
    assert dual_ascent(0.5, -0.2, 1.0) == pytest.approx(0.3)  # slack lowers lambda
    assert dual_ascent(0.1, -5.0, 1.0) == 0.0  # projected onto lambda >= 0


def test_bin_of_covers_ranges_and_is_row_major(env: LagrangeBanditEnv) -> None:
    t = env._t_amb_range
    p = env._phi_range
    assert env.bin_of(t.minimum, p.minimum) == 0
    assert env.bin_of(t.maximum, p.maximum) == 24
    assert env.bin_of(t.minimum, p.maximum) == 4
    assert env.bin_of(t.maximum, p.minimum) == 20
    assert env.multipliers.shape == (5, 5)
    assert np.all(env.multipliers == 0.5)


def test_multiplier_setter_checks_shape_and_sign(env: LagrangeBanditEnv) -> None:
    with pytest.raises(ValueError):
        env.set_multipliers(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        env.set_multipliers(-np.ones((5, 5)))


def test_requires_penalty_mode() -> None:
    with pytest.raises(ValueError, match="include_penalty"):
        LagrangeBanditEnv(_config(), BanditEnvConfig(include_penalty=False))


def test_reward_formula_and_signed_constraint(
    env: LagrangeBanditEnv, plain_env: OperatingPointBanditEnv
) -> None:
    env.set_multipliers(np.full((5, 5), 0.5))
    reward, info = env.evaluate_operating_point(**NOMINAL_POINT)
    limit = env.tightened_limit_kg_per_kg + MARGIN
    moisture = info["constraint_check"]["moisture_kg_per_kg"]
    g_tilde = (moisture - (limit - MARGIN)) / limit
    assert g_tilde < 0.0  # nominal point is far below the limit: slack, signed negative
    assert info["lagrange_g_tilde"] == pytest.approx(g_tilde)
    energy = info["energy_target"]
    assert reward == pytest.approx(-(energy + 0.5 * g_tilde))
    # With lambda = 0 and no violation the reward is the penalty arm's reward, bit for bit.
    env.set_multipliers(np.zeros((5, 5)))
    reward_zero, _ = env.evaluate_operating_point(**NOMINAL_POINT)
    reward_plain, _ = plain_env.evaluate_operating_point(**NOMINAL_POINT)
    assert reward_zero == reward_plain


def test_feasibility_is_judged_against_the_true_limit(env: LagrangeBanditEnv) -> None:
    """A point between the tightened and the true limit is feasible but has g_tilde > 0."""
    _, info = env.evaluate_operating_point(**BINDING_POINT)
    moisture = info["constraint_check"]["moisture_kg_per_kg"]
    limit = env.tightened_limit_kg_per_kg + MARGIN
    if info["feasible"]:
        assert moisture <= limit + 1e-4
    if moisture > limit - MARGIN:
        assert info["lagrange_g_tilde"] > 0.0


def test_failure_gets_bounded_value_not_the_sentinel(env: LagrangeBanditEnv) -> None:
    env.set_multipliers(np.full((5, 5), 0.5))
    reward, info = env.evaluate_operating_point(**COLD_SLOW_POINT)
    assert info["solve_failed"] is True
    assert reward == -failure_bound_kw(env.reference_table, 0.5)
    assert abs(reward) < 1.0e3  # not the 1e6 sentinel of Befund 52
    table = env.reference_table
    assert table.energy_kw.size > 100  # most of the 117 grid points are solvable


def test_relabel_matches_the_environment(env: LagrangeBanditEnv) -> None:
    env.set_multipliers(np.linspace(0.1, 2.0, 25).reshape(5, 5))
    rewards, energies, rests, gs, failed, bins = [], [], [], [], [], []
    for point in (NOMINAL_POINT, BINDING_POINT, COLD_SLOW_POINT):
        reward, info = env.evaluate_operating_point(**point)
        rewards.append(reward)
        failed.append(bool(info.get("solve_failed")))
        bins.append(info["lagrange_bin"])
        energies.append(info.get("lagrange_energy_kw", 0.0))
        rests.append(info.get("lagrange_rest_penalty_kw", 0.0))
        gs.append(info.get("lagrange_g_tilde", 0.0))
    relabelled = relabel_rewards(
        energy_kw=np.array(energies),
        rest_penalty_kw=np.array(rests),
        g_tilde=np.nan_to_num(np.array(gs)),
        solve_failed=np.array(failed),
        bin_index=np.array(bins),
        multipliers=env.multipliers,
        table=env.reference_table,
    )
    np.testing.assert_allclose(relabelled, rewards, rtol=0, atol=1e-12)


def test_failure_bound_is_the_maximum_cost() -> None:
    table = ReferenceTable(
        energy_kw=np.array([1.0, 2.0]),
        rest_penalty_kw=np.array([0.0, 10.0]),
        g_tilde=np.array([-1.0, 0.5]),
    )
    assert failure_bound_kw(table, 0.0) == 12.0
    assert failure_bound_kw(table, 4.0) == 14.0


def test_margin_metric_is_zero_for_the_oracle_policy(plain_env: OperatingPointBanditEnv) -> None:
    """The oracle sits on the limit at the 17 binding contexts of Befund 62 (margin ~ 0)."""
    from twinlink_ml.rl import load_test_points
    from twinlink_ml.rl.evaluation import moisture_margin_at_oracle_binding_points

    points = {
        (p.ambient_temperature_celsius, p.ambient_relative_humidity): p for p in load_test_points()
    }
    inlet, gas = plain_env._inlet_bound, plain_env._gas_bound

    def oracle_policy(observation: np.ndarray) -> np.ndarray:
        # the observation is float32: match the nearest grid point, not exact equality
        key = min(
            points,
            key=lambda k: abs(k[0] - float(observation[0])) + abs(k[1] - float(observation[1])),
        )
        p = points[key]
        return np.array(
            [
                2.0 * (p.oracle_inlet_temperature_celsius - inlet.minimum) / inlet.span - 1.0,
                2.0 * (p.oracle_gas_velocity_m_per_s - gas.minimum) / gas.span - 1.0,
            ]
        )

    margin, n_binding = moisture_margin_at_oracle_binding_points(plain_env, oracle_policy)
    assert n_binding == 17
    assert abs(margin) < 1e-6
