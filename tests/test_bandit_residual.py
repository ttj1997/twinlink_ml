"""Residual action and separated moisture cost of the bandit (Befund 68 closure, mechanics only)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from gymnasium import spaces

from twinlink_ml.rl import BanditEnvConfig
from twinlink_ml.rl.env import _normalise
from twinlink_ml.rl.lagrange import LagrangeBanditEnv, LagrangeConfig
from twinlink_ml.rl.residual import (
    MaskedReplayBuffer,
    ResidualActionWrapper,
    ResidualConfig,
    SeparatedLagrangeBanditEnv,
    residual_to_action,
    zero_init_actor_mean,
)
from twinlink_ml.simulation import PlaceholderConfigWarning

MARGIN = 0.005961819615175404
LIMIT = 0.15
CONTEXT = (20.0, 0.5)
FIXED = (92.98902491351069, 0.8004504372775472)


def _lagrange() -> LagrangeConfig:
    return LagrangeConfig(
        margin_kg_per_kg=MARGIN,
        initial_multiplier_kw=0.5,
        step_size_kw=0.2,
        update_interval_steps=250,
    )


def _config() -> BanditEnvConfig:
    return BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT)


@pytest.fixture(scope="module")
def separated() -> SeparatedLagrangeBanditEnv:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return SeparatedLagrangeBanditEnv(_lagrange(), _config())


@pytest.fixture(scope="module")
def plain() -> LagrangeBanditEnv:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return LagrangeBanditEnv(_lagrange(), _config())


def test_residual_to_action_zero_is_the_anchor_and_clips() -> None:
    anchor = np.array([0.0996, -0.9989])
    assert np.array_equal(residual_to_action(np.zeros(2), anchor, 0.5), anchor)
    out = residual_to_action(np.array([5.0, -5.0]), anchor, 0.5)  # delta clipped to [-1, 1] first
    assert out[0] == pytest.approx(0.0996 + 0.5)
    assert out[1] == -1.0  # the lower gas bound is reached by the clip
    assert np.all(
        np.abs(residual_to_action(np.array([1.0, 1.0]), np.array([0.9, 0.9]), 1.0)) <= 1.0
    )


def test_residual_config_rejects_bad_scale() -> None:
    for scale in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ResidualConfig(*FIXED, scale)


def test_wrapper_zero_delta_executes_the_fixed_point(plain: LagrangeBanditEnv) -> None:
    wrapper = ResidualActionWrapper(plain, ResidualConfig(*FIXED, 0.5))
    expected = np.array(
        [
            _normalise(FIXED[0], plain._inlet_bound.minimum, plain._inlet_bound.maximum),
            _normalise(FIXED[1], plain._gas_bound.minimum, plain._gas_bound.maximum),
        ]
    )
    assert np.array_equal(wrapper.anchor_normalised, expected)
    wrapper.reset(
        seed=0,
        options={
            "ambient_temperature_celsius": CONTEXT[0],
            "ambient_relative_humidity": CONTEXT[1],
        },
    )
    _, _, terminated, _, info = wrapper.step(np.zeros(2))
    assert terminated
    physical = info["action_physical_applied"]
    assert physical["inlet_temperature_celsius"] == pytest.approx(FIXED[0], abs=1e-9)
    assert physical["gas_velocity_m_per_s"] == pytest.approx(FIXED[1], abs=1e-9)
    assert info["residual_delta"] == [0.0, 0.0]


def test_zero_init_actor_gives_zero_deterministic_action() -> None:
    import gymnasium as gym
    from stable_baselines3 import SAC

    class _Dummy(gym.Env):  # type: ignore[type-arg]
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):  # type: ignore[no-untyped-def]  # noqa: ARG002
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):  # type: ignore[no-untyped-def]  # noqa: ARG002
            return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

    model = SAC("MlpPolicy", _Dummy(), seed=0, verbose=0)
    before, _ = model.predict(np.array([0.3, -0.7], dtype=np.float32), deterministic=True)
    assert np.any(before != 0.0)
    zero_init_actor_mean(model)
    for observation in (np.array([0.3, -0.7]), np.array([-1.0, 1.0]), np.zeros(2)):
        action, _ = model.predict(observation.astype(np.float32), deterministic=True)
        assert np.array_equal(action, np.zeros(2, dtype=np.float32))


def test_moisture_above_critical_gets_smooth_cost_not_the_step(
    separated: SeparatedLagrangeBanditEnv, plain: LagrangeBanditEnv
) -> None:
    kwargs = {
        "inlet_temperature_celsius": 65.0,
        "gas_velocity_m_per_s": 0.8,
        "ambient_temperature_celsius": CONTEXT[0],
        "ambient_relative_humidity": CONTEXT[1],
    }
    reward_sep, info_sep = separated.evaluate_operating_point(**kwargs)
    reward_old, info_old = plain.evaluate_operating_point(**kwargs)
    check = info_sep["constraint_check"]
    assert not check["solver_feasible"] and check["moisture_kg_per_kg"] > 0.2
    assert info_sep["masked"] is False and info_sep["mask_reason"] is None
    # the old reward carries the +1 normalised violation, the separated one does not
    assert reward_sep - reward_old == pytest.approx(separated._penalty_weight_kw * 1.0)
    assert info_sep["lagrange_g_tilde"] == pytest.approx(info_old["lagrange_g_tilde"])
    assert info_sep["feasible"] is False  # judged against the true limit as before


def test_ordinary_point_is_unchanged(
    separated: SeparatedLagrangeBanditEnv, plain: LagrangeBanditEnv
) -> None:
    kwargs = {
        "inlet_temperature_celsius": FIXED[0],
        "gas_velocity_m_per_s": FIXED[1],
        "ambient_temperature_celsius": CONTEXT[0],
        "ambient_relative_humidity": CONTEXT[1],
    }
    reward_sep, info_sep = separated.evaluate_operating_point(**kwargs)
    reward_old, _ = plain.evaluate_operating_point(**kwargs)
    assert info_sep["masked"] is False
    assert reward_sep == reward_old


def test_solver_failure_and_other_reasons_are_masked(separated: SeparatedLagrangeBanditEnv) -> None:
    failed = {"solve_failed": True, "feasible": False}
    reward, info = separated.lagrangian_reward(
        failed, ambient_temperature_celsius=CONTEXT[0], ambient_relative_humidity=CONTEXT[1]
    )
    assert info["masked"] and info["mask_reason"] == "solve_failed" and reward == 0.0
    other = {
        "solve_failed": False,
        "constraint_check": {"solver_feasible": False, "moisture_kg_per_kg": 0.1},
        "infeasibility_reasons": ["kein Trocknungsregime: Y_in=..."],
    }
    _, info = separated.lagrangian_reward(
        other, ambient_temperature_celsius=CONTEXT[0], ambient_relative_humidity=CONTEXT[1]
    )
    assert info["masked"] and info["mask_reason"] == "solver_infeasible_other"
    mixed = {
        "solve_failed": False,
        "constraint_check": {"solver_feasible": False, "moisture_kg_per_kg": 0.3},
        "infeasibility_reasons": ["Partikelfeuchte (0.3 kg/kg) ...", "Saettigungsgrad (1.2) > 1"],
    }
    _, info = separated.lagrangian_reward(
        mixed, ambient_temperature_celsius=CONTEXT[0], ambient_relative_humidity=CONTEXT[1]
    )
    assert info["masked"] and info["mask_reason"] == "solver_infeasible_other"


def test_masked_replay_buffer_drops_masked_transitions() -> None:
    space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    buffer = MaskedReplayBuffer(8, space, space, device="cpu")
    obs = np.zeros((1, 2), dtype=np.float32)
    args = (obs, obs, obs, np.array([-1.0]), np.array([True]))
    buffer.add(*args, [{"masked": True}])
    assert buffer.pos == 0 and buffer.n_masked == 1
    buffer.add(*args, [{"masked": False}])
    buffer.add(*args, [{}])
    assert buffer.pos == 2 and buffer.n_masked == 1
