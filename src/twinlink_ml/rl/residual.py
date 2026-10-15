r"""Residual action around the fixed point and a separated moisture cost (Befund 68 closure).

Two measures, each switchable on its own (the 2 x 2 ablation of
``scripts/run_rl_bandit_residual.py``):

**Residual action.** The agent outputs ``delta`` in ``[-1, 1]^2``; the executed normalised action is

.. math:: a = \mathrm{clip}\bigl(a_{fixed} + s \cdot \delta,\; -1,\; 1\bigr)

with ``a_fixed`` the fixed operating point of the quality limit (Befund 52 / S0.5, normalised by the
action bounds) and ``s`` the residual range in normalised units (:class:`ResidualConfig`). The last
layer of the actor mean is initialised to zero (:func:`zero_init_actor_mean`), so the deterministic
policy is exactly the fixed point before the first gradient step (``tanh(0) = 0``); the closed
regret fraction of the untrained policy is 0 by construction. A bound that the oracle sits on (the
gas velocity at its lower bound) is reached by the clip at a finite ``delta``, not only
asymptotically as with a tanh squashed absolute action.

**Separated moisture cost** (:class:`SeparatedLagrangeBanditEnv`). In the Lagrangian environment of
S0.3 two things were still mixed into the step the critic saw: (a) the +1 normalised violation
(50 kW) that ``check_constraints`` adds when the solver marks a solution infeasible, which for a
product moisture above the critical moisture of the kinetics is the plausibility check of
``steady_state.py`` (S0.4: the only reason on the validity map), and (b) the solver failure value.
Here a solved point whose only infeasibility reason is "product moisture above the critical
moisture" is treated like any other moisture value: its cost is the smooth, normalised
``lambda_b * g_tilde`` (``g_tilde`` is linear in the moisture, divided by the quality limit, no
step), the +1 term is removed. A solver failure (exception) and a solved point with any OTHER
infeasibility reason are a case of their own: they are MASKED -- the transition is not stored in
the replay buffer (:class:`MaskedReplayBuffer`) and the step is counted. The failure bound of S0.3
is not used.

Not changed: the energy, the penalty of the remaining constraints (bed temperature, fluidisation
window, 50 kW weight), the evaluation (always the plain environment against the TRUE limit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

from .env import _normalise
from .lagrange import LagrangeBanditEnv

__all__ = [
    "MaskedReplayBuffer",
    "ResidualActionWrapper",
    "ResidualConfig",
    "SeparatedLagrangeBanditEnv",
    "residual_to_action",
    "zero_init_actor_mean",
]

_MOISTURE_REASON_PREFIX = "Partikelfeuchte"
"""Start of the infeasibility reason of ``steady_state.py`` for a moisture outside the plausible
range (S0.4); only this reason, with the moisture ABOVE the critical moisture, is not masked."""


@dataclass(frozen=True, slots=True)
class ResidualConfig:
    """Fixed point (physical units) and residual range ``scale`` (normalised units)."""

    fixed_inlet_temperature_celsius: float
    fixed_gas_velocity_m_per_s: float
    scale: float

    def __post_init__(self) -> None:
        if not 0.0 < self.scale <= 1.0:
            raise ValueError("scale must be in (0, 1]")


def residual_to_action(
    delta: np.ndarray, anchor_normalised: np.ndarray, scale: float
) -> np.ndarray:
    """Executed normalised action ``clip(a_fixed + scale * clip(delta), -1, 1)``."""
    clipped = np.clip(np.asarray(delta, dtype=np.float64), -1.0, 1.0)
    return np.clip(anchor_normalised + scale * clipped, -1.0, 1.0)


class ResidualActionWrapper(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Maps the agent's ``delta`` to the absolute action of the wrapped bandit environment."""

    def __init__(self, env: gym.Env[np.ndarray, np.ndarray], residual: ResidualConfig) -> None:
        super().__init__(env)
        base: Any = env.unwrapped
        self._residual = residual
        self.anchor_normalised = np.array(
            [
                _normalise(
                    residual.fixed_inlet_temperature_celsius,
                    base._inlet_bound.minimum,
                    base._inlet_bound.maximum,
                ),
                _normalise(
                    residual.fixed_gas_velocity_m_per_s,
                    base._gas_bound.minimum,
                    base._gas_bound.maximum,
                ),
            ]
        )

    @property
    def residual_config(self) -> ResidualConfig:
        return self._residual

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        absolute = residual_to_action(action, self.anchor_normalised, self._residual.scale)
        observation, reward, terminated, truncated, info = self.env.step(absolute)
        info["residual_delta"] = np.asarray(action, dtype=np.float64).tolist()
        return observation, reward, terminated, truncated, info


def zero_init_actor_mean(model: Any) -> None:
    """Zero weights and bias of the actor's mean layer (stable-baselines3 SAC/TD3 actor ``mu``).

    The deterministic action is then ``tanh(0) = 0`` for every observation: the policy starts
    exactly at the fixed point. The log-std layer is left alone (the exploration is the default).
    """
    layer = model.actor.mu
    layer.weight.data.zero_()
    layer.bias.data.zero_()


class SeparatedLagrangeBanditEnv(LagrangeBanditEnv):
    """Lagrangian bandit with the moisture cost separated from the solver checks (see module)."""

    def _mask_reason(self, info: dict[str, Any]) -> str | None:
        if info.get("solve_failed"):
            return "solve_failed"
        check = info["constraint_check"]
        if check["solver_feasible"]:
            return None
        reasons = info.get("infeasibility_reasons", [])
        critical = self._base_config.material.critical_moisture_kg_per_kg
        moisture_only = bool(reasons) and all(
            str(reason).startswith(_MOISTURE_REASON_PREFIX) for reason in reasons
        )
        if moisture_only and float(check["moisture_kg_per_kg"]) > critical:
            return None
        return "solver_infeasible_other"

    def lagrangian_reward(
        self,
        info: dict[str, Any],
        *,
        ambient_temperature_celsius: float,
        ambient_relative_humidity: float,
    ) -> tuple[float, dict[str, Any]]:
        bin_index = self.bin_of(ambient_temperature_celsius, ambient_relative_humidity)
        multiplier = float(self._multipliers.reshape(-1)[bin_index])
        info["lagrange_bin"] = bin_index
        info["lagrange_multiplier_kw"] = multiplier
        reason = self._mask_reason(info)
        info["masked"] = reason is not None
        info["mask_reason"] = reason
        if reason is not None:
            info["lagrange_g_tilde"] = float("nan")
            return 0.0, info  # never stored, never used by the dual update
        energy, rest_penalty, g_tilde = self._components(info)
        if not info["constraint_check"]["solver_feasible"]:
            # moisture above the critical moisture: the +1 step is replaced by lambda * g_tilde
            rest_penalty -= self._penalty_weight_kw
        info["lagrange_energy_kw"] = energy
        info["lagrange_rest_penalty_kw"] = rest_penalty
        info["lagrange_g_tilde"] = g_tilde
        return -(energy + rest_penalty + multiplier * g_tilde), info


class MaskedReplayBuffer(ReplayBuffer):
    """Replay buffer that drops transitions whose info says ``masked`` (and counts them)."""

    n_masked: int = 0

    def add(  # type: ignore[override]
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        if infos and infos[0].get("masked"):
            self.n_masked += 1
            return
        super().add(obs, next_obs, action, reward, done, infos)
