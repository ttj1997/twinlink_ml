r"""SH1: moisture shield for the Lagrangian bandit -- a shield instead of a cliff.

Befund 68 (S0.9): the critic smears the 50 kW step that the solver check adds at the validity
boundary of the kinetics (product moisture above the critical moisture) over about 0.1 kg/kg, and
the policy stays correspondingly far before it. The shield removes the step from what the agent can
reach, instead of changing the reward function.

Before an action is executed the steady-state product moisture of the PROPOSED action is computed
(one solver call, the same solve the evaluation needs anyway). If it exceeds the tightened limit
``quality limit - m`` (the target of the Lagrange term,
:attr:`LagrangeBanditEnv.tightened_limit_kg_per_kg`), the action is pushed along a fixed
direction, the inlet temperature upwards at constant gas velocity, onto the tightened limit by a
one-dimensional root search (Brent, tolerance in kelvin); that action is executed and rewarded
with the Lagrangian reward. The replay buffer of the learner still holds the PROPOSED action
(stable-baselines3 stores the action the policy sampled, the environment only receives it),
paired with the reward of the executed one. For proposals beyond the boundary the reward is
therefore flat in the inlet temperature instead of falling off a cliff.

Why the inlet temperature: it is the direct drying lever, the steady-state moisture falls
monotonically with it (checked by ``scripts/probe_sh1_shield_probe.py``), and the gas velocity
also enters the fluidisation window, which the shield should not touch.

Limits of the shield, all counted in ``info``:
  * the proposal is solver-infeasible (``solve_failed``): the moisture is undefined, no shift,
    the Lagrange failure value applies as before;
  * even the upper inlet-temperature bound does not reach the tightened limit: the action is
    executed at the bound (``shield_saturated``), the moisture stays above the limit;
  * the shifted action may violate the bed-temperature limit, then the rest penalty of the
    Lagrangian reward applies as for any action.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq

from .env import OperatingPointBanditEnv, _denormalise, _normalise
from .lagrange import LagrangeBanditEnv

__all__ = ["ShieldConfig", "ShieldedLagrangeBanditEnv"]

_FAILED_MOISTURE_KG_PER_KG = 1.0
"""Moisture assigned to a solver failure inside the root search (cold side: too wet)."""


@dataclass(frozen=True, slots=True)
class ShieldConfig:
    """Settings of the shield. ``xtol_celsius`` is the root tolerance in the inlet temperature."""

    xtol_celsius: float = 0.05
    max_iterations: int = 50

    def __post_init__(self) -> None:
        if self.xtol_celsius <= 0.0:
            raise ValueError("xtol_celsius must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")


class ShieldedLagrangeBanditEnv(LagrangeBanditEnv):
    """:class:`LagrangeBanditEnv` whose ``step`` executes the shielded action."""

    def __init__(self, *args: Any, shield: ShieldConfig | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._shield = shield if shield is not None else ShieldConfig()

    @property
    def shield_config(self) -> ShieldConfig:
        return self._shield

    def _plain_evaluate(
        self, t_in: float, u_gas: float, t_amb: float, phi_amb: float
    ) -> tuple[float, dict[str, Any]]:
        return OperatingPointBanditEnv.evaluate_operating_point(
            self,
            inlet_temperature_celsius=t_in,
            gas_velocity_m_per_s=u_gas,
            ambient_temperature_celsius=t_amb,
            ambient_relative_humidity=phi_amb,
        )

    def shield_action(
        self, t_prop: float, u_gas: float, t_amb: float, phi_amb: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Executed inlet temperature and the plain info of the executed point.

        Returns ``(shield_record, plain_info)``; ``shield_record`` holds the executed inlet
        temperature, the flags ``active``/``saturated``, the moisture of the proposal and the
        number of solver calls.
        """
        t_max = self._inlet_bound.maximum
        target = self.tightened_limit_kg_per_kg
        n_solves = 1
        _, info = self._plain_evaluate(t_prop, u_gas, t_amb, phi_amb)
        t_exec = t_prop
        proposed_moisture = float("nan")
        active = False
        saturated = False
        if not info.get("solve_failed"):
            proposed_moisture = float(info["constraint_check"]["moisture_kg_per_kg"])
            active = proposed_moisture > target

        if active:
            calls = [0]
            known: dict[float, float] = {t_prop: proposed_moisture - target}

            def excess(t_in: float) -> float:
                if t_in in known:  # the bracket ends are not solved a second time
                    return known[t_in]
                calls[0] += 1
                _, probe = self._plain_evaluate(t_in, u_gas, t_amb, phi_amb)
                if probe.get("solve_failed"):
                    value = _FAILED_MOISTURE_KG_PER_KG - target
                else:
                    value = float(probe["constraint_check"]["moisture_kg_per_kg"]) - target
                known[t_in] = value
                return value

            f_max = excess(t_max)
            if f_max >= 0.0:
                t_exec = t_max
                saturated = True
            else:
                t_exec = float(
                    brentq(
                        excess,
                        t_prop,
                        t_max,
                        xtol=self._shield.xtol_celsius,
                        maxiter=self._shield.max_iterations,
                    )
                )
            n_solves += calls[0]
            _, info = self._plain_evaluate(t_exec, u_gas, t_amb, phi_amb)
            n_solves += 1
        record = {
            "t_exec": t_exec,
            "active": active,
            "saturated": saturated,
            "proposed_moisture": proposed_moisture,
            "n_solves": n_solves,
        }
        return record, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._disturbance is None:
            raise RuntimeError("step() vor reset() aufgerufen")
        started = time.perf_counter()
        t_amb, phi_amb = self._disturbance
        requested = np.asarray(action, dtype=np.float64)
        proposed = np.clip(requested, -1.0, 1.0)
        t_prop = _denormalise(
            float(proposed[0]), self._inlet_bound.minimum, self._inlet_bound.maximum
        )
        u_gas = _denormalise(float(proposed[1]), self._gas_bound.minimum, self._gas_bound.maximum)

        record, info = self.shield_action(t_prop, u_gas, t_amb, phi_amb)
        reward, info = self.lagrangian_reward(
            info, ambient_temperature_celsius=t_amb, ambient_relative_humidity=phi_amb
        )
        executed = np.array(
            [
                _normalise(record["t_exec"], self._inlet_bound.minimum, self._inlet_bound.maximum),
                proposed[1],
            ]
        )
        info["action_normalised_requested"] = requested.tolist()
        info["action_normalised_applied"] = executed.tolist()
        info["shield_active"] = record["active"]
        info["shield_saturated"] = record["saturated"]
        info["shield_proposed_inlet_temperature_celsius"] = t_prop
        info["shield_executed_inlet_temperature_celsius"] = record["t_exec"]
        info["shield_proposed_moisture_kg_per_kg"] = record["proposed_moisture"]
        info["shield_n_solves"] = record["n_solves"]
        info["shield_step_seconds"] = time.perf_counter() - started

        observation = np.array([t_amb, phi_amb], dtype=np.float32)
        return observation, reward, True, False, info
