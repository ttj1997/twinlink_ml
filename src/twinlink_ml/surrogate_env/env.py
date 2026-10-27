r"""Bandit of Befund 93/96 (cell ``ressep``) with a learned operating-point model as transition.

:class:`LearnedSeparatedLagrangeBanditEnv` is
:class:`~twinlink_ml.rl.residual.SeparatedLagrangeBanditEnv` with exactly one change: the steady
state of the grey-box solver is replaced by the steady prediction of an
:class:`~.model.OperatingPointModel`. Reward, Lagrange multipliers per context cell, separated
moisture cost, residual action and masking rules are inherited unchanged.

The info dict carries the fields that the inherited reward reads, built from the prediction:

* moisture = the moisture of the constraint (U3: ``mean + 2 std``), its violation against the
  quality limit and the bed-temperature violation against ``BED_TEMPERATURE_LIMIT_C`` enter
  ``normalized_violation`` with the weights of ``check_constraints``;
* the fluidisation window contributes 0: it never binds on this action box (Befund 62 a; d50 is a
  constant of the steady state), and the learned model has no d50 head;
* ``solver_feasible`` is True for a valid prediction. A point without a valid fixed point (any head,
  any member) is reported as ``solve_failed`` -- the grey-box equivalent of "no steady state" -- and
  is therefore masked by the inherited rule of the separated cell: never stored, counted. For the
  controller such a point is not admissible.

Evaluation of a policy is never done here: the truth is always the grey-box model
(``OperatingPointBanditEnv`` against the limit 0.15).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from twinlink_ml.rl.env import BanditEnvConfig
from twinlink_ml.rl.lagrange import LagrangeConfig
from twinlink_ml.rl.residual import SeparatedLagrangeBanditEnv
from twinlink_ml.simulation.operating_point import (
    BED_TEMPERATURE_LIMIT_C,
    TOLERANCE_MOISTURE_KG_PER_KG,
    TOLERANCE_TEMPERATURE_K,
)

from .model import FixedInputs, OperatingPointModel

__all__ = ["LearnedSeparatedLagrangeBanditEnv", "learned_info"]


def learned_info(
    prediction: dict[str, Any],
    *,
    quality_limit_kg_per_kg: float,
    inlet_temperature_celsius: float,
    gas_velocity_m_per_s: float,
    ambient_temperature_celsius: float,
    ambient_relative_humidity: float,
    fixed: FixedInputs,
    energy_unit: str,
) -> dict[str, Any]:
    """Info dict of one predicted point in the layout of ``OperatingPointBanditEnv``."""
    requested = {
        "inlet_temperature_celsius": inlet_temperature_celsius,
        "gas_velocity_m_per_s": gas_velocity_m_per_s,
        "nozzle_pressure_bar": fixed.nozzle_pressure_bar,
        "spray_rate_kg_per_h": fixed.spray_rate_kg_per_h,
    }
    common = {
        "action_physical_requested": requested,
        "action_physical_applied": requested,
        "ambient_temperature_celsius": ambient_temperature_celsius,
        "ambient_relative_humidity": ambient_relative_humidity,
        "learned_prediction": prediction,
    }
    if not prediction["valid"]:
        return {
            "feasible": False,
            "solve_failed": True,
            "solve_failure_reason": "learned model: no valid fixed point",
            **common,
        }
    moisture = float(prediction["moisture_for_constraint_kg_per_kg"])
    bed = float(prediction["bed_temperature_celsius"])
    moisture_violation = max(0.0, moisture - quality_limit_kg_per_kg)
    bed_violation = max(0.0, bed - BED_TEMPERATURE_LIMIT_C)
    normalized = moisture_violation / max(quality_limit_kg_per_kg, 1e-6) + bed_violation / 10.0
    check = {
        "moisture_kg_per_kg": moisture,
        "moisture_violation_kg_per_kg": moisture_violation,
        "gas_velocity_m_per_s": gas_velocity_m_per_s,
        "bed_temperature_celsius": bed,
        "bed_temperature_violation_k": bed_violation,
        "solver_feasible": True,
        "feasible": moisture_violation <= TOLERANCE_MOISTURE_KG_PER_KG
        and bed_violation <= TOLERANCE_TEMPERATURE_K,
        "normalized_violation": normalized,
    }
    return {
        "feasible": bool(check["feasible"]),
        "solve_failed": False,
        "energy_target": float(prediction["energy_kw"]),
        "energy_target_unit": energy_unit,
        "constraint_check": check,
        "infeasibility_reasons": [],
        **common,
    }


class LearnedSeparatedLagrangeBanditEnv(SeparatedLagrangeBanditEnv):
    """Cell ``ressep`` of Befund 93/96 on a learned operating-point model (see module docstring)."""

    def __init__(
        self,
        model: OperatingPointModel,
        fixed: FixedInputs,
        lagrange: LagrangeConfig,
        config: BanditEnvConfig | None = None,
    ) -> None:
        super().__init__(lagrange, config)
        if fixed.nozzle_pressure_bar != self._env_config.fixed_nozzle_pressure_bar:
            raise ValueError("nozzle pressure of the model differs from the bandit's")
        self._model = model
        self._fixed = fixed
        self.n_invalid = 0

    @property
    def model(self) -> OperatingPointModel:
        return self._model

    def _evaluate_operating_point_raw(
        self,
        *,
        inlet_temperature_celsius: float,
        gas_velocity_m_per_s: float,
        ambient_temperature_celsius: float,
        ambient_relative_humidity: float,
    ) -> tuple[float, dict[str, Any]]:
        prediction = self._model.predict(
            np.array([[inlet_temperature_celsius, gas_velocity_m_per_s]]),
            np.array([[ambient_temperature_celsius, ambient_relative_humidity]]),
        )
        point = prediction.point(0)
        point["status_counts"] = prediction.status_counts
        info = learned_info(
            point,
            quality_limit_kg_per_kg=self._quality_limit_value(),
            inlet_temperature_celsius=inlet_temperature_celsius,
            gas_velocity_m_per_s=gas_velocity_m_per_s,
            ambient_temperature_celsius=ambient_temperature_celsius,
            ambient_relative_humidity=ambient_relative_humidity,
            fixed=self._fixed,
            energy_unit=self._base_config.energy.target,
        )
        if info["solve_failed"]:
            self.n_invalid += 1
            return 0.0, info
        check = info["constraint_check"]
        reward = -(info["energy_target"] + self._penalty_weight_kw * check["normalized_violation"])
        return float(reward), info

    def _quality_limit_value(self) -> float:
        return float(self._base_config.material.quality_moisture_limit_kg_per_kg)
