"""P1 of Befund 107 with a bypass of the recovery and an optional pressurised medium.

:class:`JointLineProblem` is :class:`~twinlink_ml.surrogate_opt.line_problem.LineProblem` (read
only) with two additions:

Bypass (``medium_kg_per_h <= 0``)
    The pump is off and the recovery exchanger carries no heat: ``Q = 0``, the exhaust leaves the
    exchanger at its inlet temperature, no pump power, no condensate. The exchangers stay in the
    gas paths, so both fans see the same pressure drops as with recovery (same plant, circuit
    stagnant). With a medium flow ``> 0`` the line is :class:`LineProblem` unchanged (tested). The
    medium constraint of the bypass is ``T_amb - T_medium,max`` (the stagnant medium at ambient).
Pressurised medium (``pressurised=True``)
    No upper limit of the medium temperature (:meth:`LineMode.physical` with
    ``htm_max_celsius=None``); the medium constraint is dropped, the medium temperature and the
    saturation pressure of water at it are reported.

Side metrics added to those of :class:`LineProblem`: exhaust temperature behind the recovery
exchanger, medium temperatures, what binds the stage, the medium flow, the saturation pressure of
water at the hot medium temperature.

:class:`RestrictedProblem` fixes some decisions of a problem and leaves the others free (S0: inlet
temperature and air with the bypass; S1 and S3: the medium flow at a fixed operating point).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from twinlink_ml.flowsheet.line import LineMode, LineResult
from twinlink_ml.flowsheet.loop import StageOut
from twinlink_ml.flowsheet.thermo import dew_point
from twinlink_ml.flowsheet.validators import water_saturation_pressure_pa
from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import SiMaterial
from twinlink_ml.surrogate_opt.line_problem import LineProblem, LineSettings, _LineWithHeatLoss
from twinlink_ml.surrogate_opt.problem import (
    Box,
    Evaluation,
    ParameterSpec,
    ProblemFamily,
    SteadyProblem,
)

__all__ = [
    "BINDINGS",
    "MEDIUM_INDEX",
    "JointLineFamily",
    "JointLineProblem",
    "RestrictedProblem",
    "bypass_stage",
]

MEDIUM_INDEX = 2
"""Index of the medium flow in the decision vector (inlet temperature, air, medium)."""

BINDINGS = (
    "bypass", "no_driving_force", "approach", "medium_limit", "sink_target", "exhaust_floor",
)  # fmt: skip
"""What limits the duty of the stage; ``side["binding_index"]`` indexes this tuple."""


def bypass_stage(exhaust: c.Stream, gas: c.Stream) -> StageOut:
    """The stage with the pump off: no duty, both streams leave as they enter."""
    t_ex, t_gas = exhaust.temperature_celsius, gas.temperature_celsius
    return StageOut(
        1, 1, 0.0, t_gas, t_gas, t_ex, t_ex, None, None, "bypass", math.nan, None,
        dew_point(exhaust.humidity_kg_per_kg), 0.0, 0.0,
    )  # fmt: skip


class _LineWithBypass(_LineWithHeatLoss):
    """The line of :class:`LineProblem` whose recovery is bypassed at a medium flow of zero."""

    def recovery(
        self,
        material: SiMaterial,
        exhaust: c.Stream,
        gas: c.Stream,
        htm_kg_per_s: float,
        htm_cp_j_per_kg_k: float,
        target_celsius: float = math.inf,
    ) -> StageOut:
        if htm_kg_per_s <= 0.0:
            return bypass_stage(exhaust, gas)
        return super().recovery(
            material, exhaust, gas, htm_kg_per_s, htm_cp_j_per_kg_k, target_celsius
        )


class JointLineProblem(LineProblem):
    """P1 with the bypass and the medium variant (see the module doc)."""

    def __init__(
        self,
        settings: LineSettings,
        *,
        heater_efficiency: float,
        heat_loss_ua_w_per_k: float,
        dt_min_k: float,
        pressurised: bool,
        name: str = "P1_joint",
    ) -> None:
        super().__init__(
            settings,
            heater_efficiency=heater_efficiency,
            heat_loss_ua_w_per_k=heat_loss_ua_w_per_k,
            dt_min_k=dt_min_k,
            name=name,
        )
        self.pressurised = pressurised
        self.dt_min_k = dt_min_k
        if pressurised:
            self.mode = LineMode.physical(htm_max_celsius=None, dt_min_k=dt_min_k)
            self.constraints = tuple(c for c in self.constraints if c.name != "medium_temperature")
        self.line = _LineWithBypass(self.mode, heat_loss_ua_w_per_k)
        self._last: LineResult | None = None

    def solve(self, u: np.ndarray, w: np.ndarray) -> LineResult:
        result = super().solve(u, w)
        self._last = result
        return result

    def _evaluate(self, u: np.ndarray, w: np.ndarray) -> Evaluation:
        ev = super()._evaluate(u, w)
        r = self._last
        assert r is not None
        g = np.array(ev.constraints, dtype=np.float64)
        bypass = r.inputs.htm_kg_per_s <= 0.0
        if self.pressurised:
            g = g[:3]
        elif bypass:
            g[3] = r.inputs.fresh_gas_celsius - self.settings.medium_max_celsius
        hot = r.medium_hot_celsius
        side = dict(ev.side)
        side.update(
            {
                "exhaust_after_recovery_celsius": r.stage.exhaust_out_celsius,
                "medium_cold_celsius": r.medium_cold_celsius,
                "medium_kg_per_h": r.inputs.htm_kg_per_s * 3600.0,
                "bypass": float(bypass),
                "medium_saturation_pressure_bar": (
                    math.nan if math.isnan(hot) else water_saturation_pressure_pa(hot) / 1e5
                ),
            }
        )
        side["binding_index"] = float(BINDINGS.index(r.stage.binding))
        return Evaluation(ev.energy_kw, g, side)


class JointLineFamily(ProblemFamily):
    """:class:`JointLineProblem` at the parameters of P1, ``dt_min_k`` set by the factor."""

    def __init__(
        self,
        settings: LineSettings,
        parameters: Sequence[ParameterSpec],
        *,
        dt_min_k: float,
        pressurised: bool,
    ) -> None:
        names = {p.name for p in parameters}
        if names != {"heater_efficiency", "heat_loss_ua_w_per_k", "dt_min_k"}:
            raise ValueError(f"P1 needs heater_efficiency, heat_loss_ua_w_per_k, dt_min_k: {names}")
        self.settings = settings
        # the factor replaces the truth of the (hidden) approach; the other truths stay as in P1
        self.parameters = tuple(
            replace(p, truth=dt_min_k) if p.name == "dt_min_k" else p for p in parameters
        )
        self.dt_min_k = dt_min_k
        self.pressurised = pressurised

    @property
    def nominal_disturbance(self) -> np.ndarray:
        return np.asarray(self.settings.nominal_disturbance, dtype=np.float64)

    def build(self, values: Mapping[str, float]) -> JointLineProblem:
        v = {**self.nominal_values, **values}
        return JointLineProblem(
            self.settings,
            heater_efficiency=v["heater_efficiency"],
            heat_loss_ua_w_per_k=v["heat_loss_ua_w_per_k"],
            dt_min_k=v["dt_min_k"],
            pressurised=self.pressurised,
        )

    def truth(self) -> JointLineProblem:
        return self.build(self.truth_values)


@dataclass(frozen=True)
class _Fix:
    free: tuple[int, ...]
    full: tuple[float, ...]


class RestrictedProblem(SteadyProblem):
    """``base`` with the decisions outside ``free`` fixed at the values of ``fixed``."""

    def __init__(self, base: JointLineProblem, free: Sequence[int], fixed: Sequence[float]) -> None:
        super().__init__()
        if len(fixed) != base.decisions.dim:
            raise ValueError("fixed needs one value per decision of the base problem")
        self.base = base
        self._fix = _Fix(tuple(int(i) for i in free), tuple(float(v) for v in fixed))
        d = base.decisions
        self.name = f"{base.name}/free{list(self._fix.free)}"
        self.decisions = Box(
            tuple(d.names[i] for i in self._fix.free),
            tuple(d.lower[i] for i in self._fix.free),
            tuple(d.upper[i] for i in self._fix.free),
        )
        self.disturbances = base.disturbances
        self.constraints = base.constraints

    def full(self, u: np.ndarray) -> np.ndarray:
        """The decision vector of the base problem."""
        out = np.array(self._fix.full, dtype=np.float64)
        out[list(self._fix.free)] = np.asarray(u, dtype=np.float64)
        return out

    def _evaluate(self, u: np.ndarray, w: np.ndarray) -> Evaluation:
        return self.base.evaluate(self.full(u), w)
