r"""Controllers of the study: references, the two rules, the two MPCs and the hindsight optimum.

Every controller sees the plant, the market and the content at the start of the hour and returns
the pump heat ``Q`` and the storage duties ``(q_c, q_d)`` for the hour. The run
(:func:`twinlink_ml.storage_price.evaluate.run_period`) applies the same projection on the limits to
every controller and counts what it had to cut.

Rules (no optimisation over time; per hour the step programme of Befund 101,
``min c_U U + p_t Q - v_t (q_c - q_d)`` with ``p_t = c_el,t / COP`` and a value ``v_t`` of stored
heat, solved exactly by the vertex enumeration of :mod:`twinlink_ml.heatpump.storage`, here with
the power limits of the storage in the bounds of ``q_c`` and ``q_d``):

``R0``  ``v_t = kappa c_U`` constant (the tuned constant rule of Befund 101), ``kappa`` tuned on
        the training months. Charges with the pump whenever pump heat is cheaper than
        ``kappa c_U``, discharges whenever that displaces gas or dearer pump heat.
``R1``  schedule-aware: an hour is CHEAP when its price is at most the ``q``-quantile of the price
        information of the next ``r1_window_h`` hours (the hour itself included; information as M2:
        known until midnight, then the price of the same hour today). Cheap: ``v_t = (1 - eps) c_U``
        (pump covers the demand and charges whenever pump heat is cheaper than gas; the storage only
        displaces gas). Not cheap: ``v_t = eps c_U`` (the storage is discharged first, the pump runs
        only for the rest and only when cheaper than gas, no charging). ``q`` tuned on the training
        months.

References: ``N0`` no pump, no storage (gas only after direct recovery); ``P0`` pump without
storage (the step programme with capacity 0, i.e. pump whenever cheaper than gas).

Programmes (:func:`twinlink_ml.storage_price.lp.solve_window`, re-solved every hour, the first hour
applied, terminal value ``end_value`` per kWh left at the end of the window):

``M1``  perfect price information over the horizon (upper bound of the MPC);
``M2``  price information as R1;
``P*``  hindsight: one programme over the whole evaluated period with the true prices and the end
        credit of the evaluation (the exact optimum of the evaluation protocol; reference only).

Demand and waste heat follow the production plan and are known to every controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# The exact step solver of Befund 101 (vertex enumeration of the two-variable step programme,
# tested against HiGHS in tests/test_heatpump.py). Private there; imported, not copied, so that
# the rules of this study are the rules of Befund 101.
from ..heatpump.storage import CoupledPieces, _case_step, _case_table
from .lp import WindowSolution, solve_window
from .market import Market
from .plant import Plant

__all__ = [
    "MPC",
    "Action",
    "Controller",
    "Hindsight",
    "NoPump",
    "StepRule",
    "pump_only",
    "r0_rule",
    "r1_rule",
    "r1_values",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Action:
    """Pump heat and storage duties of one hour (kW)."""

    heat_kw: float
    charge_kw: float
    discharge_kw: float


class Controller(ABC):
    """One controller; :meth:`start` once per evaluated period, then :meth:`decide` per hour."""

    name: str

    def __init__(self, plant: Plant, market: Market) -> None:
        self.plant = plant
        self.market = market
        self.hours: IntArray = np.zeros(0, dtype=np.int64)

    def start(self, hours: IntArray) -> None:
        """Prepare a run over the consecutive hours ``hours`` (content 0 at the start)."""
        self.hours = np.asarray(hours, dtype=np.int64)

    @abstractmethod
    def decide(self, k: int, stored_kwh: float) -> Action:
        """Action for hour ``hours[k]`` given the content at its start."""


class NoPump(Controller):
    """``N0``: neither pump nor storage."""

    name = "N0"

    def decide(
        self,
        k: int,  # noqa: ARG002 - shared interface
        stored_kwh: float,  # noqa: ARG002 - shared interface
    ) -> Action:
        return Action(0.0, 0.0, 0.0)


def _slice(pieces: CoupledPieces, hours: IntArray) -> CoupledPieces:
    return CoupledPieces(
        offset_kw=pieces.offset_kw[hours],
        alpha=pieces.alpha,
        beta=pieces.beta,
        gamma=pieces.gamma,
        cop=pieces.cop,
    )


class StepRule(Controller):
    """Step programme with a value ``v_t`` of stored heat (R0, R1, P0; see module docstring)."""

    def __init__(
        self,
        plant: Plant,
        market: Market,
        name: str,
        stored_value: FloatArray,
        *,
        use_storage: bool = True,
    ) -> None:
        super().__init__(plant, market)
        self.name = name
        self.stored_value_year = np.asarray(stored_value, dtype=np.float64)
        self.use_storage = use_storage

    def start(self, hours: IntArray) -> None:
        super().start(hours)
        pl, mk = self.plant, self.market
        pieces = _slice(pl.pieces, self.hours)
        self._p = mk.electricity[self.hours] / pl.cop
        v = self.stored_value_year[self.hours]
        self._charge = _case_table(pieces, "charge", pl.size_kw, mk.heat_value, self._p, -v)
        self._discharge = _case_table(pieces, "discharge", pl.size_kw, mk.heat_value, self._p, v)

    def decide(self, k: int, stored_kwh: float) -> Action:
        pl = self.plant
        content = pl.keep * stored_kwh
        if self.use_storage:
            room = min(pl.charge_max_kw, max(0.0, (pl.capacity_kwh - content) / pl.dt_h))
            avail = min(pl.discharge_max_kw, max(0.0, content / pl.dt_h))
        else:
            room = avail = 0.0
        c_u, p = self.market.heat_value, float(self._p[k])
        fc, qc_q, qc_y = _case_step(self._charge, k, pl.size_kw, np.array([room]), c_u, p)
        fd, qd_q, qd_y = _case_step(self._discharge, k, pl.size_kw, np.array([avail]), c_u, p)
        if fc[0] <= fd[0]:
            return Action(float(qc_q[0]), float(qc_y[0]), 0.0)
        return Action(float(qd_q[0]), 0.0, float(qd_y[0]))


def r0_rule(plant: Plant, market: Market, kappa: float) -> StepRule:
    """``R0`` with ``v = kappa c_U`` (``kappa`` from the training months)."""
    v = np.full(len(market.electricity), kappa * market.heat_value)
    return StepRule(plant, market, "R0", v)


def r1_values(market: Market, quantile: float, window_h: int, epsilon: float) -> FloatArray:
    """``v_t`` of R1 for every hour (cheap: ``(1-eps) c_U``, else ``eps c_U``)."""
    info = market.known[:, :window_h]
    threshold = np.quantile(info, quantile, axis=1)
    cheap = market.electricity <= threshold + 1e-12
    c_u = market.heat_value
    return np.where(cheap, (1.0 - epsilon) * c_u, epsilon * c_u).astype(np.float64)


def r1_rule(
    plant: Plant, market: Market, quantile: float, window_h: int, epsilon: float
) -> StepRule:
    """``R1`` with the ``quantile`` from the training months."""
    if window_h > market.horizon_h:
        raise ValueError("price information shorter than the R1 window")
    return StepRule(plant, market, "R1", r1_values(market, quantile, window_h, epsilon))


def pump_only(plant: Plant, market: Market, epsilon: float) -> StepRule:
    """``P0``: pump whenever cheaper than gas, no storage."""
    v = np.full(len(market.electricity), epsilon * market.heat_value)
    return StepRule(plant, market, "P0", v, use_storage=False)


def _window(
    plant: Plant,
    market: Market,
    rows: IntArray,
    electricity: FloatArray,
    initial_kwh: float,
) -> WindowSolution:
    pc = plant.pieces
    return solve_window(
        pc.offset_kw[rows],
        pc.alpha,
        pc.beta,
        pc.gamma,
        electricity=electricity,
        heat_value=market.heat_value,
        cop=plant.cop,
        q_max_kw=plant.size_kw,
        charge_max_kw=plant.charge_max_kw,
        discharge_max_kw=plant.discharge_max_kw,
        capacity_kwh=plant.capacity_kwh,
        keep=plant.keep,
        dt_h=plant.dt_h,
        initial_kwh=initial_kwh,
        terminal_value=market.end_value,
    )


class MPC(Controller):
    """Rolling-horizon programme (``M1``: true prices, ``M2``: price information as R1)."""

    def __init__(
        self,
        plant: Plant,
        market: Market,
        *,
        horizon_h: int,
        information: Literal["perfect", "known"],
    ) -> None:
        super().__init__(plant, market)
        if information == "known" and horizon_h > market.horizon_h:
            raise ValueError("price information shorter than the horizon")
        self.horizon_h = horizon_h
        self.information = information
        self.name = f"{'M1' if information == 'perfect' else 'M2'}_{horizon_h}h"

    def decide(self, k: int, stored_kwh: float) -> Action:
        t = int(self.hours[k])
        n_year = len(self.market.electricity)
        rows = np.arange(t, min(t + self.horizon_h, n_year))
        if self.information == "perfect":
            el = self.market.electricity[rows]
        else:
            el = self.market.known[t, : len(rows)]
        plan = _window(self.plant, self.market, rows, el, stored_kwh)
        return Action(float(plan.heat_kw[0]), float(plan.charge_kw[0]), float(plan.discharge_kw[0]))


class Hindsight(Controller):
    """``P*``: one programme over the whole period, true prices, replayed hour by hour."""

    name = "P*"

    def start(self, hours: IntArray) -> None:
        super().start(hours)
        self.plan = _window(
            self.plant, self.market, self.hours, self.market.electricity[self.hours], 0.0
        )

    def decide(self, k: int, stored_kwh: float) -> Action:  # noqa: ARG002 - shared interface
        p = self.plan
        return Action(float(p.heat_kw[k]), float(p.charge_kw[k]), float(p.discharge_kw[k]))
