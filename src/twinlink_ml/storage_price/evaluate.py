r"""Evaluation protocol: one run per calendar month, the same projection for every controller.

Per hour ``t`` with content ``s`` at its start the requested action is projected on

* power limits: ``0 <= Q <= Q_max``, ``0 <= q_c <= P_c``, ``0 <= q_d <= P_d``;
* charge limits: ``q_c dt <= C - (1 - lambda dt) s`` and ``q_d dt <= (1 - lambda dt) s``;

a cut larger than ``tolerance x`` the limit (power) or ``x C`` (content) is counted as a
violation of that kind; the projected action is applied:
``s+ = (1 - lambda dt) s + (q_c - q_d) dt``. Cost of the hour ``dt (c_U U + c_el Q / COP)``, CO2
``dt (U / eta x 0.201 + Q / COP x 0.366)``. Every month starts with an empty storage; the content
left at its end is credited at ``end_value`` (EUR/kWh) and, for CO2, at the CO2 of the pump heat at
the mean factor (``0.366 / COP`` per kWh). The time of every :meth:`Controller.decide` call is
recorded (``time.perf_counter``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .controllers import Action, Controller
from .market import Market
from .plant import Plant
from .prices import Clock

__all__ = ["PeriodResult", "month_hours", "project", "run_months", "run_period"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def month_hours(clock: Clock, month: int) -> IntArray:
    """Consecutive hour indices of a calendar month."""
    idx = np.nonzero(clock.month == month)[0]
    if len(idx) == 0 or np.any(np.diff(idx) != 1):
        raise ValueError(f"month {month} is not one block of hours")
    return idx.astype(np.int64)


@dataclass(frozen=True, slots=True)
class Projection:
    """Projected action and the cuts (kW) of each kind."""

    action: Action
    cut_power_kw: float
    cut_content_kw: float


def project(plant: Plant, action: Action, stored_kwh: float) -> Projection:
    """Clip an action on the power and charge limits (module docstring)."""
    content = plant.keep * stored_kwh
    q = min(max(action.heat_kw, 0.0), plant.size_kw)
    qc0 = max(action.charge_kw, 0.0)
    qd0 = max(action.discharge_kw, 0.0)
    qc1 = min(qc0, plant.charge_max_kw)
    qd1 = min(qd0, plant.discharge_max_kw)
    # charge and discharge in one hour: the net change must fit the content limits
    room = max(0.0, (plant.capacity_kwh - content) / plant.dt_h)
    avail = max(0.0, content / plant.dt_h)
    qc = min(qc1, room + qd1)
    qd = min(qd1, avail + qc)
    cut_heat = max(action.heat_kw - plant.size_kw, -action.heat_kw, 0.0)
    cut_power = cut_heat + (qc0 - qc1) + (qd0 - qd1)
    cut_content = (qc1 - qc) + (qd1 - qd)
    return Projection(Action(q, qc, qd), cut_power, cut_content)


@dataclass(frozen=True, slots=True)
class PeriodResult:
    """Totals of one run (one month) of one controller."""

    controller: str
    month: int
    hours: int
    cost_eur: float  # including the end credit
    cost_operating_eur: float  # without the end credit
    co2_kg: float  # including the end credit
    gas_heat_kwh: float
    electricity_kwh: float
    pump_heat_kwh: float
    charged_kwh: float
    discharged_kwh: float
    loss_kwh: float
    end_kwh: float
    violations_power: int
    violations_content: int
    largest_cut_kw: float
    balance_residual_kwh: float
    decide_s_median: float
    decide_s_p95: float
    decide_s_mean: float
    decide_s_max: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def run_period(
    controller: Controller,
    plant: Plant,
    market: Market,
    hours: IntArray,
    *,
    month: int,
    tolerance: float,
    trace: dict[str, list[float]] | None = None,
) -> PeriodResult:
    """Run ``controller`` over ``hours`` from an empty storage (module docstring)."""
    controller.start(hours)
    s = 0.0
    dt = plant.dt_h
    cost = co2 = gas = elec = heat = charged = discharged = loss = 0.0
    v_power = v_content = 0
    largest = 0.0
    times = np.zeros(len(hours))
    tol_q = tolerance * max(1.0, plant.size_kw)
    tol_p = tolerance * max(1.0, plant.charge_max_kw, plant.discharge_max_kw)
    tol_s = tolerance * max(1.0, plant.capacity_kwh)
    for k, t in enumerate(hours):
        t0 = time.perf_counter()
        requested = controller.decide(k, s)
        times[k] = time.perf_counter() - t0
        pr = project(plant, requested, s)
        a = pr.action
        if pr.cut_power_kw > max(tol_q, tol_p):
            v_power += 1
        if pr.cut_content_kw * dt > tol_s:
            v_content += 1
        largest = max(largest, pr.cut_power_kw, pr.cut_content_kw)
        u = plant.utility_kw(int(t), a.heat_kw, a.charge_kw, a.discharge_kw)
        work = a.heat_kw / plant.cop
        cost += dt * (market.heat_value * u + market.electricity[t] * work)
        co2 += dt * (u * market.heat_co2 + work * market.electricity_co2)
        gas += dt * u
        elec += dt * work
        heat += dt * a.heat_kw
        charged += dt * a.charge_kw
        discharged += dt * a.discharge_kw
        loss += s * (1.0 - plant.keep)
        if trace is not None:
            trace.setdefault("stored_kwh", []).append(s)
            trace.setdefault("heat_kw", []).append(a.heat_kw)
            trace.setdefault("charge_kw", []).append(a.charge_kw)
            trace.setdefault("discharge_kw", []).append(a.discharge_kw)
            trace.setdefault("utility_kw", []).append(u)
        s = min(max(plant.keep * s + (a.charge_kw - a.discharge_kw) * dt, 0.0), plant.capacity_kwh)
    credit = s * market.end_value
    credit_co2 = s * market.electricity_co2 / plant.cop
    return PeriodResult(
        controller=controller.name,
        month=month,
        hours=len(hours),
        cost_eur=cost - credit,
        cost_operating_eur=cost,
        co2_kg=co2 - credit_co2,
        gas_heat_kwh=gas,
        electricity_kwh=elec,
        pump_heat_kwh=heat,
        charged_kwh=charged,
        discharged_kwh=discharged,
        loss_kwh=loss,
        end_kwh=s,
        violations_power=v_power,
        violations_content=v_content,
        largest_cut_kw=largest,
        balance_residual_kwh=charged - discharged - loss - s,
        decide_s_median=float(np.median(times)),
        decide_s_p95=float(np.percentile(times, 95)),
        decide_s_mean=float(np.mean(times)),
        decide_s_max=float(np.max(times)),
    )


def run_months(
    controller: Controller,
    plant: Plant,
    market: Market,
    clock: Clock,
    months: Sequence[int],
    *,
    tolerance: float,
) -> list[PeriodResult]:
    """One run per month."""
    return [
        run_period(controller, plant, market, month_hours(clock, m), month=m, tolerance=tolerance)
        for m in months
    ]
