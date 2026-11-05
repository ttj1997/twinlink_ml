r"""Follow-up study (``docs/36_campaign_network_2.md``): rolling dispatch planning P1 / P2.

**Hourly model (reduced heat cascade).** With one hot stream (source plus tank, both supply water)
the most heat the consumers can take in hour ``t`` at the hot-side heat ``H`` is the min-cut of the
problem-table cascade of K3:

    use_t(H) = min_b ( f(b) H + C_t(b) ),

``b`` over the shifted boundaries of the hour (supply, floor, start and end of every consumer),
``f(b)`` the share of the source heat above ``b``, ``C_t(b)`` the consumer demand below ``b``. Every
consumer has the same net CO2 weight per kWh (same efficiency of the replaced supply, same pump
share), so maximising the net CO2 saving is maximising the heat used, and the LP over a window is

    max sum_t u_t   s.t.  u_t <= f(b) (P + d_t - c_t) + C_t(b)  for every b,
                          E_{t+1} = (1 - loss) E_t + c_t - d_t,  0 <= E <= S,
                          0 <= c_t <= P,  0 <= d_t <= discharge_max.

Over the whole year this LP has the optimum of K3 (``target.run_k3``; checked in the tests). The
weights must be equal; :func:`cut_data` refuses other settings.

**Rolling planning.** At a planning hour the LP is solved over the window ``[t, t + horizon)`` with
the FORECAST consumer demand (no value of the content at the window end) from the TRUE content
``E_t``; until the next planning hour the plan's ``c`` and ``d`` are executed; the heat used in an
hour is ``use_t`` of the TRUE demand at the executed ``H``. A new plan is made every
``replan_every_h`` hours and at once when the observed state of a plant differs from the state the
current plan assumed for this hour (a change has happened that was not planned for now, or a
planned change is overdue). With hourly planning (``replan_every_h = 1``) every hour is a planning
hour.

* **P1** -- the standstill plan is known exactly (forecast = truth).
* **P2** -- the plan deviates: every change of state (start and end of a campaign) of the plan is
  the true one shifted by ``U[-24, +24] h`` (rounded to hours, order kept, fixed per draw; in the
  joint mode one plan for all plants). Correction: a change that has happened is taken at its true
  time; a planned change that is overdue is expected in the next hour (and the one after that again
  if it still has not come), the following planned changes keep their times but not earlier than
  that. This is a NAIVE adjustment of the plan, so P2 is a lower bound for planning with
  deviations.

With storage 0 there is nothing to plan: P1 = P2 = K3 without storage.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .config import CampaignNetworkConfig
from .profiles import build_consumers, plant_schedules

__all__ = [
    "CutData",
    "PlanForecast",
    "PlanResult",
    "TrueForecast",
    "cut_data",
    "planned_transitions",
    "rolling_plan",
    "use_of",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CutData:
    """Min-cut pieces of every hour: ``f`` (hours x pieces), demand below per plant and the rest."""

    f: FloatArray
    c_low: FloatArray
    c_plant: dict[str, FloatArray]

    def demand_below(self, on: dict[str, NDArray], rows: slice | NDArray) -> FloatArray:
        out = self.c_low[rows].copy()
        for name, base in self.c_plant.items():
            out += np.asarray(on[name], dtype=np.float64)[:, None] * base[rows]
        return out


def cut_data(config: CampaignNetworkConfig) -> CutData:
    """Cut pieces of the year for the consumers of ``config`` (process plants at full flow)."""
    etas = {p.gas_efficiency for p in config.plants}
    etas |= {config.hall.gas_efficiency, config.water.gas_efficiency}
    if len(etas) != 1:
        raise ValueError("the reduced model needs the same efficiency for every consumer")
    n = config.campaigns.hours
    full = build_consumers(config, {p.name: np.ones(n, dtype=bool) for p in config.plants})
    half = 0.5 * config.approach_k
    hs = config.source.supply_celsius - half
    hb = config.source.floor_celsius - half
    cols = [np.full(n, hs), np.full(n, hb)]
    for s in full.streams:
        cols += [s.t_in + half, s.t_out + half]
    b = np.stack(cols, axis=1)
    f = np.clip((hs - b) / (hs - hb), 0.0, 1.0)
    plants = {p.name for p in config.plants}
    c_low = np.zeros_like(b)
    c_plant: dict[str, FloatArray] = {}
    for s in full.streams:
        lo = s.t_in + half
        width = np.maximum(s.t_out - s.t_in, 0.0)
        part = s.cp[:, None] * np.clip(b - lo[:, None], 0.0, width[:, None])
        if s.name in plants:
            c_plant[s.name] = part
        else:
            c_low += part
    return CutData(f=f, c_low=c_low, c_plant=c_plant)


def use_of(f: FloatArray, c_below: FloatArray, heat: FloatArray | float) -> FloatArray:
    """Heat used per hour at the hot-side heat ``heat`` (min-cut)."""
    h = np.asarray(heat, dtype=np.float64)
    return np.asarray(np.min(f * h[..., None] + c_below, axis=-1), dtype=np.float64)


# ------------------------------------------------------------------------------ forecasts


def _transitions(on: BoolArray) -> NDArray[np.int64]:
    return np.flatnonzero(on[1:] != on[:-1]).astype(np.int64) + 1


class _Forecaster(ABC):
    """Forecast of the on/off state of every plant over a window, seen at hour ``t``."""

    @abstractmethod
    def states(self, t: int, length: int) -> dict[str, BoolArray]:
        """State of every plant in the hours ``t .. t + length - 1``."""


class TrueForecast(_Forecaster):
    """P1: the forecast is the truth."""

    def __init__(self, on: dict[str, BoolArray]) -> None:
        self.on = on

    def states(self, t: int, length: int) -> dict[str, BoolArray]:
        return {k: v[t : t + length] for k, v in self.on.items()}


class PlanForecast(_Forecaster):
    """P2: shifted plan with the naive correction of the module docstring."""

    def __init__(self, on: dict[str, BoolArray], planned: dict[str, NDArray[np.int64]]) -> None:
        self.on = on
        self.true = {k: _transitions(v) for k, v in on.items()}
        self.planned = planned
        self.init = {k: bool(v[0]) for k, v in on.items()}

    def states(self, t: int, length: int) -> dict[str, BoolArray]:
        hours = np.arange(t, t + length)
        out: dict[str, BoolArray] = {}
        for name, tau in self.true.items():
            p = self.planned[name]
            k_obs = int(np.searchsorted(tau, t, side="right"))
            later = np.arange(k_obs, p.size)
            q = np.maximum(p[k_obs:], t + 1 + (later - k_obs))
            times = np.concatenate([tau[:k_obs], q])
            count = np.searchsorted(times, hours, side="right")
            out[name] = (count % 2 == 1) ^ self.init[name]
        return out


def planned_transitions(
    config: CampaignNetworkConfig, availability: float, mode: str, draw: int
) -> dict[str, NDArray[np.int64]]:
    """Planned change times of P2: true changes shifted by U[-24, +24] h, order kept."""
    on = plant_schedules(config, availability, mode, draw)
    c = config.campaigns
    out: dict[str, NDArray[np.int64]] = {}
    cache: dict[int, NDArray[np.int64]] = {}
    for i, plant in enumerate(config.plants):
        stream = 0 if mode == "joint" else i
        if stream not in cache:
            tau = _transitions(on[plant.name])
            seq = np.random.SeedSequence([c.seed, draw, round(availability * 1000), stream, 24])
            shift = np.round(np.random.default_rng(seq).uniform(-24.0, 24.0, tau.size))
            p = (tau + shift).astype(np.int64)
            for k in range(p.size):
                low = 1 if k == 0 else p[k - 1] + 1
                p[k] = max(p[k], low)
            cache[stream] = p
        out[plant.name] = cache[stream]
    return out


# ------------------------------------------------------------------------------ rolling LP


@dataclass(frozen=True, slots=True)
class PlanResult:
    """Realised operation of a rolling planning run and its planning statistics."""

    use_kw: FloatArray
    charge_kw: FloatArray
    discharge_kw: FloatArray
    content_kwh: FloatArray
    plans: int
    lp_failures: int
    solve_seconds: float


def _solve_window(
    f: FloatArray,
    c_below: FloatArray,
    e0: float,
    power_kw: float,
    capacity: float,
    dis_max: float,
    keep: float,
    method: str,
) -> tuple[FloatArray, FloatArray, bool]:
    """Plan ``c``, ``d`` over one window (rows of ``f`` / ``c_below``)."""
    t_len, m = f.shape
    nv = 4 * t_len
    k = np.repeat(np.arange(t_len), m)
    rows = np.repeat(np.arange(t_len * m), 3)
    cols = np.stack([k, t_len + k, 2 * t_len + k], axis=1).ravel()
    ff = f.ravel()
    vals = np.stack([np.ones_like(ff), ff, -ff], axis=1).ravel()
    a_ub = coo_matrix((vals, (rows, cols)), shape=(t_len * m, nv)).tocsr()
    b_ub = (c_below + f * power_kw).ravel()
    kk = np.arange(t_len)
    er = np.concatenate([kk, kk, kk, kk[1:]])
    ec = np.concatenate([3 * t_len + kk, t_len + kk, 2 * t_len + kk, 3 * t_len + kk[:-1]])
    ev = np.concatenate(
        [np.ones(t_len), -np.ones(t_len), np.ones(t_len), -keep * np.ones(t_len - 1)]
    )
    a_eq = coo_matrix((ev, (er, ec)), shape=(t_len, nv)).tocsr()
    b_eq = np.zeros(t_len)
    b_eq[0] = keep * e0
    obj = np.concatenate([-np.ones(t_len), np.zeros(3 * t_len)])
    bounds = np.array(
        [(0.0, np.inf)] * t_len
        + [(0.0, power_kw)] * t_len
        + [(0.0, dis_max)] * t_len
        + [(0.0, capacity)] * t_len
    )
    res = linprog(obj, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method=method)
    if res.x is None:
        return np.zeros(t_len), np.zeros(t_len), False
    x = res.x
    return np.asarray(x[t_len : 2 * t_len]), np.asarray(x[2 * t_len : 3 * t_len]), True


def rolling_plan(
    config: CampaignNetworkConfig,
    cuts: CutData,
    on: dict[str, BoolArray],
    forecaster: _Forecaster,
    power_kw: float,
    storage_hours: float,
    horizon_h: int,
    replan_every_h: int,
) -> PlanResult:
    """Rolling planning of the module docstring; returns the realised operation."""
    n = cuts.f.shape[0]
    capacity = storage_hours * power_kw
    keep = 1.0 - config.storage.loss_per_h
    dis_max = config.storage.discharge_max_fraction * power_kw
    true_below = cuts.demand_below(on, slice(None))
    use = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    content = np.zeros(n + 1)
    e = 0.0
    plan_c = plan_d = np.zeros(0)
    plan_states: dict[str, BoolArray] = {}
    t0 = 0
    plans = failures = 0
    solve_s = 0.0
    for t in range(n):
        due = t == 0 or t - t0 >= replan_every_h or t - t0 >= plan_c.size
        if not due:
            for name, states in plan_states.items():
                if bool(states[t - t0]) != bool(on[name][t]):
                    due = True
                    break
        if due:
            length = min(horizon_h, n - t)
            fc = forecaster.states(t, length)
            rows = slice(t, t + length)
            below = cuts.demand_below(fc, rows)
            s0 = time.perf_counter()
            plan_c, plan_d, ok = _solve_window(
                cuts.f[rows], below, e, power_kw, capacity, dis_max, keep, config.solver.lp_method
            )
            solve_s += time.perf_counter() - s0
            plans += 1
            failures += 0 if ok else 1
            plan_states = fc
            t0 = t
        c = float(min(max(plan_c[t - t0], 0.0), power_kw, max(capacity - keep * e, 0.0)))
        d = float(min(max(plan_d[t - t0], 0.0), dis_max, keep * e + c))
        e_next = min(max(keep * e + c - d, 0.0), capacity)
        use[t] = float(use_of(cuts.f[t], true_below[t], power_kw - c + d))
        charge[t] = c
        discharge[t] = d
        content[t + 1] = e_next
        e = e_next
    return PlanResult(use, charge, discharge, content, plans, failures, solve_s)
