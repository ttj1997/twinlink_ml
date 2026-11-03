r"""K3: linear programme with full foresight over the year (upper bound of K1 and K2).

Heat-cascade (transshipment) model of the problem table, as in the pinch studies (Befund 99/100):
shifted temperatures, hot side ``T - dT/2``, cold side ``T + dT/2``. In hour ``t`` the network has
one hot stream, the source water from the supply temperature down to the floor, with the heat
``H[t] = P + d[t] - c[t]`` (source power ``P``, storage discharge ``d`` and charge ``c``; the tank
stores supply water, so the stored heat has the temperature profile of the source). Every consumer
``j`` is a cold stream cut into the intervals between the hour's boundaries; ``q[j, k]`` is the
heat it takes in interval ``k`` (at most its demand there). Heat flows only downwards:

    sum_{k above b} sum_j q[j, k] <= f(b) H[t]     for every boundary b,

``f(b)`` = share of the source heat above ``b``. Any network of counter-current exchangers with
approach >= ``dT`` (K1, K2) satisfies these constraints, so K3 >= K2 >= K1 when the storage is the
same. The model allows ideal matches (any split, partial service of any temperature range):
an upper bound, not a design.

Storage as in K2: ``E[t+1] = (1 - loss) E[t] + c[t] - d[t]``, ``0 <= E <= S``, ``E[0] = 0``, end
free, ``c <= P``, ``d <= discharge_max_fraction P``. Objective: net CO2 saving
``sum_j q_j (gas_co2 / eta_j - pump_fraction x electricity_co2)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .config import CampaignNetworkConfig
from .profiles import Consumers

__all__ = ["TargetResult", "run_k3"]

FloatArray = NDArray[np.float64]
_ROUND = 9


@dataclass(frozen=True, slots=True)
class TargetResult:
    """Heat per consumer and hour, storage flows and LP status of one K3 run."""

    duty_kw: dict[str, FloatArray]
    charge_kw: FloatArray
    discharge_kw: FloatArray
    content_kwh: FloatArray
    status: int
    message: str
    objective_kg: float
    seconds: float


def net_weight(config: CampaignNetworkConfig, gas_efficiency: float) -> float:
    """Net CO2 saving per kWh of delivered heat (kg/kWh)."""
    return (
        config.gas_co2_kg_per_kwh / gas_efficiency
        - config.pump_fraction * config.electricity_co2_kg_per_kwh
    )


def run_k3(
    consumers: Consumers,
    power_kw: float,
    storage_hours: float,
    config: CampaignNetworkConfig,
) -> TargetResult:
    """Solve the LP of the module docstring for one source power and storage size."""
    t0 = time.perf_counter()
    half = 0.5 * config.approach_k
    hs = config.source.supply_celsius - half
    hb = config.source.floor_celsius - half
    streams = consumers.streams
    n = streams[0].cp.size
    weights = [net_weight(config, s.gas_efficiency) for s in streams]
    storage = storage_hours > 0.0
    capacity = storage_hours * power_kw

    obj: list[float] = []
    upper: list[float] = []
    owner: list[tuple[int, int]] = []  # (stream, hour) of every q variable
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    # storage variables come after the q variables; their columns are fixed at the end
    store_terms: list[tuple[int, int, float]] = []  # (row, hour, f(b))
    row = 0
    for t in range(n):
        active = []
        for j, s in enumerate(streams):
            cp = float(s.cp[t])
            lo = float(s.t_in[t]) + half
            hi = float(s.t_out[t]) + half
            if cp > 0.0 and hi > lo:
                active.append((j, cp, lo, hi))
        bounds = {round(hs, _ROUND), round(hb, _ROUND)}
        for _, _, lo, hi in active:
            bounds.add(round(lo, _ROUND))
            bounds.add(round(hi, _ROUND))
        b = sorted(bounds, reverse=True)
        # q variables per interval (k between b[k] and b[k+1])
        interval_vars: list[list[int]] = []
        for k in range(len(b) - 1):
            top, bot = b[k], b[k + 1]
            ids: list[int] = []
            for j, cp, lo, hi in active:
                overlap = min(top, hi) - max(bot, lo)
                if overlap > 0.0:
                    ids.append(len(obj))
                    obj.append(-weights[j])
                    upper.append(cp * overlap)
                    owner.append((j, t))
            interval_vars.append(ids)
        # heat-flow constraints at every boundary below the top
        above: list[int] = []
        for m in range(1, len(b)):
            above = above + interval_vars[m - 1]
            f = min(max((hs - b[m]) / (hs - hb), 0.0), 1.0)
            if not above:
                continue
            for v in above:
                rows.append(row)
                cols.append(v)
                vals.append(1.0)
            rhs.append(f * power_kw)
            if storage and f > 0.0:
                store_terms.append((row, t, f))
            row += 1

    n_q = len(obj)
    n_var = n_q
    bounds_list: list[tuple[float, float]] = [(0.0, u) for u in upper]
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    if storage:
        c0, d0, e0 = n_q, n_q + n, n_q + 2 * n  # c[t], d[t], E[t+1]
        n_var = n_q + 3 * n
        obj.extend([0.0] * (3 * n))
        dis_max = config.storage.discharge_max_fraction * power_kw
        bounds_list += [(0.0, power_kw)] * n + [(0.0, dis_max)] * n + [(0.0, capacity)] * n
        for r, t, f in store_terms:
            rows += [r, r]
            cols += [c0 + t, d0 + t]
            vals += [f, -f]
        keep = 1.0 - config.storage.loss_per_h
        for t in range(n):
            # E[t+1] - keep E[t] - c[t] + d[t] = 0 (E[0] = 0)
            eq_rows += [t, t, t]
            eq_cols += [e0 + t, c0 + t, d0 + t]
            eq_vals += [1.0, -1.0, 1.0]
            if t > 0:
                eq_rows.append(t)
                eq_cols.append(e0 + t - 1)
                eq_vals.append(-keep)
    a_ub = coo_matrix((vals, (rows, cols)), shape=(row, n_var)).tocsr()
    kwargs: dict[str, object] = {}
    if storage:
        kwargs["A_eq"] = coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(n, n_var)).tocsr()
        kwargs["b_eq"] = np.zeros(n)
    res = linprog(
        np.asarray(obj),
        A_ub=a_ub,
        b_ub=np.asarray(rhs),
        bounds=bounds_list,
        method=config.solver.lp_method,
        **kwargs,
    )
    if res.x is None:
        raise RuntimeError(f"K3 LP failed: status {res.status}, {res.message}")
    x = np.asarray(res.x, dtype=np.float64)
    duty = {s.name: np.zeros(n, dtype=np.float64) for s in streams}
    q = x[:n_q]
    js = np.fromiter((o[0] for o in owner), dtype=np.int64, count=n_q)
    ts = np.fromiter((o[1] for o in owner), dtype=np.int64, count=n_q)
    for j, s in enumerate(streams):
        sel = js == j
        np.add.at(duty[s.name], ts[sel], q[sel])
    if storage:
        charge = x[n_q : n_q + n]
        discharge = x[n_q + n : n_q + 2 * n]
        content = np.concatenate([[0.0], x[n_q + 2 * n :]])
    else:
        charge = np.zeros(n)
        discharge = np.zeros(n)
        content = np.zeros(n + 1)
    return TargetResult(
        duty_kw=duty,
        charge_kw=np.asarray(charge, dtype=np.float64),
        discharge_kw=np.asarray(discharge, dtype=np.float64),
        content_kwh=np.asarray(content, dtype=np.float64),
        status=int(res.status),
        message=str(res.message),
        objective_kg=float(-res.fun),
        seconds=time.perf_counter() - t0,
    )
