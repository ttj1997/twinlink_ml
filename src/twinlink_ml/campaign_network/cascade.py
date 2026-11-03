r"""Configurations without foresight: K0 (no network), K1 (cascade rule), K2 (K1 + storage rule).

Exchanger (every stage): counter-current, constant heat-capacity flows, minimum approach ``dT`` at
both ends (with constant flows the approach is linear along the exchanger, so the ends decide).
Network water enters with ``T_w`` and heat-capacity flow ``C_w``, the consumer stream goes from
``t_in`` to ``t_out`` with ``C_c``; the duty is the largest one allowed by

* demand:   ``q <= C_c (t_out - t_in)``
* hot end:  ``q <= C_c (T_w - dT - t_in)``     (consumer outlet at most ``T_w - dT``)
* cold end: ``q <= C_w (T_w - t_in - dT)``     (water outlet at least ``t_in + dT``)
* floor:    ``q <= C_w (T_w - T_floor)``        (the source water is not cooled below the floor)

and the water leaves with ``T_w - q / C_w``.

**K0** -- no network: the source water passes the preheater of plant A only (the fixed assignment
of today, Befund 100); what A does not take goes to the cooling tower, also while A stands still.

**K1** -- network with the cascade rule (rule of practice, fixed before the run): the supply water
goes first to the stage with the highest required temperature, its return cascades to the next
lower stage. Stage order: required temperature descending (process-air preheaters of A, B, C ->
process water -> hall heating with the placeholders of the YAML), ties by the gas replaced per kWh
(lower efficiency of the replaced supply first). The preheaters of A, B, C are parallel branches
of the first stage with the water flow split in proportion to their air flows (every branch then
has the same temperatures, so the stage acts as one exchanger with the summed air flow). A
consumer that stands still (or has no demand) takes nothing; the next stage moves up and gets the
water directly.

**K2** -- K1 plus a stratified hot-water tank on the loop that stores supply water at the supply
temperature (charging diverts supply water before the consumers, discharging adds it). Rule
without foresight, hour by hour:

1. discharge if more supply water raises some stage duty: the smallest discharge that reaches
   every stage duty of the largest possible discharge (content after losses, power limit);
2. otherwise charge the largest part of the source water whose removal lowers no stage duty
   (limited by the free capacity and the source power).

Storage balance ``E[t+1] = (1 - loss) E[t] + c[t] - d[t]``, empty at the start. Because the cascade
duties do not decrease with the water flow, K2 >= K1 in every stage and hour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import CampaignNetworkConfig
from .profiles import ColdStream, Consumers

__all__ = [
    "CascadeResult",
    "Stage",
    "cascade_stages",
    "run_k0",
    "run_k1",
    "run_k2",
    "stage_duty",
]

FloatArray = NDArray[np.float64]


def stage_duty(
    t_water: FloatArray | float,
    cp_water: FloatArray | float,
    t_in: FloatArray | float,
    t_out: FloatArray | float,
    cp_cold: FloatArray | float,
    approach_k: float,
    floor_celsius: float,
) -> FloatArray:
    """Duty of one counter-current exchanger (see the module docstring), vectorised."""
    q = np.minimum.reduce(
        [
            np.asarray(cp_cold * (np.asarray(t_out) - t_in), dtype=np.float64),
            np.asarray(cp_cold * (np.asarray(t_water) - approach_k - t_in), dtype=np.float64),
            np.asarray(cp_water * (np.asarray(t_water) - t_in - approach_k), dtype=np.float64),
            np.asarray(cp_water * (np.asarray(t_water) - floor_celsius), dtype=np.float64),
        ]
    )
    return np.asarray(np.maximum(q, 0.0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Stage:
    """One cascade stage: parallel branches with the same temperatures, summed flow."""

    name: str
    members: tuple[ColdStream, ...]
    t_in: FloatArray
    t_out: FloatArray
    cp: FloatArray


def cascade_stages(consumers: Consumers) -> tuple[Stage, ...]:
    """Stages of the cascade rule K1 in their order (see the module docstring)."""
    groups: dict[str, list[ColdStream]] = {}
    for s in consumers.streams:
        groups.setdefault(s.group, []).append(s)
    stages: list[tuple[tuple[float, float], Stage]] = []
    for name, members in groups.items():
        first = members[0]
        for m in members[1:]:
            if not (np.array_equal(m.t_in, first.t_in) and np.array_equal(m.t_out, first.t_out)):
                raise ValueError(f"branches of stage {name!r} need the same temperatures")
            if m.stage_celsius != first.stage_celsius:
                raise ValueError(f"branches of stage {name!r} need the same required temperature")
        cp = np.sum([m.cp for m in members], axis=0)
        key = (-first.stage_celsius, min(m.gas_efficiency for m in members))
        stages.append((key, Stage(name, tuple(members), first.t_in, first.t_out, cp)))
    stages.sort(key=lambda item: item[0])
    return tuple(s for _, s in stages)


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """Heat delivered per consumer and hour, with the storage flows of K2 (zero otherwise)."""

    duty_kw: dict[str, FloatArray]
    charge_kw: FloatArray
    discharge_kw: FloatArray
    content_kwh: FloatArray


def _split(stage: Stage, duty: FloatArray) -> dict[str, FloatArray]:
    """Duty of a stage split to its branches in proportion to their flows."""
    out: dict[str, FloatArray] = {}
    total = stage.cp
    with np.errstate(invalid="ignore", divide="ignore"):
        for m in stage.members:
            share = np.where(total > 0.0, m.cp / np.where(total > 0.0, total, 1.0), 0.0)
            out[m.name] = np.asarray(duty * share, dtype=np.float64)
    return out


def _cascade(
    stages: tuple[Stage, ...],
    cp_water: FloatArray | float,
    config: CampaignNetworkConfig,
) -> list[FloatArray]:
    t_w = np.full_like(stages[0].cp, config.source.supply_celsius)
    cpw = np.broadcast_to(np.asarray(cp_water, dtype=np.float64), t_w.shape)
    duties: list[FloatArray] = []
    for st in stages:
        q = stage_duty(
            t_w, cpw, st.t_in, st.t_out, st.cp, config.approach_k, config.source.floor_celsius
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            t_w = np.where(cpw > 0.0, t_w - q / np.where(cpw > 0.0, cpw, 1.0), t_w)
        duties.append(q)
    return duties


def _result(stages: tuple[Stage, ...], duties: list[FloatArray], n: int) -> CascadeResult:
    duty: dict[str, FloatArray] = {}
    for st, q in zip(stages, duties, strict=True):
        duty.update(_split(st, q))
    zeros = np.zeros(n, dtype=np.float64)
    return CascadeResult(duty, zeros, zeros.copy(), np.zeros(n + 1, dtype=np.float64))


def run_k0(consumers: Consumers, power_kw: float, config: CampaignNetworkConfig) -> CascadeResult:
    """K0: the source water passes the preheater of plant A only."""
    first = config.plants[0].name
    a = consumers.by_name(first)
    stage = Stage(first, (a,), a.t_in, a.t_out, a.cp)
    duties = _cascade((stage,), power_kw / config.source.span_k, config)
    res = _result((stage,), duties, a.cp.size)
    for s in consumers.streams:
        res.duty_kw.setdefault(s.name, np.zeros(a.cp.size, dtype=np.float64))
    return res


def run_k1(consumers: Consumers, power_kw: float, config: CampaignNetworkConfig) -> CascadeResult:
    """K1: cascade rule, no storage."""
    stages = cascade_stages(consumers)
    duties = _cascade(stages, power_kw / config.source.span_k, config)
    return _result(stages, duties, stages[0].cp.size)


def _cascade_scalar(
    params: list[tuple[float, float, float]],
    cp_water: float,
    t_supply: float,
    approach: float,
    floor: float,
) -> list[float]:
    """Pure-float version of :func:`_cascade` for one hour (speed of the K2 loop)."""
    t_w = t_supply
    out: list[float] = []
    for t_in, t_out, cp_c in params:
        if cp_c <= 0.0 or cp_water <= 0.0:
            out.append(0.0)
            continue
        q = min(
            cp_c * (t_out - t_in),
            cp_c * (t_w - approach - t_in),
            cp_water * (t_w - t_in - approach),
            cp_water * (t_w - floor),
        )
        q = q if q > 0.0 else 0.0
        t_w -= q / cp_water
        out.append(q)
    return out


def run_k2(
    consumers: Consumers,
    power_kw: float,
    storage_hours: float,
    config: CampaignNetworkConfig,
) -> CascadeResult:
    """K2: cascade rule plus the storage rule of the module docstring."""
    stages = cascade_stages(consumers)
    n = stages[0].cp.size
    span = config.source.span_k
    t_s = config.source.supply_celsius
    dt_min = config.approach_k
    floor = config.source.floor_celsius
    cp0 = power_kw / span
    capacity = storage_hours * power_kw
    keep = 1.0 - config.storage.loss_per_h
    dis_max = config.storage.discharge_max_fraction * power_kw
    iters = config.solver.bisection_iterations
    tol = config.solver.duty_tolerance_kw
    t_in = [st.t_in.tolist() for st in stages]
    t_out = [st.t_out.tolist() for st in stages]
    cps = [st.cp.tolist() for st in stages]
    k = len(stages)
    duties = np.zeros((k, n), dtype=np.float64)
    charge = np.zeros(n, dtype=np.float64)
    discharge = np.zeros(n, dtype=np.float64)
    content = np.zeros(n + 1, dtype=np.float64)
    e = 0.0

    def ok(d: list[float], target: list[float]) -> bool:
        return all(a >= b - tol for a, b in zip(d, target, strict=True))

    for t in range(n):
        params = [(t_in[i][t], t_out[i][t], cps[i][t]) for i in range(k)]
        d0 = _cascade_scalar(params, cp0, t_s, dt_min, floor)
        avail = keep * e
        x = 0.0
        c = 0.0
        big_x = min(avail, dis_max)
        if big_x > tol:
            d_x = _cascade_scalar(params, cp0 + big_x / span, t_s, dt_min, floor)
            if sum(d_x) > sum(d0) + tol:
                lo, hi = 0.0, big_x
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    if ok(_cascade_scalar(params, cp0 + mid / span, t_s, dt_min, floor), d_x):
                        hi = mid
                    else:
                        lo = mid
                x = hi
                d0 = _cascade_scalar(params, cp0 + x / span, t_s, dt_min, floor)
        if x == 0.0 and capacity > 0.0:
            big_c = min(power_kw, capacity - avail)
            if power_kw - sum(d0) > tol and big_c > tol:
                if ok(_cascade_scalar(params, cp0 - big_c / span, t_s, dt_min, floor), d0):
                    c = big_c
                else:
                    lo, hi = 0.0, big_c
                    for _ in range(iters):
                        mid = 0.5 * (lo + hi)
                        if ok(_cascade_scalar(params, cp0 - mid / span, t_s, dt_min, floor), d0):
                            lo = mid
                        else:
                            hi = mid
                    c = lo
                d0 = _cascade_scalar(params, cp0 - c / span, t_s, dt_min, floor)
        e = avail + c - x
        duties[:, t] = d0
        charge[t] = c
        discharge[t] = x
        content[t + 1] = e
    duty: dict[str, FloatArray] = {}
    for i, st in enumerate(stages):
        duty.update(_split(st, duties[i]))
    return CascadeResult(duty, charge, discharge, content)
