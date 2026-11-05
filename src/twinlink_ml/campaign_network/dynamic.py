r"""Follow-up study (``docs/NN_campaign_network_2.md``): fair baseline K0b and dynamic cascade K1d.

**K0b** -- fair baseline without a network: the source water is split in FIXED shares to the
process-air preheaters of plants A, B, C, each share passing its own preheater only; what a
preheater does not take (also while its plant stands still) goes to the cooling tower; hall heating
and process water get nothing (no network). Shares in proportion to the expected mean demand of the
plants; every plant has the same availability and the same temperatures, so the shares are the
air-flow shares (A : B : C = 20 000 : 15 000 : 10 000 with the placeholders). While all plants run,
K0b equals the process-air stage of K1.

**K1d** -- dynamic cascade with the storage logic of K2b: the stages of K1 (process-air
preheaters of A, B, C as parallel branches with the flow split by air flow, process water, hall
heating) are ordered in every hour by the gas each stage could replace right now, i.e. its duty if
it got the full source water at the supply temperature (``stage_duty`` alone at the top), divided
by the efficiency of the replaced supply; descending, ties in the order of K1. The return of a
stage passes the next one as in K1. Storage (same tank as K2/K2b/K3): discharge exactly
``min(total deficit of the active stages, content after losses, power limit)`` while an active
stage has a deficit, the discharged supply water entering the cascade at the top; otherwise charge
the largest part of the source water whose removal lowers no stage duty (K2b). With storage 0, K1d
is the dynamic cascade alone. K1d >= nothing in particular by structure, but K3 >= K1d (every
series cascade with approach >= dT is feasible in the heat cascade of K3, with the same tank).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .cascade import CascadeResult, Stage, _cascade_scalar, _split, cascade_stages, stage_duty
from .config import CampaignNetworkConfig
from .profiles import Consumers

__all__ = ["fixed_shares", "run_k0b", "run_k1d"]

FloatArray = NDArray[np.float64]


def fixed_shares(config: CampaignNetworkConfig) -> dict[str, float]:
    """Shares of the source water per process plant (expected mean demand = air-flow shares)."""
    total = sum(p.flow_nm3_per_h for p in config.plants)
    return {p.name: p.flow_nm3_per_h / total for p in config.plants}


def run_k0b(consumers: Consumers, power_kw: float, config: CampaignNetworkConfig) -> CascadeResult:
    """K0b: fixed shares of the source water, one preheater each, no network."""
    n = consumers.ambient.size
    cp_total = power_kw / config.source.span_k
    duty: dict[str, FloatArray] = {s.name: np.zeros(n, dtype=np.float64) for s in consumers.streams}
    for name, share in fixed_shares(config).items():
        s = consumers.by_name(name)
        duty[name] = stage_duty(
            np.full(n, config.source.supply_celsius),
            np.full(n, share * cp_total),
            s.t_in,
            s.t_out,
            s.cp,
            config.approach_k,
            config.source.floor_celsius,
        )
    zeros = np.zeros(n, dtype=np.float64)
    return CascadeResult(duty, zeros, zeros.copy(), np.zeros(n + 1, dtype=np.float64))


def _hourly_order(stages: tuple[Stage, ...], config: CampaignNetworkConfig, cp0: float) -> NDArray:
    """Stage order per hour (rows: hours), by replaceable gas descending, ties as in K1."""
    t_s = config.source.supply_celsius
    gas = []
    for st in stages:
        alone = stage_duty(
            np.full_like(st.cp, t_s),
            np.full_like(st.cp, cp0),
            st.t_in,
            st.t_out,
            st.cp,
            config.approach_k,
            config.source.floor_celsius,
        )
        eta = min(m.gas_efficiency for m in st.members)
        gas.append(alone / eta)
    g = np.stack(gas, axis=1)  # hours x stages
    # stable sort on -g keeps the K1 order for ties
    return np.argsort(-g, axis=1, kind="stable")


def run_k1d(
    consumers: Consumers,
    power_kw: float,
    storage_hours: float,
    config: CampaignNetworkConfig,
) -> CascadeResult:
    """K1d: dynamic cascade order per hour plus the storage logic of K2b (module docstring)."""
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
    order = _hourly_order(stages, config, cp0)
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
        idx = order[t].tolist()
        params = [(t_in[i][t], t_out[i][t], cps[i][t]) for i in idx]
        d0 = _cascade_scalar(params, cp0, t_s, dt_min, floor)
        avail = keep * e
        x = 0.0
        c = 0.0
        if capacity > 0.0:
            deficit = 0.0
            for (ti, to, cp_c), q in zip(params, d0, strict=True):
                if cp_c > 0.0 and to > ti:
                    deficit += max(cp_c * (to - ti) - q, 0.0)
            x = min(avail, dis_max, deficit)
            if x > tol:
                d0 = _cascade_scalar(params, cp0 + x / span, t_s, dt_min, floor)
            else:
                x = 0.0
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
        for pos, i in enumerate(idx):
            duties[i, t] = d0[pos]
        charge[t] = c
        discharge[t] = x
        content[t + 1] = e
    duty: dict[str, FloatArray] = {}
    for i, st in enumerate(stages):
        duty.update(_split(st, duties[i]))
    return CascadeResult(duty, charge, discharge, content)
