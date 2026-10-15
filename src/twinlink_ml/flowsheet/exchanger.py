"""The heat exchanger unit and the counter-current solver behind it.

``"partner"``: both fluids leave at the same temperature (equations 2.3/2.4 of the partner's
equation document): parallel flow with unlimited area, ``Q = (T_hot,in - T_cold,in) /
(1/C_hot + 1/C_cold)``. With condensation switched on the hot side follows the condensing release
curve of :func:`~twinlink_ml.flowsheet.thermo.exhaust_profile` and the outlet temperature is found
where it equals the cold outlet (bisection; the map is monotone).

``"physical"``: counter-current, unlimited area, the largest duty at which the temperature
difference stays at least ``dT_min`` **along the exchanger** (checked at the nodes of the release
curve, not only at the ends). The solver :func:`solve_stage` is the one of Befund 90, lifted
unchanged from ``scripts/report_heat_recovery_comparison.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import Assumption, Unit
from .thermo import Profile

__all__ = [
    "ExchangerConfig",
    "ExchangerInputs",
    "ExchangerResult",
    "HeatExchanger",
    "approach_diagnostics",
    "solve_stage",
]


def min_g(profile: Profile, inverse_capacity: float, q: float) -> float:
    mask = profile.q_w <= q
    g_nodes = profile.t_c[mask] + profile.q_w[mask] * inverse_capacity
    return float(min(g_nodes.min(), profile.temperature_at(q) + q * inverse_capacity))


def solve_stage(
    profile: Profile,
    gas_capacity_w_per_k: float,
    gas_in_celsius: float,
    gas_target_celsius: float,
    htm_capacity_w_per_k: float | None,
    dt_recovery_k: float,
    dt_preheater_k: float,
) -> tuple[float, str]:
    """Largest duty [W] of one stage whose approaches stay >= the given minima; and what binds.

    Counter-current, unlimited area. ``htm_capacity_w_per_k=None``: direct exchange exhaust -> gas
    with ``dt_recovery_k`` as the minimum along the exchanger (the module's stage). Otherwise the
    circuit: recovery exchanger exhaust -> medium (``dt_recovery_k``) and pre-heating exchanger
    medium -> gas (``dt_preheater_k``); the medium returns to the recovery exchanger at ``T1`` and
    leaves it at ``T2 = T1 + Q / B``. The most favourable ``T1`` is the lowest the pre-heating
    exchanger admits, ``T1 = T_h,in + dt + max(0, Q (1/C - 1/B))`` (its cold end, or its warm end
    for a medium with more capacity than the gas), so that the condition on the recovery exchanger,
    ``min over q <= Q of (T_src(q) + q/B) - Q/B - T1 >= dt_recovery``, is monotone in ``Q``.
    """
    cg = gas_capacity_w_per_k
    demand = max(cg * (gas_target_celsius - gas_in_celsius), 0.0)
    cap_q = min(profile.total_w, demand)

    def margin(q: float) -> float:
        if htm_capacity_w_per_k is None:
            return min_g(profile, 1.0 / cg, q) - q / cg - gas_in_celsius - dt_recovery_k
        b = htm_capacity_w_per_k
        t1 = gas_in_celsius + dt_preheater_k + max(0.0, q * (1.0 / cg - 1.0 / b))
        return min_g(profile, 1.0 / b, q) - q / b - t1 - dt_recovery_k

    if cap_q <= 0.0 or margin(0.0) < 0.0:
        return 0.0, "no_driving_force"
    if margin(cap_q) >= 0.0:
        return cap_q, "sink_target" if demand <= profile.total_w else "exhaust_floor"
    lo, hi = 0.0, cap_q
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if margin(mid) >= 0.0 else (lo, mid)
        if hi - lo <= 1e-13 * cap_q:
            break
    return lo, "approach"


def approach_diagnostics(
    profile: Profile,
    q: float,
    gas_in: float,
    cg: float,
    b: float | None,
    t1: float | None,
) -> tuple[float, float | None]:
    """Smallest approach in the recovery exchanger and (circuit) in the pre-heating exchanger."""
    if q <= 0.0:
        return math.nan, None
    mask = profile.q_w <= q
    qs = np.concatenate([profile.q_w[mask], [q]])
    ts = np.concatenate([profile.t_c[mask], [profile.temperature_at(q)]])
    if b is None or t1 is None:
        return float(np.min(ts - (gas_in + (q - qs) / cg))), None
    recovery = float(np.min(ts - (t1 + (q - qs) / b)))
    t2 = t1 + q / b
    preheater = min(t1 - gas_in, t2 - (gas_in + q / cg))
    return recovery, float(preheater)


@dataclass(frozen=True)
class ExchangerConfig:
    """One exchanger: the flow arrangement and, for the counter-current one, the approach."""

    kind: Literal["equal_outlet", "counter_current"]
    dt_min_k: float = 0.0
    name: str = "exchanger"


@dataclass(frozen=True)
class ExchangerInputs:
    """Hot side as a release curve plus its capacity rate, cold side as capacity rate and inlet."""

    hot_profile: Profile
    hot_capacity_w_per_k: float
    cold_capacity_w_per_k: float
    cold_in_celsius: float
    cold_target_celsius: float = math.inf
    """Upper bound of the cold outlet (the sink's set point); only the counter-current kind
    uses it (the partner's equal-outlet closed form has no bound)."""


@dataclass(frozen=True)
class ExchangerResult:
    """Duty and outlet temperatures of one exchanger (no circuit)."""

    q_w: float
    hot_out_celsius: float
    cold_out_celsius: float
    binding: str
    min_approach_k: float


class HeatExchanger(Unit[ExchangerInputs, ExchangerResult]):
    """A single exchanger between a hot and a cold stream."""

    def __init__(self, config: ExchangerConfig, *, assumption: Assumption = "partner") -> None:
        self.config = config
        self.name = config.name
        self._assumption = assumption

    def assumptions(self) -> dict[str, Assumption]:
        return {"exchanger": self._assumption}

    def solve(self, inputs: ExchangerInputs) -> ExchangerResult:
        profile = inputs.hot_profile
        t_hot = float(profile.t_c[0])
        cc = inputs.cold_capacity_w_per_k
        t_cold = inputs.cold_in_celsius
        if self.config.kind == "equal_outlet":
            q = (t_hot - t_cold) / (1.0 / inputs.hot_capacity_w_per_k + 1.0 / cc)
            return ExchangerResult(
                q, t_hot - q / inputs.hot_capacity_w_per_k, t_cold + q / cc, "equal_outlet", 0.0
            )
        q, binding = solve_stage(
            profile,
            cc,
            t_cold,
            inputs.cold_target_celsius,
            None,
            self.config.dt_min_k,
            0.0,
        )
        hot_out = profile.temperature_at(q) if q > 0.0 else t_hot
        approach, _ = approach_diagnostics(profile, q, t_cold, cc, None, None)
        return ExchangerResult(q, hot_out, t_cold + q / cc, binding, approach)
