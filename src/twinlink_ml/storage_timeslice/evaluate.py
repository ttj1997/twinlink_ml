r"""Evaluation of one scenario: direct hot utility, ideal storage, rule, checks.

All energies are PER DAY (kWh/d of useful heat of the heater; fuel = this / efficiency) -- a week
horizon is divided by its 7 days -- so that day and week scenarios are comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import CAPACITIES_KWH, LEVELS_CELSIUS, Scenario
from .storage import simulate_rule, solve_ideal_storage
from .streams import StreamBank, site_streams
from .table import (
    cascade_profile,
    direct_hot_utility_kw,
    level_groups,
    storage_stream_temperatures,
)

__all__ = ["CapacityRow", "ScenarioResult", "evaluate_scenario"]

FloatArray = NDArray[np.float64]
_BALANCE_TOLERANCE_KWH = 1e-3
_EXACT_RELATIVE = 1e-12  # float summation order only
_MONOTONE_TOLERANCE_KWH_PER_DAY = 1e-6


@dataclass(frozen=True, slots=True)
class CapacityRow:
    """Result for one capacity (energies in kWh/d)."""

    capacity_kwh: float
    ideal_kwh_per_day: float
    ideal_level_celsius: float
    rule_kwh_per_day: float
    rule_level_celsius: float
    rule_converged: bool
    multi_level_kwh_per_day: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Result of one scenario (or of a stream subset of it)."""

    label: str
    fingerprint: str
    hot_streams: tuple[str, ...]
    cold_streams: tuple[str, ...]
    direct_kwh_per_day: float
    rows: tuple[CapacityRow, ...]
    checks: dict[str, Any] = field(default_factory=dict)

    def saving_fraction(self, row: CapacityRow) -> float:
        """``(E_direct - E_ideal) / E_direct`` (nan when there is no direct demand)."""
        if self.direct_kwh_per_day <= 0.0:
            return float("nan")
        return (self.direct_kwh_per_day - row.ideal_kwh_per_day) / self.direct_kwh_per_day

    def gap_fraction(self, row: CapacityRow) -> float:
        """``(E_rule - E_ideal) / E_direct``."""
        if self.direct_kwh_per_day <= 0.0:
            return float("nan")
        return (row.rule_kwh_per_day - row.ideal_kwh_per_day) / self.direct_kwh_per_day

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for row in self.rows:
            d = row.to_dict()
            d["saving_kwh_per_day"] = self.direct_kwh_per_day - row.ideal_kwh_per_day
            d["saving_fraction"] = self.saving_fraction(row)
            d["gap_fraction"] = self.gap_fraction(row)
            rows.append(d)
        return {
            "label": self.label,
            "fingerprint": self.fingerprint,
            "hot_streams": list(self.hot_streams),
            "cold_streams": list(self.cold_streams),
            "direct_kwh_per_day": self.direct_kwh_per_day,
            "rows": rows,
            "checks": self.checks,
        }


def evaluate_scenario(
    bank: StreamBank,
    scenario: Scenario,
    *,
    hot_names: Sequence[str] | None = None,
    cold_names: Sequence[str] | None = None,
    capacities_kwh: Sequence[float] = CAPACITIES_KWH,
    levels_celsius: Sequence[float] = LEVELS_CELSIUS,
    multi_level: bool = False,
) -> ScenarioResult:
    """Direct utility, ideal storage (best level per capacity), rule (best level per capacity).

    :param hot_names, cold_names: stream subset; default = the streams of the scenario's site.
    :param multi_level: also solve the ideal storage with ALL levels at once sharing the capacity
        (upper-bound check; slow).
    """
    site_hot, site_cold = site_streams(scenario.site)
    hot_names = tuple(site_hot if hot_names is None else hot_names)
    cold_names = tuple(site_cold if cold_names is None else cold_names)
    ph = scenario.placeholders
    hot, cold = bank.select(hot_names, cold_names)
    levels = np.asarray(levels_celsius, dtype=np.float64)
    tc, td = storage_stream_temperatures(
        levels, minimum_pinch_k=ph.minimum_pinch_k, approach_k=ph.approach_k
    )
    cascade = cascade_profile(
        hot, cold, minimum_pinch_k=ph.minimum_pinch_k, extra_points=np.concatenate([tc, td])
    )
    dt_h = bank.dt_h
    per_day = 1.0 / scenario.n_days
    direct = direct_hot_utility_kw(cascade)
    e_direct = float(np.sum(direct)) * dt_h * per_day
    caps = np.asarray(capacities_kwh, dtype=np.float64)

    ideal = np.full((len(levels), len(caps)), np.nan)
    rule = np.full((len(levels), len(caps)), np.nan)
    rule_ok = np.zeros((len(levels), len(caps)), dtype=bool)
    worst_balance = 0.0
    if e_direct > 0.0:
        for j in range(len(levels)):
            groups = level_groups(cascade, tc[j : j + 1], td[j : j + 1])
            rr = simulate_rule(
                groups, dt_h=dt_h, capacities_kwh=caps, loss_per_h=scenario.loss_per_h
            )
            rule[j] = rr.energy_kwh * per_day
            rule_ok[j] = rr.converged
            for i, cap in enumerate(caps):
                res = solve_ideal_storage(
                    groups, dt_h=dt_h, capacity_kwh=float(cap), loss_per_h=scenario.loss_per_h
                )
                ideal[j, i] = res.energy_kwh * per_day
                worst_balance = max(worst_balance, abs(res.balance_residual_kwh(dt_h)))
    else:
        ideal[:] = 0.0
        rule[:] = 0.0
        rule_ok[:] = True

    multi: FloatArray | None = None
    if multi_level and e_direct > 0.0:
        groups_all = level_groups(cascade, tc, td)
        multi = np.array(
            [
                solve_ideal_storage(
                    groups_all, dt_h=dt_h, capacity_kwh=float(c), loss_per_h=scenario.loss_per_h
                ).energy_kwh
                * per_day
                for c in caps
            ]
        )

    rows = []
    for i, cap in enumerate(caps):
        ji = int(np.argmin(ideal[:, i]))
        jr = int(np.argmin(rule[:, i]))
        rows.append(
            CapacityRow(
                capacity_kwh=float(cap),
                ideal_kwh_per_day=float(ideal[ji, i]),
                ideal_level_celsius=float(levels[ji]),
                rule_kwh_per_day=float(rule[jr, i]),
                rule_level_celsius=float(levels[jr]),
                rule_converged=bool(rule_ok[jr, i]),
                multi_level_kwh_per_day=None if multi is None else float(multi[i]),
            )
        )
    checks = _checks(rows, e_direct, caps, worst_balance, scenario.loss_per_h)
    return ScenarioResult(
        label=scenario.label,
        fingerprint=scenario.fingerprint,
        hot_streams=hot_names,
        cold_streams=cold_names,
        direct_kwh_per_day=e_direct,
        rows=tuple(rows),
        checks=checks,
    )


def _checks(
    rows: list[CapacityRow],
    e_direct: float,
    caps: FloatArray,
    worst_balance_kwh: float,
    loss_per_h: float,
) -> dict[str, Any]:
    ideal = np.array([r.ideal_kwh_per_day for r in rows])
    rule = np.array([r.rule_kwh_per_day for r in rows])
    zero = np.nonzero(caps == 0.0)[0]
    tol = _MONOTONE_TOLERANCE_KWH_PER_DAY * max(1.0, e_direct)
    return {
        "capacity_zero_equals_direct": bool(
            all(
                abs(ideal[i] - e_direct) <= _EXACT_RELATIVE * max(1.0, e_direct)
                and abs(rule[i] - e_direct) <= _EXACT_RELATIVE * max(1.0, e_direct)
                for i in zero
            )
        ),
        "ideal_non_increasing_in_capacity": bool(np.all(np.diff(ideal) <= tol)),
        "rule_not_below_ideal": bool(np.all(rule >= ideal - tol)),
        "rule_converged_all": bool(all(r.rule_converged for r in rows)),
        "ideal_not_above_direct": bool(np.all(ideal <= e_direct + tol)),
        "max_storage_balance_residual_kwh": worst_balance_kwh,
        "storage_balance_closed": bool(worst_balance_kwh <= _BALANCE_TOLERANCE_KWH),
        "loss_per_h": loss_per_h,
    }
