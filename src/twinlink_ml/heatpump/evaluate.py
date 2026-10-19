r"""Evaluation of one scenario: Stage A (heat pump, step by step) and Stage B (plus storage).

The streams come from storage test 1 (:class:`twinlink_ml.storage_timeslice.StreamFactory`,
sites A, C, A+C, Befund 99). All energies are PER DAY (kWh/d; a week horizon is divided by 7);
objectives in kg CO2 or EUR per day. Annual figures are day values x 365 -- a scaling of the
measured day, not an annual simulation.

Baseline: the pinch target (direct hot utility) of the site with maximum DIRECT heat recovery.
The heat pump is credited only with what it saves ON TOP of direct recovery; that is the
pinch-consistent comparison (a heat pump that only replaces direct recovery saves nothing).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..storage_timeslice.config import Scenario
from ..storage_timeslice.profiles import Clock
from ..storage_timeslice.streams import StreamBank, site_streams
from ..storage_timeslice.table import (
    Cascade,
    cascade_profile,
    direct_hot_utility_kw,
    storage_stream_temperatures,
)
from .config import Criterion, HeatPumpSpec, Signal
from .cop import DesignPair, design_pairs, pseudo_stream_points
from .dispatch import dispatch_steps, heat_pump_lines
from .pinch import PinchSummary, classify_placement, pinch_summary
from .signal import signal_series
from .storage import RULES, RuleName, coupled_pieces, simulate_rule, solve_foresight

__all__ = [
    "StageAResult",
    "StageBResult",
    "evaluate_stage_a",
    "evaluate_stage_b",
    "site_cascade",
]

FloatArray = NDArray[np.float64]


def site_cascade(
    bank: StreamBank,
    scenario: Scenario,
    extra_points: FloatArray,
    hot_names: Sequence[str] | None = None,
    cold_names: Sequence[str] | None = None,
) -> Cascade:
    """Cascade of the scenario's site (or a stream subset) with the given extra boundaries."""
    site_hot, site_cold = site_streams(scenario.site)
    hot, cold = bank.select(
        tuple(site_hot if hot_names is None else hot_names),
        tuple(site_cold if cold_names is None else cold_names),
    )
    return cascade_profile(
        hot,
        cold,
        minimum_pinch_k=scenario.placeholders.minimum_pinch_k,
        extra_points=extra_points,
    )


@dataclass(frozen=True, slots=True)
class StageAResult:
    """Best design pair per (size, criterion) and what it does (all per day).

    Arrays have shape ``(sizes, criteria)``.
    """

    label: str
    sizes_kw: tuple[float, ...]
    criteria: tuple[Criterion, ...]
    direct_kwh_per_day: float
    pinch: dict[str, float]
    best_pair: tuple[tuple[DesignPair | None, ...], ...]
    heat_kwh_per_day: FloatArray
    utility_saving_kwh_per_day: FloatArray
    work_kwh_per_day: FloatArray
    objective_direct_per_day: FloatArray
    objective_saving_per_day: FloatArray
    across_share: FloatArray
    pair_table: list[dict[str, Any]] = field(default_factory=list)

    @property
    def heat_ratio(self) -> FloatArray:
        """``delta U / W``: kWh of gas heat replaced per kWh of electricity (nan where W = 0)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self.work_kwh_per_day > 1e-9,
                self.utility_saving_kwh_per_day / self.work_kwh_per_day,
                np.nan,
            )

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for i, size in enumerate(self.sizes_kw):
            for j, crit in enumerate(self.criteria):
                pair = self.best_pair[i][j]
                rows.append(
                    {
                        "size_kw": size,
                        "criterion": crit.name,
                        "pair": None if pair is None else pair.label,
                        "cop": None if pair is None else pair.cop,
                        "heat_kwh_per_day": float(self.heat_kwh_per_day[i, j]),
                        "utility_saving_kwh_per_day": float(self.utility_saving_kwh_per_day[i, j]),
                        "work_kwh_per_day": float(self.work_kwh_per_day[i, j]),
                        "objective_direct_per_day": float(self.objective_direct_per_day[i, j]),
                        "objective_saving_per_day": float(self.objective_saving_per_day[i, j]),
                        "across_share": float(self.across_share[i, j]),
                    }
                )
        return {
            "label": self.label,
            "direct_kwh_per_day": self.direct_kwh_per_day,
            "pinch": self.pinch,
            "rows": rows,
            "pair_table": self.pair_table,
        }


def _pinch_stats(summary: PinchSummary, dt_h: float, per_day: float) -> dict[str, float]:
    return {
        "hot_utility_kwh_per_day": float(np.sum(summary.hot_utility_kw) * dt_h * per_day),
        "cold_utility_kwh_per_day": float(np.sum(summary.cold_utility_kw) * dt_h * per_day),
        "pinch_high_mean_shifted": float(np.nanmean(summary.pinch_high)),
        "pinch_high_min_shifted": float(np.nanmin(summary.pinch_high)),
        "pinch_high_max_shifted": float(np.nanmax(summary.pinch_high)),
        "pinch_low_mean_shifted": float(np.nanmean(summary.pinch_low)),
        "share_steps_without_hot_utility": float(np.mean(summary.hot_utility_kw <= 1e-6)),
        "share_steps_without_cold_utility": float(np.mean(summary.cold_utility_kw <= 1e-6)),
    }


def evaluate_stage_a(
    bank: StreamBank,
    scenario: Scenario,
    *,
    spec: HeatPumpSpec,
    criteria: Sequence[Criterion],
    sizes_kw: Sequence[float],
    pairs: Sequence[DesignPair] | None = None,
    keep_pair_table: bool = False,
    hot_names: Sequence[str] | None = None,
    cold_names: Sequence[str] | None = None,
) -> StageAResult:
    """Best admissible pair per size and criterion with the optimal output per step.

    :param keep_pair_table: also return the result of EVERY pair at every size for the first
        criterion (pinch-rule check; large).
    """
    ph = scenario.placeholders
    if pairs is None:
        pairs = design_pairs(spec, minimum_pinch_k=ph.minimum_pinch_k)
    pairs = list(pairs)
    cascade = site_cascade(bank, scenario, pseudo_stream_points(pairs), hot_names, cold_names)
    dt_h = bank.dt_h
    per_day = 1.0 / scenario.n_days
    direct = direct_hot_utility_kw(cascade)
    e_direct = float(np.sum(direct)) * dt_h * per_day
    summary = pinch_summary(cascade)
    sizes = np.asarray(sizes_kw, dtype=np.float64)
    c_u = np.array([c.heat_value_per_kwh for c in criteria])
    c_el = np.array([c.electricity_value_per_kwh for c in criteria])
    n_s, n_c = len(sizes), len(criteria)
    j_direct = np.broadcast_to(c_u * e_direct, (n_s, n_c)).copy()

    best_saving = np.zeros((n_s, n_c))
    best_idx = np.full((n_s, n_c), -1)
    heat = np.zeros((n_s, n_c))
    usave = np.zeros((n_s, n_c))
    work = np.zeros((n_s, n_c))
    across = np.zeros((n_s, n_c))
    table: list[dict[str, Any]] = []
    for k, pair in enumerate(pairs):
        lines = heat_pump_lines(cascade, pair.evaporator_shifted, pair.condenser_shifted, pair.cop)
        d = dispatch_steps(lines, sizes[:, None, None], c_u[None, :, None], c_el[None, :, None])
        q_day = np.sum(d.heat_kw, axis=-1) * dt_h * per_day
        u_day = np.sum(d.utility_kw, axis=-1) * dt_h * per_day
        w_day = np.sum(d.work_kw, axis=-1) * dt_h * per_day
        saving = c_u * (e_direct - u_day) - c_el * w_day
        place = classify_placement(pair.evaporator_shifted, pair.condenser_shifted, summary)
        is_across = place == "across"
        q_across = np.sum(d.heat_kw * is_across, axis=-1) * dt_h * per_day
        better = saving > best_saving + 1e-9 * np.maximum(1.0, np.abs(best_saving))
        best_saving = np.where(better, saving, best_saving)
        best_idx = np.where(better, k, best_idx)
        heat = np.where(better, q_day, heat)
        usave = np.where(better, e_direct - u_day, usave)
        work = np.where(better, w_day, work)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(q_day > 1e-9, q_across / q_day, 0.0)
        across = np.where(better, share, across)
        if keep_pair_table:
            counts = {str(p): float(np.mean(place == p)) for p in np.unique(place)}
            for i, size in enumerate(sizes):
                if size <= 0.0:
                    continue
                # forced full output in every step (placement check, criterion-free)
                u_full = lines.utility_kw(np.full(cascade.n_steps, float(size)))
                table.append(
                    {
                        "pair": pair.label,
                        "cop": pair.cop,
                        "size_kw": float(size),
                        "placement_step_shares": counts,
                        "forced_full_output_utility_saving_kwh_per_day": float(
                            np.sum(direct - u_full) * dt_h * per_day
                        ),
                        "forced_full_output_heat_kwh_per_day": float(
                            size * cascade.n_steps * dt_h * per_day
                        ),
                        "optimal_heat_kwh_per_day": float(q_day[i, 0]),
                        "optimal_utility_saving_kwh_per_day": float(e_direct - u_day[i, 0]),
                        "optimal_objective_saving_per_day": float(saving[i, 0]),
                    }
                )
    best_pair = tuple(
        tuple(pairs[int(best_idx[i, j])] if best_idx[i, j] >= 0 else None for j in range(n_c))
        for i in range(n_s)
    )
    return StageAResult(
        label=scenario.label,
        sizes_kw=tuple(float(s) for s in sizes),
        criteria=tuple(criteria),
        direct_kwh_per_day=e_direct,
        pinch=_pinch_stats(summary, dt_h, per_day),
        best_pair=best_pair,
        heat_kwh_per_day=heat,
        utility_saving_kwh_per_day=usave,
        work_kwh_per_day=work,
        objective_direct_per_day=j_direct,
        objective_saving_per_day=best_saving,
        across_share=across,
        pair_table=table,
    )


@dataclass(frozen=True, slots=True)
class StageBResult:
    """Heat pump plus storage for one (scenario, criterion, signal, size, pair); per day.

    ``objective_*`` are ``(capacities,)`` arrays of the objective per day at the best storage
    level of each method; ``levels_*`` the chosen levels.
    """

    label: str
    criterion: str
    amplitude: float
    size_kw: float
    pair: str
    cop: float
    capacities_kwh: tuple[float, ...]
    direct_kwh_per_day: float
    objective_none_per_day: float
    objective_pump_only_per_day: float
    objective_foresight_per_day: FloatArray
    level_foresight: FloatArray
    objective_daily_neutral_per_day: FloatArray | None
    objective_rules_per_day: dict[str, FloatArray]
    level_rules: dict[str, FloatArray]
    rule_converged: dict[str, NDArray[np.bool_]]
    checks: dict[str, Any]

    @property
    def objective_rule_best_per_day(self) -> FloatArray:
        return np.min(np.stack(list(self.objective_rules_per_day.values())), axis=0)

    def gap_fraction(self) -> FloatArray:
        """``(J_rule - J_foresight) / J_none`` per capacity (best rule)."""
        return (self.objective_rule_best_per_day - self.objective_foresight_per_day) / (
            self.objective_none_per_day
        )

    def storage_gain_fraction(self) -> FloatArray:
        """``(J_pump_only - J_foresight) / J_none`` per capacity."""
        return (self.objective_pump_only_per_day - self.objective_foresight_per_day) / (
            self.objective_none_per_day
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "criterion": self.criterion,
            "amplitude": self.amplitude,
            "size_kw": self.size_kw,
            "pair": self.pair,
            "cop": self.cop,
            "capacities_kwh": list(self.capacities_kwh),
            "direct_kwh_per_day": self.direct_kwh_per_day,
            "objective_none_per_day": self.objective_none_per_day,
            "objective_pump_only_per_day": self.objective_pump_only_per_day,
            "objective_foresight_per_day": self.objective_foresight_per_day.tolist(),
            "level_foresight": self.level_foresight.tolist(),
            "objective_daily_neutral_per_day": None
            if self.objective_daily_neutral_per_day is None
            else self.objective_daily_neutral_per_day.tolist(),
            "objective_rules_per_day": {
                k: v.tolist() for k, v in self.objective_rules_per_day.items()
            },
            "level_rules": {k: v.tolist() for k, v in self.level_rules.items()},
            "rule_converged": {k: v.tolist() for k, v in self.rule_converged.items()},
            "gap_fraction": self.gap_fraction().tolist(),
            "storage_gain_fraction": self.storage_gain_fraction().tolist(),
            "checks": self.checks,
        }


def evaluate_stage_b(
    bank: StreamBank,
    scenario: Scenario,
    *,
    criterion: Criterion,
    signal: Signal,
    size_kw: float,
    pair: DesignPair,
    levels_celsius: Sequence[float],
    capacities_kwh: Sequence[float],
    rules: Sequence[RuleName] = RULES,
    daily_neutral: bool = True,
    max_cycles: int = 30,
) -> StageBResult:
    """Pump only (C = 0), foresight programme and rules for every capacity (C > 0) and level."""
    ph = scenario.placeholders
    levels = np.asarray(levels_celsius, dtype=np.float64)
    tc, td = storage_stream_temperatures(
        levels, minimum_pinch_k=ph.minimum_pinch_k, approach_k=ph.approach_k
    )
    extra = np.concatenate([[pair.evaporator_shifted, pair.condenser_shifted], tc, td])
    cascade = site_cascade(bank, scenario, extra)
    dt_h = bank.dt_h
    per_day = 1.0 / scenario.n_days
    n = cascade.n_steps
    clock = Clock.of(scenario.n_days)
    elec = signal_series(criterion.electricity_value_per_kwh, signal, clock)
    c_u = criterion.heat_value_per_kwh
    direct = direct_hot_utility_kw(cascade)
    j_none = float(np.sum(c_u * direct) * dt_h)
    lines = heat_pump_lines(cascade, pair.evaporator_shifted, pair.condenser_shifted, pair.cop)
    d0 = dispatch_steps(lines, size_kw, c_u, elec)
    j_pump = float(np.sum(c_u * d0.utility_kw + elec * d0.work_kw) * dt_h)

    caps = np.asarray([c for c in capacities_kwh if c > 0.0], dtype=np.float64)
    n_l, n_c = len(levels), len(caps)
    fore = np.full((n_l, n_c), np.nan)
    neutral = np.full((n_l, n_c), np.nan) if (daily_neutral and scenario.n_days > 1) else None
    rule_obj = {r: np.full((n_l, n_c), np.nan) for r in rules}
    rule_ok = {r: np.zeros((n_l, n_c), dtype=bool) for r in rules}
    worst_balance = 0.0
    pump_only_check = 0.0
    for li in range(n_l):
        pieces = coupled_pieces(
            cascade,
            evaporator_shifted=pair.evaporator_shifted,
            condenser_shifted=pair.condenser_shifted,
            cop=pair.cop,
            charge_shifted=float(tc[li]),
            discharge_shifted=float(td[li]),
        )
        if li == 0:
            zero = solve_foresight(
                pieces,
                q_max_kw=size_kw,
                capacity_kwh=0.0,
                loss_per_h=scenario.loss_per_h,
                dt_h=dt_h,
                heat_value=c_u,
                electricity=elec,
            )
            pump_only_check = abs(zero.objective - j_pump) / max(1e-9, j_none)
        for ci, cap in enumerate(caps):
            res = solve_foresight(
                pieces,
                q_max_kw=size_kw,
                capacity_kwh=float(cap),
                loss_per_h=scenario.loss_per_h,
                dt_h=dt_h,
                heat_value=c_u,
                electricity=elec,
            )
            fore[li, ci] = res.objective
            worst_balance = max(worst_balance, abs(res.balance_residual_kwh(dt_h)))
            if neutral is not None:
                neutral[li, ci] = solve_foresight(
                    pieces,
                    q_max_kw=size_kw,
                    capacity_kwh=float(cap),
                    loss_per_h=scenario.loss_per_h,
                    dt_h=dt_h,
                    heat_value=c_u,
                    electricity=elec,
                    steps_per_day=96,
                ).objective
        for r in rules:
            run = simulate_rule(
                pieces,
                r,
                q_max_kw=size_kw,
                capacities_kwh=caps,
                loss_per_h=scenario.loss_per_h,
                dt_h=dt_h,
                heat_value=c_u,
                electricity=elec,
                max_cycles=max_cycles,
            )
            rule_obj[r][li] = run.objective
            rule_ok[r][li] = run.converged
    del n
    best_f = np.argmin(fore, axis=0) if n_c else np.zeros(0, dtype=int)
    cols = np.arange(n_c)
    obj_fore = fore[best_f, cols] * per_day if n_c else np.zeros(0)
    obj_neutral = None
    if neutral is not None and n_c:
        obj_neutral = np.min(neutral, axis=0) * per_day
    obj_rules: dict[str, FloatArray] = {}
    lvl_rules: dict[str, FloatArray] = {}
    conv: dict[str, NDArray[np.bool_]] = {}
    for r in rules:
        br = np.argmin(rule_obj[r], axis=0)
        obj_rules[r] = rule_obj[r][br, cols] * per_day
        lvl_rules[r] = levels[br]
        conv[r] = rule_ok[r][br, cols]
    tol = 1e-7 * max(1.0, j_none * per_day)
    best_rule = np.min(np.stack(list(obj_rules.values())), axis=0) if rules else obj_fore
    checks = {
        "pump_only_programme_equals_step_dispatch": bool(pump_only_check <= 1e-7),
        "pump_only_programme_relative_difference": pump_only_check,
        "foresight_not_above_pump_only": bool(np.all(obj_fore <= j_pump * per_day + tol)),
        "foresight_non_increasing_in_capacity": bool(np.all(np.diff(obj_fore) <= tol)),
        "rules_not_below_foresight": bool(np.all(best_rule >= obj_fore - tol)),
        "daily_neutral_not_below_foresight": bool(
            obj_neutral is None or np.all(obj_neutral >= obj_fore - tol)
        ),
        "pump_not_above_none": bool(j_pump <= j_none + tol / per_day),
        "max_storage_balance_residual_kwh": worst_balance,
        "rules_converged_all": bool(all(np.all(v) for v in conv.values())),
    }
    return StageBResult(
        label=scenario.label,
        criterion=criterion.name,
        amplitude=signal.amplitude,
        size_kw=float(size_kw),
        pair=pair.label,
        cop=pair.cop,
        capacities_kwh=tuple(float(c) for c in caps),
        direct_kwh_per_day=float(np.sum(direct) * dt_h * per_day),
        objective_none_per_day=j_none * per_day,
        objective_pump_only_per_day=j_pump * per_day,
        objective_foresight_per_day=obj_fore,
        level_foresight=levels[best_f] if n_c else np.zeros(0),
        objective_daily_neutral_per_day=obj_neutral,
        objective_rules_per_day=obj_rules,
        level_rules=lvl_rules,
        rule_converged=conv,
        checks=checks,
    )
