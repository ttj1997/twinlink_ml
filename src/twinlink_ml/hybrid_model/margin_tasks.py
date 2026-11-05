r"""Margin search and evaluation per draw, pooled parameter of a cell (docs/NN_hybrid_model.md).

The margins part of ``scripts/run_hybrid_evaluation.py`` used to run one job per cell (model,
master seed, ``n_train``) over all 20 draws. Since the failed run of a343840 (a draw without any
inside case gave an empty case set to the search; decision of the project lead, 2026-11-05) the
work is split per draw, with the computation unchanged:

1. **Search** per (cell, draw): per INSIDE case the needed constraint parameter
   (:func:`search_draw`, bisection of :func:`grid_rto.search_parameter`); a draw without any inside
   case has no search. Plus the box-wide moisture bias on the coarse grid
   (:func:`coarse_bias_draw`).
2. **Parameter** of the cell: the ``ceil(0.95 N)``-th smallest needed value over the inside cases
   of all draws (:func:`cell_parameter`); raise rule of Befund 111 (:func:`next_round`).
3. **Evaluation** per (cell, draw, round): EVERY case of the draw optimised at the round's parameter
   and evaluated at the truth (:func:`evaluate_draw`). If the holding share over the inside cases
   of all draws is below 95 %, the next larger needed value is the parameter of the next round.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .grid_rto import (
    BisectionSettings,
    CaseEvaluator,
    GridSettings,
    _box,
    grid_zoom,
    search_parameter,
    truth_holds,
)
from .study import Case

__all__ = [
    "CellParameter",
    "RoundDecision",
    "cell_parameter",
    "coarse_bias_draw",
    "evaluate_draw",
    "limits_ks",
    "next_round",
    "search_draw",
]

Bounds = tuple[tuple[float, float], tuple[float, float]]
BED_CHECK_C = 80.0 + 0.05  # bed limit plus the bandit's tolerance (as before the split)


def finite(x: float) -> float | None:
    """``x`` as float, ``None`` if not finite (JSON)."""
    return float(x) if np.isfinite(x) else None


def limits_ks(p: np.ndarray, limit: float, is_k: bool) -> tuple[np.ndarray, np.ndarray]:
    """Constraint per point: M4 ``(limit, k = p)``, M0-M3 ``(limit - m, 0)`` with ``m = p``."""
    if is_k:
        return np.full(p.shape, limit), p
    return limit - p, np.zeros(p.shape)


@dataclass(frozen=True, slots=True)
class DrawSearch:
    """Needed parameter and number of RTO solves per inside case (slot -> value)."""

    needed: dict[int, float]
    n_solves: dict[int, int]


def search_draw(
    model_ev: CaseEvaluator,
    truth_ev: CaseEvaluator,
    inside: np.ndarray,
    *,
    limit: float,
    is_k: bool,
    bounds: Bounds,
    grid: GridSettings,
    settings: BisectionSettings,
) -> DrawSearch:
    """Bisection of the needed parameter at the inside cases of one draw; none without any."""
    inside_idx = np.flatnonzero(np.asarray(inside, dtype=bool))
    if inside_idx.size == 0:
        return DrawSearch(needed={}, n_solves={})

    def hold_at(rows: np.ndarray, p: np.ndarray) -> np.ndarray:
        lim, ks = limits_ks(p, limit, is_k)
        actions, _ = grid_zoom(model_ev, inside_idx[rows], lim, ks, bounds, grid)
        return truth_holds(truth_ev.values(inside_idx[rows], actions), limit)

    result = search_parameter(hold_at, inside_idx.size, settings)
    return DrawSearch(
        needed=dict(zip(inside_idx.tolist(), result.needed.tolist(), strict=True)),
        n_solves=dict(zip(inside_idx.tolist(), result.n_solves.tolist(), strict=True)),
    )


def coarse_bias_draw(
    model_ev: CaseEvaluator,
    truth_ev: CaseEvaluator,
    cases: Sequence[Case],
    inside: np.ndarray,
    *,
    draw: int,
    bounds: Bounds,
    grid: GridSettings,
) -> list[dict[str, Any]]:
    """Box-wide moisture bias (model minus truth, valid points of both) on the coarse grid."""
    coarse = _box(bounds, grid.coarse)
    rows = []
    for c, case in enumerate(cases):
        cidx = np.full(coarse.shape[0], c)
        pm = model_ev.values(cidx, coarse)
        pt = truth_ev.values(cidx, coarse)
        ok = pm["valid"] & pt["valid"]
        diff = pm["moisture_kg_per_kg"][ok] - pt["moisture_kg_per_kg"][ok]
        rows.append(
            {
                "draw": draw,
                "slot": case.slot,
                "kind": case.kind,
                "inside": bool(inside[c]),
                "n_valid": int(ok.sum()),
                "n_model_invalid": int((~pm["valid"]).sum()),
                "bias": finite(diff.mean()) if diff.size else None,
                "rmse": finite(np.sqrt(np.mean(diff**2))) if diff.size else None,
            }
        )
    return rows


def evaluate_draw(
    model_ev: CaseEvaluator,
    truth_ev: CaseEvaluator,
    cases: Sequence[Case],
    inside: np.ndarray,
    oracle_cases: Sequence[dict[str, Any]],
    parameter: float,
    *,
    draw: int,
    limit: float,
    is_k: bool,
    bounds: Bounds,
    grid: GridSettings,
) -> list[dict[str, Any]]:
    """Every case of one draw optimised at ``parameter`` and evaluated at the truth."""
    k = len(cases)
    idx = np.arange(k)
    lim, ks = limits_ks(np.full(k, parameter), limit, is_k)
    actions, j = grid_zoom(model_ev, idx, lim, ks, bounds, grid)
    pred = model_ev.values(idx, actions)
    tru = truth_ev.values(idx, actions)
    holds = truth_holds(tru, limit)
    rows: list[dict[str, Any]] = []
    for c, case in enumerate(cases):
        orc = oracle_cases[c]
        e_true = tru["energy_kw"][c]
        rows.append(
            {
                "draw": draw,
                "slot": case.slot,
                "kind": case.kind,
                "inside": bool(inside[c]),
                "action": actions[c].tolist(),
                "objective": float(j[c]),
                "predicted": {f: finite(pred[f][c]) for f in pred if f != "valid"}
                | {"valid": bool(pred["valid"][c])},
                "truth": {
                    f: finite(tru[f][c])
                    for f in ("energy_kw", "moisture_kg_per_kg", "bed_temperature_celsius")
                }
                | {"valid": bool(tru["valid"][c])},
                "holds": bool(holds[c]),
                "bed_violated": bool(
                    tru["valid"][c] and tru["bed_temperature_celsius"][c] > BED_CHECK_C
                ),
                "oracle_energy_kw": orc["energy_kw"],
                "energy_excess_relative": finite((e_true - orc["energy_kw"]) / orc["energy_kw"])
                if orc["energy_kw"]
                else None,
                "moisture_bias_at_action": finite(
                    pred["moisture_kg_per_kg"][c] - tru["moisture_kg_per_kg"][c]
                ),
                # constant margin: m; ensemble: k * std at the chosen point
                "effective_margin": finite(
                    (limit - lim[c])
                    + (parameter if is_k else 0.0) * pred["moisture_std_kg_per_kg"][c]
                )
                if pred["valid"][c]
                else None,
            }
        )
    return rows


@dataclass(frozen=True, slots=True)
class CellParameter:
    """Pooled parameter of a cell: quantile, rank, candidates of the raise rule, reached."""

    n_inside: int
    rank: int
    quantile: float
    candidates: tuple[float, ...]
    reached: bool


def cell_parameter(needed: Sequence[float], holding_share: float) -> CellParameter:
    """``ceil(share N)``-th smallest needed value over the pooled inside cases (N may be 0).

    Candidates of the raise rule: the quantile, then every larger finite needed value.
    """
    values = np.asarray(list(needed), dtype=np.float64)
    n_inside = values.size
    ordered = np.sort(values)
    rank = max(1, math.ceil(holding_share * n_inside)) if n_inside else 0
    quantile = float(ordered[rank - 1]) if n_inside else math.nan
    candidates = (
        quantile,
        *sorted({float(v) for v in ordered if np.isfinite(v) and v > quantile}),
    )
    reached = bool(n_inside) and bool(np.isfinite(quantile))
    return CellParameter(n_inside, rank, quantile, candidates, reached)


@dataclass(frozen=True, slots=True)
class RoundDecision:
    """After the evaluated rounds: the next round's parameter, or the final one."""

    done: bool
    parameter: float
    raised: int


def next_round(
    cell: CellParameter, shares: Sequence[float], holding_share: float, upper: float
) -> RoundDecision:
    """Raise rule of Befund 111 over the rounds evaluated so far (``shares`` per round).

    Not reached (no inside case, or the quantile is not finite): one round at ``upper``. Otherwise
    round r evaluates candidate r; the cell is done at the first round holding in at least
    ``holding_share`` of the inside cases, or after the last candidate (``raised`` = number of
    rounds below it).
    """
    if not cell.reached:
        return RoundDecision(done=bool(shares), parameter=upper, raised=0)
    for r, share in enumerate(shares):
        if share >= holding_share:
            return RoundDecision(done=True, parameter=cell.candidates[r], raised=r)
    r = len(shares)
    if r < len(cell.candidates):
        return RoundDecision(done=False, parameter=cell.candidates[r], raised=r)
    return RoundDecision(done=True, parameter=cell.candidates[-1], raised=r)
