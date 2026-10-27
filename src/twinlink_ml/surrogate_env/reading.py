r"""Metrics and the pre-registered reading of the study docs/30 (G0 / G1 / G2).

The text of the reading is the docstring of ``scripts/analyze_surrogate_env.py``; this module holds
the arithmetic so that it can be tested. All quantities are on the grey-box model (truth) at the 25
Befund-52 grid points against the TRUE limit 0.15.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ENERGY_EXCESS_MAX",
    "FEASIBILITY_SHARE_MIN",
    "Criterion",
    "classify",
    "criterion",
    "energy_excess",
    "point_metrics",
]

FEASIBILITY_SHARE_MIN = 0.95
"""G0: the moisture limit (with every other constraint) holds in at least 95 % of the cases."""

ENERGY_EXCESS_MAX = 0.02
"""G0: at most 2 % energy above the result of the same controller on U0."""


def point_metrics(rows: Sequence[Mapping[str, Any]], limit: float) -> dict[str, float]:
    """Metrics of ONE replicate from its per-point rows.

    Each row: ``true_feasible``, ``true_solve_failed``, ``true_energy_kw``,
    ``true_moisture_kg_per_kg``, ``energy_oracle_kw`` (true optimum, limit 0.15),
    ``energy_oracle_m_kw`` (Oracle(m)), optional ``predicted`` (dict of the environment's
    prediction with ``valid``, ``energy_kw``, ``moisture_kg_per_kg``,
    ``moisture_for_constraint_kg_per_kg``).
    """
    n = len(rows)
    feasible = np.array([bool(r["true_feasible"]) for r in rows])
    solved = np.array([not bool(r["true_solve_failed"]) for r in rows])
    energy = np.array([float(r["true_energy_kw"]) for r in rows])
    moisture = np.array([float(r["true_moisture_kg_per_kg"]) for r in rows])
    e_opt = np.array([float(r["energy_oracle_kw"]) for r in rows])
    e_opt_m = np.array([float(r["energy_oracle_m_kw"]) for r in rows])
    violation = np.where(solved, np.maximum(0.0, moisture - limit), np.nan)
    out: dict[str, float] = {
        "n_points": float(n),
        "n_true_solve_failed": float((~solved).sum()),
        "feasibility_share": float(feasible.mean()),
        "violation_share": float(1.0 - feasible.mean()),
        "max_moisture_violation_kg_per_kg": float(np.nanmax(violation))
        if solved.any()
        else math.nan,
        "sum_true_energy_kw": float(energy[solved].sum()),
        "energy_over_optimum_pct": 100.0 * (energy[solved].sum() / e_opt[solved].sum() - 1.0),
        "energy_over_oracle_m_pct": 100.0 * (energy[solved].sum() / e_opt_m[solved].sum() - 1.0),
    }
    predicted = [r.get("predicted") for r in rows]
    if all(p is None for p in predicted):
        return out
    valid = np.array([p is not None and bool(p["valid"]) for p in predicted]) & solved
    out["n_prediction_invalid"] = float(
        sum(1 for p in predicted if p is not None and not bool(p["valid"]))
    )
    if valid.any():
        e_pred = np.array(
            [float(p["energy_kw"]) if v else np.nan for p, v in zip(predicted, valid, strict=True)]
        )
        x_pred = np.array(
            [
                float(p["moisture_kg_per_kg"]) if v else np.nan
                for p, v in zip(predicted, valid, strict=True)
            ]
        )
        x_con = np.array(
            [
                float(p["moisture_for_constraint_kg_per_kg"]) if v else np.nan
                for p, v in zip(predicted, valid, strict=True)
            ]
        )
        out["gap_energy_mean_kw"] = float(np.nanmean(e_pred - energy))
        out["gap_energy_pooled_pct"] = 100.0 * (np.nansum(e_pred) / energy[valid].sum() - 1.0)
        out["gap_moisture_mean_kg_per_kg"] = float(np.nanmean(x_pred - moisture))
        out["gap_moisture_constraint_mean_kg_per_kg"] = float(np.nanmean(x_con - moisture))
        model_feasible = x_con <= limit
        out["n_model_feasible_truly_infeasible"] = float(np.sum(valid & model_feasible & ~feasible))
    return out


@dataclass(frozen=True, slots=True)
class Criterion:
    """G0 criteria of one (environment, controller) against U0."""

    feasibility_share_pooled: float
    energy_excess_median: float
    energy_excess_per_replicate: tuple[float, ...]
    passes: bool


def energy_excess(energy: np.ndarray, reference_energies: Sequence[np.ndarray]) -> float:
    """``sum(E) / median_r sum(E_ref,r) - 1`` over the points solved in ``energy`` and every ref.

    ``energy (25,)`` true energy of one replicate (NaN where the truth did not solve),
    ``reference_energies``: the same arrays of the controller's U0 replicates.
    """
    mask = np.isfinite(energy)
    for ref in reference_energies:
        mask &= np.isfinite(ref)
    reference = float(np.median([ref[mask].sum() for ref in reference_energies]))
    return float(energy[mask].sum() / reference - 1.0)


def criterion(
    feasible_counts: Sequence[tuple[int, int]], excess_per_replicate: Sequence[float]
) -> Criterion:
    """Feasibility pooled over replicates and points; energy excess median over replicates.

    ``feasible_counts``: ``(n_feasible, n_points)`` per replicate; ``excess_per_replicate``: the
    :func:`energy_excess` of every replicate against the controller on U0.
    """
    n_feasible = sum(c[0] for c in feasible_counts)
    n_total = sum(c[1] for c in feasible_counts)
    share = n_feasible / n_total
    excess = tuple(float(e) for e in excess_per_replicate)
    median = float(np.median(excess))
    return Criterion(
        feasibility_share_pooled=share,
        energy_excess_median=median,
        energy_excess_per_replicate=excess,
        passes=share >= FEASIBILITY_SHARE_MIN and median <= ENERGY_EXCESS_MAX,
    )


def classify(u2: Criterion, u3: Criterion) -> str:
    """G0 if U2 passes; G1 if U2 fails and U3 passes; G2 otherwise."""
    if u2.passes:
        return "G0"
    if u3.passes:
        return "G1"
    return "G2"
