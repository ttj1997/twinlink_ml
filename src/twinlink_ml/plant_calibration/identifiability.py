r"""Local identifiability: normalised sensitivity matrix, SVD, collinearity, profile verdict.

Normalisation: :math:`S_{ij} = \partial y_i/\partial\theta_j \cdot s_j / \sigma_i` with the
parameter scale :math:`s_j` and the sensor sigma :math:`\sigma_i` of output row ``i``. Central
differences with the step ``h s_j``: :math:`S_{ij} = (y_i^+ - y_i^-)/(2 h \sigma_i)`; a trial
value outside the parameter's bounds is clipped (one-sided difference, e.g. at UA = 0).

Measures of Brun et al. (2001, Water Res. 35:3667):

* sensitivity measure :math:`\delta^{msqr}_j = \sqrt{\tfrac1n \sum_i S_{ij}^2}`;
* collinearity index of a subset ``K``: :math:`\gamma_K = 1/\sqrt{\lambda_{min}(\tilde S_K^T
  \tilde S_K)}`, :math:`\tilde S` = columns scaled to unit length (infinite if a column is 0);
* determinant measure :math:`\rho_K = \det(S_K^T S_K)^{1/(2|K|)}`.

A subset is identifiable if every :math:`\delta_j \ge \delta_{min}` and
:math:`\gamma_K \le \gamma_{max}`. Selection (:func:`select_subset`): the largest identifiable
subset of the eligible parameters (at most ``max_size``); ties: largest :math:`\rho_K`, then the
order of the names.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "SubsetScore",
    "collinearity_index",
    "delta_msqr",
    "determinant_measure",
    "fd_steps",
    "finite_difference_thetas",
    "identifiable_subsets",
    "profile_verdict",
    "select_subset",
    "sensitivity_from_outputs",
    "svd_summary",
]


def finite_difference_thetas(
    theta0: Mapping[str, float],
    names: Sequence[str],
    scales: Mapping[str, float],
    relative_step: float,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> list[dict[str, float]]:
    """``[theta0, theta0 + h s_1 e_1, theta0 - h s_1 e_1, theta0 + h s_2 e_2, ...]``.

    With ``bounds`` each trial value is clipped to the parameter's bounds (one-sided difference
    where the nominal value sits on a bound, e.g. UA = 0); :func:`fd_steps` gives the steps.
    """
    thetas = [dict(theta0)]
    for name in names:
        for sign in (1.0, -1.0):
            t = dict(theta0)
            value = theta0[name] + sign * relative_step * scales[name]
            if bounds is not None and name in bounds:
                lo, hi = bounds[name]
                value = min(max(value, lo), hi)
            t[name] = value
            thetas.append(t)
    return thetas


def fd_steps(
    thetas: Sequence[Mapping[str, float]], names: Sequence[str], scales: Mapping[str, float]
) -> np.ndarray:
    """Step (plus - minus) per parameter in units of its scale."""
    return np.array(
        [(thetas[1 + 2 * j][n] - thetas[2 + 2 * j][n]) / scales[n] for j, n in enumerate(names)]
    )


def sensitivity_from_outputs(
    outputs: np.ndarray, sigma_rows: np.ndarray, steps: np.ndarray
) -> np.ndarray:
    """Normalised ``S`` (rows x parameters) from the outputs of :func:`finite_difference_thetas`.

    ``outputs``: array ``(1 + 2 p, n_rows)``; row 0 is the base point. ``steps``: plus - minus per
    parameter in units of its scale (:func:`fd_steps`; ``2 h`` for a central difference).
    """
    y = np.asarray(outputs, dtype=np.float64)
    plus, minus = y[1::2], y[2::2]
    diff = (plus - minus) / np.asarray(steps, dtype=np.float64)[:, None]
    return diff.T / np.asarray(sigma_rows)[:, None]


def delta_msqr(s: np.ndarray) -> np.ndarray:
    """Sensitivity measure per column (RMS over rows)."""
    return np.sqrt(np.mean(np.asarray(s) ** 2, axis=0))


def collinearity_index(s_k: np.ndarray) -> float:
    """Collinearity index of the columns (infinite for a zero or dependent column)."""
    s_k = np.asarray(s_k, dtype=np.float64)
    if not np.all(np.isfinite(s_k)):
        return math.inf
    norms = np.linalg.norm(s_k, axis=0)
    if np.any(norms <= 0.0):
        return math.inf
    sn = s_k / norms
    lam = float(np.linalg.eigvalsh(sn.T @ sn).min())
    return math.inf if lam <= 1e-15 else 1.0 / math.sqrt(lam)


def determinant_measure(s_k: np.ndarray) -> float:
    """Determinant measure rho_K (0 if singular)."""
    s_k = np.asarray(s_k, dtype=np.float64)
    if not np.all(np.isfinite(s_k)):
        return 0.0
    det = float(np.linalg.det(s_k.T @ s_k))
    return 0.0 if det <= 0.0 else det ** (1.0 / (2 * s_k.shape[1]))


def svd_summary(s: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """Singular values, condition number and right singular vectors of ``S``."""
    s = np.asarray(s, dtype=np.float64)
    if not np.all(np.isfinite(s)):
        return {"finite": False}
    _, sv, vt = np.linalg.svd(s, full_matrices=False)
    tiny = sv[0] * 1e-12 if sv.size else 0.0
    condition = math.inf if sv.size == 0 or sv[-1] <= tiny else float(sv[0] / sv[-1])
    return {
        "finite": True,
        "parameters": list(names),
        "singular_values": [float(v) for v in sv],
        "condition_number": condition,
        "numerical_rank": int(np.sum(sv > tiny)),
        "right_singular_vectors": [
            {n: float(v) for n, v in zip(names, row, strict=True)} for row in vt
        ],
    }


@dataclass(frozen=True, slots=True)
class SubsetScore:
    """Scores of one parameter subset."""

    parameters: tuple[str, ...]
    gamma: float
    rho: float
    min_delta: float
    identifiable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameters": list(self.parameters),
            "gamma": self.gamma,
            "rho": self.rho,
            "min_delta": self.min_delta,
            "identifiable": self.identifiable,
        }


def identifiable_subsets(
    s: np.ndarray,
    names: Sequence[str],
    eligible: Sequence[str],
    *,
    delta_min: float,
    gamma_max: float,
    max_size: int,
) -> list[SubsetScore]:
    """Scores of every subset (size 1..``max_size``) of ``eligible``, in name order."""
    index = {n: i for i, n in enumerate(names)}
    deltas = delta_msqr(s)
    scores: list[SubsetScore] = []
    ordered = [n for n in names if n in set(eligible)]
    for size in range(1, max_size + 1):
        for subset in itertools.combinations(ordered, size):
            cols = [index[n] for n in subset]
            s_k = np.asarray(s)[:, cols]
            gamma = collinearity_index(s_k)
            md = float(np.min(deltas[cols]))
            scores.append(
                SubsetScore(
                    parameters=subset,
                    gamma=gamma,
                    rho=determinant_measure(s_k),
                    min_delta=md,
                    identifiable=bool(np.isfinite(md) and md >= delta_min and gamma <= gamma_max),
                )
            )
    return scores


def select_subset(scores: Sequence[SubsetScore], names: Sequence[str]) -> tuple[str, ...]:
    """Largest identifiable subset; ties: largest rho, then name order. Empty if none."""
    ok = [sc for sc in scores if sc.identifiable]
    if not ok:
        return ()
    rank = {n: i for i, n in enumerate(names)}
    best = max(
        ok,
        key=lambda sc: (
            len(sc.parameters),
            sc.rho,
            tuple(-rank[n] for n in sc.parameters),
        ),
    )
    return best.parameters


def _crossing(
    grid: np.ndarray, excess: np.ndarray, x_opt: float, side: str, threshold: float
) -> float | None:
    """Parameter value where ``excess`` first exceeds ``threshold`` going out from ``x_opt``."""
    sel = np.where(grid < x_opt)[0][::-1] if side == "lower" else np.where(grid > x_opt)[0]
    prev_x, prev_e = x_opt, 0.0
    for i in sel:
        if not np.isfinite(excess[i]):
            continue
        if excess[i] > threshold:
            frac = (threshold - prev_e) / (excess[i] - prev_e)
            return float(prev_x + frac * (grid[i] - prev_x))
        prev_x, prev_e = float(grid[i]), float(excess[i])
    return None


def profile_verdict(
    grid: Sequence[float],
    ssr: Sequence[float],
    *,
    x_opt: float,
    ssr_opt: float,
    bounds: tuple[float, float],
    n_residuals: int,
    n_free: int,
    threshold_chi2: float,
) -> dict[str, Any]:
    """Confidence interval from a profile of the weighted SSR (chi2 = SSR, sigma-normalised).

    Two variants: ``sensor`` (threshold on the SSR itself) and ``rescaled`` (SSR divided by
    ``s^2 = ssr_opt / (n - p)``, i.e. sigma rescaled so that the fit is "as good as the noise").
    Per side: the interpolated crossing, or ``None`` if the profile stays below the threshold up to
    the grid's end. ``practically_identifiable``: crossed on every side where the optimum is not
    on the bound.
    """
    g = np.asarray(grid, dtype=np.float64)
    e = np.asarray(ssr, dtype=np.float64) - ssr_opt
    order = np.argsort(g)
    g, e = g[order], e[order]
    dof = max(n_residuals - n_free, 1)
    s2 = max(ssr_opt / dof, 1e-300)
    lo_b, hi_b = bounds
    at_lo = x_opt <= lo_b + 1e-9 * max(1.0, abs(lo_b))
    at_hi = x_opt >= hi_b - 1e-9 * max(1.0, abs(hi_b))
    out: dict[str, Any] = {"s2": s2, "dof": dof, "x_opt": x_opt, "optimum_at_bound": at_lo or at_hi}
    for label, scale in (("sensor", 1.0), ("rescaled", s2)):
        lower = None if at_lo else _crossing(g, e / scale, x_opt, "lower", threshold_chi2)
        upper = None if at_hi else _crossing(g, e / scale, x_opt, "upper", threshold_chi2)
        identifiable = (at_lo or lower is not None) and (at_hi or upper is not None)
        out[label] = {
            "lower": lower,
            "upper": upper,
            "practically_identifiable": bool(identifiable),
            "max_excess": float(np.nanmax(e / scale)) if np.any(np.isfinite(e)) else None,
        }
    return out


Evaluator = Callable[[Sequence[Mapping[str, float]]], np.ndarray]
