r"""Weighted least squares with bounds, the constant-bias baseline, cross-check metrics, reading.

Residuals are weighted with the sensor sigma of their row (:math:`r_i = (y_i - m_i)/\sigma_i`);
the decision vector is :math:`\theta_j / s_j` (parameter scale). A solver failure at a trial point
gives a residual of 100 K per temperature (as Befund 117), divided by the row's sigma.

Bias baseline (B1 of Befund 120): the uncalibrated model plus, per measured quantity (bed,
exhaust), the mean residual measured - model over the segments of the calibration day; applied
unchanged to the other day.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "FAILURE_RESIDUAL_K",
    "FitResult",
    "apply_bias",
    "bias_offsets",
    "error_metrics",
    "fit_parameters",
    "reading",
    "weighted_residuals",
]

FAILURE_RESIDUAL_K = 100.0
QUANTITIES = ("bed", "exhaust")


def weighted_residuals(model: np.ndarray, measured: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """(model - measured) / sigma; a non-finite model value gives 100 K / sigma."""
    r = (np.asarray(model) - np.asarray(measured)) / np.asarray(sigma)
    return np.where(np.isfinite(r), r, FAILURE_RESIDUAL_K / np.asarray(sigma))


@dataclass(frozen=True)
class FitResult:
    """Result of :func:`fit_parameters`."""

    parameters: tuple[str, ...]
    theta: dict[str, float]
    ssr: float  # sum of squared weighted residuals
    n_residuals: int
    status: int
    message: str
    evaluations: int
    at_bound: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameters": list(self.parameters),
            "theta": dict(self.theta),
            "ssr": self.ssr,
            "n_residuals": self.n_residuals,
            "status": self.status,
            "message": self.message,
            "evaluations": self.evaluations,
            "at_bound": list(self.at_bound),
        }


def fit_parameters(
    outputs_for: Callable[[Mapping[str, float]], np.ndarray],
    measured: np.ndarray,
    sigma: np.ndarray,
    theta_fixed: Mapping[str, float],
    free: Sequence[str],
    *,
    scales: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
    start: Mapping[str, float],
    max_nfev: int,
) -> FitResult:
    """Fit ``free`` (others at ``theta_fixed``) by bounded weighted least squares.

    With no free parameter the result is the evaluation at ``theta_fixed``.
    """
    names = tuple(free)

    def theta_of(x: np.ndarray) -> dict[str, float]:
        t = dict(theta_fixed)
        for n, v in zip(names, x, strict=True):
            t[n] = float(v) * scales[n]
        return t

    def residuals(x: np.ndarray) -> np.ndarray:
        return weighted_residuals(outputs_for(theta_of(x)), measured, sigma)

    if not names:
        r = residuals(np.zeros(0))
        return FitResult((), dict(theta_fixed), float(np.sum(r**2)), r.size, 0, "no free", 1, ())
    lo = np.array([bounds[n][0] / scales[n] for n in names])
    hi = np.array([bounds[n][1] / scales[n] for n in names])
    x0 = np.clip(np.array([start[n] / scales[n] for n in names]), lo, hi)
    result = least_squares(
        residuals,
        x0,
        bounds=(lo, hi),
        method="trf",
        diff_step=1e-3,
        xtol=1e-8,
        ftol=1e-10,
        max_nfev=max_nfev,
    )
    final = residuals(result.x)
    at_bound = tuple(
        n
        for n, v, a, b in zip(names, result.x, lo, hi, strict=True)
        if v <= a + 1e-9 or v >= b - 1e-9
    )
    return FitResult(
        parameters=names,
        theta=theta_of(result.x),
        ssr=float(np.sum(final**2)),
        n_residuals=int(final.size),
        status=int(result.status),
        message=str(result.message),
        evaluations=int(result.nfev),
        at_bound=at_bound,
    )


def bias_offsets(model: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """Mean residual measured - model per quantity over rows ``[bed, exh, bed, exh, ...]``."""
    d = np.asarray(measured) - np.asarray(model)
    return {q: float(np.mean(d[i::2])) for i, q in enumerate(QUANTITIES)}


def apply_bias(model: np.ndarray, bias: Mapping[str, float]) -> np.ndarray:
    """Add the bias per quantity to model rows ``[bed, exh, ...]``."""
    out = np.array(model, dtype=np.float64)
    for i, q in enumerate(QUANTITIES):
        out[i::2] += bias[q]
    return out


def error_metrics(model: np.ndarray, measured: np.ndarray) -> dict[str, Any]:
    """Per quantity: errors model - measured per segment, max |e|, RMS, mean (K)."""
    e = np.asarray(model, dtype=np.float64) - np.asarray(measured, dtype=np.float64)
    out: dict[str, Any] = {}
    for i, q in enumerate(QUANTITIES):
        eq = e[i::2]
        finite = bool(np.all(np.isfinite(eq)))
        out[q] = {
            "errors_k": [float(v) for v in eq],
            "max_abs_k": float(np.max(np.abs(eq))) if finite else math.inf,
            "rms_k": float(np.sqrt(np.mean(eq**2))) if finite else math.inf,
            "mean_k": float(np.mean(eq)) if finite else math.inf,
        }
    out["max_abs_k"] = max(out[q]["max_abs_k"] for q in QUANTITIES)
    return out


def reading(
    calibrated: Mapping[str, Mapping[str, Any]],
    bias: Mapping[str, Mapping[str, Any]],
    tolerance_k: float,
) -> dict[str, Any]:
    """Pre-registered reading on the cross-check metrics per direction (see the analysis script).

    ``calibrated`` / ``bias``: direction -> :func:`error_metrics` on the other day.
    K0: calibrated within ``tolerance_k`` for bed and exhaust in every direction AND its largest
    error below the bias's. K1: largest error not below the bias's. ``between``: below the bias
    but above the tolerance somewhere (case not named in the request; no reading of its own).
    """
    e_cal = max(m["max_abs_k"] for m in calibrated.values())
    e_bias = max(m["max_abs_k"] for m in bias.values())
    within = all(m[q]["max_abs_k"] <= tolerance_k for m in calibrated.values() for q in QUANTITIES)
    better = e_cal < e_bias
    label = "K0" if (within and better) else ("K1" if not better else "between")
    return {
        "reading": label,
        "largest_cross_error_calibrated_k": e_cal,
        "largest_cross_error_bias_k": e_bias,
        "within_tolerance": within,
        "better_than_bias": better,
        "tolerance_k": tolerance_k,
    }
