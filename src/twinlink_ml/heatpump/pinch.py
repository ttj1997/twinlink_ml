r"""Simple pinch calculation for the site (A, C, A+C) and the placement rule of a heat pump.

Built on the vectorised problem table of storage test 1
(:func:`twinlink_ml.storage_timeslice.table.cascade_profile`, which equals
:func:`twinlink_ml.heatnetwork.pinch.compute_pinch_target` step by step). Per time step:

* minimum hot utility ``Q_H = max(0, -min R_b)`` and cold utility ``Q_C = R_last + Q_H`` with the
  cascade ``R_b`` (heat flowing down through boundary ``b`` before utilities);
* heat flow of the grand composite curve ``H_b = R_b + Q_H >= 0``; the PINCH is every boundary with
  ``H_b = 0`` (several pinch points or a whole pinch region are possible), reported as the highest
  and the lowest such shifted temperature.

Placement rule (Townsend and Linnhoff 1983; textbook: Kemp, "Pinch Analysis and Process
Integration"): a heat pump saves hot utility only if it takes heat BELOW the pinch and delivers it
ABOVE (``across``): then ``delta Q_H = Q_k`` and the cold utility falls by ``Q_e``. Entirely above
the pinch it only turns work into heat (``delta Q_H = W``, an electric heater with extra steps);
entirely below it saves nothing (``delta Q_H = 0``, it adds ``W`` to the cold utility). The problem
table with the heat pump as two pseudo-streams (:mod:`twinlink_ml.heatpump.cop`) reproduces this
without being told; :func:`classify_placement` labels each step for the report and the tests.

This module is deliberately small; the site pinch study that runs in parallel (study 1) is the
place to merge it later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..storage_timeslice.table import Cascade

__all__ = [
    "PLACEMENTS",
    "PinchSummary",
    "classify_placement",
    "grand_composite_kw",
    "pinch_summary",
]

FloatArray = NDArray[np.float64]
PLACEMENTS = ("across", "above", "below", "partial", "no_deficit")
_FLOW_TOLERANCE_KW = 1e-6
_TEMPERATURE_TOLERANCE_K = 1e-7


@dataclass(frozen=True, slots=True)
class PinchSummary:
    """Per-step pinch result (shifted temperatures; ``nan`` where there is no pinch point).

    :ivar hot_utility_kw, cold_utility_kw: minimum utilities.
    :ivar pinch_high, pinch_low: highest and lowest boundary with zero heat flow. With
        ``Q_H = 0`` the top boundary has zero flow (a threshold problem without heating), with
        ``Q_C = 0`` the bottom one (threshold without cooling).
    """

    hot_utility_kw: FloatArray
    cold_utility_kw: FloatArray
    pinch_high: FloatArray
    pinch_low: FloatArray


def _flows(cascade: Cascade) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    finite = np.isfinite(cascade.residual_kw)
    residual = np.where(finite, cascade.residual_kw, np.inf)
    hot = np.maximum(0.0, -np.min(residual, axis=1))
    flow = np.where(finite, cascade.residual_kw + hot[:, None], np.nan)
    return hot, flow, finite


def pinch_summary(cascade: Cascade) -> PinchSummary:
    """Utilities and pinch temperatures of every step."""
    hot, flow, finite = _flows(cascade)
    last = np.sum(finite, axis=1) - 1
    rows = np.arange(cascade.n_steps)
    cold = flow[rows, last]
    tol = _FLOW_TOLERANCE_KW * np.maximum(1.0, hot)[:, None]
    zero = finite & (np.abs(np.nan_to_num(flow, nan=np.inf)) <= tol)
    bounds = np.where(zero, cascade.boundaries, np.nan)
    any_zero = np.any(zero, axis=1)
    high = np.full(cascade.n_steps, np.nan)
    low = np.full(cascade.n_steps, np.nan)
    high[any_zero] = np.nanmax(bounds[any_zero], axis=1)
    low[any_zero] = np.nanmin(bounds[any_zero], axis=1)
    return PinchSummary(hot_utility_kw=hot, cold_utility_kw=cold, pinch_high=high, pinch_low=low)


def grand_composite_kw(cascade: Cascade, shifted_celsius: FloatArray) -> FloatArray:
    """Heat flow ``H`` of the grand composite curve at the given shifted temperatures.

    Linear between the boundaries of each step; above the top boundary ``H`` equals the hot
    utility, below the lowest the cold utility. Shape ``(steps, len(shifted_celsius))``.
    """
    temps = np.asarray(shifted_celsius, dtype=np.float64)
    hot, flow, finite = _flows(cascade)
    out = np.zeros((cascade.n_steps, len(temps)))
    for i in range(cascade.n_steps):
        b = cascade.boundaries[i, finite[i]][::-1]  # ascending
        h = flow[i, finite[i]][::-1]
        out[i] = np.interp(temps, b, h, left=h[0], right=h[-1])
    return out


def classify_placement(
    evaporator_shifted: float, condenser_shifted: float, summary: PinchSummary
) -> NDArray[np.str_]:
    """Placement of a heat pump relative to the pinch, per step (labels of :data:`PLACEMENTS`).

    ``across``: evaporator strictly below the lowest and condenser strictly above the highest
    pinch point; ``above``: evaporator at or above the highest pinch point; ``below``: condenser at
    or below the lowest; ``partial``: anything else (inside a pinch region); ``no_deficit``: no hot
    utility in that step (nothing to replace).
    """
    te, tk = float(evaporator_shifted), float(condenser_shifted)
    tol = _TEMPERATURE_TOLERANCE_K
    high, low = summary.pinch_high, summary.pinch_low
    out = np.full(len(high), "partial", dtype="<U10")
    out[(te < low - tol) & (tk > high + tol)] = "across"
    out[te >= high - tol] = "above"
    out[tk <= low + tol] = "below"
    out[summary.hot_utility_kw <= _FLOW_TOLERANCE_KW] = "no_deficit"
    return out
