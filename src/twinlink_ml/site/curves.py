r"""Problem table, network levels and heat-pump limits on piecewise-linear stream curves.

Representation
--------------
A set of stream segments ``[high, low, CP]`` (real temperatures, kW/K, the format of
:mod:`twinlink_ml.storage_timeslice.streams`) is turned into its "heat above T" function

.. math:: A(T) = \sum_i CP_i \,\max\bigl(0,\ T^{high}_i - \max(T, T^{low}_i)\bigr)
          = \sum_j w_j \max(0,\ p_j - T),

a sum of ramps with points ``p`` = all highs (weight ``+CP``) and lows (weight ``-CP``)
(:class:`Ramps`). ``A`` is piecewise linear with kinks only at the points, so every quantity below
is evaluated EXACTLY on the union of the relevant points (no temperature raster). Time averaging
is exact as well: the average of ramp sums is the ramp sum of all points with weights divided by
the number of steps (:meth:`Ramps.combine`).

Problem table (:func:`problem_table`): the same algorithm as
:func:`twinlink_ml.heatnetwork.pinch.compute_pinch_target` (shifted temperatures, cascade from the
top with start value 0); the cascade at the shifted temperature ``s`` is
``A_hot(s + dT/2) - A_cold(s - dT/2)``. ``tests/test_site.py`` checks the equality.

Composite loop at one level (:func:`network_capacity`, a diagnostic: which side limits a loop
that may pass the sources and sinks in any order and has no return cooler): a closed loop with
supply ``T_s`` and
return ``T_r`` and approach ``a`` at every exchanger. Sources heat the loop from ``T_r`` to
``T_s``; the loop is fully absorbed by the sinks while cooling back to ``T_r``. With the loop duty
``Q`` and ``phi(theta) = (T_s - theta)/(T_s - T_r)`` (loop duty above the loop temperature
``theta``) the source side needs no other heat iff
``Q phi(theta) <= A_hot(theta + a)`` for all ``theta`` (problem table, no hot utility), and the sink
side needs no cooling iff ``Q psi(theta) <= B_cold(theta - a)`` with ``psi = 1 - phi`` and ``B`` the
heat demand below a temperature (problem table from the bottom, no cold utility). Hence

.. math:: Q_{src} = \min_\theta \frac{A_{hot}(\theta + a)}{\phi(\theta)},\qquad
          Q_{sink} = \min_\theta \frac{B_{cold}(\theta - a)}{\psi(\theta)},\qquad
          Q_N = \min(Q_{src}, Q_{sink}),

the minimum taken over the loop end and the kinks (a ratio of two linear functions is monotone on
each piece). The check is the minimum approach ALONG every exchanger (counter-current), the same
condition as :func:`twinlink_ml.heatnetwork.approach.max_duty_along` (cross-checked in the tests).
With ``a = dT_min/2`` the network can never recover more than the direct target. Several loops at
once: a small LP over the loop duties (:func:`multi_loop_capacity`, HiGHS), exact on the union of
kinks. The network model of the study results is the zonal LP (:func:`zonal_network_target`):
direct recovery inside each zone, parallel connection of the zones to the loop.

Heat pump (:func:`heat_pump_limits`): from the grand composite curve ``G(s)`` (cascade plus the
minimum hot utility) an evaporator at the real temperature ``T_e`` (a cold stream, shifted
``T_e + dT/2``) can take at most ``E = min_{s <= s_e} G(s)`` and a condenser at ``T_c`` (a hot
stream, shifted ``T_c - dT/2``) can deliver at most ``U = min_{s >= s_c} G(s)`` without raising the
hot utility elsewhere. Both are zero on the wrong side of the pinch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Ramps",
    "TableResult",
    "heat_pump_limits",
    "multi_loop_capacity",
    "network_capacity",
    "problem_table",
    "zonal_network_target",
]

FloatArray = NDArray[np.float64]

_ROUND_DECIMALS = 9
"""Rounding of shifted temperatures before forming boundaries (as in ``heatnetwork.pinch``)."""


@dataclass(frozen=True, slots=True)
class Ramps:
    """``f(T) = sum_j w_j max(0, p_j - T)`` with the points sorted ascending.

    ``suffix_w[k]`` and ``suffix_wp[k]`` are the sums of ``w`` and ``w p`` over ``j >= k``.
    """

    points: FloatArray
    weights: FloatArray
    suffix_w: FloatArray
    suffix_wp: FloatArray

    @staticmethod
    def from_points(points: ArrayLike, weights: ArrayLike) -> Ramps:
        p = np.asarray(points, dtype=np.float64).ravel()
        w = np.asarray(weights, dtype=np.float64).ravel()
        if p.shape != w.shape:
            raise ValueError("points and weights must have the same length")
        keep = w != 0.0
        p, w = p[keep], w[keep]
        order = np.argsort(p, kind="stable")
        p, w = p[order], w[order]
        suffix_w = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
        suffix_wp = np.concatenate([np.cumsum((w * p)[::-1])[::-1], [0.0]])
        return Ramps(p, w, suffix_w, suffix_wp)

    @staticmethod
    def of_segments(segments: ArrayLike, scale: float = 1.0) -> Ramps:
        """Heat above ``T`` of segments ``(n, 3)`` = high, low, CP (CP 0 = padding)."""
        seg = np.asarray(segments, dtype=np.float64).reshape(-1, 3)
        seg = seg[seg[:, 2] > 0.0]
        if np.any(seg[:, 0] < seg[:, 1]):
            raise ValueError("segment high must not lie below its low")
        cp = seg[:, 2] * scale
        return Ramps.from_points(np.concatenate([seg[:, 0], seg[:, 1]]), np.concatenate([cp, -cp]))

    @staticmethod
    def combine(parts: Sequence[tuple[Ramps, float]]) -> Ramps:
        """Weighted sum ``sum_k c_k f_k`` (exact; used for time averages)."""
        if not parts:
            return Ramps.from_points(np.zeros(0), np.zeros(0))
        points = np.concatenate([r.points for r, _ in parts])
        weights = np.concatenate([r.weights * c for r, c in parts])
        return Ramps.from_points(points, weights)

    def __call__(self, temperature: ArrayLike) -> FloatArray:
        t = np.asarray(temperature, dtype=np.float64)
        k = np.searchsorted(self.points, t, side="right")
        return np.asarray(self.suffix_wp[k] - t * self.suffix_w[k], dtype=np.float64)

    @property
    def total(self) -> float:
        """Value below every point (the whole duty of the segments)."""
        if self.points.size == 0:
            return 0.0
        return float(self(self.points[0] - 1.0))

    def below(self, temperature: ArrayLike) -> FloatArray:
        """Heat BELOW ``T`` (``total - f(T)``)."""
        return self.total - self(temperature)


@dataclass(frozen=True, slots=True)
class TableResult:
    """Problem table of one step.

    :ivar shifted: shifted boundaries, descending.
    :ivar gcc_kw: grand composite curve at ``shifted`` (cascade + hot utility, non-negative).
    :ivar hot_utility_kw: minimum hot utility.
    :ivar cold_utility_kw: minimum cold utility.
    :ivar pinch_shifted: shifted pinch temperature (``nan`` without any stream).
    :ivar threshold: True if no hot utility is needed (no pinch in the usual sense).
    """

    shifted: FloatArray
    gcc_kw: FloatArray
    hot_utility_kw: float
    cold_utility_kw: float
    pinch_shifted: float
    threshold: bool


def problem_table(hot: Ramps, cold: Ramps, minimum_pinch_k: float) -> TableResult:
    """Minimum hot/cold utility, pinch and grand composite curve (module docstring)."""
    if minimum_pinch_k < 0.0:
        raise ValueError(f"minimum_pinch_k must not be negative, is {minimum_pinch_k}")
    half = minimum_pinch_k / 2.0
    raw = np.concatenate([hot.points - half, cold.points + half])
    if raw.size == 0:
        return TableResult(np.zeros(0), np.zeros(0), 0.0, 0.0, float("nan"), True)
    shifted = np.unique(np.round(raw, _ROUND_DECIMALS))[::-1]
    cascade = hot(shifted + half) - cold(shifted - half)
    index = int(np.argmin(cascade))
    hot_utility = max(0.0, -float(cascade[index]))
    gcc = cascade + hot_utility
    return TableResult(
        shifted=shifted,
        gcc_kw=gcc,
        hot_utility_kw=hot_utility,
        cold_utility_kw=float(gcc[-1]),
        pinch_shifted=float(shifted[index]),
        threshold=hot_utility <= 1e-9,
    )


def network_capacity(
    hot: Ramps,
    cold: Ramps,
    supply_celsius: ArrayLike,
    return_celsius: ArrayLike,
    approach_k: float,
) -> tuple[FloatArray, FloatArray]:
    """Largest loop duty on the source side and on the sink side, per level (kW).

    ``supply_celsius``/``return_celsius``: arrays of the same length (one entry per level).
    """
    t_s = np.atleast_1d(np.asarray(supply_celsius, dtype=np.float64))
    t_r = np.atleast_1d(np.asarray(return_celsius, dtype=np.float64))
    if t_s.shape != t_r.shape or np.any(t_s <= t_r):
        raise ValueError("supply must exceed return for every level")
    span = (t_s - t_r)[:, None]

    theta = hot.points[None, :] - approach_k
    phi = (t_s[:, None] - theta) / span
    inside = (theta >= t_r[:, None]) & (theta < t_s[:, None])
    value = hot(hot.points)[None, :]  # A_hot(theta + a) at theta = points - a
    ratio = np.where(inside, value / np.where(inside, phi, 1.0), np.inf)
    q_src = np.minimum(hot(t_r + approach_k), np.min(ratio, axis=1, initial=np.inf))

    theta_c = cold.points[None, :] + approach_k
    psi = (theta_c - t_r[:, None]) / span
    inside_c = (theta_c > t_r[:, None]) & (theta_c <= t_s[:, None])
    value_c = cold.below(cold.points)[None, :]
    ratio_c = np.where(inside_c, value_c / np.where(inside_c, psi, 1.0), np.inf)
    q_sink = np.minimum(cold.below(t_s - approach_k), np.min(ratio_c, axis=1, initial=np.inf))
    return np.maximum(q_src, 0.0), np.maximum(q_sink, 0.0)


def multi_loop_capacity(
    hot: Ramps,
    cold: Ramps,
    loops: Sequence[tuple[float, float]],
    approach_k: float,
) -> FloatArray:
    """Duties of several loops ``(supply, return)`` that maximise their sum (kW), by LP."""
    from scipy.optimize import linprog

    if not loops:
        return np.zeros(0)
    t_s = np.array([s for s, _ in loops], dtype=np.float64)
    t_r = np.array([r for _, r in loops], dtype=np.float64)
    if np.any(t_s <= t_r):
        raise ValueError("supply must exceed return for every loop")
    theta = np.unique(np.concatenate([hot.points - approach_k, cold.points + approach_k, t_s, t_r]))
    span = t_s - t_r
    phi = np.clip((t_s[None, :] - theta[:, None]) / span[None, :], 0.0, 1.0)
    psi = 1.0 - phi
    a_ub = np.vstack([phi, psi])
    b_ub = np.concatenate([hot(theta + approach_k), cold.below(theta - approach_k)])
    b_ub = np.maximum(b_ub, 0.0)
    result = linprog(-np.ones(len(loops)), A_ub=a_ub, b_ub=b_ub, bounds=(0.0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"loop LP failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def zonal_network_target(
    zones: Sequence[tuple[Ramps, Ramps]],
    supply_celsius: float,
    return_celsius: float,
    minimum_pinch_k: float,
) -> tuple[float, FloatArray]:
    r"""Minimum hot utility: direct transfer inside each zone, between zones only through the loop.

    Zone ``z`` (hot, cold ramps) keeps its own problem table with ``dT_min``; the loop
    ``(supply, return)`` enters it with approach ``dT_min/2`` on each side, i.e. as a stream at its
    real temperatures on the shifted scale (as the storage streams of
    :mod:`twinlink_ml.storage_timeslice.table`). Only the net loop import ``n_z`` of a zone matters
    (giving and taking at the same loop temperatures equals a direct match with ``dT_min``, which
    the zone has anyway); with ``phi(s) = clip((T_s - s)/(T_s - T_r), 0, 1)`` the LP is

    .. math:: \min \sum_z Q_{H,z} \quad\text{s.t.}\quad
              Q_{H,z} + n_z\,\phi(s) \ge -c_z(s)\ \ \forall z, s,\qquad \sum_z n_z = 0,\
              Q_{H,z} \ge 0,

    with ``c_z(s) = A_{hot,z}(s + dT/2) - A_{cold,z}(s - dT/2)`` the zone's cascade, on the union of
    its boundaries and the loop temperatures (exact: both sides are linear in between).

    Semantics: every zone sits on its own branch between supply and return (parallel connection,
    as consumers and feeders of a district network); a zone may reject heat it does not use --
    also imported loop heat -- to its cold utility (a return cooler that brings the water back to
    ``T_r``); fossil heat into the loop is allowed but never lowers the sum. One zone with every
    stream gives the direct target; one zone of all sources and one of all sinks lies between the
    direct target and the composite loop of :func:`network_capacity` (which allows the sources in
    series but no return cooler); see ``tests/test_site.py``.

    :returns: ``(total hot utility, net loop imports n_z)`` in kW.
    """
    from scipy.optimize import linprog

    if supply_celsius <= return_celsius:
        raise ValueError("supply must exceed return")
    half = minimum_pinch_k / 2.0
    span = supply_celsius - return_celsius
    n_z = len(zones)
    rows: list[FloatArray] = []
    rhs: list[float] = []
    for z, (hot, cold) in enumerate(zones):
        raw = np.concatenate(
            [hot.points - half, cold.points + half, [supply_celsius, return_celsius]]
        )
        s = np.unique(np.round(raw, _ROUND_DECIMALS))
        c = hot(s + half) - cold(s - half)
        phi = np.clip((supply_celsius - s) / span, 0.0, 1.0)
        keep = (phi > 0.0) & (phi < 1.0)
        for mask in (phi <= 0.0, phi >= 1.0):  # one binding row each where phi is constant
            if np.any(mask):
                j = np.flatnonzero(mask)[np.argmin(c[mask])]
                keep[j] = True
        for j in np.flatnonzero(keep):
            row = np.zeros(2 * n_z)
            row[z] = -1.0
            row[n_z + z] = -phi[j]
            rows.append(row)
            rhs.append(float(c[j]))
    objective = np.concatenate([np.ones(n_z), np.zeros(n_z)])
    a_eq = np.concatenate([np.zeros(n_z), np.ones(n_z)])[None, :]
    bounds = [(0.0, None)] * n_z + [(None, None)] * n_z
    result = linprog(
        objective,
        A_ub=np.array(rows),
        b_ub=np.array(rhs),
        A_eq=a_eq,
        b_eq=[0.0],
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"zonal network LP failed: {result.message}")
    return float(result.fun), np.asarray(result.x[n_z:], dtype=np.float64)


def heat_pump_limits(
    table: TableResult,
    evaporator_celsius: ArrayLike,
    condenser_celsius: ArrayLike,
    minimum_pinch_k: float,
) -> tuple[FloatArray, FloatArray]:
    """``(E, U)``: heat an evaporator can take, heat a condenser can deliver (kW), see docstring."""
    t_e = np.atleast_1d(np.asarray(evaporator_celsius, dtype=np.float64))
    t_c = np.atleast_1d(np.asarray(condenser_celsius, dtype=np.float64))
    if table.shifted.size == 0:
        return np.zeros_like(t_e), np.zeros_like(t_c)
    half = minimum_pinch_k / 2.0
    s_asc = table.shifted[::-1]
    g_asc = table.gcc_kw[::-1]
    s_e = t_e + half
    s_c = t_c - half
    prefix_min = np.minimum.accumulate(g_asc)
    suffix_min = np.minimum.accumulate(g_asc[::-1])[::-1]
    i_e = np.searchsorted(s_asc, s_e, side="right") - 1
    e_grid = np.where(i_e >= 0, prefix_min[np.clip(i_e, 0, None)], np.inf)
    evap = np.minimum(e_grid, np.interp(s_e, s_asc, g_asc))
    i_c = np.searchsorted(s_asc, s_c, side="left")
    u_grid = np.where(i_c < s_asc.size, suffix_min[np.clip(i_c, None, s_asc.size - 1)], np.inf)
    cond = np.minimum(u_grid, np.interp(s_c, s_asc, g_asc))
    return np.maximum(evap, 0.0), np.maximum(cond, 0.0)
