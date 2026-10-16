r"""Problem table per time step, vectorised over the steps, with the cascade kept for storage.

Same algorithm as :func:`twinlink_ml.heatnetwork.pinch.compute_pinch_target` (shifted
temperatures, interval boundaries, cascade from the top with start value 0, hot utility = largest
negative cascade value), but

* it returns the cascade ITSELF (heat flowing down through every boundary, before any utility),
  because the storage programme needs it, and
* it adds extra boundaries (the temperatures of the storage streams), because a point source or
  sink is only exact when its temperature is a boundary of the table.

Storage streams as isothermal streams
-------------------------------------
A storage at level ``T_S`` with approach ``a`` on both sides acts in the table as

* charging: an isothermal COLD stream at the shifted temperature ``T*_c = T_S + a - dT_min/2``
  (it takes heat only from hot parts with real temperature above ``T_S + a``);
* discharging: an isothermal HOT stream at ``T*_d = T_S - a + dT_min/2`` (it serves cold parts
  only below ``T_S - a``).

With charge duty ``q_c`` and discharge duty ``q_d`` the cascade value through boundary ``b``
changes by ``-q_c [b <= T*_c] + q_d [b < T*_d]`` (strict for the discharge: its heat enters only
below its own temperature). Because the stream temperatures are given, the
hot utility of a step is

.. math:: U = \max\bigl(0,\ \max_g (M_g + a_g^T q_c - b_g^T q_d)\bigr),

with one group ``g`` per region between the sorted storage temperatures and
``M_g = \max_{b \in g} (-R_b)``: only ``2K + 1`` linear pieces for ``K`` levels instead of one per
boundary. :func:`level_groups` computes ``M_g`` and the 0/1 coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Cascade",
    "LevelGroups",
    "cascade_profile",
    "direct_hot_utility_kw",
    "level_groups",
    "storage_stream_temperatures",
]

FloatArray = NDArray[np.float64]

_TOLERANCE_K = 1e-7
"""Comparison tolerance of shifted temperatures (after rounding to 9 decimals), as in ``pinch``."""
_ROUND_DECIMALS = 9
_PAD_TEMPERATURE = -1.0e9
_NO_GROUP = -np.inf


@dataclass(frozen=True, slots=True)
class Cascade:
    """Cascade of every step on its own (padded) boundary list.

    :ivar boundaries: shifted temperatures, descending per row, padded with a very low value.
    :ivar residual_kw: heat flowing down through each boundary (start value 0 at the top);
        ``+inf`` at padded entries so that ``-residual`` never wins a maximum.
    """

    boundaries: FloatArray
    residual_kw: FloatArray

    @property
    def n_steps(self) -> int:
        return int(self.boundaries.shape[0])


def storage_stream_temperatures(
    levels_celsius: FloatArray, *, minimum_pinch_k: float, approach_k: float
) -> tuple[FloatArray, FloatArray]:
    """Shifted temperatures ``(T*_c, T*_d)`` of the charging and discharging stream per level."""
    if approach_k < minimum_pinch_k / 2.0:
        raise ValueError(
            "approach_k must be at least dT_min/2 (charging level must not lie below the "
            f"discharging level), is {approach_k} for dT_min {minimum_pinch_k}"
        )
    levels = np.asarray(levels_celsius, dtype=np.float64)
    half = minimum_pinch_k / 2.0
    return levels + approach_k - half, levels - approach_k + half


def _step_cascade(
    hot: FloatArray, cold: FloatArray, minimum_pinch_k: float, extra: FloatArray
) -> tuple[FloatArray, FloatArray]:
    half = minimum_pinch_k / 2.0
    hot = hot[hot[:, 2] > 0.0]
    cold = cold[cold[:, 2] > 0.0]
    hot_high, hot_low, hot_cp = hot[:, 0] - half, hot[:, 1] - half, hot[:, 2]
    cold_high, cold_low, cold_cp = cold[:, 0] + half, cold[:, 1] + half, cold[:, 2]
    raw = np.concatenate([hot_high, hot_low, cold_high, cold_low, extra])
    bounds = np.unique(np.round(raw, _ROUND_DECIMALS))[::-1]
    upper, lower = bounds[:-1, None], bounds[1:, None]

    def active(high: FloatArray, low: FloatArray) -> NDArray[np.bool_]:
        return (high[None, :] >= upper - _TOLERANCE_K) & (low[None, :] <= lower + _TOLERANCE_K)

    cp_hot = active(hot_high, hot_low) @ hot_cp
    cp_cold = active(cold_high, cold_low) @ cold_cp
    net = (cp_hot - cp_cold) * (upper[:, 0] - lower[:, 0])
    return bounds, np.concatenate([[0.0], np.cumsum(net)])


def cascade_profile(
    hot: FloatArray, cold: FloatArray, *, minimum_pinch_k: float, extra_points: FloatArray
) -> Cascade:
    """Cascades of all steps.

    ``hot``/``cold``: ``(steps, segments, 3)`` = high, low, CP in real temperatures and kW/K;
    segments with CP 0 are padding and ignored.
    """
    if minimum_pinch_k < 0.0:
        raise ValueError(f"minimum_pinch_k must not be negative, is {minimum_pinch_k}")
    extra = np.round(np.asarray(extra_points, dtype=np.float64), _ROUND_DECIMALS)
    n_steps = hot.shape[0]
    rows = [_step_cascade(hot[i], cold[i], minimum_pinch_k, extra) for i in range(n_steps)]
    width = max((len(b) for b, _ in rows), default=1)
    boundaries = np.full((n_steps, width), _PAD_TEMPERATURE)
    residual = np.full((n_steps, width), np.inf)
    for i, (b, r) in enumerate(rows):
        boundaries[i, : len(b)] = b
        residual[i, : len(r)] = r
    return Cascade(boundaries=boundaries, residual_kw=residual)


def direct_hot_utility_kw(cascade: Cascade) -> FloatArray:
    """Minimum hot utility per step (kW): ``max(0, -min cascade)`` -- the pinch target."""
    return np.maximum(0.0, -np.min(cascade.residual_kw, axis=1))


@dataclass(frozen=True, slots=True)
class LevelGroups:
    """Linear pieces of the hot utility for a set of storage levels.

    :ivar offset_kw: ``(steps, groups)`` ``M_g`` (``-inf`` where a group has no boundary).
    :ivar charge: ``(groups, levels)`` 0/1, coefficient of ``q_c`` of a level in the piece.
    :ivar discharge: ``(groups, levels)`` 0/1, coefficient of ``q_d`` (enters with a minus sign).
    """

    offset_kw: FloatArray
    charge: FloatArray
    discharge: FloatArray

    def utility_kw(self, q_charge: FloatArray, q_discharge: FloatArray) -> FloatArray:
        """``U`` per step for charge/discharge duties of shape ``(steps, levels)``."""
        value = self.offset_kw + q_charge @ self.charge.T - q_discharge @ self.discharge.T
        return np.maximum(0.0, np.max(value, axis=1))


def level_groups(
    cascade: Cascade, charge_temperatures: FloatArray, discharge_temperatures: FloatArray
) -> LevelGroups:
    """Pieces of ``U`` for levels with the given shifted storage-stream temperatures.

    Asymmetric at the stream's own boundary, on purpose: a charging (cold) stream absorbs AT its
    temperature, so the cascade value of that boundary already contains the absorption
    (``b <= T*_c``); a discharging (hot) stream injects heat only BELOW its temperature, so the
    value arriving at that boundary from above is unchanged (``b < T*_d``). The latter matters:
    heat of the storage cannot flow upwards.
    """
    tc = np.round(np.atleast_1d(np.asarray(charge_temperatures, dtype=np.float64)), _ROUND_DECIMALS)
    td = np.round(
        np.atleast_1d(np.asarray(discharge_temperatures, dtype=np.float64)), _ROUND_DECIMALS
    )
    cut_c = tc + _TOLERANCE_K  # inclusive: b <= cut
    cut_d = td - _TOLERANCE_K  # strict at the stream temperature: b <= cut
    points = np.sort(np.concatenate([cut_c, cut_d]))[::-1]
    n_groups = len(points) + 1
    region = np.sum(cascade.boundaries[:, :, None] <= points[None, None, :], axis=2)
    neg_r = -cascade.residual_kw
    offset = np.full((cascade.n_steps, n_groups), _NO_GROUP)
    for g in range(n_groups):
        offset[:, g] = np.max(np.where(region == g, neg_r, _NO_GROUP), axis=1)
    charge = np.zeros((n_groups, len(tc)))
    discharge = np.zeros((n_groups, len(td)))
    for g in range(1, n_groups):
        charge[g] = (cut_c >= points[g - 1]).astype(np.float64)
        discharge[g] = (cut_d >= points[g - 1]).astype(np.float64)
    return LevelGroups(offset_kw=offset, charge=charge, discharge=discharge)
