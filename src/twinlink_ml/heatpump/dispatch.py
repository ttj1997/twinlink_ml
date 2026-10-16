r"""Stage A: hot utility as a function of the heat-pump output, and the best output per step.

With the evaporator and condenser as isothermal pseudo-streams
(:func:`~twinlink_ml.storage_timeslice.table.level_groups` with one "charging" stream at
``T*_e`` and one "discharging" stream at ``T*_k``), the hot utility of a step is

.. math:: U(Q) = \max\bigl(0,\ \max_g (M_g + s_g Q)\bigr), \qquad
          s_g = a_g (1 - 1/\mathrm{COP}) - b_g,

``Q`` = condenser heat, ``Q_e = Q (1 - 1/COP)`` = evaporator heat, ``W = Q / COP`` = electric
work, ``a_g, b_g`` in {0, 1}. ``U`` is convex and piecewise linear in ``Q``; slope -1 where the
pump works across the pinch (each kWh of condenser heat replaces a kWh of hot utility), slope
``-1/COP`` where its evaporator draws on heat that is itself needed (above the pinch: only the
work is gained), slope 0 where its heat cannot be used.

Per step and criterion the operator chooses ``0 <= Q <= Q_max`` to minimise
``c_U U(Q) + c_el Q / COP`` (heat value ``c_U`` per kWh of gas heat, electricity value ``c_el``);
the minimum of a convex piecewise-linear function on an interval lies at an end or at a kink, so
the candidates ``{0, Q_max, kinks}`` are evaluated exactly (no solver, vectorised).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from ..storage_timeslice.table import Cascade, level_groups

__all__ = ["HeatPumpLines", "StepDispatch", "dispatch_steps", "heat_pump_lines"]

FloatArray = NDArray[np.float64]
_BIG_NEGATIVE = -1.0e15


@dataclass(frozen=True, slots=True)
class HeatPumpLines:
    """``U(Q) = max(0, max_g(offset[t, g] + slope[g] Q))`` for one design pair.

    :ivar offset_kw: ``(steps, groups)``, ``-inf`` where a group has no boundary.
    :ivar slope: ``(groups,)``.
    :ivar cop: heating COP of the pair.
    """

    offset_kw: FloatArray
    slope: FloatArray
    cop: float

    def utility_kw(self, q_kw: FloatArray) -> FloatArray:
        """``U`` for condenser heat ``q_kw`` of shape ``(..., steps)``."""
        q = np.asarray(q_kw, dtype=np.float64)
        value = self.offset_kw + q[..., None] * self.slope
        return np.maximum(0.0, np.max(value, axis=-1))

    @property
    def direct_kw(self) -> FloatArray:
        """``U(0)``, the direct hot utility without the heat pump."""
        return np.maximum(0.0, np.max(self.offset_kw, axis=1))


def heat_pump_lines(
    cascade: Cascade, evaporator_shifted: float, condenser_shifted: float, cop: float
) -> HeatPumpLines:
    """Pieces of ``U(Q)`` for one pair; the boundaries must contain both pseudo-temperatures."""
    if cop <= 1.0:
        raise ValueError(f"cop must exceed 1, is {cop}")
    groups = level_groups(cascade, np.array([evaporator_shifted]), np.array([condenser_shifted]))
    slope = groups.charge[:, 0] * (1.0 - 1.0 / cop) - groups.discharge[:, 0]
    return HeatPumpLines(offset_kw=groups.offset_kw, slope=slope, cop=float(cop))


@dataclass(frozen=True, slots=True)
class StepDispatch:
    """Best output per step; arrays broadcast over leading dimensions ``(..., steps)``."""

    heat_kw: FloatArray
    utility_kw: FloatArray
    work_kw: FloatArray


def dispatch_steps(
    lines: HeatPumpLines,
    q_max_kw: FloatArray | float,
    heat_value: FloatArray | float,
    electricity_value: FloatArray | float,
) -> StepDispatch:
    """Optimal condenser heat per step for every combination of the leading dimensions.

    :param q_max_kw, heat_value, electricity_value: broadcastable to ``(..., steps)``
        (e.g. ``(sizes, 1, 1)``, ``(1, criteria, 1)``, ``(1, criteria, steps)``).
    Ties are broken towards the SMALLER output (candidates are scanned in ascending order).
    """
    offset = np.where(np.isfinite(lines.offset_kw), lines.offset_kw, _BIG_NEGATIVE)
    n_steps = offset.shape[0]
    m = np.concatenate([np.zeros((n_steps, 1)), offset], axis=1)  # line 0 = the zero floor
    s = np.concatenate([[0.0], lines.slope])
    kinks = []
    for i, j in combinations(range(len(s)), 2):
        if abs(s[i] - s[j]) > 1e-12:
            kinks.append((m[:, j] - m[:, i]) / (s[i] - s[j]))
    q_max = np.asarray(q_max_kw, dtype=np.float64)
    c_u = np.asarray(heat_value, dtype=np.float64)
    c_el = np.asarray(electricity_value, dtype=np.float64)
    shape = np.broadcast_shapes(q_max.shape, c_u.shape, c_el.shape, (n_steps,))
    q_max_b = np.broadcast_to(q_max, shape)
    kink_arr = np.stack(kinks, axis=-1) if kinks else np.zeros((n_steps, 0))  # (steps, K)
    cand = np.concatenate(
        [
            np.zeros((*shape, 1)),
            np.clip(
                np.broadcast_to(kink_arr, shape + kink_arr.shape[-1:]), 0.0, q_max_b[..., None]
            ),
            q_max_b[..., None],
        ],
        axis=-1,
    )
    cand = np.sort(cand, axis=-1)
    utility = np.maximum(0.0, np.max(m[:, None, :] + cand[..., None] * s, axis=-1))
    cost = (
        np.broadcast_to(c_u, shape)[..., None] * utility
        + np.broadcast_to(c_el, shape)[..., None] * cand / lines.cop
    )
    best = np.argmin(cost, axis=-1)
    q = np.take_along_axis(cand, best[..., None], axis=-1)[..., 0]
    u = np.take_along_axis(utility, best[..., None], axis=-1)[..., 0]
    return StepDispatch(heat_kw=q, utility_kw=u, work_kw=q / lines.cop)
