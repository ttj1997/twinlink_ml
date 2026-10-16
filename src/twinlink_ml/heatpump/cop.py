r"""COP of the heat pump and the admissible design grid (evaporating x condensing temperature).

``COP = min(quality * T_k / (T_k - T_e), cop_max)`` (kelvin), admissible when
``min_lift <= T_k - T_e <= max_lift`` and ``T_k <= max_condensing``. The COP is the HEATING COP:
condenser heat per unit of electric work, ``Q_k = Q_e + W``.

Pseudo-streams in the problem table (shifted temperatures, ``dT_min/2`` convention of
:mod:`twinlink_ml.storage_timeslice.table`): the evaporator at ``T*_e = T_e + a - dT_min/2`` (a
cold stream: it absorbs heat only from hot parts with real temperature above ``T_e + a``), the
condenser at ``T*_k = T_k - a + dT_min/2`` (a hot stream: it serves cold parts only below
``T_k - a``), ``a`` the approach of :class:`~twinlink_ml.heatpump.config.HeatPumpSpec`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import CONDENSING_GRID_CELSIUS, EVAPORATING_GRID_CELSIUS, HeatPumpSpec

__all__ = [
    "KELVIN",
    "DesignPair",
    "carnot_cop",
    "design_pairs",
    "heat_pump_cop",
    "pseudo_stream_points",
]

KELVIN = 273.15


def carnot_cop(evaporating_celsius: float, condensing_celsius: float) -> float:
    """Heating COP of the Carnot cycle between the two refrigerant temperatures."""
    lift = condensing_celsius - evaporating_celsius
    if lift <= 0.0:
        raise ValueError(f"condensing must exceed evaporating temperature, lift {lift}")
    return (condensing_celsius + KELVIN) / lift


def heat_pump_cop(
    evaporating_celsius: float, condensing_celsius: float, spec: HeatPumpSpec
) -> float:
    """``min(quality * COP_Carnot, cop_max)`` (no admissibility check, see :func:`design_pairs`)."""
    return min(spec.quality * carnot_cop(evaporating_celsius, condensing_celsius), spec.cop_max)


@dataclass(frozen=True, slots=True)
class DesignPair:
    """One admissible design: refrigerant temperatures, COP and the pseudo-stream temperatures."""

    evaporating_celsius: float
    condensing_celsius: float
    cop: float
    evaporator_shifted: float
    condenser_shifted: float

    @property
    def lift_k(self) -> float:
        return self.condensing_celsius - self.evaporating_celsius

    @property
    def label(self) -> str:
        return f"{self.evaporating_celsius:g}->{self.condensing_celsius:g}"


def design_pairs(
    spec: HeatPumpSpec,
    *,
    minimum_pinch_k: float,
    evaporating_celsius: Sequence[float] = EVAPORATING_GRID_CELSIUS,
    condensing_celsius: Sequence[float] = CONDENSING_GRID_CELSIUS,
) -> list[DesignPair]:
    """All admissible pairs of the grid, ordered by (evaporating, condensing)."""
    a = spec.approach(minimum_pinch_k)
    half = minimum_pinch_k / 2.0
    out: list[DesignPair] = []
    for te in evaporating_celsius:
        for tk in condensing_celsius:
            lift = tk - te
            if lift < spec.min_lift_k - 1e-9 or lift > spec.max_lift_k + 1e-9:
                continue
            if tk > spec.max_condensing_celsius + 1e-9:
                continue
            out.append(
                DesignPair(
                    evaporating_celsius=float(te),
                    condensing_celsius=float(tk),
                    cop=heat_pump_cop(float(te), float(tk), spec),
                    evaporator_shifted=float(te) + a - half,
                    condenser_shifted=float(tk) - a + half,
                )
            )
    return out


def pseudo_stream_points(pairs: Sequence[DesignPair]) -> np.ndarray:
    """All pseudo-stream temperatures of ``pairs`` (extra boundaries of the problem table)."""
    pts = [p.evaporator_shifted for p in pairs] + [p.condenser_shifted for p in pairs]
    return np.unique(np.asarray(pts, dtype=np.float64))
