r"""Power of the two fans of plant A (process-air fan, exhaust fan) against the air flow.

Model: ``P = P_meas (V / V_meas)^n`` per fan, anchored at the measured day mean of the product
(so the baseline is reproduced exactly). The exponent is the open quantity:

* ``affinity``: n = 3 for both fans -- speed-controlled fan on a system curve through the origin
  (pure friction, no static pressure); the largest electrical saving a lower flow can give.
  A PLACEHOLDER variant, not fitted.
* ``data_fit``: n from the two measured days (day means; :func:`two_point_exponent`). The data
  hold almost no variation of the air flow within a day (coefficient of variation below 0.3 %),
  so the only information is the difference between the two days, and that difference is
  confounded with the product (exhaust temperature and humidity, filter state). The exhaust fan's
  two-point exponent lies far outside any physical range (it changes by a factor of about two for
  an air-flow change of 8 %), so it is NOT used: ``data_fit`` applies the process fan's exponent
  to both fans and the study reports the exhaust fan's own value next to it.
* ``constant`` (n = 0) and ``quadratic`` (n = 2) bracket the sensitivity.

:func:`within_day_exponent` fits ``ln P`` on ``ln V`` within one day (15-minute blocks) with its
standard error, as the evidence that the within-day data cannot identify the exponent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import FanLawName

__all__ = [
    "PHYSICAL_EXPONENT_RANGE",
    "FanLaw",
    "FanModel",
    "PowerLawFan",
    "fan_law",
    "two_point_exponent",
    "within_day_exponent",
]

FloatArray = NDArray[np.float64]

PHYSICAL_EXPONENT_RANGE: tuple[float, float] = (0.0, 3.0)
"""Range of the flow exponent a fan on a fixed system can show (0: constant power, 3: affinity)."""


class FanModel(ABC):
    """Electric power of one fan as a function of the air flow relative to the measured day."""

    @abstractmethod
    def power_kw(self, air_factor: ArrayLike) -> FloatArray:
        """Electric power at ``V / V_meas = air_factor``, kW."""


@dataclass(frozen=True, slots=True)
class PowerLawFan(FanModel):
    """``P = anchor_kw * air_factor ** exponent``."""

    anchor_kw: float
    exponent: float

    def __post_init__(self) -> None:
        if self.anchor_kw < 0.0:
            raise ValueError("anchor power must be >= 0")

    def power_kw(self, air_factor: ArrayLike) -> FloatArray:
        f = np.asarray(air_factor, dtype=np.float64)
        return np.asarray(self.anchor_kw * f**self.exponent, dtype=np.float64)


def two_point_exponent(p1: float, v1: float, p2: float, v2: float) -> float:
    """Exponent ``n`` of ``P ~ V^n`` through two (flow, power) points."""
    return float(np.log(p1 / p2) / np.log(v1 / v2))


def within_day_exponent(power_kw: ArrayLike, flow: ArrayLike) -> dict[str, float]:
    """OLS of ``ln P`` on ``ln V`` within one day: slope, its standard error, flow CV."""
    x = np.log(np.asarray(flow, dtype=np.float64))
    y = np.log(np.asarray(power_kw, dtype=np.float64))
    xc = x - x.mean()
    sxx = float(np.sum(xc**2))
    slope = float(np.sum(xc * (y - y.mean())) / sxx)
    resid = y - y.mean() - slope * xc
    dof = max(len(x) - 2, 1)
    se = float(np.sqrt(np.sum(resid**2) / dof / sxx))
    v = np.asarray(flow, dtype=np.float64)
    return {
        "slope": slope,
        "standard_error": se,
        "flow_cv": float(np.std(v) / np.mean(v)),
        "flow_range_fraction": float((v.max() - v.min()) / np.mean(v)),
        "n_blocks": float(len(x)),
    }


@dataclass(frozen=True, slots=True)
class FanLaw:
    """Exponents of the process-air fan and the exhaust fan under one named law."""

    name: FanLawName
    exponent_process: float
    exponent_exhaust: float


def fan_law(name: FanLawName, data_exponent_process: float) -> FanLaw:
    """Resolve a named fan law; ``data_fit`` needs the fitted process-fan exponent."""
    if name == "data_fit":
        lo, hi = PHYSICAL_EXPONENT_RANGE
        if not lo <= data_exponent_process <= hi:
            raise ValueError(
                f"process-fan exponent {data_exponent_process} outside {PHYSICAL_EXPONENT_RANGE}"
            )
        return FanLaw(name, data_exponent_process, data_exponent_process)
    fixed = {"affinity": 3.0, "constant": 0.0, "quadratic": 2.0}
    if name not in fixed:
        raise ValueError(f"unknown fan law {name!r}")
    return FanLaw(name, fixed[name], fixed[name])
