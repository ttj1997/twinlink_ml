r"""Waste heat delivered by the preheater (storage loop of Befund 91) at another air flow.

The preheater is a counter-current exchanger, water (preheater loop) against the process air,
effectiveness-NTU with one conductance (the exchanger class of :mod:`twinlink_ml.storage_a1.
model`). Its conductance is fitted per day from the day means (:func:`fit_preheater_conductance`;
Befund 91 fitted 17.3 / 18.9 kW/K on the minute data). Two readings of how the waste-heat side
answers a change of the air flow, both kept as variants (the storage dynamics are not modelled):

* :class:`FixedWaterInlet` -- water inlet temperature and loop flow stay as measured; the duty
  follows the air capacity rate (less air -> lower duty, the rest of the compressor heat would be
  discarded). Conservative for a lower air flow.
* :class:`DutyHeld` -- the compressor heat delivered on the measured day is still taken: the
  storage loop runs warmer until the preheater transfers the measured duty, up to a water inlet of
  ``max_water_celsius`` (placeholder); above it the duty is capped there. Optimistic.

In both, the waste heat can never exceed the heat the air still needs: callers cap it at the air
heating demand minus the fan heat (:mod:`twinlink_ml.plant_a_op.plant`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..storage_a1.config import ExchangerConfig, WaterConfig
from ..storage_a1.model import CounterflowExchanger, water_capacity_rate_kw_per_k

__all__ = [
    "DutyHeld",
    "FixedWaterInlet",
    "WasteHeatModel",
    "fit_preheater_conductance",
    "preheater_conductance_rate",
]

FloatArray = NDArray[np.float64]


def preheater_conductance_rate(
    ua_kw_per_k: float, c_water_kw_per_k: ArrayLike, c_air_kw_per_k: ArrayLike
) -> FloatArray:
    """``eps * C_min`` of the preheater (duty per kelvin of water inlet minus air inlet)."""
    hx = CounterflowExchanger(ExchangerConfig(ua_kw_per_k=ua_kw_per_k))
    return np.asarray(
        hx.conductance_kw_per_k(
            np.asarray(c_water_kw_per_k, dtype=np.float64),
            np.asarray(c_air_kw_per_k, dtype=np.float64),
        ),
        dtype=np.float64,
    )


def fit_preheater_conductance(
    duty_kw: float,
    c_water_kw_per_k: float,
    c_air_kw_per_k: float,
    water_in_celsius: float,
    air_in_celsius: float,
) -> float:
    """Conductance ``UA`` that reproduces the duty (bisection; the duty rises with ``UA``)."""
    target = duty_kw / (water_in_celsius - air_in_celsius)
    c_min = min(c_water_kw_per_k, c_air_kw_per_k)
    if not 0.0 < target < c_min:
        raise ValueError(f"duty {duty_kw} kW not reachable by a counter-current exchanger")
    lo, hi = 1e-6, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if float(preheater_conductance_rate(mid, c_water_kw_per_k, c_air_kw_per_k)) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


class WasteHeatModel(ABC):
    """Duty the preheater can deliver to the process air, kW."""

    @abstractmethod
    def duty_kw(self, c_air_kw_per_k: ArrayLike, ambient_celsius: ArrayLike) -> FloatArray:
        """Available waste-heat duty at the given air capacity rate and intake temperature."""


@dataclass(frozen=True, slots=True)
class FixedWaterInlet(WasteHeatModel):
    """Water inlet and loop flow as measured; duty from effectiveness-NTU."""

    ua_kw_per_k: float
    loop_flow_m3h: float
    water_in_celsius: float
    water: WaterConfig = field(default_factory=WaterConfig)

    def duty_kw(self, c_air_kw_per_k: ArrayLike, ambient_celsius: ArrayLike) -> FloatArray:
        c_w = water_capacity_rate_kw_per_k(self.loop_flow_m3h, self.water)
        k = preheater_conductance_rate(self.ua_kw_per_k, c_w, c_air_kw_per_k)
        dt = self.water_in_celsius - np.asarray(ambient_celsius, dtype=np.float64)
        return np.asarray(np.maximum(k * dt, 0.0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DutyHeld(WasteHeatModel):
    """Measured duty kept, capped where the water inlet would exceed ``max_water_celsius``."""

    duty_measured_kw: float
    ua_kw_per_k: float
    loop_flow_m3h: float
    max_water_celsius: float
    water: WaterConfig = field(default_factory=WaterConfig)

    def duty_kw(self, c_air_kw_per_k: ArrayLike, ambient_celsius: ArrayLike) -> FloatArray:
        c_w = water_capacity_rate_kw_per_k(self.loop_flow_m3h, self.water)
        k = preheater_conductance_rate(self.ua_kw_per_k, c_w, c_air_kw_per_k)
        cap = k * (self.max_water_celsius - np.asarray(ambient_celsius, dtype=np.float64))
        return np.asarray(np.clip(cap, 0.0, self.duty_measured_kw), dtype=np.float64)
