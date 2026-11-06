"""Inputs, box, named regions and designs of the case-ranking study.

Eight inputs (decision of the project lead, F1 (a)): shared by both dryers the ambient temperature
and the relative humidity of the outside air (the same outside air feeds both); per dryer the
supply-air temperature, the spray load and the solid fraction. The absolute humidity of the supply
air follows from the relative humidity at the ambient temperature (so that no supersaturated
corners arise); both dryers and both sinks get it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from twinlink_ml.flowsheet.validators import MOLAR_MASS_RATIO

__all__ = [
    "ATM_PA",
    "INPUT_NAMES",
    "Box",
    "Interval",
    "OperatingPoint",
    "ambient_humidity_kg_per_kg",
    "ambient_relative_humidity",
    "ambient_saturation_pressure_pa",
    "sobol_points",
]

ATM_PA = 101_325.0
"""Ambient pressure of the workbook (1 atm, docs/11 section 13.1)."""

INPUT_NAMES: tuple[str, ...] = (
    "ambient_celsius",
    "ambient_relative_humidity",
    "inlet_1_celsius",
    "inlet_2_celsius",
    "spray_load_1_kg_per_kg",
    "spray_load_2_kg_per_kg",
    "solid_fraction_1",
    "solid_fraction_2",
)
"""Order of the columns of every design (8 dimensions)."""


def ambient_saturation_pressure_pa(temperature_celsius: float) -> float:
    """Saturation pressure over water, Buck (1996), as :func:`twinlink_ml.simulation.thermo`.

    Used for the outside air only: the box reaches -10 degC, below the validity range (0 degC) of
    the IAPWS-IF97 function that checks the exhausts; over water also below 0 degC (the usual
    convention for the relative humidity of air).
    """
    t = temperature_celsius
    return 611.21 * float(np.exp((18.678 - t / 234.5) * t / (257.14 + t)))


def ambient_humidity_kg_per_kg(
    temperature_celsius: float, relative_humidity: float, pressure_pa: float = ATM_PA
) -> float:
    """Absolute humidity (kg water per kg dry air) of air at ``relative_humidity``."""
    p_v = relative_humidity * ambient_saturation_pressure_pa(temperature_celsius)
    return MOLAR_MASS_RATIO * p_v / (pressure_pa - p_v)


def ambient_relative_humidity(
    temperature_celsius: float, humidity_kg_per_kg: float, pressure_pa: float = ATM_PA
) -> float:
    """Inverse of :func:`ambient_humidity_kg_per_kg`."""
    p_v = humidity_kg_per_kg * pressure_pa / (MOLAR_MASS_RATIO + humidity_kg_per_kg)
    return p_v / ambient_saturation_pressure_pa(temperature_celsius)


@dataclass(frozen=True)
class OperatingPoint:
    """One point of the input space (degC, relative humidity as a fraction, kg/kg)."""

    ambient_celsius: float
    ambient_relative_humidity: float
    inlet_1_celsius: float
    inlet_2_celsius: float
    spray_load_1_kg_per_kg: float
    spray_load_2_kg_per_kg: float
    solid_fraction_1: float
    solid_fraction_2: float

    @property
    def humidity_kg_per_kg(self) -> float:
        """Absolute humidity of the outside air = supply air of both dryers and both sinks."""
        return ambient_humidity_kg_per_kg(self.ambient_celsius, self.ambient_relative_humidity)

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in INPUT_NAMES], dtype=np.float64)

    @classmethod
    def from_array(cls, x: Sequence[float] | np.ndarray) -> OperatingPoint:
        if len(x) != len(INPUT_NAMES):
            raise ValueError(f"expected {len(INPUT_NAMES)} values, got {len(x)}")
        return cls(**{n: float(v) for n, v in zip(INPUT_NAMES, x, strict=True)})

    def with_(self, **values: float) -> OperatingPoint:
        d = {n: getattr(self, n) for n in INPUT_NAMES}
        unknown = set(values) - set(d)
        if unknown:
            raise KeyError(sorted(unknown))
        d.update({k: float(v) for k, v in values.items()})
        return OperatingPoint(**d)


@dataclass(frozen=True)
class Interval:
    """Closed interval ``[low, high]``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"empty interval [{self.low}, {self.high}]")

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def intersect(self, other: Interval) -> Interval:
        return Interval(max(self.low, other.low), min(self.high, other.high))


@dataclass(frozen=True)
class Box:
    """Axis-aligned closed box over :data:`INPUT_NAMES`."""

    intervals: tuple[Interval, ...]

    def __post_init__(self) -> None:
        if len(self.intervals) != len(INPUT_NAMES):
            raise ValueError("one interval per input")

    @classmethod
    def from_mapping(cls, ranges: Mapping[str, Sequence[float]]) -> Box:
        missing = set(INPUT_NAMES) - set(ranges)
        extra = set(ranges) - set(INPUT_NAMES)
        if missing or extra:
            raise KeyError(f"missing {sorted(missing)}, unknown {sorted(extra)}")
        return cls(tuple(Interval(float(ranges[n][0]), float(ranges[n][1])) for n in INPUT_NAMES))

    def interval(self, name: str) -> Interval:
        return self.intervals[INPUT_NAMES.index(name)]

    @property
    def low(self) -> np.ndarray:
        return np.array([i.low for i in self.intervals])

    @property
    def high(self) -> np.ndarray:
        return np.array([i.high for i in self.intervals])

    def contains(self, point: OperatingPoint) -> bool:
        return all(i.contains(v) for i, v in zip(self.intervals, point.as_array(), strict=True))

    def replace(self, **ranges: Sequence[float]) -> Box:
        out = list(self.intervals)
        for name, (lo, hi) in ranges.items():
            out[INPUT_NAMES.index(name)] = Interval(float(lo), float(hi))
        return Box(tuple(out))

    def neighbourhood(self, centre: OperatingPoint, fraction: float) -> Box:
        """``centre +- fraction * width`` per input, intersected with this box."""
        out = []
        for i, v in zip(self.intervals, centre.as_array(), strict=True):
            half = fraction * i.width
            out.append(Interval(v - half, v + half).intersect(i))
        return Box(tuple(out))

    def as_dict(self) -> dict[str, list[float]]:
        return {n: [i.low, i.high] for n, i in zip(INPUT_NAMES, self.intervals, strict=True)}


def sobol_points(box: Box, count: int, seed: int) -> np.ndarray:
    """First ``count`` points of a scrambled Sobol stream in ``box`` (shape ``(count, 8)``).

    Prefixes of one stream are nested (the convergence over ``N`` uses this); ``count`` should be
    a power of two for the balance properties of the sequence.
    """
    sampler = qmc.Sobol(len(INPUT_NAMES), scramble=True, seed=seed)
    return qmc.scale(sampler.random(count), box.low, box.high)
