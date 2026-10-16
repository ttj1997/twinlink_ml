r"""Operating-mode profiles of the scenarios (placeholders; plant A runs 24/7 in every mode).

Clock: the measured day is repeated; step ``t`` has hour of day ``h`` in [0, 24) and day index
``d`` (0 = Monday). ``weekly`` marks Saturday and Sunday (``d`` = 5, 6).

======== ================================ ======================================== =========
mode     compressors (both plants)        lines C1 / C2                            water
======== ================================ ======================================== =========
flat     1                                1 / 1                                    1
shift    1 in the shift window,           C1 in the shift window,                  window
         ``low_fraction`` otherwise       C2 from window start to ``c2_end_h``
weekly   1 Mon-Fri, ``low_fraction``      1 / 1 Mon-Fri, 0 / 0 Sat-Sun             Mon-Fri
         Sat-Sun
======== ================================ ======================================== =========
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import Mode, Placeholders

__all__ = [
    "Clock",
    "compressor_profile",
    "line_c_profile",
    "night_profile",
    "water_profile",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Clock:
    """Hour of day and day index of every step of the horizon."""

    hour: FloatArray
    day: NDArray[np.int64]

    @staticmethod
    def of(n_days: int, steps_per_day: int = 96) -> Clock:
        t = np.arange(n_days * steps_per_day)
        return Clock(hour=(t % steps_per_day) * 24.0 / steps_per_day, day=t // steps_per_day)

    @property
    def weekend(self) -> NDArray[np.bool_]:
        return np.asarray((self.day % 7) >= 5)


def _window(hour: FloatArray, start: float, end: float) -> NDArray[np.bool_]:
    return np.asarray((hour >= start) & (hour < end))


def compressor_profile(mode: Mode, clock: Clock, ph: Placeholders) -> FloatArray:
    """Relative compressor heat (1 = nominal level)."""
    ones = np.ones_like(clock.hour)
    if mode == "flat":
        return ones
    if mode == "shift":
        return np.where(_window(clock.hour, ph.shift_start_h, ph.shift_end_h), 1.0, ph.low_fraction)
    return np.where(clock.weekend, ph.low_fraction, 1.0)


def line_c_profile(mode: Mode, clock: Clock, ph: Placeholders, line: int) -> FloatArray:
    """On/off (1/0) of line C1 (``line=1``) or C2 (``line=2``)."""
    if line not in (1, 2):
        raise ValueError("line must be 1 or 2")
    if mode == "flat":
        return np.ones_like(clock.hour)
    if mode == "shift":
        end = ph.shift_end_h if line == 1 else ph.c2_end_h
        return _window(clock.hour, ph.shift_start_h, end).astype(np.float64)
    return np.where(clock.weekend, 0.0, 1.0)


def water_profile(mode: Mode, clock: Clock, ph: Placeholders) -> FloatArray:
    """Relative process-water demand."""
    if mode == "flat":
        return np.ones_like(clock.hour)
    if mode == "shift":
        return _window(clock.hour, ph.shift_start_h, ph.shift_end_h).astype(np.float64)
    return np.where(clock.weekend, 0.0, 1.0)


def night_profile(clock: Clock, day_factor: float) -> FloatArray:
    """Hall-heating time factor: 1 from 18:00 to 06:00, ``day_factor`` otherwise."""
    night = (clock.hour >= 18.0) | (clock.hour < 6.0)
    return np.where(night, 1.0, day_factor)
