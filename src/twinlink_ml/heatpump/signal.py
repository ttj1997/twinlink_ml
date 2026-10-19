r"""Time profile of the electricity value (price or CO2 factor), Stage B. PLACEHOLDER shape.

``shape(h)`` is a smooth daily pattern built from three periodic bumps -- a moderate morning peak
(08:00), a deep solar trough at midday (13:00) and an evening peak (19:30) -- normalised to mean 0
and minimum -1. It stands for the typical German day-ahead price and grid-CO2 pattern; it is not
fitted to data (placeholder). Price and CO2 use the same shape (in reality correlated, not equal).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..storage_timeslice.profiles import Clock
from .config import Signal

__all__ = ["daily_shape", "signal_series", "trailing_mean"]

FloatArray = NDArray[np.float64]

_BUMPS = ((8.0, 2.0, 0.5), (13.0, 3.0, -1.0), (19.5, 2.5, 0.8))
"""(centre h, width h, height) of the bumps of the raw shape."""


def _raw(hour: ArrayLike) -> FloatArray:
    h = np.asarray(hour, dtype=np.float64)
    out = np.zeros_like(h)
    for centre, width, height in _BUMPS:
        d = (h - centre + 12.0) % 24.0 - 12.0  # periodic distance
        out += height * np.exp(-0.5 * (d / width) ** 2)
    return out


_STEP_CENTRES = (np.arange(96) + 0.5) * 0.25
_RAW_STEPS = _raw(_STEP_CENTRES)
_MEAN = float(np.mean(_RAW_STEPS))
_DEPTH = float(_MEAN - np.min(_RAW_STEPS))


def daily_shape(hour: ArrayLike) -> FloatArray:
    """Shape of mean 0 and minimum -1 over the 96 centres of the 15-minute steps of a day."""
    return np.asarray((_raw(hour) - _MEAN) / _DEPTH, dtype=np.float64)


def signal_series(mean: float, signal: Signal, clock: Clock) -> FloatArray:
    """``mean * (1 + amplitude * shape(h)) * week(d)`` on the steps of ``clock``.

    The value of a 15-minute step is the shape at the step centre (exact mean over a day).
    ``week``: weekdays ``f``, Saturday/Sunday ``f (1 - weekend_reduction)`` with ``5 f + 2 f (1 -
    r) = 7`` (mean 1 over a week; a one-day horizon is a weekday, factor 1).
    """
    hour = clock.hour + 0.5 * (24.0 / 96.0)
    base = mean * (1.0 + signal.amplitude * daily_shape(hour))
    n_days = int(clock.day.max()) + 1 if len(clock.day) else 1
    if n_days < 7:
        return np.asarray(base, dtype=np.float64)
    r = signal.weekend_reduction
    weekday = 7.0 / (5.0 + 2.0 * (1.0 - r))
    week = np.where(clock.weekend, weekday * (1.0 - r), weekday)
    return np.asarray(base * week, dtype=np.float64)


def trailing_mean(values: FloatArray, window: int) -> FloatArray:
    """Mean of the ``window`` values BEFORE each step, cyclic (no foresight in a periodic world)."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    ext = np.concatenate([v[-window:] if window <= n else np.resize(v, window), v])
    c = np.concatenate([[0.0], np.cumsum(ext)])
    idx = np.arange(n) + window
    return np.asarray((c[idx] - c[idx - window]) / window, dtype=np.float64)
