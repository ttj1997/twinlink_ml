"""Hourly day-ahead prices, the clock of the year and the price information of the controllers.

The raw file (Energy-Charts JSON, ``unix_seconds`` and ``price`` in EUR/MWh) lives in ``local/``
and never enters the repository; :func:`load_price_year` checks its SHA-256 against the config.

Clock: one fixed offset (UTC+1) all year, so every day has 24 hours and the day boundary is 00:00
CET; hour ``i`` of the year is ``Jan 1, 00:00 + i h``.

Price information (:func:`known_price_forecast`): at hour ``t`` of day ``D`` the prices of day ``D``
are known up to its end (the day-ahead auction of ``D - 1``); every later hour is forecast by the
price of the same hour of day ``D`` ("previous day as forecast" for day ``D + 1``). This is the
information of M2, R1 and the RL agents; M1 sees the true prices.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import PriceSettings

__all__ = [
    "HOURS_PER_DAY",
    "Clock",
    "PriceYear",
    "known_price_forecast",
    "load_price_year",
    "year_clock",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
HOURS_PER_DAY = 24


@dataclass(frozen=True, slots=True)
class Clock:
    """Calendar of every hour of the year (local fixed-offset clock)."""

    month: IntArray  # 1..12
    day: IntArray  # day of the year, 0-based
    hour: IntArray  # 0..23
    weekday: IntArray  # 0 = Monday

    @property
    def n_hours(self) -> int:
        return int(self.hour.shape[0])

    @property
    def weekend(self) -> NDArray[np.bool_]:
        return np.asarray(self.weekday >= 5)


def year_clock(year: int) -> Clock:
    """Clock of all hours of ``year``."""
    start = np.datetime64(f"{year}-01-01T00", "h")
    end = np.datetime64(f"{year + 1}-01-01T00", "h")
    stamps = np.arange(start, end, np.timedelta64(1, "h"))
    days = stamps.astype("datetime64[D]")
    month = (stamps.astype("datetime64[M]").astype(np.int64) % 12 + 1).astype(np.int64)
    day = (days - days[0]).astype(np.int64)
    hour = (stamps - days).astype(np.int64)
    weekday = ((days.astype(np.int64) + 3) % 7).astype(np.int64)  # 1970-01-01 was a Thursday
    return Clock(month=month, day=day, hour=hour, weekday=weekday)


@dataclass(frozen=True, slots=True)
class PriceYear:
    """Day-ahead price of every hour (EUR/kWh) and the clock."""

    spot_eur_per_kwh: FloatArray
    clock: Clock
    raw_sha256: str
    license_info: str

    @property
    def n_hours(self) -> int:
        return int(self.spot_eur_per_kwh.shape[0])


def load_price_year(settings: PriceSettings, root: Path) -> PriceYear:
    """Read the raw file, check hash, length, spacing and alignment with the clock."""
    path = root / settings.raw_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- download it from {settings.url} (see conf/storage_price)"
        )
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != settings.sha256:
        raise ValueError(f"SHA-256 of {path} is {digest}, config expects {settings.sha256}")
    raw = json.loads(blob)
    seconds = np.asarray(raw["unix_seconds"], dtype=np.int64)
    if any(p is None for p in raw["price"]):
        raise ValueError("price series has gaps")
    price = np.asarray(raw["price"], dtype=np.float64) / 1000.0  # EUR/MWh -> EUR/kWh
    clock = year_clock(settings.year)
    if len(price) != clock.n_hours or np.any(np.diff(seconds) != 3600):
        raise ValueError("price series is not one value per hour of the year")
    first_local = np.datetime64(int(seconds[0]), "s") + np.timedelta64(
        settings.clock_utc_offset_h, "h"
    )
    if first_local != np.datetime64(f"{settings.year}-01-01T00:00:00", "s"):
        raise ValueError(f"first stamp {first_local} is not Jan 1, 00:00 of the local clock")
    return PriceYear(
        spot_eur_per_kwh=price,
        clock=clock,
        raw_sha256=digest,
        license_info=str(raw.get("license_info", "")),
    )


def known_price_forecast(price: FloatArray, clock: Clock, horizon_h: int) -> FloatArray:
    """``(hours, horizon)`` price information at every hour (see module docstring).

    Row ``t`` holds the values used for hours ``t, t+1, ..., t+horizon-1``: the true price while
    the hour lies on the day of ``t``, else the price of the same hour of the day of ``t``. Hours
    beyond the end of the year are filled the same way (only the last day is affected).
    """
    n = len(price)
    t = np.arange(n)[:, None]
    k = np.arange(horizon_h)[None, :]
    target = t + k
    start_of_day = (clock.day * HOURS_PER_DAY)[:, None]
    same_day = target < start_of_day + HOURS_PER_DAY
    hour_of_target = target % HOURS_PER_DAY
    persistence = start_of_day + hour_of_target
    index = np.where(same_day, target, persistence)
    return np.asarray(price[np.minimum(index, n - 1)], dtype=np.float64)
