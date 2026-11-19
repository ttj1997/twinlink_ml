"""Tariff, selection of the days and the day scenarios (prices and ambient air).

Tariff: day-ahead price + adder of the price study, floored at ``tariff.floor_eur_per_kwh``
(with a negative tariff a dynamic programme would be paid for wasting energy). The reference
tariff is the mean tariff of the TRAINING months; the optimisers work with the tariff relative to
it, the costs are booked with the tariff itself.

Price information (as Befund 115): the prices of the day are known at its start; the hour of the
tail lies on the following day and is forecast by the same hour of the day itself. The cost of
the tail is booked with its true price.

Ambient air: the day profiles of the headroom family ``ambient``
(:class:`~twinlink_ml.simulation.excitation.DiurnalDisturbanceDesign` with the settings of the
headroom config), one of the 5 x 5 weather grid points per day by a seeded permutation, the
profile from a seed per day. Weather and price are therefore independent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from twinlink_ml.headroom.scenarios import weather_grid
from twinlink_ml.simulation import DiurnalDisturbanceDesign
from twinlink_ml.storage_price.prices import HOURS_PER_DAY, Clock, load_price_year

from .config import LoadFollowingConfig

__all__ = [
    "DayScenario",
    "Market",
    "PriceMode",
    "build_days",
    "build_market",
    "daily_spread",
    "month_days",
    "stratified_days",
]

FloatArray = NDArray[np.float64]
PriceMode = Literal["true", "flat"]


@dataclass(frozen=True, slots=True)
class Market:
    """Hourly tariff of the year (EUR/kWh, floor applied), clock and reference values."""

    tariff: FloatArray
    clock: Clock
    reference_tariff: float
    gas_cost_eur_per_kwh_fuel: float
    floored_hours: int
    raw_sha256: str

    def months(self, which: Sequence[int]) -> list[int]:
        """Days (0-based day of the year) of the given months, in order."""
        return [d for m in which for d in month_days(self.clock, m)]


@dataclass(frozen=True, eq=False)
class DayScenario:
    """One day with its tail on the truth sample grid.

    ``disturbances`` ``(5, n_samples)`` in the order of
    :data:`~twinlink_ml.headroom.symbolic.DISTURBANCE_CHANNELS`; the spray-rate row holds the
    nominal value and is NOT used (the spray rate is a command here). ``price_true`` and
    ``price_known`` hold one tariff per hour of the day and of the tail.
    """

    name: str
    day: int
    month: int
    hours: int
    tail_hours: int
    sample_time_s: float
    disturbances: np.ndarray
    moisture_limit: np.ndarray
    price_true: np.ndarray
    price_known: np.ndarray
    price_mode: str

    @property
    def n_samples(self) -> int:
        """Number of truth samples (day and tail)."""
        return int(self.disturbances.shape[1])


def month_days(clock: Clock, month: int) -> list[int]:
    """Days (0-based day of the year) of ``month``."""
    return sorted({int(d) for d in clock.day[clock.month == month]})


def daily_spread(tariff: FloatArray, day: int) -> float:
    """Price spread of a day: maximum minus minimum of its hourly tariff."""
    hours = tariff[day * HOURS_PER_DAY : (day + 1) * HOURS_PER_DAY]
    return float(hours.max() - hours.min())


def build_market(config: LoadFollowingConfig, root: Path) -> Market:
    """Load the price year (hash checked) and build the tariff."""
    prices = load_price_year(config.prices.prices, root)
    raw = prices.spot_eur_per_kwh + config.prices.prices.adder_eur_per_kwh
    tariff = np.maximum(raw, config.tariff_floor_eur_per_kwh)
    train = np.isin(prices.clock.month, config.prices.training_months)
    return Market(
        tariff=np.asarray(tariff, dtype=np.float64),
        clock=prices.clock,
        reference_tariff=float(np.mean(tariff[train])),
        gas_cost_eur_per_kwh_fuel=config.gas_cost_eur_per_kwh_fuel,
        floored_hours=int(np.sum(raw < config.tariff_floor_eur_per_kwh)),
        raw_sha256=prices.raw_sha256,
    )


def stratified_days(
    market: Market, months: Sequence[int], per_tercile: tuple[int, int, int], seed: int
) -> list[int]:
    """Days drawn per month from the terciles of the daily price spread (low, middle, high)."""
    chosen: list[int] = []
    for month in months:
        days = month_days(market.clock, month)
        order = sorted(days, key=lambda d: (daily_spread(market.tariff, d), d))
        rng = np.random.default_rng([seed, month])
        for part, count in zip(np.array_split(np.array(order), 3), per_tercile, strict=True):
            chosen.extend(int(d) for d in rng.choice(part, size=count, replace=False))
    return sorted(chosen)


def _day_prices(
    market: Market, day: int, hours: int, tail_hours: int, mode: PriceMode
) -> tuple[np.ndarray, np.ndarray]:
    total = hours + tail_hours
    if mode == "flat":
        flat = np.full(total, market.reference_tariff)
        return flat, flat.copy()
    start = day * HOURS_PER_DAY
    n = len(market.tariff)
    index = start + np.arange(total)
    same_day = index < start + HOURS_PER_DAY
    persistence = start + index % HOURS_PER_DAY
    true = market.tariff[np.where(index < n, index, persistence)]
    known = market.tariff[np.where(same_day, index, persistence)]
    return np.asarray(true, dtype=np.float64), np.asarray(known, dtype=np.float64)


def build_days(
    config: LoadFollowingConfig,
    market: Market,
    days: Sequence[int],
    *,
    price_mode: PriceMode = "true",
) -> list[DayScenario]:
    """Day scenarios (ambient profile, limit, prices) for the given days of the year."""
    headroom, process = config.headroom, config.process
    dt = headroom.scenarios.sample_time_s
    hours, tail = config.day.hours, config.day.tail_hours
    n = round((hours + tail) * 3600.0 / dt)
    amb = headroom.scenarios.ambient
    design = DiurnalDisturbanceDesign(
        diurnal_amplitude_fraction=amb.diurnal_amplitude_fraction,
        drift_amplitude_fraction=amb.drift_amplitude_fraction,
        drift_correlation_time_s=amb.drift_correlation_time_s,
        solid_fraction_process="independent",
        solid_fraction_std_fraction=0.0,
    )
    space = process.disturbances
    t_r, rh_r, sf_r = (
        space.ambient_temperature_celsius,
        space.ambient_relative_humidity,
        space.suspension_solid_fraction,
    )
    grid = weather_grid(process, amb.grid_points_per_axis)
    seed = config.day.seed
    permutation = np.random.default_rng([seed, 0]).permutation(len(grid))
    solid = headroom.line.nominal_solid_fraction
    spray = headroom.line.nominal_spray_rate_kg_per_h
    limit = headroom.line.product_moisture_limit_kg_per_kg
    out = []
    for day in days:
        t_amb, rh = grid[int(permutation[day % len(grid)])]
        unit = np.array(
            [
                [
                    (t_amb - t_r.minimum) / (t_r.maximum - t_r.minimum),
                    (rh - rh_r.minimum) / (rh_r.maximum - rh_r.minimum),
                    (solid - sf_r.minimum) / (sf_r.maximum - sf_r.minimum),
                ]
            ]
        )
        signal = design.build(space, unit, n, dt, np.random.default_rng([seed, 1, day]))[0]
        disturbances = np.empty((5, n))
        disturbances[0:3] = signal.T
        disturbances[3] = spray
        disturbances[4] = 0.0
        true, known = _day_prices(market, day, hours, tail, price_mode)
        month = int(market.clock.month[day * HOURS_PER_DAY])
        suffix = "" if price_mode == "true" else f"/{price_mode}"
        out.append(
            DayScenario(
                name=f"day{day:03d}{suffix}",
                day=int(day),
                month=month,
                hours=hours,
                tail_hours=tail,
                sample_time_s=dt,
                disturbances=disturbances,
                moisture_limit=np.full(n, limit),
                price_true=true,
                price_known=known,
                price_mode=price_mode,
            )
        )
    return out
