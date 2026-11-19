"""Stage 0 of the load-following study: everything that needs the steady-state map only.

* :func:`map_table` -- the steady optimum over the spray-rate grid at every weather grid point;
* :func:`admissible_range` and :func:`bands` -- where a steady state within the limits exists at
  every weather point, and the two bands derived from it (main: +-20 % cut with the range; box:
  the whole range);
* :func:`hourly_curves` -- power curves over the levels of a band for every hour of given days
  (hourly mean of the true ambient air: hindsight, used for tuning on TRAINING days and for the
  descriptive figures only);
* :func:`tune_rule` -- quantile and spread of the price rule C1 on the training days;
* :func:`static_costs` -- cost of constant throughput and of the schedule on the map (daily
  target), and :func:`weekly_costs` -- the same with a weekly target (descriptive).

No truth run and no dynamics here; margins are zero unless a bound is passed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from twinlink_ml.headroom.scenarios import weather_grid

from .config import LoadFollowingConfig
from .market import DayScenario
from .steady import Carrier, SteadyMap, cost_rate, optimal_levels, rule_levels

__all__ = [
    "admissible_range",
    "bands",
    "hourly_curves",
    "map_table",
    "static_costs",
    "throughput_grid",
    "tune_rule",
    "weekly_costs",
]


def throughput_grid(config: LoadFollowingConfig) -> np.ndarray:
    """Spray-rate grid of stage 0 over the ActionBound box (contains the nominal value)."""
    box = config.process.actions.spray_rate_kg_per_h
    step = config.throughput.stage0_step_kg_per_h
    n = round((box.maximum - box.minimum) / step)
    return np.round(box.minimum + step * np.arange(n + 1), 10)


def map_table(config: LoadFollowingConfig, steady_map: SteadyMap) -> dict[str, Any]:
    """Steady optimum at every weather grid point and spray rate of the grid.

    At the moisture limit itself (margin 0): powers, set points, moisture and bed temperature;
    ``feasible_with_margin``: whether a steady optimum also exists with the bound
    ``limit - throughput.stage0_feasibility_margin``.
    """
    headroom = config.headroom
    limit = headroom.line.product_moisture_limit_kg_per_kg
    tight = limit - config.throughput.stage0_feasibility_margin_kg_per_kg
    grid = throughput_grid(config)
    weather = weather_grid(config.process, headroom.scenarios.ambient.grid_points_per_axis)
    solid = headroom.line.nominal_solid_fraction
    shape = (len(weather), len(grid))
    keys = ("total_kw", "heater_kw", "inlet_celsius", "gas_m_per_s", "moisture", "bed_celsius")
    table = {key: np.full(shape, np.nan) for key in keys}
    feasible = np.zeros(shape, dtype=bool)
    for i, (t_amb, rh) in enumerate(weather):
        d = np.array([t_amb, rh, solid, 0.0, 0.0])
        for j, spray in enumerate(grid):
            point = steady_map.point(d, float(spray), limit)
            if point is not None:
                table["total_kw"][i, j] = point.total_kw
                table["heater_kw"][i, j] = point.heater_kw
                table["inlet_celsius"][i, j] = point.command[0]
                table["gas_m_per_s"][i, j] = point.command[1]
                table["moisture"][i, j] = point.state[1] / point.state[0]
                table["bed_celsius"][i, j] = point.state[2]
            feasible[i, j] = steady_map.point(d, float(spray), tight) is not None
    return {
        "spray_kg_per_h": grid.tolist(),
        "weather": [list(w) for w in weather],
        "feasible_with_margin": feasible.tolist(),
        **{key: value.tolist() for key, value in table.items()},
    }


def admissible_range(grid: np.ndarray, feasible: np.ndarray, nominal: float) -> tuple[float, float]:
    """Largest interval of the grid around ``nominal`` that is feasible at every weather point."""
    everywhere = np.asarray(feasible, dtype=bool).all(axis=0)
    centre = int(np.argmin(np.abs(grid - nominal)))
    if not everywhere[centre]:
        raise ValueError("the nominal throughput is not admissible at every weather point")
    low = centre
    while low > 0 and everywhere[low - 1]:
        low -= 1
    high = centre
    while high < len(grid) - 1 and everywhere[high + 1]:
        high += 1
    return float(grid[low]), float(grid[high])


def bands(
    config: LoadFollowingConfig, admissible: tuple[float, float]
) -> dict[str, tuple[float, float]]:
    """Main band (+-``main_relative_band`` cut with the admissible range) and the box."""
    nominal = config.nominal_spray_rate
    width = config.throughput.main_relative_band
    main = (
        max(admissible[0], nominal * (1.0 - width)),
        min(admissible[1], nominal * (1.0 + width)),
    )
    return {"main": (round(main[0], 10), round(main[1], 10)), "box": admissible}


def hourly_curves(
    config: LoadFollowingConfig,
    steady_map: SteadyMap,
    days: Sequence[DayScenario],
    band: tuple[float, float],
    bound: float,
) -> dict[str, np.ndarray]:
    """Power curves over the band for every hour of the given days.

    ``(days, hours, levels)`` at the hourly mean of the true ambient air of each day (hindsight),
    and the levels.
    """
    levels = np.linspace(band[0], band[1], config.throughput.schedule_levels)
    spi = config.samples_per_interval * config.intervals_per_hour
    total = np.empty((len(days), days[0].hours, len(levels)))
    heater = np.empty_like(total)
    for i, day in enumerate(days):
        hourly = day.disturbances[:, : day.hours * spi].reshape(5, day.hours, spi).mean(axis=2)
        for h in range(day.hours):
            total[i, h], heater[i, h] = steady_map.curve(hourly[:, h], levels, bound)
    return {"levels": levels, "total_kw": total, "heater_kw": heater}


def _interpolated_cost(
    curves: dict[str, np.ndarray],
    day_index: int,
    spray: np.ndarray,
    prices: np.ndarray,
    carrier: Carrier,
    config: LoadFollowingConfig,
) -> float:
    levels = curves["levels"]
    total = np.array(
        [np.interp(s, levels, curves["total_kw"][day_index, h]) for h, s in enumerate(spray)]
    )
    heater = np.array(
        [np.interp(s, levels, curves["heater_kw"][day_index, h]) for h, s in enumerate(spray)]
    )
    return float(np.sum(cost_rate(total, heater, prices, carrier, config)))


def tune_rule(
    config: LoadFollowingConfig,
    days: Sequence[DayScenario],
    curves: dict[str, np.ndarray],
    band: tuple[float, float],
    carrier: Carrier,
) -> dict[str, Any]:
    """Grid search of the quantile and the spread of C1 on the steady-state map of ``days``.

    Cost = hourly tariff times the steady power at the level of the rule (hindsight ambient air,
    margin 0). Ties go to the smaller spread, then the smaller quantile.
    """
    nominal = config.nominal_spray_rate
    width = band[1] - band[0]
    best: tuple[float, float, float] | None = None
    table = []
    for fraction in config.spread_grid:
        for quantile in config.quantile_grid:
            cost = 0.0
            for i, day in enumerate(days):
                prices = day.price_known[: day.hours]
                spray = rule_levels(
                    prices, quantile, fraction * width, nominal=nominal, low=band[0], high=band[1]
                )
                cost += _interpolated_cost(curves, i, spray, prices, carrier, config)
            table.append([fraction, quantile, cost])
            if best is None or cost < best[2] - 1e-12:
                best = (fraction, quantile, cost)
    assert best is not None
    constant = sum(
        _interpolated_cost(
            curves,
            i,
            np.full(day.hours, nominal),
            day.price_known[: day.hours],
            carrier,
            config,
        )
        for i, day in enumerate(days)
    )
    return {
        "carrier": carrier,
        "spread_fraction_of_band": best[0],
        "spread_kg_per_h": best[0] * width,
        "quantile": best[1],
        "cost": best[2],
        "cost_constant": float(constant),
        "grid": table,
    }


def _schedule_cost(
    cost: np.ndarray, levels: np.ndarray, nominal: float
) -> tuple[float, np.ndarray]:
    """Optimal cost and levels for ``cost`` ``(hours, levels)`` with the mean level ``nominal``."""
    hours = cost.shape[0]
    delta = levels[1] - levels[0]
    index_sum = round(hours * (nominal - levels[0]) / delta)
    chosen = optimal_levels(cost, int(index_sum))
    return float(cost[np.arange(hours), chosen].sum()), levels[chosen]


def static_costs(
    config: LoadFollowingConfig,
    days: Sequence[DayScenario],
    curves: dict[str, np.ndarray],
    carrier: Carrier,
) -> dict[str, Any]:
    """Per day: cost of constant throughput and of the schedule on the map (daily target)."""
    nominal = config.nominal_spray_rate
    levels = curves["levels"]
    constant, schedule, spread = [], [], []
    for i, day in enumerate(days):
        prices = day.price_true[: day.hours]
        cost = cost_rate(
            curves["total_kw"][i], curves["heater_kw"][i], prices[:, None], carrier, config
        )
        constant.append(
            _interpolated_cost(curves, i, np.full(day.hours, nominal), prices, carrier, config)
        )
        best, chosen = _schedule_cost(cost, levels, nominal)
        schedule.append(best)
        spread.append(float(chosen.max() - chosen.min()))
    total_c, total_s = float(np.sum(constant)), float(np.sum(schedule))
    return {
        "carrier": carrier,
        "cost_constant": total_c,
        "cost_schedule": total_s,
        "gain": (total_c - total_s) / total_c,
        "per_day_gain": [(c - s) / c for c, s in zip(constant, schedule, strict=True)],
        "per_day_level_spread": spread,
    }


def weekly_costs(
    config: LoadFollowingConfig,
    weeks: Sequence[Sequence[DayScenario]],
    curves: Sequence[dict[str, np.ndarray]],
    carrier: Carrier,
) -> list[dict[str, Any]]:
    """Per week: constant throughput and the schedule with a daily and with a weekly target.

    All on the steady-state map with hindsight ambient air and the true prices.
    """
    nominal = config.nominal_spray_rate
    out = []
    for week, week_curves in zip(weeks, curves, strict=True):
        levels = week_curves["levels"]
        daily = static_costs(config, week, week_curves, carrier)
        cost = np.concatenate(
            [
                cost_rate(
                    week_curves["total_kw"][i],
                    week_curves["heater_kw"][i],
                    day.price_true[: day.hours, None],
                    carrier,
                    config,
                )
                for i, day in enumerate(week)
            ]
        )
        weekly, _ = _schedule_cost(cost, levels, nominal)
        out.append(
            {
                "first_day": week[0].day,
                "days": len(week),
                "cost_constant": daily["cost_constant"],
                "cost_daily_target": daily["cost_schedule"],
                "cost_weekly_target": weekly,
            }
        )
    return out
