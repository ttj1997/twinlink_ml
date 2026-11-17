r"""Controllers C0-C4 of the load-following study.

All controllers command (inlet temperature, gas velocity, spray rate) piecewise constant on the
control grid and plan for ``X <= limit - m`` with their own margin ``m``. They act during the
hours of the day; in the tail every run gets the same commands (:class:`TailLaw`, applied by the
runner), so a controller is never asked for a tail command.

C0 constant throughput (reference)
    Nominal spray rate; inlet temperature and gas velocity from the steady optimum at the current
    disturbance (R1 of Befund 105).
C1 price rule
    Two throughput levels fixed at the start of the day (:func:`~.steady.rule_levels`); set
    points from the steady optimum at the current disturbance and throughput.
C2 schedule on the steady-state map
    At every full hour: the exact programme over the hourly levels of the remaining day on the
    steady-state power curve at the CURRENT disturbance (held), with the price information and
    the remaining production target (:func:`~.steady.optimal_levels`); set points as C1.
C3 economic MPC
    At every full hour: the dynamic programme (:class:`~.dynamic.LoadOptimizer`) from the TRUE
    state over the rest of the day and the tail, current disturbance held, price information,
    remaining target as a terminal condition. It fixes the spray rate. Every interval the 2 h MPC
    of Befund 105 (the unchanged :class:`~twinlink_ml.headroom.controllers.DynamicOptimizer`,
    R3) sets inlet temperature and gas velocity from the true state, with the planned spray rate
    as a known input: so C3 reacts to the disturbance on the same 5 min grid as C0-C2.
C4 perfect preview
    One dynamic programme per day with the true prices and the true disturbance course, applied
    open loop; started at the schedule of C2 with the same information and never worse than it in
    its own model (the safeguard of R2 in Befund 105).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from twinlink_ml.headroom.controllers import DynamicOptimizer
from twinlink_ml.headroom.symbolic import step_means

from .config import LoadFollowingConfig
from .dynamic import LoadOptimizer, command_rates
from .market import DayScenario
from .steady import Carrier, SteadyMap, cost_rate, optimal_levels, rule_levels

__all__ = [
    "ConstantThroughput",
    "EconomicMpc",
    "LoadController",
    "PerfectPreview",
    "PriceRule",
    "Schedule",
    "SetPointLaw",
    "TailLaw",
    "schedule_levels",
]


class SetPointLaw:
    """Inlet temperature and gas velocity from the steady-state map.

    The steady optimum at the current disturbance and spray rate with the bound
    ``limit - margin``; holds the last set points where there is no steady optimum (counted in
    ``failures``).
    """

    def __init__(self, steady_map: SteadyMap, limit: float, margin: float) -> None:
        self._map = steady_map
        self.bound = limit - margin
        self.failures = 0

    def __call__(self, disturbance: np.ndarray, spray: float, last: np.ndarray) -> np.ndarray:
        point = self._map.point(disturbance, spray, self.bound)
        if point is None:
            self.failures += 1
            return np.asarray(last[:2], dtype=np.float64)
        return point.command


class TailLaw:
    """Commands of the tail, the same for every controller.

    Nominal spray rate, set points of C0 (steady optimum at the current disturbance) with the
    margin of the tail.
    """

    def __init__(self, steady_map: SteadyMap, limit: float, margin: float, nominal: float) -> None:
        self._law = SetPointLaw(steady_map, limit, margin)
        self._nominal = nominal

    @property
    def failures(self) -> int:
        """Intervals without a steady optimum (previous set points held)."""
        return self._law.failures

    def __call__(self, disturbance: np.ndarray, last: np.ndarray) -> np.ndarray:
        inlet, gas = self._law(disturbance, self._nominal, last)
        return np.array([inlet, gas, self._nominal])


class LoadController(ABC):
    """A controller run on one day; the runner calls :meth:`command` per interval of the day."""

    def __init__(self) -> None:
        self.info: dict[str, Any] = {"failures": 0}

    @abstractmethod
    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
        """Start a day from the true state ``(4,)`` with the last command ``(3,)``."""

    @abstractmethod
    def command(
        self,
        k: int,
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,
    ) -> np.ndarray:
        """Command ``(3,)`` for interval ``k`` of the day."""


def schedule_levels(
    steady_map: SteadyMap,
    config: LoadFollowingConfig,
    *,
    disturbances: np.ndarray,
    prices: np.ndarray,
    remaining_kg: float,
    band: tuple[float, float],
    bound: float,
    carrier: Carrier,
) -> np.ndarray:
    """Spray rate per remaining hour of the schedule on the steady-state map.

    ``disturbances`` ``(5,)`` (held over all hours) or ``(hours, 5)`` (one per hour); ``prices``
    ``(hours,)`` tariff per hour; ``remaining_kg`` the suspension still to be sprayed. The levels
    lie on the grid of ``throughput.schedule_levels`` points across ``band``; the grid sum nearest
    to the target is used, and with one hour left the level is the remainder itself.
    """
    prices = np.asarray(prices, dtype=np.float64)
    hours = len(prices)
    low, high = band
    mean = float(np.clip(remaining_kg / hours, low, high))
    if high - low < 1e-9 or hours == 1:
        return np.full(hours, mean)
    levels = np.linspace(low, high, config.throughput.schedule_levels)
    delta = levels[1] - levels[0]
    rows = np.atleast_2d(np.asarray(disturbances, dtype=np.float64))
    if rows.shape[0] == 1:
        total, heater = steady_map.curve(rows[0], levels, bound)
        total, heater = np.tile(total, (hours, 1)), np.tile(heater, (hours, 1))
    else:
        curves = [steady_map.curve(row, levels, bound) for row in rows]
        total = np.stack([c[0] for c in curves])
        heater = np.stack([c[1] for c in curves])
    cost = cost_rate(total, heater, prices[:, None], carrier, config)
    index_sum = round((remaining_kg - hours * low) / delta)
    index_sum = int(np.clip(index_sum, 0, hours * (len(levels) - 1)))
    try:
        return levels[optimal_levels(cost, index_sum)]
    except ValueError:
        return np.full(hours, mean)


class ConstantThroughput(LoadController):
    """C0."""

    def __init__(self, law: SetPointLaw, nominal: float) -> None:
        super().__init__()
        self._law = law
        self._nominal = nominal

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
        """Nothing to reset."""

    def command(
        self,
        k: int,  # noqa: ARG002 - shared interface
        state: np.ndarray,  # noqa: ARG002 - shared interface
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,  # noqa: ARG002 - shared interface
    ) -> np.ndarray:
        """Nominal spray rate, steady set points."""
        inlet, gas = self._law(disturbance, self._nominal, last_command)
        self.info["failures"] = self._law.failures
        return np.array([inlet, gas, self._nominal])


class PriceRule(LoadController):
    """C1."""

    def __init__(
        self,
        law: SetPointLaw,
        config: LoadFollowingConfig,
        band: tuple[float, float],
        quantile: float,
        spread_kg_per_h: float,
    ) -> None:
        super().__init__()
        self._law = law
        self._config = config
        self._band = band
        self._quantile = quantile
        self._spread = spread_kg_per_h
        self._levels = np.empty(0)
        self._per_hour = config.intervals_per_hour

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:  # noqa: ARG002 - shared interface
        """Fix the levels of the day from its known prices."""
        self._levels = rule_levels(
            day.price_known[: day.hours],
            self._quantile,
            self._spread,
            nominal=self._config.nominal_spray_rate,
            low=self._band[0],
            high=self._band[1],
        )

    def command(
        self,
        k: int,
        state: np.ndarray,  # noqa: ARG002 - shared interface
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,  # noqa: ARG002 - shared interface
    ) -> np.ndarray:
        """Level of the hour, steady set points."""
        spray = float(self._levels[k // self._per_hour])
        inlet, gas = self._law(disturbance, spray, last_command)
        self.info["failures"] = self._law.failures
        return np.array([inlet, gas, spray])


class Schedule(LoadController):
    """C2."""

    def __init__(
        self,
        law: SetPointLaw,
        steady_map: SteadyMap,
        config: LoadFollowingConfig,
        band: tuple[float, float],
        carrier: Carrier,
    ) -> None:
        super().__init__()
        self._law = law
        self._map = steady_map
        self._config = config
        self._band = band
        self._carrier = carrier
        self._per_hour = config.intervals_per_hour
        self._day: DayScenario | None = None
        self._spray = config.nominal_spray_rate

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:  # noqa: ARG002 - shared interface
        """Remember the day (prices, target)."""
        self._day = day

    def command(
        self,
        k: int,
        state: np.ndarray,  # noqa: ARG002 - shared interface
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,
    ) -> np.ndarray:
        """Re-plan at the full hour; level of the hour, steady set points."""
        assert self._day is not None
        day = self._day
        if k % self._per_hour == 0:
            hour = k // self._per_hour
            target = self._config.nominal_spray_rate * day.hours
            self._spray = float(
                schedule_levels(
                    self._map,
                    self._config,
                    disturbances=disturbance,
                    prices=day.price_known[hour : day.hours],
                    remaining_kg=target - sprayed_kg,
                    band=self._band,
                    bound=self._law.bound,
                    carrier=self._carrier,
                )[0]
            )
        inlet, gas = self._law(disturbance, self._spray, last_command)
        self.info["failures"] = self._law.failures
        return np.array([inlet, gas, self._spray])


def _planning_inputs(
    config: LoadFollowingConfig,
    day: DayScenario,
    first_interval: int,
    band: tuple[float, float],
    tail_commands: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Command bounds and counted intervals of a programme that runs to the end of the tail.

    Lower and upper bounds of the commands ``(3, n)`` and the indicator ``(n,)`` of the intervals
    that count for the production target, for a programme starting at ``first_interval``.
    """
    per_hour = config.intervals_per_hour
    n_day = day.hours * per_hour
    n_total = n_day + day.tail_hours * per_hour
    n = n_total - first_interval
    actions = config.process.actions
    low = np.tile(
        np.array(
            [actions.inlet_temperature_celsius.minimum, actions.gas_velocity_m_per_s.minimum, 0.0]
        )[:, None],
        (1, n),
    )
    high = np.tile(
        np.array(
            [actions.inlet_temperature_celsius.maximum, actions.gas_velocity_m_per_s.maximum, 0.0]
        )[:, None],
        (1, n),
    )
    low[2], high[2] = band
    in_day = np.arange(first_interval, n_total) < n_day
    n_tail = int(np.sum(~in_day))
    if n_tail:
        low[:, ~in_day] = tail_commands[:, -n_tail:]
        high[:, ~in_day] = tail_commands[:, -n_tail:]
    return low, high, in_day.astype(np.float64)


class PerfectPreview(LoadController):
    """C4."""

    def __init__(
        self,
        optimizer: LoadOptimizer,
        steady_map: SteadyMap,
        tail: TailLaw,
        config: LoadFollowingConfig,
        band: tuple[float, float],
        margin: float,
        reference_tariff: float,
    ) -> None:
        super().__init__()
        self._optimizer = optimizer
        self._map = steady_map
        self._tail = tail
        self._config = config
        self._band = band
        self._margin = margin
        self._reference = reference_tariff
        self._plan = np.empty((3, 0))

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
        """Solve the whole day at once with the true prices and disturbances."""
        cfg, opt = self._config, self._optimizer
        per_hour = cfg.intervals_per_hour
        n_day = day.hours * per_hour
        n_total = opt.n_intervals
        if n_total != n_day + day.tail_hours * per_hour:
            raise ValueError("day length does not match the optimiser horizon")
        spi = cfg.samples_per_interval
        dist = step_means(day.disturbances, spi // opt.n_steps)
        limit = cfg.headroom.line.product_moisture_limit_kg_per_kg
        bound = np.full(n_total * opt.n_steps, limit - self._margin)
        price = np.repeat(day.price_true / self._reference, per_hour)
        starts = day.disturbances[:, ::spi]  # disturbance at every interval start, (5, n_total)
        hourly = day.disturbances[:, : n_day * spi].reshape(5, day.hours, -1).mean(axis=2).T
        levels = schedule_levels(
            self._map,
            cfg,
            disturbances=hourly,
            prices=day.price_true[: day.hours],
            remaining_kg=cfg.nominal_spray_rate * day.hours,
            band=self._band,
            bound=limit - self._margin,
            carrier="E",
        )
        law = SetPointLaw(self._map, limit, self._margin)
        guess = np.empty((3, n_total))
        last = np.asarray(command, dtype=np.float64)
        for k in range(n_total):
            if k < n_day:
                spray = float(levels[k // per_hour])
                inlet, gas = law(starts[:, k], spray, last)
                last = np.array([inlet, gas, spray])
            else:
                last = self._tail(starts[:, k], last)
            guess[:, k] = last
        low, high, counted = _planning_inputs(cfg, day, 0, self._band, guess)
        x0 = state[[0, 1, 3]]
        target = cfg.nominal_spray_rate * day.hours
        guess_objective = opt.penalised_objective(x0, command, guess, dist, bound, price)
        started = time.perf_counter()
        try:
            plan = opt.solve(
                x0,
                command,
                dist,
                bound,
                price,
                command_low=low,
                command_high=high,
                guess_commands=guess,
                counted=counted,
                target_kg=target,
            )
            objective = opt.last_objective
            self.info.update(opt.last_stats)
            self.info["slack_max"] = opt.last_slack_max
        except RuntimeError as error:
            plan, objective = guess, np.inf
            self.info["failures"] += 1
            self.info["status"] = str(error)
        self.info["solve_s"] = time.perf_counter() - started
        self.info["objective"] = float(objective)
        self.info["guess_objective"] = float(guess_objective)
        # Safeguard as R2 of Befund 105: never a plan that the own model rates worse than the
        # start (a worse local optimum of the non-convex problem, or a failed solve). Counted.
        self.info["fallback_to_guess"] = bool(guess_objective < objective)
        self._plan = guess if guess_objective < objective else plan

    def command(
        self,
        k: int,
        state: np.ndarray,  # noqa: ARG002 - shared interface
        last_command: np.ndarray,  # noqa: ARG002 - shared interface
        disturbance: np.ndarray,  # noqa: ARG002 - shared interface
        sprayed_kg: float,  # noqa: ARG002 - shared interface
    ) -> np.ndarray:
        """Planned command of interval ``k``."""
        return self._plan[:, k]


class EconomicMpc(LoadController):
    """C3."""

    def __init__(
        self,
        optimizers: Any,
        fast: DynamicOptimizer,
        steady_map: SteadyMap,
        tail: TailLaw,
        config: LoadFollowingConfig,
        band: tuple[float, float],
        margin: float,
        reference_tariff: float,
    ) -> None:
        super().__init__()
        self._optimizers = optimizers  # callable n_intervals -> LoadOptimizer
        self._fast = fast
        self._map = steady_map
        self._tail = tail
        self._config = config
        self._band = band
        self._margin = margin
        self._reference = reference_tariff
        self._day: DayScenario | None = None
        self._plan = np.empty((3, 0))
        self._plan_start = 0
        self._rate = float(command_rates(config.process)[2])
        self.info.update(
            {"plan_failures": 0, "plan_iterations": [], "plan_s": [], "fast_iterations": []}
        )

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:  # noqa: ARG002 - shared interface
        """Forget the plan and the warm start."""
        self._day = day
        self._plan = np.empty((3, 0))
        self._fast.reset()

    # ------------------------------------------------------------------ hourly plan

    def _replan(
        self,
        k: int,
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,
    ) -> None:
        cfg, day = self._config, self._day
        assert day is not None
        per_hour = cfg.intervals_per_hour
        n_day = day.hours * per_hour
        n_total = n_day + day.tail_hours * per_hour
        n = n_total - k
        opt: LoadOptimizer = self._optimizers(n)
        hour = k // per_hour
        limit = cfg.headroom.line.product_moisture_limit_kg_per_kg
        nodes = n * opt.n_steps
        dist = np.tile(np.asarray(disturbance, dtype=np.float64)[:, None], (1, nodes))
        bound = np.full(nodes, limit - self._margin)
        price = np.repeat(day.price_known[hour:] / self._reference, per_hour)
        target = cfg.nominal_spray_rate * day.hours - sprayed_kg
        tail_command = self._tail(disturbance, last_command)
        if self._plan.shape[1] >= k - self._plan_start + n and self._plan.shape[1] > 0:
            guess = self._plan[:, k - self._plan_start :].copy()
        else:
            levels = schedule_levels(
                self._map,
                cfg,
                disturbances=disturbance,
                prices=day.price_known[hour : day.hours],
                remaining_kg=target,
                band=self._band,
                bound=limit - self._margin,
                carrier="E",
            )
            law = SetPointLaw(self._map, limit, self._margin)
            guess = np.empty((3, n))
            last = np.asarray(last_command, dtype=np.float64)
            for i in range(n):
                if k + i < n_day:
                    spray = float(levels[(k + i) // per_hour - hour])
                    inlet, gas = law(disturbance, spray, last)
                    last = np.array([inlet, gas, spray])
                else:
                    last = tail_command
                guess[:, i] = last
        guess[:, k + np.arange(n) >= n_day] = tail_command[:, None]
        low, high, counted = _planning_inputs(cfg, day, k, self._band, guess)
        started = time.perf_counter()
        try:
            plan = opt.solve(
                state[[0, 1, 3]],
                last_command,
                dist,
                bound,
                price,
                command_low=low,
                command_high=high,
                guess_commands=guess,
                counted=counted,
                target_kg=target,
            )
            self.info["plan_iterations"].append(int(opt.last_stats["iterations"]))  # type: ignore[call-overload]
        except RuntimeError:
            plan = np.clip(guess, low, high)
            self.info["plan_failures"] += 1
        self.info["plan_s"].append(time.perf_counter() - started)
        self._plan, self._plan_start = plan, k

    # ------------------------------------------------------------------ set points every interval

    def _spray_course(self, k: int, realised: float) -> np.ndarray:
        """Planned spray rate per RK4 step of the fast horizon, rate limited as in the truth."""
        cfg = self._config
        fast = self._fast
        dt = cfg.headroom.scenarios.sample_time_s
        spi = cfg.samples_per_interval
        planned = np.full(fast.n_intervals, cfg.nominal_spray_rate)
        offset = k - self._plan_start
        available = self._plan[2, offset : offset + fast.n_intervals]
        planned[: len(available)] = available
        course = np.empty(fast.n_intervals * spi)
        value = realised
        step = self._rate * dt
        for i, target in enumerate(np.repeat(planned, spi)):
            value += float(np.clip(target - value, -step, step))
            course[i] = value
        return course.reshape(fast.n_intervals * fast.n_steps, -1).mean(axis=1)

    def command(
        self,
        k: int,
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,
    ) -> np.ndarray:
        """Spray rate of the hourly plan, set points of the 2 h MPC."""
        cfg = self._config
        if k % (cfg.resolve_interval_h * cfg.intervals_per_hour) == 0:
            self._replan(k, state, last_command, disturbance, sprayed_kg)
        spray = float(self._plan[2, k - self._plan_start])
        fast = self._fast
        nodes = fast.n_intervals * fast.n_steps
        dist = np.tile(np.asarray(disturbance, dtype=np.float64)[:, None], (1, nodes))
        dist[3] = self._spray_course(k, float(last_command[2]))
        limit = cfg.headroom.line.product_moisture_limit_kg_per_kg
        bound = np.full(nodes, limit - self._margin)
        try:
            plan = fast.solve(state[[0, 1, 3]], np.asarray(last_command[:2]), dist, bound)
            inlet, gas = plan[:, 0]
            self.info["fast_iterations"].append(int(fast.last_stats["iterations"]))  # type: ignore[call-overload]
        except RuntimeError:
            self.info["failures"] += 1
            inlet, gas = last_command[:2]
        return np.array([inlet, gas, spray])
