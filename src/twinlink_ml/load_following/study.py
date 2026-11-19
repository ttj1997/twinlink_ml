"""Running the controllers C0-C4 on days: initial conditions, lockstep truth runs, records.

:func:`run_days` is the single entry point of the driver (``scripts/run_load_following.py``), of
its worker processes and of the tests. It builds the controllers of one kind for a list of days,
runs them in lockstep on one :class:`~.truth.LoadTruthPlant` batch and returns one record per day.

Initial condition (all controllers alike): the steady optimum at the first disturbance sample and
the nominal throughput with the bound ``limit - start_margin`` of the headroom config.

Day and tail: during the hours of the day the controller decides; in the tail every run gets the
commands of :class:`~.controllers.TailLaw` (nominal throughput, set points of C0 with the tail
margin). The cost of the tail counts and the path condition holds there too.

Costs per day (hourly energies of the truth times the TRUE tariff of the hour):
``cost_e`` with the whole power priced (electric heater), ``cost_g`` with a gas-fired heater
(:func:`~.steady.cost_rate`).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from twinlink_ml.headroom.controllers import DynamicOptimizer
from twinlink_ml.headroom.symbolic import steady_particle_count

from .config import LoadFollowingConfig
from .controllers import (
    ConstantThroughput,
    EconomicMpc,
    LoadController,
    PerfectPreview,
    PriceRule,
    Schedule,
    SetPointLaw,
    TailLaw,
)
from .dynamic import LoadOptimizer
from .market import DayScenario
from .steady import Carrier, SteadyMap, cost_rate
from .truth import LoadTruthPlant

__all__ = ["ControllerKind", "RunSpec", "Solvers", "initial_conditions", "run_days"]

ControllerKind = Literal["C0", "C1", "C2", "C3", "C4"]


@dataclass
class Solvers:
    """Steady map and optimisers of one process, built lazily (one set per worker process)."""

    config: LoadFollowingConfig
    reference_tariff: float
    _map: SteadyMap | None = None
    _load: dict[int, LoadOptimizer] = field(default_factory=dict)

    @property
    def steady(self) -> SteadyMap:
        """Steady-state map (C0-C2, tail, initial conditions, starts of C3 and C4)."""
        if self._map is None:
            self._map = SteadyMap(self.config)
        return self._map

    def load(self, n_intervals: int) -> LoadOptimizer:
        """Dynamic programme over ``n_intervals`` (C3, C4).

        Only the two most recently used horizon lengths are kept: C3 needs a different length at
        every full hour, and a programme per length held for the whole run would multiply the
        memory of a worker by the number of hours (building one takes about a second).
        """
        if n_intervals in self._load:
            self._load[n_intervals] = self._load.pop(n_intervals)  # most recent last
        else:
            while len(self._load) >= 2:
                self._load.pop(next(iter(self._load)))
            self._load[n_intervals] = LoadOptimizer(
                self.config.headroom, self.config.process, n_intervals
            )
        return self._load[n_intervals]

    def fast(self) -> DynamicOptimizer:
        """A fresh 2 h MPC of Befund 105 (R3) for the set points of C3 (own warm start)."""
        headroom = self.config.headroom
        return DynamicOptimizer(
            headroom, self.config.process, headroom.control.mpc_intervals, expression="MX"
        )


@dataclass(frozen=True, slots=True)
class RunSpec:
    """What is run: controller, its margin and the variant it belongs to."""

    kind: ControllerKind
    margin: float
    tail_margin: float
    band: tuple[float, float]
    band_name: str = "main"
    carrier: Carrier = "E"
    rule_quantile: float | None = None
    rule_spread_kg_per_h: float | None = None


def initial_conditions(
    solvers: Solvers, days: Sequence[DayScenario]
) -> tuple[np.ndarray, np.ndarray]:
    """Initial states ``(n, 4)`` and commands ``(n, 3)``, see the module docstring."""
    cfg = solvers.config
    line = cfg.headroom.line
    bound = line.product_moisture_limit_kg_per_kg - line.start_margin_kg_per_kg
    nominal = cfg.nominal_spray_rate
    states, commands = [], []
    for day in days:
        d0 = day.disturbances[:, 0]
        point = solvers.steady.point(d0, nominal, bound)
        if point is None:
            raise RuntimeError(f"no steady start for {day.name}")
        count = steady_particle_count(cfg.process, float(d0[2]))
        states.append([point.state[0], point.state[1], count, point.state[2]])
        commands.append([point.command[0], point.command[1], nominal])
    return np.array(states), np.array(commands)


def _build_controller(spec: RunSpec, solvers: Solvers, day: DayScenario) -> LoadController:
    cfg = solvers.config
    limit = cfg.headroom.line.product_moisture_limit_kg_per_kg
    nominal = cfg.nominal_spray_rate
    law = SetPointLaw(solvers.steady, limit, spec.margin)
    if spec.kind == "C0":
        return ConstantThroughput(law, nominal)
    if spec.kind == "C1":
        if spec.rule_quantile is None or spec.rule_spread_kg_per_h is None:
            raise ValueError("C1 needs its tuned quantile and spread")
        return PriceRule(law, cfg, spec.band, spec.rule_quantile, spec.rule_spread_kg_per_h)
    if spec.kind == "C2":
        return Schedule(law, solvers.steady, cfg, spec.band, spec.carrier)
    tail = TailLaw(solvers.steady, limit, spec.tail_margin, nominal)
    n_total = (day.hours + day.tail_hours) * cfg.intervals_per_hour
    if spec.kind == "C4":
        return PerfectPreview(
            solvers.load(n_total),
            solvers.steady,
            tail,
            cfg,
            spec.band,
            spec.margin,
            solvers.reference_tariff,
        )
    if spec.kind == "C3":
        return EconomicMpc(
            solvers.load,
            solvers.fast(),
            solvers.steady,
            tail,
            cfg,
            spec.band,
            spec.margin,
            solvers.reference_tariff,
        )
    raise ValueError(f"unknown controller {spec.kind!r}")


def _summarise(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": float(len(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "sum": float(array.sum()),
    }


def run_days(spec: RunSpec, days: Sequence[DayScenario], solvers: Solvers) -> list[dict[str, Any]]:
    """Run controller ``spec`` on ``days`` (one lockstep truth batch); one record per day."""
    cfg = solvers.config
    headroom = cfg.headroom
    limit = headroom.line.product_moisture_limit_kg_per_kg
    nominal = cfg.nominal_spray_rate
    per_hour = cfg.intervals_per_hour
    hours, tail_hours = days[0].hours, days[0].tail_hours
    n_day = hours * per_hour
    states, commands = initial_conditions(solvers, days)
    plant = LoadTruthPlant(
        cfg.process, days, states, commands, interval_s=headroom.control.interval_s
    )
    controllers = [_build_controller(spec, solvers, day) for day in days]
    tails = [TailLaw(solvers.steady, limit, spec.tail_margin, nominal) for _ in days]
    decide_s: list[list[float]] = [[] for _ in days]
    reset_s = []
    for i, (controller, day) in enumerate(zip(controllers, days, strict=True)):
        started = time.perf_counter()
        controller.reset(day, states[i], commands[i])
        reset_s.append(time.perf_counter() - started)
    last = np.array(commands, dtype=np.float64)
    for k in range(plant.n_intervals):
        disturbance = plant.current_disturbance()
        realised = plant.last_commands
        sprayed = plant.sprayed_so_far
        new = np.empty_like(last)
        for i, controller in enumerate(controllers):
            if k < n_day:
                started = time.perf_counter()
                new[i] = controller.command(
                    k, plant.state[i], realised[i], disturbance[i], float(sprayed[i])
                )
                decide_s[i].append(time.perf_counter() - started)
            else:
                new[i] = tails[i](disturbance[i], realised[i])
        plant.advance(new)
        last = new
    result = plant.result()
    spi = cfg.samples_per_interval
    low, high = spec.band
    records = []
    for i, day in enumerate(days):
        shape = (hours + tail_hours, per_hour)
        energy = result.energy_kwh_per_interval[i].reshape(shape).sum(axis=1)
        heater = result.heater_kwh_per_interval[i].reshape(shape).sum(axis=1)
        sprayed_h = result.sprayed_kg_per_interval[i].reshape(shape).sum(axis=1)
        excess = np.maximum(result.moisture[i] - result.moisture_limit[i], 0.0)
        day_samples = n_day * spi
        spray_commands = result.commands[i, :n_day, 2]
        info = dict(controllers[i].info)
        for key in ("plan_iterations", "plan_s", "fast_iterations"):
            if key in info:
                info[key] = _summarise(info[key])
        info["tail_failures"] = tails[i].failures
        solid = float(day.disturbances[2, 0])
        records.append(
            {
                "day": day.name,
                "day_of_year": day.day,
                "month": day.month,
                "price_mode": day.price_mode,
                "controller": spec.kind,
                "carrier": spec.carrier,
                "band": spec.band_name,
                "margin": float(spec.margin),
                "tail_margin": float(spec.tail_margin),
                "cost_e": float(np.sum(cost_rate(energy, heater, day.price_true, "E", cfg))),
                "cost_g": float(np.sum(cost_rate(energy, heater, day.price_true, "G", cfg))),
                "energy_kwh": float(energy.sum()),
                "heater_kwh": float(heater.sum()),
                "sprayed_day_kg": float(sprayed_h[:hours].sum()),
                "sprayed_total_kg": float(sprayed_h.sum()),
                "target_day_kg": float(nominal * hours),
                "product_dry_kg": float(sprayed_h.sum() * solid),
                "max_violation": float(excess.max()),
                "max_violation_tail": float(excess[day_samples:].max(initial=0.0)),
                "max_bed_temperature": float(result.bed_temperature[i].max()),
                "share_at_low": float(np.mean(spray_commands <= low + 1e-6)),
                "share_at_high": float(np.mean(spray_commands >= high - 1e-6)),
                "d50_um_range": [float(v) for v in result.d50_um_range[i]],
                "fluidisation_margin_min": [float(v) for v in result.fluidisation_margin_min[i]],
                "energy_kwh_per_hour": [float(f"{v:.7g}") for v in energy],
                "heater_kwh_per_hour": [float(f"{v:.7g}") for v in heater],
                "sprayed_kg_per_hour": [float(f"{v:.7g}") for v in sprayed_h],
                "moisture_max_per_hour": [
                    float(f"{v:.6g}") for v in result.moisture[i].reshape(shape[0], -1).max(axis=1)
                ],
                "decide_s": _summarise(decide_s[i]),
                "reset_s": float(reset_s[i]),
                "solver": info,
            }
        )
    return records
