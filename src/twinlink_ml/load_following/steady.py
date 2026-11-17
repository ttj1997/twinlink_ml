r"""Steady-state map over the throughput, the schedule on it (C2) and the price rule (C1).

Map: the steady optimum of inlet temperature and gas velocity at a given disturbance, spray rate
and moisture bound -- the unchanged :class:`~twinlink_ml.headroom.controllers.
SteadyStateOptimizer` of the headroom study, whose disturbance vector already carries the spray
rate. Its objective is the total power; the heater share of the optimum is read from the truth
model's own diagnostics at the optimum (:meth:`GreyBoxFluidBedGranulator.evaluate_derivative`),
so no physics is repeated here.

Cost rate of a steady point at the tariff :math:`c`:

* carrier ``E`` (electric heater, the whole power priced): :math:`c\,P`;
* carrier ``G`` (gas-fired heater): :math:`c\,(P - P_H) + c_{gas}\,P_H\,\eta_{el}/\eta_{gas}`
  with the heater power :math:`P_H` of the process config (efficiency :math:`\eta_{el}`) rebooked
  to a gas heater of efficiency :math:`\eta_{gas}` (the booking of the headroom study). The set
  points stay those of minimum total power; they are not re-optimised for the price ratio of gas
  and electricity (side variant; ``docs/NN_load_following.md``).

Schedule (:func:`optimal_levels`): hourly levels on a grid whose sum is fixed, minimising the sum
of hourly costs -- an exact dynamic programme over the level-index sum, so that a non-convex
steady-state map (falling specific energy) is handled as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from twinlink_ml.headroom.controllers import SteadyStateOptimizer
from twinlink_ml.headroom.symbolic import steady_particle_count
from twinlink_ml.simulation import GreyBoxFluidBedGranulator

from .config import LoadFollowingConfig

__all__ = [
    "Carrier",
    "MapPoint",
    "SteadyMap",
    "cost_rate",
    "optimal_levels",
    "rule_levels",
]

Carrier = Literal["E", "G"]


@dataclass(frozen=True, slots=True)
class MapPoint:
    """Steady optimum at one throughput: commands (T_in, u_gas), state, powers in kW."""

    command: np.ndarray
    state: np.ndarray
    total_kw: float
    heater_kw: float


class SteadyMap:
    """Steady optima over disturbance, spray rate and moisture bound, cached."""

    def __init__(self, config: LoadFollowingConfig) -> None:
        self._config = config
        self._process = config.process
        self._optimizer = SteadyStateOptimizer(config.headroom, config.process)
        self._simulator = GreyBoxFluidBedGranulator(config.process)
        self._nozzle = config.process.actions.nozzle_pressure_bar.nominal
        self._cache: dict[tuple[float, ...], MapPoint | None] = {}
        self.failures = 0

    def point(self, disturbance: np.ndarray, spray: float, bound: float) -> MapPoint | None:
        """Steady optimum at ``disturbance`` with the spray rate ``spray``.

        ``None`` if there is no steady state within the limits (the optimiser fails).
        """
        d = np.array(disturbance, dtype=np.float64)
        d[3] = spray
        key = (*np.round(d, 10), round(float(bound), 10))
        if key in self._cache:
            return self._cache[key]
        try:
            steady = self._optimizer.solve(d, float(bound))
        except RuntimeError:
            self.failures += 1
            self._cache[key] = None
            return None
        count = steady_particle_count(self._process, float(d[2]))
        state = np.array([[steady.state[0], steady.state[1], count, steady.state[2]]])
        action = np.array([[steady.command[0] + d[4], steady.command[1], spray, self._nozzle]])
        _, diagnostics = self._simulator.evaluate_derivative(state, action, d[None, 0:3])
        point = MapPoint(
            command=steady.command.copy(),
            state=steady.state.copy(),
            total_kw=float(np.asarray(diagnostics["power_total_kw"]).ravel()[0]),
            heater_kw=float(np.asarray(diagnostics["power_heater_kw"]).ravel()[0]),
        )
        self._cache[key] = point
        return point

    def curve(
        self, disturbance: np.ndarray, levels: np.ndarray, bound: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Total and heater power (kW) at every spray level; ``nan`` where infeasible."""
        total = np.full(len(levels), np.nan)
        heater = np.full(len(levels), np.nan)
        for j, level in enumerate(levels):
            point = self.point(disturbance, float(level), bound)
            if point is not None:
                total[j], heater[j] = point.total_kw, point.heater_kw
        return total, heater


def cost_rate(
    total_kw: np.ndarray | float,
    heater_kw: np.ndarray | float,
    tariff: np.ndarray | float,
    carrier: Carrier,
    config: LoadFollowingConfig,
) -> np.ndarray:
    """Cost rate in EUR/h of a power at a tariff, see the module docstring."""
    total = np.asarray(total_kw, dtype=np.float64)
    if carrier == "E":
        return np.asarray(tariff * total)
    if carrier == "G":
        heater = np.asarray(heater_kw, dtype=np.float64)
        rebook = config.process.energy.heater_efficiency / (
            config.headroom.control.gas_heater_efficiency
        )
        return np.asarray(
            tariff * (total - heater) + config.gas_cost_eur_per_kwh_fuel * heater * rebook
        )
    raise ValueError(f"unknown carrier {carrier!r}")


def optimal_levels(cost: np.ndarray, index_sum: int) -> np.ndarray:
    """Level index per hour, minimising ``sum_h cost[h, j_h]`` with ``sum_h j_h == index_sum``.

    ``cost`` ``(hours, levels)``; ``nan`` or ``inf`` marks an inadmissible level. Exact dynamic
    programme over the partial sums. Raises ``ValueError`` if no admissible choice has the sum.
    """
    cost = np.where(np.isfinite(cost), cost, np.inf)
    hours, levels = cost.shape
    top = hours * (levels - 1)
    if not 0 <= index_sum <= top:
        raise ValueError("index_sum outside the reachable range")
    best = np.full(top + 1, np.inf)
    best[0] = 0.0
    choice = np.zeros((hours, top + 1), dtype=np.int64)
    for h in range(hours):
        new = np.full(top + 1, np.inf)
        pick = np.zeros(top + 1, dtype=np.int64)
        for j in range(levels):
            if not np.isfinite(cost[h, j]):
                continue
            candidate = np.full(top + 1, np.inf)
            candidate[j:] = best[: top + 1 - j] + cost[h, j]
            better = candidate < new
            new[better] = candidate[better]
            pick[better] = j
        best = new
        choice[h] = pick
    if not np.isfinite(best[index_sum]):
        raise ValueError("no admissible schedule with the required sum")
    out = np.empty(hours, dtype=np.int64)
    remaining = index_sum
    for h in range(hours - 1, -1, -1):
        out[h] = choice[h, remaining]
        remaining -= int(out[h])
    return out


def rule_levels(
    prices: np.ndarray,
    quantile: float,
    spread: float,
    *,
    nominal: float,
    low: float,
    high: float,
) -> np.ndarray:
    """Spray rate per hour of the price rule C1.

    Hours whose price is at most the ``quantile`` of ``prices`` are cheap and run at the upper
    level, the others at the lower one; the levels differ by ``spread`` (kg/h) and are placed so
    that the mean is ``nominal``. The spread is reduced where a level would leave ``[low, high]``.
    """
    prices = np.asarray(prices, dtype=np.float64)
    hours = len(prices)
    cheap = prices <= np.quantile(prices, quantile)
    n_cheap = int(cheap.sum())
    if n_cheap in (0, hours) or spread <= 0.0:
        return np.full(hours, nominal)
    largest = min((nominal - low) * hours / n_cheap, (high - nominal) * hours / (hours - n_cheap))
    spread = min(spread, largest)
    lower = nominal - spread * n_cheap / hours
    return np.where(cheap, lower + spread, lower)
