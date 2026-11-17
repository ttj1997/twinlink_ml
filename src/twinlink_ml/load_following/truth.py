"""Truth evaluation with three commands: the unchanged grey-box model on a batch of days.

As :class:`twinlink_ml.headroom.truth.TruthPlant` (same model, same rate-limit rule
``|u[k] - u[k-1]| <= rate * dt`` per truth sample, path condition at every sample), with the spray
rate as a third, rate-limited command instead of a disturbance column. Recorded per control
interval: total and heater energy, the suspension mass sprayed (realised, kg); per sample: product
moisture and bed temperature; per day: the range of the median particle diameter and the smallest
fluidisation margins (diagnostics only).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from twinlink_ml.simulation import GreyBoxFluidBedGranulator, ProcessConfig

from .dynamic import command_rates
from .market import DayScenario

__all__ = ["LoadTruthPlant", "LoadTruthResult"]

_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True, eq=False)
class LoadTruthResult:
    """Record of one controller on a batch of days (first axis = day)."""

    moisture: np.ndarray  # (n, samples), kg/kg
    bed_temperature: np.ndarray  # (n, samples), degC
    moisture_limit: np.ndarray  # (n, samples)
    energy_kwh_per_interval: np.ndarray  # (n, intervals)
    heater_kwh_per_interval: np.ndarray
    sprayed_kg_per_interval: np.ndarray
    commands: np.ndarray  # (n, intervals, 3)
    d50_um_range: np.ndarray  # (n, 2)
    fluidisation_margin_min: np.ndarray  # (n, 2): lower (u - u_mf), upper (u_t - u), m/s
    sample_time_s: float


class LoadTruthPlant:
    """Lockstep simulation of a batch of equally long days."""

    def __init__(
        self,
        config: ProcessConfig,
        days: Sequence[DayScenario],
        initial_states: np.ndarray,
        initial_commands: np.ndarray,
        *,
        interval_s: float,
    ) -> None:
        lengths = {d.n_samples for d in days}
        if len(lengths) != 1:
            raise ValueError("days of one batch must have the same length")
        self._sim = GreyBoxFluidBedGranulator(config)
        self._days = list(days)
        self._dt = days[0].sample_time_s
        if abs(self._dt - config.numerics.sample_time_s) > 1e-12:
            raise ValueError("scenario sample time must equal the model sample time")
        self._spi = round(interval_s / self._dt)
        n_samples = lengths.pop()
        if n_samples % self._spi:
            raise ValueError("day length is not a multiple of the control interval")
        self.n_intervals = n_samples // self._spi
        self._dist = np.stack([d.disturbances for d in days])  # (n, 5, samples)
        self.state = np.array(initial_states, dtype=np.float64)
        self._realised = np.array(initial_commands, dtype=np.float64)
        self._rates = command_rates(config)
        self._nozzle = config.actions.nozzle_pressure_bar.nominal
        n = len(days)
        self._moisture = np.empty((n, n_samples))
        self._bed_t = np.empty((n, n_samples))
        self._energy = np.zeros((n, self.n_intervals))
        self._heater = np.zeros((n, self.n_intervals))
        self._sprayed = np.zeros((n, self.n_intervals))
        self._commands = np.empty((n, self.n_intervals, 3))
        self._d50 = np.stack([np.full(n, np.inf), np.full(n, -np.inf)], axis=1)
        self._fluid = np.full((n, 2), np.inf)
        self.k = 0

    def current_disturbance(self) -> np.ndarray:
        """Disturbance sample at the start of the current interval, ``(n, 5)``."""
        return self._dist[:, :, self.k * self._spi]

    @property
    def last_commands(self) -> np.ndarray:
        """Rate-limited commands at the end of the last sample, ``(n, 3)``."""
        return self._realised.copy()

    @property
    def sprayed_so_far(self) -> np.ndarray:
        """Suspension mass sprayed up to the current interval, kg, ``(n,)``."""
        return self._sprayed[:, : self.k].sum(axis=1)

    def advance(self, commands: np.ndarray) -> None:
        """Simulate one control interval with the given commands ``(n, 3)``."""
        if self.k >= self.n_intervals:
            raise RuntimeError("day finished")
        commands = np.asarray(commands, dtype=np.float64)
        self._commands[:, self.k] = commands
        limit = self._rates * self._dt
        hours = self._dt / _SECONDS_PER_HOUR
        for j in range(self._spi):
            index = self.k * self._spi + j
            self._realised = self._realised + np.clip(commands - self._realised, -limit, limit)
            d = self._dist[:, :, index]
            action = np.stack(
                [
                    self._realised[:, 0] + d[:, 4],
                    self._realised[:, 1],
                    self._realised[:, 2],
                    np.full(d.shape[0], self._nozzle),
                ],
                axis=-1,
            )
            result = self._sim.step(self.state, action, d[:, 0:3])
            self.state = np.asarray(result.state)
            diag = result.diagnostics
            self._moisture[:, index] = diag["particle_moisture_kg_per_kg"]
            self._bed_t[:, index] = self.state[:, 3]
            self._energy[:, self.k] += np.asarray(diag["power_total_kw"]) * hours
            self._heater[:, self.k] += np.asarray(diag["power_heater_kw"]) * hours
            self._sprayed[:, self.k] += self._realised[:, 2] * hours
            d50 = np.asarray(diag["d50_m"]) * 1e6
            self._d50[:, 0] = np.minimum(self._d50[:, 0], d50)
            self._d50[:, 1] = np.maximum(self._d50[:, 1], d50)
            self._fluid[:, 0] = np.minimum(
                self._fluid[:, 0], np.asarray(diag["fluidization_margin_lower_m_per_s"])
            )
            self._fluid[:, 1] = np.minimum(
                self._fluid[:, 1], np.asarray(diag["fluidization_margin_upper_m_per_s"])
            )
        self.k += 1

    def result(self) -> LoadTruthResult:
        """Record of the finished run."""
        if self.k != self.n_intervals:
            raise RuntimeError("run not finished")
        return LoadTruthResult(
            moisture=self._moisture,
            bed_temperature=self._bed_t,
            moisture_limit=np.stack([d.moisture_limit for d in self._days]),
            energy_kwh_per_interval=self._energy,
            heater_kwh_per_interval=self._heater,
            sprayed_kg_per_interval=self._sprayed,
            commands=self._commands,
            d50_um_range=self._d50,
            fluidisation_margin_min=self._fluid,
            sample_time_s=self._dt,
        )
