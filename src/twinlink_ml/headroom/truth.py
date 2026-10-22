"""Truth evaluation: the unchanged grey-box model (4 states, RK4) run on a batch of scenarios.

Every controller is judged here and only here. Per control interval the controllers hand over a
commanded inlet temperature and gas velocity; per truth sample (5 s) the command is rate limited
exactly as :func:`~twinlink_ml.simulation.excitation.apply_rate_limit` does (``|u[k] - u[k-1]| <=
rate * dt``), the realised inlet temperature is the limited command plus the scenario's offset,
and :meth:`GreyBoxFluidBedGranulator.step` advances the state (internal RK4 substeps of the
config). The path condition is checked at EVERY sample on the product moisture
(``particle_moisture_kg_per_kg``, continuous product stream = bed moisture).

Energy: model total power (``power_total_kw``, the objective of the bandit studies) at the end of
each sample times the sample time. CO2: :func:`~twinlink_ml.evaluation.carrier_energy.
carrier_energy` booking with a gas-fired heater (electricity 0.366, gas 0.201 kg/kWh).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from twinlink_ml.evaluation.net_energy import ELECTRICITY_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH
from twinlink_ml.simulation import GreyBoxFluidBedGranulator, ProcessConfig

from .scenarios import Scenario

__all__ = ["TruthPlant", "TruthResult"]

_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True, eq=False)
class TruthResult:
    """Per-scenario record of one controller run (arrays: first axis = scenario)."""

    moisture: np.ndarray  # (n, samples), kg/kg
    bed_temperature: np.ndarray  # (n, samples), degC
    moisture_limit: np.ndarray  # (n, samples)
    energy_kwh_per_interval: np.ndarray  # (n, intervals)
    electric_kwh_per_interval: np.ndarray
    gas_kwh_per_interval: np.ndarray
    commands: np.ndarray  # (n, intervals, 2)
    sample_time_s: float

    @property
    def energy_kwh(self) -> np.ndarray:
        """Total energy per scenario, kWh."""
        return self.energy_kwh_per_interval.sum(axis=1)

    @property
    def co2_kg(self) -> np.ndarray:
        """CO2 per scenario, kg CO2e (gas-fired heater)."""
        return ELECTRICITY_CO2_KG_PER_KWH * self.electric_kwh_per_interval.sum(
            axis=1
        ) + GAS_CO2_KG_PER_KWH * self.gas_kwh_per_interval.sum(axis=1)

    @property
    def max_violation(self) -> np.ndarray:
        """Largest exceedance of the moisture limit per scenario, kg/kg (>= 0)."""
        return np.maximum(self.moisture - self.moisture_limit, 0.0).max(axis=1)


class TruthPlant:
    """Lockstep simulation of a batch of equally long scenarios."""

    def __init__(
        self,
        config: ProcessConfig,
        scenarios: Sequence[Scenario],
        initial_states: np.ndarray,
        initial_commands: np.ndarray,
        *,
        interval_s: float,
        gas_heater_efficiency: float,
    ) -> None:
        lengths = {s.n_samples for s in scenarios}
        if len(lengths) != 1:
            raise ValueError("scenarios of one batch must have the same length")
        self._config = config
        self._sim = GreyBoxFluidBedGranulator(config)
        self._scenarios = list(scenarios)
        self._dt = scenarios[0].sample_time_s
        if abs(self._dt - config.numerics.sample_time_s) > 1e-12:
            raise ValueError("scenario sample time must equal the model sample time")
        self._spi = round(interval_s / self._dt)
        n_samples = lengths.pop()
        if n_samples % self._spi:
            raise ValueError("scenario length is not a multiple of the control interval")
        self.n_intervals = n_samples // self._spi
        self._dist = np.stack([s.disturbances for s in scenarios])  # (n, 5, samples)
        self.state = np.array(initial_states, dtype=np.float64)
        self._realised = np.array(initial_commands, dtype=np.float64)
        self._rates = np.array(
            [
                config.actions.inlet_temperature_celsius.max_rate_per_s,
                config.actions.gas_velocity_m_per_s.max_rate_per_s,
            ]
        )
        self._nozzle = config.actions.nozzle_pressure_bar.nominal
        self._process_eta = config.energy.heater_efficiency
        self._gas_eta = gas_heater_efficiency
        n = len(scenarios)
        self._moisture = np.empty((n, n_samples))
        self._bed_t = np.empty((n, n_samples))
        self._energy = np.zeros((n, self.n_intervals))
        self._electric = np.zeros((n, self.n_intervals))
        self._gas = np.zeros((n, self.n_intervals))
        self._commands = np.empty((n, self.n_intervals, 2))
        self.k = 0

    @property
    def scenarios(self) -> list[Scenario]:
        """The scenarios of the batch."""
        return self._scenarios

    def current_disturbance(self) -> np.ndarray:
        """Disturbance sample at the start of the current interval, ``(n, 5)``."""
        return self._dist[:, :, self.k * self._spi]

    def current_limit(self) -> np.ndarray:
        """Moisture limit at the start of the current interval, ``(n,)``."""
        return np.array([s.moisture_limit[self.k * self._spi] for s in self._scenarios])

    @property
    def last_commands(self) -> np.ndarray:
        """Rate-limited commands at the end of the last sample, ``(n, 2)``."""
        return self._realised.copy()

    def advance(self, commands: np.ndarray) -> None:
        """Simulate one control interval with the given commands ``(n, 2)``."""
        if self.k >= self.n_intervals:
            raise RuntimeError("scenario finished")
        commands = np.asarray(commands, dtype=np.float64)
        self._commands[:, self.k] = commands
        limit = self._rates * self._dt
        for j in range(self._spi):
            index = self.k * self._spi + j
            self._realised = self._realised + np.clip(commands - self._realised, -limit, limit)
            d = self._dist[:, :, index]
            action = np.stack(
                [
                    self._realised[:, 0] + d[:, 4],
                    self._realised[:, 1],
                    d[:, 3],
                    np.full(d.shape[0], self._nozzle),
                ],
                axis=-1,
            )
            result = self._sim.step(self.state, action, d[:, 0:3])
            self.state = np.asarray(result.state)
            diag = result.diagnostics
            self._moisture[:, index] = diag["particle_moisture_kg_per_kg"]
            self._bed_t[:, index] = self.state[:, 3]
            hours = self._dt / _SECONDS_PER_HOUR
            self._energy[:, self.k] += np.asarray(diag["power_total_kw"]) * hours
            electric = (
                np.asarray(diag["power_fan_kw"])
                + np.asarray(diag["power_pump_kw"])
                + np.asarray(diag["power_atomiser_kw"])
            )
            gas = np.asarray(diag["power_heater_kw"]) * self._process_eta / self._gas_eta
            self._electric[:, self.k] += electric * hours
            self._gas[:, self.k] += gas * hours
        self.k += 1

    @property
    def moisture_so_far(self) -> np.ndarray:
        """Product moisture of the simulated samples, ``(n, k * samples_per_interval)``."""
        return self._moisture[:, : self.k * self._spi]

    def result(self) -> TruthResult:
        """Record of the finished run."""
        if self.k != self.n_intervals:
            raise RuntimeError("run not finished")
        return TruthResult(
            moisture=self._moisture,
            bed_temperature=self._bed_t,
            moisture_limit=np.stack([s.moisture_limit for s in self._scenarios]),
            energy_kwh_per_interval=self._energy,
            electric_kwh_per_interval=self._electric,
            gas_kwh_per_interval=self._gas,
            commands=self._commands,
            sample_time_s=self._dt,
        )
