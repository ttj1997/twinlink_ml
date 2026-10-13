"""Measure of the deviation of a plant from the simulator (study A1).

How far is a plant from the standard simulator (P0)? The error of the simulator against the
plant in an OPEN simulation of a fixed reference excitation, per channel and in physical
units:

* the reference excitation is a fixed design (:data:`REFERENCE_RUNS` runs of the new-study
  excitation, 8 h each) drawn with a reserved seed, :data:`REFERENCE_SEED`, which the main run
  must not use for data, split or model;
* both machines are started in the same initial states of the schedule and driven with the same
  actions and disturbances (open simulation, no feedback of measurements);
* two versions: on the process OUTPUTS (no sensor path) and on the MEASURED channels (sensor
  path of each machine, noise switched off);
* the error is the RMSE over all runs and samples, per output channel, in the unit of the
  channel (the energy target in the unit of ``EnergyConfig.target``, bed temperature in degC,
  product moisture in kg/kg, d50 in um);
* the individual channels are the main statement. The summary is the mean over the channels
  the models predict (the targets of the data, d50 excluded unless it is a target), each
  channel normalised with the standard deviation of the P0 reference over all runs and
  samples.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..simulation.backend import FloatArray
from ..simulation.config import PlaceholderConfigWarning, ProcessConfig
from ..simulation.excitation import (
    DiurnalDisturbanceDesign,
    ExcitationConfig,
    MixedExcitationDesign,
)
from ..simulation.plant_variants import PLANT_VARIANTS, PlantVariantSimulator, PlantVariantSpec
from .experiments import DataSpec

__all__ = ["REFERENCE_RUNS", "REFERENCE_SEED", "PlantDeviation", "plant_deviation"]

REFERENCE_SEED = 20_261_012
"""Reserved seed of the reference excitation; not to be used for data, split or model seeds."""
REFERENCE_RUNS = 8


@dataclass(frozen=True, slots=True)
class PlantDeviation:
    """Deviation of one plant from P0, see the module docstring."""

    plant: str
    channels: tuple[str, ...]
    units: dict[str, str]
    summary_channels: tuple[str, ...]
    rmse_outputs: dict[str, float]
    rmse_measured: dict[str, float]
    reference_std_outputs: dict[str, float]
    reference_std_measured: dict[str, float]
    summary_outputs: float
    """Mean over ``summary_channels`` of ``rmse_outputs / reference_std_outputs``."""
    summary_measured: float
    reference_seed: int
    n_runs: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _reference_schedule(base_config: ProcessConfig, spec: DataSpec) -> Any:
    design = MixedExcitationDesign(
        ExcitationConfig(
            run_duration_s=spec.run_duration_s,
            constant_run_fraction=spec.constant_run_fraction,
        ),
        base_config.actions,
        base_config.disturbances,
        sample_time_s=base_config.numerics.sample_time_s,
        geometry=base_config.geometry,
        material=base_config.material,
        operation=base_config.operation,
        gas=base_config.gas,
        gas_velocity_cap=spec.gas_velocity_cap,
        disturbance_design=DiurnalDisturbanceDesign(
            solid_fraction_process=spec.solid_fraction_process
        ),
    )
    return design.build(REFERENCE_RUNS, seed=REFERENCE_SEED)


def _quiet(config: ProcessConfig) -> ProcessConfig:
    return dataclasses.replace(
        config,
        measurement=dataclasses.replace(
            config.measurement, relative_noise_std=0.0, absolute_noise_std_temperature_k=0.0
        ),
    )


def _simulate(
    base_config: ProcessConfig, plant: PlantVariantSpec, schedule: Any
) -> tuple[FloatArray, FloatArray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = PlantVariantSimulator(_quiet(base_config), plant)
    columns = schedule.initial_states
    initial = simulator.state_from_physical(
        columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
    )
    result = simulator.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=0,
        initial_state=initial,
    )
    return result.outputs, result.measurements


def plant_deviation(
    base_config: ProcessConfig,
    plant: PlantVariantSpec,
    spec: DataSpec,
    *,
    summary_channels: tuple[str, ...] | None = None,
) -> PlantDeviation:
    """The deviation of ``plant`` from P0 on the reference excitation.

    :param spec: the data spec of the study (sample time, run duration, solid-fraction
        process, targets); build it with ``DataSpec.new_study``
    :param summary_channels: default ``spec.target_names``
    """
    base = dataclasses.replace(
        base_config,
        numerics=dataclasses.replace(base_config.numerics, sample_time_s=spec.sample_time_s),
    )
    schedule = _reference_schedule(base, spec)
    reference_outputs, reference_measured = _simulate(base, PLANT_VARIANTS["P0"], schedule)
    outputs, measured = _simulate(base, plant, schedule)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        names = PlantVariantSimulator(base, PLANT_VARIANTS["P0"]).output_names
    chosen = spec.target_names if summary_channels is None else summary_channels
    energy_unit = base.energy.target
    units = {
        "energy_target": energy_unit,
        "bed_temperature_celsius": "degC",
        "product_moisture_kg_per_kg": "kg/kg",
        "d50_um": "um",
    }

    def rmse(a: FloatArray, b: FloatArray) -> dict[str, float]:
        return {
            name: float(np.sqrt(np.mean((a[..., i] - b[..., i]) ** 2)))
            for i, name in enumerate(names)
        }

    def spread(a: FloatArray) -> dict[str, float]:
        return {name: float(np.std(a[..., i])) for i, name in enumerate(names)}

    rmse_outputs = rmse(outputs, reference_outputs)
    rmse_measured = rmse(measured, reference_measured)
    std_outputs = spread(reference_outputs)
    std_measured = spread(reference_measured)

    def summary(rmse_by_channel: dict[str, float], std_by_channel: dict[str, float]) -> float:
        return float(np.mean([rmse_by_channel[c] / std_by_channel[c] for c in chosen]))

    return PlantDeviation(
        plant=plant.name,
        channels=tuple(names),
        units={name: units.get(name, "") for name in names},
        summary_channels=tuple(chosen),
        rmse_outputs=rmse_outputs,
        rmse_measured=rmse_measured,
        reference_std_outputs=std_outputs,
        reference_std_measured=std_measured,
        summary_outputs=summary(rmse_outputs, std_outputs),
        summary_measured=summary(rmse_measured, std_measured),
        reference_seed=REFERENCE_SEED,
        n_runs=REFERENCE_RUNS,
    )
