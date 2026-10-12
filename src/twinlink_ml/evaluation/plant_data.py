"""Data generation for a plant variant (study A1).

``experiments.generate_data`` builds the standard grey-box model itself and cannot be given a
different simulator. :func:`generate_plant_data` is the same procedure -- same excitation design,
same initial states, same run-level split, same seeds -- with the simulator being a
:class:`~twinlink_ml.simulation.plant_variants.PlantVariantSimulator`. For P0 the result is
bit-identical to ``generate_data`` (tested).

The result is a :class:`PlantData`, a :class:`~twinlink_ml.evaluation.experiments.GeneratedData`
that additionally knows its plant: its fingerprints name the plant (the process fingerprint
alone does not separate P0 from a plant that only differs in the measurement path or the
fouling), and ``with_measurements`` re-draws the measurements with the plant's sensor path,
not with the standard model's.

New studies pass ``DataSpec.new_study(...)`` as ``spec``; the function does not choose a data
configuration itself.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass

from ..data.splits import IndexBlockSplitter, split_schedule
from ..simulation.config import MeasurementConfig, PlaceholderConfigWarning, ProcessConfig
from ..simulation.excitation import (
    DiurnalDisturbanceDesign,
    ExcitationConfig,
    MixedExcitationDesign,
)
from ..simulation.plant_variants import PlantVariantSimulator, PlantVariantSpec
from ..utils.seeding import SeedBundle
from .experiments import DataSpec, GeneratedData

__all__ = ["PlantData", "generate_plant_data"]


@dataclass(frozen=True, slots=True)
class PlantData(GeneratedData):
    """Data of one plant: :class:`GeneratedData` plus the plant it was generated on."""

    plant: PlantVariantSpec
    base_config: ProcessConfig
    """The config of the simulator the plant deviates from (``process_config`` is the plant's)."""

    @property
    def fingerprints(self) -> dict[str, str]:
        simulator = _simulator(self.base_config, self.plant)
        return {
            **GeneratedData.fingerprints.fget(self),  # type: ignore[attr-defined]
            "plant/name": self.plant.name,
            "plant/fingerprint": simulator.variant_fingerprint,
        }

    def with_measurements(
        self,
        *,
        noise_seed: int | None = None,
        measurement: MeasurementConfig | None = None,
    ) -> PlantData:
        """Same runs, measurements re-drawn through the sensor path of the plant."""
        seeds = (
            self.seeds
            if noise_seed is None
            else dataclasses.replace(self.seeds, noise_seed=noise_seed)
        )
        base = (
            self.base_config
            if measurement is None
            else dataclasses.replace(self.base_config, measurement=measurement)
        )
        simulator = _simulator(base, self.plant)
        measurements = simulator.remeasure(self.trajectory.outputs, seed=seeds.noise_seed)
        trajectory = dataclasses.replace(
            self.trajectory,
            measurements=measurements,
            metadata={**self.trajectory.metadata, "seed": seeds.noise_seed},
        )
        return dataclasses.replace(
            self,
            trajectory=trajectory,
            process_config=simulator.config,
            base_config=base,
            seeds=seeds,
        )


def _simulator(base_config: ProcessConfig, plant: PlantVariantSpec) -> PlantVariantSimulator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return PlantVariantSimulator(base_config, plant)


def generate_plant_data(
    base_config: ProcessConfig,
    plant: PlantVariantSpec,
    spec: DataSpec,
    *,
    master_seed: int | None = None,
    seeds: SeedBundle | None = None,
) -> PlantData:
    """Excitation, simulation on the plant, split -- like ``generate_data``.

    Exactly one of ``master_seed`` and ``seeds``, as there. The excitation is designed on the
    BASE config (the action and disturbance ranges, geometry and material of the design are the
    simulator's, not the plant's): every plant is excited with the same schedule for the same
    seeds, so differences between plants are differences of the plant.
    """
    if (master_seed is None) == (seeds is None):
        raise ValueError("exactly one of master_seed and seeds")
    if seeds is None:
        assert master_seed is not None
        seeds = SeedBundle.from_master(master_seed)
    base = dataclasses.replace(
        base_config,
        numerics=dataclasses.replace(base_config.numerics, sample_time_s=spec.sample_time_s),
    )
    simulator = _simulator(base, plant)

    design = MixedExcitationDesign(
        ExcitationConfig(
            run_duration_s=spec.run_duration_s,
            constant_run_fraction=spec.constant_run_fraction,
        ),
        base.actions,
        base.disturbances,
        sample_time_s=base.numerics.sample_time_s,
        geometry=base.geometry,
        material=base.material,
        operation=base.operation,
        disturbance_design=(
            None
            if spec.solid_fraction_process == "diurnal"
            else DiurnalDisturbanceDesign(solid_fraction_process=spec.solid_fraction_process)
        ),
    )
    schedule = design.build(spec.pool_size, seed=seeds.data_seed)

    initial = None
    if schedule.initial_states is not None:
        columns = schedule.initial_states
        initial = simulator.state_from_physical(
            columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        )
    trajectory = simulator.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=seeds.noise_seed,
        initial_state=initial,
    )
    splitter = IndexBlockSplitter(n_train=spec.n_train_max, n_val=spec.n_val, n_test=spec.n_test)
    return PlantData(
        trajectory=trajectory,
        schedule=schedule,
        split=split_schedule(schedule, splitter),
        process_config=simulator.config,
        spec=spec,
        seeds=seeds,
        plant=plant,
        base_config=base,
    )
