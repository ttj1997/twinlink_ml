r"""Measured process responses for the soft sensor F5, and its training data (docs/29).

Three responses of the grey-box model can be measured on a plant besides the commands and the
disturbances:

``bed_temperature_celsius``
    the bed temperature (state, output channel 1 of :class:`GreyBoxFluidBedGranulator`);
``exhaust_temperature_celsius``
    the outlet temperature of :func:`~twinlink_ml.simulation.exhaust.reconstruct_exhaust_state`,
    ``bed temperature - exhaust.outlet_duct_heat_loss_k`` (well-mixed bed, model assumption; the
    loss is 0.0 in the configs of this study, so the exhaust temperature equals the bed
    temperature and differs only by its own sensor noise);
``exhaust_absolute_humidity_kg_per_kg``
    the absolute humidity of the exhaust air (diagnostic ``outlet_humidity_kg_per_kg`` of the
    model, at the end of each sample).

All three are measured with the EXISTING sensor model (:meth:`GreyBoxFluidBedGranulator.measure`,
``MeasurementConfig`` of the process config: PT1 with ``sensor_time_constant_s``, relative Gaussian
noise, plus the absolute temperature noise on temperature channels): the bed temperature exactly as
channel 1 of the model's outputs; exhaust temperature and humidity as channels 1 and 2 of a second,
independent sensor instance (temperature and moisture-type channel of the sensor model).

* Training data: :func:`with_exhaust_channels` appends the two exhaust channels (noise-free and
  measured) to the outputs of a generated pool; the bed temperature is the pool's existing
  measurement. :func:`randomised_data` re-simulates a pool with one parameter draw per run
  (domain randomisation, F5-DR) on the same excitation, initial states and split.
* Closed loop: :class:`ResponseSensor` measures the true responses at the end of every 25 s block
  of one pair, with its own per-pair random streams.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence

import numpy as np

from twinlink_ml.evaluation.experiments import GeneratedData
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    MeasurementConfig,
    PlaceholderConfigWarning,
    ProcessConfig,
)
from twinlink_ml.simulation.base import Trajectory

__all__ = [
    "EXHAUST_HUMIDITY",
    "EXHAUST_TEMPERATURE",
    "RESPONSE_NAMES",
    "ResponseSensor",
    "randomised_data",
    "with_exhaust_channels",
]

BED_TEMPERATURE = "bed_temperature_celsius"
EXHAUST_TEMPERATURE = "exhaust_temperature_celsius"
EXHAUST_HUMIDITY = "exhaust_absolute_humidity_kg_per_kg"
RESPONSE_NAMES: tuple[str, ...] = (BED_TEMPERATURE, EXHAUST_TEMPERATURE, EXHAUST_HUMIDITY)
"""Order of the true responses handed over by the closed loop."""

_TEMPERATURE_CHANNEL = 1  # channel of the sensor model with the absolute temperature noise
_MOISTURE_CHANNEL = 2  # moisture-type channel (relative noise only)


def _pseudo_outputs(temperature: np.ndarray, humidity: np.ndarray | None) -> np.ndarray:
    out = np.zeros((*np.shape(temperature), 4))
    out[..., _TEMPERATURE_CHANNEL] = temperature
    if humidity is not None:
        out[..., _MOISTURE_CHANNEL] = humidity
    return out


def with_exhaust_channels(data: GeneratedData, noise_seed: int) -> GeneratedData:
    """The pool with exhaust temperature and humidity appended to outputs and measurements.

    The measured values come from :meth:`remeasure` of a sensor model with the pool's own process
    config (sample time and ``MeasurementConfig``), seeded with ``noise_seed`` (independent of the
    pool's noise seed); the existing outputs, measurements and everything else stay unchanged.
    """
    trajectory = data.trajectory
    names = tuple(trajectory.metadata["output_names"])
    if EXHAUST_TEMPERATURE in names or EXHAUST_HUMIDITY in names:
        raise ValueError("the pool already has exhaust channels")
    bed = trajectory.outputs[..., names.index(BED_TEMPERATURE)]
    exhaust_t = bed - data.process_config.exhaust.outlet_duct_heat_loss_k
    exhaust_y = np.asarray(trajectory.diagnostics["outlet_humidity_kg_per_kg"], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        sensor = GreyBoxFluidBedGranulator(data.process_config)
    measured = sensor.remeasure(_pseudo_outputs(exhaust_t, exhaust_y), seed=noise_seed)
    outputs = np.concatenate([trajectory.outputs, np.stack([exhaust_t, exhaust_y], -1)], axis=-1)
    measurements = np.concatenate(
        [trajectory.measurements, measured[..., [_TEMPERATURE_CHANNEL, _MOISTURE_CHANNEL]]],
        axis=-1,
    )
    metadata = {
        **trajectory.metadata,
        "output_names": (*names, EXHAUST_TEMPERATURE, EXHAUST_HUMIDITY),
        "exhaust_noise_seed": noise_seed,
    }
    return dataclasses.replace(
        data,
        trajectory=dataclasses.replace(
            trajectory, outputs=outputs, measurements=measurements, metadata=metadata
        ),
    )


def randomised_data(
    data: GeneratedData, run_configs: Sequence[ProcessConfig], noise_seeds: Sequence[int]
) -> GeneratedData:
    """The pool re-simulated with one process config per run (domain randomisation).

    Same requested actions, disturbances and initial states as the pool's excitation schedule
    (:func:`~twinlink_ml.evaluation.experiments.generate_data`), same split and spec; run ``r`` is
    simulated alone with ``run_configs[r]`` (sample time of the pool) and sensor seed
    ``noise_seeds[r]``. ``process_config`` of the result stays the pool's (the nominal one); the
    per-run configs are recorded by the caller.
    """
    schedule = data.schedule
    n_runs = len(schedule.run_id)
    if len(run_configs) != n_runs or len(noise_seeds) != n_runs:
        raise ValueError("one config and one noise seed per run")
    numerics = data.process_config.numerics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        nominal = GreyBoxFluidBedGranulator(data.process_config)
        initial = None
        if schedule.initial_states is not None:
            c = schedule.initial_states
            initial = nominal.state_from_physical(c[:, 0], c[:, 1], c[:, 2], c[:, 3])
        parts: list[Trajectory] = []
        for r, (config, seed) in enumerate(zip(run_configs, noise_seeds, strict=True)):
            if config.numerics.sample_time_s != numerics.sample_time_s:
                config = dataclasses.replace(
                    config,
                    numerics=dataclasses.replace(
                        config.numerics, sample_time_s=numerics.sample_time_s
                    ),
                )
            simulator = GreyBoxFluidBedGranulator(config)
            parts.append(
                simulator.rollout(
                    schedule.actions[r : r + 1],
                    schedule.disturbances[r : r + 1],
                    run_id=(schedule.run_id[r],),
                    seed=int(seed),
                    initial_state=None if initial is None else initial[r : r + 1],
                )
            )
    first = parts[0]
    stacked = Trajectory(
        run_id=tuple(p.run_id[0] for p in parts),
        time_s=first.time_s,
        actions=np.concatenate([p.actions for p in parts]),
        disturbances=np.concatenate([p.disturbances for p in parts]),
        states=np.concatenate([p.states for p in parts]),
        outputs=np.concatenate([p.outputs for p in parts]),
        measurements=np.concatenate([p.measurements for p in parts]),
        diagnostics={
            k: np.concatenate([p.diagnostics[k] for p in parts]) for k in first.diagnostics
        },
        metadata={**first.metadata, "seed": "per run", "domain_randomised": True},
        requested_actions=(
            None
            if first.requested_actions is None
            else np.concatenate([p.requested_actions for p in parts])  # type: ignore[misc]
        ),
        projection_active=first.projection_active,
        n_projected_steps=(
            None
            if first.n_projected_steps is None
            else np.concatenate([p.n_projected_steps for p in parts])  # type: ignore[misc]
        ),
    )
    return dataclasses.replace(data, trajectory=stacked)


class ResponseSensor:
    """The existing sensor model on the measured responses of one pair, on the feedback grid.

    ``names``: the responses the soft sensor uses, a subsequence of :data:`RESPONSE_NAMES` in that
    order. Two sensor instances (as in :func:`with_exhaust_channels`): the bed temperature as
    channel 1, exhaust temperature and humidity as channels 1 and 2 of the second; each has its
    own random generator (seeds derived per pair by the caller), so a pair sees the same noise at
    every margin.
    """

    def __init__(
        self,
        process: ProcessConfig,
        measurement: MeasurementConfig,
        interval_s: float,
        names: Sequence[str],
        *,
        bed_seed: int,
        exhaust_seed: int,
    ) -> None:
        names = tuple(names)
        if not names or any(n not in RESPONSE_NAMES for n in names):
            raise ValueError(f"responses must be among {RESPONSE_NAMES}")
        if list(names) != [n for n in RESPONSE_NAMES if n in names]:
            raise ValueError("responses must keep the order of RESPONSE_NAMES")
        if not measurement.enabled:
            raise ValueError("the response sensor needs an enabled measurement config")
        numerics = dataclasses.replace(process.numerics, sample_time_s=interval_s)
        config = dataclasses.replace(process, numerics=numerics, measurement=measurement)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            self._bed = GreyBoxFluidBedGranulator(config)
            self._exhaust = GreyBoxFluidBedGranulator(config)
        self._bed_rng = np.random.default_rng(bed_seed)
        self._exhaust_rng = np.random.default_rng(exhaust_seed)
        self.names = names
        self._columns = [RESPONSE_NAMES.index(n) for n in names]

    def measure(self, true_responses: np.ndarray) -> np.ndarray:
        """Measured values ``(m, len(names))`` from true responses ``(m, 3)``, blocks in order.

        Both sensors advance at every block (also for a response that is not used), so the
        random streams do not depend on the subset of names.
        """
        values = np.asarray(true_responses, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(RESPONSE_NAMES):
            raise ValueError(f"true responses must have shape (m, {len(RESPONSE_NAMES)})")
        out = np.empty((len(values), len(RESPONSE_NAMES)))
        for j, (bed, exhaust_t, exhaust_y) in enumerate(values):
            b = np.asarray(self._bed.measure(_pseudo_outputs(np.array([bed]), None), self._bed_rng))
            e = np.asarray(
                self._exhaust.measure(
                    _pseudo_outputs(np.array([exhaust_t]), np.array([exhaust_y])),
                    self._exhaust_rng,
                )
            )
            out[j] = (
                b[0, _TEMPERATURE_CHANNEL],
                e[0, _TEMPERATURE_CHANNEL],
                e[0, _MOISTURE_CHANNEL],
            )
        return out[:, self._columns]
