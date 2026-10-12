"""Hybrid model and simulator-only predictor (study A1).

The hybrid model gets the open simulation of the grey-box model (P0) as additional input
channels: for every run, the simulator is driven with the plant's inputs (actions and
disturbances) and its prediction of each target is appended to the known exogenous inputs of
every window, in the history and over the horizon. The network itself is an ordinary
:class:`~twinlink_ml.models.transfer.TransferSequenceModel` with more exogenous channels (its
residual head with zero start still begins as persistence).

What the channel is
-------------------
* **Inputs only.** The simulator sees the set points (planned actions) and the disturbances of
  the run, never a measurement of the plant. The channel is therefore as legitimately "known in
  advance" as the planned set points.
* **Initial state.** The state of the plant at the start of a run is unknown. The simulator
  starts in the STEADY STATE at the set points and disturbances at the start of the run
  (:func:`stationary_initial_states`, the solver of ``simulation.steady_state`` -- inputs only,
  no plant measurement). Where the solver fails the run starts in the nominal state; the number
  of such runs is recorded (``SimulatorChannelInfo``).
* **Value.** The noise-free MEASUREMENT of P0: its outputs through its own sensor path with
  the noise switched off. This is what the standard simulator predicts as a reading, and the
  same series is the prediction of :class:`SimulatorOnlyModel`.
* **Cost.** The channel depends on the inputs only: it is computed once per data pool, not per
  learning-curve point and seed.

The channels are appended to ``Trajectory.disturbances`` (names ``sim_<target>``), so the
unchanged ``TrajectoryWindowDataset.from_trajectory`` windows and standardises them like any
exogenous input.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from ..data.windows import SequenceDataset, Standardizer, WindowBatch
from ..evaluation.experiments import GeneratedData
from ..simulation.backend import FloatArray
from ..simulation.base import Trajectory
from ..simulation.config import PlaceholderConfigWarning, ProcessConfig
from ..simulation.greybox import GreyBoxFluidBedGranulator
from ..simulation.operating_point import DEFAULT_SEARCH_MAX_ITERATIONS, DEFAULT_SEARCH_TOLERANCE
from ..simulation.steady_state import solve_steady_state
from .base import SequenceModel

__all__ = [
    "SIM_PREFIX",
    "SimulatorChannelInfo",
    "SimulatorOnlyModel",
    "simulator_channels",
    "stationary_initial_states",
    "with_simulator_channels",
]

SIM_PREFIX = "sim_"

D = TypeVar("D", bound=GeneratedData)


@dataclass(frozen=True, slots=True)
class SimulatorChannelInfo:
    """Record of one :func:`simulator_channels` computation."""

    n_runs: int
    n_stationary: int
    """Runs started in the steady state at their set points."""
    n_nominal_fallback: int
    """Runs started in the nominal state because the steady-state solver failed."""
    n_infeasible_stationary: int
    """Of ``n_stationary``: the solution is flagged infeasible by the solver (still used)."""
    channel_names: tuple[str, ...]
    solver_max_iterations: int
    solver_tolerance: float
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class _SetpointSimulator(GreyBoxFluidBedGranulator):
    """The grey-box model whose "nominal" point is a given action and disturbance.

    ``solve_steady_state`` solves at the nominal point of the simulator it is given; this lets
    it solve at any set point without a change to the solver.
    """

    def __init__(self, config: ProcessConfig) -> None:
        super().__init__(config)
        self._point: tuple[FloatArray, FloatArray] | None = None

    def at(self, action: FloatArray, disturbance: FloatArray) -> _SetpointSimulator:
        self._point = (
            np.asarray(action, dtype=np.float64).reshape(1, -1),
            np.asarray(disturbance, dtype=np.float64).reshape(1, -1),
        )
        return self

    def nominal_action(self, n_batch: int = 1) -> FloatArray:
        if self._point is None:
            return super().nominal_action(n_batch)
        return np.tile(self._point[0], (n_batch, 1))

    def nominal_disturbance(self, n_batch: int = 1) -> FloatArray:
        if self._point is None:
            return super().nominal_disturbance(n_batch)
        return np.tile(self._point[1], (n_batch, 1))


def stationary_initial_states(
    config: ProcessConfig,
    actions_at_start: FloatArray,
    disturbances_at_start: FloatArray,
    *,
    max_iterations: int = DEFAULT_SEARCH_MAX_ITERATIONS,
    tolerance: float = DEFAULT_SEARCH_TOLERANCE,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Steady state at the given set points and disturbances, per run.

    :param actions_at_start, disturbances_at_start: ``(n_runs, n_action)`` /
        ``(n_runs, n_disturbance)``, the values of the first sample of every run
    :returns: ``(states (n_runs, 4), solved (n_runs,) bool, feasible (n_runs,) bool)``; for a
        run where the solver fails the state is the nominal initial state and ``solved`` False
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = _SetpointSimulator(config)
    nominal = simulator.backend.to_numpy(simulator.initial_state(1))[0]
    n_runs = actions_at_start.shape[0]
    states = np.tile(nominal, (n_runs, 1))
    solved = np.zeros(n_runs, dtype=bool)
    feasible = np.zeros(n_runs, dtype=bool)
    for run in range(n_runs):
        try:
            solution = solve_steady_state(
                simulator.at(actions_at_start[run], disturbances_at_start[run]),
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
        except RuntimeError:
            continue
        states[run] = solution.state
        solved[run] = True
        feasible[run] = bool(solution.feasible)
    return states, solved, feasible


def simulator_channels(
    base_config: ProcessConfig,
    trajectory: Trajectory,
    target_names: tuple[str, ...],
) -> tuple[FloatArray, SimulatorChannelInfo]:
    """The noise-free P0 measurement of ``target_names`` for every run of ``trajectory``.

    :param base_config: config of the standard simulator (P0), with the sample time of the
        trajectory
    :returns: ``(channels (n_runs, n_steps, n_targets), info)``
    """
    import warnings

    started = time.perf_counter()
    # The simulator is driven with the SET POINTS (the requested actions where the plant
    # recorded them, i.e. with an active safety projection): the simulator applies its own
    # projection at its own state. Without a projection both series are identical.
    set_points = (
        trajectory.requested_actions
        if trajectory.requested_actions is not None
        else trajectory.actions
    )
    states, solved, feasible = stationary_initial_states(
        base_config, set_points[:, 0, :], trajectory.disturbances[:, 0, :]
    )
    quiet = dataclasses.replace(
        base_config,
        measurement=dataclasses.replace(
            base_config.measurement,
            relative_noise_std=0.0,
            absolute_noise_std_temperature_k=0.0,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(quiet)
    result = simulator.rollout(
        set_points,
        trajectory.disturbances,
        run_id=trajectory.run_id,
        seed=0,
        initial_state=simulator.backend.asarray(states),
    )
    columns = [simulator.output_names.index(name) for name in target_names]
    channels = result.measurements[..., columns]
    info = SimulatorChannelInfo(
        n_runs=int(states.shape[0]),
        n_stationary=int(solved.sum()),
        n_nominal_fallback=int((~solved).sum()),
        n_infeasible_stationary=int((solved & ~feasible).sum()),
        channel_names=tuple(f"{SIM_PREFIX}{name}" for name in target_names),
        solver_max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
        solver_tolerance=DEFAULT_SEARCH_TOLERANCE,
        seconds=time.perf_counter() - started,
    )
    return channels, info


def with_simulator_channels(data: D, *, base_config: ProcessConfig | None = None) -> D:
    """The data of ``data`` with the simulator channels of its targets as exogenous inputs.

    ``data`` may be a ``GeneratedData`` or a ``PlantData``; the same type is returned. The
    simulator is the standard one (P0): ``base_config`` defaults to ``data.base_config`` for a
    plant, else to ``data.process_config``. The :class:`SimulatorChannelInfo` is stored in
    ``trajectory.metadata["simulator_channels"]``.
    """
    trajectory = data.trajectory
    names = tuple(trajectory.metadata.get("disturbance_names", ()))
    if any(name.startswith(SIM_PREFIX) for name in names):
        raise ValueError("the trajectory already has simulator channels")
    if base_config is None:
        base_config = getattr(data, "base_config", None) or data.process_config
    channels, info = simulator_channels(base_config, trajectory, data.spec.target_names)
    augmented = dataclasses.replace(
        trajectory,
        disturbances=np.concatenate([trajectory.disturbances, channels], axis=-1),
        metadata={
            **trajectory.metadata,
            "disturbance_names": names + info.channel_names,
            "simulator_channels": info.as_dict(),
        },
    )
    return dataclasses.replace(data, trajectory=augmented)


class SimulatorOnlyModel(SequenceModel):
    """Predicts each target as the simulator's channel of the data -- no training.

    ``fit`` only takes the scalers of the dataset (to convert the channel, which is
    standardised like any input, to the standardised target). Needs a dataset built from
    :func:`with_simulator_channels`.
    """

    def __init__(self) -> None:
        self._channel_index: tuple[int, ...] = ()
        self._scalers: tuple[Standardizer, Standardizer] | None = None

    @property
    def name(self) -> str:
        return "simulator_only"

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        del seed, validation
        names = train.exogenous_names
        try:
            self._channel_index = tuple(
                names.index(f"{SIM_PREFIX}{target}") for target in train.target_names
            )
        except ValueError:
            raise ValueError(
                "the dataset has no simulator channels for its targets; "
                "build the data with with_simulator_channels"
            ) from None
        self._scalers = (train.input_scaler, train.target_scaler)

    def predict(self, batch: WindowBatch) -> FloatArray:
        if self._scalers is None:
            raise RuntimeError("model not fitted")
        input_scaler, target_scaler = self._scalers
        index = np.asarray(self._channel_index, dtype=np.intp)
        simulated = (
            batch.future_exogenous[..., index] * input_scaler.scale[index]
            + (input_scaler.mean[index])
        )
        return np.asarray(target_scaler.transform(simulated), dtype=np.float64)
