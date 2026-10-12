r"""Growing data pool for active learning and the learning curves of study B1.

``generate_data`` is bound to the pool size: it builds one schedule, simulates all of it and
splits it by index blocks. Active learning needs the opposite: a fixed set of *candidate plans*
(inputs only) from which runs are added round by round, and the truth of a candidate must not
exist before it is chosen. :class:`GrowingPool` holds the candidates as an
``ExcitationSchedule`` (actions, disturbances, initial states: inputs) and simulates a run only
in :meth:`GrowingPool.add`. :meth:`GrowingPool.candidate_inputs` returns inputs only; there is
no accessor for the outputs of an unselected candidate, so an acquisition function cannot use
the truth of a candidate by construction.

Same simulation path as ``generate_data``
-----------------------------------------
:func:`prepare_process_config` and :func:`simulate_schedule` repeat the 15 lines of
``generate_data`` (sample time override, simulator, initial states, rollout) instead of
changing that shared function. The test ``test_full_pool_equals_generate_data`` pins that the
full pool built in chunks is bit-identical to ``generate_data`` (run ids, actions, states,
outputs, disturbances, and the measurements of :meth:`GrowingPool.legacy_trajectory`).

Sensor noise
------------
The sensor noise of ``rollout`` is one random stream over the whole batch, drawn per time step:
the noise of a run depends on which runs share its batch. Chunked simulation therefore cannot
reproduce the noise of ``generate_data`` run by run, and a pool that redraws the noise of all
runs whenever it grows would change the data of earlier rounds. The pool draws the noise **per
run** from a stream seeded by ``(noise_seed, run_id)``: a run keeps its measurements however
the pool grows. ``legacy_trajectory`` reproduces the batch stream of ``generate_data`` for
the test only and is not used by a study.
"""

from __future__ import annotations

import dataclasses
import hashlib
import warnings
from collections.abc import Mapping, Sequence
from functools import reduce

import numpy as np

from ..data.splits import RunSplit, SplitName
from ..data.windows import FloatArray, TrajectoryWindowDataset, WindowConfig
from ..simulation.base import Trajectory
from ..simulation.config import PlaceholderConfigWarning, ProcessConfig
from ..simulation.excitation import ExcitationSchedule
from ..simulation.greybox import GreyBoxFluidBedGranulator
from .experiments import DataSpec

__all__ = [
    "GrowingPool",
    "concatenate_trajectories",
    "prepare_process_config",
    "simulate_schedule",
    "window_dataset",
    "windows_for_test_set",
]

_UNUSED_ID = "__unused__"


def prepare_process_config(process_config: ProcessConfig, spec: DataSpec) -> ProcessConfig:
    """The config as ``generate_data`` runs it: the sample time of the spec."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return dataclasses.replace(
            process_config,
            numerics=dataclasses.replace(process_config.numerics, sample_time_s=spec.sample_time_s),
        )


def _simulator(config: ProcessConfig) -> GreyBoxFluidBedGranulator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return GreyBoxFluidBedGranulator(config)


def simulate_schedule(
    config: ProcessConfig,
    schedule: ExcitationSchedule,
    *,
    noise_seed: int,
    initial_states: FloatArray | None = None,
) -> Trajectory:
    """Rollout of a schedule exactly as ``generate_data`` does it (batch noise stream).

    ``config`` must already carry the sample time (:func:`prepare_process_config`).
    ``initial_states`` (state vectors, not physical columns) override the schedule's physical
    initial conditions, used by test set T3 (start in a steady state).
    """
    simulator = _simulator(config)
    initial = None
    if initial_states is not None:
        initial = simulator.backend.asarray(np.asarray(initial_states, dtype=np.float64))
    elif schedule.initial_states is not None:
        columns = schedule.initial_states
        initial = simulator.state_from_physical(
            columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        )
    return simulator.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=noise_seed,
        initial_state=initial,
    )


def _per_run_seed(noise_seed: int, run_id: str) -> int:
    digest = hashlib.sha256(f"{noise_seed}:{run_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def concatenate_trajectories(parts: Sequence[Trajectory]) -> Trajectory:
    """Runs of several trajectories (same time grid) in one, in the order given."""
    if not parts:
        raise ValueError("nothing to concatenate")
    return reduce(lambda left, right: left.concatenate(right), parts)


def window_dataset(
    trajectory: Trajectory,
    window: WindowConfig,
    *,
    role: SplitName,
    scalers: tuple | None = None,  # type: ignore[type-arg]
) -> TrajectoryWindowDataset:
    """Windows of all runs of ``trajectory`` as the TRAIN or TEST role.

    ``from_trajectory`` selects runs through a ``RunSplit``; this wrapper puts every run into
    the requested role (and a placeholder id into the other, which is never looked up).
    Test sets must be given the scalers of the training set.
    """
    ids = tuple(trajectory.run_id)
    if role is SplitName.TRAIN:
        split = RunSplit(train=ids, val=(), test=(_UNUSED_ID,), splitter_digest="all-train")
    elif role is SplitName.TEST:
        split = RunSplit(train=(_UNUSED_ID,), val=(), test=ids, splitter_digest="all-test")
    else:
        raise ValueError("role must be TRAIN or TEST")
    return TrajectoryWindowDataset.from_trajectory(trajectory, split, role, window, scalers=scalers)


class GrowingPool:
    """Candidate plans (inputs) plus the runs chosen so far (inputs and truth)."""

    def __init__(
        self,
        process_config: ProcessConfig,
        spec: DataSpec,
        candidates: ExcitationSchedule,
        *,
        noise_seed: int,
    ) -> None:
        self._config = prepare_process_config(process_config, spec)
        self._spec = spec
        self._candidates = candidates
        self._noise_seed = noise_seed
        self._selected: list[int] = []
        self._runs: dict[int, Trajectory] = {}

    # ------------------------------------------------------------------ inputs only

    @property
    def n_candidates(self) -> int:
        return self._candidates.n_runs

    @property
    def selected(self) -> tuple[int, ...]:
        """Indices of the chosen candidates, in the order they were added."""
        return tuple(self._selected)

    @property
    def remaining(self) -> tuple[int, ...]:
        chosen = set(self._selected)
        return tuple(i for i in range(self.n_candidates) if i not in chosen)

    def candidate_inputs(self, indices: Sequence[int]) -> ExcitationSchedule:
        """Actions, disturbances and initial conditions of candidates. Inputs only."""
        c = self._candidates
        rows = list(indices)
        return ExcitationSchedule(
            run_id=tuple(c.run_id[i] for i in rows),
            actions=c.actions[rows],
            disturbances=c.disturbances[rows],
            initial_states=None if c.initial_states is None else c.initial_states[rows],
            signal_kinds=tuple(c.signal_kinds[i] for i in rows),
            metadata=dict(c.metadata),
        )

    # ------------------------------------------------------------------ with truth

    def add(self, indices: Sequence[int]) -> None:
        """Simulate the given candidates and add them to the pool, in the order given."""
        new = list(indices)
        if len(set(new)) != len(new) or set(new) & set(self._selected):
            raise ValueError("a candidate can be added only once")
        if not all(0 <= i < self.n_candidates for i in new):
            raise ValueError("candidate index out of range")
        if not new:
            return
        sub = self.candidate_inputs(new)
        batch = simulate_schedule(self._config, sub, noise_seed=self._noise_seed)
        simulator = _simulator(self._config)
        for position, index in enumerate(new):
            run = batch.select(position)
            measurements = simulator.remeasure(
                run.outputs, seed=_per_run_seed(self._noise_seed, run.run_id[0])
            )
            self._runs[index] = dataclasses.replace(run, measurements=measurements)
            self._selected.append(index)

    def trajectory(self, n: int | None = None) -> Trajectory:
        """The first ``n`` chosen runs (all by default), per-run noise."""
        order = self._selected if n is None else self._selected[:n]
        return concatenate_trajectories([self._runs[i] for i in order])

    def legacy_trajectory(self) -> Trajectory:
        """All chosen runs with the batch noise stream of ``generate_data`` (test only)."""
        trajectory = self.trajectory()
        simulator = _simulator(self._config)
        measurements = simulator.remeasure(trajectory.outputs, seed=self._noise_seed)
        return dataclasses.replace(trajectory, measurements=measurements)

    def windows(
        self, n: int | None = None, *, window: WindowConfig | None = None
    ) -> TrajectoryWindowDataset:
        """Training windows of the first ``n`` chosen runs."""
        return window_dataset(
            self.trajectory(n),
            window if window is not None else self._spec.window,
            role=SplitName.TRAIN,
        )


def windows_for_test_set(
    trajectory: Trajectory,
    window: WindowConfig,
    scalers: tuple,  # type: ignore[type-arg]
) -> TrajectoryWindowDataset:
    """Windows of a test trajectory, normalised with the training scalers."""
    return window_dataset(trajectory, window, role=SplitName.TEST, scalers=scalers)


def pool_summary(pool: GrowingPool) -> Mapping[str, object]:
    """Small description for run records."""
    return {"n_candidates": pool.n_candidates, "selected": list(pool.selected)}
