r"""Windowing, normalisation and the abstraction :class:`SequenceDataset`.

Window structure -- the actual design decision
------------------------------------------------------
A window consists of **three** blocks, not two:

.. code-block:: text

    |<--- history_length L --->|<--- horizon H --->|
    history_inputs             future_exogenous
    [u, z, y_gemessen]         [u, z]
                               future_targets  [y]

The reason is the multi-step rollout. Actions and disturbances are **known** over the
prediction horizon -- the actions because a controller sets them, the disturbances
because they are measured (but not controllable). Only the targets have to be
propagated autoregressively by the model. A window format that does not contain
``future_exogenous`` forces either an extrapolation of the actions (physically wrong) or
a special path in the model code for rollouts. With this format the one-step error is
the horizon slice :math:`h = 1` of the same window -- one definition, two metrics.

Normalisation
-------------
Statistics come **exclusively** from the training runs of the respective learning-curve
point. If they are computed once over the full pool and reused for the small point, the
model there has information from all runs. That raises exactly the left end of the
learning curve, i.e. the statement of M2.

Observability -- inline measurement or soft sensor
-------------------------------------------------
``WindowConfig.observed_target_names`` decides whether a target may appear in the window
history. The difference is not a detail but the question whether the model can exist in
the target system at all:

* **Measured inline** -- the history contains the past measurements. The model has an
  *anchor*: it does not have to predict the level difference between the runs (from the
  scattered initial states), it is in the history. Persistence is then a defined
  baseline.
* **Not measured (soft sensor)** -- the channel is replaced in the history by the
  standardised training mean, i.e. by zero. The model has no anchor and has to infer the
  level from the remaining measurements. Persistence is **undefined** there; the
  parameter-free reference is the mean prediction
  (:class:`~twinlink_ml.models.baselines.ClimatologyModel`).

The zero filling happens in the windowing and not in the model. Otherwise every
architecture would have to implement the case distinction correctly itself, and a model
that forgets it would be handed a non-existent measurement for free -- a leak that looks
like a good soft sensor in the metrics. A test compares the history against the
measurement array and fires if an unobserved channel appears there.

Target: measurement or truth
--------------------------------
The default is the **measurement** (``target_source="measurement"``), because only that
exists at the plant. The evaluator additionally reports against the noise-free output,
where available -- the difference between the two numbers is the irreducible part and
limits how low a learning curve can fall at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np

from ..simulation.backend import FloatArray
from ..simulation.base import Trajectory
from .splits import RunSplit, SplitName

__all__ = [
    "FutureActions",
    "SequenceDataset",
    "Standardizer",
    "TargetSource",
    "TrajectoryWindowDataset",
    "WindowBatch",
    "WindowConfig",
]

TargetSource = Literal["measurement", "truth"]
FutureActions = Literal["realised", "planned"]
"""Which action series enters ``future_exogenous``: the ``realised`` one (after the safety
projection of the simulator) or the ``planned`` one (the set points as requested, before any
projection). See :attr:`WindowConfig.future_actions`."""

_SCALE_FLOOR: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class Standardizer:
    r"""Channel-wise standardisation :math:`\tilde x = (x - \mu) / \sigma`.

    Attributes
    ----------
    mean, scale
        Shape ``(n_channels,)``. ``scale`` is clipped from below, so that constant
        channels -- such as an action in a constant-run design -- do not cause a division
        by zero.
    """

    mean: FloatArray
    scale: FloatArray

    def __post_init__(self) -> None:
        if self.mean.shape != self.scale.shape or self.mean.ndim != 1:
            raise ValueError("mean und scale müssen dieselbe Form (n_channels,) haben")

    @classmethod
    def fit(cls, values: FloatArray) -> Standardizer:
        """Estimate from an array of shape ``(..., n_channels)``."""
        flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        mean = flat.mean(axis=0)
        scale = np.maximum(flat.std(axis=0), _SCALE_FLOOR)
        return cls(mean=mean, scale=scale)

    def transform(self, values: FloatArray) -> FloatArray:
        """Standardise."""
        return (values - self.mean) / self.scale

    def inverse(self, values: FloatArray) -> FloatArray:
        """Back to engineering units."""
        return values * self.scale + self.mean

    def select(self, indices: Sequence[int]) -> Standardizer:
        """Subset of the channels."""
        index = np.asarray(indices, dtype=np.intp)
        return Standardizer(mean=self.mean[index], scale=self.scale[index])

    @property
    def digest(self) -> str:
        """Fingerprint of the statistics, so that the run log can check where they came from."""
        import hashlib

        payload = np.concatenate([self.mean, self.scale]).tobytes()
        return hashlib.blake2b(payload, digest_size=8).hexdigest()


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Window parameters.

    Attributes
    ----------
    history_length
        :math:`L` sampling steps of the past. **A modelling decision, not a
        hyperparameter**: with three decades of spread in the time constants it determines
        which mode the model can see at all.
    horizon
        :math:`H` steps of prediction horizon for the rollout. The one-step error is the
        slice :math:`h = 1`.
    stride
        Distance between consecutive window starts within a run. Overlap is not critical
        as long as splits are at run level.
    target_names
        Targets as a subset of ``Trajectory`` output names. At least one quality variable
        besides ``energy_target``, otherwise one compares optimisation noise.
    target_source
        ``"measurement"`` (default) or ``"truth"``.
    observed_target_names
        Subset of ``target_names`` that is **measured inline** at the plant and may
        therefore appear in the window history. ``None`` means: all. Unobserved targets
        are replaced in the history by the standardised training mean (i.e. 0) -- see the
        module docstring, section "Observability".
    future_actions
        Which action series appears in ``future_exogenous``. ``"realised"`` (default,
        behaviour up to Befund 81): the action realised by the simulator, i.e. with the
        gas velocity projection active the one limited to ``[u_mf(d50), u_t(d50)]``.
        This clipping depends on the true granule state and thereby carries information
        about the state into the "known" future inputs (review D1, 2026-10-09, 30.6 % of
        the test windows affected). ``"planned"``: the **planned set points**
        (``Trajectory.requested_actions``), as a controller specifies them before the plant
        limits them -- without state information. In both cases the history
        (``history_inputs``) contains the realised action: it is in the past and observable
        at the plant. Without an active projection (``requested_actions is None``) both
        series are identical and the option has no effect.
    measured_input_names
        Measured process responses that enter the model as additional known inputs (follow-up
        sensor-value study, docs/29): output channels of the trajectory, taken from
        ``Trajectory.measurements`` (the sensor model, never the noise-free outputs), appended
        after the disturbances in ``history_inputs`` and ``future_exogenous`` and in
        ``exogenous_names``. A response at step ``t`` is measured when the estimate of step ``t``
        is formed, so in ``future_exogenous`` it is a realised value, not a forecast; this suits a
        soft sensor (estimate of the current target), not a multi-step forecast. Must not contain a
        target. Default ``()``: no extra input, behaviour bit for bit as before the option
        (test).
    """

    history_length: int
    horizon: int
    target_names: tuple[str, ...]
    stride: int = 1
    target_source: TargetSource = "measurement"
    observed_target_names: tuple[str, ...] | None = None
    future_actions: FutureActions = "realised"
    measured_input_names: tuple[str, ...] = ()

    @property
    def observed_targets(self) -> tuple[str, ...]:
        """Targets measured inline, in column order."""
        if self.observed_target_names is None:
            return self.target_names
        return tuple(name for name in self.target_names if name in self.observed_target_names)

    @property
    def unobserved_targets(self) -> tuple[str, ...]:
        """Targets without inline measurement -- the soft-sensor task."""
        observed = set(self.observed_targets)
        return tuple(name for name in self.target_names if name not in observed)

    @property
    def is_fully_observed(self) -> bool:
        """Whether every target appears as a measurement in the history."""
        return not self.unobserved_targets

    def __post_init__(self) -> None:
        if self.history_length < 1:
            raise ValueError("history_length muss >= 1 sein")
        if self.horizon < 1:
            raise ValueError("horizon muss >= 1 sein")
        if self.stride < 1:
            raise ValueError("stride muss >= 1 sein")
        if not self.target_names:
            raise ValueError("mindestens eine Zielgröße angeben")
        if self.observed_target_names is not None:
            unknown = set(self.observed_target_names) - set(self.target_names)
            if unknown:
                raise ValueError(
                    f"observed_target_names enthält Namen, die keine Zielgrößen sind: "
                    f"{sorted(unknown)}"
                )
        leaked = set(self.measured_input_names) & set(self.target_names)
        if leaked:
            raise ValueError(f"measured_input_names must not contain targets: {sorted(leaked)}")
        if len(set(self.measured_input_names)) != len(self.measured_input_names):
            raise ValueError("measured_input_names contains duplicates")


def future_actions_of(trajectory: Trajectory, config: WindowConfig) -> FloatArray:
    """Action series for ``future_exogenous`` according to ``config.future_actions``.

    ``"planned"`` returns ``trajectory.requested_actions`` if the simulation recorded it
    (projection active), otherwise the realised actions, which are then identical.
    """
    if config.future_actions == "planned" and trajectory.requested_actions is not None:
        return trajectory.requested_actions
    return trajectory.actions


@dataclass(frozen=True, slots=True)
class WindowBatch:
    """A batch of windows, all quantities standardised.

    Attributes
    ----------
    history_inputs
        ``(batch, L, n_u + n_z + n_y)`` -- actions, disturbances and observed targets.
    future_exogenous
        ``(batch, H, n_u + n_z)`` -- known inputs over the horizon.
    future_targets
        ``(batch, H, n_y)`` -- reference values according to ``target_source``.
    future_targets_truth
        ``(batch, H, n_y)`` or ``None`` -- noise-free reference, for diagnostics only.
    run_index, window_start
        Origin of every window, for run-level aggregation of the metrics.
    """

    history_inputs: FloatArray
    future_exogenous: FloatArray
    future_targets: FloatArray
    run_index: FloatArray
    window_start: FloatArray
    future_targets_truth: FloatArray | None = None

    @property
    def n_windows(self) -> int:
        """Number of windows in the batch."""
        return int(self.history_inputs.shape[0])


class SequenceDataset(ABC):
    """Abstraction over windowed sequence data.

    Behind this ABC the Dyssol or measurement data set is later plugged in, without any
    change to model or evaluation code.
    """

    @property
    @abstractmethod
    def config(self) -> WindowConfig: ...

    @property
    @abstractmethod
    def n_windows(self) -> int: ...

    @property
    @abstractmethod
    def exogenous_names(self) -> tuple[str, ...]:
        """Names of the action and disturbance channels."""

    @property
    @abstractmethod
    def target_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def input_scaler(self) -> Standardizer:
        """Standardisation of the ``history_inputs`` channels."""

    @property
    @abstractmethod
    def target_scaler(self) -> Standardizer:
        """Standardisation of the targets; for the back-transformation in the evaluator."""

    @abstractmethod
    def batch(self, indices: Sequence[int]) -> WindowBatch: ...

    @property
    def autoregressive_index(self) -> tuple[int, ...]:
        """Positions of the targets within the ``history_inputs`` channel axis.

        The channels also exist for unobserved targets -- there they are filled with the
        training mean. The number of channels thus stays the same across both operating
        modes, and an architecture does not have to carry a case distinction: a model
        that propagates its own state does so in the same way in both cases, it only
        starts without an anchor.
        """
        offset = len(self.exogenous_names)
        return tuple(offset + i for i in range(len(self.target_names)))

    @property
    def observed_target_names(self) -> tuple[str, ...]:
        """Targets measured inline."""
        return self.config.observed_targets

    @property
    def unobserved_target_names(self) -> tuple[str, ...]:
        """Targets without inline measurement."""
        return self.config.unobserved_targets

    @property
    def is_fully_observed(self) -> bool:
        """Whether every target appears as a measurement in the history."""
        return self.config.is_fully_observed

    def full_batch(self) -> WindowBatch:
        """All windows in one batch. For baselines and small data sets."""
        return self.batch(range(self.n_windows))

    def iter_batches(
        self, batch_size: int, *, rng: np.random.Generator | None = None
    ) -> Iterator[WindowBatch]:
        """Batch-wise iteration, optionally in random order."""
        order = np.arange(self.n_windows)
        if rng is not None:
            rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            yield self.batch(order[start : start + batch_size].tolist())


@dataclass(slots=True)
class TrajectoryWindowDataset(SequenceDataset):
    """Windows over the runs of one split of a :class:`Trajectory`.

    The windows are formed **lazily** from an index: with 128 runs of 1152 steps each
    and ``stride=1`` there are about 130,000 windows, whose materialised form occupies a
    multiple of the raw data.

    Construction via :meth:`from_trajectory`, so that the normalisation is not
    accidentally estimated from the wrong split.
    """

    _config: WindowConfig
    _exogenous_names: tuple[str, ...]
    _target_names: tuple[str, ...]
    _inputs: FloatArray
    _exogenous: FloatArray
    _targets: FloatArray
    _targets_truth: FloatArray | None
    _index: FloatArray
    _input_scaler: Standardizer
    _target_scaler: Standardizer
    run_ids: tuple[str, ...] = field(default=())

    # ---------------------------------------------------------------- Construction

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        split: RunSplit,
        which: SplitName,
        config: WindowConfig,
        *,
        scalers: tuple[Standardizer, Standardizer] | None = None,
    ) -> TrajectoryWindowDataset:
        """Build the data set for one split.

        Parameters
        ----------
        scalers
            ``(input_scaler, target_scaler)``. For ``val`` and ``test`` the scalers of the
            corresponding training data set **must** be passed. If omitted, they are
            estimated from the data set's own runs, which is correct for ``train`` and a
            leak for ``test``.
        """
        output_names = tuple(trajectory.metadata.get("output_names", ()))
        if not output_names:
            raise ValueError("Trajectory.metadata enthält keine output_names")
        missing = set(config.target_names) - set(output_names)
        if missing:
            raise ValueError(f"Unbekannte Zielgrößen: {sorted(missing)}")

        wanted = set(split.as_mapping[which])
        positions = [i for i, run_id in enumerate(trajectory.run_id) if run_id in wanted]
        if len(positions) != len(wanted):
            found = {trajectory.run_id[i] for i in positions}
            raise ValueError(f"Läufe fehlen in der Trajektorie: {sorted(wanted - found)}")
        # Preserve the order of the split -- it defines the learning-curve nesting.
        order = {trajectory.run_id[i]: i for i in positions}
        positions = [order[run_id] for run_id in split.as_mapping[which]]
        rows = np.asarray(positions, dtype=np.intp)

        target_cols = np.asarray(
            [output_names.index(name) for name in config.target_names], dtype=np.intp
        )
        exogenous = np.concatenate(
            [trajectory.actions[rows], trajectory.disturbances[rows]], axis=-1
        )
        future_exogenous = np.concatenate(
            [future_actions_of(trajectory, config)[rows], trajectory.disturbances[rows]], axis=-1
        )
        if config.measured_input_names:
            missing_inputs = set(config.measured_input_names) - set(output_names)
            if missing_inputs:
                raise ValueError(f"unknown measured inputs: {sorted(missing_inputs)}")
            measured_cols = np.asarray(
                [output_names.index(name) for name in config.measured_input_names], dtype=np.intp
            )
            measured_inputs = trajectory.measurements[rows][..., measured_cols]
            exogenous = np.concatenate([exogenous, measured_inputs], axis=-1)
            future_exogenous = np.concatenate([future_exogenous, measured_inputs], axis=-1)
        observed = (
            trajectory.measurements[rows][..., target_cols]
            if config.target_source == "measurement"
            else trajectory.outputs[rows][..., target_cols]
        )
        truth = trajectory.outputs[rows][..., target_cols]

        inputs = np.concatenate([exogenous, observed], axis=-1)

        input_scaler, target_scaler = (
            scalers
            if scalers is not None
            else (Standardizer.fit(inputs), Standardizer.fit(observed))
        )
        standardised = input_scaler.transform(inputs)

        # Observability: channels not measured inline carry no information into the
        # history. Masking happens **after** the standardisation, so that zero corresponds
        # exactly to the mean of the training split and the scaler stays the same as in the
        # fully observed case -- otherwise the two operating modes would not be
        # comparable, because the normalisation of the inputs would have changed as well.
        unobserved = set(config.unobserved_targets)
        offset = exogenous.shape[-1]
        masked_channels = [
            offset + index for index, name in enumerate(config.target_names) if name in unobserved
        ]
        if masked_channels:
            standardised[..., masked_channels] = 0.0

        n_runs, n_steps, _ = inputs.shape
        span = config.history_length + config.horizon
        if n_steps < span:
            raise ValueError(
                f"Läufe haben {n_steps} Schritte, ein Fenster braucht "
                f"{span} (history_length + horizon)"
            )
        starts = np.arange(0, n_steps - span + 1, config.stride, dtype=np.intp)
        index = np.stack(
            [
                np.repeat(np.arange(n_runs, dtype=np.intp), len(starts)),
                np.tile(starts, n_runs),
            ],
            axis=-1,
        )

        exogenous_names = (
            tuple(trajectory.metadata.get("action_names", ()))
            + tuple(trajectory.metadata.get("disturbance_names", ()))
            + config.measured_input_names
        )
        return cls(
            _config=config,
            _exogenous_names=exogenous_names,
            _target_names=config.target_names,
            _inputs=standardised,
            _exogenous=input_scaler.select(range(len(exogenous_names))).transform(future_exogenous),
            _targets=target_scaler.transform(observed),
            _targets_truth=target_scaler.transform(truth),
            _index=index,
            _input_scaler=input_scaler,
            _target_scaler=target_scaler,
            run_ids=tuple(trajectory.run_id[i] for i in positions),
        )

    # -------------------------------------------------------------------- Interface

    @property
    def config(self) -> WindowConfig:
        """Window parameters."""
        return self._config

    @property
    def n_windows(self) -> int:
        """Number of windows."""
        return int(self._index.shape[0])

    @property
    def exogenous_names(self) -> tuple[str, ...]:
        """Action and disturbance names."""
        return self._exogenous_names

    @property
    def target_names(self) -> tuple[str, ...]:
        """Target names."""
        return self._target_names

    @property
    def input_scaler(self) -> Standardizer:
        """Scaler of the input channels."""
        return self._input_scaler

    @property
    def target_scaler(self) -> Standardizer:
        """Scaler of the targets."""
        return self._target_scaler

    @property
    def scalers(self) -> tuple[Standardizer, Standardizer]:
        """Both scalers, for passing on to the val and test data sets."""
        return self._input_scaler, self._target_scaler

    def batch(self, indices: Sequence[int]) -> WindowBatch:
        """Form the windows for the given indices."""
        selected = self._index[np.asarray(indices, dtype=np.intp)]
        runs = selected[:, 0]
        starts = selected[:, 1]
        history_length = self._config.history_length
        horizon = self._config.horizon

        history_offsets = np.arange(history_length, dtype=np.intp)
        future_offsets = np.arange(history_length, history_length + horizon, dtype=np.intp)
        history_time = starts[:, None] + history_offsets[None, :]
        future_time = starts[:, None] + future_offsets[None, :]
        run_column = runs[:, None]

        return WindowBatch(
            history_inputs=self._inputs[run_column, history_time],
            future_exogenous=self._exogenous[run_column, future_time],
            future_targets=self._targets[run_column, future_time],
            future_targets_truth=(
                None
                if self._targets_truth is None
                else self._targets_truth[run_column, future_time]
            ),
            run_index=runs.astype(np.float64),
            window_start=starts.astype(np.float64),
        )
