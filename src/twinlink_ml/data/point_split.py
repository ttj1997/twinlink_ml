r"""Point-level (window-/observation-level) split -- Kräuter's comparison variant.

Deliberately **outside** the :class:`~twinlink_ml.data.splits.RunSplitter` ABC. That ABC
assigns whole runs -- ``RunSplit`` has the invariant that no run lies in two splits, and
``TrajectoryWindowDataset.from_trajectory`` filters rows exclusively by run membership. A
point-level split, by contrast, distributes individual **windows** independently of their
run membership -- windows of the same run can end up on different sides. That is a
different granularity, not another ``RunSplitter`` implementation.

What it is good for: Kräuter, Zhou & Heinrich (ABS23) split their training/test data
randomly over individual observations, not over batches (see
``docs/04_stand_ap1.md``, section "Was das Paper des Vorgängers ändert"). With fine
sampling, the neighbouring point of a test observation therefore almost always lies in the
training data -- a window leak against which the run-level splits of this project
(``HashRunSplitter``, ``IndexBlockSplitter``) form a barrier. This module builds the
comparison variant in order to measure the gap between the two splits on the same data
set -- not to establish it as a second regular split strategy in the project.

Normalisation: the same invariant as for the run-level splits -- the scaler is estimated
exclusively from the windows assigned to ``train``, never from ``val``/``test``. Because a
point-level split only arises after windowing (it cuts across runs), the scaler cannot be
estimated here from the full run array before windowing as in
:meth:`~twinlink_ml.data.windows.TrajectoryWindowDataset.from_trajectory` -- instead it is
estimated from the (still unstandardised) ``history_inputs``/``future_targets`` of the
train windows, one level later than in the run-level case. The invariant itself ("never
from val/test") stays identical.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..simulation.base import Trajectory
from .splits import SplitDegenerateError, SplitName, SplitRatios, stable_unit_hash
from .windows import Standardizer, TrajectoryWindowDataset, WindowConfig, future_actions_of

__all__ = ["PointSplitter", "windowed_point_split"]

_HASH_NAMESPACE = "twinlink_ml/point_split/v1"


@dataclass(frozen=True, slots=True)
class PointSplitter:
    """Assigns individual windows to train/val/test, independently of run membership.

    Structurally identical to :class:`~twinlink_ml.data.splits.HashRunSplitter`, only one
    level deeper: instead of ``(run_id, seed) -> Split`` it is ``(run_id, window_start, seed)
    -> Split``. Every window is assigned independently of every other, without knowledge
    of the window pool -- stable under pool growth, for the same reason as documented
    there.

    Attributes
    ----------
    seed
        Seed of the assignment.
    ratios
        Target fractions of the three splits, in expectation (see
        :class:`~twinlink_ml.data.splits.SplitRatios`).
    """

    seed: int
    ratios: SplitRatios = field(default_factory=SplitRatios)

    def assign(self, run_id: str, window_start: int) -> SplitName:
        """Split of a single window, without knowledge of the rest of the window pool."""
        value = stable_unit_hash(
            "point_assign", self.seed, run_id, window_start, namespace=_HASH_NAMESPACE
        )
        train_edge, val_edge = self.ratios.edges
        if value < train_edge:
            return SplitName.TRAIN
        if value < val_edge:
            return SplitName.VAL
        return SplitName.TEST


def _assemble_raw(
    trajectory: Trajectory, run_ids: Sequence[str], config: WindowConfig
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Raw (unstandardised) channel arrays over the given run pool.

    Mirrors the corresponding section of
    :meth:`TrajectoryWindowDataset.from_trajectory`, but without the standardisation
    there -- here that only happens after windowing, see the module docstring.

    Returns
    -------
    tuple
        ``(rows, exogenous, future_exogenous, observed, truth, exogenous_names,
        target_names)``. ``rows`` are the trajectory rows in the order of
        ``run_ids``; ``future_exogenous`` follows ``config.future_actions`` (see
        :class:`WindowConfig`), ``exogenous`` is the realised series of the history.
    """
    output_names = tuple(trajectory.metadata.get("output_names", ()))
    if not output_names:
        raise ValueError("Trajectory.metadata enthält keine output_names")
    missing = set(config.target_names) - set(output_names)
    if missing:
        raise ValueError(f"Unbekannte Zielgrößen: {sorted(missing)}")

    wanted = set(run_ids)
    positions = [i for i, run_id in enumerate(trajectory.run_id) if run_id in wanted]
    if len(positions) != len(wanted):
        found = {trajectory.run_id[i] for i in positions}
        raise ValueError(f"Läufe fehlen in der Trajektorie: {sorted(wanted - found)}")
    order = {trajectory.run_id[i]: i for i in positions}
    ordered_positions = [order[run_id] for run_id in run_ids]
    rows = np.asarray(ordered_positions, dtype=np.intp)

    target_cols = np.asarray(
        [output_names.index(name) for name in config.target_names], dtype=np.intp
    )
    exogenous = np.concatenate([trajectory.actions[rows], trajectory.disturbances[rows]], axis=-1)
    future_exogenous = np.concatenate(
        [future_actions_of(trajectory, config)[rows], trajectory.disturbances[rows]], axis=-1
    )
    observed = (
        trajectory.measurements[rows][..., target_cols]
        if config.target_source == "measurement"
        else trajectory.outputs[rows][..., target_cols]
    )
    truth = trajectory.outputs[rows][..., target_cols]

    exogenous_names = tuple(trajectory.metadata.get("action_names", ())) + tuple(
        trajectory.metadata.get("disturbance_names", ())
    )
    return rows, exogenous, future_exogenous, observed, truth, exogenous_names, config.target_names


def _window_index(n_runs: int, n_steps: int, config: WindowConfig) -> np.ndarray:
    """Window index over the full run pool, ``(n_windows, 2)`` as ``(run_row, start)``.

    Same convention as ``TrajectoryWindowDataset._index``.
    """
    span = config.history_length + config.horizon
    if n_steps < span:
        raise ValueError(
            f"Läufe haben {n_steps} Schritte, ein Fenster braucht {span} (history_length + horizon)"
        )
    starts = np.arange(0, n_steps - span + 1, config.stride, dtype=np.intp)
    return np.stack(
        [
            np.repeat(np.arange(n_runs, dtype=np.intp), len(starts)),
            np.tile(starts, n_runs),
        ],
        axis=-1,
    )


def _slice_windows(array: np.ndarray, index: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """``array[run, start + offsets]`` for every row of ``index``, vectorised."""
    runs = index[:, 0:1]
    starts = index[:, 1:2]
    time = starts + offsets[None, :]
    return np.asarray(array[runs, time])


def windowed_point_split(
    trajectory: Trajectory,
    run_ids: Sequence[str],
    config: WindowConfig,
    splitter: PointSplitter,
    *,
    min_per_split: int = 2,
) -> tuple[TrajectoryWindowDataset, TrajectoryWindowDataset | None, TrajectoryWindowDataset]:
    """Build point-level train/val/test data sets over the same run pool.

    Builds **one** window pool over all ``run_ids`` (in contrast to the run-level path,
    which separates by split before windowing), assigns every window individually via
    :meth:`PointSplitter.assign`, and estimates the standardisation exclusively from the
    train windows -- see the module docstring.

    Returns
    -------
    tuple
        ``(train, val, test)``, all three :class:`TrajectoryWindowDataset` instances with
        identical (jointly standardised) backing arrays, but disjoint window indices.
        ``val`` is ``None`` if ``splitter.ratios.val == 0``.
    """
    _, exogenous, future_exogenous, observed, truth, exogenous_names, target_names = _assemble_raw(
        trajectory, run_ids, config
    )
    inputs_raw = np.concatenate([exogenous, observed], axis=-1)
    n_runs, n_steps, _ = inputs_raw.shape
    index = _window_index(n_runs, n_steps, config)

    assignments = np.array(
        [splitter.assign(run_ids[run_row], int(start)).value for run_row, start in index]
    )
    masks = {name: assignments == name.value for name in SplitName}
    for name in (SplitName.TRAIN, SplitName.TEST):
        n_windows = int(masks[name].sum())
        if n_windows < min_per_split:
            raise SplitDegenerateError(
                f"{name.value} hat {n_windows} Fenster, Minimum ist {min_per_split}"
            )
    n_val_windows = int(masks[SplitName.VAL].sum())
    if splitter.ratios.val > 0.0 and n_val_windows < min_per_split:
        raise SplitDegenerateError(f"val hat {n_val_windows} Fenster, Minimum ist {min_per_split}")

    history_offsets = np.arange(config.history_length, dtype=np.intp)
    future_offsets = np.arange(
        config.history_length, config.history_length + config.horizon, dtype=np.intp
    )

    train_index = index[masks[SplitName.TRAIN]]
    train_history_raw = _slice_windows(inputs_raw, train_index, history_offsets)
    train_future_target_raw = _slice_windows(observed, train_index, future_offsets)
    input_scaler = Standardizer.fit(train_history_raw)
    target_scaler = Standardizer.fit(train_future_target_raw)

    standardised_inputs = input_scaler.transform(inputs_raw)
    standardised_exogenous = input_scaler.select(range(len(exogenous_names))).transform(
        future_exogenous
    )
    standardised_targets = target_scaler.transform(observed)
    standardised_targets_truth = target_scaler.transform(truth)

    unobserved = set(config.unobserved_targets)
    offset = exogenous.shape[-1]
    masked_channels = [offset + i for i, name in enumerate(target_names) if name in unobserved]
    if masked_channels:
        standardised_inputs[..., masked_channels] = 0.0

    datasets: dict[SplitName, TrajectoryWindowDataset] = {}
    for name in SplitName:
        if name is SplitName.VAL and splitter.ratios.val == 0.0:
            continue
        split_index = index[masks[name]]
        datasets[name] = TrajectoryWindowDataset(
            _config=config,
            _exogenous_names=exogenous_names,
            _target_names=target_names,
            _inputs=standardised_inputs,
            _exogenous=standardised_exogenous,
            _targets=standardised_targets,
            _targets_truth=standardised_targets_truth,
            _index=split_index,
            _input_scaler=input_scaler,
            _target_scaler=target_scaler,
            run_ids=tuple(run_ids),
        )

    return datasets[SplitName.TRAIN], datasets.get(SplitName.VAL), datasets[SplitName.TEST]
