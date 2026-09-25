r"""Punktweiser (fenster-/beobachtungsweiser) Split -- Kräuters Vergleichsvariante.

Bewusst **außerhalb** der :class:`~twinlink_ml.data.splits.RunSplitter`-ABC. Diese ABC
ordnet ganze Läufe zu -- ``RunSplit`` hat als Invariante, dass kein Lauf in zwei Splits
liegt, und ``TrajectoryWindowDataset.from_trajectory`` filtert Zeilen ausschließlich über
Lauf-Zugehörigkeit. Ein punktweiser Split verteilt dagegen einzelne **Fenster** unabhängig
von ihrer Lauf-Zugehörigkeit -- Fenster desselben Laufs können auf verschiedenen Seiten
landen. Das ist eine andere Granularität, keine weitere ``RunSplitter``-Implementierung.

Wozu das gut ist: Kräuter, Zhou & Heinrich (ABS23) teilen ihre Trainings-/Testdaten
zufällig über Einzelbeobachtungen auf, nicht über Chargen (siehe
``docs/04_stand_ap1.md``, Abschnitt "Was das Paper des Vorgängers ändert"). Bei feiner
Abtastung liegt der Nachbarpunkt einer Testbeobachtung damit fast immer im Training --
ein Fenster-Leck, gegen das die laufweisen Splits dieses Projekts (``HashRunSplitter``,
``IndexBlockSplitter``) gerade eine Barriere bilden. Dieses Modul baut die
Vergleichsvariante, um den Abstand zwischen beiden Aufteilungen auf demselben Datensatz
zu messen -- nicht, um sie als zweite reguläre Splitstrategie im Projekt zu etablieren.

Normierung: dieselbe Invariante wie bei den laufweisen Splits -- der Skalierer wird
ausschließlich aus den als ``train`` zugeordneten Fenstern geschätzt, nie aus
``val``/``test``. Weil ein punktweiser Split erst nach der Fensterung entsteht (er trennt
quer über Läufe), kann der Skalierer hier nicht wie bei
:meth:`~twinlink_ml.data.windows.TrajectoryWindowDataset.from_trajectory` vor der
Fensterung aus dem vollen Lauf-Array geschätzt werden -- er wird stattdessen aus den
(noch unstandardisierten) ``history_inputs``/``future_targets`` der train-Fenster
geschätzt, eine Ebene später als im laufweisen Fall. Die Invariante selbst ("nie aus
val/test") bleibt identisch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..simulation.base import Trajectory
from .splits import SplitDegenerateError, SplitName, SplitRatios, stable_unit_hash
from .windows import Standardizer, TrajectoryWindowDataset, WindowConfig

__all__ = ["PointSplitter", "windowed_point_split"]

_HASH_NAMESPACE = "twinlink_ml/point_split/v1"


@dataclass(frozen=True, slots=True)
class PointSplitter:
    """Weist einzelne Fenster train/val/test zu, unabhängig von der Lauf-Zugehörigkeit.

    Strukturell identisch zu :class:`~twinlink_ml.data.splits.HashRunSplitter`, nur eine
    Ebene tiefer: statt ``(run_id, seed) -> Split`` ist es ``(run_id, window_start, seed)
    -> Split``. Jedes Fenster wird unabhängig von jedem anderen zugeordnet, ohne Kenntnis
    des Fenster-Pools -- stabil unter Pool-Wachstum, aus demselben Grund wie dort
    dokumentiert.

    Attributes
    ----------
    seed
        Seed der Zuordnung.
    ratios
        Zielanteile der drei Splits, im Erwartungswert (siehe
        :class:`~twinlink_ml.data.splits.SplitRatios`).
    """

    seed: int
    ratios: SplitRatios = field(default_factory=SplitRatios)

    def assign(self, run_id: str, window_start: int) -> SplitName:
        """Split eines einzelnen Fensters, ohne Kenntnis des übrigen Fenster-Pools."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Rohe (unstandardisierte) Kanal-Arrays über den gegebenen Lauf-Pool.

    Spiegelt den entsprechenden Abschnitt von
    :meth:`TrajectoryWindowDataset.from_trajectory`, aber ohne die dortige
    Standardisierung -- die geschieht hier erst nach der Fensterung, siehe
    Moduldocstring.

    Returns
    -------
    tuple
        ``(rows, exogenous, observed, truth, exogenous_names, target_names)``. ``rows``
        sind die Trajektorie-Zeilen in der Reihenfolge von ``run_ids``.
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
    observed = (
        trajectory.measurements[rows][..., target_cols]
        if config.target_source == "measurement"
        else trajectory.outputs[rows][..., target_cols]
    )
    truth = trajectory.outputs[rows][..., target_cols]

    exogenous_names = tuple(trajectory.metadata.get("action_names", ())) + tuple(
        trajectory.metadata.get("disturbance_names", ())
    )
    return rows, exogenous, observed, truth, exogenous_names, config.target_names


def _window_index(n_runs: int, n_steps: int, config: WindowConfig) -> np.ndarray:
    """Fenster-Index über den vollen Lauf-Pool, ``(n_windows, 2)`` als ``(run_row, start)``.

    Identische Konvention zu ``TrajectoryWindowDataset._index``.
    """
    span = config.history_length + config.horizon
    if n_steps < span:
        raise ValueError(
            f"Läufe haben {n_steps} Schritte, ein Fenster braucht {span} "
            "(history_length + horizon)"
        )
    starts = np.arange(0, n_steps - span + 1, config.stride, dtype=np.intp)
    return np.stack(
        [
            np.repeat(np.arange(n_runs, dtype=np.intp), len(starts)),
            np.tile(starts, n_runs),
        ],
        axis=-1,
    )


def _slice_windows(
    array: np.ndarray, index: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """``array[run, start + offsets]`` für jede Zeile von ``index``, vektorisiert."""
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
    """Punktweise train/val/test-Datensätze über denselben Lauf-Pool bilden.

    Baut **einen** Fenster-Pool über alle ``run_ids`` (im Gegensatz zum laufweisen Pfad,
    der schon vor der Fensterung nach Split trennt), ordnet jedes Fenster einzeln über
    :meth:`PointSplitter.assign` zu, und schätzt die Standardisierung ausschließlich aus
    den train-Fenstern -- siehe Moduldocstring.

    Returns
    -------
    tuple
        ``(train, val, test)``, alle drei :class:`TrajectoryWindowDataset`-Instanzen mit
        identischen (gemeinsam standardisierten) Trägerarrays, aber disjunkten
        Fenster-Indizes. ``val`` ist ``None``, wenn ``splitter.ratios.val == 0``.
    """
    rows, exogenous, observed, truth, exogenous_names, target_names = _assemble_raw(
        trajectory, run_ids, config
    )
    inputs_raw = np.concatenate([exogenous, observed], axis=-1)
    n_runs, n_steps, _ = inputs_raw.shape
    index = _window_index(n_runs, n_steps, config)

    assignments = np.array(
        [
            splitter.assign(run_ids[run_row], int(start)).value
            for run_row, start in index
        ]
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
        raise SplitDegenerateError(
            f"val hat {n_val_windows} Fenster, Minimum ist {min_per_split}"
        )

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
    standardised_exogenous = input_scaler.select(range(len(exogenous_names))).transform(exogenous)
    standardised_targets = target_scaler.transform(observed)
    standardised_targets_truth = target_scaler.transform(truth)

    unobserved = set(config.unobserved_targets)
    offset = exogenous.shape[-1]
    masked_channels = [
        offset + i for i, name in enumerate(target_names) if name in unobserved
    ]
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
