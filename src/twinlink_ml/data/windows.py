r"""Fensterung, Normalisierung und die Abstraktion :class:`SequenceDataset`.

Fensterstruktur -- die eigentliche Entwurfsentscheidung
------------------------------------------------------
Ein Fenster besteht aus **drei** Blöcken, nicht aus zwei:

.. code-block:: text

    |<--- history_length L --->|<--- horizon H --->|
    history_inputs             future_exogenous
    [u, z, y_gemessen]         [u, z]
                               future_targets  [y]

Der Grund ist der Mehrschritt-Rollout. Stell- und Störgrößen sind über den
Vorhersagehorizont **bekannt** -- die Stellgrößen, weil ein Regler sie vorgibt, die
Störgrößen, weil sie gemessen (aber nicht steuerbar) sind. Nur die Zielgrößen muss das
Modell autoregressiv fortschreiben. Ein Fensterformat, das ``future_exogenous`` nicht
enthält, erzwingt entweder eine Extrapolation der Stellgrößen (physikalisch falsch) oder
einen Sonderpfad im Modellcode für Rollouts. Mit diesem Format ist der
Ein-Schritt-Fehler der Horizontschnitt :math:`h = 1` desselben Fensters -- eine
Definition, zwei Metriken.

Normalisierung
--------------
Statistiken kommen **ausschließlich** aus den Trainingsläufen der jeweiligen
Lernkurven-Stützstelle. Werden sie einmal über den vollen Pool berechnet und für die
kleine Stützstelle weiterverwendet, hat das Modell dort Information aus allen Läufen.
Das hebt genau das linke Ende der Lernkurve an, also die Aussage von M2.

Beobachtbarkeit -- Inline-Messung oder Soft-Sensor
-------------------------------------------------
``WindowConfig.observed_target_names`` entscheidet, ob eine Zielgröße in der
Fensterhistorie stehen darf. Der Unterschied ist keine Feinheit, sondern die Frage, ob das
Modell im Zielsystem überhaupt existieren kann:

* **Inline gemessen** -- die Historie enthält die vergangenen Messwerte. Das Modell hat
  einen *Anker*: den Niveauunterschied zwischen den Läufen (aus den gestreuten
  Anfangszuständen) muss es nicht vorhersagen, er steht in der Historie. Persistenz ist
  dann eine definierte Baseline.
* **Nicht gemessen (Soft Sensor)** -- der Kanal wird in der Historie durch den
  standardisierten Trainingsmittelwert ersetzt, also durch Null. Das Modell hat keinen
  Anker und muss das Niveau aus den übrigen Messungen erschließen. Persistenz ist dort
  **undefiniert**; die parameterfreie Referenz ist die Mittelwertvorhersage
  (:class:`~twinlink_ml.models.baselines.ClimatologyModel`).

Die Null-Füllung passiert in der Fensterung und nicht im Modell. Sonst müsste jede
Architektur die Fallunterscheidung selbst korrekt implementieren, und ein Modell, das sie
vergisst, bekäme eine nicht existierende Messung geschenkt -- ein Leck, das in den
Metriken wie ein guter Soft Sensor aussieht. Ein Test vergleicht die Historie gegen das
Messarray und schlägt an, wenn ein nicht beobachteter Kanal dort auftaucht.

Zielgröße: Messung oder Wahrheit
--------------------------------
Standard ist die **Messung** (``target_source="measurement"``), denn nur die existiert
an der Anlage. Der Evaluator berichtet zusätzlich gegen die rauschfreie Ausgangsgröße,
soweit vorhanden -- die Differenz beider Zahlen ist der irreduzible Anteil und
begrenzt, wie tief eine Lernkurve überhaupt fallen kann.
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
    "SequenceDataset",
    "Standardizer",
    "TargetSource",
    "TrajectoryWindowDataset",
    "WindowBatch",
    "WindowConfig",
]

TargetSource = Literal["measurement", "truth"]

_SCALE_FLOOR: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class Standardizer:
    r"""Kanalweise Standardisierung :math:`\tilde x = (x - \mu) / \sigma`.

    Attributes
    ----------
    mean, scale
        Form ``(n_channels,)``. ``scale`` ist nach unten abgeschnitten, damit konstante
        Kanäle -- etwa eine Stellgröße in einem Konstantlauf-Design -- keine Division
        durch Null erzeugen.
    """

    mean: FloatArray
    scale: FloatArray

    def __post_init__(self) -> None:
        if self.mean.shape != self.scale.shape or self.mean.ndim != 1:
            raise ValueError("mean und scale müssen dieselbe Form (n_channels,) haben")

    @classmethod
    def fit(cls, values: FloatArray) -> Standardizer:
        """Aus einem Array der Form ``(..., n_channels)`` schätzen."""
        flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        mean = flat.mean(axis=0)
        scale = np.maximum(flat.std(axis=0), _SCALE_FLOOR)
        return cls(mean=mean, scale=scale)

    def transform(self, values: FloatArray) -> FloatArray:
        """Standardisieren."""
        return (values - self.mean) / self.scale

    def inverse(self, values: FloatArray) -> FloatArray:
        """Zurück in Ingenieurseinheiten."""
        return values * self.scale + self.mean

    def select(self, indices: Sequence[int]) -> Standardizer:
        """Teilmenge der Kanäle."""
        index = np.asarray(indices, dtype=np.intp)
        return Standardizer(mean=self.mean[index], scale=self.scale[index])

    @property
    def digest(self) -> str:
        """Fingerprint der Statistiken, damit im Run-Log prüfbar ist, woraus sie kamen."""
        import hashlib

        payload = np.concatenate([self.mean, self.scale]).tobytes()
        return hashlib.blake2b(payload, digest_size=8).hexdigest()


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Fensterparameter.

    Attributes
    ----------
    history_length
        :math:`L` Abtastschritte Vergangenheit. **Modellentscheidung, kein
        Hyperparameter**: bei drei Dekaden Spreizung der Zeitkonstanten legt sie fest,
        welche Mode das Modell überhaupt sehen kann.
    horizon
        :math:`H` Schritte Vorhersagehorizont für den Rollout. Der Ein-Schritt-Fehler
        ist der Schnitt :math:`h = 1`.
    stride
        Abstand aufeinanderfolgender Fensteranfänge innerhalb eines Laufs.
        Überlappung ist unkritisch, solange Splits auf Lauf-Ebene liegen.
    target_names
        Zielgrößen als Teilmenge von ``Trajectory``-Ausgangsnamen. Mindestens eine
        Qualitätsgröße neben ``energy_target``, sonst vergleicht man Optimierungsrauschen.
    target_source
        ``"measurement"`` (Standard) oder ``"truth"``.
    observed_target_names
        Teilmenge von ``target_names``, die an der Anlage **inline gemessen** wird und
        deshalb in der Fensterhistorie stehen darf. ``None`` heißt: alle. Nicht
        beobachtete Zielgrößen werden in der Historie durch den standardisierten
        Trainingsmittelwert (also 0) ersetzt -- siehe Modul-Docstring, Abschnitt
        "Beobachtbarkeit".
    """

    history_length: int
    horizon: int
    target_names: tuple[str, ...]
    stride: int = 1
    target_source: TargetSource = "measurement"
    observed_target_names: tuple[str, ...] | None = None

    @property
    def observed_targets(self) -> tuple[str, ...]:
        """Inline gemessene Zielgrößen in Spaltenreihenfolge."""
        if self.observed_target_names is None:
            return self.target_names
        return tuple(name for name in self.target_names if name in self.observed_target_names)

    @property
    def unobserved_targets(self) -> tuple[str, ...]:
        """Zielgrößen ohne Inline-Messung -- die Soft-Sensor-Aufgabe."""
        observed = set(self.observed_targets)
        return tuple(name for name in self.target_names if name not in observed)

    @property
    def is_fully_observed(self) -> bool:
        """Ob jede Zielgröße als Messung in der Historie steht."""
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


@dataclass(frozen=True, slots=True)
class WindowBatch:
    """Ein Stapel Fenster, alle Größen standardisiert.

    Attributes
    ----------
    history_inputs
        ``(batch, L, n_u + n_z + n_y)`` -- Stell-, Stör- und beobachtete Zielgrößen.
    future_exogenous
        ``(batch, H, n_u + n_z)`` -- bekannte Eingänge über den Horizont.
    future_targets
        ``(batch, H, n_y)`` -- Sollwerte gemäß ``target_source``.
    future_targets_truth
        ``(batch, H, n_y)`` oder ``None`` -- rauschfreie Referenz, nur für Diagnose.
    run_index, window_start
        Herkunft jedes Fensters, für laufweise Aggregation der Metriken.
    """

    history_inputs: FloatArray
    future_exogenous: FloatArray
    future_targets: FloatArray
    run_index: FloatArray
    window_start: FloatArray
    future_targets_truth: FloatArray | None = None

    @property
    def n_windows(self) -> int:
        """Anzahl Fenster im Stapel."""
        return int(self.history_inputs.shape[0])


class SequenceDataset(ABC):
    """Abstraktion über gefensterte Sequenzdaten.

    Hinter dieser ABC wird später der Dyssol- oder Messdaten-Datensatz eingehängt, ohne
    Änderung an Modell- oder Evaluationscode.
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
        """Namen der Stell- und Störgrößenkanäle."""

    @property
    @abstractmethod
    def target_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def input_scaler(self) -> Standardizer:
        """Standardisierung der ``history_inputs``-Kanäle."""

    @property
    @abstractmethod
    def target_scaler(self) -> Standardizer:
        """Standardisierung der Zielgrößen; für die Rückrechnung im Evaluator."""

    @abstractmethod
    def batch(self, indices: Sequence[int]) -> WindowBatch: ...

    @property
    def autoregressive_index(self) -> tuple[int, ...]:
        """Positionen der Zielgrößen innerhalb der ``history_inputs``-Kanalachse.

        Die Kanäle existieren auch für nicht beobachtete Zielgrößen -- sie sind dort mit
        dem Trainingsmittelwert gefüllt. Damit bleibt die Kanalzahl über beide
        Betriebsarten gleich, und eine Architektur muss keine Fallunterscheidung tragen:
        ein Modell, das seinen eigenen Zustand fortschreibt, tut das in beiden Fällen
        gleich, es startet nur ohne Anker.
        """
        offset = len(self.exogenous_names)
        return tuple(offset + i for i in range(len(self.target_names)))

    @property
    def observed_target_names(self) -> tuple[str, ...]:
        """Inline gemessene Zielgrößen."""
        return self.config.observed_targets

    @property
    def unobserved_target_names(self) -> tuple[str, ...]:
        """Zielgrößen ohne Inline-Messung."""
        return self.config.unobserved_targets

    @property
    def is_fully_observed(self) -> bool:
        """Ob jede Zielgröße als Messung in der Historie steht."""
        return self.config.is_fully_observed

    def full_batch(self) -> WindowBatch:
        """Alle Fenster in einem Stapel. Für Baselines und kleine Datensätze."""
        return self.batch(range(self.n_windows))

    def iter_batches(
        self, batch_size: int, *, rng: np.random.Generator | None = None
    ) -> Iterator[WindowBatch]:
        """Stapelweise Iteration, optional in zufälliger Reihenfolge."""
        order = np.arange(self.n_windows)
        if rng is not None:
            rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            yield self.batch(order[start : start + batch_size].tolist())


@dataclass(slots=True)
class TrajectoryWindowDataset(SequenceDataset):
    """Fenster über die Läufe eines Splits einer :class:`Trajectory`.

    Die Fenster werden **lazy** aus einem Index gebildet: bei 128 Läufen à 1152
    Schritten und ``stride=1`` gibt es rund 130 000 Fenster, deren materialisierte Form
    ein Vielfaches der Rohdaten belegt.

    Konstruktion über :meth:`from_trajectory`, damit die Normalisierung nicht versehentlich
    aus dem falschen Split geschätzt wird.
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

    # ---------------------------------------------------------------- Konstruktion

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
        """Datensatz für einen Split bauen.

        Parameters
        ----------
        scalers
            ``(input_scaler, target_scaler)``. Für ``val`` und ``test`` **zwingend** die
            Skalierer des zugehörigen Trainings-Datensatzes übergeben. Ohne Angabe
            werden sie aus den eigenen Läufen geschätzt, was für ``train`` richtig und
            für ``test`` ein Leck ist.
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
        # Reihenfolge des Splits erhalten -- sie definiert die Lernkurven-Verschachtelung.
        order = {trajectory.run_id[i]: i for i in positions}
        positions = [order[run_id] for run_id in split.as_mapping[which]]
        rows = np.asarray(positions, dtype=np.intp)

        target_cols = np.asarray(
            [output_names.index(name) for name in config.target_names], dtype=np.intp
        )
        exogenous = np.concatenate(
            [trajectory.actions[rows], trajectory.disturbances[rows]], axis=-1
        )
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

        # Beobachtbarkeit: nicht inline gemessene Kanäle tragen keine Information in die
        # Historie. Maskiert wird **nach** der Standardisierung, damit die Null exakt dem
        # Mittelwert des Trainings-Splits entspricht und der Skalierer derselbe bleibt wie
        # im vollbeobachteten Fall -- sonst wären die beiden Betriebsarten nicht
        # vergleichbar, weil sich auch die Normierung der Eingänge geändert hätte.
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

        exogenous_names = tuple(trajectory.metadata.get("action_names", ())) + tuple(
            trajectory.metadata.get("disturbance_names", ())
        )
        return cls(
            _config=config,
            _exogenous_names=exogenous_names,
            _target_names=config.target_names,
            _inputs=standardised,
            _exogenous=input_scaler.select(range(len(exogenous_names))).transform(exogenous),
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
        """Fensterparameter."""
        return self._config

    @property
    def n_windows(self) -> int:
        """Anzahl Fenster."""
        return int(self._index.shape[0])

    @property
    def exogenous_names(self) -> tuple[str, ...]:
        """Stell- und Störgrößennamen."""
        return self._exogenous_names

    @property
    def target_names(self) -> tuple[str, ...]:
        """Zielgrößennamen."""
        return self._target_names

    @property
    def input_scaler(self) -> Standardizer:
        """Skalierer der Eingangskanäle."""
        return self._input_scaler

    @property
    def target_scaler(self) -> Standardizer:
        """Skalierer der Zielgrößen."""
        return self._target_scaler

    @property
    def scalers(self) -> tuple[Standardizer, Standardizer]:
        """Beide Skalierer, zur Weitergabe an val- und test-Datensätze."""
        return self._input_scaler, self._target_scaler

    def batch(self, indices: Sequence[int]) -> WindowBatch:
        """Fenster zu den gegebenen Indizes bilden."""
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
