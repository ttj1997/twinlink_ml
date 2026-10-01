"""Abstraktion ``ProcessSimulator`` und Trajektorien-Container.

Diese Schnittstelle ist der eigentliche Ertrag von Schritt 1: sie ist gleichzeitig die
**Schnittstellenspezifikation gegenüber TUHH-A und dem Software-Integrationspartner**. Wer einen
Datensatz
liefern kann, der als :class:`Trajectory` darstellbar ist, kann ohne Änderung am
Trainingscode eingehängt werden.

Geplante Implementierungen:

* :class:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator` -- jetzt
* ``DyssolSimulator`` -- ab Verfügbarkeit der Python-Bibliothek des
  Software-Integrationspartners (M7)
* ``ReplaySimulator`` -- Wiedergabe gemessener Anwendungspartner-/Labordaten als Trajektorien
* ``SurrogateSimulator`` -- gelerntes Modell als Environment für TD3/SAC

Der Rollout ist als Template-Methode in der Basisklasse implementiert: eine neue
Datenquelle muss nur ``step`` und die Namenslisten bereitstellen (Open-Closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backend import Array, ArrayBackend, FloatArray

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class StepResult:
    """Ergebnis eines Abtastschritts."""

    state: Array
    """Zustandsvektor, Form ``(batch, n_state)``."""

    outputs: Array
    """Ausgangs-/Zielgrößen, Form ``(batch, n_output)``."""

    diagnostics: dict[str, Array]
    """Nicht-Ziel-Größen für Plausibilisierung und Constraint-Prüfung."""


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Ein Simulationslauf (oder ein Batch davon).

    Die Achsenkonvention ist ``(batch, time, channel)`` -- identisch zu der, die die
    Fensterung in ``data/`` erwartet. ``run_id`` ist die Einheit, auf der die Splits
    gebildet werden (**nie** auf Fensterebene, siehe Arbeitsplan Schritt 5).
    """

    run_id: tuple[str, ...]
    time_s: FloatArray
    actions: FloatArray
    disturbances: FloatArray
    states: FloatArray
    outputs: FloatArray
    measurements: FloatArray
    diagnostics: dict[str, FloatArray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    requested_actions: FloatArray | None = None
    """Aktion VOR einer Sicherheitsprojektion (z. B. Fluidisierungsfenster), Form wie
    ``actions``. ``actions`` bleibt die REALISIERTE Aktion -- sie speist Modellmerkmale und
    Massenbilanz und muss zum integrierten Zustand passen. Nur populiert, wenn
    ``projection_active`` -- auch dann, wenn die Projektion an keinem Punkt gegriffen hat
    (dann bitgleich zu ``actions``). ``None`` heißt: kein Projektionsmechanismus konfiguriert,
    nicht nur "hat nicht gegriffen" -- siehe ``projection_active``."""
    projection_active: bool = False
    """Ob ein Projektionsmechanismus fuer diesen Rollout konfiguriert war -- eine Eigenschaft
    der Simulation (des Prozesssimulators/seiner Config), nicht eines einzelnen Laufs: pro
    Rollout entweder aktiv oder nicht, deshalb ein Skalar und kein Array, obwohl ``actions``/
    ``states``/etc. Batch-Felder sind. Ob die Projektion an einem konkreten Lauf tatsächlich
    gegriffen hat, steht in ``n_projected_steps`` -- "aktiv" und "hat eingegriffen" sind zwei
    verschiedene Fakten, siehe Feld-Docstrings."""
    n_projected_steps: IntArray | None = None
    """Anzahl Schritte je Lauf, an denen die Projektion tatsaechlich eingegriffen hat, Form
    ``(n_batch,)``, dtype int. Anders als ``projection_active`` IST dies laufspezifisch (ein
    Lauf kann durchgehend im zulaessigen Fenster bleiben, ein anderer nicht). ``None`` nur
    wenn ``projection_active`` False ist; sonst immer ein Array (ggf. voller Nullen)."""

    def __post_init__(self) -> None:
        n_batch, n_time = self.actions.shape[0], self.actions.shape[1]
        if len(self.run_id) != n_batch:
            raise ValueError(
                f"run_id hat {len(self.run_id)} Einträge, Batch-Dimension ist {n_batch}"
            )
        for name in ("disturbances", "states", "outputs", "measurements"):
            array = getattr(self, name)
            if array.shape[:2] != (n_batch, n_time):
                raise ValueError(
                    f"{name} hat Form {array.shape}, erwartet ({n_batch}, {n_time}, ...)"
                )
        if self.time_s.shape[-1] != n_time:
            raise ValueError("time_s passt nicht zur Zeitdimension")
        if self.projection_active:
            if self.requested_actions is None or self.n_projected_steps is None:
                raise ValueError(
                    "projection_active=True verlangt requested_actions und "
                    "n_projected_steps -- 'aktiv' ohne Wirkungsnachweis waere nicht "
                    "auditierbar"
                )
            if self.requested_actions.shape != self.actions.shape:
                raise ValueError(
                    f"requested_actions hat Form {self.requested_actions.shape}, "
                    f"erwartet {self.actions.shape} wie actions"
                )
            if self.n_projected_steps.shape != (n_batch,):
                raise ValueError(
                    f"n_projected_steps hat Form {self.n_projected_steps.shape}, "
                    f"erwartet ({n_batch},)"
                )
        elif self.requested_actions is not None or self.n_projected_steps is not None:
            raise ValueError(
                "requested_actions/n_projected_steps gesetzt, aber projection_active=False -- "
                "widersprüchlicher Zustand (Struktur ohne Bedeutung), siehe Feld-Docstrings"
            )

    @property
    def n_runs(self) -> int:
        return int(self.actions.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.actions.shape[1])

    def select(self, index: int) -> Trajectory:
        """Einzelnen Lauf herauslösen, Batch-Dimension bleibt erhalten."""
        sl = slice(index, index + 1)
        return Trajectory(
            run_id=(self.run_id[index],),
            time_s=self.time_s,
            actions=self.actions[sl],
            disturbances=self.disturbances[sl],
            states=self.states[sl],
            outputs=self.outputs[sl],
            measurements=self.measurements[sl],
            diagnostics={k: v[sl] for k, v in self.diagnostics.items()},
            metadata=dict(self.metadata),
            requested_actions=(
                None if self.requested_actions is None else self.requested_actions[sl]
            ),
            projection_active=self.projection_active,
            n_projected_steps=(
                None if self.n_projected_steps is None else self.n_projected_steps[sl]
            ),
        )

    def concatenate(self, other: Trajectory) -> Trajectory:
        """Läufe zusammenführen. Vorbedingung: gleiches Zeitraster und ``projection_active``.

        Unterschiedliches ``projection_active`` wird nicht stillschweigend zusammengefuehrt --
        das wuerde Laeufe mit und ohne Sicherheitsprojektion ununterscheidbar vermischen, ein
        Datenintegritaetsfehler derselben Art wie der bereits geprueften Zeitraster-Vorbedingung.
        """
        if not np.allclose(self.time_s, other.time_s):
            raise ValueError("Trajektorien mit unterschiedlichem Zeitraster")
        if self.projection_active != other.projection_active:
            raise ValueError(
                "Trajektorien mit unterschiedlichem projection_active koennen nicht "
                "zusammengefuehrt werden -- das wuerde projizierte und unprojizierte "
                "Laeufe ununterscheidbar vermischen"
            )
        shared = set(self.diagnostics) & set(other.diagnostics)
        requested_actions = None
        n_projected_steps = None
        if self.projection_active:
            assert self.requested_actions is not None
            assert other.requested_actions is not None
            assert self.n_projected_steps is not None
            assert other.n_projected_steps is not None
            requested_actions = np.concatenate(
                [self.requested_actions, other.requested_actions], axis=0
            )
            n_projected_steps = np.concatenate(
                [self.n_projected_steps, other.n_projected_steps], axis=0
            )
        return Trajectory(
            run_id=self.run_id + other.run_id,
            time_s=self.time_s,
            actions=np.concatenate([self.actions, other.actions], axis=0),
            disturbances=np.concatenate([self.disturbances, other.disturbances], axis=0),
            states=np.concatenate([self.states, other.states], axis=0),
            outputs=np.concatenate([self.outputs, other.outputs], axis=0),
            measurements=np.concatenate([self.measurements, other.measurements], axis=0),
            diagnostics={
                k: np.concatenate([self.diagnostics[k], other.diagnostics[k]], axis=0)
                for k in sorted(shared)
            },
            metadata={**self.metadata, **other.metadata},
            requested_actions=requested_actions,
            projection_active=self.projection_active,
            n_projected_steps=n_projected_steps,
        )


class ProcessSimulator(ABC):
    """Quelle von Trajektorien für Training und Evaluation."""

    @property
    @abstractmethod
    def backend(self) -> ArrayBackend: ...

    @property
    @abstractmethod
    def action_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def disturbance_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def state_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def output_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def sample_time_s(self) -> float: ...

    @abstractmethod
    def initial_state(self, n_batch: int) -> Array:
        """Zustandsvektor der Form ``(batch, n_state)`` im Anfangsbetriebspunkt."""

    @abstractmethod
    def step(self, state: Array, action: Array, disturbance: Array) -> StepResult:
        """Ein Abtastschritt von ``t`` nach ``t + sample_time_s``."""

    @abstractmethod
    def measure(self, outputs: Array, rng: np.random.Generator) -> Array:
        """Sensormodell: verrauschte, verzögerte Beobachtung der Ausgangsgrößen."""

    def time_constants(self, state: Array, action: Array, disturbance: Array) -> dict[str, float]:
        """Zeitkonstanten im linearisierten Arbeitspunkt, falls das Modell sie liefert.

        Optionale Fähigkeit statt Pflichtmethode: das Grey-Box-Modell kann seine
        Jacobi-Matrix numerisch differenzieren, ein künftiger Dyssol-Adapter oder ein
        gelerntes Surrogat womöglich nicht. Die Standardimplementierung wirft, damit
        `simulator.time_constants(...)` unter mypy strict typgeprüft bleibt -- ohne diese
        Deklaration bräuchte jeder Aufrufer ein `# type: ignore[attr-defined]`, das bei der
        nächsten Refaktorierung so leicht unbemerkt falsch wird wie richtig.

        :raises NotImplementedError: wenn der Simulator keine Zeitkonstanten unterstützt.
            Aufrufer, die das behandeln müssen, prüfen vorher mit
            ``hasattr(simulator, "time_constants")`` -- das bleibt gültig, weil die Methode
            auf jeder Instanz existiert, nur eben manchmal nicht implementiert ist.
        """
        raise NotImplementedError(f"{type(self).__name__} liefert keine Zeitkonstanten")

    def reset_sensor(self) -> None:  # noqa: B027 - absichtlich nicht abstrakt
        """Internen Zustand des Sensormodells verwerfen.

        Notwendig, weil das Sensormodell ein Verzögerungsglied erster Ordnung enthält und
        damit **Zustand über die Zeitschritte hinweg** trägt. Wird dieselbe
        Simulatorinstanz zweimal ausgerollt, startet der Filter sonst mit dem Endzustand
        des vorigen Laufs: eine Kopplung zwischen Läufen, die es im Prozess nicht gibt
        und die als Fehler nirgends sichtbar wäre. Die Standardimplementierung tut
        nichts -- ein Simulator ohne Sensordynamik hat nichts zurückzusetzen.
        """

    @property
    def projection_active(self) -> bool:
        """Ob :meth:`project_action` fuer diesen Simulator tatsaechlich etwas clippt.

        Generischer Schalter-Zustand fuer :meth:`rollout`: bestimmt, ob die Buchfuehrung
        (``requested_actions``/``n_projected_steps`` auf der resultierenden
        :class:`Trajectory`) ueberhaupt gefuehrt wird. Default False -- ein Simulator ohne
        Sicherheitsprojektion (Dyssol-Adapter, Replay, Surrogat) muss nichts überschreiben.
        """
        return False

    def project_action(self, state: Array, action: Array, disturbance: Array) -> Array:
        """Stoffspezifische Sicherheitsprojektion, VOR dem Schritt am wahren Zustand.

        Generischer Aufrufpunkt in :meth:`rollout`, direkt vor ``step()`` -- Vorgabe aus dem
        Auftrag: Margen wie Fluidisierungsgrenzen sind zustandsabhängig und müssen aus dem
        WAHREN, unverrauschten Zustand berechnet werden (nie aus einer Messung), unmittelbar
        vor dem Integrationsschritt, dessen Aktion sie einschränken. Die generische
        Basisimplementierung tut nichts -- Simulatoren mit einer solchen Projektion (siehe
        :meth:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator.project_action`)
        überschreiben sowohl diese Methode als auch :attr:`projection_active`.
        """
        del state, disturbance  # ungenutzt in der generischen No-Op-Basisimplementierung
        return action

    def remeasure(self, outputs: FloatArray, *, seed: int) -> FloatArray:
        """Sensormodell erneut auf bereits berechnete Ausgangsgrößen anwenden.

        Warum das eine eigene Methode ist und nicht ein zweiter ``rollout``: das
        Prozessmodell ist deterministisch, der Seed betrifft **ausschließlich** das
        Sensormodell. Ein Rausch-Seed oder ein Rauschniveau zu variieren erfordert
        deshalb keine neue Integration. Für Schritt 5 ist das der Unterschied zwischen
        einer Simulation je Datensatz-Realisierung und einer je Kombination aus
        Datensatz, Rausch-Seed und Rauschniveau.

        Die Schleife über die Zeit ist absichtlich dieselbe wie in :meth:`rollout` --
        eine vektorisierte Variante würde den Zufallsgenerator in anderer Reihenfolge
        verbrauchen und wäre nicht bitgleich zum Rollout. Ein Test sichert die Gleichheit
        zu.
        """
        self.reset_sensor()
        backend = self.backend
        rng = np.random.default_rng(seed)
        values = np.asarray(outputs, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("outputs muss die Form (batch, time, channel) haben")
        return np.stack(
            [
                backend.to_numpy(self.measure(backend.asarray(values[:, k, :]), rng))
                for k in range(values.shape[1])
            ],
            axis=1,
        )

    def rollout(
        self,
        actions: FloatArray,
        disturbances: FloatArray,
        *,
        run_id: tuple[str, ...] | None = None,
        seed: int = 0,
        initial_state: Array | None = None,
    ) -> Trajectory:
        """Trajektorie zu vorgegebenen Anregungs- und Störsignalen.

        Template-Methode: die Schleife ist hier festgelegt, die Physik in ``step``.

        :param actions: Form ``(batch, time, n_action)`` in Ingenieurseinheiten
        :param disturbances: Form ``(batch, time, n_disturbance)``
        :param seed: Seed **ausschließlich** für das Sensormodell. Das Prozessmodell
            selbst ist deterministisch -- Reproduzierbarkeit hängt damit nur an
            (Config, Anregung, Seed).
        """
        self._validate_inputs(actions, disturbances)
        n_batch, n_time, _ = actions.shape
        backend = self.backend
        rng = np.random.default_rng(seed)

        state = self.initial_state(n_batch) if initial_state is None else initial_state
        self.reset_sensor()
        states: list[FloatArray] = []
        outputs: list[FloatArray] = []
        measurements: list[FloatArray] = []
        diagnostics: dict[str, list[FloatArray]] = {}

        # Generischer Projektions-Schalter: Buchfuehrung (requested_actions/
        # n_projected_steps) wird nur gefuehrt, wenn ueberhaupt konfiguriert -- sonst bleibt
        # der Rollout so leicht wie zuvor und die Trajectory bitgleich zum Vor-Projektions-
        # Stand (siehe Trajectory-Feld-Docstrings und den Regressionstest dazu).
        projection_active = self.projection_active
        realised_actions_steps: list[FloatArray] = []
        requested_actions_steps: list[FloatArray] = []
        n_projected_steps = np.zeros(n_batch, dtype=np.int64) if projection_active else None

        for k in range(n_time):
            requested_action = backend.asarray(actions[:, k, :])
            disturbance = backend.asarray(disturbances[:, k, :])
            # `state` ist hier der interne, WAHRE Simulatorzustand dieser Iteration --
            # unveraendert von self.measure() weiter unten. project_action() darf sich
            # deshalb nie auf eine Messung stuetzen, siehe dessen Vertrag.
            action = self.project_action(state, requested_action, disturbance)
            action_np = backend.to_numpy(action)
            realised_actions_steps.append(action_np)
            if projection_active:
                requested_np = backend.to_numpy(requested_action)
                requested_actions_steps.append(requested_np)
                assert n_projected_steps is not None
                n_projected_steps = n_projected_steps + np.any(action_np != requested_np, axis=-1)
            result = self.step(state, action, disturbance)
            state = result.state
            states.append(backend.to_numpy(state))
            outputs.append(backend.to_numpy(result.outputs))
            measurements.append(backend.to_numpy(self.measure(result.outputs, rng)))
            for key, value in result.diagnostics.items():
                diagnostics.setdefault(key, []).append(backend.to_numpy(value))

        requested_actions = np.stack(requested_actions_steps, axis=1) if projection_active else None

        return Trajectory(
            run_id=run_id if run_id is not None else tuple(f"run_{i:05d}" for i in range(n_batch)),
            time_s=np.arange(1, n_time + 1, dtype=np.float64) * self.sample_time_s,
            # REALISIERTE Aktion (nach project_action()) -- speist Modellmerkmale und
            # Massenbilanz und muss zum integrierten Zustand passen. Bei
            # projection_active=False bitgleich zum rohen `actions`-Argument, weil
            # project_action() dann fuer jeden Schritt die Identitaet ist.
            actions=np.stack(realised_actions_steps, axis=1),
            disturbances=np.asarray(disturbances, dtype=np.float64),
            states=np.stack(states, axis=1),
            outputs=np.stack(outputs, axis=1),
            measurements=np.stack(measurements, axis=1),
            diagnostics={k: np.stack(v, axis=1) for k, v in diagnostics.items()},
            metadata=self.run_metadata(seed=seed),
            requested_actions=requested_actions,
            projection_active=projection_active,
            n_projected_steps=n_projected_steps,
        )

    def run_metadata(self, *, seed: int) -> dict[str, Any]:
        """Wird in jede Trajektorie geschrieben und später ins Experiment-Tracking."""
        return {
            "simulator": type(self).__name__,
            "backend": self.backend.name,
            "sample_time_s": self.sample_time_s,
            "seed": seed,
            "action_names": self.action_names,
            "disturbance_names": self.disturbance_names,
            "state_names": self.state_names,
            "output_names": self.output_names,
        }

    def _validate_inputs(self, actions: FloatArray, disturbances: FloatArray) -> None:
        if actions.ndim != 3 or disturbances.ndim != 3:
            raise ValueError(
                "actions und disturbances müssen die Form (batch, time, channel) haben"
            )
        if actions.shape[:2] != disturbances.shape[:2]:
            raise ValueError("actions und disturbances haben unterschiedliche Batch-/Zeitform")
        if actions.shape[2] != len(self.action_names):
            raise ValueError(
                f"{actions.shape[2]} Stellgrößen übergeben, erwartet {len(self.action_names)}: "
                f"{self.action_names}"
            )
        if disturbances.shape[2] != len(self.disturbance_names):
            raise ValueError(
                f"{disturbances.shape[2]} Störgrößen übergeben, "
                f"erwartet {len(self.disturbance_names)}: {self.disturbance_names}"
            )
