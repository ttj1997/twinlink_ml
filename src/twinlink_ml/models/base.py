"""Abstraktion :class:`SequenceModel` -- gemeinsames Interface aller Vorhersagemodelle.

Der Vertrag ist bewusst schmal und **rollout-zentriert**: ein Modell bekommt einen
:class:`~twinlink_ml.data.windows.WindowBatch` und liefert die Zielgrößen über den
gesamten Horizont. Es gibt keine separate Ein-Schritt-Methode, weil der
Ein-Schritt-Fehler nur der Horizontschnitt :math:`h = 1` ist. Ein Interface mit zwei
Methoden würde erlauben, dass Ein-Schritt- und Rollout-Pfad auseinanderlaufen -- der
klassische Weg zu einem Modell, das die Nebenmetrik gewinnt und im Rollout divergiert.

Reihenfolge in AP 1: dieses Interface wird zuerst von den drei rechenfreien Baselines
implementiert (Persistenz, statische Regression, linearer ARX) und erst danach vom GRU.
Umgekehrt formt sich das Interface um das neuronale Netz, und die Baselines passen nur
mit Sonderfällen hinein.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..data.windows import SequenceDataset, WindowBatch
from ..simulation.backend import FloatArray

__all__ = ["SequenceModel"]


class SequenceModel(ABC):
    """Vorhersagemodell für Zielgrößentrajektorien."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Kurzname für Logs und Lernkurvenlegenden."""

    @abstractmethod
    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        """Anpassen an den Trainings-Datensatz.

        Parameters
        ----------
        train
            Trainings-Datensatz. Alle Statistiken, die das Modell braucht, kommen aus
            diesem Objekt -- nie aus dem vollen Lauf-Pool.
        seed
            Modell-Seed aus dem ``SeedBundle``, getrennt von Daten- und Rausch-Seed.
        validation
            Optional für Early Stopping. Bei den analytischen Baselines ungenutzt; sie
            haben keinen Trainingsverlauf, der gestoppt werden könnte.
        """

    @abstractmethod
    def predict(self, batch: WindowBatch) -> FloatArray:
        """Zielgrößen über den Horizont, Form ``(batch, H, n_targets)``, standardisiert.

        Standardisiert, nicht in Ingenieurseinheiten: die Rückrechnung gehört in den
        Evaluator, der den Skalierer des Trainings-Datensatzes kennt. Sonst müsste jedes
        Modell die Rückrechnung selbst korrekt implementieren.
        """

    @property
    def n_parameters(self) -> int:
        """Anzahl freier Parameter. Für die Einordnung der Lernkurven relevant."""
        return 0

    def describe(self) -> dict[str, Any]:
        """Konfiguration für das Experiment-Tracking."""
        return {"model": self.name, "n_parameters": self.n_parameters}

    def training_diagnostics(self) -> dict[str, Any]:
        """Trainingsverlauf-Kennzahlen für das Tracking, leer ohne Trainingsschleife.

        Modelle ohne iterative Optimierung (Persistenz, ARX, NARX, ...) haben keinen
        Trainingsverlauf, der gestoppt werden könnte -- das leere Dict ist für sie
        korrekt, nicht nur ein Platzhalter.
        """
        return {}
