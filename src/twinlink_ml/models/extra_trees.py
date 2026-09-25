r"""Extra Trees -- methodische Reproduktion von Kräuter, Zhou & Heinrich (ABS23).

**Kein Kandidat des Architekturvergleichs.** Dieses Modell steht bewusst nicht in
``models/registry.py``/``MODEL_SLOTS`` -- es dient ausschließlich dazu, den
Splitvergleich aus ``docs/04_stand_ap1.md`` (Abschnitt "Was das Paper des Vorgängers
ändert") auf demselben Datensatz zu messen: fenster-/punktweiser Split (Kräuters
Methodik, siehe :mod:`twinlink_ml.data.point_split`) gegen laufweisen Split (Standard
dieses Projekts). Der Abstand zwischen beiden Zahlen ist das Ergebnis, nicht eine
Extra-Trees-Genauigkeit für sich genommen. Siehe ``scripts/run_extra_trees_split_comparison.py``.

Was übernommen wird
--------------------
* :class:`sklearn.ensemble.ExtraTreesRegressor`, **ein** Regressor für alle Zielgrößen
  (native Multi-Output-Unterstützung von Baumregressoren -- kein
  ``MultiOutputRegressor``-Wrapper nötig).
* Merkmale: aktueller Wert sowie erste und zweite Differenz über :math:`t, t{-}1, t{-}2`
  (numerische Differenzen, kein Δt-normierter Schätzer) -- **kein** langes Fenster wie bei
  GRU/LSTM/xLSTM/Transformer/Mamba. Kanäle: alle über die Historie beobachtbaren Größen,
  also Stell- und Störgrößen (``exogenous_names``) **und** inline gemessene Zielgrößen
  (``observed_target_names``) -- analog zur Selbstrückführung in
  :class:`~twinlink_ml.models.baselines.LinearARXModel` und
  :class:`~twinlink_ml.models.narx.RandomFeatureNARX`. Nicht beobachtete Zielgrößenkanäle
  (Soft-Sensor-Fall) bleiben ausgeschlossen: sie stehen in der Historie konstant auf dem
  Trainingsmittelwert, eine Ableitung davon wäre eine Ableitung von Rauschen um eine
  Konstante.
* Ein-Schritt-Vorhersage (:math:`h = 1`), keine Mehrschritt-Rollout-Bewertung -- die
  Fensterkonfiguration für diesen Vergleich muss ``horizon=1`` setzen
  (:meth:`ExtraTreesModel.fit` prüft das).

Was bewusst NICHT übernommen wird
----------------------------------
* **Keine PCA.** Kräuters 12 Hauptkomponenten waren die Antwort auf 74 Rohmerkmale; der
  hiesige Merkmalsraum (Kanäle × 3) ist von vornherein klein, eine Dimensionsreduktion
  hätte hier keine Entsprechung.
* **Keine zentrierte Savitzky-Golay-Glättung der Zielgröße** -- nicht kausal, in einer
  Simulation ohne eigene Sensorrauschen-Pipeline (siehe
  :meth:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator.measure`, rein
  kausales PT1-Verzögerungsglied plus additives Rauschen) nicht sinnvoll übertragbar.
* **Keine Ausreißerentfernung** -- kein Sensorartefakt in der Simulation.

Hyperparameter
--------------
Kräuters eigene ``ExtraTreesRegressor``-Hyperparameter (Baumzahl, Tiefe, ...) sind in
``02_prozess_setup.md``/``04_stand_ap1.md`` nicht überliefert -- nur das Ergebnis
(R² = 0.994) und die Vorverarbeitung (PCA) stehen dort. Die Voreinstellungen hier sind
deshalb **sklearn-Standardwerte** (``n_estimators=100``, ``max_depth=None``, ...): dieser
Vergleich soll die Wirkung des Splits isolieren, nicht eine Baumkonfiguration
nachträglich optimieren.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.windows import SequenceDataset, WindowBatch
from ..simulation.backend import FloatArray
from .base import SequenceModel

__all__ = ["ExtraTreesModel"]


class ExtraTreesModel(SequenceModel):
    r"""Extra Trees auf Wert + 1./2. Differenz, Ein-Schritt-Vorhersage.

    Parameters
    ----------
    n_estimators
        Anzahl Bäume im Ensemble. sklearn-Standard.
    max_depth
        Maximale Baumtiefe. ``None`` (sklearn-Standard) heißt unbegrenzt, bis
        ``min_samples_leaf`` greift.
    min_samples_leaf
        sklearn-Standard.
    n_jobs
        Parallelität von ``fit``/``predict``. ``-1`` nutzt alle Kerne -- anders als bei
        den übrigen Modellen dieses Projekts kostet das hier keine Reproduzierbarkeit:
        Extra Trees hat keinen sequenziellen Optimierungspfad, den Threading-Reihenfolge
        beeinflussen könnte.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators muss >= 1 sein")
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._n_jobs = n_jobs
        self._regressor: Any = None
        self._feature_channels: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        """Kurzname mit Baumzahl."""
        return f"extra_trees{self._n_estimators}"

    @property
    def n_parameters(self) -> int:
        """Gesamtzahl der Knoten über alle Bäume.

        Kein Gewicht im ANN-Sinn, aber die vergleichbare Kapazitätskennzahl für ein
        Baumensemble.
        """
        if self._regressor is None:
            return 0
        return int(sum(tree.tree_.node_count for tree in self._regressor.estimators_))

    def _resolve_feature_channels(self, dataset: SequenceDataset) -> tuple[int, ...]:
        """Kanalindizes in ``history_inputs``, deren Wert + Differenzen Merkmale werden.

        Alle exogenen Kanäle plus die inline gemessenen Zielgrößen -- siehe Moduldocstring.
        """
        n_exogenous = len(dataset.exogenous_names)
        observed = set(dataset.observed_target_names)
        target_indices = tuple(
            n_exogenous + i
            for i, name in enumerate(dataset.target_names)
            if name in observed
        )
        return tuple(range(n_exogenous)) + target_indices

    def _features(self, history_inputs: FloatArray) -> FloatArray:
        """Wert + erste + zweite Differenz über die letzten drei Zeitschritte.

        ``history_inputs[:, -1]`` ist :math:`t`, ``[:, -2]`` ist :math:`t{-}1`,
        ``[:, -3]`` ist :math:`t{-}2` -- numerische Differenzen, kein Δt-normierter
        Ableitungsschätzer, wie bei Kräuter.
        """
        channels = list(self._feature_channels)
        current = history_inputs[:, -1, channels]
        previous = history_inputs[:, -2, channels]
        before = history_inputs[:, -3, channels]
        first_difference = current - previous
        second_difference = first_difference - (previous - before)
        return np.concatenate([current, first_difference, second_difference], axis=-1)

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        """Ein-Schritt-Regression auf Wert-/Differenzmerkmalen.

        ``validation`` bleibt ungenutzt: Extra Trees hat keinen Trainingsverlauf, der
        gestoppt werden könnte -- derselbe Grund wie bei den analytischen Baselines.
        """
        del validation
        if train.config.horizon != 1:
            raise ValueError(
                f"ExtraTreesModel ist eine Ein-Schritt-Aufgabe (horizon=1 wie bei "
                f"Kräuter), Fensterkonfiguration hat horizon={train.config.horizon}"
            )
        if train.config.history_length < 3:
            raise ValueError(
                f"Merkmale brauchen t, t-1, t-2 -- history_length={train.config.history_length}"
                " ist kleiner als 3"
            )
        self._feature_channels = self._resolve_feature_channels(train)

        from sklearn.ensemble import ExtraTreesRegressor

        batch = train.full_batch()
        design = self._features(batch.history_inputs)
        target = batch.future_targets[:, 0, :]
        self._regressor = ExtraTreesRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            n_jobs=self._n_jobs,
            random_state=seed,
        )
        self._regressor.fit(design, target)

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Direkte Ein-Schritt-Vorhersage, kein autoregressiver Rollout."""
        if self._regressor is None:
            raise RuntimeError("Modell ist nicht angepasst")
        if batch.future_exogenous.shape[1] != 1:
            raise ValueError(
                "ExtraTreesModel wertet nur h=1 aus, Batch hat horizon="
                f"{batch.future_exogenous.shape[1]}"
            )
        design = self._features(batch.history_inputs)
        prediction = np.asarray(self._regressor.predict(design), dtype=np.float64)
        if prediction.ndim == 1:
            prediction = prediction[:, None]
        return prediction[:, None, :]

    def describe(self) -> dict[str, Any]:
        """Konfiguration für das Tracking."""
        return {
            **super().describe(),
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "min_samples_leaf": self._min_samples_leaf,
        }
