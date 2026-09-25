r"""Metriken und Evaluationsprotokoll (Arbeitsplan Schritt 5).

Rangfolge der Metriken -- bewusst gegen die übliche Praxis
---------------------------------------------------------
1. **Mehrschritt-Rollout-Fehler** über mehrere Horizonte -- primär.
2. Ein-Schritt-Fehler -- nur noch Nebenmetrik. Bei :math:`\Delta t = 5` s liegt der
   Persistenz-NRMSE bei 0.016; ein Vergleich auf dieser Metrik misst Seed-Varianz.
3. Inferenzzeit -- Echtzeitfähigkeit ist ein Auswahlkriterium in AP 1.

Normierung
----------
.. math:: \mathrm{NRMSE} = \frac{\sqrt{\overline{(y - \hat y)^2}}}{\sigma_y}

mit :math:`\sigma_y` aus dem **Evaluations**-Split, nicht aus dem Training. Damit ist
1.0 die Schwelle, unterhalb derer ein Modell überhaupt besser ist als die Vorhersage des
Mittelwerts, und die Zahl ist über Zielgrößen mit völlig verschiedenen Einheiten
(kW, kg/kg, µm) vergleichbar.

Zwei Referenzen pro Zahl
------------------------
``nrmse`` misst gegen die verrauschte Messung -- das ist, was an der Anlage prüfbar ist.
``nrmse_truth`` misst gegen die rauschfreie Ausgangsgröße des Grey-Box-Modells. Die
Differenz beider ist der irreduzible Anteil und damit die **Sättigungshöhe der
Lernkurve**. Ohne diese zweite Zahl liest man eine Lernkurve, die bei 0.12 abflacht, als
Modellgrenze, obwohl sie die Rauschgrenze sein kann.

Laufweise Normierung ist die Primärzahl, nicht die gepoolte
-----------------------------------------------------------
Wird :math:`\sigma_y` über alle Läufe gepoolt geschätzt, enthält es die
Niveauunterschiede **zwischen** den Läufen. Die entstehen durch die gestreuten
Anfangszustände (Anregungsdesign, Schritt 2) und müssen von keinem Modell vorhergesagt
werden -- sie stehen in der Fensterhistorie. Der Nenner ist damit künstlich groß.

Gemessen am Grey-Box-Modell, :math:`\Delta t = 25` s, ARX(4,2), :math:`h = 24`:

============================  ========  ==========
Zielgröße                     gepoolt   pro Lauf
============================  ========  ==========
``energy_target``               0.133      0.154
``product_moisture_kg_per_kg``  0.043      1.354
``d50_um``                      0.026      0.091
============================  ========  ==========

Bei der Produktfeuchte wechselt damit die Aussage: gepoolt liest sich die Zeile als
"praktisch gelöst", laufweise liegt der Fehler über der Streuung innerhalb des Laufs --
das Modell schlägt den Mittelwert des eigenen Laufs nicht. Der Effekt ist bei
``energy_target`` klein und bei den Qualitätsgrößen groß, also genau dort, wo der
Architekturvergleich stattfindet. Deshalb ist der **Median über Läufe** die berichtete
Zahl und die gepoolte nur Nebeninformation.

Fenster desselben Laufs sind zusätzlich stark korreliert; die laufweise Aggregation gibt
damit auch eine belastbarere Streuungsangabe als die Fensterzahl suggeriert.

Absoluter RMSE zusätzlich zur NRMSE
------------------------------------
Der Nenner ``np.std(reference)`` in :func:`nrmse` wird **je Lauf** neu geschätzt (siehe
oben) und streut selbst stark zwischen den Läufen -- gemessen an einem Konstantlauf-Holdout
des Grey-Box-Modells (``scripts/probe_e0_normalisation_artifact.py``,
``artifacts/runs/probe_e0_normalisation_artifact.json``) um den Faktor 26.6 bei ``d50_um``
und um den Faktor 2292.7 bei ``product_moisture_kg_per_kg`` (Minimum zu Maximum über alle
geprüften Läufe, Seed 1, GRU/LSTM, h=1/h=36). Die NRMSE allein ist damit nicht in einen
physikalischen Fehler übersetzbar: derselbe NRMSE-Wert kann aus einem kleinen absoluten
Fehler bei einem ruhigen Lauf (kleiner Nenner) oder einem großen absoluten Fehler bei einem
stark streuenden Lauf (großer Nenner) stammen. :func:`per_run_rmse_median` und
``EvaluationResult.rmse_run_median_physical`` liefern deshalb zusätzlich den unnormierten
RMSE in physikalischen Einheiten (nach ``target_scaler.inverse()``, unabhängig vom
``physical_units``-Schalter von :class:`RolloutEvaluator`) -- als Ergänzung, nicht als
Ersatz: die laufweise NRMSE bleibt die Primärzahl für Modellvergleiche, weil sie Zielgrößen
unterschiedlicher Einheit vergleichbar macht; der absolute RMSE beantwortet die davon
getrennte Frage, wie groß der Fehler in der Einheit der Zielgröße tatsächlich ist.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.windows import SequenceDataset, WindowBatch
from ..models.base import SequenceModel
from ..simulation.backend import FloatArray

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "RolloutEvaluator",
    "nrmse",
    "per_run_nrmse",
    "per_run_rmse_median",
    "pooled_nrmse",
]


def nrmse(
    prediction: FloatArray, reference: FloatArray, *, axis: tuple[int, ...] = (0,)
) -> FloatArray:
    """NRMSE, normiert auf die Streuung der Referenz entlang derselben Achsen."""
    residual = prediction - reference
    rmse = np.sqrt(np.mean(residual**2, axis=axis))
    scale = np.std(reference, axis=axis)
    return np.where(scale > 0.0, rmse / np.maximum(scale, 1e-30), np.nan)


def pooled_nrmse(
    prediction: FloatArray,
    reference: FloatArray,
    horizons: Sequence[int],
    names: Sequence[str],
) -> dict[str, dict[int, float]]:
    r"""NRMSE, gepoolt über alle Fenster und die Schritte :math:`1..h`.

    Kumulativ und nicht punktuell: der Rollout-Fehler bei :math:`h` soll den Weg dorthin
    enthalten. Ein Modell, das bei :math:`h = 12` zufällig wieder nahe an der Wahrheit
    liegt, nachdem es dazwischen divergiert ist, darf nicht gut aussehen.

    Modulfunktion und nicht Methode, damit dieselbe Normierungskonvention auch außerhalb
    eines Modellvergleichs benutzbar ist -- etwa für die Rauschgrenze in
    :mod:`twinlink_ml.evaluation.protocol`, die kein Modell auswertet.
    """
    result: dict[str, dict[int, float]] = {name: {} for name in names}
    for horizon in horizons:
        values = nrmse(prediction[:, :horizon, :], reference[:, :horizon, :], axis=(0, 1))
        for column, name in enumerate(names):
            result[name][horizon] = float(values[column])
    return result


def per_run_nrmse(
    prediction: FloatArray,
    reference: FloatArray,
    runs: FloatArray,
    horizons: Sequence[int],
    names: Sequence[str],
) -> dict[str, dict[int, tuple[float, float, float]]]:
    """NRMSE je Lauf, zusammengefasst als ``(Median, Minimum, Maximum)``.

    Die **Primärzahl** des Protokolls, siehe Modul-Docstring: der Nenner enthält damit
    nur die Streuung innerhalb eines Laufs und nicht die Niveauunterschiede zwischen den
    Läufen, die aus den gestreuten Anfangszuständen kommen.
    """
    result: dict[str, dict[int, tuple[float, float, float]]] = {name: {} for name in names}
    unique = np.unique(runs)
    for horizon in horizons:
        per_run = np.stack(
            [
                nrmse(
                    prediction[runs == run, :horizon, :],
                    reference[runs == run, :horizon, :],
                    axis=(0, 1),
                )
                for run in unique
            ],
            axis=0,
        )
        for column, name in enumerate(names):
            column_values = per_run[:, column]
            result[name][horizon] = (
                float(np.nanmedian(column_values)),
                float(np.nanmin(column_values)),
                float(np.nanmax(column_values)),
            )
    return result


def per_run_rmse_median(
    prediction: FloatArray,
    reference: FloatArray,
    runs: FloatArray,
    horizons: Sequence[int],
    names: Sequence[str],
) -> dict[str, dict[int, float]]:
    """Unnormierter RMSE je Lauf, Median über die Läufe -- Ergänzung zur NRMSE.

    ``prediction``/``reference`` müssen bereits in physikalischen Einheiten vorliegen
    (``target_scaler.inverse()``), unabhängig davon, ob der Aufrufer die NRMSE selbst
    standardisiert oder physikalisch berechnet -- siehe Moduldocstring, Abschnitt
    "Absoluter RMSE zusätzlich zur NRMSE". Nur der Median, keine Spannweite: die
    Streuung des absoluten Fehlers über Läufe wird bereits über den NRMSE-Nenner
    (``nrmse_run_min``/``_max``) sichtbar, hier geht es um das Niveau.
    """
    result: dict[str, dict[int, float]] = {name: {} for name in names}
    unique = np.unique(runs)
    for horizon in horizons:
        per_run = np.stack(
            [
                np.sqrt(
                    np.mean(
                        (prediction[runs == run, :horizon, :] - reference[runs == run, :horizon, :])
                        ** 2,
                        axis=(0, 1),
                    )
                )
                for run in unique
            ],
            axis=0,
        )
        for column, name in enumerate(names):
            result[name][horizon] = float(np.nanmedian(per_run[:, column]))
    return result


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Ergebnis einer Evaluation, flach genug für MLflow-/W&B-Metriken.

    Attributes
    ----------
    model_name
        Kurzname des Modells.
    target_names
        Zielgrößen in Spaltenreihenfolge.
    horizons
        Ausgewertete Horizonte in Abtastschritten; ``1`` ist der Ein-Schritt-Fehler.
    nrmse_by_horizon
        ``{Zielgröße: {Horizont: NRMSE}}`` gegen die Messung, über alle Fenster gepoolt.
    nrmse_truth_by_horizon
        Dasselbe gegen die rauschfreie Referenz, oder leer.
    nrmse_run_spread
        ``{Zielgröße: {Horizont: (Median, Minimum, Maximum)}}`` über Läufe -- die
        Primärzahl.
    nrmse_truth_run_spread
        Dasselbe gegen die rauschfreie Referenz. Ohne diese Zahl ist die Sättigungshöhe
        einer Lernkurve nicht in derselben Normierung lesbar wie die Kurve selbst.
    rmse_run_median_physical
        ``{Zielgröße: {Horizont: Median-RMSE}}`` über Läufe, unnormiert, in physikalischen
        Einheiten -- Ergänzung zur NRMSE, siehe Moduldocstring "Absoluter RMSE zusätzlich
        zur NRMSE". Immer in physikalischen Einheiten, unabhängig vom
        ``physical_units``-Schalter von :class:`RolloutEvaluator`.
    inference_us_per_window
        Inferenzzeit je Fenster für den vollen Horizont, in Mikrosekunden.
    n_windows, n_runs
        Umfang der Auswertung.
    """

    model_name: str
    target_names: tuple[str, ...]
    horizons: tuple[int, ...]
    nrmse_by_horizon: dict[str, dict[int, float]]
    nrmse_truth_by_horizon: dict[str, dict[int, float]]
    nrmse_run_spread: dict[str, dict[int, tuple[float, float, float]]]
    inference_us_per_window: float
    n_windows: int
    n_runs: int
    nrmse_truth_run_spread: dict[str, dict[int, tuple[float, float, float]]] = field(
        default_factory=dict
    )
    rmse_run_median_physical: dict[str, dict[int, float]] = field(default_factory=dict)

    def as_flat_metrics(self) -> dict[str, float]:
        """Flache Schlüssel-Wert-Paare für das Experiment-Tracking.

        Enthält beide Normierungen und beide Referenzen. Die laufweisen Kennzahlen sind
        hier und nicht erst im Treiber, damit es nur **eine** Stelle gibt, an der ein
        Metrikname entsteht -- sonst laufen Log-Schlüssel und Auswertung auseinander.
        """
        flat: dict[str, float] = {
            "inference_us_per_window": self.inference_us_per_window,
            "n_windows": float(self.n_windows),
            "n_test_runs": float(self.n_runs),
        }
        for target, pooled_by_horizon in self.nrmse_by_horizon.items():
            for horizon, value in pooled_by_horizon.items():
                flat[f"nrmse/{target}/h{horizon}"] = value
        for target, truth_by_horizon in self.nrmse_truth_by_horizon.items():
            for horizon, value in truth_by_horizon.items():
                flat[f"nrmse_truth/{target}/h{horizon}"] = value
        # Eigener Variablenname je Schleife, absichtlich: dieselbe Bezeichnung für einen
        # `dict[int, float]` weiter oben und einen `dict[int, tuple[float, float, float]]`
        # hier würde mypy als inkompatible Neuzuweisung ablehnen -- das ist keine
        # Formalie, sondern derselbe Fehler, den ein Mensch beim Lesen auch verwechseln
        # würde: eine gepoolte Zahl ist keine Median/Min/Max-Spanne.
        for prefix, spread_by_target in (
            ("nrmse_run", self.nrmse_run_spread),
            ("nrmse_truth_run", self.nrmse_truth_run_spread),
        ):
            for target, spread_by_horizon in spread_by_target.items():
                for horizon, (median, minimum, maximum) in spread_by_horizon.items():
                    flat[f"{prefix}_median/{target}/h{horizon}"] = median
                    flat[f"{prefix}_min/{target}/h{horizon}"] = minimum
                    flat[f"{prefix}_max/{target}/h{horizon}"] = maximum
        for target, rmse_by_horizon in self.rmse_run_median_physical.items():
            for horizon, value in rmse_by_horizon.items():
                flat[f"rmse_run_median_physical/{target}/h{horizon}"] = value
        return flat

    def format(self) -> str:
        """Tabelle für Run-Logs."""
        header = f"{self.model_name}  ({self.n_runs} Läufe, {self.n_windows} Fenster)"
        columns = "".join(f"{f'h={h}':>10}" for h in self.horizons)
        lines = [header, f"{'Zielgröße':<28}{columns}"]
        for target in self.target_names:
            primary = "".join(
                f"{self.nrmse_run_spread[target][h][0]:>10.4f}" for h in self.horizons
            )
            lines.append(f"{target:<28}{primary}")
            pooled = "".join(f"{self.nrmse_by_horizon[target][h]:>10.4f}" for h in self.horizons)
            lines.append(f"{'  gepoolt (nicht primär)':<28}{pooled}")
            if target in self.nrmse_truth_run_spread:
                against_truth = "".join(
                    f"{self.nrmse_truth_run_spread[target][h][0]:>10.4f}" for h in self.horizons
                )
                lines.append(f"{'  gegen rauschfrei':<28}{against_truth}")
        lines.append(f"Inferenz: {self.inference_us_per_window:.1f} µs/Fenster")
        return "\n".join(lines)


class Evaluator(ABC):
    """Abstraktion über Evaluationsprotokolle."""

    @abstractmethod
    def evaluate(self, model: SequenceModel, dataset: SequenceDataset) -> EvaluationResult:
        """Modell auf einem Datensatz bewerten."""


@dataclass(frozen=True, slots=True)
class RolloutEvaluator(Evaluator):
    """Standardprotokoll: NRMSE über mehrere Horizonte, laufweise Streuung, Inferenzzeit.

    Attributes
    ----------
    horizons
        Auszuwertende Horizonte. ``None`` wählt automatisch ``1``, ein Viertel, die
        Hälfte und den vollen Horizont des Datensatzes.
    physical_units
        Metriken nach Rückrechnung in Ingenieurseinheiten berechnen. Für den NRMSE
        rechnerisch nahezu gleichwertig, aber die Rückrechnung deckt Fehler in der
        Skalierer-Weitergabe zwischen den Splits auf -- die häufigste stille
        Leakage-Variante nach dem Split selbst.
    batch_size
        Stapelgröße der Auswertung.
    """

    horizons: tuple[int, ...] | None = None
    physical_units: bool = True
    batch_size: int = 4096

    def _horizons_for(self, dataset: SequenceDataset) -> tuple[int, ...]:
        if self.horizons is not None:
            invalid = [h for h in self.horizons if not 1 <= h <= dataset.config.horizon]
            if invalid:
                raise ValueError(
                    f"Horizonte {invalid} liegen außerhalb 1..{dataset.config.horizon}"
                )
            return self.horizons
        full = dataset.config.horizon
        return tuple(sorted({1, max(1, full // 4), max(1, full // 2), full}))

    def evaluate(self, model: SequenceModel, dataset: SequenceDataset) -> EvaluationResult:
        """Rollout auswerten."""
        horizons = self._horizons_for(dataset)
        scaler = dataset.target_scaler
        collected: list[tuple[FloatArray, FloatArray, FloatArray | None, FloatArray]] = []
        elapsed = 0.0
        n_windows = 0

        for batch in self._batches(dataset):
            start = time.perf_counter()
            prediction = model.predict(batch)
            elapsed += time.perf_counter() - start
            n_windows += batch.n_windows
            collected.append(
                (
                    prediction,
                    batch.future_targets,
                    batch.future_targets_truth,
                    batch.run_index,
                )
            )

        prediction_raw = np.concatenate([item[0] for item in collected], axis=0)
        reference_raw = np.concatenate([item[1] for item in collected], axis=0)
        truth_raw = (
            np.concatenate([item[2] for item in collected], axis=0)
            if all(item[2] is not None for item in collected)
            else None
        )
        runs = np.concatenate([item[3] for item in collected], axis=0)

        # Fuer rmse_run_median_physical wird IMMER in physikalische Einheiten
        # zurueckgerechnet, unabhaengig vom physical_units-Schalter unten -- ein
        # unnormierter Fehler in standardisierten Einheiten waere nicht interpretierbar.
        prediction_physical = scaler.inverse(prediction_raw)
        reference_physical = scaler.inverse(reference_raw)

        if self.physical_units:
            prediction = prediction_physical
            reference = reference_physical
            truth = None if truth_raw is None else scaler.inverse(truth_raw)
        else:
            prediction = prediction_raw
            reference = reference_raw
            truth = truth_raw

        names = dataset.target_names
        by_horizon = pooled_nrmse(prediction, reference, horizons, names)
        truth_by_horizon = {} if truth is None else pooled_nrmse(prediction, truth, horizons, names)
        spread = per_run_nrmse(prediction, reference, runs, horizons, names)
        truth_spread = (
            {} if truth is None else per_run_nrmse(prediction, truth, runs, horizons, names)
        )
        rmse_median_physical = per_run_rmse_median(
            prediction_physical, reference_physical, runs, horizons, names
        )

        return EvaluationResult(
            model_name=model.name,
            target_names=names,
            horizons=horizons,
            nrmse_by_horizon=by_horizon,
            nrmse_truth_by_horizon=truth_by_horizon,
            nrmse_run_spread=spread,
            inference_us_per_window=1e6 * elapsed / max(n_windows, 1),
            n_windows=n_windows,
            n_runs=int(np.unique(runs).size),
            nrmse_truth_run_spread=truth_spread,
            rmse_run_median_physical=rmse_median_physical,
        )

    def _batches(self, dataset: SequenceDataset) -> list[WindowBatch]:
        return list(dataset.iter_batches(self.batch_size))

    def describe(self) -> dict[str, Any]:
        """Protokollparameter für das Tracking."""
        return {
            "evaluator": type(self).__name__,
            "horizons": self.horizons,
            "physical_units": self.physical_units,
        }
