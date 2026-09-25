r"""Pflicht-Baselines (Arbeitsplan Schritt 4). Kein PyTorch, keine Trainingsschleife.

Warum drei und nicht eine
-------------------------
Die drei Baselines schließen sich gegenseitig in ihren Schwächen ein, und genau das macht
sie zusammen aussagekräftig:

* :class:`PersistenceModel` nutzt **nur** die letzte Beobachtung und **keinen** bekannten
  zukünftigen Eingang. Bei feiner Abtastung ist sie deshalb fast unschlagbar auf
  Ein-Schritt-Ebene und verliert jeden Stellgrößensprung -- für ``energy_target`` die
  falsche Referenz.
* :class:`StaticRegressionModel` nutzt **nur** die bekannten Eingänge zum jeweiligen
  Zeitpunkt und **keine** Historie. Sie ist horizontunabhängig: ihr Fehler bei
  :math:`h = 24` ist derselbe wie bei :math:`h = 1`. Für eine Zielgröße mit
  :math:`R^2_{\text{statisch}} = 0.987` ist sie die eigentliche Hürde.
* :class:`LinearARXModel` nutzt beides und ist damit die erste Baseline, die überhaupt
  Dynamik abbildet. Sie ist die Referenz, die ein Sequenzmodell schlagen muss, um
  Nichtlinearität zu rechtfertigen -- nicht Persistenz.

Ein Sequenzmodell, das nur Persistenz schlägt, hat nichts gezeigt. Ein Sequenzmodell, das
die statische Regression auf ``energy_target`` nicht schlägt, zeigt korrekt an, dass dort
nichts Dynamisches zu lernen ist.

Alle drei arbeiten im standardisierten Raum und implementieren dasselbe
:class:`~twinlink_ml.models.base.SequenceModel`-Interface wie später GRU, xLSTM und Mamba.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..data.windows import SequenceDataset, WindowBatch
from ..simulation.backend import FloatArray
from .base import SequenceModel

__all__ = [
    "ClimatologyModel",
    "LinearARXModel",
    "PersistenceModel",
    "StaticRegressionModel",
]

_MAX_FIT_ROWS = 200_000
"""Obergrenze der Zeilen in der Entwurfsmatrix. Darüber wird deterministisch gedünnt."""


def _ridge_solve(design: FloatArray, target: FloatArray, ridge: float) -> FloatArray:
    r"""Least-Squares mit Tikhonov-Regularisierung.

    .. math:: \boldsymbol\theta = (X^\top X + \lambda I)^{-1} X^\top y

    Die Regularisierung ist nicht kosmetisch: bei kleinen Stützstellen der Lernkurve
    (8 Läufe) und Konstantläufen im Design ist :math:`X^\top X` schlecht konditioniert,
    und eine unregularisierte Lösung schwankt zwischen den Seeds stärker als das
    eigentliche Modellsignal. Der Achsenabschnitt wird **nicht** bestraft.
    """
    n_features = design.shape[1]
    penalty = np.eye(n_features) * ridge
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    solution = np.linalg.solve(gram, design.T @ target)
    return np.asarray(solution, dtype=np.float64)


def _thin(n_rows: int, limit: int = _MAX_FIT_ROWS) -> slice:
    """Deterministische Ausdünnung -- gleichmäßig, nicht die ersten ``limit`` Zeilen."""
    if n_rows <= limit:
        return slice(None)
    return slice(None, None, int(np.ceil(n_rows / limit)))


def _max_lag(n_lags_target: int, n_lags_exogenous: int) -> int:
    """Mindestens nötige Fensterhistorie einer Verzögerungsordnung.

    Geteilt zwischen :class:`LinearARXModel` und
    :class:`~twinlink_ml.models.narx.RandomFeatureNARX`, die beide dieselbe
    Verzögerungskonvention verwenden (siehe ``narx.py``-Moduldocstring).
    """
    return max(n_lags_target, n_lags_exogenous)


def _initial_autoregressive_state(
    history_inputs: FloatArray,
    *,
    n_lags_target: int,
    n_lags_exogenous: int,
    n_exogenous: int,
    autoregressive_index: Sequence[int],
) -> tuple[FloatArray, FloatArray]:
    """Anfangszustand des Rollouts aus der Fensterhistorie schneiden.

    ``(vergangene Zielgrößen, vergangene Eingänge)``, je die letzten ``n_lags_*``
    Schritte. Geteilt zwischen :class:`LinearARXModel` und
    :class:`~twinlink_ml.models.narx.RandomFeatureNARX` -- beide Modelle mit derselben
    Verzögerungsordnung sehen exakt dieselbe Vorgeschichte.
    """
    targets_index = list(autoregressive_index)
    past_targets = history_inputs[:, -n_lags_target:, targets_index].copy()
    exogenous_history = history_inputs[..., :n_exogenous]
    past_exogenous = (
        exogenous_history[:, exogenous_history.shape[1] - n_lags_exogenous :, :].copy()
        if n_lags_exogenous
        else exogenous_history[:, :0, :]
    )
    return past_targets, past_exogenous


def _autoregressive_rollout(
    batch: WindowBatch,
    *,
    n_lags_target: int,
    n_lags_exogenous: int,
    n_exogenous: int,
    autoregressive_index: Sequence[int],
    predict_step: Callable[[FloatArray, FloatArray, FloatArray], FloatArray],
) -> FloatArray:
    r"""Autoregressiver Rollout über den Horizont, mit Verzögerungsfortschreibung.

    Genau die eine Stelle, an der sich :class:`LinearARXModel` und
    :class:`~twinlink_ml.models.narx.RandomFeatureNARX` unterscheiden dürfen, ist
    ``predict_step`` -- die Abbildung von Verzögerungen auf eine Ein-Schritt-Vorhersage
    (linear bzw. über Zufallsmerkmale). Historien-Slicing und die Fortschreibung von
    ``past_targets``/``past_exogenous`` sind für jedes Modell mit dieser
    Verzögerungskonvention identisch und stehen deshalb hier und nur hier -- das
    garantiert strukturell, was ``narx.py`` bisher nur im Docstring behauptete
    ("Struktur, Verzögerungen und Rollout sind identisch zum linearen ARX").

    Parameters
    ----------
    predict_step
        ``(past_targets, past_exogenous, current) -> Vorhersage (batch, n_y)``.
    """
    past_targets, past_exogenous = _initial_autoregressive_state(
        batch.history_inputs,
        n_lags_target=n_lags_target,
        n_lags_exogenous=n_lags_exogenous,
        n_exogenous=n_exogenous,
        autoregressive_index=autoregressive_index,
    )
    horizon = batch.future_exogenous.shape[1]
    predictions = np.empty((batch.n_windows, horizon, past_targets.shape[-1]), dtype=np.float64)
    for step in range(horizon):
        current = batch.future_exogenous[:, step, :]
        prediction = predict_step(past_targets, past_exogenous, current)
        predictions[:, step, :] = prediction
        past_targets = np.concatenate([past_targets[:, 1:, :], prediction[:, None, :]], axis=1)
        if n_lags_exogenous:
            past_exogenous = np.concatenate([past_exogenous[:, 1:, :], current[:, None, :]], axis=1)
    return predictions


class PersistenceModel(SequenceModel):
    r"""Naive Referenz: :math:`\hat y(t+h) = y(t)` für alle :math:`h`.

    Kein Parameter, kein Fit. Bemerkenswert bei :math:`\Delta t = 5` s: der
    Persistenz-NRMSE liegt dort bei 1.6 %, weshalb Architekturunterschiede in der
    Seed-Varianz verschwinden. Die Abtastzeit ist deshalb ein Versuchsparameter erster
    Ordnung, kein Implementierungsdetail.
    """

    def __init__(self) -> None:
        self._autoregressive_index: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        """Kurzname."""
        return "persistence"

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        """Merkt sich nur, welche Eingangskanäle die Zielgrößen sind.

        Bricht ab, wenn eine Zielgröße nicht inline gemessen wird. Der Grund ist keine
        Bequemlichkeit: ohne Messung enthält der Historienkanal den Trainingsmittelwert,
        und "die letzte Beobachtung halten" wäre dann die Mittelwertvorhersage unter
        falschem Namen. Sie hätte einen anderen Fehler, würde aber weiter als
        Auflösungsreferenz gelesen -- und die Schwelle wäre still falsch.
        """
        del seed, validation
        if not train.is_fully_observed:
            raise ValueError(
                "Persistenz ist ohne Inline-Messung undefiniert (nicht beobachtet: "
                f"{list(train.unobserved_target_names)}). Als parameterfreie Referenz "
                "ClimatologyModel verwenden."
            )
        self._autoregressive_index = train.autoregressive_index

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Letzte Beobachtung über den ganzen Horizont halten."""
        last = batch.history_inputs[:, -1, list(self._autoregressive_index)]
        return np.repeat(last[:, None, :], batch.future_exogenous.shape[1], axis=1)


class ClimatologyModel(SequenceModel):
    r"""Mittelwertvorhersage: :math:`\hat y(t+h) = \bar y_{\text{train}}`.

    Die parameterfreie Referenz für den **Soft-Sensor-Fall**, in dem Persistenz keine
    Bedeutung hat. Sie schätzt nur eine Lage und keine Beziehung; ihre Streuung über die
    Seeds ist damit -- wie bei der Persistenz -- reine Datensatz-Realisierung und taugt
    als Auflösungsschwelle (siehe :mod:`twinlink_ml.evaluation.resolution`).

    Warum sie auch im vollbeobachteten Fall mitlaufen sollte: der laufweise normierte
    NRMSE ist gegen die Streuung **innerhalb** eines Laufs normiert. Eine
    Mittelwertvorhersage liegt deshalb nicht bei 1.0, sondern bei
    :math:`\sqrt{1 + (\bar y_{\text{run}} - \bar y)^2/\sigma_{\text{run}}^2}` --
    also um so höher, je stärker die Läufe im Niveau streuen. Der Abstand dieser Zahl zu
    1.0 ist damit ein direktes Maß dafür, wie viel des Fehlers eines Modells reine
    Niveauzuordnung ist. Bei einem Design mit gestreuten Anfangszuständen ist das der
    Unterschied zwischen "das Modell versteht die Dynamik nicht" und "es findet das
    Niveau nicht".
    """

    def __init__(self) -> None:
        self._mean: FloatArray | None = None

    @property
    def name(self) -> str:
        """Kurzname."""
        return "climatology"

    @property
    def n_parameters(self) -> int:
        """Eine Lage je Zielgröße."""
        return 0 if self._mean is None else int(self._mean.size)

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        """Mittelwert der Zielgrößen über die Trainingsfenster."""
        del seed, validation
        batch = train.full_batch()
        self._mean = np.asarray(
            batch.future_targets.reshape(-1, batch.future_targets.shape[-1]).mean(axis=0),
            dtype=np.float64,
        )

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Konstante Vorhersage über den Horizont."""
        if self._mean is None:
            raise RuntimeError("Modell ist nicht angepasst")
        horizon = batch.future_exogenous.shape[1]
        return np.tile(self._mean[None, None, :], (batch.n_windows, horizon, 1))


class StaticRegressionModel(SequenceModel):
    r"""Regression der Zielgrößen auf die **aktuellen** Eingänge, ohne jede Historie.

    .. math:: \hat y(t+h) = \boldsymbol\theta^\top \phi\bigl(\mathbf u(t+h),
              \mathbf z(t+h)\bigr)

    Pflicht-Baseline nach Befund 2 aus ``04_stand_ap1.md``: ``energy_target`` ist zu
    98.7 % statisch erklärbar. Persistenz allein lässt hier jede Architektur gut
    aussehen, weil sie jeden Stellgrößensprung verliert.

    Die quadratischen Terme sind wichtig, damit **Nichtlinearität nicht mit Dynamik
    verwechselt** wird: eine rein lineare statische Baseline lässt ein Sequenzmodell
    besser erscheinen, obwohl der Gewinn aus der Nichtlinearität und nicht aus der
    Zeitstruktur kommt. Identisch zur Konvention in
    :func:`~twinlink_ml.simulation.diagnostics.static_explained_variance`.

    Parameters
    ----------
    quadratic
        Quadratische Terme mitführen.
    ridge
        Regularisierungsparameter.
    """

    def __init__(self, *, quadratic: bool = True, ridge: float = 1e-6) -> None:
        self._quadratic = quadratic
        self._ridge = ridge
        self._coefficients: FloatArray | None = None

    @property
    def name(self) -> str:
        """Kurzname."""
        return "static_quadratic" if self._quadratic else "static_linear"

    @property
    def n_parameters(self) -> int:
        """Anzahl Koeffizienten."""
        return 0 if self._coefficients is None else int(self._coefficients.size)

    def _features(self, exogenous: FloatArray) -> FloatArray:
        flat = exogenous.reshape(-1, exogenous.shape[-1])
        parts = [flat, flat**2] if self._quadratic else [flat]
        parts.append(np.ones((flat.shape[0], 1)))
        return np.concatenate(parts, axis=-1)

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        """Koeffizienten aus allen Horizontschritten aller Trainingsfenster schätzen."""
        del seed, validation
        batch = train.full_batch()
        design = self._features(batch.future_exogenous)
        target = batch.future_targets.reshape(-1, batch.future_targets.shape[-1])
        keep = _thin(design.shape[0])
        self._coefficients = _ridge_solve(design[keep], target[keep], self._ridge)

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Für jeden Horizontschritt getrennt auswerten -- horizontunabhängig."""
        if self._coefficients is None:
            raise RuntimeError("Modell ist nicht angepasst")
        n_windows, horizon, _ = batch.future_exogenous.shape
        prediction = self._features(batch.future_exogenous) @ self._coefficients
        return prediction.reshape(n_windows, horizon, -1)

    def describe(self) -> dict[str, Any]:
        """Konfiguration für das Tracking."""
        return {**super().describe(), "quadratic": self._quadratic, "ridge": self._ridge}


class LinearARXModel(SequenceModel):
    r"""Linearer ARX: autoregressiv in den Zielgrößen, mit bekannten Eingängen.

    .. math:: \hat{\mathbf y}(t+1) = \sum_{i=1}^{p} A_i\, \mathbf y(t+1-i)
              + \sum_{j=0}^{q} B_j\, \mathbf x(t+1-j) + \mathbf c

    Wichtig ist der Term :math:`j = 0`: der Eingang zum Vorhersagezeitpunkt ist bekannt
    (der Regler gibt die Stellgrößen vor, die Störgrößen sind messbar). Ein ARX ohne
    diesen Term wäre künstlich schwach und würde die Hürde für die Sequenzmodelle senken.

    Angepasst wird ausschließlich auf **Ein-Schritt**-Vorhersage, ausgewertet über den
    ganzen Horizont durch autoregressives Fortschreiben. Das ist der ehrliche Weg: das
    Modell sieht im Rollout seine eigenen Fehler, nicht die Wahrheit. Wer stattdessen pro
    Horizont ein eigenes Direktmodell anpasst, bekommt kleinere Zahlen und ein Verfahren,
    das an der Anlage nicht existiert.

    Parameters
    ----------
    n_lags_target
        :math:`p`, Anzahl Zielgrößen-Verzögerungen.
    n_lags_exogenous
        :math:`q`, zusätzliche Eingangsverzögerungen über :math:`j = 0` hinaus.
    ridge
        Regularisierungsparameter.
    """

    def __init__(
        self, *, n_lags_target: int = 4, n_lags_exogenous: int = 2, ridge: float = 1e-4
    ) -> None:
        if n_lags_target < 1:
            raise ValueError("n_lags_target muss >= 1 sein")
        if n_lags_exogenous < 0:
            raise ValueError("n_lags_exogenous darf nicht negativ sein")
        self._n_lags_target = n_lags_target
        self._n_lags_exogenous = n_lags_exogenous
        self._ridge = ridge
        self._coefficients: FloatArray | None = None
        self._autoregressive_index: tuple[int, ...] = ()
        self._n_exogenous = 0

    @property
    def name(self) -> str:
        """Kurzname mit Ordnung."""
        return f"arx_p{self._n_lags_target}_q{self._n_lags_exogenous}"

    @property
    def n_parameters(self) -> int:
        """Anzahl Koeffizienten."""
        return 0 if self._coefficients is None else int(self._coefficients.size)

    @property
    def required_history(self) -> int:
        """Mindestens nötige Fensterhistorie."""
        return _max_lag(self._n_lags_target, self._n_lags_exogenous)

    def _design_row(
        self, past_targets: FloatArray, past_exogenous: FloatArray, current: FloatArray
    ) -> FloatArray:
        """Entwurfszeile aus Verzögerungen bauen.

        Parameters
        ----------
        past_targets
            ``(batch, p, n_y)``, zeitlich aufsteigend; letzte Spalte ist :math:`y(t)`.
        past_exogenous
            ``(batch, q, n_x)``, zeitlich aufsteigend, bis :math:`x(t)`.
        current
            ``(batch, n_x)``, der bekannte Eingang :math:`x(t+1)`.
        """
        n_windows = past_targets.shape[0]
        parts = [
            past_targets.reshape(n_windows, -1),
            current,
            past_exogenous.reshape(n_windows, -1),
            np.ones((n_windows, 1)),
        ]
        return np.concatenate(parts, axis=-1)

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        """Ein-Schritt-Regression über alle Trainingsfenster."""
        del seed, validation
        self._autoregressive_index = train.autoregressive_index
        self._n_exogenous = len(train.exogenous_names)
        if train.config.history_length < self.required_history:
            raise ValueError(
                f"history_length={train.config.history_length} ist kleiner als die "
                f"ARX-Ordnung {self.required_history}"
            )
        batch = train.full_batch()
        targets_index = list(self._autoregressive_index)
        history_targets = batch.history_inputs[..., targets_index]
        history_exogenous = batch.history_inputs[..., : self._n_exogenous]

        design = self._design_row(
            history_targets[:, -self._n_lags_target :, :],
            history_exogenous[:, history_exogenous.shape[1] - self._n_lags_exogenous :, :]
            if self._n_lags_exogenous
            else history_exogenous[:, :0, :],
            batch.future_exogenous[:, 0, :],
        )
        target = batch.future_targets[:, 0, :]
        keep = _thin(design.shape[0])
        self._coefficients = _ridge_solve(design[keep], target[keep], self._ridge)

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Autoregressiver Rollout über den Horizont."""
        if self._coefficients is None:
            raise RuntimeError("Modell ist nicht angepasst")
        # Lokal binden statt self._coefficients im Closure lesen: mypy narrowt die
        # Optionalität nicht über eine verschachtelte Funktion hinweg (Instanzattribute
        # könnten sich theoretisch zwischen Prüfung und Aufruf ändern).
        coefficients = self._coefficients
        return _autoregressive_rollout(
            batch,
            n_lags_target=self._n_lags_target,
            n_lags_exogenous=self._n_lags_exogenous,
            n_exogenous=self._n_exogenous,
            autoregressive_index=self._autoregressive_index,
            predict_step=lambda past_targets, past_exogenous, current: (
                self._design_row(past_targets, past_exogenous, current) @ coefficients
            ),
        )

    def describe(self) -> dict[str, Any]:
        """Konfiguration für das Tracking."""
        return {
            **super().describe(),
            "n_lags_target": self._n_lags_target,
            "n_lags_exogenous": self._n_lags_exogenous,
            "ridge": self._ridge,
        }
