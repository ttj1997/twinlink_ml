r"""Nichtlineares NARX mit Zufallsmerkmalen (Arbeitsplan Schritt 6, Vorstufe).

Warum dieses Modell zwischen linearem ARX und GRU gehört
--------------------------------------------------------
Nach Schritt 4 steht auf der Feuchte fest: die beste Baseline ist die Persistenz, und der
lineare ARX-Rollout ist schlechter als sie. Der nächste Schritt wäre normalerweise ein
GRU. Damit vergleicht man aber **drei Änderungen gleichzeitig** -- Nichtlinearität,
gelernter Zustand und gradientenbasiertes Training -- und weiß bei einem Gewinn nicht,
welche davon ihn erzeugt hat. Dieses Modell ändert genau eine:

.. math::
   \hat y(t+1) = y(t) + W^\top \phi\bigl(\mathbf h(t)\bigr), \qquad
   \phi_j(\mathbf h) = \cos\bigl(\boldsymbol\omega_j^\top \mathbf h + b_j\bigr)

Struktur, Verzögerungen und Rollout sind identisch zum linearen ARX; nur die
Regressionsbasis ist eine nichtlineare Zufallsabbildung (Random Fourier Features nach
Rahimi/Recht, in der Praxis identisch zu einer Extreme Learning Machine mit cos-Aktivierung).
:math:`\boldsymbol\omega` und :math:`b` werden **gezogen und nicht trainiert**, die
Ausgabeschicht ist eine Ridge-Regression in geschlossener Form. Damit gilt:

* **Ein Sequenzmodell muss dieses Modell schlagen, nicht den linearen ARX.** Sonst zeigt es
  nur, dass die Aufgabe nichtlinear ist -- was seit Schritt 1 bekannt ist.
* Es ist das **erste Modell mit einem echten Modell-Seed**. Bei Persistenz, statischer
  Regression und ARX ist :math:`\sigma_{\text{model}}` exakt Null; die Varianzzerlegung
  aus Schritt 5 war damit an keinem Fall geprüft, in dem sie etwas zu zeigen hat.
* Es kostet keine GPU und keine Trainingsschleife. Fehlt PyTorch, trägt es den
  Architekturvergleich allein.

Residual-Parametrisierung
-------------------------
Der Kopf sagt :math:`\Delta y` vorher, nicht :math:`y`. Bei Nullgewichten ist das Modell
exakt die Persistenz-Baseline -- dieselbe Vorgabe wie für alle Architekturen in Schritt 6,
und hier ohne Torch prüfbar (ein Test setzt die Ridge-Strafe gegen unendlich und vergleicht
gegen :class:`~twinlink_ml.models.baselines.PersistenceModel`).

Fallstrick, der hier sichtbar wird
----------------------------------
Im Soft-Sensor-Fall enthält der Historienkanal der nicht gemessenen Zielgröße konstant den
Trainingsmittelwert. Zufallsmerkmale einer konstanten Eingangsrichtung sind selbst konstant;
die Ridge-Regression schiebt ihre Gewichte gegen Null, und das Modell benutzt seine eigene
Rückführung nicht. **Das ist kein Fehler, sondern die korrekte Konsequenz von
Ein-Schritt-Anpassung**: das Modell hat im Training nie einen von Null verschiedenen Wert
in diesem Kanal gesehen. Ein Netz, das auf den Rollout-Verlust trainiert, sieht seine
eigenen Vorhersagen und kann diese Rückführung lernen. Genau darin liegt der strukturelle
Vorteil der Sequenzmodelle im Soft-Sensor-Fall -- und er ist an dieser Stelle als
Erwartung formuliert, bevor die erste Zahl vorliegt.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.windows import SequenceDataset, WindowBatch
from ..simulation.backend import FloatArray
from .base import SequenceModel
from .baselines import _MAX_FIT_ROWS, _autoregressive_rollout, _max_lag, _ridge_solve, _thin

__all__ = ["RandomFeatureNARX"]


class RandomFeatureNARX(SequenceModel):
    r"""NARX with drawn random features and a ridge output layer.

    Parameters
    ----------
    n_features
        Number of random features :math:`M`. The capacity parameter. Above the number of
        training windows the model becomes unstable even with ridge; with eight runs this is
        reachable, which is why the default is deliberately moderate.
    n_lags_target, n_lags_exogenous
        Lag order, identical to the convention of the linear ARX -- the comparison
        is meant to isolate the basis function and not change the order at the same time.
    length_scale
        Bandwidth :math:`\ell` of the random frequencies,
        :math:`\boldsymbol\omega \sim \mathcal N(0, \ell^{-2} I)`. In the standardised
        input space 1.0 is the neutral choice. Too small an :math:`\ell` produces
        high-frequency features that overfit immediately with few runs -- this is the
        actual hyperparameter of this model.
    ridge
        Regularisation of the output layer. Larger than for the linear ARX, because the
        design matrix is wider and worse conditioned.
    include_linear
        Include the standardised inputs in addition to the random features. This
        makes the linear ARX a special case (random weights pushed to zero), and the
        model cannot be *worse* than it if the regularisation fits.
    """

    def __init__(
        self,
        *,
        n_features: int = 256,
        n_lags_target: int = 4,
        n_lags_exogenous: int = 2,
        length_scale: float = 1.0,
        ridge: float = 1e-2,
        include_linear: bool = True,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features muss >= 1 sein")
        if n_lags_target < 1:
            raise ValueError("n_lags_target muss >= 1 sein")
        if n_lags_exogenous < 0:
            raise ValueError("n_lags_exogenous darf nicht negativ sein")
        if length_scale <= 0.0:
            raise ValueError("length_scale muss > 0 sein")
        self._n_features = n_features
        self._n_lags_target = n_lags_target
        self._n_lags_exogenous = n_lags_exogenous
        self._length_scale = length_scale
        self._ridge = ridge
        self._include_linear = include_linear
        self._frequencies: FloatArray | None = None
        self._offsets: FloatArray | None = None
        self._coefficients: FloatArray | None = None
        self._autoregressive_index: tuple[int, ...] = ()
        self._anchor_mask: FloatArray | None = None
        self._n_exogenous = 0

    # ------------------------------------------------------------------- Interface

    @property
    def name(self) -> str:
        """Short name with capacity."""
        return f"narx_rf{self._n_features}"

    @property
    def n_parameters(self) -> int:
        """Only the output layer is fitted; the features are drawn."""
        return 0 if self._coefficients is None else int(self._coefficients.size)

    @property
    def required_history(self) -> int:
        """Minimum window history required."""
        return _max_lag(self._n_lags_target, self._n_lags_exogenous)

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        """Draw features, solve the output layer in closed form.

        ``validation`` remains unused: there is no training history that could be
        stopped. This is the reason why this model still leaves the difference between the
        validation operating modes at exactly zero -- the substantive number
        for it remains tied to an iteratively trained model.
        """
        del validation
        self._autoregressive_index = train.autoregressive_index
        self._n_exogenous = len(train.exogenous_names)
        observed = set(train.observed_target_names)
        self._anchor_mask = np.asarray(
            [1.0 if name in observed else 0.0 for name in train.target_names],
            dtype=np.float64,
        )
        if train.config.history_length < self.required_history:
            raise ValueError(
                f"history_length={train.config.history_length} ist kleiner als die "
                f"NARX-Ordnung {self.required_history}"
            )

        batch = train.full_batch()
        raw = self._raw_inputs(batch.history_inputs, batch.future_exogenous[:, 0, :])
        generator = np.random.default_rng(seed)
        self._frequencies = generator.normal(
            0.0, 1.0 / self._length_scale, size=(raw.shape[-1], self._n_features)
        )
        self._offsets = generator.uniform(0.0, 2.0 * np.pi, size=self._n_features)

        design = self._design(raw)
        residual = batch.future_targets[:, 0, :] - self._anchor(batch.history_inputs)
        keep = _thin(design.shape[0], _MAX_FIT_ROWS)
        self._coefficients = _ridge_solve(design[keep], residual[keep], self._ridge)

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Autoregressive rollout, residual on the respective last state."""
        if self._coefficients is None:
            raise RuntimeError("Modell ist nicht angepasst")
        # Bind locally instead of reading self._coefficients in the closure: mypy does not
        # narrow the optionality across a nested function.
        coefficients = self._coefficients

        def predict_step(
            past_targets: FloatArray, past_exogenous: FloatArray, current: FloatArray
        ) -> FloatArray:
            raw = self._assemble(past_targets, past_exogenous, current)
            anchor = self._mask_anchor(past_targets[:, -1, :])
            return anchor + self._design(raw) @ coefficients

        return _autoregressive_rollout(
            batch,
            n_lags_target=self._n_lags_target,
            n_lags_exogenous=self._n_lags_exogenous,
            n_exogenous=self._n_exogenous,
            autoregressive_index=self._autoregressive_index,
            predict_step=predict_step,
        )

    def describe(self) -> dict[str, Any]:
        """Configuration for tracking."""
        return {
            **super().describe(),
            "n_features": self._n_features,
            "n_lags_target": self._n_lags_target,
            "n_lags_exogenous": self._n_lags_exogenous,
            "length_scale": self._length_scale,
            "ridge": self._ridge,
            "include_linear": self._include_linear,
        }

    # -------------------------------------------------------------------- internal

    def _anchor(self, history: FloatArray) -> FloatArray:
        """Reference point of the residual parametrisation: the last observation."""
        return self._mask_anchor(history[:, -1, list(self._autoregressive_index)])

    def _mask_anchor(self, values: FloatArray) -> FloatArray:
        r"""Restrict the anchor to the inline-measured channels.

        **The most important special case of this model.** The residual parametrisation
        :math:`\hat y(t+1) = y(t) + \Delta` presupposes that :math:`y(t)` is known.
        If the target is not measured inline, its place in the rollout is taken by the
        model's own prediction of the previous step -- and the prediction becomes an
        **integrator without feedback**: every estimation error of the increment adds up
        over the horizon, and over 24 steps the model drifts away. Measured on
        product moisture without inline measurement, this was the difference between NRMSE 3.8
        (mean prediction) and 40 to 60.

        For unobserved channels the anchor is therefore constantly zero, i.e. the
        standardised training mean: the prediction there is **direct** and not
        incremental. Thus the requirement of the work plan -- zero weights correspond to
        the best parameter-free baseline -- holds in both operating modes, only this
        baseline is persistence in one case and the mean prediction in the other.
        """
        if self._anchor_mask is None:
            return values
        return values * self._anchor_mask[None, :]

    def _raw_inputs(self, history: FloatArray, current: FloatArray) -> FloatArray:
        targets_index = list(self._autoregressive_index)
        past_targets = history[:, -self._n_lags_target :, targets_index]
        exogenous_history = history[..., : self._n_exogenous]
        past_exogenous = (
            exogenous_history[:, exogenous_history.shape[1] - self._n_lags_exogenous :, :]
            if self._n_lags_exogenous
            else exogenous_history[:, :0, :]
        )
        return self._assemble(past_targets, past_exogenous, current)

    @staticmethod
    def _assemble(
        past_targets: FloatArray, past_exogenous: FloatArray, current: FloatArray
    ) -> FloatArray:
        n_windows = past_targets.shape[0]
        return np.concatenate(
            [
                past_targets.reshape(n_windows, -1),
                current,
                past_exogenous.reshape(n_windows, -1),
            ],
            axis=-1,
        )

    def _design(self, raw: FloatArray) -> FloatArray:
        r"""Design matrix: random features, optionally linear, plus intercept.

        The factor :math:`\sqrt{2/M}` is the usual normalisation, so that the inner product
        of the features approximates the Gaussian kernel and the scale of the features does not
        grow with :math:`M` -- otherwise the ridge penalty would depend on the number of features
        and a capacity study would shift the regularisation at the same time.
        """
        assert self._frequencies is not None and self._offsets is not None
        projection = raw @ self._frequencies + self._offsets[None, :]
        features = np.sqrt(2.0 / self._n_features) * np.cos(projection)
        parts = [features]
        if self._include_linear:
            parts.append(raw)
        parts.append(np.ones((raw.shape[0], 1), dtype=np.float64))
        return np.concatenate(parts, axis=-1)
