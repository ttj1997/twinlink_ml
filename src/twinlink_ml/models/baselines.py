r"""Mandatory baselines (work plan step 4). No PyTorch, no training loop.

Why three and not one
---------------------
The three baselines enclose each other in their weaknesses, and exactly that makes
them meaningful together:

* :class:`PersistenceModel` uses **only** the last observation and **no** known
  future input. With fine sampling it is therefore almost unbeatable at the
  one-step level and misses every action step -- for ``energy_target`` the
  wrong reference.
* :class:`StaticRegressionModel` uses **only** the known inputs at the respective
  time and **no** history. It is horizon-independent: its error at
  :math:`h = 24` is the same as at :math:`h = 1`. For a target with
  :math:`R^2_{\text{statisch}} = 0.987` it is the actual hurdle.
* :class:`LinearARXModel` uses both and is thus the first baseline that represents
  dynamics at all. It is the reference a sequence model must beat in order to
  justify nonlinearity -- not persistence.

A sequence model that only beats persistence has shown nothing. A sequence model that
does not beat the static regression on ``energy_target`` correctly indicates that there
is nothing dynamic to learn there.

All three work in the standardised space and implement the same
:class:`~twinlink_ml.models.base.SequenceModel` interface as GRU, xLSTM and Mamba later.
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
    r"""Least squares with Tikhonov regularisation.

    .. math:: \boldsymbol\theta = (X^\top X + \lambda I)^{-1} X^\top y

    The regularisation is not cosmetic: at small support points of the learning curve
    (8 runs) and with constant runs in the design, :math:`X^\top X` is ill-conditioned,
    and an unregularised solution fluctuates between the seeds more than the
    actual model signal. The intercept is **not** penalised.
    """
    n_features = design.shape[1]
    penalty = np.eye(n_features) * ridge
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    solution = np.linalg.solve(gram, design.T @ target)
    return np.asarray(solution, dtype=np.float64)


def _thin(n_rows: int, limit: int = _MAX_FIT_ROWS) -> slice:
    """Deterministic thinning -- uniform, not the first ``limit`` rows."""
    if n_rows <= limit:
        return slice(None)
    return slice(None, None, int(np.ceil(n_rows / limit)))


def _max_lag(n_lags_target: int, n_lags_exogenous: int) -> int:
    """Minimum window history required by a lag order.

    Shared between :class:`LinearARXModel` and
    :class:`~twinlink_ml.models.narx.RandomFeatureNARX`, which both use the same
    lag convention (see the ``narx.py`` module docstring).
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
    """Cut the initial state of the rollout from the window history.

    ``(vergangene Zielgrößen, vergangene Eingänge)``, each the last ``n_lags_*``
    steps. Shared between :class:`LinearARXModel` and
    :class:`~twinlink_ml.models.narx.RandomFeatureNARX` -- both models with the same
    lag order see exactly the same past.
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
    r"""Autoregressive rollout over the horizon, with lag update.

    The one and only place where :class:`LinearARXModel` and
    :class:`~twinlink_ml.models.narx.RandomFeatureNARX` may differ is
    ``predict_step`` -- the mapping from lags to a one-step prediction
    (linear or via random features, respectively). History slicing and the update of
    ``past_targets``/``past_exogenous`` are identical for every model with this
    lag convention and therefore live here and only here -- this
    guarantees structurally what ``narx.py`` previously only claimed in its docstring
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
    r"""Naive reference: :math:`\hat y(t+h) = y(t)` for all :math:`h`.

    No parameter, no fit. Notable at :math:`\Delta t = 5` s: the
    persistence NRMSE there is 1.6 %, which is why architecture differences vanish in the
    seed variance. The sampling time is therefore a first-order experimental parameter,
    not an implementation detail.
    """

    def __init__(self) -> None:
        self._autoregressive_index: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        """Short name."""
        return "persistence"

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        """Only remembers which input channels are the targets.

        Aborts if a target is not measured inline. The reason is not
        convenience: without a measurement the history channel contains the training mean,
        and "holding the last observation" would then be the mean prediction under a
        false name. It would have a different error, but would still be read as the
        resolution reference -- and the threshold would be silently wrong.
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
        """Hold the last observation over the whole horizon."""
        last = batch.history_inputs[:, -1, list(self._autoregressive_index)]
        return np.repeat(last[:, None, :], batch.future_exogenous.shape[1], axis=1)


class ClimatologyModel(SequenceModel):
    r"""Mean prediction: :math:`\hat y(t+h) = \bar y_{\text{train}}`.

    The parameter-free reference for the **soft sensor case**, in which persistence has no
    meaning. It estimates only a location and no relation; its spread over the
    seeds is thus -- as with persistence -- pure dataset realisation and serves
    as a resolution threshold (see :mod:`twinlink_ml.evaluation.resolution`).

    Why it should also run along in the fully observed case: the run-level normalised
    NRMSE is normalised against the spread **within** a run. A
    mean prediction therefore does not lie at 1.0, but at
    :math:`\sqrt{1 + (\bar y_{\text{run}} - \bar y)^2/\sigma_{\text{run}}^2}` --
    i.e. the higher, the more the runs spread in level. The distance of this number from
    1.0 is thus a direct measure of how much of a model's error is pure
    level assignment. In a design with scattered initial states this is the
    difference between "the model does not understand the dynamics" and "it does not find
    the level".
    """

    def __init__(self) -> None:
        self._mean: FloatArray | None = None

    @property
    def name(self) -> str:
        """Short name."""
        return "climatology"

    @property
    def n_parameters(self) -> int:
        """One location per target."""
        return 0 if self._mean is None else int(self._mean.size)

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        """Mean of the targets over the training windows."""
        del seed, validation
        batch = train.full_batch()
        self._mean = np.asarray(
            batch.future_targets.reshape(-1, batch.future_targets.shape[-1]).mean(axis=0),
            dtype=np.float64,
        )

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Constant prediction over the horizon."""
        if self._mean is None:
            raise RuntimeError("Modell ist nicht angepasst")
        horizon = batch.future_exogenous.shape[1]
        return np.tile(self._mean[None, None, :], (batch.n_windows, horizon, 1))


class StaticRegressionModel(SequenceModel):
    r"""Regression of the targets on the **current** inputs, without any history.

    .. math:: \hat y(t+h) = \boldsymbol\theta^\top \phi\bigl(\mathbf u(t+h),
              \mathbf z(t+h)\bigr)

    Mandatory baseline after Befund 2 in ``04_stand_ap1.md``: ``energy_target`` is
    98.7 % explainable statically. Persistence alone makes every architecture look good
    here, because it misses every action step.

    The quadratic terms are important so that **nonlinearity is not confused with
    dynamics**: a purely linear static baseline makes a sequence model appear
    better, although the gain comes from the nonlinearity and not from the
    temporal structure. Identical to the convention in
    :func:`~twinlink_ml.simulation.diagnostics.static_explained_variance`.

    Parameters
    ----------
    quadratic
        Include quadratic terms.
    ridge
        Regularisation parameter.
    """

    def __init__(self, *, quadratic: bool = True, ridge: float = 1e-6) -> None:
        self._quadratic = quadratic
        self._ridge = ridge
        self._coefficients: FloatArray | None = None

    @property
    def name(self) -> str:
        """Short name."""
        return "static_quadratic" if self._quadratic else "static_linear"

    @property
    def n_parameters(self) -> int:
        """Number of coefficients."""
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
        """Estimate coefficients from all horizon steps of all training windows."""
        del seed, validation
        batch = train.full_batch()
        design = self._features(batch.future_exogenous)
        target = batch.future_targets.reshape(-1, batch.future_targets.shape[-1])
        keep = _thin(design.shape[0])
        self._coefficients = _ridge_solve(design[keep], target[keep], self._ridge)

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Evaluate separately for each horizon step -- horizon-independent."""
        if self._coefficients is None:
            raise RuntimeError("Modell ist nicht angepasst")
        n_windows, horizon, _ = batch.future_exogenous.shape
        prediction = self._features(batch.future_exogenous) @ self._coefficients
        return prediction.reshape(n_windows, horizon, -1)

    def describe(self) -> dict[str, Any]:
        """Configuration for tracking."""
        return {**super().describe(), "quadratic": self._quadratic, "ridge": self._ridge}


class LinearARXModel(SequenceModel):
    r"""Linear ARX: autoregressive in the targets, with known inputs.

    .. math:: \hat{\mathbf y}(t+1) = \sum_{i=1}^{p} A_i\, \mathbf y(t+1-i)
              + \sum_{j=0}^{q} B_j\, \mathbf x(t+1-j) + \mathbf c

    The term :math:`j = 0` is important: the input at the prediction time is known
    (the controller sets the actions, the disturbances are measurable). An ARX without
    this term would be artificially weak and would lower the hurdle for the sequence models.

    Fitting is done exclusively on **one-step** prediction, evaluated over the
    whole horizon by autoregressive continuation. This is the honest way: the
    model sees its own errors in the rollout, not the truth. Whoever instead fits a
    separate direct model per horizon gets smaller numbers and a procedure
    that does not exist at the plant.

    Parameters
    ----------
    n_lags_target
        :math:`p`, number of target lags.
    n_lags_exogenous
        :math:`q`, additional input lags beyond :math:`j = 0`.
    ridge
        Regularisation parameter.
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
        """Short name with order."""
        return f"arx_p{self._n_lags_target}_q{self._n_lags_exogenous}"

    @property
    def n_parameters(self) -> int:
        """Number of coefficients."""
        return 0 if self._coefficients is None else int(self._coefficients.size)

    @property
    def required_history(self) -> int:
        """Minimum window history required."""
        return _max_lag(self._n_lags_target, self._n_lags_exogenous)

    def _design_row(
        self, past_targets: FloatArray, past_exogenous: FloatArray, current: FloatArray
    ) -> FloatArray:
        """Build a design row from lags.

        Parameters
        ----------
        past_targets
            ``(batch, p, n_y)``, ascending in time; last column is :math:`y(t)`.
        past_exogenous
            ``(batch, q, n_x)``, ascending in time, up to :math:`x(t)`.
        current
            ``(batch, n_x)``, the known input :math:`x(t+1)`.
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
        """One-step regression over all training windows."""
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
        """Autoregressive rollout over the horizon."""
        if self._coefficients is None:
            raise RuntimeError("Modell ist nicht angepasst")
        # Bind locally instead of reading self._coefficients in the closure: mypy does not
        # narrow the optionality across a nested function (instance attributes could in
        # theory change between the check and the call).
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
        """Configuration for tracking."""
        return {
            **super().describe(),
            "n_lags_target": self._n_lags_target,
            "n_lags_exogenous": self._n_lags_exogenous,
            "ridge": self._ridge,
        }
