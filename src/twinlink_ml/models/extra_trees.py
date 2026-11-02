r"""Extra Trees -- methodological reproduction of Kräuter, Zhou & Heinrich (ABS23).

**Not a candidate of the architecture comparison.** This model is deliberately not in
``models/registry.py``/``MODEL_SLOTS`` -- it serves exclusively to measure the
split comparison from ``docs/04_stand_ap1.md`` (section "Was das Paper des Vorgängers
ändert") on the same dataset: window-/point-level split (Kräuter's
methodology, see :mod:`twinlink_ml.data.point_split`) against run-level split (the standard
of this project). The gap between the two numbers is the result, not an
Extra Trees accuracy taken on its own. See ``scripts/run_extra_trees_split_comparison.py``.

What is adopted
---------------
* :class:`sklearn.ensemble.ExtraTreesRegressor`, **one** regressor for all targets
  (native multi-output support of tree regressors -- no
  ``MultiOutputRegressor`` wrapper needed).
* Features: current value as well as first and second difference over :math:`t, t{-}1, t{-}2`
  (numerical differences, not a Δt-normalised estimator) -- **no** long window as with
  GRU/LSTM/xLSTM/Transformer/Mamba. Channels: all quantities observable via the history,
  i.e. actions and disturbances (``exogenous_names``) **and** inline-measured targets
  (``observed_target_names``) -- analogous to the self-feedback in
  :class:`~twinlink_ml.models.baselines.LinearARXModel` and
  :class:`~twinlink_ml.models.narx.RandomFeatureNARX`. Unobserved target channels
  (soft sensor case) remain excluded: they sit constantly at the training mean in the
  history; a derivative of that would be a derivative of noise around a
  constant.
* One-step prediction (:math:`h = 1`), no multi-step rollout evaluation -- the
  window configuration for this comparison must set ``horizon=1``
  (:meth:`ExtraTreesModel.fit` checks this).

What is deliberately NOT adopted
--------------------------------
* **No PCA.** Kräuter's 12 principal components were the answer to 74 raw features; the
  feature space here (channels × 3) is small from the outset, a dimensionality reduction
  would have no counterpart here.
* **No centred Savitzky-Golay smoothing of the target** -- not causal, not meaningfully
  transferable to a simulation without its own sensor-noise pipeline (see
  :meth:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator.measure`, purely
  causal PT1 lag element plus additive noise).
* **No outlier removal** -- no sensor artefact in the simulation.

Hyperparameters
---------------
Kräuter's own ``ExtraTreesRegressor`` hyperparameters (number of trees, depth, ...) are not
recorded in ``02_prozess_setup.md``/``04_stand_ap1.md`` -- only the result
(R² = 0.994) and the preprocessing (PCA) are given there. The defaults here are
therefore **sklearn defaults** (``n_estimators=100``, ``max_depth=None``, ...): this
comparison is meant to isolate the effect of the split, not to optimise a tree configuration
after the fact.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.windows import SequenceDataset, WindowBatch
from ..simulation.backend import FloatArray
from .base import SequenceModel

__all__ = ["ExtraTreesModel"]


class ExtraTreesModel(SequenceModel):
    r"""Extra Trees on value + 1st/2nd difference, one-step prediction.

    Parameters
    ----------
    n_estimators
        Number of trees in the ensemble. sklearn default.
    max_depth
        Maximum tree depth. ``None`` (sklearn default) means unlimited, until
        ``min_samples_leaf`` takes effect.
    min_samples_leaf
        sklearn default.
    n_jobs
        Parallelism of ``fit``/``predict``. ``-1`` uses all cores -- unlike with
        the other models of this project, this costs no reproducibility here:
        Extra Trees has no sequential optimisation path that the threading order
        could influence.
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
        """Short name with number of trees."""
        return f"extra_trees{self._n_estimators}"

    @property
    def n_parameters(self) -> int:
        """Total number of nodes over all trees.

        Not a weight in the ANN sense, but the comparable capacity metric for a
        tree ensemble.
        """
        if self._regressor is None:
            return 0
        return int(sum(tree.tree_.node_count for tree in self._regressor.estimators_))

    def _resolve_feature_channels(self, dataset: SequenceDataset) -> tuple[int, ...]:
        """Channel indices in ``history_inputs`` whose value + differences become features.

        All exogenous channels plus the inline-measured targets -- see module docstring.
        """
        n_exogenous = len(dataset.exogenous_names)
        observed = set(dataset.observed_target_names)
        target_indices = tuple(
            n_exogenous + i for i, name in enumerate(dataset.target_names) if name in observed
        )
        return tuple(range(n_exogenous)) + target_indices

    def _features(self, history_inputs: FloatArray) -> FloatArray:
        """Value + first + second difference over the last three time steps.

        ``history_inputs[:, -1]`` is :math:`t`, ``[:, -2]`` is :math:`t{-}1`,
        ``[:, -3]`` is :math:`t{-}2` -- numerical differences, not a Δt-normalised
        derivative estimator, as with Kräuter.
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
        """One-step regression on value/difference features.

        ``validation`` remains unused: Extra Trees has no training history that
        could be stopped -- the same reason as for the analytical baselines.
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
        """Direct one-step prediction, no autoregressive rollout."""
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
        """Configuration for tracking."""
        return {
            **super().describe(),
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "min_samples_leaf": self._min_samples_leaf,
        }
