"""Abstraction :class:`SequenceModel` -- common interface of all prediction models.

The contract is deliberately narrow and **rollout-centred**: a model receives a
:class:`~twinlink_ml.data.windows.WindowBatch` and returns the targets over the
entire horizon. There is no separate one-step method, because the
one-step error is just the horizon slice :math:`h = 1`. An interface with two
methods would allow the one-step and rollout paths to drift apart -- the
classic route to a model that wins the secondary metric and diverges in the rollout.

Order in AP 1: this interface is implemented first by the three computation-free baselines
(persistence, static regression, linear ARX) and only afterwards by the GRU.
The other way round, the interface shapes itself around the neural network, and the
baselines only fit in with special cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..data.windows import SequenceDataset, WindowBatch
from ..simulation.backend import FloatArray

__all__ = ["SequenceModel"]


class SequenceModel(ABC):
    """Prediction model for target trajectories."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for logs and learning-curve legends."""

    @abstractmethod
    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
    ) -> None:
        """Fit to the training dataset.

        Parameters
        ----------
        train
            Training dataset. All statistics the model needs come from
            this object -- never from the full run pool.
        seed
            Model seed from the ``SeedBundle``, separate from the data and noise seeds.
        validation
            Optional, for early stopping. Unused by the analytical baselines; they
            have no training history that could be stopped.
        """

    @abstractmethod
    def predict(self, batch: WindowBatch) -> FloatArray:
        """Targets over the horizon, shape ``(batch, H, n_targets)``, standardised.

        Standardised, not in engineering units: the back-transformation belongs in the
        evaluator, which knows the scaler of the training dataset. Otherwise every
        model would have to implement the back-transformation correctly itself.
        """

    @property
    def n_parameters(self) -> int:
        """Number of free parameters. Relevant for placing the learning curves."""
        return 0

    def describe(self) -> dict[str, Any]:
        """Configuration for experiment tracking."""
        return {"model": self.name, "n_parameters": self.n_parameters}

    def training_diagnostics(self) -> dict[str, Any]:
        """Training-history metrics for tracking, empty without a training loop.

        Models without iterative optimisation (persistence, ARX, NARX, ...) have no
        training history that could be stopped -- the empty dict is correct for
        them, not just a placeholder.
        """
        return {}
