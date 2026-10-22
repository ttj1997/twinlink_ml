r"""Outlier processes of the sensor, on top of the existing sensor model (PT1 + Gaussian noise).

Why a separate process and not a larger noise level: the lab data of the predecessor project lost
17-34 % of the raw points per batch as outliers (docs/04). Gaussian noise of any level does not
describe that; the rank order of the architectures is known to be robust against the *level* of
Gaussian noise (step 5) but was never tested against outliers.

Both processes return an additive error array of the shape of ``scale`` (``(runs, time)``) and the
mask of the affected samples. A measured value ``y_m[k]`` becomes ``y_m[k] + e[k]``; ``e`` is zero
outside the mask.

* :class:`IndependentOutliers`: every sample is an outlier with probability ``rate``,
  independently; ``e[k] = scale[k] * t_nu``.
* :class:`BurstOutliers`: two-state Markov chain with the same stationary share ``rate``; a burst
  ends with probability ``1 / mean_length`` per step (geometric length, mean ``mean_length``),
  one starts with probability ``rate / (1 - rate) / mean_length``; the first state is drawn from
  the stationary distribution. One error per burst, ``scale`` taken at its first sample, held
  constant over the burst (offset).

The random numbers are drawn for every sample regardless of ``rate``, so with one seed the
independent outliers of a smaller rate are a subset of those of a larger rate (common random
numbers across the rates). No clipping: a heavy-tailed error may produce a non-physical value, as
a faulty sensor does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..simulation.backend import FloatArray

__all__ = [
    "BurstOutliers",
    "IndependentOutliers",
    "NoOutliers",
    "OutlierProcess",
    "OutlierSample",
]


@dataclass(frozen=True, slots=True)
class OutlierSample:
    """Additive error and mask of one draw."""

    error: FloatArray
    mask: np.ndarray

    @property
    def realised_rate(self) -> float:
        """Share of affected samples."""
        return float(self.mask.mean()) if self.mask.size else 0.0


class OutlierProcess(ABC):
    """An outlier process: error array for a given per-sample scale."""

    kind: str = "abstract"

    @abstractmethod
    def sample(self, scale: FloatArray, rng: np.random.Generator) -> OutlierSample:
        """Draw the error for ``scale`` of shape ``(runs, time)``."""

    def describe(self) -> dict[str, float | str]:
        """For the run log."""
        return {"kind": self.kind}


@dataclass(frozen=True, slots=True)
class NoOutliers(OutlierProcess):
    """No outliers: the measurement is returned unchanged (``p = 0``)."""

    kind = "none"

    def sample(self, scale: FloatArray, rng: np.random.Generator) -> OutlierSample:
        del rng  # no draw: p = 0 must leave the measurement bit-identical
        shape = np.shape(scale)
        return OutlierSample(error=np.zeros(shape), mask=np.zeros(shape, dtype=bool))

    def describe(self) -> dict[str, float | str]:
        return {"kind": self.kind, "rate": 0.0}


@dataclass(frozen=True, slots=True)
class IndependentOutliers(OutlierProcess):
    """Independent outliers with Student-t error."""

    rate: float
    degrees_of_freedom: float = 3.0
    kind = "independent"

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate < 1.0:
            raise ValueError(f"rate must lie in [0, 1), got {self.rate}")

    def sample(self, scale: FloatArray, rng: np.random.Generator) -> OutlierSample:
        scale = np.asarray(scale, dtype=np.float64)
        uniform = rng.random(scale.shape)
        t = rng.standard_t(self.degrees_of_freedom, size=scale.shape)
        mask = uniform < self.rate
        return OutlierSample(error=np.where(mask, scale * t, 0.0), mask=mask)

    def describe(self) -> dict[str, float | str]:
        return {"kind": self.kind, "rate": self.rate, "t_dof": self.degrees_of_freedom}


@dataclass(frozen=True, slots=True)
class BurstOutliers(OutlierProcess):
    """Bursts of geometric length with one constant Student-t offset per burst."""

    rate: float
    mean_length: float = 10.0
    degrees_of_freedom: float = 3.0
    kind = "burst"

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate < 1.0:
            raise ValueError(f"rate must lie in [0, 1), got {self.rate}")
        if self.mean_length < 1.0:
            raise ValueError("mean_length must be >= 1")
        if self.start_probability > 1.0:
            raise ValueError("rate too large for this mean burst length")

    @property
    def end_probability(self) -> float:
        """Probability that a burst ends after a step."""
        return 1.0 / self.mean_length

    @property
    def start_probability(self) -> float:
        """Probability that a burst starts, chosen so that the stationary share is ``rate``."""
        return self.rate / (1.0 - self.rate) * self.end_probability

    def sample(self, scale: FloatArray, rng: np.random.Generator) -> OutlierSample:
        scale = np.asarray(scale, dtype=np.float64)
        n_runs, n_steps = scale.shape
        uniform = rng.random((n_runs, n_steps))
        t = rng.standard_t(self.degrees_of_freedom, size=(n_runs, n_steps))
        mask = np.zeros((n_runs, n_steps), dtype=bool)
        error = np.zeros((n_runs, n_steps))
        if n_steps == 0 or self.rate == 0.0:
            return OutlierSample(error=error, mask=mask)
        state = uniform[:, 0] < self.rate
        offset = np.where(state, scale[:, 0] * t[:, 0], 0.0)
        mask[:, 0] = state
        error[:, 0] = offset
        for k in range(1, n_steps):
            starts = ~state & (uniform[:, k] < self.start_probability)
            stays = state & (uniform[:, k] >= self.end_probability)
            offset = np.where(starts, scale[:, k] * t[:, k], np.where(stays, offset, 0.0))
            state = starts | stays
            mask[:, k] = state
            error[:, k] = offset
        return OutlierSample(error=error, mask=mask)

    def describe(self) -> dict[str, float | str]:
        return {
            "kind": self.kind,
            "rate": self.rate,
            "t_dof": self.degrees_of_freedom,
            "mean_burst_length": self.mean_length,
        }
