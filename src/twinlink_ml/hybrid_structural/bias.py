r"""M2b: the refitted grey-box model (M2) plus a constant bias of the product moisture.

The bias is the mean of ``measured moisture - replay of the refitted model`` over every sample of
the ``n_train`` training runs (open-loop replay from the run start, as the refit itself). It is
added to the moisture of the steady map and of the replay; energy and bed temperature are those of
M2. With a bias of exactly zero the map IS the M2 map (tested).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from twinlink_ml.evaluation.experiments import GeneratedData
from twinlink_ml.hybrid_model.maps import MapPrediction, SteadyMap
from twinlink_ml.hybrid_model.residual import replay
from twinlink_ml.simulation import ProcessConfig

__all__ = ["BiasedMap", "training_bias"]


class BiasedMap(SteadyMap):
    """A steady map with a constant added to its moisture."""

    def __init__(self, base: SteadyMap, moisture_bias_kg_per_kg: float) -> None:
        if not np.isfinite(moisture_bias_kg_per_kg):
            raise ValueError("the moisture bias must be finite")
        self.base = base
        self.moisture_bias_kg_per_kg = float(moisture_bias_kg_per_kg)

    def predict(self, actions: np.ndarray, contexts: np.ndarray, *, wide: bool) -> MapPrediction:
        prediction = self.base.predict(actions, contexts, wide=wide)
        return dataclasses.replace(
            prediction,
            moisture_kg_per_kg=prediction.moisture_kg_per_kg + self.moisture_bias_kg_per_kg,
        )


def training_bias(
    data: GeneratedData, config: ProcessConfig, rows: list[int], target_names: Sequence[str]
) -> dict[str, float]:
    """Mean of ``measured - replay with config`` per target over the runs ``rows``."""
    names = list(data.trajectory.metadata["output_names"])
    cols = [names.index(t) for t in target_names]
    measured = data.trajectory.measurements[rows][..., cols]
    replayed = replay(data, config, rows).outputs[..., cols]
    mean = (measured - replayed).reshape(-1, len(cols)).mean(axis=0)
    return {t: float(v) for t, v in zip(target_names, mean, strict=True)}
