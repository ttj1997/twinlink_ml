"""Errors of one model against three references: noise-free truth, disturbed, processed signal.

The model sees the processed measurements (``dataset``). The same windows are cut from the disturbed
data set (``disturbed``, same split, same window configuration, same scalers), so window ``i`` of
both data sets is the same run and start; only the target values differ. Metric as Befund 94:
per-run RMSE over the first ``h`` steps in physical units, median over the runs
(:func:`~twinlink_ml.evaluation.evaluator.per_run_rmse_median`).

* ``rmse_run_median_physical_truth`` -- against ``trajectory.outputs`` (main metric of the study);
* ``rmse_run_median_physical_disturbed`` -- against the measurement with outliers, before any
  filter (what a check at the plant would see);
* ``rmse_run_median_physical_processed`` -- against the processed measurement; without a filter
  equal to the disturbed one, and at ``p = 0`` without a filter the metric of Befund 94.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..evaluation.evaluator import per_run_rmse_median

if TYPE_CHECKING:
    from ..data.windows import TrajectoryWindowDataset
    from ..models.base import SequenceModel

__all__ = ["REFERENCES", "evaluate_against_references"]

REFERENCES = ("truth", "disturbed", "processed")
BATCH_SIZE = 4096


def evaluate_against_references(
    model: SequenceModel,
    dataset: TrajectoryWindowDataset,
    disturbed: TrajectoryWindowDataset,
    *,
    channel: str,
    horizons: Sequence[int],
) -> dict[str, float]:
    """Flat metrics ``rmse_run_median_physical_<reference>/<channel>/h<h>``."""
    if dataset.n_windows != disturbed.n_windows or dataset.run_ids != disturbed.run_ids:
        raise ValueError("dataset and disturbed must have the same windows")
    column = dataset.target_names.index(channel)
    scaler = dataset.target_scaler
    predictions, truths, processed, disturbed_targets, runs = [], [], [], [], []
    for start in range(0, dataset.n_windows, BATCH_SIZE):
        indices = list(range(start, min(start + BATCH_SIZE, dataset.n_windows)))
        batch = dataset.batch(indices)
        other = disturbed.batch(indices)
        if not np.array_equal(batch.run_index, other.run_index) or not np.array_equal(
            batch.window_start, other.window_start
        ):
            raise ValueError("window index differs between dataset and disturbed")
        if batch.future_targets_truth is None:
            raise ValueError("dataset carries no noise-free targets")
        predictions.append(model.predict(batch))
        truths.append(batch.future_targets_truth)
        processed.append(batch.future_targets)
        disturbed_targets.append(other.future_targets)
        runs.append(batch.run_index)
    run_index = np.concatenate(runs)
    prediction = scaler.inverse(np.concatenate(predictions))[..., [column]]
    references = {
        "truth": np.concatenate(truths),
        "disturbed": np.concatenate(disturbed_targets),
        "processed": np.concatenate(processed),
    }
    metrics: dict[str, float] = {}
    for name in REFERENCES:
        reference = scaler.inverse(references[name])[..., [column]]
        result = per_run_rmse_median(prediction, reference, run_index, horizons, (channel,))
        for horizon, value in result[channel].items():
            metrics[f"rmse_run_median_physical_{name}/{channel}/h{horizon}"] = float(value)
    return metrics
