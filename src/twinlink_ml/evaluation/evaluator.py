r"""Metrics and evaluation protocol (work plan step 5).

Ranking of the metrics -- deliberately against common practice
---------------------------------------------------------------
1. **Multi-step rollout error** over several horizons -- primary.
2. One-step error -- only a secondary metric. At :math:`\Delta t = 5` s the
   persistence NRMSE is 0.016; a comparison on this metric measures seed variance.
3. Inference time -- real-time capability is a selection criterion in AP 1.

Normalisation
-------------
.. math:: \mathrm{NRMSE} = \frac{\sqrt{\overline{(y - \hat y)^2}}}{\sigma_y}

with :math:`\sigma_y` from the **evaluation** split, not from training. Thus 1.0 is the
threshold below which a model is better at all than predicting the mean, and the number
is comparable across targets with completely different units (kW, kg/kg, µm).

Two references per number
-------------------------
``nrmse`` measures against the noisy measurement -- that is what can be checked at the
plant. ``nrmse_truth`` measures against the noise-free output of the grey-box model. The
difference between the two is the irreducible part and thus the **saturation level of the
learning curve**. Without this second number, a learning curve that flattens at 0.12 is
read as a model limit, although it may be the noise limit.

Run-level normalisation is the primary number, not the pooled one
-----------------------------------------------------------------
If :math:`\sigma_y` is estimated pooled over all runs, it contains the level differences
**between** the runs. They arise from the scattered initial states (excitation design,
step 2) and do not have to be predicted by any model -- they are in the window history.
The denominator is thus artificially large.

Measured on the grey-box model, :math:`\Delta t = 25` s, ARX(4,2), :math:`h = 24`:

============================  ========  ==========
Target                        pooled    per run
============================  ========  ==========
``energy_target``               0.133      0.154
``product_moisture_kg_per_kg``  0.043      1.354
``d50_um``                      0.026      0.091
============================  ========  ==========

For product moisture this changes the statement: pooled, the row reads as "practically
solved"; run-level, the error lies above the spread within the run -- the model does not
beat the mean of its own run. The effect is small for ``energy_target`` and large for the
quality variables, i.e. exactly where the architecture comparison takes place. Therefore
the **median over runs** is the reported number and the pooled one is only secondary
information.

Windows of the same run are in addition strongly correlated; the run-level aggregation
thus also gives a more robust spread figure than the number of windows suggests.

Absolute RMSE in addition to the NRMSE
--------------------------------------
The denominator ``np.std(reference)`` in :func:`nrmse` is estimated anew **per run** (see
above) and itself scatters strongly between the runs -- measured on a constant-run holdout
of the grey-box model (``scripts/probe_e0_normalisation_artifact.py``,
``artifacts/runs/probe_e0_normalisation_artifact.json``) by a factor of 26.6 for ``d50_um``
and by a factor of 2292.7 for ``product_moisture_kg_per_kg`` (minimum to maximum over all
checked runs, seed 1, GRU/LSTM, h=1/h=36). The NRMSE alone can therefore not be translated
into a physical error: the same NRMSE value can come from a small absolute error in a calm
run (small denominator) or a large absolute error in a strongly scattering run (large
denominator). :func:`per_run_rmse_median` and
``EvaluationResult.rmse_run_median_physical`` therefore additionally provide the
unnormalised RMSE in physical units (after ``target_scaler.inverse()``, independent of the
``physical_units`` switch of :class:`RolloutEvaluator`) -- as a complement, not as a
replacement: the run-level NRMSE remains the primary number for model comparisons,
because it makes targets of different units comparable; the absolute RMSE answers the
separate question of how large the error actually is in the unit of the target.
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
    """NRMSE, normalised by the spread of the reference along the same axes."""
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
    r"""NRMSE, pooled over all windows and the steps :math:`1..h`.

    Cumulative and not pointwise: the rollout error at :math:`h` should contain the path
    there. A model that happens to be close to the truth again at :math:`h = 12` after
    diverging in between must not look good.

    A module function and not a method, so that the same normalisation convention can
    also be used outside a model comparison -- for instance for the noise limit in
    :mod:`twinlink_ml.evaluation.protocol`, which evaluates no model.
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
    """NRMSE per run, summarised as ``(Median, Minimum, Maximum)``.

    The **primary number** of the protocol, see the module docstring: the denominator
    thereby contains only the spread within a run and not the level differences between
    the runs, which come from the scattered initial states.
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
    """Unnormalised RMSE per run, median over the runs -- complement to the NRMSE.

    ``prediction``/``reference`` must already be in physical units
    (``target_scaler.inverse()``), regardless of whether the caller computes the NRMSE
    itself standardised or physically -- see the module docstring, section "Absolute RMSE
    in addition to the NRMSE". Only the median, no range: the spread of the absolute error
    over runs is already visible via the NRMSE denominator (``nrmse_run_min``/``_max``),
    here the point is the level.
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
    """Result of an evaluation, flat enough for MLflow/W&B metrics.

    Attributes
    ----------
    model_name
        Short name of the model.
    target_names
        Targets in column order.
    horizons
        Evaluated horizons in sampling steps; ``1`` is the one-step error.
    nrmse_by_horizon
        ``{Zielgröße: {Horizont: NRMSE}}`` against the measurement, pooled over all windows.
    nrmse_truth_by_horizon
        The same against the noise-free reference, or empty.
    nrmse_run_spread
        ``{Zielgröße: {Horizont: (Median, Minimum, Maximum)}}`` over runs -- the
        primary number.
    nrmse_truth_run_spread
        The same against the noise-free reference. Without this number the saturation
        level of a learning curve cannot be read in the same normalisation as the curve
        itself.
    rmse_run_median_physical
        ``{Zielgröße: {Horizont: Median-RMSE}}`` over runs, unnormalised, in physical
        units -- complement to the NRMSE, see the module docstring "Absolute RMSE in
        addition to the NRMSE". Always in physical units, independent of the
        ``physical_units`` switch of :class:`RolloutEvaluator`.
    inference_us_per_window
        Inference time per window for the full horizon, in microseconds.
    n_windows, n_runs
        Size of the evaluation.
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
        """Flat key-value pairs for experiment tracking.

        Contains both normalisations and both references. The run-level figures are here
        and not only in the driver, so that there is only **one** place where a metric
        name is created -- otherwise log keys and analysis drift apart.
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
        # A separate variable name per loop, on purpose: the same name for a
        # `dict[int, float]` further up and a `dict[int, tuple[float, float, float]]`
        # here would be rejected by mypy as an incompatible reassignment -- that is not a
        # formality but the same mistake a human reader would also make: a pooled number
        # is not a median/min/max range.
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
        """Table for run logs."""
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
    """Abstraction over evaluation protocols."""

    @abstractmethod
    def evaluate(self, model: SequenceModel, dataset: SequenceDataset) -> EvaluationResult:
        """Evaluate a model on a data set."""


@dataclass(frozen=True, slots=True)
class RolloutEvaluator(Evaluator):
    """Standard protocol: NRMSE over several horizons, run-level spread, inference time.

    Attributes
    ----------
    horizons
        Horizons to evaluate. ``None`` automatically selects ``1``, a quarter, half and
        the full horizon of the data set.
    physical_units
        Compute metrics after back-transformation into engineering units. For the NRMSE
        numerically almost equivalent, but the back-transformation exposes errors in
        passing the scalers between the splits -- the most frequent silent leakage
        variant after the split itself.
    batch_size
        Batch size of the evaluation.
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
        """Evaluate the rollout."""
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

        # For rmse_run_median_physical the values are ALWAYS converted back into physical
        # units, independent of the physical_units switch below -- an unnormalised error
        # in standardised units would not be interpretable.
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
        """Protocol parameters for tracking."""
        return {
            "evaluator": type(self).__name__,
            "horizons": self.horizons,
            "physical_units": self.physical_units,
        }
