"""Apply an outlier process and an optional causal filter to a generated data set (no simulation).

The existing sensor model (PT1 + Gaussian noise) is left exactly as it is: the outliers are added to
the measurements that :meth:`~twinlink_ml.evaluation.experiments.GeneratedData.with_measurements` /
``remeasure`` produce, only on one channel. The scale of the outlier error is a multiple of the
standard deviation of the existing Gaussian noise at that sample, ``relative_noise_std x |PT1
output|`` (the greybox sensor model draws its noise with exactly that standard deviation). The PT1
output is obtained from the sensor model itself (``remeasure`` with both noise levels set to 0), so
the filter recursion is not duplicated here.

Two data sets come out:

* ``disturbed``: measurements with outliers (what the plant sensor delivers);
* ``processed``: ``disturbed`` after the optional causal filter -- the measurements every model
  sees, in the input history AND as training target (as at the plant). Without a filter it is
  ``disturbed`` itself.

The noise-free simulator output (``trajectory.outputs``) is never touched; it is the reference of
the main metric.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.seeding import _derive
from .hampel import CausalHampelFilter
from .process import BurstOutliers, IndependentOutliers, NoOutliers, OutlierProcess

if TYPE_CHECKING:
    from ..evaluation.experiments import GeneratedData
    from ..simulation.backend import FloatArray
    from .config import OutlierStudyConfig

__all__ = [
    "Condition",
    "ContaminatedData",
    "apply_condition",
    "existing_noise_std",
    "outlier_process",
    "outlier_seed",
]


@dataclass(frozen=True, slots=True)
class Condition:
    """One data condition of the study: outlier rate and type, preprocessing."""

    rate: float
    kind: str
    """``none`` (only with ``rate == 0``), ``independent`` or ``burst``."""
    filter_window: int | None
    """Window of the causal Hampel filter, ``None`` = no preprocessing."""

    def __post_init__(self) -> None:
        if self.kind not in ("none", "independent", "burst"):
            raise ValueError(f"unknown outlier kind {self.kind!r}")
        if (self.kind == "none") != (self.rate == 0.0):
            raise ValueError("kind 'none' if and only if rate == 0")
        if self.filter_window is not None and self.filter_window < 1:
            raise ValueError("filter_window must be >= 1")

    @property
    def filter_label(self) -> str:
        return "none" if self.filter_window is None else f"hampel{self.filter_window}"

    @property
    def id(self) -> str:
        """Stable identifier, e.g. ``p0.25_burst_hampel8``."""
        return f"p{self.rate:g}_{self.kind}_{self.filter_label}"

    @classmethod
    def parse(cls, text: str) -> Condition:
        rate, kind, filt = text.split("_")
        window = None if filt == "none" else int(filt.removeprefix("hampel"))
        return cls(rate=float(rate.removeprefix("p")), kind=kind, filter_window=window)

    def as_parameters(self) -> dict[str, Any]:
        return {
            "condition/id": self.id,
            "outlier/rate": self.rate,
            "outlier/kind": self.kind,
            "preprocess/filter": self.filter_label,
            "preprocess/hampel_window": self.filter_window,
        }


@dataclass(frozen=True, slots=True)
class ContaminatedData:
    """Disturbed and processed data set of one condition, with diagnostics."""

    disturbed: GeneratedData
    processed: GeneratedData
    diagnostics: dict[str, Any]


def outlier_process(condition: Condition, config: OutlierStudyConfig) -> OutlierProcess:
    """The outlier process of a condition."""
    settings = config.outliers
    if condition.kind == "none":
        return NoOutliers()
    if condition.kind == "independent":
        return IndependentOutliers(
            rate=condition.rate, degrees_of_freedom=settings.t_degrees_of_freedom
        )
    return BurstOutliers(
        rate=condition.rate,
        mean_length=settings.mean_burst_length_steps,
        degrees_of_freedom=settings.t_degrees_of_freedom,
    )


def outlier_seed(data: GeneratedData, kind: str) -> int:
    """Seed of the outlier draw: derived from the noise seed of the data set and the type.

    Not from the rate: with the same seed the independent outliers of p = 0.05 are a subset of those
    of p = 0.25 (common random numbers). The fixed constant-run holdout (seed 999) thus gets one
    fixed outlier realisation per type, as it has one fixed noise realisation.
    """
    return _derive(data.seeds.noise_seed, "outlier", kind)


def _channel_index(data: GeneratedData, channel: str) -> int:
    names = tuple(data.trajectory.metadata.get("output_names", ()))
    if channel not in names:
        raise ValueError(f"channel {channel!r} not among the outputs {names}")
    return names.index(channel)


def existing_noise_std(data: GeneratedData, channel: str) -> FloatArray:
    """Standard deviation of the existing Gaussian noise per sample, shape ``(runs, time)``."""
    measurement = data.process_config.measurement
    if not measurement.enabled:
        return np.zeros(data.trajectory.outputs.shape[:2])
    noiseless = dataclasses.replace(
        measurement, relative_noise_std=0.0, absolute_noise_std_temperature_k=0.0
    )
    pt1 = data.with_measurements(measurement=noiseless).trajectory.measurements
    return measurement.relative_noise_std * np.abs(pt1[..., _channel_index(data, channel)])


def _with_channel(data: GeneratedData, index: int, values: FloatArray) -> GeneratedData:
    measurements = np.array(data.trajectory.measurements, dtype=np.float64, copy=True)
    measurements[..., index] = values
    return dataclasses.replace(
        data, trajectory=dataclasses.replace(data.trajectory, measurements=measurements)
    )


def apply_condition(
    data: GeneratedData, condition: Condition, config: OutlierStudyConfig
) -> ContaminatedData:
    """Outliers on ``config.channel``, then the optional causal Hampel filter on that channel."""
    index = _channel_index(data, config.channel)
    process = outlier_process(condition, config)
    seed = outlier_seed(data, condition.kind)
    if condition.kind == "none":
        disturbed = data
        realised_rate = 0.0
    else:
        scale = config.outliers.scale_factor * existing_noise_std(data, config.channel)
        sample = process.sample(scale, np.random.default_rng(seed))
        disturbed = _with_channel(
            data, index, data.trajectory.measurements[..., index] + sample.error
        )
        realised_rate = sample.realised_rate
    diagnostics: dict[str, Any] = {
        **condition.as_parameters(),
        "outlier/seed": seed if condition.kind != "none" else None,
        "outlier/realised_rate": realised_rate,
        **{f"outlier/process_{k}": v for k, v in process.describe().items()},
    }
    if condition.filter_window is None:
        diagnostics["preprocess/replaced_fraction"] = 0.0
        return ContaminatedData(disturbed=disturbed, processed=disturbed, diagnostics=diagnostics)
    hampel = CausalHampelFilter(window=condition.filter_window, threshold=config.hampel.threshold)
    result = hampel.apply(disturbed.trajectory.measurements[..., index])
    diagnostics["preprocess/replaced_fraction"] = result.replaced_fraction
    diagnostics["preprocess/hampel_threshold"] = hampel.threshold
    processed = _with_channel(disturbed, index, result.values)
    return ContaminatedData(disturbed=disturbed, processed=processed, diagnostics=diagnostics)
