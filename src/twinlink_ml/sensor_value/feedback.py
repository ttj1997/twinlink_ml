r"""Feedback variants F0-F3 and the causal first-order bias filter.

The controller corrects its nominal model with a zeroth-order bias
:math:`b`, the first-order filtered difference between a feedback value of the product moisture
and the model output (:mod:`.loop`):

.. math:: e_j = y_j - \hat x_j, \qquad b_j = a\, b_{j-1} + (1 - a)\, e_j, \qquad
          a = \exp(-\Delta t / \tau)

with the feedback grid :math:`\Delta t` (25 s) and the time constant :math:`\tau` (30 min,
placeholder). The first update sets :math:`b = e` (the filter has no earlier value). The filter
is causal by construction: :math:`b_j` uses :math:`e_0 \dots e_j` only.

Feedback values :math:`y_j` on the 25 s grid, one object per pair (draw, scenario):

F0 :class:`NoFeedback`
    none; :math:`b \equiv 0`.
F1 :class:`IdealMeasurement`
    the true product moisture at the end of each 25 s block.
F2 :class:`InlineSensor`
    the existing sensor model (:meth:`GreyBoxFluidBedGranulator.measure`: PT1 with
    ``sensor_time_constant_s`` plus relative Gaussian noise, ``MeasurementConfig`` of the process
    config) run at 25 s, plus independent outliers (:class:`~twinlink_ml.outliers.process.
    IndependentOutliers`, Student-t, scale = ``scale_factor`` x the std of the Gaussian noise at
    that sample, ``relative_noise_std x |PT1 output|``, as Befund 109); no outlier filter, no
    clipping.
    The PT1 output is taken from a noise-free copy of the same sensor model. Random numbers come
    from two per-pair generators, so a pair sees the same noise at every margin and in every batch.
F3 :class:`SoftSensorFeedback`
    the moisture head of a GRU soft sensor (:mod:`.softsensor`) from the exogenous channels.
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod

import numpy as np

from twinlink_ml.outliers.process import IndependentOutliers
from twinlink_ml.simulation import GreyBoxFluidBedGranulator, MeasurementConfig, ProcessConfig

from .softsensor import SoftSensor

__all__ = [
    "BiasFilter",
    "Feedback",
    "IdealMeasurement",
    "InlineSensor",
    "NoFeedback",
    "SoftSensorFeedback",
]

_MOISTURE_CHANNEL = 2  # position of the product moisture in the grey-box outputs


class BiasFilter:
    """Causal first-order filter of the model error, see the module docstring."""

    def __init__(self, time_constant_s: float, interval_s: float) -> None:
        if time_constant_s <= 0.0 or interval_s <= 0.0:
            raise ValueError("time constant and interval must be positive")
        self.decay = math.exp(-interval_s / time_constant_s)
        self.value = 0.0
        self.updates = 0

    def update(self, error: float) -> float:
        """Filter one error sample; returns the new bias."""
        if self.updates == 0:
            self.value = float(error)
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * float(error)
        self.updates += 1
        return self.value


class Feedback(ABC):
    """Feedback of one pair on the 25 s grid."""

    name: str = "abstract"
    uses_model_output: bool = True
    """False only for F0: no feedback, the parallel model need not be simulated."""

    @abstractmethod
    def values(self, true_moisture: np.ndarray, exogenous: np.ndarray) -> np.ndarray:
        """Feedback values ``(m,)`` for ``m`` consecutive 25 s blocks.

        ``true_moisture``: true product moisture at the end of each block; ``exogenous``:
        ``(m, 7)`` block means of the exogenous channels in the order of the soft-sensor data
        (actions, then disturbances). Called once per control interval, blocks in time order.
        """


class NoFeedback(Feedback):
    """F0."""

    name = "F0"
    uses_model_output = False

    def values(self, true_moisture: np.ndarray, exogenous: np.ndarray) -> np.ndarray:
        del true_moisture, exogenous
        raise RuntimeError("F0 has no feedback")


class IdealMeasurement(Feedback):
    """F1: the true moisture."""

    name = "F1"

    def values(self, true_moisture: np.ndarray, exogenous: np.ndarray) -> np.ndarray:
        del exogenous
        return np.asarray(true_moisture, dtype=np.float64).copy()


class InlineSensor(Feedback):
    """F2: existing sensor model plus independent outliers, see the module docstring."""

    name = "F2"

    def __init__(
        self,
        process: ProcessConfig,
        measurement: MeasurementConfig,
        interval_s: float,
        *,
        outlier_rate: float,
        degrees_of_freedom: float,
        scale_factor: float,
        noise_seed: int,
        outlier_seed: int,
    ) -> None:
        if not measurement.enabled:
            raise ValueError("the inline sensor needs an enabled measurement config")
        numerics = dataclasses.replace(process.numerics, sample_time_s=interval_s)
        noisy = dataclasses.replace(process, numerics=numerics, measurement=measurement)
        clean = dataclasses.replace(
            noisy,
            measurement=dataclasses.replace(
                measurement, relative_noise_std=0.0, absolute_noise_std_temperature_k=0.0
            ),
        )
        self._sensor = GreyBoxFluidBedGranulator(noisy)
        self._pt1 = GreyBoxFluidBedGranulator(clean)
        self._relative_std = measurement.relative_noise_std
        self._scale_factor = scale_factor
        self._outliers = IndependentOutliers(
            rate=outlier_rate, degrees_of_freedom=degrees_of_freedom
        )
        self._noise_rng = np.random.default_rng(noise_seed)
        self._outlier_rng = np.random.default_rng(outlier_seed)
        self._unused_rng = np.random.default_rng(0)  # noise-free copy: draws are multiplied by 0
        self.outliers_drawn = 0
        self.samples = 0

    def values(self, true_moisture: np.ndarray, exogenous: np.ndarray) -> np.ndarray:
        del exogenous
        out = np.empty(len(true_moisture))
        for j, x in enumerate(np.asarray(true_moisture, dtype=np.float64)):
            outputs = np.zeros((1, 4))
            outputs[0, _MOISTURE_CHANNEL] = x
            measured = np.asarray(self._sensor.measure(outputs, self._noise_rng))
            pt1 = np.asarray(self._pt1.measure(outputs, self._unused_rng))
            scale = self._scale_factor * self._relative_std * np.abs(pt1[:, [_MOISTURE_CHANNEL]])
            draw = self._outliers.sample(scale, self._outlier_rng)
            out[j] = measured[0, _MOISTURE_CHANNEL] + draw.error[0, 0]
            self.outliers_drawn += int(draw.mask.sum())
            self.samples += 1
        return out


class SoftSensorFeedback(Feedback):
    """F3: GRU soft sensor (moisture head) on the exogenous channels."""

    name = "F3"

    def __init__(self, sensor: SoftSensor, seed_label: str) -> None:
        self._sensor = sensor
        self.name = f"F3:{seed_label}"
        self._history: np.ndarray | None = None

    def values(self, true_moisture: np.ndarray, exogenous: np.ndarray) -> np.ndarray:
        del true_moisture  # the soft sensor never sees the moisture
        exogenous = np.asarray(exogenous, dtype=np.float64)
        length = self._sensor.history_length
        if self._history is None:
            # the plant ran at the first block's inputs before (steady pre-history)
            self._history = np.repeat(exogenous[:1], length, axis=0)
        rows = np.concatenate([self._history, exogenous], axis=0)
        histories = np.stack([rows[j : j + length] for j in range(len(exogenous))])
        estimates = self._sensor.estimate(histories, exogenous)
        self._history = rows[-length:]
        return estimates
