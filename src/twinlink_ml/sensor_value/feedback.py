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

Follow-up study (docs/29_sensor_value_2.md):

F4 :class:`LabSample`
    periodic laboratory sample: the true moisture at the sample instant plus Gaussian error,
    available after an analysis delay; the bias is updated only when a result arrives,
    :math:`b \leftarrow b + g\,(y_s - (\hat x_s + b_s))` with the prediction of the corrected
    model at the sample instant, and held in between (:class:`SampleAndHoldBias`).
F5 :class:`SoftSensorFeedback` with a :class:`~.responses.ResponseSensor`
    the moisture head of a GRU whose inputs are the exogenous channels PLUS measured process
    responses (bed temperature, exhaust temperature, exhaust humidity, through the existing sensor
    model); the bias filter is the one of F1-F3.

Every variant hands its values to the bias through :meth:`Feedback.feed`; for F0-F3 this is the
loop of the original study unchanged (same arithmetic, test against digests of the old code).
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from twinlink_ml.outliers.process import IndependentOutliers
from twinlink_ml.simulation import GreyBoxFluidBedGranulator, MeasurementConfig, ProcessConfig

from .responses import ResponseSensor
from .softsensor import SoftSensor

__all__ = [
    "BiasFilter",
    "Feedback",
    "IdealMeasurement",
    "InlineSensor",
    "LabSample",
    "NoFeedback",
    "SampleAndHoldBias",
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


class SampleAndHoldBias:
    """Bias of F4: zero until the first result, changed only by :meth:`apply`, held in between."""

    def __init__(self) -> None:
        self.value = 0.0
        self.updates = 0

    def apply(self, innovation: float, gain: float) -> float:
        """``b <- b + gain * innovation``; returns the new bias."""
        self.value = self.value + gain * float(innovation)
        self.updates += 1
        return self.value


class Feedback(ABC):
    """Feedback of one pair on the 25 s grid."""

    name: str = "abstract"
    uses_model_output: bool = True
    """False only for F0: no feedback, the parallel model need not be simulated."""
    needs_responses: bool = False
    """True for F5: the loop hands over the true process responses at the block ends."""

    def bias_filter(self, time_constant_s: float, interval_s: float) -> Any:
        """The bias estimator of this variant (F0-F3, F5: the first-order filter)."""
        return BiasFilter(time_constant_s, interval_s)

    def feed(
        self,
        bias: Any,
        true_moisture: np.ndarray,
        model_moisture: np.ndarray,
        exogenous: np.ndarray,
        responses: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update ``bias`` from the blocks of one control interval; return the errors ``y - x``.

        ``model_moisture``: the parallel nominal model at the block ends; ``responses``: ``(m, 3)``
        true bed temperature, exhaust temperature and exhaust humidity at the block ends (only
        for ``needs_responses``). The errors (feedback value minus true moisture) go into the
        record; F4 returns them only for the samples taken in these blocks.
        """
        del responses
        y = self.values(true_moisture, exogenous)
        for value, x_hat in zip(y, model_moisture, strict=True):
            bias.update(value - x_hat)
        return y - true_moisture

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
    """F3: GRU soft sensor (moisture head) on the exogenous channels; F5 with measured responses.

    ``responses``: the sensor of the measured process responses of the pair (required exactly when
    the soft sensor was trained with ``measured_input_names``); their measured values are appended
    to the exogenous block values in the order of the checkpoint.
    """

    name = "F3"

    def __init__(
        self, sensor: SoftSensor, seed_label: str, responses: ResponseSensor | None = None
    ) -> None:
        if bool(sensor.measured_input_names) != (responses is not None):
            raise ValueError("a response sensor is needed exactly for measured inputs")
        if responses is not None and responses.names != sensor.measured_input_names:
            raise ValueError(
                f"response sensor {responses.names} differs from the checkpoint "
                f"{sensor.measured_input_names}"
            )
        self._sensor = sensor
        self._responses = responses
        self.needs_responses = responses is not None
        self.name = f"{'F5' if responses is not None else 'F3'}:{seed_label}"
        self._history: np.ndarray | None = None

    def feed(
        self,
        bias: Any,
        true_moisture: np.ndarray,
        model_moisture: np.ndarray,
        exogenous: np.ndarray,
        responses: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._responses is None:
            return super().feed(bias, true_moisture, model_moisture, exogenous)
        if responses is None:
            raise ValueError("F5 needs the process responses")
        measured = self._responses.measure(responses)
        y = self._estimate(np.concatenate([np.asarray(exogenous, np.float64), measured], axis=1))
        for value, x_hat in zip(y, model_moisture, strict=True):
            bias.update(value - x_hat)
        return y - true_moisture

    def values(self, true_moisture: np.ndarray, exogenous: np.ndarray) -> np.ndarray:
        del true_moisture  # the soft sensor never sees the moisture
        if self._responses is not None:
            raise RuntimeError("F5 is fed through feed() with the process responses")
        return self._estimate(exogenous)

    def _estimate(self, exogenous: np.ndarray) -> np.ndarray:
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


class LabSample(Feedback):
    """F4: periodic laboratory sample with analysis delay, see the module docstring.

    Time is counted in feedback blocks (25 s) from the start of the closed loop (start of the
    pre-roll). A sample is taken at the end of blocks ``1, 1 + s, 1 + 2 s, ...`` (``s`` = sampling
    interval in blocks; the first sample at the first feedback instant), its result
    ``y = x_true + N(0, sigma^2)`` arrives ``d`` blocks later (analysis delay) and updates the bias
    at the end of that block with the innovation against the corrected prediction at the sample
    instant, ``y - (x_model + b)`` with ``x_model`` and ``b`` of the sample instant. The controller
    reads the bias at each interval start (as for every variant). Random numbers from one per-pair
    generator (common random numbers over the margins).
    """

    name = "F4"

    def __init__(
        self,
        *,
        sampling_interval_s: float,
        delay_s: float,
        error_std_kg_per_kg: float,
        gain: float,
        interval_s: float,
        seed: int,
        label: str,
    ) -> None:
        every = sampling_interval_s / interval_s
        delay = delay_s / interval_s
        if every != round(every) or delay != round(delay) or every < 1 or delay < 0:
            raise ValueError("sampling interval and delay must be whole numbers of blocks")
        if not 0.0 < gain <= 1.0 or error_std_kg_per_kg < 0.0:
            raise ValueError("gain in (0, 1], error std >= 0")
        self.every = round(every)
        self.delay = round(delay)
        self.error_std = error_std_kg_per_kg
        self.gain = gain
        self.name = f"F4:{label}"
        self._rng = np.random.default_rng(seed)
        self._block = 0
        self._pending: list[tuple[int, float, float, float]] = []  # arrival, y, x_model, b
        self.samples_taken = 0
        self.results_arrived = 0
        self.innovations: list[float] = []

    def bias_filter(self, time_constant_s: float, interval_s: float) -> SampleAndHoldBias:
        del time_constant_s, interval_s
        return SampleAndHoldBias()

    def values(self, true_moisture: np.ndarray, exogenous: np.ndarray) -> np.ndarray:
        del true_moisture, exogenous
        raise RuntimeError("F4 is fed through feed()")

    def feed(
        self,
        bias: Any,
        true_moisture: np.ndarray,
        model_moisture: np.ndarray,
        exogenous: np.ndarray,
        responses: np.ndarray | None = None,
    ) -> np.ndarray:
        del exogenous, responses
        errors = []
        for x_true, x_model in zip(
            np.asarray(true_moisture, np.float64),
            np.asarray(model_moisture, np.float64),
            strict=True,
        ):
            self._block += 1
            if (self._block - 1) % self.every == 0:
                y = float(x_true) + float(self._rng.normal(0.0, self.error_std))
                self._pending.append((self._block + self.delay, y, float(x_model), bias.value))
                self.samples_taken += 1
                errors.append(y - float(x_true))
            while self._pending and self._pending[0][0] <= self._block:
                _, y, x_model_s, b_s = self._pending.pop(0)
                innovation = y - (x_model_s + b_s)
                bias.apply(innovation, self.gain)
                self.innovations.append(innovation)
                self.results_arrived += 1
        return np.asarray(errors, dtype=np.float64)
