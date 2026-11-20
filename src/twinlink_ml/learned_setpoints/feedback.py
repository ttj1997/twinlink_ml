r"""Moisture feedback for the learned steady-state maps (factor "feedback" of family M).

Scheme of Befund 111 with the learned map in the place of the nominal model: the set-point law
plans with the bound

.. math:: X_{map} \le X_{limit} - m - b ,

where :math:`b` estimates the error of the map, measured moisture minus predicted moisture. The
prediction is the STEADY prediction of the map at the inputs of the 25 s block (block means of the
realised commands and of the ambient state): the map has no dynamics, so during a transient the
difference also holds the transient, which the filter smooths.

Level ``fb``
    inline moisture measurement through the existing sensor model
    (:class:`twinlink_ml.sensor_value.feedback.InlineSensor` with outlier rate 0: first-order lag
    and Gaussian noise of the measurement config), first-order bias filter with the time constant
    of Befund 111.
Level ``lab``
    laboratory sample, the unchanged :class:`twinlink_ml.sensor_value.feedback.LabSample` (F4 of
    Befund 112: true moisture plus Gaussian error, result after the analysis delay, bias changed
    by ``gain`` times the innovation and held in between), with the sampling interval of this
    study.

Both levels are given a pre-roll before the day (length of Befund 111, the first disturbance
sample held, nominal throughput), in which the bias settles; neither cost nor violations of the
pre-roll count. Random numbers come from generators seeded per day, so a day sees the same noise at
every margin.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import numpy as np

from twinlink_ml.load_following.controllers import SetPointLaw
from twinlink_ml.sensor_value.feedback import Feedback, InlineSensor, LabSample

from .config import LearnedSetpointsConfig
from .maps import TableMap

__all__ = [
    "FEEDBACK_LEVELS",
    "BiasedSetPointLaw",
    "FeedbackLevel",
    "MoistureFeedback",
    "build_feedback",
]

FeedbackLevel = Literal["none", "fb", "lab"]
FEEDBACK_LEVELS: tuple[str, ...] = ("none", "fb", "lab")


class BiasedSetPointLaw(SetPointLaw):
    """:class:`SetPointLaw` whose bound is lowered by a bias; with bias 0 it is the base class."""

    def __init__(self, steady_map: Any, limit: float, margin: float) -> None:
        self._base_bound = limit - margin
        self.bias = 0.0
        super().__init__(steady_map, limit, margin)

    @property
    def bound(self) -> float:
        """Moisture bound of the map: ``limit - margin - bias``."""
        return self._base_bound - self.bias

    @bound.setter
    def bound(self, value: float) -> None:
        self._base_bound = float(value)


class MoistureFeedback:
    """Bias of one day-run, updated from the blocks of each control interval."""

    def __init__(
        self, feedback: Feedback, table_map: TableMap, config: LearnedSetpointsConfig
    ) -> None:
        self._feedback = feedback
        self._map = table_map
        settings = config.feedback
        self._filter = feedback.bias_filter(
            settings.bias_time_constant_s, settings.measurement_interval_s
        )
        self._last_prediction: float | None = None
        self.invalid_predictions = 0
        self.trace: list[float] = []

    @property
    def bias(self) -> float:
        """Current bias estimate, kg/kg."""
        return float(self._filter.value)

    def update(self, exogenous: np.ndarray, moisture: np.ndarray) -> None:
        """Feed the blocks of one interval: block-mean inputs ``(m, 7)``, true moisture ``(m,)``."""
        predicted = np.empty(len(moisture))
        for j, row in enumerate(exogenous):
            values, ok = self._map.response(row[0], row[1], row[2], row[4], row[5])
            if ok:
                self._last_prediction = float(values[1])
            else:
                self.invalid_predictions += 1
            if self._last_prediction is None:
                # no valid prediction yet: the measurement itself, i.e. no information
                self._last_prediction = float(moisture[j])
            predicted[j] = self._last_prediction
        self._feedback.feed(self._filter, np.asarray(moisture, np.float64), predicted, exogenous)
        self.trace.append(self.bias)


def build_feedback(
    level: FeedbackLevel, table_map: TableMap, config: LearnedSetpointsConfig, day_of_year: int
) -> MoistureFeedback | None:
    """Feedback object of one day-run at ``level`` (``None`` for the level without feedback)."""
    if level == "none":
        return None
    settings = config.feedback
    process = config.load_following.process
    if level == "fb":
        feedback: Feedback = InlineSensor(
            process,
            dataclasses.replace(process.measurement, enabled=True),
            settings.measurement_interval_s,
            outlier_rate=0.0,
            degrees_of_freedom=3.0,
            scale_factor=1.0,
            noise_seed=int(
                np.random.SeedSequence([settings.noise_seed, day_of_year, 1]).generate_state(1)[0]
            ),
            outlier_seed=int(
                np.random.SeedSequence([settings.noise_seed, day_of_year, 2]).generate_state(1)[0]
            ),
        )
    elif level == "lab":
        feedback = LabSample(
            sampling_interval_s=settings.lab_sampling_interval_s,
            delay_s=settings.lab_delay_s,
            error_std_kg_per_kg=settings.lab_error_std_kg_per_kg,
            gain=settings.lab_gain,
            interval_s=settings.measurement_interval_s,
            seed=int(
                np.random.SeedSequence([settings.noise_seed, day_of_year, 3]).generate_state(1)[0]
            ),
            label=f"{settings.lab_sampling_interval_s / 3600.0:g}h",
        )
    else:
        raise ValueError(f"unknown feedback level {level!r}")
    return MoistureFeedback(feedback, table_map, config)
