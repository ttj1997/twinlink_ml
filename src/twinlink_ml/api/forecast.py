"""Forecast service: history and planned inputs to the predicted quantities over the horizon.

The service wraps a model bundle (:mod:`twinlink_ml.api.bundle`): in the architecture comparison
(Befund 94) the per-target GRU composite is the recommended default, and the linear ARX is the
fallback that needs no PyTorch. What the service adds to ``SequenceModel.predict`` is only the
boundary: engineering units in, engineering units out, with the checks a model cannot make itself.

What is rejected (never repaired silently)
------------------------------------------
* a sample time that is not the one of the model (the service does not resample),
* a history shorter than the window of the model,
* a horizon longer than the one the model was trained for,
* an observation mode that is not the one of the model, or a missing measured history of a
  quantity the model reads.

What is only remarked (a notice in ``meta``)
--------------------------------------------
* an input outside the range of the training data (the model then extrapolates),
* a measured history the model does not read,
* the use of the fallback model.

Fallback
--------
The fallback is used when the primary model cannot be loaded because its extra is not installed,
or when its prediction contains a non-finite value. It is never used to hide a rejected request.

Equality with the research code
-------------------------------
The window is built with the same arithmetic as ``TrajectoryWindowDataset`` (standardise with the
scalers of the training data set, then set the history of an unmeasured quantity to zero), so the
response equals ``target_scaler.inverse(model.predict(batch))`` bit for bit (test).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np

from ..data.windows import WindowBatch
from ..models.base import SequenceModel
from ..simulation.backend import FloatArray
from .base import (
    API_VERSION,
    SCHEMA_VERSION,
    MissingExtraError,
    ModelFailureError,
    Notice,
    RequestRejectedError,
    ResponseMeta,
    Service,
    ValueRange,
)
from .bundle import BundleManifest, ModelBundle, instantiate_model
from .schemas import (
    EXOGENOUS_FIELDS,
    TARGET_FIELD_OF_MODEL_NAME,
    ForecastDescription,
    ForecastRequest,
    ForecastResponse,
    ObservationMode,
    ProcessInputSeries,
)

__all__ = ["ForecastService", "ModelLoader"]

ModelLoader = Callable[[ModelBundle], SequenceModel]
"""Turns a bundle into a fitted model; injected so that a host can supply its own loader."""

_SAMPLE_TIME_TOLERANCE_S = 1e-9


def _observation_mode(manifest: BundleManifest) -> ObservationMode:
    return "soft_sensor" if manifest.unobserved_target_names else "inline"


def _check_schema_compatible(manifest: BundleManifest) -> None:
    """A bundle must have exactly the channels of schema version 1."""
    if set(manifest.exogenous_names) != set(EXOGENOUS_FIELDS) or len(
        manifest.exogenous_names
    ) != len(EXOGENOUS_FIELDS):
        raise ValueError(
            "the inputs of the bundle are not those of schema version "
            f"{SCHEMA_VERSION}: {manifest.exogenous_names}"
        )
    if manifest.measured_input_names:
        raise ValueError(
            "a bundle with measured process responses as inputs needs a later schema version: "
            f"{manifest.measured_input_names}"
        )
    unknown = set(manifest.target_names) - set(TARGET_FIELD_OF_MODEL_NAME)
    if unknown:
        raise ValueError(
            f"schema version {SCHEMA_VERSION} has no field for the targets {sorted(unknown)}"
        )


def _interface(manifest: BundleManifest) -> tuple[object, ...]:
    """What a fallback must share with the primary model to answer the same requests."""
    return (
        manifest.exogenous_names,
        manifest.target_names,
        manifest.observed_target_names,
        manifest.sample_time_s,
        manifest.window.horizon,
    )


def _matrix(series: ProcessInputSeries, names: tuple[str, ...]) -> FloatArray:
    """``(n_samples, n_channels)`` in the channel order of the model."""
    return np.stack([np.asarray(getattr(series, name), dtype=np.float64) for name in names], -1)


class ForecastService(Service[ForecastRequest, ForecastResponse]):
    """Multi-step forecast of energy target, bed temperature and product moisture.

    Parameters
    ----------
    bundle
        The primary model.
    fallback
        Optional second model with the same channels, sample time and horizon.
    model_loader
        How a bundle becomes a model (default :func:`~twinlink_ml.api.bundle.instantiate_model`).
    """

    name: ClassVar[str] = "forecast"
    request_model = ForecastRequest
    response_model = ForecastResponse
    description_model = ForecastDescription

    def __init__(
        self,
        bundle: ModelBundle,
        fallback: ModelBundle | None = None,
        *,
        model_loader: ModelLoader = instantiate_model,
    ) -> None:
        _check_schema_compatible(bundle.manifest)
        self._bundle = bundle
        self._fallback_bundle = fallback
        self._fallback_model: SequenceModel | None = None
        if fallback is not None:
            _check_schema_compatible(fallback.manifest)
            if _interface(fallback.manifest) != _interface(bundle.manifest):
                raise ValueError(
                    "the fallback must share inputs, targets, measured targets, sample time and "
                    "horizon with the primary model"
                )
            self._fallback_model = model_loader(fallback)
        self._model: SequenceModel | None
        self._missing_extra: MissingExtraError | None = None
        try:
            self._model = model_loader(bundle)
        except MissingExtraError as error:
            if fallback is None:
                raise
            self._model = None
            self._missing_extra = error

    # ------------------------------------------------------------------ interface

    def describe(self) -> ForecastDescription:
        """Version, fingerprints, window and the measured histories the model needs."""
        manifest = self._bundle.manifest
        fallback = None if self._fallback_bundle is None else self._fallback_bundle.manifest
        return ForecastDescription(
            service=self.name,
            api_version=API_VERSION,
            schema_version=SCHEMA_VERSION,
            model_fingerprint=manifest.fingerprint,
            model_label=manifest.model_label,
            model_scope=manifest.model_scope,
            parameter_status=manifest.parameter_status,
            required_extras=("api", "torch")
            if manifest.model_kind == "composite_torch"
            else ("api",),
            valid_input_ranges={
                TARGET_FIELD_OF_MODEL_NAME.get(name, name): value
                for name, value in manifest.input_ranges.items()
            },
            sample_time_s=manifest.sample_time_s,
            history_length=manifest.window.history_length,
            maximum_horizon=manifest.window.horizon,
            observation_mode=_observation_mode(manifest),
            predicted_fields=tuple(TARGET_FIELD_OF_MODEL_NAME[n] for n in manifest.target_names),
            required_measured_fields=tuple(
                TARGET_FIELD_OF_MODEL_NAME[n] for n in manifest.observed_target_names
            ),
            fallback_model_label=None if fallback is None else fallback.model_label,
            fallback_model_fingerprint=None if fallback is None else fallback.fingerprint,
        )

    def handle(self, request: ForecastRequest) -> ForecastResponse:
        """Check the request against the model, predict, return engineering units."""
        manifest = self._bundle.manifest
        notices: list[Notice] = []
        self._check_grid_and_mode(request, manifest)
        measured = self._measured_history(request, manifest, notices)
        notices.extend(self._range_notices(request, measured, manifest))

        history_length = manifest.window.history_length
        horizon = request.planned.n_samples
        history = _matrix(request.history, manifest.exogenous_names)[-history_length:]
        planned = _matrix(request.planned, manifest.exogenous_names)

        prediction: FloatArray | None = None
        reason = ""
        used = self._bundle
        if self._model is not None:
            prediction = self._predict(self._model, manifest, history, measured, planned)
            if not np.all(np.isfinite(prediction)):
                prediction = None
                reason = "the primary model returned a non-finite value"
        else:
            reason = str(self._missing_extra)
        if prediction is None:
            if self._fallback_model is None or self._fallback_bundle is None:
                raise ModelFailureError(reason)
            used = self._fallback_bundle
            prediction = self._predict(
                self._fallback_model, used.manifest, history, measured, planned
            )
            if not np.all(np.isfinite(prediction)):
                raise ModelFailureError("the fallback model returned a non-finite value")
            notices.append(
                Notice(
                    code="fallback_model_used",
                    message=f"{used.manifest.model_label} answered instead of "
                    f"{manifest.model_label}: {reason}",
                )
            )

        fields = {
            TARGET_FIELD_OF_MODEL_NAME[name]: tuple(prediction[:, column].tolist())
            for column, name in enumerate(used.manifest.target_names)
        }
        sample_time = manifest.sample_time_s
        return ForecastResponse(
            meta=ResponseMeta(
                schema_version=SCHEMA_VERSION,
                service=self.name,
                api_version=API_VERSION,
                model_fingerprint=used.manifest.fingerprint,
                model_label=used.manifest.model_label,
                model_scope=used.manifest.model_scope,
                parameter_status=used.manifest.parameter_status,
                notices=tuple(notices),
            ),
            fallback_used=used is not self._bundle,
            sample_time_s=sample_time,
            time_offset_s=tuple(sample_time * (step + 1) for step in range(horizon)),
            **fields,
        )

    # ------------------------------------------------------------------- internal

    @staticmethod
    def _check_grid_and_mode(request: ForecastRequest, manifest: BundleManifest) -> None:
        if abs(request.sample_time_s - manifest.sample_time_s) > _SAMPLE_TIME_TOLERANCE_S:
            raise RequestRejectedError(
                f"the model works on a grid of {manifest.sample_time_s} s, the request is on "
                f"{request.sample_time_s} s; resample before the call"
            )
        mode = _observation_mode(manifest)
        if request.observation_mode != mode:
            raise RequestRejectedError(
                f"the model is a '{mode}' model, the request says '{request.observation_mode}'"
            )
        window = manifest.window
        if request.history.n_samples < window.history_length:
            raise RequestRejectedError(
                f"the model needs a history of {window.history_length} samples, got "
                f"{request.history.n_samples}"
            )
        if request.planned.n_samples > window.horizon:
            raise RequestRejectedError(
                f"the model was trained for a horizon of {window.horizon} samples, the request "
                f"plans {request.planned.n_samples}"
            )

    @staticmethod
    def _measured_history(
        request: ForecastRequest, manifest: BundleManifest, notices: list[Notice]
    ) -> dict[str, FloatArray]:
        """Measured history per observed target, cut to the window; checks presence and length."""
        observed = set(manifest.observed_target_names)
        history_length = manifest.window.history_length
        measured: dict[str, FloatArray] = {}
        ignored: list[str] = []
        for model_name, field in TARGET_FIELD_OF_MODEL_NAME.items():
            series = getattr(request.measured_history, field)
            if model_name in observed:
                if series is None:
                    raise RequestRejectedError(
                        f"the model reads the measured history of '{field}', which is missing"
                    )
                if len(series) != request.history.n_samples:
                    raise RequestRejectedError(
                        f"'{field}' has {len(series)} samples, the history has "
                        f"{request.history.n_samples}"
                    )
                measured[model_name] = np.asarray(series, dtype=np.float64)[-history_length:]
            elif series is not None:
                ignored.append(field)
        if ignored:
            notices.append(
                Notice(
                    code="measurement_ignored",
                    message="the model does not read the measured history of: "
                    + ", ".join(ignored),
                )
            )
        return measured

    @staticmethod
    def _range_notices(
        request: ForecastRequest, measured: dict[str, FloatArray], manifest: BundleManifest
    ) -> list[Notice]:
        """One notice naming every input that leaves the range of the training data."""
        outside: list[str] = []

        def check(label: str, values: FloatArray, allowed: ValueRange | None) -> None:
            if allowed is None:
                return
            if float(values.min()) < allowed.minimum or float(values.max()) > allowed.maximum:
                outside.append(
                    f"{label} [{float(values.min())!r}, {float(values.max())!r}] outside "
                    f"[{allowed.minimum!r}, {allowed.maximum!r}]"
                )

        ranges = manifest.input_ranges
        history_length = manifest.window.history_length
        for name in manifest.exogenous_names:
            history = np.asarray(getattr(request.history, name), dtype=np.float64)
            check(f"history.{name}", history[-history_length:], ranges.get(name))
            planned = np.asarray(getattr(request.planned, name), dtype=np.float64)
            check(f"planned.{name}", planned, ranges.get(name))
        for name, values in measured.items():
            check(f"measured_history.{TARGET_FIELD_OF_MODEL_NAME[name]}", values, ranges.get(name))
        if not outside:
            return []
        return [
            Notice(
                code="input_outside_training_range",
                message="the model extrapolates: " + "; ".join(outside),
            )
        ]

    @staticmethod
    def _predict(
        model: SequenceModel,
        manifest: BundleManifest,
        history: FloatArray,
        measured: dict[str, FloatArray],
        planned: FloatArray,
    ) -> FloatArray:
        """One window through the model: ``(horizon, n_targets)`` in engineering units."""
        input_scaler = manifest.input_scaler.to_standardizer()
        target_scaler = manifest.target_scaler.to_standardizer()
        n_exogenous = len(manifest.exogenous_names)
        # The same arithmetic as TrajectoryWindowDataset.from_trajectory: concatenate inputs and
        # targets, standardise, then set the history of an unmeasured target to zero (the
        # standardised training mean). The placeholder zeros below never reach the model.
        targets = np.stack(
            [
                measured.get(name, np.zeros(history.shape[0], dtype=np.float64))
                for name in manifest.target_names
            ],
            axis=-1,
        )
        inputs = np.concatenate([history, targets], axis=-1)[None, ...]
        standardised = input_scaler.transform(inputs)
        masked = [
            n_exogenous + index
            for index, name in enumerate(manifest.target_names)
            if name not in measured
        ]
        if masked:
            standardised[..., masked] = 0.0
        future = input_scaler.select(range(n_exogenous)).transform(planned[None, ...])
        batch = WindowBatch(
            history_inputs=standardised,
            future_exogenous=future,
            future_targets=np.zeros((1, planned.shape[0], len(manifest.target_names))),
            run_index=np.zeros(1),
            window_start=np.zeros(1),
        )
        return np.asarray(target_scaler.inverse(model.predict(batch))[0], dtype=np.float64)
