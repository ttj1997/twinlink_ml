"""Calibration service: stationary sections to a bias per measured quantity, optionally parameters.

The service wraps the calibration of the plant-near model (``twinlink_ml.plant_calibration``,
Befund 123, on the model of Befund 117). The model is evaluated at the measured operating points
of the sections; what is compared are the two temperatures it predicts, bed and exhaust.

Bias first
----------
The default answer is the **constant bias** per measured quantity (mean of measured minus model
over the sections) of the uncalibrated model. That is the reading of the studies, not caution:
with few points a bias is a hard baseline (Befund 120), and in Befund 123 a stationary calibration
was no better on the other day than the bias, because per day only two directions of the
parameter space are determinable. Parameters are therefore calibrated only when the request names
them, never the kinetics or the heat capacity (the configuration marks those as not calibratable),
at most as many as the configuration allows, and every such answer carries a notice.

Which model
-----------
The plant-near model of plant A, not the laboratory line of the forecast and set-point services.
A calibration result belongs to the model it was computed with: its bias must not be applied to
the forecasts or set points of another model. Those two services therefore do not take a
calibration result as input; a service on the plant-near model that does would be a further step
(``docs/NN_module_api.md``, section 9).

Equality with the research code
-------------------------------
The service calls ``evaluate_rows``, ``bias_offsets``, ``apply_bias``, ``error_metrics`` and
``fit_parameters`` in the order of ``scripts/run_plant_calibration.py`` (one dictionary of
velocity guesses per fit, a fresh one for the prediction at the result), so its numbers equal a
direct call bit for bit (test). The research modules are imported when the service is used, not
when the package is imported.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from .base import (
    API_VERSION,
    PLANT_A_SCOPE,
    SCHEMA_VERSION,
    NoSolutionError,
    Notice,
    RequestRejectedError,
    ResponseMeta,
    Service,
    ValueRange,
)
from .schemas import (
    CalibratedParameters,
    CalibrationDescription,
    CalibrationRequest,
    CalibrationResponse,
    TemperatureBias,
    TemperatureErrors,
)

if TYPE_CHECKING:
    from ..plant_calibration.config import ParameterSpec, PlantCalibrationConfig
    from ..plant_calibration.model import ModelContext
    from ..plant_scale.model import PlantPoint

__all__ = ["CalibrationService"]

_CALIBRATABLE_KIND = "steady"
_MODEL_LABEL = "plant_near_steady_model"


def _errors(metrics: Mapping[str, Any]) -> TemperatureErrors:
    """The numbers of ``plant_calibration.error_metrics`` as a schema object."""
    return TemperatureErrors(
        bed_temperature_rms_k=metrics["bed"]["rms_k"],
        bed_temperature_maximum_k=metrics["bed"]["max_abs_k"],
        exhaust_temperature_rms_k=metrics["exhaust"]["rms_k"],
        exhaust_temperature_maximum_k=metrics["exhaust"]["max_abs_k"],
    )


class CalibrationService(Service[CalibrationRequest, CalibrationResponse]):
    """Bias and optional parameter calibration of the plant-near steady model.

    Parameters
    ----------
    context
        Process configuration, solid fraction and solver settings of the plant-near model.
    config
        Candidate parameters with scales, bounds and start values, sensor sigma, product roles.
    """

    name: ClassVar[str] = "calibration"
    request_model = CalibrationRequest
    response_model = CalibrationResponse
    description_model = CalibrationDescription

    def __init__(
        self,
        context: ModelContext,
        config: PlantCalibrationConfig,
        *,
        model_scope: str = PLANT_A_SCOPE,
    ) -> None:
        self._context = context
        self._config = config
        self._scope = model_scope

    @classmethod
    def from_config(cls, path: Path | str | None = None) -> CalibrationService:
        """The plant-near model of the calibration study (``conf/plant_calibration``)."""
        from ..plant_calibration.config import load_plant_calibration_config
        from ..plant_calibration.model import ModelContext
        from ..plant_scale.config import load_plant_scale_config
        from ..simulation import PlaceholderConfigWarning, load_process_config

        config = (
            load_plant_calibration_config() if path is None else load_plant_calibration_config(path)
        )
        plant = load_plant_scale_config(config.plant_scale_config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            base = load_process_config(plant.process_config)
        steady_kwargs = {
            "normal_density_kg_per_m3": plant.data.normal_density_dry_air_kg_per_m3,
            "inlet_humidity_kg_per_kg": plant.placeholders.inlet_absolute_humidity_kg_per_kg,
            "feed_celsius": plant.placeholders.feed_temperature_celsius,
            "iterations": plant.calibration.velocity_iterations,
            "relative_tolerance": plant.calibration.velocity_relative_tolerance,
        }
        context = ModelContext(base, plant.placeholders.solid_fraction_reference, steady_kwargs)
        return cls(context, config)

    # ------------------------------------------------------------------ interface

    @property
    def products(self) -> tuple[str, ...]:
        """Product roles the model distinguishes."""
        return tuple(sorted(set(self._config.products.values())))

    @property
    def calibratable(self) -> tuple[ParameterSpec, ...]:
        """Parameters the configuration allows to calibrate on stationary sections."""
        return tuple(p for p in self._config.parameters if p.kind == _CALIBRATABLE_KIND)

    @property
    def model_fingerprint(self) -> str:
        """Fingerprint of the process configuration and of the calibration settings."""
        config, context = self._config, self._context
        settings = {
            "solid_fraction": context.solid_fraction,
            "steady": {key: context.steady_kwargs[key] for key in sorted(context.steady_kwargs)},
            "parameters": [
                [p.name, p.nominal, p.scale, list(p.bounds), p.kind, p.product]
                for p in config.parameters
            ],
            "sigma": [config.sigma_bed_k, config.sigma_exhaust_k],
            "start": {
                key: config.calibration.start[key] for key in sorted(config.calibration.start)
            },
            "max_nfev": config.calibration.max_nfev,
            "max_parameters": config.sensitivity.max_parameters,
        }
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
        return f"process:{context.base.fingerprint}/calibration:{digest[:16]}"

    def describe(self) -> CalibrationDescription:
        """Version, fingerprint, product roles and the parameters that can be calibrated."""
        calibratable = self.calibratable
        return CalibrationDescription(
            service=self.name,
            api_version=API_VERSION,
            schema_version=SCHEMA_VERSION,
            model_fingerprint=self.model_fingerprint,
            model_label=_MODEL_LABEL,
            model_scope=self._scope,
            parameter_status=self._context.base.provenance.status,
            required_extras=("api",),
            valid_input_ranges={
                p.name: ValueRange(minimum=p.bounds[0], maximum=p.bounds[1]) for p in calibratable
            },
            products=self.products,
            calibratable_parameters=tuple(p.name for p in calibratable),
            maximum_parameters=self._config.sensitivity.max_parameters,
            bed_temperature_sigma_k=self._config.sigma_bed_k,
            exhaust_temperature_sigma_k=self._config.sigma_exhaust_k,
        )

    def handle(self, request: CalibrationRequest) -> CalibrationResponse:
        """Bias of the uncalibrated model on the sections; the fit when parameters are named."""
        from ..plant_calibration.calibration import (
            apply_bias,
            bias_offsets,
            error_metrics,
            fit_parameters,
        )
        from ..plant_calibration.model import evaluate_rows

        config, context = self._config, self._context
        rows = self._rows(request)
        free = self._checked_parameters(request)
        measured = np.array(
            [
                value
                for section in request.sections
                for value in (section.bed_temperature_celsius, section.exhaust_temperature_celsius)
            ],
            dtype=np.float64,
        )
        nominal = config.nominal()
        uncalibrated = evaluate_rows(context, rows, nominal, {})
        failed = [i for i in range(len(rows)) if not np.isfinite(uncalibrated[2 * i])]
        if failed:
            raise NoSolutionError(
                f"the model has no steady state at the sections {failed} (counted from 0)"
            )
        bias = bias_offsets(uncalibrated, measured)
        with_bias = apply_bias(uncalibrated, bias)

        notices: list[Notice] = []
        if len(rows) == 1:
            notices.append(
                Notice(
                    code="bias_from_one_section",
                    message="the bias equals the residual of the single section; its error with "
                    "bias is zero by construction",
                )
            )
        parameters: CalibratedParameters | None = None
        if free:
            sigma = np.array(
                [config.sigma_bed_k, config.sigma_exhaust_k] * len(rows), dtype=np.float64
            )
            guesses: dict[str, float] = {}
            fit = fit_parameters(
                lambda theta: evaluate_rows(context, rows, theta, guesses),
                measured,
                sigma,
                nominal,
                free,
                scales={p.name: p.scale for p in config.parameters},
                bounds={p.name: p.bounds for p in config.parameters},
                start={
                    name: config.calibration.start.get(name, config.spec(name).nominal)
                    for name in free
                },
                max_nfev=config.calibration.max_nfev,
            )
            calibrated = evaluate_rows(context, rows, fit.theta, {})
            parameters = CalibratedParameters(
                values={name: fit.theta[name] for name in free},
                at_bound=fit.at_bound,
                weighted_squared_residual_sum=fit.ssr,
                n_residuals=fit.n_residuals,
                evaluations=fit.evaluations,
                solver_message=fit.message,
                errors=_errors(error_metrics(calibrated, measured)),
            )
            notices.append(
                Notice(
                    code="calibration_not_validated",
                    message="parameters fitted on stationary sections describe these sections; in "
                    "the calibration study they were no better on another day than the constant "
                    "bias. Check them on sections that were not used before relying on them",
                )
            )
            if fit.at_bound:
                notices.append(
                    Notice(
                        code="parameter_at_bound",
                        message="at a bound of its range: " + ", ".join(fit.at_bound),
                    )
                )

        temperature_bias = TemperatureBias(
            bed_temperature_bias_k=bias["bed"], exhaust_temperature_bias_k=bias["exhaust"]
        )
        return CalibrationResponse(
            meta=ResponseMeta(
                schema_version=SCHEMA_VERSION,
                service=self.name,
                api_version=API_VERSION,
                model_fingerprint=self.model_fingerprint,
                model_label=_MODEL_LABEL,
                model_scope=self._scope,
                parameter_status=context.base.provenance.status,
                notices=tuple(notices),
            ),
            n_sections=len(rows),
            model_bed_temperature_celsius=tuple(uncalibrated[0::2].tolist()),
            model_exhaust_temperature_celsius=tuple(uncalibrated[1::2].tolist()),
            bias=temperature_bias,
            errors_uncalibrated=_errors(error_metrics(uncalibrated, measured)),
            errors_with_bias=_errors(error_metrics(with_bias, measured)),
            parameters=parameters,
            calibration_fingerprint=self._result_fingerprint(temperature_bias, parameters),
        )

    # ------------------------------------------------------------------- internal

    def _rows(self, request: CalibrationRequest) -> list[tuple[PlantPoint, str]]:
        """Operating points of the research code; the label is the position of the section."""
        from ..plant_scale.model import PlantPoint

        known = self.products
        rows: list[tuple[PlantPoint, str]] = []
        for index, section in enumerate(request.sections):
            if section.product not in known:
                raise RequestRejectedError(
                    f"section {index}: unknown product role {section.product!r}, known are "
                    f"{list(known)}"
                )
            point = PlantPoint(
                label=f"section_{index}",
                ambient_celsius=section.ambient_temperature_celsius,
                process_air_nm3h=section.process_air_flow_nm3_per_h,
                atomising_air_nm3h=section.atomising_air_flow_nm3_per_h,
                supply_celsius=section.supply_air_temperature_celsius,
                spray_kg_h=section.spray_rate_kg_per_h,
                bed_celsius=section.bed_temperature_celsius,
                exhaust_celsius=section.exhaust_temperature_celsius,
            )
            rows.append((point, section.product))
        return rows

    def _checked_parameters(self, request: CalibrationRequest) -> tuple[str, ...]:
        """The requested parameters, or a rejection that says why they cannot be fitted."""
        free = request.calibrate_parameters
        if not free:
            return ()
        allowed = {p.name: p for p in self.calibratable}
        unknown = [name for name in free if name not in allowed]
        if unknown:
            raise RequestRejectedError(
                f"not calibratable on stationary sections: {unknown}; allowed are {sorted(allowed)}"
            )
        limit = self._config.sensitivity.max_parameters
        if len(free) > limit:
            raise RequestRejectedError(f"at most {limit} parameters can be calibrated at once")
        n_residuals = 2 * len(request.sections)
        if len(free) > n_residuals:
            raise RequestRejectedError(
                f"{len(free)} parameters cannot be fitted to {n_residuals} measured values"
            )
        present = {section.product for section in request.sections}
        absent = [
            name
            for name in free
            if allowed[name].product is not None and allowed[name].product not in present
        ]
        if absent:
            raise RequestRejectedError(
                f"no section of the product of {absent}; the parameter has no effect on the data"
            )
        return free

    def _result_fingerprint(
        self, bias: TemperatureBias, parameters: CalibratedParameters | None
    ) -> str:
        payload = {
            "model": self.model_fingerprint,
            "bias": bias.model_dump(mode="json"),
            "parameters": None if parameters is None else dict(sorted(parameters.values.items())),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
