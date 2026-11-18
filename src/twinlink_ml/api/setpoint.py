"""Set-point service: disturbances, limits and margin to inlet temperature and gas velocity.

The service wraps the steady-state optimiser of the dynamic headroom study
(``twinlink_ml.headroom.controllers.SteadyStateOptimizer``, controller R1 of Befund 105): minimum
total power of the laboratory line at the current disturbance, with the steady product moisture at
or below ``limit - margin`` and the bed temperature below its limit. Spray rate (the throughput)
and nozzle pressure are not decided.

One optimiser per request
-------------------------
``SteadyStateOptimizer`` keeps the last solution as the start of the next solve. That is right for
a controller that follows a scenario and wrong for a stateless service: the answer would depend on
the requests before it (measured: differences of the order of 1e-10 in the commands). The service
therefore builds a fresh optimiser for every request, so that a response equals a direct call on a
new ``SteadyStateOptimizer`` bit for bit (test) and never depends on history. Building costs about
one second; a quasi-stationary recommendation is asked for at most every few minutes.

Bounds
------
The box of the manipulated variables and the bed-temperature limit come from the injected
configuration. A request may narrow them, never widen them: outside the box the model was neither
built nor checked.

Validity
--------
The model is the laboratory line with placeholder parameters; every response says so in
``meta.parameter_status``. A disturbance outside the ranges of the process configuration is
answered with a notice, not rejected: the optimiser extrapolates the model there.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

import numpy as np

from ..headroom.config import HeadroomStudyConfig, load_headroom_config, study_process_config
from ..simulation.config import ActionBound, ProcessConfig
from .base import (
    API_VERSION,
    SCHEMA_VERSION,
    NoSolutionError,
    Notice,
    RequestRejectedError,
    ResponseMeta,
    Service,
    ValueRange,
    require_extra,
)
from .schemas import (
    AppliedLimits,
    SetpointDescription,
    SetpointRequest,
    SetpointResponse,
    TimedSetpoint,
)

if TYPE_CHECKING:
    from ..headroom.controllers import SteadyPoint

__all__ = ["OptimizerFactory", "SetpointService", "SteadyOptimizer"]

DISTURBANCE_FIELDS: tuple[str, ...] = (
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
    "suspension_solid_fraction",
    "spray_rate_kg_per_h",
    "inlet_temperature_offset_k",
)
"""Order of the disturbance vector of the optimiser
(``twinlink_ml.headroom.symbolic.DISTURBANCE_CHANNELS``; checked against it at the first solve)."""


class SteadyOptimizer(Protocol):
    """What the service needs from an optimiser: one solve."""

    def solve(self, disturbance: np.ndarray, moisture_bound: float) -> SteadyPoint:
        """Steady optimum for a disturbance vector and a moisture bound."""
        ...


OptimizerFactory = Callable[[HeadroomStudyConfig, ProcessConfig], SteadyOptimizer]


def _default_optimizer_factory(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> SteadyOptimizer:
    """The research optimiser. Imports CasADi, hence only here."""
    require_extra("casadi", "optimal-control")
    from ..headroom.controllers import SteadyStateOptimizer
    from ..headroom.symbolic import DISTURBANCE_CHANNELS

    if DISTURBANCE_CHANNELS != DISTURBANCE_FIELDS:
        raise RuntimeError(
            "the disturbance channels of the optimiser changed: "
            f"{DISTURBANCE_CHANNELS} against {DISTURBANCE_FIELDS}"
        )
    return SteadyStateOptimizer(study, process)


def _narrowed(bound: ActionBound, low: float | None, high: float | None, label: str) -> ActionBound:
    """``bound`` with a narrower box; a wider one is rejected."""
    minimum = bound.minimum if low is None else low
    maximum = bound.maximum if high is None else high
    if minimum < bound.minimum or maximum > bound.maximum:
        raise RequestRejectedError(
            f"{label}: the request asks for [{minimum}, {maximum}], the model is valid in "
            f"[{bound.minimum}, {bound.maximum}]; bounds may only be narrowed"
        )
    if minimum > maximum:
        raise RequestRejectedError(f"{label}: minimum {minimum} exceeds maximum {maximum}")
    if minimum == bound.minimum and maximum == bound.maximum:
        return bound
    return dataclasses.replace(
        bound, minimum=minimum, maximum=maximum, nominal=min(max(bound.nominal, minimum), maximum)
    )


_CHECKED_REQUEST_FIELDS: tuple[str, ...] = (
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
    "suspension_solid_fraction",
    "spray_rate_kg_per_h",
)


def _model_ranges(process: ProcessConfig) -> dict[str, ValueRange]:
    """Ranges of the disturbances and box of the manipulated variables of the configuration."""
    actions, disturbances = process.actions, process.disturbances
    return {
        "ambient_temperature_celsius": ValueRange(
            minimum=disturbances.ambient_temperature_celsius.minimum,
            maximum=disturbances.ambient_temperature_celsius.maximum,
        ),
        "ambient_relative_humidity": ValueRange(
            minimum=disturbances.ambient_relative_humidity.minimum,
            maximum=disturbances.ambient_relative_humidity.maximum,
        ),
        "suspension_solid_fraction": ValueRange(
            minimum=disturbances.suspension_solid_fraction.minimum,
            maximum=disturbances.suspension_solid_fraction.maximum,
        ),
        "spray_rate_kg_per_h": ValueRange(
            minimum=actions.spray_rate_kg_per_h.minimum, maximum=actions.spray_rate_kg_per_h.maximum
        ),
        "inlet_temperature_celsius": ValueRange(
            minimum=actions.inlet_temperature_celsius.minimum,
            maximum=actions.inlet_temperature_celsius.maximum,
        ),
        "gas_velocity_m_per_s": ValueRange(
            minimum=actions.gas_velocity_m_per_s.minimum,
            maximum=actions.gas_velocity_m_per_s.maximum,
        ),
    }


class SetpointService(Service[SetpointRequest, SetpointResponse]):
    """Quasi-stationary set-point recommendation for one line.

    Parameters
    ----------
    study
        Settings of the line (fixed nozzle pressure, bed-temperature limit).
    process
        Process configuration of the line.
    optimizer_factory
        Builds the optimiser for one request (default: the steady-state optimiser of the headroom
        study, which needs the extra ``optimal-control``).
    """

    name: ClassVar[str] = "setpoint"
    request_model = SetpointRequest
    response_model = SetpointResponse
    description_model = SetpointDescription

    def __init__(
        self,
        study: HeadroomStudyConfig,
        process: ProcessConfig,
        *,
        optimizer_factory: OptimizerFactory = _default_optimizer_factory,
    ) -> None:
        self._study = study
        self._process = process
        self._optimizer_factory = optimizer_factory

    @classmethod
    def from_config(
        cls,
        path: Path | str | None = None,
        *,
        optimizer_factory: OptimizerFactory = _default_optimizer_factory,
    ) -> SetpointService:
        """The laboratory line of the headroom study (``conf/headroom/dynamic_headroom.yaml``)."""
        study = load_headroom_config() if path is None else load_headroom_config(path)
        return cls(study, study_process_config(study), optimizer_factory=optimizer_factory)

    # ------------------------------------------------------------------ interface

    @property
    def model_fingerprint(self) -> str:
        """Fingerprints of the process configuration and of the line settings."""
        return f"process:{self._process.fingerprint}/study:{self._study.fingerprint}"

    def describe(self) -> SetpointDescription:
        """Version, fingerprint, box of the manipulated variables and ranges of the disturbances."""
        return SetpointDescription(
            service=self.name,
            api_version=API_VERSION,
            schema_version=SCHEMA_VERSION,
            model_fingerprint=self.model_fingerprint,
            model_label="steady_state_optimizer",
            parameter_status=self._process.provenance.status,
            required_extras=("api", "optimal-control"),
            valid_input_ranges=_model_ranges(self._process),
            decided_fields=("inlet_temperature_celsius", "gas_velocity_m_per_s"),
            fixed_nozzle_pressure_bar=self._study.line.nozzle_pressure_bar,
            default_bed_temperature_limit_celsius=self._study.line.bed_temperature_limit_celsius,
            setpoints_per_response=1,
        )

    def handle(self, request: SetpointRequest) -> SetpointResponse:
        """Steady optimum for the request, from a fresh optimiser."""
        study, process = self._effective_configuration(request)
        disturbance = np.array(
            [getattr(request, name) for name in DISTURBANCE_FIELDS], dtype=np.float64
        )
        moisture_bound = (
            request.product_moisture_limit_kg_per_kg - request.moisture_margin_kg_per_kg
        )
        optimizer = self._optimizer_factory(study, process)
        try:
            point = optimizer.solve(disturbance, moisture_bound)
        except RuntimeError as error:
            raise NoSolutionError(str(error)) from error

        inlet, gas = process.actions.inlet_temperature_celsius, process.actions.gas_velocity_m_per_s
        return SetpointResponse(
            meta=ResponseMeta(
                schema_version=SCHEMA_VERSION,
                service=self.name,
                api_version=API_VERSION,
                model_fingerprint=self.model_fingerprint,
                model_label="steady_state_optimizer",
                parameter_status=process.provenance.status,
                notices=tuple(self._range_notices(request)),
            ),
            setpoints=(
                TimedSetpoint(
                    time_offset_s=0.0,
                    inlet_temperature_celsius=float(point.command[0]),
                    gas_velocity_m_per_s=float(point.command[1]),
                ),
            ),
            applied_limits=AppliedLimits(
                inlet_temperature_minimum_celsius=inlet.minimum,
                inlet_temperature_maximum_celsius=inlet.maximum,
                gas_velocity_minimum_m_per_s=gas.minimum,
                gas_velocity_maximum_m_per_s=gas.maximum,
                bed_temperature_limit_celsius=study.line.bed_temperature_limit_celsius,
                moisture_bound_kg_per_kg=moisture_bound,
            ),
            expected_bed_temperature_celsius=float(point.state[2]),
            expected_product_moisture_kg_per_kg=float(point.state[1] / point.state[0]),
            expected_total_power_kw=float(point.power_kw),
        )

    # ------------------------------------------------------------------- internal

    def _effective_configuration(
        self, request: SetpointRequest
    ) -> tuple[HeadroomStudyConfig, ProcessConfig]:
        """Configuration with the bounds of the request; the injected objects when none is given."""
        actions = self._process.actions
        inlet = _narrowed(
            actions.inlet_temperature_celsius,
            request.inlet_temperature_minimum_celsius,
            request.inlet_temperature_maximum_celsius,
            "inlet temperature",
        )
        gas = _narrowed(
            actions.gas_velocity_m_per_s,
            request.gas_velocity_minimum_m_per_s,
            request.gas_velocity_maximum_m_per_s,
            "gas velocity",
        )
        process = self._process
        if (
            inlet is not actions.inlet_temperature_celsius
            or gas is not actions.gas_velocity_m_per_s
        ):
            process = dataclasses.replace(
                process,
                actions=dataclasses.replace(
                    actions, inlet_temperature_celsius=inlet, gas_velocity_m_per_s=gas
                ),
            )
        study = self._study
        limit = request.bed_temperature_limit_celsius
        if limit is not None and limit != study.line.bed_temperature_limit_celsius:
            if limit > study.line.bed_temperature_limit_celsius:
                raise RequestRejectedError(
                    f"bed temperature limit: the request asks for {limit} degC, the "
                    f"configuration allows {study.line.bed_temperature_limit_celsius} degC; "
                    "bounds may only be narrowed"
                )
            study = dataclasses.replace(
                study, line=dataclasses.replace(study.line, bed_temperature_limit_celsius=limit)
            )
        return study, process

    def _range_notices(self, request: SetpointRequest) -> list[Notice]:
        """One notice naming every disturbance outside the ranges of the process configuration."""
        ranges = _model_ranges(self._process)
        outside = [
            f"{name} {getattr(request, name)!r} outside "
            f"[{ranges[name].minimum!r}, {ranges[name].maximum!r}]"
            for name in _CHECKED_REQUEST_FIELDS
            if not ranges[name].minimum <= getattr(request, name) <= ranges[name].maximum
        ]
        if not outside:
            return []
        return [
            Notice(
                code="input_outside_model_range",
                message="the optimiser extrapolates the model: " + "; ".join(outside),
            )
        ]
