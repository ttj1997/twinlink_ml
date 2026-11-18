"""Request and response schemas of the services (schema version 1).

Conventions
-----------
* **Units are part of the field name** (``_celsius``, ``_k``, ``_m_per_s``, ``_kg_per_h``,
  ``_bar``, ``_kg_per_kg``, ``_kw``, ``_s``). The two dimensionless quantities
  (``ambient_relative_humidity``, ``suspension_solid_fraction``) are fractions between 0 and 1;
  they keep the names of the process configuration.
* **No plant, partner or channel names.** The fields are roles of the process model. Which plant
  channel feeds a role is decided outside this package, in a role file of the kind of
  ``conf/plant/plant_local.yaml.example``.
* **Time series** are ordered oldest sample first, on a regular grid with the sample time given
  in the request. A service rejects a grid that is not the one of its model; it never resamples.
* Every request carries ``schema_version``; every response carries ``meta``
  (:class:`~twinlink_ml.api.base.ResponseMeta`).

Schema version 1 covers the interface of the laboratory line of the grey-box model: four
manipulated variables, three disturbances, three predicted quantities. A model with other channels
(for example a soft sensor with measured process responses as inputs, ``docs/29``) needs a new
schema version and is rejected by the services of this version.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import ApiModel, ResponseMeta, ServiceDescription, VersionedRequest

__all__ = [
    "EXOGENOUS_FIELDS",
    "TARGET_FIELD_OF_MODEL_NAME",
    "AppliedLimits",
    "ForecastDescription",
    "ForecastRequest",
    "ForecastResponse",
    "MeasuredTargetSeries",
    "ObservationMode",
    "ProcessInputSeries",
    "SetpointDescription",
    "SetpointRequest",
    "SetpointResponse",
    "TimedSetpoint",
]

ObservationMode = Literal["inline", "soft_sensor"]
"""``inline``: every predicted quantity is measured and its history is part of the request.
``soft_sensor``: at least one predicted quantity is not measured (the product moisture at a plant
without an inline sensor); the model infers it from the inputs alone."""

EXOGENOUS_FIELDS: tuple[str, ...] = (
    "inlet_temperature_celsius",
    "gas_velocity_m_per_s",
    "spray_rate_kg_per_h",
    "nozzle_pressure_bar",
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
    "suspension_solid_fraction",
)
"""Manipulated variables and disturbances of schema version 1; equal to the channel names of the
process configuration (``ProcessConfig.actions`` and ``ProcessConfig.disturbances``)."""

TARGET_FIELD_OF_MODEL_NAME: dict[str, str] = {
    "energy_target": "energy_target_kw",
    "bed_temperature_celsius": "bed_temperature_celsius",
    "product_moisture_kg_per_kg": "product_moisture_kg_per_kg",
}
"""Schema field of each predicted quantity. Only the energy target is renamed: the model calls it
``energy_target`` without a unit; it is the instantaneous total power in kW."""

_Series = tuple[float, ...]


# ---------------------------------------------------------------------- forecast


class ProcessInputSeries(ApiModel):
    """Manipulated variables and disturbances on the model grid, oldest sample first."""

    inlet_temperature_celsius: _Series
    gas_velocity_m_per_s: _Series
    spray_rate_kg_per_h: _Series
    nozzle_pressure_bar: _Series
    ambient_temperature_celsius: _Series
    ambient_relative_humidity: _Series = Field(
        description="Relative humidity of the ambient air as a fraction between 0 and 1."
    )
    suspension_solid_fraction: _Series = Field(
        description="Solid mass fraction of the sprayed suspension, between 0 and 1."
    )

    @model_validator(mode="after")
    def _equal_lengths(self) -> ProcessInputSeries:
        lengths = {name: len(getattr(self, name)) for name in EXOGENOUS_FIELDS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"all series must have the same length, got {lengths}")
        if next(iter(lengths.values())) < 1:
            raise ValueError("series must not be empty")
        return self

    @property
    def n_samples(self) -> int:
        """Number of samples of every series."""
        return len(self.inlet_temperature_celsius)


class MeasuredTargetSeries(ApiModel):
    """Measured history of the predicted quantities, same grid and length as ``history``.

    A quantity that is not measured is left out (``null``). Which quantities a model needs is
    stated by :class:`ForecastDescription`.
    """

    energy_target_kw: _Series | None = None
    bed_temperature_celsius: _Series | None = None
    product_moisture_kg_per_kg: _Series | None = None


class ForecastRequest(VersionedRequest):
    """History and planned inputs for a forecast over the horizon.

    ``history`` holds the realised manipulated variables and the measured disturbances up to now;
    ``planned`` the set points as planned and the disturbances as expected by the caller from the
    next sample on. Its length is the requested horizon, at most the horizon the model was
    trained for.
    """

    sample_time_s: float = Field(gt=0.0)
    observation_mode: ObservationMode
    history: ProcessInputSeries
    measured_history: MeasuredTargetSeries = MeasuredTargetSeries()
    planned: ProcessInputSeries


class ForecastResponse(ApiModel):
    """Predicted quantities over the horizon; ``null`` for a quantity the model does not predict.

    ``time_offset_s`` counts from the last sample of the history; the first prediction is one
    sample time later.
    """

    meta: ResponseMeta
    fallback_used: bool
    sample_time_s: float
    time_offset_s: _Series
    energy_target_kw: _Series | None = None
    bed_temperature_celsius: _Series | None = None
    product_moisture_kg_per_kg: _Series | None = None


class ForecastDescription(ServiceDescription):
    """Self-description of the forecast service."""

    sample_time_s: float
    history_length: int
    maximum_horizon: int
    observation_mode: ObservationMode
    predicted_fields: tuple[str, ...]
    required_measured_fields: tuple[str, ...]
    fallback_model_label: str | None = None
    fallback_model_fingerprint: str | None = None


# ---------------------------------------------------------------------- setpoint


class SetpointRequest(VersionedRequest):
    """Disturbances, limits and margin for one steady set-point recommendation.

    The optimiser minimises the total power of the line while the steady product moisture stays
    at or below ``product_moisture_limit_kg_per_kg - moisture_margin_kg_per_kg``. The spray rate
    is the throughput and is given, not decided. The optional bounds may only narrow the box of
    the process configuration.
    """

    ambient_temperature_celsius: float
    ambient_relative_humidity: float = Field(
        ge=0.0, le=1.0, description="Relative humidity as a fraction between 0 and 1."
    )
    suspension_solid_fraction: float = Field(gt=0.0, lt=1.0)
    spray_rate_kg_per_h: float = Field(gt=0.0)
    inlet_temperature_offset_k: float = Field(
        default=0.0,
        description="Known additive offset between the realised and the commanded inlet "
        "temperature; 0 when unknown.",
    )
    product_moisture_limit_kg_per_kg: float = Field(gt=0.0)
    moisture_margin_kg_per_kg: float = Field(ge=0.0)
    inlet_temperature_minimum_celsius: float | None = None
    inlet_temperature_maximum_celsius: float | None = None
    gas_velocity_minimum_m_per_s: float | None = None
    gas_velocity_maximum_m_per_s: float | None = None
    bed_temperature_limit_celsius: float | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SetpointRequest:
        if self.moisture_margin_kg_per_kg >= self.product_moisture_limit_kg_per_kg:
            raise ValueError("moisture_margin_kg_per_kg must be smaller than the moisture limit")
        pairs = (
            (self.inlet_temperature_minimum_celsius, self.inlet_temperature_maximum_celsius),
            (self.gas_velocity_minimum_m_per_s, self.gas_velocity_maximum_m_per_s),
        )
        for low, high in pairs:
            if low is not None and high is not None and low > high:
                raise ValueError("a minimum bound exceeds its maximum bound")
        return self


class TimedSetpoint(ApiModel):
    """Set points valid from ``time_offset_s`` after the request.

    ``spray_rate_kg_per_h`` is ``null`` when the service does not decide the throughput.
    """

    time_offset_s: float = Field(ge=0.0)
    inlet_temperature_celsius: float
    gas_velocity_m_per_s: float
    spray_rate_kg_per_h: float | None = None


class AppliedLimits(ApiModel):
    """The bounds the optimiser actually used (process configuration narrowed by the request)."""

    inlet_temperature_minimum_celsius: float
    inlet_temperature_maximum_celsius: float
    gas_velocity_minimum_m_per_s: float
    gas_velocity_maximum_m_per_s: float
    bed_temperature_limit_celsius: float
    moisture_bound_kg_per_kg: float = Field(
        description="Limit minus margin: the moisture the optimiser plans for."
    )


class SetpointResponse(ApiModel):
    """Recommended set points as a time series and the steady state the model expects for them.

    The quasi-stationary optimiser returns one entry (``time_offset_s = 0``); the list form leaves
    room for a set-point trajectory without a change of the schema.
    """

    meta: ResponseMeta
    setpoints: tuple[TimedSetpoint, ...]
    applied_limits: AppliedLimits
    expected_bed_temperature_celsius: float
    expected_product_moisture_kg_per_kg: float
    expected_total_power_kw: float


class SetpointDescription(ServiceDescription):
    """Self-description of the set-point service."""

    decided_fields: tuple[str, ...]
    fixed_nozzle_pressure_bar: float
    default_bed_temperature_limit_celsius: float
    setpoints_per_response: int
