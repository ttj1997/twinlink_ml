"""Request and response schemas of the services (schema version 1).

Conventions
-----------
* **Units are part of the field name** (``_celsius``, ``_k``, ``_m_per_s``, ``_kg_per_h``,
  ``_nm3_per_h``, ``_bar``, ``_kg_per_kg``, ``_kw``, ``_kwh``, ``_kw_per_k``, ``_kg``, ``_s``,
  ``_per_h``). Dimensionless quantities are named as what they are: the fractions
  ``ambient_relative_humidity`` and ``suspension_solid_fraction`` keep the names of the process
  configuration; ``utilisation``, ``replaced_supply_efficiency`` and
  ``pump_electricity_fraction`` are ratios; ``weighted_squared_residual_sum`` is a sum of squared
  residuals in units of the sensor sigma.
* **No plant, partner or channel names.** The fields are roles of the process model. Which plant
  channel feeds a role is decided outside this package, in a role file of the kind of
  ``conf/plant/plant_local.yaml.example``.
* **Time series** are ordered oldest sample first, on a regular grid with the sample time given
  in the request. A service rejects a grid that is not the one of its model; it never resamples.
* Every request carries ``schema_version``; every response carries ``meta``
  (:class:`~twinlink_ml.api.base.ResponseMeta`).

Schema version 1 covers, for the forecast and the set point, the interface of the laboratory
line of the grey-box model: four manipulated variables, three disturbances, three predicted
quantities. A model with other channels (for example a soft sensor with measured process responses
as inputs, ``docs/29``) needs a new schema version and is rejected by the services of this version.
The calibration works on the roles of the plant-near steady model, the heat distribution on cold
streams on an hourly grid.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import ApiModel, ResponseMeta, ServiceDescription, VersionedRequest

__all__ = [
    "EXOGENOUS_FIELDS",
    "TARGET_FIELD_OF_MODEL_NAME",
    "AppliedLimits",
    "CalibratedParameters",
    "CalibrationDescription",
    "CalibrationRequest",
    "CalibrationResponse",
    "ForecastDescription",
    "ForecastRequest",
    "ForecastResponse",
    "HeatDistributionDescription",
    "HeatDistributionMode",
    "HeatDistributionRequest",
    "HeatDistributionResponse",
    "HeatKeyFigures",
    "HeatSink",
    "MeasuredTargetSeries",
    "ObservationMode",
    "ProcessInputSeries",
    "RecoveryCaseDescription",
    "RecoveryCaseRequest",
    "RecoveryCaseResponse",
    "SetpointDescription",
    "SetpointRequest",
    "SetpointResponse",
    "SinkHeat",
    "SteadySection",
    "StorageRule",
    "TemperatureBias",
    "TemperatureErrors",
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


# ------------------------------------------------------------------- calibration


class SteadySection(ApiModel):
    """Means of one stationary section of plant operation.

    The roles are those of the plant-near model (Befund 117): intake air, process air, atomising
    air, supply air, spray rate, and the two measured temperatures the model is compared with.
    """

    product: str = Field(description="Role of the product, as named by the service description.")
    ambient_temperature_celsius: float
    process_air_flow_nm3_per_h: float = Field(gt=0.0)
    atomising_air_flow_nm3_per_h: float = Field(ge=0.0)
    supply_air_temperature_celsius: float
    spray_rate_kg_per_h: float = Field(gt=0.0)
    bed_temperature_celsius: float
    exhaust_temperature_celsius: float


class CalibrationRequest(VersionedRequest):
    """Stationary sections, and optionally the parameters to calibrate on them.

    Without ``calibrate_parameters`` the answer is the constant bias per measured quantity, the
    default. Parameters are calibrated only when they are named.
    """

    sections: tuple[SteadySection, ...] = Field(min_length=1)
    calibrate_parameters: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _distinct_parameters(self) -> CalibrationRequest:
        if len(set(self.calibrate_parameters)) != len(self.calibrate_parameters):
            raise ValueError("calibrate_parameters contains a name twice")
        return self


class TemperatureBias(ApiModel):
    """Mean of measured minus model over the sections, per measured quantity."""

    bed_temperature_bias_k: float
    exhaust_temperature_bias_k: float


class TemperatureErrors(ApiModel):
    """Model minus measured over the sections: root mean square and largest absolute error."""

    bed_temperature_rms_k: float
    bed_temperature_maximum_k: float
    exhaust_temperature_rms_k: float
    exhaust_temperature_maximum_k: float


class CalibratedParameters(ApiModel):
    """Result of the weighted least-squares fit; the keys of ``values`` carry the units."""

    values: dict[str, float]
    at_bound: tuple[str, ...]
    weighted_squared_residual_sum: float
    n_residuals: int
    evaluations: int
    solver_message: str
    errors: TemperatureErrors


class CalibrationResponse(ApiModel):
    """Bias of the uncalibrated model, its errors, and the calibrated parameters when asked for.

    ``calibration_fingerprint`` identifies this result (model, bias and parameters); a caller that
    stores the result stores the fingerprint with it.
    """

    meta: ResponseMeta
    n_sections: int
    model_bed_temperature_celsius: _Series
    model_exhaust_temperature_celsius: _Series
    bias: TemperatureBias
    errors_uncalibrated: TemperatureErrors
    errors_with_bias: TemperatureErrors
    parameters: CalibratedParameters | None = None
    calibration_fingerprint: str


class CalibrationDescription(ServiceDescription):
    """Self-description of the calibration service."""

    products: tuple[str, ...]
    calibratable_parameters: tuple[str, ...]
    maximum_parameters: int
    bed_temperature_sigma_k: float
    exhaust_temperature_sigma_k: float


# ------------------------------------------------------------- heat distribution

HeatDistributionMode = Literal["cascade_rule", "planning"]
"""``cascade_rule``: assignment hour by hour without foresight (rule of Befund 118).
``planning``: one linear programme over the whole request with the on/off plan (Befund 119)."""

StorageRule = Literal["k2", "k2b"]
"""Discharge rule of the tank in the mode ``cascade_rule`` (Befund 118 and its addendum)."""


class HeatSink(ApiModel):
    """A consumer of network heat: a cold stream from inlet to outlet temperature, hour by hour.

    ``heat_capacity_flow_kw_per_k`` is the flow while the consumer runs; ``running`` says in which
    hours it does (default: always). ``group`` joins consumers that are parallel branches of one
    cascade stage; they need the same temperatures and the same required temperature.
    """

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=40)
    group: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=40)
    required_temperature_celsius: float
    inlet_temperature_celsius: _Series
    outlet_temperature_celsius: _Series
    heat_capacity_flow_kw_per_k: _Series
    replaced_supply_efficiency: float = Field(
        gt=0.0, le=1.0, description="Efficiency of the heat supply the network heat replaces."
    )
    running: tuple[bool, ...] | None = None

    @model_validator(mode="after")
    def _equal_lengths(self) -> HeatSink:
        lengths = {
            len(self.inlet_temperature_celsius),
            len(self.outlet_temperature_celsius),
            len(self.heat_capacity_flow_kw_per_k),
        }
        if self.running is not None:
            lengths.add(len(self.running))
        if len(lengths) != 1 or next(iter(lengths)) < 1:
            raise ValueError(f"sink {self.name!r}: all series need the same, positive length")
        if min(self.heat_capacity_flow_kw_per_k) < 0.0:
            raise ValueError(f"sink {self.name!r}: a heat-capacity flow is negative")
        return self

    @property
    def n_samples(self) -> int:
        """Number of hours of every series."""
        return len(self.inlet_temperature_celsius)


class HeatDistributionRequest(VersionedRequest):
    """One source of waste heat, the consumers and, with a tank, how it is operated.

    The grid is hourly. ``storage_capacity_kwh = 0`` means no tank; ``storage_content_kwh`` is the
    content of the tank at the beginning of the first hour (default: empty). The content after
    every hour comes back in the response, so the state of the tank stays with the caller.
    """

    mode: HeatDistributionMode = "cascade_rule"
    sample_time_s: float = Field(default=3600.0, gt=0.0)
    source_power_kw: float = Field(gt=0.0)
    storage_capacity_kwh: float = Field(default=0.0, ge=0.0)
    storage_content_kwh: float = Field(default=0.0, ge=0.0)
    storage_rule: StorageRule = "k2"
    sinks: tuple[HeatSink, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> HeatDistributionRequest:
        names = [sink.name for sink in self.sinks]
        if len(set(names)) != len(names):
            raise ValueError("sink names must be distinct")
        if len({sink.n_samples for sink in self.sinks}) != 1:
            raise ValueError("all sinks need series of the same length")
        if self.storage_content_kwh > self.storage_capacity_kwh:
            raise ValueError("storage_content_kwh exceeds storage_capacity_kwh")
        return self


class SinkHeat(ApiModel):
    """Heat delivered to one consumer, hour by hour."""

    name: str
    heat_delivered_kw: _Series


class HeatKeyFigures(ApiModel):
    """Sums over the request (the key figures of Befund 118)."""

    source_energy_kwh: float
    heat_used_kwh: float
    utilisation: float = Field(description="Heat used as a fraction of the source energy.")
    replaced_gas_kwh: float
    co2_replaced_gas_kg: float
    pump_electricity_kwh: float
    co2_pump_kg: float
    co2_net_saving_kg: float
    storage_charged_kwh: float
    storage_discharged_kwh: float
    storage_content_end_kwh: float


class HeatDistributionResponse(ApiModel):
    """Assignment (mode ``cascade_rule``) or planned operation (mode ``planning``).

    ``sink_heat`` is ``null`` in the mode ``planning``: the reduced model of the planning knows
    the heat used per hour, not its split to the consumers. ``storage_content_kwh`` has one entry
    more than the other series (content before the first and after every hour).
    """

    meta: ResponseMeta
    mode: HeatDistributionMode
    sample_time_s: float
    sink_heat: tuple[SinkHeat, ...] | None = None
    total_heat_used_kw: _Series
    storage_charge_kw: _Series
    storage_discharge_kw: _Series
    storage_content_kwh: _Series
    key_figures: HeatKeyFigures
    lp_failures: int | None = None


class HeatDistributionDescription(ServiceDescription):
    """Self-description of the heat-distribution service."""

    modes: tuple[str, ...]
    storage_rules: tuple[str, ...]
    sample_time_s: float
    source_supply_temperature_celsius: float
    source_return_floor_celsius: float
    minimum_approach_k: float
    storage_loss_per_h: float = Field(description="Share of the content lost per hour.")
    pump_electricity_fraction: float = Field(
        description="Pump electricity per unit of heat delivered."
    )


# ------------------------------------------------------------------ recovery case


class RecoveryCaseRequest(VersionedRequest):
    """Exhaust temperatures of the two dryers of the workbook arrangement, and the ambient air."""

    exhaust_1_temperature_celsius: float
    exhaust_2_temperature_celsius: float
    ambient_temperature_celsius: float


class RecoveryCaseResponse(ApiModel):
    """Recommended heat-recovery case with what the recommendation rests on.

    ``recommended_case``: 1 (each exhaust to its own sink), 2 (crossed), or ``null`` where the
    study names no case. ``case_1_expected_best`` is the statement the hit rates refer to.
    """

    meta: ResponseMeta
    recommended_case: Literal[1, 2] | None
    case_1_expected_best: bool
    exhaust_temperature_difference_k: float
    exhaust_2_usable_excess_k: float = Field(
        description="Exhaust 2 minus ambient minus the offset of the rule; positive: usable heat."
    )
    reference: str
    valid_for: str
    hit_rate_physical_unpressurised: float
    hit_rate_physical_pressurised: float
    hit_rate_basis: str


class RecoveryCaseDescription(ServiceDescription):
    """Self-description of the recovery-case rule."""

    cases: tuple[int, ...]
    difference_threshold_k: float
    usable_excess_offset_k: float
    reference: str
    valid_for: str
    hit_rate_physical_unpressurised: float
    hit_rate_physical_pressurised: float
    hit_rate_basis: str
