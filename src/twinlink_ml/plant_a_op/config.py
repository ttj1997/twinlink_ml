r"""Placeholders, search space and case definition of the operating-point study of plant A.

Question of the study (``docs/23_plant_a_operating_point.md``): how much CO2 would another
combination of process-air flow and supply-air temperature at plant A save against today's
operation, counting the steam AND the electricity of both fans?

Everything that is not a measurement is a PLACEHOLDER. :data:`PLACEHOLDERS` names each one with
its reference value, band, source and status; the values of a case enter its fingerprint
(:func:`~twinlink_ml.simulation.config.config_fingerprint`). Results are always reported as grids
over the most important placeholders, never as a single number without its labels.

Roles, not tags: nothing here names a measurement point, a company, a person or a product.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..evaluation.net_energy import ELECTRICITY_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH
from ..simulation.config import GasConfig, config_fingerprint

__all__ = [
    "AIR_FRACTION_MIN_GRID",
    "AMBIENT_SHIFTS_K",
    "BOILER_EFFICIENCY_GRID",
    "ELECTRICITY_CO2_2030_KG_PER_KWH",
    "ELECTRICITY_CO2_GRID",
    "PLACEHOLDERS",
    "SOLID_FRACTION_GRID",
    "Calibration",
    "ConstraintVariant",
    "DryerPlaceholders",
    "EnergyPlaceholders",
    "FanLawName",
    "PlaceholderSpec",
    "SearchSpace",
    "WasteHeatVariant",
]

Calibration = Literal["own_day", "other_day"]
"""Day on which the heat-loss conductance of the dryer is fitted: the product's own day
(in-sample, the baseline is reproduced exactly) or the other measured day (the check of the
transfer between days)."""

ConstraintVariant = Literal["a", "b", "a_and_b"]
"""Proxy for the product moisture (not measured): (a) exhaust temperature not below the observed
minimum of the product, (b) relative exhaust humidity not above the observed maximum, or both."""

WasteHeatVariant = Literal["fixed_water_inlet", "duty_held"]
"""How the preheater (waste heat of the storage loop) reacts to another air flow:
``fixed_water_inlet`` -- water inlet temperature and loop flow as measured, duty from
effectiveness-NTU with the fitted conductance (less air -> less waste heat taken);
``duty_held`` -- the waste-heat duty of the measured day is kept (the storage loop would run warmer)
as long as the water inlet needed stays below ``max_preheater_water_celsius``."""

FanLawName = Literal["data_fit", "affinity", "constant", "quadratic"]
"""Fan power against air flow, ``P = P_meas (V / V_meas)^n``: ``data_fit`` (exponent of the
process fan from the two measured days, see :mod:`twinlink_ml.plant_a_op.fans`), ``affinity``
(n = 3, speed control on a system curve through the origin), ``constant`` (n = 0, no fan saving)
and ``quadratic`` (n = 2) as sensitivity."""

SOLID_FRACTION_GRID: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75)
AMBIENT_SHIFTS_K: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0)
AIR_FRACTION_MIN_GRID: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
BOILER_EFFICIENCY_GRID: tuple[float, ...] = (0.85, 0.88, 0.92)
ELECTRICITY_CO2_2030_KG_PER_KWH = 0.20
"""Placeholder for a 2030 electricity mix (no source; order of magnitude of a strongly
decarbonised grid). The base value is the project factor 0.366 of ``carrier_energy``."""
ELECTRICITY_CO2_GRID: tuple[float, ...] = (
    ELECTRICITY_CO2_KG_PER_KWH,
    ELECTRICITY_CO2_2030_KG_PER_KWH,
)


@dataclass(frozen=True, slots=True)
class PlaceholderSpec:
    """One placeholder: reference value, band, source and status."""

    value: float | str
    band: tuple[float, float] | tuple[str, ...]
    source: str
    status: str = "placeholder"


PLACEHOLDERS: dict[str, PlaceholderSpec] = {
    "dryer.solid_fraction": PlaceholderSpec(
        0.65,
        (0.60, 0.75),
        "not measured. Lower end from the dryer energy balance itself (heat-loss conductance >= 0 "
        "needs >= 0.59 / 0.63 on product 1 / 2); a suspension density of 1339 kg/m3 (control-"
        "system screenshot, a third date, product unknown) corresponds to 0.42-0.76 for a solid "
        "density of 2500-1500 kg/m3; feeds of fluidised-bed spray granulation are typically "
        "concentrated suspensions (order 0.4-0.7, textbook order of magnitude, no specific "
        "source). Same value for both products (assumption).",
    ),
    "dryer.intake_relative_humidity": PlaceholderSpec(
        0.75, (0.5, 0.9), "not measured; same assumption as the storage model (Befund 91)"
    ),
    "dryer.feed_temperature_celsius": PlaceholderSpec(
        30.0, (20.0, 50.0), "not measured; temperature of the sprayed suspension"
    ),
    "dryer.atomising_air_temperature_celsius": PlaceholderSpec(
        25.0, (15.0, 40.0), "not measured; compressed atomising air after its cooler"
    ),
    "dryer.room_temperature_celsius": PlaceholderSpec(
        20.0, (10.0, 30.0), "not measured; reference of the dryer heat loss (hall)"
    ),
    "dryer.cp_solid_j_per_kg_k": PlaceholderSpec(
        1200.0, (800.0, 1600.0), "default of MaterialConfig, no source for these products"
    ),
    "dryer.normal_density_kg_per_nm3": PlaceholderSpec(
        1.293,
        (1.293, 1.293),
        "textbook value; moist-air correction neglected (as storage_a1)",
        status="textbook",
    ),
    "energy.fan_heat_fraction": PlaceholderSpec(
        0.90, (0.8, 1.0), "share of the process-fan power that heats the supply air (storage_a1)"
    ),
    "energy.boiler_efficiency": PlaceholderSpec(
        0.88, (0.85, 0.92), "steam from a gas boiler; band of the task, no plant value"
    ),
    "energy.electricity_co2_kg_per_kwh": PlaceholderSpec(
        ELECTRICITY_CO2_KG_PER_KWH,
        (ELECTRICITY_CO2_2030_KG_PER_KWH, ELECTRICITY_CO2_KG_PER_KWH),
        "project factor of carrier_energy (base); 0.20 is a 2030 placeholder without source",
        status="project factor / placeholder",
    ),
    "energy.gas_co2_kg_per_kwh": PlaceholderSpec(
        GAS_CO2_KG_PER_KWH,
        (GAS_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH),
        "project factor of carrier_energy",
        status="project factor",
    ),
    "energy.max_preheater_water_celsius": PlaceholderSpec(
        80.0,
        (70.0, 90.0),
        "only in the variant duty_held: highest water inlet of the preheater (compressor heat "
        "at about 85 degC minus an exchanger approach; Befund 99 used 85 -> 45 degC)",
    ),
    "fan.law": PlaceholderSpec(
        "data_fit",
        ("data_fit", "affinity", "constant", "quadratic"),
        "the data hold almost no variation of the air flow (within-day CV below 0.3 %); the "
        "only lever is the difference between the two days, confounded with the product",
    ),
    "constraint.air_fraction_min": PlaceholderSpec(
        0.8, (0.6, 0.9), "placeholder for the fluidisation limit (minimum air flow / today's)"
    ),
    "search.air_factor": PlaceholderSpec(
        "0.6-1.1", ("-40 %", "+10 %"), "search range of the task", status="setting"
    ),
    "search.supply_celsius": PlaceholderSpec(
        "70-100", ("70", "100"), "search range of the task", status="setting"
    ),
}
"""Every placeholder of the study with reference value, band, source and status."""


@dataclass(frozen=True, slots=True)
class DryerPlaceholders:
    """Unmeasured inputs of the dryer balance (see :data:`PLACEHOLDERS`)."""

    solid_fraction: float = 0.65
    intake_relative_humidity: float = 0.75
    feed_temperature_celsius: float = 30.0
    atomising_air_temperature_celsius: float = 25.0
    room_temperature_celsius: float = 20.0
    cp_solid_j_per_kg_k: float = 1200.0
    cp_water_liquid_j_per_kg_k: float = 4180.0
    normal_density_kg_per_nm3: float = 1.293
    gas: GasConfig = field(default_factory=GasConfig)

    def __post_init__(self) -> None:
        if not 0.0 <= self.solid_fraction < 1.0:
            raise ValueError(f"solid_fraction must be in [0, 1), is {self.solid_fraction}")
        if not 0.0 <= self.intake_relative_humidity <= 1.0:
            raise ValueError("intake_relative_humidity must be in [0, 1]")
        positive = (
            "cp_solid_j_per_kg_k",
            "cp_water_liquid_j_per_kg_k",
            "normal_density_kg_per_nm3",
        )
        for name in positive:
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class EnergyPlaceholders:
    """Conversion of the heat and power flows into CO2 (see :data:`PLACEHOLDERS`)."""

    fan_heat_fraction: float = 0.90
    boiler_efficiency: float = 0.88
    electricity_co2_kg_per_kwh: float = ELECTRICITY_CO2_KG_PER_KWH
    gas_co2_kg_per_kwh: float = GAS_CO2_KG_PER_KWH
    max_preheater_water_celsius: float = 80.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.fan_heat_fraction <= 1.0:
            raise ValueError("fan_heat_fraction must be in [0, 1]")
        if not 0.0 < self.boiler_efficiency <= 1.0:
            raise ValueError("boiler_efficiency must be in (0, 1]")
        if self.electricity_co2_kg_per_kwh < 0.0 or self.gas_co2_kg_per_kwh < 0.0:
            raise ValueError("CO2 factors must be >= 0")


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Box of the decision variables and the grid of the search.

    ``air_factor`` = process-air flow relative to the measured day mean. Coarse grid, then one
    local refinement around the coarse optimum (``refine`` times finer, +-2 coarse steps). The
    measured operating point itself is always a candidate, so the saving is never negative.
    """

    air_factor_min: float = 0.6
    air_factor_max: float = 1.1
    supply_min_celsius: float = 70.0
    supply_max_celsius: float = 100.0
    air_factor_step: float = 0.005
    supply_step_k: float = 0.1
    refine: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.air_factor_min < self.air_factor_max:
            raise ValueError("air factor range is empty")
        if not self.supply_min_celsius < self.supply_max_celsius:
            raise ValueError("supply temperature range is empty")
        if self.air_factor_step <= 0.0 or self.supply_step_k <= 0.0 or self.refine < 1:
            raise ValueError("grid steps must be positive")


def fingerprint_of(*parts: Any) -> str:
    """Fingerprint of any combination of the dataclasses above (and plain values)."""
    payload = {
        str(i): (asdict(p) if hasattr(p, "__dataclass_fields__") else p)
        for i, p in enumerate(parts)
    }
    return config_fingerprint(payload)
