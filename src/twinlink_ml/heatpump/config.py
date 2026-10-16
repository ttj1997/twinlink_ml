r"""Placeholders of the heat-pump study: machine, prices, emission factors, signal, capital.

Every number here that is not a partner statement or a repository constant is a PLACEHOLDER with a
central value, a band and a source status (:data:`PLACEHOLDER_BANDS`, docstrings below). A
configuration is REALISTIC when all its placeholders lie inside their bands (:func:`is_realistic`);
values outside a band are reported, never counted in a gate. The streams of the site (sources and
sinks) are those of storage test 1 (:mod:`twinlink_ml.storage_timeslice`, Befund 99) and keep their
own placeholders and bands there.

Roles, not tags: nothing here names a company, a product, a device number or a measuring point.

Units: temperatures degC, powers kW, energies kWh, prices EUR/kWh, CO2 kg/kWh (EUR/t for the CO2
price). Useful heat ``U`` is heat delivered to the process; fuel = ``U / eta``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from ..evaluation.net_energy import ELECTRICITY_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH
from ..simulation.config import config_fingerprint

__all__ = [
    "CONDENSING_GRID_CELSIUS",
    "ELECTRICITY_CO2_2030_KG_PER_KWH",
    "ELECTRICITY_CO2_KG_PER_KWH",
    "EVAPORATING_GRID_CELSIUS",
    "GAS_CO2_KG_PER_KWH",
    "HEAT_PUMP_SIZES_KW",
    "PLACEHOLDER_BANDS",
    "SITE_CO2_T_PER_A",
    "SITE_ELECTRICITY_GWH_PER_A",
    "SITE_GAS_GWH_PER_A",
    "Capital",
    "Criterion",
    "Economics",
    "HeatPumpSpec",
    "Signal",
    "co2_criterion",
    "cost_criterion",
    "is_realistic",
]

ELECTRICITY_CO2_2030_KG_PER_KWH = 0.20
"""Electricity CO2 factor, variant "2030" (PLACEHOLDER, band 0.10-0.30 kg/kWh): order of magnitude
of the grid mix after the coal phase-down and 80 % renewables as planned for 2030 (user
specification "e.g. 0.20"); the repository factor 0.366 (``evaluation.net_energy``, project
description, UBA) is the base."""

SITE_GAS_GWH_PER_A = 30.0
SITE_ELECTRICITY_GWH_PER_A = 15.0
SITE_CO2_T_PER_A = 11500.0
"""Partner statements (order of magnitude) for the whole site; context only (shares of savings)."""

HEAT_PUMP_SIZES_KW: tuple[float, ...] = (0.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0)
"""Heating (condenser) capacity grid, kW (user: 0-1500 kW)."""

EVAPORATING_GRID_CELSIUS: tuple[float, ...] = tuple(float(t) for t in range(5, 75, 5))
"""Evaporating temperatures of the design grid (5 ... 70 degC)."""

CONDENSING_GRID_CELSIUS: tuple[float, ...] = tuple(float(t) for t in range(60, 95, 5))
"""Condensing temperatures of the design grid (60 ... 90 degC; user: lift to 60-90 degC)."""

PLACEHOLDER_BANDS: dict[str, tuple[float, float]] = {
    "quality": (0.45, 0.55),
    "min_lift_k": (15.0, 30.0),
    "max_lift_k": (40.0, 80.0),
    "max_condensing_celsius": (80.0, 100.0),
    "cop_max": (6.0, 10.0),
    "heat_pump_approach_k": (5.0, 20.0),
    "gas_eur_per_kwh": (0.025, 0.060),
    "electricity_eur_per_kwh": (0.10, 0.25),
    "co2_eur_per_t": (55.0, 300.0),
    "heater_efficiency": (0.75, 0.95),
    "electricity_co2_kg_per_kwh": (0.10, 0.40),
    "capex_eur_per_kw": (400.0, 1000.0),
    "price_amplitude": (0.0, 0.6),
    "co2_amplitude": (0.0, 0.4),
    "weekend_reduction": (0.0, 0.2),
}
"""Band of every numeric placeholder of this module (gate: realistic = inside all bands)."""


@dataclass(frozen=True, slots=True)
class HeatPumpSpec:
    r"""Heat-pump model: ``COP = min(quality * COP_Carnot, cop_max)``, two isothermal exchangers.

    ``COP_Carnot = T_k / (T_k - T_e)`` in kelvin with the condensing temperature ``T_k`` and the
    evaporating temperature ``T_e`` of the refrigerant.

    * ``quality`` (Guetegrad) 0.50, band 0.45-0.55: user specification; consistent with the range
      of industrial heat pumps in published reviews (e.g. Arpagaus et al., Energy 152, 2018:
      about 0.4-0.6). PLACEHOLDER.
    * ``min_lift_k`` 20 K, band 15-30: below about 20 K lift a constant quality overstates the COP
      (compressor and exchanger losses dominate); PLACEHOLDER.
    * ``max_lift_k`` 60 K, band 40-80: typical limit of a single-stage machine (data sheets of
      industrial machines, order of magnitude); PLACEHOLDER.
    * ``max_condensing_celsius`` 90 degC, band 80-100: user frame "lift to 60-90 degC"; standard
      industrial refrigerants reach about 90-100 degC; PLACEHOLDER.
    * ``cop_max`` 8, band 6-10: cap against the Carnot artefact at small lifts; PLACEHOLDER.
    * ``approach_k`` ``None`` = ``dT_min/2``: refrigerant side condensing/evaporating has a very
      high heat-transfer coefficient, its stream contribution is about 0, so the whole approach
      ``dT_min/2`` sits on the process side (stream-contribution convention of pinch analysis);
      band 5-20 K, variants ``dT_min`` and 5 K. PLACEHOLDER.

    In the problem table the evaporator is an isothermal COLD stream (it takes heat only from hot
    parts with real temperature above ``T_e + a``), the condenser an isothermal HOT stream (it
    serves cold parts only below ``T_k - a``), exactly as the storage streams of storage test 1.
    """

    quality: float = 0.50
    min_lift_k: float = 20.0
    max_lift_k: float = 60.0
    max_condensing_celsius: float = 90.0
    cop_max: float = 8.0
    approach_k: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.quality <= 1.0:
            raise ValueError(f"quality must be in (0, 1], is {self.quality}")
        if not 0.0 < self.min_lift_k < self.max_lift_k:
            raise ValueError("need 0 < min_lift_k < max_lift_k")
        if self.cop_max <= 1.0:
            raise ValueError("cop_max must exceed 1")
        if self.approach_k is not None and self.approach_k <= 0.0:
            raise ValueError("approach_k must be positive")

    def approach(self, minimum_pinch_k: float) -> float:
        """Approach of each heat-pump exchanger to the process streams."""
        return minimum_pinch_k / 2.0 if self.approach_k is None else self.approach_k


@dataclass(frozen=True, slots=True)
class Economics:
    r"""Prices, emission factors and the heater efficiency (all PLACEHOLDERS unless marked).

    * ``gas_eur_per_kwh`` 0.040 EUR/kWh fuel WITHOUT the CO2 price, band 0.025-0.060: order of
      magnitude of German industrial gas prices 2025/26 (commodity, grid, levies) as commonly
      published (Destatis / BDEW price statistics); not checked against one table. PLACEHOLDER.
    * ``electricity_eur_per_kwh`` 0.16 EUR/kWh, band 0.10-0.25: mean industrial electricity price
      including grid fees and levies after reliefs, same kind of source and status. PLACEHOLDER.
      The CO2 cost of electricity is inside this price (not charged again).
    * ``co2_eur_per_t`` 55 EUR/t today, 150 / 300 EUR/t as 2030 variants: user specification
      (partner expectation 55 -> 300 EUR/t by 2030). Applies to gas only.
    * ``heater_efficiency`` 0.85, band 0.75-0.95: useful heat per kWh gas, one value for the steam
      heater of plant A (boiler plus distribution, lower end; the partner data case of Befund 60
      uses 0.80) and the direct gas burner of plant C (upper end). PLACEHOLDER.
    * ``electricity_co2_kg_per_kwh`` 0.366 (repository factor, base) or 0.20 (variant 2030);
      ``gas_co2_kg_per_kwh`` 0.201 (repository factor, not varied).
    """

    gas_eur_per_kwh: float = 0.040
    electricity_eur_per_kwh: float = 0.16
    co2_eur_per_t: float = 55.0
    heater_efficiency: float = 0.85
    electricity_co2_kg_per_kwh: float = ELECTRICITY_CO2_KG_PER_KWH
    gas_co2_kg_per_kwh: float = GAS_CO2_KG_PER_KWH

    def __post_init__(self) -> None:
        if not 0.0 < self.heater_efficiency <= 1.0:
            raise ValueError("heater_efficiency must be in (0, 1]")
        if min(self.gas_eur_per_kwh, self.electricity_eur_per_kwh, self.co2_eur_per_t) < 0.0:
            raise ValueError("prices must not be negative")

    @property
    def gas_cost_per_kwh_heat(self) -> float:
        """EUR per kWh of useful heat from gas, CO2 price included."""
        fuel = self.gas_eur_per_kwh + self.co2_eur_per_t * self.gas_co2_kg_per_kwh / 1000.0
        return fuel / self.heater_efficiency

    @property
    def gas_co2_per_kwh_heat(self) -> float:
        """CO2 in kg per kWh of useful heat from gas."""
        return self.gas_co2_kg_per_kwh / self.heater_efficiency


@dataclass(frozen=True, slots=True)
class Criterion:
    """One objective: value of a kWh of gas heat and mean value of a kWh of electricity.

    ``kind`` is ``"co2"`` (kg) or ``"cost"`` (EUR). The electricity value is the MEAN of the
    signal; Stage B multiplies it by a time profile (:class:`Signal`).
    """

    name: str
    kind: Literal["co2", "cost"]
    heat_value_per_kwh: float
    electricity_value_per_kwh: float

    def __post_init__(self) -> None:
        if self.kind not in ("co2", "cost"):
            raise ValueError(f"unknown criterion kind {self.kind!r}")
        if self.heat_value_per_kwh <= 0.0 or self.electricity_value_per_kwh < 0.0:
            raise ValueError("criterion values must be positive (heat) and >= 0 (electricity)")

    @property
    def breakeven_cop(self) -> float:
        """COP above which a kWh of heat pump heat that replaces gas heat 1:1 pays."""
        return self.electricity_value_per_kwh / self.heat_value_per_kwh

    @property
    def unit(self) -> str:
        return "kg CO2" if self.kind == "co2" else "EUR"


def co2_criterion(name: str, economics: Economics) -> Criterion:
    """CO2 criterion of ``economics`` (kg per kWh)."""
    return Criterion(
        name, "co2", economics.gas_co2_per_kwh_heat, economics.electricity_co2_kg_per_kwh
    )


def cost_criterion(name: str, economics: Economics) -> Criterion:
    """Cost criterion of ``economics`` (EUR per kWh, CO2 price on gas)."""
    return Criterion(
        name, "cost", economics.gas_cost_per_kwh_heat, economics.electricity_eur_per_kwh
    )


@dataclass(frozen=True, slots=True)
class Capital:
    """Annualised capital cost of the heat pump (PLACEHOLDERS).

    ``capex_eur_per_kw`` 600 EUR per kW heating capacity installed and integrated, band 400-1000:
    order of magnitude of large industrial heat pumps in technology catalogues (e.g. the Danish
    Energy Agency catalogue for large heat pumps, about 0.5-1 MEUR/MW); annuity over 15 years at
    6 % plus 2 %/a maintenance. Only for the payback column; the operating savings do not use it.
    """

    capex_eur_per_kw: float = 600.0
    lifetime_years: float = 15.0
    interest_rate: float = 0.06
    maintenance_fraction: float = 0.02

    @property
    def annuity_factor(self) -> float:
        r, n = self.interest_rate, self.lifetime_years
        if r == 0.0:
            return 1.0 / n
        return float(r / (1.0 - (1.0 + r) ** (-n)))

    def annual_cost_eur(self, size_kw: float) -> float:
        """Annualised capital plus maintenance, EUR/a."""
        return size_kw * self.capex_eur_per_kw * (self.annuity_factor + self.maintenance_fraction)


@dataclass(frozen=True, slots=True)
class Signal:
    r"""Time profile of the electricity value (price or CO2 factor) in Stage B.

    ``c_t = mean * (1 + amplitude * shape(h)) * week(d)`` with ``shape`` of mean 0 and minimum -1
    (:func:`twinlink_ml.heatpump.signal.daily_shape`), so ``amplitude`` is the depth of the
    daily trough relative to the mean (``amplitude <= 1`` keeps the signal non-negative). ``week``
    lowers Saturday and Sunday by ``weekend_reduction`` relative to the weekdays, mean 1 over a
    week. PLACEHOLDERS: shape "night moderate, morning peak, solar trough at midday, evening peak"
    (typical German day-ahead and grid-CO2 pattern, not fitted to data); amplitude bands: price
    0-0.6 (spot-indexed energy share of an industrial tariff; 0 for a fixed tariff), CO2 0-0.4
    (hourly average grid factor); weekend reduction 0.1, band 0-0.2.
    """

    amplitude: float = 0.0
    weekend_reduction: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.amplitude <= 1.0:
            raise ValueError("amplitude must be in [0, 1]")
        if not 0.0 <= self.weekend_reduction < 1.0:
            raise ValueError("weekend_reduction must be in [0, 1)")


def is_realistic(
    *,
    spec: HeatPumpSpec,
    economics: Economics,
    minimum_pinch_k: float,
    criterion_kind: Literal["co2", "cost"] | None = None,
    signal: Signal | None = None,
    capital: Capital | None = None,
) -> bool:
    """All placeholders of this module inside :data:`PLACEHOLDER_BANDS`."""
    values: dict[str, float] = {
        "quality": spec.quality,
        "min_lift_k": spec.min_lift_k,
        "max_lift_k": spec.max_lift_k,
        "max_condensing_celsius": spec.max_condensing_celsius,
        "cop_max": spec.cop_max,
        "heat_pump_approach_k": spec.approach(minimum_pinch_k),
        "gas_eur_per_kwh": economics.gas_eur_per_kwh,
        "electricity_eur_per_kwh": economics.electricity_eur_per_kwh,
        "co2_eur_per_t": economics.co2_eur_per_t,
        "heater_efficiency": economics.heater_efficiency,
        "electricity_co2_kg_per_kwh": economics.electricity_co2_kg_per_kwh,
    }
    if capital is not None:
        values["capex_eur_per_kw"] = capital.capex_eur_per_kw
    if signal is not None:
        values["weekend_reduction"] = signal.weekend_reduction
        if criterion_kind == "cost":
            values["price_amplitude"] = signal.amplitude
        elif criterion_kind == "co2":
            values["co2_amplitude"] = signal.amplitude
    return all(
        PLACEHOLDER_BANDS[name][0] - 1e-12 <= value <= PLACEHOLDER_BANDS[name][1] + 1e-12
        for name, value in values.items()
    )


def fingerprint_of(*parts: Any) -> str:
    """Config fingerprint of dataclass parts (for the records)."""
    return config_fingerprint({f"part{i}": asdict(p) for i, p in enumerate(parts)})
