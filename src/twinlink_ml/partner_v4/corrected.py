"""Partner model v4 in SI with the unit errors of the sheet removed.

Same structure as the workbook (two dryers with their own balances, exhaust streams, heater
with gas), same material properties (the partner's values are taken as given, including
``cp_gas = 1.2 kJ/(kg K)``; that value is a question to the partner, not corrected here).
Only the arithmetic is made dimensionally consistent. Internal units are SI: kg/s, kg/kg
(relative to dry air), J, W, Pa; temperatures in degC and named so.

The Antoine coefficients are those of the sheet with the decimal point restored (the sheet
stores 101962 for 10.1962 etc., see :mod:`.deviations`). They give ``p`` in Pa.

Gas consumption is a **computation from the heating power** (:func:`gas_mass_flow_kg_per_s`),
not a quantity to be predicted: ``m_gas = P_heat / (eta * H_u)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .inputs import SiDryer, SiMaterial

__all__ = [
    "ANTOINE_HIGH",
    "ANTOINE_LOW",
    "DryerResult",
    "Stream",
    "dew_point_celsius",
    "dryer_balance",
    "dryer_enthalpy_residual_w",
    "exhaust_stream",
    "gas_mass_flow_kg_per_s",
    "heat_capacity_rate_w_per_k",
    "heater_power_w",
    "saturation_humidity_kg_per_kg",
    "saturation_pressure_pa",
    "stream_enthalpy_j_per_kg_dry",
    "stream_heat_release_w",
    "temperature_at_released_heat",
]

MOLAR_MASS_RATIO_WATER_AIR = 0.622
"""``M_water / M_air``; relates humidity ratio and vapour partial pressure."""

ANTOINE_LOW = (10.1962, 1730.63, 233.426)
"""``log10(p/Pa) = A - B / (C + T/degC)``, range 1-99 degC (sheet ``Balances!J4:J6``)."""

ANTOINE_HIGH = (10.2651, 1810.94, 244.485)
"""Same, range 100-374 degC (sheet ``Balances!K4:K6``)."""


def saturation_pressure_pa(temperature_celsius: float) -> float:
    """Saturation pressure of water in Pa from the partner's Antoine coefficients."""
    a, b, c = ANTOINE_LOW if temperature_celsius < 100.0 else ANTOINE_HIGH
    return 10.0 ** (a - b / (c + temperature_celsius))


def dew_point_celsius(humidity_kg_per_kg: float, pressure_pa: float) -> float:
    """Dew point of air with the humidity ratio ``Y`` (inverse Antoine)."""
    p_vapour = pressure_pa * humidity_kg_per_kg / (MOLAR_MASS_RATIO_WATER_AIR + humidity_kg_per_kg)
    a, b, c = ANTOINE_HIGH if p_vapour >= 101_000.0 else ANTOINE_LOW
    return b / (a - math.log10(p_vapour)) - c


def saturation_humidity_kg_per_kg(temperature_celsius: float, pressure_pa: float) -> float:
    """Saturation humidity ratio at ``T`` and total pressure (``inf`` at or above boiling)."""
    p_sat = saturation_pressure_pa(temperature_celsius)
    if p_sat >= pressure_pa:
        return math.inf
    return MOLAR_MASS_RATIO_WATER_AIR * p_sat / (pressure_pa - p_sat)


@dataclass(frozen=True)
class DryerResult:
    """Result of one dryer balance in SI."""

    outlet_humidity_kg_per_kg: float
    outlet_temperature_celsius: float
    product_solids_kg_per_s: float
    evaporation_kg_per_s: float


def dryer_balance(d: SiDryer, m: SiMaterial) -> DryerResult:
    """Mass and enthalpy balance of one dryer (adiabatic, product leaves at exhaust T).

    Same model as ``Balances!F4/F5``: spray water minus the residual moisture of the product
    (equilibrium moisture) is evaporated into the air; reference state 0 degC with liquid
    water at zero enthalpy and vapour at ``h_fg`` + ``cp_v * T``.
    """
    w = m.equilibrium_moisture_kg_per_kg
    xs = d.spray_solid_fraction_kg_per_kg
    solids = d.spray_mass_flow_kg_per_s * xs
    evaporation = d.spray_mass_flow_kg_per_s * (1.0 - xs * (1.0 + w))
    y_in = d.air_humidity_kg_per_kg
    air = d.air_mass_flow_kg_per_s
    y_out = y_in + evaporation / air
    numerator = air * (
        (m.cp_gas_j_per_kg_k + y_in * m.cp_vapour_j_per_kg_k) * d.air_temperature_celsius
        + (y_in - y_out) * m.enthalpy_of_evaporation_j_per_kg
    ) + d.spray_mass_flow_kg_per_s * d.spray_temperature_celsius * (
        (1.0 - xs) * m.cp_liquid_j_per_kg_k + xs * m.cp_solid_j_per_kg_k
    )
    denominator = air * (m.cp_gas_j_per_kg_k + y_out * m.cp_vapour_j_per_kg_k) + solids * (
        m.cp_solid_j_per_kg_k + w * m.cp_liquid_j_per_kg_k
    )
    return DryerResult(
        outlet_humidity_kg_per_kg=y_out,
        outlet_temperature_celsius=numerator / denominator,
        product_solids_kg_per_s=solids,
        evaporation_kg_per_s=evaporation,
    )


def dryer_enthalpy_residual_w(d: SiDryer, m: SiMaterial, result: DryerResult) -> float:
    """Enthalpy flow out minus in [W], written independently of :func:`dryer_balance`.

    In: dry air, its vapour, the spray (liquid water at ``T_spray`` plus solids).
    Out: dry air, all vapour (evaporated included, at ``h_fg + cp_v T``), product solids with
    their residual liquid water, all at the exhaust temperature. Zero for a closed balance.
    """
    t_out = result.outlet_temperature_celsius
    air = d.air_mass_flow_kg_per_s
    xs = d.spray_solid_fraction_kg_per_kg
    spray = d.spray_mass_flow_kg_per_s
    enthalpy_in = (
        air * m.cp_gas_j_per_kg_k * d.air_temperature_celsius
        + air
        * d.air_humidity_kg_per_kg
        * (m.enthalpy_of_evaporation_j_per_kg + m.cp_vapour_j_per_kg_k * d.air_temperature_celsius)
        + spray * (1.0 - xs) * m.cp_liquid_j_per_kg_k * d.spray_temperature_celsius
        + spray * xs * m.cp_solid_j_per_kg_k * d.spray_temperature_celsius
    )
    vapour_out = air * result.outlet_humidity_kg_per_kg
    # The spray water is booked as liquid at T_spray; the evaporated part leaves as vapour
    # (with h_fg), the rest stays in the product as residual liquid.
    residual_water = result.product_solids_kg_per_s * m.equilibrium_moisture_kg_per_kg
    enthalpy_out = (
        air * m.cp_gas_j_per_kg_k * t_out
        + vapour_out * (m.enthalpy_of_evaporation_j_per_kg + m.cp_vapour_j_per_kg_k * t_out)
        + residual_water * m.cp_liquid_j_per_kg_k * t_out
        + result.product_solids_kg_per_s * m.cp_solid_j_per_kg_k * t_out
    )
    return enthalpy_out - enthalpy_in


# ---------------------------------------------------------------------------------------
# Streams, heaters, gas


@dataclass(frozen=True)
class Stream:
    """Moist-air stream: dry-air mass flow, humidity ratio, temperature."""

    dry_air_kg_per_s: float
    humidity_kg_per_kg: float
    temperature_celsius: float


def exhaust_stream(d: SiDryer, result: DryerResult) -> Stream:
    """Exhaust of a dryer as a stream (heat source)."""
    return Stream(
        d.air_mass_flow_kg_per_s,
        result.outlet_humidity_kg_per_kg,
        result.outlet_temperature_celsius,
    )


def heat_capacity_rate_w_per_k(stream: Stream, m: SiMaterial) -> float:
    """``m_dry * (cp_g + Y * cp_v)`` in W/K (sensible, no condensation)."""
    return stream.dry_air_kg_per_s * (
        m.cp_gas_j_per_kg_k + stream.humidity_kg_per_kg * m.cp_vapour_j_per_kg_k
    )


def heater_power_w(
    stream: Stream, t_in_celsius: float, t_out_celsius: float, m: SiMaterial
) -> float:
    """Heating power [W] to bring the stream from ``t_in`` to ``t_out`` (sensible)."""
    return heat_capacity_rate_w_per_k(stream, m) * (t_out_celsius - t_in_celsius)


def gas_mass_flow_kg_per_s(power_w: float, m: SiMaterial) -> float:
    """Natural-gas mass flow [kg/s] for a heating power: ``P / (eta * H_u)``.

    Derived quantity, not a target of prediction. ``E_gas = m_gas * H_u = P / eta`` is the
    fuel power that the per-carrier objective (D5) charges with the gas CO2 factor.
    """
    return power_w / (m.heater_efficiency * m.lower_heating_value_j_per_kg)


# ---------------------------------------------------------------------------------------
# Heat release of a (possibly condensing) exhaust stream


def stream_enthalpy_j_per_kg_dry(
    humidity_kg_per_kg: float,
    temperature_celsius: float,
    pressure_pa: float,
    m: SiMaterial,
) -> float:
    """Enthalpy per kg dry air incl. condensate [J/kg], reference 0 degC liquid water.

    Above the dew point: dry air + vapour (``h_fg + cp_v T``). Below: the saturated vapour
    and the condensate (liquid, at the stream temperature) -- the condensate stays in the
    balance, so the function is continuous at the dew point.
    """
    y_sat = saturation_humidity_kg_per_kg(temperature_celsius, pressure_pa)
    y_vapour = min(humidity_kg_per_kg, y_sat)
    y_liquid = humidity_kg_per_kg - y_vapour
    t = temperature_celsius
    return (
        m.cp_gas_j_per_kg_k * t
        + y_vapour * (m.enthalpy_of_evaporation_j_per_kg + m.cp_vapour_j_per_kg_k * t)
        + y_liquid * m.cp_liquid_j_per_kg_k * t
    )


def stream_heat_release_w(
    stream: Stream, t_cold_celsius: float, pressure_pa: float, m: SiMaterial
) -> float:
    """Heat released when the stream is cooled to ``t_cold`` incl. condensation [W]."""
    h_in = stream_enthalpy_j_per_kg_dry(
        stream.humidity_kg_per_kg, stream.temperature_celsius, pressure_pa, m
    )
    h_out = stream_enthalpy_j_per_kg_dry(stream.humidity_kg_per_kg, t_cold_celsius, pressure_pa, m)
    return stream.dry_air_kg_per_s * (h_in - h_out)


def temperature_at_released_heat(
    stream: Stream,
    released_w: float,
    t_cold_celsius: float,
    pressure_pa: float,
    m: SiMaterial,
) -> float:
    """Temperature of the stream after it has released ``released_w`` (bisection, monotone)."""
    h_in = stream_enthalpy_j_per_kg_dry(
        stream.humidity_kg_per_kg, stream.temperature_celsius, pressure_pa, m
    )
    target = h_in - released_w / stream.dry_air_kg_per_s
    lo, hi = t_cold_celsius, stream.temperature_celsius
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if stream_enthalpy_j_per_kg_dry(stream.humidity_kg_per_kg, mid, pressure_pa, m) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
