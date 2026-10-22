"""Plausibility checks of a dryer line, reusable for any source of numbers.

Each check takes plain numbers (from the flowsheet, from a partner table, from a Dyssol stream),
so the same check can be applied to the partner's Excel values, to the Dyssol values and to what
the units of this package compute. A check returns one :class:`Finding`; ``severity`` is
``"ok"``, ``"warning"`` (physically possible, but it needs a measure: a pressurised circuit) or
``"violation"`` (not possible as stated).

(a) Exhaust state (:func:`check_exhaust_saturation`,
    :func:`check_exhaust_above_adiabatic_saturation`):
    the exhaust humidity may not exceed the saturation humidity at the exhaust temperature, and
    the exhaust temperature may not be below the adiabatic saturation temperature of the dryer's
    inlet air. The second bound is strict for an adiabatic dryer whose liquid enters at the
    adiabatic saturation temperature; a colder spray and the heating of the solid move the true
    bound a little down, so a small undercut is reported with its size (``margin_k``) and is
    read with that in mind.
(b) Product mass (:func:`check_product_mass`): the product (solid plus residual moisture) may
    not be lighter than the solid fed with the spray.
(c) Medium temperature (:func:`check_medium_boiling`): a liquid heat transfer medium (water)
    above its boiling point at the circuit pressure (1 atm unless stated) boils; the warning
    states the saturation pressure the circuit would need at least (no safety margin added).

Water properties are those of IAPWS-IF97, region 4 (saturation pressure, equation 30, and its
backward equation 31), valid from 0 to 373.946 degC. The humidity ratio uses the ratio of the
molar masses of water and dry air, 18.015 / 28.965 = 0.62198. Enthalpies are referred to
0 degC with liquid water at zero enthalpy, as in the dryer block (equation 1.4 of the partner's
equation document): ``dh_v(T) = dh_0 + (cp_v - cp_l) T``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from twinlink_ml.partner_v4.inputs import ATM_PA, SiMaterial

__all__ = [
    "MOLAR_MASS_RATIO",
    "Finding",
    "SaturationProps",
    "Severity",
    "adiabatic_saturation_temperature_celsius",
    "check_exhaust_above_adiabatic_saturation",
    "check_exhaust_saturation",
    "check_medium_boiling",
    "check_product_mass",
    "saturation_humidity_kg_per_kg",
    "water_boiling_point_celsius",
    "water_saturation_pressure_pa",
]

Severity = Literal["ok", "warning", "violation"]

MOLAR_MASS_RATIO = 18.015 / 28.965
"""``M_water / M_dry_air``."""

_KELVIN = 273.15
_T_MIN_C = 0.0
_T_CRIT_C = 647.096 - _KELVIN
_N = (
    0.11670521452767e4,
    -0.72421316703206e6,
    -0.17073846940092e2,
    0.12020824702470e5,
    -0.32325550322333e7,
    0.14915108613530e2,
    -0.48232657361591e4,
    0.40511340542057e6,
    -0.23855557567849,
    0.65017534844798e3,
)


def water_saturation_pressure_pa(temperature_celsius: float) -> float:
    """Saturation pressure of water [Pa], IAPWS-IF97 equation 30 (0 to 373.946 degC)."""
    if not _T_MIN_C <= temperature_celsius <= _T_CRIT_C:
        raise ValueError(f"{temperature_celsius} degC is outside 0 .. {_T_CRIT_C:.3f} degC")
    n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = _N
    t = temperature_celsius + _KELVIN
    theta = t + n9 / (t - n10)
    a = theta * theta + n1 * theta + n2
    b = n3 * theta * theta + n4 * theta + n5
    c = n6 * theta * theta + n7 * theta + n8
    return (2.0 * c / (-b + math.sqrt(b * b - 4.0 * a * c))) ** 4 * 1e6


def water_boiling_point_celsius(pressure_pa: float) -> float:
    """Saturation temperature of water [degC] at ``pressure_pa``, IAPWS-IF97 equation 31."""
    if not 611.213 <= pressure_pa <= 22.064e6:
        raise ValueError(f"{pressure_pa} Pa is outside the saturation line")
    n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = _N
    beta = (pressure_pa / 1e6) ** 0.25
    e = beta * beta + n3 * beta + n6
    f = n1 * beta * beta + n4 * beta + n7
    g = n2 * beta * beta + n5 * beta + n8
    d = 2.0 * g / (-f - math.sqrt(f * f - 4.0 * e * g))
    t = (n10 + d - math.sqrt((n10 + d) ** 2 - 4.0 * (n9 + n10 * d))) / 2.0
    return t - _KELVIN


def saturation_humidity_kg_per_kg(temperature_celsius: float, pressure_pa: float = ATM_PA) -> float:
    """Saturation humidity ratio [kg water / kg dry air]; ``inf`` at or above the boiling point."""
    p_s = water_saturation_pressure_pa(max(temperature_celsius, _T_MIN_C))
    if p_s >= pressure_pa:
        return math.inf
    return MOLAR_MASS_RATIO * p_s / (pressure_pa - p_s)


@dataclass(frozen=True)
class SaturationProps:
    """Heat capacities [J/(kg K)] and evaporation enthalpy at 0 degC [J/kg] of one model."""

    cp_gas_j_per_kg_k: float
    cp_vapour_j_per_kg_k: float
    cp_liquid_j_per_kg_k: float
    evaporation_enthalpy_0c_j_per_kg: float

    @classmethod
    def from_material(cls, material: SiMaterial) -> SaturationProps:
        return cls(
            material.cp_gas_j_per_kg_k,
            material.cp_vapour_j_per_kg_k,
            material.cp_liquid_j_per_kg_k,
            material.enthalpy_of_evaporation_j_per_kg,
        )

    def latent_heat_j_per_kg(self, temperature_celsius: float) -> float:
        slope = self.cp_vapour_j_per_kg_k - self.cp_liquid_j_per_kg_k
        return self.evaporation_enthalpy_0c_j_per_kg + slope * temperature_celsius


def adiabatic_saturation_temperature_celsius(
    temperature_celsius: float,
    humidity_kg_per_kg: float,
    props: SaturationProps,
    pressure_pa: float = ATM_PA,
) -> float:
    """Temperature at which air of ``(T, Y)`` leaves saturated from an adiabatic saturator.

    Root of ``(cp_g + Y cp_v)(T - T_as) = (Y_sat(T_as) - Y) dh_v(T_as)`` (bisection; the
    left side falls and the right side rises with ``T_as``).
    """
    cp_moist = props.cp_gas_j_per_kg_k + humidity_kg_per_kg * props.cp_vapour_j_per_kg_k

    def g(t_as: float) -> float:
        y_s = saturation_humidity_kg_per_kg(t_as, pressure_pa)
        if math.isinf(y_s):
            return -math.inf
        return cp_moist * (temperature_celsius - t_as) - (y_s - humidity_kg_per_kg) * (
            props.latent_heat_j_per_kg(t_as)
        )

    lo = _T_MIN_C
    hi = min(temperature_celsius, water_boiling_point_celsius(pressure_pa))
    if g(lo) <= 0.0:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) > 0.0 else (lo, mid)
        if hi - lo <= 1e-10:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Finding:
    """Result of one check."""

    check: str
    severity: Severity
    message: str
    values: Mapping[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.severity == "ok"

    def as_dict(self) -> dict[str, object]:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "values": dict(self.values),
        }


def check_exhaust_saturation(
    exhaust_celsius: float,
    exhaust_humidity_kg_per_kg: float,
    *,
    pressure_pa: float = ATM_PA,
    rel_tol: float = 0.0,
) -> Finding:
    """(a) ``Y_out <= Y_sat(T_out)``; violation if the exhaust holds more water than saturated."""
    y_sat = saturation_humidity_kg_per_kg(exhaust_celsius, pressure_pa)
    values = {
        "exhaust_celsius": exhaust_celsius,
        "exhaust_humidity_kg_per_kg": exhaust_humidity_kg_per_kg,
        "saturation_humidity_kg_per_kg": y_sat,
        "relative_saturation": exhaust_humidity_kg_per_kg / y_sat if y_sat > 0.0 else math.inf,
    }
    if exhaust_humidity_kg_per_kg > y_sat * (1.0 + rel_tol):
        return Finding(
            "exhaust_saturation",
            "violation",
            f"exhaust holds {exhaust_humidity_kg_per_kg * 1e3:.2f} g/kg at {exhaust_celsius:.2f} "
            f"degC, saturation is {y_sat * 1e3:.2f} g/kg",
            values,
        )
    return Finding("exhaust_saturation", "ok", "exhaust below saturation", values)


def check_exhaust_above_adiabatic_saturation(
    exhaust_celsius: float,
    inlet_celsius: float,
    inlet_humidity_kg_per_kg: float,
    props: SaturationProps,
    *,
    pressure_pa: float = ATM_PA,
    tol_k: float = 0.0,
) -> Finding:
    """(a) ``T_out >= T_as`` of the inlet air; violation if the exhaust is colder."""
    t_as = adiabatic_saturation_temperature_celsius(
        inlet_celsius, inlet_humidity_kg_per_kg, props, pressure_pa
    )
    margin = exhaust_celsius - t_as
    values = {
        "exhaust_celsius": exhaust_celsius,
        "adiabatic_saturation_celsius": t_as,
        "margin_k": margin,
    }
    if margin < -tol_k:
        return Finding(
            "exhaust_above_adiabatic_saturation",
            "violation",
            f"exhaust at {exhaust_celsius:.2f} degC is {-margin:.2f} K below the adiabatic "
            f"saturation temperature of the inlet air ({t_as:.2f} degC)",
            values,
        )
    return Finding(
        "exhaust_above_adiabatic_saturation", "ok", "exhaust above adiabatic saturation", values
    )


def check_product_mass(
    product_kg_per_s: float,
    spray_kg_per_s: float,
    solid_fraction_kg_per_kg: float,
    *,
    rel_tol: float = 0.0,
) -> Finding:
    """(b) product mass >= solid fed with the spray; violation if solid is lost."""
    solid = spray_kg_per_s * solid_fraction_kg_per_kg
    values = {
        "product_kg_per_s": product_kg_per_s,
        "solid_feed_kg_per_s": solid,
        "ratio": product_kg_per_s / solid if solid > 0.0 else math.inf,
    }
    if product_kg_per_s < solid * (1.0 - rel_tol):
        return Finding(
            "product_mass",
            "violation",
            f"product {product_kg_per_s * 3600:.3f} kg/h is lighter than the solid fed "
            f"({solid * 3600:.3f} kg/h)",
            values,
        )
    return Finding("product_mass", "ok", "product not lighter than the solid feed", values)


def check_medium_boiling(medium_celsius: float, *, pressure_pa: float = ATM_PA) -> Finding:
    """(c) medium (water) above its boiling point at ``pressure_pa``: warning, pressure needed."""
    boiling = water_boiling_point_celsius(pressure_pa)
    values: dict[str, float] = {
        "medium_celsius": medium_celsius,
        "boiling_point_celsius": boiling,
        "circuit_pressure_pa": pressure_pa,
    }
    if medium_celsius <= boiling:
        return Finding("medium_boiling", "ok", "medium below its boiling point", values)
    needed = water_saturation_pressure_pa(medium_celsius)
    values["required_pressure_pa"] = needed
    return Finding(
        "medium_boiling",
        "warning",
        f"medium at {medium_celsius:.2f} degC is above the boiling point at "
        f"{pressure_pa / 1e5:.5f} bar ({boiling:.2f} degC); the circuit needs at least "
        f"{needed / 1e5:.3f} bar absolute (saturation pressure, no margin)",
        values,
    )
