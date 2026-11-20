"""Python rebuild of the stationary Dyssol units and of the flowsheets built from them.

Three units, each a closed formula (source: the units delivered on 2026-11-19, kept in
``local/``; equation numbers as in the heat-recovery unit):

* **dryer**: the feed is dried down to the equilibrium moisture ``X_eq`` (dry basis); energy
  balance with liquid water at the normal boiling point ``T_b`` as the reference state and
  constant heat capacities,
  ``T_out = T_b + (H_in - M_evap h_vap) / C_out``, gas and product leave at ``T_out``;
* **heat recovery**: hot gas, medium and cold gas coupled by equations 4.1 / 4.2,
  ``A (T_hot - T2) = B (T2 - T1)``, ``C (T1 - T_cold) = B (T2 - T1)`` with the capacity rates
  ``A``, ``B``, ``C`` of hot gas, medium, cold gas; the hot gas leaves at ``T2``, the cold gas at
  ``T1`` (ideal exchangers, no condensation);
* **heater**: heats to a set temperature; duty from the specific enthalpy of the mixture, which
  Dyssol tabulates on an equidistant temperature grid and interpolates linearly
  (:class:`EnthalpyTable`); outside the temperature range of a compound's correlation the
  correlation is evaluated at the nearest bound; fuel = duty / (efficiency x calorific value).

Because every dryer inlet is held at its heater's set temperature, no flowsheet of this family
has a real recycle: dryers first, then the recoveries, then the heaters.

``preheated_by[i]`` names the line whose dryer exhaust preheats the fresh gas of line ``i``
(``(0,)`` for one line with its own recovery, ``(1, 0)`` for two lines that exchange).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from twinlink_ml.flowsheet.validators import saturation_humidity_kg_per_kg

from .materials import DyssolProperties

__all__ = [
    "KELVIN",
    "EnthalpyTable",
    "LineInputs",
    "LineOutputs",
    "dryer",
    "evaluate",
    "heat_recovery",
    "heater",
]

KELVIN = 273.15


@dataclass(frozen=True)
class LineInputs:
    """Inputs of one line (one fresh-gas feed, heater, dryer, and the circuit of its exhaust)."""

    dry_gas_kg_per_h: float
    gas_humidity_g_per_kg: float
    gas_celsius: float
    dryer_inlet_celsius: float
    spray_kg_per_h: float
    solid_fraction: float
    spray_celsius: float
    medium_kg_per_h: float
    """Flow of the medium in the heat recovery that cools THIS line's dryer exhaust."""
    equilibrium_moisture_g_per_kg: float
    heater_efficiency_percent: float
    calorific_value_kj_per_kg: float


@dataclass(frozen=True)
class LineOutputs:
    """Results of one line."""

    dryer_outlet_celsius: float
    dryer_outlet_humidity_g_per_kg: float
    exhaust_celsius: float
    """This line's dryer exhaust after its heat recovery (the workbook's exhaust temperature)."""
    preheated_celsius: float
    """This line's fresh gas after the preheater (heater inlet)."""
    heat_recovered_kw: float
    """Heat taken from this line's exhaust."""
    heater_duty_kw: float
    fuel_kg_per_h: float
    product_kg_per_h: float
    saturation_ratio: float
    """Humidity of the dryer exhaust over the saturation humidity at its temperature (1 atm);
    above 1 the unit's result is physically inadmissible (the unit has no saturation limit)."""

    @property
    def oversaturated(self) -> bool:
        """True if the dryer exhaust holds more water than saturated air can."""
        return self.saturation_ratio > 1.0


@dataclass(frozen=True)
class EnthalpyTable:
    """Specific enthalpy of a gas mixture as Dyssol tabulates it (linear between grid points)."""

    properties: DyssolProperties
    minimum_k: float = 173.0
    maximum_k: float = 1273.0
    intervals: int = 100

    def _exact(self, kelvin: np.ndarray, vapour_share: float) -> np.ndarray:
        a, c = self.properties.gas_enthalpy_linear
        s = self.properties.vapour_enthalpy_shomate
        tg = np.clip(kelvin, *self.properties.gas_enthalpy_range_k)
        t = np.clip(kelvin, *self.properties.vapour_enthalpy_range_k)
        vapour = (
            s[0] * t + s[1] * t**2 / 2 + s[2] * t**3 / 3 + s[3] * t**4 / 4 - s[4] / t + s[5] - s[6]
        )
        return np.asarray((1.0 - vapour_share) * (a * tg + c) + vapour_share * vapour)

    def enthalpy_j_per_kg(self, kelvin: float, vapour_share: float) -> float:
        """Enthalpy of humid gas with the given mass share of vapour at ``kelvin``."""
        grid = np.linspace(self.minimum_k, self.maximum_k, self.intervals + 1)
        return float(np.interp(kelvin, grid, self._exact(grid, vapour_share)))


def dryer(
    props: DyssolProperties,
    gas_kg_per_s: float,
    vapour_kg_per_s: float,
    gas_in_k: float,
    solid_kg_per_s: float,
    liquid_kg_per_s: float,
    feed_k: float,
    equilibrium_moisture: float,
) -> tuple[float, float, float]:
    """Outlet temperature [K], vapour flow out and liquid left in the product [kg/s]."""
    evaporated = max(liquid_kg_per_s - solid_kg_per_s * equilibrium_moisture, 0.0)
    liquid_out = liquid_kg_per_s - evaporated
    vapour_out = vapour_kg_per_s + evaporated
    t_b = props.boiling_point_k
    h_in = (gas_kg_per_s * props.cp_gas + vapour_kg_per_s * props.cp_vapour) * (gas_in_k - t_b) + (
        solid_kg_per_s * props.cp_solid + liquid_kg_per_s * props.cp_liquid
    ) * (feed_k - t_b)
    c_out = (
        gas_kg_per_s * props.cp_gas
        + vapour_out * props.cp_vapour
        + solid_kg_per_s * props.cp_solid
        + liquid_out * props.cp_liquid
    )
    t_out = t_b + (h_in - evaporated * props.heat_of_vaporization_j_per_kg) / c_out
    return t_out, vapour_out, liquid_out


def heat_recovery(
    props: DyssolProperties,
    hot: tuple[float, float, float],
    cold: tuple[float, float, float],
    medium_kg_per_s: float,
) -> tuple[float, float, float]:
    """Cold-gas outlet ``T1`` [K], hot-gas outlet ``T2`` [K] and recovered heat [W].

    ``hot`` and ``cold`` are ``(dry gas kg/s, vapour kg/s, temperature K)``.
    """
    a = hot[0] * props.cp_gas + hot[1] * props.cp_vapour
    c = cold[0] * props.cp_gas + cold[1] * props.cp_vapour
    b = medium_kg_per_s * props.cp_liquid
    d = a * b + a * c + b * c
    t1 = (a * b * hot[2] + (a + b) * c * cold[2]) / d
    t2 = (b * c * cold[2] + a * (b + c) * hot[2]) / d
    return t1, t2, b * (t2 - t1)


def heater(
    table: EnthalpyTable,
    gas_kg_per_s: float,
    vapour_kg_per_s: float,
    inlet_k: float,
    outlet_k: float,
    efficiency: float,
    calorific_value_kj_per_kg: float,
) -> tuple[float, float]:
    """Heat duty [kW] and fuel consumption [kg/h]."""
    total = gas_kg_per_s + vapour_kg_per_s
    share = vapour_kg_per_s / total
    duty_kw = (
        total
        * (table.enthalpy_j_per_kg(outlet_k, share) - table.enthalpy_j_per_kg(inlet_k, share))
        / 1000.0
    )
    return duty_kw, duty_kw * 3600.0 / (efficiency * calorific_value_kj_per_kg)


def evaluate(
    props: DyssolProperties, lines: Sequence[LineInputs], preheated_by: Sequence[int]
) -> tuple[LineOutputs, ...]:
    """All lines of a flowsheet in which every exhaust preheats exactly one fresh-gas feed."""
    n = len(lines)
    if sorted(preheated_by) != list(range(n)):
        raise ValueError("preheated_by must be a permutation of the line indices")
    table = EnthalpyTable(props)
    gas = [ln.dry_gas_kg_per_h / 3600.0 for ln in lines]
    vapour = [g * ln.gas_humidity_g_per_kg / 1000.0 for g, ln in zip(gas, lines, strict=True)]
    dried = []
    for ln, g, v in zip(lines, gas, vapour, strict=True):
        spray = ln.spray_kg_per_h / 3600.0
        solid = spray * ln.solid_fraction
        t_out, vapour_out, liquid_out = dryer(
            props,
            g,
            v,
            ln.dryer_inlet_celsius + KELVIN,
            solid,
            spray - solid,
            ln.spray_celsius + KELVIN,
            ln.equilibrium_moisture_g_per_kg / 1000.0,
        )
        dried.append((t_out, vapour_out, solid + liquid_out))
    recovered: dict[int, tuple[float, float, float]] = {}
    for cold_line, hot_line in enumerate(preheated_by):
        t_out, vapour_out, _ = dried[hot_line]
        recovered[hot_line] = heat_recovery(
            props,
            (gas[hot_line], vapour_out, t_out),
            (gas[cold_line], vapour[cold_line], lines[cold_line].gas_celsius + KELVIN),
            lines[hot_line].medium_kg_per_h / 3600.0,
        )
    outputs = []
    for i, ln in enumerate(lines):
        t_out, vapour_out, product = dried[i]
        _, t2, heat = recovered[i]
        t1 = recovered[preheated_by[i]][0]
        duty, fuel = heater(
            table,
            gas[i],
            vapour[i],
            t1,
            ln.dryer_inlet_celsius + KELVIN,
            ln.heater_efficiency_percent / 100.0,
            ln.calorific_value_kj_per_kg,
        )
        humidity = vapour_out / gas[i]
        saturation = saturation_humidity_kg_per_kg(t_out - KELVIN)
        outputs.append(
            LineOutputs(
                dryer_outlet_celsius=t_out - KELVIN,
                dryer_outlet_humidity_g_per_kg=humidity * 1000.0,
                exhaust_celsius=t2 - KELVIN,
                preheated_celsius=t1 - KELVIN,
                heat_recovered_kw=heat / 1000.0,
                heater_duty_kw=duty,
                fuel_kg_per_h=fuel,
                product_kg_per_h=product * 3600.0,
                saturation_ratio=humidity / saturation if saturation > 0.0 else float("inf"),
            )
        )
    return tuple(outputs)
