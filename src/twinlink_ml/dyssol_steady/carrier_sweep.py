"""Cost-optimal flow of the medium of the heat recovery of one line (descriptive study).

Design: ``docs/49_carrier_sweep.md``. One line with its own heat recovery is evaluated with the
rebuild of the stationary Dyssol units (:func:`twinlink_ml.dyssol_steady.reference.evaluate`) over
the flow of the heat-transfer medium, at several outside temperatures:

    cost(m) = c_fuel * fuel(m) * H_u + c_el * P_pump(m)          [EUR/h]

The units have ideal exchangers, so the recovered heat only grows with the medium flow; an
optimum exists only through the pump. The pump is a PLACEHOLDER: pressure drop quadratic in the
flow through a reference point, efficiency and density as in
:class:`twinlink_ml.flowsheet.auxiliary.AuxiliaryConfig` (:class:`PumpLaw`).

The inputs of the line are those of a delivered sheet; they are read from a local file (template
``conf/dyssol_steady/carrier_sweep_local.yaml.example``) and never enter the repository. The
sheet's gas flow is the dry gas (Befund 130).

*Over-saturation.* A point is over-saturated when the gas holds more water than saturated air at
1 atm at the dryer outlet or at the exhaust after the heat recovery (the units have neither a
saturation limit nor condensation). Every scanned point carries this flag.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize_scalar

from twinlink_ml.flowsheet.auxiliary import pump_power_w
from twinlink_ml.flowsheet.validators import MOLAR_MASS_RATIO, saturation_humidity_kg_per_kg
from twinlink_ml.partner_v4.inputs import ATM_PA

from .materials import DyssolProperties
from .reference import LineInputs, LineOutputs, evaluate

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "SHEET_FIELDS",
    "CostCurve",
    "Prices",
    "PumpLaw",
    "SweepConfig",
    "cost_curve",
    "fresh_gas_at",
    "fresh_gas_saturation_g_per_kg",
    "line_inputs",
    "load_config",
    "optimum",
    "recompute",
    "saturation_ratios",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "dyssol_steady" / "carrier_sweep.yaml"

SHEET_FIELDS: tuple[str, ...] = (
    "gas_kg_per_s",
    "gas_moisture_kg_per_s",
    "gas_celsius",
    "dryer_inlet_celsius",
    "spray_kg_per_s",
    "spray_solid_kg_per_s",
    "spray_celsius",
    "heater_efficiency_percent",
    "calorific_value_kj_per_kg",
    "equilibrium_moisture_g_per_kg",
)
"""Inputs of one line as the sheet gives them (flows in kg/s; the solid as a mass flow)."""


# ----------------------------------------------------------------------------- config
@dataclass(frozen=True)
class PumpLaw:
    """Circulation pump of one medium circuit (PLACEHOLDER, no source).

    ``dp(m) = reference_pressure_drop_pa * factor * (m / reference_flow_kg_per_h)**2`` and the
    shaft power ``m dp / (rho eta)`` of :func:`twinlink_ml.flowsheet.auxiliary.pump_power_w`,
    i.e. cubic in the flow.
    """

    reference_pressure_drop_pa: float
    reference_flow_kg_per_h: float
    efficiency: float
    density_kg_per_m3: float

    def __post_init__(self) -> None:
        if min(self.reference_pressure_drop_pa, self.reference_flow_kg_per_h) <= 0.0:
            raise ValueError("the reference point of the pump must be positive")
        if not 0.0 < self.efficiency <= 1.0 or self.density_kg_per_m3 <= 0.0:
            raise ValueError("pump efficiency in (0, 1] and a positive density are required")

    def power_kw(self, medium_kg_per_h: float, pressure_factor: float = 1.0) -> float:
        """Electric power of the pump [kW] at a medium flow."""
        ratio = medium_kg_per_h / self.reference_flow_kg_per_h
        pressure = self.reference_pressure_drop_pa * pressure_factor * ratio**2
        return (
            pump_power_w(
                medium_kg_per_h / 3600.0, pressure, self.density_kg_per_m3, self.efficiency
            )
            / 1000.0
        )


@dataclass(frozen=True)
class Prices:
    """Prices and CO2 factors (placeholders of Befund 115 / Befund 101)."""

    gas_eur_per_kwh: float
    co2_eur_per_t: float
    electricity_eur_per_kwh: float
    gas_co2_kg_per_kwh: float
    electricity_co2_kg_per_kwh: float

    @property
    def fuel_eur_per_kwh(self) -> float:
        """Price of the fuel energy, CO2 price included."""
        return self.gas_eur_per_kwh + self.co2_eur_per_t * self.gas_co2_kg_per_kwh / 1000.0


@dataclass(frozen=True)
class SweepConfig:
    """Settings of the study (``conf/dyssol_steady/carrier_sweep.yaml``)."""

    version: int
    flow_max_kg_per_h: float
    flow_step_kg_per_h: float
    ambient_celsius: tuple[float, ...]
    neighbour_factors: tuple[float, ...]
    pressure_factors: tuple[float, ...]
    pump: PumpLaw
    prices: Prices
    raw: Mapping[str, Any] = field(repr=False)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the parsed YAML (canonical JSON)."""
        text = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def flows(self) -> np.ndarray:
        """Scanned medium flows [kg/h], zero and the upper bound included."""
        steps = round(self.flow_max_kg_per_h / self.flow_step_kg_per_h)
        if not math.isclose(steps * self.flow_step_kg_per_h, self.flow_max_kg_per_h):
            raise ValueError("the flow step must divide the upper bound")
        return np.linspace(0.0, self.flow_max_kg_per_h, steps + 1)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> SweepConfig:
    """Read and check the study YAML."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scan, pump, prices = raw["scan"], raw["pump"], raw["prices"]
    factors = tuple(float(f) for f in pump["pressure_factors"])
    if 1.0 not in factors:
        raise ValueError("pressure_factors must contain 1.0 (the placeholder itself)")
    return SweepConfig(
        version=int(raw["version"]),
        flow_max_kg_per_h=float(scan["flow_max_kg_per_h"]),
        flow_step_kg_per_h=float(scan["flow_step_kg_per_h"]),
        ambient_celsius=tuple(float(t) for t in scan["ambient_celsius"]),
        neighbour_factors=tuple(float(f) for f in scan["neighbour_factors"]),
        pressure_factors=factors,
        pump=PumpLaw(
            reference_pressure_drop_pa=float(pump["reference_pressure_drop_pa"]),
            reference_flow_kg_per_h=float(pump["reference_flow_kg_per_h"]),
            efficiency=float(pump["efficiency"]),
            density_kg_per_m3=float(pump["density_kg_per_m3"]),
        ),
        prices=Prices(
            gas_eur_per_kwh=float(prices["gas_eur_per_kwh"]),
            co2_eur_per_t=float(prices["co2_eur_per_t"]),
            electricity_eur_per_kwh=float(prices["electricity_eur_per_kwh"]),
            gas_co2_kg_per_kwh=float(prices["gas_co2_kg_per_kwh"]),
            electricity_co2_kg_per_kwh=float(prices["electricity_co2_kg_per_kwh"]),
        ),
        raw=raw,
    )


# ----------------------------------------------------------------------------- line
def line_inputs(sheet: Mapping[str, Any], medium_kg_per_h: float = 0.0) -> LineInputs:
    """Inputs of the rebuild from the input block of the sheet (its gas flow is the dry gas)."""
    if set(sheet) != set(SHEET_FIELDS):
        raise ValueError(f"the input block needs exactly the fields {SHEET_FIELDS}")
    values: dict[str, float] = {}
    for key in SHEET_FIELDS:
        value = sheet[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{key}: expected a number, found {value!r}")
        values[key] = float(value)
    return LineInputs(
        dry_gas_kg_per_h=values["gas_kg_per_s"] * 3600.0,
        gas_humidity_g_per_kg=1000.0 * values["gas_moisture_kg_per_s"] / values["gas_kg_per_s"],
        gas_celsius=values["gas_celsius"],
        dryer_inlet_celsius=values["dryer_inlet_celsius"],
        spray_kg_per_h=values["spray_kg_per_s"] * 3600.0,
        solid_fraction=values["spray_solid_kg_per_s"] / values["spray_kg_per_s"],
        spray_celsius=values["spray_celsius"],
        medium_kg_per_h=medium_kg_per_h,
        equilibrium_moisture_g_per_kg=values["equilibrium_moisture_g_per_kg"],
        heater_efficiency_percent=values["heater_efficiency_percent"],
        calorific_value_kj_per_kg=values["calorific_value_kj_per_kg"],
    )


def recompute(props: DyssolProperties, inputs: LineInputs) -> LineOutputs:
    """One line with its own heat recovery through the rebuild."""
    return evaluate(props, (inputs,), (0,))[0]


def saturation_ratios(outputs: LineOutputs) -> tuple[float, float]:
    """Humidity over the saturation humidity (1 atm) at the dryer outlet and at the exhaust."""
    saturation = saturation_humidity_kg_per_kg(outputs.exhaust_celsius)
    humidity = outputs.dryer_outlet_humidity_g_per_kg / 1000.0
    exhaust = humidity / saturation if saturation > 0.0 else math.inf
    return outputs.saturation_ratio, exhaust


def fresh_gas_saturation_g_per_kg(ambient_celsius: float) -> float:
    """Saturation humidity of the outside air [g/kg dry air] at 1 atm.

    From 0 degC on the repository's function (IAPWS-IF97), which is held at its value of 0 degC
    below; there the Magnus form over ice, ``611.2 Pa exp(22.46 t / (272.62 + t))`` (WMO guide
    No. 8), is used instead.
    """
    if ambient_celsius >= 0.0:
        return 1000.0 * saturation_humidity_kg_per_kg(ambient_celsius)
    pressure = 611.2 * math.exp(22.46 * ambient_celsius / (272.62 + ambient_celsius))
    return 1000.0 * MOLAR_MASS_RATIO * pressure / (ATM_PA - pressure)


def fresh_gas_at(inputs: LineInputs, ambient_celsius: float) -> tuple[LineInputs, bool]:
    """The line with fresh gas at an outside temperature, and whether the humidity was capped.

    The humidity of the fresh gas stays at the sheet's value but not above the saturation
    humidity at the outside temperature (:func:`fresh_gas_saturation_g_per_kg`).
    """
    limit = fresh_gas_saturation_g_per_kg(ambient_celsius)
    capped = inputs.gas_humidity_g_per_kg > limit
    return (
        replace(
            inputs,
            gas_celsius=ambient_celsius,
            gas_humidity_g_per_kg=min(inputs.gas_humidity_g_per_kg, limit),
        ),
        capped,
    )


# ----------------------------------------------------------------------------- cost
@dataclass(frozen=True)
class CostCurve:
    """Fuel, recovered heat and over-saturation of one line over the medium flow."""

    flows_kg_per_h: np.ndarray
    fuel_kw: np.ndarray
    """Chemical power of the fuel."""
    heat_recovered_kw: np.ndarray
    oversaturated: np.ndarray
    inputs: LineInputs
    pump: PumpLaw
    prices: Prices
    properties: DyssolProperties = field(repr=False)

    def fuel_kw_at(self, medium_kg_per_h: float) -> float:
        """Chemical power of the fuel at any medium flow (closed formula, not interpolated)."""
        outputs = recompute(self.properties, replace(self.inputs, medium_kg_per_h=medium_kg_per_h))
        return outputs.fuel_kg_per_h * self.inputs.calorific_value_kj_per_kg / 3600.0

    def cost_eur_per_h(self, medium_kg_per_h: float, pressure_factor: float = 1.0) -> float:
        """Fuel plus pump electricity at any medium flow."""
        return self.prices.fuel_eur_per_kwh * self.fuel_kw_at(
            medium_kg_per_h
        ) + self.prices.electricity_eur_per_kwh * self.pump.power_kw(
            medium_kg_per_h, pressure_factor
        )

    def costs_eur_per_h(self, pressure_factor: float = 1.0) -> np.ndarray:
        """Fuel plus pump electricity on the grid."""
        pump = np.asarray([self.pump.power_kw(m, pressure_factor) for m in self.flows_kg_per_h])
        return np.asarray(
            self.prices.fuel_eur_per_kwh * self.fuel_kw + self.prices.electricity_eur_per_kwh * pump
        )

    def co2_kg_per_h(self, medium_kg_per_h: float, pressure_factor: float = 1.0) -> float:
        """CO2 of the fuel and of the pump electricity at any medium flow."""
        return self.prices.gas_co2_kg_per_kwh * self.fuel_kw_at(
            medium_kg_per_h
        ) + self.prices.electricity_co2_kg_per_kwh * self.pump.power_kw(
            medium_kg_per_h, pressure_factor
        )


def cost_curve(
    props: DyssolProperties,
    inputs: LineInputs,
    flows_kg_per_h: np.ndarray,
    pump: PumpLaw,
    prices: Prices,
) -> CostCurve:
    """Evaluate the rebuild on a grid of medium flows."""
    fuel, heat, over = [], [], []
    for medium in flows_kg_per_h:
        outputs = recompute(props, replace(inputs, medium_kg_per_h=float(medium)))
        fuel.append(outputs.fuel_kg_per_h * inputs.calorific_value_kj_per_kg / 3600.0)
        heat.append(outputs.heat_recovered_kw)
        over.append(max(saturation_ratios(outputs)) > 1.0)
    return CostCurve(
        flows_kg_per_h=np.asarray(flows_kg_per_h, dtype=float),
        fuel_kw=np.asarray(fuel),
        heat_recovered_kw=np.asarray(heat),
        oversaturated=np.asarray(over),
        inputs=inputs,
        pump=pump,
        prices=prices,
        properties=props,
    )


def optimum(
    curve: CostCurve, pressure_factor: float, neighbour_factors: Sequence[float]
) -> dict[str, Any]:
    """Cost-optimal medium flow and the cost next to it.

    The optimum of the grid is refined between its two neighbouring grid points with the closed
    formula (bounded scalar minimisation); an optimum at an end of the grid stays there.

    ``neighbours``: cost at ``factor x optimum`` (evaluated with the closed formula, also above
    the scanned range) as an increase over the optimal cost, in percent of the optimal cost and
    in percent of the saving the optimum makes against no medium flow.
    """
    costs = curve.costs_eur_per_h(pressure_factor)
    best = int(np.argmin(costs))
    flow = float(curve.flows_kg_per_h[best])
    cost = float(costs[best])
    if 0 < best < len(costs) - 1:
        refined = minimize_scalar(
            lambda m: curve.cost_eur_per_h(float(m), pressure_factor),
            bounds=(float(curve.flows_kg_per_h[best - 1]), float(curve.flows_kg_per_h[best + 1])),
            method="bounded",
            options={"xatol": 1e-6 * float(curve.flows_kg_per_h[-1])},
        )
        if float(refined.fun) < cost:
            flow, cost = float(refined.x), float(refined.fun)
    saving = float(costs[0]) - cost
    upper = float(curve.flows_kg_per_h[-1])
    neighbours = []
    for factor in neighbour_factors:
        other = curve.cost_eur_per_h(factor * flow, pressure_factor)
        neighbours.append(
            {
                "factor": factor,
                "flow_kg_per_h": factor * flow,
                "above_scanned_range": factor * flow > upper,
                "cost_eur_per_h": other,
                "increase_percent_of_cost": 100.0 * (other - cost) / cost,
                "increase_percent_of_saving": (
                    100.0 * (other - cost) / saving if saving > 0.0 else None
                ),
            }
        )
    co2_none = curve.co2_kg_per_h(0.0, pressure_factor)
    at_optimum = recompute(curve.properties, replace(curve.inputs, medium_kg_per_h=flow))
    heat = at_optimum.heat_recovered_kw
    pump_kw = curve.pump.power_kw(flow, pressure_factor)
    return {
        "pressure_factor": pressure_factor,
        "flow_kg_per_h": flow,
        "flow_share_of_scan": flow / upper,
        "at_upper_bound": best == len(costs) - 1,
        "at_zero": best == 0,
        "cost_eur_per_h": cost,
        "cost_no_flow_eur_per_h": float(costs[0]),
        "saving_percent_of_no_flow_cost": 100.0 * saving / float(costs[0]),
        "pump_kw": pump_kw,
        "pump_share_of_cost_percent": 100.0 * curve.prices.electricity_eur_per_kwh * pump_kw / cost,
        "heat_recovered_kw": heat,
        "heat_recovered_share_of_scan_max": (
            heat / float(curve.heat_recovered_kw.max())
            if curve.heat_recovered_kw.max() > 0.0
            else None
        ),
        "co2_change_percent_of_no_flow": 100.0
        * (curve.co2_kg_per_h(flow, pressure_factor) - co2_none)
        / co2_none,
        "oversaturated_at_optimum": max(saturation_ratios(at_optimum)) > 1.0,
        "neighbours": neighbours,
    }
