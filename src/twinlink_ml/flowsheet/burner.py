"""The burner unit: the gas heater that brings the process gas to the dryer's inlet temperature.

The duty is ``C (T_target - T_in)`` with ``C = M (cp_air + Y cp_vapour)`` of the gas properties in
use. The fuel follows from the duty with the efficiency and the heating value of the sheet; the
switch ``gas_formula`` decides how:

``"partner"``
    the formula of ``Scenarios`` row 26 / 39 **as built**:
    ``air * (T_out - T_in) * (cp_g + Y_gkg cp_v) / (H_u * eta_percent / 1000)`` with ``Y`` in g/kg
    and the efficiency in percent: the ``1000`` sits at the efficiency (``/1000`` instead of
    ``/100``: factor 10) and the moisture enters in g/kg (factor 10.89 for 6 g/kg and the sheet's
    ``cp``). Total factor 108.9 too high (docs/11, 7.3).
``"physical"``
    corrected: ``C (T_target - T_in) / (eta H_u)`` in kg/kg (the formula of the equation document,
    3.2).

The corrected fuel is always computed too (``fuel_corrected_kg_per_s``): the ranking of the
scenarios and the CO2 are never taken from the as-built formula.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import SiMaterial

from .base import Assumption, Unit
from .thermo import GasProps, capacity

__all__ = ["Burner", "BurnerConfig", "BurnerInputs", "BurnerOut"]


@dataclass(frozen=True)
class BurnerConfig:
    """Material (efficiency, heating value), gas properties in use and the gas-formula switch."""

    material: SiMaterial
    props: GasProps
    gas_formula: Assumption = "partner"
    name: str = "burner"


@dataclass(frozen=True)
class BurnerInputs:
    """Process gas entering the burner and the set point it has to reach."""

    gas: c.Stream
    """Process gas leaving the pre-heating train."""
    target_celsius: float


@dataclass(frozen=True)
class BurnerOut:
    """Duty and fuel flow of one burner."""

    duty_w: float
    """Thermal duty ``C (T_target - T_in)``, not negative."""
    fuel_kg_per_s: float
    """With the formula the switch selects."""
    fuel_corrected_kg_per_s: float
    """Always the corrected formula (the basis of ranking and CO2)."""


class Burner(Unit[BurnerInputs, BurnerOut]):
    """Gas heater of one sink."""

    def __init__(self, config: BurnerConfig) -> None:
        self.config = config
        self.name = config.name

    def assumptions(self) -> dict[str, Assumption]:
        return {"gas_formula": self.config.gas_formula}

    def fuel_power_w(self, fuel_kg_per_s: float) -> float:
        """Chemical power of a fuel flow, ``m H_u``."""
        return fuel_kg_per_s * self.config.material.lower_heating_value_j_per_kg

    def solve(self, inputs: BurnerInputs) -> BurnerOut:
        m, p = self.config.material, self.config.props
        gas = inputs.gas
        t_in, t_out = gas.temperature_celsius, inputs.target_celsius
        duty = max(capacity(gas, p) * (t_out - t_in), 0.0)
        corrected = duty / (m.heater_efficiency * m.lower_heating_value_j_per_kg)
        if self.config.gas_formula == "physical":
            return BurnerOut(duty, corrected, corrected)
        # as built, in the sheet's units (kg/h, g/kg, kJ, percent); the result back in kg/s
        air_kg_per_h = gas.dry_air_kg_per_s * 3600.0
        as_built_kg_per_h = (
            air_kg_per_h
            * (t_out - t_in)
            * (p.cp_air / 1e3 + gas.humidity_kg_per_kg * 1e3 * p.cp_vapour / 1e3)
        ) / (m.lower_heating_value_j_per_kg / 1e3 * m.heater_efficiency * 100.0 / 1000.0)
        return BurnerOut(duty, as_built_kg_per_h / 3600.0, corrected)
