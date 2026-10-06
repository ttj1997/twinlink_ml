"""Inputs of the partner's workbook ``Model_v4.xlsx`` -- in the sheet's units and in SI.

Two containers on purpose:

* :class:`ExcelInputs` holds the numbers exactly as the sheet ``Inputs`` states them
  (kg/h, g/kg, kJ, percent, atm). The as-built reproduction (:mod:`.as_built`) works on it,
  because the sheet formulas are arithmetic on exactly these numbers.
* :class:`PartnerV4Inputs` holds the same data in SI (kg/s, kg/kg, J, fraction, Pa). The
  corrected model (:mod:`.corrected`) works on it.

:meth:`ExcelInputs.to_si` is the only place where units are converted.

The workbook is confidential partner material and is never in the repository; the reader
:func:`load_excel_inputs` takes a path (``local/`` or elsewhere) and imports ``openpyxl``
only when called (optional dependency ``plant-data``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ATM_PA",
    "ExcelDryerInputs",
    "ExcelInputs",
    "ExcelMaterialInputs",
    "PartnerV4Inputs",
    "SiAmbient",
    "SiDryer",
    "SiMaterial",
    "load_excel_inputs",
]

ATM_PA = 101325.0
"""One standard atmosphere in Pa (the sheet states the pressure in atm)."""


@dataclass(frozen=True)
class ExcelDryerInputs:
    """One dryer block of the sheet ``Inputs`` (rows 3-8 and 10-15), sheet units."""

    air_mass_flow_kg_per_h: float
    air_moisture_g_per_kg: float
    air_temperature_celsius: float
    spray_mass_flow_kg_per_h: float
    spray_solid_fraction_kg_per_kg: float
    """Labelled ``gm/kg`` in the sheet; the value 0.4 is a mass fraction (kg/kg)."""
    spray_temperature_celsius: float


@dataclass(frozen=True)
class ExcelMaterialInputs:
    """Material properties of the sheet ``Inputs`` (``G3:I12``), sheet units."""

    cp_gas_kj_per_kg_k: float
    calorific_value_kj_per_kg: float
    heater_efficiency_percent: float
    cp_liquid_kj_per_kg_k: float
    cp_solid_kj_per_kg_k: float
    cp_vapour_kj_per_kg_k: float
    equilibrium_moisture_g_per_kg: float
    enthalpy_of_evaporation_kj_per_kg: float
    cp_heating_medium_kj_per_kg_k: float


@dataclass(frozen=True)
class ExcelInputs:
    """All inputs of the sheet ``Inputs`` in the sheet's own units."""

    dryer_1: ExcelDryerInputs
    dryer_2: ExcelDryerInputs
    material: ExcelMaterialInputs
    ambient_temperature_celsius: float
    ambient_pressure_atm: float

    def to_si(self) -> PartnerV4Inputs:
        """Convert to SI (kg/s, kg/kg, J, fraction, Pa); temperatures stay in degC."""
        m = self.material
        return PartnerV4Inputs(
            dryer_1=_dryer_si(self.dryer_1),
            dryer_2=_dryer_si(self.dryer_2),
            material=SiMaterial(
                cp_gas_j_per_kg_k=m.cp_gas_kj_per_kg_k * 1e3,
                lower_heating_value_j_per_kg=m.calorific_value_kj_per_kg * 1e3,
                heater_efficiency=m.heater_efficiency_percent / 100.0,
                cp_liquid_j_per_kg_k=m.cp_liquid_kj_per_kg_k * 1e3,
                cp_solid_j_per_kg_k=m.cp_solid_kj_per_kg_k * 1e3,
                cp_vapour_j_per_kg_k=m.cp_vapour_kj_per_kg_k * 1e3,
                equilibrium_moisture_kg_per_kg=m.equilibrium_moisture_g_per_kg / 1e3,
                enthalpy_of_evaporation_j_per_kg=m.enthalpy_of_evaporation_kj_per_kg * 1e3,
            ),
            ambient=SiAmbient(
                temperature_celsius=self.ambient_temperature_celsius,
                pressure_pa=self.ambient_pressure_atm * ATM_PA,
            ),
        )


@dataclass(frozen=True)
class SiDryer:
    """One dryer in SI. Humidity is relative to dry air (kg water per kg dry air)."""

    air_mass_flow_kg_per_s: float
    air_humidity_kg_per_kg: float
    air_temperature_celsius: float
    spray_mass_flow_kg_per_s: float
    spray_solid_fraction_kg_per_kg: float
    spray_temperature_celsius: float


@dataclass(frozen=True)
class SiMaterial:
    """Material properties in SI (J, kg, K); ``heater_efficiency`` is a fraction."""

    cp_gas_j_per_kg_k: float
    lower_heating_value_j_per_kg: float
    heater_efficiency: float
    cp_liquid_j_per_kg_k: float
    cp_solid_j_per_kg_k: float
    cp_vapour_j_per_kg_k: float
    equilibrium_moisture_kg_per_kg: float
    enthalpy_of_evaporation_j_per_kg: float


@dataclass(frozen=True)
class SiAmbient:
    """Ambient air: temperature in degC, pressure in Pa."""

    temperature_celsius: float
    pressure_pa: float


@dataclass(frozen=True)
class PartnerV4Inputs:
    """The inputs of the partner model in SI."""

    dryer_1: SiDryer
    dryer_2: SiDryer
    material: SiMaterial
    ambient: SiAmbient


def _dryer_si(d: ExcelDryerInputs) -> SiDryer:
    return SiDryer(
        air_mass_flow_kg_per_s=d.air_mass_flow_kg_per_h / 3600.0,
        air_humidity_kg_per_kg=d.air_moisture_g_per_kg / 1e3,
        air_temperature_celsius=d.air_temperature_celsius,
        spray_mass_flow_kg_per_s=d.spray_mass_flow_kg_per_h / 3600.0,
        spray_solid_fraction_kg_per_kg=d.spray_solid_fraction_kg_per_kg,
        spray_temperature_celsius=d.spray_temperature_celsius,
    )


def load_excel_inputs(path: str | Path) -> ExcelInputs:
    """Read the sheet ``Inputs`` of ``Model_v4.xlsx`` (needs ``openpyxl``).

    Cell addresses are fixed by the workbook layout and are the only coupling to it.
    """
    import openpyxl

    sheet = openpyxl.load_workbook(Path(path), data_only=True)["Inputs"]

    def cell(address: str) -> float:
        value = sheet[address].value
        if not isinstance(value, int | float):
            raise ValueError(f"Inputs!{address} is not numeric: {value!r}")
        return float(value)

    def dryer(first_row: int) -> ExcelDryerInputs:
        r = first_row
        return ExcelDryerInputs(
            air_mass_flow_kg_per_h=cell(f"E{r}"),
            air_moisture_g_per_kg=cell(f"E{r + 1}"),
            air_temperature_celsius=cell(f"E{r + 2}"),
            spray_mass_flow_kg_per_h=cell(f"E{r + 3}"),
            spray_solid_fraction_kg_per_kg=cell(f"E{r + 4}"),
            spray_temperature_celsius=cell(f"E{r + 5}"),
        )

    return ExcelInputs(
        dryer_1=dryer(3),
        dryer_2=dryer(10),
        material=ExcelMaterialInputs(
            cp_gas_kj_per_kg_k=cell("I4"),
            calorific_value_kj_per_kg=cell("I5"),
            heater_efficiency_percent=cell("I6"),
            cp_liquid_kj_per_kg_k=cell("I7"),
            cp_solid_kj_per_kg_k=cell("I8"),
            cp_vapour_kj_per_kg_k=cell("I9"),
            equilibrium_moisture_g_per_kg=cell("I10"),
            enthalpy_of_evaporation_kj_per_kg=cell("I11"),
            cp_heating_medium_kj_per_kg_k=cell("I12"),
        ),
        ambient_temperature_celsius=cell("E17"),
        ambient_pressure_atm=cell("E18"),
    )
