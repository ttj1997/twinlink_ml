"""Fixed values of the workbook ``TwinLink Excel.xlsx`` and the mapping of a point to the inputs.

Every value is the one committed in docs/11 section 13.1 (the workbook is not read; nothing from
``local/`` beyond what docs/11 states). Held fixed in the study: dry-air flows 120 / 100 kg/h,
circuit flows 40 / 30 kg/h (circuit *i* belongs to exhaust *i*, docs/11 section 13.4), the
properties, calorific value 45 000 kJ/kg, heater efficiency 80 %, spray temperature 20 degC of both
dryers, 1 atm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from twinlink_ml.partner_v4.inputs import ExcelDryerInputs, ExcelInputs, ExcelMaterialInputs
from twinlink_ml.partner_v4_1.inputs import ExcelInputsV41

from .space import OperatingPoint, ambient_relative_humidity

__all__ = ["WORKBOOK", "WORKBOOK_POINT", "WorkbookConstants", "excel_inputs"]


@dataclass(frozen=True)
class WorkbookConstants:
    """The inputs of the workbook that the study does not vary (sheet units)."""

    air_kg_per_h: tuple[float, float] = (120.0, 100.0)
    """``Inputs!E3``, ``E10``: dry-air flow of dryer 1 (spray tower) and dryer 2 (fluidised bed)."""
    htm_flow_kg_per_h: tuple[float, float] = (40.0, 30.0)
    """``Inputs!E20``, ``E21``: circuit of exhaust 1 and of exhaust 2."""
    spray_celsius: tuple[float, float] = (20.0, 20.0)
    ambient_pressure_atm: float = 1.0
    material: ExcelMaterialInputs = field(
        default_factory=lambda: ExcelMaterialInputs(
            cp_gas_kj_per_kg_k=1.006,
            calorific_value_kj_per_kg=45_000.0,
            heater_efficiency_percent=80.0,
            cp_liquid_kj_per_kg_k=4.185,
            cp_solid_kj_per_kg_k=1.0,
            cp_vapour_kj_per_kg_k=1.906,
            equilibrium_moisture_g_per_kg=2.0,
            enthalpy_of_evaporation_kj_per_kg=2500.0,
            cp_heating_medium_kj_per_kg_k=4.185,
        )
    )


WORKBOOK = WorkbookConstants()

WORKBOOK_HUMIDITY_KG_PER_KG = 0.006
"""Supply-air humidity of both dryers in the workbook (6 g/kg, docs/11 section 13.1)."""

WORKBOOK_POINT = OperatingPoint(
    ambient_celsius=20.0,
    ambient_relative_humidity=ambient_relative_humidity(20.0, WORKBOOK_HUMIDITY_KG_PER_KG),
    inlet_1_celsius=300.0,
    inlet_2_celsius=150.0,
    spray_load_1_kg_per_kg=12.0 / 120.0,
    spray_load_2_kg_per_kg=3.0 / 100.0,
    solid_fraction_1=0.4,
    solid_fraction_2=0.4,
)
"""The input point of the workbook in the coordinates of the study (6 g/kg at 20 degC = the
relative humidity stored here)."""


def excel_inputs(
    point: OperatingPoint,
    htm_flow_kg_per_h: tuple[float, float] | None = None,
    constants: WorkbookConstants = WORKBOOK,
) -> ExcelInputsV41:
    """Sheet inputs of ``point`` (the varied values) and ``constants`` (everything else)."""
    y_g_per_kg = point.humidity_kg_per_kg * 1e3
    air_1, air_2 = constants.air_kg_per_h
    d1 = ExcelDryerInputs(
        air_mass_flow_kg_per_h=air_1,
        air_moisture_g_per_kg=y_g_per_kg,
        air_temperature_celsius=point.inlet_1_celsius,
        spray_mass_flow_kg_per_h=point.spray_load_1_kg_per_kg * air_1,
        spray_solid_fraction_kg_per_kg=point.solid_fraction_1,
        spray_temperature_celsius=constants.spray_celsius[0],
    )
    d2 = ExcelDryerInputs(
        air_mass_flow_kg_per_h=air_2,
        air_moisture_g_per_kg=y_g_per_kg,
        air_temperature_celsius=point.inlet_2_celsius,
        spray_mass_flow_kg_per_h=point.spray_load_2_kg_per_kg * air_2,
        spray_solid_fraction_kg_per_kg=point.solid_fraction_2,
        spray_temperature_celsius=constants.spray_celsius[1],
    )
    flows = constants.htm_flow_kg_per_h if htm_flow_kg_per_h is None else htm_flow_kg_per_h
    plant = ExcelInputs(
        d1, d2, constants.material, point.ambient_celsius, constants.ambient_pressure_atm
    )
    return ExcelInputsV41(plant, (float(flows[0]), float(flows[1])))
