"""Formula-for-formula reproduction of ``Model_v4.xlsx`` as it is -- bugs included.

Purpose: reproduce the workbook's numbers so that deviations can be traced to a cause. Every
function mirrors one cell formula and works in the sheet's units (kg/h, g/kg, kJ, percent).
**Nothing here is physically corrected**; the corrected model is :mod:`.corrected`.

Cell references in the docstrings are to ``Model_v4.xlsx`` (state of the draft with
``#REF!`` cells). Cells that evaluate to ``#REF!`` in the workbook are not reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass

from .inputs import ExcelDryerInputs, ExcelInputs, ExcelMaterialInputs

__all__ = [
    "AsBuiltDryer",
    "dryer_as_built",
    "heater_gas_kg_per_h_as_built",
    "preheater_energy_kw_as_built",
    "preheater_outlet_temperature_as_built",
    "scenario_cells_as_built",
    "total_fuel_as_built",
]


@dataclass(frozen=True)
class AsBuiltDryer:
    """Results of one dryer block of the sheet ``Balances`` (sheet units)."""

    outlet_moisture_g_per_kg: float
    """``Balances!F4`` / ``F11``."""
    outlet_temperature_celsius: float
    """``Balances!F5`` / ``F12``."""
    product_mass_flow_kg_per_h: float
    """``Balances!E6`` / ``E13``: ``spray rate * solid fraction`` (dry solids)."""


def dryer_as_built(d: ExcelDryerInputs, m: ExcelMaterialInputs) -> AsBuiltDryer:
    """``Balances!E6, F4, F5`` (dryer 1) and ``E13, F11, F12`` (dryer 2).

    F4 = E4 + 1000 * spray * (1 - x_s * (1 + w_eq/1000)) / air
    F5 = (air * ((cp_g + Y_in/1000 * cp_v) * T_in + ((Y_in - Y_out)/1000) * h_fg)
          + spray * T_spray * ((1 - x_s) * cp_l + x_s * cp_s))
         / (air * (cp_g + Y_out/1000 * cp_v) + spray * x_s * (cp_s + w_eq/1000 * cp_l))
    """
    air, y_in, t_in = d.air_mass_flow_kg_per_h, d.air_moisture_g_per_kg, d.air_temperature_celsius
    spray, xs, t_sp = (
        d.spray_mass_flow_kg_per_h,
        d.spray_solid_fraction_kg_per_kg,
        (d.spray_temperature_celsius),
    )
    w_eq = m.equilibrium_moisture_g_per_kg
    y_out = y_in + 1000.0 * spray * (1.0 - xs * (1.0 + w_eq / 1000.0)) / air
    numerator = air * (
        (m.cp_gas_kj_per_kg_k + (y_in / 1000.0) * m.cp_vapour_kj_per_kg_k) * t_in
        + ((y_in - y_out) / 1000.0) * m.enthalpy_of_evaporation_kj_per_kg
    ) + spray * t_sp * ((1.0 - xs) * m.cp_liquid_kj_per_kg_k + xs * m.cp_solid_kj_per_kg_k)
    denominator = air * (m.cp_gas_kj_per_kg_k + (y_out / 1000.0) * m.cp_vapour_kj_per_kg_k) + (
        spray * xs * (m.cp_solid_kj_per_kg_k + (w_eq / 1000.0) * m.cp_liquid_kj_per_kg_k)
    )
    return AsBuiltDryer(
        outlet_moisture_g_per_kg=y_out,
        outlet_temperature_celsius=numerator / denominator,
        product_mass_flow_kg_per_h=xs * spray,
    )


def heater_gas_kg_per_h_as_built(
    air_kg_per_h: float,
    t_in_celsius: float,
    t_out_celsius: float,
    moisture_g_per_kg: float,
    m: ExcelMaterialInputs,
) -> float:
    """``Scenarios!E21`` (and ``G21, I21, K21, O21, E34, G34, I34, K34, M34, O34``).

    ``= air * (T_out - T_in) * (cp_g + Y * cp_v) / ((H_u * 1e6) * (eta / 100))``

    As built: ``Y`` enters in g/kg (6 instead of 0.006); ``H_u`` (already kJ/kg) is multiplied
    by 1e6. The numerator is kJ/h, so without the 1e6 the quotient would already be kg/h.
    """
    return (
        air_kg_per_h
        * (t_out_celsius - t_in_celsius)
        * (m.cp_gas_kj_per_kg_k + moisture_g_per_kg * m.cp_vapour_kj_per_kg_k)
    ) / ((m.calorific_value_kj_per_kg * 1_000_000.0) * (m.heater_efficiency_percent / 100.0))


def total_fuel_as_built(gas_heater_1: float, gas_heater_2: float) -> float:
    """``Scenarios!E36`` etc.: ``(heater 1 + heater 2) * 3600``; the sheet labels it kg/h."""
    return (gas_heater_1 + gas_heater_2) * 3600.0


def preheater_outlet_temperature_as_built(
    t_in_celsius: float,
    energy_kw: float,
    air_kg_per_h: float,
    moisture_g_per_kg: float,
    m: ExcelMaterialInputs,
    *,
    cap_celsius: float | None = None,
) -> float:
    """``Scenarios!F25`` (uncapped) and ``H25, L25, P25`` (``MIN`` with the heater target).

    ``= T_in + (Q * 1000) / (air * (cp_g + Y * cp_v))`` with ``Y`` in g/kg and ``Q`` in kW.
    As built: ``* 1000`` where kW -> kJ/h needs ``* 3600``, and ``Y`` in g/kg.
    """
    t_out = t_in_celsius + (energy_kw * 1000.0) / (
        air_kg_per_h * (m.cp_gas_kj_per_kg_k + moisture_g_per_kg * m.cp_vapour_kj_per_kg_k)
    )
    return t_out if cap_celsius is None else min(t_out, cap_celsius)


def preheater_energy_kw_as_built(
    t_in_celsius: float,
    t_out_celsius: float,
    air_kg_per_h: float,
    moisture_g_per_kg: float,
    m: ExcelMaterialInputs,
) -> float:
    """``Scenarios!E26`` (``G26, K26, O26, K30, O30``): ``air * dT * (cp_g + Y*cp_v) / 1000``.

    As built: ``/ 1000`` where kJ/h -> kW needs ``/ 3600``, and ``Y`` in g/kg.
    """
    return (
        air_kg_per_h
        * (t_out_celsius - t_in_celsius)
        * (m.cp_gas_kj_per_kg_k + moisture_g_per_kg * m.cp_vapour_kj_per_kg_k)
    ) / 1000.0


def scenario_cells_as_built(inputs: ExcelInputs) -> dict[str, float]:
    """The finished cells of the sheet ``Scenarios`` (rows 21, 34, 36) as the sheet computes.

    Only cells that are numbers in the workbook; cells that are ``#REF!`` or empty are absent.
    Heater inlet/outlet temperatures are taken from the sheet cells as they are:

    * cases 1-3: ``E20 = F12``, ``G20 = H12`` refer to empty cells -> 0; ``I20``/``J20`` are
      empty -> 0 (so ``I21 = 0``),
    * cases 4 and 6: 15 -> 300 degC (``K20 = O20 = Inputs!E17``),
    * case 5: ``M18`` holds a bare ``=``, ``M21`` is empty,
    * heater 2: cases 3 and 5 are 15 -> 300 degC (``I33 = M33 = Inputs!E17``); the others
      depend on ``#REF!``.
    """
    m = inputs.material
    d1, d2 = inputs.dryer_1, inputs.dryer_2
    t_amb = inputs.ambient_temperature_celsius
    t_target_1, t_target_2 = d1.air_temperature_celsius, d2.air_temperature_celsius
    y1 = d1.air_moisture_g_per_kg  # E19 = E11 = Inputs!E4, g/kg
    y2 = d2.air_moisture_g_per_kg  # E32 = E24 = Inputs!E11, g/kg
    air1, air2 = d1.air_mass_flow_kg_per_h, d2.air_mass_flow_kg_per_h

    cells: dict[str, float] = {}
    # Row 21, heater 1. (inlet, outlet) per column as the sheet has them.
    heater_1 = {
        "E": (0.0, t_target_1),
        "G": (0.0, t_target_1),
        "I": (0.0, 0.0),
        "K": (t_amb, t_target_1),
        "O": (t_amb, t_target_1),
    }
    for col, (t_in, t_out) in heater_1.items():
        cells[f"{col}21"] = heater_gas_kg_per_h_as_built(air1, t_in, t_out, y1, m)
    # Row 34, heater 2: only the columns without a #REF! dependency.
    for col in ("I", "M"):
        cells[f"{col}34"] = heater_gas_kg_per_h_as_built(air2, t_amb, t_target_2, y2, m)
    # Row 36: (row 21 + row 34) * 3600; an empty row-21 cell counts as 0 (case 5).
    cells["I36"] = total_fuel_as_built(cells["I21"], cells["I34"])
    cells["M36"] = total_fuel_as_built(0.0, cells["M34"])
    return cells
