"""Deviation list: workbook cell, workbook value, rebuilt value, cause, hand calculation.

"Rebuilt value" is the value of :mod:`.corrected` for the same cell with the *same inputs the
cell itself uses* (so a wrong temperature reference stays visible as its own item and is not
mixed into the arithmetic items). Where a cell is ``#REF!`` the workbook's formula is
evaluated at a stated test value and the item says so.

The workbook is confidential partner material and is read from a path; this module contains
no workbook content apart from cell addresses, the formulas' structure and numbers that the
corrected formulas reproduce.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from . import as_built as ab
from . import cascade as k
from . import corrected as c
from .inputs import ExcelInputs
from .workbook import CellInfo

__all__ = ["CELLS_TO_READ", "Deviation", "build_deviations"]

_BALANCE_CELLS = ["Balances!F4", "Balances!F5", "Balances!F11", "Balances!F12"]
_HEATER_1 = {"E": "F", "G": "H", "I": "J", "K": "L", "O": "P"}
_HEATER_2 = {"I": "J", "M": "N"}
_ANTOINE_CELLS = [f"Balances!{col}{row}" for col in "JK" for row in (4, 5, 6)]

CELLS_TO_READ: list[str] = (
    _BALANCE_CELLS
    + [f"Scenarios!{col}{row}" for col in "EGIKMO" for row in (18, 20, 21, 33, 34, 36)]
    + [f"Scenarios!{col}20" for col in "FHJLNP"]
    + [f"Scenarios!{col}33" for col in "FHJLNP"]
    + [f"Scenarios!{col}{row}" for col in "EGIKMO" for row in (13, 17, 25, 26, 29, 30)]
    + [f"Scenarios!{col}25" for col in "FHJLNP"]
    + [f"Scenarios!{col}29" for col in "FHJLNP"]
    + _ANTOINE_CELLS
    + ["Inputs!D7", "Inputs!D14", "Scenarios!D6", "Scenarios!D7", "Scenarios!M18"]
    + [f"Scenarios!{col}12" for col in "FHJN"]
)


@dataclass(frozen=True)
class Deviation:
    """One line of the deviation list."""

    item: str
    cell: str
    quantity: str
    unit: str
    workbook_value: float | str | None
    rebuilt_value: float | str | None
    cause: str
    hand_calculation: str
    formula: str | None = None

    @property
    def ratio(self) -> float | None:
        """``workbook / rebuilt`` for numbers, else ``None``."""
        a, b = self.workbook_value, self.rebuilt_value
        if isinstance(a, int | float) and isinstance(b, int | float) and b not in (0, 0.0):
            return a / b
        return None

    def as_dict(self) -> dict[str, object]:
        """Dictionary for the JSON artifact (ratio included)."""
        return {**asdict(self), "ratio": self.ratio}


def _num(cell: CellInfo | None) -> float:
    """Numeric value of a cell; an empty cell counts as 0 as in Excel arithmetic."""
    value = None if cell is None else cell.value
    return float(value) if isinstance(value, int | float) else 0.0


def _gas_hand(air: float, t_in: float, t_out: float, y: float, cp: float, hu: float) -> str:
    cpm = cp + y * 2.0
    p_kw = air / 3600.0 * cpm * (t_out - t_in)
    return (
        f"P = {air:g}/3600 kg/s * ({cp:g} + {y:g}*2) kJ/(kg K) * ({t_out:g} - {t_in:g}) K = "
        f"{p_kw:.4f} kW; m_gas = P / (0.8 * {hu:g} kJ/kg) = {p_kw / (0.8 * hu):.4e} kg/s = "
        f"{p_kw / (0.8 * hu) * 3600:.5f} kg/h"
    )


def build_deviations(excel: ExcelInputs, cells: dict[str, CellInfo]) -> list[Deviation]:
    """Build the deviation list for ``excel`` against the workbook cells ``cells``.

    ``cells`` must contain every address of :data:`CELLS_TO_READ` (see
    :func:`~.workbook.read_cells`).
    """
    si = excel.to_si()
    m, ms = excel.material, si.material
    out: list[Deviation] = []

    # --- 1. Dryer balances: verified, no deviation ------------------------------------
    for address, dryer_si in zip(
        _BALANCE_CELLS, (si.dryer_1, si.dryer_1, si.dryer_2, si.dryer_2), strict=True
    ):
        res = c.dryer_balance(dryer_si, ms)
        is_moisture = address.endswith(("F4", "F11"))
        value = (
            res.outlet_humidity_kg_per_kg * 1e3 if is_moisture else res.outlet_temperature_celsius
        )
        resid = c.dryer_enthalpy_residual_w(dryer_si, ms, res)
        out.append(
            Deviation(
                item="B1" if is_moisture else "B2",
                cell=address,
                quantity="outlet moisture" if is_moisture else "outlet temperature",
                unit="g/kg" if is_moisture else "degC",
                workbook_value=cells[address].value,
                rebuilt_value=value,
                cause="none -- verified (rebuild identical; independent enthalpy balance closes "
                f"to {resid:.1e} W)",
                hand_calculation=(
                    "Y_out = 6 + 1000*5*(1-0.4*1.002)/50 = 6 + 59.92 = 65.92 g/kg; "
                    "T_out = (50*((1.2+0.006*2)*300 - 0.05992*2500) + 5*15*(0.6*4.2+0.4*1)) / "
                    "(50*(1.2+0.06592*2) + 2*(1+0.002*4.2)) = 10909 / 68.6088 = 159.003 degC"
                    if is_moisture or address.endswith(("F5", "F12"))
                    else ""
                ),
                formula=cells[address].formula,
            )
        )

    # --- 2. Gas formula (rows 21 and 34), total (row 36) ------------------------------
    air1, y1 = excel.dryer_1.air_mass_flow_kg_per_h, excel.dryer_1.air_moisture_g_per_kg
    air2, y2 = excel.dryer_2.air_mass_flow_kg_per_h, excel.dryer_2.air_moisture_g_per_kg
    hu, cp = m.calorific_value_kj_per_kg, m.cp_gas_kj_per_kg_k
    rebuilt_gas: dict[str, float] = {}
    for col, out_col in _HEATER_1.items():
        t_in, t_out = (
            _num(cells.get(f"Scenarios!{col}20")),
            _num(cells.get(f"Scenarios!{out_col}20")),
        )
        stream = c.Stream(
            si.dryer_1.air_mass_flow_kg_per_s, si.dryer_1.air_humidity_kg_per_kg, t_in
        )
        power = c.heater_power_w(stream, t_in, t_out, ms)
        rebuilt = c.gas_mass_flow_kg_per_s(power, ms) * 3600.0
        rebuilt_gas[f"{col}21"] = rebuilt
        cell = cells[f"Scenarios!{col}21"]
        out.append(
            Deviation(
                item=f"G-{col}21",
                cell=f"Scenarios!{col}21",
                quantity=(
                    f"gas, heater 1, case {'EGIKMO'.index(col) + 1}, "
                    f"inlet {t_in:g} -> {t_out:g} degC"
                ),
                unit="kg/h",
                workbook_value=cell.value,
                rebuilt_value=rebuilt,
                cause=_gas_cause(t_in, t_out),
                hand_calculation=_gas_hand(air1, t_in, t_out, y1 / 1e3, cp, hu),
                formula=cell.formula,
            )
        )
    for col, out_col in _HEATER_2.items():
        t_in, t_out = (
            _num(cells.get(f"Scenarios!{col}33")),
            _num(cells.get(f"Scenarios!{out_col}33")),
        )
        stream = c.Stream(
            si.dryer_2.air_mass_flow_kg_per_s, si.dryer_2.air_humidity_kg_per_kg, t_in
        )
        rebuilt = c.gas_mass_flow_kg_per_s(c.heater_power_w(stream, t_in, t_out, ms), ms) * 3600.0
        rebuilt_gas[f"{col}34"] = rebuilt
        cell = cells[f"Scenarios!{col}34"]
        out.append(
            Deviation(
                item=f"G-{col}34",
                cell=f"Scenarios!{col}34",
                quantity=(
                    f"gas, heater 2, case {'EGIKMO'.index(col) + 1}, "
                    f"inlet {t_in:g} -> {t_out:g} degC"
                ),
                unit="kg/h",
                workbook_value=cell.value,
                rebuilt_value=rebuilt,
                cause=_gas_cause(t_in, t_out),
                hand_calculation=_gas_hand(air2, t_in, t_out, y2 / 1e3, cp, hu),
                formula=cell.formula,
            )
        )
    cp_factor = (cp + y1 * m.cp_vapour_kj_per_kg_k) / (cp + y1 / 1e3 * m.cp_vapour_kj_per_kg_k)
    for col in ("I", "M"):
        cell = cells[f"Scenarios!{col}36"]
        rebuilt = rebuilt_gas.get(f"{col}21", 0.0) + rebuilt_gas[f"{col}34"]
        out.append(
            Deviation(
                item=f"T-{col}36",
                cell=f"Scenarios!{col}36",
                quantity=f"total fuel, case {'EGIKMO'.index(col) + 1}",
                unit="kg/h",
                workbook_value=cell.value,
                rebuilt_value=rebuilt,
                cause=(
                    "three unit errors multiply: moisture entered as 6 (g/kg) instead of 0.006 "
                    f"in the heat capacity (factor {cp_factor:.4f} too high), calorific value "
                    "(already kJ/kg) times 1e6 (factor 1e-6), and kg/h times 3600 (factor 3600): "
                    f"{cp_factor:.4f} * 1e-6 * 3600 = {cp_factor * 1e-6 * 3600:.4f}, i.e. the "
                    f"sheet is {1 / (cp_factor * 1e-6 * 3600):.1f} times too low. "
                    + (
                        "Row 21 of this case is 0 because heater 1's temperatures are empty cells."
                        if col == "I"
                        else "Row 21 of this case does not exist (M18 is a bare '=')."
                    )
                ),
                hand_calculation=f"{rebuilt:.5f} kg/h = row 21 "
                f"({rebuilt_gas.get(f'{col}21', 0.0):.5f}) "
                f"+ row 34 ({rebuilt_gas[f'{col}34']:.5f}), both in kg/h, no factor 3600",
                formula=cell.formula,
            )
        )

    # --- 3. Heat-recovery rows 25-30 (cells are #REF!; formulas evaluated at Q = 1 kW) --
    q_kw = 1.0
    t_rise_sheet = ab.preheater_outlet_temperature_as_built(0.0, q_kw, air2, y2, m)
    stream2 = c.Stream(si.dryer_2.air_mass_flow_kg_per_s, si.dryer_2.air_humidity_kg_per_kg, 0.0)
    t_rise_si = q_kw * 1e3 / c.heat_capacity_rate_w_per_k(stream2, ms)
    out.append(
        Deviation(
            item="R-F25",
            cell="Scenarios!F25 (and H25, L25, P25, L29, P29)",
            quantity="temperature rise for a recovered heat of 1 kW (the cells are #REF!)",
            unit="K",
            workbook_value=t_rise_sheet,
            rebuilt_value=t_rise_si,
            cause=(
                "kW is converted with * 1000 where kW -> kJ/h needs * 3600 (factor 3.6 too low), "
                "and the moisture enters as g/kg in the heat capacity (factor "
                f"{cp_factor:.4f}); product {3.6 * cp_factor:.2f}"
            ),
            hand_calculation=(
                f"sheet: dT = 1*1000 / (50*(1.2+6*2)) = {t_rise_sheet:.4f} K; "
                f"correct: dT = 1*3600 / (50*(1.2+0.006*2)) = {t_rise_si:.3f} K"
            ),
            formula=(cells["Scenarios!F25"].formula),
        )
    )
    e26_sheet_kw = ab.preheater_energy_kw_as_built(0.0, t_rise_sheet, air2, y2, m)
    e26_si_kw = c.heat_capacity_rate_w_per_k(stream2, ms) * t_rise_sheet / 1e3
    out.append(
        Deviation(
            item="R-E26",
            cell="Scenarios!E26 (and G26, K26, O26, K30, O30)",
            quantity="heat reported for the temperature rise the sheet itself computes (1 kW case)",
            unit="kW",
            workbook_value=e26_sheet_kw,
            rebuilt_value=e26_si_kw,
            cause=(
                "the two recovery formulas use the same wrong heat capacity and the same wrong "
                "factor, so the reported energy returns exactly the input (1 kW) while the "
                "temperature that goes into the heater (E33 = F25) corresponds to only "
                f"{e26_si_kw:.4f} kW: energy bookkeeping and heater input disagree"
            ),
            hand_calculation=(
                f"sheet: E26 = 50*{t_rise_sheet:.4f}*(1.2+6*2)/1000 = {e26_sheet_kw:.4f} kW; "
                f"correct for that dT: 50/3600*{t_rise_sheet:.4f}*1.212 = {e26_si_kw:.5f} kW"
            ),
            formula=cells["Scenarios!E26"].formula,
        )
    )

    # --- 4. c_p of the gas -------------------------------------------------------------
    cp_ref = 1.0
    p = (
        si.dryer_1.air_mass_flow_kg_per_s
        * (cp_ref * 1e3 + si.dryer_1.air_humidity_kg_per_kg * ms.cp_vapour_j_per_kg_k)
        * 285.0
    )
    gas = c.gas_mass_flow_kg_per_s(p, ms) * 3600.0
    out.append(
        Deviation(
            item="P-cp",
            cell="Inputs!I4",
            quantity="gas for 50 kg/h air, 15 -> 300 degC, with c_p,gas as a question",
            unit="kg/h",
            workbook_value=None,
            rebuilt_value=gas,
            cause=(
                "c_p of 1.2 kJ/(kg K) is high for air (about 1.0 over 15-300 degC); this is a "
                "question to the partner, the corrected model keeps 1.2. Rebuilt with 1.2: "
                f"{rebuilt_gas['K21']:.5f} kg/h; with 1.0: {gas:.5f} kg/h "
                f"({(gas / rebuilt_gas['K21'] - 1) * 100:+.1f} %)"
            ),
            hand_calculation=_gas_hand(air1, 15.0, 300.0, y1 / 1e3, cp_ref, hu),
        )
    )

    # --- 5. Antoine coefficients -------------------------------------------------------
    correct = {
        "J4": 10.1962,
        "J5": 1730.63,
        "J6": 233.426,
        "K4": 10.2651,
        "K5": 1810.94,
        "K6": 244.485,
    }
    for key, target in correct.items():
        cell = cells[f"Balances!{key}"]
        is_text = isinstance(cell.value, str)
        out.append(
            Deviation(
                item=f"A-{key}",
                cell=f"Balances!{key}",
                quantity="Antoine coefficient " + "ABC"["456".index(key[1])],
                unit="-",
                workbook_value=cell.value,
                rebuilt_value=target,
                cause=(
                    "stored as text (works in arithmetic by coercion, fails in SUM/functions)"
                    if is_text
                    else "decimal point lost: stored value is "
                    f"{float(cell.value) / target:.0f} times the coefficient"  # type: ignore[arg-type]
                    + " (probably a decimal/thousands-separator locale effect, to be confirmed)"
                ),
                hand_calculation=(
                    f"coefficient set A = {correct['J4']}, B = {correct['J5']}, "
                    f"C = {correct['J6']} "
                    f"(1-99 degC); check: log10(p/Pa) = A - B/(C+T) gives "
                    f"{c.saturation_pressure_pa(99.99):.0f} Pa at 100 degC (101325 Pa expected)"
                    if key == "J4"
                    else ""
                ),
            )
        )
    stored_log_p = 101962 - 1730.63 / (233426 + 100.0)
    out.append(
        Deviation(
            item="A-use",
            cell="Balances!J4:K6 (not referenced by any formula yet)",
            quantity="log10(p/Pa) at 100 degC with the stored coefficients",
            unit="-",
            workbook_value=stored_log_p,
            rebuilt_value=math.log10(c.saturation_pressure_pa(100.0)),
            cause="with the stored A the exponent is about 1e5, so p overflows (#NUM!)",
            hand_calculation=(
                f"stored: 101962 - 1730.63/(233426+100) = {stored_log_p:.2f}; "
                "corrected: 10.2651 - 1810.94/(244.485+100) = "
                f"{math.log10(c.saturation_pressure_pa(100.0)):.4f}"
            ),
        )
    )

    # --- 6. Unit labels ----------------------------------------------------------------
    labels = [
        (
            "Inputs!D7",
            "gm/kg",
            0.4,
            "kg/kg",
            "value 0.4 is a mass fraction (400 g/kg); the formulas use it as kg/kg",
        ),
        ("Inputs!D14", "gm/kg", 0.4, "kg/kg", "same as Inputs!D7"),
        ("Scenarios!D6", "kg/s", 50.0, "kg/h", "value 50 is kg/h (equals Scenarios!D3)"),
        ("Scenarios!D7", "kg/kg", 65.92, "g/kg", "value 65.92 is g/kg (equals Scenarios!D4)"),
    ]
    for address, label, number, unit, note in labels:
        out.append(
            Deviation(
                item="U-" + address.split("!")[1],
                cell=address,
                quantity="unit label",
                unit=unit,
                workbook_value=label,
                rebuilt_value=unit,
                cause=note,
                hand_calculation=f"value {number:g}",
            )
        )

    # --- 7. Structure: empty references, inconsistent references ------------------------
    proposal = k.proposed_cell_values(si)
    for col, rowref in (("E", "F12"), ("G", "H12")):
        out.append(
            Deviation(
                item=f"S-{col}20",
                cell=f"Scenarios!{col}20",
                quantity=f"heater 1 inlet temperature, case {'EGIKMO'.index(col) + 1}",
                unit="degC",
                workbook_value=_num(cells.get(f"Scenarios!{col}20")),
                rebuilt_value=proposal[f"{col}20"],
                cause=f"{col}20 refers to {rowref} (outlet of pre-heater low 1), which is an empty "
                "cell: the heater is computed from 0 degC. Value on the right: our proposal "
                "(sheet structure, assumptions A1-A5)",
                hand_calculation="see docs/11, section 4",
                formula=cells[f"Scenarios!{col}20"].formula,
            )
        )
    out.append(
        Deviation(
            item="S-I20",
            cell="Scenarios!I20, J20, I19, I21(case 3); M18..M21 (case 5)",
            quantity="heater 1 of cases 3 and 5",
            unit="-",
            workbook_value="I20/J20 empty (I21 = 0); M18 holds a bare '=', M19:M21 missing",
            rebuilt_value="case 3: 15 -> 159 degC (proposal); case 5: same",
            cause="draft state: totals I36 and M36 count only heater 2",
            hand_calculation="",
        )
    )
    out.append(
        Deviation(
            item="S-cap",
            cell="Scenarios!F25 vs H25, L25, P25",
            quantity="cap at the heater target temperature",
            unit="-",
            workbook_value="F25 has no MIN(...); H25, L25, P25 have",
            rebuilt_value="all four capped",
            cause="without the cap a surplus of heat would drive the sink above 300 degC",
            hand_calculation="",
            formula=cells["Scenarios!F25"].formula,
        )
    )
    out.append(
        Deviation(
            item="S-ref",
            cell="Scenarios!E26 ; K26, O26, K30, O30",
            quantity="moisture reference in the heat capacity",
            unit="-",
            workbook_value="E26 uses Inputs!E4 (dryer 1); K26, O26, K30, O30 use G11 (case 2)",
            rebuilt_value="own block: Inputs!E11 / K24 / O24",
            cause="references to a neighbouring column; numerically identical today (all 6 g/kg), "
            "wrong as soon as the dryers differ",
            hand_calculation="",
        )
    )
    # --- 8. Basis of the moisture (assumption to confirm) -------------------------------
    d1 = si.dryer_1
    dry_air = d1.air_mass_flow_kg_per_s / (1.0 + d1.air_humidity_kg_per_kg)
    d1_dry = type(d1)(
        dry_air,
        d1.air_humidity_kg_per_kg,
        d1.air_temperature_celsius,
        d1.spray_mass_flow_kg_per_s,
        d1.spray_solid_fraction_kg_per_kg,
        d1.spray_temperature_celsius,
    )
    res_dry = c.dryer_balance(d1_dry, ms)
    out.append(
        Deviation(
            item="Q-basis",
            cell="Inputs!E3 / Balances!F4",
            quantity="outlet moisture if the 50 kg/h were moist air rather than dry air",
            unit="g/kg",
            workbook_value=cells["Balances!F4"].value,
            rebuilt_value=res_dry.outlet_humidity_kg_per_kg * 1e3,
            cause="the balance treats the air flow as dry-air flow; if the 50 kg/h include the "
            "6 g/kg moisture, dry air is 50/1.006 kg/h (question to the partner, not a correction)",
            hand_calculation=f"dry air = 50/1.006 = {dry_air * 3600:.3f} kg/h",
        )
    )
    return out


def _gas_cause(t_in: float, t_out: float) -> str:
    base = (
        "moisture enters as 6 (g/kg) instead of 0.006 in c_p (13.2 instead of 1.212 kJ/(kg K)); "
        "calorific value (already kJ/kg) multiplied by 1e6"
    )
    if t_out == 0.0 and t_in == 0.0:
        return base + "; inlet and outlet temperatures are empty cells -> 0"
    if t_in == 0.0:
        return base + "; the inlet temperature refers to an empty cell (0 degC)"
    return base
