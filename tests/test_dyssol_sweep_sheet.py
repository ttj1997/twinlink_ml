"""Sweep sheet of the one-line flowsheets: reader, readings of the gas flow, comparison.

The synthetic sheet uses the laboratory-scale inputs already committed in docs/11 and invented
process names. The test on the confidential workbook is skipped when it is missing.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from twinlink_ml.dyssol_steady.materials import (
    DEFAULT_HEAT_CAPACITY,
    DEFAULT_MOLAR_MASS,
    DyssolProperties,
)
from twinlink_ml.dyssol_steady.reference import KELVIN, LineInputs, evaluate
from twinlink_ml.dyssol_steady.sweep_sheet import (
    GAS_FLOW_READINGS,
    RESULT_KEYS,
    SheetLine,
    compare,
    line_inputs,
    load_sweep_sheet,
    parse_sweep_rows,
    sweep_table,
    totals,
    verdict,
)

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", ROOT / "local"))
WORKBOOK = LOCAL / "partner" / "dyssol_2026-11-24" / "Simulation results_flow rates.xlsx"

PROPS = DyssolProperties(
    cp_gas=1006.0,
    cp_vapour=1906.0,
    cp_liquid=4185.0,
    cp_solid=DEFAULT_HEAT_CAPACITY,
    heat_of_vaporization_j_per_kg=40660.5 / DEFAULT_MOLAR_MASS,
    boiling_point_k=373.15,
    gas_enthalpy_linear=(1005.0, 0.0),
    vapour_enthalpy_shomate=(
        1670.36,
        0.379262,
        0.000377093,
        -1.40685e-07,
        4.5594e06,
        -1.3926e07,
        -1.34234e07,
    ),
    gas_enthalpy_range_k=(200.0, 2773.0),
    vapour_enthalpy_range_k=(290.0, 1700.0),
)
LAB = LineInputs(
    dry_gas_kg_per_h=120.0,
    gas_humidity_g_per_kg=6.0,
    gas_celsius=20.0,
    dryer_inlet_celsius=300.0,
    spray_kg_per_h=12.0,
    solid_fraction=0.4,
    spray_celsius=20.0,
    medium_kg_per_h=0.0,
    equilibrium_moisture_g_per_kg=2.0,
    heater_efficiency_percent=80.0,
    calorific_value_kj_per_kg=45000.0,
)
LAB2 = replace(LAB, dry_gas_kg_per_h=100.0, dryer_inlet_celsius=150.0, spray_kg_per_h=3.0)
FLOWS = (0.0, 20.0, 40.0)
NAMES = ("alpha", "beta")


def _sheet_line(line: LineInputs) -> SheetLine:
    """The sheet's statement of a line when its gas flow is the dry gas."""
    gas = line.dry_gas_kg_per_h / 3600.0
    return SheetLine(
        gas_kg_per_s=gas,
        gas_water_kg_per_s=gas * line.gas_humidity_g_per_kg / 1000.0,
        gas_celsius=line.gas_celsius,
        dryer_inlet_celsius=line.dryer_inlet_celsius,
        spray_kg_per_s=line.spray_kg_per_h / 3600.0,
        spray_solid_kg_per_s=line.spray_kg_per_h * line.solid_fraction / 3600.0,
        spray_celsius=line.spray_celsius,
        heater_efficiency_percent=line.heater_efficiency_percent,
        calorific_value_kj_per_kg=line.calorific_value_kj_per_kg,
    )


def _table(line: LineInputs) -> dict[str, tuple[float, ...]]:
    return sweep_table(
        [evaluate(PROPS, [replace(line, medium_kg_per_h=m)], (0,))[0] for m in FLOWS]
    )


def _rounded(value: float) -> float:
    return math.floor(value * 10.0 + 0.5) / 10.0


def _truncated(value: float) -> float:
    return math.floor(value * 10.0 + 1e-9) / 10.0


def _reported_tables(cut: Any = _rounded) -> list[dict[str, list[float | None]]]:
    tables: list[dict[str, list[float | None]]] = []
    for line in (LAB, LAB2):
        table: dict[str, list[float | None]] = {
            k: [cut(v) for v in vs] for k, vs in _table(line).items()
        }
        table["medium_hot_celsius"][0] = None
        table["medium_cold_celsius"][0] = None
        tables.append(table)
    return tables


def _rows(tables: list[dict[str, list[float | None]]]) -> list[list[Any]]:
    """Rows in the layout of the sheet (merged cells empty, as openpyxl returns them)."""
    width = 11 + len(FLOWS)

    def row(**cells: Any) -> list[Any]:
        out: list[Any] = [None] * width
        for column, value in cells.items():
            out[ord(column) - ord("A")] = value
        return out

    def values(cells: list[Any], reported: list[float | None]) -> list[Any]:
        cells[11:] = ["-" if v is None else v for v in reported]
        return cells

    inputs: list[list[Any]] = []
    for k, (name, line) in enumerate(zip(NAMES, (LAB, LAB2), strict=True), start=1):
        s = _sheet_line(line)
        inputs += [
            row(
                A=f"Process {k}",
                B=name,
                C="Heat recovery",
                D="Fresh gas",
                E="Mass flow rate",
                F="kg/s",
                G=s.gas_kg_per_s,
            ),
            row(E="Moisture content", F="kg/s", G=s.gas_water_kg_per_s),
            row(E="Temperature", F="K", G=s.gas_celsius + KELVIN),
            row(C="Dryer ", D="Process air", E="Mass flow rate", F="kg/s", G=s.gas_kg_per_s),
            row(E="Moisture content", F="kg/s", G=s.gas_water_kg_per_s),
            row(E="Temperature", F="°C", G=s.dryer_inlet_celsius),
            row(D="Spray rate", E="Mass flow rate", F="kg/s", G=s.spray_kg_per_s),
            row(E="Solid fraction", F="kg/s", G=s.spray_solid_kg_per_s),
            row(E="Temperature", F="K", G=s.spray_celsius + KELVIN),
            row(C="Heater", D="Efficiency", F="%", G=s.heater_efficiency_percent),
            row(D="Calorific value", F="kJ/kg", G=s.calorific_value_kj_per_kg),
        ]
    labels = {
        "product_celsius": "Product stream temperature (°C)",
        "exhaust_celsius": "Exhaust gas temperature (°C)",
        "recovered_kw": "Enegry recovery (kW)",
        "fuel_kg_per_h": "Fuel consumption(kg/h)",
    }
    results: list[dict[str, Any]] = []
    for key in ("product_celsius", "exhaust_celsius", "recovered_kw"):
        for p, name in enumerate(NAMES):
            results.append({"I": labels[key] if p == 0 else None, "J": name, "v": tables[p][key]})
    for p, name in enumerate(NAMES):
        results.append(
            {
                "I": "Heat transfer media temperature (°C)" if p == 0 else None,
                "J": name,
                "K": "Hot side ",
                "v": tables[p]["medium_hot_celsius"],
            }
        )
        results.append({"K": "Cold side", "v": tables[p]["medium_cold_celsius"]})
    for p, name in enumerate(NAMES):
        key = "fuel_kg_per_h"
        results.append({"I": labels[key] if p == 0 else None, "J": name, "v": tables[p][key]})
    for cells, result in zip(inputs, results, strict=False):
        for column in ("I", "J", "K"):
            if result.get(column) is not None:
                cells[ord(column) - ord("A")] = result[column]
        values(cells, result["v"])
    header = row(A="Process ", C="Input parameters", I="Heating transfer medium flow rate (kg/hr)")
    header[11:] = ["No flow", *FLOWS[1:]]
    return [
        header,
        row(C="Component", D="Stream", E="Parameters", F="Inlet", I="Results"),
        *inputs,
        row(),
        row(A="Other parameters"),
        row(A="Equilibrium moisture content ", D="2 gm/kg"),
        row(A="Materials used ", D="Al2O3 and H2O"),
        row(A="Condensation ", D="No"),
    ]


def test_reader_returns_inputs_flows_and_results() -> None:
    tables = _reported_tables()
    sheet = parse_sweep_rows(_rows(tables))
    assert sheet.medium_kg_per_h == FLOWS
    assert sheet.equilibrium_moisture_g_per_kg == 2.0
    assert sheet.condensation is False
    assert len(sheet.lines) == 2
    for found, line in zip(sheet.lines, (LAB, LAB2), strict=True):
        expected = _sheet_line(line)
        for name in SheetLine.__dataclass_fields__:
            assert getattr(found, name) == pytest.approx(getattr(expected, name), rel=1e-12)
    assert set(sheet.results) == set(RESULT_KEYS)
    for key in RESULT_KEYS:
        for p in range(2):
            assert list(sheet.results[key][p]) == tables[p][key]
    assert sheet.results["medium_hot_celsius"][1][0] is None


def test_the_two_readings_of_the_gas_flow() -> None:
    sheet = parse_sweep_rows(_rows(_reported_tables()))
    dry = line_inputs(sheet, 0, "dry")
    for name in LineInputs.__dataclass_fields__:
        assert getattr(dry, name) == pytest.approx(getattr(LAB, name), rel=1e-12)
    humid = line_inputs(sheet, 0, "humid_total")
    water = LAB.dry_gas_kg_per_h * LAB.gas_humidity_g_per_kg / 1000.0
    assert humid.dry_gas_kg_per_h == pytest.approx(LAB.dry_gas_kg_per_h - water)
    assert humid.gas_humidity_g_per_kg == pytest.approx(1000.0 * water / humid.dry_gas_kg_per_h)
    assert humid.spray_kg_per_h == pytest.approx(LAB.spray_kg_per_h)
    assert GAS_FLOW_READINGS == ("dry", "humid_total")
    with pytest.raises(ValueError, match="unknown reading"):
        line_inputs(sheet, 0, "wet")


def test_comparison_separates_rounded_truncated_and_shifted_sheets() -> None:
    ours = _table(LAB)
    rounded = {k: compare(ours[k], _reported_tables(_rounded)[0][k]) for k in RESULT_KEYS}
    total = totals(rounded)
    assert total["n"] == 4 * len(FLOWS) + 2 * (len(FLOWS) - 1)
    assert total["within_half"] == total["n"]
    assert all(entry["equal_if_rounded"] == entry["n"] for entry in rounded.values())
    assert verdict(total) == "S1"

    truncated = {k: compare(ours[k], _reported_tables(_truncated)[0][k]) for k in RESULT_KEYS}
    assert all(entry["equal_if_truncated"] == entry["n"] for entry in truncated.values())
    assert all(entry["min_signed"] >= -1e-9 for entry in truncated.values())
    assert totals(truncated)["within_one"] == total["n"]
    assert verdict(totals(truncated)) in ("S1", "S2")

    shifted = {
        k: compare([v + 0.3 for v in ours[k]], _reported_tables(_rounded)[0][k])
        for k in RESULT_KEYS
    }
    assert totals(shifted)["within_one"] == 0
    assert verdict(totals(shifted)) == "S3"


def test_comparison_by_hand() -> None:
    entry = compare([1.04, 1.06, 1.15], [1.0, 1.0, None])
    assert entry["n"] == 2
    assert entry["within_half"] == 1
    assert entry["within_one"] == 2
    assert entry["equal_if_rounded"] == 1
    assert entry["equal_if_truncated"] == 2
    assert entry["min_signed"] == pytest.approx(0.04)
    assert entry["max_signed"] == pytest.approx(0.06)
    assert entry["max_relative_percent"] == pytest.approx(6.0)
    assert verdict({"n": 2, "within_half": 1, "within_one": 2}) == "S2"
    with pytest.raises(ValueError, match="no reported value"):
        compare([1.0], [None])


def test_medium_temperatures_are_the_outlet_temperatures_of_the_two_gases() -> None:
    out = evaluate(PROPS, [replace(LAB, medium_kg_per_h=40.0)], (0,))[0]
    table = sweep_table([out])
    assert table["medium_hot_celsius"] == (out.exhaust_celsius,)
    assert table["medium_cold_celsius"] == (out.preheated_celsius,)
    no_flow = sweep_table([evaluate(PROPS, [LAB], (0,))[0]])
    assert no_flow["recovered_kw"] == (0.0,)
    assert no_flow["exhaust_celsius"] == no_flow["product_celsius"]


def test_reader_rejects_incomplete_or_unexpected_sheets() -> None:
    rows = _rows(_reported_tables())
    assert parse_sweep_rows(rows).medium_kg_per_h == FLOWS

    def changed(row: int, column: str, value: Any) -> list[list[Any]]:
        copy = [list(r) for r in rows]
        copy[row][ord(column) - ord("A")] = value
        return copy

    with pytest.raises(ValueError, match="not in kg/h"):
        parse_sweep_rows(changed(0, "I", "Heating transfer medium flow rate (t/d)"))
    with pytest.raises(ValueError, match="more than 1 decimal"):
        parse_sweep_rows(changed(2, "M", 20.25))
    with pytest.raises(ValueError, match="fresh gas and process air differ"):
        parse_sweep_rows(changed(5, "G", 1.0))
    with pytest.raises(ValueError, match="does not name"):
        parse_sweep_rows(changed(3, "J", "gamma"))
    with pytest.raises(ValueError, match="without the unit"):
        parse_sweep_rows(changed(6, "I", "Enegry recovery (MJ)"))
    with pytest.raises(ValueError, match="is not a temperature"):
        parse_sweep_rows(changed(4, "F", "bar"))
    with pytest.raises(ValueError, match="processes incomplete"):
        parse_sweep_rows(changed(12, "I", "Something else"))
    with pytest.raises(ValueError, match="equilibrium moisture"):
        parse_sweep_rows(changed(len(rows) - 3, "D", "2 %"))
    with pytest.raises(ValueError, match="neither yes nor no"):
        parse_sweep_rows(changed(len(rows) - 1, "D", "maybe"))


@pytest.mark.skipif(not WORKBOOK.exists(), reason="confidential workbook not in local/")
def test_partner_workbook_is_read_completely() -> None:
    sheet = load_sweep_sheet(WORKBOOK)
    assert len(sheet.lines) == 2
    assert sheet.medium_kg_per_h[0] == 0.0
    assert len(sheet.medium_kg_per_h) == 11
    assert sheet.condensation is False
    for key in RESULT_KEYS:
        for p in range(2):
            reported = sheet.results[key][p]
            assert len(reported) == 11
            missing = [v is None for v in reported]
            assert missing == ([True] + [False] * 10 if key.startswith("medium") else [False] * 11)
