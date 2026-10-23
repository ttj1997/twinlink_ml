"""Six-case network with the new inputs: reader, pinch bound, flowsheet options, reproduction.

The synthetic tests need no partner data. The others need the confidential files from ``local/``
(``TWINLINK_LOCAL_DIR`` names the directory, e.g. in a worktree; never copied, never committed):
``partner/TwinLink Excel.xlsx``, ``partner/Dyssol_Excel simulation_results.xlsx``,
``partner/Model v4.1_2.xlsx``; each is skipped when its file is missing.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from twinlink_ml.flowsheet import Flowsheet, FlowsheetConfig, Switches
from twinlink_ml.flowsheet.pinch_target import SinkDemand, exhaust_segments, network_pinch_target
from twinlink_ml.flowsheet.sheet import sheet_scenario_cells
from twinlink_ml.flowsheet.thermo import GasProps, capacity, exhaust_profile, partner_props
from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4_1 import deviations as dev
from twinlink_ml.partner_v4_1.comparison import Reported
from twinlink_ml.partner_v4_1.inputs import ExcelInputsV41, load_excel_inputs_v41
from twinlink_ml.partner_v4_1.simulation_sheet import CASES, parse_simulation_rows

LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", Path(__file__).resolve().parents[1] / "local"))
PHYSICAL = Switches(exchanger="physical", condensation="physical", gas_formula="physical")
PROPS = GasProps("test", 1006.0, 1906.0)


def _first(*names: str) -> Path | None:
    return next((LOCAL / n for n in names if (LOCAL / n).exists()), None)


WORKBOOK = _first("partner/TwinLink Excel.xlsx", "partner/TwinLink_Excel.xlsx")
RESULTS = _first(
    "partner/Dyssol_Excel simulation_results.xlsx", "partner/Dyssol_Excel simulation results.xlsx"
)
PREVIOUS = _first("partner/Model v4.1_2.xlsx", "partner/Model_v4.1_2.xlsx")


def _need(path: Path | None) -> Path:
    if path is None:
        pytest.skip("partner file of the six-case delivery not available in local/")
    pytest.importorskip("openpyxl")
    return path


# ---------------------------------------------------------------------- reader (synthetic)
def _synthetic_rows(skip: str | None = None) -> list[tuple[object, ...]]:
    rows: list[list[object]] = [[None] * 23 for _ in range(14)]
    rows[0][9], rows[1][9] = "Scenarios", "Results"
    rows[2][:8] = [
        "Process 1",
        "Heat recovery",
        "Fresh gas",
        "Mass flow rate",
        "kg/hr",
        120,
        "kg/s",
        0.03,
    ]
    rows[3][3:6] = ["Temperature", "C", 20]
    rows[4][1:6] = ["Heating transfer medium", None, "Flow rate", "kg/hr", 40]
    rows[5][1:6] = ["Heater", "Efficiency", None, "%", 80]
    labels: list[tuple[str, Callable[[int, int], tuple[object, object]]]] = [
        ("Product stream temperature (C)", lambda _n, k: (100.0 + k, 101.0 + k)),
        ("Exhaust gas temperature (C)", lambda n, _k: (90.0 + n, 91.0 + n)),
        ("Enegry recovery (kW)", lambda _n, k: (1.0 + 0.1 * k, 1.05 + 0.1 * k)),
        (
            "Heat transfer media temperature (C)",
            lambda n, k: (f"{80 + n} - {60 + k}", f"{81 + n} - {61 + k}"),
        ),
        ("Fuel consumption(kg/h)", lambda _n, k: (0.5 * k, 0.51 * k)),
    ]
    r = 2
    for label, f in labels:
        if label == skip:
            continue
        for k in (1, 2):
            if r >= len(rows):
                rows.append([None] * 23)
            row = rows[r]
            row[9] = label if k == 1 else None
            row[10] = k
            for n in CASES:
                e, d = f(n, k)
                row[11 + 2 * (n - 1)], row[12 + 2 * (n - 1)] = e, d
            r += 1
    total: list[object] = [None] * 23
    total[9], total[10] = "Total fuel consumption", "(kg/h)"
    for n in CASES:
        total[11 + 2 * (n - 1)], total[12 + 2 * (n - 1)] = 1.5, 1.53
    rows.append(total)
    return [tuple(x) for x in rows]


def test_simulation_reader_parses_inputs_results_pairs_and_total() -> None:
    sheet = parse_simulation_rows(_synthetic_rows())
    assert sheet.value(1, "Heat recovery", "Fresh gas", "Mass flow rate") == 120.0
    assert sheet.value(1, "Heat recovery", "Fresh gas", "Temperature") == 20.0
    assert sheet.value(1, "Heating transfer medium", "", "Flow rate") == 40.0
    assert sheet.value(1, "Heater", "Efficiency", "") == 80.0
    r = sheet.results["recovered_kw"][3][2]
    assert (r.excel, r.dyssol) == pytest.approx((1.2, 1.25), abs=1e-12)
    assert sheet.results["medium_hot_celsius"][4][1] == Reported(84.0, 85.0)
    assert sheet.results["medium_cold_celsius"][4][2] == Reported(62.0, 63.0)
    assert sheet.results["exhaust_out_celsius"][6][1] == Reported(96.0, 97.0)
    assert sheet.total_fuel[5] == Reported(1.5, 1.53)


def test_simulation_reader_raises_on_a_missing_row() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        parse_simulation_rows(_synthetic_rows(skip="Fuel consumption(kg/h)"))


# ---------------------------------------------------------------------- pinch bound (synthetic)
def test_exhaust_segments_carry_the_whole_release_curve() -> None:
    ex = c.Stream(120 / 3600, 0.066, 137.6)
    profile = exhaust_profile(ex, PROPS, "condensing")
    segments = exhaust_segments("x", ex, PROPS)
    assert sum(s.duty_kw() for s in segments) == pytest.approx(profile.total_w / 1e3, rel=1e-12)
    assert segments[0].high_temperature_celsius == pytest.approx(137.6)


def test_network_pinch_target_against_hand_calculation() -> None:
    # one sensible exhaust, one sink, the exhaust has the larger capacity: the cold end pinches
    # and the sink takes heat up to T_ex - 2 dT ... hand: Q = C_sink * (min(T_set, ...) - T_in)
    ex = c.Stream(200 / 3600, 0.0, 120.0)
    gas = c.Stream(100 / 3600, 0.0, 20.0)
    res = network_pinch_target(
        [("ex", ex)], [SinkDemand("s", gas, 300.0)], PROPS, dt_min_k_per_exchanger=10.0,
        kind="linear",
    )  # fmt: skip
    cg = capacity(gas, PROPS) / 1e3
    assert res.total_demand_kw == pytest.approx(cg * 280.0)
    assert res.max_recovered_kw == pytest.approx(cg * (120.0 - 20.0 - 20.0), rel=1e-9)
    capped = network_pinch_target(
        [("ex", ex)], [SinkDemand("s", gas, 300.0)], PROPS, dt_min_k_per_exchanger=10.0,
        htm_max_celsius=60.0, kind="linear",
    )  # fmt: skip
    assert capped.fixed_burner_kw == pytest.approx(cg * (300.0 - 50.0))
    assert capped.max_recovered_kw == pytest.approx(cg * (50.0 - 20.0), rel=1e-9)


# ---------------------------------------------------------------------- flowsheet options
def _inputs() -> ExcelInputsV41:
    from twinlink_ml.partner_v4.inputs import ExcelDryerInputs, ExcelInputs, ExcelMaterialInputs

    plant = ExcelInputs(
        ExcelDryerInputs(120.0, 6.0, 300.0, 12.0, 0.4, 20.0),
        ExcelDryerInputs(100.0, 6.0, 150.0, 3.0, 0.4, 20.0),
        ExcelMaterialInputs(1.006, 45000.0, 80.0, 4.185, 1.0, 1.906, 2.0, 2500.0, 4.185),
        20.0,
        1.0,
    )
    return ExcelInputsV41(plant, (40.0, 30.0))


def test_flowsheet_options_default_to_the_earlier_behaviour() -> None:
    si = _inputs().to_si()
    old = Flowsheet(FlowsheetConfig(si, PHYSICAL))
    new = Flowsheet(FlowsheetConfig(si, PHYSICAL, saturation_limit=False, htm_max_celsius=None))
    for n in (0, *CASES):
        assert new.evaluate(n).fuel_corrected_kg_per_h == old.evaluate(n).fuel_corrected_kg_per_h
    assert old.stage_model.htm_max_celsius is None
    assert old.dryer_unit(1).config.saturation_limit is False


def test_flowsheet_options_reach_the_units() -> None:
    si = _inputs().to_si()
    fs = Flowsheet(FlowsheetConfig(si, PHYSICAL, saturation_limit=True, htm_max_celsius=95.0))
    assert fs.stage_model.htm_max_celsius == 95.0
    assert fs.dryer_unit(2).config.saturation_limit is True
    hot = [s.htm_hot_celsius for n in CASES for s in fs.evaluate(n).stages]
    assert max(h for h in hot if h is not None) <= 95.0 + 1e-9
    with pytest.raises(ValueError, match="counter-current"):
        FlowsheetConfig(si, Switches(), htm_max_celsius=95.0)


def test_pinch_bound_is_not_above_the_best_case_in_both_medium_variants() -> None:
    si = _inputs().to_si()
    plant = si.plant
    for limit in (95.0, None):
        fs = Flowsheet(FlowsheetConfig(si, PHYSICAL, saturation_limit=True, htm_max_celsius=limit))
        sinks = [
            SinkDemand(
                f"s{j}",
                c.Stream(d.air_mass_flow_kg_per_s, d.air_humidity_kg_per_kg, 20.0),
                d.air_temperature_celsius,
            )
            for j, d in ((1, plant.dryer_1), (2, plant.dryer_2))
        ]
        ex = fs.exhausts()
        target = network_pinch_target(
            [("1", ex[1]), ("2", ex[2])], sinks, partner_props(plant.material),
            dt_min_k_per_exchanger=10.0, htm_max_celsius=limit,
        )  # fmt: skip
        best = min(fs.evaluate(n).burner_duty_kw for n in CASES)
        assert target.min_burner_kw <= best + 1e-9
        assert target.total_demand_kw == pytest.approx(fs.evaluate(0).burner_duty_kw, rel=1e-12)


# ---------------------------------------------------------------------- with the partner files
def test_new_workbook_has_the_formulas_of_4_1_2_and_is_reproduced_exactly() -> None:
    import openpyxl

    new, old = _need(WORKBOOK), _need(PREVIOUS)
    a, b = openpyxl.load_workbook(old), openpyxl.load_workbook(new)
    for name in b.sheetnames:
        if name == "Inputs":
            continue
        addresses = {x.coordinate for ws in (a[name], b[name]) for r in ws.iter_rows() for x in r}
        assert all(a[name][k].value == b[name][k].value for k in addresses), name
    inputs = load_excel_inputs_v41(new)
    cells = sheet_scenario_cells(
        Flowsheet(FlowsheetConfig(inputs.to_si(), Switches(gas_formula="physical")))
    )
    excel = dev.read_formula_cells(new, "Scenarios")
    assert set(cells) == set(excel)
    for address, (_, value) in excel.items():
        assert cells[address] == pytest.approx(value, rel=1e-12, abs=1e-12), address


def test_case_2_with_swapped_medium_flows_is_near_dyssol() -> None:
    from twinlink_ml.partner_v4_1.simulation_sheet import load_simulation_sheet

    inputs = load_excel_inputs_v41(_need(WORKBOOK))
    sim = load_simulation_sheet(_need(RESULTS))
    swapped = inputs.with_htm_flow((inputs.htm_flow_kg_per_h[1], inputs.htm_flow_kg_per_h[0]))
    for flows, tol in ((swapped, 0.02), (inputs, None)):
        fs = Flowsheet(FlowsheetConfig(flows.to_si(), Switches(gas_formula="physical")))
        by_source = {s.source: s.q_w / 1e3 for s in fs.evaluate(2).stages}
        reported = sim.results["recovered_kw"][2]
        # index 1 of the sheet's energy row is the sink-1 stage, i.e. exhaust 2 in case 2
        dev_ = max(abs(by_source[2] - reported[1].dyssol), abs(by_source[1] - reported[2].dyssol))
        if tol is None:
            assert dev_ > 0.05
        else:
            assert dev_ < tol
