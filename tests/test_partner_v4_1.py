"""Tests of the reconstruction of the partner workbook ``Model_v4.1.xlsx``.

Separate from ``test_partner_v4.py`` (``Model_v4.xlsx``), which is unchanged.

* **Workbook tests** read ``Model v4.1.xlsx`` (also accepted: ``Model_v4_1.xlsx``) from
  ``local/`` (confidential, never committed; the directory can be named with the environment
  variable ``TWINLINK_LOCAL_DIR``, e.g. in a worktree) and compare the as-built reproduction
  with every number Excel stored, and the document-based model with the same cells. Skipped when
  the file is missing.
* **Model tests** need no workbook: hand calculations, closure of every balance of the
  equation document, the structure of the six cases, the stage order, the assumption of equal
  outlet temperatures.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import ExcelDryerInputs, ExcelInputs, ExcelMaterialInputs
from twinlink_ml.partner_v4_1 import as_built as ab
from twinlink_ml.partner_v4_1 import cascade as k
from twinlink_ml.partner_v4_1 import deviations as dev
from twinlink_ml.partner_v4_1 import layout as lay
from twinlink_ml.partner_v4_1 import loops
from twinlink_ml.partner_v4_1.inputs import ExcelInputsV41, load_excel_inputs_v41

LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", Path(__file__).resolve().parents[1] / "local"))


def _find(*names: str) -> Path | None:
    for name in names:
        if (LOCAL / name).exists():
            return LOCAL / name
    return None


WORKBOOK = _find("Model v4.1.xlsx", "Model_v4_1.xlsx")
EQUATIONS = _find("Flow diagram and balance equations v4.docx")
NEW_VERSION = (
    "NEW VERSION OF THE PARTNER WORKBOOK -- reconcile the reconstruction with it "
    "(docs/11, section 7). This is not a code error. "
)


def default_inputs() -> ExcelInputsV41:
    """The sheet ``Inputs`` of ``Model_v4.1.xlsx`` as of the draft, typed in by hand."""
    plant = ExcelInputs(
        dryer_1=ExcelDryerInputs(100.0, 6.0, 150.0, 6.0, 0.4, 15.0),
        dryer_2=ExcelDryerInputs(100.0, 6.0, 300.0, 12.0, 0.4, 15.0),
        material=ExcelMaterialInputs(1.2, 45000.0, 80.0, 4.2, 1.0, 2.0, 2.0, 2500.0, 4.2),
        ambient_temperature_celsius=15.0,
        ambient_pressure_atm=1.0,
    )
    return ExcelInputsV41(plant, (20.0, 20.0))


# --- workbook tests ------------------------------------------------------------------


@pytest.fixture(scope="module")
def workbook():
    if WORKBOOK is None:
        pytest.skip("local/Model v4.1.xlsx not available")
    pytest.importorskip("openpyxl")
    return load_excel_inputs_v41(WORKBOOK), WORKBOOK


def test_inputs_are_the_recorded_draft(workbook):
    """State of the draft the reconstruction was made against (see ``NEW_VERSION``)."""
    inputs, _ = workbook
    assert inputs == default_inputs(), NEW_VERSION + "The sheet Inputs changed."


@pytest.mark.parametrize("sheet", ["Scenarios", "Balances", "Balance"])
def test_as_built_reproduces_every_stored_number(workbook, sheet):
    inputs, path = workbook
    excel = dev.read_formula_cells(path, sheet)
    built = (
        ab.scenario_cells_as_built(inputs)
        if sheet == "Scenarios"
        else ab.balances_cells_as_built(inputs)
    )
    if sheet == "Balance":  # the duplicate sheet only has columns A:F
        built = {a: v for a, v in built.items() if re.fullmatch(r"[A-F]\d+", a)}
    assert set(excel) == set(built), NEW_VERSION + "Numeric cells appeared or disappeared."
    for address, (_, stored) in excel.items():
        assert built[address] == pytest.approx(stored, rel=1e-12, abs=1e-12), address


def test_document_model_agrees_with_the_workbook_except_in_the_gas_rows(workbook):
    """Temperatures, flows and energies follow the document; rows 26, 39, 41 do not."""
    inputs, path = workbook
    si = inputs.to_si()
    cases = {n: k.to_case_cells(k.evaluate_case(si, n)) for n in range(1, 7)}
    document = lay.scenario_cells(k.plant_cells(si), cases)
    excel = dev.read_formula_cells(path, "Scenarios")
    differing = {
        a for a, (_, v) in excel.items() if document[a] != pytest.approx(v, rel=1e-9, abs=1e-12)
    }
    assert differing, "the gas rows are expected to differ"
    assert {int(re.sub(r"[A-Z]+", "", a)) for a in differing} == {26, 39, 41}
    for address in differing:
        assert excel[address][1] / document[address] == pytest.approx(13.2 / 1.212 * 10.0, rel=1e-9)


def test_document_balances_agree_with_the_workbook(workbook):
    inputs, path = workbook
    document = k.balances_cells(inputs.to_si())
    for address, (_, stored) in dev.read_formula_cells(path, "Balances").items():
        assert document[address] == pytest.approx(stored, rel=1e-12, abs=1e-12), address


def test_no_stored_number_in_scenarios_is_negative(workbook):
    """Finding (docs/11, 7.5): the draft has no negative recovery (or any negative number)."""
    _, path = workbook
    negative = {a: v for a, (_, v) in dev.read_formula_cells(path, "Scenarios").items() if v < 0.0}
    assert not negative, NEW_VERSION + f"Negative numbers in Scenarios: {negative}"


def test_gas_formula_text_is_still_the_recorded_one(workbook):
    """The deviation 'efficiency /1000, moisture in g/kg' is about this exact formula text."""
    _, path = workbook
    formula, _ = dev.read_formula_cells(path, "Scenarios")["E26"]
    assert formula == "=E15*(F25-E25)*(Inputs!I4+E16*Inputs!I9)/(Inputs!I5*Inputs!I6/1000)", (
        NEW_VERSION + "The gas formula of Scenarios row 26 changed (finding G)."
    )


def test_inputs_column_k_and_the_sheet_balance_are_still_unused(workbook):
    """Finding: ``Inputs!K3:K8`` and the sheet ``Balance`` are referenced by no formula."""
    _, path = workbook
    scan = dev.reference_scan(path)
    assert scan["inputs_column_k_referenced_by"] == [], NEW_VERSION + "Inputs column K is used."
    assert scan["balance_sheet_referenced_by"] == [], NEW_VERSION + "Sheet Balance is used."
    assert scan["balance_repeats_balances_a1_f15"], NEW_VERSION + "Balance differs from Balances."


def _linear(node: ET.Element) -> str:
    """Flatten one OMML element of the equation document to text (fractions as ``((n)/(d))``)."""
    m = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    tag = node.tag
    if tag == m + "t":
        return node.text or ""
    if tag == m + "f":
        return f"(({_linear(node.find(m + 'num'))})/({_linear(node.find(m + 'den'))}))"
    if tag == m + "sSub":
        return f"{_linear(node.find(m + 'e'))}_{{{_linear(node.find(m + 'sub'))}}}"
    if tag.endswith("Pr"):
        return ""
    return "".join(_linear(child) for child in node)


def _equation_texts() -> list[str]:
    with zipfile.ZipFile(EQUATIONS) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    m = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    return [re.sub(r"\s+", "", _linear(e)) for e in root.iter(m + "oMath")]


def test_equation_document_still_has_the_two_typos():
    """(2.1) right side ``T_w,in`` for ``T_w,out``; (3.1) left ``T_g,out`` for ``T_g,in``."""
    if EQUATIONS is None:
        pytest.skip("equation document not available")
    texts = _equation_texts()
    eq_2_1 = next(t for t in texts if "M_{c}C_{p,l}T_{c,out}" in t)
    right = eq_2_1.split("=", 1)[1]
    assert "M_{w}C_{p,w}T_{w,in}" in right and "T_{w,out}" not in right, (
        "EQUATION DOCUMENT CHANGED -- (2.1) no longer has the typo; update docs/11 section 7."
    )
    eq_3_1 = next(t for t in texts if "M_{f}LHVη_{th}=" in t)
    left = eq_3_1.split("=", 1)[0]
    assert "C_{p,v}T_{g,out}" in left, (
        "EQUATION DOCUMENT CHANGED -- (3.1) no longer has the typo; update docs/11 section 7."
    )


# --- model tests: the equations of the document -----------------------------------------


def _streams(inputs: ExcelInputsV41):
    si = inputs.to_si()
    m = si.plant.material
    exhaust = c.exhaust_stream(si.plant.dryer_1, c.dryer_balance(si.plant.dryer_1, m))
    gas = c.Stream(si.plant.dryer_2.air_mass_flow_kg_per_s, 0.006, 15.0)
    return si, m, exhaust, gas


def test_loop_by_hand_case_1_source_1():
    """Exhaust 1 (100 kg/h, 41.952 g/kg, 72.2786 degC) into sink 1 (100 kg/h, 6 g/kg, 15 degC).

    Hand (kJ/h per K): A = 100 (1.2 + 0.041952*2) = 128.3904, B = 20*4.2 = 84,
    C = 100 (1.2 + 0.006*2) = 121.2; Q = (72.2786 - 15) / (1/A + 1/B + 1/C)
    = 57.2786 / 0.0279445 = 2049.7 kJ/h = 0.56937 kW.
    T1 = 15 + Q/C = 31.912, T2 = T1 + Q/B... -> T2 = 72.2786 - Q/A = 56.314.
    """
    si, m, exhaust, gas = _streams(default_inputs())
    r = loops.equal_outlet_loop(exhaust, gas, 20 / 3600, si.cp_htm_j_per_kg_k, m)
    assert exhaust.temperature_celsius == pytest.approx(72.27856833576736)
    assert r.recovered_w * 3.6 == pytest.approx(2049.74, abs=0.01)  # W -> kJ/h
    assert r.sink_out_celsius == pytest.approx(31.912, abs=1e-3)
    assert r.exhaust_out_celsius == pytest.approx(56.3137, abs=1e-4)


def test_loop_closes_every_balance_independently():
    """(4.1), (4.2) as written, the energy balance of the three streams, equal outlets."""
    si, m, exhaust, gas = _streams(default_inputs())
    flow = 33.0 / 3600
    r = loops.equal_outlet_loop(exhaust, gas, flow, si.cp_htm_j_per_kg_k, m)
    a, b, cc = loops.loop_capacity_rates(exhaust, gas, flow, si.cp_htm_j_per_kg_k, m)
    t_c, t_h = exhaust.temperature_celsius, gas.temperature_celsius
    assert a * (t_c - r.exhaust_out_celsius) == pytest.approx(
        b * (r.exhaust_out_celsius - r.sink_out_celsius)
    )
    assert cc * (r.sink_out_celsius - t_h) == pytest.approx(
        b * (r.exhaust_out_celsius - r.sink_out_celsius)
    )
    # closed form: series of three conductances
    assert r.recovered_w == pytest.approx((t_c - t_h) / (1 / a + 1 / b + 1 / cc), rel=1e-12)
    # what the exhaust loses equals what the gas gains
    assert a * (t_c - r.exhaust_out_celsius) == pytest.approx(cc * (r.sink_out_celsius - t_h))


def test_equation_2_4_is_the_mixing_temperature_of_parallel_flow():
    """Equal outlet temperatures = co-current, unlimited area: ``(C1 T1 + C2 T2) / (C1 + C2)``."""
    si, m, exhaust, _ = _streams(default_inputs())
    b = 20 / 3600 * si.cp_htm_j_per_kg_k
    a = c.heat_capacity_rate_w_per_k(exhaust, m)
    t = loops.equation_2_4_celsius(exhaust, b, 15.0, m)
    assert t == pytest.approx((a * exhaust.temperature_celsius + b * 15.0) / (a + b), rel=1e-14)
    assert 15.0 < t < exhaust.temperature_celsius


def test_condensing_exchanger_closes_equation_2_1_and_agrees_with_2_3():
    """A cold, large medium flow makes the exhaust condense; (2.1) corrected and (2.3) agree."""
    si, m, exhaust, _ = _streams(default_inputs())
    p = si.plant.ambient.pressure_pa
    cp_w = si.cp_htm_j_per_kg_k
    flow, t_w_in = 400.0 / 3600, 10.0
    r = loops.equal_outlet_exchanger(exhaust, flow, cp_w, t_w_in, p, m)
    assert r.condensate_kg_per_s > 0.0
    assert r.outlet_celsius < loops.exhaust_dew_point_celsius(exhaust, p)
    # (2.2) condensate; (2.3) with the resulting outlet humidity reproduces the outlet temperature
    assert r.condensate_kg_per_s == pytest.approx(
        exhaust.dry_air_kg_per_s * (exhaust.humidity_kg_per_kg - r.outlet_humidity_kg_per_kg)
    )
    t_23 = loops.equation_2_3_celsius(exhaust, r.outlet_humidity_kg_per_kg, flow * cp_w, t_w_in, m)
    assert t_23 == pytest.approx(r.outlet_celsius, abs=1e-9)
    # (2.1) written out, T_w,out on the right (the typo of the document corrected)
    mg, y_in, y_out = (
        exhaust.dry_air_kg_per_s,
        exhaust.humidity_kg_per_kg,
        r.outlet_humidity_kg_per_kg,
    )
    dh, cpg, cpv, cpl = (
        m.enthalpy_of_evaporation_j_per_kg,
        m.cp_gas_j_per_kg_k,
        m.cp_vapour_j_per_kg_k,
        m.cp_liquid_j_per_kg_k,
    )
    t_in, t = exhaust.temperature_celsius, r.outlet_celsius
    left = mg * (cpg * t_in + y_in * (dh + cpv * t_in)) + flow * cp_w * t_w_in
    right = (
        mg * (cpg * t + y_out * (dh + cpv * t)) + flow * cp_w * t + r.condensate_kg_per_s * cpl * t
    )
    assert left == pytest.approx(right, rel=1e-9)
    # (2.5)
    assert r.recovered_w == pytest.approx(flow * cp_w * (t - t_w_in))


def test_without_condensation_2_3_reduces_to_2_4():
    si, m, exhaust, _ = _streams(default_inputs())
    b = 20 / 3600 * si.cp_htm_j_per_kg_k
    assert loops.equation_2_3_celsius(
        exhaust, exhaust.humidity_kg_per_kg, b, 15.0, m
    ) == pytest.approx(loops.equation_2_4_celsius(exhaust, b, 15.0, m), rel=1e-14)


def test_heater_equation_3_2_closes_3_1_corrected():
    si, m, _, gas = _streams(default_inputs())
    assert abs(loops.heater_enthalpy_residual_w(gas, 300.0, m)) < 1e-9


def test_gas_by_hand_100_kg_per_h_from_15_to_300():
    """Document: 100*285*1.212 / (45000*0.8) = 34542 / 36000 = 0.9595 kg/h.

    Workbook formula (Scenarios row 26/39): 100*285*13.2 / (45000*80/1000) = 376200 / 3600
    = 104.5 kg/h; ratio 108.91 = 10.891 (moisture in g/kg) * 10 (efficiency / 1000).
    """
    r = dev.gas_hand_calculation(default_inputs(), 100.0, 15.0, 300.0)
    assert r["as_built_kg_per_h"] == pytest.approx(104.5)
    assert r["document_kg_per_h"] == pytest.approx(34542.0 / 36000.0)
    assert r["ratio"] == pytest.approx(108.91, abs=0.005)
    assert r["moisture_factor"] == pytest.approx(13.2 / 1.212)
    assert r["efficiency_factor"] == pytest.approx(10.0)


def test_gas_cp_of_one_instead_of_1_2_would_lower_the_gas_by_16_5_percent():
    """Question to the partner, not a correction: ``cp_gas`` 1.2 vs about 1.0 for air."""
    si, m, _, gas = _streams(default_inputs())
    base = loops.heater_gas_kg_per_s(gas, 300.0, m)
    cp1 = ExcelMaterialInputs(1.0, 45000.0, 80.0, 4.2, 1.0, 2.0, 2.0, 2500.0, 4.2)
    plant = default_inputs().plant
    other = ExcelInputsV41(
        ExcelInputs(plant.dryer_1, plant.dryer_2, cp1, 15.0, 1.0), (20.0, 20.0)
    ).to_si()
    lower = loops.heater_gas_kg_per_s(gas, 300.0, other.plant.material)
    assert lower / base - 1.0 == pytest.approx(-0.165, abs=0.001)


# --- model tests: the six cases ----------------------------------------------------------


def test_every_source_is_used_once_in_every_case_and_the_structure_is_the_drawing():
    """Drawings: 1 separate, 2 crossed, 3/5 both into sink 1 (2,1 / 1,2), 4/6 into sink 2."""
    assert lay.CASES == {
        1: ((1,), (2,)),
        2: ((2,), (1,)),
        3: ((2, 1), ()),
        4: ((), (2, 1)),
        5: ((1, 2), ()),
        6: ((), (1, 2)),
    }
    si = default_inputs().to_si()
    for case in range(1, 7):
        sources = sorted(s.source for s in k.evaluate_case(si, case).stages)
        assert sources == [1, 2]


def test_as_built_equals_document_model_outside_the_gas_rows():
    """Without the workbook: the sheet's K-system is (4.1)/(4.2); only rows 26/39/41 differ."""
    inputs = default_inputs()
    built = ab.scenario_cells_as_built(inputs)
    si = inputs.to_si()
    cases = {n: k.to_case_cells(k.evaluate_case(si, n)) for n in range(1, 7)}
    document = lay.scenario_cells(k.plant_cells(si), cases)
    assert set(built) == set(document)
    for address, value in built.items():
        row = int(re.sub(r"[A-Z]+", "", address))
        if row in (26, 39, 41):
            assert value / document[address] == pytest.approx(108.91, abs=0.005), address
        else:
            assert value == pytest.approx(document[address], rel=1e-12), address


def test_recovered_heat_is_the_gas_saved():
    """Per case: (gas without - gas with recovery) * eta * H_u = sum of the stage energies."""
    si = default_inputs().to_si()
    m = si.plant.material
    base = k.evaluate_case(si, 0).total_gas_kg_per_h / 3600
    for case in range(1, 7):
        result = k.evaluate_case(si, case)
        saved = (base - result.total_gas_kg_per_h / 3600) * m.heater_efficiency
        assert saved * m.lower_heating_value_j_per_kg == pytest.approx(
            result.total_recovered_w, rel=1e-9
        )


def test_no_condensation_in_any_case_at_the_sheet_flows():
    """(4.1) neglects condensation; at 20 kg/h the exhaust stays above its dew point."""
    si = default_inputs().to_si()
    for case in range(1, 7):
        for stage in k.evaluate_case(si, case).stages:
            assert not stage.condenses, (case, stage.source)
            assert stage.exhaust_out_celsius > stage.exhaust_dew_point_celsius + 5.0


def test_htm_flow_is_adjustable_and_more_flow_recovers_more_in_a_cold_stage():
    inputs = default_inputs()
    low = k.evaluate_case(inputs.with_htm_flow((10.0, 10.0)).to_si(), 1).total_recovered_w
    high = k.evaluate_case(inputs.with_htm_flow((40.0, 40.0)).to_si(), 1).total_recovered_w
    assert 0.0 < low < high


def test_stage_order_case_6_cold_source_first_recovers_more_than_reversed():
    """Case 6 has source 1 (72.3 degC, cold) first; the reversed order is the order of case 4.

    Sheet flows (20 kg/h): gas to heater 2 inlet 62.71 degC (cold first) against 57.19 degC
    (hot first); no negative recovery in either order.
    """
    si = default_inputs().to_si()
    drawn = k.evaluate_case(si, 6)
    reversed_ = k.evaluate_case(si, 6, order=((), (2, 1)))
    assert drawn.sink_2.heater_in_celsius == pytest.approx(62.7056, abs=1e-3)
    assert reversed_.sink_2.heater_in_celsius == pytest.approx(57.1854, abs=1e-3)
    assert reversed_.stages == k.evaluate_case(si, 4).stages
    assert drawn.total_gas_kg_per_h < reversed_.total_gas_kg_per_h
    assert min(s.recovered_w for s in drawn.stages + reversed_.stages) > 0.0


def test_negative_recovery_needs_the_hot_source_first_and_a_large_circuit_flow():
    """Equal outlet temperatures reverse a stage when the gas enters warmer than the exhaust.

    That happens in the second stage of the hot-first order (cases 3/4) once the circuit flow is
    large enough for the first stage to lift the gas above the cold exhaust (72.3 degC); the
    cold-first orders (cases 5/6) stay positive at any flow.
    """
    inputs = default_inputs()
    flow = dev.break_even_htm_flow(inputs, 4)
    assert flow is not None and 100.0 < flow < 200.0
    # at the break-even flow the first stage lifts the gas exactly to the cold exhaust temperature
    si = inputs.with_htm_flow((flow, flow)).to_si()
    first, second = k.evaluate_case(si, 4).sink_2.stages
    assert first.sink_out_celsius == pytest.approx(second.exhaust_in_celsius, abs=1e-6)
    assert abs(second.recovered_w) < 1e-6
    assert dev.break_even_htm_flow(inputs, 6) is None
    assert dev.break_even_htm_flow(inputs, 5) is None
    assert dev.break_even_htm_flow(inputs, 6, ((), (2, 1))) == pytest.approx(flow, rel=1e-9)


def test_counter_current_bound_is_never_below_the_equal_outlet_stage():
    """The document's assumption underestimates: bound >= equal-outlet heat (stage 1, case totals).

    A second stage is not comparable one to one: it sees the gas the first stage delivered, and
    in the bound that gas is warmer (in cases 3/4 it is already above the cold exhaust: 0 kW).
    """
    rows = dev.equal_outlet_vs_counter_current(default_inputs())
    assert len(rows) == 12
    for row in rows:
        if row["stage_of_case"] == 1 or row["case"] in (1, 2):
            assert row["counter_current_kw"] >= row["equal_outlet_kw"] - 1e-12
    for case in range(1, 7):
        in_case = [r for r in rows if r["case"] == case]
        assert sum(r["counter_current_kw"] for r in in_case) >= sum(
            r["equal_outlet_kw"] for r in in_case
        )
    case_1 = next(r for r in rows if r["case"] == 1 and r["source"] == 1)
    # hand: min(A, B, C) = B = 84 kJ/(h K) -> 84 * 57.2786 / 3600 = 1.3365 kW (stage 1)
    assert case_1["counter_current_kw"] == pytest.approx(84 * 57.27856833576736 / 3600, rel=1e-9)
    assert case_1[
        "condenses"
    ]  # the bound's exhaust (34.8 degC) falls below its dew point (36.9 degC)


def test_counter_current_single_exchanger_limit():
    """Unlimited counter-current: the smaller capacity rate leaves at the other's inlet."""
    si, m, exhaust, gas = _streams(default_inputs())
    flow = 20 / 3600
    r = loops.counter_current_infinite_area_loop(exhaust, gas, flow, si.cp_htm_j_per_kg_k, m)
    a, b, cc = loops.loop_capacity_rates(exhaust, gas, flow, si.cp_htm_j_per_kg_k, m)
    assert b < min(a, cc)  # the medium is the smallest: Q = B (T_c,in - T_h,in)
    assert r.recovered_w == pytest.approx(b * (exhaust.temperature_celsius - 15.0), rel=1e-9)
