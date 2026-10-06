"""Tests of the reconstruction of the partner workbook ``Model_v4.xlsx``.

Two groups:

* **Workbook tests** read ``local/Model_v4.xlsx`` (confidential, never committed) and compare
  the as-built reproduction with the values Excel stored for the FINISHED cells (Balances F4,
  F5, F11, F12; Scenarios rows 21, 34, 36). Skipped when the file is missing.
* **Model tests** need no workbook: hand calculations, mass/enthalpy closure, the unit errors.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from twinlink_ml.partner_v4 import as_built as ab
from twinlink_ml.partner_v4 import cascade as k
from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import (
    ExcelDryerInputs,
    ExcelInputs,
    ExcelMaterialInputs,
    load_excel_inputs,
)

WORKBOOK = Path(__file__).resolve().parents[1] / "local" / "Model_v4.xlsx"


def default_inputs() -> ExcelInputs:
    """The sheet ``Inputs`` as of the draft (both dryers identical), typed in by hand."""
    dryer = ExcelDryerInputs(50.0, 6.0, 300.0, 5.0, 0.4, 15.0)
    return ExcelInputs(
        dryer_1=dryer,
        dryer_2=dryer,
        material=ExcelMaterialInputs(1.2, 45000.0, 80.0, 4.2, 1.0, 2.0, 2.0, 2500.0, 4.2),
        ambient_temperature_celsius=15.0,
        ambient_pressure_atm=1.0,
    )


# --- workbook tests ------------------------------------------------------------------


@pytest.fixture(scope="module")
def workbook_values():
    if not WORKBOOK.exists():
        pytest.skip("local/Model_v4.xlsx not available")
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.load_workbook(WORKBOOK, data_only=True)
    return load_excel_inputs(WORKBOOK), wb


def test_balances_match_workbook(workbook_values):
    inputs, wb = workbook_values
    sheet = wb["Balances"]
    d1 = ab.dryer_as_built(inputs.dryer_1, inputs.material)
    d2 = ab.dryer_as_built(inputs.dryer_2, inputs.material)
    assert d1.outlet_moisture_g_per_kg == pytest.approx(sheet["F4"].value, rel=1e-12)
    assert d1.outlet_temperature_celsius == pytest.approx(sheet["F5"].value, rel=1e-12)
    assert d2.outlet_moisture_g_per_kg == pytest.approx(sheet["F11"].value, rel=1e-12)
    assert d2.outlet_temperature_celsius == pytest.approx(sheet["F12"].value, rel=1e-12)


def test_scenario_finished_cells_match_workbook(workbook_values):
    inputs, wb = workbook_values
    sheet = wb["Scenarios"]
    cells = ab.scenario_cells_as_built(inputs)
    assert set(cells) == {"E21", "G21", "I21", "K21", "O21", "I34", "M34", "I36", "M36"}
    for address, value in cells.items():
        assert sheet[address].value == pytest.approx(value, rel=1e-12, abs=1e-18), address


def test_partner_workbook_is_still_the_draft_with_open_cells(workbook_values):
    """NEW VERSION OF THE PARTNER WORKBOOK -> reconcile the reconstruction (not a code error).

    This test records the state of the draft the reconstruction was made against: the cells below
    are ``#REF!``. If it fails, the partner has filled one of them. Nothing in the code is broken:
    compare the new cell values with ``proposed_cell_values`` (``docs/11``, section 3), update the
    reconstruction and the deviation list, then update the cell list here.
    """
    _, wb = workbook_values
    sheet = wb["Scenarios"]
    watched = ("F25", "H25", "L25", "P25", "E34", "G34", "K34", "O34", "E36", "K36")
    filled = {a: sheet[a].value for a in watched if sheet[a].value != "#REF!"}
    assert not filled, (
        "NEW VERSION OF THE PARTNER WORKBOOK -- reconcile the reconstruction with it "
        "(docs/11, section 3; compare with partner_v4.cascade.proposed_cell_values). "
        f"This is not a code error. Cells that are no longer #REF!: {filled}"
    )


# --- model tests ---------------------------------------------------------------------


def test_as_built_dryer_hand_values():
    r = ab.dryer_as_built(default_inputs().dryer_1, default_inputs().material)
    assert r.outlet_moisture_g_per_kg == pytest.approx(65.92)
    assert r.outlet_temperature_celsius == pytest.approx(10909.0 / 68.6088, rel=1e-9)


def test_corrected_dryer_equals_as_built_and_closes_balances():
    e = default_inputs()
    si = e.to_si()
    res = c.dryer_balance(si.dryer_1, si.material)
    old = ab.dryer_as_built(e.dryer_1, e.material)
    assert res.outlet_humidity_kg_per_kg * 1e3 == pytest.approx(old.outlet_moisture_g_per_kg)
    assert res.outlet_temperature_celsius == pytest.approx(old.outlet_temperature_celsius)
    assert abs(c.dryer_enthalpy_residual_w(si.dryer_1, si.material, res)) < 1e-9
    # water: spray water = evaporated + residual product moisture
    water = si.dryer_1.spray_mass_flow_kg_per_s * (1 - 0.4)
    residual = res.product_solids_kg_per_s * 0.002
    assert res.evaporation_kg_per_s + residual == pytest.approx(water)


def test_gas_for_50_kg_per_h_from_15_to_300_is_048_kg_per_h():
    si = default_inputs().to_si()
    stream = c.Stream(50 / 3600, 0.006, 15.0)
    power = c.heater_power_w(stream, 15.0, 300.0, si.material)
    assert power == pytest.approx(4797.5, rel=1e-9)  # W
    gas_kg_per_h = c.gas_mass_flow_kg_per_s(power, si.material) * 3600
    assert gas_kg_per_h == pytest.approx(0.47975, rel=1e-9)


def test_sheet_total_is_25_5_times_too_low():
    e = default_inputs()
    cells = ab.scenario_cells_as_built(e)
    right = (
        c.gas_mass_flow_kg_per_s(
            c.heater_power_w(c.Stream(50 / 3600, 0.006, 15.0), 15.0, 300.0, e.to_si().material),
            e.to_si().material,
        )
        * 3600
    )
    assert right / cells["M36"] == pytest.approx(25.5, abs=0.05)
    # decomposition: (1.2+6*2)/(1.2+0.006*2) * 1e-6 * 3600
    assert cells["M36"] / right == pytest.approx(13.2 / 1.212 * 1e-6 * 3600, rel=1e-9)


def test_recovery_formula_factor_and_round_trip():
    m = default_inputs().material
    t = ab.preheater_outlet_temperature_as_built(15.0, 1.0, 50.0, 6.0, m)
    assert t - 15.0 == pytest.approx(1000 / (50 * 13.2))
    # the sheet's energy formula gives back the 1 kW although dT is 39.2 times too small
    assert ab.preheater_energy_kw_as_built(15.0, t, 50.0, 6.0, m) == pytest.approx(1.0)
    right = 1.0 * 3600 / (50 * 1.212)
    assert right / (t - 15.0) == pytest.approx(3.6 * 13.2 / 1.212, rel=1e-9)


def test_antoine_with_restored_decimals_gives_boiling_point():
    assert c.saturation_pressure_pa(99.99) == pytest.approx(101325.0, rel=0.01)
    assert c.saturation_pressure_pa(100.0) == pytest.approx(101325.0, rel=0.02)
    # continuity at the switch of the two ranges
    assert c.saturation_pressure_pa(99.999) == pytest.approx(
        c.saturation_pressure_pa(100.0), rel=0.01
    )
    # stored coefficients (decimal point lost) are unusable
    assert 101962 - 1730.63 / (233426 + 100.0) > 1e5


def test_dew_point_inverts_saturation_humidity():
    p = 101325.0
    t = 45.0
    y = c.saturation_humidity_kg_per_kg(t, p)
    assert c.dew_point_celsius(y, p) == pytest.approx(t, abs=1e-6)
    assert math.isinf(c.saturation_humidity_kg_per_kg(130.0, p))


def test_heat_release_continuous_at_dew_point_and_monotone():
    si = default_inputs().to_si()
    stream = c.exhaust_stream(si.dryer_1, c.dryer_balance(si.dryer_1, si.material))
    p = si.ambient.pressure_pa
    t_dp = c.dew_point_celsius(stream.humidity_kg_per_kg, p)
    h = lambda t: c.stream_enthalpy_j_per_kg_dry(stream.humidity_kg_per_kg, t, p, si.material)  # noqa: E731
    assert h(t_dp - 1e-6) == pytest.approx(h(t_dp + 1e-6), rel=1e-6)
    assert h(20.0) < h(40.0) < h(60.0) < h(150.0)
    q = c.stream_heat_release_w(stream, 15.0, p, si.material)
    assert q == pytest.approx(4559.3, rel=1e-3)
    # inverse
    t_back = c.temperature_at_released_heat(stream, 0.5 * q, 15.0, p, si.material)
    released = stream.dry_air_kg_per_s * (h(stream.temperature_celsius) - h(t_back))
    assert released == pytest.approx(0.5 * q, rel=1e-6)


def test_cascade_cases_hand_values():
    si = default_inputs().to_si()
    none = k.evaluate_case(si, 0)
    assert none.total_gas_kg_per_h == pytest.approx(2 * 0.47975, rel=1e-9)
    one = k.evaluate_case(si, 1)
    # both sinks reach the exhaust temperature (159.003 degC): heater from there to 300
    t_ex = 159.00292673826098
    gas_each = 50 / 3600 * 1212.0 * (300 - t_ex) / (0.8 * 45e6) * 3600
    assert one.total_gas_kg_per_h == pytest.approx(2 * gas_each, rel=1e-9)
    # crossed case is identical for identical dryers, and case 3 = case 5, 4 = 6
    assert k.evaluate_case(si, 2).total_gas_kg_per_h == pytest.approx(one.total_gas_kg_per_h)
    assert k.evaluate_case(si, 3).total_gas_kg_per_h == pytest.approx(
        k.evaluate_case(si, 5).total_gas_kg_per_h
    )
    # second stage of a cascade adds nothing: the sink is already at the source temperature
    assert k.evaluate_case(si, 3).sink_1.stages[1].power_w == 0.0


def test_counter_current_is_never_better_than_sheet_structure():
    si = default_inputs().to_si()
    for case in range(1, 7):
        sheet = k.evaluate_case(si, case, mode="sheet").total_gas_kg_per_h
        cc0 = k.evaluate_case(si, case, mode="counter_current", dt_min_k=0.0).total_gas_kg_per_h
        cc20 = k.evaluate_case(si, case, mode="counter_current", dt_min_k=20.0).total_gas_kg_per_h
        assert sheet <= cc0 * (1 + 1e-9)
        assert cc0 <= cc20 * (1 + 1e-9)


def test_proposal_covers_the_open_cells():
    cells = k.proposed_cell_values(default_inputs().to_si())
    for address in (
        "F25",
        "H25",
        "L25",
        "P25",
        "E26",
        "K26",
        "O26",
        "K29",
        "L29",
        "O29",
        "P29",
        "K30",
        "O30",
        "E33",
        "G33",
        "K33",
        "O33",
        "E34",
        "G34",
        "K34",
        "O34",
        "E36",
        "G36",
        "K36",
        "O36",
        "E20",
        "G20",
        "I20",
        "J20",
        "F12",
        "H12",
    ):
        assert address in cells, address
    # row 36 is the sum of the heaters (kg/h, no factor 3600)
    assert cells["E36"] == pytest.approx(cells["E21"] + cells["E34"])


# --- order of the cascade (synthetic, deliberately different dryers) -------------------


def synthetic_unequal_dryers(*, swap: bool = False) -> ExcelInputs:
    """SYNTHETIC inputs, not partner data: the two dryers differ on purpose.

    Dryer "big": 100 kg/h air, 5 kg/h spray (exhaust 225.14 degC). Dryer "hot": 50 kg/h air,
    1 kg/h spray (exhaust 268.89 degC, small flow). All other values as the partner's draft.
    ``swap=True`` exchanges the roles of dryer 1 and 2.
    """
    big = ExcelDryerInputs(100.0, 6.0, 300.0, 5.0, 0.4, 15.0)
    hot = ExcelDryerInputs(50.0, 6.0, 300.0, 1.0, 0.4, 15.0)
    d1, d2 = (hot, big) if swap else (big, hot)
    return ExcelInputs(
        dryer_1=d1,
        dryer_2=d2,
        material=default_inputs().material,
        ambient_temperature_celsius=15.0,
        ambient_pressure_atm=1.0,
    )


def _gas_kg_per_h(air_kg_per_h: float, t_in: float, t_out: float) -> float:
    """Gas of a heater: air * (1.2 + 0.006*2) kJ/(kg K) * dT / (0.8 * 45000 kJ/kg)."""
    return air_kg_per_h * 1.212 * (t_out - t_in) / (0.8 * 45000.0)


def test_exhaust_temperatures_of_the_synthetic_dryers_by_hand():
    """Hand calculation of the two exhaust temperatures (sheet formula; 0.6*4.2 + 0.4*1 = 2.92).

    Big (100 kg/h, spray 5): Y = 6 + 1000*5*0.5992/100 = 35.96 g/kg;
      numerator 100*(1.212*300 - 29.96e-3*2500) + 5*15*2.92 = 100*288.7 + 219 = 29089,
      denominator 100*(1.2 + 0.03596*2) + 2*1.0084 = 129.2088; T = 225.14 degC.
    Hot (50 kg/h, spray 1): Y = 6 + 1000*1*0.5992/50 = 17.984 g/kg;
      numerator 50*(363.6 - 11.984e-3*2500) + 1*15*2.92 = 16682 + 43.8 = 16725.8,
      denominator 50*(1.2 + 0.017984*2) + 0.4*1.0084 = 62.20176; T = 268.89 degC.
    """
    e = synthetic_unequal_dryers()
    big = ab.dryer_as_built(e.dryer_1, e.material)
    hot = ab.dryer_as_built(e.dryer_2, e.material)
    assert big.outlet_moisture_g_per_kg == pytest.approx(35.96)
    assert hot.outlet_moisture_g_per_kg == pytest.approx(17.984)
    assert big.outlet_temperature_celsius == pytest.approx(29089.0 / 129.2088, rel=1e-9)
    assert hot.outlet_temperature_celsius == pytest.approx(16725.8 / 62.20176, rel=1e-9)
    assert big.outlet_temperature_celsius < hot.outlet_temperature_celsius


def test_cascade_order_matters_cases_3_and_5_for_unequal_dryers():
    """Sink 1 (big, 100 kg/h) with the hotter small source 2 (synthetic inputs, see above).

    Case 3 (order 2,1): source 2 first offers 4.612 kW (energy-limited, sink to about 152 degC),
    then source 1 lifts the sink to its own exhaust temperature T1 = 225.14 degC; the hotter
    source 2 is used up before. Final sink temperature T1.
    Case 5 (order 1,2): source 1 first lifts the sink to T1 (7.07 kW); then the hotter source 2
    lifts it further to its own exhaust temperature T2 = 268.89 degC. Final sink temperature T2.
    Heater 1 gas = 100*1.212*(300 - T_final)/36000 kg/h; heater 2 has no recovery in these
    cases (50 kg/h, 15 -> 300 degC: 0.47975 kg/h).
    """
    e = synthetic_unequal_dryers()
    si = e.to_si()
    t1, t2 = 29089.0 / 129.2088, 16725.8 / 62.20176
    heater_2 = _gas_kg_per_h(50.0, 15.0, 300.0)
    case3 = k.evaluate_case(si, 3)
    case5 = k.evaluate_case(si, 5)
    assert case3.sink_1.heater_in_celsius == pytest.approx(t1, rel=1e-9)
    assert case5.sink_1.heater_in_celsius == pytest.approx(t2, rel=1e-9)
    assert case3.total_gas_kg_per_h == pytest.approx(_gas_kg_per_h(100.0, t1, 300.0) + heater_2)
    assert case5.total_gas_kg_per_h == pytest.approx(_gas_kg_per_h(100.0, t2, 300.0) + heater_2)
    assert case5.total_gas_kg_per_h < case3.total_gas_kg_per_h - 0.1
    # stage by stage: case 5 stage 1 is the whole lift to T1 (C = 100/3600*1212 W/K)
    assert case5.sink_1.stages[0].power_w == pytest.approx(100 / 3600 * 1212 * (t1 - 15.0))
    assert case3.sink_1.stages[0].source == 2 and case5.sink_1.stages[0].source == 1


def test_cascade_order_matters_cases_4_and_6_when_the_roles_are_swapped():
    """Same synthetic dryers with dryer 1 and 2 exchanged: sink 2 is now the big one (100 kg/h).

    The hot small source is now source 1. Case 4 (order 2,1): the big source 2 first lifts the
    sink to its own 225.14 degC, then the hotter source 1 lifts it to 268.89 degC. Case 6 (order
    1,2): source 1 first (energy-limited, about 152 degC), then source 2 up to 225.14 degC.
    Final 268.89 vs 225.14 degC; heater 1 (50 kg/h) has no recovery in these cases.
    """
    si = synthetic_unequal_dryers(swap=True).to_si()
    t_big, t_hot = 29089.0 / 129.2088, 16725.8 / 62.20176
    heater_1 = _gas_kg_per_h(50.0, 15.0, 300.0)  # sink 1 has no recovery in cases 4 and 6
    case4 = k.evaluate_case(si, 4)
    case6 = k.evaluate_case(si, 6)
    assert case4.sink_2.heater_in_celsius == pytest.approx(t_hot, rel=1e-9)
    assert case6.sink_2.heater_in_celsius == pytest.approx(t_big, rel=1e-9)
    assert case4.total_gas_kg_per_h == pytest.approx(_gas_kg_per_h(100.0, t_hot, 300.0) + heater_1)
    assert case6.total_gas_kg_per_h == pytest.approx(_gas_kg_per_h(100.0, t_big, 300.0) + heater_1)
    assert case4.total_gas_kg_per_h < case6.total_gas_kg_per_h - 0.1


def test_order_is_visible_in_the_stage_powers_even_when_the_total_is_equal():
    """With the hot source on the OTHER side the totals of 4 and 6 coincide, the stages do not.

    Unswapped synthetic dryers, sink 2 (hot, small flow): its own source is the hot one, so both
    orders end at T2 = 268.89 degC (equal total), but the stage powers differ.
    """
    si = synthetic_unequal_dryers().to_si()
    case4, case6 = k.evaluate_case(si, 4), k.evaluate_case(si, 6)
    assert case4.total_gas_kg_per_h == pytest.approx(case6.total_gas_kg_per_h)
    assert case4.sink_2.stages[0].power_w != pytest.approx(case6.sink_2.stages[0].power_w)
