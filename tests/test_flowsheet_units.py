"""Tests of the units and the flowsheet of the partner model (``twinlink_ml.flowsheet``).

No workbook needed: the inputs of ``Model_v4.1.xlsx`` are typed in (as in ``test_partner_v4_1``).
The workbook comparison is ``test_flowsheet_workbook.py``.

* every unit against the code that already exists (dryer balance, as-built gas formula,
  equal-outlet loop, the counter-current solver),
* the flowsheet against the artifact of Befund 90 (``heat_recovery_comparison_r2.json``): with the
  exhausts pinned to its source variants, all switches ``"physical"`` reproduce its columns b and c
  and all ``"partner"`` its column a, in both conventions of ``dT_min``,
* structure: mirror pairs, the walk, switches that change flags only.
"""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from twinlink_ml.evaluation.net_energy import ELECTRICITY_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH
from twinlink_ml.flowsheet import (
    SWITCH_NAMES,
    Flowsheet,
    FlowsheetConfig,
    PhysicalParameters,
    Switches,
)
from twinlink_ml.flowsheet.bed_state import BedState, bed_state_at_partner_inputs
from twinlink_ml.flowsheet.burner import Burner, BurnerConfig, BurnerInputs
from twinlink_ml.flowsheet.decomposition import rank_groups, ranking_text, walk
from twinlink_ml.flowsheet.dryer import DryerBlock, DryerConfig, DryerInputs
from twinlink_ml.flowsheet.exchanger import ExchangerConfig, ExchangerInputs, HeatExchanger
from twinlink_ml.flowsheet.loop import HeatTransferLoop, LoopInputs, StageModel
from twinlink_ml.flowsheet.thermo import REPO_PROPS, capacity, exhaust_profile, partner_props
from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import ExcelDryerInputs, ExcelInputs, ExcelMaterialInputs
from twinlink_ml.partner_v4_1 import as_built as ab
from twinlink_ml.partner_v4_1 import loops
from twinlink_ml.partner_v4_1.inputs import ExcelInputsV41

ROOT = Path(__file__).resolve().parents[1]
BEFUND_90 = ROOT / "artifacts" / "runs" / "heat_recovery_comparison_r2.json"


def v41_inputs() -> ExcelInputsV41:
    """The sheet ``Inputs`` of ``Model_v4.1.xlsx`` (draft), typed in by hand."""
    plant = ExcelInputs(
        dryer_1=ExcelDryerInputs(100.0, 6.0, 150.0, 6.0, 0.4, 15.0),
        dryer_2=ExcelDryerInputs(100.0, 6.0, 300.0, 12.0, 0.4, 15.0),
        material=ExcelMaterialInputs(1.2, 45000.0, 80.0, 4.2, 1.0, 2.0, 2.0, 2500.0, 4.2),
        ambient_temperature_celsius=15.0,
        ambient_pressure_atm=1.0,
    )
    return ExcelInputsV41(plant, (20.0, 20.0))


def config(**switches: str) -> FlowsheetConfig:
    return FlowsheetConfig(v41_inputs().to_si(), Switches(**switches))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(BEFUND_90.read_text(encoding="utf-8"))


# --- units against the code that exists ----------------------------------------------------


@pytest.mark.parametrize("dryer_no", [1, 2])
def test_dryer_block_partner_is_the_workbook_balance(dryer_no: int) -> None:
    si = v41_inputs().to_si()
    d = si.plant.dryer_1 if dryer_no == 1 else si.plant.dryer_2
    unit = DryerBlock(
        DryerConfig(
            material=si.plant.material,
            ambient=si.plant.ambient,
            props=partner_props(si.plant.material),
            heat_loss_ua_w_per_k=0.0,
            product_moisture_kg_per_kg=si.plant.material.equilibrium_moisture_kg_per_kg,
        )
    )
    out = unit.solve(DryerInputs(d))
    ref = c.dryer_balance(d, si.plant.material)
    assert out.exhaust.temperature_celsius == pytest.approx(
        ref.outlet_temperature_celsius, rel=1e-12
    )
    assert out.exhaust.humidity_kg_per_kg == pytest.approx(ref.outlet_humidity_kg_per_kg, rel=1e-12)
    assert out.evaporation_kg_per_s == pytest.approx(ref.evaporation_kg_per_s, rel=1e-12)


def test_dryer_block_heat_loss_and_product_moisture_lower_and_raise_the_exhaust() -> None:
    """Loss cools the exhaust; a wetter product keeps water (less evaporation) and warms it."""
    si = v41_inputs().to_si()
    d = si.plant.dryer_2

    def run(ua: float, xp: float) -> c.Stream:
        cfg = DryerConfig(si.plant.material, si.plant.ambient, REPO_PROPS, ua, xp)
        return DryerBlock(cfg).solve(DryerInputs(d)).exhaust

    base = run(0.0, 0.002)
    assert run(2.0, 0.002).temperature_celsius < base.temperature_celsius
    wet = run(0.0, 0.0566)
    assert wet.temperature_celsius > base.temperature_celsius
    assert wet.humidity_kg_per_kg < base.humidity_kg_per_kg


def test_burner_as_built_is_the_sheet_formula_and_corrected_is_108_9_times_lower() -> None:
    si = v41_inputs().to_si()
    ex = v41_inputs()
    m = si.plant.material
    gas = c.Stream(100 / 3600, 0.006, 15.0)
    sheet = ab.gas_kg_per_h_as_built(100.0, 15.0, 300.0, 6.0, ex.plant.material)
    unit = Burner(BurnerConfig(m, partner_props(m), "partner"))
    out = unit.solve(BurnerInputs(gas, 300.0))
    assert out.fuel_kg_per_s * 3600.0 == pytest.approx(sheet, rel=1e-13)
    fixed = Burner(BurnerConfig(m, partner_props(m), "physical")).solve(BurnerInputs(gas, 300.0))
    assert fixed.fuel_kg_per_s * 3600.0 == pytest.approx(34542.0 / 36000.0, rel=1e-12)
    assert out.fuel_kg_per_s / fixed.fuel_kg_per_s == pytest.approx(108.91, abs=0.005)
    assert out.fuel_corrected_kg_per_s == fixed.fuel_corrected_kg_per_s
    assert out.duty_w == pytest.approx(fixed.duty_w)


def test_burner_does_not_run_backwards() -> None:
    m = v41_inputs().to_si().plant.material
    hot = c.Stream(100 / 3600, 0.006, 200.0)
    out = Burner(BurnerConfig(m, partner_props(m), "physical")).solve(BurnerInputs(hot, 150.0))
    assert out.duty_w == 0.0 and out.fuel_corrected_kg_per_s == 0.0


def _stage_inputs() -> tuple[LoopInputs, c.Stream]:
    si = v41_inputs().to_si()
    exhaust = c.exhaust_stream(
        si.plant.dryer_1, c.dryer_balance(si.plant.dryer_1, si.plant.material)
    )
    gas = c.Stream(100 / 3600, 0.006, 15.0)
    inp = LoopInputs(1, 1, exhaust, gas, 20 / 3600, si.cp_htm_j_per_kg_k, 150.0)
    return inp, exhaust


def test_loop_equal_outlet_is_the_document_loop() -> None:
    si = v41_inputs().to_si()
    inp, exhaust = _stage_inputs()
    stage = HeatTransferLoop(
        StageModel("a", partner_props(si.plant.material), "equal_outlet")
    ).solve(inp)
    ref = loops.equal_outlet_loop(
        exhaust, inp.gas, inp.htm_kg_per_s, si.cp_htm_j_per_kg_k, si.plant.material
    )
    assert stage.q_w == pytest.approx(ref.recovered_w, rel=1e-12)
    assert stage.gas_out_celsius == pytest.approx(ref.sink_out_celsius, rel=1e-12)
    assert stage.exhaust_out_celsius == pytest.approx(ref.exhaust_out_celsius, rel=1e-12)
    assert stage.htm_hot_celsius == pytest.approx(ref.exhaust_out_celsius, rel=1e-12)


def test_loop_counter_current_beats_equal_outlet_and_keeps_the_approach() -> None:
    si = v41_inputs().to_si()
    inp, _ = _stage_inputs()
    partner = HeatTransferLoop(StageModel("a", REPO_PROPS, "equal_outlet")).solve(inp)
    physical = HeatTransferLoop(
        StageModel("c", REPO_PROPS, "counter_current", "sensible", 10.0, "each")
    ).solve(inp)
    assert physical.q_w > partner.q_w > 0.0
    assert physical.min_approach_recovery_k >= 10.0 - 1e-9
    assert physical.min_approach_preheater_k is not None
    assert physical.min_approach_preheater_k >= 10.0 - 1e-9
    assert si.cp_htm_j_per_kg_k == 4200.0


def test_loop_refuses_a_direct_stage() -> None:
    with pytest.raises(ValueError):
        HeatTransferLoop(StageModel("d", REPO_PROPS, "direct"))


def test_condensing_equal_outlet_matches_linear_without_condensation_and_exceeds_it_with() -> None:
    """At 20 kg/h nothing condenses (same q); with a cold, large flow the latent heat adds."""
    si = v41_inputs().to_si()
    inp, _ = _stage_inputs()
    props = partner_props(si.plant.material)
    lin = StageModel("lin", props, "equal_outlet", "linear")
    con = StageModel("con", props, "equal_outlet", "condensing")
    a = HeatTransferLoop(lin).solve(inp)
    b = HeatTransferLoop(con).solve(inp)
    assert b.q_w == pytest.approx(a.q_w, rel=1e-9)
    assert b.condensate_modelled_kg_per_h == 0.0
    # a cold, large gas stream and a large medium flow pull the exhaust below its dew point
    big = replace(inp, htm_kg_per_s=400 / 3600, gas=c.Stream(400 / 3600, 0.006, 0.0))
    a2 = HeatTransferLoop(lin).solve(big)
    b2 = HeatTransferLoop(con).solve(big)
    assert b2.q_w > a2.q_w
    assert b2.condensate_modelled_kg_per_h > 0.0
    assert a2.condensate_unmodelled_kg_per_h > 0.0 and b2.condensate_unmodelled_kg_per_h == 0.0
    # the condensing circuit closes: exhaust release at its outlet equals the medium's gain
    profile = exhaust_profile(inp.exhaust, props, "condensing")
    assert profile.temperature_at(b2.q_w) == pytest.approx(b2.exhaust_out_celsius, abs=1e-9)
    assert b2.exhaust_out_celsius == pytest.approx(b2.htm_hot_celsius, abs=1e-9)


def test_heat_exchanger_unit_partner_and_physical() -> None:
    si = v41_inputs().to_si()
    _, exhaust = _stage_inputs()
    props = partner_props(si.plant.material)
    profile = exhaust_profile(exhaust, props, "linear")
    inputs = ExchangerInputs(profile, capacity(exhaust, props), 30.0, 15.0, 150.0)
    equal = HeatExchanger(ExchangerConfig("equal_outlet")).solve(inputs)
    assert equal.hot_out_celsius == pytest.approx(equal.cold_out_celsius, rel=1e-12)
    counter = HeatExchanger(ExchangerConfig("counter_current", 5.0)).solve(inputs)
    assert counter.q_w > equal.q_w
    assert counter.min_approach_k >= 5.0 - 1e-9
    assert counter.cold_out_celsius < exhaust.temperature_celsius


# --- the flowsheet: structure -------------------------------------------------------------


def test_switches_are_validated_and_hashable() -> None:
    with pytest.raises(ValueError):
        Switches(exchanger="fancy")  # type: ignore[arg-type]
    assert Switches.all_partner() != Switches.all_physical()
    assert len({Switches.all_partner(), Switches.all_physical(), Switches()}) == 2
    assert Switches().with_("cp_gas", "physical").cp_gas == "physical"
    with pytest.raises(KeyError):
        Switches().with_("nonsense", "physical")
    with pytest.raises(ValueError):
        PhysicalParameters(convention="whole")  # type: ignore[arg-type]


def test_physical_product_moisture_needs_the_bed_state() -> None:
    with pytest.raises(ValueError):
        Flowsheet(config(product_moisture="physical"))


def test_topology_is_the_case_drawing() -> None:
    assert Flowsheet.topology(0).sources_of_sink == ((), ())
    assert Flowsheet.topology(3).sources_of_sink == ((2, 1), ())
    assert Flowsheet.topology(6).sources_of_sink == ((), (1, 2))
    assert Flowsheet.topology(5).stage_count == 2


def _bed_for_tests(artifact: dict) -> BedState:
    lab = artifact["laboratory_model_dryer_2"]
    return BedState(**{k: lab[k] for k in BedState.__dataclass_fields__})


def all_states(artifact: dict) -> list[tuple[str, Switches]]:
    """All-partner, all-physical and every state of both walks."""
    out = [("partner", Switches.all_partner()), ("physical", Switches.all_physical())]
    for label, order in (("forward", SWITCH_NAMES), ("reverse", tuple(reversed(SWITCH_NAMES)))):
        s = Switches.all_partner()
        for name in order:
            s = s.with_(name, "physical")
            out.append((f"{label}:{name}", s))
    return out


@pytest.mark.parametrize("convention", ["per_exchanger", "through_circuit"])
def test_mirror_pairs_have_the_same_total_fuel_in_every_state(
    artifact: dict, convention: str
) -> None:
    """Scenarios 1/2, 3/4, 5/6 mirror each other (equal air flow and humidity of both sinks)."""
    bed = _bed_for_tests(artifact)
    for label, sw in all_states(artifact):
        cfg = FlowsheetConfig(
            v41_inputs().to_si(),
            sw,
            PhysicalParameters(convention=convention),  # type: ignore[arg-type]
            bed=bed,
        )
        fs = Flowsheet(cfg)
        r = fs.evaluate_all()
        for a, b in ((1, 2), (3, 4), (5, 6)):
            for metric in (
                "fuel_kg_per_h",
                "fuel_corrected_kg_per_h",
                "recovered_kw",
                "co2_kg_per_h",
            ):
                assert getattr(r[a], metric) == pytest.approx(getattr(r[b], metric), rel=1e-9), (
                    label,
                    a,
                    b,
                    metric,
                )


def test_electricity_is_visible_in_the_co2_and_is_a_placeholder_output() -> None:
    r = Flowsheet(config(gas_formula="physical")).evaluate(1)
    e = r.carrier_energy
    assert e.electric_kw > 0.0
    assert r.co2_kg_per_h == pytest.approx(
        ELECTRICITY_CO2_KG_PER_KWH * e.electric_kw + GAS_CO2_KG_PER_KWH * e.gas_kw
    )
    assert r.co2_kg_per_h > r.co2_fuel_only_kg_per_h
    # gas power = corrected fuel times the heating value
    assert e.gas_kw == pytest.approx(r.fuel_corrected_kg_per_h / 3600.0 * 45000.0, rel=1e-12)
    # the placeholders never enter the heat calculation
    other = replace(
        config(gas_formula="physical"),
        auxiliary=replace(config().auxiliary, loop_pressure_drop_pa=0.0),
    )
    r2 = Flowsheet(other).evaluate(1)
    assert r2.recovered_kw == r.recovered_kw
    assert r2.auxiliary.pumps_w == 0.0 < r.auxiliary.pumps_w


def test_recovery_lowers_the_co2_of_the_partner_model() -> None:
    fs = Flowsheet(config())
    base = fs.evaluate(0).co2_kg_per_h
    for case in range(1, 7):
        assert fs.evaluate(case).co2_kg_per_h < base


def test_the_gas_formula_switch_changes_fuel_only() -> None:
    a = Flowsheet(config(gas_formula="partner")).evaluate_all()
    b = Flowsheet(config(gas_formula="physical")).evaluate_all()
    for case in range(7):
        assert a[case].recovered_kw == b[case].recovered_kw
        assert a[case].fuel_corrected_kg_per_h == b[case].fuel_corrected_kg_per_h
        assert a[case].co2_kg_per_h == b[case].co2_kg_per_h
        assert a[case].fuel_kg_per_h / b[case].fuel_kg_per_h == pytest.approx(108.91, abs=0.005)


def test_boiling_check_is_a_flag_and_never_a_cap() -> None:
    """Befund 90: the medium leaves the recovery exchanger at 102.6 / 107.1 degC (6 stages)."""
    off = Flowsheet(config()).evaluate_all()
    on = Flowsheet(config(boiling_check="physical")).evaluate_all()
    flagged = {c_: on[c_].htm_boiling_stages for c_ in range(1, 7)}
    assert all(off[c_].htm_boiling_stages is None for c_ in range(7))
    assert flagged == dict.fromkeys(range(1, 7), 1)
    assert sum(flagged.values()) == 6
    assert on[0].htm_boiling_stages == 0
    for case in range(7):
        assert on[case].recovered_kw == off[case].recovered_kw
        assert on[case].fuel_kg_per_h == off[case].fuel_kg_per_h
    assert on[1].htm_hot_max_celsius == pytest.approx(102.61, abs=0.005)
    assert on[5].htm_hot_max_celsius == pytest.approx(107.14, abs=0.005)


# --- the flowsheet against Befund 90 ----------------------------------------------------------


def _pins(artifact: dict, variant: str) -> dict[int, tuple[float, float]]:
    ex = artifact["source_variants"][variant]["exhausts"]
    return {int(k): (v["temperature_celsius"], v["humidity_g_per_kg"] / 1e3) for k, v in ex.items()}


VARIANTS = ["i_partner_exhausts", "ii_laboratory_model_dryer_2"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_partner_switches_reproduce_column_a_of_befund_90(artifact: dict, variant: str) -> None:
    """All ``"partner"`` with the corrected gas formula = model ``a`` (gas in kg/h, recovery in kW).

    The pinned exhausts are the source variant's; the partner properties and equal-outlet closed
    form then give model ``a`` exactly (the artifact's ``a`` uses them for the stages as well).
    """
    cfg = FlowsheetConfig(
        v41_inputs().to_si(),
        Switches(gas_formula="physical"),
        pinned_exhausts=_pins(artifact, variant),
    )
    fs = Flowsheet(cfg)
    ref = artifact["source_variants"][variant]["wu1"]["a"]
    for case in range(1, 7):
        r = fs.evaluate(case)
        assert r.recovered_kw == pytest.approx(ref[str(case)]["recovered_kw"], rel=1e-9, abs=1e-12)
        assert r.fuel_kg_per_h == pytest.approx(ref[str(case)]["gas_kg_per_h"], rel=1e-9)
        for mine, theirs in zip(r.stages, ref[str(case)]["stages"], strict=True):
            assert mine.q_w / 1e3 == pytest.approx(theirs["recovered_kw"], rel=1e-9, abs=1e-12)
            assert mine.htm_hot_celsius == pytest.approx(theirs["htm_hot_celsius"], rel=1e-9)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dt", [20.0, 10.0, 5.0])
@pytest.mark.parametrize("convention", ["per_exchanger", "through_circuit"])
@pytest.mark.parametrize("model", ["b", "c"])
def test_physical_switches_reproduce_columns_b_and_c_of_befund_90(
    artifact: dict, variant: str, dt: float, convention: str, model: str
) -> None:
    """Exchanger, ``cp_gas``, gas formula (and condensation for ``c``) ``"physical"``.

    Befund 90's main convention is the approach *through the circuit* (key ``dt<K>``); "per
    exchanger" is its sensitivity (key ``dt<K>_per_exchanger``). Exhausts pinned to the source
    variant (Befund 90 does not recompute them with the repository's properties).
    """
    cfg = FlowsheetConfig(
        v41_inputs().to_si(),
        Switches(
            exchanger="physical",
            cp_gas="physical",
            condensation="physical" if model == "c" else "partner",
            gas_formula="physical",
            boiling_check="physical",
        ),
        PhysicalParameters(dt_min_k=dt, convention=convention),  # type: ignore[arg-type]
        pinned_exhausts=_pins(artifact, variant),
    )
    fs = Flowsheet(cfg)
    key = f"dt{dt:g}" + ("_per_exchanger" if convention == "per_exchanger" else "")
    ref = artifact["source_variants"][variant]["wu1"][key][model]
    for case in range(1, 7):
        r = fs.evaluate(case)
        assert r.recovered_kw == pytest.approx(ref[str(case)]["recovered_kw"], rel=1e-9, abs=1e-12)
        assert r.fuel_kg_per_h == pytest.approx(ref[str(case)]["gas_kg_per_h"], rel=1e-9)
        assert len(r.stages) == len(ref[str(case)]["stages"])
        flags = 0
        for mine, theirs in zip(r.stages, ref[str(case)]["stages"], strict=True):
            assert mine.q_w / 1e3 == pytest.approx(theirs["recovered_kw"], rel=1e-9, abs=1e-12)
            assert mine.gas_out_celsius == pytest.approx(theirs["gas_out_celsius"], rel=1e-9)
            assert mine.binding == theirs["binding"]
            flags += mine.above_boiling(100.0)
            assert mine.above_boiling(100.0) == theirs["htm_above_boiling_at_1_atm"]
        assert r.htm_boiling_stages == flags


def test_the_bed_state_is_the_one_of_befund_90(artifact: dict) -> None:
    """The lifted solve gives Befund 90's laboratory-model state (an extrapolation)."""
    live = bed_state_at_partner_inputs()
    ref = artifact["laboratory_model_dryer_2"]
    for key in (
        "gas_velocity_m_per_s",
        "exhaust_temperature_celsius",
        "exhaust_humidity_kg_per_kg",
        "product_moisture_kg_per_kg",
    ):
        assert getattr(live, key) == pytest.approx(ref[key], rel=1e-9), key
    assert live.feasible and live.config_fingerprint_base == ref["config_fingerprint_base"]


# --- the walk ---------------------------------------------------------------------------------


@pytest.mark.parametrize("order", [SWITCH_NAMES, tuple(reversed(SWITCH_NAMES))])
def test_walk_changes_one_switch_per_step_and_ends_where_the_other_order_ends(
    artifact: dict, order: tuple[str, ...]
) -> None:
    bed = _bed_for_tests(artifact)
    base = FlowsheetConfig(v41_inputs().to_si(), bed=bed)
    steps = walk(base, order)
    assert len(steps) == len(SWITCH_NAMES) + 1
    assert steps[0].switches == Switches.all_partner()
    assert steps[-1].switches == Switches.all_physical()
    for prev, step in pairwise(steps):
        diff = [n for n in SWITCH_NAMES if getattr(prev.switches, n) != getattr(step.switches, n)]
        assert diff == [step.changed]
    other = walk(base, tuple(reversed(order)))
    for case in range(1, 7):
        assert steps[-1].results[case].recovered_kw == pytest.approx(
            other[-1].results[case].recovered_kw, rel=1e-12
        )
        assert steps[0].results[case].recovered_kw == pytest.approx(
            other[0].results[case].recovered_kw, rel=1e-12
        )


def test_walk_rejects_a_wrong_order() -> None:
    with pytest.raises(ValueError):
        walk(config(), ("exchanger",))


def test_rank_groups_ties_and_text() -> None:
    groups = rank_groups({1: 1.0, 2: 1.0, 3: 3.0, 4: 3.0 + 1e-12, 5: 2.0, 6: 2.0})
    assert groups == ((1, 2), (5, 6), (3, 4))
    assert ranking_text(groups) == "1=2 < 5=6 < 3=4"
    assert rank_groups({1: 1.0, 2: 2.0}, descending=True) == ((2,), (1,))
