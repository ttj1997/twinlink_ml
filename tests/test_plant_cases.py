"""The six cases at plant inputs: network rebuild, series topology, points, readings, records.

Synthetic data only: two invented laboratory-scale lines stand for the lines of the deliveries
(line 1 hotter and smaller, line 2 over-saturated as delivered), invented segment means for the
measured days, invented rows for the plant sheet. No plant value, no value of a partner file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twinlink_ml.dyssol_steady import plant_cases as pcs
from twinlink_ml.dyssol_steady import plant_check as pc
from twinlink_ml.dyssol_steady.flowsheet import (
    DyssolFlowsheet,
    discover_network,
    discover_topology,
    extract_outputs,
    nominal_inputs,
)
from twinlink_ml.dyssol_steady.materials import (
    DEFAULT_HEAT_CAPACITY,
    DEFAULT_MOLAR_MASS,
    DyssolProperties,
)
from twinlink_ml.dyssol_steady.network import (
    CASES,
    evaluate_network,
    exhaust_capacity_kw_per_k,
    sink_capacity_kw_per_k,
)
from twinlink_ml.dyssol_steady.reference import KELVIN, LineInputs, evaluate, heat_recovery
from twinlink_ml.partner_v4.cascade import CASES as WORKBOOK_CASES
from twinlink_ml.partner_v4_1.comparison import DyssolFile, DyssolStream, DyssolUnit

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
LINE_1 = LineInputs(
    dry_gas_kg_per_h=80.3,
    gas_humidity_g_per_kg=6.1,
    gas_celsius=11.3,
    dryer_inlet_celsius=158.0,
    spray_kg_per_h=3.1,
    solid_fraction=0.47,
    spray_celsius=29.1,
    medium_kg_per_h=31.0,
    equilibrium_moisture_g_per_kg=31.0,
    heater_efficiency_percent=79.0,
    calorific_value_kj_per_kg=44100.0,
)
LINE_2 = replace(
    LINE_1,
    dry_gas_kg_per_h=123.4,
    dryer_inlet_celsius=81.7,
    spray_kg_per_h=6.9,
    solid_fraction=0.41,
    medium_kg_per_h=9.7,
)
"""Over-saturated as delivered; admissible with the reference solid fraction."""
POINT: pcs.Point = (LINE_1, LINE_2)
REFERENCE: pcs.Point = (LINE_1, replace(LINE_2, solid_fraction=0.65))
DENSITY = 1.2922
EXACT, FRACTION = 1e-9, 0.01
MAX_RH = 0.90
NAMES = {"KA": "Air", "KL": "H2O", "KV": "H2O_gas", "KS": "Al2O3"}
ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "runs" / "plant_cases_v1.json"


def _segment(label: str, air: float, supply: float, spray: float) -> pc.MeasuredPoint:
    return pc.MeasuredPoint(
        label=label,
        process_air_nm3h=air,
        supply_celsius=supply,
        spray_kg_h=spray,
        ambient_celsius=4.7,
        exhaust_celsius=50.0,
        bed_celsius=57.0,
    )


SEGMENTS = {
    "day_1": [_segment("day_1_s1", 101.3, 85.9, 3.1), _segment("day_1_s2", 102.9, 86.3, 3.3)],
    "day_2": [_segment("day_2_s1", 96.1, 83.7, 6.1), _segment("day_2_s2", 97.3, 84.1, 6.3)],
}
LABELS = [p.label for pts in SEGMENTS.values() for p in pts]


# ----------------------------------------------------------------------------- network rebuild
def test_cases_are_the_wiring_of_the_workbook() -> None:
    assert CASES[0] == ((), ())
    assert sorted(CASES) == [0, 1, 2, 3, 4, 5, 6]
    for case, sinks in WORKBOOK_CASES.items():
        assert CASES[case] == tuple(tuple(s - 1 for s in sink) for sink in sinks)
    assert CASES[1] == ((0,), (1,))
    assert CASES[2] == ((1,), (0,))
    assert CASES[4] == ((), (1, 0))


@pytest.mark.parametrize(("case", "pairing"), [(1, (0, 1)), (2, (1, 0))])
def test_network_equals_the_existing_rebuild_for_cases_1_and_2(case, pairing) -> None:
    for point in (POINT, REFERENCE):
        ours = evaluate_network(PROPS, point, CASES[case]).lines
        assert ours == evaluate(PROPS, point, pairing)  # the same arithmetic: equal to the last bit


def test_network_without_recovery_and_with_stages_in_series() -> None:
    none = evaluate_network(PROPS, REFERENCE, CASES[0])
    assert none.stages == ()
    assert none.recovered_kw == 0.0
    for line, out in zip(REFERENCE, none.lines, strict=True):
        assert out.preheated_celsius == pytest.approx(line.gas_celsius)
        assert out.exhaust_celsius == out.dryer_outlet_celsius
        assert out.heat_recovered_kw == 0.0

    cascade = evaluate_network(PROPS, REFERENCE, CASES[4])  # sink 2: exhaust 2, then exhaust 1
    first, second = cascade.stages
    assert (first.exhaust, first.sink, second.exhaust, second.sink) == (1, 1, 0, 1)
    assert first.cold_in_celsius == pytest.approx(LINE_2.gas_celsius)
    assert second.cold_in_celsius == first.cold_out_celsius
    assert cascade.lines[1].preheated_celsius == pytest.approx(second.cold_out_celsius)
    assert cascade.lines[0].preheated_celsius == pytest.approx(LINE_1.gas_celsius)
    assert cascade.lines[0].fuel_kg_per_h == pytest.approx(none.lines[0].fuel_kg_per_h)
    assert cascade.lines[1].fuel_kg_per_h < none.lines[1].fuel_kg_per_h
    # the second stage by hand, with the unit function and the first stage's outlet
    gas = [ln.dry_gas_kg_per_h / 3600 for ln in REFERENCE]
    vapour_in = gas[1] * REFERENCE[1].gas_humidity_g_per_kg / 1000
    vapour_out = gas[0] * none.lines[0].dryer_outlet_humidity_g_per_kg / 1000
    t1, t2, q = heat_recovery(
        PROPS,
        (gas[0], vapour_out, none.lines[0].dryer_outlet_celsius + KELVIN),
        (gas[1], vapour_in, first.cold_out_celsius + KELVIN),
        LINE_1.medium_kg_per_h / 3600,
    )
    assert second.heat_kw == pytest.approx(q / 1000)
    assert second.cold_out_celsius == pytest.approx(t1 - KELVIN)
    assert cascade.lines[0].exhaust_celsius == pytest.approx(t2 - KELVIN)
    assert cascade.fuel_kg_per_h == pytest.approx(sum(o.fuel_kg_per_h for o in cascade.lines))


def test_hot_exhaust_first_gives_a_stage_with_negative_recovery() -> None:
    hot_first = evaluate_network(PROPS, REFERENCE, CASES[6])  # sink 2: exhaust 1, then exhaust 2
    assert hot_first.stages[1].heat_kw < 0.0
    assert hot_first.negative_stages == 1
    assert evaluate_network(PROPS, REFERENCE, CASES[4]).negative_stages == 0


def test_network_admissibility_and_wiring_errors() -> None:
    assert not evaluate_network(PROPS, POINT, CASES[1]).dryers_admissible
    assert evaluate_network(PROPS, REFERENCE, CASES[1]).admissible
    # an exhaust near saturation that is cooled below its dew point by the recovery
    wet = replace(REFERENCE[1], solid_fraction=0.62, medium_kg_per_h=500.0)
    out = evaluate_network(PROPS, (LINE_1, wet), CASES[1])
    assert out.dryers_admissible
    assert not out.admissible
    with pytest.raises(ValueError, match="wiring"):
        evaluate_network(PROPS, POINT, ((0,), (0,)))
    with pytest.raises(ValueError, match="wiring"):
        evaluate_network(PROPS, POINT, ((0,),))
    assert sink_capacity_kw_per_k(PROPS, LINE_2) > sink_capacity_kw_per_k(PROPS, LINE_1)
    assert exhaust_capacity_kw_per_k(PROPS, LINE_1) > sink_capacity_kw_per_k(PROPS, LINE_1)


# ----------------------------------------------------------------------------- topology
def _fake_network(lines: pcs.Point, wiring: tuple[tuple[int, ...], ...]) -> DyssolFile:
    """A ``DyssolFile`` as the reader returns it, filled with the results of the rebuild."""
    out = evaluate_network(PROPS, lines, wiring)
    streams: dict[str, DyssolStream] = {}
    units: dict[str, DyssolUnit] = {}

    def stream(
        name: str, mass: float, celsius: float, vapour: float = 0.0, solid: float = 0.0
    ) -> str:
        streams[name] = DyssolStream(
            name,
            f"key-{name}",
            3600.0,
            mass,
            celsius,
            {
                "Solid": solid,
                "Liquid": (1.0 - solid) if solid else 0.0,
                "Gas": 0.0 if solid else 1.0,
            },
            {
                "Solid": (0.0, 0.0, 0.0, 1.0),
                "Liquid": (0.0, 1.0, 0.0, 0.0),
                "Gas": (1.0 - vapour, 0.0, vapour, 0.0),
            },
        )
        return f"key-{name}"

    heater_in: dict[int, str] = {}
    for i, (ln, res) in enumerate(zip(lines, out.lines, strict=True)):
        y = ln.gas_humidity_g_per_kg / 1000
        total = ln.dry_gas_kg_per_h * (1 + y) / 3600
        y_out = res.dryer_outlet_humidity_g_per_kg / 1000
        wet = ln.dry_gas_kg_per_h * (1 + y_out) / 3600
        k_gas = stream(f"Gas in {i + 1}", total, ln.gas_celsius, y / (1 + y))
        k_hot = stream(f"Dryer in {i + 1}", total, ln.dryer_inlet_celsius, y / (1 + y))
        k_feed = stream(
            f"Feed {i + 1}", ln.spray_kg_per_h / 3600, ln.spray_celsius, solid=ln.solid_fraction
        )
        k_out = stream(f"Dryer out {i + 1}", wet, res.dryer_outlet_celsius, y_out / (1 + y_out))
        stream(f"Gas out {i + 1}", wet, res.exhaust_celsius, y_out / (1 + y_out))
        k_prod = stream(
            f"Product {i + 1}", res.product_kg_per_h / 3600, res.dryer_outlet_celsius, solid=0.9
        )
        cold = k_gas
        for source in wiring[i]:
            stage = next(s for s in out.stages if s.exhaust == source and s.sink == i)
            warm = stream(f"Pre {i + 1}.{source + 1}", total, stage.cold_out_celsius, y / (1 + y))
            units[f"Recovery {source + 1}"] = DyssolUnit(
                f"Recovery {source + 1}",
                {
                    "Heat recovery inlet": f"key-Dryer out {source + 1}",
                    "Preheater inlet": cold,
                    "Heat recovery outlet": f"key-Gas out {source + 1}",
                    "Preheater outlet": warm,
                },
                {"Medium mass flow": (lines[source].medium_kg_per_h,)},
                {"Heat recovered [kW]": stage.heat_kw},
            )
            cold = warm
        heater_in[i] = cold
        units[f"Gas inlet {i + 1}"] = DyssolUnit(
            f"Gas inlet {i + 1}", {"InletMaterial": k_gas}, {}, {}
        )
        units[f"Suspension {i + 1}"] = DyssolUnit(
            f"Suspension {i + 1}", {"InletMaterial": k_feed}, {}, {}
        )
        units[f"Heater {i + 1}"] = DyssolUnit(
            f"Heater {i + 1}",
            {"InPort": heater_in[i], "OutPort": k_hot},
            {
                "Efficiency": (ln.heater_efficiency_percent,),
                "Calorific Value": (ln.calorific_value_kj_per_kg,),
                "Outlet Temperature": (ln.dryer_inlet_celsius,),
            },
            {"Heat duty [kW]": res.heater_duty_kw, "Fuel consumption [kg/h]": res.fuel_kg_per_h},
        )
        units[f"Dryer {i + 1}"] = DyssolUnit(
            f"Dryer {i + 1}",
            {"GasIn": k_hot, "Feed": k_feed, "GasOut": k_out, "Product": k_prod},
            {"Equilibrium moisture content": (ln.equilibrium_moisture_g_per_kg,)},
            {},
        )
    return DyssolFile(("KA", "KL", "KV", "KS"), streams, units)


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5, 6])
def test_series_topology_inputs_and_outputs_are_read_from_the_file(case) -> None:
    file = _fake_network(REFERENCE, CASES[case])
    topology = discover_network(file)
    assert topology.sources_of_sink == CASES[case]
    assert [ln.dryer for ln in topology.lines] == ["Dryer 1", "Dryer 2"]
    for got, want in zip(nominal_inputs(file, topology, NAMES), REFERENCE, strict=True):
        assert asdict(got) == pytest.approx(asdict(want))
    expected = evaluate_network(PROPS, REFERENCE, CASES[case]).lines
    for got, want in zip(extract_outputs(file, topology, NAMES), expected, strict=True):
        assert asdict(got) == pytest.approx(asdict(want))


def test_series_topology_agrees_with_the_existing_one_and_keeps_it_the_default() -> None:
    file = _fake_network(REFERENCE, CASES[2])
    assert discover_network(file).lines == discover_topology(file).lines
    assert discover_topology(file).preheated_by == (1, 0)
    with pytest.raises(ValueError):  # a cascade stays outside the existing family
        discover_topology(_fake_network(REFERENCE, CASES[4]))
    import inspect

    default = inspect.signature(DyssolFlowsheet.__init__).parameters["discover"].default
    assert default is discover_topology


def test_series_topology_rejects_a_missing_recovery() -> None:
    file = _fake_network(REFERENCE, CASES[3])
    units = dict(file.units)
    del units["Recovery 2"]
    with pytest.raises(ValueError):
        discover_network(DyssolFile(file.compounds, file.streams, units))


# ----------------------------------------------------------------------------- points
def test_lines_are_assigned_by_the_stored_dryer_inlet() -> None:
    stored = (replace(LINE_1, dryer_inlet_celsius=160.0), replace(LINE_2, dryer_inlet_celsius=80.0))
    assert pcs.assign_lines([LINE_2, LINE_1], stored) == (1, 0)  # reversed numbering of a sheet
    assert pcs.assign_lines([LINE_1, LINE_2], stored) == (0, 1)
    with pytest.raises(ValueError, match="one-to-one"):
        pcs.assign_lines([LINE_1, LINE_1], stored)


def test_exchanged_sizes_keep_both_dryer_exhausts() -> None:
    one, two = pcs.exchange_sizes(REFERENCE)
    assert one.dry_gas_kg_per_h == pytest.approx(REFERENCE[1].dry_gas_kg_per_h)
    assert two.dry_gas_kg_per_h == pytest.approx(REFERENCE[0].dry_gas_kg_per_h)
    assert (one.medium_kg_per_h, two.medium_kg_per_h) == (31.0, 9.7)
    before = pcs.evaluate_units(PROPS, REFERENCE)
    after = pcs.evaluate_units(PROPS, (one, two))
    assert after.exhaust_celsius == pytest.approx(before.exhaust_celsius)
    assert after.saturation == pytest.approx(before.saturation)
    assert after.sink_capacity_kw_per_k == pytest.approx(before.sink_capacity_kw_per_k[::-1])
    scaled = pcs.exchange_sizes(REFERENCE, scale_circuits=True)
    assert scaled[0].medium_kg_per_h == pytest.approx(31.0 * 123.4 / 80.3)


def _points() -> dict[str, pcs.Point]:
    return pcs.build_points(
        POINT,
        SEGMENTS,
        solid_reference=0.65,
        solid_band=(0.6, 0.7),
        humidity_reference_g_per_kg=0.024,
        normal_density_kg_per_m3=DENSITY,
    )


def test_points_of_the_study() -> None:
    points = _points()
    assert points["S"] == POINT
    assert points["R"] == REFERENCE
    assert len(points) == 1 + 1 + 4 + 1 + 4 + 2 * (1 + 4)
    one, two = points["M|day_2_s1"]
    assert one == replace(LINE_1, gas_celsius=4.7)  # line 1 as the sheet, the day's fresh gas
    assert two.dry_gas_kg_per_h == pytest.approx(96.1 * DENSITY)
    assert (two.dryer_inlet_celsius, two.spray_kg_per_h, two.gas_celsius) == (83.7, 6.1, 4.7)
    assert (two.solid_fraction, two.gas_humidity_g_per_kg) == (0.65, 0.024)
    assert two.equilibrium_moisture_g_per_kg == LINE_2.equilibrium_moisture_g_per_kg
    assert points["M|day_2_s1|sf=0.7"][1].solid_fraction == 0.7
    assert points["R|sf=0.6"][1].solid_fraction == 0.6
    assert points["X|M|day_1_s2"] == pcs.exchange_sizes(points["M|day_1_s2"])
    flows = pcs.with_flows(points["R"], (5.0, 7.0))
    assert (flows[0].medium_kg_per_h, flows[1].medium_kg_per_h) == (5.0, 7.0)


# ----------------------------------------------------------------------------- models
def test_units_model_at_the_invented_points() -> None:
    delivered = pcs.evaluate_units(PROPS, POINT)
    assert delivered.saturation[1] > 1.0  # line 2 over-saturated as delivered
    assert not any(c.admissible for c in delivered.cases)
    result = pcs.evaluate_units(PROPS, REFERENCE)
    assert [c.case for c in result.cases] == [0, 1, 2, 3, 4, 5, 6]
    assert all(c.admissible for c in result.cases)
    assert result.exhaust_celsius[0] > result.exhaust_celsius[1]
    assert result.sink_capacity_kw_per_k[1] > result.sink_capacity_kw_per_k[0]
    fuel = result.fuel()
    assert fuel[0] > fuel[1] > fuel[2]  # the warm exhaust belongs to the larger sink
    assert result.cases[6].negative_stages == 1


def test_workbook_model_needs_one_fresh_gas_and_follows_the_same_order() -> None:
    with pytest.raises(ValueError, match="gas_celsius"):
        pcs.workbook_inputs((LINE_1, replace(LINE_2, gas_celsius=3.0)))
    inputs = pcs.workbook_inputs(REFERENCE)
    assert inputs.htm_flow_kg_per_h == (31.0, 9.7)
    assert inputs.plant.material.equilibrium_moisture_g_per_kg == 31.0
    assert inputs.plant.material.cp_gas_kj_per_kg_k == 1.006  # properties of the workbook
    assert inputs.plant.dryer_2.spray_solid_fraction_kg_per_kg == 0.65
    for mode in ("partner", "physical_unpressurised_95"):
        result = pcs.evaluate_workbook(REFERENCE, mode, MAX_RH)
        assert result.model == f"W:{mode}"
        fuel = result.fuel()
        assert fuel[0] > fuel[1] > fuel[2]
        assert all(c.admissible for c in result.cases)
    details = pcs.workbook_case_details(REFERENCE, "partner", 2)
    total = pcs.evaluate_workbook(REFERENCE, "partner", MAX_RH).fuel()[2]
    assert sum(details["fuel_kg_per_h"]) == pytest.approx(total)
    assert not all(c.admissible for c in pcs.evaluate_workbook(POINT, "partner", MAX_RH).cases)


# ----------------------------------------------------------------------------- readings
def _result(fuel: list[float], admissible: list[bool] | None = None) -> pcs.ModelResult:
    admissible = admissible or [True] * 7
    return pcs.ModelResult(
        model=pcs.MODEL_U,
        exhaust_celsius=(100.0, 40.0),
        saturation=(0.1, 0.5),
        sink_capacity_kw_per_k=(7.0, 11.0),
        cases=tuple(
            pcs.CaseResult(k, f, 0.0, a, 0)
            for k, (f, a) in enumerate(zip(fuel, admissible, strict=True))
        ),
    )


def test_case_one_against_the_best_case() -> None:
    entry = pcs.case_one(_result([100.0, 90.0, 80.0, 95.0, 81.0, 99.0, 97.0]), EXACT, FRACTION)
    assert entry["best_case"] == 2
    assert entry["order"] == [2, 4, 1, 3, 6, 5]
    assert not entry["best_or_practically_equal"]
    assert entry["regret_case_1_percent_of_best_saving"] == pytest.approx(50.0)
    assert entry["regret_case_1_percent_of_best_fuel"] == pytest.approx(12.5)
    assert entry["saving_percent"]["2"] == pytest.approx(20.0)
    # practical tie: the gap is below 1 % of the best saving
    tie = pcs.case_one(_result([100.0, 80.1, 80.0, 95.0, 95.0, 99.0, 97.0]), EXACT, FRACTION)
    assert tie["best_or_practically_equal"]
    assert not pcs.case_one(_result([100.0, 80.3, 80.0, 95.0, 95.0, 99.0, 97.0]), EXACT, FRACTION)[
        "best_or_practically_equal"
    ]
    # the best case is taken among the admissible ones; an inadmissible case 1 is not best
    masked = _result(
        [100.0, 70.0, 80.0, 95.0, 81.0, 99.0, 97.0], [True, False, True, True, True, True, True]
    )
    assert pcs.case_one(masked, EXACT, FRACTION)["best_case"] == 2
    assert not pcs.case_one(masked, EXACT, FRACTION)["best_or_practically_equal"]
    none = pcs.case_one(_result([100.0] * 7, [False] * 7), EXACT, FRACTION)
    assert none["best_case"] is None
    assert not none["best_or_practically_equal"]


def test_reading_codes() -> None:
    assert pcs.reading_f([True, True]) == "F0"
    assert pcs.reading_f([True, False]) == "F1"
    assert pcs.reading_c([True, True]) == "C0"
    assert pcs.reading_c([False, False]) == "C1"
    assert pcs.reading_c([True, False]) == "C2"
    with pytest.raises(ValueError):
        pcs.reading_c([])
    assert pcs.reading_a({"U": False, "W": False}, {"U": True, "W": True}) == "A0"
    assert pcs.reading_a({"U": False, "W": False}, {"U": True, "W": False}) == "A1"
    assert pcs.reading_a({"U": True, "W": False}, {"U": True, "W": True}) == "not applicable"


def test_rules_at_one_point() -> None:
    result = _result([100.0, 90.0, 80.0, 95.0, 81.0, 99.0, 97.0])
    assert pcs.worded_rule_predicts_case_1(result)  # exhaust 1 is the warmer one
    colder = replace(result, exhaust_celsius=(40.0, 100.0))
    assert not pcs.worded_rule_predicts_case_1(colder)
    # the physical modes split at -0.54 K, the partner mode and U at -15.5 K
    near = replace(result, exhaust_celsius=(90.0, 100.0))
    assert pcs.worded_rule_predicts_case_1(near)
    assert not pcs.worded_rule_predicts_case_1(replace(near, model="W:physical_pressurised"))
    assert pcs.generalised_rule_case((100.0, 40.0), (7.0, 11.0)) == 2
    assert pcs.generalised_rule_case((100.0, 40.0), (11.0, 7.0)) == 1
    assert pcs.generalised_rule_case((40.0, 100.0), (7.0, 11.0)) == 1
    assert pcs.generalised_rule_case((40.0, 40.0), (7.0, 11.0)) is None
    assert pcs.pair_choice(result, EXACT) == 2
    assert pcs.pair_choice(_result([100.0, 80.0, 80.0, 95.0, 81.0, 99.0, 97.0]), EXACT) is None


def test_pair_rule_hit_rates_over_many_points() -> None:
    fuel = np.array(
        [
            [10.0, 8.0, 9.0, 9.5, 9.5, 9.5, 9.5],  # case 1 better, exhaust 1 warmer, sink 1 larger
            [10.0, 9.0, 8.0, 9.5, 9.5, 9.5, 9.5],  # case 2 better, exhaust 2 warmer
            [10.0, 9.0, 8.0, 9.5, 9.5, 9.5, 9.5],  # case 2 better, exhaust 1 warmer by 0.2 K
            [10.0, 8.0, 8.0, 9.5, 9.5, 9.5, 9.5],  # exact tie: not decided
            [10.0, 8.0, 8.001, 9.5, 9.5, 9.5, 9.5],  # decided, inside the practical tie
        ]
    )
    exhaust = np.array([[90.0, 60.0], [60.0, 90.0], [60.2, 60.0], [70.0, 60.0], [70.0, 60.0]])
    capacity = np.tile([12.0, 10.0], (5, 1))
    rates = pcs.pair_rule_hit_rates(
        fuel, exhaust, capacity, exact=EXACT, fraction=FRACTION, worded_threshold_k=-0.54
    )
    assert rates["points"] == 5
    assert rates["generalised"]["decided"] == {"n": 4, "hits": 3, "rate": 0.75}
    assert rates["generalised"]["clear"]["n"] == 3
    assert rates["worded"]["decided"]["hits"] == 3
    assert rates["rules_differ"]["decided_points"] == 0
    assert rates["sink_1_larger_share"] == 1.0
    # with the sizes the other way round the generalised rule turns, the worded one does not
    turned = pcs.pair_rule_hit_rates(
        fuel, exhaust, capacity[:, ::-1], exact=EXACT, fraction=FRACTION, worded_threshold_k=-0.54
    )
    assert turned["generalised"]["decided"]["hits"] == 1
    assert turned["worded"]["decided"]["hits"] == 3
    assert turned["rules_differ"] == {
        "decided_points": 4,
        "generalised_right": 1,
        "worded_right": 3,
    }


# ----------------------------------------------------------------------------- plant sheet
def _sheet_rows() -> list[list[Any]]:
    def block(number: int, name: str, gas: float, inlet: float, spray: float, medium: float):
        e = None
        return [
            [
                f"Process {number}",
                name,
                "Heat recovery",
                "Fresh gas",
                "Mass flow rate",
                "kg/hr",
                gas,
            ],
            [e, e, e, e, "Moisture content", "gm/kg", 9.5],
            [e, e, e, e, "Temperature", "°C", 21.0],
            [e, e, "Heating transfer medium", e, "Flow rate", "kg/hr", medium],
            [e, e, "Dryer ", "Process air", "Mass flow rate", "kg/hr", gas],
            [e, e, e, e, "Moisture content", "gm/kg", 9.5],
            [e, e, e, e, "Temperature", "°C", inlet],
            [e, e, e, "Spray rate", "Mass flow rate", "kg/hr", spray],
            [e, e, e, e, "Solid fraction", "-", 0.45],
            [e, e, e, e, "Temperature", "°C", 31.0],
            [e, e, "Heater", "Efficiency", e, "%", 79.0],
            [e, e, e, "Calorific value", e, "kJ/kg", 44100.0],
            [e] * 7,
        ]

    rows = block(1, "Tower", 80.0, 158.0, 3.0, 31.0) + block(2, "Bed", 120.0, 82.0, 7.0, 9.0)
    rows = [row + [None] * (16 - len(row)) for row in rows]
    rows[0][10], rows[0][12], rows[0][14] = "Scenarios", "Case 1", "Case 2"
    rows[1][10] = "Results"
    rows[1][12:16] = ["Excel", "Dyssol", "Excel", "Dyssol"]
    rows[2][10:16] = ["Enegry recovery (kW)", "Tower", 2.5, 2.52, 3.1, None]
    rows[3][11:16] = ["Bed", 0.4, 0.41, 0.3, None]
    rows[4][10:16] = ["Fuel consumption(kg/h)", "Tower", 0.76, 0.75, 0.95, None]
    rows[5][11:16] = ["Bed", 0.37, 0.36, 0.19, None]
    rows[6][10:16] = ["Total fuel consumption", "(kg/h)", 1.13, 1.11, 1.14, 0]
    return rows


def test_plant_sheet_reader() -> None:
    sheet = pcs.parse_plant_sheet(_sheet_rows())
    assert len(sheet.lines) == 2
    line = sheet.line_inputs(1, 31.0)
    assert (line.dry_gas_kg_per_h, line.gas_humidity_g_per_kg, line.gas_celsius) == (
        120.0,
        9.5,
        21.0,
    )
    assert (line.dryer_inlet_celsius, line.spray_kg_per_h, line.solid_fraction) == (82.0, 7.0, 0.45)
    assert (line.spray_celsius, line.medium_kg_per_h) == (31.0, 9.0)
    assert (line.heater_efficiency_percent, line.calorific_value_kj_per_kg) == (79.0, 44100.0)
    assert line.equilibrium_moisture_g_per_kg == 31.0
    assert sheet.fuel[1] == {"excel": (0.76, 0.37), "dyssol": (0.75, 0.36)}
    assert sheet.fuel[2] == {"excel": (0.95, 0.19), "dyssol": (None, None)}
    assert sheet.recovered[2]["excel"] == (3.1, 0.3)
    assert sheet.total[1] == {"excel": 1.13, "dyssol": 1.11}
    assert sheet.total[2] == {"excel": 1.14, "dyssol": None}  # a sum over empty cells is no total
    broken = _sheet_rows()
    broken[6][5] = "K"
    with pytest.raises(ValueError, match="expected"):
        pcs.parse_plant_sheet(broken)
    missing = [row[:10] + [None] * 6 for row in _sheet_rows()]
    with pytest.raises(ValueError, match="header row"):
        pcs.parse_plant_sheet(missing)


def test_within_rounding() -> None:
    assert pcs.within_rounding(1.1349, 1.13, 2)
    assert not pcs.within_rounding(1.1351, 1.13, 2)
    assert pcs.within_rounding(1.139, 1.13, 2, halves=2)
    assert not pcs.within_rounding(1.141, 1.13, 2, halves=2)


# ----------------------------------------------------------------------------- records
GRID = ("grid|0", "grid|1")
PREDICTIONS = {
    "F": dict.fromkeys(pcs.MODELS, "F0"),
    "C": dict.fromkeys(pcs.MODELS, "C1"),
    "A": "A0",
    "A_models": (pcs.MODEL_U, "W:physical_unpressurised_95"),
}


@pytest.fixture(scope="module")
def records() -> tuple[dict[str, Any], dict[str, Any]]:
    points = _points()
    read_points = ["R", *(f"M|{label}" for label in LABELS)]
    stored = (31.0, 9.7)
    cells: dict[tuple[str, str], pcs.Point] = {}
    for tag, point in points.items():
        cells[tag, pcs.READ_SETTING] = pcs.with_flows(point, stored)
        if tag in ("S", "X", *read_points):
            for setting, flow in zip(GRID, (10.0, 40.0), strict=True):
                cells[tag, setting] = pcs.with_flows(point, (flow, flow))
    cells["X|scaled", pcs.READ_SETTING] = pcs.exchange_sizes(
        pcs.with_flows(points["R"], stored), scale_circuits=True
    )
    results: dict[str, dict[str, dict[str, pcs.ModelResult]]] = {}
    for (tag, setting), point in cells.items():
        entry = results.setdefault(tag, {}).setdefault(setting, {})
        entry[pcs.MODEL_U] = pcs.evaluate_units(PROPS, point)
        for mode in ("partner", "physical_unpressurised_95", "physical_pressurised"):
            entry[f"W:{mode}"] = pcs.evaluate_workbook(point, mode, MAX_RH)
    return pcs.summarise(
        results=results,
        read_points=read_points,
        groups={"R": ["R"], **{d: [f"M|{p.label}" for p in pts] for d, pts in SEGMENTS.items()}},
        grid_settings=GRID,
        band=("0.6", "0.7"),
        exact=EXACT,
        fraction=FRACTION,
        sensitivity=(0.005, 0.02),
        predictions=PREDICTIONS,
        points={
            tag: point for (tag, setting), point in cells.items() if setting == pcs.READ_SETTING
        },
    )


def test_readings_at_the_invented_points(records) -> None:
    repo, _ = records
    units = repo["models"][pcs.MODEL_U]
    assert repo["headline_model"] == pcs.MODEL_U
    assert units["reading_f"] == {
        "code": "F0",
        "cells": 30,
        "admissible_cells": 30,
        "admissible_cells_per_case": dict.fromkeys("123456", 5),
    }
    assert units["reading_c"]["code"] == "C1"
    assert units["reading_c"]["per_group"] == {"R": "C1", "day_1": "C1", "day_2": "C1"}
    assert set(units["reading_c"]["best_case_per_point"].values()) == {2}
    assert units["worded_rule"]["points_where_it_predicts_case_1"] == 5
    assert units["worded_rule"]["points_where_it_predicts_case_1_and_case_1_is_best_or_equal"] == 0
    rule = units["generalised_rule"]
    assert "post-hoc" in rule["status"]
    assert (rule["points"], rule["decided_points"], rule["hits"]) == (6, 6, 6)
    assert rule["per_point"]["X"]["generalised_rule_case"] == 1
    assert units["exchanged_sizes"]["X"]["best_or_practically_equal"]
    assert repo["reading_a"]["code"] == "A0"
    assert repo["predictions"]["A"] == {"predicted": "A0", "found": "A0", "confirmed": True}
    assert repo["predictions"]["C"][pcs.MODEL_U]["confirmed"]
    assert set(units["band"]) == {"0.6", "0.7"}
    assert len(units["grid"]["R"]["best_case_per_flow"]) == 2
    assert not units["as_delivered"]["S"]["dryers_admissible"]
    assert sorted(units["as_delivered"]["S"]["order_of_the_formula"]) == [1, 2, 3, 4, 5, 6]
    assert set(repo["models"]) == set(pcs.MODELS)


def test_repository_record_names_no_absolute_quantity(records) -> None:
    repo, local = records
    allowed = {"exact_kg_per_h"}

    def keys(node: Any) -> set[str]:
        if isinstance(node, dict):
            return set(map(str, node)) | {k for v in node.values() for k in keys(v)}
        if isinstance(node, list):
            return {k for v in node for k in keys(v)}
        return set()

    forbidden = ("kg_per_h", "celsius", "_kw", "flow", "humidity_g", "inlet")
    named = {k for k in keys(repo) if any(f in k for f in forbidden)} - allowed
    assert named <= {"best_case_per_flow", "case_1_best_or_practically_equal_per_flow"}
    json.dumps(repo)  # serialisable
    text = json.dumps(repo)
    for value in (LINE_1.dry_gas_kg_per_h, LINE_2.dry_gas_kg_per_h, LINE_1.dryer_inlet_celsius):
        assert f"{value}" not in text
    assert "points" in local
    assert "results" in local
    assert local["points"]["R"][1]["solid_fraction"] == 0.65
    assert "fuel_kg_per_h" in json.dumps(local)


# ----------------------------------------------------------------------------- the artifact
def _span(values: list[float]) -> tuple[float, float]:
    return round(min(values), 2), round(max(values), 2)


def test_artifact_holds_the_numbers_of_the_document_and_of_the_lab_book_entry() -> None:
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert art["provenance"]["git/dirty"] is False
    assert art["provenance"]["code/root_checked"] is True
    control = art["control"]
    assert control["read_units"] is True
    assert control["read_workbook"] is True
    assert control["a_extended_rebuild"] == {
        "cells": 158,
        "max_relative_difference": 0.0,
        "tolerance": 1e-12,
        "held": True,
    }
    assert control["b_segments"]["saturation_ratio_max_relative_deviation"] == 0.0
    dyssol = control["c_dyssol"]
    assert (dyssol["dyssol_runs"], dyssol["points"], dyssol["points_compared"]) == (
        1106,
        1896,
        1896,
    )
    assert dyssol["failures"] == []
    assert f"{dyssol['max_temperature_k']:.1e}" == "2.3e-13"
    assert f"{dyssol['max_relative_percent']:.1e}" == "5.6e-11"
    excel = control["d_workbook_against_excel"]["flow_settings"]
    assert excel["stored"]["totals_within_rounding"] == 6
    assert excel["stored"]["fuel_values_within_rounding"] == [12, 12]
    assert excel["stored"]["recovered_heats_within_rounding"] == [12, 12]
    assert round(excel["stored"]["largest_total_difference_in_units_of_the_last_decimal"], 2) == 0.9
    assert excel["stated"]["totals_within_rounding"] == 0
    assert round(excel["stated"]["largest_total_difference_in_units_of_the_last_decimal"]) == 193

    days = art["test_set"]["groups"]
    assert len(art["test_set"]["read_points"]) == 9
    orders = {
        pcs.MODEL_U: [2, 4, 1, 6, 3, 5],
        "W:partner": [2, 4, 1, 6, 3, 5],
        "W:physical_unpressurised_95": [2, 4, 6, 1, 3, 5],
        "W:physical_pressurised": [2, 4, 6, 1, 3, 5],
    }
    regrets = {
        pcs.MODEL_U: (9.51, (8.10, 8.28), (9.62, 10.05), 2.44, (2.39, 2.40), (2.45, 2.49)),
        "W:partner": (9.56, (8.06, 8.26), (9.56, 10.05), 2.45, (2.37, 2.39), (2.43, 2.48)),
        "W:physical_unpressurised_95": (
            23.62,
            (23.56, 25.37),
            (23.91, 24.75),
            11.46,
            (14.19, 16.71),
            (11.28, 11.72),
        ),
    }
    regrets["W:physical_pressurised"] = regrets["W:physical_unpressurised_95"]
    for model in pcs.MODELS:
        entry = art["models"][model]
        assert entry["reading_f"]["code"] == "F0"
        assert (entry["reading_f"]["cells"], entry["reading_f"]["admissible_cells"]) == (54, 54)
        assert entry["reading_c"]["code"] == "C1"
        assert entry["reading_c"]["case_1_best_or_practically_equal"] == 0
        assert entry["reading_c"]["sensitivity"] == {"0.005": "C1", "0.02": "C1"}
        assert set(entry["reading_c"]["best_case_per_point"].values()) == {2}
        assert all(p["order"] == orders[model] for p in entry["points"].values())
        assert entry["worded_rule"]["points_where_it_predicts_case_1"] == 9
        assert (
            entry["worded_rule"]["points_where_it_predicts_case_1_and_case_1_is_best_or_equal"] == 0
        )
        rule = entry["generalised_rule"]
        assert (rule["points"], rule["decided_points"], rule["hits"]) == (10, 10, 10)
        assert rule["per_point"]["R"]["generalised_rule_case"] == 2
        assert rule["per_point"]["X"]["generalised_rule_case"] == 1
        exchanged = entry["exchanged_sizes"]
        assert exchanged["X"]["best_case"] == 1
        assert exchanged["X_scaled_circuits"]["best_case"] == 1
        assert exchanged["measured_points_with_case_1_best_or_practically_equal"] == 8
        for label in ("0.634", "0.7"):
            assert entry["band"][label] == {
                "reading_f": "F0",
                "reading_c": "C1",
                "best_cases": ["2"],
            }
        assert entry["band"]["0.6"]["reading_c"] == "C1"
        r_saving, day_1, day_2, r_fuel, fuel_1, fuel_2 = regrets[model]
        saving = {t: p["regret_case_1_percent_of_best_saving"] for t, p in entry["points"].items()}
        fuel = {t: p["regret_case_1_percent_of_best_fuel"] for t, p in entry["points"].items()}
        assert round(saving["R"], 2) == r_saving
        assert round(fuel["R"], 2) == r_fuel
        assert _span([saving[t] for t in days["day_1"]]) == day_1
        assert _span([saving[t] for t in days["day_2"]]) == day_2
        assert _span([fuel[t] for t in days["day_1"]]) == fuel_1
        assert _span([fuel[t] for t in days["day_2"]]) == fuel_2
    units = art["models"][pcs.MODEL_U]
    assert units["band"]["0.6"] == {"reading_f": "F1", "reading_c": "C1", "best_cases": ["2", "6"]}
    assert art["models"]["W:partner"]["band"]["0.6"]["reading_f"] == "F0"
    assert units["exchanged_sizes"]["X"]["order"] == [1, 3, 2, 5, 4, 6]
    physical = art["models"]["W:physical_unpressurised_95"]
    assert physical["exchanged_sizes"]["X"]["order"] == [1, 3, 5, 2, 4, 6]

    def savings(model: str, case: str) -> tuple[float, tuple[float, float], tuple[float, float]]:
        points = art["models"][model]["points"]
        return (
            round(points["R"]["saving_percent"][case], 2),
            _span([points[t]["saving_percent"][case] for t in days["day_1"]]),
            _span([points[t]["saving_percent"][case] for t in days["day_2"]]),
        )

    assert savings(pcs.MODEL_U, "2") == (20.42, (22.49, 22.74), (19.83, 20.32))
    assert savings(pcs.MODEL_U, "1") == (18.48, (20.62, 20.90), (17.83, 18.36))
    assert savings("W:physical_unpressurised_95", "2") == (32.67, (37.58, 39.72), (31.30, 32.90))
    assert art["reading_a"]["code"] == "A0"
    assert all(c["confirmed"] for c in art["predictions"]["F"].values())
    assert all(c["confirmed"] for c in art["predictions"]["C"].values())
    assert art["predictions"]["A"]["confirmed"]

    # negative recovery: cases 5 and 6 at R and on day 2 (case 6 not at the last segment)
    for model in (pcs.MODEL_U, "W:partner"):
        points = art["models"][model]["points"]
        assert points["R"]["negative_stages"] == {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 1}
        assert all(sum(points[t]["negative_stages"].values()) == 0 for t in days["day_1"])
        assert [points[t]["negative_stages"]["5"] for t in days["day_2"]] == [1, 1, 1, 1]
        assert [points[t]["negative_stages"]["6"] for t in days["day_2"]] == [1, 1, 1, 0]
        grid = art["models"][model]["grid"]
        assert all(
            set(grid[t]["best_case_per_flow"]) == {2} for t in art["test_set"]["read_points"]
        )
    for model in ("W:physical_unpressurised_95", "W:physical_pressurised"):
        points = art["models"][model]["points"]
        assert all(sum(p["negative_stages"].values()) == 0 for p in points.values())
        grid = art["models"][model]["grid"]
        for tag in art["test_set"]["read_points"]:
            flags = grid[tag]["case_1_best_or_practically_equal_per_flow"]
            assert len(flags) == 10
            assert all(flags[:6])
            assert sum(flags) == (7 if tag == "M|day_1_s2" else 6)

    # as delivered
    assert round(units["as_delivered"]["S"]["saturation"][1], 3) == 2.015
    assert units["as_delivered"]["S"]["order_of_the_formula"] == [2, 4, 1, 3, 6, 5]
    assert not units["as_delivered"]["S"]["dryers_admissible"]
    assert physical["as_delivered"]["S"]["saturation"][1] == 1.0
    delivery = art["delivery"]
    p0 = delivery["P0_units"]
    assert round(p0["saturation"][1], 3) == 2.309
    assert p0["order_of_the_formula"] == [2, 4, 1, 3, 6, 5]
    assert [round(p0["saving_percent"][k], 2) for k in "1234"] == [15.29, 17.52, 15.24, 17.5]
    assert delivery["plant_sheet_dyssol_columns_within_rounding"] == {
        "fuel": [5, 8],
        "total": [3, 4],
        "recovered": [0, 8],
    }
    stored = delivery["stored_results_against_rebuild"]
    exact = [k for k, v in stored.items() if v["max_temperature_k"] == 0.0]
    assert exact == ["case 2", "case 4", "case 6"]
    others = [v["max_temperature_k"] for k, v in stored.items() if k not in exact]
    assert (round(min(others), 3), round(max(others), 3)) == (0.010, 0.030)
    assert round(max(v["max_relative_percent"] for v in stored.values()), 2) == 1.75
    assert delivery["sheet_numbering_reversed_between_the_deliveries"] is True
    assert delivery["stated_and_stored_circuit_flows_equal"] is False

    # the generalised rule at the points of Befund 121
    rule_121 = art["generalised_rule_at_befund_121"]
    assert rule_121["recomputed"] is False
    assert rule_121["sets"]["full"]["admissible_points"] == 16803
    assert rule_121["sets"]["rule_test"]["admissible_points"] == 4185
    expected = {
        ("full", "physical_unpressurised_95"): (14841, 0.9725, 0.9735, 13, 28),
        ("rule_test", "physical_unpressurised_95"): (3716, 0.9701, 0.9701, 5, 5),
        ("full", "physical_pressurised"): (16756, 0.9736, 0.9743, 21, 32),
        ("rule_test", "physical_pressurised"): (4178, 0.9706, 0.9703, 9, 8),
        ("full", "partner"): (16803, 0.8398, 0.9311, 54, 1589),
        ("rule_test", "partner"): (4185, 0.8315, 0.9259, 18, 413),
    }
    for (label, mode), (n, generalised, worded, g_right, w_right) in expected.items():
        rates = rule_121["sets"][label]["modes"][mode]
        assert rates["generalised"]["decided"]["n"] == n
        assert round(rates["generalised"]["decided"]["rate"], 4) == generalised
        assert round(rates["worded"]["decided"]["rate"], 4) == worded
        assert rates["rules_differ"]["generalised_right"] == g_right
        assert rates["rules_differ"]["worded_right"] == w_right
        assert rates["sink_1_larger_share"] == 1.0
        if mode != "partner":
            assert 0.992 <= rates["generalised"]["clear"]["rate"] <= 0.9945
            assert 0.992 <= rates["worded"]["clear"]["rate"] <= 0.9945
    partner = rule_121["sets"]["full"]["modes"]["partner"]
    assert round(partner["generalised"]["clear"]["rate"], 3) == 0.982
    assert round(partner["worded"]["clear"]["rate"], 4) == 0.9997
    tree = rule_121["stored_tree_test_accuracy_strictly_best"]
    assert (round(min(tree.values()), 3), round(max(tree.values()), 3)) == (0.966, 0.978)
