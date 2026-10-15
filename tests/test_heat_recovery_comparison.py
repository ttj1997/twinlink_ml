"""Tests of ``scripts/report_heat_recovery_comparison.py``.

Partner model v4.1 against an independent calculation of the heat recovery (docs/11, section 10).

The rebuild of the intermediate circuit is checked against the code that already exists:
``evaluate_cascade`` (direct exchange, minimum approach along the exchanger, both recovery modes)
and ``partner_v4_1.loops`` (equal outlet, counter-current bound). The inputs are typed in by
hand (the partner workbook is not needed).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import ExcelDryerInputs, ExcelInputs, ExcelMaterialInputs
from twinlink_ml.partner_v4_1 import cascade as k
from twinlink_ml.partner_v4_1 import loops
from twinlink_ml.partner_v4_1.inputs import ExcelInputsV41

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_heat_recovery_comparison.py"
_spec = importlib.util.spec_from_file_location("report_heat_recovery_comparison", SCRIPT)
assert _spec is not None and _spec.loader is not None
hr = importlib.util.module_from_spec(_spec)
sys.modules["report_heat_recovery_comparison"] = hr
_spec.loader.exec_module(hr)


def v41_inputs() -> ExcelInputsV41:
    plant = ExcelInputs(
        dryer_1=ExcelDryerInputs(100.0, 6.0, 150.0, 6.0, 0.4, 15.0),
        dryer_2=ExcelDryerInputs(100.0, 6.0, 300.0, 12.0, 0.4, 15.0),
        material=ExcelMaterialInputs(1.2, 45000.0, 80.0, 4.2, 1.0, 2.0, 2.0, 2500.0, 4.2),
        ambient_temperature_celsius=15.0,
        ambient_pressure_atm=1.0,
    )
    return ExcelInputsV41(plant, (20.0, 20.0))


@pytest.fixture(scope="module")
def plant() -> hr.Plant:
    return hr.plant_from_inputs(v41_inputs().to_si())


def test_partner_model_as_delivered_reproduces_the_document_model(plant: hr.Plant) -> None:
    """Model ``a`` is the document model of docs/11, 7.4 (1.4140, 1.2363, 1.2720, 1.2534 kg/h)."""
    si = v41_inputs().to_si()
    model = hr.Model("a", plant.partner, "equal_outlet")
    for case in range(7):
        mine = hr.run_scenario(plant, model, case)
        reference = k.evaluate_case(si, case)
        assert mine.gas_kg_per_h == pytest.approx(reference.total_gas_kg_per_h, rel=1e-12)
        assert mine.recovered_kw * 1e3 == pytest.approx(
            reference.total_recovered_w, rel=1e-12, abs=1e-9
        )
        for a, b in zip(mine.stages, reference.stages, strict=True):
            assert a.gas_out_celsius == pytest.approx(b.sink_out_celsius, rel=1e-12)
            assert a.exhaust_out_celsius == pytest.approx(b.exhaust_out_celsius, rel=1e-12)
    assert hr.run_scenario(plant, model, 0).gas_kg_per_h == pytest.approx(1.4140, abs=5e-5)
    assert hr.run_scenario(plant, model, 1).gas_kg_per_h == pytest.approx(1.2363, abs=5e-5)


@pytest.mark.parametrize("dt", [20.0, 10.0, 5.0, 1e-4])
@pytest.mark.parametrize("mode", ["sensible", "condensing"])
def test_direct_exchange_equals_the_module(plant: hr.Plant, dt: float, mode: str) -> None:
    """Direct exchange with the approach along the exchanger = ``evaluate_cascade``."""
    for case in range(7):
        reference = hr.module_direct(plant, case, dt, mode)  # type: ignore[arg-type]
        model = hr.Model("d", hr.REPO, "direct", mode, dt)  # type: ignore[arg-type]
        mine = hr.run_scenario(plant, model, case)
        assert mine.gas_kg_per_h == pytest.approx(reference["gas_kg_per_h"], abs=1e-9)
        for stage, q_kw in zip(mine.stages, reference["stage_q_kw"], strict=True):
            assert stage.q_w / 1e3 == pytest.approx(q_kw, abs=1e-9)


def test_circuit_without_approach_equals_the_counter_current_bound(plant: hr.Plant) -> None:
    """Circuit, counter-current, ``dT_min`` 0, no dew-point limit = ``loops`` (own closure)."""
    si = v41_inputs().to_si()
    material = si.plant.material
    for source in (1, 2):
        for t_in in (15.0, 40.0, 57.0, 80.0):
            for flow in (5.0, 20.0, 60.0, 150.0, 600.0):
                exhaust = plant.exhausts[source]
                gas = c.Stream(plant.sink_air[1].dry_air_kg_per_s, 0.006, t_in)
                reference = loops.counter_current_infinite_area_loop(
                    exhaust, gas, flow / 3600.0, si.cp_htm_j_per_kg_k, material
                )
                profile = hr.exhaust_profile(exhaust, plant.partner, "linear")
                q, _ = hr.solve_stage(
                    profile,
                    hr.capacity(gas, plant.partner),
                    t_in,
                    1e9,
                    flow / 3600.0 * si.cp_htm_j_per_kg_k,
                    0.0,
                    0.0,
                )
                assert q == pytest.approx(reference.recovered_w, abs=1e-6)


def test_equal_outlet_stage_equals_loops(plant: hr.Plant) -> None:
    si = v41_inputs().to_si()
    exhaust = plant.exhausts[2]
    gas = plant.sink_air[1]
    reference = loops.equal_outlet_loop(
        exhaust, gas, 20.0 / 3600.0, si.cp_htm_j_per_kg_k, si.plant.material
    )
    model = hr.Model("a", plant.partner, "equal_outlet")
    stage = hr.run_stage(plant, model, 2, 1, gas, 20.0 / 3600.0)
    assert stage.q_w == pytest.approx(reference.recovered_w, rel=1e-12)
    assert stage.gas_out_celsius == pytest.approx(reference.sink_out_celsius, rel=1e-12)
    assert stage.exhaust_out_celsius == pytest.approx(reference.exhaust_out_celsius, rel=1e-12)


@pytest.mark.parametrize("profile", ["sensible", "condensing"])
@pytest.mark.parametrize("dt", [20.0, 10.0, 5.0])
@pytest.mark.parametrize("flow", [20.0, 60.0, 150.0])
def test_circuit_stage_holds_the_approach_everywhere(
    plant: hr.Plant, profile: str, dt: float, flow: float
) -> None:
    """The reported minimum approach of both exchangers is >= dT_min/2, on a fine grid too."""
    model = hr.Model("c", hr.REPO, "counter_current", profile, dt)  # type: ignore[arg-type]
    gas = plant.sink_air[1]
    for source in (1, 2):
        stage = hr.run_stage(plant, model, source, 1, gas, flow / 3600.0)
        if stage.q_w == 0.0:
            continue
        assert stage.min_approach_recovery_k >= dt / 2.0 - 1e-6
        assert stage.min_approach_preheater_k is not None
        assert stage.min_approach_preheater_k >= dt / 2.0 - 1e-6
        prof = hr.exhaust_profile(plant.exhausts[source], hr.REPO, model.profile, nodes=20000)
        b = flow / 3600.0 * plant.htm_cp_j_per_kg_k
        t1, q = stage.htm_cold_celsius, stage.q_w
        assert t1 is not None
        grid = [q * n / 4000.0 for n in range(4001)]
        approach = min(prof.temperature_at(x) - (t1 + (q - x) / b) for x in grid)
        assert approach >= dt / 2.0 - 1e-6


def test_stage_is_the_largest_one(plant: hr.Plant) -> None:
    """A little more duty violates the approach (the bisection found the limit)."""
    model = hr.Model("c", hr.REPO, "counter_current", "condensing", 10.0)
    gas = plant.sink_air[1]
    stage = hr.run_stage(plant, model, 2, 1, gas, 20.0 / 3600.0)
    assert stage.binding == "approach"
    profile = hr.exhaust_profile(plant.exhausts[2], hr.REPO, "condensing")
    b = 20.0 / 3600.0 * plant.htm_cp_j_per_kg_k
    cg = hr.capacity(gas, hr.REPO)
    for bigger in (stage.q_w * 1.0001, stage.q_w + 0.01):
        t1 = 15.0 + 5.0 + max(0.0, bigger * (1.0 / cg - 1.0 / b))
        worst = min(
            profile.temperature_at(x) - (t1 + (bigger - x) / b)
            for x in [bigger * n / 4000.0 for n in range(4001)]
        )
        assert worst < 5.0 - 1e-9


def test_more_approach_never_recovers_more(plant: hr.Plant) -> None:
    for case in range(1, 7):
        values = [
            hr.run_scenario(
                plant, hr.Model("c", hr.REPO, "counter_current", "condensing", dt), case
            ).recovered_kw
            for dt in (0.0, 5.0, 10.0, 20.0, 40.0)
        ]
        assert values == sorted(values, reverse=True)


def test_condensation_only_adds(plant: hr.Plant) -> None:
    for dt in (20.0, 10.0, 5.0):
        for case in range(1, 7):
            b = hr.run_scenario(
                plant, hr.Model("b", hr.REPO, "counter_current", "sensible", dt), case
            )
            cc = hr.run_scenario(
                plant, hr.Model("c", hr.REPO, "counter_current", "condensing", dt), case
            )
            assert cc.recovered_kw >= b.recovered_kw - 1e-9


def test_no_stage_cools_the_gas_with_the_approach_model(plant: hr.Plant) -> None:
    for case in range(1, 7):
        model = hr.Model("c", hr.REPO, "counter_current", "condensing", 10.0)
        assert all(not s.cools_gas for s in hr.run_scenario(plant, model, case).stages)


def test_energy_closure_of_the_scenario(plant: hr.Plant) -> None:
    """Recovered heat + heater duty = demand of the sinks (any model, any case)."""
    for model in (
        hr.Model("a", plant.partner, "equal_outlet"),
        hr.Model("c", hr.REPO, "counter_current", "condensing", 5.0),
    ):
        for case in range(7):
            r = hr.run_scenario(plant, model, case)
            assert r.recovered_kw + sum(r.heater_kw.values()) == pytest.approx(
                r.demand_kw, rel=1e-12
            )


def test_pinch_bound_is_not_exceeded_by_any_wiring(plant: hr.Plant) -> None:
    """The bound at the same approach holds for the circuit and for direct exchange."""
    for dt in (20.0, 10.0, 5.0):
        for case in range(1, 7):
            bound = hr.pinch_bound(plant, case, hr.REPO, dt, "condensing")["recovered_kw"]
            for kind in ("counter_current", "direct"):
                model = hr.Model("x", hr.REPO, kind, "condensing", dt)  # type: ignore[arg-type]
                assert hr.run_scenario(plant, model, case).recovered_kw <= bound + 1e-6


def test_pinch_bound_falls_with_the_approach(plant: hr.Plant) -> None:
    values = [
        hr.pinch_bound(plant, 1, hr.REPO, dt, "condensing")["recovered_kw"]
        for dt in (0.0, 5.0, 10.0, 20.0)
    ]
    assert values == sorted(values, reverse=True)
    assert math.isfinite(values[-1])


def test_consistency_check_finds_the_known_negative_recovery_of_the_intermediate_version() -> None:
    """Positive control: docs/11, 8.1 -- scenario 6, second stage, -0.0563 kW."""
    plant = hr.plant_from_inputs(hr.intermediate_inputs().to_si())
    report = hr.consistency_check(plant, plant.partner, "intermediate")
    found = [
        v
        for v in report["violations"]
        if v["case"] == 6 and v["check"].startswith("stage cools") and v["recovered_kw"] < 0.0
    ]
    assert len(found) == 1
    assert found[0]["recovered_kw"] == pytest.approx(-0.0563, abs=5e-5)


def test_consistency_check_v41_has_no_energy_or_second_law_violation(plant: hr.Plant) -> None:
    report = hr.consistency_check(plant, plant.partner, "v4.1")
    kinds = {v["check"] for v in report["violations"]}
    assert not any(
        kind.startswith(("energy balance", "second law", "total recovery")) for kind in kinds
    )


def test_profile_of_a_source_below_its_floor_is_empty() -> None:
    cold = c.Stream(0.0278, 0.04, 30.0)
    assert hr.exhaust_profile(cold, hr.REPO, "sensible").total_w == 0.0


# --- consistent property set (Befund 90, addendum) ---------------------------------------------


def test_consistent_properties_recompute_the_exhausts_for_the_repository_models_only() -> None:
    """b, c, d get dryer exhausts with the repository's gas properties; a keeps the partner's."""
    si = v41_inputs().to_si()
    mixed = hr.plant_from_inputs(si)
    consistent = hr.plant_from_inputs(si, consistent_properties=True)
    assert mixed.exhausts_for(hr.REPO) is mixed.exhausts
    assert consistent.exhausts == mixed.exhausts
    assert consistent.exhausts_for(consistent.partner) is consistent.exhausts
    repo = consistent.exhausts_for(hr.REPO)
    assert repo[1].temperature_celsius < mixed.exhausts[1].temperature_celsius - 10.0
    assert repo[2].temperature_celsius < mixed.exhausts[2].temperature_celsius - 20.0
    a = hr.Model("a", consistent.partner, "equal_outlet")
    assert hr.run_scenario(consistent, a, 1).gas_kg_per_h == pytest.approx(
        hr.run_scenario(mixed, a, 1).gas_kg_per_h, rel=1e-14
    )
    c_model = hr.Model("c", hr.REPO, "counter_current", "condensing", 10.0)
    assert (
        hr.run_scenario(consistent, c_model, 1).recovered_kw
        < hr.run_scenario(mixed, c_model, 1).recovered_kw
    )


def test_consistent_properties_keep_the_laboratory_exhaust_of_dryer_2_for_both_sets() -> None:
    si = v41_inputs().to_si()
    plant = hr.plant_from_inputs(si, {2: (103.485, 0.0752)}, consistent_properties=True)
    for exhausts in (plant.exhausts, plant.exhausts_for(hr.REPO)):
        assert exhausts[2].temperature_celsius == 103.485
        assert exhausts[2].humidity_kg_per_kg == 0.0752
