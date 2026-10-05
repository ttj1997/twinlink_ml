"""M2: site CO2 of lines coupled by the heat network (mechanics, no optimisation run)."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest

from twinlink_ml.evaluation import site_network as sn
from twinlink_ml.evaluation import site_oracle as so
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.operating_point import check_constraints
from twinlink_ml.simulation.oracle_search import evaluate_point

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"


def _load_script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(SCRIPTS / "verify"))
    spec = importlib.util.spec_from_file_location(f"{name}_for_test", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


def _state(config, name: str, t_in: float, u_gas: float, t_amb: float, phi: float):
    solution = evaluate_point(
        t_in, u_gas, 1.0, t_amb, phi, max_iterations=30, tolerance=1e-7, base_config=config
    )
    return solution, sn.line_state(name, solution, config, t_amb=t_amb, phi_amb=phi, t_in=t_in)


def test_partner_constants_equal_the_pinch_benchmark() -> None:
    partner = _load_script("run_pinch_benchmark")
    assert sn.EXTERNAL_SOURCE_Q1 == partner.SOURCE_1
    assert sn.EXTERNAL_SINK_S1 == partner.SINK_1
    assert partner.SOURCE_2.dry_air_mass_flow_kg_per_s == sn.PARTNER_LINE_EXHAUST_FLOW_KG_PER_S
    assert sn.SiteSpec(1.0).heater_efficiency == partner.HEATER_EFFICIENCY
    assert (
        sn.SiteSpec(1.0).fuel_lower_heating_value_j_per_kg
        == partner.FUEL_LOWER_HEATING_VALUE_J_PER_KG
    )


def test_allocation_sets() -> None:
    one, two = sn.all_allocations(1), sn.all_allocations(2)
    assert (len(one), len(two)) == (9, 343)
    assert sn.fixed_allocation(1) in one and sn.all_to_all_allocation(1) in one
    assert sn.fixed_allocation(2) in two and sn.all_to_all_allocation(2) in two
    assert sn.fixed_allocation(1) == (("Q1", "S1"), ("LA", "SA"))


def test_scale_factor_matches_the_m0_artifact(config) -> None:
    import json

    solution, _ = _state(
        config,
        "A",
        config.actions.inlet_temperature_celsius.nominal,
        config.actions.gas_velocity_m_per_s.nominal,
        config.disturbances.ambient_temperature_celsius.nominal,
        config.disturbances.ambient_relative_humidity.nominal,
    )
    k = sn.scale_factor(solution.dry_air_mass_flow_kg_per_s)
    stored = json.loads(
        (REPO_ROOT / "artifacts" / "runs" / "m0_line_b_network.json").read_text("utf-8")
    )
    assert k == pytest.approx(stored["scale_factor_k"], rel=1e-9)


def test_two_line_site_reproduces_the_m0_script(config) -> None:
    """The generalisation gives the numbers of ``analyze_m0_line_b_network._evaluate_scenario``."""
    m0 = _load_script("analyze_m0_line_b_network")
    sol_a, a = _state(config, "A", 88.0, 0.9, 12.0, 0.6)
    sol_b, b = _state(config, "B", 95.0, 1.1, 12.0, 0.6)
    k = 271.79706373319294

    def as_row(line: sn.LineState) -> dict[str, float]:
        return {
            "t_amb": line.ambient_temperature_celsius,
            "phi_amb": 0.6,
            "t_in": line.inlet_temperature_celsius,
            "t_out": line.outlet_temperature_celsius,
            "y_out": line.outlet_humidity_kg_per_kg,
            "m_dry": line.dry_air_mass_flow_kg_per_s,
            "y_in": line.inlet_humidity_kg_per_kg,
            "electric_kw": line.electric_kw,
        }

    expected = m0._evaluate_scenario(as_row(a), as_row(b), k, m0._allocations())
    results, rejected = sn.evaluate_site((a, b), sn.SiteSpec(k), sn.all_allocations(2))
    best = min(results, key=lambda r: r.co2_kg_per_h)
    assert rejected == expected["n_rejected"]
    assert len(results) == expected["n_allocations_evaluated"]
    assert best.co2_kg_per_h == pytest.approx(expected["co2_best_kg_per_h"], rel=1e-12)
    assert [list(c) for c in best.connections] == expected["best_connections"]


def test_site_result_formulas(config) -> None:
    _, line = _state(config, "A", 90.0, 1.2, 15.0, 0.5)
    spec = sn.SiteSpec(271.79706373319294)
    results, rejected = sn.evaluate_site((line,), spec, sn.all_allocations(1))
    assert rejected == 0 and len(results) == 9
    for r in results:
        assert r.co2_kg_per_h == pytest.approx(0.366 * r.electric_kw + 0.201 * r.gas_kw)
        assert r.kwh_sum_kw == pytest.approx(r.electric_kw + r.gas_kw)
        assert r.gas_kw_lines <= r.gas_kw + 1e-9
        assert r.recovered_kw_lines <= r.recovered_kw_total + 1e-9
    # a hotter inlet needs more fuel at the same routing
    _, hotter = _state(config, "A", 110.0, 1.2, 15.0, 0.5)
    hot_results, _ = sn.evaluate_site((hotter,), spec, [sn.fixed_allocation(1)])
    cold_results, _ = sn.evaluate_site((line,), spec, [sn.fixed_allocation(1)])
    assert hot_results[0].gas_kw_lines > cold_results[0].gas_kw_lines


def test_search_objective_is_best_site_co2_plus_penalty(config) -> None:
    spec = so.SiteOracleSpec(0.15 - 0.005961819615175404, sn.SiteSpec(271.79706373319294))
    so.init_worker(config)
    x = (88.0, 0.9, 1.0)
    solution, line = _state(config, "A", *x[:2], 12.0, 0.6)
    results, _ = sn.evaluate_site((line,), spec.site, sn.all_allocations(1))
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=spec.quality_threshold_kg_per_kg,
        gas_velocity_m_per_s=0.9,
    )
    expected = (
        min(r.co2_kg_per_h for r in results) + so._PENALTY_WEIGHT * check.normalized_violation
    )
    assert so.penalized_objective(x, 12.0, 0.6, spec) == pytest.approx(expected, rel=1e-12)


def test_site_point_reports_the_three_routings(config) -> None:
    spec = so.SiteOracleSpec(0.15, sn.SiteSpec(271.79706373319294))
    out = so.evaluate_site_point(
        config, spec, t_in=90.0, u_gas=1.0, p_nozzle=1.0, t_amb=15.0, phi_amb=0.5
    )
    assert not out["solve_failed"]
    best = out["routing_best"]["co2_kg_per_h"]
    assert best <= out["routing_fixed"]["co2_kg_per_h"] + 1e-12
    assert best <= out["routing_all_to_all"]["co2_kg_per_h"] + 1e-12
    assert out["n_allocations_evaluated"] + out["n_allocations_rejected"] == 9
