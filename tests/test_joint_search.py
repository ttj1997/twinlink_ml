"""M3' joint stage: response tables of the lines and the nested search over allocations."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import joint_search as js
from twinlink_ml.evaluation import site_network as sn
from twinlink_ml.evaluation.line_tables import LineTable, build_line_table
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.operating_point import check_constraints
from twinlink_ml.simulation.oracle_search import evaluate_point

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
K = 271.79706373319294
THRESHOLD = 0.15 - 0.005961819615175404
T_GRID = np.array([80.0, 100.0, 120.0])
U_GRID = np.array([1.26, 1.9, 2.5])
T_AMB, PHI = 15.0, 0.55


@pytest.fixture(scope="module")
def configs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(CONFIG_PATH)
    return {"A": base, "B": sn.with_spray_rate(base, 2.0)}


@pytest.fixture(scope="module")
def tables(configs):
    return {
        name: build_line_table(
            name,
            configs[name],
            t_amb=T_AMB,
            phi_amb=PHI,
            t_in_grid=T_GRID,
            u_grid=U_GRID,
            p_nozzle=1.0,
        )
        for name in ("A", "B")
    }


def test_table_nodes_are_the_true_solutions(configs, tables) -> None:
    table = tables["A"]
    for t_in in T_GRID:
        for u_gas in U_GRID:
            solution = evaluate_point(
                float(t_in),
                float(u_gas),
                1.0,
                T_AMB,
                PHI,
                max_iterations=30,
                tolerance=1e-7,
                base_config=configs["A"],
            )
            state, violation = table.state(float(t_in), float(u_gas), THRESHOLD)
            assert state.outlet_humidity_kg_per_kg == solution.outlet_humidity_kg_per_kg
            assert state.dry_air_mass_flow_kg_per_s == solution.dry_air_mass_flow_kg_per_s
            check = check_constraints(
                solution, critical_moisture_kg_per_kg=THRESHOLD, gas_velocity_m_per_s=float(u_gas)
            )
            if solution.feasible:
                assert violation == pytest.approx(check.normalized_violation, abs=1e-12)


def test_bilinear_midpoint_is_the_mean_of_the_corners(tables) -> None:
    table = tables["B"]
    mid = table.interpolate(90.0, 1.58)
    corners = table.values[0:2, 0:2].reshape(4, -1).mean(axis=0)
    assert np.allclose(mid, corners)


def test_a_failed_node_makes_its_cells_unusable(tables) -> None:
    table = tables["A"]
    values = table.values.copy()
    values[2, 2] = np.nan
    broken = LineTable(
        **{**{f: getattr(table, f) for f in table.__dataclass_fields__}, "values": values}
    )
    assert broken.state(100.0, 1.9, THRESHOLD) is None
    assert broken.state(110.0, 2.2, THRESHOLD) is None
    assert broken.state(80.0, 1.26, THRESHOLD) is not None
    restored = LineTable.from_dict(broken.to_dict())
    assert np.array_equal(restored.values, broken.values, equal_nan=True)


def test_equivalent_order_variants() -> None:
    a = (("Q1", "S1"), ("Q1", "SA"), ("Q1", "SB"))
    b = (("Q1", "S1"), ("Q1", "SB"), ("Q1", "SA"))
    assert js.equivalent_order_variant(a, b) and not js.equivalent_order_variant(a, a)
    assert not js.equivalent_order_variant(a, a[:2])


def _spec(cap):
    return SiteOracleSpec(THRESHOLD, js.site_spec_with_cap(sn.SiteSpec(K), cap))


def test_site_spec_with_cap_keeps_everything_else() -> None:
    base = sn.SiteSpec(K)
    capped = js.site_spec_with_cap(base, 12.5)
    assert capped.shared_cap_kw == 12.5 and capped.scale_factor == K
    assert js.site_spec_with_cap(capped, None) == base


def test_s2_is_the_best_allocation_at_fixed_set_points(configs) -> None:
    spec = _spec(40.0)
    allocations = sn.all_allocations(2, order_variants=True)
    x = [80.0, 1.26, 90.0, 1.26]
    best = js.s2_evaluation(x, configs, T_AMB, PHI, spec, allocations)
    for allocation in allocations[::37]:
        single = js.evaluate_true(x, allocation, configs, T_AMB, PHI, spec)
        assert best["co2_kg_per_h"] <= single["co2_kg_per_h"] + 1e-12


def test_the_nested_search_runs_end_to_end(configs, tables) -> None:
    allocations = [
        sn.fixed_allocation(2),
        (("Q1", "S1"), ("Q1", "SA"), ("Q1", "SB"), ("LA", "S1"), ("LB", "S1")),
        (("Q1", "S1"), ("Q1", "SB"), ("Q1", "SA"), ("LA", "S1"), ("LB", "S1")),
        (("Q1", "S1"), ("LA", "SB"), ("LB", "SA")),
    ]
    bounds = [(60.0, 120.0), (1.26, 2.5), (60.0, 120.0), (1.26, 2.5)]
    spec = _spec(None)
    record = js.nested_search(
        allocations=allocations,
        table_a=tables["A"],
        table_b=tables["B"],
        configs=configs,
        t_amb=T_AMB,
        phi_amb=PHI,
        spec=spec,
        bounds=bounds,
        seed=0,
        s2_allocation=allocations[0],
        search_maxiter=2,
        polish_maxiter=2,
    )
    assert len(record["table_values"]) == len(allocations)
    # no limit: the two order variants are the same objective, the second is not searched again
    assert record["table_values"][1] == record["table_values"][2]
    assert record["n_skipped_duplicates"] >= 1 or record["n_polished"] >= 1
    valid = [
        e["polished_value"] for e in record["polished"].values() if e["polished_value"] is not None
    ]
    assert record["m3_value"] == pytest.approx(min(valid), rel=1e-12)
    assert math.isfinite(record["table_error_max_kg_per_h"])
    assert record["s2_allocation_index"] == 0 and str(0) in record["polished"]
    assert record["window_final_kg_per_h"] >= record["window_initial_kg_per_h"]
