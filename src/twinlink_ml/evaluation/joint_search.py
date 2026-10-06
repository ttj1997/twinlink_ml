r"""Nested search of the joint optimum of two lines over all allocations (stage M3', scheme C).

THE RULES BELOW WERE FIXED BEFORE ANY RUN OF THE JOINT STAGE (project lead, 2026-10-06).

Problem: four set points (inlet temperature and gas velocity of the lines A and B; the nozzle
pressure
of both at its lower bound 1.0 bar, D6) against the site CO2 of one ALLOCATION (every source
feeds at
least one sink; 441 sets including the order of the two stages Q1 -> SA, Q1 -> SB) with the duty
limit
of the shared exchanger, plus the constraint penalty. Discrete and continuous variables are NOT
mixed
in one DE run: for EVERY allocation the four set points are optimised, then the best allocation is
chosen, so that a missing coupling gain cannot be a search error of a mixed search.

Scheme C (the direct search on true solves for 441 allocations would take about 780 h):

1. every allocation: DE + Nelder-Mead (budget of Befund 52 / M1 / M2: popsize 10, maxiter 50,
   tolerance 1e-4, Nelder-Mead refinement) on the response tables of the two lines
   (:mod:`twinlink_ml.evaluation.line_tables`, bilinear), giving a table optimum ``x_i`` and a table
   value ``V_i``; the penalty weight and the failure value are the laboratory ones times the scale
   factor ``k`` (Befund 70 c), the penalty of a cell is the sum of both lines;
2. every allocation with ``V_i <= V_min + W`` (``W`` = 5 kg CO2e/h, the polishing window) is
   polished with
   TRUE solves: the table error ``E_i = |V_i - f_true(x_i)|`` is measured at the table optimum,
   then a
   Nelder-Mead search on true solves starts there; of two allocations that differ only in the order
   of the two Q1 stages and have the same table value (within 1e-6) only one is polished; the
   allocation chosen by S2 is always polished (decomposition), its error is not part of ``E_max``;
3. error rule: if ``E_max`` (largest table error over the polished allocations of the window)
   exceeds
   ``W / 2``, the window is widened to ``W = 4 * E_max`` and the search repeats the polishing for
   the
   newly included allocations, until ``E_max <= W / 2`` (or all allocations are polished); window
   and
   ``E_max`` are reported per scenario and case;
4. the joint optimum M3' is the polished allocation with the lowest true value. Value = site CO2
   of the
   best allocation plus the penalty; the CO2 itself and the feasibility of both lines (true check
   against the tightened limit) are reported.

S2 ("each line on its own, then the best allocation"): the set points of the two lines are the
optima of
each line ignoring the network (A: arm S of M2, B: the optimum of line B on its own, same
method), at the
nozzle pressure 1.0 bar; S2 is the best of the 441 allocations at these fixed set points, with
the same
limit. Coupling gain ``G = value(S2) - value(M3')`` (kg CO2e/h); split by the allocation: ``G_set
= value(S2) -
value(best set points for the allocation of S2)`` (the gain of the set points alone) and
``G_alloc = G - G_set``
(the further gain from another allocation).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from twinlink_ml.simulation.config import ProcessConfig
from twinlink_ml.simulation.operating_point import check_constraints
from twinlink_ml.simulation.oracle_search import OptimizationBudget, evaluate_point, run_de_nm

from .line_tables import LineTable
from .site_network import Allocation, SiteSpec, evaluate_site, line_state
from .site_oracle import SiteOracleSpec, penalty_weight, solve_failure_value

__all__ = [
    "POLISH_WINDOW_KG_PER_H",
    "WINDOW_WIDENING_FACTOR",
    "SearchRecord",
    "equivalent_order_variant",
    "evaluate_true",
    "nested_search",
    "s2_evaluation",
    "table_objective",
]

POLISH_WINDOW_KG_PER_H = 5.0
ERROR_HALF_WINDOW = 0.5
WINDOW_WIDENING_FACTOR = 4.0
DUPLICATE_TOLERANCE = 1.0e-6
BOUNDS_T_IN = (60.0, 120.0)

SearchRecord = dict[str, Any]


def table_objective(
    x: Any, allocation: Allocation, table_a: LineTable, table_b: LineTable, spec: SiteOracleSpec
) -> float:
    """Site CO2 of one allocation plus the penalty of both lines, from the tables."""
    threshold = spec.quality_threshold_kg_per_kg
    a = table_a.state(float(x[0]), float(x[1]), threshold)
    b = table_b.state(float(x[2]), float(x[3]), threshold)
    if a is None or b is None:
        return solve_failure_value(spec)
    results, _ = evaluate_site((a[0], b[0]), spec.site, [allocation])
    if not results:
        return solve_failure_value(spec)
    return float(results[0].co2_kg_per_h + penalty_weight(spec) * (a[1] + b[1]))


def evaluate_true(
    x: Any,
    allocation: Allocation | list[Allocation],
    configs: dict[str, ProcessConfig],
    t_amb: float,
    phi_amb: float,
    spec: SiteOracleSpec,
    p_nozzle: tuple[float, float] = (1.0, 1.0),
) -> dict[str, Any]:
    """True solves of both lines at the set points ``x`` and the site result of the allocation(s).

    ``allocation`` may be a list; the best of them (lowest CO2) is returned.
    """
    out: dict[str, Any] = {"solve_failed": False}
    states = []
    violations = []
    checks = []
    for name, (t_in, u_gas), p in (
        ("A", (float(x[0]), float(x[1])), p_nozzle[0]),
        ("B", (float(x[2]), float(x[3])), p_nozzle[1]),
    ):
        try:
            solution = evaluate_point(
                t_in,
                u_gas,
                p,
                t_amb,
                phi_amb,
                max_iterations=spec.search_max_iterations,
                tolerance=spec.search_tolerance,
                base_config=configs[name],
            )
        except (RuntimeError, ValueError):
            return {"solve_failed": True, "value": solve_failure_value(spec)}
        check = check_constraints(
            solution,
            critical_moisture_kg_per_kg=spec.quality_threshold_kg_per_kg,
            gas_velocity_m_per_s=u_gas,
        )
        checks.append(check)
        violations.append(check.normalized_violation)
        states.append(
            line_state(name, solution, configs[name], t_amb=t_amb, phi_amb=phi_amb, t_in=t_in)
        )
    allocations = allocation if isinstance(allocation, list) else [allocation]
    results, _ = evaluate_site((states[0], states[1]), spec.site, allocations)
    if not results:
        return {"solve_failed": True, "value": solve_failure_value(spec)}
    best = min(results, key=lambda r: r.co2_kg_per_h)
    out.update(
        {
            "value": float(best.co2_kg_per_h + penalty_weight(spec) * sum(violations)),
            "co2_kg_per_h": best.co2_kg_per_h,
            "co2_lines_kg_per_h": best.co2_lines_kg_per_h,
            "kwh_sum_kw": best.kwh_sum_kw,
            "allocation": [list(c) for c in best.connections],
            "feasible_a": checks[0].feasible,
            "feasible_b": checks[1].feasible,
            "moisture_a": checks[0].moisture_kg_per_kg,
            "moisture_b": checks[1].moisture_kg_per_kg,
            "recovered_kw_external_to_lines": best.recovered_kw_external_to_lines,
            "set_points": [float(v) for v in x],
        }
    )
    return out


def _true_value(
    x: Any,
    allocation: Allocation,
    configs: dict[str, ProcessConfig],
    t_amb: float,
    phi_amb: float,
    spec: SiteOracleSpec,
) -> float:
    return float(evaluate_true(x, allocation, configs, t_amb, phi_amb, spec)["value"])


def equivalent_order_variant(a: Allocation, b: Allocation) -> bool:
    """Same connections, only the order of the stages differs."""
    return a != b and sorted(a) == sorted(b)


def s2_evaluation(
    set_points: Any,
    configs: dict[str, ProcessConfig],
    t_amb: float,
    phi_amb: float,
    spec: SiteOracleSpec,
    allocations: list[Allocation],
) -> dict[str, Any]:
    """S2: the best of all allocations at the fixed set points of the two lines (own optima)."""
    return evaluate_true(set_points, allocations, configs, t_amb, phi_amb, spec)


def nested_search(
    *,
    allocations: list[Allocation],
    table_a: LineTable,
    table_b: LineTable,
    configs: dict[str, ProcessConfig],
    t_amb: float,
    phi_amb: float,
    spec: SiteOracleSpec,
    bounds: list[tuple[float, float]],
    seed: int,
    s2_allocation: Allocation | None,
    window_kg_per_h: float = POLISH_WINDOW_KG_PER_H,
    polish: bool = True,
    search_maxiter: int = 50,
    polish_maxiter: int = 200,
) -> SearchRecord:
    """Steps 1-4 of the module docstring for one scenario and one limit."""
    budget = OptimizationBudget(
        seed=seed,
        popsize=10,
        maxiter=search_maxiter,
        de_tol=1.0e-4,
        search_max_iterations=spec.search_max_iterations,
        search_tolerance=spec.search_tolerance,
        workers=1,
    )
    table_x: list[np.ndarray] = []
    table_v: list[float] = []
    first_of_key: dict[tuple[Any, ...], int] = {}
    for position, allocation in enumerate(allocations):
        key = tuple(sorted(allocation))
        if spec.site.shared_cap_kw is None and key in first_of_key:
            # without a limit the order of the stages cannot matter: the same objective
            table_x.append(table_x[first_of_key[key]])
            table_v.append(table_v[first_of_key[key]])
            continue
        first_of_key.setdefault(key, position)
        de, refine, _, _ = run_de_nm(
            table_objective, bounds, (allocation, table_a, table_b, spec), budget, de_workers=1
        )
        best = refine if refine.fun <= de.fun else de
        table_x.append(np.asarray(best.x, dtype=np.float64))
        table_v.append(float(best.fun))
    v_min = min(table_v)
    record: SearchRecord = {
        "seed": seed,
        "table_value_min": v_min,
        "table_values": table_v,
        "table_optima": [x.tolist() for x in table_x],
    }
    if not polish:
        return record

    order = np.argsort(table_v)
    polished: dict[int, dict[str, Any]] = {}
    skipped_duplicates: list[int] = []
    window = window_kg_per_h
    widened = False
    s2_index = allocations.index(s2_allocation) if s2_allocation in allocations else None

    def polish_one(i: int, forced: bool) -> None:
        x0 = table_x[i]
        value_at_table_optimum = _true_value(x0, allocations[i], configs, t_amb, phi_amb, spec)
        entry: dict[str, Any] = {
            "allocation": [list(c) for c in allocations[i]],
            "table_value": table_v[i],
            "table_optimum": x0.tolist(),
            "true_value_at_table_optimum": value_at_table_optimum,
            "forced": forced,
        }
        failed = value_at_table_optimum >= solve_failure_value(spec)
        entry["true_solve_failed_at_table_optimum"] = bool(failed)
        entry["table_error"] = None if failed else abs(table_v[i] - value_at_table_optimum)
        if failed:
            entry["polished_value"] = None
            polished[i] = entry
            return
        result = minimize(
            _true_value,
            x0,
            args=(allocations[i], configs, t_amb, phi_amb, spec),
            method="Nelder-Mead",
            bounds=bounds,
            options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": polish_maxiter, "adaptive": True},
        )
        entry["polished_value"] = float(result.fun)
        entry["polished_optimum"] = [float(v) for v in result.x]
        entry["polish_nit"] = int(result.nit)
        entry["polish_nfev"] = int(result.nfev)
        polished[i] = entry

    while True:
        for i in order:
            i = int(i)
            if table_v[i] > v_min + window or i in polished or i in skipped_duplicates:
                continue
            twin = next(
                (
                    j
                    for j in polished
                    if equivalent_order_variant(allocations[i], allocations[j])
                    and abs(table_v[i] - table_v[j]) <= DUPLICATE_TOLERANCE
                ),
                None,
            )
            if twin is not None:
                skipped_duplicates.append(i)
                continue
            polish_one(i, forced=False)
        if s2_index is not None and s2_index not in polished:
            polish_one(s2_index, forced=True)
        errors = [
            e["table_error"]
            for e in polished.values()
            if not e["forced"] and e["table_error"] is not None
        ]
        e_max = max(errors) if errors else 0.0
        all_done = all((i in polished or i in skipped_duplicates) for i in range(len(allocations)))
        if e_max <= ERROR_HALF_WINDOW * window or all_done:
            break
        window = WINDOW_WIDENING_FACTOR * e_max
        widened = True

    valid = [(i, e) for i, e in polished.items() if e.get("polished_value") is not None]
    best_i, best_e = min(valid, key=lambda item: item[1]["polished_value"])
    final = evaluate_true(
        best_e["polished_optimum"], allocations[best_i], configs, t_amb, phi_amb, spec
    )
    record.update(
        {
            "window_initial_kg_per_h": window_kg_per_h,
            "window_final_kg_per_h": window,
            "window_widened": widened,
            "table_error_max_kg_per_h": e_max,
            "table_error_rule_met": bool(e_max <= ERROR_HALF_WINDOW * window),
            "n_polished": len(polished),
            "n_skipped_duplicates": len(skipped_duplicates),
            "n_true_failed_at_table_optimum": sum(
                e["true_solve_failed_at_table_optimum"] for e in polished.values()
            ),
            "polished": {str(i): e for i, e in polished.items()},
            "chosen_index": best_i,
            "m3": final,
            "m3_value": final["value"],
            "s2_allocation_index": s2_index,
            "value_best_setpoints_for_s2_allocation": (
                polished[s2_index]["polished_value"] if s2_index in polished else None
            ),
        }
    )
    if math.isnan(record["m3_value"]):
        raise RuntimeError("joint optimum is NaN")
    return record


def site_spec_with_cap(site: SiteSpec, cap_kw: float | None) -> SiteSpec:
    """The site settings with the duty limit of the shared exchanger (``None``: no limit)."""
    return SiteSpec(
        scale_factor=site.scale_factor,
        t_kalt_celsius=site.t_kalt_celsius,
        dt_min_k=site.dt_min_k,
        heater_efficiency=site.heater_efficiency,
        fuel_lower_heating_value_j_per_kg=site.fuel_lower_heating_value_j_per_kg,
        external_source=site.external_source,
        external_sink=site.external_sink,
        shared_cap_kw=cap_kw,
    )
