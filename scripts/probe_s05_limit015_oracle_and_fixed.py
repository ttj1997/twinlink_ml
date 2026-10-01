r"""S0.5: reference values for the product-moisture quality limit 0.15 kg/kg.

``product_moisture_limit_kg_per_kg = 0.15``; the drying kinetics keep ``critical_moisture =
0.20`` (decoupled mechanics of Befund 64, ``probe_margin_energy_cost_decoupled``, imported
unchanged; ``study_operating_point_regret`` untouched). The tightened limit of the Lagrange
agent, 0.15 - m, then lies 0.056 kg/kg (9.4 m) below the solver validity boundary at 0.20
(S0.4: ``steady_state.py`` flags every solution above ``material.critical_moisture``).

Computed for this limit, on the 5 x 5 Befund 52 grid with the SAME optimiser budget as Befund 52
(differential evolution popsize 10, maxiter 50, tolerance 1e-4, seed 0, Nelder-Mead refinement):

1. the per-instance oracle at the 25 grid points (decoupled threshold 0.15),
2. the fixed operating point: one setpoint, feasible at all 25 points, minimal mean energy
   (the BB3(a) search of Befund 52 with the threshold as a parameter).

The artifact has the structure of ``bb1_bb3_operating_point_regret.json`` (``grid_results``,
``fixed_point.per_grid_point``), so ``twinlink_ml.rl.load_test_points(path)`` reads it as the
test set of the S0.6 gate against the limit 0.15.

Runtime estimate (before the run, 16 logical CPUs): the oracle took 16.7 min per boundary in
the FA3 run (33.4 min for two boundaries, Befund 65 FA3, 25 points in two waves of 16 workers).
The fixed-point search evaluates the 25 contexts per candidate in the pool: DE with population
30 and up to 51 generations (about 1500 candidates) at about 0.2-0.4 s per candidate (two waves
of 0.16 s solves) is 5-10 min, the refinement a few minutes more. Total estimate 25-40 min.

    python scripts/probe_s05_limit015_oracle_and_fixed.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import probe_margin_energy_cost as ka1  # noqa: E402
import probe_margin_energy_cost_decoupled as dec  # noqa: E402
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact  # noqa: E402

from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402

QUALITY_LIMIT_KG_PER_KG = 0.15
ARTIFACT_NAME = "s05_limit015_oracle_and_fixed"


def _fixed_worker(
    x: tuple[float, float, float],
    t_amb: float,
    phi_amb: float,
    max_iterations: int,
    tolerance: float,
    threshold: float,
) -> tuple[float, float] | None:
    """One grid point for one fixed candidate (pool task), decoupled threshold."""
    regret = dec._REGRET_MODULE
    assert regret is not None and regret._BASE_CONFIG is not None
    t_in, u_gas, p_nozzle = x
    try:
        solution = regret.evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=max_iterations,
            tolerance=tolerance,
            base_config=regret._BASE_CONFIG,
        )
    except (RuntimeError, ValueError):
        return None
    check = regret.check_constraints(
        solution, critical_moisture_kg_per_kg=threshold, gas_velocity_m_per_s=u_gas
    )
    objective = regret._objective_components(solution, regret._BASE_CONFIG)["objective_value"]
    return float(objective), check.normalized_violation


def _fixed_objective(
    x: np.ndarray,
    grid_points: list[tuple[float, float]],
    pool: Any,
    max_iterations: int,
    tolerance: float,
    threshold: float,
    sentinel_kw: float,
    penalty_weight_kw: float,
) -> float:
    tasks = [
        (tuple(float(v) for v in x), t_amb, phi, max_iterations, tolerance, threshold)
        for t_amb, phi in grid_points
    ]
    energies: list[float] = []
    violation_sum = 0.0
    for outcome in pool.starmap(_fixed_worker, tasks):
        if outcome is None:
            energies.append(sentinel_kw)
            violation_sum += 5.0
        else:
            energies.append(outcome[0])
            violation_sum += outcome[1]
    return float(np.mean(energies)) + penalty_weight_kw * violation_sum


def find_fixed_point_decoupled(
    regret: Any,
    base_config: Any,
    grid_points: list[tuple[float, float]],
    budget: Any,
    threshold: float,
) -> dict[str, Any]:
    bounds = regret.action_bounds(base_config)
    start = time.perf_counter()
    with Pool(budget.workers, initializer=dec._init_worker, initargs=(base_config,)) as pool:
        args = (
            grid_points,
            pool,
            budget.search_max_iterations,
            budget.search_tolerance,
            threshold,
            regret._SOLVE_FAILURE_PENALTY_KW,
            regret._PENALTY_WEIGHT_KW,
        )
        de_result, refine_result, de_elapsed, refine_elapsed = regret._optimize(
            _fixed_objective, bounds, args, budget, de_workers=1
        )
    best_x = refine_result.x if refine_result.fun <= de_result.fun else de_result.x
    t_in, u_gas, p_nozzle = (float(v) for v in best_x)

    per_point: list[dict[str, Any]] = []
    all_feasible = True
    regret._set_base_config(base_config)
    for t_amb, phi in grid_points:
        try:
            solution = regret.evaluate_point(
                t_in,
                u_gas,
                p_nozzle,
                t_amb,
                phi,
                max_iterations=budget.search_max_iterations,
                tolerance=budget.search_tolerance,
                base_config=base_config,
            )
        except (RuntimeError, ValueError) as exc:
            per_point.append(
                {
                    "ambient_temperature_celsius": t_amb,
                    "ambient_relative_humidity": phi,
                    "feasible": False,
                    "solve_failed": True,
                    "solve_failure_reason": str(exc),
                }
            )
            all_feasible = False
            continue
        check = regret.check_constraints(
            solution, critical_moisture_kg_per_kg=threshold, gas_velocity_m_per_s=u_gas
        )
        all_feasible = all_feasible and check.feasible
        components = regret._objective_components(solution, base_config)
        per_point.append(
            {
                "ambient_temperature_celsius": t_amb,
                "ambient_relative_humidity": phi,
                "feasible": check.feasible,
                "solve_failed": False,
                "energy_target": components["objective_value"],
                "particle_moisture_kg_per_kg": solution.particle_moisture_kg_per_kg,
                "infeasibility_reasons": list(solution.infeasibility_reasons),
                "constraint_check": check.as_dict(),
            }
        )
    return {
        "inlet_temperature_celsius": t_in,
        "gas_velocity_m_per_s": u_gas,
        "nozzle_pressure_bar": p_nozzle,
        "feasible_everywhere": all_feasible,
        "de_converged": bool(de_result.success),
        "de_message": str(de_result.message),
        "de_elapsed_s": de_elapsed,
        "refine_converged": bool(refine_result.success),
        "refine_elapsed_s": refine_elapsed,
        "wall_elapsed_s": time.perf_counter() - start,
        "per_grid_point": per_point,
    }


def main() -> int:
    require_clean_tree()
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    base_config = load_process_config(ka1.CONFIG_PATH)
    assert base_config.material.critical_moisture_kg_per_kg == 0.20
    # The kinetics stay at the config value; the quality limit is the explicit threshold below.

    grid52 = json.loads(
        (RUNS_DIR / "study_operating_point_regret_win_repro.json").read_text("utf-8")
    )
    regret = ka1._load_regret_module()
    workers = cpu_count()
    budget = regret.OptimizationBudget(
        seed=0,
        popsize=regret.DEFAULT_POPSIZE,
        maxiter=regret.DEFAULT_MAXITER,
        de_tol=regret.DEFAULT_DE_TOL,
        search_max_iterations=regret.DEFAULT_SEARCH_MAX_ITERATIONS,
        search_tolerance=regret.DEFAULT_SEARCH_TOLERANCE,
        workers=workers,
    )
    budget_dict = dataclasses.asdict(budget)
    for key in (
        "de_tol",
        "maxiter",
        "popsize",
        "search_max_iterations",
        "search_tolerance",
        "seed",
    ):
        assert budget_dict[key] == grid52["budget"][key], key
    grid_points = regret.build_grid(base_config, 5)

    oracle = dec.run_oracle_grid_decoupled(
        base_config,
        regret,
        grid_size=5,
        workers=workers,
        budget=budget,
        effective_critical_moisture_kg_per_kg=QUALITY_LIMIT_KG_PER_KG,
        label="oracle_limit_0.15",
    )
    fixed = find_fixed_point_decoupled(
        regret, base_config, grid_points, budget, QUALITY_LIMIT_KG_PER_KG
    )
    write_artifact(
        ARTIFACT_NAME,
        {
            "_provenance": provenance(),
            "quality_limit_kg_per_kg": QUALITY_LIMIT_KG_PER_KG,
            "critical_moisture_kg_per_kg_kinetics": base_config.material.critical_moisture_kg_per_kg,
            "config_fingerprint_kinetics_config": base_config.fingerprint,
            "budget": budget_dict,
            "budget_befund52": grid52["budget"],
            "grid_results": oracle,
            "fixed_point": fixed,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
