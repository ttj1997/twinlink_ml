r"""B1: anchors of test set T3 (deployment profiles) -- optimum and moisture-limit operating points.

For four ambient conditions (quartiles of the disturbance ranges) the constrained steady-state
optimum of the legacy objective (``energy_target``, as Befund 52) is searched with the mechanics
of Befund 52 / M1 (``carrier_oracle.optimize_point``: DE popsize 10, maxiter 50, seed 0,
Nelder-Mead, search precision 30 / 1e-7, corrected solver bracket of Befund 78/79) at

* lower gas-velocity bound 1.26 m/s (laboratory minimum; a run parameter, config default unchanged),
* quality limit 0.15 kg/kg on the product moisture (``MaterialConfig.product_moisture_limit_kg_per_kg``
  field semantics; the kinetics keep ``critical_moisture_kg_per_kg``).

A second operating point per ambient condition lies on the spray-rate line through the optimum, at
the spray rate where the steady-state product moisture is HALF the M1 soft-sensor margin
(0.005961819615175404 kg/kg) below the quality limit ("near the limit").

Output: ``artifacts/runs/<name>.json`` (never overwritten) read by
``twinlink_ml.evaluation.b1_testsets.build_t3``.

    tools/run_here.sh scripts/compute_b1_t3_anchors.py --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (
    CONFIG_PATH,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.evaluation.b1_testsets import (
    SEARCH_MAX_ITERATIONS,
    SEARCH_TOLERANCE,
    find_limit_spray,
    steady_state_at,
)
from twinlink_ml.evaluation.carrier_oracle import OracleSpec, run_oracle_grid
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.oracle_search import (
    OptimizationBudget,
    action_bounds,
    with_gas_velocity_minimum,
)

QUALITY_LIMIT = 0.15
GAS_VELOCITY_MIN = 1.26
MARGIN_M1 = 0.005961819615175404
AMBIENT_UNITS = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="b1_t3_anchors")
    arguments = parser.parse_args()
    if arguments.workers > 8:
        raise SystemExit("at most 8 workers")
    require_clean_tree()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    if (RUNS_DIR / f"{arguments.name}.json").exists():
        raise SystemExit("artifact exists; use a new --name")

    t_range = config.disturbances.ambient_temperature_celsius
    rh_range = config.disturbances.ambient_relative_humidity
    grid = [
        (
            t_range.minimum + u_t * (t_range.maximum - t_range.minimum),
            rh_range.minimum + u_rh * (rh_range.maximum - rh_range.minimum),
        )
        for u_t, u_rh in AMBIENT_UNITS
    ]
    budget = OptimizationBudget(
        seed=0,
        popsize=10,
        maxiter=50,
        de_tol=1.0e-4,
        search_max_iterations=SEARCH_MAX_ITERATIONS,
        search_tolerance=SEARCH_TOLERANCE,
        workers=arguments.workers,
    )
    spec = OracleSpec(
        quality_threshold_kg_per_kg=QUALITY_LIMIT,
        objective="legacy_power_kw",
        search_max_iterations=SEARCH_MAX_ITERATIONS,
        search_tolerance=SEARCH_TOLERANCE,
    )
    bounds = with_gas_velocity_minimum(action_bounds(config), GAS_VELOCITY_MIN)
    start = time.perf_counter()
    results = run_oracle_grid(
        config, grid, bounds, budget, spec, workers=arguments.workers, label="b1 T3 optimum"
    )
    oracle_elapsed = time.perf_counter() - start

    spray_nominal = config.actions.spray_rate_kg_per_h.nominal
    solid_nominal = config.disturbances.suspension_solid_fraction.nominal
    anchors = []
    for (u_t, u_rh), (t_amb, rh), result in zip(AMBIENT_UNITS, grid, results, strict=True):
        if result.get("solve_failed") or not result.get("feasible"):
            raise SystemExit(f"oracle infeasible at ambient {t_amb}, {rh}: {result}")
        optimum = {
            "t_in": result["inlet_temperature_celsius"],
            "u_gas": result["gas_velocity_m_per_s"],
            "spray": spray_nominal,
            "p_nozzle": result["nozzle_pressure_bar"],
        }
        disturbance = (t_amb, rh, solid_nominal)
        opt_solution = steady_state_at(
            config,
            (optimum["t_in"], optimum["u_gas"], optimum["spray"], optimum["p_nozzle"]),
            disturbance,
        )
        spray, moisture = find_limit_spray(
            config,
            (optimum["t_in"], optimum["u_gas"], optimum["spray"], optimum["p_nozzle"]),
            disturbance,
            target_moisture_kg_per_kg=QUALITY_LIMIT - 0.5 * MARGIN_M1,
        )
        anchors.append(
            {
                "ambient_unit": [u_t, u_rh],
                "ambient": [t_amb, rh],
                "optimum": optimum,
                "optimum_moisture_kg_per_kg": opt_solution.particle_moisture_kg_per_kg,
                "optimum_binds_gas_lower": abs(optimum["u_gas"] - GAS_VELOCITY_MIN) < 1e-3,
                "optimum_energy_target_kw": result.get("legacy_energy_target_kw"),
                "limit": {**optimum, "spray": spray},
                "limit_moisture_kg_per_kg": moisture,
            }
        )
        print(
            f"ambient {t_amb:.1f} C / {rh:.2f}: optimum {optimum}, moisture "
            f"{opt_solution.particle_moisture_kg_per_kg:.4f}; limit spray {spray:.3f} "
            f"(moisture {moisture:.4f})",
            flush=True,
        )
    write_artifact(
        arguments.name,
        {
            "_provenance": provenance(),
            "purpose": "anchors of B1 test set T3",
            "quality_limit_kg_per_kg": QUALITY_LIMIT,
            "u_gas_min": GAS_VELOCITY_MIN,
            "margin_m1_kg_per_kg": MARGIN_M1,
            "limit_target_kg_per_kg": QUALITY_LIMIT - 0.5 * MARGIN_M1,
            "objective": "legacy_power_kw (energy_target, Befund 52)",
            "budget": {
                "seed": budget.seed,
                "popsize": budget.popsize,
                "maxiter": budget.maxiter,
                "de_tol": budget.de_tol,
                "search_max_iterations": budget.search_max_iterations,
                "search_tolerance": budget.search_tolerance,
            },
            "oracle_elapsed_s": oracle_elapsed,
            "workers": arguments.workers,
            "anchors": anchors,
        },
    )


if __name__ == "__main__":
    main()
