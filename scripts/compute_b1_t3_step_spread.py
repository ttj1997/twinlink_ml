r"""B1: spread of the oracle set points over the 25 disturbance scenarios (step sizes of T3, family 2).

Decision of the project lead (2026-10-12): family 2 of test set T3 makes larger steps across the
moisture limit; the step size comes from how far the oracle set points move between the 25
disturbance scenarios of Befund 52 (realistic movements of an optimiser). This script computes
exactly that: the per-scenario optimum of the legacy objective (``energy_target``) with the
mechanics of Befund 52 / M1 (``carrier_oracle``: DE popsize 10, maxiter 50, seed 0, Nelder-Mead,
precision 30 / 1e-7, corrected solver bracket of Befund 78/79) at the lower gas-velocity bound
1.26 m/s and the quality limit 0.15 kg/kg (decoupled field), on the 5 x 5 ambient grid of Befund 52.
Spray rate is not optimised by the oracle (line A, 1.5 kg/h); its step is derived in
``b1_testsets`` from the largest relative spread of the other channels.

Output: ``artifacts/runs/<name>.json`` (never overwritten): per-scenario set points and, per channel,
range (max - min), standard deviation, interquartile range and 10-90 % range.

    tools/run_here.sh scripts/compute_b1_t3_step_spread.py --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

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
)
from twinlink_ml.evaluation.carrier_oracle import OracleSpec, run_oracle_grid
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.oracle_search import (
    OptimizationBudget,
    action_bounds,
    build_grid,
    with_gas_velocity_minimum,
)

QUALITY_LIMIT = 0.15
GAS_VELOCITY_MIN = 1.26
CHANNELS = ("inlet_temperature_celsius", "gas_velocity_m_per_s", "nozzle_pressure_bar")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="b1_t3_step_spread")
    arguments = parser.parse_args()
    if arguments.workers > 8:
        raise SystemExit("at most 8 workers")
    require_clean_tree()
    if (RUNS_DIR / f"{arguments.name}.json").exists():
        raise SystemExit("artifact exists; use a new --name")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    grid = build_grid(config, 5)
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
        config, grid, bounds, budget, spec, workers=arguments.workers, label="b1 T3 step spread"
    )
    elapsed = time.perf_counter() - start

    feasible = [r for r in results if r.get("feasible") and not r.get("solve_failed")]
    scenarios = [
        {
            "ambient_temperature_celsius": r["ambient_temperature_celsius"],
            "ambient_relative_humidity": r["ambient_relative_humidity"],
            **{c: r[c] for c in CHANNELS},
            "particle_moisture_kg_per_kg": r["particle_moisture_kg_per_kg"],
        }
        for r in feasible
    ]
    spread: dict[str, dict[str, float]] = {}
    for channel in CHANNELS:
        values = np.array([s[channel] for s in scenarios])
        q10, q25, q75, q90 = np.percentile(values, [10, 25, 75, 90])
        spread[channel] = {
            "range": float(values.max() - values.min()),
            "std": float(values.std(ddof=1)),
            "iqr": float(q75 - q25),
            "p10_p90": float(q90 - q10),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        print(f"{channel}: {spread[channel]}", flush=True)
    write_artifact(
        arguments.name,
        {
            "_provenance": provenance(),
            "purpose": "step sizes of T3 family 2 (spread of oracle set points over 25 scenarios)",
            "quality_limit_kg_per_kg": QUALITY_LIMIT,
            "u_gas_min": GAS_VELOCITY_MIN,
            "n_grid_points": len(grid),
            "n_feasible": len(feasible),
            "oracle_elapsed_s": elapsed,
            "workers": arguments.workers,
            "spread": spread,
            "scenarios": scenarios,
        },
    )


if __name__ == "__main__":
    main()
