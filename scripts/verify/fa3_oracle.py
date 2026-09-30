"""FA3 (1/2) -- per-instance oracle with a tightened quality limit, decoupled mechanics of Befund 64.

A controller that keeps a margin ``m`` to the moisture limit operates like an optimiser against the
constraint ``0.20 - m`` (Befund 64). For the two soft-sensor margins of the second addendum of Befund 65
(excited-trained 0.0016282 kg/kg, constant-trained 0.0044651 kg/kg; 3 x the global RMSE of Befund 62 HA1)
the 25 per-instance oracle setpoints of the Befund 52 grid (5 x 5) are re-solved with the SAME optimiser
budget as Befund 52 (differential evolution popsize 10, maxiter 50, tolerance 1e-4, seed 0, refinement) and
the SAME material: ``material.critical_moisture_kg_per_kg`` stays 0.20 kg/kg (drying kinetics unchanged),
only the threshold handed to the quality constraint moves (``probe_margin_energy_cost_decoupled``,
imported unchanged; ``study_operating_point_regret`` untouched).

Runtime estimate (before the run): Befund 52 logged 520-580 s wall per grid point for the same search; with
16 workers 25 points take two waves, about 20 min per margin, about 40 min for both (a sequential run of the
two margins, the machine otherwise idle).

    python scripts/verify/fa3_oracle.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import warnings
from multiprocessing import cpu_count
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_margin_energy_cost as ka1  # noqa: E402
import probe_margin_energy_cost_decoupled as dec  # noqa: E402
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact  # noqa: E402

from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402


def main() -> None:
    require_clean_tree()
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    base_config = load_process_config(ka1.CONFIG_PATH)
    assert base_config.material.critical_moisture_kg_per_kg == 0.20

    grid52 = json.loads(
        (RUNS_DIR / "study_operating_point_regret_win_repro.json").read_text("utf-8")
    )
    assert grid52["grid_size"] == 5
    ref = json.loads((RUNS_DIR / "probe_soft_sensor_margin_requirement.json").read_text("utf-8"))
    margins = {
        row["label"]: row["margin_kg_per_kg"]
        for row in ref["merged_dataset_sorted_by_margin"]
        if row["label"] in ("excited_lstm_3rmse", "constant_lstm_3rmse")
    }
    assert len(margins) == 2

    regret_module = ka1._load_regret_module()
    workers = cpu_count()
    budget = regret_module.OptimizationBudget(
        seed=0,
        popsize=regret_module.DEFAULT_POPSIZE,
        maxiter=regret_module.DEFAULT_MAXITER,
        de_tol=regret_module.DEFAULT_DE_TOL,
        search_max_iterations=regret_module.DEFAULT_SEARCH_MAX_ITERATIONS,
        search_tolerance=regret_module.DEFAULT_SEARCH_TOLERANCE,
        workers=workers,
    )
    b52 = grid52["budget"]
    bd = dataclasses.asdict(budget)
    for key in (
        "de_tol",
        "maxiter",
        "popsize",
        "search_max_iterations",
        "search_tolerance",
        "seed",
    ):
        assert bd[key] == b52[key], (key, bd[key], b52[key])

    fresh = [(t, p) for t, p in regret_module.build_grid(base_config, 5)]
    ref_grid = [
        (g["ambient_temperature_celsius"], g["ambient_relative_humidity"])
        for g in grid52["grid_results"]
    ]
    assert all(
        abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12
        for a, b in zip(fresh, ref_grid, strict=True)
    )

    results = {}
    for label, margin in margins.items():
        results[label] = {
            "margin_kg_per_kg": margin,
            "effective_threshold_kg_per_kg": 0.20 - margin,
            "grid_results": dec.run_oracle_grid_decoupled(
                base_config,
                regret_module,
                grid_size=5,
                workers=workers,
                budget=budget,
                effective_critical_moisture_kg_per_kg=0.20 - margin,
                label=label,
            ),
        }
        n_ok = sum(
            1
            for r in results[label]["grid_results"]
            if r.get("feasible") and not r.get("solve_failed")
        )
        print(label, "feasible", n_ok, "/ 25", flush=True)
    write_artifact(
        "verify_fa3_tightened_oracle",
        {
            "_provenance": provenance(),
            "config_fingerprint": base_config.fingerprint,
            "budget": bd,
            "budget_befund52": b52,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
