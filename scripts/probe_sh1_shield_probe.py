r"""SH1 (read-only probe, no training): justify the push direction and measure the shield's cost.

Environment of arm C (quality limit 0.15, Lagrange term, lambda_0 0.5, m = 0.005961819615175404,
5 x 5 cells, eta 0.2 kW). Test contexts: the 25 Befund 52 grid points. Two parts:

  (a) Direction. For every context and gas velocity u in {0.8, 1.2, 1.6, 2.0, 2.5} m/s the
      steady-state product moisture on a grid of 25 inlet temperatures in [60, 120] degC. The
      push direction (inlet temperature upwards) is justified iff the moisture never increases
      along the grid (tolerance 1e-9 kg/kg) among the solved points. Reported: violations, solver
      failures per inlet temperature.
  (b) Cost and behaviour of the shield. For every context, every gas velocity above and every
      proposed inlet temperature in {60, 70, 80, 90, 100, 110} degC: ``shield_action`` with the
      default tolerance (0.05 K), wall time, solver calls, whether it is active, saturated,
      moisture and bed temperature at the executed point.

Decision rule, fixed before the run: the direction is kept iff (a) shows no violation of the
monotonicity. If it does, nothing is trained and the direction is put to the project lead.
Estimate of the run time before this probe (ad-hoc timing of 60 solver calls, not an artifact):
0.17 s per solver call, 0.46 s per training step without shield (4600-4800 s per 10 000-step run
of arm C, ``wall_s_since_training_start``), so about 0.29 s per step for the SAC update.

    python scripts/probe_sh1_shield_probe.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "artifacts" / "runs"
REFERENCE_015 = RUNS / "s05_limit015_oracle_and_fixed.json"
MARGIN_M = 0.005961819615175404
LIMIT = 0.15
GAS_VELOCITIES = (0.8, 1.2, 1.6, 2.0, 2.5)
GRID_T_IN = tuple(np.linspace(60.0, 120.0, 25))
PROPOSED_T_IN = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0)
MONOTONE_TOLERANCE = 1e-9
WORKERS = 12

_ENV: Any = None


def _env() -> Any:
    global _ENV
    if _ENV is None:
        from twinlink_ml.rl import BanditEnvConfig
        from twinlink_ml.rl.lagrange import LagrangeConfig
        from twinlink_ml.rl.shield import ShieldedLagrangeBanditEnv

        warnings.filterwarnings("ignore")
        lagrange = LagrangeConfig(
            margin_kg_per_kg=MARGIN_M,
            initial_multiplier_kw=0.5,
            step_size_kw=0.2,
            update_interval_steps=250,
            context_bins=(5, 5),
        )
        _ENV = ShieldedLagrangeBanditEnv(
            lagrange, BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT)
        )
    return _ENV


def _task(task: tuple[float, float, float]) -> dict[str, Any]:
    t_amb, phi, _ = task
    env = _env()
    grid: list[dict[str, Any]] = []
    shield: list[dict[str, Any]] = []
    for u in GAS_VELOCITIES:
        moistures: list[float | None] = []
        for t_in in GRID_T_IN:
            _, info = env._plain_evaluate(float(t_in), u, t_amb, phi)
            moistures.append(
                None
                if info.get("solve_failed")
                else float(info["constraint_check"]["moisture_kg_per_kg"])
            )
        grid.append({"u_gas": u, "moisture": moistures})
        for t_prop in PROPOSED_T_IN:
            started = time.perf_counter()
            record, info = env.shield_action(t_prop, u, t_amb, phi)
            seconds = time.perf_counter() - started
            row: dict[str, Any] = {
                "u_gas": u,
                "t_proposed": t_prop,
                "seconds": seconds,
                "n_solves": record["n_solves"],
                "active": record["active"],
                "saturated": record["saturated"],
                "proposed_moisture": record["proposed_moisture"],
                "t_executed": record["t_exec"],
                "solve_failed_proposal": bool(np.isnan(record["proposed_moisture"])),
            }
            if not info.get("solve_failed"):
                check = info["constraint_check"]
                row["executed_moisture"] = float(check["moisture_kg_per_kg"])
                row["executed_bed_temperature"] = float(check["bed_temperature_celsius"])
                row["executed_bed_violation_k"] = float(check["bed_temperature_violation_k"])
            shield.append(row)
    return {"t_amb": t_amb, "phi": phi, "grid": grid, "shield": shield}


def main() -> int:
    require_clean_tree()
    from twinlink_ml.rl import load_test_points

    points = load_test_points(REFERENCE_015)
    tasks = [(p.ambient_temperature_celsius, p.ambient_relative_humidity, 0.0) for p in points]
    started = time.perf_counter()
    with Pool(WORKERS) as pool:
        results = pool.map(_task, tasks, chunksize=1)
    wall = time.perf_counter() - started

    violations = []
    failures_per_t = np.zeros(len(GRID_T_IN), dtype=int)
    n_curves = 0
    for r in results:
        for g in r["grid"]:
            n_curves += 1
            solved = [
                (t, x) for t, x in zip(GRID_T_IN, g["moisture"], strict=True) if x is not None
            ]
            for (t0, x0), (t1, x1) in pairwise(solved):
                if x1 > x0 + MONOTONE_TOLERANCE:
                    violations.append(
                        {
                            "t_amb": r["t_amb"],
                            "phi": r["phi"],
                            "u_gas": g["u_gas"],
                            "t0": t0,
                            "t1": t1,
                        }
                    )
            for k, x in enumerate(g["moisture"]):
                failures_per_t[k] += x is None
    rows = [dict(row, t_amb=r["t_amb"], phi=r["phi"]) for r in results for row in r["shield"]]
    active = [row for row in rows if row["active"]]
    solved_exec = [row for row in active if "executed_moisture" in row]
    payload = {
        "_provenance": provenance(),
        "setup": {
            "arm": "C environment (limit 0.15, Lagrange, m)",
            "gas_velocities": list(GAS_VELOCITIES),
            "t_in_grid": list(GRID_T_IN),
            "proposed_t_in": list(PROPOSED_T_IN),
            "contexts": len(results),
            "workers": WORKERS,
            "wall_s_total": wall,
        },
        "direction": {
            "n_curves": n_curves,
            "n_monotonicity_violations": len(violations),
            "violations": violations[:50],
            "solver_failures_per_t_in": dict(
                zip([f"{t:.1f}" for t in GRID_T_IN], failures_per_t.tolist(), strict=True)
            ),
            "direction_kept": not violations,
        },
        "shield": {
            "n_combinations": len(rows),
            "n_solver_failed_proposals": int(sum(row["solve_failed_proposal"] for row in rows)),
            "n_active": len(active),
            "n_saturated": sum(row["saturated"] for row in active),
            "active_seconds_median": float(np.median([row["seconds"] for row in active])),
            "active_seconds_p90": float(np.percentile([row["seconds"] for row in active], 90)),
            "active_seconds_max": float(max(row["seconds"] for row in active)),
            "active_n_solves_median": float(np.median([row["n_solves"] for row in active])),
            "active_n_solves_max": int(max(row["n_solves"] for row in active)),
            "inactive_seconds_median": float(
                np.median([row["seconds"] for row in rows if not row["active"]])
            ),
            "executed_distance_to_target_max_kg_per_kg": float(
                max(
                    abs(row["executed_moisture"] - (LIMIT - MARGIN_M))
                    for row in solved_exec
                    if not row["saturated"]
                )
            ),
            "n_executed_bed_temperature_violation": int(
                sum(row["executed_bed_violation_k"] > 0.0 for row in solved_exec)
            ),
            "rows": rows,
        },
    }
    write_artifact("sh1_shield_probe", payload)
    print(
        json.dumps({k: v for k, v in payload["direction"].items() if k != "violations"}, indent=1)
    )
    print(json.dumps({k: v for k, v in payload["shield"].items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
