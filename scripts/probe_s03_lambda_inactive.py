r"""S0.3 diagnosis after the gate: lambda stayed 0 in every cell of every run.

Reads the committed per-step traces of the Lagrange arms (5x5 cells and global) and counts how
often the signed constraint was positive (product moisture above the tightened limit). Then
evaluates the 13 x 9 reference grid at the nominal disturbance (the grid of the solver-failure
bound) and reports, at every grid point with g_tilde > 0, whether the solver solution is
infeasible (``solver_feasible`` False, which adds the fixed +1.0 to the normalised violation and
therefore 50 kW to the "rest" penalty that the Lagrange arm kept, as in the penalty arm).

Report only: no setting, no gate constant and no training run is changed by this script.

    python scripts/probe_s03_lambda_inactive.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact

from twinlink_ml.rl.lagrange import (
    REFERENCE_GRID_T_IN,
    REFERENCE_GRID_U_GAS,
    LagrangeBanditEnv,
    LagrangeConfig,
)
from twinlink_ml.simulation import PlaceholderConfigWarning

RUNS = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
ARMS = ("rl_bandit_lagrange_s03", "rl_bandit_lagrange_s03_global")
SEEDS = (0, 1, 2, 3, 4)
MARGIN_KG_PER_KG = 0.005961819615175404


def trace_counts(prefix: str) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        path = RUNS / f"{prefix}_trace_seed{seed}.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        solved = [r for r in rows if not r["solve_failed"]]
        g = np.array([r["g_tilde"] for r in solved])
        moisture = np.array([r["moisture_kg_per_kg"] for r in solved])
        per_seed[str(seed)] = {
            "n_steps": len(rows),
            "n_solve_failed": len(rows) - len(solved),
            "n_steps_g_tilde_positive": int((g > 0.0).sum()),
            "g_tilde_max": float(g.max()),
            "moisture_p95_kg_per_kg": float(np.percentile(moisture, 95)),
            "moisture_max_kg_per_kg": float(moisture.max()),
        }
    return per_seed


def reference_grid() -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    env = LagrangeBanditEnv(LagrangeConfig(MARGIN_KG_PER_KG, 0.5, 0.2, 250))
    points: list[dict[str, Any]] = []
    for t_in in REFERENCE_GRID_T_IN:
        for u_gas in REFERENCE_GRID_U_GAS:
            _, info = env.evaluate_operating_point(
                inlet_temperature_celsius=float(t_in),
                gas_velocity_m_per_s=float(u_gas),
                ambient_temperature_celsius=env._t_amb_range.nominal,
                ambient_relative_humidity=env._phi_range.nominal,
            )
            if info.get("solve_failed"):
                points.append({"t_in": float(t_in), "u_gas": float(u_gas), "solve_failed": True})
                continue
            check = info["constraint_check"]
            points.append(
                {
                    "t_in": float(t_in),
                    "u_gas": float(u_gas),
                    "solve_failed": False,
                    "energy_kw": info["lagrange_energy_kw"],
                    "rest_penalty_kw": info["lagrange_rest_penalty_kw"],
                    "g_tilde": info["lagrange_g_tilde"],
                    "moisture_kg_per_kg": check["moisture_kg_per_kg"],
                    "solver_feasible": bool(check["solver_feasible"]),
                }
            )
    solved = [p for p in points if not p["solve_failed"]]
    positive = [p for p in solved if p["g_tilde"] > 0.0]
    return {
        "disturbance": "nominal (ambient temperature and humidity at their nominal values)",
        "n_points": len(points),
        "n_solve_failed": len(points) - len(solved),
        "n_g_tilde_positive": len(positive),
        "n_g_tilde_positive_with_solver_infeasible": sum(
            1 for p in positive if not p["solver_feasible"]
        ),
        "n_g_tilde_positive_with_rest_penalty": sum(
            1 for p in positive if p["rest_penalty_kw"] > 0.0
        ),
        "points_g_tilde_positive": positive,
        "closest_to_tightened_limit": sorted(solved, key=lambda p: abs(p["g_tilde"]))[:6],
    }


def main() -> int:
    require_clean_tree()
    payload = {
        "_provenance": provenance(),
        "trace_counts": {prefix: trace_counts(prefix) for prefix in ARMS},
        "reference_grid": reference_grid(),
    }
    write_artifact("s03_lambda_inactive_diagnosis", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
