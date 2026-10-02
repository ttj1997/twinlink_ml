r"""S0.6 diagnosis after the gate: why lambda stayed inactive, and where the agents sit.

Report only. Reads the committed traces, lambda histories and the artifacts ``s03_gate``,
``s06_gate``. No training, no setting is changed.

For the four Lagrange arms (A-D) per seed:
  * the largest lambda over all updates and cells, and the number of updates in which any
    cell's lambda went up;
  * the number of training steps with ``g_tilde > 0``, how many of them after step 2500, and
    the median moisture of those steps;
  * over all (update window x context cell) groups with at least one solved step: the share
    whose mean ``g_tilde`` is positive (the quantity the dual ascent step uses);
  * the median product moisture of the steps 7500-10000.

For every arm (including the penalty arms): the distance of the moisture to the SOLVER
VALIDITY BOUNDARY 0.20 kg/kg (S0.4) at the oracle-binding contexts, derived from the gate
artifacts as ``0.20 - limit + distance_to_limit``.

    python scripts/probe_s06_diagnosis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact

RUNS = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
SEEDS = (0, 1, 2, 3, 4)
UPDATE_INTERVAL = 250
VALIDITY_BOUNDARY = 0.20
ARMS = {
    "A_L020_lam05": "rl_bandit_lagrange_s03",
    "B_L020_lam0": "rl_bandit_lagrange_s06_l020_lam0",
    "C_L015_lam05": "rl_bandit_lagrange_s06_l015_lam05",
    "D_L015_lam0": "rl_bandit_lagrange_s06_l015_lam0",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def arm_diagnosis(prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in SEEDS:
        trace = _rows(RUNS / f"{prefix}_trace_seed{seed}.jsonl")
        history = _rows(RUNS / f"{prefix}_lambda_seed{seed}.jsonl")
        matrix = np.array([h["multipliers_kw"] for h in history])
        increases = int(np.sum(np.any(np.diff(matrix, axis=0) > 1e-12, axis=1)))
        solved = [r for r in trace if not r["solve_failed"]]
        positive = [r for r in solved if r["g_tilde"] > 0.0]
        groups: dict[tuple[int, int], list[float]] = {}
        for r in solved:
            groups.setdefault(((r["step"] - 1) // UPDATE_INTERVAL, r["bin"]), []).append(
                r["g_tilde"]
            )
        means = np.array([np.mean(v) for v in groups.values()])
        late = [r["moisture_kg_per_kg"] for r in solved if r["step"] > 7500]
        out[str(seed)] = {
            "max_lambda_ever_kw": float(matrix.max()),
            "lambda_at_first_update_max_kw": float(matrix[0].max()),
            "n_updates_with_any_increase": increases,
            "n_steps_g_tilde_positive": len(positive),
            "n_steps_g_tilde_positive_after_step_2500": sum(
                1 for r in positive if r["step"] > 2500
            ),
            "median_moisture_of_positive_steps": (
                float(np.median([r["moisture_kg_per_kg"] for r in positive])) if positive else None
            ),
            "n_window_cell_groups": len(means),
            "share_groups_with_positive_mean_g_tilde": float(np.mean(means > 0.0)),
            "median_moisture_steps_7500_10000": float(np.median(late)),
        }
    return out


def validity_distance() -> dict[str, Any]:
    s06 = json.loads((RUNS / "s06_gate.json").read_text(encoding="utf-8"))
    s03 = json.loads((RUNS / "s03_gate.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for name, arm in s06["arms"].items():
        limit = arm["limit_kg_per_kg"]
        per_seed = [
            VALIDITY_BOUNDARY - limit + d
            for d in arm["distance_to_limit_binding_kg_per_kg"]["per_seed"]
        ]
        out[name] = {"limit": limit, "per_seed": per_seed, "median": float(np.median(per_seed))}
    pen = s03["penalty"]["moisture_margin_binding_median_kg_per_kg"]["per_seed"]
    out["penalty_L020_s03"] = {
        "limit": 0.20,
        "per_seed": pen,
        "median": float(np.median(pen)),
    }
    return out


def main() -> int:
    require_clean_tree()
    payload = {
        "_provenance": provenance(),
        "arms": {name: arm_diagnosis(prefix) for name, prefix in ARMS.items()},
        "distance_to_validity_boundary_0_20_at_binding_contexts": validity_distance(),
    }
    write_artifact("s06_diagnosis", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
