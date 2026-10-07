r"""Robustness re-measurement of the verdict of Befund 73 with the direct five-seed search (all 25 scenarios).

The verdict of Befund 73 (coupling gain of the control NOT demonstrated, median 0.384 kg CO2e/h) STANDS AS
MEASURED; this is a re-measurement with a more robust search (direct DE + Nelder-Mead on true solves, five
seeds, a shortlist of allocations per scenario, S2 from five seeds per line and the best of 25 pairs), box and
test set as in Befund 73: 25 Befund 52 ambient points, lines A and B, scaled site, no exchanger limit (the
control), inlet temperature 60-120 degC, gas velocity lower bound 1.26 m/s. FIXED HERE BEFORE THE RUNS
(project lead, 2026-10-07):

* Inputs: the evaluation artifacts of the three direct runs (scenarios 16, 17, 22, 23, 24; 18, 19; the other
  18 with shortlist cap 5), produced by ``analyze_m3x_confirm.py``.
* Per scenario ``G = CO2(best S2) - CO2(best M3')`` in kg CO2e/h of the site, where best S2 is the minimum
  over the 25 pairs of line seeds and best M3' the minimum over the five seeds (scenarios in which either
  solution violates a constraint are masked and counted).
* Seed spread: the largest, over the scenarios, of (largest minus smallest M3' value over the five seeds).
* Verdict: ``twinlink_ml.evaluation.coupling_gain.coupling_gain_verdict`` UNCHANGED (median G at least 1.0,
  median at least 3 times the seed spread, G above 0.2 at least 60 % of the scenarios).
* Descriptive (no rule): G with the median instead of the best M3' over the seeds; the difference to the G of
  Befund 73 per scenario; the number of scenarios where the direct search finds a better M3' than the
  table search (difference above the resolution 0.2 kg/h).

    python scripts/analyze_m3x_robust.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

from twinlink_ml.evaluation import coupling_gain as cg

SUFFIX = "umin126"
INPUTS = (
    f"m3x_confirm_evaluation_{SUFFIX}",
    f"m3x_confirm_s18_19_evaluation_{SUFFIX}",
    f"m3x_robust_rest_evaluation_{SUFFIX}",
)
RESOLUTION = cg.GAIN_PER_SCENARIO_MIN_KG_PER_H


def _load(name: str) -> dict[str, Any]:
    return json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))


def main() -> int:
    require_clean_tree()
    scenarios: dict[int, dict[str, Any]] = {}
    for name in INPUTS:
        for s, v in _load(name)["scenarios"].items():
            if int(s) in scenarios:
                raise SystemExit(f"scenario {s} twice in the inputs")
            scenarios[int(s)] = v
    if sorted(scenarios) != list(range(25)):
        raise SystemExit(f"scenarios incomplete: {sorted(scenarios)}")
    control = {r["grid_index"]: r for r in _load(f"m3_joint_control_{SUFFIX}")["scenarios"]}
    rows: dict[int, dict[str, Any]] = {}
    for s, v in sorted(scenarios.items()):
        seeds = v["per_seed_m3"]
        valid = bool(v["best_s2"]["feasible"]) and all(e["feasible"] for e in seeds.values())
        m3_values = [e["value"] for e in seeds.values()]
        s2_value = v["best_s2"]["value"]
        best = min(m3_values)
        rows[s] = {
            "valid": valid,
            "g_best_kg_per_h": s2_value - best,
            "g_median_seed_kg_per_h": s2_value - float(np.median(m3_values)),
            "m3_spread_kg_per_h": max(m3_values) - min(m3_values),
            "g_befund73_kg_per_h": control[s]["s2_co2_kg_per_h"] - control[s]["m3_co2_kg_per_h"],
            "s2_co2_lines_befund73": control[s]["s2_co2_lines_kg_per_h"],
        }
    valid_rows = {s: r for s, r in rows.items() if r["valid"]}
    spread = max(r["m3_spread_kg_per_h"] for r in rows.values())
    gains = [r["g_best_kg_per_h"] for r in valid_rows.values()]
    verdict = cg.coupling_gain_verdict(
        gains, spread, [r["s2_co2_lines_befund73"] for r in valid_rows.values()]
    )
    old = [r["g_befund73_kg_per_h"] for r in valid_rows.values()]
    improved = [
        s
        for s, r in valid_rows.items()
        if r["g_best_kg_per_h"] - r["g_befund73_kg_per_h"] > RESOLUTION
    ]
    payload = {
        "_provenance": provenance(),
        "note": "re-measurement; the verdict of Befund 73 stands as measured",
        "inputs": list(INPUTS),
        "n_masked": len(rows) - len(valid_rows),
        "rows": {str(s): r for s, r in rows.items()},
        "verdict_direct_five_seeds": verdict,
        "median_g_befund73": float(np.median(old)),
        "median_g_with_median_seed": float(
            np.median([r["g_median_seed_kg_per_h"] for r in valid_rows.values()])
        ),
        "scenarios_where_direct_search_better_than_befund73_by_more_than_resolution": improved,
    }
    write_artifact(f"m3x_robust_verdict_{SUFFIX}", payload)
    print(json.dumps(verdict, indent=1))
    print("improved:", improved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
