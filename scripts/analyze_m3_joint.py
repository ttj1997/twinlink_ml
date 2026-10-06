r"""M3' joint stage at 1.26 m/s: evaluation by the rule fixed beforehand.

Pure evaluation of the artifacts of ``run_m3_joint.py`` (``m3_joint_<case>_umin126`` for the cases control,
c0.5, c1.0, c1.5; ``m3_joint_seeds_umin126``; ``m3_check_umin126``). Test set (always named): the 25
Befund 52 ambient grid points; lines A (1.5 kg/h) and B (2.0 kg/h), scaled site, lower gas-velocity bound
1.26 m/s only. THE EVALUATION RULES ARE FIXED HERE BEFORE ANY RUN OF THE JOINT STAGE (project lead,
2026-10-06):

* THE CONTROL (no limit; the two lines share the limited residual heat of Q1 without any synthetic
  assumption) IS THE MAIN RESULT. The coupling gain per scenario is ``G = CO2(S2) - CO2(M3')`` in kg CO2e/h
  of the site, over the scenarios in which both S2 and M3' satisfy the constraints of both lines (true check
  against the tightened limit; the others are counted and masked). The verdict is that of
  ``twinlink_ml.evaluation.coupling_gain.coupling_gain_verdict`` with the seed spread measured on the three
  scenarios (largest spread of the M3' value over the DE seeds 0, 1, 2, in kg CO2e/h, for the control), and G
  is also given relative to the CO2 attributable to the two lines (in S2).
* ``G(c) - G(control)`` for c = 0.5, 1.0, 1.5 is the additional effect of the synthetic exchanger limit; it
  is reported descriptively (median, range, number of scenarios with a positive difference) next to the
  seed spread of that case, WITHOUT a decision rule.
* Per scenario and case: whether M3' chooses another allocation than S2 (exact tuple, the same set of
  connections, and, because many allocations are equivalent (connections that carry no heat), whether the
  change of the allocation brings a material gain: ``G_alloc`` above 0.2 kg CO2e/h), and the split of G into the gain of the set points for the allocation of S2 and the further
  gain from another allocation (separating gain from the allocation from gain from the set points).
* Table accuracy rule: at the three check scenarios and for every case the table-based M3' value must not
  exceed the value of the direct true-solve DE for the same allocation by more than 3 times the measured seed
  spread of that case; failures are listed (the table is then refined at that scenario or the scenario is run
  with a shortlist and reported). The largest table error at the polished points, the final polishing window
  and the number of widened scenarios are reported per case (rule of ``joint_search``: error at most half the
  window).
* Free nozzle pressure (six variables) at one scenario: the value against the four-variable value of the
  control; a difference above the resolution (the larger of 0.2 kg CO2e/h and the seed spread) is reported.

    python scripts/analyze_m3_joint.py
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
CASES = ("control", "c0.5", "c1.0", "c1.5")
SEED_SCENARIOS = (2, 12, 17)
ACCURACY_FACTOR = 3.0
RESOLUTION_FLOOR_KG_PER_H = cg.GAIN_PER_SCENARIO_MIN_KG_PER_H


def _load(name: str) -> dict[str, Any]:
    return json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))


def _valid(row: dict[str, Any]) -> bool:
    return "search" in row and row["s2_feasible"] and row["m3_feasible"]


def main() -> int:
    require_clean_tree()
    joint = {case: _load(f"m3_joint_{case}_{SUFFIX}") for case in CASES}
    seeds = _load(f"m3_joint_seeds_{SUFFIX}")
    check = _load(f"m3_check_{SUFFIX}")
    by_case = {c: {r["grid_index"]: r for r in joint[c]["scenarios"]} for c in CASES}

    spread: dict[str, Any] = {}
    for case in CASES:
        per_scenario = {}
        for s in SEED_SCENARIOS:
            values = [by_case[case][s]["m3_value"]]
            values += [
                r["m3_value"]
                for r in seeds["records"]
                if r["case"] == case and r["grid_index"] == s and "m3_value" in r
            ]
            per_scenario[str(s)] = {"values": values, "spread": max(values) - min(values)}
        spread[case] = {
            "per_scenario": per_scenario,
            "largest": max(v["spread"] for v in per_scenario.values()),
        }

    cases_out: dict[str, Any] = {}
    for case in CASES:
        rows = list(by_case[case].values())
        valid = [r for r in rows if _valid(r)]
        gains_co2 = [r["s2_co2_kg_per_h"] - r["m3_co2_kg_per_h"] for r in valid]
        entry: dict[str, Any] = {
            "n_scenarios": len(rows),
            "n_valid_both_feasible": len(valid),
            "n_masked": len(rows) - len(valid),
            "gain_penalised_value_kg_per_h": {
                str(r["grid_index"]): r.get("gain_kg_per_h") for r in rows
            },
            "table_error_max_kg_per_h": max(
                (r["search"]["table_error_max_kg_per_h"] for r in rows if "search" in r),
                default=None,
            ),
            "n_window_widened": sum(
                bool(r["search"]["window_widened"]) for r in rows if "search" in r
            ),
            "window_final_max_kg_per_h": max(
                (r["search"]["window_final_kg_per_h"] for r in rows if "search" in r), default=None
            ),
            "n_polished_median": float(
                np.median([r["search"]["n_polished"] for r in rows if "search" in r])
            ),
            "n_other_allocation_exact_tuple": sum(not r["same_allocation"] for r in valid),
            "n_other_connections_exact_set": sum(not r["same_connections"] for r in valid),
            "n_allocation_gain_material": sum(
                (r["gain_from_other_allocation_kg_per_h"] or 0.0) > RESOLUTION_FLOOR_KG_PER_H
                for r in valid
            ),
            "gain_setpoints_only_median": float(
                np.median(
                    [
                        r["gain_setpoints_for_s2_allocation_kg_per_h"]
                        for r in valid
                        if r["gain_setpoints_for_s2_allocation_kg_per_h"] is not None
                    ]
                )
            ),
            "gain_other_allocation_median": float(
                np.median(
                    [
                        r["gain_from_other_allocation_kg_per_h"]
                        for r in valid
                        if r["gain_from_other_allocation_kg_per_h"] is not None
                    ]
                )
            ),
            "per_scenario": [
                {
                    "grid_index": r["grid_index"],
                    "gain_co2_kg_per_h": (r["s2_co2_kg_per_h"] - r["m3_co2_kg_per_h"])
                    if _valid(r)
                    else None,
                    "gain_relative_to_line_co2": r.get("gain_relative_to_line_co2"),
                    "allocation_gain_material": (
                        r.get("gain_from_other_allocation_kg_per_h") or 0.0
                    )
                    > RESOLUTION_FLOOR_KG_PER_H,
                    "same_allocation": r.get("same_allocation"),
                    "same_connections": r.get("same_connections"),
                    "gain_setpoints": r.get("gain_setpoints_for_s2_allocation_kg_per_h"),
                    "gain_other_allocation": r.get("gain_from_other_allocation_kg_per_h"),
                    "valid": _valid(r),
                }
                for r in rows
            ],
        }
        if case == "control":
            entry["verdict"] = cg.coupling_gain_verdict(
                gains_co2,
                spread["control"]["largest"],
                [r["s2_co2_lines_kg_per_h"] for r in valid],
            )
        else:
            diffs = []
            for s, r in by_case[case].items():
                c0 = by_case["control"].get(s)
                if _valid(r) and c0 is not None and _valid(c0):
                    diffs.append(
                        (r["s2_co2_kg_per_h"] - r["m3_co2_kg_per_h"])
                        - (c0["s2_co2_kg_per_h"] - c0["m3_co2_kg_per_h"])
                    )
            entry["additional_effect_of_the_limit"] = {
                "n": len(diffs),
                "median_kg_per_h": float(np.median(diffs)) if diffs else None,
                "min_kg_per_h": min(diffs) if diffs else None,
                "max_kg_per_h": max(diffs) if diffs else None,
                "n_positive": sum(d > 0.0 for d in diffs),
                "seed_spread_kg_per_h": spread[case]["largest"],
            }
        cases_out[case] = entry

    accuracy = []
    for r in check["results"]:
        if r["kind"] != "direct4":
            continue
        limit = ACCURACY_FACTOR * spread[r["case"]]["largest"]
        excess = r["table_based_value"] - r["direct_value"]
        accuracy.append(
            {
                "case": r["case"],
                "grid_index": r["grid_index"],
                "table_based_value": r["table_based_value"],
                "direct_value": r["direct_value"],
                "excess_of_table_over_direct": excess,
                "allowed": limit,
                "met": bool(excess <= limit),
            }
        )
    six = [r for r in check["results"] if r["kind"] == "six_variables"]
    resolution = max(RESOLUTION_FLOOR_KG_PER_H, spread["control"]["largest"])
    six_out = [
        {
            "grid_index": r["grid_index"],
            "six_variable_value": r["direct_value"],
            "four_variable_value": r["table_based_value"],
            "difference": r["table_based_value"] - r["direct_value"],
            "resolution": resolution,
            "within_resolution": bool(
                abs(r["table_based_value"] - r["direct_value"]) <= resolution
            ),
        }
        for r in six
    ]
    payload = {
        "_provenance": provenance(),
        "test_set": "25 Befund-52 grid points; lines A and B, scaled site; lower bound 1.26 m/s",
        "rule": {
            "main_result": "control (no limit)",
            "gain_median_min_kg_per_h": cg.GAIN_MEDIAN_MIN_KG_PER_H,
            "seed_factor": cg.GAIN_SEED_FACTOR,
            "per_scenario_min_kg_per_h": cg.GAIN_PER_SCENARIO_MIN_KG_PER_H,
            "scenario_share_min": cg.GAIN_SCENARIO_SHARE_MIN,
            "accuracy_factor": ACCURACY_FACTOR,
        },
        "seed_spread": spread,
        "cases": cases_out,
        "accuracy_check": accuracy,
        "accuracy_rule_met_everywhere": bool(all(a["met"] for a in accuracy)),
        "six_variable_check": six_out,
    }
    write_artifact(f"m3_joint_evaluation_{SUFFIX}", payload)
    print(json.dumps(cases_out["control"]["verdict"], indent=1))
    for case in CASES[1:]:
        print(case, json.dumps(cases_out[case]["additional_effect_of_the_limit"]))
    print("accuracy rule met:", payload["accuracy_rule_met_everywhere"], "six variables:", six_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
