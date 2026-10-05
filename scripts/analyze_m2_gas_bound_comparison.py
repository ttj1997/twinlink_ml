r"""M2: allocation against set-point effect at the two lower bounds of the gas velocity (D16).

Pure evaluation of two committed artifacts of ``scripts/run_m2_site_network.py``:
``m2_site_network_l015_umin126`` (main case, laboratory minimum 1.26 m/s) and
``m2_site_network_l015_umin080`` (comparison to Befund 52 to 69, config default 0.80 m/s). Single line A,
gas heater, quality limit 0.15, margin m = 0.005961819615175404 kg/kg, the 25 Befund 52 ambient grid
points as test set, scaled site with the fixed external source and sink (a scaling computation).

Per bound, the effects of the arms (F, F+R, F best, S, M of the run script) on the CO2 of the site, the
CO2 attributable to the line and the kWh sum, as median over the points that are feasible in all arms
(masked points are counted, never replaced):

  routing effect (rule)   F -> F+R         (all-to-all routing at the fixed set points)
  routing effect (best)   F -> F best      (best of the 9 allocations at the fixed set points)
  set-point effect        F best -> S      (per-scenario optimum ignoring the network)
  joint effect            S -> M           (joint optimum against the site CO2)
  total effect            F -> M

STABILITY RULE across the bounds, fixed before the run (project lead, 2026-10-05, applied here to the
relative effects above): a statement is STABLE iff it points in the same direction at both bounds
AND its figure changes by less than half (``|x_1.26 - x_0.80| < 0.5 * |x_0.80|``); otherwise both
columns are reported and the bound stands in every heading. Direction classes: negative (the arm
lowers the CO2), none (``|x| < 1e-6``), positive. If the figure at 0.80 is "none", the statement is
stable iff the figure at 1.26 is "none" too. The statements: S1 routing effect (best), S2 set-point
effect, S3 joint effect, S4 total effect; each for the site CO2, the line CO2 and the kWh sum. Also
reported, descriptive: the share of the routing effect in the total effect, the number of points where
M is worse than S in the site CO2 (an unconverged DE), the number of points where M sits on the lower
gas bound, the best allocations per arm.

    python scripts/analyze_m2_gas_bound_comparison.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

ARTIFACTS = {"1.26": "m2_site_network_l015_umin126", "0.80": "m2_site_network_l015_umin080"}
NONE_THRESHOLD = 1.0e-6
STABLE_RELATIVE_CHANGE = 0.5
STATEMENTS = {
    "S1_routing_effect_best": "routing_effect_best",
    "S2_setpoint_effect": "setpoint_effect",
    "S3_joint_effect": "joint_effect",
    "S4_total_effect": "total_effect",
}
KEYS = ("site_co2", "line_co2", "site_kwh")


def _direction(x: float) -> str:
    return "none" if abs(x) < NONE_THRESHOLD else ("negative" if x < 0.0 else "positive")


def _judge(x_126: float, x_080: float) -> dict[str, Any]:
    same = _direction(x_126) == _direction(x_080)
    if _direction(x_080) == "none":
        stable = same
        change = 0.0 if same else float("inf")
    else:
        change = abs(x_126 - x_080) / abs(x_080)
        stable = bool(same and change < STABLE_RELATIVE_CHANGE)
    return {
        "value_1.26": x_126,
        "value_0.80": x_080,
        "direction_1.26": _direction(x_126),
        "direction_0.80": _direction(x_080),
        "same_direction": same,
        "relative_change": change,
        "stable": bool(stable),
    }


def _connection_counts(artifact: dict[str, Any], arm: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in artifact["per_point"]:
        if row["feasible_in_all_arms"]:
            counter[json.dumps(row["best_connections"][arm])] += 1
    return dict(counter)


def main() -> int:
    require_clean_tree()
    loaded = {
        bound: json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))
        for bound, name in ARTIFACTS.items()
    }
    for bound, artifact in loaded.items():
        assert artifact["gas_velocity_lower_bound_m_per_s"] == float(bound)
        assert artifact["gas_velocity_lower_bound_is_run_parameter"] is True
        assert artifact["quality_limit_kg_per_kg"] == 0.15
    verdicts: dict[str, Any] = {}
    for statement, effect in STATEMENTS.items():
        figures = {
            key: _judge(
                loaded["1.26"]["summary"][key][effect]["rel_median"],
                loaded["0.80"]["summary"][key][effect]["rel_median"],
            )
            for key in KEYS
        }
        verdicts[statement] = {
            "figures": figures,
            "stable": all(f["stable"] for f in figures.values()),
        }
    unstable = [s for s, v in verdicts.items() if not v["stable"]]
    descriptive: dict[str, Any] = {}
    for bound, artifact in loaded.items():
        summary = artifact["summary"]
        routing = summary["site_co2"]["routing_effect_best"]["abs_median"]
        total = summary["site_co2"]["total_effect"]["abs_median"]
        descriptive[bound] = {
            "n_points": summary["n_points"],
            "n_feasible_in_all_arms": summary["n_feasible_in_all_arms"],
            "n_masked": summary["n_masked"],
            "n_m_worse_than_s": summary["n_m_worse_than_s"],
            "n_m_on_lower_gas_bound": summary["n_m_on_lower_gas_bound"],
            "n_s_on_lower_gas_bound": summary["n_s_on_lower_gas_bound"],
            "n_m_binding_moisture": summary["n_m_binding_moisture"],
            "routing_share_of_total_site_co2_abs_medians": (
                float(routing / total) if total != 0.0 else None
            ),
            "effects_rel_median_site_co2": {
                e: summary["site_co2"][e]["rel_median"] for e in STATEMENTS.values()
            },
            "effects_rel_median_line_co2": {
                e: summary["line_co2"][e]["rel_median"] for e in STATEMENTS.values()
            },
            "effects_abs_median_site_co2_kg_per_h": {
                e: summary["site_co2"][e]["abs_median"] for e in STATEMENTS.values()
            },
            "best_allocations": {
                arm: _connection_counts(artifact, arm) for arm in ("F_best", "S", "M")
            },
            "arm_failures": {
                arm: {
                    "solve_failed": sum(
                        bool(p.get("solve_failed")) for p in artifact[f"arm_{arm}"]
                    ),
                    "infeasible": sum(
                        (not p.get("solve_failed")) and not p["feasible"]
                        for p in artifact[f"arm_{arm}"]
                    ),
                }
                for arm in ("F", "S", "M")
            },
        }
    payload = {
        "_provenance": provenance(),
        "test_set": "25 Befund-52 grid points; line A, gas heater, limit 0.15, margin m, scaled site",
        "source_artifacts": ARTIFACTS,
        "stability_rule": {
            "same_direction": True,
            "relative_change_below": STABLE_RELATIVE_CHANGE,
            "none_threshold": NONE_THRESHOLD,
        },
        "statements": verdicts,
        "overall": "stable" if not unstable else "partly stable",
        "unstable_statements": unstable,
        "descriptive": descriptive,
    }
    write_artifact("m2_gas_bound_comparison", payload)
    print(json.dumps({s: v["stable"] for s, v in verdicts.items()}, indent=1), payload["overall"])
    print(json.dumps(descriptive, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
