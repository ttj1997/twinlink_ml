r"""M1 sensitivity to the lower bound of the gas velocity: 0.80 m/s (config placeholder) against 1.26 m/s.

Pure evaluation of two committed artifacts of ``scripts/run_m1_margin_cost.py`` (single line A, gas
heater, objective ``co2_gas_heater``, quality limit 0.15 kg/kg main case, margin m =
0.005961819615175404 kg/kg, 5 x 5 grid, same optimiser budget): ``m1_margin_cost_l015`` (bound 0.80, the
config default) and ``m1_margin_cost_l015_umin126`` (bound 1.26 m/s = laboratory minimum, run
parameter, config default not changed). Test set (always named): the 25 Befund 52 grid points.

Quantities, per bound, CO2 and kWh sum from the same solutions, over the points feasible in all three
arms (infeasible or solver-failed points are masked and counted, never replaced by a sentinel):

  regret of the fixed point F against the oracle M0, Befund 52 definition ``(F - M0) / M0`` per point
  (median, 95th percentile, maximum) and pooled ``sum(F - M0) / sum(M0)``; absolute median;
  cost of the margin ``(M - M0) / M0`` (median, maximum);
  remaining control benefit ``(F - M) / (F - M0)`` (median of the per-point ratios and pooled);
  descriptive: points where M0 binds the moisture limit, points where the M0 / M oracle sits on the
  lower gas-velocity bound (within 1e-3 m/s), gas velocity of the fixed point, distance of M to the
  validity boundary, solver failures / infeasible points / non-converged flags per arm.

STABILITY RULE, fixed before the run (project lead, 2026-10-05). A statement counts as STABLE iff it
points in the same direction at both bounds AND the figure behind it changes by less than half
(``|x_1.26 - x_0.80| < 0.5 * |x_0.80|``); otherwise the lower bound stands as a condition in every
heading and caption. The statements and their directions:

  S1  "adapting the operating point saves energy": the regret of F against M0 is positive
      (median and pooled, CO2 and kWh);
  S2  "the margin costs energy": the cost of the margin is positive (median, CO2 and kWh);
  S3  "most of the control benefit survives the margin": the remaining fraction is above 0.5
      (median and pooled, CO2 and kWh).

Each figure is judged on its own; a statement is stable iff all its figures are. The overall reading is
"stable" iff S1-S3 are all stable, "partly stable" otherwise with the unstable statements named.

RUN-TIME ESTIMATE of the new run, made before it (``run_m1_margin_cost.py --stage arms
--quality-limit 0.15 --gas-velocity-min 1.26``, 16 logical CPUs, nothing else running): the arms of
``m1_margin_cost_l015`` took, measured, M0 oracle 245.6 cpu-min over 16 workers (about 15 min wall, the
slowest point 11 min), M oracle 337.7 cpu-min (about 21 min wall, slowest point 19 min), fixed-point
search 1116 s (18.6 min); so about 55 min, 45-80 min allowing for another bound changing the
optimiser path. Another job on the machine stretches this proportionally.

    python scripts/analyze_m1_gas_bound_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

ARTIFACTS = {
    "0.80": "m1_margin_cost_l015",
    "1.26": "m1_margin_cost_l015_umin126",
}
CONFIG_DEFAULT_BOUND = (
    0.8  # conf/process/greybox_gab_placeholder.yaml; artifact m1_margin_cost_l015
)
AT_BOUND_TOLERANCE_M_PER_S = 1.0e-3
STABLE_RELATIVE_CHANGE = 0.5
REMAINING_THRESHOLD = 0.5
KEYS = (("co2_kg_per_h", "CO2"), ("kwh_sum_kw", "kWh"))


def _arm_counts(arm: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "n_points": len(arm),
        "n_solve_failed": sum(bool(r.get("solve_failed")) for r in arm),
        "n_infeasible": sum(
            (not r.get("solve_failed")) and not r.get("feasible", False) for r in arm
        ),
        "n_not_converged": sum(not r.get("converged", True) for r in arm),
    }


def _bound(artifact: dict[str, Any]) -> float:
    """Lower gas-velocity bound of a run; the 0.80 artifact predates the field (config default)."""
    return float(artifact.get("gas_velocity_lower_bound_m_per_s", CONFIG_DEFAULT_BOUND))


def _figures(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in artifact["per_point"] if r["feasible_in_all_arms"]]
    out: dict[str, Any] = {"n_points_used": len(rows), "n_points_masked": 25 - len(rows)}
    for key, label in KEYS:
        f = np.array([r[key]["F"] for r in rows])
        m0 = np.array([r[key]["M0"] for r in rows])
        m = np.array([r[key]["M"] for r in rows])
        regret_rel = (f - m0) / np.abs(m0)
        remaining = (f - m) / (f - m0)
        cost_rel = (m - m0) / np.abs(m0)
        out[label] = {
            "regret_rel_median": float(np.median(regret_rel)),
            "regret_rel_p95": float(np.percentile(regret_rel, 95)),
            "regret_rel_max": float(np.max(regret_rel)),
            "regret_rel_pooled": float(np.sum(f - m0) / np.sum(np.abs(m0))),
            "regret_abs_median": float(np.median(f - m0)),
            "cost_rel_median": float(np.median(cost_rel)),
            "cost_rel_max": float(np.max(cost_rel)),
            "remaining_median": float(np.median(remaining)),
            "remaining_pooled": float(np.sum(f - m) / np.sum(f - m0)),
        }
    bound = _bound(artifact)
    m0_arm, m_arm = artifact["arm_M0_oracle"], artifact["arm_M_oracle"]
    out["descriptive"] = {
        "gas_velocity_lower_bound_m_per_s": bound,
        "n_m0_binding_moisture": sum(bool(r["m0_binding_moisture"]) for r in rows),
        "n_m0_on_lower_gas_bound": sum(
            abs(r["gas_velocity_m_per_s"] - bound) < AT_BOUND_TOLERANCE_M_PER_S
            for r in m0_arm
            if not r.get("solve_failed")
        ),
        "n_m_on_lower_gas_bound": sum(
            abs(r["gas_velocity_m_per_s"] - bound) < AT_BOUND_TOLERANCE_M_PER_S
            for r in m_arm
            if not r.get("solve_failed")
        ),
        "fixed_point": {
            k: artifact["arm_F_fixed_point"][k]
            for k in ("inlet_temperature_celsius", "gas_velocity_m_per_s", "nozzle_pressure_bar")
        },
        "m0_gas_velocity_min_max": [
            min(r["gas_velocity_m_per_s"] for r in m0_arm),
            max(r["gas_velocity_m_per_s"] for r in m0_arm),
        ],
        "distance_of_m_to_validity_boundary_at_binding_kg_per_kg": artifact["summary"]
        .get("distance_to_validity_boundary_at_m0_binding_points", {})
        .get("m_arm_median_kg_per_kg"),
        "arm_counts": {
            "F": _arm_counts(artifact["arm_F_fixed_point"]["per_grid_point"]),
            "M0": _arm_counts(m0_arm),
            "M": _arm_counts(m_arm),
        },
        "fixed_point_feasible_everywhere": artifact["fixed_point_feasible_everywhere"],
    }
    return out


def _judge(a: float, b: float, kind: str) -> dict[str, Any]:
    """Direction at both bounds and relative change of figure ``b`` against ``a``."""
    reference = REMAINING_THRESHOLD if kind == "remaining" else 0.0
    same_direction = (a - reference > 0.0) == (b - reference > 0.0)
    change = abs(b - a) / abs(a) if a != 0.0 else float("inf")
    return {
        "value_0.80": a,
        "value_1.26": b,
        "relative_change": change,
        "same_direction": bool(same_direction),
        "stable": bool(same_direction and change < STABLE_RELATIVE_CHANGE),
    }


def main() -> int:
    require_clean_tree()
    loaded = {
        bound: json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))
        for bound, name in ARTIFACTS.items()
    }
    assert _bound(loaded["0.80"]) == 0.8
    assert _bound(loaded["1.26"]) == 1.26
    assert loaded["1.26"]["gas_velocity_lower_bound_is_run_parameter"] is True
    assert loaded["1.26"]["quality_limit_kg_per_kg"] == loaded["0.80"]["quality_limit_kg_per_kg"]
    assert loaded["1.26"]["margin_kg_per_kg"] == loaded["0.80"]["margin_kg_per_kg"]
    figures = {bound: _figures(a) for bound, a in loaded.items()}

    statements = {
        "S1_adapting_the_operating_point_saves_energy": [
            (label, name, "regret")
            for label in ("CO2", "kWh")
            for name in ("regret_rel_median", "regret_rel_pooled")
        ],
        "S2_the_margin_costs_energy": [
            (label, "cost_rel_median", "cost") for label in ("CO2", "kWh")
        ],
        "S3_most_of_the_control_benefit_survives_the_margin": [
            (label, name, "remaining")
            for label in ("CO2", "kWh")
            for name in ("remaining_median", "remaining_pooled")
        ],
    }
    verdicts: dict[str, Any] = {}
    for statement, items in statements.items():
        judged = {
            f"{label}:{name}": _judge(
                figures["0.80"][label][name], figures["1.26"][label][name], kind
            )
            for label, name, kind in items
        }
        verdicts[statement] = {
            "figures": judged,
            "stable": all(j["stable"] for j in judged.values()),
        }
    unstable = [s for s, v in verdicts.items() if not v["stable"]]
    payload = {
        "_provenance": provenance(),
        "test_set": "25 Befund-52 grid points; line A, gas heater, quality limit 0.15, margin m",
        "source_artifacts": ARTIFACTS,
        "stability_rule": {
            "same_direction": True,
            "relative_change_below": STABLE_RELATIVE_CHANGE,
            "remaining_reference": REMAINING_THRESHOLD,
        },
        "figures": figures,
        "statements": verdicts,
        "overall": "stable" if not unstable else "partly stable",
        "unstable_statements": unstable,
    }
    write_artifact("m1_gas_bound_sensitivity", payload)
    print(json.dumps({k: figures[k] for k in figures}, indent=1)[:6000])
    print(json.dumps({s: v["stable"] for s, v in verdicts.items()}, indent=1), payload["overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
