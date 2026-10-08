r"""Evaluation of the protection of the five large gains (``run_m3x_confirm.py``, artifact ``m3x_confirm_umin126``).

Criteria as fixed in the module docstring of ``run_m3x_confirm.py`` before the run (restated here):

(a) per scenario: the best S2 is the minimum over all 25 pairs of the per-line seeds (A seed i, B seed j) of
    the best allocation at the set points of the pair (true solves, 1.0 bar); M3'(seed) is the minimum over
    the shortlisted allocations of the direct joint search of that seed; ``G_s = value(best S2) -
    value(M3'(s))``; the gain HOLDS iff ``min_s G_s >= 1.0 kg CO2e/h``, is WEAK iff it is at least 0.2, else NOT
    CONFIRMED; overall CONFIRMED iff at least three of the five scenarios hold.
(b) at the best S2, the best M3' and the M3' of every seed: moisture of each line and its distance to the
    tightened limit, the quality limit and the kinetics validity boundary, the bed temperature against 80
    degC, and the set points at the boundaries of the action box.
(c) descriptive: allocation, set points, CO2 split (electricity, fuel of S1 / SA / SB), power of every stage,
    the exhaust of both lines, for the best S2 and the best M3' (re-evaluated with true solves).

    python scripts/analyze_m3x_confirm.py [--input NAME --output NAME [--t-max 100]]

Defaults: the first run (artifact ``m3x_confirm_umin126``, evaluation ``m3x_confirm_evaluation_umin126``).
For the sensitivity run (``--t-max 100``) the pre-fixed criterion is added to the output: the gain is NOT
bound to the box limit iff the minimum over the seeds of G exceeds 1 kg CO2e/h in at least four of the five
scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_m3_joint as m3  # noqa: E402
import run_m3x_confirm as c  # noqa: E402
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact  # noqa: E402

from twinlink_ml.evaluation import joint_search as js  # noqa: E402
from twinlink_ml.evaluation import site_network as sn  # noqa: E402
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec  # noqa: E402
from twinlink_ml.heatnetwork.balance import evaluate_cascade  # noqa: E402
from twinlink_ml.heatnetwork.routing import FixedRouting  # noqa: E402
from twinlink_ml.simulation.operating_point import BED_TEMPERATURE_LIMIT_C  # noqa: E402

QUALITY_LIMIT = m3.QUALITY_LIMIT
CRITICAL_MOISTURE = 0.20
T_BOUNDS = (60.0, 120.0)
U_BOUNDS = (m3.BOUND, 2.5)
P_BOUNDS = (1.0, 4.0)
T_TOL, U_TOL, P_TOL = 0.05, 1.0e-3, 1.0e-3


def _boundary_flags(x: list[float], p: tuple[float, float] = (1.0, 1.0)) -> dict[str, list[str]]:
    flags: dict[str, list[str]] = {"A": [], "B": []}
    for name, (t, u), pn in (("A", (x[0], x[1]), p[0]), ("B", (x[2], x[3]), p[1])):
        if abs(t - T_BOUNDS[0]) < T_TOL:
            flags[name].append("inlet temperature at the lower bound 60")
        if abs(t - T_BOUNDS[1]) < T_TOL:
            flags[name].append(f"inlet temperature at the upper bound {T_BOUNDS[1]:g}")
        if abs(u - U_BOUNDS[0]) < U_TOL:
            flags[name].append("gas velocity at the lower bound 1.26")
        if abs(u - U_BOUNDS[1]) < U_TOL:
            flags[name].append("gas velocity at the upper bound 2.5")
        if abs(pn - P_BOUNDS[0]) < P_TOL:
            flags[name].append("nozzle pressure at the lower bound 1.0 (fixed, D6)")
    return flags


def _describe(
    x: list[float],
    allocation: Any,
    configs: dict[str, Any],
    t_amb: float,
    phi: float,
    k: float,
) -> dict[str, Any]:
    """Re-evaluation with true solves: limits, flags, CO2 split, stage powers, exhausts."""
    spec = SiteOracleSpec(m3.THRESHOLD, js.site_spec_with_cap(sn.SiteSpec(k), None))
    alloc = tuple(tuple(c) for c in allocation)
    from twinlink_ml.simulation.operating_point import check_constraints
    from twinlink_ml.simulation.oracle_search import evaluate_point

    lines = []
    lines_out: dict[str, Any] = {}
    for name, (t_in, u) in (("A", (x[0], x[1])), ("B", (x[2], x[3]))):
        sol = evaluate_point(
            t_in, u, 1.0, t_amb, phi, max_iterations=30, tolerance=1e-7, base_config=configs[name]
        )
        check = check_constraints(
            sol, critical_moisture_kg_per_kg=m3.THRESHOLD, gas_velocity_m_per_s=u
        )
        lines.append(sn.line_state(name, sol, configs[name], t_amb=t_amb, phi_amb=phi, t_in=t_in))
        moisture = sol.particle_moisture_kg_per_kg
        lines_out[name] = {
            "inlet_temperature_celsius": t_in,
            "gas_velocity_m_per_s": u,
            "moisture_kg_per_kg": moisture,
            "distance_to_tightened_limit_kg_per_kg": m3.THRESHOLD - moisture,
            "distance_to_quality_limit_kg_per_kg": QUALITY_LIMIT - moisture,
            "distance_to_kinetics_validity_boundary_kg_per_kg": CRITICAL_MOISTURE - moisture,
            "bed_temperature_celsius": sol.bed_temperature_celsius,
            "bed_temperature_margin_to_80_k": BED_TEMPERATURE_LIMIT_C - sol.bed_temperature_celsius,
            "outlet_temperature_celsius": lines[-1].outlet_temperature_celsius,
            "outlet_humidity_kg_per_kg": lines[-1].outlet_humidity_kg_per_kg,
            "dew_point_celsius": sol.dew_point_celsius,
            "dry_air_flow_lab_kg_per_s": lines[-1].dry_air_mass_flow_kg_per_s,
            "electric_kw_lab": lines[-1].electric_kw,
            "feasible": check.feasible,
        }
    network, sources, sinks = sn._network(tuple(lines), spec.site)
    stages = FixedRouting(alloc).route(network)
    cascade = evaluate_cascade(network, stages, sources, sinks)
    h_u = spec.site.fuel_lower_heating_value_j_per_kg
    fuel_kw = {
        name: float(cascade.fuel_mass_flow_kg_per_s[i, 0] * h_u / 1000.0)
        for i, name in enumerate(cascade.sink_names)
    }
    electric_kw = float(k * sum(line.electric_kw for line in lines))
    co2 = {
        "electricity": 0.366 * electric_kw,
        **{f"fuel_{name}": 0.201 * kw for name, kw in fuel_kw.items()},
    }
    return {
        "set_points": x,
        "allocation": [list(c) for c in alloc],
        "box_boundaries": _boundary_flags(x),
        "lines": lines_out,
        "stage_power_kw": {
            f"{s}->{d}": float(cascade.stage_recovered_power_kw[i, 0])
            for i, (s, d) in enumerate(
                zip(cascade.stage_source_names, cascade.stage_sink_names, strict=False)
            )
        },
        "fuel_kw_by_sink": fuel_kw,
        "electric_kw_scaled": electric_kw,
        "co2_kg_per_h_by_part": co2,
        "co2_kg_per_h_total": float(sum(co2.values())),
    }


def main() -> int:
    global T_BOUNDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=f"m3x_confirm_{c.SUFFIX}")
    parser.add_argument("--output", default=f"m3x_confirm_evaluation_{c.SUFFIX}")
    parser.add_argument("--t-max", type=float, default=120.0)
    arguments = parser.parse_args()
    T_BOUNDS = (60.0, arguments.t_max)
    require_clean_tree()
    warnings.filterwarnings("ignore")
    art = json.loads((RUNS_DIR / f"{arguments.input}.json").read_text("utf-8"))
    configs, k, grid, _ = m3._setup()
    allocations = sn.all_allocations(2, order_variants=True)
    records = art["records"]
    out: dict[str, Any] = {}
    for s in art["scenarios"]:
        t_amb, phi = grid[s]
        spec = SiteOracleSpec(m3.THRESHOLD, js.site_spec_with_cap(sn.SiteSpec(k), None))
        line_sets: dict[str, dict[int, list[float]]] = {"A": {}, "B": {}}
        for r in records:
            if r["kind"] == "line" and r["scenario"] == s:
                line_sets[r["line"]][r["seed"]] = r["set_points"]
        s2_pairs = []
        for i, sa in line_sets["A"].items():
            for j, sb in line_sets["B"].items():
                x = [sa[0], sa[1], sb[0], sb[1]]
                ev = js.evaluate_true(x, allocations, configs, t_amb, phi, spec)
                s2_pairs.append(
                    {
                        "seed_a": i,
                        "seed_b": j,
                        "value": ev["value"],
                        "co2": ev["co2_kg_per_h"],
                        "allocation": ev["allocation"],
                        "set_points": x,
                        "feasible": bool(ev["feasible_a"] and ev["feasible_b"]),
                    }
                )
        best_s2 = min(s2_pairs, key=lambda p: p["value"])
        joint = [r for r in records if r["kind"] == "joint" and r["scenario"] == s]
        per_seed = {}
        for seed in art["seeds"]:
            mine = [r for r in joint if r["seed"] == seed]
            best = min(mine, key=lambda r: r["value"])
            per_seed[seed] = {
                "value": best["value"],
                "co2": best["detail"]["co2_kg_per_h"],
                "allocation": best["allocation"],
                "set_points": best["detail"]["set_points"],
                "feasible": bool(best["detail"]["feasible_a"] and best["detail"]["feasible_b"]),
                "values_by_allocation": [round(r["value"], 4) for r in mine],
                "box_boundaries": _boundary_flags(best["detail"]["set_points"]),
                "moisture_a": best["detail"]["moisture_a"],
                "moisture_b": best["detail"]["moisture_b"],
            }
        gains = {seed: best_s2["value"] - v["value"] for seed, v in per_seed.items()}
        worst = min(gains.values())
        verdict = (
            "confirmed"
            if worst >= c.CONFIRMED_MIN_KG_PER_H
            else ("weak" if worst >= c.WEAK_MIN_KG_PER_H else "not confirmed")
        )
        best_seed = min(per_seed, key=lambda sd: per_seed[sd]["value"])
        out[str(s)] = {
            "ambient": [t_amb, phi],
            "best_s2": best_s2,
            "s2_pairs_value_min_median_max": [
                min(p["value"] for p in s2_pairs),
                float(np.median([p["value"] for p in s2_pairs])),
                max(p["value"] for p in s2_pairs),
            ],
            "per_seed_m3": per_seed,
            "gain_per_seed": gains,
            "gain_min_over_seeds": worst,
            "gain_max_over_seeds": max(gains.values()),
            "verdict": verdict,
            "description_best_s2": _describe(
                best_s2["set_points"], best_s2["allocation"], configs, t_amb, phi, k
            ),
            "description_best_m3": _describe(
                per_seed[best_seed]["set_points"],
                per_seed[best_seed]["allocation"],
                configs,
                t_amb,
                phi,
                k,
            ),
        }
    n_confirmed = sum(v["verdict"] == "confirmed" for v in out.values())
    payload = {
        "_provenance": provenance(),
        "criteria": {
            "confirmed_min_kg_per_h": c.CONFIRMED_MIN_KG_PER_H,
            "weak_min_kg_per_h": c.WEAK_MIN_KG_PER_H,
            "overall_min_confirmed_scenarios": c.OVERALL_MIN_CONFIRMED,
        },
        "scenarios": out,
        "n_confirmed": n_confirmed,
        "overall_confirmed": bool(n_confirmed >= c.OVERALL_MIN_CONFIRMED),
        "inlet_temperature_upper_bound_celsius": arguments.t_max,
        "sensitivity_criterion": {
            "rule": "gain not bound to the box limit iff min-over-seeds G >= 1 kg/h in at least 4 of 5",
            "n_scenarios_above_1": n_confirmed,
            "not_bound_to_box_limit": bool(n_confirmed >= 4),
        },
    }
    write_artifact(arguments.output, payload)
    for s, v in out.items():
        print(s, v["verdict"], {sd: round(g, 3) for sd, g in v["gain_per_seed"].items()})
    print("overall confirmed:", payload["overall_confirmed"], n_confirmed, "of", len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
