r"""M2: one line plus the fixed external source and sink, scaled -- allocation against set-point effect.

Docs/09 MO2 stage M2, arms F, F+R, S, M, at ONE lower bound of the gas velocity per call (D16:
1.26 m/s main case, 0.80 m/s comparison to Befund 52 to 69; the bound is a run parameter, the config
default is not changed). Single line A (spray 1.5 kg/h), objective CO2 of the SITE per carrier (gas
heater, D5) with the kWh sum from the same solution, quality limit 0.15 kg/kg, margin m =
0.005961819615175404 kg/kg, 5 x 5 ambient grid of Befund 52 (test set), same optimiser budget as
M1 (DE popsize 10, maxiter 50, seed 0, Nelder-Mead, search precision 30 / 1e-7, three variables).

Site (``twinlink_ml.evaluation.site_network``): sources Q1 (external, fixed) and LA (the line's
exhaust, scaled), sinks S1 (external, fixed) and SA (the line's inlet air, scaled), condensing
cascade, ``T_kalt`` 29.1 degC, ``dT_min`` 20 K, ``eta_H`` 0.80, factors 0.366 / 0.201 kg CO2e/kWh.
Scale factor ``k`` = 8 kg/s / dry-air flow of line A at the config nominal point (271.797, the value
of the M0 artifact); a scaling computation, not a plant statement. Allocations (D7): every source
feeds at least one sink, 9 sets.

Arms (all at the same lower bound; solver failures are masked and counted, nothing is replaced):

  F    the fixed operating point of M1 at this bound (artifact ``m1_margin_cost_l015`` or
       ``..._umin126``, arm F; searched in M1 over the same 25 points, feasible at ``L_q``, without
       the network), routing "Fall 1": each source to its own sink (Q1 -> S1, LA -> SA);
  F+R  the same set points, rule-based routing. The rule "highest exhaust temperature to the highest
       sink demand" coincides with Fall 1 for two sources and two sinks (Q1 is the hottest source
       and S1 the largest demand), so it is degenerate here; the rule used is instead "every source
       feeds every sink" (all-to-all, a pre-set simple rule, decision of this stage, flagged in the
       finding). Also reported, as an extra column: the best of the 9 allocations at the F set points
       ("F best");
  S    the per-scenario optimum of the line IGNORING the network (the arm M of M1 at this bound,
       tightened limit ``L_q - m``), best of the 9 allocations at those set points;
  M    the joint optimum: DE + Nelder-Mead on the line's three set points against the site CO2 of the
       best allocation (evaluated inside every objective call), tightened limit ``L_q - m``.

Reported (descriptive, no gate), per scenario and pooled, for the CO2 of the site, the CO2
attributable to the line (its electricity and the fuel of its own sink) and the kWh sum: routing
effect F -> F+R and F -> F best, set-point effect F best -> S, joint effect S -> M, total F -> M;
absolute and relative to F; points where the optimum sits on the lower gas bound; which allocations
are best; recovered heat to the line; points where M is worse than S in the site CO2 (a sign of the
unconverged DE, counted). The DE is not converged by design (as Befund 52).

Deviations from the decided design, flagged: (1) the direct DE is the primary method in this stage
(single line, three variables, the same cost as the M1 oracle); the response tables of D13 are built
at M3 with the check step; (2) the nozzle pressure stays a search variable as in M1 (D6 fixes it for
the two-line stages), so that F and S can be taken from M1 unchanged; the optima of M1 have p in
1.000-1.011 bar.

Penalty scale (found in the first attempt, artifact ``m2_attempt1_penalty_unscaled_l015_umin126``): the
line part of the site objective is ``k`` times the laboratory one, so the constraint penalty and the
solver-failure value of the ordinal DE are multiplied by ``k`` as well
(``site_oracle.penalty_weight``); unscaled, the network credit outweighed the penalty and the joint
optimum sat at the inlet-temperature bound with the tightened limit violated at 11 of 25 points
(1.26 m/s). The run-time estimate below is unchanged by it.

Run-time estimate (before the run, 16 logical CPUs, nothing else running): the M arm of M1 took
337.7 cpu-min over 16 workers at bound 0.80 (about 21 min wall) and 16.5 min at 1.26; the site
objective adds at most about 12 ms (9 cascades, 1.3 ms each, M0) to a 170-210 ms solve, i.e. at most
7 %. So about 17-23 min per bound, the evaluation of F, F+R, S a few seconds, in total 35-46 min for
the two bounds run one after the other, up to 65 min if the machine is shared.

    PYTHONPATH=<worktree>/src python scripts/run_m2_site_network.py --gas-velocity-min 1.26
    PYTHONPATH=<worktree>/src python scripts/run_m2_site_network.py --gas-velocity-min 0.80
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation.site_network import SiteSpec, scale_factor  # noqa: E402
from twinlink_ml.evaluation.site_oracle import (  # noqa: E402
    SiteOracleSpec,
    evaluate_site_point,
    run_site_grid,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402
from twinlink_ml.simulation.operating_point import (  # noqa: E402
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
)
from twinlink_ml.simulation.oracle_search import (  # noqa: E402
    OptimizationBudget,
    action_bounds,
    build_grid,
    with_gas_velocity_minimum,
)

MARGIN_KG_PER_KG = 0.005961819615175404
QUALITY_LIMIT_KG_PER_KG = 0.15
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
M1_ARTIFACTS = {0.80: "m1_margin_cost_l015", 1.26: "m1_margin_cost_l015_umin126"}
M0_ARTIFACT = "m0_line_b_network"
AT_BOUND_TOLERANCE_M_PER_S = 1.0e-3
KEYS = (
    ("site_co2", "co2_kg_per_h"),
    ("line_co2", "co2_lines_kg_per_h"),
    ("site_kwh", "kwh_sum_kw"),
)


def _budget(workers: int) -> OptimizationBudget:
    grid52 = json.loads(
        (RUNS_DIR / "study_operating_point_regret_win_repro.json").read_text("utf-8")
    )
    budget = OptimizationBudget(
        seed=0,
        popsize=10,
        maxiter=50,
        de_tol=1.0e-4,
        search_max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
        search_tolerance=DEFAULT_SEARCH_TOLERANCE,
        workers=workers,
    )
    committed = grid52["budget"]
    for key in (
        "de_tol",
        "maxiter",
        "popsize",
        "search_max_iterations",
        "search_tolerance",
        "seed",
    ):
        assert dataclasses.asdict(budget)[key] == committed[key], key
    return budget


def _point(
    config: Any, spec: SiteOracleSpec, setpoint: dict[str, float], t_amb: float, phi: float
) -> dict[str, Any]:
    return evaluate_site_point(
        config,
        spec,
        t_in=setpoint["inlet_temperature_celsius"],
        u_gas=setpoint["gas_velocity_m_per_s"],
        p_nozzle=setpoint["nozzle_pressure_bar"],
        t_amb=t_amb,
        phi_amb=phi,
    )


def _value(point: dict[str, Any], routing: str, key: str) -> float | None:
    block = point.get(f"routing_{routing}")
    return None if point.get("solve_failed") or block is None else float(block[key])


def _delta(before: float | None, after: float | None) -> dict[str, float] | None:
    if before is None or after is None:
        return None
    return {"abs": after - before, "rel": (after - before) / abs(before)}


def analyse(arms: dict[str, list[dict[str, Any]]], bound: float) -> dict[str, Any]:
    """Per-scenario effects and their summary; infeasible or failed points are masked, counted."""
    f, s, m = arms["F"], arms["S"], arms["M"]
    rows = []
    for i, (pf, ps, pm) in enumerate(zip(f, s, m, strict=True)):
        ok = all(
            (not p.get("solve_failed"))
            and p.get("feasible", False)
            and p["routing_best"] is not None
            for p in (pf, ps, pm)
        )
        row: dict[str, Any] = {
            "grid_index": i,
            "ambient_temperature_celsius": pf["ambient_temperature_celsius"],
            "ambient_relative_humidity": pf["ambient_relative_humidity"],
            "feasible_in_all_arms": ok,
        }
        if ok:
            for name, key in KEYS:
                v = {
                    "F": _value(pf, "fixed", key),
                    "F+R": _value(pf, "all_to_all", key),
                    "F_best": _value(pf, "best", key),
                    "S": _value(ps, "best", key),
                    "M": _value(pm, "best", key),
                }
                row[name] = {
                    **v,
                    "routing_effect_rule": _delta(v["F"], v["F+R"]),
                    "routing_effect_best": _delta(v["F"], v["F_best"]),
                    "setpoint_effect": _delta(v["F_best"], v["S"]),
                    "joint_effect": _delta(v["S"], v["M"]),
                    "total_effect": _delta(v["F"], v["M"]),
                }
            row["best_connections"] = {
                "F_best": pf["routing_best"]["connections"],
                "S": ps["routing_best"]["connections"],
                "M": pm["routing_best"]["connections"],
            }
            row["recovered_kw_lines"] = {
                "F": pf["routing_fixed"]["recovered_kw_lines"],
                "S": ps["routing_best"]["recovered_kw_lines"],
                "M": pm["routing_best"]["recovered_kw_lines"],
            }
            row["setpoints"] = {
                name: [
                    p["inlet_temperature_celsius"],
                    p["gas_velocity_m_per_s"],
                    p["nozzle_pressure_bar"],
                ]
                for name, p in (("F", pf), ("S", ps), ("M", pm))
            }
            row["m_on_lower_gas_bound"] = bool(
                abs(pm["gas_velocity_m_per_s"] - bound) < AT_BOUND_TOLERANCE_M_PER_S
            )
            row["s_on_lower_gas_bound"] = bool(
                abs(ps["gas_velocity_m_per_s"] - bound) < AT_BOUND_TOLERANCE_M_PER_S
            )
            row["m_binding_moisture"] = pm["constraint_check"]["binding_moisture"]
            row["m_worse_than_s_site_co2"] = bool(
                row["site_co2"]["M"] > row["site_co2"]["S"] * (1.0 + 1.0e-9)
            )
        rows.append(row)
    ok_rows = [r for r in rows if r["feasible_in_all_arms"]]
    summary: dict[str, Any] = {
        "n_points": len(rows),
        "n_feasible_in_all_arms": len(ok_rows),
        "n_masked": len(rows) - len(ok_rows),
        "n_m_on_lower_gas_bound": sum(r["m_on_lower_gas_bound"] for r in ok_rows),
        "n_s_on_lower_gas_bound": sum(r["s_on_lower_gas_bound"] for r in ok_rows),
        "n_m_binding_moisture": sum(bool(r["m_binding_moisture"]) for r in ok_rows),
        "n_m_worse_than_s": sum(r["m_worse_than_s_site_co2"] for r in ok_rows),
    }
    for name, _ in KEYS:
        entry: dict[str, Any] = {}
        for effect in (
            "routing_effect_rule",
            "routing_effect_best",
            "setpoint_effect",
            "joint_effect",
            "total_effect",
        ):
            absolute = np.array([r[name][effect]["abs"] for r in ok_rows])
            relative = np.array([r[name][effect]["rel"] for r in ok_rows])
            entry[effect] = {
                "abs_median": float(np.median(absolute)),
                "abs_min": float(np.min(absolute)),
                "abs_max": float(np.max(absolute)),
                "rel_median": float(np.median(relative)),
                "rel_min": float(np.min(relative)),
                "rel_max": float(np.max(relative)),
                "n_negative": int(np.sum(absolute < 0.0)),
            }
        base = np.array([r[name]["F"] for r in ok_rows])
        entry["F_median"] = float(np.median(base))
        summary[name] = entry
    return {"per_point": rows, "summary": summary}


def run(gas_velocity_min: float) -> int:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    config = load_process_config(CONFIG_PATH)
    workers = cpu_count()
    budget = _budget(workers)
    grid = build_grid(config, 5)
    bounds = action_bounds(config)
    config_default_min = bounds[1][0]
    bounds = with_gas_velocity_minimum(bounds, gas_velocity_min)
    key = round(gas_velocity_min, 2)
    m1 = json.loads((RUNS_DIR / f"{M1_ARTIFACTS[key]}.json").read_text("utf-8"))
    assert m1["quality_limit_kg_per_kg"] == QUALITY_LIMIT_KG_PER_KG
    assert m1["margin_kg_per_kg"] == MARGIN_KG_PER_KG
    assert m1["fixed_point_feasible_everywhere"]
    m0 = json.loads((RUNS_DIR / f"{M0_ARTIFACT}.json").read_text("utf-8"))

    from twinlink_ml.simulation.oracle_search import evaluate_point

    nominal = evaluate_point(
        config.actions.inlet_temperature_celsius.nominal,
        config.actions.gas_velocity_m_per_s.nominal,
        config.actions.nozzle_pressure_bar.nominal,
        config.disturbances.ambient_temperature_celsius.nominal,
        config.disturbances.ambient_relative_humidity.nominal,
        max_iterations=30,
        tolerance=1e-7,
        base_config=config,
    )
    k = scale_factor(nominal.dry_air_mass_flow_kg_per_s)
    assert abs(k - m0["scale_factor_k"]) < 1e-9 * k, "scale factor differs from the M0 artifact"
    site = SiteSpec(k)
    spec_free = SiteOracleSpec(QUALITY_LIMIT_KG_PER_KG, site)
    spec_tight = SiteOracleSpec(QUALITY_LIMIT_KG_PER_KG - MARGIN_KG_PER_KG, site)

    fixed = m1["arm_F_fixed_point"]
    f_setpoint = {
        name: fixed[name]
        for name in ("inlet_temperature_celsius", "gas_velocity_m_per_s", "nozzle_pressure_bar")
    }
    f_arm = [_point(config, spec_free, f_setpoint, t, phi) for t, phi in grid]
    s_arm = [
        _point(
            config, spec_tight, r, r["ambient_temperature_celsius"], r["ambient_relative_humidity"]
        )
        for r in m1["arm_M_oracle"]
    ]
    m_arm = run_site_grid(
        config, grid, bounds, budget, spec_tight, workers=workers, label=f"M umin {key:.2f}"
    )
    analysis = analyse({"F": f_arm, "S": s_arm, "M": m_arm}, bounds[1][0])
    tag = f"l015_umin{round(key * 100):03d}"
    write_artifact(
        f"m2_site_network_{tag}",
        {
            "_provenance": provenance(),
            "quality_limit_kg_per_kg": QUALITY_LIMIT_KG_PER_KG,
            "margin_kg_per_kg": MARGIN_KG_PER_KG,
            "gas_velocity_lower_bound_m_per_s": bounds[1][0],
            "gas_velocity_lower_bound_config_default_m_per_s": config_default_min,
            "gas_velocity_lower_bound_is_run_parameter": True,
            "scale_factor_k": k,
            "site": dataclasses.asdict(site),
            "budget": dataclasses.asdict(budget),
            "config_fingerprint": config.fingerprint,
            "source_artifact_m1": M1_ARTIFACTS[key],
            "fixed_point_setpoint": f_setpoint,
            "summary": analysis["summary"],
            "per_point": analysis["per_point"],
            "arm_F": f_arm,
            "arm_S": s_arm,
            "arm_M": m_arm,
        },
    )
    print(json.dumps(analysis["summary"], indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gas-velocity-min",
        type=float,
        required=True,
        choices=sorted(M1_ARTIFACTS),
        help="lower bound of the gas velocity for this run in m/s (D16: 1.26 main, 0.80 comparison)",
    )
    arguments = parser.parse_args()
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT), (
        "twinlink_ml is imported from outside this checkout; set PYTHONPATH=<checkout>/src"
    )
    return run(arguments.gas_velocity_min)


if __name__ == "__main__":
    raise SystemExit(main())
