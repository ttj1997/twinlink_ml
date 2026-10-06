r"""M3' joint stage at the lower gas-velocity bound 1.26 m/s: two lines, one shared limited exchanger.

Released by the project lead on 2026-10-06 (scheme C, lower bound 1.26 m/s only; 0.80 m/s only after
consultation). Lines A (spray 1.5 kg/h) and B (2.0 kg/h, LB3), scaled site of M2 (fixed external source Q1
and sink S1, condensing cascade, ``T_kalt`` 29.1 degC, ``dT_min`` 20 K, ``k`` = 271.797, a scaling
computation), quality limit 0.15, margin m = 0.005961819615175404 kg/kg (both lines, D8), nozzle pressure
1.0 bar (D6), 25 Befund 52 ambient points (test set), gas heaters, CO2 per carrier with the kWh sum from the
same solution. All rules of the nested search are in ``twinlink_ml.evaluation.joint_search`` and those of
the verdict in ``twinlink_ml.evaluation.coupling_gain``, both committed before any run.

Stages (one call each; the artifact of a stage is committed before the next stage starts, the tree has to
be clean):

  tables   response tables of the two lines at every scenario: 25 inlet temperatures (60-120 degC, step
           2.5) x 9 gas velocities (1.26-2.5 m/s), nozzle pressure 1.0 bar          -> m3_line_tables_umin126
  lineb    the optimum of line B on its own (as the arm M of M2: three variables, the line objective, the
           tightened limit, the Befund 52 budget)                                    -> m3_line_b_optimum_umin126
  joint    --case control | c0.5 | c1.0 | c1.5: per scenario S2 and the nested search M3'
                                                                                      -> m3_joint_<case>_umin126
  seeds    the whole procedure with DE seeds 1 and 2 at the scenarios 2, 12, 17 for every case (seed 0 is
           the joint stage)                                                         -> m3_joint_seeds_umin126
  check    (a) the direct DE (true solves, four variables, seed 0) for the allocation chosen by M3' at the
           same three scenarios and every case, to test the tables; (b) at scenario 12 for the control, the
           six-variable run with free nozzle pressures (D6)                           -> m3_check_umin126

Order of the cases: the control first, then c = 0.5, 1.0, 1.5 (project lead). The seed spread is measured
before the evaluation, which is ``analyze_m3_joint.py``.

THE CONTROL IS THE MAIN RESULT (project lead, 2026-10-06, fixed before the run). The two lines share the
limited residual heat of Q1 without any synthetic assumption (stage 0: the site CO2 is not separable also
without a limit, median mixed difference 6.2 kg CO2e/h). The control case is therefore the case WITHOUT a
limit (``SiteSpec.shared_cap_kw`` = None, identical to a limit above the demand, which stage 0 showed bit for
bit). A positive coupling gain G in the control is physical, not an error of the build, and the run does NOT
stop for it. The pre-registered threshold rule (``coupling_gain``) applies to G(control). The additional
effect of the synthetic limit, ``G(c) - G(control)`` for c = 0.5, 1.0, 1.5, is reported descriptively with the
seed spread and without a decision rule. The check of the build is the bit-identical regression test of the
M2 results. Limits: ``Q_cap = c * U`` with ``U`` the power from Q1 into one line in the best single-line
allocation of M2 at 95 degC (as in stage 0), per scenario.

Per scenario and case reported: S2 and M3' (value, CO2, CO2 attributable to the two lines, feasibility),
G in kg CO2e/h and relative to the CO2 of the two lines, whether M3' chooses another allocation than S2,
and the split of G into the gain of the set points for the allocation of S2 and the further gain from
another allocation; the table error at the polished points, window and number of polished allocations.

Run-time estimate (before the run, from measured costs; docs/09 "re-estimate"): tables about 4 min wall,
line B about 13 min, per case 25 scenarios on 16 workers about 20-45 min (table searches about 5-7 min and
the polishing of the allocations of the window about 2-3 min each), seeds about 50 min, check about 30
min; in total about 4-5 h wall. Another job on the machine stretches this.

    PYTHONPATH=<worktree>/src python scripts/run_m3_joint.py --stage tables
    PYTHONPATH=<worktree>/src python scripts/run_m3_joint.py --stage lineb
    PYTHONPATH=<worktree>/src python scripts/run_m3_joint.py --stage joint --case control   # then c0.5 ...
    PYTHONPATH=<worktree>/src python scripts/run_m3_joint.py --stage seeds
    PYTHONPATH=<worktree>/src python scripts/run_m3_joint.py --stage check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_m1_margin_cost as m1  # noqa: E402
from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation import joint_search as js  # noqa: E402
from twinlink_ml.evaluation import site_network as sn  # noqa: E402
from twinlink_ml.evaluation.carrier_oracle import OracleSpec, run_oracle_grid  # noqa: E402
from twinlink_ml.evaluation.line_tables import LineTable, build_line_table  # noqa: E402
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402
from twinlink_ml.simulation.oracle_search import (  # noqa: E402
    action_bounds,
    build_grid,
    evaluate_point,
    run_de_nm,
    with_gas_velocity_minimum,
)

BOUND = 1.26
SUFFIX = "umin126"
QUALITY_LIMIT = 0.15
MARGIN = m1.MARGIN_KG_PER_KG
THRESHOLD = QUALITY_LIMIT - MARGIN
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
T_GRID = np.arange(60.0, 120.0 + 1e-9, 2.5)
U_GRID = np.linspace(BOUND, 2.5, 9)
SPRAY = {"A": 1.5, "B": 2.0}
CASES: dict[str, float | None] = {"control": None, "c0.5": 0.5, "c1.0": 1.0, "c1.5": 1.5}
SEED_SCENARIOS = (2, 12, 17)
EXTRA_SEEDS = (1, 2)
D6_SCENARIO = 12
UPTAKE_T_IN = 95.0
M2_ARTIFACT = "m2_site_network_l015_umin126"

_STATE: dict[str, Any] = {}


def _init(configs: dict[str, Any]) -> None:
    _STATE["configs"] = configs
    _STATE["allocations"] = sn.all_allocations(2, order_variants=True)


def _setup() -> tuple[dict[str, Any], float, list[tuple[float, float]], list[tuple[float, float]]]:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    base = load_process_config(CONFIG_PATH)
    configs = {name: sn.with_spray_rate(base, rate) for name, rate in SPRAY.items()}
    nominal = evaluate_point(
        base.actions.inlet_temperature_celsius.nominal,
        base.actions.gas_velocity_m_per_s.nominal,
        base.actions.nozzle_pressure_bar.nominal,
        base.disturbances.ambient_temperature_celsius.nominal,
        base.disturbances.ambient_relative_humidity.nominal,
        max_iterations=30,
        tolerance=1e-7,
        base_config=base,
    )
    k = sn.scale_factor(nominal.dry_air_mass_flow_kg_per_s)
    stored = json.loads((RUNS_DIR / "m0_line_b_network.json").read_text("utf-8"))["scale_factor_k"]
    assert abs(k - stored) < 1e-9 * k
    grid = build_grid(base, 5)
    return configs, k, grid, with_gas_velocity_minimum(action_bounds(base), BOUND)


# ----------------------------------------------------------------------------------------- tables
def _table_task(task: tuple[int, float, float]) -> dict[str, Any]:
    index, t_amb, phi = task
    out: dict[str, Any] = {"grid_index": index, "t_amb": t_amb, "phi": phi}
    for name in ("A", "B"):
        table = build_line_table(
            name,
            _STATE["configs"][name],
            t_amb=t_amb,
            phi_amb=phi,
            t_in_grid=T_GRID,
            u_grid=U_GRID,
            p_nozzle=1.0,
        )
        out[name] = table.to_dict()
    return out


def stage_tables() -> int:
    configs, k, grid, _ = _setup()
    tasks = [(i, t, p) for i, (t, p) in enumerate(grid)]
    start = time.perf_counter()
    with Pool(cpu_count(), initializer=_init, initargs=(configs,)) as pool:
        results = pool.map(_table_task, tasks, chunksize=1)
    failed = {
        name: sum(any(v is None for v in r[name]["values"]) for r in results) for name in ("A", "B")
    }
    write_artifact(
        f"m3_line_tables_{SUFFIX}",
        {
            "_provenance": provenance(),
            "gas_velocity_lower_bound_m_per_s": BOUND,
            "nozzle_pressure_bar": 1.0,
            "scale_factor_k": k,
            "t_in_grid": [float(x) for x in T_GRID],
            "u_grid": [float(x) for x in U_GRID],
            "spray_rates_kg_per_h": SPRAY,
            "wall_s": time.perf_counter() - start,
            "scenarios_with_failed_nodes": failed,
            "scenarios": results,
        },
    )
    print("tables done", failed, f"{time.perf_counter() - start:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------ line B
def stage_lineb() -> int:
    configs, k, grid, bounds = _setup()
    workers = cpu_count()
    budget = m1._budget(workers)
    spec = OracleSpec(THRESHOLD, "co2_gas_heater", m1.GAS_HEATER_EFFICIENCY)
    results = run_oracle_grid(
        configs["B"], grid, bounds, budget, spec, workers=workers, label="line B alone"
    )
    write_artifact(
        f"m3_line_b_optimum_{SUFFIX}",
        {
            "_provenance": provenance(),
            "gas_velocity_lower_bound_m_per_s": BOUND,
            "spray_rate_kg_per_h": SPRAY["B"],
            "quality_threshold_kg_per_kg": THRESHOLD,
            "objective": "co2_gas_heater (line only, no network)",
            "budget": budget.__dict__,
            "config_fingerprint": configs["B"].fingerprint,
            "n_failed": sum(bool(r.get("solve_failed")) for r in results),
            "n_infeasible": sum((not r.get("solve_failed")) and not r["feasible"] for r in results),
            "grid_results": results,
        },
    )
    return 0


# ------------------------------------------------------------------------------------------- joint
def _uptake_one_line(config: Any, k: float, t_amb: float, phi: float) -> float | None:
    try:
        solution = evaluate_point(
            UPTAKE_T_IN,
            BOUND,
            1.0,
            t_amb,
            phi,
            max_iterations=30,
            tolerance=1e-7,
            base_config=config,
        )
    except (RuntimeError, ValueError):
        return None
    line = sn.line_state("A", solution, config, t_amb=t_amb, phi_amb=phi, t_in=UPTAKE_T_IN)
    results, _ = sn.evaluate_site(
        (line,), sn.SiteSpec(k), [(("Q1", "S1"), ("Q1", "SA"), ("LA", "S1"))]
    )
    return results[0].recovered_kw_external_to_lines if results else None


def _joint_task(task: dict[str, Any]) -> dict[str, Any]:
    configs = _STATE["configs"]
    allocations = _STATE["allocations"]
    k = task["k"]
    index, t_amb, phi = task["grid_index"], task["t_amb"], task["phi"]
    factor = task["cap_factor"]
    u_one = _uptake_one_line(configs["A"], k, t_amb, phi)
    cap_kw = None if factor is None else (None if u_one is None else factor * u_one)
    spec = SiteOracleSpec(THRESHOLD, js.site_spec_with_cap(sn.SiteSpec(k), cap_kw))
    set_a, set_b = task["set_a"], task["set_b"]
    s2 = js.s2_evaluation(
        [set_a[0], set_a[1], set_b[0], set_b[1]], configs, t_amb, phi, spec, allocations
    )
    out: dict[str, Any] = {
        "grid_index": index,
        "case": task["case"],
        "seed": task["seed"],
        "t_amb": t_amb,
        "phi": phi,
        "uptake_one_line_kw": u_one,
        "cap_kw": cap_kw,
        "s2": s2,
    }
    if s2["solve_failed"]:
        out["error"] = "S2 could not be evaluated"
        return out
    s2_allocation = tuple(tuple(c) for c in s2["allocation"])
    bounds = [(60.0, 120.0), (BOUND, 2.5), (60.0, 120.0), (BOUND, 2.5)]
    start = time.perf_counter()
    record = js.nested_search(
        allocations=allocations,
        table_a=LineTable.from_dict(task["table_a"]),
        table_b=LineTable.from_dict(task["table_b"]),
        configs=configs,
        t_amb=t_amb,
        phi_amb=phi,
        spec=spec,
        bounds=bounds,
        seed=task["seed"],
        s2_allocation=s2_allocation,
    )
    m3 = record["m3"]
    gain = s2["value"] - record["m3_value"]
    set_value = record["value_best_setpoints_for_s2_allocation"]
    g_set = None if set_value is None else s2["value"] - set_value
    out.update(
        {
            "search": record,
            "wall_s": time.perf_counter() - start,
            "gain_kg_per_h": gain,
            "gain_setpoints_for_s2_allocation_kg_per_h": g_set,
            "gain_from_other_allocation_kg_per_h": None if g_set is None else gain - g_set,
            "gain_relative_to_line_co2": gain / s2["co2_lines_kg_per_h"],
            "same_allocation": [list(c) for c in s2_allocation] == m3["allocation"],
            "same_connections": sorted(s2_allocation) == sorted(tuple(c) for c in m3["allocation"]),
            "s2_value": s2["value"],
            "m3_value": record["m3_value"],
            "s2_co2_kg_per_h": s2["co2_kg_per_h"],
            "m3_co2_kg_per_h": m3["co2_kg_per_h"],
            "s2_co2_lines_kg_per_h": s2["co2_lines_kg_per_h"],
            "m3_co2_lines_kg_per_h": m3["co2_lines_kg_per_h"],
            "s2_feasible": bool(s2["feasible_a"] and s2["feasible_b"]),
            "m3_feasible": bool(m3["feasible_a"] and m3["feasible_b"]),
        }
    )
    return out


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    tables = json.loads((RUNS_DIR / f"m3_line_tables_{SUFFIX}.json").read_text("utf-8"))
    line_b = json.loads((RUNS_DIR / f"m3_line_b_optimum_{SUFFIX}.json").read_text("utf-8"))
    m2 = json.loads((RUNS_DIR / f"{M2_ARTIFACT}.json").read_text("utf-8"))
    return tables, line_b, m2


def _tasks(
    case: str, seed: int, scenarios: tuple[int, ...] | None, k: float
) -> list[dict[str, Any]]:
    tables, line_b, m2 = _load_inputs()
    out = []
    for entry in tables["scenarios"]:
        i = entry["grid_index"]
        if scenarios is not None and i not in scenarios:
            continue
        a = m2["arm_S"][i]
        b = line_b["grid_results"][i]
        out.append(
            {
                "grid_index": i,
                "t_amb": entry["t_amb"],
                "phi": entry["phi"],
                "case": case,
                "seed": seed,
                "cap_factor": CASES[case],
                "k": k,
                "table_a": entry["A"],
                "table_b": entry["B"],
                "set_a": (a["inlet_temperature_celsius"], a["gas_velocity_m_per_s"]),
                "set_b": (b["inlet_temperature_celsius"], b["gas_velocity_m_per_s"]),
            }
        )
    return out


def _run_tasks(
    name: str, tasks: list[dict[str, Any]], configs: dict[str, Any], resume: bool
) -> list[dict[str, Any]]:
    scratch = RUNS_DIR / f"_m3_{name}.jsonl"
    done: dict[Any, dict[str, Any]] = {}
    if resume and scratch.exists():
        for line in scratch.read_text("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["case"], row["grid_index"], row["seed"])] = row
    elif scratch.exists():
        raise SystemExit(f"{scratch} exists; use --resume or remove it")
    todo = [t for t in tasks if (t["case"], t["grid_index"], t["seed"]) not in done]
    print(f"{name}: {len(done)} done, {len(todo)} to run", flush=True)
    start = time.perf_counter()
    with (
        Pool(cpu_count(), initializer=_init, initargs=(configs,)) as pool,
        scratch.open("a", encoding="utf-8") as handle,
    ):
        for n, row in enumerate(pool.imap_unordered(_joint_task, todo), start=1):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done[(row["case"], row["grid_index"], row["seed"])] = row
            print(
                f"  [{n}/{len(todo)}] {row['case']} scenario {row['grid_index']} seed {row['seed']} "
                f"({time.perf_counter() - start:.0f} s)",
                flush=True,
            )
    return sorted(done.values(), key=lambda r: (r["case"], r["seed"], r["grid_index"]))


def stage_joint(case: str, resume: bool) -> int:
    configs, k, _, _ = _setup()
    rows = _run_tasks(f"joint_{case}", _tasks(case, 0, None, k), configs, resume)
    write_artifact(
        f"m3_joint_{case}_{SUFFIX}",
        {
            "_provenance": provenance(),
            "case": case,
            "cap_factor": CASES[case],
            "gas_velocity_lower_bound_m_per_s": BOUND,
            "scale_factor_k": k,
            "quality_threshold_kg_per_kg": THRESHOLD,
            "scenarios": rows,
        },
    )
    return 0


def stage_seeds(resume: bool) -> int:
    configs, k, _, _ = _setup()
    tasks = [
        t for case in CASES for seed in EXTRA_SEEDS for t in _tasks(case, seed, SEED_SCENARIOS, k)
    ]
    rows = _run_tasks("seeds", tasks, configs, resume)
    write_artifact(
        f"m3_joint_seeds_{SUFFIX}",
        {
            "_provenance": provenance(),
            "seed_scenarios": list(SEED_SCENARIOS),
            "extra_seeds": list(EXTRA_SEEDS),
            "records": rows,
        },
    )
    return 0


# ------------------------------------------------------------------------------------------ check
def _direct_task(task: dict[str, Any]) -> dict[str, Any]:
    configs = _STATE["configs"]
    spec = SiteOracleSpec(THRESHOLD, js.site_spec_with_cap(sn.SiteSpec(task["k"]), task["cap_kw"]))
    allocation = tuple(tuple(c) for c in task["allocation"])
    bounds = [(60.0, 120.0), (BOUND, 2.5), (60.0, 120.0), (BOUND, 2.5)]
    budget = m1._budget(1)
    start = time.perf_counter()
    if task["kind"] == "direct4":
        de, refine, _, _ = run_de_nm(
            js._true_value,
            bounds,
            (allocation, configs, task["t_amb"], task["phi"], spec),
            budget,
            de_workers=1,
        )
        best = refine if refine.fun <= de.fun else de
        final = js.evaluate_true(best.x, allocation, configs, task["t_amb"], task["phi"], spec)
    else:  # six variables, free nozzle pressures
        bounds6 = [bounds[0], bounds[1], (1.0, 4.0), bounds[2], bounds[3], (1.0, 4.0)]

        def objective(x: Any) -> float:
            return float(
                js.evaluate_true(
                    [x[0], x[1], x[3], x[4]],
                    allocation,
                    configs,
                    task["t_amb"],
                    task["phi"],
                    spec,
                    p_nozzle=(float(x[2]), float(x[5])),
                )["value"]
            )

        de, refine, _, _ = run_de_nm(objective, bounds6, (), budget, de_workers=1)
        best = refine if refine.fun <= de.fun else de
        final = {"value": float(best.fun), "set_points": [float(v) for v in best.x]}
    return {
        "kind": task["kind"],
        "case": task["case"],
        "grid_index": task["grid_index"],
        "allocation": task["allocation"],
        "direct_value": final["value"],
        "detail": final,
        "wall_s": time.perf_counter() - start,
    }


def stage_check() -> int:
    configs, k, _, _ = _setup()
    tasks = []
    for case in CASES:
        joint = json.loads((RUNS_DIR / f"m3_joint_{case}_{SUFFIX}.json").read_text("utf-8"))
        for row in joint["scenarios"]:
            if row["grid_index"] in SEED_SCENARIOS and "search" in row:
                tasks.append(
                    {
                        "kind": "direct4",
                        "case": case,
                        "grid_index": row["grid_index"],
                        "t_amb": row["t_amb"],
                        "phi": row["phi"],
                        "k": k,
                        "cap_kw": row["cap_kw"],
                        "allocation": row["search"]["m3"]["allocation"],
                        "table_based_value": row["m3_value"],
                    }
                )
            if case == "control" and row["grid_index"] == D6_SCENARIO and "search" in row:
                tasks.append(
                    {
                        "kind": "six_variables",
                        "case": case,
                        "grid_index": row["grid_index"],
                        "t_amb": row["t_amb"],
                        "phi": row["phi"],
                        "k": k,
                        "cap_kw": row["cap_kw"],
                        "allocation": row["search"]["m3"]["allocation"],
                        "table_based_value": row["m3_value"],
                    }
                )
    with Pool(min(cpu_count(), len(tasks)), initializer=_init, initargs=(configs,)) as pool:
        results = pool.map(_direct_task, tasks, chunksize=1)
    for r, t in zip(results, tasks, strict=True):
        r["table_based_value"] = t["table_based_value"]
    write_artifact(
        f"m3_check_{SUFFIX}",
        {"_provenance": provenance(), "results": results},
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage", choices=("tables", "lineb", "joint", "seeds", "check"), required=True
    )
    parser.add_argument("--case", choices=sorted(CASES), default="control")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    if arguments.stage == "tables":
        return stage_tables()
    if arguments.stage == "lineb":
        return stage_lineb()
    if arguments.stage == "joint":
        return stage_joint(arguments.case, arguments.resume)
    if arguments.stage == "seeds":
        return stage_seeds(arguments.resume)
    return stage_check()


if __name__ == "__main__":
    raise SystemExit(main())
