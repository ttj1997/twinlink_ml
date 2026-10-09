r"""Re-fixed coupling test M4-R after the corrections A1 and K1 (Befund 79). FIXED BEFORE ANY COMPUTATION.

Released by the project lead on 2026-10-09 (draft ``docs/14_draft_m4r_coupling_test.md``, open points decided:
name M4-R; five seeds of the full nested search; 20 K first, then 10 K in the same process; the order variants
searched separately). This file and the section "Re-fixed coupling test M4-R" of
``docs/09_entwurf_modellbasierte_optimierung.md`` are committed before any scenario is computed. Nothing of it
has been computed at the time of the commit.

Why: Befund 77 (M4-C) confirmed H1, but Befund 78 showed that its solutions rest on the warm-end check of
``dT_min`` (A1) and on the solver bound ``T_amb + 10 K`` (K1); both are corrected (Befund 79). The candidate list
of M4-C was taken from optima of the faulty model and is invalid: M3' is searched over ALL 441 interconnections.

Hypothesis H1 (wording unchanged): at warm and humid ambient air the joint optimum of the two lines (M3') saves
more than 1.0 kg CO2e/h of the site against "each line on its own, then the best interconnection" (S2), in the
great majority of the scenarios of group W.

Scenarios (as M4-C, none among the 25 Befund 52 points): W (decision) T_amb 30.0 / 32.5 degC x phi 0.60 / 0.72 /
0.82; N (negative control, no rule) (32.5; 0.30), (30.0; 0.35); H (descriptive) (20.0; 0.82), (25.0; 0.82).

Settings: lines A (1.5 kg/h) and B (2.0 kg/h), scaled site (k = 271.797), Q1 and S1 of the partner data case,
``T_kalt`` 29.1 degC, ``eta_H`` 0.80, ``H_u`` 45 MJ/kg, quality limit 0.15, margin m = 0.005961819615175404 kg/kg
for both lines, nozzle pressure 1.0 bar, lower gas-velocity bound 1.26 m/s, inlet box 60-100 degC; corrected model
(``approach_check="along_exchanger"``, solver with the fallback bracket); ``dT_min`` = 20 K (main case), then
10 K (sensitivity) in the same process.

No exchanger limit. ``nested_search`` would treat the two orders of the stages Q1 -> SA, Q1 -> SB as the same
problem when ``shared_cap_kw`` is None (finding N1, Befund 79 f); to search them separately without changing
``src/``, the run uses ``shared_cap_kw`` = 1e12 kW, a limit that cannot bind (the uptake of the lines is below
1e3 kW): every value equals the one without a limit (as for the control of Befund 71). This is CHECKED: every best
S2 and every M3' solution is re-evaluated with ``shared_cap_kw`` = None and must give the same value (``==``).

Method:
  S2   for each line and DE seed 0-4 the optimum of the line on its own (line objective of M1/M2, three
       variables, box 60-100 degC, u >= 1.26 m/s, Befund 52 budget; network-independent, shared by both cases);
       S2 of a pair = the best of the 441 interconnections at the set points of the pair (1.0 bar, true solves,
       the cascade of the case); best S2 = minimum over the 25 pairs.
  M3'  the full nested search of ``joint_search.nested_search`` (scheme C, every rule unchanged): response tables
       of both lines per scenario (inlet 60-100 degC in steps of 2.5 K, gas velocity 1.26-2.5 m/s, 9 nodes,
       1.0 bar, corrected solver); for each of the 441 interconnections DE + Nelder-Mead on the tables (Befund
       52 budget); polishing with true solves within the window 5 kg CO2e/h, error rule, duplicate rule, the S2
       interconnection always polished; five DE seeds 0-4 of the whole procedure.
  G    ``G_s = value(best S2) - value(M3'(seed s))``; ``G = min_s G_s`` (the weakest seed).

CRITERION K_R (main, ``dT_min`` 20 K, box 60-100 degC): G >= 1.0 kg CO2e/h in at least 5 of the 6 scenarios of W,
with all solutions (best S2, M3' of every seed) feasible. If K_R is missed: H1 is NOT confirmed in the corrected
model; the gains of Befunde 73, 76, 77 do not survive the corrections; no recomputation and no change of a
threshold afterwards.

Reported without a decision rule: the 10 K case evaluated the same way (the count of K_R reported, no verdict);
G from the BEST of the five seeds per scenario (``G_best = value(best S2) - min_s value(M3'(seed s))``, project
lead); N (expectation G about zero, stated, not tested); H; the check of the tables (direct DE on true solves,
four variables, seed 0, for the interconnection of the best M3' at (30.0; 0.82) and (32.5; 0.82), 20 K only);
for the best S2 and M3': smallest approach and its position per stage, condensate, solver fallback used, box and
gas-velocity bounds, product moisture.

Run-time estimate (before the run, from ``probe_cascade_runtime_2``): about 4 h wall per case on 16 workers, about
10 h in total (range 7-14 h). Long run: started independently of the session, log file, ``--resume``.

    PYTHONPATH=src python scripts/run_m4r_test.py --dt 20 [--resume]
    PYTHONPATH=src python scripts/run_m4r_test.py --dt 10 --resume
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_m1_margin_cost as m1  # noqa: E402
import run_m3_joint as m3  # noqa: E402
import run_m3x_confirm as c  # noqa: E402
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
from twinlink_ml.evaluation.line_tables import LineTable, build_line_table  # noqa: E402
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec  # noqa: E402
from twinlink_ml.heatnetwork.balance import evaluate_cascade  # noqa: E402
from twinlink_ml.heatnetwork.routing import FixedRouting  # noqa: E402
from twinlink_ml.simulation import NumpyBackend  # noqa: E402
from twinlink_ml.simulation.oracle_search import evaluate_point, run_de_nm  # noqa: E402
from twinlink_ml.simulation.thermo import absolute_humidity_kg_per_kg  # noqa: E402

GROUPS: dict[str, list[tuple[float, float]]] = {
    "W": [(30.0, 0.60), (30.0, 0.72), (30.0, 0.82), (32.5, 0.60), (32.5, 0.72), (32.5, 0.82)],
    "N": [(32.5, 0.30), (30.0, 0.35)],
    "H": [(20.0, 0.82), (25.0, 0.82)],
}
T_MAX = 100.0
T_GRID = np.arange(60.0, T_MAX + 1e-9, 2.5)
U_GRID = np.linspace(m3.BOUND, 2.5, 9)
SEEDS = (0, 1, 2, 3, 4)
DT_CASES = (20.0, 10.0)
UNBINDING_CAP_KW = 1.0e12
GAIN_MIN_KG_PER_H = 1.0
MIN_SCENARIOS_OF_W = 5
CHECK_SCENARIOS = ((30.0, 0.82), (32.5, 0.82))
SCRATCH = RUNS_DIR / "_m4r_test_umin126.jsonl"
BOUNDS4 = [(60.0, T_MAX), (m3.BOUND, 2.5), (60.0, T_MAX), (m3.BOUND, 2.5)]
_STATE: dict[str, Any] = {}


def scenarios() -> list[dict[str, Any]]:
    out = []
    for group, points in GROUPS.items():
        for t_amb, phi in points:
            out.append({"id": len(out), "group": group, "t_amb": t_amb, "phi": phi})
    return out


def _site(k: float, dt: float, cap: float | None = UNBINDING_CAP_KW) -> sn.SiteSpec:
    return sn.SiteSpec(k, dt_min_k=dt, shared_cap_kw=cap)


def _spec(k: float, dt: float, cap: float | None = UNBINDING_CAP_KW) -> SiteOracleSpec:
    return SiteOracleSpec(c.THRESHOLD, _site(k, dt, cap))


def _init(configs: dict[str, Any]) -> None:
    c._init(configs)
    _STATE["configs"] = configs
    _STATE["allocations"] = sn.all_allocations(2, order_variants=True)


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["kind"],
        row.get("dt"),
        row["scenario"],
        row.get("seed"),
        row.get("line"),
        row.get("pair"),
    )


def _table_task(task: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    table = build_line_table(
        task["line"],
        _STATE["configs"][task["line"]],
        t_amb=task["t_amb"],
        phi_amb=task["phi"],
        t_in_grid=T_GRID,
        u_grid=U_GRID,
        p_nozzle=1.0,
    )
    return {**task, "table": table.to_dict(), "wall_s": time.perf_counter() - start}


def _s2_task(task: dict[str, Any]) -> dict[str, Any]:
    spec = _spec(task["k"], task["dt"])
    ev = js.evaluate_true(
        task["x"], _STATE["allocations"], _STATE["configs"], task["t_amb"], task["phi"], spec
    )
    return {**{k_: task[k_] for k_ in ("kind", "dt", "scenario", "pair", "x")}, "evaluation": ev}


def _joint_task(task: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    spec = _spec(task["k"], task["dt"])
    record = js.nested_search(
        allocations=_STATE["allocations"],
        table_a=LineTable.from_dict(task["table_a"]),
        table_b=LineTable.from_dict(task["table_b"]),
        configs=_STATE["configs"],
        t_amb=task["t_amb"],
        phi_amb=task["phi"],
        spec=spec,
        bounds=BOUNDS4,
        seed=task["seed"],
        s2_allocation=tuple(tuple(p) for p in task["s2_allocation"]),
    )
    for key in ("table_values", "table_optima"):
        record.pop(key, None)  # 441 entries each; the polished entries carry what is needed
    return {
        **{k_: task[k_] for k_ in ("kind", "dt", "scenario", "seed")},
        "search": record,
        "wall_s": time.perf_counter() - start,
    }


def _check_task(task: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    spec = _spec(task["k"], task["dt"])
    allocation = tuple(tuple(p) for p in task["allocation"])
    budget = dataclasses.replace(m1._budget(1), seed=0)
    de, refine, _, _ = run_de_nm(
        js._true_value,
        BOUNDS4,
        (allocation, _STATE["configs"], task["t_amb"], task["phi"], spec),
        budget,
        de_workers=1,
    )
    best = refine if refine.fun <= de.fun else de
    final = js.evaluate_true(
        best.x, allocation, _STATE["configs"], task["t_amb"], task["phi"], spec
    )
    return {
        **{k_: task[k_] for k_ in ("kind", "dt", "scenario")},
        "allocation": task["allocation"],
        "direct_value": final["value"],
        "direct_set_points": final.get("set_points"),
        "table_based_value": task["table_based_value"],
        "wall_s": time.perf_counter() - start,
    }


def _dispatch(task: dict[str, Any]) -> dict[str, Any]:
    kind = task["kind"]
    if kind == "table":
        return _table_task(task)
    if kind == "line":
        return c._task({**task, "t_max": T_MAX})
    if kind == "s2":
        return _s2_task(task)
    if kind == "joint":
        return _joint_task(task)
    return _check_task(task)


def _stage(pool: Any, tasks: list[dict[str, Any]], done: dict, handle: Any, label: str) -> None:
    todo = [t for t in tasks if _key(t) not in done]
    print(f"{label}: {len(tasks) - len(todo)} done, {len(todo)} to run", flush=True)
    start = time.perf_counter()
    for n, row in enumerate(pool.imap_unordered(_dispatch, todo), start=1):
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        done[_key(row)] = row
        print(
            f"  {label} [{n}/{len(todo)}] scenario {row['scenario']} seed {row.get('seed')} "
            f"({time.perf_counter() - start:.0f} s)",
            flush=True,
        )


def _diagnostics(
    x: list[float], allocation: list[list[str]], t_amb: float, phi: float, k: float, dt: float
) -> dict:
    """Stage approach, condensate, solver fallback, bounds and moisture of one solution."""
    configs = _STATE["configs"]
    lines, out = [], {}
    for name, (t_in, u) in (("A", x[:2]), ("B", x[2:])):
        solution = evaluate_point(
            t_in, u, 1.0, t_amb, phi, max_iterations=30, tolerance=1e-7, base_config=configs[name]
        )
        lines.append(
            sn.line_state(name, solution, configs[name], t_amb=t_amb, phi_amb=phi, t_in=t_in)
        )
        out[name] = {
            "inlet_celsius": t_in,
            "gas_velocity_m_per_s": u,
            "on_inlet_box": bool(t_in <= 60.0 + 1e-3 or t_in >= T_MAX - 1e-3),
            "on_gas_bound": bool(u <= m3.BOUND + 1e-3),
            "bed_temperature_celsius": solution.bed_temperature_celsius,
            "fallback_used": solution.bed_temperature_fallback_used,
            "moisture_kg_per_kg": solution.particle_moisture_kg_per_kg,
            "drying_regime": solution.drying_regime,
        }
    network, sources, sinks = sn._network((lines[0], lines[1]), _site(k, dt, None))
    stages = FixedRouting(tuple(tuple(p) for p in allocation)).route(network)
    result = evaluate_cascade(network, stages, sources, sinks)
    out["stages"] = [
        {
            "source": result.stage_source_names[i],
            "sink": result.stage_sink_names[i],
            "duty_kw": float(result.stage_recovered_power_kw[i, 0]),
            "min_approach_k": float(result.stage_min_approach_k[i, 0]),
            "pinch_position": float(result.stage_pinch_position[i, 0]),
        }
        for i in range(len(stages))
    ]
    out["condensate_kg_per_s_scaled"] = dict(
        zip(
            result.source_names,
            [float(v) for v in result.source_condensate_kg_per_s[:, 0]],
            strict=True,
        )
    )
    return out


def _evaluate(done: dict, k: float, dt: float) -> dict[str, Any]:
    backend = NumpyBackend()
    configs = _STATE["configs"]
    allocations = _STATE["allocations"]
    results: dict[str, Any] = {}
    cap_check = 0
    for s in scenarios():
        pairs = [
            r
            for r in done.values()
            if r["kind"] == "s2" and r["dt"] == dt and r["scenario"] == s["id"]
        ]
        best_pair = min(pairs, key=lambda r: r["evaluation"]["value"])
        best_s2 = best_pair["evaluation"]
        joint = {
            r["seed"]: r
            for r in done.values()
            if r["kind"] == "joint" and r["dt"] == dt and r["scenario"] == s["id"]
        }
        per_seed = {}
        for seed, r in sorted(joint.items()):
            m3_ = r["search"]["m3"]
            per_seed[seed] = {
                "value": r["search"]["m3_value"],
                "allocation": m3_["allocation"],
                "set_points": m3_["set_points"],
                "feasible": bool(m3_["feasible_a"] and m3_["feasible_b"]),
                "moisture_a": m3_["moisture_a"],
                "moisture_b": m3_["moisture_b"],
                "window_final_kg_per_h": r["search"]["window_final_kg_per_h"],
                "n_polished": r["search"]["n_polished"],
                "table_error_max_kg_per_h": r["search"]["table_error_max_kg_per_h"],
            }
        # the limit of 1e12 kW must not bind: the same value without a limit
        no_cap = _spec(k, dt, None)
        for value, x, allocation in [
            (best_s2["value"], best_s2["set_points"], best_s2["allocation"]),
            *[(e["value"], e["set_points"], e["allocation"]) for e in per_seed.values()],
        ]:
            check = js.evaluate_true(
                x, tuple(tuple(p) for p in allocation), configs, s["t_amb"], s["phi"], no_cap
            )
            if check["value"] != value:
                raise SystemExit(
                    f"scenario {s['id']}: the limit of 1e12 kW changed a value ({check['value']} != {value})"
                )
            cap_check += 1
        gains = {seed: best_s2["value"] - e["value"] for seed, e in per_seed.items()}
        y = float(
            absolute_humidity_kg_per_kg(
                backend, configs["A"].gas, np.float64(s["t_amb"]), np.float64(s["phi"])
            )
        )
        best_seed = min(per_seed, key=lambda seed: per_seed[seed]["value"])
        results[str(s["id"])] = {
            **s,
            "absolute_humidity_ambient_kg_per_kg": y,
            "best_s2": {
                "value": best_s2["value"],
                "co2": best_s2["co2_kg_per_h"],
                "allocation": best_s2["allocation"],
                "set_points": best_s2["set_points"],
                "feasible": bool(best_s2["feasible_a"] and best_s2["feasible_b"]),
                "pair": best_pair["pair"],
                "diagnostics": _diagnostics(
                    best_s2["set_points"], best_s2["allocation"], s["t_amb"], s["phi"], k, dt
                ),
            },
            "per_seed_m3": per_seed,
            "gain_per_seed": gains,
            "gain_min_over_seeds": min(gains.values()),
            "gain_best_seed": best_s2["value"] - per_seed[best_seed]["value"],
            "gain_median_over_seeds": float(np.median(list(gains.values()))),
            "best_m3_diagnostics": _diagnostics(
                per_seed[best_seed]["set_points"],
                per_seed[best_seed]["allocation"],
                s["t_amb"],
                s["phi"],
                k,
                dt,
            ),
            "feasible_all": bool(
                best_s2["feasible_a"]
                and best_s2["feasible_b"]
                and all(e["feasible"] for e in per_seed.values())
            ),
        }
    w = [r for r in results.values() if r["group"] == "W"]
    n_hold = sum(r["gain_min_over_seeds"] >= GAIN_MIN_KG_PER_H and r["feasible_all"] for r in w)
    return {
        "scenarios": results,
        "n_scenarios_of_W_holding": n_hold,
        "n_scenarios_of_W_holding_best_seed": sum(
            r["gain_best_seed"] >= GAIN_MIN_KG_PER_H and r["feasible_all"] for r in w
        ),
        "criterion_met": bool(n_hold >= MIN_SCENARIOS_OF_W) if dt == 20.0 else None,
        "n_value_checks_without_limit": cap_check,
        "n_allocations": len(allocations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dt", type=float, choices=DT_CASES, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    dt = arguments.dt
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    configs, k, _, _ = m3._setup()
    done: dict[tuple[Any, ...], dict[str, Any]] = {}
    if SCRATCH.exists():
        if not arguments.resume:
            raise SystemExit(f"{SCRATCH} exists; use --resume or remove it")
        for line in SCRATCH.read_text("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[_key(row)] = row
    start = time.perf_counter()
    scen = scenarios()
    with (
        Pool(cpu_count(), initializer=_init, initargs=(configs,)) as pool,
        SCRATCH.open("a", encoding="utf-8") as handle,
    ):
        _init(configs)
        _stage(
            pool,
            [
                {
                    "kind": "table",
                    "scenario": s["id"],
                    "line": line,
                    "t_amb": s["t_amb"],
                    "phi": s["phi"],
                }
                for s in scen
                for line in ("A", "B")
            ],
            done,
            handle,
            "tables",
        )
        _stage(
            pool,
            [
                {
                    "kind": "line",
                    "scenario": s["id"],
                    "line": line,
                    "seed": seed,
                    "t_amb": s["t_amb"],
                    "phi": s["phi"],
                }
                for s in scen
                for line in ("A", "B")
                for seed in SEEDS
            ],
            done,
            handle,
            "line optima (S2)",
        )
        s2_tasks = []
        for s in scen:
            sets = {"A": {}, "B": {}}
            for r in done.values():
                if r["kind"] == "line" and r["scenario"] == s["id"]:
                    sets[r["line"]][r["seed"]] = r["set_points"]
            for i in SEEDS:
                for j in SEEDS:
                    sa, sb = sets["A"][i], sets["B"][j]
                    s2_tasks.append(
                        {
                            "kind": "s2",
                            "dt": dt,
                            "scenario": s["id"],
                            "pair": i * 10 + j,
                            "x": [sa[0], sa[1], sb[0], sb[1]],
                            "t_amb": s["t_amb"],
                            "phi": s["phi"],
                            "k": k,
                        }
                    )
        _stage(pool, s2_tasks, done, handle, f"S2 {dt:g} K")
        tables = {
            (r["scenario"], r["line"]): r["table"] for r in done.values() if r["kind"] == "table"
        }
        joint_tasks = []
        order = sorted(
            scen, key=lambda s: {"N": 0, "W": 1, "H": 2}[s["group"]]
        )  # longest (dry, warm) first
        for s in order:
            pairs = [
                r
                for r in done.values()
                if r["kind"] == "s2" and r["dt"] == dt and r["scenario"] == s["id"]
            ]
            best = min(pairs, key=lambda r: r["evaluation"]["value"])
            for seed in SEEDS:
                joint_tasks.append(
                    {
                        "kind": "joint",
                        "dt": dt,
                        "scenario": s["id"],
                        "seed": seed,
                        "t_amb": s["t_amb"],
                        "phi": s["phi"],
                        "k": k,
                        "table_a": tables[(s["id"], "A")],
                        "table_b": tables[(s["id"], "B")],
                        "s2_allocation": best["evaluation"]["allocation"],
                    }
                )
        _stage(pool, joint_tasks, done, handle, f"M3' {dt:g} K")
        checks = []
        if dt == 20.0:
            for s in scen:
                if (s["t_amb"], s["phi"]) in CHECK_SCENARIOS:
                    joint = [
                        r
                        for r in done.values()
                        if r["kind"] == "joint" and r["dt"] == dt and r["scenario"] == s["id"]
                    ]
                    best = min(joint, key=lambda r: r["search"]["m3_value"])
                    checks.append(
                        {
                            "kind": "check",
                            "dt": dt,
                            "scenario": s["id"],
                            "t_amb": s["t_amb"],
                            "phi": s["phi"],
                            "k": k,
                            "allocation": best["search"]["m3"]["allocation"],
                            "table_based_value": best["search"]["m3_value"],
                        }
                    )
            _stage(pool, checks, done, handle, "check of the tables")
    evaluation = _evaluate(done, k, dt)
    check_rows = [r for r in done.values() if r["kind"] == "check" and r["dt"] == dt]
    write_artifact(
        f"m4r_test_dt{dt:g}_umin126",
        {
            "_provenance": provenance(_site(k, dt)),
            "case_dt_min_k": dt,
            "criterion": "K_R"
            if dt == 20.0
            else "none (sensitivity; the K_R count is reported without a verdict)",
            "criterion_rule": f"G = min over seeds >= {GAIN_MIN_KG_PER_H} kg/h in at least {MIN_SCENARIOS_OF_W} of 6 W",
            "box_inlet_temperature_celsius": [60.0, T_MAX],
            "gas_velocity_lower_bound_m_per_s": m3.BOUND,
            "unbinding_cap_kw": UNBINDING_CAP_KW,
            "scale_factor_k": k,
            "seeds": list(SEEDS),
            **evaluation,
            "check_of_the_tables": check_rows,
            "wall_s_this_call": time.perf_counter() - start,
        },
    )
    print(
        "K_R count (min over seeds):",
        evaluation["n_scenarios_of_W_holding"],
        "best seed:",
        evaluation["n_scenarios_of_W_holding_best_seed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
