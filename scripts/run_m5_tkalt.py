r"""Sensitivity of the confirmed gain (Befund 77) to the cold-side temperature ``T_kalt`` of the cascade.

Released by the project lead on 2026-10-09. DESCRIPTIVE, no decision rule. Laboratory box (inlet temperature
60-100 degC), lower gas-velocity bound 1.26 m/s (0.80 m/s is NOT computed), lines A (1.5 kg/h) and B (2.0
kg/h), scaled site (k = 271.797), no exchanger limit, margin m for both lines, nozzle pressure 1.0 bar, and
the method of M4-C (``run_m4_confirm_test.py``, unchanged): S2 = per-line optima for the DE seeds 0-4, best S2
= minimum over the 25 pairs of the best of the 441 interconnections with true solves; M3' = joint DE +
Nelder-Mead of the four set points on true solves for every candidate interconnection and every seed 0-4,
best over the candidates; ``G_s = value(best S2) - value(M3'(seed s))``.

Scenarios (those of M4-C): three of group W, (30.0; 0.60), (30.0; 0.82), (32.5; 0.82), and the control N
(32.5; 0.30). ``T_kalt`` = 25.0, 29.1 and 35.0 degC as a RUN PARAMETER (``SiteSpec.t_kalt_celsius``); the
default (29.1) is not changed.

Reuse (deterministic, documented): the per-line searches do not depend on ``T_kalt`` (the line objective is the
gas-heater CO2 of one line alone), so the set points of the lines are taken from the records of M4-C
(``m4c_confirm_lab_umin126``, same scenarios, seeds, box); the results at ``T_kalt`` = 29.1 degC are those of
M4-C (the S2 evaluation is recomputed at 29.1 as an integrity check of this code path: it must equal the
S2 value of M4-C to 1e-6 kg/h). The candidate list per scenario and ``T_kalt``: the two interconnections of
M4-C (chosen by M3' at the scenarios 18, 19, 22, 23, 24) plus the interconnection of the best S2 AT THIS
``T_kalt``.

Reported per scenario and ``T_kalt`` (descriptive): G (minimum and maximum over the seeds), the dew points of
the exhaust of both lines (at the best M3' and the best S2), the condensate flow per source, and, for the
solution of M3' at 29.1 degC with fixed set points and interconnection, the condensate of each line over a
sweep of ``T_kalt`` from 15 to 45 degC in steps of 1 K: the temperature above which the condensation of the hot
line stops.

Run-time estimate: see the lab-book addendum to Befund 77 (about 3.2 h on 16 workers; 2 values of ``T_kalt`` x
4 scenarios x 3 interconnections x 5 seeds = at most 120 joint searches of about 25 min).

    PYTHONPATH=<worktree>/src python scripts/run_m5_tkalt.py [--resume]
"""

from __future__ import annotations

import argparse
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

import run_m3_joint as m3  # noqa: E402
import run_m3x_confirm as c  # noqa: E402
import run_m4_confirm_test as m4  # noqa: E402
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
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec  # noqa: E402
from twinlink_ml.heatnetwork.balance import evaluate_cascade  # noqa: E402
from twinlink_ml.heatnetwork.routing import FixedRouting  # noqa: E402
from twinlink_ml.simulation import NumpyBackend  # noqa: E402
from twinlink_ml.simulation.oracle_search import evaluate_point, run_de_nm  # noqa: E402
from twinlink_ml.simulation.thermo import dew_point_celsius  # noqa: E402

M4C_ARTIFACT = "m4c_confirm_lab_umin126"
SCENARIO_IDS = (0, 2, 5, 6)  # M4-C ids: (30; 0.60), (30; 0.82), (32.5; 0.82), N (32.5; 0.30)
T_KALT_RUN = (25.0, 35.0)
T_KALT_REFERENCE = 29.1
T_KALT_ALL = (25.0, 29.1, 35.0)
SWEEP = tuple(float(t) for t in range(15, 46))
T_MAX = 100.0
SEEDS = c.SEEDS
SUFFIX = c.SUFFIX
INTEGRITY_TOLERANCE_KG_PER_H = 1.0e-6


def _spec(k: float, t_kalt: float) -> SiteOracleSpec:
    site = sn.SiteSpec(k, t_kalt_celsius=t_kalt)
    return SiteOracleSpec(c.THRESHOLD, js.site_spec_with_cap(site, None))


def _s2_task(task: dict[str, Any]) -> dict[str, Any]:
    ev = js.evaluate_true(
        task["x"],
        c._STATE["allocations"],
        c._STATE["configs"],
        task["t_amb"],
        task["phi"],
        _spec(task["k"], task["t_kalt"]),
    )
    return {
        "kind": "s2",
        "t_kalt": task["t_kalt"],
        "scenario": task["scenario"],
        "seed": task["seed_a"] * 10 + task["seed_b"],
        "value": float(ev["value"]),
        "co2": float(ev["co2_kg_per_h"]),
        "allocation": ev["allocation"],
        "set_points": task["x"],
        "feasible": bool(ev["feasible_a"] and ev["feasible_b"]),
    }


def _joint_task(task: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    import dataclasses

    from twinlink_ml.evaluation import joint_search as js_

    budget = dataclasses.replace(c.m1._budget(1), seed=task["seed"])
    spec = _spec(task["k"], task["t_kalt"])
    allocation = tuple(tuple(a) for a in task["allocation"])
    bounds = [(60.0, T_MAX), (m3.BOUND, 2.5), (60.0, T_MAX), (m3.BOUND, 2.5)]
    de, refine, _, _ = run_de_nm(
        js_._true_value,
        bounds,
        (allocation, c._STATE["configs"], task["t_amb"], task["phi"], spec),
        budget,
        de_workers=1,
    )
    best = refine if refine.fun <= de.fun else de
    final = js_.evaluate_true(
        best.x, allocation, c._STATE["configs"], task["t_amb"], task["phi"], spec
    )
    return {
        "kind": "joint",
        "t_kalt": task["t_kalt"],
        "scenario": task["scenario"],
        "seed": task["seed"],
        "allocation_index": task["allocation_index"],
        "allocation": task["allocation"],
        "value": float(final["value"]),
        "detail": final,
        "wall_s": time.perf_counter() - start,
    }


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["kind"], row["t_kalt"], row["scenario"], row["seed"], row.get("allocation_index"))


def _run(
    pool: Any, tasks: list[dict[str, Any]], function: Any, done: dict, handle: Any, label: str
) -> None:
    todo = [t for t in tasks if _key(t) not in done]
    print(f"{label}: {len(tasks) - len(todo)} done, {len(todo)} to run", flush=True)
    start = time.perf_counter()
    for n, row in enumerate(pool.imap_unordered(function, todo), start=1):
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        done[_key(row)] = row
        print(
            f"  {label} [{n}/{len(todo)}] T_kalt {row['t_kalt']} scenario {row['scenario']} "
            f"({time.perf_counter() - start:.0f} s)",
            flush=True,
        )


def describe(
    x: list[float],
    allocation: Any,
    configs: dict[str, Any],
    t_amb: float,
    phi: float,
    k: float,
    t_kalt: float,
) -> dict[str, Any]:
    """True re-evaluation: exhaust dew points of both lines, condensate per source, stage powers."""
    spec = _spec(k, t_kalt)
    backend = NumpyBackend()
    gas = configs["A"].gas
    alloc = tuple(tuple(a) for a in allocation)
    lines = []
    out: dict[str, Any] = {"set_points": x, "t_kalt": t_kalt, "lines": {}}
    for name, (t_in, u) in (("A", (x[0], x[1])), ("B", (x[2], x[3]))):
        sol = evaluate_point(
            t_in, u, 1.0, t_amb, phi, max_iterations=30, tolerance=1e-7, base_config=configs[name]
        )
        state = sn.line_state(name, sol, configs[name], t_amb=t_amb, phi_amb=phi, t_in=t_in)
        lines.append(state)
        dew = float(dew_point_celsius(backend, gas, np.float64(state.outlet_humidity_kg_per_kg)))
        out["lines"][name] = {
            "inlet_temperature_celsius": t_in,
            "gas_velocity_m_per_s": u,
            "outlet_temperature_celsius": state.outlet_temperature_celsius,
            "outlet_humidity_kg_per_kg": state.outlet_humidity_kg_per_kg,
            "exhaust_dew_point_celsius": dew,
        }
    network, sources, sinks = sn._network(tuple(lines), spec.site)
    cascade = evaluate_cascade(network, FixedRouting(alloc).route(network), sources, sinks)
    out["condensate_kg_per_h_by_source"] = {
        name: float(cascade.source_condensate_kg_per_s[i, 0] * 3600.0)
        for i, name in enumerate(cascade.source_names)
    }
    out["stage_power_kw"] = {
        f"{s}->{d}": float(cascade.stage_recovered_power_kw[i, 0])
        for i, (s, d) in enumerate(
            zip(cascade.stage_source_names, cascade.stage_sink_names, strict=False)
        )
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    configs, k, _, _ = m3._setup()
    m4c = json.loads((RUNS_DIR / f"{M4C_ARTIFACT}.json").read_text("utf-8"))
    scenarios = {i: m4c["scenarios"][str(i)] for i in SCENARIO_IDS}
    line_sets: dict[int, dict[str, dict[int, list[float]]]] = {
        i: {"A": {}, "B": {}} for i in SCENARIO_IDS
    }
    for r in m4c["records"]:
        if r["kind"] == "line" and r["scenario"] in line_sets:
            line_sets[r["scenario"]][r["line"]][r["seed"]] = r["set_points"]
    fixed = m4.fixed_candidates()
    name = f"m5_tkalt_sensitivity_{SUFFIX}"
    scratch = RUNS_DIR / f"_{name}.jsonl"
    done: dict[tuple[Any, ...], dict[str, Any]] = {}
    if arguments.resume and scratch.exists():
        for line in scratch.read_text("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[_key(row)] = row
    elif scratch.exists():
        raise SystemExit(f"{scratch} exists; use --resume or remove it")
    with (
        Pool(cpu_count(), initializer=c._init, initargs=(configs,)) as pool,
        scratch.open("a", encoding="utf-8") as handle,
    ):
        s2_tasks = [
            {
                "kind": "s2",
                "t_kalt": t,
                "scenario": i,
                "seed": a * 10 + b,
                "seed_a": a,
                "seed_b": b,
                "x": [
                    line_sets[i]["A"][a][0],
                    line_sets[i]["A"][a][1],
                    line_sets[i]["B"][b][0],
                    line_sets[i]["B"][b][1],
                ],
                "t_amb": scenarios[i]["t_amb"],
                "phi": scenarios[i]["phi"],
                "k": k,
            }
            for t in T_KALT_ALL
            for i in SCENARIO_IDS
            for a in SEEDS
            for b in SEEDS
        ]
        _run(pool, s2_tasks, _s2_task, done, handle, "s2")
        joint_tasks = []
        candidates: dict[tuple[float, int], list[Any]] = {}
        for t in T_KALT_RUN:
            for i in SCENARIO_IDS:
                pairs = [
                    r
                    for r in done.values()
                    if r["kind"] == "s2" and r["t_kalt"] == t and r["scenario"] == i
                ]
                best = min(pairs, key=lambda r: r["value"])
                listing = {json.dumps(a): a for a in fixed}
                listing.setdefault(json.dumps(best["allocation"]), best["allocation"])
                candidates[(t, i)] = list(listing.values())
                for index, allocation in enumerate(candidates[(t, i)]):
                    for seed in SEEDS:
                        joint_tasks.append(
                            {
                                "kind": "joint",
                                "t_kalt": t,
                                "scenario": i,
                                "seed": seed,
                                "allocation_index": index,
                                "allocation": allocation,
                                "t_amb": scenarios[i]["t_amb"],
                                "phi": scenarios[i]["phi"],
                                "k": k,
                            }
                        )
        _run(pool, joint_tasks, _joint_task, done, handle, "joint")
    # ------------------------------------------------------------------ assembly
    results: dict[str, Any] = {}
    integrity: dict[str, float] = {}
    for i in SCENARIO_IDS:
        ref = scenarios[i]
        per_t: dict[str, Any] = {}
        for t in T_KALT_ALL:
            pairs = [
                r
                for r in done.values()
                if r["kind"] == "s2" and r["t_kalt"] == t and r["scenario"] == i
            ]
            best_s2 = min(pairs, key=lambda r: r["value"])
            if t == T_KALT_REFERENCE:
                integrity[str(i)] = abs(best_s2["value"] - ref["best_s2"]["value"])
                per_seed = {
                    seed: {
                        "value": e["value"],
                        "allocation": e["allocation"],
                        "set_points": e["set_points"],
                    }
                    for seed, e in ref["per_seed_m3"].items()
                }
            else:
                per_seed = {}
                for seed in SEEDS:
                    mine = [
                        r
                        for r in done.values()
                        if r["kind"] == "joint"
                        and r["t_kalt"] == t
                        and r["scenario"] == i
                        and r["seed"] == seed
                    ]
                    top = min(mine, key=lambda r: r["value"])
                    per_seed[seed] = {
                        "value": top["value"],
                        "allocation": top["allocation"],
                        "set_points": top["detail"]["set_points"],
                        "feasible": bool(
                            top["detail"]["feasible_a"] and top["detail"]["feasible_b"]
                        ),
                    }
            gains = {str(s): best_s2["value"] - e["value"] for s, e in per_seed.items()}
            top_seed = min(per_seed, key=lambda s: per_seed[s]["value"])
            top_m3 = per_seed[top_seed]
            per_t[str(t)] = {
                "best_s2_value": best_s2["value"],
                "gain_per_seed": gains,
                "gain_min": min(gains.values()),
                "gain_max": max(gains.values()),
                "best_m3_value": top_m3["value"],
                "best_m3_allocation": top_m3["allocation"],
                "best_m3_set_points": top_m3["set_points"],
                "description_best_m3": describe(
                    top_m3["set_points"],
                    top_m3["allocation"],
                    configs,
                    ref["t_amb"],
                    ref["phi"],
                    k,
                    t,
                ),
                "description_best_s2": describe(
                    best_s2["set_points"],
                    best_s2["allocation"],
                    configs,
                    ref["t_amb"],
                    ref["phi"],
                    k,
                    t,
                ),
                "candidates": candidates.get((t, i)),
            }
        # sweep of T_kalt at the fixed solution of M3' at 29.1 degC
        base = per_t[str(T_KALT_REFERENCE)]
        sweep = {
            str(t): describe(
                base["best_m3_set_points"],
                base["best_m3_allocation"],
                configs,
                ref["t_amb"],
                ref["phi"],
                k,
                t,
            )["condensate_kg_per_h_by_source"]
            for t in SWEEP
        }
        results[str(i)] = {
            "group": ref["group"],
            "t_amb": ref["t_amb"],
            "phi": ref["phi"],
            "absolute_humidity_ambient_kg_per_kg": ref["absolute_humidity_ambient_kg_per_kg"],
            "by_t_kalt": per_t,
            "condensate_sweep_at_fixed_solution_of_29_1": sweep,
        }
    write_artifact(
        name,
        {
            "_provenance": provenance(),
            "note": "descriptive, no decision rule; lines reused from m4c_confirm_lab_umin126; 29.1 degC = M4-C",
            "integrity_s2_difference_at_29_1_kg_per_h": integrity,
            "integrity_ok": bool(
                all(v <= INTEGRITY_TOLERANCE_KG_PER_H for v in integrity.values())
            ),
            "scale_factor_k": k,
            "scenarios": results,
            "records": sorted(
                done.values(),
                key=lambda r: (
                    r["kind"],
                    r["t_kalt"],
                    r["scenario"],
                    r["seed"],
                    r.get("allocation_index") or 0,
                ),
            ),
        },
    )
    print("integrity", integrity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
