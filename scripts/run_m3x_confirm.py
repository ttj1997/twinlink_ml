r"""Protection of the five large gains of the M3' control (Befund 73, EXPLORATORY): direct, many seeds.

Released by the project lead on 2026-10-07. Only the scenarios 16, 17, 22, 23, 24 of the 25 Befund 52
ambient points, lower gas-velocity bound 1.26 m/s only (0.80 m/s is NOT computed), lines A (1.5 kg/h) and B
(2.0 kg/h), scaled site (k = 271.797, a scaling computation), no exchanger limit (the control), margin m for
both lines (D8, an assumption for B), nozzle pressure 1.0 bar for the joint optimum (D6). The five were
singled out after the results of the joint stage; their gains are exploratory until they survive this
protection and, afterwards, a confirmatory test formulated beforehand.

(a) S2 and M3' DIRECTLY (no tables), five seeds each (DE seeds 0-4):
  * S2: the optimum of EACH line on its own (the line objective of M1/M2: three variables, tightened limit,
    Befund 52 budget, ``carrier_oracle``) for every seed and both lines; S2 of a pair of seeds (i for A, j for B)
    is the best of the 441 allocations at the set points of the pair, evaluated at 1.0 bar with true
    solves. **The BEST S2 is the minimum over all 25 pairs** (the strongest S2 the search finds).
  * M3': for every allocation of a shortlist (see below) and every seed the joint DE + Nelder-Mead of the four
    set points on TRUE solves (budget of Befund 52, popsize 10, maxiter 50); M3'(seed) is the best over the
    shortlist.
  * Shortlist per scenario (fixed here): the allocation M3' chose in the joint stage, every allocation with a
    polished value within 1.0 kg CO2e/h of it (one per distinct value, 0.01), the allocations M3' chose with the
    other seeds of the joint stage where they exist (scenario 17), and the allocation of S2 of the joint
    stage.
  * Per scenario: ``G_s = value(best S2) - value(M3'(seed s))`` for each seed. **The gain HOLDS (confirmed) iff
    ``min_s G_s`` is at least 1.0 kg CO2e/h** (the median threshold of the stage rule); WEAK iff it is at least
    0.2 kg/h (the per-scenario threshold); otherwise NOT CONFIRMED. **Overall the gains are CONFIRMED iff at
    least three of the five scenarios are confirmed.** (Both thresholds are those of ``coupling_gain``; the
    overall count is fixed here.)

(b) At the best S2 and at the best M3' of every scenario (and at the M3' of every seed): the distance of the
    product moisture of each line to the tightened limit (0.15 - m), to the quality limit 0.15 and to the
    validity boundary of the kinetics (the critical moisture 0.20), the bed temperature against 80 degC, and
    which set points lie at the boundaries of the action box (inlet temperature 60 / 120 degC, gas velocity
    1.26 / 2.5 m/s, nozzle pressure 1.0 / 4.0 bar), within 0.05 K, 1e-3 m/s and 1e-3 bar. A gain that lies at a
    boundary is reported.

(c) The physical explanation is descriptive: for the best S2 and the best M3' the allocation, the set points,
    the CO2 split into electricity and the fuel of the three sinks, the power of every stage, the exhaust of
    both lines (``analyze_m3x_confirm.py``).

Run-time estimate (before the run; measured costs of Befund 73: 0.17 s per solve alone, 0.34 s with 16 workers
busy): the per-line searches are single-solve problems (about 1100 calls, about 7 min each): 2 lines x 5
seeds x 5 scenarios = 50 runs; the joint searches are two-solve problems (about 2300 calls, about 25 min each):
about 11 shortlisted allocations x 5 seeds = 55 runs; together about 105 runs on 16 workers, about 2.5 h wall
(range 2-4 h, the polishing of the main stage took up to 2.7 h for single scenarios; here no tables and no
polishing).

Extension of 2026-10-07 (project lead; the first run above covered 16, 17, 22, 23, 24 by a mix-up of the
order: the five LARGEST gains of Befund 73 are 18, 19, 22, 23, 24). The script is now parameterised; the
defaults reproduce the first run exactly (scenarios 16, 17, 22, 23, 24, box 60-120 degC, no cap):

* ``--scenarios 18,19 --name m3x_confirm_s18_19_umin126``: the two largest gains not yet protected, same
  method and criteria.
* ``--t-max 100 --scenarios 18,19,22,23,24 --name m3x_sens_tmax100_umin126``: sensitivity to the upper bound of
  the inlet temperature of BOTH lines, 100 degC (the laboratory range 50-100 degC), as a RUN PARAMETER, the
  default box (60-120 degC) is not changed; the S2 line searches use the same box. Same shortlists as at 120.
  Fixed beforehand: the gain is NOT bound to the box limit iff G (minimum over the seeds against the best S2,
  as above) exceeds 1 kg CO2e/h in at least four of the five scenarios.
* ``--scenarios <the other 18> --shortlist-cap 5 --name m3x_robust_rest_umin126``: robustness re-measurement of
  the verdict of Befund 73 on all 25 scenarios (together with the first run, 16, 17, 22, 23, 24, and the
  18/19 run, all at the box 60-120 degC and 1.26 m/s; evaluated by ``analyze_m3x_robust.py``). Shortlist cap
  5 per scenario (cost: the uncapped rule would give up to 15 allocations at scenarios 15, 20, 21 where the
  gain is about zero and many allocations are tied): kept first the allocations chosen by M3' (all seeds of
  the joint stage) and by S2, then the others by ascending polished value; the cap can only lower M3'
  (a conservative bias against a gain, in tied scenarios).

    PYTHONPATH=<worktree>/src python scripts/run_m3x_confirm.py [--resume] [--scenarios ..] [--t-max ..]
        [--shortlist-cap N] [--name ..]
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

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_m1_margin_cost as m1  # noqa: E402
import run_m3_joint as m3  # noqa: E402
from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation import carrier_oracle as co  # noqa: E402
from twinlink_ml.evaluation import joint_search as js  # noqa: E402
from twinlink_ml.evaluation import site_network as sn  # noqa: E402
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec  # noqa: E402
from twinlink_ml.simulation.oracle_search import run_de_nm  # noqa: E402

SCENARIOS: tuple[int, ...] = (16, 17, 22, 23, 24)
T_MAX = 120.0
SHORTLIST_CAP: int | None = None
ARTIFACT = f"m3x_confirm_{m3.SUFFIX}"
SEEDS = (0, 1, 2, 3, 4)
SHORTLIST_WINDOW = 1.0
SUFFIX = m3.SUFFIX
THRESHOLD = m3.THRESHOLD
MARGIN = m3.MARGIN
CONFIRMED_MIN_KG_PER_H = 1.0
WEAK_MIN_KG_PER_H = 0.2
OVERALL_MIN_CONFIRMED = 3

_STATE: dict[str, Any] = {}


def _init(configs: dict[str, Any]) -> None:
    _STATE["configs"] = configs
    _STATE["allocations"] = sn.all_allocations(2, order_variants=True)


def _allocation_key(allocation: Any) -> tuple[tuple[str, str], ...]:
    return tuple((str(a), str(b)) for a, b in allocation)


def shortlists() -> dict[int, list[list[list[str]]]]:
    """Allocations per scenario for the direct joint searches (rule in the module docstring)."""
    control = json.loads((RUNS_DIR / f"m3_joint_control_{SUFFIX}.json").read_text("utf-8"))
    seeds = json.loads((RUNS_DIR / f"m3_joint_seeds_{SUFFIX}.json").read_text("utf-8"))
    rows = {r["grid_index"]: r for r in control["scenarios"]}
    out: dict[int, list[list[list[str]]]] = {}
    for s in SCENARIOS:
        row = rows[s]
        chosen: dict[tuple[tuple[str, str], ...], list[list[str]]] = {}
        capped: dict[tuple[tuple[str, str], ...], list[list[str]]] = {}
        polished = [
            e for e in row["search"]["polished"].values() if e.get("polished_value") is not None
        ]
        best = min(e["polished_value"] for e in polished)
        seen_values: set[float] = set()
        near: list[list[list[str]]] = []
        for e in sorted(polished, key=lambda e: e["polished_value"]):
            if (
                e["polished_value"] <= best + SHORTLIST_WINDOW
                and round(e["polished_value"], 2) not in seen_values
            ):
                seen_values.add(round(e["polished_value"], 2))
                near.append(e["allocation"])
        priority = [row["search"]["m3"]["allocation"]]
        for r in seeds["records"]:
            if r["case"] == "control" and r["grid_index"] == s and "search" in r:
                priority.append(r["search"]["m3"]["allocation"])
        priority.append(row["s2"]["allocation"])
        for allocation in priority + near:
            chosen.setdefault(_allocation_key(allocation), allocation)
        if SHORTLIST_CAP is not None:
            for allocation in priority + near:
                if len(capped) < SHORTLIST_CAP:
                    capped.setdefault(_allocation_key(allocation), allocation)
            chosen = capped
        out[s] = list(chosen.values())
    return out


def _task(task: dict[str, Any]) -> dict[str, Any]:
    configs = _STATE["configs"]
    start = time.perf_counter()
    budget = dataclasses.replace(m1._budget(1), seed=task["seed"])
    if task["kind"] == "line":
        spec = co.OracleSpec(THRESHOLD, "co2_gas_heater", m1.GAS_HEATER_EFFICIENCY)
        co.init_worker(configs[task["line"]])
        base_bounds = [(60.0, task["t_max"]), (m3.BOUND, 2.5), (1.0, 4.0)]
        result = co.optimize_point(task["t_amb"], task["phi"], base_bounds, budget, spec)
        return {
            **{k: task[k] for k in ("kind", "line", "seed", "scenario")},
            "set_points": [
                result["inlet_temperature_celsius"],
                result["gas_velocity_m_per_s"],
                result["nozzle_pressure_bar"],
            ],
            "feasible": result.get("feasible"),
            "solve_failed": result.get("solve_failed"),
            "objective": result.get("objective"),
            "wall_s": time.perf_counter() - start,
        }
    spec = SiteOracleSpec(THRESHOLD, js.site_spec_with_cap(sn.SiteSpec(task["k"]), None))
    allocation = tuple(tuple(c) for c in task["allocation"])
    t_max = task["t_max"]
    bounds = [(60.0, t_max), (m3.BOUND, 2.5), (60.0, t_max), (m3.BOUND, 2.5)]
    de, refine, _, _ = run_de_nm(
        js._true_value,
        bounds,
        (allocation, configs, task["t_amb"], task["phi"], spec),
        budget,
        de_workers=1,
    )
    best = refine if refine.fun <= de.fun else de
    final = js.evaluate_true(best.x, allocation, configs, task["t_amb"], task["phi"], spec)
    return {
        **{k: task[k] for k in ("kind", "seed", "scenario", "allocation_index")},
        "allocation": task["allocation"],
        "value": float(final["value"]),
        "detail": final,
        "wall_s": time.perf_counter() - start,
    }


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["kind"],
        row.get("line"),
        row["seed"],
        row["scenario"],
        row.get("allocation_index"),
    )


def main() -> int:
    global SCENARIOS, T_MAX, SHORTLIST_CAP
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scenarios", default=None, help="comma-separated grid indices")
    parser.add_argument("--t-max", type=float, default=T_MAX)
    parser.add_argument("--shortlist-cap", type=int, default=None)
    parser.add_argument("--name", default=ARTIFACT)
    arguments = parser.parse_args()
    if arguments.scenarios:
        SCENARIOS = tuple(int(v) for v in arguments.scenarios.split(","))
    T_MAX = arguments.t_max
    SHORTLIST_CAP = arguments.shortlist_cap
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    configs, k, grid, _ = m3._setup()
    lists = shortlists()
    tasks: list[dict[str, Any]] = []
    for s in SCENARIOS:
        t_amb, phi = grid[s]
        for index, allocation in enumerate(lists[s]):
            for seed in SEEDS:
                tasks.append(
                    {
                        "kind": "joint",
                        "scenario": s,
                        "seed": seed,
                        "allocation_index": index,
                        "allocation": allocation,
                        "t_amb": t_amb,
                        "phi": phi,
                        "k": k,
                        "t_max": T_MAX,
                    }
                )
    for s in SCENARIOS:
        t_amb, phi = grid[s]
        for line in ("A", "B"):
            for seed in SEEDS:
                tasks.append(
                    {
                        "kind": "line",
                        "line": line,
                        "scenario": s,
                        "seed": seed,
                        "t_amb": t_amb,
                        "phi": phi,
                        "t_max": T_MAX,
                    }
                )
    scratch = RUNS_DIR / f"_{arguments.name}.jsonl"
    done: dict[tuple[Any, ...], dict[str, Any]] = {}
    if arguments.resume and scratch.exists():
        for line in scratch.read_text("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[_key(row)] = row
    elif scratch.exists():
        raise SystemExit(f"{scratch} exists; use --resume or remove it")
    todo = [t for t in tasks if _key(t) not in done]
    print(f"{len(done)} done, {len(todo)} to run of {len(tasks)}", flush=True)
    start = time.perf_counter()
    with (
        Pool(cpu_count(), initializer=_init, initargs=(configs,)) as pool,
        scratch.open("a", encoding="utf-8") as handle,
    ):
        for n, row in enumerate(pool.imap_unordered(_task, todo), start=1):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done[_key(row)] = row
            print(
                f"  [{n}/{len(todo)}] {row['kind']} scenario {row['scenario']} seed {row['seed']} "
                f"({time.perf_counter() - start:.0f} s)",
                flush=True,
            )
    write_artifact(
        arguments.name,
        {
            "_provenance": provenance(),
            "inlet_temperature_upper_bound_celsius": T_MAX,
            "shortlist_cap": SHORTLIST_CAP,
            "scenarios": list(SCENARIOS),
            "seeds": list(SEEDS),
            "shortlists": {str(s): lists[s] for s in SCENARIOS},
            "scale_factor_k": k,
            "quality_threshold_kg_per_kg": THRESHOLD,
            "records": sorted(done.values(), key=lambda r: (r["kind"], r["scenario"], r["seed"])),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
