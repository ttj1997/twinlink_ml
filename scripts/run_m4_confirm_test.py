r"""Confirmatory test of the coupling gain at warm and humid ambient air (M4-C). FIXED BEFORE ANY COMPUTATION.

Status: the large gains of the joint optimum (Befund 73, scenarios 18, 19, 22, 23, 24) are EXPLORATORY in their
origin; Befund 76 protected them against the search (five seeds, best S2) but did not formulate them beforehand.
This test is formulated, fixed and committed (this file and the section "Confirmatory test M4-C" of
``docs/09_entwurf_modellbasierte_optimierung.md``) before any of its scenarios has been computed. Nothing of
it has been computed at the time of the commit.

Hypothesis H1: at warm and humid ambient air the joint optimum of the two lines (M3') saves more than 1.0 kg
CO2e/h of the site against "each line on its own, then the best interconnection" (S2), in the great majority
of the scenarios of group W, with the same method as in Befund 76, in scenarios that are not among the 25
Befund 52 points.

Scenarios (ambient temperature degC, relative humidity), none among the 25 grid points (grid temperatures 5,
12.5, 20, 27.5, 35; humidities 0.25, 0.40, 0.55, 0.70, 0.85):
  * W (decision rule): T in {30.0, 32.5} x phi in {0.60, 0.72, 0.82}: six scenarios.
  * N (negative control, descriptive, no decision rule; the expectation G about zero is stated, not tested):
    (32.5, 0.30), (30.0, 0.35).
  * H (descriptive, no decision rule): mild and humid (20.0, 0.82) and (25.0, 0.82), to separate whether the gain
    depends on the temperature or on the absolute humidity. For EVERY scenario the absolute humidity of the
    ambient air (kg/kg dry air, ``thermo.absolute_humidity_kg_per_kg``) is reported next to G.

Method (that of Befund 76, unchanged): lines A (1.5 kg/h) and B (2.0 kg/h), scaled site (k = 271.797), no
exchanger limit, cold-side temperature 29.1 degC, lower gas-velocity bound 1.26 m/s (0.80 m/s is NOT
computed), margin m for both lines, nozzle pressure 1.0 bar for the joint optimum; S2: the optimum of each
line on its own for the DE seeds 0-4, S2 of a pair of seeds = the best of the 441 interconnections at the
set points of the pair with true solves, **best S2 = minimum over the 25 pairs**; M3': for every
interconnection of the candidate list and every seed 0-4 the joint DE + Nelder-Mead of the four set points on
true solves (budget of Befund 52); M3'(seed) = the best over the candidate list; ``G_s = value(best S2) -
value(M3'(seed s))``; ``G = min_s G_s``.

Candidate list per scenario (fixed): the interconnections that M3' chose in the best solution of the scenarios
18, 19, 22, 23, 24 at the box 60-120 degC AND at the box 60-100 degC (read from the evaluation artifacts of
Befund 76; there are two distinct ones), plus the interconnection of the best S2 of the scenario itself.

Two boxes of the inlet temperature (a run parameter ``--box``; the default box is not changed):
  * ``lab``: 60-100 degC (the laboratory range 50-100 degC). **K_lab (MAIN criterion): G >= 1.0 kg CO2e/h in
    at least 5 of the 6 scenarios of W.** The statement for the conference rests on K_lab: above 100 degC the
    model extrapolates (as for the lower bound of the gas velocity). Groups N and H are computed at this box
    only.
  * ``box``: 60-120 degC. **K_box (second criterion): G >= 1.0 kg CO2e/h in at least 5 of the 6 scenarios of
    W.** Group W only. Both criteria are reported.

If K_lab is missed: the gain is NOT confirmed, the five scenarios stay EXPLORATORY; no recomputation and no
change of a threshold afterwards.

Run-time estimate: see ``docs/09`` (about 7 h on 16 workers).

    PYTHONPATH=<worktree>/src python scripts/run_m4_confirm_test.py --box lab [--resume]
    PYTHONPATH=<worktree>/src python scripts/run_m4_confirm_test.py --box box [--resume]
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
from twinlink_ml.simulation import NumpyBackend  # noqa: E402
from twinlink_ml.simulation.thermo import absolute_humidity_kg_per_kg  # noqa: E402

GROUPS: dict[str, list[tuple[float, float]]] = {
    "W": [(30.0, 0.60), (30.0, 0.72), (30.0, 0.82), (32.5, 0.60), (32.5, 0.72), (32.5, 0.82)],
    "N": [(32.5, 0.30), (30.0, 0.35)],
    "H": [(20.0, 0.82), (25.0, 0.82)],
}
BOXES = {"lab": (100.0, ("W", "N", "H")), "box": (120.0, ("W",))}
GRID_TEMPERATURES = (5.0, 12.5, 20.0, 27.5, 35.0)
GRID_HUMIDITIES = (0.25, 0.40, 0.55, 0.70, 0.85)
SEEDS = c.SEEDS
GAIN_MIN_KG_PER_H = 1.0
MIN_SCENARIOS_OF_W = 5
CANDIDATE_SOURCES = (
    ("m3x_confirm_s18_19_evaluation_umin126", (18, 19)),
    ("m3x_confirm_evaluation_umin126", (22, 23, 24)),
    ("m3x_sens_tmax100_evaluation_umin126", (18, 19, 22, 23, 24)),
)


def scenario_list(box: str) -> list[dict[str, Any]]:
    out = []
    for group in BOXES[box][1]:
        for t_amb, phi in GROUPS[group]:
            assert not (t_amb in GRID_TEMPERATURES and phi in GRID_HUMIDITIES)
            out.append({"id": len(out), "group": group, "t_amb": t_amb, "phi": phi})
    return out


def fixed_candidates() -> list[list[list[str]]]:
    found: dict[str, list[list[str]]] = {}
    for name, scenarios in CANDIDATE_SOURCES:
        art = json.loads((RUNS_DIR / f"{name}.json").read_text("utf-8"))
        for s in scenarios:
            seeds = art["scenarios"][str(s)]["per_seed_m3"]
            best = min(seeds.values(), key=lambda e: e["value"])
            found.setdefault(json.dumps(best["allocation"]), best["allocation"])
    return list(found.values())


def _s2_pair(task: dict[str, Any]) -> dict[str, Any]:
    spec = SiteOracleSpec(c.THRESHOLD, js.site_spec_with_cap(sn.SiteSpec(task["k"]), None))
    x = task["x"]
    ev = js.evaluate_true(
        x, c._STATE["allocations"], c._STATE["configs"], task["t_amb"], task["phi"], spec
    )
    return {
        "kind": "s2",
        "scenario": task["scenario"],
        "seed": task["seed_a"] * 10 + task["seed_b"],
        "seed_a": task["seed_a"],
        "seed_b": task["seed_b"],
        "value": float(ev["value"]),
        "co2": float(ev["co2_kg_per_h"]),
        "allocation": ev["allocation"],
        "set_points": x,
        "feasible": bool(ev["feasible_a"] and ev["feasible_b"]),
    }


def _stage(pool: Any, tasks: list[dict[str, Any]], done: dict, handle: Any, label: str) -> None:
    todo = [t for t in tasks if c._key(t) not in done]
    print(f"{label}: {len(tasks) - len(todo)} done, {len(todo)} to run", flush=True)
    start = time.perf_counter()
    function = _s2_pair if label == "s2" else c._task
    for n, row in enumerate(pool.imap_unordered(function, todo), start=1):
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        done[c._key(row)] = row
        print(
            f"  {label} [{n}/{len(todo)}] scenario {row['scenario']} ({time.perf_counter() - start:.0f} s)",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--box", choices=sorted(BOXES), required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    t_max = BOXES[arguments.box][0]
    configs, k, _, _ = m3._setup()
    scenarios = scenario_list(arguments.box)
    name = f"m4c_confirm_{arguments.box}_{c.SUFFIX}"
    scratch = RUNS_DIR / f"_{name}.jsonl"
    done: dict[tuple[Any, ...], dict[str, Any]] = {}
    if arguments.resume and scratch.exists():
        for line in scratch.read_text("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[c._key(row)] = row
    elif scratch.exists():
        raise SystemExit(f"{scratch} exists; use --resume or remove it")
    fixed = fixed_candidates()
    with (
        Pool(cpu_count(), initializer=c._init, initargs=(configs,)) as pool,
        scratch.open("a", encoding="utf-8") as handle,
    ):
        line_tasks = [
            {
                "kind": "line",
                "line": line,
                "scenario": s["id"],
                "seed": seed,
                "t_amb": s["t_amb"],
                "phi": s["phi"],
                "t_max": t_max,
            }
            for s in scenarios
            for line in ("A", "B")
            for seed in SEEDS
        ]
        _stage(pool, line_tasks, done, handle, "line")
        s2_tasks = []
        for s in scenarios:
            sets: dict[str, dict[int, list[float]]] = {"A": {}, "B": {}}
            for row in done.values():
                if row["kind"] == "line" and row["scenario"] == s["id"]:
                    sets[row["line"]][row["seed"]] = row["set_points"]
            for i, sa in sets["A"].items():
                for j, sb in sets["B"].items():
                    s2_tasks.append(
                        {
                            "scenario": s["id"],
                            "seed_a": i,
                            "seed_b": j,
                            "x": [sa[0], sa[1], sb[0], sb[1]],
                            "t_amb": s["t_amb"],
                            "phi": s["phi"],
                            "k": k,
                            "kind": "s2",
                            "seed": i * 10 + j,
                        }
                    )
        _stage(pool, s2_tasks, done, handle, "s2")
        joint_tasks = []
        candidates: dict[int, list[list[list[str]]]] = {}
        for s in scenarios:
            pairs = [r for r in done.values() if r["kind"] == "s2" and r["scenario"] == s["id"]]
            best = min(pairs, key=lambda r: r["value"])
            listing = {json.dumps(a): a for a in fixed}
            listing.setdefault(json.dumps(best["allocation"]), best["allocation"])
            candidates[s["id"]] = list(listing.values())
            for index, allocation in enumerate(candidates[s["id"]]):
                for seed in SEEDS:
                    joint_tasks.append(
                        {
                            "kind": "joint",
                            "scenario": s["id"],
                            "seed": seed,
                            "allocation_index": index,
                            "allocation": allocation,
                            "t_amb": s["t_amb"],
                            "phi": s["phi"],
                            "k": k,
                            "t_max": t_max,
                        }
                    )
        _stage(pool, joint_tasks, done, handle, "joint")
    backend = NumpyBackend()
    results: dict[str, Any] = {}
    for s in scenarios:
        pairs = [r for r in done.values() if r["kind"] == "s2" and r["scenario"] == s["id"]]
        best = min(pairs, key=lambda r: r["value"])
        per_seed = {}
        for seed in SEEDS:
            mine = [
                r
                for r in done.values()
                if r["kind"] == "joint" and r["scenario"] == s["id"] and r["seed"] == seed
            ]
            top = min(mine, key=lambda r: r["value"])
            per_seed[seed] = {
                "value": top["value"],
                "allocation": top["allocation"],
                "set_points": top["detail"]["set_points"],
                "feasible": bool(top["detail"]["feasible_a"] and top["detail"]["feasible_b"]),
                "moisture_a": top["detail"]["moisture_a"],
                "moisture_b": top["detail"]["moisture_b"],
            }
        gains = {seed: best["value"] - e["value"] for seed, e in per_seed.items()}
        y = float(
            absolute_humidity_kg_per_kg(
                backend, configs["A"].gas, np.float64(s["t_amb"]), np.float64(s["phi"])
            )
        )
        results[str(s["id"])] = {
            **s,
            "absolute_humidity_ambient_kg_per_kg": y,
            "candidates": candidates[s["id"]],
            "best_s2": {
                k_: best[k_] for k_ in ("value", "co2", "allocation", "set_points", "feasible")
            },
            "per_seed_m3": per_seed,
            "gain_per_seed": gains,
            "gain_min_over_seeds": min(gains.values()),
            "gain_median_over_seeds": float(np.median(list(gains.values()))),
            "feasible_all": bool(
                best["feasible"] and all(e["feasible"] for e in per_seed.values())
            ),
        }
    w = [r for r in results.values() if r["group"] == "W"]
    n_hold = sum(r["gain_min_over_seeds"] >= GAIN_MIN_KG_PER_H and r["feasible_all"] for r in w)
    criterion = "K_lab" if arguments.box == "lab" else "K_box"
    write_artifact(
        name,
        {
            "_provenance": provenance(),
            "box_inlet_temperature_celsius": [60.0, t_max],
            "criterion": criterion,
            "criterion_rule": f"G >= {GAIN_MIN_KG_PER_H} kg/h in at least {MIN_SCENARIOS_OF_W} of 6 scenarios of W",
            "n_scenarios_of_W_holding": n_hold,
            "criterion_met": bool(n_hold >= MIN_SCENARIOS_OF_W),
            "scale_factor_k": k,
            "scenarios": results,
            "records": sorted(
                done.values(),
                key=lambda r: (r["kind"], r["scenario"], r["seed"], r.get("allocation_index") or 0),
            ),
        },
    )
    print(criterion, "met" if n_hold >= MIN_SCENARIOS_OF_W else "NOT met", n_hold, "of 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
