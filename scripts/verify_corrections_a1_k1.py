r"""Verification of the corrections K1 and A1 (Befund 79): the figures of the lab-book entry, from committed code.

(a) K1 bit identity: every distinct stored line solution of the artifacts of Befunde 69, 73, 76, 77 (the 1037 of
    ``review_k1_solver_bracket``, computed with the solver before Befund 79) is re-solved with the corrected
    solver at the search precision; bed temperature and product moisture are compared with ``==``; the use of
    the fallback interval and the feasibility are counted.
(b) A1 legacy: every S2 and joint record of ``m4c_confirm_lab_umin126`` is re-evaluated with
    ``approach_check="warm_end"`` (stored interconnection); the stored CO2 must come out with ``==``.
(c) A1 discretisation: the stage duty at 400 profile nodes against 40000 nodes on 400 random condensing stages
    (seed 20261009; the population of ``tests/test_heatnetwork_approach.py``).
(d) The grid of the check script of Befund 78 (0.01 K, no exact dew-point node) against the corrected cascade at
    the best M3' of seed 0 of the scenario (30.0; 0.82), all 441 interconnections, for the grid steps 0.01,
    0.001 and 0.0001 K: the largest CO2 difference shrinks with the step if it is the error of that grid.
(e) Order variants without an exchanger limit: at every reported solution of ``m4c_confirm_lab_umin126`` (best
    S2, M3' of every seed), the pairs of interconnections that differ only in the order of the stages
    ``Q1 -> SA`` and ``Q1 -> SB``; how many give a different site CO2 (more than 1e-6 kg/h) and by how much, with
    the warm-end check and with the check along the exchanger. Bears on the shortcut of
    ``joint_search.nested_search`` that treats them as the same problem when there is no limit.

    PYTHONPATH=src python scripts/verify_corrections_a1_k1.py
"""

from __future__ import annotations

import dataclasses
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

import check_review_a1_cascade_approach as a1  # noqa: E402
import run_m3_joint as m3  # noqa: E402
from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation import site_network as sn  # noqa: E402
from twinlink_ml.heatnetwork.balance import SinkState, SourceState, evaluate_cascade  # noqa: E402
from twinlink_ml.heatnetwork.config import (  # noqa: E402
    HeatNetworkConfig,
    HeatSinkConfig,
    HeatSourceConfig,
)
from twinlink_ml.simulation import PlaceholderConfigWarning  # noqa: E402
from twinlink_ml.simulation.oracle_search import evaluate_point  # noqa: E402

ARTIFACT = "verify_corrections_a1_k1"
_STATE: dict[str, Any] = {}


def _init(configs: dict[str, Any]) -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    _STATE["configs"] = configs


def _k1_task(entry: dict[str, Any]) -> tuple[bool, bool, bool, bool, bool]:
    solution = evaluate_point(
        *entry["set_point"],
        entry["t_amb"],
        entry["phi"],
        max_iterations=30,
        tolerance=1e-7,
        base_config=_STATE["configs"][entry["line"]],
    )
    return (
        solution.bed_temperature_celsius == entry["bed_temperature_celsius"],
        solution.particle_moisture_kg_per_kg == entry["moisture_kg_per_kg"],
        solution.bed_temperature_fallback_used,
        solution.feasible,
        bool(entry["on_bound"]),
    )


def part_a(pool: Any) -> dict[str, Any]:
    stored = json.loads((RUNS_DIR / "review_k1_solver_bracket.json").read_text("utf-8"))[
        "solutions"
    ]
    distinct: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in stored:
        distinct.setdefault(
            (entry["line"], *entry["set_point"], entry["t_amb"], entry["phi"]), entry
        )
    rows = pool.map(_k1_task, list(distinct.values()), chunksize=8)
    return {
        "n_distinct_solutions": len(rows),
        "n_bed_temperature_identical": sum(r[0] for r in rows),
        "n_moisture_identical": sum(r[1] for r in rows),
        "n_fallback_used": sum(r[2] for r in rows),
        "n_feasible": sum(r[3] for r in rows),
        "n_on_old_bound": sum(r[4] for r in rows),
        "n_on_old_bound_feasible": sum(r[3] for r in rows if r[4]),
    }


def part_b(k: float) -> dict[str, Any]:
    art = json.loads((RUNS_DIR / "m4c_confirm_lab_umin126.json").read_text("utf-8"))
    legacy = sn.SiteSpec(k, approach_check="warm_end")
    n, identical = 0, 0
    for s in art["scenarios"].values():
        records = [
            r for r in art["records"] if r["scenario"] == s["id"] and r["kind"] in ("s2", "joint")
        ]
        for r in records:
            x = r["set_points"] if r["kind"] == "s2" else r["detail"]["set_points"]
            allocation = tuple(
                tuple(p)
                for p in (r["allocation"] if r["kind"] == "s2" else r["detail"]["allocation"])
            )
            co2 = r["co2"] if r["kind"] == "s2" else r["detail"]["co2_kg_per_h"]
            prep = a1.prepared(x, s["t_amb"], s["phi"])
            result, _ = sn.evaluate_site(prep.lines, legacy, [allocation])
            n += 1
            identical += result[0].co2_kg_per_h == co2
    return {"n_records": n, "n_co2_identical": identical}


def part_c() -> dict[str, Any]:
    rng = np.random.default_rng(20261009)
    errors_abs, errors_rel = [], []
    for _ in range(400):
        source = HeatSourceConfig(
            "s",
            float(rng.uniform(0.5, 20)),
            float(rng.uniform(35, 130)),
            float(rng.uniform(0.01, 0.09)),
        )
        sink = HeatSinkConfig(
            "k",
            float(rng.uniform(0.5, 40)),
            0.005,
            float(rng.uniform(5, 35)),
            float(rng.uniform(60, 300)),
        )
        base = HeatNetworkConfig(
            sources=(source,),
            sinks=(sink,),
            minimum_pinch_k=float(rng.choice([10.0, 20.0])),
            recovery_mode="condensing",
            cold_temperature_celsius=float(rng.uniform(10, 32)),
        )
        duty = {}
        for nodes in (400, 40000):
            result = evaluate_cascade(
                dataclasses.replace(base, approach_profile_nodes=nodes),
                (("s", "k"),),
                (SourceState.from_config(source),),
                (SinkState.from_config(sink),),
            )
            duty[nodes] = float(result.stage_recovered_power_kw[0, 0])
        if duty[40000] > 0.0:
            errors_abs.append(abs(duty[400] - duty[40000]))
            errors_rel.append(abs(duty[400] - duty[40000]) / duty[40000])
    return {
        "n_stages_with_duty": len(errors_abs),
        "max_abs_kw": max(errors_abs),
        "max_rel": max(errors_rel),
        "median_rel": float(np.median(errors_rel)),
    }


def part_d(k: float) -> dict[str, Any]:
    art = json.loads((RUNS_DIR / "m4c_confirm_lab_umin126.json").read_text("utf-8"))
    s = art["scenarios"]["2"]
    record = s["per_seed_m3"]["0"]
    allocations = [tuple(tuple(p) for p in a) for a in sn.all_allocations(2, order_variants=True)]
    out = {}
    for step in (0.01, 0.001, 0.0001):
        a1.PROFILE_STEP_K = step
        a1._STATE["prepared"] = {}  # profiles depend on the grid step
        prep = a1.prepared(record["set_points"], s["t_amb"], s["phi"])
        results, _ = sn.evaluate_site(prep.lines, sn.SiteSpec(k), allocations)
        out[f"{step:g}"] = max(
            abs(a1.cascade(prep, a, "along", 20.0)["co2"] - r.co2_kg_per_h)
            for a, r in zip(allocations, results, strict=True)
        )
    a1.PROFILE_STEP_K = 0.01
    return {"max_abs_co2_difference_kg_per_h_by_grid_step_k": out}


def part_e(k: float) -> dict[str, Any]:
    art = json.loads((RUNS_DIR / "m4c_confirm_lab_umin126.json").read_text("utf-8"))
    allocations = [tuple(tuple(p) for p in a) for a in sn.all_allocations(2, order_variants=True)]
    out = {}
    for mode in ("warm_end", "along_exchanger"):
        worst, n_diff, n_pairs = 0.0, 0, 0
        for s in art["scenarios"].values():
            for record in [s["best_s2"], *s["per_seed_m3"].values()]:
                prep = a1.prepared(record["set_points"], s["t_amb"], s["phi"])
                results, _ = sn.evaluate_site(
                    prep.lines, sn.SiteSpec(k, approach_check=mode), allocations
                )
                groups: dict[tuple[Any, ...], list[float]] = {}
                for allocation, r in zip(allocations, results, strict=True):
                    groups.setdefault(tuple(sorted(allocation)), []).append(r.co2_kg_per_h)
                for values in groups.values():
                    if len(values) > 1:
                        n_pairs += 1
                        difference = max(values) - min(values)
                        worst = max(worst, difference)
                        n_diff += difference > 1e-6
        out[mode] = {
            "n_pairs": n_pairs,
            "n_differing_by_more_than_1e-6": n_diff,
            "largest_difference_kg_per_h": worst,
        }
    return out


def main() -> int:
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    start = time.perf_counter()
    configs, k, _, _ = m3._setup()
    a1._init(configs, k)
    with Pool(cpu_count(), initializer=_init, initargs=(configs,)) as pool:
        result_a = part_a(pool)
    payload = {
        "_provenance": provenance(sn.SiteSpec(k)),
        "a_k1_bit_identity": result_a,
        "b_a1_legacy_reproduction_m4c": part_b(k),
        "c_a1_discretisation_400_vs_40000_nodes": part_c(),
        "d_check_script_grid_vs_corrected_cascade": part_d(k),
        "e_order_variants_without_limit": part_e(k),
    }
    payload["wall_s"] = time.perf_counter() - start
    write_artifact(ARTIFACT, payload)
    print(
        json.dumps({key: value for key, value in payload.items() if key != "_provenance"}, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
