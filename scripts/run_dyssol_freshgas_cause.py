"""Add-on to the fresh-gas study: medium-flow range or fresh-gas inputs as the cause.

Is the range of the medium flow, or are the fresh-gas inputs, the reason why the quadratic
regression misses the thresholds? Python rebuild only.

Design: ``docs/NN_dyssol_freshgas.md``, section 17; package ``twinlink_ml.dyssol_steady.cause``.

**ADDED AFTER THE FACT** (decision of the project lead, 2026-11-24, after the results of the main
run were known: reading D1, the quadratic regression has no ``N*`` up to 512 runs, prediction
``N* = 64`` not confirmed). Against Befund 128 two things had changed at once: the fresh-gas state
became an input (7 instead of 5 inputs) and the flow of the heat transfer medium went from the
narrow range of Befund 128 to the full range from no flow. This add-on separates them.

**No Dyssol run.** Step 0 of the main run found Dyssol and the Python rebuild equal to rounding at
all 1291 points of the box of the main run. Every variant box below lies inside that box (``V-b``
as its slice at the stored fresh-gas state). All data here come from the rebuild.

**Variants** (same design in each: pool of 1024 and test set of 256 scrambled Sobol points with
the seeds of the main run, N = 8 ... 512, seeds 1-5, thresholds 1 K and 1 %; only the quadratic
regression and the MLP with the setting selected for the main run):

* ``V-a``  seven inputs as in the main run, but the medium flow in the interval of the box of
  Befund 128;
* ``V-b``  the five inputs of Befund 128 (fresh gas fixed at the stored state), with the box of
  Befund 128 except for the medium flow, which takes the full range of the main run;
* ``V-c``  seven inputs and the box of the main run (control).

Test set of a variant: **"in-box Sobol test set, variant V-x"** (its own 256 points).

PRE-REGISTERED READING (fixed in this docstring before the first run of this script)
----------------------------------------------------------------------------------
Per variant and model the rule of docs/45, section 6: reading numbers = 95th percentile over the
test set of the absolute error of the exhaust temperature after the heat recovery [K] and of the
relative error of the fuel [%]; a model meets the thresholds at N if the median over the five
seeds of both is at or below 1 K and 1 %; ``N*`` = the smallest N from which on it meets them at
every larger N. "Reaches the thresholds" = has an ``N*``.

**Control first.** ``V-c`` must reproduce the main run: the same points (digest of the box equal,
same seeds), and ``N*`` of the quadratic regression and of the MLP as in
``dyssol_freshgas_v1_analysis`` (none and 256). The largest difference of the reading numbers to
the cells of ``dyssol_freshgas_v1_training`` is reported (the main run fitted Dyssol results, this
script the rebuild's). If ``V-c`` does not reproduce the main run, the add-on is not read.

**Reading K** on the quadratic regression:

* **K0** -- it reaches the thresholds in ``V-a`` and not in ``V-b``: the cause is the range of the
  medium flow, not the additional inputs;
* **K1** -- the reverse (not in ``V-a``, but in ``V-b``): the cause is the fresh-gas inputs;
* **K2** -- otherwise (in both or in neither).

**Hypothesis of the project lead** (stated with the decision): the recovery of an ideal
counter-current exchanger has a kink where the capacity rates of both sides are equal
(``Q = min(C_hot, C_cold) x dT_max``); with the full range of the medium flow the kink lies in the
box, in the box of Befund 128 it does not.

**Known before the run, from the equations of the unit** (``reference.heat_recovery``, equations
4.1 / 4.2; no number of this add-on had been computed): the unit is not that exchanger. Its
recovered heat is ``Q = (T_hot - T_cold) / (1/A + 1/B + 1/C)`` with the capacity rates of exhaust,
medium and fresh gas, a smooth saturation in the medium flow without a kink. Reading K does not
depend on the mechanism: it separates the range of the medium flow from the additional inputs. A
verdict K0 would therefore confirm the cause named by the hypothesis (the range of the medium
flow) and not its mechanism (a kink); what the split below shows is reported next to it.

**Reported WITHOUT a reading:** per variant the share of pool and test points on each side of
capacity ratio 1 (``B / min(A, C)``: medium below / at or above the smaller gas stream) and in the
bins of ``cause.RATIO_EDGES``; the reading numbers of the quadratic regression (and of the MLP)
separately per side and per bin; ``N*`` and the learning curve of the MLP per variant;
over-saturated points per variant (flag of docs/45).

**Seen before (disclosure).** All results of the main run; the equations of the unit; rough sizes
of the capacity rates from the local box. No fit on a variant box, no number of this add-on.

Usage: ``tools/run_here.sh scripts/run_dyssol_freshgas_cause.py --name <name> [--resume]
[--dry-run] [--workers 4]``. Cells are cached per unit in ``<out>/_<name>_cache.jsonl``; the
artifact ``artifacts/runs/<name>.json.gz`` holds errors, counts, shares and digests -- no input
value. ``--dry-run``: every cell with 5 optimiser steps, output in ``<out>``, never a result.
Local files and folders as ``scripts/run_dyssol_steady.py``; clean tree required.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))
sys.path.insert(0, str(SCRIPTS))

import run_dyssol_steady as run  # noqa: E402
from _common import RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.dyssol_steady import study  # noqa: E402
from twinlink_ml.dyssol_steady.cause import (  # noqa: E402
    MEDIUM,
    RATIO_EDGES,
    capacity_ratio,
    cause_reading,
    ratio_groups,
    with_interval_of,
)
from twinlink_ml.dyssol_steady.design import Box, apply_point, sobol_points  # noqa: E402
from twinlink_ml.dyssol_steady.flowsheet import ReferenceFlowsheet  # noqa: E402
from twinlink_ml.dyssol_steady.materials import sha256_of  # noqa: E402

CASE = "one_line"
MAIN_CONFIG = "dyssol_freshgas.yaml"
MAIN_TRAINING = "dyssol_freshgas_v1_training"
MAIN_ANALYSIS = "dyssol_freshgas_v1_analysis"
MODELS: tuple[str, ...] = ("quadratic", "mlp")
VARIANTS: dict[str, str] = {
    "V-a": "7 inputs, medium flow in the interval of the box of Befund 128",
    "V-b": "5 inputs of Befund 128 (fresh gas fixed), full range of the medium flow",
    "V-c": "7 inputs, box of the main run (control)",
}
NUMBERS = ("exhaust_p95_k", "fuel_p95_percent")


def variant_boxes(
    main: run.Context, befund_128: run.Context
) -> dict[str, tuple[Box, tuple[str, ...]]]:
    """Box and varied inputs of every variant, composed from the two local boxes."""
    stored = main.nominal[0]
    inside = main.box.contains(np.asarray([run.point_row([stored], main.config.varied)]))
    if not bool(inside.all()):
        raise SystemExit("the stored point (fresh-gas state of V-b) lies outside the main box")
    return {
        "V-a": (with_interval_of(main.box, befund_128.box, MEDIUM), main.config.varied),
        "V-b": (with_interval_of(befund_128.box, main.box, MEDIUM), befund_128.config.varied),
        "V-c": (main.box, main.config.varied),
    }


def evaluated(
    ctx: run.Context, box: Box, varied: tuple[str, ...], n: int, seed: int
) -> dict[str, Any]:
    """Sobol points of a variant through the rebuild: scaled inputs, targets, ratio, flags."""
    reference = ReferenceFlowsheet(ctx.properties, ctx.topology.preheated_by)
    x = sobol_points(box, n, seed)
    lines = [apply_point(ctx.nominal, row, varied) for row in x]
    outputs = [reference.evaluate(point).outputs for point in lines]
    return {
        "x": box.to_unit_symmetric(x),
        "y": study.target_matrix(outputs),
        "ratio": np.asarray(
            [
                capacity_ratio(ctx.properties, point[0], out[0])
                for point, out in zip(lines, outputs, strict=True)
            ]
        ),
        "oversaturated": np.asarray([any(o.oversaturated for o in out) for out in outputs]),
    }


def shares(ratio: np.ndarray) -> dict[str, int]:
    """Number of points per group of the capacity ratio."""
    return {label: int(mask.sum()) for label, mask in ratio_groups(ratio).items()}


def group_table(cells: list[dict[str, Any]], config: study.StudyConfig) -> dict[str, Any]:
    """Per model, group and N: points and the median over the seeds of both reading numbers."""
    table: dict[str, Any] = {}
    for model in MODELS:
        table[model] = {}
        labels = sorted({g for c in cells if c["model"] == model for g in c["metrics_by_group"]})
        for label in labels:
            table[model][label] = {}
            for n in config.budgets:
                group = [
                    c["metrics_by_group"][label]
                    for c in cells
                    if c["model"] == model and c["n"] == n and label in c["metrics_by_group"]
                ]
                table[model][label][str(n)] = {
                    "points": group[0]["points"],
                    **{k: float(np.median([g[k] for g in group])) for k in NUMBERS},
                    "max_abs_exhaust_k": float(
                        np.median([g["max_abs/line1/exhaust_celsius"] for g in group])
                    ),
                    "max_rel_fuel_percent": float(
                        np.median([g["max_rel/total_fuel"] for g in group])
                    ),
                }
    return table


def control_against_main(
    cells: list[dict[str, Any]], n_star: dict[str, int | None], box: Box
) -> dict[str, Any]:
    """Check that the control variant reproduces the main run (same points, same ``N*``)."""
    with gzip.open(RUNS_DIR / f"{MAIN_TRAINING}.json.gz", "rt", encoding="utf-8") as fh:
        training = json.load(fh)
    analysis = json.loads((RUNS_DIR / f"{MAIN_ANALYSIS}.json").read_text(encoding="utf-8"))
    main = {(c["model"], c["n"], c["seed"]): c["metrics"] for c in training["cells"]}
    difference = {
        model: {
            k: max(
                abs(c["metrics"][k] - main[(model, c["n"], c["seed"])][k])
                for c in cells
                if c["model"] == model
            )
            for k in NUMBERS
        }
        for model in MODELS
    }
    main_n_star = {model: analysis["reading"]["n_star"][model] for model in MODELS}
    same_box = training["header"]["box_sha256"] == box.sha256()
    return {
        "same_box_as_main_run": same_box,
        "n_star_main_run": main_n_star,
        "n_star_control": n_star,
        "largest_difference_of_reading_numbers": difference,
        "reproduces": bool(same_box and n_star == main_n_star),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cores = (os.cpu_count() or 2) // 2
    if not 1 <= args.workers <= max(cores, 1):
        raise SystemExit(f"--workers must be between 1 and the physical cores ({cores})")
    require_clean_tree()
    ctx = run.build_context(CASE, MAIN_CONFIG, sheet=False)  # the sheet is not loaded
    befund_128 = run.build_context(CASE)
    config = ctx.config
    boxes = variant_boxes(ctx, befund_128)
    settings = asdict(config.mlp_settings(CASE))
    if args.dry_run:
        settings["steps"] = run.DRY_STEPS
    head = {
        "stage": "cause",
        "name": args.name,
        "dry_run": args.dry_run,
        "git/commit": git("rev-parse", "HEAD"),
        "config_fingerprint": config.fingerprint,
        "config_fingerprint_befund_128": befund_128.config.fingerprint,
        "box_sha256": {key: box.sha256() for key, (box, _) in boxes.items()},
        "source_sha256": sha256_of(ctx.source),
    }
    print(f"add-on cause, boxes {json.dumps(head['box_sha256'])}", flush=True)
    cache = run.Cache(run.OUT / f"_{args.name}_cache.jsonl", head, args.resume)
    names = study.target_names(1)
    t0 = time.perf_counter()
    variants: dict[str, Any] = {}
    all_cells: list[dict[str, Any]] = []
    n_star_quadratic: dict[str, int | None] = {}
    for key, (box, varied) in boxes.items():
        pool = evaluated(ctx, box, varied, config.pool_size, config.pool_seed)
        test = evaluated(ctx, box, varied, config.test_size, config.test_seed)
        payload = {
            "x_pool": pool["x"],
            "y_pool": pool["y"],
            "x_test": test["x"],
            "y_test": test["y"],
            "admissible": ~test["oversaturated"],
            "available": np.arange(config.pool_size),
            "names": names,
            "groups": ratio_groups(test["ratio"]),
        }
        tasks = [
            {
                "id": f"{key}-{model}-n{n}-seed{seed}",
                "variant": key,
                "model": model,
                "n": int(n),
                "seed": int(seed),
                "settings": settings if model == "mlp" else None,
            }
            for model in MODELS
            for n in config.budgets
            for seed in config.seeds
        ]
        cells = run.run_fits(tasks, payload, cache, args.workers)
        tables = {model: study.learning_table(cells, config, model) for model in MODELS}
        n_star = {model: tables[model][1] for model in MODELS}
        n_star_quadratic[key] = n_star["quadratic"]
        variants[key] = {
            "description": VARIANTS[key],
            "varied": list(varied),
            "coefficients_quadratic": int(
                study.make_model("quadratic", None, 0).features(np.zeros((1, len(varied)))).shape[1]
            ),
            "test_set": f"in-box Sobol test set, variant {key}",
            "points": {"pool": config.pool_size, "test": config.test_size},
            "oversaturated": {
                "pool": int(pool["oversaturated"].sum()),
                "test": int(test["oversaturated"].sum()),
            },
            "capacity_ratio_groups": {"pool": shares(pool["ratio"]), "test": shares(test["ratio"])},
            "n_star": n_star,
            "table": {model: tables[model][0] for model in MODELS},
            "by_capacity_ratio": group_table(cells, config),
        }
        if key == "V-c":
            variants[key]["control"] = control_against_main(cells, n_star, box)
        print(f"{key}: N* {json.dumps(n_star)}", flush=True)
        for model in MODELS:
            for n in config.budgets:
                row = tables[model][0][n]
                print(
                    f"  {key} {model:9s} N={n:4d}  exhaust P95 "
                    f"{row['exhaust_p95_k']['median']:8.4f} K  fuel P95 "
                    f"{row['fuel_p95_percent']['median']:8.4f} %  seeds "
                    f"{row['seeds_meeting_both']}/{len(config.seeds)}  met={row['met']}",
                    flush=True,
                )
        all_cells += cells
    control = variants["V-c"]["control"]
    result = cause_reading(n_star_quadratic, bool(control["reproduces"]))
    print("control:", json.dumps(control))
    print("reading K:", json.dumps(result), "(DRY RUN, not a result)" if args.dry_run else "")
    target = (run.OUT if args.dry_run else RUNS_DIR) / f"{args.name}.json.gz"
    keep = ("id", "variant", "model", "n", "seed", "metrics", "metrics_by_group", "final_loss")
    run.write_gz(
        target,
        {
            "provenance": provenance(ignore_paths=[target]),
            "header": head,
            "source": "python rebuild (no Dyssol run)",
            "added_after_the_fact": "decision of the project lead, 2026-11-24, after the "
            "results of dyssol_freshgas_v1_analysis were known",
            "materials": ctx.materials,
            "design": {
                "pool_seed": config.pool_seed,
                "test_seed": config.test_seed,
                "budgets": list(config.budgets),
                "seeds": list(config.seeds),
                "models": list(MODELS),
                "ratio_edges": [str(e) for e in RATIO_EDGES],
            },
            "mlp_settings": {**settings, "hidden": list(settings["hidden"])},
            "thresholds": {
                "exhaust_p95_k": config.threshold_exhaust_k,
                "fuel_p95_percent": config.threshold_fuel_percent,
            },
            "variants": variants,
            "reading": result,
            "workers": args.workers,
            "elapsed_seconds": time.perf_counter() - t0,
            "peak_memory_mb_per_worker": max(c["peak_memory_mb"] for c in all_cells),
            "cells": [{k: c[k] for k in keep if k in c} for c in all_cells],
        },
    )


if __name__ == "__main__":
    main()
