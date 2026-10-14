r"""Confirmation of the pilot settings on the NEW data (preset version 2): small, not a full pilot.

The pilots (``a1_pilot_pp.json``, ``a1_pilot2_pp_pp2.json``) were run on data of preset version 1 (no
per-run cap of the gas velocity; the safety projection acted on 4-8 % of the samples, see
``a1_action_limits_pilot1.json``). The excitation of all studies has since been changed (Befund 86: per-run
cap of the gas velocity at ``u_t`` of the initial diameter, projection kept as a counted guard,
``DataSpec.new_study`` preset version 2). This script checks whether the settings chosen in pilot 2 still hold
on the new data, on the two pilot plants PP and PP2. Not a re-selection.

Written before the run (this file is committed before it is executed; clean tree, ``git/dirty`` is logged).
The rule below is fixed here, before any computation.

What is run (everything with ``DataSpec.new_study``, i.e. version 2; pilot machinery of
``scripts/run_a1_pilot2.py`` reused unchanged: same models, windows, process config, pilot seeds, criterion C)
-------------------------------------------------------------------------------------------------------
* Pretraining anew on the new P0 data: 256 train + 32 validation + 224 test runs, master seed 201 (the pilot
  seed; a different data set now), early stopping on the validation runs.
* PP and PP2 pools, 16 / 8 / 8 runs, master seeds 11-13, N = 2 and 16: 12 cells per setting, as in pilot 2.
* Fine-tuning (``fine_tune``, all parameters trained, pretraining scalers kept): the selected setting of pilot 2
  (learning rate 3e-4, 320 epochs) and its direct neighbours in the grid -- learning rate {1e-4, 3e-4, 1e-3} x
  epochs {160, 320, 640}, i.e. 9 settings including the diagonal neighbours.
* Step budgets: "from scratch" 4000 steps (the pilot-2 optimum of that method) and ONE neighbour, the neighbour
  with the lower pilot-2 score of that method (8000 steps: 0.2735 against 0.2825 at 2000); "hybrid" 8000 steps
  and one neighbour, the one with the lower pilot-2 score (16000: 0.2616 against 0.2945 at 4000).
* The score of a setting is the mean criterion C over the 12 cells (as in the pilot; validation runs of the pilot
  plants, which are in no fit). No test run is evaluated.

RULE (fixed in advance, per part)
---------------------------------
The chosen setting stays if its score is at most 3 % above the best score among the settings tested for that
part (fine-tuning: the 9 settings; "from scratch": its two budgets; "hybrid": its two budgets). Otherwise it is
REPORTED (the main run is then not started on the old setting without a decision). The result per part and the
scores of all tested settings are recorded.

Also recorded per pool: how often the projection acts (the guard; expected 0) and the simulator's own
projection in the hybrid channel.

    tools/run_here.sh scripts/run_a1_confirm.py            # the run (writes the artifact)
    tools/run_here.sh scripts/run_a1_confirm.py --smoke    # tiny dry run, writes nothing
"""

from __future__ import annotations

import argparse
import itertools
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

import run_a1_pilot2 as pilot2
from _common import provenance, require_clean_tree, write_artifact
from run_a1_pilot_pp import generate, load_config

from twinlink_ml.evaluation.experiments import NEW_STUDY_PRESET_VERSION
from twinlink_ml.models.hybrid import with_simulator_channels
from twinlink_ml.simulation.plant_variants import PLANT_VARIANTS

OUTPUT_NAME = "a1_confirm_preset2"
CHOSEN_FINETUNE = {"freeze": "full", "lr": 3e-4, "epochs": 320}
FINETUNE_LRS = (1e-4, 3e-4, 1e-3)
FINETUNE_EPOCHS = (160, 320, 640)
CHOSEN_STEPS = {"from_scratch": 4000, "hybrid": 8000}
TESTED_STEPS = {"from_scratch": (4000, 8000), "hybrid": (8000, 16000)}
TOLERANCE = 0.03


def smoke_overrides() -> dict[str, Any]:
    """Smoke sizes: the same computation, tiny."""
    return {
        "lrs": (1e-3, 3e-3),
        "epochs": (1, 2),
        "chosen": {"freeze": "full", "lr": 1e-3, "epochs": 1},
        "tested_steps": {"from_scratch": (2, 4), "hybrid": (4, 8)},
        "chosen_steps": {"from_scratch": 2, "hybrid": 4},
    }


def judge(score_by_setting: dict[Any, float], chosen: Any) -> dict[str, Any]:
    best_setting = min(score_by_setting, key=lambda k: score_by_setting[k])
    best = score_by_setting[best_setting]
    chosen_score = score_by_setting[chosen]
    return {
        "chosen": str(chosen),
        "chosen_score": chosen_score,
        "best_tested": str(best_setting),
        "best_score": best,
        "excess_over_best_pct": 100.0 * (chosen_score / best - 1.0),
        "stays": bool(chosen_score <= (1.0 + TOLERANCE) * best),
        "all_scores": {str(k): v for k, v in score_by_setting.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny dry run, writes nothing")
    arguments = parser.parse_args()
    plan = pilot2.make_plan(arguments.smoke)
    if not plan.smoke:
        require_clean_tree()
    lrs, epochs = FINETUNE_LRS, FINETUNE_EPOCHS
    chosen_ft = dict(CHOSEN_FINETUNE)
    tested_steps, chosen_steps = dict(TESTED_STEPS), dict(CHOSEN_STEPS)
    if plan.smoke:
        o = smoke_overrides()
        lrs, epochs, chosen_ft = o["lrs"], o["epochs"], o["chosen"]
        tested_steps, chosen_steps = o["tested_steps"], o["chosen_steps"]
    if (
        plan.pool_spec.new_study_version != NEW_STUDY_PRESET_VERSION
        or NEW_STUDY_PRESET_VERSION != 2
    ):
        raise RuntimeError("the data are not of the new-study preset version 2")

    import torch

    config = load_config()
    timings: dict[str, Any] = {}
    workdir_context = tempfile.TemporaryDirectory(prefix="a1_confirm_")
    workdir = Path(workdir_context.name)
    started_all = time.perf_counter()

    checkpoint, pretraining = pilot2.pretrain(plan, config, workdir, timings)
    print(f"pretrained: {timings['pretrain_fit_s']:.0f} s", flush=True)

    # the guard, per pool
    guards: dict[str, Any] = {}
    p0 = generate(config, "P0", plan.pretrain_spec, pilot2.PRETRAIN_SEED)
    guards["P0_pretraining_pool"] = {
        "preset_version": p0.spec.new_study_version,
        **p0.projection_guard,
    }
    for plant, pool in itertools.product(pilot2.PLANTS, plan.pools):
        data = generate(config, plant, plan.pool_spec, pool)
        hybrid = with_simulator_channels(data)
        guards[f"{plant}_pool{pool}"] = {
            "preset_version": data.spec.new_study_version,
            **data.projection_guard,
            "simulator_in_hybrid_channel": {
                k: v
                for k, v in hybrid.trajectory.metadata["simulator_channels"].items()
                if k.startswith("simulator_") or k in ("n_nominal_fallback", "n_stationary")
            },
        }
    violated = {k: g for k, g in guards.items() if g["projected_steps"] != 0}
    print("projection guard:", {k: g["projected_steps"] for k, g in guards.items()}, flush=True)

    # fine-tuning settings
    ft_tasks = [
        {
            "kind": "finetune",
            "plant": plant,
            "pool": pool,
            "n": n,
            "freeze": "full",
            "lr": lr,
            "epochs": ep,
        }
        for plant, pool, n, lr, ep in itertools.product(
            pilot2.PLANTS, plan.pools, plan.sizes, lrs, epochs
        )
    ]
    # step budgets
    budget_tasks = [
        {
            "kind": "budget",
            "plant": plant,
            "pool": pool,
            "n": n,
            "method": method,
            "steps": steps,
        }
        for method in pilot2.METHODS
        for plant, pool, n, steps in itertools.product(
            pilot2.PLANTS, plan.pools, plan.sizes, tested_steps[method]
        )
    ]
    tasks = sorted(ft_tasks + budget_tasks, key=lambda t: -t.get("steps", t.get("epochs", 0) * 10))
    started = time.perf_counter()
    results = pilot2.run_tasks(plan, checkpoint, tasks)
    timings["confirm_grid_wall_s"] = time.perf_counter() - started
    timings["confirm_cells"] = len(results)

    finetune = [r for r in results if r["kind"] == "finetune"]
    budget = [r for r in results if r["kind"] == "budget"]
    ft_scores = {
        (lr, ep): float(
            np.mean(
                [r["val"]["criterion_C"] for r in finetune if r["lr"] == lr and r["epochs"] == ep]
            )
        )
        for lr, ep in itertools.product(lrs, epochs)
    }
    verdicts = {"finetune": judge(ft_scores, (chosen_ft["lr"], chosen_ft["epochs"]))}
    for method in pilot2.METHODS:
        scores = {
            s: float(
                np.mean(
                    [
                        r["val"]["criterion_C"]
                        for r in budget
                        if r["method"] == method and r["steps"] == s
                    ]
                )
            )
            for s in tested_steps[method]
        }
        verdicts[method] = judge(scores, chosen_steps[method])
    per_plant = {
        "finetune": {
            plant: {
                f"{lr}/{ep}": float(
                    np.mean(
                        [
                            r["val"]["criterion_C"]
                            for r in finetune
                            if r["plant"] == plant and r["lr"] == lr and r["epochs"] == ep
                        ]
                    )
                )
                for lr, ep in itertools.product(lrs, epochs)
            }
            for plant in pilot2.PLANTS
        }
    }
    all_stay = all(v["stays"] for v in verdicts.values())
    timings["wall_total_s"] = time.perf_counter() - started_all
    payload = {
        "_provenance": provenance(),
        "design": {
            "rule": f"chosen setting stays if its score <= {1 + TOLERANCE:.2f} x the best tested score",
            "preset_version_expected": 2,
            "chosen_finetune": chosen_ft,
            "chosen_steps": chosen_steps,
            "tested_finetune_lrs": list(lrs),
            "tested_finetune_epochs": list(epochs),
            "tested_steps": {k: list(v) for k, v in tested_steps.items()},
            "pool_seeds": list(plan.pools),
            "pretrain_seed": pilot2.PRETRAIN_SEED,
            "pretrain_spec": plan.pretrain_spec.as_parameters(),
            "pool_spec": plan.pool_spec.as_parameters(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "verdicts": verdicts,
        "all_settings_stay": all_stay,
        "per_plant_scores": per_plant,
        "projection_guard_per_pool": guards,
        "projection_guard_violations": violated,
        "pretraining": pretraining,
        "cells": results,
        "deviation_of_pilot_plants": {
            name: pilot2.plant_deviation(
                config, PLANT_VARIANTS[name], plan.deviation_spec
            ).as_dict()
            for name in pilot2.PLANTS
        },
        "timings": timings,
    }
    print(
        "verdicts:",
        {k: (v["stays"], round(v["excess_over_best_pct"], 2)) for k, v in verdicts.items()},
    )
    if plan.smoke:
        print("smoke run finished; all stay:", all_stay, "| guard violations:", len(violated))
        workdir_context.cleanup()
        return
    write_artifact(OUTPUT_NAME, payload)
    workdir_context.cleanup()


if __name__ == "__main__":
    main()
