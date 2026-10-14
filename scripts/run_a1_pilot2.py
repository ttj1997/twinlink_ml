r"""Pilot A1.3b: fine-tuning setting and training budget, selected on TWO pilot plants (PP and PP2).

Why a second pilot (decision of the project lead after the first one, ``a1_pilot_pp.json``): the
first pilot selected the smallest learning rate and the fewest epochs of its grid on PP, whose
summary deviation from P0 (0.13) is about half that of the main variants; a very cautious setting
chosen there adapts too little under strong deviation and would artificially support HA2. PP2 is a
second pilot plant with a summary deviation of about 0.3 (``plant_variants.PP2``, set a priori
from the deviation measure alone). Neither PP nor PP2 is part of the main run. The first artifact
stays as it is.

Written before the run (this file is committed before it is executed; clean tree, ``git/dirty`` is
logged). Nothing below is changed after the first look at a result.

Design (fixed here)
-------------------
* Data, model, windows, process config: as in ``scripts/run_a1_pilot_pp.py`` (one LSTM, hidden 64,
  1 layer, three targets, horizon 36, history 32, stride 8, 25 s, projection ON, fingerprint checked).
* Pretraining (as decided for the main run): P0 pool of 256 train + 32 validation + 224 test runs
  (a pool of 512 keeps the Sobol balance; the large test block is only the in-distribution check of
  the pretrained model), the same excitation as the plants (``DataSpec.new_study``), early stopping on
  the validation runs (defaults: 200 epochs, patience 20). Master seed ``PRETRAIN_SEED`` = 201, NOT one
  of the seeds planned for the main run.
* Plant pools: PP and PP2, 16 train / 8 validation / 8 test runs each, master seeds ``POOL_SEEDS`` =
  11-13 (own seeds: not those of the first pilot, not those planned for the main run). N = 2 and 16.
* SELECTION rule, for both parts: the criterion C of a cell is the mean over the three channels of
  ``median over runs of the RMSE at h = 36 (physical units) / standard deviation of the channel in
  the plant's validation measurements``; the validation runs of the pilot plants are in no fit. The
  score of a setting is the mean of C over PP AND PP2, the two N and the three pools (12 cells); the
  lowest score is selected. TEST runs are evaluated only for the selected settings and the references,
  after the selection. Test set named: the 8 excited test runs of each pilot pool.
* Part 1, fine-tuning (``TransferSequenceModel.fine_tune``, scalers of the pretraining kept, new AdamW,
  no validation inside the fit, last epoch kept): freeze {none, core frozen} x learning rate {1e-5,
  3e-5, 1e-4, 3e-4, 1e-3, 3e-3} x epochs {1, 2, 3, 5, 10, 20, 40, 80}. EXTENSION UPWARD while the
  selected setting lies at the upper edge in learning rate or epochs: learning rate 1e-2, 3e-2, 1e-1;
  epochs 160, 320, 640 (each new level crossed with the whole other axis and both freeze options; at
  most three rounds). The lower edge (1e-5, 1 epoch) is the end of the grid by decision and is not
  extended; if the selection ends there it is reported.
* Part 2, training budget of "from scratch" and "hybrid" (a new LSTM on the N plant runs, scalers
  estimated on them, learning rate 1e-3, no validation inside the fit, last epoch kept; the hybrid
  gets the three simulator channels of ``models.hybrid``): budget in GRADIENT STEPS {500, 1000, 2000,
  4000}, realised as ``epochs = max(1, round(steps / steps_per_epoch))`` with ``steps_per_epoch =
  ceil(windows / 256)`` (the realised number of steps is recorded). One setting for both methods: the
  score of a budget is the mean over both methods of their mean C (12 cells each); each method's own
  best budget is recorded too. Same upward rule: while the selected budget is the largest, add 8000,
  then 16000 (at most two rounds).
* References on validation and test of every pilot pool: persistence, the simulator alone, the
  pretrained model without adaptation.
* Run times as in the first pilot (first-use cost of torch measured apart); 8 worker processes with
  one torch thread, the main process ``MAIN_THREADS``. The deviation measure of PP and PP2 is recorded.

    tools/run_here.sh scripts/run_a1_pilot2.py            # the run (writes the artifact)
    tools/run_here.sh scripts/run_a1_pilot2.py --smoke    # tiny dry run, writes nothing
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import platform
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact
from run_a1_pilot_pp import (
    FREEZE_GRID,
    TARGETS,
    channel_std,
    evaluate,
    generate,
    load_config,
    warm_up,
)

from twinlink_ml.data.splits import SplitName
from twinlink_ml.evaluation.experiments import DataSpec, build_datasets
from twinlink_ml.evaluation.plant_data import PlantData
from twinlink_ml.evaluation.plant_deviation import plant_deviation
from twinlink_ml.models.baselines import PersistenceModel
from twinlink_ml.models.hybrid import SimulatorOnlyModel, with_simulator_channels
from twinlink_ml.models.torch_sequence import TorchSequenceConfig
from twinlink_ml.models.transfer import (
    TransferSequenceModel,
    build_datasets_with_scalers,
    load_pretrained,
)
from twinlink_ml.simulation.plant_variants import PLANT_VARIANTS
from twinlink_ml.utils.checkpoints import save_checkpoint

OUTPUT_NAME = "a1_pilot2_pp_pp2"
PRETRAIN_SEED = 201
POOL_SEEDS = (11, 12, 13)
PLANTS = ("PP", "PP2")
SIZES = (2, 16)
LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
EXTRA_LEARNING_RATES = (1e-2, 3e-2, 1e-1)
EPOCHS = (1, 2, 3, 5, 10, 20, 40, 80)
EXTRA_EPOCHS = (160, 320, 640)
STEP_BUDGETS = (500, 1000, 2000, 4000)
EXTRA_STEP_BUDGETS = (8000, 16000)
BATCH_SIZE = 256
SCRATCH_LEARNING_RATE = 1e-3
WORKERS = 8
MAIN_THREADS = 8
MAX_EXTENSION_ROUNDS = 3
METHODS = ("from_scratch", "hybrid")


@dataclasses.dataclass(frozen=True)
class Plan:
    """Sizes and grids of the run; the smoke run is the same computation, smaller."""

    smoke: bool
    pretrain_spec: DataSpec
    pool_spec: DataSpec
    deviation_spec: DataSpec
    sizes: tuple[int, ...]
    pools: tuple[int, ...]
    learning_rates: tuple[float, ...]
    extra_learning_rates: tuple[float, ...]
    epochs: tuple[int, ...]
    extra_epochs: tuple[int, ...]
    step_budgets: tuple[int, ...]
    extra_step_budgets: tuple[int, ...]
    max_rounds: int
    pretrain_epochs: int


def make_plan(smoke: bool) -> Plan:
    common = {"target_names": TARGETS}
    if smoke:
        window = {"history_length": 8, "horizon": 4, "run_duration_s": 1_500.0}
        return Plan(
            smoke=True,
            pretrain_spec=DataSpec.new_study(n_train_max=4, n_val=2, n_test=2, **window, **common),
            pool_spec=DataSpec.new_study(n_train_max=4, n_val=2, n_test=2, **window, **common),
            deviation_spec=DataSpec.new_study(run_duration_s=1_500.0, **common),
            sizes=(2, 4),
            pools=(11,),
            learning_rates=(1e-4, 1e-3),
            extra_learning_rates=(3e-3,),
            epochs=(1, 2),
            extra_epochs=(4,),
            step_budgets=(2, 4),
            extra_step_budgets=(8,),
            max_rounds=1,
            pretrain_epochs=2,
        )
    window = {"history_length": 32, "horizon": 36, "stride": 8}
    return Plan(
        smoke=False,
        pretrain_spec=DataSpec.new_study(n_train_max=256, n_val=32, n_test=224, **window, **common),
        pool_spec=DataSpec.new_study(n_train_max=16, n_val=8, n_test=8, **window, **common),
        deviation_spec=DataSpec.new_study(**common),
        sizes=SIZES,
        pools=POOL_SEEDS,
        learning_rates=LEARNING_RATES,
        extra_learning_rates=EXTRA_LEARNING_RATES,
        epochs=EPOCHS,
        extra_epochs=EXTRA_EPOCHS,
        step_budgets=STEP_BUDGETS,
        extra_step_budgets=EXTRA_STEP_BUDGETS,
        max_rounds=MAX_EXTENSION_ROUNDS,
        pretrain_epochs=200,
    )


# ------------------------------------------------------------------------- workers

_WORKER: dict[str, Any] = {}


def _worker_init(plan: Plan, checkpoint: str) -> None:
    import torch

    torch.set_num_threads(1)
    _WORKER.update(
        plan=plan,
        checkpoint=checkpoint,
        config=load_config(),
        data={},
        hybrid={},
        warmup_s=warm_up(),
    )


def _data(plant: str, pool: int) -> PlantData:
    cache = _WORKER["data"]
    if (plant, pool) not in cache:
        cache[(plant, pool)] = generate(_WORKER["config"], plant, _WORKER["plan"].pool_spec, pool)
    return cache[(plant, pool)]


def _hybrid(plant: str, pool: int) -> PlantData:
    cache = _WORKER["hybrid"]
    if (plant, pool) not in cache:
        cache[(plant, pool)] = with_simulator_channels(_data(plant, pool))
    return cache[(plant, pool)]


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    """One cell: kind "finetune" (freeze, lr, epochs) or "budget" (method, steps)."""
    data = _data(task["plant"], task["pool"])
    base: dict[str, Any] = {**task, "worker_warmup_s": _WORKER["warmup_s"]}
    if task["kind"] == "finetune":
        model = load_pretrained(Path(_WORKER["checkpoint"]))
        assert model.pretrained_scalers is not None
        train, validation, test = build_datasets_with_scalers(
            data, task["n"], model.pretrained_scalers
        )
        epochs = task["epochs"]
        started = time.perf_counter()
        model.fine_tune(
            train,
            seed=data.seeds.model_seed,
            learning_rate=task["lr"],
            max_epochs=epochs,
            freeze=FREEZE_GRID[task["freeze"]],
        )
        elapsed = time.perf_counter() - started
        base["realised_steps"] = int(epochs * math.ceil(train.n_windows / BATCH_SIZE))
    else:
        source = _hybrid(task["plant"], task["pool"]) if task["method"] == "hybrid" else data
        train, validation, test = build_datasets(source, task["n"])
        steps_per_epoch = math.ceil(train.n_windows / BATCH_SIZE)
        epochs = max(1, round(task["steps"] / steps_per_epoch))
        model = TransferSequenceModel(
            TorchSequenceConfig(
                core="lstm",
                max_epochs=epochs,
                patience=0,
                learning_rate=SCRATCH_LEARNING_RATE,
                batch_size=BATCH_SIZE,
            )
        )
        started = time.perf_counter()
        model.fit(train, seed=source.seeds.model_seed)
        elapsed = time.perf_counter() - started
        base["epochs"] = int(epochs)
        base["realised_steps"] = int(epochs * steps_per_epoch)
    assert validation is not None
    base.update(
        fit_s=elapsed,
        n_train_windows=int(train.n_windows),
        final_train_loss=float(model.training_history[-1]["train_loss"]),
        val=evaluate(model, validation, channel_std(data, SplitName.VAL)),
    )
    if task.get("eval_test"):
        base["test"] = evaluate(model, test, channel_std(data, SplitName.TEST))
    return base


def run_tasks(plan: Plan, checkpoint: Path, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tasks:
        return []
    with ProcessPoolExecutor(
        max_workers=min(WORKERS, len(tasks)),
        initializer=_worker_init,
        initargs=(plan, str(checkpoint)),
    ) as pool:
        return list(pool.map(run_task, tasks, chunksize=1))


# -------------------------------------------------------------------- the stages


def pretrain(
    plan: Plan, config: Any, workdir: Path, timings: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    import torch

    torch.set_num_threads(MAIN_THREADS)
    timings["warmup_main_s"] = warm_up()
    started = time.perf_counter()
    data = generate(config, "P0", plan.pretrain_spec, PRETRAIN_SEED)
    timings["generate_p0_pretrain_pool_s"] = time.perf_counter() - started
    timings["pretrain_pool_runs"] = data.spec.pool_size
    train, validation, test = build_datasets(data, data.spec.n_train_max)
    model = TransferSequenceModel(TorchSequenceConfig(core="lstm", max_epochs=plan.pretrain_epochs))
    started = time.perf_counter()
    model.fit(train, seed=data.seeds.model_seed, validation=validation)
    timings["pretrain_fit_s"] = time.perf_counter() - started
    diagnostics = model.training_diagnostics()
    timings["pretrain_epochs_run"] = diagnostics.get("epochs_run")
    timings["pretrain_train_windows"] = int(train.n_windows)
    timings["pretrain_s_per_epoch"] = timings["pretrain_fit_s"] / max(
        len(model.training_history), 1
    )
    in_distribution = evaluate(model, test, channel_std(data, SplitName.TEST))
    path = workdir / "pretrained_p0_lstm.pt"
    save_checkpoint(
        path,
        model,
        model_label="a1_pilot2_pretrained_p0_lstm",
        run_uid="a1_pilot2_pretrained_p0_lstm",
        target_names=train.target_names,
        exogenous_names=train.exogenous_names,
        n_exogenous=len(train.exogenous_names),
        input_scaler=train.input_scaler,
        target_scaler=train.target_scaler,
        process_fingerprint=data.process_config.fingerprint,
        data_spec_params=data.spec.as_parameters(),
        seed_bundle_params=data.seeds.as_parameters(),
        git_commit=provenance()["git/commit"],
    )
    return path, {
        "model": model.describe(),
        "training": {k: v for k, v in diagnostics.items() if not k.startswith("loss_history")},
        "loss_history_train": diagnostics.get("loss_history_train"),
        "loss_history_val": diagnostics.get("loss_history_val"),
        "p0_test_in_distribution": in_distribution,
        "seed_bundle": data.seeds.as_parameters(),
        "split_digest": data.split.digest,
    }


def finetune_tasks(
    plan: Plan, freezes: tuple[str, ...], lrs: list[float], epochs: list[int], done: set[Any]
) -> list[dict[str, Any]]:
    tasks = []
    for plant, pool, n, freeze, lr, ep in itertools.product(
        PLANTS, plan.pools, plan.sizes, freezes, lrs, epochs
    ):
        key = (plant, pool, n, freeze, lr, ep)
        if key not in done:
            tasks.append(
                {
                    "kind": "finetune",
                    "plant": plant,
                    "pool": pool,
                    "n": n,
                    "freeze": freeze,
                    "lr": lr,
                    "epochs": ep,
                }
            )
    return tasks


def score_table(results: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Mean criterion C per setting (``keys`` of the cells), sorted ascending; non-finite = inf."""
    scores: dict[tuple[Any, ...], list[float]] = {}
    for r in results:
        c = r["val"]["criterion_C"]
        scores.setdefault(tuple(r[k] for k in keys), []).append(c if np.isfinite(c) else np.inf)
    table = [
        {**dict(zip(keys, setting, strict=True)), "score": float(np.mean(v)), "n_cells": len(v)}
        for setting, v in scores.items()
    ]
    table.sort(key=lambda row: row["score"])
    return table


def per_plant_scores(
    results: list[dict[str, Any]], keys: tuple[str, ...], best: dict[str, Any]
) -> dict[str, float]:
    out = {}
    for plant in PLANTS:
        cells = [
            r["val"]["criterion_C"]
            for r in results
            if r["plant"] == plant and all(r[k] == best[k] for k in keys)
        ]
        out[plant] = float(np.mean(cells))
    return out


def part1_finetune(
    plan: Plan, checkpoint: Path, timings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    freezes = tuple(FREEZE_GRID)
    lrs, epochs = list(plan.learning_rates), list(plan.epochs)
    results: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    started = time.perf_counter()
    for round_index in range(plan.max_rounds + 1):
        done = {(r["plant"], r["pool"], r["n"], r["freeze"], r["lr"], r["epochs"]) for r in results}
        tasks = finetune_tasks(plan, freezes, lrs, epochs, done)
        results += run_tasks(plan, checkpoint, tasks)
        keys = ("freeze", "lr", "epochs")
        table = score_table(results, keys)
        best = table[0]
        at_upper_lr = best["lr"] == max(lrs)
        at_upper_epochs = best["epochs"] == max(epochs)
        rounds.append(
            {
                "round": round_index,
                "n_new_cells": len(tasks),
                "learning_rates": list(lrs),
                "epochs": list(epochs),
                "selected": best,
                "at_upper_lr": at_upper_lr,
                "at_upper_epochs": at_upper_epochs,
            }
        )
        print(f"fine-tune round {round_index}: {len(tasks)} cells, selected {best}", flush=True)
        if round_index == plan.max_rounds or not (at_upper_lr or at_upper_epochs):
            break
        if at_upper_lr:
            larger = [x for x in plan.extra_learning_rates if x > max(lrs)]
            if larger:
                lrs.append(larger[0])
        if at_upper_epochs:
            larger_e = [x for x in plan.extra_epochs if x > max(epochs)]
            if larger_e:
                epochs.append(larger_e[0])
    timings["finetune_grid_wall_s"] = time.perf_counter() - started
    timings["finetune_cells"] = len(results)
    selection = {
        "selected": best,
        "per_plant_score_of_selected": per_plant_scores(results, keys, best),
        "ranking_top10": table[:10],
        "table": table,
        "rounds": rounds,
        "lr_at_lower_edge": best["lr"] == min(lrs),
        "epochs_at_lower_edge": best["epochs"] == min(epochs),
        "lr_at_upper_edge_after_extensions": best["lr"] == max(lrs),
        "epochs_at_upper_edge_after_extensions": best["epochs"] == max(epochs),
    }
    return results, selection


def part2_budget(
    plan: Plan, checkpoint: Path, timings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    budgets = list(plan.step_budgets)
    results: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    started = time.perf_counter()
    extras = list(plan.extra_step_budgets)
    for round_index in range(len(extras) + 1):
        done = {(r["plant"], r["pool"], r["n"], r["method"], r["steps"]) for r in results}
        tasks = [
            {"kind": "budget", "plant": plant, "pool": pool, "n": n, "method": method, "steps": s}
            for plant, pool, n, method, s in itertools.product(
                PLANTS, plan.pools, plan.sizes, METHODS, budgets
            )
            if (plant, pool, n, method, s) not in done
        ]
        # the longest runs first (better load balance)
        tasks.sort(key=lambda t: -t["steps"])
        results += run_tasks(plan, checkpoint, tasks)
        per_method = {
            m: score_table([r for r in results if r["method"] == m], ("steps",)) for m in METHODS
        }
        combined = [
            {
                "steps": s,
                "score": float(
                    np.mean(
                        [
                            next(x["score"] for x in per_method[m] if x["steps"] == s)
                            for m in METHODS
                        ]
                    )
                ),
            }
            for s in budgets
        ]
        combined.sort(key=lambda row: row["score"])
        best = combined[0]
        rounds.append({"round": round_index, "budgets": list(budgets), "selected": best})
        print(f"budget round {round_index}: selected {best}", flush=True)
        if best["steps"] != max(budgets) or round_index == len(extras):
            break
        budgets.append(extras[round_index])
    timings["budget_grid_wall_s"] = time.perf_counter() - started
    timings["budget_cells"] = len(results)
    selection = {
        "selected": best,
        "combined_table": combined,
        "per_method_table": per_method,
        "per_method_best_steps": {m: per_method[m][0]["steps"] for m in METHODS},
        "rounds": rounds,
        "at_lower_edge": best["steps"] == min(budgets),
        "at_upper_edge_after_extensions": best["steps"] == max(budgets),
    }
    return results, selection


def references(plan: Plan, config: Any, checkpoint: Path) -> list[dict[str, Any]]:
    out = []
    for plant, pool in itertools.product(PLANTS, plan.pools):
        data = generate(config, plant, plan.pool_spec, pool)
        hybrid_data = with_simulator_channels(data)
        pretrained = load_pretrained(checkpoint)
        assert pretrained.pretrained_scalers is not None
        _, p_val, p_test = build_datasets_with_scalers(
            data, plan.sizes[0], pretrained.pretrained_scalers
        )
        _, own_val, own_test = build_datasets(data, plan.sizes[0])
        sim_train, sim_val, sim_test = build_datasets(hybrid_data, plan.sizes[0])
        persistence = PersistenceModel()
        persistence.fit(own_val, seed=0)
        simulator_only = SimulatorOnlyModel()
        simulator_only.fit(sim_train, seed=0)
        std_val, std_test = channel_std(data, SplitName.VAL), channel_std(data, SplitName.TEST)
        for name, model, val, test in (
            ("persistence", persistence, own_val, own_test),
            ("simulator_only", simulator_only, sim_val, sim_test),
            ("pretrained_zero_shot", pretrained, p_val, p_test),
        ):
            out.append(
                {
                    "reference": name,
                    "plant": plant,
                    "pool": pool,
                    "val": evaluate(model, val, std_val),
                    "test": evaluate(model, test, std_test),
                    "simulator_channels": hybrid_data.trajectory.metadata["simulator_channels"]
                    if name == "simulator_only"
                    else None,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny dry run, writes nothing")
    arguments = parser.parse_args()
    plan = make_plan(arguments.smoke)
    if not plan.smoke:
        require_clean_tree()

    import torch

    config = load_config()
    timings: dict[str, Any] = {}
    workdir_context = tempfile.TemporaryDirectory(prefix="a1_pilot2_")
    workdir = Path(workdir_context.name)
    wall_start = time.perf_counter()

    checkpoint, pretraining = pretrain(plan, config, workdir, timings)
    print(f"pretrained: {timings['pretrain_fit_s']:.0f} s", flush=True)

    finetune, finetune_selection = part1_finetune(plan, checkpoint, timings)
    chosen = finetune_selection["selected"]
    budget, budget_selection = part2_budget(plan, checkpoint, timings)
    chosen_steps = budget_selection["selected"]["steps"]

    # test evaluation ONLY for the selected settings (after both selections)
    final_tasks: list[dict[str, Any]] = [
        {
            "kind": "finetune",
            "plant": plant,
            "pool": pool,
            "n": n,
            "freeze": chosen["freeze"],
            "lr": chosen["lr"],
            "epochs": chosen["epochs"],
            "eval_test": True,
        }
        for plant, pool, n in itertools.product(PLANTS, plan.pools, plan.sizes)
    ] + [
        {
            "kind": "budget",
            "plant": plant,
            "pool": pool,
            "n": n,
            "method": method,
            "steps": chosen_steps,
            "eval_test": True,
        }
        for plant, pool, n, method in itertools.product(PLANTS, plan.pools, plan.sizes, METHODS)
    ]
    final = run_tasks(plan, checkpoint, final_tasks)
    earlier = {
        ("finetune", r["plant"], r["pool"], r["n"], r["freeze"], r["lr"], r["epochs"]): r
        for r in finetune
    } | {("budget", r["plant"], r["pool"], r["n"], r["method"], r["steps"]): r for r in budget}
    for r in final:
        key = (
            ("finetune", r["plant"], r["pool"], r["n"], r["freeze"], r["lr"], r["epochs"])
            if r["kind"] == "finetune"
            else ("budget", r["plant"], r["pool"], r["n"], r["method"], r["steps"])
        )
        r["val_equals_selection_run"] = (
            r["val"]["criterion_C"] == earlier[key]["val"]["criterion_C"]
        )

    refs = references(plan, config, checkpoint)
    deviations = {
        name: plant_deviation(config, PLANT_VARIANTS[name], plan.deviation_spec).as_dict()
        for name in PLANTS
    }
    timings["wall_total_s"] = time.perf_counter() - wall_start
    worker_warmups = sorted({round(r["worker_warmup_s"], 2) for r in finetune + budget})
    timings["worker_warmup_s"] = worker_warmups
    for kind, cells, key in (
        ("finetune", finetune, "epochs"),
        ("budget", budget, "realised_steps"),
    ):
        timings[f"{kind}_fit_s_total"] = float(sum(r["fit_s"] for r in cells))
        timings[f"{kind}_seconds_per_{'epoch' if kind == 'finetune' else 'step'}_by_n"] = {
            str(n): float(
                np.sum([r["fit_s"] for r in cells if r["n"] == n])
                / np.sum(
                    [
                        r[key] if kind == "finetune" else r["realised_steps"]
                        for r in cells
                        if r["n"] == n
                    ]
                )
            )
            for n in plan.sizes
        }

    payload = {
        "_provenance": provenance(),
        "design": {
            "pretrain_seed": PRETRAIN_SEED,
            "pool_seeds": list(plan.pools),
            "plants": list(PLANTS),
            "sizes": list(plan.sizes),
            "pretrain_spec": plan.pretrain_spec.as_parameters(),
            "pool_spec": plan.pool_spec.as_parameters(),
            "targets": list(TARGETS),
            "scratch_learning_rate": SCRATCH_LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "main_threads": MAIN_THREADS,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "pretraining": pretraining,
        "finetune_selection": finetune_selection,
        "budget_selection": budget_selection,
        "finetune_grid": finetune,
        "budget_grid": budget,
        "selected_final": final,
        "references": refs,
        "deviation_of_pilot_plants": deviations,
        "timings": timings,
    }
    if plan.smoke:
        print("smoke run finished; keys:", sorted(payload))
        print("selected fine-tune:", chosen, "| budget:", budget_selection["selected"])
        print("timings:", {k: v for k, v in timings.items() if not isinstance(v, dict | list)})
        workdir_context.cleanup()
        return
    write_artifact(OUTPUT_NAME, payload)
    workdir_context.cleanup()


if __name__ == "__main__":
    main()
