r"""Pilot A1.3: fine-tuning a pretrained LSTM on the pilot plant PP (N = 2 and 16 runs).

Question of study A1: how much does a simulator help a data-poor model when the plant is
structurally different, and from how many plant runs on no longer? The pilot fixes the
fine-tuning setting that the main run then applies UNCHANGED to every plant variant, and measures
run times. It is run on PP only (the variant that is not part of the main run), so that the
setting is not tuned on a variant the main run reports.

Written before the run (this file is committed before it is executed; clean tree, ``git/dirty``
is logged). Nothing below is changed after the first look at a result.

Design (fixed here)
-------------------
* Plants: P0 pool for the pretraining (excited design of ``DataSpec.new_study``, 64 train / 32
  validation / 32 test runs, 8 h at 25 s, master seed ``PRETRAIN_SEED``); PP pool for the
  adaptation (16 / 8 / 8 runs, master seeds ``POOL_SEEDS``). Process config
  ``greybox_gab_placeholder.yaml`` with the gas-velocity projection ON, as in Befunde 51/58
  (fingerprint ``bddca37f716f5cfc`` at 25 s, checked). Targets energy target, product moisture,
  bed temperature; horizon 36 (900 s), history 32, stride 8.
* Model: ONE LSTM (hidden 64, 1 layer, the ``TorchSequenceConfig`` defaults) with the three
  targets -- not the per-target composite of Befund 26 (``CompositeSequenceModel`` has no
  ``fine_tune``); a deliberate difference from Befunde 51/58, named in the report. Pretrained on
  P0 with early stopping on the P0 validation runs (defaults: 200 epochs, patience 20).
* Fine-tuning: ``TransferSequenceModel.fine_tune`` from the pretrained weights, the scalers of the
  pretraining kept (``build_datasets_with_scalers``), a new AdamW per cell, no validation set
  inside the fit (full epoch budget, last epoch kept -- the setting at small N). Grid: freeze
  {none ("full"), core frozen ("head_only")} x learning rate {1e-4, 3e-4, 1e-3, 3e-3} x epochs
  {5, 10, 20, 40, 80} x N {2, 16} x 3 PP pools (master seeds 1-3; the model seed is the
  ``model_seed`` of the pool's seed bundle).
* SELECTION. The pilot is the only place where validation runs of the plant exist (the 8
  validation runs of the PP pool; they are NOT used inside any fit). Criterion C of a cell: the
  mean over the three channels of
  ``median over runs of the RMSE at h = 36 (physical units) / standard deviation of the channel
  in the PP validation measurements``. Setting = (freeze, learning rate, epochs); score = mean of
  C over the two N and the three pools. The setting with the lowest score is selected, one
  setting for both N (the main run has no validation runs at small N either). If the minimum
  lies at the edge of the grid in learning rate or epochs this is reported (the grid is NOT
  extended afterwards without a new decision). The TEST runs of PP are evaluated ONLY for the
  selected setting and the references, after the selection.
* Test set named: the 8 excited test runs of the PP pool (``constant_run_fraction`` of
  ``DataSpec`` default, i.e. the excited design with its share of constant runs); no constant-run
  holdout in the pilot.
* References (val and test): persistence, the simulator alone (``SimulatorOnlyModel``), the
  pretrained model without any adaptation ("zero shot").
* Run times (the second purpose; the first-use cost of torch in every process is measured apart
  by ``warm_up`` and is not part of any timed fit): data generation (P0 pool, PP pool), the simulator channels of
  the hybrid model, pretraining, fine-tuning per epoch and per cell, a from-scratch fit and a
  hybrid fit at N = 16 with a PROVISIONAL budget of ``PROVISIONAL_EPOCHS`` epochs without
  validation (timing only; their metrics are recorded but not a result: the budget of these two
  methods in the main run is a separate decision), inference time. The grid runs on 8 worker
  processes with 1 torch thread each; the main process uses ``MAIN_THREADS`` threads.
* Also recorded: the deviation measure of every plant variant (reference excitation, reserved
  seed) for the design of the main run.

    tools/run_here.sh scripts/run_a1_pilot_pp.py            # the run (writes the artifact)
    tools/run_here.sh scripts/run_a1_pilot_pp.py --smoke    # tiny dry run, writes nothing
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import platform
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import CONFIG_PATH, provenance, require_clean_tree, write_artifact

from twinlink_ml.data.splits import SplitName
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.evaluation.experiments import DataSpec, build_datasets
from twinlink_ml.evaluation.plant_data import PlantData, generate_plant_data
from twinlink_ml.evaluation.plant_deviation import plant_deviation
from twinlink_ml.models.baselines import PersistenceModel
from twinlink_ml.models.hybrid import SimulatorOnlyModel, with_simulator_channels
from twinlink_ml.models.torch_sequence import TorchSequenceConfig
from twinlink_ml.models.transfer import (
    FreezeSpec,
    TransferSequenceModel,
    build_datasets_with_scalers,
    load_pretrained,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.plant_variants import PLANT_VARIANTS
from twinlink_ml.utils.checkpoints import save_checkpoint
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent

OUTPUT_NAME = "a1_pilot_pp"
PRETRAIN_SEED = 101
POOL_SEEDS = (1, 2, 3)
SIZES = (2, 16)
HORIZON = 36
TARGETS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
FREEZE_GRID = {"full": FreezeSpec(), "head_only": FreezeSpec(core=True)}
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3)
EPOCHS = (5, 10, 20, 40, 80)
PROVISIONAL_EPOCHS = 100
WORKERS = 8
MAIN_THREADS = 4
FINGERPRINT = "bddca37f716f5cfc"
VARIANTS_FOR_DEVIATION = ("P0", "P1", "P2a", "P2b", "P3", "PP")


# --------------------------------------------------------------------------- set-up


@dataclasses.dataclass(frozen=True)
class Plan:
    """Sizes of the run; the smoke run is the same computation, smaller."""

    smoke: bool
    pretrain_spec: DataSpec
    pool_spec: DataSpec
    deviation_spec: DataSpec
    sizes: tuple[int, ...]
    pools: tuple[int, ...]
    learning_rates: tuple[float, ...]
    epochs: tuple[int, ...]
    freezes: tuple[str, ...]
    provisional_epochs: int
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
            pools=(1,),
            learning_rates=(1e-3,),
            epochs=(2, 3),
            freezes=("full", "head_only"),
            provisional_epochs=2,
            pretrain_epochs=3,
        )
    window = {"history_length": 32, "horizon": HORIZON, "stride": 8}
    return Plan(
        smoke=False,
        pretrain_spec=DataSpec.new_study(n_train_max=64, n_val=32, n_test=32, **window, **common),
        pool_spec=DataSpec.new_study(n_train_max=16, n_val=8, n_test=8, **window, **common),
        deviation_spec=DataSpec.new_study(**common),
        sizes=SIZES,
        pools=POOL_SEEDS,
        learning_rates=LEARNING_RATES,
        epochs=EPOCHS,
        freezes=tuple(FREEZE_GRID),
        provisional_epochs=PROVISIONAL_EPOCHS,
        pretrain_epochs=200,
    )


def load_config() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    return dataclasses.replace(
        config,
        numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True),
    )


def generate(config: Any, plant: str, spec: DataSpec, seed: int) -> PlantData:
    data = generate_plant_data(config, PLANT_VARIANTS[plant], spec, master_seed=seed)
    if not fingerprints_equivalent(data.base_config.fingerprint, FINGERPRINT):
        raise RuntimeError(
            f"process fingerprint {data.base_config.fingerprint!r}, expected {FINGERPRINT!r}"
        )
    return data


# ------------------------------------------------------------------------ metrics


def channel_std(data: PlantData, split: SplitName) -> dict[str, float]:
    """Standard deviation of each target in the measurements of the runs of ``split``."""
    names = list(data.trajectory.metadata["output_names"])
    ids = list(data.split.as_mapping[split])
    rows = [data.trajectory.run_id.index(run) for run in ids]
    return {
        target: float(np.std(data.trajectory.measurements[rows][..., names.index(target)]))
        for target in TARGETS
    }


def evaluate(model: Any, dataset: Any, std: dict[str, float]) -> dict[str, Any]:
    """RMSE (median over runs, physical) at h = 1 and the full horizon, criterion C, timing."""
    horizon = int(dataset.config.horizon)
    result = RolloutEvaluator(horizons=(1, horizon)).evaluate(model, dataset)
    rmse = {
        target: {
            "h1": float(result.rmse_run_median_physical[target][1]),
            f"h{horizon}": float(result.rmse_run_median_physical[target][horizon]),
        }
        for target in TARGETS
    }
    nrmse = {target: float(result.nrmse_run_spread[target][horizon][0]) for target in TARGETS}
    criterion = float(np.mean([rmse[t][f"h{horizon}"] / std[t] for t in TARGETS]))
    return {
        "horizon": horizon,
        "rmse_run_median_physical": rmse,
        "nrmse_run_median": nrmse,
        "criterion_C": criterion,
        "inference_us_per_window": float(result.inference_us_per_window),
        "n_windows": int(result.n_windows),
        "n_runs": int(result.n_runs),
    }


# ------------------------------------------------------------------------- workers

_WORKER: dict[str, Any] = {}


def warm_up() -> float:
    """First-use cost of torch (kernel set-up, here also slow file access), measured apart.

    One forward/backward pass of a small LSTM with an AdamW step and gradient clipping, so that
    the timed fits do not contain the one-off initialisation of the process.
    """
    import torch

    started = time.perf_counter()
    net = torch.nn.LSTM(input_size=8, hidden_size=64, batch_first=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        output, _ = net(torch.randn(16, 32, 8))
        output.pow(2).mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    return time.perf_counter() - started


def _worker_init(plan: Plan, checkpoint: str) -> None:
    import torch

    torch.set_num_threads(1)
    _WORKER.update(
        plan=plan, checkpoint=checkpoint, config=load_config(), data={}, warmup_s=warm_up()
    )


def _pool_data(pool_seed: int) -> PlantData:
    cache = _WORKER["data"]
    if pool_seed not in cache:
        cache[pool_seed] = generate(_WORKER["config"], "PP", _WORKER["plan"].pool_spec, pool_seed)
    return cache[pool_seed]


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """One fine-tuning cell: (pool, N, freeze, lr, epochs); val metrics, test only on request."""
    data = _pool_data(cell["pool"])
    model = load_pretrained(Path(_WORKER["checkpoint"]))
    assert model.pretrained_scalers is not None
    train, validation, test = build_datasets_with_scalers(data, cell["n"], model.pretrained_scalers)
    assert validation is not None
    started = time.perf_counter()
    model.fine_tune(
        train,
        seed=data.seeds.model_seed,
        learning_rate=cell["lr"],
        max_epochs=cell["epochs"],
        freeze=FREEZE_GRID[cell["freeze"]],
    )
    elapsed = time.perf_counter() - started
    out: dict[str, Any] = {
        **cell,
        "finetune_s": elapsed,
        "worker_warmup_s": _WORKER["warmup_s"],
        "n_train_windows": int(train.n_windows),
        "final_train_loss": float(model.training_history[-1]["train_loss"]),
        "val": evaluate(model, validation, channel_std(data, SplitName.VAL)),
    }
    if cell.get("eval_test"):
        out["test"] = evaluate(model, test, channel_std(data, SplitName.TEST))
    return out


def run_cells(plan: Plan, checkpoint: Path, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    workers = min(WORKERS, len(cells))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(plan, str(checkpoint))
    ) as pool:
        return list(pool.map(run_cell, cells, chunksize=1))


# --------------------------------------------------------------------- the stages


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

    std = channel_std(data, SplitName.TEST)
    in_distribution = evaluate(model, test, std)

    path = workdir / "pretrained_p0_lstm.pt"
    save_checkpoint(
        path,
        model,
        model_label="a1_pilot_pretrained_p0_lstm",
        run_uid="a1_pilot_pretrained_p0_lstm",
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


def grid_cells(plan: Plan) -> list[dict[str, Any]]:
    return [
        {"pool": pool, "n": n, "freeze": freeze, "lr": lr, "epochs": epochs}
        for pool, n, freeze, lr, epochs in itertools.product(
            plan.pools, plan.sizes, plan.freezes, plan.learning_rates, plan.epochs
        )
    ]


def select(plan: Plan, results: list[dict[str, Any]]) -> dict[str, Any]:
    """The setting with the lowest mean criterion over N and pools (rule in the docstring)."""
    scores: dict[tuple[str, float, int], list[float]] = {}
    for r in results:
        c = r["val"]["criterion_C"]
        scores.setdefault((r["freeze"], r["lr"], r["epochs"]), []).append(
            c if np.isfinite(c) else np.inf
        )
    table = [
        {"freeze": f, "lr": lr, "epochs": ep, "score": float(np.mean(v)), "n_cells": len(v)}
        for (f, lr, ep), v in scores.items()
    ]
    table.sort(key=lambda row: row["score"])
    best = table[0]
    return {
        "selected": best,
        "ranking_top10": table[:10],
        "table": table,
        "lr_at_grid_edge": best["lr"] in (min(plan.learning_rates), max(plan.learning_rates))
        and len(plan.learning_rates) > 1,
        "epochs_at_grid_edge": best["epochs"] in (min(plan.epochs), max(plan.epochs))
        and len(plan.epochs) > 1,
    }


def references(plan: Plan, config: Any, checkpoint: Path) -> list[dict[str, Any]]:
    """Persistence, simulator alone and the pretrained model without adaptation, val and test."""
    out = []
    for pool in plan.pools:
        data = generate(config, "PP", plan.pool_spec, pool)
        hybrid_data = with_simulator_channels(data)
        pretrained = load_pretrained(checkpoint)
        assert pretrained.pretrained_scalers is not None
        _, p_val, p_test = build_datasets_with_scalers(
            data, plan.sizes[0], pretrained.pretrained_scalers
        )
        _, own_val, own_test = build_datasets(data, plan.sizes[0])
        sim_train, sim_val, sim_test = build_datasets(hybrid_data, plan.sizes[0])
        persistence = PersistenceModel()
        persistence.fit(own_val, seed=0)  # parameter free
        simulator_only = SimulatorOnlyModel()
        simulator_only.fit(sim_train, seed=0)
        std_val, std_test = channel_std(data, SplitName.VAL), channel_std(data, SplitName.TEST)
        assert p_val is not None and own_val is not None and sim_val is not None
        for name, model, val, test in (
            ("persistence", persistence, own_val, own_test),
            ("simulator_only", simulator_only, sim_val, sim_test),
            ("pretrained_zero_shot", pretrained, p_val, p_test),
        ):
            out.append(
                {
                    "reference": name,
                    "pool": pool,
                    "val": evaluate(model, val, std_val),
                    "test": evaluate(model, test, std_test),
                }
            )
    return out


def provisional_fits(plan: Plan, config: Any, timings: dict[str, Any]) -> dict[str, Any]:
    """From-scratch and hybrid LSTM at N = 16 (pool 1), provisional budget: TIMING ONLY."""
    import torch

    torch.set_num_threads(1)
    timings["warmup_1thread_s"] = warm_up()
    n = plan.sizes[-1]
    pool = plan.pools[0]
    data = generate(config, "PP", plan.pool_spec, pool)
    started = time.perf_counter()
    hybrid_data = with_simulator_channels(data)
    timings["simulator_channels_s"] = time.perf_counter() - started
    timings["simulator_channels_runs"] = data.trajectory.n_runs
    channel_info = hybrid_data.trajectory.metadata["simulator_channels"]

    out: dict[str, Any] = {"n": n, "pool": pool, "epochs": plan.provisional_epochs}
    std_test = channel_std(data, SplitName.TEST)
    for name, source in (("from_scratch", data), ("hybrid", hybrid_data)):
        train, _, test = build_datasets(source, n)
        model = TransferSequenceModel(
            TorchSequenceConfig(core="lstm", max_epochs=plan.provisional_epochs, patience=0)
        )
        started = time.perf_counter()
        model.fit(train, seed=source.seeds.model_seed)
        elapsed = time.perf_counter() - started
        out[name] = {
            "fit_s": elapsed,
            "s_per_epoch": elapsed / plan.provisional_epochs,
            "n_train_windows": int(train.n_windows),
            "n_exogenous": len(train.exogenous_names),
            "test_NOT_A_RESULT": evaluate(model, test, std_test),
        }
    out["simulator_channels"] = channel_info
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
    workdir_context = tempfile.TemporaryDirectory(prefix="a1_pilot_")
    workdir = Path(workdir_context.name)
    wall_start = time.perf_counter()

    checkpoint, pretraining = pretrain(plan, config, workdir, timings)
    print(f"pretrained: {timings['pretrain_fit_s']:.0f} s", flush=True)

    started = time.perf_counter()
    cells = grid_cells(plan)
    grid = run_cells(plan, checkpoint, cells)
    timings["grid_wall_s"] = time.perf_counter() - started
    timings["grid_cells"] = len(cells)
    timings["grid_workers"] = min(WORKERS, len(cells))
    timings["finetune_cell_s_total"] = float(sum(r["finetune_s"] for r in grid))
    timings["worker_warmup_s"] = sorted({round(r["worker_warmup_s"], 2) for r in grid})
    timings["finetune_s_per_epoch"] = {
        str(n): float(
            np.sum([r["finetune_s"] for r in grid if r["n"] == n])
            / np.sum([r["epochs"] for r in grid if r["n"] == n])
        )
        for n in plan.sizes
    }
    print(f"grid: {len(cells)} cells in {timings['grid_wall_s']:.0f} s", flush=True)

    selection = select(plan, grid)
    chosen = selection["selected"]
    print("selected:", chosen, flush=True)

    final_cells = [
        {
            "pool": pool,
            "n": n,
            "freeze": chosen["freeze"],
            "lr": chosen["lr"],
            "epochs": chosen["epochs"],
            "eval_test": True,
        }
        for pool, n in itertools.product(plan.pools, plan.sizes)
    ]
    final = run_cells(plan, checkpoint, final_cells)
    by_key = {(r["pool"], r["n"], r["freeze"], r["lr"], r["epochs"]): r for r in grid}
    for r in final:
        earlier = by_key[(r["pool"], r["n"], r["freeze"], r["lr"], r["epochs"])]
        # determinism across processes: the re-run of the selected cell reproduces its val metric
        r["val_equals_grid_run"] = r["val"]["criterion_C"] == earlier["val"]["criterion_C"]

    refs = references(plan, config, checkpoint)
    provisional = provisional_fits(plan, config, timings)
    deviations = {
        name: plant_deviation(config, PLANT_VARIANTS[name], plan.deviation_spec).as_dict()
        for name in VARIANTS_FOR_DEVIATION
    }
    timings["wall_total_s"] = time.perf_counter() - wall_start

    payload = {
        "_provenance": provenance(),
        "design": {
            "pretrain_seed": PRETRAIN_SEED,
            "pool_seeds": list(plan.pools),
            "sizes": list(plan.sizes),
            "freeze_grid": list(plan.freezes),
            "learning_rates": list(plan.learning_rates),
            "epochs": list(plan.epochs),
            "pretrain_spec": plan.pretrain_spec.as_parameters(),
            "pool_spec": plan.pool_spec.as_parameters(),
            "targets": list(TARGETS),
            "horizon": HORIZON,
            "provisional_epochs": plan.provisional_epochs,
            "process_fingerprint": FINGERPRINT,
            "workers": WORKERS,
            "main_threads": MAIN_THREADS,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "pretraining": pretraining,
        "selection": selection,
        "grid": grid,
        "selected_final": final,
        "references": refs,
        "provisional_fits_timing_only": provisional,
        "deviation_of_plants": deviations,
        "timings": timings,
    }
    if plan.smoke:
        print("smoke run finished; keys:", sorted(payload))
        print("timings:", {k: v for k, v in timings.items() if not isinstance(v, dict)})
        workdir_context.cleanup()
        return
    write_artifact(OUTPUT_NAME, payload)
    workdir_context.cleanup()


if __name__ == "__main__":
    main()
