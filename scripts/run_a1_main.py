r"""Main run of study A1: the value of a simulator for a data-poor model on a different plant.

Design, hypotheses and the criteria are in ``docs/NN_a1_transfer_main_run_design.md``; this file only
carries them out. Written before the run (committed before it is executed; clean tree, ``git/dirty`` is
logged in the provenance of the artifact).

What is computed
----------------
* Plants P0, P1, P2a, P2b, P3 x seeds 1-5 (master seed = plant pool, noise and model seed; the same seeds for
  every method and plant) x N = 2, 4, 8, 16, 32, 64 plant runs (nested prefixes of the pool of 64 train runs).
* Methods, one LSTM (hidden 64, one layer, three targets) for every neural method:
  from scratch (4000 gradient steps), hybrid (open P0 simulation as three more input channels, 16000 steps:
  the edge of the tested range, chosen by the pre-registered rule of the confirmation; the choice is driven by
  the strongly scattering N = 2 cells), plus a SENSITIVITY without a hypothesis of its own: the hybrid with
  8000 steps at N = 2 and 4 (method ``hybrid_steps8000``),
  pretraining + fine-tuning (one pretrained model per seed on P0 data, 256 train + 32 validation runs, master
  seed 100 + seed; then ``fine_tune``, all parameters, learning rate 3e-4, 320 epochs, pretraining scalers
  kept), the simulator alone, and the baselines persistence and ARX. The settings are those of pilot 2
  (``a1_pilot2_pp_pp2.json``) as confirmed on the new data (``a1_confirm_preset2.json``; the hybrid budget
  16000 is the neighbour that beat 8000 by 13 %).
* Every time measured in this run (training, inference) carries ``timing_context``: taken under parallel load,
  NOT a result. The inference time per method is measured once after the run on an idle machine (one thread,
  warm-up excluded).
* Data only with ``DataSpec.new_study`` (preset version 2: planned future set points, own solid-fraction
  process, per-run cap of the gas velocity at ``u_t`` of the initial diameter, projection kept as a counted
  guard). Per plant and seed one pool of 128 runs: 64 train, 32 reserved validation runs (in no fit), 32 test
  runs; plus one fixed pool of constant runs per plant (master seed ``HOLDOUT_SEED``, 16 test runs).
* Test sets (named in every record): ``excited`` (the 32 excited test runs of the plant's own pool, primary) and
  ``constant_holdout`` (16 constant runs of the same plant; secondary, never pooled with the primary).
* Metrics per record: median over test runs of the RMSE in physical units at h = 1 and h = 36 per channel,
  the run-median NRMSE at h = 36, the criterion C of the pilots, inference time per window; training time,
  number of epochs and gradient steps.

Guard (checked BEFORE any training of a unit and recorded in every record)
-------------------------------------------------------------------------
Every pool a unit generates must carry ``data/new_study_version`` = 2 and must show 0 projected steps. If not,
the unit writes the file ``STOP`` into the work directory and raises; the orchestrator halts the run and reports
(running units end at their next record). The projection of the simulator inside the hybrid channel is
recorded per pool (not a stop criterion).

Units and resume
----------------
A unit is (pretrain, seed), or (scratch | hybrid | finetune | baselines, plant, seed). Records are appended to
``<workdir>/units/<unit>.jsonl`` one at a time; with ``--resume`` finished records are skipped, so the same
command continues a stopped run. The work directory lies outside the repository (the tree stays clean). At the
end the records are collected into ``artifacts/runs/a1_main_records.json`` (never overwritten).

    # first start (detached, with a log):
    tools/run_here.sh scripts/run_a1_main.py --workdir C:/Users/<user>/tl_a1_main --workers 8
    # continue after a stop:
    tools/run_here.sh scripts/run_a1_main.py --workdir C:/Users/<user>/tl_a1_main --workers 8 --resume
    tools/run_here.sh scripts/run_a1_main.py --smoke --workdir <tmp>    # tiny dry run, writes no artifact.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact
from run_a1_pilot_pp import TARGETS, channel_std, evaluate, generate, load_config, warm_up

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation.experiments import NEW_STUDY_PRESET_VERSION, DataSpec, build_datasets
from twinlink_ml.evaluation.plant_data import PlantData
from twinlink_ml.models.baselines import LinearARXModel, PersistenceModel
from twinlink_ml.models.hybrid import SimulatorOnlyModel, with_simulator_channels
from twinlink_ml.models.torch_sequence import TorchSequenceConfig
from twinlink_ml.models.transfer import (
    FreezeSpec,
    TransferSequenceModel,
    build_datasets_with_scalers,
    load_pretrained,
)
from twinlink_ml.utils.checkpoints import save_checkpoint
from twinlink_ml.utils.tracking import environment_fingerprint

OUTPUT_NAME = "a1_main_records"
PLANTS = ("P0", "P1", "P2a", "P2b", "P3")
SEEDS = (1, 2, 3, 4, 5)
SIZES = (2, 4, 8, 16, 32, 64)
HOLDOUT_SEED = 999
PRETRAIN_SEED_OFFSET = 100
FINETUNE_LEARNING_RATE = 3e-4
FINETUNE_EPOCHS = 320
STEPS = {"from_scratch": 4000, "hybrid": 16000}
SENSITIVITY_METHOD = "hybrid_steps8000"
SENSITIVITY_STEPS = 8000
SENSITIVITY_SIZES = (2, 4)
TIMING_CONTEXT = {
    "parallel_load": True,
    "use": "NOT a result: measured while other jobs ran on the machine; the inference time is "
    "measured separately after the run",
}
SCRATCH_LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_WORKERS = 8
CHECKPOINT_WAIT_S = 4 * 3600


@dataclasses.dataclass(frozen=True)
class Plan:
    """Sizes of the run; the smoke run is the same computation, smaller."""

    smoke: bool
    pretrain_spec: DataSpec
    pool_spec: DataSpec
    holdout_spec: DataSpec
    plants: tuple[str, ...]
    seeds: tuple[int, ...]
    sizes: tuple[int, ...]
    finetune_epochs: int
    steps: dict[str, int]
    sensitivity_sizes: tuple[int, ...]
    sensitivity_steps: int
    pretrain_epochs: int


def make_plan(smoke: bool) -> Plan:
    common = {"target_names": TARGETS}
    if smoke:
        window = {"history_length": 8, "horizon": 4, "run_duration_s": 1_500.0}
        pool = DataSpec.new_study(n_train_max=4, n_val=2, n_test=2, **window, **common)
        return Plan(
            smoke=True,
            pretrain_spec=pool,
            pool_spec=pool,
            holdout_spec=dataclasses.replace(
                pool, n_train_max=2, n_val=2, n_test=2, constant_run_fraction=1.0
            ),
            plants=("P0", "P3"),
            seeds=(1,),
            sizes=(2, 4),
            finetune_epochs=2,
            steps={"from_scratch": 4, "hybrid": 6},
            sensitivity_sizes=(2, 4),
            sensitivity_steps=3,
            pretrain_epochs=2,
        )
    window = {"history_length": 32, "horizon": 36, "stride": 8}
    pool = DataSpec.new_study(n_train_max=64, n_val=32, n_test=32, **window, **common)
    return Plan(
        smoke=False,
        pretrain_spec=DataSpec.new_study(n_train_max=256, n_val=32, n_test=224, **window, **common),
        pool_spec=pool,
        holdout_spec=dataclasses.replace(
            pool, n_train_max=8, n_val=8, n_test=16, constant_run_fraction=1.0
        ),
        plants=PLANTS,
        seeds=SEEDS,
        sizes=SIZES,
        finetune_epochs=FINETUNE_EPOCHS,
        steps=dict(STEPS),
        sensitivity_sizes=SENSITIVITY_SIZES,
        sensitivity_steps=SENSITIVITY_STEPS,
        pretrain_epochs=200,
    )


# ------------------------------------------------------------------------- helpers


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class GuardViolationError(RuntimeError):
    """A pool is not of preset version 2 or shows projected steps: the run must stop."""


def unit_name(unit: dict[str, Any]) -> str:
    if unit["kind"] == "pretrain":
        return f"pretrain__s{unit['seed']}"
    return f"{unit['kind']}__{unit['plant']}__s{unit['seed']}"


def append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_default) + "\n")


def _default(value: object) -> object:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_guard(data: PlantData, label: str, workdir: Path) -> dict[str, Any]:
    """Version 2 and 0 projected steps -- else STOP file and an exception."""
    guard = {"preset_version": data.spec.new_study_version, **data.projection_guard}
    if guard["preset_version"] != NEW_STUDY_PRESET_VERSION or guard["projected_steps"] != 0:
        reason = {"pool": label, **guard}
        (workdir / "STOP").write_text(json.dumps(reason), encoding="utf-8")
        raise GuardViolationError(f"guard violated at {label}: {reason}")
    return guard


def stop_requested(workdir: Path) -> bool:
    return (workdir / "STOP").exists()


_PROCESS: dict[str, Any] = {}


def process_setup() -> dict[str, Any]:
    """Once per worker process: one torch thread, first-use cost measured apart, environment record."""
    if not _PROCESS:
        import torch

        torch.set_num_threads(1)
        _PROCESS["warmup_s"] = warm_up()
        _PROCESS["env"] = environment_fingerprint()
        _PROCESS["config"] = load_config()
    return _PROCESS


def pools(
    plant: str, seed: int, plan: Plan, workdir: Path
) -> tuple[PlantData, PlantData, dict[str, Any]]:
    config = process_setup()["config"]
    excited = generate(config, plant, plan.pool_spec, seed)
    holdout = generate(config, plant, plan.holdout_spec, HOLDOUT_SEED)
    guards = {
        "excited": check_guard(excited, f"{plant}/s{seed}/excited", workdir),
        "constant_holdout": check_guard(holdout, f"{plant}/constant_holdout", workdir),
    }
    return excited, holdout, guards


def holdout_test(holdout: PlantData, scalers: Any, window: Any) -> TrajectoryWindowDataset:
    return TrajectoryWindowDataset.from_trajectory(
        holdout.trajectory, holdout.split, SplitName.TEST, window, scalers=scalers
    )


def result_records(
    unit: dict[str, Any],
    method: str,
    n: int,
    model: Any,
    tests: dict[str, tuple[Any, dict[str, float]]],
    data: PlantData,
    holdout: PlantData,
    guards: dict[str, Any],
    fit: dict[str, Any],
) -> list[dict[str, Any]]:
    """One record per test set."""
    params = {
        **data.spec.as_parameters(),
        **data.fingerprints,
        **data.seeds.as_parameters(),
        "holdout/seed": HOLDOUT_SEED,
        "holdout/split_digest": holdout.split.digest,
        "n_train": n,
        "model/label": method,
        **{f"model/{k}": v for k, v in model.describe().items()},
    }
    records = []
    for test_set, (dataset, std) in tests.items():
        records.append(
            {
                "kind": "result",
                "plant": unit["plant"],
                "seed": unit["seed"],
                "method": method,
                "n": n,
                "test_set": test_set,
                "metrics": evaluate(model, dataset, std),
                "fit": fit,
                "timing_context": TIMING_CONTEXT,
                "params": params,
                "projection_guard": guards,
            }
        )
    return records


def test_sets(
    test_excited: Any, std_excited: dict[str, float], test_hold: Any, std_hold: dict[str, float]
) -> dict[str, tuple[Any, dict[str, float]]]:
    return {
        "excited": (test_excited, std_excited),
        "constant_holdout": (test_hold, std_hold),
    }


def unit_start(unit: dict[str, Any], out: Path, guards: dict[str, Any] | None = None) -> None:
    setup = process_setup()
    append(
        out,
        {
            "kind": "unit_start",
            "unit": unit_name(unit),
            "time": datetime.now().isoformat(),
            "warmup_s": setup["warmup_s"],
            "env": setup["env"],
            "projection_guard": guards,
            "pid": os.getpid(),
        },
    )


def done_keys(records: list[dict[str, Any]]) -> set[tuple[str, int]]:
    seen: dict[tuple[str, int], set[str]] = {}
    for r in records:
        if r.get("kind") == "result":
            seen.setdefault((r["method"], r["n"]), set()).add(r["test_set"])
    return {key for key, sets in seen.items() if sets >= {"excited", "constant_holdout"}}


# ------------------------------------------------------------------------- the units


def run_pretrain(unit: dict[str, Any], plan: Plan, workdir: Path) -> dict[str, Any]:
    seed = unit["seed"]
    checkpoint = workdir / "ckpt" / f"pretrained_s{seed}.pt"
    out = workdir / "units" / f"{unit_name(unit)}.jsonl"
    if checkpoint.exists():
        return {"unit": unit_name(unit), "skipped": True}
    config = process_setup()["config"]
    master = PRETRAIN_SEED_OFFSET + seed
    data = generate(config, "P0", plan.pretrain_spec, master)
    guard = check_guard(data, f"P0/pretrain/s{seed}", workdir)
    unit_start(unit, out, {"pretraining_pool": guard})
    train, validation, test = build_datasets(data, data.spec.n_train_max)
    model = TransferSequenceModel(TorchSequenceConfig(core="lstm", max_epochs=plan.pretrain_epochs))
    started = time.perf_counter()
    model.fit(train, seed=data.seeds.model_seed, validation=validation)
    fit_s = time.perf_counter() - started
    diagnostics = model.training_diagnostics()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    save_checkpoint(
        temporary,
        model,
        model_label=f"a1_main_pretrained_p0_lstm_s{seed}",
        run_uid=f"a1_main_pretrained_p0_lstm_s{seed}",
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
    temporary.replace(checkpoint)
    append(
        out,
        {
            "kind": "pretrain",
            "seed": seed,
            "master_seed": master,
            "fit_s": fit_s,
            "timing_context": TIMING_CONTEXT,
            "training": {k: v for k, v in diagnostics.items() if not k.startswith("loss_history")},
            "p0_test_in_distribution": evaluate(model, test, channel_std(data, SplitName.TEST)),
            "params": {
                **data.spec.as_parameters(),
                **data.fingerprints,
                **data.seeds.as_parameters(),
            },
            "projection_guard": {"pretraining_pool": guard},
        },
    )
    return {"unit": unit_name(unit), "fit_s": fit_s}


def run_neural(unit: dict[str, Any], plan: Plan, workdir: Path) -> dict[str, Any]:
    """from_scratch, hybrid or finetune for one plant and seed, all N."""
    kind, plant, seed = unit["kind"], unit["plant"], unit["seed"]
    out = workdir / "units" / f"{unit_name(unit)}.jsonl"
    started_unit = time.perf_counter()
    excited, holdout, guards = pools(plant, seed, plan, workdir)
    unit_start(unit, out, guards)
    already = done_keys(read_records(out))
    window = excited.spec.window
    std_ex, std_hold = channel_std(excited, SplitName.TEST), channel_std(holdout, SplitName.TEST)

    if kind == "hybrid":
        source, hold_source = with_simulator_channels(excited), with_simulator_channels(holdout)
        method = "hybrid"
    else:
        source, hold_source = excited, holdout
        method = "from_scratch" if kind == "scratch" else "pretrain_finetune"
    pretrained_path = workdir / "ckpt" / f"pretrained_s{seed}.pt"
    if kind == "finetune":
        waited = 0.0
        while not pretrained_path.exists():
            if stop_requested(workdir) or waited > CHECKPOINT_WAIT_S:
                raise RuntimeError(f"no pretrained model for seed {seed}")
            time.sleep(30)
            waited += 30

    jobs = [(method, n, plan.steps.get(method)) for n in plan.sizes]
    if kind == "hybrid":
        jobs += [
            (SENSITIVITY_METHOD, n, plan.sensitivity_steps)
            for n in plan.sensitivity_sizes
            if n in plan.sizes
        ]
    for job_method, n, job_steps in jobs:
        if stop_requested(workdir):
            raise GuardViolationError("STOP file present")
        if (job_method, n) in already:
            continue
        if kind == "finetune":
            model = load_pretrained(pretrained_path)
            assert model.pretrained_scalers is not None
            train, _, test = build_datasets_with_scalers(source, n, model.pretrained_scalers)
            started = time.perf_counter()
            model.fine_tune(
                train,
                seed=source.seeds.model_seed,
                learning_rate=FINETUNE_LEARNING_RATE,
                max_epochs=plan.finetune_epochs,
                freeze=FreezeSpec(),
            )
            fit_s = time.perf_counter() - started
            epochs = plan.finetune_epochs
            steps_per_epoch = math.ceil(train.n_windows / BATCH_SIZE)
        else:
            train, _, test = build_datasets(source, n)
            steps_per_epoch = math.ceil(train.n_windows / BATCH_SIZE)
            epochs = max(1, round(job_steps / steps_per_epoch))
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
            fit_s = time.perf_counter() - started
        fit = {
            "fit_s": fit_s,
            "epochs": int(epochs),
            "gradient_steps": int(epochs * steps_per_epoch),
            "n_train_windows": int(train.n_windows),
            "final_train_loss": float(model.training_history[-1]["train_loss"]),
        }
        sets = test_sets(test, std_ex, holdout_test(hold_source, train.scalers, window), std_hold)
        for record in result_records(
            unit, job_method, n, model, sets, excited, holdout, guards, fit
        ):
            append(out, record)
        log(f"{unit_name(unit)} {job_method} N={n} done (fit {fit_s:.0f} s)")
    return {"unit": unit_name(unit), "wall_s": time.perf_counter() - started_unit}


def run_baselines(unit: dict[str, Any], plan: Plan, workdir: Path) -> dict[str, Any]:
    """Persistence, ARX per N, the simulator alone; also the pool records (guard, simulator channel)."""
    plant, seed = unit["plant"], unit["seed"]
    out = workdir / "units" / f"{unit_name(unit)}.jsonl"
    started_unit = time.perf_counter()
    excited, holdout, guards = pools(plant, seed, plan, workdir)
    unit_start(unit, out, guards)
    already = done_keys(read_records(out))
    window = excited.spec.window
    std_ex, std_hold = channel_std(excited, SplitName.TEST), channel_std(holdout, SplitName.TEST)

    hybrid_excited, hybrid_holdout = (
        with_simulator_channels(excited),
        with_simulator_channels(holdout),
    )
    if not any(r.get("kind") == "pool" for r in read_records(out)):
        for name, data, hybrid in (
            ("excited", excited, hybrid_excited),
            ("constant_holdout", holdout, hybrid_holdout),
        ):
            append(
                out,
                {
                    "kind": "pool",
                    "plant": plant,
                    "seed": seed,
                    "pool": name,
                    "projection_guard": data.projection_guard,
                    "preset_version": data.spec.new_study_version,
                    "simulator_in_hybrid_channel": hybrid.trajectory.metadata["simulator_channels"],
                    "params": {**data.spec.as_parameters(), **data.fingerprints},
                },
            )

    def one(
        method: str,
        n: int,
        model: Any,
        train: Any,
        test: Any,
        hold_source: Any,
        fit: dict[str, Any],
    ) -> None:
        sets = test_sets(test, std_ex, holdout_test(hold_source, train.scalers, window), std_hold)
        for record in result_records(unit, method, n, model, sets, excited, holdout, guards, fit):
            append(out, record)

    # parameter-free methods, n = 0
    if ("persistence", 0) not in already:
        train, _, test = build_datasets(excited, plan.sizes[0])
        model = PersistenceModel()
        model.fit(train, seed=0)
        one("persistence", 0, model, train, test, holdout, {"fit_s": 0.0})
    if ("simulator_only", 0) not in already:
        train, _, test = build_datasets(hybrid_excited, plan.sizes[0])
        model = SimulatorOnlyModel()
        model.fit(train, seed=0)
        one("simulator_only", 0, model, train, test, hybrid_holdout, {"fit_s": 0.0})
    for n in plan.sizes:
        if stop_requested(workdir):
            raise GuardViolationError("STOP file present")
        if ("arx", n) in already:
            continue
        train, _, test = build_datasets(excited, n)
        model = LinearARXModel()
        started = time.perf_counter()
        model.fit(train, seed=excited.seeds.model_seed)
        one("arx", n, model, train, test, holdout, {"fit_s": time.perf_counter() - started})
    return {"unit": unit_name(unit), "wall_s": time.perf_counter() - started_unit}


def run_unit(unit: dict[str, Any], plan: Plan, workdir_text: str) -> dict[str, Any]:
    """Entry point of a worker process."""
    workdir = Path(workdir_text)
    log(f"START {unit_name(unit)}")
    try:
        if stop_requested(workdir):
            raise GuardViolationError("STOP file present")
        if unit["kind"] == "pretrain":
            result = run_pretrain(unit, plan, workdir)
        elif unit["kind"] == "baselines":
            result = run_baselines(unit, plan, workdir)
        else:
            result = run_neural(unit, plan, workdir)
    except GuardViolationError:
        raise
    except Exception:
        log(f"ERROR in {unit_name(unit)}:\n{traceback.format_exc()}")
        raise
    log(f"DONE {unit_name(unit)} {result}")
    return result


# ------------------------------------------------------------------------ orchestration


def all_units(plan: Plan) -> list[dict[str, Any]]:
    """Longest first; the pretrain units first, the fine-tune units after the long ones."""
    units: list[dict[str, Any]] = [{"kind": "pretrain", "seed": s} for s in plan.seeds]
    for kind in ("hybrid", "scratch"):
        units += [{"kind": kind, "plant": p, "seed": s} for s in plan.seeds for p in plan.plants]
    for kind in ("finetune", "baselines"):
        units += [{"kind": kind, "plant": p, "seed": s} for s in plan.seeds for p in plan.plants]
    return units


def unit_finished(unit: dict[str, Any], plan: Plan, workdir: Path) -> bool:
    if unit["kind"] == "pretrain":
        return (workdir / "ckpt" / f"pretrained_s{unit['seed']}.pt").exists()
    records = read_records(workdir / "units" / f"{unit_name(unit)}.jsonl")
    keys = done_keys(records)
    if unit["kind"] == "baselines":
        need = {("persistence", 0), ("simulator_only", 0)} | {("arx", n) for n in plan.sizes}
    else:
        method = {"hybrid": "hybrid", "scratch": "from_scratch", "finetune": "pretrain_finetune"}[
            unit["kind"]
        ]
        need = {(method, n) for n in plan.sizes}
        if unit["kind"] == "hybrid":
            need |= {(SENSITIVITY_METHOD, n) for n in plan.sensitivity_sizes if n in plan.sizes}
    return need <= keys


def collect(plan: Plan, workdir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for unit in all_units(plan):
        records += read_records(workdir / "units" / f"{unit_name(unit)}.jsonl")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny dry run, no artifact")
    arguments = parser.parse_args()
    plan = make_plan(arguments.smoke)
    workers = max(1, min(arguments.workers, MAX_WORKERS))
    workdir: Path = arguments.workdir
    if NEW_STUDY_PRESET_VERSION != 2 or plan.pool_spec.new_study_version != 2:
        log("the data spec is not of preset version 2: abort")
        return 2
    if not plan.smoke:
        require_clean_tree()
    provenance_file = workdir / "provenance.json"
    if workdir.exists() and any(workdir.iterdir()) and not arguments.resume:
        log(f"{workdir} is not empty: use --resume to continue or choose a new directory")
        return 2
    workdir.mkdir(parents=True, exist_ok=True)
    now = provenance()
    if provenance_file.exists():
        earlier = json.loads(provenance_file.read_text(encoding="utf-8"))
        if earlier["git/commit"] != now["git/commit"] and not plan.smoke:
            log("the commit differs from the first start of this run: abort")
            return 2
    else:
        provenance_file.write_text(json.dumps(now, indent=1), encoding="utf-8")
    if (workdir / "STOP").exists():
        log("STOP file present: resolve the cause, remove the file, then --resume")
        return 3

    units = [u for u in all_units(plan) if not unit_finished(u, plan, workdir)]
    log(f"{len(units)} unit(s) to run on {workers} worker(s); workdir {workdir}")
    failed: list[str] = []
    halted = False
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_unit, unit, plan, str(workdir)): unit for unit in units}
        for future in as_completed(futures):
            unit = futures[future]
            try:
                future.result()
            except GuardViolationError as error:
                halted = True
                log(f"HALT: {error}")
                for other in futures:
                    other.cancel()
            except Exception as error:
                failed.append(unit_name(unit))
                log(f"FAILED {unit_name(unit)}: {error!r}")
    if halted:
        log("run halted by the guard; see STOP in the work directory")
        return 3
    remaining = [u for u in all_units(plan) if not unit_finished(u, plan, workdir)]
    if failed or remaining:
        log(f"not finished: failed {failed}, unfinished {[unit_name(u) for u in remaining]}")
        return 1
    log(f"all units finished in {time.perf_counter() - started:.0f} s")
    if plan.smoke:
        log(f"smoke run finished; {len(collect(plan, workdir))} lines collected")
        return 0
    first = json.loads(provenance_file.read_text(encoding="utf-8"))
    write_artifact(
        OUTPUT_NAME,
        {"_provenance": first, "n_workers": workers, "records": collect(plan, workdir)},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
