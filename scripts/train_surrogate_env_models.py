r"""Train the GRU composite models of the learned environments U1-U3 (study docs/30_surrogate_env.md).

Pipeline of Befund 94, arm N, unchanged (``run_arch_rerun_worker.arm_spec("N")``:
``DataSpec.new_study()``, preset version 2, 25 s sample time, history 32, horizon 36, targets energy,
moisture, bed temperature; process config ``conf/process/greybox_gab_placeholder.yaml`` with the
gas-velocity projection, process fingerprint ``bddca37f716f5cfc``; own excited pool per master seed;
run-level splits; ``ValidationPolicy.FIXED_POOL``; GRU factory of ``run_arch_learning_curve_worker``,
``max_epochs`` 200; one torch thread).

Models (master seeds 1, 2, 3 = data seeds; replicate r of the study uses master seed r):

* ``n16_m0``  U1: ``n_train`` 16, model seed of the seed bundle (= the GRU of Befund 94).
* ``n64_m0``  U2: ``n_train`` 64, model seed of the seed bundle (= the GRU of Befund 94).
* ``n64_m1`` .. ``n64_m4``  the four further members of the U3 ensemble: SAME data and split as
  ``n64_m0`` (only the training seed differs: ``_derive(model_seed, "surrogate_env_ensemble", j)``),
  so the ensemble of ``n64_m0 .. n64_m4`` is trained on 64 runs, not on 5 x 64.

Reproduction check (stop condition of the study): for ``m0`` the RolloutEvaluator metrics of both
test sets (excited, constant-run holdout seed 999; own pool, planned window = the primary variant)
and the training diagnostics are compared with the records of Befund 94
(``artifacts/runs/rerun_full_N.jsonl``); the largest absolute difference is stored.

Writes ``artifacts/checkpoints/surrogate_env/gru_N_seed<s>_<tag>.pt`` (gitignored) and the record
``artifacts/runs/<name>.json`` with the SHA-256 of every checkpoint (the drivers check it), test
errors in physical units, the input ranges of the training runs and the reproduction check.

    tools/run_here.sh scripts/train_surrogate_env_models.py --name surrogate_env_models_v1 --workers 9
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

ROOT = SCRIPTS.parent
PROCESS_FINGERPRINT = "bddca37f716f5cfc"
SEEDS = (1, 2, 3)
ENSEMBLE_MEMBERS = 5
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints" / "surrogate_env"
BEFUND94_RECORDS = ROOT / "artifacts" / "runs" / "rerun_full_N.jsonl"


def model_tags() -> list[tuple[int, int, int]]:
    """``(master_seed, n_train, member)`` of every model of the study."""
    tags = [(s, 16, 0) for s in SEEDS]
    tags += [(s, 64, j) for s in SEEDS for j in range(ENSEMBLE_MEMBERS)]
    return tags


def checkpoint_file(seed: int, n_train: int, member: int) -> Path:
    return CHECKPOINT_DIR / f"gru_N_seed{seed}_n{n_train}_m{member}.pt"


def _befund94_record(seed: int, n_train: int, test_set: str) -> dict[str, Any]:
    with BEFUND94_RECORDS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            p = record["params"]
            if (
                p["model/label"] == "gru"
                and p["master_seed"] == seed
                and p["n_train"] == n_train
                and p["eval/primary"]
                and p["test_set"] == test_set
            ):
                return record
    raise KeyError((seed, n_train, test_set))


def _max_difference(new: dict[str, Any], old: dict[str, Any]) -> tuple[float, int]:
    """Largest absolute difference over the shared finite numeric (or list) entries.

    Timing entries (``inference_us_per_window``, anything with ``elapsed``) are machine-load
    measurements, not results, and are left out.
    """
    worst, n = 0.0, 0
    for key, value in new.items():
        if key not in old or "inference" in key or "elapsed" in key:
            continue
        a = np.asarray(value, dtype=np.float64).ravel()
        b = np.asarray(old[key], dtype=np.float64).ravel()
        if a.shape != b.shape:
            return math.inf, n
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            return math.inf, n
        if mask.any():
            worst = max(worst, float(np.max(np.abs(a[mask] - b[mask]))))
        n += 1
    return worst, n


def _physical_errors(model: Any, dataset: Any) -> dict[str, float]:
    """RMSE and bias per target (physical units, against the noise-free truth) at h = 1 and h = 36."""
    batch = dataset.batch(range(dataset.n_windows))
    predicted = dataset.target_scaler.inverse(model.predict(batch))
    truth = dataset.target_scaler.inverse(batch.future_targets_truth)
    out: dict[str, float] = {}
    for k, target in enumerate(dataset.target_names):
        for h in (1, predicted.shape[1]):
            err = predicted[:, h - 1, k] - truth[:, h - 1, k]
            out[f"rmse/{target}/h{h}"] = float(np.sqrt(np.mean(err**2)))
            out[f"bias/{target}/h{h}"] = float(np.mean(err))
    return out


def train_one(seed: int, n_train: int, member: int, smoke: bool) -> dict[str, Any]:
    """Train, save and check one model; returns its record."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    import run_arch_learning_curve_worker as old_worker
    import run_arch_rerun_worker as rerun_worker
    from _common import git

    from twinlink_ml.data.splits import SplitName
    from twinlink_ml.data.windows import TrajectoryWindowDataset
    from twinlink_ml.evaluation import ValidationPolicy, build_datasets
    from twinlink_ml.evaluation.evaluator import RolloutEvaluator
    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
    from twinlink_ml.utils.checkpoints import save_checkpoint
    from twinlink_ml.utils.seeding import _derive

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(old_worker.CONFIG_PATH)
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )
    spec = rerun_worker.arm_spec("N", smoke=smoke)
    started = time.perf_counter()
    own = rerun_worker._generate(config, spec, seed)
    holdout = rerun_worker._generate(
        config, rerun_worker.holdout_spec(spec), rerun_worker.HOLDOUT_SEED
    )
    if not smoke and own.process_config.fingerprint != PROCESS_FINGERPRINT:
        raise RuntimeError(f"process/fingerprint {own.process_config.fingerprint!r}")
    size = n_train if not smoke else rerun_worker.SMOKE_SIZE
    train, validation, test = build_datasets(own, size, policy=ValidationPolicy.FIXED_POOL)
    model = old_worker._build_model(
        old_worker._architecture_factory("gru"), 200 if not smoke else 2
    )
    training_seed = (
        own.seeds.model_seed
        if member == 0
        else _derive(own.seeds.model_seed, "surrogate_env_ensemble", member)
    )
    fit_start = time.perf_counter()
    model.fit(train, seed=training_seed, validation=validation)
    fit_s = time.perf_counter() - fit_start
    holdout_test = TrajectoryWindowDataset.from_trajectory(
        holdout.trajectory, holdout.split, SplitName.TEST, spec.window, scalers=train.scalers
    )
    path = checkpoint_file(seed, n_train, member)
    if smoke:
        path = path.with_name("_smoke_" + path.name)
    if path.exists():
        raise SystemExit(f"{path} exists -- checkpoints are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        path,
        model,
        model_label="gru",
        run_uid=path.stem,
        target_names=train.target_names,
        exogenous_names=train.exogenous_names,
        n_exogenous=len(train.exogenous_names),
        input_scaler=train.input_scaler,
        target_scaler=train.target_scaler,
        process_fingerprint=own.process_config.fingerprint,
        data_spec_params=spec.as_parameters(),
        seed_bundle_params=own.seeds.as_parameters(),
        git_commit=git("rev-parse", "HEAD"),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    evaluator = RolloutEvaluator(horizons=(1, spec.horizon))
    metrics = {
        "excited": dict(evaluator.evaluate(model, test).as_flat_metrics()),
        "constant_holdout": dict(evaluator.evaluate(model, holdout_test).as_flat_metrics()),
    }
    diagnostics = {f"training/{k}": v for k, v in model.training_diagnostics().items()}
    reproduction: dict[str, Any] | None = None
    if member == 0 and not smoke:
        reproduction = {}
        for test_set, values in metrics.items():
            old = _befund94_record(seed, n_train, test_set)
            diff_m, n_m = _max_difference(values, old["metrics"])
            diff_t, n_t = _max_difference(diagnostics, old["params"])
            reproduction[test_set] = {
                "max_abs_difference_metrics": diff_m,
                "n_metrics_compared": n_m,
                "max_abs_difference_training": diff_t,
                "n_training_entries_compared": n_t,
                "befund94_checkpoint_path": old["params"].get("checkpoint/path"),
            }
    rows = [own.trajectory.run_id.index(r) for r in own.split.subset(size).train]
    actions = own.trajectory.actions[rows]
    disturbances = own.trajectory.disturbances[rows]
    outputs = own.trajectory.outputs[rows]
    output_names = list(own.trajectory.metadata["output_names"])
    return {
        "master_seed": seed,
        "n_train": n_train,
        "member": member,
        "training_seed": training_seed,
        "model_seed_of_bundle": own.seeds.model_seed,
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest,
        "fit_s": fit_s,
        "total_s": time.perf_counter() - started,
        "process_fingerprint": own.process_config.fingerprint,
        "fingerprints": own.fingerprints,
        "split_digest": own.split.digest,
        "data_spec": spec.as_parameters(),
        "exogenous_names": list(train.exogenous_names),
        "target_names": list(train.target_names),
        "training": model.training_diagnostics(),
        "metrics": metrics,
        "physical_errors": {
            "excited": _physical_errors(model, test),
            "constant_holdout": _physical_errors(model, holdout_test),
        },
        "reproduction_befund94": reproduction,
        "train_ranges": {
            "action_names": list(own.trajectory.metadata.get("action_names", [])),
            "actions_min": actions.min(axis=(0, 1)).tolist(),
            "actions_max": actions.max(axis=(0, 1)).tolist(),
            "disturbances_min": disturbances.min(axis=(0, 1)).tolist(),
            "disturbances_max": disturbances.max(axis=(0, 1)).tolist(),
            "output_names": output_names,
            "outputs_min": outputs.min(axis=(0, 1)).tolist(),
            "outputs_max": outputs.max(axis=(0, 1)).tolist(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--smoke", action="store_true", help="tiny spec, no artifact")
    args = parser.parse_args()

    from _common import RUNS_DIR, provenance, require_clean_tree

    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    output = RUNS_DIR / f"{args.name}.json"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten")
    if not args.smoke:
        require_clean_tree()
    prov = provenance("not used")
    tags = model_tags() if not args.smoke else [(1, 16, 0), (1, 64, 1)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(train_one, s, n, j, args.smoke) for s, n, j in tags]
        records = [f.result() for f in futures]
    payload = {
        "provenance": prov,
        "study": "learned sequence models as environments (docs/30_surrogate_env.md)",
        "smoke": args.smoke,
        "models": records,
        "runtime_s": time.time() - started,
    }
    if args.smoke:
        print(json.dumps(payload, default=str)[:3000])
        return 0
    output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
