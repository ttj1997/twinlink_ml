r"""Train M1, M3, M4 and fit M2 of the study docs/NN_hybrid_model.md.

Pre-registration of the study: docstring of ``scripts/analyze_hybrid_model.py`` (committed before
the first run). Settings: ``conf/hybrid_model/hybrid_model.yaml``.

Per draw d (the 20 draws of Befund 111) and master seed s: the truth pool of the Befund 94 pipeline
(arm N, ``DataSpec.new_study()``, preset version 2; 64 / 32 / 32 runs, 8 h each, 25 s) generated
with the truth config of draw d; run-level splits and ``ValidationPolicy.FIXED_POOL`` exactly as
Befund 94, ``n_train`` in {4, 8, 16, 32} (``split.subset``). Tasks:

``M1``      GRU of the Befund 94 pipeline on the truth pool (factory ``gru``, ``max_epochs`` 200, one
            torch thread); training seed = the model seed of the seed bundle (as Befund 94).
``M3``      the same architecture and budget on the RESIDUAL pool ``truth - nominal replay``
            (:mod:`twinlink_ml.hybrid_model.residual`), the same training seed as M1.
``M4m<j>``  j = 1..4: further members of the ensemble M4 (member 0 = the M3 network), training seed
            ``_derive(model_seed, "hybrid_model_ensemble", j)``, same data and split.
``M2``      least-squares refit of critical moisture and nu_max on the measured targets of the
            training runs (:mod:`twinlink_ml.hybrid_model.refit`).

Order of the tasks (so that a cut by the budget rule -- first ``n_train`` 32, then M4 -- removes the
tail): M1, M3, M2 for ``n_train`` <= 16; M4 members for ``n_train`` <= 16; everything for 32.
Every finished task appends one record to ``artifacts/runs/<name>.jsonl``; ``--resume`` skips the
tasks that have one. Checkpoints ``artifacts/checkpoints/hybrid_model/<name>/`` (gitignored, copied to
``C:/twinlink_runs/hybrid-model/`` after the run); the record holds their SHA-256.

    tools/run_here.sh scripts/train_hybrid_model.py --name hybrid_model_training_v1 --workers 14
    tools/run_here.sh scripts/train_hybrid_model.py --name ... --only-draws 0 --only-n 4 32  (pilot)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

ROOT = SCRIPTS.parent
CHECKPOINT_ROOT = ROOT / "artifacts" / "checkpoints" / "hybrid_model"
EXPECTED_PROCESS_FINGERPRINT_NOMINAL = "bddca37f716f5cfc"

_POOLS: dict[tuple[int, int], Any] = {}


def task_list(config: Any, draws: list[int], sizes: list[int]) -> list[tuple[int, int, int, str]]:
    """``(draw, seed, n_train, role)`` in the order of the module docstring."""
    seeds = config.master_seeds
    early = [n for n in sizes if n <= 16]
    late = [n for n in sizes if n > 16]
    members = [f"M4m{j}" for j in range(1, config.ensemble_members)]
    tasks: list[tuple[int, int, int, str]] = []
    for group_sizes, roles in (
        (early, ["M1", "M3", "M2"]),
        (early, members),
        (late, ["M1", "M3", "M2", *members]),
    ):
        for n in sorted(group_sizes, reverse=True):
            for role in roles:
                for d in draws:
                    for s in seeds:
                        tasks.append((d, s, n, role))
    return tasks


def task_key(d: int, s: int, n: int, role: str) -> str:
    return f"d{d:02d}_s{s}_n{n}_{role}"


def _pools(setup: Any, draw: int, seed: int) -> tuple[Any, Any]:
    """Truth pool and residual pool of (draw, seed), cached per worker."""
    key = (draw, seed)
    if key not in _POOLS:
        import run_arch_rerun_worker as rerun_worker

        from twinlink_ml.hybrid_model.residual import replay, residual_data

        _POOLS.clear()
        spec = rerun_worker.arm_spec("N")
        truth = rerun_worker._generate(setup.data_truth(draw), spec, seed)
        nominal_replay = replay(truth, setup.data_nominal)
        _POOLS[key] = (truth, residual_data(truth, nominal_replay), nominal_replay)
    truth, residual, nominal_replay = _POOLS[key]
    return truth, (residual, nominal_replay)


def run_task(name: str, d: int, s: int, n: int, role: str) -> dict[str, Any]:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    import run_arch_learning_curve_worker as old_worker
    from _common import git

    from twinlink_ml.evaluation import ValidationPolicy, build_datasets
    from twinlink_ml.hybrid_model.refit import fit_kinetics
    from twinlink_ml.hybrid_model.residual import run_rows
    from twinlink_ml.hybrid_model.study import build_setup
    from twinlink_ml.utils.checkpoints import save_checkpoint
    from twinlink_ml.utils.seeding import _derive

    started = time.perf_counter()
    setup = build_setup()
    config = setup.config
    truth, (residual, nominal_replay) = _pools(setup, d, s)
    train_ids = list(truth.split.subset(n).train)
    rows = run_rows(truth, train_ids)
    record: dict[str, Any] = {
        "key": task_key(d, s, n, role),
        "draw": d,
        "master_seed": s,
        "n_train": n,
        "role": role,
        "draw_values": setup.draw_values(d),
        "fingerprints": truth.fingerprints,
        "split_digest": truth.split.digest,
        "train_run_ids": train_ids,
        "nominal_replay_projected_steps": int(np.sum(nominal_replay.n_projected_steps))
        if nominal_replay.n_projected_steps is not None
        else None,
        "truth_projected_steps": int(np.sum(truth.trajectory.n_projected_steps))
        if truth.trajectory.n_projected_steps is not None
        else None,
    }
    if role == "M2":
        target_names = list(truth.spec.target_names)
        fit = fit_kinetics(setup.data_nominal, truth, rows, target_names, config.refit)
        record["refit"] = fit
        record["elapsed_s"] = time.perf_counter() - started
        return record

    data = truth if role == "M1" else residual
    train, validation, _ = build_datasets(data, n, policy=ValidationPolicy.FIXED_POOL)
    model = old_worker._build_model(old_worker._architecture_factory("gru"), config.max_epochs)
    member = 0 if role in ("M1", "M3") else int(role[3:])
    seed = (
        truth.seeds.model_seed
        if member == 0
        else _derive(truth.seeds.model_seed, config.ensemble_seed_label, member)
    )
    fit_start = time.perf_counter()
    model.fit(train, seed=seed, validation=validation)
    fit_s = time.perf_counter() - fit_start
    path = CHECKPOINT_ROOT / name / f"{task_key(d, s, n, role)}.pt"
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
        process_fingerprint=truth.process_config.fingerprint,
        data_spec_params=truth.spec.as_parameters(),
        seed_bundle_params=truth.seeds.as_parameters(),
        git_commit=git("rev-parse", "HEAD"),
    )
    record.update(
        {
            "member": member,
            "training_seed": seed,
            "residual_targets": role != "M1",
            "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "training": model.training_diagnostics(),
            "fit_s": fit_s,
            "elapsed_s": time.perf_counter() - started,
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-draws", type=int, nargs="*", default=None)
    parser.add_argument("--only-n", type=int, nargs="*", default=None)
    parser.add_argument("--only-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--only-roles", nargs="*", default=None)
    args = parser.parse_args()

    from _common import RUNS_DIR, provenance, require_clean_tree

    from twinlink_ml.hybrid_model.study import build_setup
    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    output = RUNS_DIR / f"{args.name}.jsonl"
    require_clean_tree(ignore_paths=[output])
    setup = build_setup()
    config = setup.config
    done: set[str] = set()
    if output.exists():
        if not args.resume:
            raise SystemExit(f"{output} exists -- use --resume or a new name")
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("kind") == "task":
                    done.add(row["key"])
    draws = args.only_draws if args.only_draws is not None else list(range(len(setup.draws)))
    sizes = args.only_n if args.only_n is not None else list(config.n_train)
    tasks = [
        t
        for t in task_list(config, draws, sizes)
        if task_key(*t) not in done
        and (args.only_seeds is None or t[1] in args.only_seeds)
        and (args.only_roles is None or t[3] in args.only_roles)
    ]
    header = {
        "kind": "header",
        "provenance": provenance("not used", ignore_paths=[output]),
        "study": "hybrid model against pure GRU (docs/NN_hybrid_model.md)",
        "config": config.raw,
        "draws": list(setup.draws),
        "n_tasks_planned": len(tasks),
        "selection": {
            "draws": draws,
            "n_train": sizes,
            "seeds": args.only_seeds,
            "roles": args.only_roles,
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
    print(f"{len(tasks)} tasks, {len(done)} done before", flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_task, args.name, *t): t for t in tasks}
        finished = 0
        while pending:
            complete, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in complete:
                t = pending.pop(future)
                record = {"kind": "task", **future.result()}
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, default=float) + "\n")
                finished += 1
                print(
                    f"[{finished}/{len(tasks)}] {task_key(*t)} {record['elapsed_s']:.0f} s "
                    f"(wall {time.time() - started:.0f} s)",
                    flush=True,
                )
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "footer", "runtime_s": time.time() - started}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
