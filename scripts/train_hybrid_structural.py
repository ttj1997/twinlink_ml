r"""Train M1 and M3 and fit M2 / M2b of the study docs/41_hybrid_structural.md.

Pre-registration of the study: docstring of ``scripts/analyze_hybrid_structural.py`` (committed
before the first run). Settings: ``conf/hybrid_structural/hybrid_structural.yaml``.

Per draw d (the 20 draws of Befund 111) and master seed s (1-5): the truth pool of the Befund 94
pipeline (arm N, ``DataSpec.new_study()``, preset version 2; 64 / 32 / 32 runs, 8 h each, 25 s)
simulated with the STRUCTURALLY deviating truth of draw d (material parameters of the draw plus
evaporation of a fixed share of the spray water above the bed,
``hybrid_structural.truth.FreeboardEvaporationGranulator``); run-level splits and
``ValidationPolicy.FIXED_POOL`` exactly as Befund 94 and 122, ``n_train`` in {4, 8, 16}. Units:

``M1``  GRU of the Befund 94 pipeline on the truth pool (factory ``gru``, ``max_epochs`` 200, one
        torch thread); training seed = the model seed of the seed bundle (as Befund 94 and 122).
``M3``  the same architecture and budget on the RESIDUAL pool ``truth - nominal replay``
        (``hybrid_model.residual``), the same training seed as M1. The nominal model is the prior
        knowledge WITHOUT the freeboard term.
``M2``  least-squares refit of critical moisture and nu_max of the prior model on the measured
        targets of the training runs (``hybrid_model.refit``, as Befund 122), and for **M2b** the
        constant bias ``mean(measured - replay of the refitted model)`` over the same runs
        (``hybrid_structural.bias.training_bias``; only the moisture bias is used).

Order of the units (so that a cut by the budget rule -- first ``n_train`` 4 -- removes the tail):
``n_train`` 16, 8, 4; within one ``n_train`` M1, M3, M2. M0 needs no training.

Long-run rules (``CLAUDE.md``): every finished unit appends one record to
``artifacts/runs/<name>.jsonl.gz`` (one gzip member per record, synced); a failing unit is
recorded as an error and the others go on; ``--resume`` skips the units with an ``ok`` record;
progress per finished unit with the worker's memory peak. ``--dry-run`` covers ALL units at
minimal cost (one epoch, three refit evaluations), one task per (draw, seed) running its nine units
through the same code; it writes to a name starting with ``_`` (gitignored). Checkpoints go to
``C:/twinlink_runs/hybrid-structural/checkpoints/<name>/`` (outside the repository; the record
holds path and SHA-256) and are never overwritten.

    tools/run_here.sh scripts/train_hybrid_structural.py --name _hybrid_structural_dry_training --dry-run --workers 8
    tools/run_here.sh scripts/train_hybrid_structural.py --name hybrid_structural_training_v1 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

ROOT = SCRIPTS.parent
CHECKPOINT_ROOT = Path("C:/twinlink_runs/hybrid-structural/checkpoints")
ROLES = ("M1", "M3", "M2")
DRY_EPOCHS = 1
DRY_REFIT_NFEV = 3

Unit = tuple[int, int, int, str]

_SETUP: Any = None
_CONFIG_PATH: str | None = None  # None: the config of Befund 124; set by --config (follow-up study)
_POOLS: dict[tuple[int, int], Any] = {}


def use_config(path: str | None) -> None:
    """Select the study config of this process (before the set-up is built)."""
    global _CONFIG_PATH
    if _SETUP is not None and path != _CONFIG_PATH:
        raise RuntimeError("another study set-up is cached in this process")
    _CONFIG_PATH = path


def setup() -> Any:
    """The structural study set-up, built once per process."""
    global _SETUP
    if _SETUP is None:
        from twinlink_ml.hybrid_structural.study import (
            build_structural_setup,
            load_structural_config,
        )

        config = load_structural_config(_CONFIG_PATH) if _CONFIG_PATH else None
        _SETUP = build_structural_setup(config)
    return _SETUP


def unit_list(draws: list[int], sizes: list[int], seeds: list[int]) -> list[Unit]:
    """``(draw, seed, n_train, role)`` in the order of the module docstring."""
    return [
        (d, s, n, role)
        for n in sorted(sizes, reverse=True)
        for role in ROLES
        for d in draws
        for s in seeds
    ]


def unit_key(d: int, s: int, n: int, role: str) -> str:
    return f"d{d:02d}_s{s}_n{n}_{role}"


def _pools(s: Any, draw: int, seed: int) -> tuple[Any, Any, Any]:
    """Truth pool, residual pool and nominal replay of (draw, seed); one pair cached per worker."""
    key = (draw, seed)
    if key not in _POOLS:
        import run_arch_rerun_worker as rerun_worker

        from twinlink_ml.hybrid_model.residual import replay, residual_data

        _POOLS.clear()
        truth = s.generate(s.data_truth(draw), rerun_worker.arm_spec("N"), seed)
        nominal_replay = replay(truth, s.data_nominal)
        _POOLS[key] = (truth, residual_data(truth, nominal_replay), nominal_replay)
    truth, residual, nominal_replay = _POOLS[key]
    return truth, residual, nominal_replay


def run_unit(
    name: str, checkpoint_root: str, dry: bool, d: int, s: int, n: int, role: str
) -> dict[str, Any]:
    """One unit; raises on failure (the caller records the error)."""
    import dataclasses

    import run_arch_learning_curve_worker as old_worker
    from _common import git

    from twinlink_ml.evaluation import ValidationPolicy, build_datasets
    from twinlink_ml.hybrid_model.refit import fit_kinetics, refit_config
    from twinlink_ml.hybrid_model.residual import run_rows
    from twinlink_ml.hybrid_structural.bias import training_bias
    from twinlink_ml.utils.checkpoints import save_checkpoint

    started = time.perf_counter()
    st = setup()
    config = st.config
    truth, residual, nominal_replay = _pools(st, d, s)
    pool_s = time.perf_counter() - started
    train_ids = list(truth.split.subset(n).train)
    rows = run_rows(truth, train_ids)

    def projected(trajectory: Any) -> int | None:
        steps = trajectory.n_projected_steps
        return None if steps is None else int(np.sum(steps))

    record: dict[str, Any] = {
        "key": unit_key(d, s, n, role),
        "draw": d,
        "master_seed": s,
        "n_train": n,
        "role": role,
        "draw_values": st.draw_values(d),
        "structure": st.structure_record(),
        "fingerprints": truth.fingerprints,
        "split_digest": truth.split.digest,
        "train_run_ids": train_ids,
        "nominal_replay_projected_steps": projected(nominal_replay),
        "truth_projected_steps": projected(truth.trajectory),
        "pool_s": pool_s,
        "dry_run": dry,
    }
    if role == "M2":
        target_names = list(truth.spec.target_names)
        settings = (
            dataclasses.replace(config.refit, max_nfev=DRY_REFIT_NFEV) if dry else config.refit
        )
        fit = fit_kinetics(st.data_nominal, truth, rows, target_names, settings)
        fitted = refit_config(st.data_nominal, fit["names"], fit["theta"])
        bias = training_bias(truth, fitted, rows, target_names)
        record["refit"] = fit
        record["bias"] = bias  # M2b uses the moisture entry only
        record["moisture_bias_kg_per_kg"] = bias["product_moisture_kg_per_kg"]
        record["elapsed_s"] = time.perf_counter() - started
        return record

    data = truth if role == "M1" else residual
    train, validation, _ = build_datasets(data, n, policy=ValidationPolicy.FIXED_POOL)
    epochs = DRY_EPOCHS if dry else config.max_epochs
    model = old_worker._build_model(old_worker._architecture_factory("gru"), epochs)
    seed = truth.seeds.model_seed
    fit_start = time.perf_counter()
    model.fit(train, seed=seed, validation=validation)
    fit_s = time.perf_counter() - fit_start
    path = Path(checkpoint_root) / name / f"{unit_key(d, s, n, role)}.pt"
    if path.exists():
        raise RuntimeError(f"{path} exists -- checkpoints are never overwritten")
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
            "member": 0,
            "training_seed": seed,
            "residual_targets": role != "M1",
            "max_epochs": epochs,
            "checkpoint": path.as_posix(),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "training": model.training_diagnostics(),
            "fit_s": fit_s,
            "elapsed_s": time.perf_counter() - started,
        }
    )
    return record


def run_units(
    name: str,
    checkpoint_root: str,
    dry: bool,
    units: list[Unit],
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    """Run the units of one task in a worker; a failing unit becomes an error record."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    use_config(config_path)
    import run_hybrid_evaluation as evaluation
    import torch

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    out = []
    for d, s, n, role in units:
        started = time.perf_counter()
        try:
            record = {"status": "ok", **run_unit(name, checkpoint_root, dry, d, s, n, role)}
        except BaseException:  # recorded and reported, the run goes on
            record = {
                "status": "error",
                "key": unit_key(d, s, n, role),
                "traceback": traceback.format_exc(),
                "elapsed_s": time.perf_counter() - started,
            }
        record["memory"] = evaluation.peak_memory_mb()
        record["worker_pid"] = os.getpid()
        out.append(record)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-root", default=str(CHECKPOINT_ROOT))
    parser.add_argument(
        "--config",
        default=None,
        help="study config (default: conf/hybrid_structural/hybrid_structural.yaml)",
    )
    parser.add_argument("--only-draws", type=int, nargs="*", default=None)
    parser.add_argument("--only-n", type=int, nargs="*", default=None)
    parser.add_argument("--only-seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    from _common import RUNS_DIR, provenance, require_clean_tree

    from twinlink_ml.hybrid_structural.records import append_record, read_records, repair_tail
    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    if args.dry_run != args.name.startswith("_"):
        raise SystemExit("a dry run (and only a dry run) writes to a name starting with '_'")
    if args.workers > (os.cpu_count() or 2) // 2:
        raise SystemExit("at most as many workers as physical cores")
    output = RUNS_DIR / f"{args.name}.jsonl.gz"
    require_clean_tree(ignore_paths=[output])
    if output.exists() and not args.resume:
        raise SystemExit(f"{output} exists -- use --resume or a new name")
    removed = repair_tail(output)
    config_path = None if args.config is None else str(Path(args.config).resolve())
    use_config(config_path)
    st = setup()
    config = st.config
    done = {
        r["key"]
        for r in read_records(output)
        if r.get("kind") == "task" and r.get("status") == "ok"
    }
    draws = args.only_draws if args.only_draws is not None else list(range(len(st.draws)))
    sizes = args.only_n if args.only_n is not None else list(config.n_train)
    seeds = args.only_seeds if args.only_seeds is not None else list(config.master_seeds)
    units = [u for u in unit_list(draws, sizes, seeds) if unit_key(*u) not in done]
    if args.dry_run:  # one task per (draw, seed): the pool is generated once for its units
        groups: dict[tuple[int, int], list[Unit]] = {}
        for u in units:
            groups.setdefault((u[0], u[1]), []).append(u)
        tasks = list(groups.values())
    else:
        tasks = [[u] for u in units]
    output.parent.mkdir(parents=True, exist_ok=True)
    append_record(
        output,
        {
            "kind": "header",
            "provenance": provenance("not used", ignore_paths=[output]),
            "study": st.study_label,
            "config": config.raw,
            "structure": st.structure_record(),
            "draws": list(st.draws),
            "n_units_planned": len(units),
            "dry_run": args.dry_run,
            "workers": args.workers,
            "checkpoint_root": Path(args.checkpoint_root).as_posix(),
            "selection": {"draws": draws, "n_train": sizes, "seeds": seeds},
            "resumed_with": len(done),
            "incomplete_tail_bytes_removed": removed,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    print(f"{len(units)} units in {len(tasks)} tasks, {len(done)} done before", flush=True)
    started = time.time()
    n_done = n_errors = 0
    broken = False
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(run_units, args.name, args.checkpoint_root, args.dry_run, t, config_path): t
            for t in tasks
        }
        while pending:
            complete, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in complete:
                task = pending.pop(future)
                try:
                    records = future.result()
                except BrokenProcessPool:
                    # a worker died (e.g. out of memory): its units have no record; resume later
                    broken = True
                    records = [
                        {
                            "status": "error",
                            "key": unit_key(*u),
                            "traceback": traceback.format_exc(),
                            "elapsed_s": 0.0,
                            "memory": {},
                        }
                        for u in task
                    ]
                for record in records:
                    ok = record["status"] == "ok"
                    append_record(output, {"kind": "task" if ok else "error", **record})
                    n_done += ok
                    n_errors += not ok
                    mem = record.get("memory", {})
                    print(
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{n_done} done, {n_errors} errors, "
                        f"{len(units) - n_done - n_errors} open] {record['key']} "
                        f"{record['status']} {record['elapsed_s']:.0f} s"
                        f" worker {record.get('worker_pid')}"
                        f" peak commit {mem.get('peak_commit_mb', float('nan')):.0f} MB"
                        f" (wall {time.time() - started:.0f} s)",
                        flush=True,
                    )
                    if not ok:
                        print(record["traceback"], flush=True)
    append_record(
        output,
        {
            "kind": "footer",
            "runtime_s": time.time() - started,
            "n_done": n_done,
            "n_errors": n_errors,
            "pool_broken": broken,
        },
    )
    print(f"finished: {n_done} units, {n_errors} errors", flush=True)
    return 1 if n_errors or broken else 0


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    raise SystemExit(main())
