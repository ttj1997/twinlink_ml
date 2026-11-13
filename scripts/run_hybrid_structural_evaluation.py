r"""Evaluation of the study docs/NN_hybrid_structural.md: oracle, margins, one-step and rollout errors.

Pre-registration: docstring of ``scripts/analyze_hybrid_structural.py`` (committed before the first
run). Settings: ``conf/hybrid_structural/hybrid_structural.yaml``.

The task functions, the scheduler of the margins (search per draw, pooled 95 % quantile of the
cell, evaluation per draw and round, raise rule of Befund 111) and the per-task bookkeeping are
those of ``scripts/run_hybrid_evaluation.py`` (Befund 122), called unchanged. This driver only

* installs the set-up of this study in every worker (``hybrid_structural.study.StructuralSetup``:
  the truth with evaporation above the bed for every data pool and every steady evaluation; the
  prior knowledge without it), before any task function runs;
* runs the models M0, M1, M2, M2b and M3 (no M4);
* writes every record at once to ``artifacts/runs/<name>.jsonl.gz`` (one gzip member per record)
  and reads its inputs -- the training records and the oracle -- as committed ``.jsonl.gz`` files:
  the clean-tree check exempts the run's own output and nothing else.

``oracle``   per draw the optimum of the TRUTH at every case, the classification (b) for every
             (seed, ``n_train``), and the discretisation check against the 8-start Nelder-Mead
             at 10 interior cases.                                -> ``<prefix>_oracle``
``margins``  per cell (model, master seed, ``n_train``): search and evaluation per draw.
                                                                 -> ``<prefix>_margins_per_draw``
``errors``   per (draw, seed): one-step (h1) and rollout (h36) errors of every model and
             ``n_train`` on the excited test set, the constant-run holdout and the constant-run
             holdout with widened ambient ranges.                 -> ``<prefix>_errors``

Long-run rules (``CLAUDE.md``): ``--resume`` skips the units with an ``ok`` record; an exception
is recorded per task (its cell stops, the others go on); every log line carries the worker's
memory peak. ``--dry-run minimal`` (1 bisection step, 3 x 3 grid, no zoom) runs every margins
task at minimal cost; ``--dry-run coarse`` the same with the full coarse grid (the largest
prediction batch, hence the memory peak). Dry runs write to a name starting with ``_``.
Budget rule of the request: first ``n_train`` 4 is dropped (``--only-n 8 16``), then M0
(``--only-models M1 M2 M2b M3``).

    tools/run_here.sh scripts/run_hybrid_structural_evaluation.py --part oracle --workers 8
    tools/run_here.sh scripts/run_hybrid_structural_evaluation.py --part margins --workers 8
    tools/run_here.sh scripts/run_hybrid_structural_evaluation.py --part errors --workers 8
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_hybrid_evaluation as base  # noqa: E402 - needs the script directory on the path

ROOT = SCRIPTS.parent
PARTS = ("oracle", "margins", "errors")


def setup() -> Any:
    """Install the structural set-up as THE set-up of the Befund 122 task functions.

    ``run_hybrid_evaluation.setup()`` returns its cached set-up if there is one; this function
    puts the set-up of this study there (with its digest, checked at the start of every task by
    ``reset_task_state``). It fails if another set-up is already cached in this process.
    """
    from twinlink_ml.hybrid_structural.study import StructuralSetup, build_structural_setup

    if base._SETUP is None:
        base._SETUP = build_structural_setup()
        base._SETUP_DIGEST = base._digest(base._SETUP)
    elif not isinstance(base._SETUP, StructuralSetup):
        raise RuntimeError("another study set-up is cached in this process")
    return base._SETUP


def _with_setup(fn: Any, args: tuple[Any, ...]) -> dict[str, Any]:
    setup()
    result: dict[str, Any] = fn(*args)
    return result


def run_task(fn: Any, args: tuple[Any, ...]) -> dict[str, Any]:
    """One task in a worker: set-up of this study first, then the task of Befund 122."""
    return base.run_task(_with_setup, (fn, args))


def output_name(part: str, prefix: str, name: str | None) -> str:
    default = f"{prefix}_margins_per_draw" if part == "margins" else f"{prefix}_{part}"
    return name or default


class PackedScheduler(base.Scheduler):
    """The scheduler of Befund 122 writing packed records."""

    def write(self, record: dict[str, Any]) -> None:
        from twinlink_ml.hybrid_structural.records import append_record

        append_record(self.output, record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=PARTS, required=True)
    parser.add_argument("--prefix", default="hybrid_structural_v1")
    parser.add_argument("--training", default="hybrid_structural_training_v1")
    parser.add_argument(
        "--name",
        default=None,
        help="output name (default <prefix>_<part>, margins: <prefix>_margins_per_draw)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", choices=tuple(base.DRY_RUNS), default=None)
    parser.add_argument(
        "--max-batch",
        type=int,
        default=256,
        help="margins: points per map call (memory; 0 = one call per request)",
    )
    parser.add_argument("--only-models", nargs="*", default=None)
    parser.add_argument("--only-n", type=int, nargs="*", default=None)
    parser.add_argument("--only-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--only-draws", type=int, nargs="*", default=None)
    args = parser.parse_args()

    from _common import RUNS_DIR, provenance, require_clean_tree

    from twinlink_ml.hybrid_model.grid_rto import BisectionSettings, GridSettings
    from twinlink_ml.hybrid_structural.records import append_record, read_records, repair_tail
    from twinlink_ml.hybrid_structural.study import MODELS
    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    name = output_name(args.part, args.prefix, args.name)
    if args.dry_run and not name.startswith("_"):
        raise SystemExit("a dry run writes to a name starting with '_' (gitignored)")
    if args.workers > (os.cpu_count() or 2) // 2:
        raise SystemExit("at most as many workers as physical cores")
    models = args.only_models or list(MODELS)
    if set(models) - set(MODELS):
        raise SystemExit(f"models of this study: {MODELS}")
    output = RUNS_DIR / f"{name}.jsonl.gz"
    oracle = None if args.part == "oracle" else f"{args.prefix}_oracle"
    # no exception besides the run's own output: the inputs are committed (or gitignored dry runs)
    require_clean_tree(ignore_paths=[output])
    if output.exists() and not args.resume:
        raise SystemExit(f"{output} exists -- use --resume or a new name")
    inputs = [RUNS_DIR / f"{n}.jsonl.gz" for n in (args.training, oracle) if n is not None]
    if args.part == "oracle":
        inputs = []  # the oracle needs neither training records nor checkpoints
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"input {path} is missing")
        if (RUNS_DIR / path.name.removesuffix(".gz")).exists():
            raise SystemExit(f"{path.name} exists packed and plain -- keep the packed file")
    removed = repair_tail(output)
    s = setup()
    cfg = s.config
    seeds = args.only_seeds or list(cfg.master_seeds)
    sizes = args.only_n or list(cfg.n_train)
    draws = tuple(args.only_draws if args.only_draws is not None else range(len(s.draws)))
    grid, constant = cfg.grid, cfg.margin_constant
    if args.dry_run:
        over = dict(base.DRY_RUNS[args.dry_run])
        over["coarse"] = over["coarse"] or cfg.grid.coarse
        grid = GridSettings(**over)
        constant = BisectionSettings(upper=constant.upper, tolerance=constant.upper)
    run = base.MarginRun(
        f"{args.prefix}_oracle", args.training, grid, constant, cfg.margin_k, args.max_batch or None
    )

    done = {
        r["key"]: r
        for r in read_records(output)
        if r.get("kind") in ("task", "job", "cell") and r.get("status", "ok") == "ok"
    }
    append_record(
        output,
        {
            "kind": "header",
            "provenance": provenance("not used", ignore_paths=[output]),
            "study": "hybrid against GRU against refit, structurally deviating truth "
            "(docs/NN_hybrid_structural.md)",
            "part": args.part,
            "training": args.training,
            "oracle": oracle,
            "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            "config": cfg.raw,
            "structure": {"mode": "evaporation", "share": s.share},
            "dry_run": args.dry_run,
            "settings": {
                "grid": dataclasses.asdict(grid),
                "constant": dataclasses.asdict(constant),
                "max_batch": run.max_batch,
            },
            "workers": args.workers,
            "selection": {"models": models, "n": sizes, "seeds": seeds, "draws": list(draws)},
            "resumed_with": len(done),
            "incomplete_tail_bytes_removed": removed,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    sched = PackedScheduler(output, run, cfg.holding_share)
    if args.part == "oracle":
        for i in range(cfg.discretisation_n_cases):
            if f"discretisation_{i}" not in done:
                sched.add((0,), f"discretisation_{i}", base.discretisation_job, (i,))
        for d in draws:
            if f"oracle_d{d:02d}" not in done:
                sched.add((1,), f"oracle_d{d:02d}", base.oracle_job, (d,))
    elif args.part == "margins":
        for m in models:
            for n in sizes:
                for seed in seeds:
                    sched.add_cell(base.Cell(m, seed, n, draws), done)
    else:
        for d in draws:
            for seed in seeds:
                if f"errors_d{d:02d}_s{seed}" not in done:
                    sched.add(
                        (0,), f"errors_d{d:02d}_s{seed}", base.errors_job, (args.training, d, seed)
                    )
    n_cells = len(sched.cells)
    print(
        f"{len(sched.ready)} tasks ready, {len(done)} records done before"
        + (f", {n_cells} cells ({sched.n_cells_done} done)" if n_cells else ""),
        flush=True,
    )

    started = time.time()
    in_flight: dict[Any, str] = {}
    broken = False
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while (sched.ready and not broken) or in_flight:
            while sched.ready and len(in_flight) < args.workers and not broken:
                task = sched.pop()
                if task is None:
                    break
                key, fn, task_args = task
                in_flight[pool.submit(run_task, fn, task_args)] = key
            if not in_flight:
                break
            complete, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in complete:
                key = in_flight.pop(future)
                try:
                    record = future.result()
                except BrokenProcessPool:
                    # a worker died (e.g. out of memory): every task in flight is lost; stop
                    # scheduling, record the error, resume later
                    broken = True
                    record = {
                        "status": "error",
                        "traceback": traceback.format_exc(),
                        "elapsed_s": 0.0,
                        "memory": {},
                    }
                kind = "job" if args.part != "margins" else "task"
                if record["status"] == "ok":
                    sched.write({"kind": kind, **record})
                    sched.n_done += 1
                    if args.part == "margins":
                        sched.complete(key, record)
                else:
                    sched.write({"kind": "error", "key": key, **record})
                    sched.n_errors += 1
                    cell = sched.owner.get(key)
                    if cell is not None:
                        cell.blocked = True
                mem = record.get("memory", {})
                print(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{sched.n_done} done, "
                    f"{sched.n_errors} errors, {len(sched.ready) + len(in_flight)} open"
                    + (f", cells {sched.n_cells_done}/{n_cells}" if n_cells else "")
                    + f"] {key} {record['status']} {record['elapsed_s']:.0f} s"
                    f" worker {record.get('worker_pid')}"
                    f" commit {mem.get('commit_mb', math.nan):.0f} MB"
                    f" peak commit {mem.get('peak_commit_mb', math.nan):.0f} MB"
                    f" (wall {time.time() - started:.0f} s)",
                    flush=True,
                )
                if record["status"] != "ok":
                    print(record["traceback"], flush=True)
    blocked = [c.key for c in sched.cells.values() if c.blocked]
    sched.write(
        {
            "kind": "footer",
            "runtime_s": time.time() - started,
            "n_done": sched.n_done,
            "n_errors": sched.n_errors,
            "cells_done": sched.n_cells_done,
            "cells_blocked": blocked,
            "pool_broken": broken,
        }
    )
    print(
        f"finished: {sched.n_done} tasks, {sched.n_errors} errors"
        + (f", cells {sched.n_cells_done}/{n_cells}, blocked {blocked}" if n_cells else ""),
        flush=True,
    )
    return 1 if sched.n_errors or broken else 0


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    raise SystemExit(main())
