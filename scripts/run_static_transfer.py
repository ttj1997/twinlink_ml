"""Static transfer study: pilot, dry run and main run (docs/37_static_transfer.md).

Settings and placeholders: ``conf/static_transfer/static_transfer.yaml``; package
``twinlink_ml.static_transfer``; reading (pre-registered): docstring of
``scripts/analyze_static_transfer.py``.

Phases
------
``--phase pilot --name static_transfer_pilot_v1``
    Pretrains one MLP per pilot seed (11-13), then evaluates every setting of the pilot grids on
    the pilot plant PP-st (N = 4 and 16, its own validation set). Selection: lowest mean criterion
    C (mean over seeds, N and targets of RMSE / primary noise). A selected value at the edge of its
    grid dimension extends that dimension geometrically (at most ``pilot.max_extensions`` times; an
    edge at 0 is a natural bound and not extended); new combinations are evaluated and the
    selection repeated. Output: ``artifacts/runs/<name>.json`` with every setting, the rounds and
    the selection. The selection is then copied into the ``selected`` block of the YAML (own
    commit) before the main run.
``--phase main --name static_transfer_v1``
    Checks that ``selected`` equals the selection of the pilot artifact named there, pretrains one
    MLP per main seed (1-5), then runs every unit (plant x condition x seed; 6 x 3 x 5 = 90):
    all methods at every N, scored on the four test sets. ``--conditions`` restricts the
    conditions (budget rule of the project lead: first drop ``clean_high``).
``--phase main --dry-run --name static_transfer_dryrun_v1``
    Every unit at minimal cost (pretraining 2 epochs on 256 points, every fit at most 5 optimiser
    steps, L1 at most 3 function evaluations): surfaces edge cases, records time and peak memory
    per worker. Its numbers are never results; the deviation table and the coverage it records
    are exact (they do not depend on any fit).

Every finished task is appended to ``artifacts/runs/_<name>_cache.jsonl`` (gitignored);
``--resume`` reuses it (same commit and config required). A failing unit is logged with its
traceback and does not stop the others. Pretrained nets go to the gitignored
``artifacts/checkpoints/static_transfer/<name>/``. Clean tree required (provenance); start from
the worktree with ``tools/run_here.sh scripts/run_static_transfer.py ...``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, _json_default, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.static_transfer import study  # noqa: E402
from twinlink_ml.static_transfer.balance import plant_model, source_model  # noqa: E402
from twinlink_ml.static_transfer.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    INPUT_NAMES,
    TARGET_NAMES,
    StaticTransferConfig,
    load_config,
)
from twinlink_ml.static_transfer.design import TEST_SET_NAMES  # noqa: E402

REPO_ROOT = SCRIPTS.parent
CHECKPOINTS = REPO_ROOT / "artifacts" / "checkpoints" / "static_transfer"
_WORKER: dict[str, Any] = {}


# ----------------------------------------------------------------------------- workers
def _init(ctx: study.StudyContext, checkpoints: dict[int, str]) -> None:
    import torch

    torch.set_num_threads(1)
    _WORKER["ctx"] = ctx
    nets = {}
    for seed, path in checkpoints.items():
        net = study.fresh_net(ctx.config, seed)
        net.load_state_dict(torch.load(path, weights_only=True))
        nets[seed] = net
    _WORKER["nets"] = nets


def _pretrain_worker(args: tuple[StaticTransferConfig, int, bool, str]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    config, seed, smoke, path = args
    net, record = study.pretraining_task(config, seed, smoke=smoke)
    torch.save(net.state_dict(), path)
    record["checkpoint"] = str(Path(path).relative_to(REPO_ROOT))
    record["peak_memory_mb"] = study.peak_memory_mb()
    return record


def _task_worker(task: tuple[Any, ...]) -> dict[str, Any]:
    ctx: study.StudyContext = _WORKER["ctx"]
    t0 = time.perf_counter()
    try:
        if task[0] == "pilot":
            _, method, setting_json, steps_cap = task
            result = study.pilot_task(
                ctx, method, json.loads(setting_json), _WORKER["nets"], steps_cap=steps_cap
            )
        elif task[0] == "unit":
            _, plant, condition, seed, steps_cap, l1_cap = task
            result = study.run_unit(
                ctx,
                plant,
                condition,
                int(seed),
                _WORKER["nets"][int(seed)],
                steps_cap=steps_cap,
                l1_max_nfev=l1_cap,
            )
        else:
            raise ValueError(task[0])
        return {"task": list(task), "ok": True, "result": result, "pid": os.getpid()}
    except Exception:
        return {
            "task": list(task),
            "ok": False,
            "error": traceback.format_exc(),
            "seconds": time.perf_counter() - t0,
            "pid": os.getpid(),
        }


# ----------------------------------------------------------------------------- cache
class Cache:
    """Append-only JSONL of finished tasks; first line is the header."""

    def __init__(self, path: Path, header: dict[str, Any], resume: bool) -> None:
        self.path = path
        self.done: dict[str, dict[str, Any]] = {}
        if path.exists():
            if not resume:
                raise SystemExit(f"{path} exists -- pass --resume or use a new name")
            with path.open(encoding="utf-8") as fh:
                first = json.loads(fh.readline())
                if first.get("header") != header:
                    raise SystemExit(f"{path} was written by another commit/config: {first}")
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("ok", True):
                        self.done[json.dumps(rec["task"])] = rec
            print(f"resume: {len(self.done)} cached tasks", flush=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"header": header}) + "\n", encoding="utf-8")

    def add(self, rec: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=_json_default) + "\n")
        if rec.get("ok", True):
            self.done[json.dumps(rec["task"])] = rec


def _run_pool(
    tasks: list[tuple[Any, ...]],
    cache: Cache,
    workers: int,
    ctx: study.StudyContext,
    checkpoints: dict[int, str],
) -> list[dict[str, Any]]:
    todo = [t for t in tasks if json.dumps(list(t)) not in cache.done]
    failures: list[dict[str, Any]] = []
    print(f"{len(todo)} of {len(tasks)} tasks to run, {workers} workers", flush=True)
    if not todo:
        return failures
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(ctx, checkpoints)) as pool:
        futures = [pool.submit(_task_worker, t) for t in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            cache.add(rec)
            elapsed = time.perf_counter() - t0
            eta = elapsed / k * (len(todo) - k)
            status = "ok" if rec["ok"] else "FAILED"
            print(
                f"[{elapsed:7.0f} s, eta {eta:6.0f} s] {k}/{len(todo)} {rec['task']} {status}",
                flush=True,
            )
            if not rec["ok"]:
                print(rec["error"], flush=True)
                failures.append(rec)
    return failures


def _pretrain(
    config: StaticTransferConfig,
    seeds: list[int],
    name: str,
    smoke: bool,
    cache: Cache,
    workers: int,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    folder = CHECKPOINTS / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = {s: str(folder / f"pretrain_seed{s}.pt") for s in seeds}
    records = []
    todo = []
    for s in seeds:
        key = json.dumps(["pretrain", s])
        if key in cache.done and Path(paths[s]).exists():
            records.append(cache.done[key]["result"])
        else:
            todo.append(s)
    if todo:
        print(f"pretraining seeds {todo}", flush=True)
        with ProcessPoolExecutor(min(workers, len(todo))) as pool:
            futures = {pool.submit(_pretrain_worker, (config, s, smoke, paths[s])): s for s in todo}
            for fut in as_completed(futures):
                s = futures[fut]
                rec = fut.result()
                cache.add({"task": ["pretrain", s], "ok": True, "result": rec})
                records.append(rec)
                print(
                    f"pretrained seed {s}: {rec['epochs']} epochs, {rec['seconds']:.0f} s",
                    flush=True,
                )
    return paths, sorted(records, key=lambda r: r["seed"])


# ----------------------------------------------------------------------------- reports
def coverage(ctx: study.StudyContext) -> dict[str, Any]:
    """Share of each point set inside the pretraining box and its range per input."""
    box = ctx.config.boxes["pretraining"]
    sets = {k: v.x for k, v in ctx.tests.items()}
    sets.update({f"design_seed{s}": d.x for s, d in ctx.designs.items()})
    out: dict[str, Any] = {
        "pretraining_box": {
            n: [lo, hi] for n, lo, hi in zip(INPUT_NAMES, box.lower, box.upper, strict=True)
        }
    }
    for name, x in sets.items():
        out[name] = {
            "inside_pretraining_box": float(np.mean(box.contains(x))),
            "range": {
                n: [float(x[:, j].min()), float(x[:, j].max())] for j, n in enumerate(INPUT_NAMES)
            },
        }
    out["admissibility"] = {
        k: {"candidates": v.candidates, "admissible": v.admissible} for k, v in ctx.tests.items()
    }
    return out


def deviation_table(ctx: study.StudyContext) -> dict[str, Any]:
    """L0 against each plant (noise-free, clean inputs) per test set and target, also in sigma."""
    config = ctx.config
    source = source_model(config.source)
    out: dict[str, Any] = {}
    for spec in (*config.plants, config.pilot_plant):
        plant = plant_model(config.source, spec)
        affected = config.affected_targets.get(spec.name, ())
        rows = {}
        for k in TEST_SET_NAMES:
            x = ctx.tests[k].x
            truth = plant.predict(x)
            rmse = np.sqrt(np.mean((source.predict(x) - truth) ** 2, axis=0))
            sigma = study.criterion_scale(config, truth)
            ratio = rmse / sigma
            rows[k] = {
                "rmse": rmse.tolist(),
                "sigma": sigma.tolist(),
                "rmse_over_sigma": ratio.tolist(),
                "below_2_sigma_on_affected": [
                    t for j, t in enumerate(TARGET_NAMES) if t in affected and ratio[j] < 2.0
                ],
            }
        out[spec.name] = rows
    return out


def _check_selection(config: StaticTransferConfig) -> dict[str, Any]:
    name = config.raw["selected"].get("pilot_artifact")
    if not name:
        raise SystemExit(
            "selected.pilot_artifact is empty -- run the pilot and fill the YAML first"
        )
    pilot = json.loads((RUNS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    for method, chosen in pilot["selection"].items():
        if config.selected(method) != {k: float(v) for k, v in chosen["setting"].items()}:
            raise SystemExit(f"selected.{method} differs from the pilot artifact {name}")
    if pilot["config_fingerprint_without_selection"] != config.fingerprint_without_selection:
        raise SystemExit("the YAML changed outside the selected block since the pilot")
    return {"pilot_artifact": name, "pilot_commit": pilot["provenance"]["git/commit"]}


# ----------------------------------------------------------------------------- phases
def run_pilot(args: argparse.Namespace, config: StaticTransferConfig) -> None:
    out_path = RUNS_DIR / f"{args.name}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
    require_clean_tree()
    prov = provenance()
    pilot = config.raw["pilot"]
    seeds = [int(s) for s in pilot["seeds"]]
    header = {"commit": git("rev-parse", "HEAD"), "config": config.fingerprint, "phase": "pilot"}
    cache = Cache(RUNS_DIR / f"_{args.name}_cache.jsonl", header, args.resume)
    t0 = time.perf_counter()
    ctx = study.build_context(config, seeds)
    checkpoints, pre_records = _pretrain(config, seeds, args.name, False, cache, args.workers)

    grids = {
        m: {k: [float(v) for v in vals] for k, vals in g.items()} for m, g in pilot["grids"].items()
    }
    extensions: dict[str, dict[str, int]] = {m: {} for m in grids}
    rounds: list[dict[str, Any]] = []
    final: dict[str, bool] = dict.fromkeys(grids, False)
    failures: list[dict[str, Any]] = []
    while not all(final.values()):
        tasks = [
            ("pilot", m, json.dumps(s, sort_keys=True), None)
            for m in grids
            if not final[m]
            for s in study.grid_settings(grids[m])
        ]
        failures += _run_pool(tasks, cache, args.workers, ctx, checkpoints)
        round_info: dict[str, Any] = {}
        for m in grids:
            if final[m]:
                continue
            results = [
                cache.done[json.dumps(["pilot", m, json.dumps(s, sort_keys=True), None])]["result"]
                for s in study.grid_settings(grids[m])
                if json.dumps(["pilot", m, json.dumps(s, sort_keys=True), None]) in cache.done
            ]
            best = study.select_setting(results)
            new_grid, extensions[m], extended = study.extend_grid(
                grids[m], best["setting"], extensions[m], int(pilot["max_extensions"])
            )
            round_info[m] = {
                "best": best["setting"],
                "criterion": best["criterion"],
                "extended": extended,
            }
            if extended:
                grids[m] = new_grid
            else:
                final[m] = True
        rounds.append(round_info)
        print(f"pilot round {len(rounds)}: {round_info}", flush=True)

    selection = {}
    all_results = {}
    for m in grids:
        results = [
            cache.done[json.dumps(["pilot", m, json.dumps(s, sort_keys=True), None])]["result"]
            for s in study.grid_settings(grids[m])
            if json.dumps(["pilot", m, json.dumps(s, sort_keys=True), None]) in cache.done
        ]
        best = study.select_setting(results)
        at_edge = {
            k: float(best["setting"][k]) in (min(v), max(v))
            and len(v) > 1
            and float(best["setting"][k]) != 0.0
            for k, v in grids[m].items()
        }
        selection[m] = {
            "setting": best["setting"],
            "criterion": best["criterion"],
            "at_edge_after_extensions": at_edge,
        }
        all_results[m] = sorted(results, key=lambda r: r["criterion"])
    payload = {
        "provenance": prov,
        "study": "static_transfer",
        "phase": "pilot",
        "config_path": str(DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)),
        "config_fingerprint": config.fingerprint,
        "config_fingerprint_without_selection": config.fingerprint_without_selection,
        "config": config.raw,
        "pilot_plant": config.pilot_plant.name,
        "pretraining": pre_records,
        "final_grids": grids,
        "extensions": extensions,
        "rounds": rounds,
        "selection": selection,
        "results": all_results,
        "failures": failures,
        "wall_seconds": time.perf_counter() - t0,
    }
    out_path.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    print(f"wrote {out_path}")


def run_main(args: argparse.Namespace, config: StaticTransferConfig) -> None:
    suffix = ".json" if args.dry_run else ".json.gz"
    out_path = RUNS_DIR / f"{args.name}{suffix}"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
    require_clean_tree()
    prov = provenance()
    selection_check: dict[str, Any] | None = None
    if args.dry_run:
        for m, grid in config.raw["pilot"]["grids"].items():
            if config.selected(m) is None:  # before the pilot: the first value of each grid
                config.raw["selected"][m] = {k: v[0] for k, v in grid.items()}
    else:
        selection_check = _check_selection(config)
    conditions = args.conditions or [c.name for c in config.conditions]
    plants = [p.name for p in config.plants]
    seeds = list(config.seeds)
    header = {
        "commit": git("rev-parse", "HEAD"),
        "config": config.fingerprint,
        "phase": "main",
        "dry_run": args.dry_run,
        "conditions": conditions,
    }
    cache = Cache(RUNS_DIR / f"_{args.name}_cache.jsonl", header, args.resume)
    t0 = time.perf_counter()
    ctx = study.build_context(config, seeds)
    print(f"context built in {time.perf_counter() - t0:.0f} s", flush=True)
    checkpoints, pre_records = _pretrain(
        config, seeds, args.name, args.dry_run, cache, args.workers
    )
    steps_cap = 5 if args.dry_run else None
    l1_cap = 3 if args.dry_run else None
    tasks = [
        ("unit", p, c, s, steps_cap, l1_cap) for c in conditions for p in plants for s in seeds
    ]
    failures = _run_pool(tasks, cache, args.workers, ctx, checkpoints)
    units = [
        cache.done[json.dumps(list(t))]["result"]
        for t in tasks
        if json.dumps(list(t)) in cache.done
    ]
    payload = {
        "provenance": prov,
        "study": "static_transfer",
        "phase": "main",
        "dry_run": args.dry_run,
        "config_path": str(DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)),
        "config_fingerprint": config.fingerprint,
        "config": config.raw,
        "selection_check": selection_check,
        "conditions": conditions,
        "plants": plants,
        "seeds": seeds,
        "test_sets": {k: v.x for k, v in ctx.tests.items()},
        "designs": {str(s): d.x for s, d in ctx.designs.items()},
        "coverage": coverage(ctx),
        "deviation_table": deviation_table(ctx),
        "pretraining": pre_records,
        "units": units,
        "failures": failures,
        "unit_seconds": [u["seconds"] for u in units],
        "peak_memory_mb_per_worker_max": max(
            (u["peak_memory_mb"] for u in units), default=float("nan")
        ),
        "wall_seconds": time.perf_counter() - t0,
        "timing_context": f"workers = {args.workers} on this machine; not a result",
    }
    if args.dry_run:
        out_path.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    else:
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, default=_json_default)
    print(
        f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB); {len(failures)} failed units"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=["pilot", "main"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.workers > 8:
        raise SystemExit("at most 8 workers (physical cores of this machine)")
    config = load_config()
    if args.phase == "pilot":
        run_pilot(args, config)
    else:
        run_main(args, config)


if __name__ == "__main__":
    main()
