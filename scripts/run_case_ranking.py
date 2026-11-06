r"""Case-ranking study: the six cases of the partner workbook over the operating range (driver).

docs/NN_case_ranking.md; settings ``conf/case_ranking/case_ranking.yaml``; package
:mod:`twinlink_ml.case_ranking`. Every unit (a chunk of a Sobol set in one variant, a 1D cut, a row
of a 2D cut) evaluates its points in the three modes of docs/11 section 13.6 and is appended to
``artifacts/runs/_<name>_cache.jsonl`` as soon as it is done (``--resume`` reuses it; same commit
and config required). A failing unit is logged with its traceback and does not discard the others.

``--dry-run``: every unit with ``run.dry_run_points_per_chunk`` points; reports the seconds per
point by variant, the projected wall time of the main run with ``--workers`` and the peak memory per
worker. Output ``artifacts/runs/<name>.json`` (record) and ``<name>.npz`` (arrays,
:func:`twinlink_ml.case_ranking.study.assemble`). Never overwritten. Clean tree required; start
with ``tools/run_here.sh scripts/run_case_ranking.py --name ... [--dry-run] [--resume]``.
"""

from __future__ import annotations

import argparse
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

from _common import (  # noqa: E402
    NO_HEAT_NETWORK,
    RUNS_DIR,
    _json_default,
    git,
    provenance,
    require_clean_tree,
)

from twinlink_ml.case_ranking import study  # noqa: E402
from twinlink_ml.case_ranking.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    CaseRankingConfig,
    load_config,
)

REPO_ROOT = SCRIPTS.parent
_WORKER: dict[str, Any] = {}


def _init(config: CaseRankingConfig, dry_run: bool) -> None:
    _WORKER["config"] = config
    _WORKER["dry_run"] = dry_run


def _task_worker(task: tuple[Any, ...]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = study.run_task(_WORKER["config"], task, _WORKER["dry_run"])
        return {"task": list(task), "ok": True, "result": result, "pid": os.getpid()}
    except Exception:
        return {
            "task": list(task),
            "ok": False,
            "error": traceback.format_exc(),
            "seconds": time.perf_counter() - t0,
            "pid": os.getpid(),
        }


class Cache:
    """Append-only JSONL of finished units; the first line is the header."""

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
            print(f"resume: {len(self.done)} cached units", flush=True)
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
    config: CaseRankingConfig,
    dry_run: bool,
) -> list[dict[str, Any]]:
    todo = [t for t in tasks if json.dumps(list(t)) not in cache.done]
    # Long units first (optimised variant), so that the pool ends evenly.
    todo.sort(key=lambda t: (t[1] != "optimised", t[0] != "set"))
    failures: list[dict[str, Any]] = []
    print(f"{len(todo)} of {len(tasks)} units to run, {workers} workers", flush=True)
    if not todo:
        return failures
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(config, dry_run)) as pool:
        futures = [pool.submit(_task_worker, t) for t in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            cache.add(rec)
            elapsed = time.perf_counter() - t0
            eta = elapsed / k * (len(todo) - k)
            status = "ok" if rec["ok"] else "FAILED"
            extra = (
                f" {len(rec['result']['points'])} pts, {rec['result']['point_errors']} point errors"
                if rec["ok"]
                else ""
            )
            print(
                f"[{elapsed:7.0f} s, eta {eta:6.0f} s] {k}/{len(todo)} {rec['task']} "
                f"{status}{extra}",
                flush=True,
            )
            if not rec["ok"]:
                print(rec["error"], flush=True)
                failures.append(rec)
    return failures


def _timing(
    config: CaseRankingConfig, units: dict[str, dict[str, Any]], workers: int
) -> dict[str, Any]:
    """Seconds per point by (variant, kind) and the projection to the main run (dry run only)."""
    per: dict[str, list[float]] = {}
    for key, rec in units.items():
        t = json.loads(key)
        n = len(rec["result"]["points"])
        if n:
            per.setdefault(f"{t[1]}/{t[0]}", []).append(rec["result"]["seconds"] / n)
    sec = {k: float(np.median(v)) for k, v in per.items()}
    main_points: dict[str, int] = {}
    for t in study.task_list(config):
        main_points[f"{t[1]}/{t[0]}"] = main_points.get(f"{t[1]}/{t[0]}", 0) + len(
            study.task_points(config, t, dry_run=False)
        )
    cpu = {k: main_points[k] * sec.get(k, float("nan")) for k in main_points}
    total = float(sum(cpu.values()))
    return {
        "seconds_per_point_median": sec,
        "main_points": main_points,
        "main_cpu_seconds": cpu,
        "main_cpu_seconds_total": total,
        "main_wall_seconds_projected": total / workers,
        "workers": workers,
        "note": "projection = CPU seconds / workers (no overhead); timing, not a result",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    config = load_config()
    if args.workers > config.max_workers:
        raise SystemExit(f"at most {config.max_workers} workers (physical cores of this machine)")
    json_path = RUNS_DIR / f"{args.name}.json"
    npz_path = RUNS_DIR / f"{args.name}.npz"
    for p in (json_path, npz_path):
        if p.exists():
            raise SystemExit(f"{p} exists -- artifacts are never overwritten")
    require_clean_tree()
    prov = provenance(NO_HEAT_NETWORK)
    header = {
        "commit": git("rev-parse", "HEAD"),
        "config": config.fingerprint,
        "dry_run": args.dry_run,
    }
    cache = Cache(RUNS_DIR / f"_{args.name}_cache.jsonl", header, args.resume)
    t0 = time.perf_counter()
    tasks = study.task_list(config)
    failures = _run_pool(tasks, cache, args.workers, config, args.dry_run)
    wall = time.perf_counter() - t0
    done = {k: v for k, v in cache.done.items() if v.get("ok")}
    point_errors = sum(v["result"]["point_errors"] for v in done.values())
    payload: dict[str, Any] = {
        "provenance": prov,
        "study": "case_ranking",
        "dry_run": args.dry_run,
        "config_path": str(DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)),
        "config_fingerprint": config.fingerprint,
        "config": config.raw,
        "regions": {k: v.as_dict() for k, v in config.regions.items()},
        "box": config.box.as_dict(),
        "units_total": len(tasks),
        "units_done": len(done),
        "failures": failures,
        "point_errors": point_errors,
        "peak_memory_mb_per_worker_max": max(
            (v["result"]["peak_memory_mb"] for v in done.values()), default=float("nan")
        ),
        "wall_seconds": wall,
        "timing_context": f"workers = {args.workers} on this machine; not a result",
    }
    if args.dry_run:
        payload["timing"] = _timing(config, done, args.workers)
    arrays = study.assemble(config, {k: v["result"] for k, v in done.items()}, args.dry_run)
    np.savez_compressed(npz_path, **arrays)
    payload["arrays"] = npz_path.name
    payload["array_keys"] = sorted(arrays)
    json_path.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    print(
        f"wrote {json_path}; {len(failures)} failed units, {point_errors} point errors, "
        f"{wall:.0f} s"
    )
    if args.dry_run:
        t = payload["timing"]
        print(
            f"projected main run: {t['main_wall_seconds_projected'] / 60:.1f} min wall with "
            f"{args.workers} workers; peak memory per worker "
            f"{payload['peak_memory_mb_per_worker_max']:.0f} MB"
        )


if __name__ == "__main__":
    main()
