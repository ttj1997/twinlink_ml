"""Operating point and heat recovery, jointly or one after the other (docs/31_joint_recovery.md).

Settings: ``conf/joint_recovery/joint_recovery.yaml`` (reads P1, the test states and the optimiser
settings of ``conf/surrogate_opt/surrogate_rto.yaml``); package ``twinlink_ml.joint_recovery``;
reading (pre-registered): docstring of ``scripts/analyze_joint_recovery.py``.

Model: P1 of Befund 107, line heater -> dryer -> recovery of the flowsheet package, physical mode
(counter-current, saturation limit on, condensation in the recovery exchanger allowed), truth
parameters of P1 with the factor ``dT_min`` in {5, 10, 20} K, medium unpressurised (<= 95 degC)
or pressurised. Strategies S0-S3 per test state: module doc of ``twinlink_ml.joint_recovery.study``.

Tasks (process pool, at most ``--workers``):

phase 1
    ``reference/<dT>/<medium>``: :func:`twinlink_ml.surrogate_opt.study.reference` on the
    combination's family (test states, ``u*`` = S2 with the medium in its box, ``u0``);
    ``S0/<i>``: S0 at test state ``i`` (bypass; independent of ``dT_min`` and the medium).
phase 2
    ``media/<dT>/<medium>/<i>``: the medium flow optimal at S0's operating point (S1) and at
    ``u0`` (S3).

The assembled table (:func:`twinlink_ml.joint_recovery.study.assemble`) and every raw task record
go into ``artifacts/runs/<name>.json.gz`` (written once, never overwritten). Finished tasks are
appended to ``artifacts/runs/_<name>_cache.jsonl`` (gitignored); ``--resume`` reuses it (same
commit and config). ``--smoke`` runs a reduced study into ``--smoke-out`` (scratch, no artifact).
Run from the worktree: ``tools/run_here.sh scripts/run_joint_recovery.py --name joint_recovery_v1``.
Task times are wall-clock seconds of each worker; other computations may run on the machine at
the same time, so they are recorded as measured under parallel load.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gzip
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, _json_default, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.joint_recovery.study import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    JointRecoveryConfig,
    assemble,
    build_families,
    joint_test_states,
    load_joint_config,
    medium_optimum,
    strategy_s0,
)
from twinlink_ml.surrogate_opt.study import reference  # noqa: E402

MAX_WORKERS = 6
_WORKER: dict[str, Any] = {}


def _smoke(config: JointRecoveryConfig) -> JointRecoveryConfig:
    raw = copy.deepcopy(config.base.raw)
    raw["optimizer"]["truth"] = {"candidates": 128, "polish": 2}
    raw["optimizer"]["truth_check"] = {"candidates": 128, "polish": 2, "seed": 7}
    raw["optimizer"]["model"] = {"candidates": 64, "polish": 2}
    base = dataclasses.replace(config.base, test_count=4, raw=raw)
    return dataclasses.replace(
        config, base=base, medium_main={"candidates": 32, "polish": 2, "seed": 0},
        medium_check={"candidates": 32, "polish": 2, "seed": 7},
    )  # fmt: skip


def _config(smoke: bool) -> JointRecoveryConfig:
    config = load_joint_config()
    return _smoke(config) if smoke else config


def _init(smoke: bool) -> None:
    warnings.simplefilter("ignore")
    config = _config(smoke)
    _WORKER["config"] = config
    _WORKER["families"] = build_families(config)
    _WORKER["tw"] = joint_test_states(config)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _run(task: tuple[Any, ...], context: dict[str, Any]) -> dict[str, Any]:
    config: JointRecoveryConfig = _WORKER["config"]
    families, tw = _WORKER["families"], _WORKER["tw"]
    t0 = time.perf_counter()
    kind = task[0]
    if kind == "reference":
        out: Any = reference(families[(task[1], task[2])], config.base)
    elif kind == "S0":
        out = strategy_s0(families[config.combos[0]], tw[task[1]], config)
    elif kind == "media":
        family, i = families[(task[1], task[2])], task[3]
        out = {
            "S1": medium_optimum(family, tw[i], context["s0_u"], config),
            "S3": medium_optimum(family, tw[i], context["u0"], config),
        }
    else:
        raise ValueError(kind)
    return {
        "task": list(task),
        "seconds": time.perf_counter() - t0,
        "pid": os.getpid(),
        "result": _jsonable(out),
    }


def _pool_run(
    tasks: list[tuple[tuple[Any, ...], dict[str, Any]]],
    workers: int,
    smoke: bool,
    done: dict[str, dict[str, Any]],
    cache_path: Path,
    t0: float,
) -> None:
    todo = [(t, c) for t, c in tasks if json.dumps(list(t)) not in done]
    print(f"{len(todo)} tasks, {workers} workers", flush=True)
    if not todo:
        return
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(smoke,)) as pool:
        futures = {pool.submit(_run, t, c): t for t, c in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            done[json.dumps(rec["task"])] = rec
            with cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(
                f"[{time.perf_counter() - t0:7.0f} s] {rec['task']} {rec['seconds']:.1f} s",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-out", type=Path)
    args = ap.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS} (other studies run in parallel)")

    config = _config(args.smoke)
    if args.smoke:
        if args.smoke_out is None:
            raise SystemExit("--smoke needs --smoke-out (a scratch file, not an artifact)")
        out_path, cache_path = args.smoke_out, args.smoke_out.with_suffix(".cache.jsonl")
        prov: dict[str, Any] = {"smoke": True, "git/commit": git("rev-parse", "HEAD")}
    else:
        require_clean_tree()
        out_path = RUNS_DIR / f"{args.name}.json.gz"
        if out_path.exists():
            raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
        cache_path = RUNS_DIR / f"_{args.name}_cache.jsonl"
        prov = provenance()
    header = {"commit": git("rev-parse", "HEAD"), "config": config.fingerprint, "smoke": args.smoke}

    done: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        if not args.resume:
            raise SystemExit(f"{cache_path} exists -- pass --resume or use a new name")
        with cache_path.open(encoding="utf-8") as fh:
            first = json.loads(fh.readline())
            if first.get("header") != header:
                raise SystemExit(f"{cache_path} was written by another commit/config: {first}")
            for line in fh:
                rec = json.loads(line)
                done[json.dumps(rec["task"])] = rec
        print(f"resume: {len(done)} cached tasks")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"header": header}) + "\n", encoding="utf-8")

    families = build_families(config)
    tw = joint_test_states(config)
    n = len(tw)
    t0 = time.perf_counter()

    # phase 1: references (S2 with the medium in its box, u0) first, they are the longest
    phase1: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        (("reference", dt, m), {}) for dt, m in config.combos
    ] + [(("S0", i), {}) for i in range(n)]
    _pool_run(phase1, args.workers, args.smoke, done, cache_path, t0)

    def result(*task: Any) -> Any:
        return done[json.dumps(list(task))]["result"]

    s0 = [result("S0", i) for i in range(n)]
    refs = {(dt, m): result("reference", dt, m) for dt, m in config.combos}
    phase2 = [
        (("media", dt, m, i), {"s0_u": s0[i]["best"]["u"], "u0": refs[(dt, m)]["u0"]})
        for dt, m in config.combos
        for i in range(n)
    ]
    _pool_run(phase2, args.workers, args.smoke, done, cache_path, t0)
    wall = time.perf_counter() - t0

    warnings.simplefilter("ignore")
    media = {(dt, m, i): result("media", dt, m, i) for dt, m in config.combos for i in range(n)}
    table = assemble(config, families, tw, s0, refs, media)
    task_seconds = [r["seconds"] for r in done.values()]
    payload = {
        "provenance": prov,
        "study": "joint_recovery",
        "config_path": str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[2])),
        "config_fingerprint": config.fingerprint,
        "config": config.raw,
        "base_config_path": config.base_path,
        "base_config_fingerprint": config.base.fingerprint,
        "test_w": np.asarray(tw),
        "parameters": {
            f"{dt:g}K/{m}": [dataclasses.asdict(p) for p in families[(dt, m)].parameters]
            for dt, m in config.combos
        },
        "constraints": {
            f"{dt:g}K/{m}": [dataclasses.asdict(c) for c in families[(dt, m)].truth().constraints]
            for dt, m in config.combos
        },
        "timing": {
            "measured_under_parallel_load": True,
            "workers": args.workers,
            "wall_seconds": wall,
            "task_seconds_sum": float(sum(task_seconds)),
        },
        "table": table,
        "records": sorted(done.values(), key=lambda r: json.dumps(r["task"])),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, default=_json_default)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
