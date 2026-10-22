"""Surrogate-based steady RTO with few true evaluations: learning curves over N (docs/25).

Study document: ``docs/25_surrogate_rto.md``; settings and placeholders:
``conf/surrogate_opt/surrogate_rto.yaml``; reading (pre-registered, per problem): docstring of
``scripts/analyze_surrogate_rto.py``.

Question: how many evaluations of the true steady model does a surrogate need so that the
optimisation on it hits the true energy optimum feasibly? Metric: regret at the TRUE model (not
the prediction error). True models (read only): P1 line heater -> dryer -> recovery of the
flowsheet package, physical mode; P2 steady grey-box model of the lab line with the R1 solver
(docs/24). Package: ``twinlink_ml.surrogate_opt``.

Tasks (all independent, run in a process pool, heaviest first):

``reference/<P>``
    50 test states (Sobol, fixed seed, own sequence), ``u*`` per state with its convergence check,
    the fixed operating point ``u0`` (as R0) with its margin, all scored on the truth.
``M1|M2|M3/<P>/seed``
    nested Sobol designs of the seed, fitted at every N, optimised at every kappa, scored.
``M4/<P>/seed``
    grey-box fit of the fitted parameters at every N (the hidden one stays nominal), optimised on
    the fitted physics model, scored.
``M2a/<P>/kappa/seed``
    GP with adaptive design (8 Sobol start points, then sequential at the predicted optima of new
    training disturbances), the GP at every N scored.

Output: ``artifacts/runs/<name>.json.gz`` (written once at the end, never overwritten). Every
finished task is appended to ``artifacts/runs/_<name>_cache.jsonl`` (gitignored); ``--resume``
reuses it (same commit and config required). ``--smoke`` runs a reduced study into the scratch
file given by ``--smoke-out`` (no artifact, no provenance check). Run from the worktree with
``tools/run_here.sh scripts/run_surrogate_rto.py --name surrogate_rto_v1``.
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

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, _json_default, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.surrogate_opt.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    SurrogateStudyConfig,
    build_problems,
    load_study_config,
)
from twinlink_ml.surrogate_opt.study import (  # noqa: E402
    adaptive_arm,
    blackbox_arm,
    greybox_arm,
    make_test_states,
    reference,
)

_WORKER: dict[str, Any] = {}


def _smoke(config: SurrogateStudyConfig) -> SurrogateStudyConfig:
    raw = copy.deepcopy(config.raw)
    raw["optimizer"]["truth"] = {"candidates": 256, "polish": 3}
    raw["optimizer"]["truth_check"] = {"candidates": 256, "polish": 3, "seed": 7}
    raw["models"]["m2"]["restarts"] = 0
    return dataclasses.replace(
        config, seeds=(0,), budgets=(8, 16), kappas=(0.0, 2.0), test_count=4, raw=raw
    )


def _init(smoke: bool) -> None:
    warnings.simplefilter("ignore")
    config = load_study_config()
    _WORKER["config"] = _smoke(config) if smoke else config
    _WORKER["problems"] = build_problems(_WORKER["config"])


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _run(task: tuple[Any, ...]) -> dict[str, Any]:
    config: SurrogateStudyConfig = _WORKER["config"]
    kind, key = task[0], task[1]
    family = _WORKER["problems"][key]
    t0 = time.perf_counter()
    lines: list[str] = []
    log = lines.append
    if kind == "reference":
        out: Any = reference(family, config)
    else:
        tw = make_test_states(family, config)
        if kind in ("M1", "M2", "M3"):
            out = blackbox_arm(family, kind, task[2], config, tw, log)
        elif kind == "M4":
            out = greybox_arm(family, task[2], config, tw, log)
        elif kind == "M2a":
            out = adaptive_arm(family, task[2], task[3], config, tw, log)
        else:
            raise ValueError(kind)
    return {
        "task": list(task),
        "seconds": time.perf_counter() - t0,
        "pid": os.getpid(),
        "result": _jsonable(out),
    }


def _tasks(config: SurrogateStudyConfig, problems: list[str]) -> list[tuple[Any, ...]]:
    tasks: list[tuple[Any, ...]] = []
    for p in problems:
        tasks += [("M4", p, s) for s in config.seeds]
        tasks.append(("reference", p))
        tasks += [("M2a", p, k, s) for k in config.kappas for s in config.seeds]
        tasks += [(m, p, s) for m in ("M2", "M3", "M1") for s in config.seeds]
    # P1's M4 (one line solve per model call, 50 optimisations per N) first
    return sorted(tasks, key=lambda t: (t[0] != "M4" or t[1] != "P1",))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--problems", nargs="+", default=["P1", "P2"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-out", type=Path)
    args = ap.parse_args()

    base = load_study_config()
    config = _smoke(base) if args.smoke else base
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

    tasks = [t for t in _tasks(config, args.problems) if json.dumps(list(t)) not in done]
    print(f"{len(tasks)} tasks, {args.workers} workers")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.smoke,)) as pool:
        futures = {pool.submit(_run, t): t for t in tasks}
        for fut in as_completed(futures):
            rec = fut.result()
            done[json.dumps(rec["task"])] = rec
            with cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(
                f"[{time.perf_counter() - t0:7.0f} s] {rec['task']} {rec['seconds']:.0f} s",
                flush=True,
            )

    families = build_problems(config)
    payload = {
        "provenance": prov,
        "study": "surrogate_rto",
        "config_path": str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[2])),
        "config_fingerprint": config.fingerprint,
        "config": config.raw,
        "settings": {
            "seeds": config.seeds,
            "budgets": config.budgets,
            "kappas": config.kappas,
            "test_count": config.test_count,
        },
        "problems": {
            k: {
                "decisions": f.truth().decisions.as_dict(),
                "disturbances": f.truth().disturbances.as_dict(),
                "constraints": [dataclasses.asdict(c) for c in f.truth().constraints],
                "parameters": [dataclasses.asdict(p) for p in f.parameters],
            }
            for k, f in families.items()
            if k in args.problems
        },
        "wall_seconds": time.perf_counter() - t0,
        "records": sorted(done.values(), key=lambda r: json.dumps(r["task"])),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, default=_json_default)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
