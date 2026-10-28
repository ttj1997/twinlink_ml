"""Storage-price study: rules, MPC and RL for heat pump plus storage under the hourly day-ahead price.

Study document: ``docs/32_storage_price.md``; settings and placeholders:
``conf/storage_price/storage_price.yaml``; reading (pre-registered, committed before the run): the
docstring of ``scripts/analyze_storage_price.py``.

Jobs (one process each, at most 6 workers -- other studies run in parallel on the machine):

* ``rules:<case>`` for the main case and every variant of the YAML: tune ``kappa`` (R0) and ``q``
  (R1) on the 8 training months, then run N0, P0, R0, R1, M1 and M2 (every horizon of the config)
  and the hindsight optimum P* on the 4 test months (one run per month, empty storage at the start,
  end credit). The main case also records N0, P0, R0 and R1 on the training months (in-sample).
* ``rl:<algorithm>:<seed>`` (main case only): SAC / TD3 (stable-baselines3) trained on the training
  months for ``rl.total_steps`` steps; learning curve every ``rl.eval_every_steps`` steps on the
  training and the test months; the FINAL policy is the one evaluated (no selection on any curve).
  Checkpoints to ``artifacts/checkpoints/storage_price/<name>/`` (gitignored; copied to
  ``C:/twinlink_runs/storage-price/`` after the run).

Times per decision are measured inside every run and therefore UNDER PARALLEL LOAD (the workers of
this run plus whatever else runs on the machine); ``scripts/measure_storage_price_timing.py``
re-measures them in one process at the end.

Budget rule (task): extrapolate the run time from ``--probe-steps`` (marked as measured under
parallel load); above 30 h drop TD3 first (``--drop-algorithms TD3``), then the seeds to 2
(``--seeds 0,1``). Cuts are recorded in the artifact (``budget``).

Output: ``artifacts/runs/<name>.json.gz`` (once at the end, never overwritten); every finished job
is appended to ``artifacts/runs/_<name>_cache.jsonl`` (gitignored), ``--resume`` reuses it. Run from
the worktree: ``tools/run_here.sh scripts/run_storage_price.py --name storage_price_v1``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import REPO_ROOT, RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.storage_price import (  # noqa: E402
    build_case,
    evaluate_controllers,
    load_storage_price_config,
    rule_and_mpc_controllers,
    tune_r0,
    tune_r1,
    variant_names,
)
from twinlink_ml.storage_price.config import DEFAULT_CONFIG_PATH  # noqa: E402
from twinlink_ml.storage_price.controllers import NoPump, pump_only, r0_rule, r1_rule  # noqa: E402

MAX_WORKERS = 6
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "storage_price"
TIMING_NOTE = (
    "measured under parallel load: {workers} worker processes of this run, other studies may "
    "run on the machine at the same time"
)


# ------------------------------------------------------------------------------ worker side


def _worker_init() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def job_rules(config_path: str, variant: str) -> dict[str, Any]:
    """Tuning on the training months and the test runs of one case."""
    t0 = time.perf_counter()
    cfg = load_storage_price_config(config_path, variant)
    case = build_case(cfg, REPO_ROOT)
    t1 = time.perf_counter()
    r0 = tune_r0(case)
    r1 = tune_r1(case)
    t2 = time.perf_counter()
    ctrls = rule_and_mpc_controllers(case, r0["kappa"], r1["quantile"])
    test = evaluate_controllers(case, ctrls, cfg.test_months)
    t3 = time.perf_counter()
    out: dict[str, Any] = {
        "key": f"rules:{variant}",
        "variant": variant,
        "fingerprint": cfg.fingerprint,
        "cop": case.plant.cop,
        "heat_value_eur_per_kwh": case.market.heat_value,
        "end_value_eur_per_kwh": case.market.end_value,
        "r0_tuning": r0,
        "r1_tuning": r1,
        "test": test,
        "seconds": {"build": t1 - t0, "tuning": t2 - t1, "test": t3 - t2},
    }
    if variant == "main":
        c = cfg.controllers
        train_ctrls = [
            NoPump(case.plant, case.market),
            pump_only(case.plant, case.market, c.stored_value_epsilon),
            r0_rule(case.plant, case.market, r0["kappa"]),
            r1_rule(case.plant, case.market, r1["quantile"], c.r1_window_h, c.stored_value_epsilon),
        ]
        out["train"] = evaluate_controllers(case, train_ctrls, cfg.training_months)
    return out


def job_rl(
    config_path: str, algorithm: str, seed: int, total_steps: int, checkpoint_dir: str
) -> dict[str, Any]:
    """Training and evaluation of one agent on the main case."""
    from twinlink_ml.storage_price.train import train_agent

    t0 = time.perf_counter()
    cfg = load_storage_price_config(config_path, "main")
    case = build_case(cfg, REPO_ROOT)
    res = train_agent(
        case,
        algorithm,
        seed,
        total_steps=total_steps,
        checkpoint_dir=Path(checkpoint_dir),
        log=lambda m: print(m, flush=True),
    )
    res["key"] = f"rl:{algorithm}:{seed}"
    res["fingerprint"] = cfg.fingerprint
    res["seconds_total"] = time.perf_counter() - t0
    return res


# ------------------------------------------------------------------------------ driver side


class Cache:
    """Append-only JSONL of finished jobs with a header that must match on ``--resume``."""

    def __init__(self, path: Path, header: dict[str, Any], resume: bool) -> None:
        self.path = path
        self.done: dict[str, dict[str, Any]] = {}
        if path.exists():
            if not resume:
                raise SystemExit(f"{path} exists -- pass --resume or use a new name")
            lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
            if lines[0].get("header") != header:
                raise SystemExit(f"{path}: header differs from this run, cannot resume")
            self.done = {x["key"]: x for x in lines[1:]}
            print(f"resume: {len(self.done)} jobs from {path.name}")
        else:
            path.write_text(json.dumps({"header": header}) + "\n", encoding="utf-8")

    def add(self, result: dict[str, Any]) -> None:
        self.done[result["key"]] = result
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")


def _price_summary(config_path: str) -> dict[str, Any]:
    cfg = load_storage_price_config(config_path, "main")
    case = build_case(cfg, REPO_ROOT)
    p = case.prices
    month = p.clock.month
    spot = p.spot_eur_per_kwh
    out: dict[str, Any] = {
        "raw_file": cfg.prices.raw_file,
        "sha256": p.raw_sha256,
        "license_info": p.license_info,
        "source": cfg.prices.source,
        "url": cfg.prices.url,
        "retrieved": cfg.prices.retrieved,
        "hours": p.n_hours,
    }
    for label, months in (
        ("year", tuple(range(1, 13))),
        ("train", cfg.training_months),
        ("test", cfg.test_months),
    ):
        sel = np.isin(month, months)
        daily = spot[sel].reshape(-1, 24)
        out[f"spot_{label}"] = {
            "mean_eur_per_kwh": float(np.mean(spot[sel])),
            "min_eur_per_kwh": float(np.min(spot[sel])),
            "max_eur_per_kwh": float(np.max(spot[sel])),
            "negative_hours": int(np.sum(spot[sel] < 0.0)),
            "mean_daily_spread_eur_per_kwh": float(np.mean(daily.max(1) - daily.min(1))),
        }
    return out


def _run_jobs(
    pool: ProcessPoolExecutor, jobs: list[tuple[str, Any, tuple[Any, ...]]], cache: Cache
) -> None:
    pending: dict[Future[dict[str, Any]], str] = {}
    for key, fn, args in jobs:
        if key in cache.done:
            continue
        pending[pool.submit(fn, *args)] = key
    started = time.time()
    while pending:
        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
        for fut in finished:
            key = pending.pop(fut)
            res = fut.result()
            cache.add(res)
            print(f"[{(time.time() - started) / 60:6.1f} min] done {key}", flush=True)


def _probe_estimate(results: list[dict[str, Any]], cfg: Any, workers: int) -> dict[str, Any]:
    """Extrapolation of the full run from the probe jobs (marked as measured under load)."""
    rl = cfg.rl
    rules = [r for r in results if r["key"].startswith("rules:")]
    agents = [r for r in results if r["key"].startswith("rl:")]
    rules_s = max(sum(r["seconds"].values()) for r in rules)
    n_cases = 1 + len(variant_names(DEFAULT_CONFIG_PATH))
    per_algo: dict[str, dict[str, float]] = {}
    for a in sorted({r["algorithm"] for r in agents}):
        mine = [r for r in agents if r["algorithm"] == a]
        steps = mine[0]["total_steps"]
        learn = steps - min(rl.learning_starts, steps)
        s_per_step = float(np.median([r["train_seconds"] for r in mine])) / max(1, steps)
        s_per_learn = s_per_step * steps / max(1, learn)
        evals = len(mine[0]["learning_curve"]) + 1
        s_per_eval = float(np.median([r["eval_seconds"] for r in mine])) / evals
        n_evals = math.ceil(rl.total_steps / rl.eval_every_steps) + 1
        job = rl.total_steps * s_per_learn + n_evals * s_per_eval
        per_algo[a] = {
            "probe_steps": steps,
            "seconds_per_step_incl_updates": s_per_learn,
            "seconds_per_evaluation": s_per_eval,
            "job_hours": job / 3600.0,
        }
    # the full configured set of jobs; an algorithm that was not probed counts as the slowest one
    longest = max(p["job_hours"] for p in per_algo.values())
    rl_jobs = len(rl.algorithms) * len(rl.seeds)
    total_h = (math.ceil(n_cases / workers) * rules_s) / 3600.0 + math.ceil(
        rl_jobs / workers
    ) * longest
    return {
        "note": TIMING_NOTE.format(workers=workers),
        "rules_job_hours": rules_s / 3600.0,
        "cases": n_cases,
        "rl": per_algo,
        "rl_jobs": rl_jobs,
        "workers": workers,
        "estimated_total_hours": total_h,
        "budget_hours": 30.0,
        "within_budget": total_h <= 30.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-dir", default=str(RUNS_DIR))
    parser.add_argument("--probe-steps", type=int, default=None, help="probe run (budget)")
    parser.add_argument("--drop-algorithms", default="", help="budget rule, recorded")
    parser.add_argument("--seeds", default=None, help="budget rule (e.g. 0,1), recorded")
    parser.add_argument("--budget-probe", default=None, help="probe JSON, copied into the record")
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS} (parallel studies)")

    cfg = load_storage_price_config(args.config, "main")
    out_dir = Path(args.out_dir).resolve()
    probe = args.probe_steps is not None
    artifact_run = out_dir == RUNS_DIR.resolve() and not probe
    if out_dir == RUNS_DIR.resolve() and probe:
        raise SystemExit("probe runs must not write into artifacts/runs")
    dropped = [a for a in args.drop_algorithms.split(",") if a]
    if set(dropped) - set(cfg.rl.algorithms):
        raise SystemExit(f"unknown algorithms {dropped}")
    algorithms = [a for a in cfg.rl.algorithms if a not in dropped]
    seeds = list(cfg.rl.seeds) if args.seeds is None else [int(s) for s in args.seeds.split(",")]
    if set(seeds) - set(cfg.rl.seeds):
        raise SystemExit("--seeds must be a subset of the configured seeds")
    total_steps = args.probe_steps if probe else cfg.rl.total_steps
    suffix = "_probe" if probe else ""
    output = out_dir / f"{args.name}{suffix}.json.gz"
    cache_path = out_dir / f"_{args.name}{suffix}_cache.jsonl"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if artifact_run:
        require_clean_tree([output, cache_path])
    else:
        print(f"probe run into {out_dir}: not an artifact")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINT_ROOT / f"{args.name}{suffix}"
    prov = provenance("not used", ignore_paths=[output, cache_path])
    header = {
        "commit": git("rev-parse", "HEAD"),
        "fingerprint": cfg.fingerprint,
        "total_steps": total_steps,
        "algorithms": algorithms,
        "seeds": seeds,
    }
    cache = Cache(cache_path, header, args.resume)
    cases = ["main"] if probe else ["main", *variant_names(args.config)]
    jobs: list[tuple[str, Any, tuple[Any, ...]]] = [
        (f"rules:{v}", job_rules, (args.config, v)) for v in cases
    ]
    jobs += [
        (f"rl:{a}:{s}", job_rl, (args.config, a, s, total_steps, str(ckpt_dir)))
        for a in algorithms
        for s in seeds
    ]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        _run_jobs(pool, jobs, cache)
    results = [cache.done[k] for k, _, _ in jobs]
    if probe:
        estimate = _probe_estimate(results, cfg, args.workers)
        payload = {"probe": estimate, "header": header, "provenance": prov}
        (out_dir / f"{args.name}_probe.json").write_text(json.dumps(payload, indent=1))
        print(json.dumps(estimate, indent=1))
        return 0
    budget: dict[str, Any] = {
        "dropped_algorithms": dropped,
        "seeds": seeds,
        "rule": "above 30 h: drop TD3 first, then seeds to 2",
    }
    if args.budget_probe:
        budget["probe"] = json.loads(Path(args.budget_probe).read_text(encoding="utf-8"))["probe"]
    payload: dict[str, Any] = {
        "provenance": prov,
        "study": "storage price: rule, MPC or RL (docs/32_storage_price.md)",
        "config": cfg.raw,
        "variant_configs": {
            v: load_storage_price_config(args.config, v).raw for v in cases if v != "main"
        },
        "fingerprints": {v: load_storage_price_config(args.config, v).fingerprint for v in cases},
        "prices": _price_summary(args.config),
        "split": {"training_months": cfg.training_months, "test_months": cfg.test_months},
        "timing_note": TIMING_NOTE.format(workers=args.workers),
        "workers": args.workers,
        "budget": budget,
        "checkpoint_dir": str(ckpt_dir.relative_to(REPO_ROOT)),
        "rules": {r["variant"]: r for r in results if r["key"].startswith("rules:")},
        "rl": {r["key"]: r for r in results if r["key"].startswith("rl:")},
        "runtime_s": time.time() - started,
    }
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
