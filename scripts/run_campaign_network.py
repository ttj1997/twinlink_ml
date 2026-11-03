"""Campaign network: does a heat network pay when the consumers do not run all the time.

Study document: ``docs/35_campaign_network.md``; settings and placeholders:
``conf/campaign_network/campaign_network.yaml``; model ``twinlink_ml.campaign_network``; reading
(pre-registered, committed before the run): the docstring of ``scripts/analyze_campaign_network.py``.

Jobs (one process each):

* ``draw:<variant>:<a>:<mode>:<draw>`` -- K0, K1, K2 (every storage size > 0) and K3 (every
  storage size, 0 included) for every source power on one schedule draw. At ``a = 1`` every plant
  runs all year, so all draws and both modes are the same schedule: computed once
  (``draw:<variant>:1:independent:0``) and used for every draw and mode in the analysis.
* ``sweep:<variant>`` -- descriptive (declared before the run): K1 and K3 without storage at
  ``a = 1`` over a fine grid of source powers (saturation of the consumers, reference Befund 100).

Budget rule (task, wall-clock): above 30 h drop storage size 24 h first, then the draws to 3. The
cost is measured on a probe run first (``--probe``, not an artifact); cuts are recorded.

Output: ``artifacts/runs/<name>.json.gz`` (once at the end, never overwritten); every finished job
is appended to ``artifacts/runs/_<name>_cache.jsonl`` (gitignored), ``--resume`` reuses it. Run from
the worktree: ``tools/run_here.sh scripts/run_campaign_network.py --name campaign_network_v1``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.campaign_network import (  # noqa: E402
    build_consumers,
    evaluate_draw,
    key_figures,
    load_campaign_network_config,
    plant_schedules,
    run_k1,
    run_k3,
    variant_names,
)
from twinlink_ml.campaign_network.config import DEFAULT_CONFIG_PATH  # noqa: E402

MAX_WORKERS = 6
SWEEP_POWERS_KW = tuple(float(p) for p in range(100, 3001, 100))
"""Fine grid of the descriptive saturation sweep (kW)."""


def _worker_init() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def job_draw(
    config_path: str, variant: str, a: float, mode: str, draw: int, sizes: tuple[float, ...]
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_campaign_network_config(config_path, variant)
    res = evaluate_draw(cfg, a, mode, draw, storage_levels=sizes)
    res["key"] = f"draw:{variant}:{a:g}:{mode}:{draw}"
    res["variant"] = variant
    res["fingerprint"] = cfg.fingerprint
    res["seconds"] = time.perf_counter() - t0
    return res


def job_sweep(config_path: str, variant: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_campaign_network_config(config_path, variant)
    c = build_consumers(cfg, plant_schedules(cfg, 1.0, "independent", 0))
    rows = []
    for p in SWEEP_POWERS_KW:
        k1 = key_figures(run_k1(c, p, cfg), c, p, cfg)
        k3 = key_figures(run_k3(c, p, 0.0, cfg), c, p, cfg)
        rows.append({"power_kw": p, "K1": k1, "K3": k3})
    return {
        "key": f"sweep:{variant}",
        "variant": variant,
        "fingerprint": cfg.fingerprint,
        "rows": rows,
        "demand_kwh": {s.name: float(s.demand_kw.sum()) for s in c.streams},
        "seconds": time.perf_counter() - t0,
    }


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


def _jobs(
    config_path: str, variants: list[str], draws: int, sizes: tuple[float, ...]
) -> list[tuple[str, Any, tuple[Any, ...]]]:
    jobs: list[tuple[str, Any, tuple[Any, ...]]] = []
    for v in variants:
        cfg = load_campaign_network_config(config_path, v)
        for a in cfg.campaigns.availability_levels:
            if a >= 1.0:
                jobs.append(
                    (
                        f"draw:{v}:{a:g}:independent:0",
                        job_draw,
                        (config_path, v, a, "independent", 0, sizes),
                    )
                )
                continue
            for mode in cfg.campaigns.modes:
                for d in range(draws):
                    jobs.append(
                        (
                            f"draw:{v}:{a:g}:{mode}:{d}",
                            job_draw,
                            (config_path, v, a, mode, d, sizes),
                        )
                    )
        jobs.append((f"sweep:{v}", job_sweep, (config_path, v)))
    return jobs


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
            cache.add(fut.result())
            print(f"[{(time.time() - started) / 60:6.1f} min] done {key}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-dir", default=str(RUNS_DIR))
    parser.add_argument(
        "--probe", action="store_true", help="time a few jobs (budget), no artifact"
    )
    parser.add_argument("--drop-storage-24", action="store_true", help="budget rule, recorded")
    parser.add_argument("--draws", type=int, default=None, help="budget rule (3), recorded")
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")

    cfg = load_campaign_network_config(args.config, "main")
    variants = ["main", *variant_names(args.config)]
    sizes = tuple(s for s in cfg.storage.hours_levels if not (args.drop_storage_24 and s == 24.0))
    draws = cfg.campaigns.draws if args.draws is None else args.draws
    out_dir = Path(args.out_dir).resolve()
    if args.probe and out_dir == RUNS_DIR.resolve():
        raise SystemExit("probe runs must not write into artifacts/runs (use --out-dir)")
    suffix = "_probe" if args.probe else ""
    output = out_dir / f"{args.name}{suffix}.json.gz"
    cache_path = out_dir / f"_{args.name}{suffix}_cache.jsonl"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    artifact_run = out_dir == RUNS_DIR.resolve() and not args.probe
    if artifact_run:
        require_clean_tree([output, cache_path])
    else:
        print(f"run into {out_dir}: not an artifact")
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance("not used", ignore_paths=[output, cache_path])
    header = {
        "commit": git("rev-parse", "HEAD"),
        "fingerprints": {
            v: load_campaign_network_config(args.config, v).fingerprint for v in variants
        },
        "storage_hours": list(sizes),
        "draws": draws,
    }
    cache = Cache(cache_path, header, args.resume)
    jobs = _jobs(args.config, variants, draws, sizes)
    if args.probe:
        # one job per availability level of the main case, timed in one worker per job
        jobs = [j for j in jobs if j[0].startswith("draw:main:") and j[0].endswith(":0")][:4]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        _run_jobs(pool, jobs, cache)
    results = [cache.done[k] for k, _, _ in jobs]
    elapsed = time.time() - started
    if args.probe:
        n_full = len(_jobs(args.config, variants, cfg.campaigns.draws, cfg.storage.hours_levels))
        per_job = max(r["seconds"] for r in results)
        estimate = {
            "probe_jobs": [r["key"] for r in results],
            "max_job_seconds": per_job,
            "full_jobs": n_full,
            "workers": args.workers,
            "estimated_total_hours": per_job * -(-n_full // args.workers) / 3600.0,
            "budget_hours": 30.0,
        }
        (out_dir / f"{args.name}_probe.json").write_text(json.dumps(estimate, indent=1))
        print(json.dumps(estimate, indent=1))
        return 0
    payload: dict[str, Any] = {
        "provenance": prov,
        "study": "campaign network: heat network with intermittent consumers (docs/35_campaign_network.md)",
        "config": cfg.raw,
        "variant_configs": {v: load_campaign_network_config(args.config, v).raw for v in variants},
        "fingerprints": header["fingerprints"],
        "budget": {
            "storage_hours": list(sizes),
            "draws": draws,
            "rule": "above 30 h wall-clock: drop storage 24 h first, then draws to 3",
        },
        "a1_note": "a = 1: one schedule (all on), computed once, valid for every draw and mode",
        "workers": args.workers,
        "draws": {r["key"]: r for r in results if r["key"].startswith("draw:")},
        "sweeps": {r["variant"]: r for r in results if r["key"].startswith("sweep:")},
        "sweep_powers_kw": list(SWEEP_POWERS_KW),
        "runtime_s": elapsed,
    }
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
