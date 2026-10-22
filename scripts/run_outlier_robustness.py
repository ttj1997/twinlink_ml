r"""Driver of the outlier robustness study (``docs/27_outlier_robustness.md``): all jobs, at most 8 workers.

Design (fixed beforehand; settings in ``conf/sensor/outliers_placeholder.yaml``, placeholders):

* outlier rate p in {0, 0.05, 0.25}; type in {independent, burst} for p > 0;
* preprocessing in {none, causal Hampel (window 8)}, identical for every model;
* models persistence, ARX, GRU, LSTM, Transformer; ``n_train`` 16 and 64; seeds 1-5; protocol of
  arm N of Befund 94 (``run_outlier_robustness_worker.py``);
* side check: GRU only, p = 0.25, both types, Hampel windows 4 and 16.

Budget rule (task of 2026-10-22): the run time is extrapolated from a probe run before the start.
If it is above 12 h, first the bursts are restricted to p = 0.25 (``--bursts-only-at-025``), then
the window side check is dropped (``--skip-window-check``). Both cuts are flags so that the plan of
the actual run is recorded in the timing file.

Job order: tier 1 = the conditions of the pre-registered readings (p = 0 without filter, p = 0.25
both types with and without filter); tier 2 = p = 0 with filter (control) and p = 0.05; tier 3 = the
window side check. Within a tier the longest jobs first. A job is one condition and one seed with
either GRU, LSTM and the two baselines at both budgets, or the Transformer at one budget.

    tools/run_here.sh scripts/run_outlier_robustness.py --run-name main --workers 8 [--resume]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_outlier_robustness_worker as worker

from twinlink_ml.outliers import Condition, load_outlier_config

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
SIZES = (16, 64)
SEEDS = (1, 2, 3, 4, 5)
MAX_WORKERS = 8
FILTER_WINDOW = 8
# core-hours per job from the probe run of 2026-10-22 (only used for the order and the estimate;
# the estimate actually used is printed by --estimate and recorded in docs/27)
COST_H = {"small": 0.45, ("transformer", 16): 1.6, ("transformer", 64): 5.0, "side": 0.25}


@dataclass(frozen=True)
class Job:
    """One worker process."""

    condition: Condition
    seed: int
    models: tuple[str, ...]
    sizes: tuple[int, ...]
    tier: int

    @property
    def name(self) -> str:
        models = "-".join(self.models)
        sizes = "-".join(str(n) for n in self.sizes)
        return f"{self.condition.id}_{models}_n{sizes}_seed{self.seed}"

    @property
    def expected_records(self) -> int:
        return worker.expected_record_count(list(self.models), list(self.sizes))

    @property
    def cost_h(self) -> float:
        if self.tier == 3:
            return COST_H["side"] * len(self.sizes)
        if self.models == ("transformer",):
            return COST_H[("transformer", self.sizes[0])]
        return COST_H["small"]

    def command(self, output: Path) -> list[str]:
        return [
            sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_outlier_robustness_worker.py"),
            "--condition", self.condition.id,
            "--seed", str(self.seed),
            "--models", *self.models,
            "--sizes", *(str(n) for n in self.sizes),
            "--output", str(output),
        ]  # fmt: skip


def main_conditions(*, bursts_only_at_025: bool) -> list[tuple[Condition, int]]:
    """``(condition, tier)`` of the main grid."""
    rates = load_outlier_config().outliers.rates
    out: list[tuple[Condition, int]] = []
    for window in (None, FILTER_WINDOW):
        out.append((Condition(0.0, "none", window), 1 if window is None else 2))
    for rate in rates:
        if rate == 0.0:
            continue
        for kind in ("independent", "burst"):
            if kind == "burst" and bursts_only_at_025 and rate != 0.25:
                continue
            for window in (None, FILTER_WINDOW):
                out.append((Condition(rate, kind, window), 1 if rate == 0.25 else 2))
    return out


def side_conditions() -> list[Condition]:
    windows = load_outlier_config().hampel.side_check_windows
    return [Condition(0.25, kind, w) for kind in ("independent", "burst") for w in windows]


def build_jobs(*, seeds: list[int], bursts_only_at_025: bool, skip_window_check: bool) -> list[Job]:
    jobs: list[Job] = []
    for condition, tier in main_conditions(bursts_only_at_025=bursts_only_at_025):
        for seed in seeds:
            jobs.append(Job(condition, seed, ("persistence", "arx", "gru", "lstm"), SIZES, tier))
            for n in SIZES:
                jobs.append(Job(condition, seed, ("transformer",), (n,), tier))
    if not skip_window_check:
        for condition in side_conditions():
            for seed in seeds:
                jobs.append(Job(condition, seed, ("gru",), SIZES, 3))
    return sorted(jobs, key=lambda job: (job.tier, -job.cost_h, job.name))


def estimate_hours(jobs: list[Job], workers: int) -> tuple[float, float]:
    """(core-hours, wall-hours by list scheduling in job order)."""
    finish = [0.0] * workers
    for job in jobs:
        index = finish.index(min(finish))
        finish[index] += job.cost_h
    return sum(job.cost_h for job in jobs), max(finish)


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break  # truncated last line: the job counts as incomplete
    return records


def job_complete(path: Path, job: Job) -> bool:
    return len(read_records(path)) == job.expected_records


def check_provenance(existing_commits: set[str]) -> str:
    from twinlink_ml.utils.code_location import require_code_location
    from twinlink_ml.utils.tracking import environment_fingerprint

    require_code_location(REPO_ROOT)
    fingerprint = environment_fingerprint()
    if fingerprint["git/dirty"]:
        raise SystemExit(
            "working tree is dirty -- commit first (CLAUDE.md: clean tree before a run)"
        )
    commit = str(fingerprint["git/commit"])
    stale = existing_commits - {commit}
    if stale:
        raise SystemExit(
            f"existing records carry git/commit {sorted(stale)}, HEAD is {commit}: resuming would "
            "mix two code states"
        )
    return commit


def merge(work_dir: Path, jobs: list[Job], target: Path) -> int:
    if target.exists():
        raise SystemExit(
            f"{target} exists -- artifacts are never overwritten; use a new --run-name"
        )
    seen: set[tuple] = set()
    lines = []
    for job in jobs:
        for record in read_records(work_dir / f"{job.name}.jsonl"):
            p = record["params"]
            if p["git/dirty"]:
                raise SystemExit(f"{job.name}: git/dirty")
            key = (
                p["condition/id"],
                p["master_seed"],
                p["n_train"],
                p["model/label"],
                p["test_set"],
            )
            if key in seen:
                raise SystemExit(f"duplicate record {key}")
            seen.add(key)
            lines.append(json.dumps(record))
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--bursts-only-at-025", action="store_true")
    parser.add_argument("--skip-window-check", action="store_true")
    parser.add_argument("--resume", action="store_true", help="keep the complete jobs of the run")
    parser.add_argument(
        "--estimate", action="store_true", help="print the plan and the estimate only"
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")

    jobs = build_jobs(
        seeds=arguments.seeds,
        bursts_only_at_025=arguments.bursts_only_at_025,
        skip_window_check=arguments.skip_window_check,
    )
    core_h, wall_h = estimate_hours(jobs, arguments.workers)
    conditions = sorted({job.condition.id for job in jobs})
    print(
        f"{len(jobs)} jobs, {len(conditions)} conditions, estimate {core_h:.1f} core-hours, "
        f"{wall_h:.1f} h wall with {arguments.workers} workers",
        flush=True,
    )
    if arguments.estimate:
        for condition in conditions:
            print(f"  {condition}")
        return

    prefix = f"outlier_robustness_{arguments.run_name}"
    target = RUNS_DIR / f"{prefix}.jsonl.gz"
    timing_path = RUNS_DIR / f"{prefix}_timing.json"
    for path in (target, timing_path):
        if path.exists():
            raise SystemExit(f"{path} exists -- use a new --run-name")
    work_dir = RUNS_DIR / f"_outlier_{arguments.run_name}"  # underscore: git-ignored scratch
    if work_dir.exists() and not arguments.resume and any(work_dir.glob("*.jsonl")):
        raise SystemExit(f"{work_dir} has records -- pass --resume or use a new --run-name")
    work_dir.mkdir(parents=True, exist_ok=True)

    existing_commits = {
        r["params"].get("git/commit")
        for job in jobs
        if job_complete(work_dir / f"{job.name}.jsonl", job)
        for r in read_records(work_dir / f"{job.name}.jsonl")
    }
    commit = check_provenance({c for c in existing_commits if c})
    todo = [job for job in jobs if not job_complete(work_dir / f"{job.name}.jsonl", job)]
    print(f"commit {commit[:8]}: {len(todo)} of {len(jobs)} jobs to run", flush=True)

    timing_file = work_dir / "timing.json"
    timing: dict[str, dict] = (
        json.loads(timing_file.read_text(encoding="utf-8")) if timing_file.exists() else {}
    )
    env = {**os.environ, "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    running: list[tuple[Job, subprocess.Popen, float]] = []
    failed: list[str] = []
    queue = list(todo)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    start = time.perf_counter()
    while queue or running:
        while queue and len(running) < arguments.workers:
            job = queue.pop(0)
            out = work_dir / f"{job.name}.jsonl"
            if out.exists():
                out.unlink()  # incomplete earlier attempt: start the job from scratch
            log = (work_dir / f"{job.name}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                job.command(out), cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            running.append((job, process, time.perf_counter()))
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} started {job.name}", flush=True)
        time.sleep(15)
        for entry in list(running):
            job, process, began = entry
            if process.poll() is None:
                continue
            running.remove(entry)
            seconds = time.perf_counter() - began
            ok = process.returncode == 0 and job_complete(work_dir / f"{job.name}.jsonl", job)
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} {'finished' if ok else 'FAILED'} "
                f"{job.name} in {seconds:.0f} s ({len(queue)} queued, {len(running)} running)",
                flush=True,
            )
            if ok:
                timing[job.name] = {
                    "wall_s": seconds,
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                timing_file.write_text(json.dumps(timing, indent=1), encoding="utf-8")
            else:
                failed.append(job.name)
    session_wall_min = (time.perf_counter() - start) / 60.0
    print(f"wall time of this session {session_wall_min:.1f} min", flush=True)
    sessions_file = work_dir / "sessions.json"
    sessions = (
        json.loads(sessions_file.read_text(encoding="utf-8")) if sessions_file.exists() else []
    )
    sessions.append({"started": started_at, "wall_min": session_wall_min, "jobs_run": len(todo)})
    sessions_file.write_text(json.dumps(sessions, indent=1), encoding="utf-8")
    if failed:
        raise SystemExit(f"failed jobs: {failed}; see the logs in {work_dir}; re-run with --resume")
    from twinlink_ml.utils.tracking import environment_fingerprint

    fingerprint = environment_fingerprint()
    n = merge(work_dir, jobs, target)
    print(f"wrote {target} ({n} records)")
    timing_path.write_text(
        json.dumps(
            {
                "provenance": fingerprint,
                "workers": arguments.workers,
                "plan": {
                    "bursts_only_at_025": arguments.bursts_only_at_025,
                    "skip_window_check": arguments.skip_window_check,
                    "conditions": conditions,
                    "seeds": arguments.seeds,
                    "estimate_core_h": core_h,
                    "estimate_wall_h": wall_h,
                },
                "sessions": sessions,
                "jobs": timing,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {timing_path}")


if __name__ == "__main__":
    main()
