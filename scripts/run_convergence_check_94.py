r"""Driver of the convergence check of Befund 94 (addendum on the epoch cap): arm N at ``max_epochs`` 600.

Pre-registration (questions, cells, metrics, readings): docstring of ``analyze_convergence_check_94.py``,
committed together with this driver before any record exists.

What is trained: arm N of Befund 94 (``DataSpec.new_study()``, preset version 2), seeds 1-5, GRU, LSTM and
Transformer, ``n_train`` 16 and 64, with ``run_arch_rerun_worker.py`` unchanged. **The only change against
Befund 94 is** ``--net-max-epochs 600`` **instead of 200**; patience (20), learning rate, batch size, every
other hyperparameter, the data, the splits, the holdout (seed 999) and the evaluation variants are those of
the worker. The training loop has no learning-rate schedule, so the first 200 epochs of every head are
those of Befund 94 (checked by the reproduction gate of the analysis).

Jobs are (architecture, ``n_train``, seed), one worker process each with one thread, at most ``--workers``
at a time, longest first. Checkpoints go to ``artifacts/checkpoints/conv600/N`` (the checkpoints of Befund 94
are not touched). Single-job files and logs go to the git-ignored ``artifacts/runs/_conv600_<run-name>/``;
the merged records are written compressed to ``artifacts/runs/conv600_<run-name>_N.jsonl.gz`` (the
uncompressed records would exceed 5 MB) and the wall times to ``conv600_<run-name>_timing.json``. Neither is
ever overwritten.

Cost estimate (made before the run from the committed records ``rerun_full_N.jsonl``: seconds per epoch and
head = ``training/elapsed_s`` / sum of ``epochs_run`` over the three heads, under eight concurrent workers as
in Befund 94; a head that reached the cap of 200 is assumed to run to 600 (worst case) or to 400 (middle)):
134 core-hours worst case, 94 middle; the longest single job (Transformer, ``n_train`` 64, seed 2) 25.1 h
worst case, 16.7 h middle. Seconds per epoch and head, GRU / LSTM / Transformer: 0.74 / 0.80 / 13.3 at
``n_train`` 16 and 2.1 / 2.6 / 54 at 64 (Transformer / GRU 18x and 26x, not 14x). With eight workers the
wall time is bounded by the longest job, about 25 h worst case, below the limit of 36 h set by the project
lead, so both budgets are run (above 36 h only ``n_train`` 64 would have been run, ``--sizes 64``).

    tools/run_here.sh scripts/run_convergence_check_94.py --run-name main --workers 8
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

import run_arch_rerun_worker as worker

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
ARCHITECTURES = ("gru", "lstm", "transformer")
SIZES = (16, 64)
SEEDS = (1, 2, 3, 4, 5)
NET_MAX_EPOCHS = 600
MAX_WORKERS = 8
# worst-case hours per job from the cost estimate in the docstring (only used for the order)
COST_H = {"transformer": {16: 6.0, 64: 20.0}, "lstm": {16: 0.3, 64: 1.0}, "gru": {16: 0.3, 64: 0.9}}


@dataclass(frozen=True)
class Job:
    """One worker process: one architecture, one budget, one seed of arm N."""

    architecture: str
    n_train: int
    seed: int

    @property
    def name(self) -> str:
        return f"N_{self.architecture}_n{self.n_train}_seed{self.seed}"

    @property
    def expected_records(self) -> int:
        return worker.expected_record_count("N", [self.n_train], 1)

    @property
    def cost_h(self) -> float:
        return COST_H[self.architecture][self.n_train]

    def command(self, output: Path, checkpoint_dir: Path) -> list[str]:
        return [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "run_arch_rerun_worker.py"),
            "--arm", "N",
            "--seed", str(self.seed),
            "--sizes", str(self.n_train),
            "--architectures", self.architecture,
            "--skip-baselines",
            "--net-max-epochs", str(NET_MAX_EPOCHS),
            "--checkpoint-dir", str(checkpoint_dir),
            "--output", str(output),
        ]  # fmt: skip


def build_jobs(sizes: list[int], seeds: list[int]) -> list[Job]:
    jobs = [Job(a, n, s) for a in ARCHITECTURES for n in sizes for s in seeds]
    return sorted(jobs, key=lambda job: (-job.cost_h, job.name))


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
            f"existing records carry git/commit {sorted(stale)}, HEAD is {commit}: resuming would mix "
            "two code states"
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
            if int(p["model/net_max_epochs"]) != NET_MAX_EPOCHS:
                raise SystemExit(f"{job.name}: net_max_epochs {p['model/net_max_epochs']}")
            key = (
                p["master_seed"],
                p["n_train"],
                p["model/label"],
                p["test_set"],
                p["eval/pool"],
                p["eval/window"],
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
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES), choices=SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")

    prefix = f"conv600_{arguments.run_name}"
    target = RUNS_DIR / f"{prefix}_N.jsonl.gz"
    timing_path = RUNS_DIR / f"{prefix}_timing.json"
    for path in (target, timing_path):
        if path.exists():
            raise SystemExit(f"{path} exists -- use a new --run-name")
    work_dir = RUNS_DIR / f"_conv600_{arguments.run_name}"  # underscore: git-ignored scratch
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = REPO_ROOT / "artifacts" / "checkpoints" / "conv600" / "N"
    jobs = build_jobs(arguments.sizes, arguments.seeds)

    existing_commits = {
        r["params"].get("git/commit")
        for job in jobs
        if job_complete(work_dir / f"{job.name}.jsonl", job)
        for r in read_records(work_dir / f"{job.name}.jsonl")
    }
    commit = check_provenance({c for c in existing_commits if c})
    todo = [job for job in jobs if not job_complete(work_dir / f"{job.name}.jsonl", job)]
    print(
        f"commit {commit[:8]}: {len(todo)} of {len(jobs)} jobs to run, worst-case estimate "
        f"{sum(j.cost_h for j in todo):.1f} core-hours, {arguments.workers} workers",
        flush=True,
    )

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
                job.command(out, checkpoint_dir),
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
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
                f"{job.name} in {seconds:.0f} s",
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
        raise SystemExit(f"failed jobs: {failed}; see the logs in {work_dir}; re-run to resume")
    from twinlink_ml.utils.tracking import environment_fingerprint

    fingerprint = environment_fingerprint()  # before the merged files exist: clean tree expected
    n = merge(work_dir, jobs, target)
    print(f"wrote {target} ({n} records)")
    timing_path.write_text(
        json.dumps(
            {
                "provenance": fingerprint,
                "workers": arguments.workers,
                "net_max_epochs": NET_MAX_EPOCHS,
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
