r"""Driver of the re-run of the architecture comparison (``docs/15_plan_rerun_architecture_comparison.md``).

Starts ``run_arch_rerun_worker.py`` as independent processes, one thread each, at most ``--workers``
at a time (hard limit 8; the plan and the project lead cap it at 4 while studies A and B run). Jobs are
(arm, seed, method group): the Transformer is its own job because it dominates the cost; the other methods
(baselines, GRU, LSTM, xLSTM, Mamba) of an (arm, seed) form one job. Jobs are scheduled longest first
(costs from the committed timing records of Befund 58, only used for the order).

Budgets per arm (decision of the project lead, 2026-10-12): A0 and A1 at ``n_train`` 8 and 64, N at all four.

Resume: a job is complete if its JSONL has exactly the expected number of records; an incomplete job is
run again from scratch (its old file is replaced). Provenance guard: the working tree must be clean, and
records already present must carry the current ``git/commit``; otherwise abort. Merging into one JSONL per
arm checks for duplicate keys.

    tools/run_here.sh scripts/run_arch_rerun_sweep.py --run-name pilot --arms A0 A1 N --seeds 1 \
        --sizes-a 8 --sizes-n 8 --workers 4
"""

from __future__ import annotations

import argparse
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
MAX_WORKERS = 8
# seconds per (method, n_train), medians of Befund 58 (only for the order of the jobs)
COST_S = {
    "transformer": {8: 3267, 16: 5610, 32: 9304, 64: 14698},
    "xlstm": {8: 418, 16: 1095, 32: 1579, 64: 3477},
    "lstm": {8: 553, 16: 891, 32: 2949, 64: 1284},
    "gru": {8: 527, 16: 865, 32: 1502, 64: 1534},
    "mamba": {8: 292, 16: 761, 32: 1078, 64: 1992},
}
OTHERS = ("gru", "lstm", "xlstm", "mamba")


@dataclass(frozen=True)
class Job:
    """One worker process: an arm, a seed and a method group."""

    arm: str
    seed: int
    group: str  # "transformer" or "others"
    sizes: tuple[int, ...]

    @property
    def name(self) -> str:
        return f"{self.arm}_seed{self.seed}_{self.group}"

    @property
    def architectures(self) -> tuple[str, ...]:
        return ("transformer",) if self.group == "transformer" else OTHERS

    @property
    def skip_baselines(self) -> bool:
        return self.group == "transformer"

    @property
    def n_methods(self) -> int:
        return len(self.architectures) + (0 if self.skip_baselines else 3)

    @property
    def expected_records(self) -> int:
        return worker.expected_record_count(self.arm, list(self.sizes), self.n_methods)

    @property
    def cost_s(self) -> float:
        return sum(COST_S[m][n] for m in self.architectures for n in self.sizes) + 60.0


def build_jobs(
    arms: list[str], seeds: list[int], sizes_a: list[int], sizes_n: list[int]
) -> list[Job]:
    jobs = []
    for arm in arms:
        sizes = tuple(sizes_n if arm == "N" else sizes_a)
        for seed in seeds:
            for group in ("transformer", "others"):
                jobs.append(Job(arm, seed, group, sizes))
    return sorted(jobs, key=lambda job: -job.cost_s)


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


def check_provenance(existing_commits: set[str], ignore_paths: list[Path] | None = None) -> str:
    from twinlink_ml.utils.tracking import environment_fingerprint

    # ignore_paths: untracked files of the driver itself (its log, the preflight artifact)
    fingerprint = environment_fingerprint(ignore_paths=ignore_paths or [])
    if fingerprint["git/dirty"]:
        raise SystemExit(
            "working tree is dirty -- commit first (CLAUDE.md: clean tree before a run)"
        )
    commit = fingerprint["git/commit"]
    stale = existing_commits - {commit}
    if stale:
        raise SystemExit(
            f"existing records carry git/commit {sorted(stale)}, HEAD is {commit}: resuming would mix "
            "two code states"
        )
    return str(commit)


def merge(
    work_dir: Path, jobs: list[Job], arms: list[str], prefix: str, runs_dir: Path
) -> list[Path]:
    written = []
    for arm in arms:
        seen: set[tuple] = set()
        lines = []
        for job in jobs:
            if job.arm != arm:
                continue
            for record in read_records(work_dir / f"{job.name}.jsonl"):
                p = record["params"]
                key = (
                    p["master_seed"],
                    p["n_train"],
                    p["model/label"],
                    p["test_set"],
                    p["eval/pool"],
                    p["eval/window"],
                )
                if key in seen:
                    raise SystemExit(f"duplicate record {key} in arm {arm}")
                seen.add(key)
                lines.append(json.dumps(record))
        target = runs_dir / f"{prefix}_{arm}.jsonl"
        if target.exists():
            raise SystemExit(
                f"{target} exists -- artifacts are never overwritten; use a new --run-name"
            )
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arms", nargs="+", choices=worker.ARMS, default=list(worker.ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--sizes-a", type=int, nargs="+", default=[8, 64], help="A0 and A1")
    parser.add_argument("--sizes-n", type=int, nargs="+", default=[8, 16, 32, 64], help="N")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument(
        "--require-preflight",
        default=None,
        help="name of the preflight artifact (artifacts/runs/<name>.json) that must exist with "
        "passed = true before anything is trained",
    )
    parser.add_argument(
        "--ignore-untracked",
        type=Path,
        nargs="*",
        default=[],
        help="untracked files of the driver that do not make the tree dirty (log, preflight artifact)",
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")

    runs_dir = REPO_ROOT / "artifacts" / "runs"
    if arguments.require_preflight:
        gate_path = runs_dir / f"{arguments.require_preflight}.json"
        if not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8")).get(
            "passed"
        ):
            raise SystemExit(
                f"preflight artifact {gate_path.name} missing or not passed: nothing is trained"
            )
    work_dir = runs_dir / f"_rerun_{arguments.run_name}"  # underscore: git-ignored scratch
    work_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(arguments.arms, arguments.seeds, arguments.sizes_a, arguments.sizes_n)

    existing_commits = {
        r["params"].get("git/commit")
        for job in jobs
        for r in read_records(work_dir / f"{job.name}.jsonl")
        if job_complete(work_dir / f"{job.name}.jsonl", job)
    }
    commit = check_provenance({c for c in existing_commits if c}, arguments.ignore_untracked)
    todo = [job for job in jobs if not job_complete(work_dir / f"{job.name}.jsonl", job)]
    total_core_h = sum(job.cost_s for job in todo) / 3600.0
    print(
        f"commit {commit[:8]}: {len(todo)} of {len(jobs)} jobs to run, estimate {total_core_h:.1f} "
        f"core-hours (committed timing records), {arguments.workers} workers",
        flush=True,
    )

    env = {**os.environ, "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    running: list[tuple[Job, subprocess.Popen, float]] = []
    failed: list[str] = []
    queue = list(todo)
    start = time.perf_counter()
    while queue or running:
        while queue and len(running) < arguments.workers:
            job = queue.pop(0)
            out = work_dir / f"{job.name}.jsonl"
            if out.exists():
                out.unlink()  # incomplete earlier attempt: start the job from scratch
            command = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "scripts" / "run_arch_rerun_worker.py"),
                "--arm", job.arm,
                "--seed", str(job.seed),
                "--output", str(out),
                "--sizes", *map(str, job.sizes),
                "--architectures", *job.architectures,
            ]  # fmt: skip
            if job.skip_baselines:
                command.append("--skip-baselines")
            log = (work_dir / f"{job.name}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            running.append((job, process, time.perf_counter()))
            print(f"started {job.name} ({len(running)} running)", flush=True)
        time.sleep(15)
        for entry in list(running):
            job, process, began = entry
            if process.poll() is None:
                continue
            running.remove(entry)
            seconds = time.perf_counter() - began
            ok = process.returncode == 0 and job_complete(work_dir / f"{job.name}.jsonl", job)
            print(f"{'finished' if ok else 'FAILED'} {job.name} in {seconds:.0f} s", flush=True)
            if not ok:
                failed.append(job.name)
    print(f"wall time {(time.perf_counter() - start) / 60.0:.1f} min", flush=True)
    if failed:
        raise SystemExit(f"failed jobs: {failed}; see the logs in {work_dir}; re-run to resume")
    if not arguments.no_merge:
        for path in merge(work_dir, jobs, arguments.arms, f"rerun_{arguments.run_name}", runs_dir):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
