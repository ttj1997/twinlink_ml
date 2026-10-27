"""Time per decision of every controller of the storage-price study, in ONE process.

The run (``scripts/run_storage_price.py``) measures the times inside parallel workers, i.e. under
load. This script repeats the test-month runs of the main case one controller after the other in a
single process (one thread for BLAS and torch) and records the time of every ``decide`` call. Other
studies may still run on the machine: the CPU load of the machine is sampled before and after every
controller (Windows ``Win32_Processor.LoadPercentage`` via PowerShell) and recorded, so the numbers are marked either way.

The tuned parameters and the RL checkpoints are taken from the run artifact; the cost of every run
is compared with the artifact (a determinism check, the largest relative difference is recorded).

Output: ``artifacts/runs/<name>.json`` (never overwritten). Run from the worktree:
``tools/run_here.sh scripts/measure_storage_price_timing.py --artifact storage_price_v1
--name storage_price_v1_timing``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import REPO_ROOT, RUNS_DIR, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.storage_price import (  # noqa: E402
    build_case,
    load_storage_price_config,
    rule_and_mpc_controllers,
    run_months,
)
from twinlink_ml.storage_price.env import PolicyController  # noqa: E402


def _load() -> float | None:
    """Machine CPU load in % (None if it cannot be read)."""
    import subprocess

    cmd = "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return float(out.stdout.strip().replace(",", "."))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    out_path = RUNS_DIR / f"{args.name}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
    require_clean_tree([out_path])
    with gzip.open(RUNS_DIR / f"{args.artifact}.json.gz", "rt", encoding="utf-8") as handle:
        art = json.load(handle)
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import SAC, TD3

    cfg = load_storage_price_config()
    if cfg.fingerprint != art["fingerprints"]["main"]:
        raise SystemExit("config fingerprint differs from the run artifact")
    case = build_case(cfg, REPO_ROOT)
    main_rec = art["rules"]["main"]
    ctrls = rule_and_mpc_controllers(
        case, main_rec["r0_tuning"]["kappa"], main_rec["r1_tuning"]["quantile"]
    )
    reference = dict(main_rec["test"])
    ckpt_dir = REPO_ROOT / art["checkpoint_dir"]
    loaders = {"SAC": SAC, "TD3": TD3}
    for run in art["rl"].values():
        model = loaders[run["algorithm"]].load(ckpt_dir / run["checkpoint_final"], device="cpu")
        name = f"{run['algorithm']}_s{run['seed']}"
        ctrls.append(PolicyController(case.plant, case.market, case.prices.clock, model, name))
        reference[name] = run["test"]
    tol = cfg.controllers.violation_tolerance_relative
    results: dict[str, Any] = {}
    for ctrl in ctrls:
        before = _load()
        t0 = time.perf_counter()
        rows = run_months(
            ctrl, case.plant, case.market, case.prices.clock, cfg.test_months, tolerance=tol
        )
        wall = time.perf_counter() - t0
        after = _load()
        ref = reference[ctrl.name]
        diff = max(
            abs(r.cost_eur - x["cost_eur"]) / abs(x["cost_eur"])
            for r, x in zip(rows, ref, strict=True)
        )
        results[ctrl.name] = {
            "decide_ms_median": float(np.median([r.decide_s_median for r in rows]) * 1e3),
            "decide_ms_mean": float(np.mean([r.decide_s_mean for r in rows]) * 1e3),
            "decide_ms_p95_max_month": float(max(r.decide_s_p95 for r in rows) * 1e3),
            "decide_ms_max": float(max(r.decide_s_max for r in rows) * 1e3),
            "decisions": int(sum(r.hours for r in rows)),
            "wall_s": wall,
            "machine_cpu_percent_before_after": [before, after],
            "largest_relative_cost_difference_to_run": diff,
        }
        print(
            f"{ctrl.name:10s} {results[ctrl.name]['decide_ms_median']:.4f} ms  load {before}/{after} %  diff {diff:.1e}",
            flush=True,
        )
    payload = {
        "provenance": provenance("not used", ignore_paths=[out_path]),
        "artifact": args.artifact,
        "note": (
            "single process, one thread (BLAS, torch), controllers one after the other; machine "
            "CPU load sampled before and after every controller (other studies may run)"
        ),
        "controllers": results,
    }
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
