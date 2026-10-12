r"""B1 main run driver: learning curves of the arms B0-B5r, resumable, at most 8 workers.

Draft: ``docs/`` (B1 main-run draft). One job = one arm and one master seed (all N, nested). A job
appends its records to the JSONL log as soon as it is done, followed by a ``job_done`` marker; a
resumed run (``--resume``) skips the jobs that have a marker and discards nothing (the records of an
interrupted job have no marker and are ignored by the analysis, and the job is run again under a new
log line group). The first line is a header with the provenance, the data specification and the
fingerprints of the test sets.

    tools/run_here.sh scripts/run_b1_main.py --log artifacts/runs/b1_main.jsonl --workers 8
    tools/run_here.sh scripts/run_b1_main.py --log artifacts/runs/b1_main.jsonl --resume

Run only after the release of the draft, from a clean tree, on a local drive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (
    CONFIG_PATH,
    REPO_ROOT,
    provenance,
    require_clean_tree,
)

from twinlink_ml.evaluation.b1_learning_curve import ARMS, CurveConfig, run_arm
from twinlink_ml.evaluation.b1_testsets import (
    HoldoutSet,
    build_t1,
    build_t2,
    build_t3,
)
from twinlink_ml.evaluation.experiments import DataSpec
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

TARGETS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
_STATE: dict[str, Any] = {}


def _init(config: Any, spec: DataSpec, holdouts: list[HoldoutSet], curve: CurveConfig) -> None:
    import torch

    torch.set_num_threads(1)
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    _STATE.update(config=config, spec=spec, holdouts=holdouts, curve=curve)


def _job(task: tuple[str, int]) -> tuple[str, int, list[dict[str, Any]], float]:
    arm, seed = task
    start = time.perf_counter()
    records = run_arm(
        arm, seed, _STATE["config"], _STATE["spec"], _STATE["holdouts"], _STATE["curve"]
    )
    return arm, seed, records, time.perf_counter() - start


def _holdout_digest(holdout: HoldoutSet) -> str:
    return hashlib.sha256(holdout.trajectory.actions.tobytes()).hexdigest()[:12]


def _done_jobs(log: Path) -> set[tuple[str, int]]:
    if not log.exists():
        return set()
    done = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("type") == "job_done":
                done.add((record["arm"], record["master_seed"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--step-spread", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.workers > 8:
        raise SystemExit("at most 8 workers")
    if arguments.log.exists() and not arguments.resume:
        raise SystemExit("log exists; use --resume to continue it or a new --log name")
    if not arguments.resume:
        require_clean_tree()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    spec = DataSpec.new_study(
        n_train_max=max(arguments.sizes), n_val=0, n_test=0, target_names=TARGETS
    )
    curve = CurveConfig(sizes=tuple(arguments.sizes))
    anchors = json.loads(arguments.anchors.read_text(encoding="utf-8"))
    spread = json.loads(arguments.step_spread.read_text(encoding="utf-8"))

    holdouts = [
        build_t1(config, spec),
        build_t2(config, spec),
        build_t3(config, spec, anchors, spread),
    ]
    done = _done_jobs(arguments.log)
    tasks = [
        (arm, seed) for arm in arguments.arms for seed in arguments.seeds if (arm, seed) not in done
    ]
    # the expensive arm first so that the tail of the run is short
    tasks.sort(key=lambda t: (t[0] != "b5", t[0] != "b5r", t[1]))
    print(f"{len(tasks)} jobs to run, {len(done)} done", flush=True)

    arguments.log.parent.mkdir(parents=True, exist_ok=True)
    with arguments.log.open("a", encoding="utf-8") as log:
        if not arguments.resume or not done:
            header = {
                "type": "header",
                "provenance": provenance(),
                "repo_root": REPO_ROOT.name,
                "data_spec": spec.as_parameters(),
                "curve": {
                    "sizes": list(curve.sizes),
                    "models": list(curve.models),
                    "n_members": curve.n_members,
                    "per_plan": curve.per_plan,
                    "horizons": list(curve.horizons),
                },
                "seeds": arguments.seeds,
                "arms": arguments.arms,
                "workers": arguments.workers,
                "anchors": str(arguments.anchors.name),
                "step_spread": str(arguments.step_spread.name),
                "test_sets": {
                    h.name: {"n_runs": h.trajectory.n_runs, "digest": _holdout_digest(h)}
                    for h in holdouts
                },
            }
            log.write(json.dumps(header) + "\n")
            log.flush()
        context = mp.get_context("spawn")
        wall = time.perf_counter()
        with context.Pool(
            arguments.workers, initializer=_init, initargs=(config, spec, holdouts, curve)
        ) as pool:
            for finished, (arm, seed, records, elapsed) in enumerate(
                pool.imap_unordered(_job, tasks), start=1
            ):
                for record in records:
                    log.write(json.dumps(record) + "\n")
                log.write(
                    json.dumps(
                        {"type": "job_done", "arm": arm, "master_seed": seed, "elapsed_s": elapsed}
                    )
                    + "\n"
                )
                log.flush()
                print(
                    f"[{finished}/{len(tasks)}] {arm} seed {seed}: {elapsed:.0f} s "
                    f"(wall {time.perf_counter() - wall:.0f} s)",
                    flush=True,
                )


if __name__ == "__main__":
    mp.freeze_support()
    main()
