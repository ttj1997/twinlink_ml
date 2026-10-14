r"""Pre-flight gate of the full re-run: data checks BEFORE any training (docs/15 sections 4 and 12).

Generates the pools of the full run for all seeds and checks them against the old records of Befunde 51/58
(arm A0 = the old records) without training a single model:

* **A1** (legacy trajectories, planned future inputs): ``process/fingerprint`` equivalent to the old one,
  ``excitation/fingerprint``, ``split/digest`` and ``seed/*`` equal to the old records of that seed; the
  projection acts as before (guard active, > 0 projected steps: the trajectories are the legacy ones);
  persistence (reads the history only) reproduces the old persistence records of that seed within
  ``--tolerance`` at ``n_train`` 8 and 64 on both test sets.
* **N** (preset version 2): ``data/new_study_version`` 2, ``data/gas_velocity_cap`` ``initial_diameter``,
  the process fingerprint of the old records, an excitation fingerprint different from the old one, and the
  projection guard **0** projected steps in the pool and in the constant-run holdout pool.

Exit status 1 and a clear message if anything fails; the run is then not started. The artifact
``<name>.json`` is written from a clean tree.

    tools/run_here.sh scripts/preflight_rerun.py --name rerun_full_preflight
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_arch_rerun_worker as worker
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from check_rerun_gate import max_relative_deviation

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models import PersistenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent

FINGERPRINTS = ("excitation/fingerprint", "split/digest", "seed/data", "seed/noise", "seed/model")
SIZES = (8, 64)


def load_old() -> list[dict[str, Any]]:
    path = RUNS_DIR / "arch_learning_curve_projected.jsonl"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _gen(config: Any, spec: Any, seed: int) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        # the holdout pool has 36 runs, not a power of 2: the Sobol balance warning is expected
        warnings.filterwarnings("ignore", message="The balance properties of Sobol")
        return generate_data(config, spec, master_seed=seed)


def persistence_metrics(
    pool: Any, holdout: Any, spec: Any, n_train: int
) -> dict[str, dict[str, float]]:
    train, _, test = build_datasets(pool, n_train)
    constant = TrajectoryWindowDataset.from_trajectory(
        holdout.trajectory, holdout.split, SplitName.TEST, spec.window, scalers=train.scalers
    )
    model = PersistenceModel()
    model.fit(train, seed=pool.seeds.model_seed, validation=None)
    evaluator = RolloutEvaluator(horizons=(1, spec.horizon))
    return {
        "excited": dict(evaluator.evaluate(model, test).as_flat_metrics()),
        "constant_holdout": dict(evaluator.evaluate(model, constant).as_flat_metrics()),
    }


def check_a1(config: Any, seed: int, old: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    failures: list[str] = []
    spec = worker.arm_spec("A1")
    pool = _gen(config, spec, seed)
    holdout = _gen(config, worker.holdout_spec(spec), worker.HOLDOUT_SEED)
    reference = next(
        r["params"]
        for r in old
        if int(r["params"]["master_seed"]) == seed and r["params"]["n_train"] == 8
    )
    if not fingerprints_equivalent(
        pool.process_config.fingerprint, str(reference["process/fingerprint"])
    ):
        failures.append(f"A1 seed {seed}: process fingerprint differs from the old one")
    new_params = {**pool.fingerprints, **pool.seeds.as_parameters()}
    for name in FINGERPRINTS:
        if new_params.get(name) != reference.get(name):
            failures.append(
                f"A1 seed {seed}: {name} {new_params.get(name)!r} != old {reference.get(name)!r}"
            )
    guard = pool.projection_guard
    if not guard["projection_active"] or guard["projected_steps"] <= 0:
        failures.append(
            f"A1 seed {seed}: the legacy pool should be limited by the projection: {guard}"
        )
    deviations = {}
    for n_train in SIZES:
        new = persistence_metrics(pool, holdout, spec, n_train)
        for test_set, metrics in new.items():
            record = next(
                r
                for r in old
                if r["params"]["model/label"] == "persistence"
                and int(r["params"]["master_seed"]) == seed
                and int(r["params"]["n_train"]) == n_train
                and r["params"]["test_set"] == test_set
            )
            worst, where, count = max_relative_deviation(metrics, record["metrics"])
            deviations[f"n{n_train}/{test_set}"] = {"worst": worst, "at": where, "metrics": count}
            if worst > tolerance:
                failures.append(
                    f"A1 seed {seed} n={n_train} {test_set}: persistence {where} off by {worst:.2e}"
                )
    return {"seed": seed, "guard_pool": guard, "guard_holdout": holdout.projection_guard,
            "persistence_vs_old": deviations, "failures": failures}  # fmt: skip


def check_n(config: Any, seed: int, old: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    spec = worker.arm_spec("N")
    pool = _gen(config, spec, seed)
    holdout = _gen(config, worker.holdout_spec(spec), worker.HOLDOUT_SEED)
    reference = next(
        r["params"]
        for r in old
        if int(r["params"]["master_seed"]) == seed and r["params"]["n_train"] == 8
    )
    if spec.new_study_version != 2 or spec.gas_velocity_cap != "initial_diameter":
        failures.append(f"N seed {seed}: spec is not preset version 2 with the cap")
    if pool.fingerprints.get("preset/new_study_version") != "2":
        failures.append(f"N seed {seed}: preset version missing from the fingerprints")
    if not fingerprints_equivalent(
        pool.process_config.fingerprint, str(reference["process/fingerprint"])
    ):
        failures.append(f"N seed {seed}: process fingerprint differs from the old one")
    if pool.fingerprints["excitation/fingerprint"] == reference["excitation/fingerprint"]:
        failures.append(f"N seed {seed}: excitation fingerprint equals the old one")
    for label, data in (("pool", pool), ("holdout pool", holdout)):
        guard = data.projection_guard
        if not guard["projection_active"]:
            failures.append(f"N seed {seed} {label}: projection is not on as a guard")
        if guard["projected_steps"] != 0 or guard["runs_with_projection"] != 0:
            failures.append(f"N seed {seed} {label}: the guard acted: {guard}")
    return {"seed": seed, "guard_pool": pool.projection_guard, "guard_holdout": holdout.projection_guard,
            "excitation_fingerprint": pool.fingerprints["excitation/fingerprint"], "failures": failures}  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--tolerance", type=float, default=1e-6)
    arguments = parser.parse_args()
    require_clean_tree()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(worker.old_worker.CONFIG_PATH)
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )
    old = load_old()
    a1, n_arm = [], []
    for seed in arguments.seeds:
        a1.append(check_a1(config, seed, old, arguments.tolerance))
        n_arm.append(check_n(config, seed, old))
        print(f"seed {seed} checked", flush=True)
    failures = [f for entry in a1 + n_arm for f in entry["failures"]]
    write_artifact(
        arguments.name,
        {
            "_provenance": provenance(),
            "tolerance": arguments.tolerance,
            "seeds": arguments.seeds,
            "a1": a1,
            "n": n_arm,
            "failures": failures,
            "passed": not failures,
        },
    )
    print("PREFLIGHT PASSED" if not failures else "PREFLIGHT FAILED")
    for failure in failures[:40]:
        print("  ", failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
