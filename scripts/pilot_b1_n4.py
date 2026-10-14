r"""B1 pilot: B1 (random steps) and B2 (two-scale multi-level PRBS) at N = 4 runs, run times.

Purpose: measure run times of every stage (data generation, test sets, windows, fit, evaluation)
with 8 worker processes, for the run-time estimate of the main run (B1.5). The error figures are
NOT a result and are not used for any rank statement (N = 4, five seeds, two designs).

Protocol (same as the draft of the main run, reduced):
* data: ``DataSpec.new_study(n_train_max=4, n_val=0, n_test=0)``; pool of the plan with the
  seed-dependent data seed; ``ValidationPolicy.NONE`` (no validation runs, fixed epoch budget:
  all N runs are training runs, so N is the cost);
* models: persistence (parameter free), ARX, LSTM (composite, one network per target);
* test sets T1 (neutral mix), T2 (constant operation), T3 (deployment profiles), generated once
  from fixed test seeds, disjoint from all training runs (checked);
* five seeds (master seeds 1-5), the same seeds for both designs (paired);
* metric: run-median RMSE in physical units at horizons 1 and 24 (CLAUDE.md: compare in physical
  units); NRMSE against the measurement is recorded too.

    tools/run_here.sh scripts/pilot_b1_n4.py --anchors artifacts/runs/b1_t3_anchors.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (
    CONFIG_PATH,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.data.splits import SplitName
from twinlink_ml.evaluation.b1_testsets import (
    HoldoutSet,
    assert_disjoint,
    build_t1,
    build_t2,
    build_t3,
)
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.evaluation.experiments import DataSpec
from twinlink_ml.evaluation.growing_pool import (
    GrowingPool,
    prepare_process_config,
    window_dataset,
)
from twinlink_ml.models.registry import build_factories
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.excitation_information import (
    arx_fisher_information,
    fir_fisher_information,
    normalise_actions,
    spectral_coverage,
)
from twinlink_ml.simulation.excitation_plans import make_plan
from twinlink_ml.utils.seeding import SeedBundle

N_RUNS = 4
DESIGNS = ("b1", "b2")
MODELS = ("persistence", "arx", "lstm")
HORIZONS = (1, 24)
MASTER_SEEDS = (1, 2, 3, 4, 5)

_STATE: dict[str, Any] = {}


def _init(config: Any, spec: DataSpec, holdouts: list[HoldoutSet]) -> None:
    import torch

    torch.set_num_threads(1)
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    _STATE.update(config=config, spec=spec, holdouts=holdouts)


def _job(task: tuple[str, int]) -> dict[str, Any]:
    design, master_seed = task
    config, spec, holdouts = _STATE["config"], _STATE["spec"], _STATE["holdouts"]
    seeds = SeedBundle.from_master(master_seed)
    timings: dict[str, float] = {}
    start = time.perf_counter()

    prepared = prepare_process_config(config, spec)
    plan = make_plan(
        design,
        prepared,
        sample_time_s=spec.sample_time_s,
        run_duration_s=spec.run_duration_s,
        solid_fraction_process=spec.solid_fraction_process,
    )
    schedule = plan.build(N_RUNS, seed=seeds.data_seed)
    pool = GrowingPool(config, spec, schedule, noise_seed=seeds.noise_seed)
    pool.add(list(range(N_RUNS)))
    trajectory = pool.trajectory()
    timings["generate_s"] = time.perf_counter() - start
    assert_disjoint(trajectory, holdouts)

    train = window_dataset(trajectory, spec.window, role=SplitName.TRAIN)
    tests = {
        h.name: window_dataset(
            h.trajectory, spec.window, role=SplitName.TEST, scalers=train.scalers
        )
        for h in holdouts
    }

    bounds = prepared.actions.bounds
    lower = np.array([b.minimum for b in bounds])
    span = np.array([b.span for b in bounds])
    normalised = normalise_actions(trajectory.actions, lower, span)
    outputs = trajectory.outputs[..., [0, 2]]
    standardised = (outputs - outputs.mean(axis=(0, 1))) / outputs.std(axis=(0, 1))
    timings_info = time.perf_counter()
    spectral = spectral_coverage(
        normalised,
        spec.sample_time_s,
        band_low_hz=plan.timings.band_low_hz,
        band_high_hz=plan.timings.band_high_hz,
    )
    fir = fir_fisher_information(normalised)
    arx = arx_fisher_information(normalised, standardised)
    timings["information_s"] = time.perf_counter() - timings_info
    information = {
        "spectral_coverage": spectral.coverage,
        "spectral_flatness": spectral.flatness,
        "max_abs_cross_correlation": spectral.max_abs_cross_correlation,
        "fir_logdet_per_parameter": fir.logdet_per_parameter,
        "fir_lambda_min": fir.lambda_min,
        "fir_condition": fir.condition,
        "arx_logdet_per_parameter": arx.logdet_per_parameter,
        "arx_lambda_min": arx.lambda_min,
        "arx_condition": arx.condition,
    }

    factories = build_factories(MODELS)
    evaluator = RolloutEvaluator(horizons=HORIZONS)
    results: dict[str, dict[str, Any]] = {}
    for label, factory in factories.items():
        model = factory()  # type: ignore[operator]
        fit_start = time.perf_counter()
        model.fit(train, seed=seeds.model_seed, validation=None)
        fit_s = time.perf_counter() - fit_start
        eval_start = time.perf_counter()
        per_set = {
            name: evaluator.evaluate(model, dataset).as_flat_metrics()
            for name, dataset in tests.items()
        }
        results[label] = {
            "fit_s": fit_s,
            "evaluate_s": time.perf_counter() - eval_start,
            "n_parameters": model.n_parameters,
            "metrics": per_set,
        }
    timings["total_s"] = time.perf_counter() - start
    return {
        "design": design,
        "master_seed": master_seed,
        "n_train": N_RUNS,
        "n_train_windows": train.n_windows,
        "timings": timings,
        "information": information,
        "models": results,
    }


def _summary(records: list[dict[str, Any]]) -> str:
    lines = [
        "run-median RMSE, physical units, median over seeds [min .. max]  (pilot, NOT a result)"
    ]
    for test in ("T1", "T2", "T3"):
        for target in ("energy_target", "product_moisture_kg_per_kg"):
            for horizon in HORIZONS:
                key = f"rmse_run_median_physical/{target}/h{horizon}"
                row = [f"{test} {target:<28} h{horizon:<2}"]
                for model in MODELS:
                    cells = []
                    for design in DESIGNS:
                        values = [
                            r["models"][model]["metrics"][test][key]
                            for r in records
                            if r["design"] == design
                        ]
                        cells.append(
                            f"{design}: {np.median(values):.4g} [{min(values):.3g}..{max(values):.3g}]"
                        )
                    row.append(f"{model}: " + ", ".join(cells))
                lines.append("  ".join(row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="pilot_b1_b2_n4")
    arguments = parser.parse_args()
    if arguments.workers > 8:
        raise SystemExit("at most 8 workers")
    require_clean_tree()
    if (RUNS_DIR / f"{arguments.name}.json").exists():
        raise SystemExit("artifact exists; use a new --name")
    anchors = json.loads(arguments.anchors.read_text(encoding="utf-8"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    spec = DataSpec.new_study(n_train_max=N_RUNS, n_val=0, n_test=0)
    stage: dict[str, float] = {}

    start = time.perf_counter()
    t1 = build_t1(config, spec)
    stage["t1_s"] = time.perf_counter() - start
    start = time.perf_counter()
    t2 = build_t2(config, spec)
    stage["t2_s"] = time.perf_counter() - start
    start = time.perf_counter()
    t3 = build_t3(config, spec, anchors)
    stage["t3_s"] = time.perf_counter() - start
    holdouts = [t1, t2, t3]
    print(f"test sets: {stage}", flush=True)

    tasks = [(design, seed) for seed in MASTER_SEEDS for design in DESIGNS]
    wall_start = time.perf_counter()
    context = mp.get_context("spawn")
    with context.Pool(
        arguments.workers, initializer=_init, initargs=(config, spec, holdouts)
    ) as pool:
        records = []
        for done, record in enumerate(pool.imap_unordered(_job, tasks), start=1):
            records.append(record)
            print(
                f"[{done}/{len(tasks)}] {record['design']} seed {record['master_seed']}: "
                f"{record['timings']['total_s']:.0f} s "
                f"(lstm fit {record['models']['lstm']['fit_s']:.0f} s)",
                flush=True,
            )
    wall_s = time.perf_counter() - wall_start
    records.sort(key=lambda r: (r["design"], r["master_seed"]))
    print(_summary(records), flush=True)
    write_artifact(
        arguments.name,
        {
            "_provenance": provenance(),
            "purpose": "run-time pilot of study B1 (B1 and B2 at N=4); error figures are not a result",
            "data_spec": spec.as_parameters(),
            "workers": arguments.workers,
            "n_jobs": len(tasks),
            "wall_s_jobs": wall_s,
            "test_set_build_s": stage,
            "test_sets": {h.name: {"n_runs": h.trajectory.n_runs} for h in holdouts},
            "models": list(MODELS),
            "horizons": list(HORIZONS),
            "records": records,
        },
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
