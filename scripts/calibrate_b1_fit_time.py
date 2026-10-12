r"""B1: calibration of the LSTM fit time against the pool size (three targets), for the main-run estimate.

The pilot measured the LSTM fit at N = 4 with two targets. The main run uses three targets (energy,
moisture, bed temperature) and N up to 32. This script times the composite LSTM (repository defaults,
200 epochs, no validation, one thread) on a B1 pool at N = 4, 8, 16, 32 in one process, and N = 32
with eight processes at the same time (the contended time of the main run). Nothing else.

    tools/run_here.sh scripts/calibrate_b1_fit_time.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (
    CONFIG_PATH,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.data.splits import SplitName
from twinlink_ml.evaluation.experiments import DataSpec
from twinlink_ml.evaluation.growing_pool import (
    GrowingPool,
    prepare_process_config,
    window_dataset,
)
from twinlink_ml.models.registry import build_factories
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.excitation_plans import make_plan
from twinlink_ml.utils.seeding import SeedBundle

TARGETS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
SIZES = (4, 8, 16, 32)


def _fit_time(task: tuple[int, int]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    n_train, master_seed = task
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    spec = DataSpec.new_study(n_train_max=32, n_val=0, n_test=0, target_names=TARGETS)
    seeds = SeedBundle.from_master(master_seed)
    prepared = prepare_process_config(config, spec)
    plan = make_plan(
        "b1",
        prepared,
        sample_time_s=spec.sample_time_s,
        run_duration_s=spec.run_duration_s,
        solid_fraction_process=spec.solid_fraction_process,
    )
    pool = GrowingPool(
        config, spec, plan.build(32, seed=seeds.data_seed), noise_seed=seeds.noise_seed
    )
    pool.add(list(range(n_train)))
    train = window_dataset(pool.trajectory(), spec.window, role=SplitName.TRAIN)
    model = build_factories(("lstm",))["lstm"]()  # type: ignore[operator]
    start = time.perf_counter()
    model.fit(train, seed=seeds.model_seed, validation=None)
    return {
        "n_train": n_train,
        "n_windows": train.n_windows,
        "fit_s": time.perf_counter() - start,
        "epochs_run": model.training_diagnostics().get("epochs_run"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="calibrate_b1_fit_time")
    arguments = parser.parse_args()
    require_clean_tree()
    if (RUNS_DIR / f"{arguments.name}.json").exists():
        raise SystemExit("artifact exists; use a new --name")
    context = mp.get_context("spawn")
    serial = []
    for n_train in SIZES:
        with context.Pool(1) as pool:
            serial.append(pool.map(_fit_time, [(n_train, 1)])[0])
        print(f"serial: {serial[-1]}", flush=True)
    with context.Pool(8) as pool:
        start = time.perf_counter()
        contended = pool.map(_fit_time, [(32, seed) for seed in range(1, 9)])
        wall = time.perf_counter() - start
    print(f"8 concurrent N=32: {[round(c['fit_s']) for c in contended]} (wall {wall:.0f} s)")
    write_artifact(
        arguments.name,
        {
            "_provenance": provenance(),
            "purpose": "fit time of the 3-target composite LSTM against N, serial and 8 concurrent",
            "targets": list(TARGETS),
            "serial": serial,
            "contended_n32": contended,
            "contended_wall_s": wall,
        },
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
