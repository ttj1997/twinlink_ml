r"""Train the GRU soft sensors with measured process responses (F5) of the follow-up study (docs/29).

The pipeline of the predecessor's F3 (``scripts/train_sensor_value_soft_sensor.py``: Befund 94 arm N,
``DataSpec.new_study()`` preset version 2, process config ``conf/process/greybox_gab_placeholder.yaml``
with the gas-velocity projection, own excited pool per master seed, run-level splits, validation
pool, operating mode ``SOFT_SENSOR``, moisture head alone, ``n_train`` 64, ``max_epochs`` 200, model
seed of the pool's seed bundle), extended only by the input list: the window configuration carries
``measured_input_names`` (``WindowConfig``), so the moisture head sees the seven exogenous channels
PLUS the measured process responses (``twinlink_ml.sensor_value.responses``; existing sensor model).

Three training variants per master seed (settings ``conf/sensor_value/sensor_value_2.yaml``):

* ``nom``  -- the pool of the nominal model (the F3 pool of the same seed), inputs bed temperature,
  exhaust temperature, exhaust humidity;
* ``DR``   -- domain randomisation: the same excitation, initial states and split, every run
  re-simulated with its own draw of the seven parameters (critical moisture, nu_max, five GAB) from
  the same bands as the 20 test draws but from another scrambled Sobol' sequence (seed of the
  config); the draws are checked to be disjoint from the 20 test draws (no common point; smallest
  distance in the unit cube recorded);
* ``DRoY`` -- the DR pool without the exhaust humidity as input.

Writes one checkpoint per (variant, seed) to ``artifacts/checkpoints/sensor_value_2/`` (gitignored)
and the record ``artifacts/runs/<name>.json`` with the SHA-256 of every checkpoint, the training
diagnostics and the test errors of the soft sensor in kg/kg against the noise-free moisture (h = 1
and h = 36) on three test sets: the excited test runs of the nominal pool, the excited test runs of
the randomised pool (both of the same seed) and the constant-run holdout (seed 999, nominal), and
the input ranges of the measured responses in the training runs. Run with
``tools/run_here.sh scripts/train_sensor_value_2_soft_sensor.py --name sensor_value_2_soft_sensor_v1``
from a clean tree.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

ROOT = SCRIPTS.parent
PROCESS_FINGERPRINT = "bddca37f716f5cfc"
TARGET = "product_moisture_kg_per_kg"
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints" / "sensor_value_2"


def _windows_error(model: Any, dataset: Any) -> dict[str, float]:
    """RMSE in kg/kg against the noise-free moisture at h = 1 and h = H, pooled and run median."""
    batch = dataset.batch(range(dataset.n_windows))
    prediction = model.predict(batch)
    scaler = dataset.target_scaler
    predicted = scaler.inverse(prediction)[..., 0]
    truth = scaler.inverse(batch.future_targets_truth)[..., 0]
    runs = batch.run_index.astype(int)
    out: dict[str, float] = {}
    for h in (1, predicted.shape[1]):
        err = predicted[:, h - 1] - truth[:, h - 1]
        per_run = [float(np.sqrt(np.mean(err[runs == r] ** 2))) for r in np.unique(runs)]
        out[f"rmse_pooled_h{h}"] = float(np.sqrt(np.mean(err**2)))
        out[f"rmse_run_median_h{h}"] = float(np.median(per_run))
        out[f"bias_h{h}"] = float(np.mean(err))
    return out


def randomised_draws(follow_up: Any, training_config: Any) -> tuple[list[Any], list[int], dict]:
    """Per-run draws of the DR pool and the disjointness check against the 20 test draws."""
    from twinlink_ml.sensitivity.config import load_sensitivity_config
    from twinlink_ml.sensor_value import parameter_draws
    from twinlink_ml.sensor_value.config import DrawSettings

    base = follow_up.base
    sensitivity = load_sensitivity_config(base.sensitivity_config)
    test_draws, _ = parameter_draws(base.draws, sensitivity, training_config)
    pool_size = rerun_spec().pool_size
    settings = DrawSettings(
        n=pool_size,
        seed=follow_up.soft.randomisation_seed,
        sequence_length=follow_up.soft.randomisation_sequence_length,
        parameters=base.draws.parameters,
    )
    if settings.seed == base.draws.seed:
        raise SystemExit("the randomisation seed must differ from the seed of the test draws")
    draws, rejected = parameter_draws(settings, sensitivity, training_config)
    test_units = np.array([[d.unit[p] for p in settings.parameters] for d in test_draws])
    units = np.array([[d.unit[p] for p in settings.parameters] for d in draws])
    distances = np.linalg.norm(units[:, None, :] - test_units[None, :, :], axis=-1)
    check = {
        "common_points": int(np.sum(distances == 0.0)),
        "min_unit_distance": float(distances.min()),
        "test_draw_seed": base.draws.seed,
        "randomisation_seed": settings.seed,
    }
    if check["common_points"]:
        raise SystemExit(f"randomised draws are not disjoint from the test draws: {check}")
    return draws, rejected, check


def rerun_spec(smoke: bool = False) -> Any:
    import run_arch_rerun_worker as rerun_worker

    return rerun_worker.arm_spec("N", smoke=smoke)


def train_one(
    variant: dict[str, Any], seed: int, name: str, n_train: int, max_epochs: int, smoke: bool
) -> dict[str, Any]:
    """Train and save one soft sensor; returns its record."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    import run_arch_learning_curve_worker as old_worker
    import run_arch_rerun_worker as rerun_worker
    import run_outlier_robustness_worker as outlier_worker
    from _common import git

    from twinlink_ml.data.splits import SplitName
    from twinlink_ml.data.windows import TrajectoryWindowDataset
    from twinlink_ml.evaluation import ValidationPolicy, build_datasets
    from twinlink_ml.evaluation.protocol import ObservationMode, spec_for_mode
    from twinlink_ml.sensor_value.config import load_follow_up_config
    from twinlink_ml.sensor_value.responses import randomised_data, with_exhaust_channels
    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
    from twinlink_ml.utils.checkpoints import save_checkpoint
    from twinlink_ml.utils.seeding import _derive

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(old_worker.CONFIG_PATH)
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )
    follow_up = load_follow_up_config()
    spec = rerun_worker.arm_spec("N", smoke=smoke)
    started = time.perf_counter()
    own = outlier_worker._generate(config, spec, seed)
    holdout = outlier_worker._generate(
        config, rerun_worker.holdout_spec(spec), rerun_worker.HOLDOUT_SEED
    )
    if not smoke and own.process_config.fingerprint != PROCESS_FINGERPRINT:
        raise RuntimeError(f"process/fingerprint {own.process_config.fingerprint!r}")
    exhaust_seed = _derive(seed, "F5/exhaust_noise")
    nominal_pool = with_exhaust_channels(own, exhaust_seed)
    holdout_pool = with_exhaust_channels(
        holdout, _derive(rerun_worker.HOLDOUT_SEED, "F5/exhaust_noise")
    )
    draws, rejected, disjoint = randomised_draws(follow_up, own.process_config)
    if smoke:
        draws = draws[: len(own.schedule.run_id)]
    generation_start = time.perf_counter()
    randomised_pool = with_exhaust_channels(
        randomised_data(
            own,
            [d.process for d in draws[: len(own.schedule.run_id)]],
            [_derive(seed, "F5/DR/noise", r) for r in range(len(own.schedule.run_id))],
        ),
        exhaust_seed,
    )
    randomisation_s = time.perf_counter() - generation_start
    pool = randomised_pool if variant["data"] == "randomised" else nominal_pool
    inputs = tuple(variant["inputs"])

    def soft_view(data: Any) -> Any:
        moisture = outlier_worker.moisture_view(data, TARGET)
        return dataclasses.replace(
            moisture, spec=spec_for_mode(moisture.spec, ObservationMode.SOFT_SENSOR)
        )

    own_soft = soft_view(pool)
    if own_soft.spec.window.observed_targets:
        raise RuntimeError("the moisture channel must be unobserved")
    window = dataclasses.replace(own_soft.spec.window, measured_input_names=inputs)
    train, validation, test = build_datasets(
        own_soft, n_train, policy=ValidationPolicy.FIXED_POOL, window=window
    )
    model = old_worker._build_model(old_worker._architecture_factory("gru"), max_epochs)
    fit_start = time.perf_counter()
    model.fit(train, seed=own.seeds.model_seed, validation=validation)
    fit_s = time.perf_counter() - fit_start

    def test_set(data: Any) -> Any:
        view = soft_view(data)
        return TrajectoryWindowDataset.from_trajectory(
            view.trajectory, view.split, SplitName.TEST, window, scalers=train.scalers
        )

    label = variant["label"]
    path = CHECKPOINT_DIR / f"{name}_{label}_seed{seed}.pt"
    if path.exists():
        raise SystemExit(f"{path} exists -- checkpoints are never overwritten; use a new name")
    data_spec = {
        **own_soft.spec.as_parameters(),
        "data/observed_target_names": [],
        "data/measured_input_names": list(inputs),
        "data/training_data": variant["data"],
    }
    save_checkpoint(
        path,
        model,
        model_label="gru",
        run_uid=f"{name}_{label}_seed{seed}",
        target_names=train.target_names,
        exogenous_names=train.exogenous_names,
        n_exogenous=len(train.exogenous_names),
        input_scaler=train.input_scaler,
        target_scaler=train.target_scaler,
        process_fingerprint=own.process_config.fingerprint,
        data_spec_params=data_spec,
        seed_bundle_params=own.seeds.as_parameters(),
        git_commit=git("rev-parse", "HEAD"),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    split = own.split.subset(n_train)
    rows = [pool.trajectory.run_id.index(r) for r in split.train]
    names = list(pool.trajectory.metadata["output_names"])
    measured = pool.trajectory.measurements[rows][..., [names.index(n) for n in inputs]]
    moisture = pool.trajectory.outputs[rows][..., names.index(TARGET)]
    return {
        "variant": label,
        "seed": seed,
        "training_data": variant["data"],
        "measured_input_names": list(inputs),
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest,
        "fit_s": fit_s,
        "randomisation_s": randomisation_s,
        "total_s": time.perf_counter() - started,
        "n_train": n_train,
        "n_val": 0 if validation is None else len(own.split.val),
        "model_seed": own.seeds.model_seed,
        "process_fingerprint": own.process_config.fingerprint,
        "fingerprints": own.fingerprints,
        "exhaust_noise_seed": exhaust_seed,
        "data_spec": data_spec,
        "exogenous_names": list(train.exogenous_names),
        "training": model.training_diagnostics(),
        "model": model.describe(),
        "test_excited_nominal": _windows_error(model, test_set(nominal_pool)),
        "test_excited_randomised": _windows_error(model, test_set(randomised_pool)),
        "test_constant_holdout_nominal": _windows_error(model, test_set(holdout_pool)),
        "test_excited_own": _windows_error(model, test),
        "randomised_draws": {
            "rejected_sequence_positions": rejected,
            "disjoint_from_test_draws": disjoint,
            "unit_points": [[round(v, 10) for v in d.unit.values()] for d in draws],
        },
        "train_ranges": {
            "measured_min": measured.min(axis=(0, 1)).tolist(),
            "measured_max": measured.max(axis=(0, 1)).tolist(),
            "moisture_min": float(moisture.min()),
            "moisture_max": float(moisture.max()),
            "moisture_share_above_0.12": float(np.mean(moisture > 0.12)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny spec, no artifact")
    parser.add_argument("--seeds", default=None, help="comma list (budget rule: fewer seeds)")
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    import warnings as _w

    from _common import RUNS_DIR, provenance, require_clean_tree

    from twinlink_ml.sensor_value.config import load_follow_up_config
    from twinlink_ml.simulation import PlaceholderConfigWarning
    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    with _w.catch_warnings():
        _w.simplefilter("ignore", PlaceholderConfigWarning)
        follow_up = load_follow_up_config()
    soft = follow_up.soft
    if soft.architecture != "gru":
        raise SystemExit("only the GRU is set up")
    output = RUNS_DIR / f"{args.name}.json"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten")
    if not args.smoke:
        require_clean_tree()
    prov = provenance("not used")
    seeds = list(soft.training_seeds)
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        if not set(seeds) <= set(soft.training_seeds):
            raise SystemExit("seeds must be a subset of the configured training seeds")
    jobs = [
        ({"label": v.label, "data": v.data, "inputs": list(v.inputs)}, seed)
        for seed in seeds
        for v in soft.variants
    ]
    started = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [
            pool.submit(
                train_one,
                variant,
                seed,
                args.name if not args.smoke else f"_smoke_{args.name}",
                soft.n_train if not args.smoke else 4,
                soft.max_epochs if not args.smoke else 2,
                args.smoke,
            )
            for variant, seed in jobs
        ]
        records = [f.result() for f in futures]
    payload = {
        "provenance": prov,
        "study": "sensor value, follow-up (docs/29_sensor_value_2.md), soft sensors F5",
        "follow_up_fingerprint": follow_up.fingerprint,
        "sensor_value_fingerprint": follow_up.base.fingerprint,
        "smoke": args.smoke,
        "seeds": seeds,
        "records": records,
        "runtime_s": time.time() - started,
    }
    if args.smoke:
        print(json.dumps(payload, indent=1, default=str)[:3000])
        for r in records:
            Path(ROOT / r["checkpoint"]).unlink()
        return 0
    output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
