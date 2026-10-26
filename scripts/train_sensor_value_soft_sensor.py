r"""Train the GRU soft sensors (feedback variant F3) of the sensor-value study (docs/28).

Existing pipeline of Befund 94, arm N (``run_arch_rerun_worker.arm_spec("N")``:
``DataSpec.new_study()``, preset version 2, 25 s sample time, history 32, horizon 36; process
config ``conf/process/greybox_gab_placeholder.yaml`` with the gas-velocity projection, process
fingerprint ``bddca37f716f5cfc``; own excited pool per master seed; run-level splits; validation
pool; ``max_epochs`` 200), operating mode ``SOFT_SENSOR`` (``spec_for_mode``), moisture head alone
(``moisture_view`` of the outlier study: the composite network's moisture network with its seed
derived from (model seed, target name), i.e. exactly the moisture head of a three-target soft-sensor
network). Trained ONLY on data of the nominal model: the process config of Befund 94 has the same
material parameters as the nominal model of the controller (checked below). ``n_train`` 64,
master seeds from ``conf/sensor_value/sensor_value.yaml`` (1, 2, 3).

Writes one checkpoint per seed to ``artifacts/checkpoints/sensor_value/<name>_seed<k>.pt``
(gitignored, as all checkpoints) and the record ``artifacts/runs/<name>.json`` with the SHA-256 of
every checkpoint (the main run checks it), the training diagnostics, test errors of the soft sensor
in kg/kg against the noise-free moisture (h = 1 and h = 36; test sets excited and constant-run
holdout, seed 999) and the input ranges of the training data next to the operating point of the
study line (domain check). Run with ``tools/run_here.sh scripts/train_sensor_value_soft_sensor.py
--name sensor_value_soft_sensor_v1`` from a clean tree.
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


def train_one(seed: int, name: str, n_train: int, max_epochs: int, smoke: bool) -> dict[str, Any]:
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
    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
    from twinlink_ml.utils.checkpoints import save_checkpoint

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(old_worker.CONFIG_PATH)
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )
    spec = rerun_worker.arm_spec("N", smoke=smoke)
    started = time.perf_counter()
    own = outlier_worker._generate(config, spec, seed)
    holdout = outlier_worker._generate(
        config, rerun_worker.holdout_spec(spec), rerun_worker.HOLDOUT_SEED
    )
    if not smoke and own.process_config.fingerprint != PROCESS_FINGERPRINT:
        raise RuntimeError(f"process/fingerprint {own.process_config.fingerprint!r}")

    def soft_view(data: Any) -> Any:
        moisture = outlier_worker.moisture_view(data, TARGET)
        return dataclasses.replace(
            moisture, spec=spec_for_mode(moisture.spec, ObservationMode.SOFT_SENSOR)
        )

    own_soft, holdout_soft = soft_view(own), soft_view(holdout)
    if own_soft.spec.window.observed_targets:
        raise RuntimeError("the moisture channel must be unobserved")
    train, validation, test = build_datasets(own_soft, n_train, policy=ValidationPolicy.FIXED_POOL)
    model = old_worker._build_model(old_worker._architecture_factory("gru"), max_epochs)
    fit_start = time.perf_counter()
    model.fit(train, seed=own.seeds.model_seed, validation=validation)
    fit_s = time.perf_counter() - fit_start
    holdout_test = TrajectoryWindowDataset.from_trajectory(
        holdout_soft.trajectory,
        holdout_soft.split,
        SplitName.TEST,
        holdout_soft.spec.window,
        scalers=train.scalers,
    )
    directory = ROOT / "artifacts" / "checkpoints" / "sensor_value"
    path = directory / f"{name}_seed{seed}.pt"
    if path.exists():
        raise SystemExit(f"{path} exists -- checkpoints are never overwritten; use a new name")
    data_spec = {**own_soft.spec.as_parameters(), "data/observed_target_names": []}
    save_checkpoint(
        path,
        model,
        model_label="gru",
        run_uid=f"{name}_seed{seed}",
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
    rows = [own.trajectory.run_id.index(r) for r in split.train]
    actions = own.trajectory.actions[rows]
    disturbances = own.trajectory.disturbances[rows]
    moisture = own.trajectory.outputs[rows][
        ..., list(own.trajectory.metadata["output_names"]).index(TARGET)
    ]
    return {
        "seed": seed,
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest,
        "fit_s": fit_s,
        "total_s": time.perf_counter() - started,
        "n_train": n_train,
        "n_val": 0 if validation is None else len(own.split.val),
        "model_seed": own.seeds.model_seed,
        "process_fingerprint": own.process_config.fingerprint,
        "fingerprints": own.fingerprints,
        "data_spec": own_soft.spec.as_parameters(),
        "exogenous_names": list(train.exogenous_names),
        "training": model.training_diagnostics(),
        "model": model.describe(),
        "test_excited": _windows_error(model, test),
        "test_constant_holdout": _windows_error(model, holdout_test),
        "train_ranges": {
            "actions_min": actions.min(axis=(0, 1)).tolist(),
            "actions_max": actions.max(axis=(0, 1)).tolist(),
            "disturbances_min": disturbances.min(axis=(0, 1)).tolist(),
            "disturbances_max": disturbances.max(axis=(0, 1)).tolist(),
            "moisture_min": float(moisture.min()),
            "moisture_max": float(moisture.max()),
            "moisture_share_above_0.12": float(np.mean(moisture > 0.12)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny spec, no artifact")
    args = parser.parse_args()
    import warnings as _w

    from _common import RUNS_DIR, provenance, require_clean_tree

    from twinlink_ml.headroom.config import load_headroom_config, study_process_config
    from twinlink_ml.sensor_value import load_sensor_value_config
    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    cfg = load_sensor_value_config()
    fb = cfg.feedback
    if fb.soft_sensor_architecture != "gru":
        raise SystemExit("only the GRU is set up")
    output = RUNS_DIR / f"{args.name}.json"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten")
    if not args.smoke:
        require_clean_tree()
    prov = provenance("not used")
    # nominal model of the controller == material of the training data (only the study settings
    # of the headroom line differ: limit, nozzle fixed point, no action clipping, no measurement)
    with _w.catch_warnings():
        _w.simplefilter("ignore", PlaceholderConfigWarning)
        study = load_headroom_config(cfg.headroom_config)
        nominal = study_process_config(study)
        base = load_process_config(
            SCRIPTS.parent / "conf" / "process" / "greybox_gab_placeholder.yaml"
        )
    same_material = (
        dataclasses.replace(
            base.material,
            product_moisture_limit_kg_per_kg=nominal.material.product_moisture_limit_kg_per_kg,
        )
        == nominal.material
    )
    if not same_material:
        raise SystemExit("training process and nominal model differ in the material parameters")
    seeds = list(fb.soft_sensor_training_seeds)
    started = time.time()
    with ProcessPoolExecutor(max_workers=len(seeds)) as pool:
        futures = [
            pool.submit(
                train_one,
                seed,
                args.name if not args.smoke else f"_smoke_{args.name}",
                fb.soft_sensor_n_train if not args.smoke else 4,
                fb.soft_sensor_max_epochs if not args.smoke else 2,
                args.smoke,
            )
            for seed in seeds
        ]
        records = [f.result() for f in futures]
    payload = {
        "provenance": prov,
        "study": "sensor value (docs/28_sensor_value.md), soft sensor F3",
        "sensor_value_fingerprint": cfg.fingerprint,
        "same_material_as_nominal_model": same_material,
        "operating_point_of_study_line": {
            "nozzle_pressure_bar": study.line.nozzle_pressure_bar,
            "nominal_spray_rate_kg_per_h": study.line.nominal_spray_rate_kg_per_h,
            "nominal_solid_fraction": study.line.nominal_solid_fraction,
            "moisture_limit_kg_per_kg": study.line.product_moisture_limit_kg_per_kg,
        },
        "smoke": args.smoke,
        "seeds": records,
        "runtime_s": time.time() - started,
    }
    if args.smoke:
        print(json.dumps(payload, indent=1, default=str)[:4000])
        return 0
    output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
