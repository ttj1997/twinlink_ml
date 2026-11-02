r"""GA2: Is the moisture soft sensor as good at the moisture limit as on average.

Motivation: Befund 64 computes with ONE global error (`rmse_run_median_physical`,
`product_moisture_kg_per_kg`, LSTM, `n_train=64`, `test_set=constant_holdout`, `h=36`,
median over 5 seeds: 0.000543 kg/kg excited-trained / 0.001488 kg/kg constant-trained,
Befund 62 HA1). The optimum of the control studies lies at 17 of 25 grid points (Befund 62)
exactly on the moisture limit (0.20 kg/kg) -- a global error says nothing about whether the
soft sensor is just as good EXACTLY THERE as in the rest of the value range.

No checkpoint of the excited-trained LSTM from Befund 51/58 is committed
(`.gitignore:38` excludes `artifacts/checkpoints/`; the `checkpoint/path` values logged in
`artifacts/runs/arch_learning_curve_projected.jsonl` point to a foreign machine and
do not exist in this checkout) -- therefore this script trains 1-3 seeds ANEW, with the
same protocol as Befund 51/58
(`scripts/run_arch_learning_curve_worker.py`, whose `_excited_spec`/`_holdout_spec`/
`HOLDOUT_SEED`/`CONFIG_PATH`/`TARGET_NAMES` are imported unchanged -- no line
changed there), but ONLY for the LSTM architecture at `n_train=64` (not the full
budget/architecture matrix), and additionally captures the raw errors per window/step that
`RolloutEvaluator.evaluate` computes internally but does not persist (only aggregated
metrics are logged, see `src/twinlink_ml/evaluation/evaluator.py::RolloutEvaluator.
evaluate`).

What is measured
----------------
a) Distribution of the TRUE product moisture (`future_targets_truth`, noise-free, pooled over
   all windows and steps 1..36) in the training split (`n_train=64`), in the excited
   test split and in the constant-run holdout -- the share in fixed moisture classes each,
   among them 0.18-0.22 kg/kg (the class around the moisture limit).
b) Error of the LSTM (against the NOISY reference `future_targets_truth` -> no:
   `future_targets`, the same reference as `rmse_run_median_physical` in the existing
   pipeline -- see `RolloutEvaluator.evaluate`, `reference_physical`) by classes of the
   TRUE moisture (`truth_physical`), on the constant-run holdout, pooled over all
   windows and steps 1..36 per class.
c) Error in the class around 0.20 kg/kg (0.18-0.22) against the global (class-unweighted)
   error on the same test set.

Identical protocol to Befund 51/58, ONE permitted deviation: only the LSTM architecture
at `n_train=64` is computed (not baselines, not the other budget points/architectures)
-- Befund 51/58 already computed this cell under the same protocol (RMSE values cited in
Befund 62 HA1); this script reproduces it and EXTENDS it by raw-error arrays,
without changing the configuration fingerprint (`bddca37f716f5cfc`, checked against
`generate_data`, exactly as in the worker).

Runtime estimate: Befund 51's complete sweep (4 budget points x 8 architectures +
3 baselines, 5 seeds in parallel) took 3.96 h (14250.8 s) for 5 jobs. This script
computes only ONE architecture at ONE (the largest) budget point per seed -- the
data generation (pool simulation, fixed per seed) is the dominant, unchanged cost share;
the actual training is a fraction of the sweep's training share. Seed 1 is computed first
as a timed pilot (like Befund 51's own procedure: estimated upfront, then
measured at the first point), before seeds 2/3 follow.

Usage::

    python scripts/probe_soft_sensor_error_by_moisture_class.py --seeds 1 \
        --output artifacts/runs/probe_soft_sensor_error_by_moisture_class_seed1.json

    python scripts/probe_soft_sensor_error_by_moisture_class.py --seeds 1 2 3 \
        --output artifacts/runs/probe_soft_sensor_error_by_moisture_class.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import run_arch_learning_curve_worker as worker

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset, WindowBatch
from twinlink_ml.evaluation import ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import per_run_rmse_median
from twinlink_ml.models.composite import CompositeLstmSequenceModel, CompositeSequenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.checkpoints import checkpoint_path, save_checkpoint
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_soft_sensor_error_by_moisture_class.json"
N_TRAIN = 64
MOISTURE_INDEX = worker.TARGET_NAMES.index("product_moisture_kg_per_kg")
EXPECTED_FINGERPRINT = "bddca37f716f5cfc"
"""Wie `run_arch_learning_curve_worker.py` -- dieselbe gepruefte BA0(c)-Grundlage."""

# Fixed moisture classes [kg/kg), last interval open. 0.18/0.22 bracket the
# moisture limit (0.20 kg/kg) symmetrically -- the class in which, according to Befund 62,
# the optimum sits at 17 of 25 grid points.
CLASS_EDGES: tuple[float, ...] = (0.0, 0.10, 0.14, 0.18, 0.22, 0.26, np.inf)
BOUNDARY_CLASS_LOW = 0.18
BOUNDARY_CLASS_HIGH = 0.22


def _class_labels(edges: tuple[float, ...]) -> list[str]:
    labels = []
    for lo, hi in itertools.pairwise(edges):
        hi_label = "inf" if np.isinf(hi) else f"{hi:.2f}"
        labels.append(f"[{lo:.2f},{hi_label})")
    return labels


CLASS_LABELS = _class_labels(CLASS_EDGES)


def _collect_predictions(
    model: CompositeSequenceModel, dataset: TrajectoryWindowDataset, *, batch_size: int = 4096
) -> dict[str, Any]:
    """Like `RolloutEvaluator.evaluate`, but the physical raw fields are
    returned instead of only aggregated metrics -- `RolloutEvaluator.evaluate` itself
    stays unchanged (`src/twinlink_ml/evaluation/evaluator.py` not touched), this
    function only duplicates the array construction (collect batches, concatenate,
    `scaler.inverse`), identical to its body."""
    scaler = dataset.target_scaler
    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    runs: list[np.ndarray] = []
    batches: list[WindowBatch] = list(dataset.iter_batches(batch_size))
    if not batches:
        raise RuntimeError("Datensatz liefert keine Fenster.")
    for batch in batches:
        if batch.future_targets_truth is None:
            raise RuntimeError(
                "Datensatz enthaelt keine rauschfreie Referenz (future_targets_truth)."
            )
        predictions.append(model.predict(batch))
        references.append(batch.future_targets)
        truths.append(batch.future_targets_truth)
        runs.append(batch.run_index)
    prediction_raw = np.concatenate(predictions, axis=0)
    reference_raw = np.concatenate(references, axis=0)
    truth_raw = np.concatenate(truths, axis=0)
    run_index = np.concatenate(runs, axis=0)
    return {
        "prediction_physical": scaler.inverse(prediction_raw),
        "reference_physical": scaler.inverse(reference_raw),
        "truth_physical": scaler.inverse(truth_raw),
        "run_index": run_index,
    }


def _moisture_channel(
    arrays: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts `product_moisture_kg_per_kg` (channel `MOISTURE_INDEX`) from prediction,
    reference and truth, each flattened to 1D over (windows x steps 1..36), plus
    `run_index` expanded to the same length (one run index was stored per WINDOW,
    here repeated for each of the `horizon` steps of that window) -- needed for the
    run-level median aggregation (see `_rmse_per_run_median`)."""
    pred = arrays["prediction_physical"][..., MOISTURE_INDEX].reshape(-1)
    ref = arrays["reference_physical"][..., MOISTURE_INDEX].reshape(-1)
    truth = arrays["truth_physical"][..., MOISTURE_INDEX].reshape(-1)
    horizon = arrays["prediction_physical"].shape[1]
    run_index = np.repeat(arrays["run_index"], horizon)
    return pred, ref, truth, run_index


def _rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))


def _rmse_per_run_median(
    prediction: np.ndarray, reference: np.ndarray, run_index: np.ndarray
) -> float | None:
    """Like `twinlink_ml.evaluation.evaluator.per_run_rmse_median` (the definition behind
    Befund 62 HA1's 0.000543 kg/kg number: RMSE per run, then median over runs) -- NOT
    the pooled/flattened RMSE over all windows/steps (see the module docstring of
    `evaluator.py`, section "Run-level normalisation is the primary number"). This function
    is the primary metric for GA2c, because it stays comparable with Befund 62 HA1's
    definition -- the pooled value (`_rmse`) is only side information here."""
    if prediction.size == 0:
        return None
    per_run = [
        _rmse(prediction[run_index == run], reference[run_index == run])
        for run in np.unique(run_index)
        if np.count_nonzero(run_index == run) > 0
    ]
    return float(np.median(per_run)) if per_run else None


def _moisture_distribution(arrays: dict[str, Any]) -> dict[str, Any]:
    truth = arrays["truth_physical"][..., MOISTURE_INDEX].reshape(-1)
    counts, _ = np.histogram(truth, bins=CLASS_EDGES)
    total = int(truth.size)
    boundary_mask = (truth >= BOUNDARY_CLASS_LOW) & (truth < BOUNDARY_CLASS_HIGH)
    return {
        "n_points": total,
        "min_kg_per_kg": float(np.min(truth)),
        "max_kg_per_kg": float(np.max(truth)),
        "mean_kg_per_kg": float(np.mean(truth)),
        "class_edges_kg_per_kg": [*list(CLASS_EDGES[:-1]), None],
        "class_labels": CLASS_LABELS,
        "class_counts": [int(c) for c in counts],
        "class_fractions": [float(c) / total for c in counts],
        "boundary_class_fraction": float(np.count_nonzero(boundary_mask)) / total,
    }


def _error_by_class(
    prediction: np.ndarray, reference: np.ndarray, truth: np.ndarray, run_index: np.ndarray
) -> dict[str, Any]:
    global_rmse = _rmse(prediction, reference)
    global_rmse_per_run_median = _rmse_per_run_median(prediction, reference, run_index)
    boundary_mask = (truth >= BOUNDARY_CLASS_LOW) & (truth < BOUNDARY_CLASS_HIGH)
    n_boundary = int(np.count_nonzero(boundary_mask))
    boundary_rmse = (
        _rmse(prediction[boundary_mask], reference[boundary_mask]) if n_boundary > 0 else None
    )
    boundary_rmse_per_run_median = (
        _rmse_per_run_median(
            prediction[boundary_mask], reference[boundary_mask], run_index[boundary_mask]
        )
        if n_boundary > 0
        else None
    )
    boundary_n_runs = int(np.unique(run_index[boundary_mask]).size) if n_boundary > 0 else 0

    per_class: list[dict[str, Any]] = []
    for label, lo, hi in zip(CLASS_LABELS, CLASS_EDGES[:-1], CLASS_EDGES[1:], strict=True):
        mask = (truth >= lo) & (truth < hi)
        n = int(np.count_nonzero(mask))
        per_class.append(
            {
                "class": label,
                "n_points": n,
                "rmse_pooled_kg_per_kg": _rmse(prediction[mask], reference[mask])
                if n > 0
                else None,
                "rmse_per_run_median_kg_per_kg": (
                    _rmse_per_run_median(prediction[mask], reference[mask], run_index[mask])
                    if n > 0
                    else None
                ),
            }
        )

    return {
        "n_points_total": int(truth.size),
        "global_rmse_pooled_kg_per_kg": global_rmse,
        "global_rmse_per_run_median_kg_per_kg": global_rmse_per_run_median,
        "boundary_class_n_points": n_boundary,
        "boundary_class_n_runs": boundary_n_runs,
        "boundary_class_rmse_pooled_kg_per_kg": boundary_rmse,
        "boundary_class_rmse_per_run_median_kg_per_kg": boundary_rmse_per_run_median,
        "boundary_vs_global_ratio_pooled": (
            boundary_rmse / global_rmse if boundary_rmse is not None and global_rmse > 0 else None
        ),
        "boundary_vs_global_ratio_per_run_median": (
            boundary_rmse_per_run_median / global_rmse_per_run_median
            if boundary_rmse_per_run_median is not None
            and global_rmse_per_run_median is not None
            and global_rmse_per_run_median > 0
            else None
        ),
        "note": (
            "per_run_median ist die Primaerzahl (vergleichbar mit Befund 62 HA1s "
            "0,000543 kg/kg), pooled ist Nebeninformation -- siehe evaluator.py-Moduldocstring "
            "'Laufweise Normierung ist die Primärzahl'."
        ),
        "per_class": per_class,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=worker.CONFIG_PATH)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Trainings-Seeds (Befund-51/58-Nummerierung)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=1, help="torch.set_num_threads (wie Befund 51/58)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=REPO_ROOT / "artifacts" / "checkpoints"
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    import torch

    torch.set_num_threads(arguments.num_threads)

    env = environment_fingerprint()
    if env["git/dirty"]:
        print("WARNUNG: Arbeitsbaum ist dirty -- git/dirty wird im Artefakt vermerkt.", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )

    excited_spec = worker._excited_spec()
    holdout_spec = worker._holdout_spec(excited_spec)

    print("=" * 88)
    print("GA2: Soft-Sensor-Fehler nach Feuchteklasse (LSTM, n_train=64, h=36, angeregt trainiert)")
    print(f"Config   {arguments.config}")
    print(f"Seeds    {arguments.seeds}")
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    print("=" * 88, flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        holdout = generate_data(config, holdout_spec, master_seed=worker.HOLDOUT_SEED)

    per_seed: dict[str, Any] = {}
    total_start = time.perf_counter()
    for seed in arguments.seeds:
        seed_start = time.perf_counter()
        print(f"\n--- Seed {seed}: Datenerzeugung ---", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            excited = generate_data(config, excited_spec, master_seed=seed)
        if not fingerprints_equivalent(excited.process_config.fingerprint, EXPECTED_FINGERPRINT):
            raise RuntimeError(
                f"Seed {seed}: process/fingerprint={excited.process_config.fingerprint!r}, "
                f"erwartet {EXPECTED_FINGERPRINT!r} -- Config weicht von Befund 51/58 ab."
            )
        data_elapsed = time.perf_counter() - seed_start

        train, validation, test_excited = build_datasets(
            excited, N_TRAIN, policy=ValidationPolicy.FIXED_POOL
        )
        test_constant = TrajectoryWindowDataset.from_trajectory(
            holdout.trajectory,
            holdout.split,
            SplitName.TEST,
            excited_spec.window,
            scalers=train.scalers,
        )

        print(f"--- Seed {seed}: Training (LSTM, n_train={N_TRAIN}) ---", flush=True)
        model = CompositeLstmSequenceModel()
        train_start = time.perf_counter()
        model.fit(train, seed=excited.seeds.model_seed, validation=validation)
        train_elapsed = time.perf_counter() - train_start

        run_uid = f"lstm_ga2_seed{seed}_n{N_TRAIN}"
        ckpt_path = checkpoint_path(arguments.checkpoint_dir, run_uid)
        save_checkpoint(
            ckpt_path,
            model,
            model_label="lstm",
            run_uid=run_uid,
            target_names=train.target_names,
            exogenous_names=train.exogenous_names,
            n_exogenous=len(train.exogenous_names),
            input_scaler=train.input_scaler,
            target_scaler=train.target_scaler,
            process_fingerprint=excited.process_config.fingerprint,
            data_spec_params=excited_spec.as_parameters(),
            seed_bundle_params=excited.seeds.as_parameters(),
            git_commit=env["git/commit"],
        )

        print(f"--- Seed {seed}: Auswertung ---", flush=True)
        train_arrays = _collect_predictions(model, train)
        test_excited_arrays = _collect_predictions(model, test_excited)
        test_constant_arrays = _collect_predictions(model, test_constant)

        pred_c, ref_c, truth_c, run_index_c = _moisture_channel(test_constant_arrays)
        error_by_class = _error_by_class(pred_c, ref_c, truth_c, run_index_c)

        # Cross-check against Befund 62 HA1: rmse_run_median_physical, product_moisture, h=36,
        # test_set=constant_holdout -- the same definition (median over runs of the
        # unnormalised RMSE), here directly from the captured arrays instead of via
        # RolloutEvaluator (which calls the same function internally).
        rmse_median_by_horizon = per_run_rmse_median(
            test_constant_arrays["prediction_physical"],
            test_constant_arrays["reference_physical"],
            test_constant_arrays["run_index"],
            horizons=(worker.HORIZON,),
            names=worker.TARGET_NAMES,
        )
        cross_check_rmse = rmse_median_by_horizon["product_moisture_kg_per_kg"][worker.HORIZON]

        per_seed[str(seed)] = {
            "seed": seed,
            "process_fingerprint": excited.process_config.fingerprint,
            "split_digest": excited.split.digest,
            "data_generation_elapsed_s": data_elapsed,
            "training_elapsed_s": train_elapsed,
            "checkpoint_path": str(ckpt_path),
            "moisture_distribution": {
                "train": _moisture_distribution(train_arrays),
                "test_excited": _moisture_distribution(test_excited_arrays),
                "test_constant_holdout": _moisture_distribution(test_constant_arrays),
            },
            "error_by_class_test_constant_holdout": error_by_class,
            "cross_check_rmse_run_median_physical_kg_per_kg_h36_constant_holdout": cross_check_rmse,
            "cross_check_reference_befund62_ha1_excited_median_5seeds_kg_per_kg": 0.000542738535806486,
        }

        seed_elapsed = time.perf_counter() - seed_start
        print(
            f"Seed {seed}: fertig ({seed_elapsed / 60.0:.1f} min) -- "
            f"global RMSE (Median/Lauf)={error_by_class['global_rmse_per_run_median_kg_per_kg']:.6f} kg/kg, "
            f"Grenzklasse [{BOUNDARY_CLASS_LOW},{BOUNDARY_CLASS_HIGH}) RMSE (Median/Lauf)="
            f"{error_by_class['boundary_class_rmse_per_run_median_kg_per_kg']}, "
            f"Verhaeltnis={error_by_class['boundary_vs_global_ratio_per_run_median']}, "
            f"Cross-Check (h=36, Median/Lauf)={cross_check_rmse:.6f} kg/kg "
            f"(Befund 62 HA1: 0.000543 kg/kg)",
            flush=True,
        )

    total_elapsed = time.perf_counter() - total_start

    # Aggregation over seeds (median), only if more than one seed was computed. The primary
    # number is per_run_median (comparable with Befund 62 HA1), pooled only side information.
    aggregate: dict[str, Any] | None = None
    if len(arguments.seeds) > 1:
        global_rmses = [
            per_seed[str(s)]["error_by_class_test_constant_holdout"][
                "global_rmse_per_run_median_kg_per_kg"
            ]
            for s in arguments.seeds
        ]
        boundary_rmses = [
            per_seed[str(s)]["error_by_class_test_constant_holdout"][
                "boundary_class_rmse_per_run_median_kg_per_kg"
            ]
            for s in arguments.seeds
        ]
        boundary_rmses_finite = [v for v in boundary_rmses if v is not None]
        aggregate = {
            "n_seeds": len(arguments.seeds),
            "median_global_rmse_per_run_median_kg_per_kg": float(np.median(global_rmses)),
            "median_boundary_class_rmse_per_run_median_kg_per_kg": (
                float(np.median(boundary_rmses_finite)) if boundary_rmses_finite else None
            ),
        }
        if (
            aggregate["median_boundary_class_rmse_per_run_median_kg_per_kg"] is not None
            and aggregate["median_global_rmse_per_run_median_kg_per_kg"] > 0
        ):
            aggregate["median_boundary_vs_global_ratio"] = (
                aggregate["median_boundary_class_rmse_per_run_median_kg_per_kg"]
                / aggregate["median_global_rmse_per_run_median_kg_per_kg"]
            )

    output: dict[str, Any] = {
        "environment": env,
        "config_path": str(arguments.config),
        "n_train": N_TRAIN,
        "horizon": worker.HORIZON,
        "target_names": list(worker.TARGET_NAMES),
        "moisture_index": MOISTURE_INDEX,
        "class_edges_kg_per_kg": [*list(CLASS_EDGES[:-1]), None],
        "boundary_class_kg_per_kg": [BOUNDARY_CLASS_LOW, BOUNDARY_CLASS_HIGH],
        "holdout_seed": worker.HOLDOUT_SEED,
        "per_seed": per_seed,
        "aggregate_over_seeds": aggregate,
        "total_elapsed_s": total_elapsed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nGesamtlaufzeit: {total_elapsed / 60.0:.1f} min")
    print(f"Ergebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
