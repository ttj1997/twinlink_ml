r"""F3+F4: pure inference (no training) on artifacts/runs/arch_gab_constant_holdout.jsonl.

NO network training -- GRU/LSTM are loaded exclusively from existing checkpoints
(artifacts/checkpoints/{gru,lstm}_seed{1..10}.pt) and evaluated in the forward pass.
Persistence/ARX(p=4,q=2) are closed-form baselines (no gradient method). NO
change to src/.

Primary metric: nrmse_run_median (normalised run-level), NOT the pooled nrmse -- see
evaluator.py:166-168 ("the primary number").
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models import LinearARXModel, PersistenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.simulation.config import MaterialConfig
from twinlink_ml.utils.checkpoints import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
LOG_PATH = REPO_ROOT / "artifacts" / "runs" / "arch_gab_constant_holdout.jsonl"
HOLDOUT_SEED = 999
BASELINE_MASTER_SEED = 1
SEEDS = tuple(range(1, 11))
ARCHITECTURES = ("gru", "lstm")

# Two separate constants instead of a shared one: this script checks two
# DIFFERENT provenances against each other, not the same value twice.
# EXPECTED_FP_FRESH is the fingerprint of the holdout data freshly generated HERE (current
# code state). EXPECTED_FP_CHECKPOINT is the fingerprint that was written into the
# artifacts/checkpoints/{gru,lstm}_seed{1..10}.pt files at training time (permanent, never
# recomputed on loading, see utils/checkpoints.py::load_checkpoint) -- these
# checkpoints were trained BEFORE NumericsConfig.enforce_gas_velocity_projection was
# introduced and therefore stay at the old value forever.
#
# The identification "40aae2b4526a5ed7 corresponds to 82f65f576a2e5253" is covered ONLY by the
# bit-identity regression test test_gas_velocity_projection_disabled_matches_legacy_
# baseline (tests/test_gas_velocity_projection.py): it shows that both
# fingerprints yield the same trajectories with the projection switched off. Without this
# evidence the evaluation here would be invalid -- see the runtime check below
# (enforce_gas_velocity_projection must be OFF, otherwise a checkpoint trained on unprojected
# data is evaluated on a different distribution without this showing up in a
# fingerprint comparison -- EXPECTED_FP_FRESH would then simply deviate "as expected"
# from the (updated) reference value).
EXPECTED_FP_FRESH = "40aae2b4526a5ed7"
EXPECTED_FP_CHECKPOINT = "82f65f576a2e5253"


def _excited_spec() -> DataSpec:
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=36,
        stride=8,
    )


def _holdout_spec(excited: DataSpec) -> DataSpec:
    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def _violation_masks(trajectory: Trajectory, material: MaterialConfig) -> dict[str, np.ndarray]:
    diagnostics = trajectory.diagnostics
    output_names = trajectory.metadata["output_names"]
    bed_temp_idx = output_names.index("bed_temperature_celsius")
    return {
        "u_above_u_terminal": diagnostics["fluidization_margin_upper_m_per_s"] < 0.0,
        "u_below_u_mf": diagnostics["fluidization_margin_lower_m_per_s"] < 0.0,
        "T_bed_above_80C": trajectory.outputs[:, :, bed_temp_idx] > 80.0,
        "moisture_above_critical": diagnostics["particle_moisture_kg_per_kg"]
        > material.critical_moisture_kg_per_kg,
    }


def _load_logged_records() -> list[dict]:
    with LOG_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _logged_metric(
    records: list[dict], *, label: str, seed: int, test_set: str
) -> dict[str, float] | None:
    for rec in records:
        p = rec["params"]
        if (
            p.get("model/label") == label
            and p.get("master_seed") == seed
            and p.get("test_set") == test_set
        ):
            return rec["metrics"]
    return None


def main() -> None:
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    # The gru/lstm checkpoints were trained on UNPROJECTED data. If this script
    # evaluates them on projected data, that is a distribution break that no
    # fingerprint comparison shows any more (both constants would then simply deviate
    # from each other "as expected") -- hence a separate, explicit check.
    if config.numerics.enforce_gas_velocity_projection:
        raise RuntimeError(
            "enforce_gas_velocity_projection ist AN, aber die gru/lstm-Checkpoints in "
            f"{CHECKPOINT_DIR} wurden auf unprojizierten Daten trainiert -- eine Auswertung "
            "auf projizierten Holdout-Daten waere ein Verteilungsbruch zwischen Training "
            "und Auswertung, den kein Fingerprint-Vergleich mehr anzeigt. Schalter in "
            f"{CONFIG_PATH} ausschalten, oder Checkpoints mit aktivierter Projektion neu "
            "trainieren."
        )

    excited_spec = _excited_spec()
    holdout_spec = _holdout_spec(excited_spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        holdout = generate_data(config, holdout_spec, master_seed=HOLDOUT_SEED)

    actual_fresh_fp = holdout.process_config.fingerprint
    assert actual_fresh_fp == EXPECTED_FP_FRESH, actual_fresh_fp
    assert len(holdout.split.test) == 32, len(holdout.split.test)
    n_test = len(holdout.split.test)
    print(f"Holdout rekonstruiert: fp={holdout.process_config.fingerprint}, n_test={n_test}")

    masks = _violation_masks(holdout.trajectory, config.material)
    run_id = holdout.trajectory.run_id
    n_runs = holdout.trajectory.n_runs
    any_violation = dict.fromkeys(run_id, False)
    per_criterion_affected = {}
    for crit, mask in masks.items():
        hits = {run_id[i]: bool(mask[i].any()) for i in range(n_runs)}
        per_criterion_affected[crit] = sum(hits.values())
        for rid, flag in hits.items():
            if flag:
                any_violation[rid] = True
    n_violating = sum(any_violation.values())
    masked_test_ids = tuple(r for r in holdout.split.test if not any_violation[r])
    print(f"Holdout-Verletzungen je Kriterium (von {n_runs}): {per_criterion_affected}")
    print(
        f"Holdout gesamt: {n_runs}, verletzend (irgendein Kriterium): {n_violating}, "
        f"verbleibend: {len(masked_test_ids)}"
    )

    window = excited_spec.window
    evaluator = RolloutEvaluator(horizons=(1, excited_spec.horizon))
    logged = _load_logged_records()

    result: dict[str, Any] = {
        "holdout_n_runs": n_runs,
        "holdout_violating_any": n_violating,
        "holdout_surviving": len(masked_test_ids),
        "per_criterion_affected": per_criterion_affected,
    }

    targets = excited_spec.target_names
    horizons = (1, excited_spec.horizon)

    def _extract(metrics: dict[str, float]) -> dict:
        out = {}
        for target in targets:
            for h in horizons:
                out[f"nrmse_run_median/{target}/h{h}"] = metrics.get(
                    f"nrmse_run_median/{target}/h{h}"
                )
                out[f"nrmse/{target}/h{h}"] = metrics.get(f"nrmse/{target}/h{h}")
        out["n_test_runs"] = metrics.get("n_test_runs")
        return out

    # ---------------------------------------------------------------- F3: gru/lstm
    f3: dict = {}
    for arch in ARCHITECTURES:
        f3[arch] = {}
        for seed in SEEDS:
            ckpt_path = CHECKPOINT_DIR / f"{arch}_seed{seed}.pt"
            model, meta = load_checkpoint(ckpt_path)
            actual_fp = meta["process_fingerprint"]
            assert actual_fp == EXPECTED_FP_CHECKPOINT, (arch, seed, actual_fp)

            scalers = (meta["input_scaler"], meta["target_scaler"])
            test_unmasked = TrajectoryWindowDataset.from_trajectory(
                holdout.trajectory, holdout.split, SplitName.TEST, window, scalers=scalers
            )
            masked_split = dataclasses.replace(holdout.split, test=masked_test_ids)
            test_masked = TrajectoryWindowDataset.from_trajectory(
                holdout.trajectory, masked_split, SplitName.TEST, window, scalers=scalers
            )

            unmasked_metrics = _extract(evaluator.evaluate(model, test_unmasked).as_flat_metrics())
            masked_metrics = _extract(evaluator.evaluate(model, test_masked).as_flat_metrics())

            logged_metrics = _logged_metric(
                logged, label=arch, seed=seed, test_set="constant_holdout"
            )
            reproduction = {}
            if logged_metrics is not None:
                for target in targets:
                    for h in horizons:
                        key = f"nrmse_run_median/{target}/h{h}"
                        reproduction[key] = {
                            "logged": logged_metrics.get(key),
                            "recomputed_unmasked": unmasked_metrics.get(key),
                        }

            f3[arch][str(seed)] = {
                "unmasked": unmasked_metrics,
                "masked": masked_metrics,
                "reproduction_check": reproduction,
            }
        print(f"-- {arch}: {len(SEEDS)} Seeds ausgewertet --")
    result["f3"] = f3

    # ---------------------------------------------------------------- F4: persistence/arx
    print(
        f"-- Baselines (persistence, arx_p4_q2), Trainingspool "
        f"master_seed={BASELINE_MASTER_SEED} --"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        excited = generate_data(config, excited_spec, master_seed=BASELINE_MASTER_SEED)
    train, validation, _test_excited = build_datasets(
        excited, excited_spec.n_train_max, policy=ValidationPolicy.FIXED_POOL
    )

    test_unmasked = TrajectoryWindowDataset.from_trajectory(
        holdout.trajectory, holdout.split, SplitName.TEST, window, scalers=train.scalers
    )
    masked_split = dataclasses.replace(holdout.split, test=masked_test_ids)
    test_masked = TrajectoryWindowDataset.from_trajectory(
        holdout.trajectory, masked_split, SplitName.TEST, window, scalers=train.scalers
    )

    f4: dict = {}
    for label, factory in (("persistence", PersistenceModel), ("arx_p4_q2", LinearARXModel)):
        model = factory()
        model.fit(train, seed=excited.seeds.model_seed, validation=validation)
        unmasked_metrics = _extract(evaluator.evaluate(model, test_unmasked).as_flat_metrics())
        masked_metrics = _extract(evaluator.evaluate(model, test_masked).as_flat_metrics())
        logged_metrics = _logged_metric(
            logged, label=label, seed=BASELINE_MASTER_SEED, test_set="constant_holdout"
        )
        reproduction = {}
        if logged_metrics is not None:
            for target in targets:
                for h in horizons:
                    key = f"nrmse_run_median/{target}/h{h}"
                    reproduction[key] = {
                        "logged": logged_metrics.get(key),
                        "recomputed_unmasked": unmasked_metrics.get(key),
                    }
        f4[label] = {
            "unmasked": unmasked_metrics,
            "masked": masked_metrics,
            "reproduction_check": reproduction,
        }
    result["f4"] = f4

    result["_elapsed_s"] = time.monotonic() - t0
    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_f3_f4_arch_holdout_inference.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\ngeschrieben: {out_path} ({result['_elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
