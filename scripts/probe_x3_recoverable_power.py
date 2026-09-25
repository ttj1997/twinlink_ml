r"""X3: rückgewinnbare Leistung Q_rec aus den X2-Checkpoints (Befund 50).

KEIN Training -- GRU wird ausschließlich aus den Checkpoints geladen
(``artifacts/checkpoints/gru_seed{1,2,3}.pt``, erzeugt von
``run_outlet_temperature_gru_protocol.py``). Persistenz/statische Baseline sind
geschlossene Fits (kein Gradientenverfahren), hier frisch gegen dieselben, deterministisch
reproduzierten Seed-Daten angepasst -- identisches Muster zu
``probe_f3_f4_arch_holdout_inference.py``.

Ausgewertet wird ausschließlich gegen bereits abgeschlossene X2-Ergebnisse (alle drei Seeds
fertig, siehe ``artifacts/runs/outlet_temperature_gru_protocol.jsonl``) -- wie angewiesen
nicht gegen Zwischenergebnisse.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.recoverable_power import evaluate_recoverable_power
from twinlink_ml.models import PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.checkpoints import load_checkpoint
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_x3_recoverable_power.json"

SEEDS = (1, 2, 3)
HOLDOUT_SEED = 999
TARGET_NAMES = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
EXPECTED_FINGERPRINT = "6ecedc6bd993fb3d"
"""Muss mit dem in den Checkpoints gespeicherten ``process_fingerprint`` übereinstimmen --
siehe ``docs/04_stand_ap1.md``, Fingerprint-Historie (Projektion AUS, nach ExhaustConfig)."""


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
        target_names=TARGET_NAMES,
    )


def _holdout_spec(excited: DataSpec) -> DataSpec:
    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    excited_spec = _excited_spec()
    holdout_spec = _holdout_spec(excited_spec)

    results: list[dict[str, object]] = []

    for seed in SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            excited = generate_data(config, excited_spec, master_seed=seed)
        if not fingerprints_equivalent(excited.process_config.fingerprint, EXPECTED_FINGERPRINT):
            raise RuntimeError(
                f"Seed {seed}: process/fingerprint={excited.process_config.fingerprint!r}, "
                f"erwartet {EXPECTED_FINGERPRINT!r} -- Config weicht vom X2-Lauf ab."
            )

        train, validation, test_excited = build_datasets(
            excited, excited_spec.n_train_max, policy=ValidationPolicy.FIXED_POOL
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            holdout = generate_data(config, holdout_spec, master_seed=HOLDOUT_SEED)
        test_constant = TrajectoryWindowDataset.from_trajectory(
            holdout.trajectory,
            holdout.split,
            SplitName.TEST,
            excited_spec.window,
            scalers=train.scalers,
        )

        test_sets = (
            ("excited", excited, test_excited),
            ("constant_holdout", holdout, test_constant),
        )

        baseline_factories = (("persistence", PersistenceModel), ("static", StaticRegressionModel))
        for label, factory in baseline_factories:
            model = factory()
            model.fit(train, seed=excited.seeds.model_seed, validation=validation)
            for test_name, source_data, test_dataset in test_sets:
                q_rec = evaluate_recoverable_power(model, source_data, test_dataset)
                print(f"[seed={seed}] {label}/{test_name}\n{q_rec.format()}\n", flush=True)
                results.append(
                    {
                        "seed": seed,
                        "model": label,
                        "test_set": test_name,
                        "q_rec": q_rec.as_flat_metrics(),
                    }
                )

        ckpt_path = CHECKPOINT_DIR / f"gru_seed{seed}.pt"
        gru_model, meta = load_checkpoint(ckpt_path)
        if not fingerprints_equivalent(meta["process_fingerprint"], EXPECTED_FINGERPRINT):
            raise RuntimeError(
                f"Seed {seed}: Checkpoint-Fingerprint {meta['process_fingerprint']!r} weicht "
                f"von {EXPECTED_FINGERPRINT!r} ab -- Checkpoint stammt nicht vom aktuellen X2-Lauf."
            )
        scalers = (meta["input_scaler"], meta["target_scaler"])
        gru_test_excited = TrajectoryWindowDataset.from_trajectory(
            excited.trajectory, excited.split, SplitName.TEST, excited_spec.window, scalers=scalers
        )
        gru_test_constant = TrajectoryWindowDataset.from_trajectory(
            holdout.trajectory, holdout.split, SplitName.TEST, excited_spec.window, scalers=scalers
        )
        for test_name, source_data, test_dataset in (
            ("excited", excited, gru_test_excited),
            ("constant_holdout", holdout, gru_test_constant),
        ):
            q_rec = evaluate_recoverable_power(gru_model, source_data, test_dataset)
            print(f"[seed={seed}] gru/{test_name}\n{q_rec.format()}\n", flush=True)
            results.append(
                {
                    "seed": seed,
                    "model": "gru",
                    "test_set": test_name,
                    "q_rec": q_rec.as_flat_metrics(),
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fertig. {len(results)} Records geschrieben nach {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
