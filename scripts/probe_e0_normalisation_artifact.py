r"""E0: Ist der d50-Maskierungseffekt (F3) ein Normierungsartefakt?.

KEIN Netztraining -- reine Inferenz auf vorhandenen Checkpoints (gru_seed1.pt,
lstm_seed1.pt). KEINE Aenderung an src/. Nutzt evaluator.nrmse()/per_run_nrmse()
(evaluator.py:79-147) unveraendert als Referenz fuer den per-Lauf-Nenner, berechnet ihn
hier aber direkt (np.std je Lauf), um ihn getrennt von der RMSE auszuweisen -- die
bestehende Funktion gibt nur das fertige Verhaeltnis zurueck.
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.simulation.config import MaterialConfig
from twinlink_ml.utils.checkpoints import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
HOLDOUT_SEED = 999
SEED_FOR_MODELS = 1

# Zwei getrennte Konstanten statt einer gemeinsamen: dieses Skript prueft zwei
# UNTERSCHIEDLICHE Provenienzen gegeneinander, nicht denselben Wert zweimal.
# EXPECTED_FP_FRESH ist der Fingerprint der HIER frisch erzeugten Holdout-Daten (aktueller
# Code-Stand). EXPECTED_FP_CHECKPOINT ist der Fingerprint, der beim Training in die
# artifacts/checkpoints/{gru,lstm}_seed1.pt-Dateien geschrieben wurde (permanent, wird beim
# Laden nie neu berechnet, siehe utils/checkpoints.py::load_checkpoint) -- diese Checkpoints
# wurden VOR Einfuehrung von NumericsConfig.enforce_gas_velocity_projection trainiert und
# bleiben deshalb fuer immer beim alten Wert.
#
# Die Gleichsetzung "40aae2b4526a5ed7 entspricht 82f65f576a2e5253" ist NUR durch den
# Bitgleichheits-Regressionstest test_gas_velocity_projection_disabled_matches_legacy_
# baseline (tests/test_gas_velocity_projection.py) gedeckt: er belegt, dass beide
# Fingerprints bei ausgeschalteter Projektion dieselben Trajektorien liefern. Ohne diesen
# Beleg waere die Auswertung hier ungueltig -- siehe die Laufzeitpruefung in main()
# (enforce_gas_velocity_projection muss AUS sein, sonst wertet ein auf unprojizierten Daten
# trainiertes Checkpoint auf einer anderen Verteilung aus, ohne dass sich das an einem
# Fingerprint-Vergleich zeigen wuerde).
EXPECTED_FP_FRESH = "40aae2b4526a5ed7"
EXPECTED_FP_CHECKPOINT = "82f65f576a2e5253"


def _excited_spec() -> DataSpec:
    return DataSpec(
        sample_time_s=25.0, run_duration_s=28_800.0, n_train_max=64, n_val=32, n_test=32,
        history_length=32, horizon=36, stride=8,
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


def main() -> None:
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    # Die gru/lstm-Checkpoints wurden auf UNPROJIZIERTEN Daten trainiert. Wertet dieses
    # Skript sie auf projizierten Daten aus, ist das ein Verteilungsbruch, den kein
    # Fingerprint-Vergleich mehr anzeigt (beide Konstanten wuerden dann einfach "wie
    # erwartet" voneinander abweichen) -- deshalb eine eigene, explizite Pruefung.
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
    assert holdout.process_config.fingerprint == EXPECTED_FP_FRESH

    masks = _violation_masks(holdout.trajectory, config.material)
    test_ids = set(holdout.split.test)
    run_id = holdout.trajectory.run_id
    any_violation = {}
    for mask in masks.values():
        for i, rid in enumerate(run_id):
            if rid in test_ids:
                any_violation[rid] = any_violation.get(rid, False) or bool(mask[i].any())
    violating = sorted(r for r, v in any_violation.items() if v)
    clean = sorted(r for r, v in any_violation.items() if not v)
    print(f"Holdout: {len(test_ids)} Laeufe, {len(violating)} verletzend, {len(clean)} sauber")

    window = excited_spec.window
    targets = excited_spec.target_names
    horizons = (1, excited_spec.horizon)

    result: dict = {"n_violating": len(violating), "n_clean": len(clean), "by_architecture": {}}

    for arch in ("gru", "lstm"):
        ckpt_path = CHECKPOINT_DIR / f"{arch}_seed{SEED_FOR_MODELS}.pt"
        model, meta = load_checkpoint(ckpt_path)
        assert meta["process_fingerprint"] == EXPECTED_FP_CHECKPOINT
        scalers = (meta["input_scaler"], meta["target_scaler"])
        dataset = TrajectoryWindowDataset.from_trajectory(
            holdout.trajectory, holdout.split, SplitName.TEST, window, scalers=scalers
        )
        batch = dataset.full_batch()
        prediction = model.predict(batch)
        reference = batch.future_targets
        run_index = batch.run_index
        scaler = dataset.target_scaler
        prediction_phys = scaler.inverse(prediction)
        reference_phys = scaler.inverse(reference)

        run_ids_by_row = dataset.run_ids  # local row index -> run_id, in dataset order
        per_run: dict = {}
        for local_row, rid in enumerate(run_ids_by_row):
            row_mask = run_index == local_row
            per_run[rid] = {}
            for h in horizons:
                ref_h = reference_phys[row_mask][:, :h, :]
                pred_h = prediction_phys[row_mask][:, :h, :]
                scale = np.std(ref_h, axis=(0, 1))
                rmse = np.sqrt(np.mean((pred_h - ref_h) ** 2, axis=(0, 1)))
                per_run[rid][f"h{h}"] = {
                    "scale": {t: float(scale[i]) for i, t in enumerate(targets)},
                    "rmse": {t: float(rmse[i]) for i, t in enumerate(targets)},
                }

        def _agg(
            run_list: list[str], h: str, field: str, target: str, *, per_run: dict = per_run
        ) -> dict:
            values = [per_run[r][h][field][target] for r in run_list]
            arr = np.asarray(values)
            return {
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        arch_result: dict = {}
        for h in (f"h{horizons[0]}", f"h{horizons[1]}"):
            arch_result[h] = {}
            for target in targets:
                arch_result[h][target] = {
                    "scale_violating": _agg(violating, h, "scale", target),
                    "scale_clean": _agg(clean, h, "scale", target),
                    "rmse_violating": _agg(violating, h, "rmse", target),
                    "rmse_clean": _agg(clean, h, "rmse", target),
                }
        result["by_architecture"][arch] = arch_result
        print(f"-- {arch} fertig --")

    result["_elapsed_s"] = time.monotonic() - t0
    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_e0_normalisation_artifact.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\ngeschrieben: {out_path} ({result['_elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
