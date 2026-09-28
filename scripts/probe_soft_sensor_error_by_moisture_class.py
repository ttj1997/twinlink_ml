r"""GA2: Ist der Feuchte-Soft-Sensor an der Feuchtegrenze genauso gut wie im Mittel.

Anlass: Befund 64 rechnet mit EINEM globalen Fehler (`rmse_run_median_physical`,
`product_moisture_kg_per_kg`, LSTM, `n_train=64`, `test_set=constant_holdout`, `h=36`,
Median ueber 5 Seeds: 0,000543 kg/kg angeregt trainiert / 0,001488 kg/kg konstant trainiert,
Befund 62 HA1). Das Optimum der Regelungsstudien liegt an 17 von 25 Gitterpunkten (Befund 62)
genau auf der Feuchtegrenze (0,20 kg/kg) -- ein globaler Fehler sagt nichts darueber, ob der
Soft Sensor GENAU DORT genauso gut ist wie im Rest des Wertebereichs.

Kein Checkpoint der angeregt trainierten LSTM aus Befund 51/58 ist committet
(`.gitignore:38` schliesst `artifacts/checkpoints/` aus; die in
`artifacts/runs/arch_learning_curve_projected.jsonl` protokollierten `checkpoint/path`-Werte
zeigen auf eine fremde Maschine und existieren in diesem Checkout nicht) -- deshalb trainiert
dieses Skript 1-3 Seeds NEU, mit demselben Protokoll wie Befund 51/58
(`scripts/run_arch_learning_curve_worker.py`, dessen `_excited_spec`/`_holdout_spec`/
`HOLDOUT_SEED`/`CONFIG_PATH`/`TARGET_NAMES` unveraendert importiert werden -- keine Zeile
dort geaendert), aber NUR fuer die LSTM-Architektur bei `n_train=64` (nicht die volle
Budget/Architektur-Matrix) und faengt zusaetzlich die Rohfehler pro Fenster/Schritt ab, die
`RolloutEvaluator.evaluate` intern berechnet, aber nicht persistiert (nur aggregierte
Metriken werden geloggt, siehe `src/twinlink_ml/evaluation/evaluator.py::RolloutEvaluator.
evaluate`).

Was gemessen wird
------------------
a) Verteilung der WAHREN Produktfeuchte (`future_targets_truth`, rauschfrei, ueber alle
   Fenster und Schritte 1..36 gepoolt) im Trainings-Split (`n_train=64`), im angeregten
   Test-Split und im Konstantlauf-Holdout -- je Anteil in festen Feuchteklassen, u.a.
   0,18-0,22 kg/kg (die Klasse um die Feuchtegrenze).
b) Fehler des LSTM (gegen die VERRAUSCHTE Referenz `future_targets_truth` -> nein:
   `future_targets`, dieselbe Referenz wie `rmse_run_median_physical` in der bestehenden
   Pipeline -- siehe `RolloutEvaluator.evaluate`, `reference_physical`) nach Klassen der
   WAHREN Feuchte (`truth_physical`), auf dem Konstantlauf-Holdout, gepoolt ueber alle
   Fenster und Schritte 1..36 je Klasse.
c) Fehler in der Klasse um 0,20 kg/kg (0,18-0,22) gegen den globalen (klassenungewichteten)
   Fehler auf demselben Testsatz.

Identisches Protokoll wie Befund 51/58, EINE zulaessige Abweichung: nur die LSTM-Architektur
bei `n_train=64` wird gerechnet (nicht Baselines, nicht die anderen Budgetpunkte/Architekturen)
-- Befund 51/58 haben diese Zelle bereits unter demselben Protokoll gerechnet (RMSE-Werte in
Befund 62 HA1 zitiert); dieses Skript reproduziert und ERWEITERT sie um Rohfehler-Arrays,
ohne den Konfigurations-Fingerprint zu aendern (`bddca37f716f5cfc`, gegen `generate_data`
geprueft, exakt wie im Worker).

Laufzeitschaetzung: Befund 51s vollstaendiger Sweep (4 Budgetpunkte x 8 Architekturen +
3 Baselines, 5 Seeds parallel) brauchte 3,96 h (14250,8 s) fuer 5 Jobs. Dieses Skript
rechnet je Seed nur EINE Architektur bei EINEM (dem groessten) Budgetpunkt -- die
Datenerzeugung (Pool-Simulation, pro Seed fix) ist der dominante, unveraenderte Kostenanteil;
das eigentliche Training ist ein Bruchteil des Sweep-Trainingsanteils. Seed 1 wird zuerst
als zeitgestoppter Pilot gerechnet (wie Befund 51s eigenes Vorgehen: vorab geschaetzt, dann
am ersten Punkt gemessen), bevor Seed 2/3 folgen.

Aufruf::

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

# Feste Feuchteklassen [kg/kg), letztes Intervall offen. 0,18/0,22 klammern die
# Feuchtegrenze (0,20 kg/kg) symmetrisch ein -- die Klasse, in der laut Befund 62 das
# Optimum an 17 von 25 Gitterpunkten sitzt.
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
    """Wie `RolloutEvaluator.evaluate`, aber die physikalischen Rohfelder werden
    zurueckgegeben statt nur aggregierte Metriken -- `RolloutEvaluator.evaluate` selbst
    bleibt unveraendert (`src/twinlink_ml/evaluation/evaluator.py` nicht angefasst), diese
    Funktion dupliziert nur die Array-Konstruktion (Batches sammeln, konkatenieren,
    `scaler.inverse`), identisch zu deren Koerper."""
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
            raise RuntimeError("Datensatz enthaelt keine rauschfreie Referenz (future_targets_truth).")
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


def _moisture_channel(arrays: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extrahiert `product_moisture_kg_per_kg` (Kanal `MOISTURE_INDEX`) aus Vorhersage,
    Referenz und Wahrheit, jeweils zu 1D ueber (Fenster x Schritte 1..36) geglaettet, plus
    `run_index` auf dieselbe Laenge expandiert (ein Laufindex war je FENSTER gespeichert,
    hier auf jeden der `horizon` Schritte dieses Fensters wiederholt) -- noetig fuer die
    laufweise Median-Aggregation (siehe `_rmse_per_run_median`)."""
    pred = arrays["prediction_physical"][..., MOISTURE_INDEX].reshape(-1)
    ref = arrays["reference_physical"][..., MOISTURE_INDEX].reshape(-1)
    truth = arrays["truth_physical"][..., MOISTURE_INDEX].reshape(-1)
    horizon = arrays["prediction_physical"].shape[1]
    run_index = np.repeat(arrays["run_index"], horizon)
    return pred, ref, truth, run_index


def _rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))


def _rmse_per_run_median(prediction: np.ndarray, reference: np.ndarray, run_index: np.ndarray) -> float | None:
    """Wie `twinlink_ml.evaluation.evaluator.per_run_rmse_median` (die Definition hinter
    Befund 62 HA1s 0,000543-kg/kg-Zahl: RMSE je Lauf, dann Median ueber Laeufe) -- NICHT
    der gepoolte/geglaettete RMSE ueber alle Fenster/Schritte (siehe Moduldocstring von
    `evaluator.py`, Abschnitt "Laufweise Normierung ist die Primärzahl"). Diese Funktion
    ist die primaere Kennzahl fuer GA2c, weil sie mit Befund 62 HA1s Definition
    vergleichbar bleibt -- der gepoolte Wert (`_rmse`) ist hier nur Nebeninformation."""
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
    boundary_rmse = _rmse(prediction[boundary_mask], reference[boundary_mask]) if n_boundary > 0 else None
    boundary_rmse_per_run_median = (
        _rmse_per_run_median(prediction[boundary_mask], reference[boundary_mask], run_index[boundary_mask])
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
                "rmse_pooled_kg_per_kg": _rmse(prediction[mask], reference[mask]) if n > 0 else None,
                "rmse_per_run_median_kg_per_kg": (
                    _rmse_per_run_median(prediction[mask], reference[mask], run_index[mask]) if n > 0 else None
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=worker.CONFIG_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1], help="Trainings-Seeds (Befund-51/58-Nummerierung)")
    parser.add_argument("--num-threads", type=int, default=1, help="torch.set_num_threads (wie Befund 51/58)")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "artifacts" / "checkpoints")
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

        train, validation, test_excited = build_datasets(excited, N_TRAIN, policy=ValidationPolicy.FIXED_POOL)
        test_constant = TrajectoryWindowDataset.from_trajectory(
            holdout.trajectory, holdout.split, SplitName.TEST, excited_spec.window, scalers=train.scalers
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

        # Cross-Check gegen Befund 62 HA1: rmse_run_median_physical, product_moisture, h=36,
        # test_set=constant_holdout -- dieselbe Definition (Median ueber Laeufe des
        # unnormierten RMSE), hier direkt aus den abgefangenen Arrays statt ueber
        # RolloutEvaluator (der dieselbe Funktion intern aufruft).
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

    # Aggregation ueber Seeds (Median), nur wenn mehr als ein Seed gerechnet wurde. Primaerzahl
    # ist per_run_median (vergleichbar mit Befund 62 HA1), pooled nur Nebeninformation.
    aggregate: dict[str, Any] | None = None
    if len(arguments.seeds) > 1:
        global_rmses = [
            per_seed[str(s)]["error_by_class_test_constant_holdout"]["global_rmse_per_run_median_kg_per_kg"]
            for s in arguments.seeds
        ]
        boundary_rmses = [
            per_seed[str(s)]["error_by_class_test_constant_holdout"]["boundary_class_rmse_per_run_median_kg_per_kg"]
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
