r"""Nachfrage zu Schritt 3.

Ist NRMSE=21.53 (xLSTM, energy_target, h=36, worst run) ein
Instabilitaets-Fall (NaN/Inf) oder nur ein schwerer Testlauf -- und ist er xLSTM-
spezifisch oder teilen GRU/Mamba dieselbe Streuung auf demselben Lauf?

Reproduziert exakt dieselben Daten/Seeds wie ``run_xlstm_reduced_comparison.py``
(gleicher Fingerprint/Split/Seed-Check), trainiert xLSTM, GRU und Mamba erneut
(deterministisch, ``torch.set_num_threads(1)`` -- reproduziert die bereits geloggten
Werte bitgleich, dient hier als Sanity-Check UND als Quelle fuer die vollen
Pro-Lauf-NRMSE-Arrays, die ``EvaluationResult`` nicht persistiert, nur Median/Min/Max).

Fuer den Lauf mit der xLSTM-maximalen NRMSE (energy_target, h=36):
  (a) rohe Vorhersagewerte auf Endlichkeit pruefen (np.isfinite, nicht nur "sieht
      schlecht aus")
  (b) GRU- und Mamba-NRMSE auf GENAU demselben run_index, derselbe Zielgroesse/Horizont
      zeigen -- gemeinsam schwerer Lauf oder xLSTM-spezifisch?

Kein Volltraining im Sinne einer neuen Kostenmessung -- exakt die Produktionsgroesse
(n_train=16), weil das Phaenomen selbst nur bei Produktionsgroesse auftrat.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import torch

from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import nrmse, per_run_nrmse
from twinlink_ml.models.composite import (
    CompositeGruSequenceModel,
    CompositeMambaSequenceModel,
    CompositeXLSTMSequenceModel,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
SEED = 1
N_TRAIN = 16

# Fingerprint aktualisiert von 82f65f576a2e5253 auf 40aae2b4526a5ed7: NumericsConfig.
# enforce_gas_velocity_projection hinzugefuegt (Default AUS). ProcessConfig.fingerprint
# hasht die gesamte Config, daher aendert sich der Wert auch bei unveraendertem Verhalten.
# Aequivalenz belegt durch test_gas_velocity_projection_disabled_matches_legacy_baseline
# (tests/test_gas_velocity_projection.py) -- siehe dort fuer den Bitgleichheits-Beleg.
EXPECTED = {
    "process/fingerprint": "40aae2b4526a5ed7",
    "split/digest": "5b9628a151344f37",
    "seed/data": 57731891,
    "seed/noise": 1868603354,
    "seed/model": 2509151119,
}


def _spec() -> DataSpec:
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


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    spec = _spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=SEED)

    actual = {
        "process/fingerprint": data.process_config.fingerprint,
        "split/digest": data.split.digest,
        "seed/data": data.seeds.data_seed,
        "seed/noise": data.seeds.noise_seed,
        "seed/model": data.seeds.model_seed,
    }
    mismatches = {k: (v, actual[k]) for k, v in EXPECTED.items() if v != actual[k]}
    if mismatches:
        raise RuntimeError(f"Daten/Seeds weichen ab: {mismatches}")
    _log("Fingerprint/Split/Seeds bitgleich bestaetigt.")

    train, validation, test = build_datasets(data, N_TRAIN, policy=ValidationPolicy.FIXED_POOL)
    _log(
        f"{train.n_windows} Trainingsfenster, {test.n_windows} Testfenster, "
        f"{len(np.unique(test.full_batch().run_index))} Testlaeufe"
    )

    scaler = test.target_scaler
    target_names = test.target_names
    energy_index = target_names.index("energy_target")

    factories = {
        "xlstm": CompositeXLSTMSequenceModel,
        "gru": CompositeGruSequenceModel,
        "mamba": CompositeMambaSequenceModel,
    }

    per_run_by_model: dict[str, dict] = {}
    raw_predictions_by_model: dict[str, tuple] = {}

    for label, factory in factories.items():
        _log(f"-- Modell {label!r}: Training (Reproduktion) --")
        start = time.perf_counter()
        model = factory()
        model.fit(train, seed=data.seeds.model_seed, validation=validation, verbose=False)
        elapsed = time.perf_counter() - start
        _log(f"-- Modell {label!r}: fertig ({elapsed:.1f}s) --")

        batch = test.full_batch()
        prediction_scaled = model.predict(batch)
        prediction = scaler.inverse(prediction_scaled)
        reference = scaler.inverse(batch.future_targets)
        runs = batch.run_index

        spread = per_run_nrmse(prediction, reference, runs, (1, 36), target_names)
        _log(
            f"   Sanity-Check (muss zu xlstm_reduced_comparison.jsonl passen): "
            f"energy_target h36 median={spread['energy_target'][36][0]:.4f} "
            f"min={spread['energy_target'][36][1]:.4f} max={spread['energy_target'][36][2]:.4f}"
        )

        per_run_by_model[label] = {"prediction": prediction, "reference": reference, "runs": runs}
        raw_predictions_by_model[label] = (prediction, reference, runs)

    # -------------------------------------------- (a) welcher run_index ist der schlimmste?
    xlstm_pred, xlstm_ref, xlstm_runs = raw_predictions_by_model["xlstm"]
    unique_runs = np.unique(xlstm_runs)
    per_run_energy_h36 = []
    for run in unique_runs:
        mask = xlstm_runs == run
        value = nrmse(
            xlstm_pred[mask, :36, energy_index : energy_index + 1],
            xlstm_ref[mask, :36, energy_index : energy_index + 1],
            axis=(0, 1),
        )[0]
        per_run_energy_h36.append((run, value))
    per_run_energy_h36.sort(key=lambda item: item[1], reverse=True)

    print("\n=== xLSTM: energy_target, h=36, NRMSE je Testlauf (absteigend, Top 5) ===")
    for run, value in per_run_energy_h36[:5]:
        print(f"  run_index={int(run)}: NRMSE={value:.4f}")

    worst_run, worst_value = per_run_energy_h36[0]
    print(f"\nSchlimmster Lauf: run_index={int(worst_run)}, NRMSE={worst_value:.4f}")

    # ---------------------------------------------------------------- (a) Endlichkeit pruefen
    mask = xlstm_runs == worst_run
    worst_prediction = xlstm_pred[mask, :36, energy_index]
    worst_reference = xlstm_ref[mask, :36, energy_index]
    print(
        "\n=== (a) Endlichkeitspruefung auf dem schlimmsten Lauf (xLSTM, energy_target, h<=36) ==="
    )
    print(f"  Fenster in diesem Lauf: {mask.sum()}")
    print(f"  Vorhersage: isfinite ueberall = {bool(np.all(np.isfinite(worst_prediction)))}")
    print(
        f"  Vorhersage: min={worst_prediction.min():.4f}  max={worst_prediction.max():.4f}  "
        f"mean={worst_prediction.mean():.4f}"
    )
    print(
        f"  Referenz (Messung):  min={worst_reference.min():.4f}  "
        f"max={worst_reference.max():.4f}  mean={worst_reference.mean():.4f}"
    )
    n_nan = int(np.isnan(worst_prediction).sum())
    n_inf = int(np.isinf(worst_prediction).sum())
    print(f"  NaN-Werte: {n_nan}, Inf-Werte: {n_inf}")

    # ------------------------------------------------ (b) GRU/Mamba auf demselben run_index
    print(f"\n=== (b) GRU/Mamba auf DEMSELBEN run_index={int(worst_run)}, energy_target, h=36 ===")
    for label in ("xlstm", "gru", "mamba"):
        pred, ref, runs = raw_predictions_by_model[label]
        run_mask = runs == worst_run
        value = nrmse(
            pred[run_mask, :36, energy_index : energy_index + 1],
            ref[run_mask, :36, energy_index : energy_index + 1],
            axis=(0, 1),
        )[0]
        # Vergleich: NRMSE ueber ALLE anderen Laeufe (Median), zur Einordnung
        other_values = []
        for run in unique_runs:
            if run == worst_run:
                continue
            m = runs == run
            v = nrmse(
                pred[m, :36, energy_index : energy_index + 1],
                ref[m, :36, energy_index : energy_index + 1],
                axis=(0, 1),
            )[0]
            other_values.append(v)
        median_other = float(np.median(other_values))
        print(
            f"  {label:<8} run={int(worst_run)}: NRMSE={value:.4f}   "
            f"(Median ueber die uebrigen {len(other_values)} Laeufe: {median_other:.4f}, "
            f"Verhaeltnis {value / median_other:.2f}x)"
        )


if __name__ == "__main__":
    main()
