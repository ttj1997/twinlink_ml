r"""AC2: prueft, ob `h36 < h1` (Befund 50i) durch die Fensterpopulation erklaerbar ist.

KEIN Training -- GRU wird aus den X2-Checkpoints geladen (``artifacts/checkpoints/
gru_seed{1,2,3}.pt``), Persistenz/statische Baseline frisch angepasst (billig,
geschlossene Form, identisches Muster zu ``probe_x3_recoverable_power.py``).

Hintergrund
-----------
``WindowConfig.horizon`` ist bei der Fenstererzeugung FEST
(``data/windows.py::TrajectoryWindowDataset.from_trajectory``: ``span =
history_length + horizon``, ``starts = arange(0, n_steps - span + 1, stride)``) -- ein
groesserer Horizont braucht mehr Restlaenge je Lauf, also WENIGER moegliche
Fensteranfaenge. X2 baute genau EIN Fensterset mit ``horizon=36``; das dort berichtete
``h=1`` ist eine Zeitschritt-Teilmenge DESSELBEN Fenstersets (Slice ``[:1]`` der
36-Schritt-Vorhersage), nicht ein eigenes, groesseres ``h=1``-Fensterset. Befund 50(i)
fragt: waere der GRU-Einschritt-Fehler anders, wenn man ihn auf der tatsaechlichen
(groesseren) ``h=1``-Population misst statt auf der (kleineren) Teilmenge, die auch bei
``h=36`` noch existiert -- und verschwindet ``h36 < h1`` dann?

Vorgehen: je Testpool zwei Fenstersets bauen -- A (``horizon=36``, wie X2) und B
(``horizon=1``, sonst identisch: ``history_length``/``stride``/``target_names``/
Skalierer). Beide verwenden ``starts = arange(0, n_steps - span + 1, stride)`` mit
demselben ``stride``/Ursprung -- A's Startmenge ist deshalb exakt eine Teilmenge von B's
Startmenge (kleineres ``span`` bei B -> hoehere obere Grenze), keine Matching-Heuristik
noetig: die "h=36-Teilmenge" von B ist B eingeschraenkt auf ``window_start <=
max(A.window_start)``.

Referenz: rauschfreie Wahrheit (``future_targets_truth``), wie die bereits berichteten
Befund-50-Zahlen (``exhaust_rmse_run_median_physical``/``nrmse_truth_run_median`` --
siehe deren Identitaet, Befund 50i).
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import per_run_rmse_median
from twinlink_ml.models import PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.checkpoints import load_checkpoint
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_ac2_h1_window_population.json"

SEEDS = (1, 2, 3)
HOLDOUT_SEED = 999
TARGET_NAMES = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
EXPECTED_FINGERPRINT_PROJECTION_OFF = "6ecedc6bd993fb3d"
CHANNEL = "bed_temperature_celsius"


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


def _rmse_by_population(
    prediction_physical: np.ndarray,
    truth_physical: np.ndarray,
    run_index: np.ndarray,
    channel_index: int,
    *,
    keep: np.ndarray | None = None,
) -> tuple[float, int]:
    """RMSE [K] (``per_run_rmse_median``, h=1, ein Kanal), optional auf ``keep`` beschraenkt.

    ``keep`` ist eine boolesche Maske ueber die Fensterachse -- damit laesst sich
    dieselbe Auswertung auf eine Teilmenge der Fenster einschraenken, ohne den
    Datensatz neu zu bauen.
    """
    if keep is not None:
        prediction_physical = prediction_physical[keep]
        truth_physical = truth_physical[keep]
        run_index = run_index[keep]
    pred = prediction_physical[:, :1, channel_index : channel_index + 1]
    truth = truth_physical[:, :1, channel_index : channel_index + 1]
    result = per_run_rmse_median(pred, truth, run_index, (1,), (CHANNEL,))
    return result[CHANNEL][1], int(prediction_physical.shape[0])


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    excited_spec = _excited_spec()
    holdout_spec = _holdout_spec(excited_spec)
    window_config_h36 = excited_spec.window
    window_config_h1 = dataclasses.replace(window_config_h36, horizon=1)

    results: list[dict[str, object]] = []

    for seed in SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            excited = generate_data(config, excited_spec, master_seed=seed)
        if not fingerprints_equivalent(
            excited.process_config.fingerprint, EXPECTED_FINGERPRINT_PROJECTION_OFF
        ):
            raise RuntimeError(
                f"Seed {seed}: process/fingerprint={excited.process_config.fingerprint!r}, "
                f"erwartet {EXPECTED_FINGERPRINT_PROJECTION_OFF!r}."
            )

        train, validation, test_excited_h36 = build_datasets(
            excited, excited_spec.n_train_max, policy=ValidationPolicy.FIXED_POOL
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            holdout = generate_data(config, holdout_spec, master_seed=HOLDOUT_SEED)

        baseline_models = {}
        baseline_factories = (("persistence", PersistenceModel), ("static", StaticRegressionModel))
        for label, factory in baseline_factories:
            model = factory()
            model.fit(train, seed=excited.seeds.model_seed, validation=validation)
            baseline_models[label] = model

        ckpt_path = CHECKPOINT_DIR / f"gru_seed{seed}.pt"
        gru_model, meta = load_checkpoint(ckpt_path)
        if not fingerprints_equivalent(
            meta["process_fingerprint"], EXPECTED_FINGERPRINT_PROJECTION_OFF
        ):
            raise RuntimeError(f"Seed {seed}: Checkpoint-Fingerprint weicht ab.")
        gru_scalers = (meta["input_scaler"], meta["target_scaler"])

        for test_name, trajectory, split in (
            ("excited", excited.trajectory, excited.split),
            ("constant_holdout", holdout.trajectory, holdout.split),
        ):
            channel_index = TARGET_NAMES.index(CHANNEL)

            for label, model in (
                ("persistence", baseline_models["persistence"]),
                ("static", baseline_models["static"]),
                ("gru", gru_model),
            ):
                scalers = train.scalers if label != "gru" else gru_scalers

                dataset_h36 = TrajectoryWindowDataset.from_trajectory(
                    trajectory, split, SplitName.TEST, window_config_h36, scalers=scalers
                )
                dataset_h1 = TrajectoryWindowDataset.from_trajectory(
                    trajectory, split, SplitName.TEST, window_config_h1, scalers=scalers
                )

                batch_h36 = dataset_h36.full_batch()
                batch_h1 = dataset_h1.full_batch()

                pred_h36 = dataset_h36.target_scaler.inverse(model.predict(batch_h36))
                truth_h36 = dataset_h36.target_scaler.inverse(batch_h36.future_targets_truth)
                pred_h1 = dataset_h1.target_scaler.inverse(model.predict(batch_h1))
                truth_h1 = dataset_h1.target_scaler.inverse(batch_h1.future_targets_truth)

                max_start_h36 = float(np.max(batch_h36.window_start))
                keep_subset = batch_h1.window_start <= max_start_h36

                rmse_h36_native, n_h36 = _rmse_by_population(
                    pred_h36, truth_h36, batch_h36.run_index, channel_index
                )
                rmse_h1_all, n_h1_all = _rmse_by_population(
                    pred_h1, truth_h1, batch_h1.run_index, channel_index
                )
                rmse_h1_subset, n_h1_subset = _rmse_by_population(
                    pred_h1, truth_h1, batch_h1.run_index, channel_index, keep=keep_subset
                )

                results.append(
                    {
                        "seed": seed,
                        "test_set": test_name,
                        "model": label,
                        "n_windows_h36_native": n_h36,
                        "n_windows_h1_all": n_h1_all,
                        "n_windows_h1_h36_subset": n_h1_subset,
                        "rmse_k_h36_dataset_at_h1": rmse_h36_native,
                        "rmse_k_h1_dataset_all_windows": rmse_h1_all,
                        "rmse_k_h1_dataset_h36_subset": rmse_h1_subset,
                    }
                )
                print(
                    f"[seed={seed}] {test_name}/{label}: "
                    f"n(h36)={n_h36} n(h1,all)={n_h1_all} n(h1,subset)={n_h1_subset}  "
                    f"RMSE[K]: h36-Datensatz@h1={rmse_h36_native:.4f}  "
                    f"h1-Datensatz,alle={rmse_h1_all:.4f}  "
                    f"h1-Datensatz,h36-Teilmenge={rmse_h1_subset:.4f}",
                    flush=True,
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fertig. {len(results)} Records geschrieben nach {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
