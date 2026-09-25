r"""AD1: prueft, ob der GRU-Abfall h1->h36 (Befund 50i) einem Zustandsschaetzer folgt.

KEIN Training -- GRU wird aus den X2-Checkpoints geladen (``artifacts/checkpoints/
gru_seed{1,2,3}.pt``), Persistenz/statische Baseline frisch angepasst (billig,
geschlossene Form, identisches Muster zu ``probe_x3_recoverable_power.py``). Fenster-
population wie in AC2 verdrahtet: EIN Fensterset mit ``horizon=36`` (wie X2), alle 36
Horizonte als Slices ``[:h]`` desselben Fenstersets -- keine Populationsverschiebung
zwischen den Horizonten.

Modell
------
``err(h) = floor + A * exp(-h * dt / tau)``, ``dt = 25 s`` (Abtastzeit). Wenn der
h1->h36-Abfall daher kommt, dass das Modell mit wachsender Historie im eigenen
rekurrenten Zustand zunehmend sicherer wird (Zustandsschaetzer-Erklaerung, Befund 50i),
sollte ``tau`` im Modenband des gekoppelten Wassermasse/Betttemperatur-Paars liegen
(Befund 49: 30,2-359,9 s am Referenzpunkt, 133-1195 s ueber den Betriebsbereich) --
nicht bei den Baselines (kein rekurrenter Zustand, kein Grund fuer denselben Abfall).

Fit je (Modell, Seed, Testmenge) einzeln, nicht ueber Seeds gemittelt: eine gemittelte
Kurve wuerde Seed-zu-Seed-Streuung in tau/A verstecken, die fuer die Bewertung "trägt
die Erklaerung" relevant ist.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

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
OUTPUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_ad1_exponential_decay.json"

SEEDS = (1, 2, 3)
HOLDOUT_SEED = 999
TARGET_NAMES = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
EXPECTED_FINGERPRINT_PROJECTION_OFF = "6ecedc6bd993fb3d"
CHANNEL = "bed_temperature_celsius"
SAMPLE_TIME_S = 25.0
HORIZONS = tuple(range(1, 37))

REFERENCE_FLOOR_K = 0.181
REFERENCE_A_K = 0.143
REFERENCE_TAU_S = 143.0


def _excited_spec() -> DataSpec:
    return DataSpec(
        sample_time_s=SAMPLE_TIME_S,
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


def _exponential_model(h: np.ndarray, floor: float, amplitude: float, tau: float) -> np.ndarray:
    return floor + amplitude * np.exp(-h * SAMPLE_TIME_S / tau)


def _fit_exponential(err_by_h: np.ndarray) -> dict[str, object]:
    """Fit ``err(h) = floor + A*exp(-h*dt/tau)``, meldet Nichtkonvergenz statt abzustuerzen."""
    h = np.asarray(HORIZONS, dtype=np.float64)
    floor_guess = float(np.min(err_by_h))
    amplitude_guess = float(err_by_h[0] - floor_guess)
    initial = (floor_guess, amplitude_guess, 100.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            params, _covariance = curve_fit(
                _exponential_model,
                h,
                err_by_h,
                p0=initial,
                bounds=([0.0, -np.inf, 1e-6], [np.inf, np.inf, np.inf]),
                maxfev=20_000,
            )
    except (RuntimeError, OptimizeWarning) as exc:
        return {"converged": False, "reason": str(exc)}

    floor, amplitude, tau = (float(v) for v in params)
    fitted = _exponential_model(h, floor, amplitude, tau)
    residual = err_by_h - fitted
    rms_residual = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((err_by_h - np.mean(err_by_h)) ** 2))
    r2 = float("nan") if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return {
        "converged": True,
        "floor_k": floor,
        "amplitude_k": amplitude,
        "tau_s": tau,
        "rms_residual_k": rms_residual,
        "r2": r2,
    }


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    excited_spec = _excited_spec()
    holdout_spec = _holdout_spec(excited_spec)
    channel_index = TARGET_NAMES.index(CHANNEL)

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

        train, validation, _test_excited = build_datasets(
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

        models = {
            "persistence": (baseline_models["persistence"], train.scalers),
            "static": (baseline_models["static"], train.scalers),
            "gru": (gru_model, (meta["input_scaler"], meta["target_scaler"])),
        }

        for test_name in ("excited", "constant_holdout"):
            trajectory = excited.trajectory if test_name == "excited" else holdout.trajectory
            split = excited.split if test_name == "excited" else holdout.split
            for label, (model, scalers) in models.items():
                dataset = TrajectoryWindowDataset.from_trajectory(
                    trajectory, split, SplitName.TEST, excited_spec.window, scalers=scalers
                )
                batch = dataset.full_batch()
                prediction = dataset.target_scaler.inverse(model.predict(batch))
                truth = dataset.target_scaler.inverse(batch.future_targets_truth)

                pred_channel = prediction[..., channel_index : channel_index + 1]
                truth_channel = truth[..., channel_index : channel_index + 1]
                per_horizon = per_run_rmse_median(
                    pred_channel, truth_channel, batch.run_index, HORIZONS, (CHANNEL,)
                )
                err_by_h = np.array([per_horizon[CHANNEL][h] for h in HORIZONS], dtype=np.float64)

                fit = _fit_exponential(err_by_h)
                results.append(
                    {
                        "seed": seed,
                        "test_set": test_name,
                        "model": label,
                        "err_by_h_k": err_by_h.tolist(),
                        "fit": fit,
                    }
                )
                if fit["converged"]:
                    print(
                        f"[seed={seed}] {test_name}/{label}: floor={fit['floor_k']:.4f} K  "
                        f"A={fit['amplitude_k']:.4f} K  tau={fit['tau_s']:.1f} s  "
                        f"rms_residual={fit['rms_residual_k']:.4f} K  r2={fit['r2']:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[seed={seed}] {test_name}/{label}: Fit NICHT konvergent "
                        f"({fit['reason']})",
                        flush=True,
                    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fertig. {len(results)} Records geschrieben nach {OUTPUT_PATH}")
    print(
        f"Referenz (Vier-Punkt-Fit): floor={REFERENCE_FLOOR_K} K, A={REFERENCE_A_K} K, "
        f"tau={REFERENCE_TAU_S} s"
    )


if __name__ == "__main__":
    main()
