r"""E2/E3/E4: Lag-Tiefen-Kurve (FIR/ARX) plus punktweise Kontrollbasis.

KEIN Netztraining -- reine OLS-Regression (np.linalg.lstsq), konsistent mit
simulation.diagnostics.static_explained_variance() (unveraendert, nicht aufgerufen, nur
als Vorbild fuer die Merkmalslogik). KEINE Aenderung an src/.

Datenbasis: fp 40aae2b4526a5ed7 (vor Einfuehrung von enforce_gas_velocity_projection:
82f65f576a2e5253 -- Aequivalenz belegt durch test_gas_velocity_projection_disabled_
matches_legacy_baseline, tests/test_gas_velocity_projection.py), Δt=25s,
greybox_gab_placeholder.yaml, Konstantlauf-Holdout wie arch_gab_constant_holdout.jsonl
(HOLDOUT_SEED=999), 3 Trainingspool-Seeds (1,2,3).

FIR: u(t),...,u(t-k), alle vier Stellgroessen, linear+quadratisch elementweise, kein
Kreuzprodukt, keine Standardisierung, Intercept.
ARX: FIR-Merkmale plus y(t-1)..y(t-k) des jeweiligen Zielkanals (linear).

R2_in-sample: Fit und Auswertung auf demselben Trainingspool (excited.split.train, 64
Laeufe). R2_holdout: dieselben Koeffizienten, aber Varianz/Residuen auf dem fixen
Konstantlauf-Holdout (32 Laeufe, seed 999) berechnet -- Referenzvarianz aus dem Holdout
selbst (wie in static_explained_variance).
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.evaluation import DataSpec, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.simulation.config import MaterialConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
HOLDOUT_SEED = 999
TRAIN_SEEDS = (1, 2, 3)
K_VALUES = (1, 2, 4, 8, 16, 32, 43)
K_VALUES_E3 = (1, 8, 43)
EXPECTED_FP = "40aae2b4526a5ed7"


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


def _violation_mask(trajectory: Trajectory, material: MaterialConfig) -> np.ndarray:
    """Punktweise Verletzungsmaske (d50-Basis, Kriterien wie D2/F1), Form (n_runs, n_steps)."""
    diagnostics = trajectory.diagnostics
    output_names = trajectory.metadata["output_names"]
    bed_temp_idx = output_names.index("bed_temperature_celsius")
    return (
        (diagnostics["fluidization_margin_upper_m_per_s"] < 0.0)
        | (diagnostics["fluidization_margin_lower_m_per_s"] < 0.0)
        | (trajectory.outputs[:, :, bed_temp_idx] > 80.0)
        | (diagnostics["particle_moisture_kg_per_kg"] > material.critical_moisture_kg_per_kg)
    )


def _fir_design(
    actions: np.ndarray, k: int, *, point_mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FIR-Merkmalsmatrix ueber alle Laeufe, Form (n_samples, 8*(k+1)+1), plus Ziel-Slicer.

    ``actions``: (n_runs, n_steps, 4). Fuer jeden Lauf werden Zeitpunkte t=k..n_steps-1
    verwendet (Lag k braucht k Vorlaufschritte). ``point_mask`` (n_runs, n_steps), falls
    gegeben: Punkte mit True an Position t werden zusaetzlich ausgeschlossen (E3).
    Rueckgabe: (design, valid_run_idx, valid_t_idx) -- letztere zur Ziel-Extraktion.
    """
    n_runs, n_steps, n_actions = actions.shape
    run_idx_list = []
    t_idx_list = []
    feat_list = []
    for run in range(n_runs):
        ts = np.arange(k, n_steps)
        if point_mask is not None:
            keep = ~point_mask[run, ts]
            ts = ts[keep]
        if ts.size == 0:
            continue
        # (n_valid, k+1, n_actions)
        lags = np.stack([actions[run, ts - i, :] for i in range(k + 1)], axis=1)
        lin = lags.reshape(ts.size, -1)
        quad = lin**2
        feat_list.append(np.concatenate([lin, quad], axis=-1))
        run_idx_list.append(np.full(ts.size, run))
        t_idx_list.append(ts)
    features = np.concatenate(feat_list, axis=0)
    run_idx = np.concatenate(run_idx_list, axis=0)
    t_idx = np.concatenate(t_idx_list, axis=0)
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=-1)
    return design, run_idx, t_idx


def _y_lags(
    outputs_channel: np.ndarray, run_idx: np.ndarray, t_idx: np.ndarray, k: int
) -> np.ndarray:
    """y(t-1)..y(t-k) fuer dieselben (run, t)-Paare wie die FIR-Design-Zeilen."""
    if k == 0:
        return np.zeros((run_idx.size, 0))
    cols = [outputs_channel[run_idx, t_idx - i] for i in range(1, k + 1)]
    return np.stack(cols, axis=-1)


def _r2(design: np.ndarray, target: np.ndarray, coefficients: np.ndarray) -> float:
    variance = float(np.var(target))
    if variance <= 0.0:
        return float("nan")
    residual = target - design @ coefficients
    return 1.0 - float(np.var(residual)) / variance


def main() -> None:
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    excited_spec = _excited_spec()
    holdout_spec = _holdout_spec(excited_spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        holdout = generate_data(config, holdout_spec, master_seed=HOLDOUT_SEED)
    assert holdout.process_config.fingerprint == EXPECTED_FP
    # split.test Block, siehe IndexBlockSplitter
    holdout_actions = holdout.trajectory.actions[4:36]
    holdout_outputs = holdout.trajectory.outputs[4:36]
    holdout_point_mask = _violation_mask(holdout.trajectory, config.material)[4:36]
    print(f"Holdout: {holdout_actions.shape[0]} Laeufe, fp={holdout.process_config.fingerprint}")

    channels = holdout.trajectory.metadata["output_names"]  # 4 Kanaele
    print(f"Zielkanaele: {channels}")

    pools: dict[int, dict] = {}
    for seed in TRAIN_SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            excited = generate_data(config, excited_spec, master_seed=seed)
        assert excited.process_config.fingerprint == EXPECTED_FP
        train_actions = excited.trajectory.actions[:64]  # split.train Block
        train_outputs = excited.trajectory.outputs[:64]
        train_point_mask = _violation_mask(excited.trajectory, config.material)[:64]
        pools[seed] = {
            "train_actions": train_actions,
            "train_outputs": train_outputs,
            "train_point_mask": train_point_mask,
        }
        print(f"seed={seed}: Trainingspool gebaut, {time.monotonic() - t0:.1f}s seit Start")

    results_e2: dict = {"FIR": {}, "ARX": {}}
    for k in K_VALUES:
        t_k = time.monotonic()
        results_e2["FIR"][str(k)] = {c: [] for c in channels}
        results_e2["ARX"][str(k)] = {c: [] for c in channels}
        n_features_fir = 8 * (k + 1) + 1
        n_features_arx = n_features_fir + k

        design_holdout_fir, run_idx_ho, t_idx_ho = _fir_design(holdout_actions, k)

        for seed in TRAIN_SEEDS:
            pool = pools[seed]
            design_train_fir, run_idx_tr, t_idx_tr = _fir_design(pool["train_actions"], k)
            targets_train = pool["train_outputs"][run_idx_tr, t_idx_tr, :]  # (n, 4)
            targets_holdout = holdout_outputs[run_idx_ho, t_idx_ho, :]

            # FIR: alle 4 Kanaele gemeinsam per lstsq (mehrspaltige rechte Seite)
            coeff_fir, *_ = np.linalg.lstsq(design_train_fir, targets_train, rcond=None)
            for c_idx, channel in enumerate(channels):
                r2_in = _r2(design_train_fir, targets_train[:, c_idx], coeff_fir[:, c_idx])
                r2_ho = _r2(design_holdout_fir, targets_holdout[:, c_idx], coeff_fir[:, c_idx])
                results_e2["FIR"][str(k)][channel].append(
                    {
                        "seed": seed,
                        "r2_in_sample": r2_in,
                        "r2_holdout": r2_ho,
                        "n_features": n_features_fir,
                    }
                )

            # ARX: je Kanal eigene y-Lags
            for c_idx, channel in enumerate(channels):
                y_lags_train = _y_lags(pool["train_outputs"][:, :, c_idx], run_idx_tr, t_idx_tr, k)
                design_train_arx = np.concatenate(
                    [design_train_fir[:, :-1], y_lags_train, design_train_fir[:, -1:]], axis=-1
                )
                y_lags_ho = _y_lags(holdout_outputs[:, :, c_idx], run_idx_ho, t_idx_ho, k)
                design_ho_arx = np.concatenate(
                    [design_holdout_fir[:, :-1], y_lags_ho, design_holdout_fir[:, -1:]], axis=-1
                )

                target_arx = targets_train[:, c_idx]
                coeff_arx, *_ = np.linalg.lstsq(design_train_arx, target_arx, rcond=None)
                r2_in = _r2(design_train_arx, targets_train[:, c_idx], coeff_arx)
                r2_ho = _r2(design_ho_arx, targets_holdout[:, c_idx], coeff_arx)
                results_e2["ARX"][str(k)][channel].append(
                    {
                        "seed": seed,
                        "r2_in_sample": r2_in,
                        "r2_holdout": r2_ho,
                        "n_features": n_features_arx,
                    }
                )
        elapsed_total = time.monotonic() - t0
        print(f"k={k}: fertig, {time.monotonic() - t_k:.1f}s (kumuliert {elapsed_total:.1f}s)")

    # --------------------------------------------------------------------------- E3
    print("\n=== E3: punktweise Maskierung, FIR, k in {1,8,43} ===")
    results_e3: dict = {}
    for k in K_VALUES_E3:
        t_k = time.monotonic()
        results_e3[str(k)] = {c: [] for c in channels}
        design_ho_masked, run_idx_ho_m, t_idx_ho_m = _fir_design(
            holdout_actions, k, point_mask=holdout_point_mask
        )
        n_dropped_holdout = (
            holdout_actions.shape[0] * (holdout_actions.shape[1] - k) - run_idx_ho_m.size
        )
        targets_ho_masked = holdout_outputs[run_idx_ho_m, t_idx_ho_m, :]

        per_seed_dropped_train = []
        for seed in TRAIN_SEEDS:
            pool = pools[seed]
            design_tr_masked, run_idx_tr_m, t_idx_tr_m = _fir_design(
                pool["train_actions"], k, point_mask=pool["train_point_mask"]
            )
            n_total_train = pool["train_actions"].shape[0] * (pool["train_actions"].shape[1] - k)
            per_seed_dropped_train.append(n_total_train - run_idx_tr_m.size)
            targets_tr_masked = pool["train_outputs"][run_idx_tr_m, t_idx_tr_m, :]

            coeff, *_ = np.linalg.lstsq(design_tr_masked, targets_tr_masked, rcond=None)
            for c_idx, channel in enumerate(channels):
                r2_in = _r2(design_tr_masked, targets_tr_masked[:, c_idx], coeff[:, c_idx])
                r2_ho = _r2(design_ho_masked, targets_ho_masked[:, c_idx], coeff[:, c_idx])
                results_e3[str(k)][channel].append(
                    {
                        "seed": seed,
                        "r2_in_sample": r2_in,
                        "r2_holdout": r2_ho,
                        "n_points_dropped_train": per_seed_dropped_train[-1],
                        "n_points_dropped_holdout": n_dropped_holdout,
                    }
                )
        print(f"E3 k={k}: fertig, {time.monotonic() - t_k:.1f}s")

    # --------------------------------------------------------------------------- Aggregation
    def _agg(entries: list[dict], field: str) -> dict:
        arr = np.asarray([e[field] for e in entries], dtype=np.float64)
        return {
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    e2_summary: dict = {"FIR": {}, "ARX": {}}
    for family in ("FIR", "ARX"):
        for k in K_VALUES:
            e2_summary[family][str(k)] = {}
            for channel in channels:
                entries = results_e2[family][str(k)][channel]
                e2_summary[family][str(k)][channel] = {
                    "r2_in_sample": _agg(entries, "r2_in_sample"),
                    "r2_holdout": _agg(entries, "r2_holdout"),
                    "n_features": entries[0]["n_features"],
                }

    e3_summary: dict = {}
    for k in K_VALUES_E3:
        e3_summary[str(k)] = {}
        for channel in channels:
            entries = results_e3[str(k)][channel]
            e3_summary[str(k)][channel] = {
                "r2_in_sample": _agg(entries, "r2_in_sample"),
                "r2_holdout": _agg(entries, "r2_holdout"),
                "n_points_dropped_train_per_seed": [e["n_points_dropped_train"] for e in entries],
                "n_points_dropped_holdout": entries[0]["n_points_dropped_holdout"],
            }

    # --------------------------------------------------------------------------- E4
    e4: dict = {}
    ref_k = K_VALUES[-1]
    for channel in channels:
        ref_r2 = e2_summary["FIR"][str(ref_k)][channel]["r2_holdout"]["median"]
        smallest_k = None
        for k in K_VALUES:
            r2 = e2_summary["FIR"][str(k)][channel]["r2_holdout"]["median"]
            if abs(r2 - ref_r2) < 0.01:
                smallest_k = k
                break
        e4[channel] = {"smallest_k_within_0.01_of_k43": smallest_k, "r2_holdout_at_k43": ref_r2}

    result = {
        "channels": list(channels),
        "e2": e2_summary,
        "e3": e3_summary,
        "e4": e4,
        "_elapsed_s": time.monotonic() - t0,
    }
    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_e2_e3_e4_lag_depth.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"e4": e4}, indent=2))
    print(f"\ngeschrieben: {out_path} ({result['_elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
