r"""L1: does the moisture finding (48) depend on the clipping of ``max_driving_humidity_kg_per_kg``.

NO network training -- pure OLS regression (``np.linalg.lstsq``), the same feature logic
as ``scripts/probe_e2_e3_e4_lag_depth.py`` (E3, FIR, k in {1, 8, 43}, point-wise
masking by the same criteria as Befund 46/D2/F1). NO change to ``src/``.

Background: Befund 48 shows for ``product_moisture_kg_per_kg`` at k=43 a large
R2_holdout jump through masking of the violation points (0.354 -> 0.785, caveat:
comparison across two samples). ``max_driving_humidity_kg_per_kg`` (default 0.50
kg/kg, config ``material``, see Befund 46) limits the evaporation rate
(``thermo.evaporation_rate_kg_per_s``, ``y_effective = min(y_saturation, max_driving_
humidity)``) and is validated nowhere. Open question: is the R2 jump an
artifact of exactly this clipping, or does it persist when the clipping hardly
binds any more?

Procedure: the same data set/the same masking code as E3, but ``material.
max_driving_humidity_kg_per_kg`` varied in {0.50, 1.00, 5.00} (0.50 = current
default, 5.00 = practically never binding, Y_sat(T) < 5.0 kg/kg in the entire reachable
range). In addition, per configuration:

* Clipping fraction: fraction of the raw trajectory points (3 training-pool seeds + holdout,
  before the FIR windowing) with ``Y_sat(T_bed) > max_driving_humidity`` -- the condition
  under which ``evaporation_rate_kg_per_s`` actually sits at the kink of the minimum.
* Fraction ``bed_temperature_celsius > 78.8`` C -- fixed physical threshold
  (Y_sat(78.8 C) ~= 0.50 kg/kg at 101325 Pa, see ``thermo.saturation_humidity_kg_
  per_kg``), independent of the config parameter. Coincides approximately with the clipping
  fraction at 0.50 kg/kg (the same boundary), but drops away from the clipping threshold
  at 1.00/5.00 kg/kg, while the fraction of points above 78.8 C itself stays nearly
  constant (the threshold is a property of the excitation/dynamics, not a
  function of the clipping parameter).
* R2_holdout per target channel (``energy_target``, ``bed_temperature_celsius``,
  ``product_moisture_kg_per_kg``, ``d50_um``) at k in {1, 8, 43}, masked and
  unmasked (median over 3 seeds), as well as Var(y) and residual variance on the holdout in
  physical units -- checkable against Befund 47/48 for the default case 0.50.

AB2/AC1 extension (docs/04_stand_ap1.md, Befund 47/50): originally only
``product_moisture_kg_per_kg`` (Befund 48). Split mechanics, feature set
(``_fir_design``) and point selection (``_violation_mask``) UNCHANGED -- the
FIR design matrix depends only on the actions/disturbances, not on the
target channel, and is therefore built ONCE per ``k``/configuration/masking level and
reused across all four channels (no second recomputation pass, only one
additional ``lstsq``/evaluation per channel). The ``product_moisture_kg_per_kg`` numbers
must therefore stay bit-identical to the previous, committed state -- this is a
built-in cross-check of the extension itself, not just an expectation.
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
from twinlink_ml.simulation.config import MaterialConfig, ProcessConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
HOLDOUT_SEED = 999
TRAIN_SEEDS = (1, 2, 3)
K_VALUES_E3 = (1, 8, 43)
MAX_DRIVING_HUMIDITY_VALUES = (0.50, 1.00, 5.00)
BED_TEMPERATURE_THRESHOLD_CELSIUS = 78.8
TARGET_CHANNELS = (
    "energy_target",
    "bed_temperature_celsius",
    "product_moisture_kg_per_kg",
    "d50_um",
)


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
    """Pointwise violation mask (d50 basis, criteria as in D2/F1/Befund 46).

    Shape ``(n_runs, n_steps)``. Unchanged compared with
    ``probe_e2_e3_e4_lag_depth.py`` -- does not depend on
    ``max_driving_humidity_kg_per_kg``, so that the masking stays comparable across all three
    configurations of this script.
    """
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
    """FIR feature matrix, unchanged compared with ``probe_e2_e3_e4_lag_depth.py``."""
    n_runs, n_steps, _n_actions = actions.shape
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


def _r2_and_variances(
    design: np.ndarray, target: np.ndarray, coefficients: np.ndarray
) -> tuple[float, float, float]:
    variance = float(np.var(target))
    residual = target - design @ coefficients
    residual_variance = float(np.var(residual))
    r2 = float("nan") if variance <= 0.0 else 1.0 - residual_variance / variance
    return r2, variance, residual_variance


def _load_config(max_driving_humidity: float) -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(CONFIG_PATH)
    return dataclasses.replace(
        base,
        material=dataclasses.replace(
            base.material, max_driving_humidity_kg_per_kg=max_driving_humidity
        ),
    )


def _clip_and_threshold_fractions(
    slices: list[tuple[np.ndarray, np.ndarray]], *, max_driving_humidity: float
) -> tuple[float, float]:
    """Clipping share and share of ``bed_temperature_celsius`` > threshold, over all points.

    ``slices`` are ``(y_saturation, bed_temperature_celsius)`` arrays, already cut to
    exactly the points that also enter the FIR regression (holdout
    test block ``[4:36]``, training pool blocks ``[:64]``) -- not the full generated
    pool (which with ``excited_spec`` also contains unused val/test runs).
    """
    clip_hits = 0
    threshold_hits = 0
    total = 0
    for y_saturation, bed_temp in slices:
        clip_hits += int(np.sum(y_saturation > max_driving_humidity))
        threshold_hits += int(np.sum(bed_temp > BED_TEMPERATURE_THRESHOLD_CELSIUS))
        total += y_saturation.size
    return clip_hits / total, threshold_hits / total


def run_for_max_driving_humidity(max_driving_humidity: float) -> dict:
    config = _load_config(max_driving_humidity)
    excited_spec = _excited_spec()
    holdout_spec = _holdout_spec(excited_spec)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        holdout = generate_data(config, holdout_spec, master_seed=HOLDOUT_SEED)
    holdout_actions = holdout.trajectory.actions[4:36]
    holdout_outputs = holdout.trajectory.outputs[4:36]
    holdout_point_mask = _violation_mask(holdout.trajectory, config.material)[4:36]
    channels = holdout.trajectory.metadata["output_names"]
    target_indices = {name: channels.index(name) for name in TARGET_CHANNELS}
    bed_temp_idx = channels.index("bed_temperature_celsius")

    slices: list[tuple[np.ndarray, np.ndarray]] = [
        (
            holdout.trajectory.diagnostics["saturation_humidity_kg_per_kg"][4:36],
            holdout_outputs[:, :, bed_temp_idx],
        )
    ]

    pools: dict[int, dict] = {}
    for seed in TRAIN_SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            excited = generate_data(config, excited_spec, master_seed=seed)
        train_actions = excited.trajectory.actions[:64]
        train_outputs = excited.trajectory.outputs[:64]
        train_point_mask = _violation_mask(excited.trajectory, config.material)[:64]
        pools[seed] = {
            "train_actions": train_actions,
            "train_outputs": train_outputs,
            "train_point_mask": train_point_mask,
        }
        slices.append(
            (
                excited.trajectory.diagnostics["saturation_humidity_kg_per_kg"][:64],
                train_outputs[:, :, bed_temp_idx],
            )
        )

    clip_fraction, threshold_fraction = _clip_and_threshold_fractions(
        slices, max_driving_humidity=max_driving_humidity
    )

    # A separate list per channel (result) -- the design matrices below are
    # channel-independent (depend only on actions/disturbances) and are built exactly ONCE
    # per k/seed/masking level, then reused across all four channels
    # (module docstring, AB2/AC1 extension).
    unmasked_by_channel: dict[str, list[dict[str, float]]] = {c: [] for c in TARGET_CHANNELS}
    masked_by_channel: dict[str, list[dict[str, float]]] = {c: [] for c in TARGET_CHANNELS}
    for k in K_VALUES_E3:
        design_ho_unmasked, run_idx_ho_u, t_idx_ho_u = _fir_design(holdout_actions, k)
        design_ho_masked, run_idx_ho_m, t_idx_ho_m = _fir_design(
            holdout_actions, k, point_mask=holdout_point_mask
        )

        train_designs: dict[int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for seed in TRAIN_SEEDS:
            pool = pools[seed]
            train_designs[seed] = {
                "unmasked": _fir_design(pool["train_actions"], k),
                "masked": _fir_design(
                    pool["train_actions"], k, point_mask=pool["train_point_mask"]
                ),
            }

        for channel in TARGET_CHANNELS:
            target_idx = target_indices[channel]
            targets_ho_unmasked = holdout_outputs[run_idx_ho_u, t_idx_ho_u, target_idx]
            targets_ho_masked = holdout_outputs[run_idx_ho_m, t_idx_ho_m, target_idx]

            r2_unmasked, r2_masked = [], []
            var_unmasked, resid_var_unmasked = [], []
            var_masked, resid_var_masked = [], []
            for seed in TRAIN_SEEDS:
                pool = pools[seed]
                design_tr_unmasked, run_idx_tr_u, t_idx_tr_u = train_designs[seed]["unmasked"]
                targets_tr_unmasked = pool["train_outputs"][run_idx_tr_u, t_idx_tr_u, target_idx]
                coeff_unmasked, *_ = np.linalg.lstsq(
                    design_tr_unmasked, targets_tr_unmasked, rcond=None
                )
                r2_ho, var_ho, resid_var_ho = _r2_and_variances(
                    design_ho_unmasked, targets_ho_unmasked, coeff_unmasked
                )
                r2_unmasked.append(r2_ho)
                var_unmasked.append(var_ho)
                resid_var_unmasked.append(resid_var_ho)

                design_tr_masked, run_idx_tr_m, t_idx_tr_m = train_designs[seed]["masked"]
                targets_tr_masked = pool["train_outputs"][run_idx_tr_m, t_idx_tr_m, target_idx]
                coeff_masked, *_ = np.linalg.lstsq(design_tr_masked, targets_tr_masked, rcond=None)
                r2_ho_m, var_ho_m, resid_var_ho_m = _r2_and_variances(
                    design_ho_masked, targets_ho_masked, coeff_masked
                )
                r2_masked.append(r2_ho_m)
                var_masked.append(var_ho_m)
                resid_var_masked.append(resid_var_ho_m)

            unmasked_by_channel[channel].append(
                {
                    "k": k,
                    "r2_holdout_median": float(np.median(r2_unmasked)),
                    "var_y_median": float(np.median(var_unmasked)),
                    "var_y_min": float(np.min(var_unmasked)),
                    "var_y_max": float(np.max(var_unmasked)),
                    "residual_var_median": float(np.median(resid_var_unmasked)),
                    "residual_var_min": float(np.min(resid_var_unmasked)),
                    "residual_var_max": float(np.max(resid_var_unmasked)),
                }
            )
            masked_by_channel[channel].append(
                {
                    "k": k,
                    "r2_holdout_median": float(np.median(r2_masked)),
                    "var_y_median": float(np.median(var_masked)),
                    "var_y_min": float(np.min(var_masked)),
                    "var_y_max": float(np.max(var_masked)),
                    "residual_var_median": float(np.median(resid_var_masked)),
                    "residual_var_min": float(np.min(resid_var_masked)),
                    "residual_var_max": float(np.max(resid_var_masked)),
                }
            )

    channels_result: dict[str, dict] = {}
    for channel in TARGET_CHANNELS:
        unmasked_by_k = unmasked_by_channel[channel]
        masked_by_k = masked_by_channel[channel]
        k43_unmasked = next(e for e in unmasked_by_k if e["k"] == 43)
        k43_masked = next(e for e in masked_by_k if e["k"] == 43)
        channels_result[channel] = {
            "unmasked_by_k": unmasked_by_k,
            "masked_by_k": masked_by_k,
            "k43_r2_holdout_diff_masked_minus_unmasked": (
                k43_masked["r2_holdout_median"] - k43_unmasked["r2_holdout_median"]
            ),
            "k43_var_y_unmasked": k43_unmasked["var_y_median"],
            "k43_var_y_masked": k43_masked["var_y_median"],
            "k43_residual_var_unmasked": k43_unmasked["residual_var_median"],
            "k43_residual_var_masked": k43_masked["residual_var_median"],
        }

    return {
        "max_driving_humidity_kg_per_kg": max_driving_humidity,
        "fingerprint": config.fingerprint,
        "clip_fraction": clip_fraction,
        "bed_temperature_above_78_8_fraction": threshold_fraction,
        "channels": channels_result,
    }


def main() -> None:
    t0 = time.monotonic()
    results = [run_for_max_driving_humidity(value) for value in MAX_DRIVING_HUMIDITY_VALUES]
    for entry in results:
        print(
            f"max_driving_humidity={entry['max_driving_humidity_kg_per_kg']:.2f}: "
            f"Klippung={entry['clip_fraction']:.4%}, "
            f"T_bed>78.8C={entry['bed_temperature_above_78_8_fraction']:.4%}"
        )
        for channel, result in entry["channels"].items():
            print(
                f"  {channel:28s} dR2(k=43)="
                f"{result['k43_r2_holdout_diff_masked_minus_unmasked']:+.3f}  "
                f"var_y(unmasked, k=43)={result['k43_var_y_unmasked']:.4g}  "
                f"residual_var(unmasked, k=43)={result['k43_residual_var_unmasked']:.4g}"
            )
    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_l1_clip_sensitivity.json"
    out_path.write_text(
        json.dumps({"results": results, "_elapsed_s": time.monotonic() - t0}, indent=2),
        encoding="utf-8",
    )
    print(f"geschrieben: {out_path} ({time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main()
