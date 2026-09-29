"""PR4 / Befund 39 -- window-wise vs run-wise split: same model, same data, only the split differs.

1. Recompute the abstract's "29 to 68 % lower error" from the committed records
   (``extra_trees_split_comparison.jsonl``) and check which fields differ between the two arms.
2. Independent re-run with the audit's own code: regenerate the data (process config with the
   projection OFF and the three targets of the record, history 3, horizon 1, stride 1), build
   both arms from raw arrays (own windowing, own scaler, own random window assignment), fit
   sklearn's ExtraTreesRegressor (100 trees, seed from the seed bundle) and score with a
   run-wise NRMSE median. Not bit-identical (window assignment is an independent random draw),
   so the comparison is of the medians over 5 seeds and their ranges.
3. Runtime estimate (NOT executed) for repeating the comparison with the LSTM.

    python scripts/verify/pr4_befund39.py
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from _common import provenance, read_jsonl, require_clean_tree, write_artifact
from _data import load_config

SEEDS = (1, 2, 3, 4, 5)
TARGETS3 = ("energy_target", "product_moisture_kg_per_kg", "d50_um")
OUT_COL = {"energy_target": 0, "product_moisture_kg_per_kg": 2, "d50_um": 3}
L, HORIZON = 3, 1


def records_summary() -> dict[str, Any]:
    recs = read_jsonl("extra_trees_split_comparison")
    by_arm = {
        arm: sorted(
            (r for r in recs if r["tags"]["split_kind"] == arm),
            key=lambda r: r["params"]["master_seed"],
        )
        for arm in ("run_level", "point_level")
    }
    params_a, params_b = by_arm["run_level"], by_arm["point_level"]
    differing = sorted(
        {
            k
            for a, b in zip(params_a, params_b, strict=True)
            for k in a["params"]
            if a["params"][k] != b["params"].get(k)
        }
    )
    out: dict[str, Any] = {"params_that_differ_between_arms": differing}
    ratios = {}
    for ch in TARGETS3:
        key = f"nrmse_run_median/{ch}/h1"
        run = [r["metrics"][key] for r in params_a]
        pt = [r["metrics"][key] for r in params_b]
        ratios[ch] = {
            "run_level_median": float(np.median(run)),
            "point_level_median": float(np.median(pt)),
            "ratio_point_over_run": float(np.median(pt) / np.median(run)),
            "reduction_percent": float(100 * (1 - np.median(pt) / np.median(run))),
            "run_level_range": [min(run), max(run)],
            "point_level_range": [min(pt), max(pt)],
            "ranges_disjoint": bool(max(pt) < min(run)),
        }
    out["from_records"] = ratios
    out["n_train_windows"] = {
        arm: [r["params"]["n_train_windows"] for r in v] for arm, v in by_arm.items()
    }
    out["model"] = {
        arm: sorted({r["params"]["model/model"] for r in v}) for arm, v in by_arm.items()
    }
    out["data_fingerprints_equal_between_arms"] = all(
        a["params"][k] == b["params"][k]
        for a, b in zip(params_a, params_b, strict=True)
        for k in (
            "process/fingerprint",
            "excitation/fingerprint",
            "split/digest",
            "seed/data",
            "seed/noise",
            "seed/model",
        )
    )
    out["record_target_names"] = params_a[0]["params"]["data/target_names"]
    out["record_git_commit"] = sorted({r["params"]["git/commit"] for r in recs})
    out["record_process_fingerprint"] = sorted({r["params"]["process/fingerprint"] for r in recs})
    return out


def windows(measure: np.ndarray, inputs: np.ndarray, run_ids: np.ndarray) -> tuple[np.ndarray, ...]:
    """History (3 steps of inputs|targets), one-step target, run label -- stride 1."""
    n_runs, n_steps, _ = inputs.shape
    hist, tgt, run = [], [], []
    for r in range(n_runs):
        starts = np.arange(0, n_steps - (L + HORIZON) + 1)
        hist.append(np.stack([inputs[r, s : s + L] for s in starts]))
        tgt.append(measure[r, starts + L])
        run.append(np.full(len(starts), run_ids[r]))
    return np.concatenate(hist), np.concatenate(tgt), np.concatenate(run)


def features(hist: np.ndarray) -> np.ndarray:
    current, previous, before = hist[:, -1], hist[:, -2], hist[:, -3]
    first = current - previous
    second = first - (previous - before)
    return np.concatenate([current, first, second], axis=-1)


def nrmse_run_median(pred: np.ndarray, ref: np.ndarray, run: np.ndarray) -> np.ndarray:
    values = []
    for r in np.unique(run):
        sel = run == r
        rmse = np.sqrt(np.mean((pred[sel] - ref[sel]) ** 2, axis=0))
        values.append(rmse / np.std(ref[sel], axis=0))
    return np.nanmedian(np.stack(values), axis=0)


def rerun_seed(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor

    from twinlink_ml.evaluation.experiments import DataSpec, generate_data

    spec = DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=L,
        horizon=HORIZON,
        stride=1,
        target_names=TARGETS3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = generate_data(load_config(projection=False), spec, master_seed=seed)
    t = data.trajectory
    pos = {rid: i for i, rid in enumerate(t.run_id)}
    train_rows = np.asarray([pos[r] for r in data.split.train])
    test_rows = np.asarray([pos[r] for r in data.split.test])
    cols = [OUT_COL[c] for c in TARGETS3]

    def arrays(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        observed = t.measurements[rows][..., cols]
        inputs = np.concatenate([t.actions[rows], t.disturbances[rows], observed], axis=-1)
        return inputs, observed

    def fit_predict(
        train: tuple[np.ndarray, ...],
        test: tuple[np.ndarray, ...],
        tscale: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        tmean, tstd = tscale
        model = ExtraTreesRegressor(n_estimators=100, random_state=data.seeds.model_seed, n_jobs=-1)
        model.fit(features(train[0]), (train[1] - tmean) / tstd)
        return model.predict(features(test[0])) * tstd + tmean

    # arm A: run-wise (64 train runs, 32 test runs), scaler statistics of the train runs
    tr_inputs, tr_observed = arrays(train_rows)
    te_inputs, te_observed = arrays(test_rows)
    train_a = windows(tr_observed, tr_inputs, np.arange(len(train_rows)))
    test_a = windows(te_observed, te_inputs, np.arange(len(test_rows)))
    flat = tr_observed.reshape(-1, 3)
    pred_a = fit_predict(train_a, test_a, (flat.mean(0), flat.std(0)))
    result_a = nrmse_run_median(pred_a, test_a[1], test_a[2])

    # arm B: window-wise over the same 96 runs, random 2:1 assignment of single windows
    pool_rows = np.concatenate([train_rows, test_rows])
    p_inputs, p_observed = arrays(pool_rows)
    hist, tgt, run = windows(p_observed, p_inputs, np.arange(len(pool_rows)))
    rng = np.random.default_rng(10_000 + seed)
    is_train = rng.random(len(run)) < 64 / 96
    tscale_b = (tgt[is_train].mean(0), tgt[is_train].std(0))
    pred_b = fit_predict(
        (hist[is_train], tgt[is_train], run[is_train]),
        (hist[~is_train], tgt[~is_train], run[~is_train]),
        tscale_b,
    )
    result_b = nrmse_run_median(pred_b, tgt[~is_train], run[~is_train])
    return {
        "seed": seed,
        "n_train_windows": {"run_level": len(train_a[2]), "point_level": int(is_train.sum())},
        "nrmse_run_median_h1": {
            ch: {"run_level": float(result_a[i]), "point_level": float(result_b[i])}
            for i, ch in enumerate(TARGETS3)
        },
    }


def lstm_time_estimate() -> dict[str, Any]:
    recs = read_jsonl("arch_learning_curve_projected")
    rows = [
        r
        for r in recs
        if r["tags"]["model"] == "lstm"
        and r["params"]["n_train"] == 64
        and r["tags"]["test_set"] == "excited"
    ]
    elapsed = float(np.median([r["params"]["training/elapsed_s"] for r in rows]))
    epochs = float(
        np.median(
            [
                np.mean(
                    [
                        r["params"][f"training/epochs_run/{c}"]
                        for c in (
                            "energy_target",
                            "product_moisture_kg_per_kg",
                            "bed_temperature_celsius",
                        )
                    ]
                )
                for r in rows
            ]
        )
    )
    steps = float(
        np.median(
            [
                np.mean(
                    [
                        r["params"][f"training/gradient_steps/{c}"]
                        for c in (
                            "energy_target",
                            "product_moisture_kg_per_kg",
                            "bed_temperature_celsius",
                        )
                    ]
                )
                for r in rows
            ]
        )
    )
    windows_n64, seq_ref, seq_new, windows_new, batch = 64 * 136, 32 + 36, 3 + 1, 73_536, 256
    heads = 3
    sec_per_step_ref = elapsed / (
        steps * heads
    )  # per gradient step, per head, at 68-step sequences
    per_window_step = elapsed / (epochs * heads * windows_n64 * seq_ref)
    new_steps_per_epoch = np.ceil(windows_new / batch)
    fit_lower = per_window_step * epochs * heads * windows_new * seq_new
    fit_upper = sec_per_step_ref * new_steps_per_epoch * epochs * heads
    return {
        "basis_records": "arch_learning_curve_projected, lstm, n_train=64 (Linux, 4 jobs in parallel), median over 5 seeds",
        "reference_fit_elapsed_s": elapsed,
        "reference_mean_epochs_per_head": epochs,
        "reference_gradient_steps_per_head": steps,
        "reference_seconds_per_gradient_step_per_head": sec_per_step_ref,
        "new_geometry": {
            "history": 3,
            "horizon": 1,
            "train_windows_per_fit": windows_new,
            "sequence_steps": seq_new,
            "gradient_steps_per_epoch": float(new_steps_per_epoch),
        },
        "one_fit_lower_bound_s_(cost_scales_with_sequence_length)": float(fit_lower),
        "one_fit_upper_bound_s_(cost_scales_with_gradient_steps_only)": float(fit_upper),
        "assumption": "same epochs as the n_train=64 excited fits (early stopping needs a validation block, which the ExtraTrees comparison does not have); 3 heads",
        "full_repetition_5_seeds_x_2_arms_sequential_h": [
            float(fit_lower * 10 / 3600),
            float(fit_upper * 10 / 3600),
        ],
        "full_repetition_with_4_parallel_jobs_h": [
            float(fit_lower * 10 / 3600 / 4 * 1.2),
            float(fit_upper * 10 / 3600 / 4 * 1.2),
        ],
        "not_executed": True,
    }


def main() -> None:
    require_clean_tree()
    import sklearn

    report: dict[str, Any] = {
        "_provenance": provenance(),
        "sklearn_version_(installed_isolated_via_PYTHONPATH,_absent_from_the_environment)": sklearn.__version__,
        "records": records_summary(),
    }
    rerun = [rerun_seed(s) for s in SEEDS]
    report["rerun_per_seed"] = rerun
    summary = {}
    for ch in TARGETS3:
        run = [r["nrmse_run_median_h1"][ch]["run_level"] for r in rerun]
        pt = [r["nrmse_run_median_h1"][ch]["point_level"] for r in rerun]
        summary[ch] = {
            "run_level_median": float(np.median(run)),
            "point_level_median": float(np.median(pt)),
            "ratio": float(np.median(pt) / np.median(run)),
            "reduction_percent": float(100 * (1 - np.median(pt) / np.median(run))),
            "run_level_range": [min(run), max(run)],
            "point_level_range": [min(pt), max(pt)],
            "ranges_disjoint": bool(max(pt) < min(run)),
        }
    report["rerun_summary"] = summary
    report["lstm_repetition_runtime_estimate"] = lstm_time_estimate()
    write_artifact("verify_pr4_befund39", report)
    print(
        {
            ch: (
                round(v["reduction_percent"], 1),
                report["records"]["from_records"][ch]["reduction_percent"],
            )
            for ch, v in summary.items()
        }
    )


if __name__ == "__main__":
    main()
