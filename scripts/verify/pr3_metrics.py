"""PR3 -- metrics and baselines recomputed from raw arrays.

Independent implementation (no import of ``twinlink_ml.evaluation`` evaluator/protocol/resolution
or of any ``scripts/analyze_*``):

* windows cut by the audit from the regenerated trajectories (history 32, horizon 36, stride 8),
  standardised with training-only statistics computed by the audit;
* **persistence** from the raw last measurement, **static regression** as a ridge fit of the audit
  (quadratic features of the future exogenous inputs, lambda 1e-6, intercept unpenalised, row
  thinning as documented in the baselines docstring), no import of the baseline classes;
* **neural models**: the original weights are loaded read-only from the archive checkpoints
  (``load_checkpoint``, i.e. the model classes are used, the evaluation is the audit's own):
  predictions on the audit's windows, back-transformation with the audit's scalers, per-run RMSE
  over the pooled forecast steps 1..h, median over runs -- compared with the records;
* R^2 with one estimator (pooled, 1 - SSE/SST in physical units) for static, persistence and the
  networks on the same windows, next to the in-sample estimator behind the quoted R^2_static;
* noise floor and regime shares (moisture above critical, bed temperature above 80 degC,
  projection active) on the forecast steps of the two test sets.

    python scripts/verify/pr3_metrics.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import numpy as np
from _common import (
    ARCHIVE_ROOT,
    TARGETS,
    provenance,
    read_jsonl,
    require_clean_tree,
    write_artifact,
)
from _data import HOLDOUT_SEED, make_data

SEEDS = (1, 2, 3, 4, 5)
BUDGETS = (8, 16, 32, 64)
L, H, STRIDE = 32, 36, 8
COLS = [0, 2, 1]  # output columns for TARGETS = (energy, moisture, bed temperature)
CKPT = ARCHIVE_ROOT / "artifacts" / "checkpoints"
CKPT_EXT = ARCHIVE_ROOT / "artifacts" / "checkpoints_arch_learning_curve_extended"
LOCAL_CKPT = Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints"
CRITICAL_MOISTURE = 0.2


# --------------------------------------------------------------------------- data windows
def train_scaler(data: Any, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = {rid: i for i, rid in enumerate(data.trajectory.run_id)}
    rows = np.asarray([pos[r] for r in data.split.subset(n).train])
    t = data.trajectory
    observed = t.measurements[rows][..., COLS]
    inputs = np.concatenate([t.actions[rows], t.disturbances[rows], observed], axis=-1)
    flat = inputs.reshape(-1, inputs.shape[-1])
    tflat = observed.reshape(-1, observed.shape[-1])
    return (
        flat.mean(0),
        np.maximum(flat.std(0), 1e-12),
        tflat.mean(0),
        np.maximum(tflat.std(0), 1e-12),
    )


def make_windows(data: Any, ids: tuple[str, ...], scaler: tuple[np.ndarray, ...]) -> dict[str, Any]:
    mean, scale, tmean, tscale = scaler
    pos = {rid: i for i, rid in enumerate(data.trajectory.run_id)}
    rows = np.asarray([pos[r] for r in ids])
    t = data.trajectory
    observed = t.measurements[rows][..., COLS]
    truth = t.outputs[rows][..., COLS]
    inputs = np.concatenate([t.actions[rows], t.disturbances[rows], observed], axis=-1)
    std = (inputs - mean) / scale
    n_steps = inputs.shape[1]
    starts = np.arange(0, n_steps - (L + H) + 1, STRIDE)
    idx = [(r, s) for r in range(len(rows)) for s in starts]
    run = np.asarray([r for r, _ in idx])
    projected = None
    if t.requested_actions is not None:
        projected = (t.actions[rows] != t.requested_actions[rows]).any(axis=-1)
    moisture = t.outputs[rows][..., 2]
    tbed = t.outputs[rows][..., 1]
    return {
        "hist": np.stack([std[r, s : s + L] for r, s in idx]),
        "fut_exo": np.stack([std[r, s + L : s + L + H, :7] for r, s in idx]),
        "ref": np.stack([observed[r, s + L : s + L + H] for r, s in idx]),
        "truth": np.stack([truth[r, s + L : s + L + H] for r, s in idx]),
        "last": np.stack([observed[r, s + L - 1] for r, s in idx]),
        "run": run,
        "start": np.asarray([s for _, s in idx]),
        "tmean": tmean,
        "tscale": tscale,
        "moisture_fut": np.stack([moisture[r, s + L : s + L + H] for r, s in idx]),
        "tbed_fut": np.stack([tbed[r, s + L : s + L + H] for r, s in idx]),
        "projected_fut": None
        if projected is None
        else np.stack([projected[r, s + L : s + L + H] for r, s in idx]),
    }


# ------------------------------------------------------------------------------- metrics
def rmse_run_median(pred: np.ndarray, ref: np.ndarray, run: np.ndarray, h: int) -> np.ndarray:
    per_run = [
        np.sqrt(np.mean((pred[run == r, :h] - ref[run == r, :h]) ** 2, axis=(0, 1)))
        for r in np.unique(run)
    ]
    return np.median(np.stack(per_run), axis=0)


def r2_pooled(pred: np.ndarray, ref: np.ndarray, h: int) -> np.ndarray:
    p, y = pred[:, :h], ref[:, :h]
    sse = ((y - p) ** 2).sum(axis=(0, 1))
    sst = ((y - y.mean(axis=(0, 1))) ** 2).sum(axis=(0, 1))
    return 1.0 - sse / sst


# ------------------------------------------------------------------------------ baselines
def persistence(w: dict[str, Any]) -> np.ndarray:
    return np.repeat(w["last"][:, None, :], H, axis=1)


def static_predict(
    train_w: dict[str, Any], test_w: dict[str, Any], *, thin: bool = True
) -> np.ndarray:
    def features(exo: np.ndarray) -> np.ndarray:
        flat = exo.reshape(-1, exo.shape[-1])
        return np.concatenate([flat, flat**2, np.ones((flat.shape[0], 1))], axis=-1)

    design = features(train_w["fut_exo"])
    target = ((train_w["ref"] - train_w["tmean"]) / train_w["tscale"]).reshape(-1, 3)
    if thin and design.shape[0] > 200_000:
        step = int(np.ceil(design.shape[0] / 200_000))
        design, target = design[::step], target[::step]
    penalty = np.eye(design.shape[1]) * 1e-6
    penalty[-1, -1] = 0.0
    theta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    pred_std = (features(test_w["fut_exo"]) @ theta).reshape(-1, H, 3)
    return pred_std * test_w["tscale"] + test_w["tmean"]


def net_predict(path: Path, w: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Predictions of a stored network on the audit's windows, physical units."""
    from twinlink_ml.data.windows import WindowBatch
    from twinlink_ml.utils.checkpoints import load_checkpoint

    model, meta = load_checkpoint(path)
    outputs = []
    for a in range(0, len(w["run"]), 1024):
        sl = slice(a, a + 1024)
        batch = WindowBatch(
            history_inputs=w["hist"][sl],
            future_exogenous=w["fut_exo"][sl],
            future_targets=(w["ref"][sl] - w["tmean"]) / w["tscale"],
            run_index=w["run"][sl].astype(np.float64),
            window_start=w["start"][sl].astype(np.float64),
        )
        outputs.append(model.predict(batch))
    pred_std = np.concatenate(outputs, axis=0)
    return pred_std * w["tscale"] + w["tmean"], meta


# ------------------------------------------------------------------------------- records
def record_table() -> dict[tuple[str, str, int, int, str], dict[str, Any]]:
    """(file, model, seed, n_train, test_set) -> record."""
    table: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    for name in (
        "arch_learning_curve_projected",
        "arch_learning_curve_projected_extended",
        "arch_learning_curve_constant_trained",
        "arch_epoch_budget_probe600",
    ):
        for r in read_jsonl(name):
            key = (
                name,
                r["tags"]["model"],
                r["params"]["master_seed"],
                r["params"]["n_train"],
                r["tags"]["test_set"],
            )
            table[key] = r
    return table


def compare_cell(
    cells: list[dict[str, Any]],
    group: str,
    label: str,
    computed: np.ndarray,
    record: dict[str, Any],
    test_set: str,
) -> None:
    for j, name in enumerate(TARGETS):
        for h in (1, H):
            rec = record["metrics"][f"rmse_run_median_physical/{name}/h{h}"]
            cells.append(
                {
                    "group": group,
                    "label": label,
                    "test_set": test_set,
                    "channel": name,
                    "h": h,
                    "record": rec,
                    "audit": float(computed[h][j]),
                    "rel_diff": float(abs(computed[h][j] - rec) / abs(rec)) if rec else None,
                }
            )


def summarise(cells: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = collections.defaultdict(list)
    for c in cells:
        if c["rel_diff"] is not None:
            groups[c["group"]].append(c["rel_diff"])
    return {
        g: {
            "n_cells": len(v),
            "max_rel_diff": float(np.max(v)),
            "median_rel_diff": float(np.median(v)),
            "n_cells_above_1e-3": int(np.sum(np.asarray(v) > 1e-3)),
            "n_cells_above_1e-2": int(np.sum(np.asarray(v) > 1e-2)),
        }
        for g, v in groups.items()
    }


def main() -> None:
    import torch

    require_clean_tree()
    torch.set_num_threads(4)
    records = record_table()
    pools = {("excited", s): make_data("excited", s) for s in SEEDS}
    pools.update({("constant_train", s): make_data("constant_train", s) for s in SEEDS})
    hold = make_data("holdout", HOLDOUT_SEED)
    cells: list[dict[str, Any]] = []
    r2_rows: list[dict[str, Any]] = []
    regime: dict[str, Any] = {}

    # windows cache: per (arm, seed, n) train scaler; test windows depend only on the scaler
    def windows(arm: str, seed: int, n: int) -> dict[str, dict[str, Any]]:
        data = pools[(arm, seed)]
        sc = train_scaler(data, n)
        return {
            "train": make_windows(data, data.split.subset(n).train, sc),
            "own_test": make_windows(data, data.split.test, sc),
            "holdout": make_windows(hold, hold.split.test, sc),
        }

    # ---------------- baselines: persistence + static, all seeds, budgets, both arms ------------
    for arm, own_label, file_name in (
        ("excited", "excited", "arch_learning_curve_projected"),
        ("constant_train", "own_pool", "arch_learning_curve_constant_trained"),
    ):
        for seed in SEEDS:
            for n in BUDGETS:
                w = windows(arm, seed, n)
                for test_set, tw in (
                    (own_label, w["own_test"]),
                    ("constant_holdout", w["holdout"]),
                ):
                    for model, pred in (
                        ("persistence", persistence(tw)),
                        ("static", static_predict(w["train"], tw)),
                    ):
                        computed = {
                            h: rmse_run_median(pred, tw["ref"], tw["run"], h) for h in (1, H)
                        }
                        compare_cell(
                            cells,
                            f"{model}/{arm}",
                            f"{model} seed{seed} n{n}",
                            computed,
                            records[(file_name, model, seed, n, test_set)],
                            test_set,
                        )
                    if n == 64:
                        for model, pred in (
                            ("persistence", persistence(tw)),
                            ("static", static_predict(w["train"], tw)),
                        ):
                            for ref_name, ref in (
                                ("measurement", tw["ref"]),
                                ("truth", tw["truth"]),
                            ):
                                r2 = r2_pooled(pred, ref, H)
                                r2_rows.append(
                                    {
                                        "arm": arm,
                                        "seed": seed,
                                        "test_set": test_set,
                                        "model": model,
                                        "reference": ref_name,
                                        "h": H,
                                        "r2": dict(zip(TARGETS, map(float, r2), strict=False)),
                                    }
                                )
            print("baselines", arm, seed, flush=True)

    # ---------------- networks from checkpoints (read-only archive) ----------------------------
    def run_ckpt(
        path: Path,
        arm: str,
        seed: int,
        n: int,
        model_label: str,
        file_name: str,
        group: str,
        own_label: str,
    ) -> None:
        w = windows(arm, seed, n)
        for test_set, tw in ((own_label, w["own_test"]), ("constant_holdout", w["holdout"])):
            pred, _ = net_predict(path, tw)
            computed = {h: rmse_run_median(pred, tw["ref"], tw["run"], h) for h in (1, H)}
            key = (file_name, model_label, seed, n, test_set)
            if key in records:
                compare_cell(cells, group, f"{path.stem}", computed, records[key], test_set)
            if n == 64:
                for ref_name, ref in (("measurement", tw["ref"]), ("truth", tw["truth"])):
                    r2 = r2_pooled(pred, ref, H)
                    r2_rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "test_set": test_set,
                            "model": f"{group}:{model_label}",
                            "reference": ref_name,
                            "h": H,
                            "r2": dict(zip(TARGETS, map(float, r2), strict=False)),
                        }
                    )

    import torch as _torch

    skipped: list[str] = []
    for path in sorted(CKPT.glob("*_seed?_n*.pt")):
        stem = path.stem
        constant = stem.startswith("constant_trained_")
        core = stem.removeprefix("constant_trained_")
        arch, seed_part, n_part = core.split("_")
        seed, n = int(seed_part[4:]), int(n_part[1:])
        commit = _torch.load(path, weights_only=False)["git_commit"][:8]
        if constant and commit == "1b85ce4c":
            run_ckpt(
                path,
                "constant_train",
                seed,
                n,
                arch,
                "arch_learning_curve_constant_trained",
                "constant_trained_gru_lstm",
                "own_pool",
            )
        elif not constant and commit == "0b3eaafc" and arch in ("gru", "lstm"):
            run_ckpt(
                path,
                "excited",
                seed,
                n,
                arch,
                "arch_learning_curve_projected",
                "excited_gru_lstm_linux_ckpt",
                "excited",
            )
        elif not constant and commit == "7e3fb1d8" and arch in ("lstm", "transformer"):
            run_ckpt(
                path,
                "excited",
                seed,
                n,
                arch,
                "arch_epoch_budget_probe600",
                "probe600_600_epochs",
                "excited",
            )
        else:
            skipped.append(f"{stem} (commit {commit})")
        print("ckpt", stem, flush=True)
    for path in sorted(CKPT_EXT.glob("*_seed?_n*.pt")):
        arch, seed_part, n_part = path.stem.split("_")
        seed, n = int(seed_part[4:]), int(n_part[1:])
        if n in (64,) or (seed == 1 and n == 8):
            run_ckpt(
                path,
                "excited",
                seed,
                n,
                arch,
                "arch_learning_curve_projected_extended",
                "extended_transformer_xlstm_mamba",
                "excited",
            )
            print("ckpt ext", path.stem, flush=True)
    for s in (1, 2, 3):
        path = LOCAL_CKPT / f"lstm_ga2_seed{s}_n64.pt"
        if path.exists():
            run_ckpt(
                path,
                "excited",
                s,
                64,
                "lstm",
                "arch_learning_curve_projected",
                "ga2_lstm_local_retrain_vs_linux_record",
                "excited",
            )

    # ---------------- in-sample estimator behind the quoted R2_static ---------------------------
    def in_sample_r2(data: Any) -> dict[str, float]:
        t = data.trajectory
        n_runs, n_steps, _ = t.actions.shape
        x = np.concatenate([t.actions, t.disturbances], axis=-1).reshape(n_runs * n_steps, -1)
        design = np.concatenate([x, x**2, np.ones((x.shape[0], 1))], axis=-1)
        out = {}
        for name, col in zip(TARGETS, COLS, strict=True):
            y = t.outputs[..., col].reshape(-1)
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            out[name] = float(1.0 - np.var(y - design @ beta) / np.var(y))
        return out

    in_sample = {f"excited_seed{s}": in_sample_r2(pools[("excited", s)]) for s in SEEDS}
    in_sample["constant_holdout"] = in_sample_r2(hold)

    # ---------------- noise floor and regime shares on the forecast steps ---------------------
    for arm_key, source, ids_fn in (
        ("excited_test", lambda s: pools[("excited", s)], lambda d: d.split.test),
        ("constant_holdout_test", lambda _s: hold, lambda d: d.split.test),
    ):
        rows = []
        for s in SEEDS if arm_key == "excited_test" else (1,):
            d = source(s)
            tw = make_windows(d, ids_fn(d), train_scaler(pools[("excited", s)], 64))
            per_run_noise = [
                np.sqrt(
                    np.mean(
                        (tw["ref"][tw["run"] == r, :, :] - tw["truth"][tw["run"] == r, :, :]) ** 2,
                        axis=(0, 1),
                    )
                )
                for r in np.unique(tw["run"])
            ]
            rows.append(
                {
                    "seed": s,
                    "share_forecast_steps_moisture_above_critical": float(
                        (tw["moisture_fut"] > CRITICAL_MOISTURE).mean()
                    ),
                    "share_forecast_steps_bed_temperature_above_80": float(
                        (tw["tbed_fut"] > 80).mean()
                    ),
                    "share_forecast_steps_projection_active": float(tw["projected_fut"].mean()),
                    "noise_floor_rmse_measurement_vs_truth_run_median": dict(
                        zip(TARGETS, map(float, np.median(per_run_noise, axis=0)), strict=False)
                    ),
                    "moisture_truth_median_kg_per_kg": float(np.median(tw["moisture_fut"])),
                }
            )
        regime[arm_key] = rows

    report = {
        "_provenance": provenance(),
        "summary_rmse_run_median_physical_vs_records": summarise(cells),
        "checkpoints_skipped_(commit_differs_from_record_run)": skipped,
        "cells": cells,
        "r2_rows": r2_rows,
        "r2_in_sample_estimator_of_quoted_R2_static": in_sample,
        "regime_and_noise_floor_on_test_forecast_steps": regime,
    }
    write_artifact("verify_pr3_metrics", report)
    print(json.dumps(report["summary_rmse_run_median_physical_vs_records"], indent=1))


if __name__ == "__main__":
    main()
