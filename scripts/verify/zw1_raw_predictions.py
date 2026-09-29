"""ZW1 -- raw moisture predictions of the LSTM (n_train=64) on the constant-run holdout.

Two arms, five seeds each, from the archive checkpoints (read-only, nothing is trained):

* ``excited``  -- ``lstm_seed{1..5}_n64.pt`` (commit 7e3fb1d8, the 600-epoch probe). The moisture
  head early-stopped before epoch 200 in all five seeds (80/182/53/99/65 epochs), so it equals the
  200-epoch network of Befund 51; the check against the Befund 51 record is part of the artifact.
  Seeds 1-3 are additionally scored from the local GA2 retrain (``lstm_ga2_seed*_n64.pt``).
* ``constant`` -- ``constant_trained_lstm_seed{1..5}_n64.pt`` (Befund 55, commit 1b85ce4c).

Per (arm, seed) one ``.npz`` (all 4352 holdout windows x 36 forecast steps): true (noise-free)
moisture, measured moisture (the reference of ``rmse_run_median_physical``), prediction, run index,
window start; run IDs in the JSON artifact. Step ``k`` (1..36) is column ``k-1``; "h=1" is column 0,
"h=36" all 36 columns pooled. The ``.npz`` files are git-ignored (size, see artifact); their SHA-256
is in the artifact and they are regenerated deterministically by this script.

    python scripts/verify/zw1_raw_predictions.py
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
from _common import (  # noqa: E402
    ARCHIVE_ROOT,
    REPO_ROOT,
    provenance,
    read_jsonl,
    require_clean_tree,
    write_artifact,
)
from _data import HOLDOUT_SEED, make_data  # noqa: E402
from pr3_metrics import make_windows, net_predict, rmse_run_median, train_scaler  # noqa: E402

SEEDS = (1, 2, 3, 4, 5)
MOIST = 1  # index of product_moisture_kg_per_kg in TARGETS
CKPT = ARCHIVE_ROOT / "artifacts" / "checkpoints"
LOCAL = REPO_ROOT / "artifacts" / "checkpoints"
RAW_DIR = REPO_ROOT / "artifacts" / "runs" / "raw_zw1"


def main() -> None:
    import torch

    require_clean_tree()
    torch.set_num_threads(1)
    RAW_DIR.mkdir(exist_ok=True)
    hold = make_data("holdout", HOLDOUT_SEED)
    test_ids = list(hold.split.test)
    records = {
        "excited": {
            r["params"]["master_seed"]: r
            for r in read_jsonl("arch_learning_curve_projected")
            if r["tags"]["model"] == "lstm"
            and r["params"]["n_train"] == 64
            and r["tags"]["test_set"] == "constant_holdout"
        },
        "constant": {
            r["params"]["master_seed"]: r
            for r in read_jsonl("arch_learning_curve_constant_trained")
            if r["tags"]["model"] == "lstm"
            and r["params"]["n_train"] == 64
            and r["tags"]["test_set"] == "constant_holdout"
        },
    }
    key = "rmse_run_median_physical/product_moisture_kg_per_kg/h36"
    sets: list[dict] = []
    for arm, data_kind, prefix in (
        ("excited", "excited", "lstm"),
        ("constant", "constant_train", "constant_trained_lstm"),
    ):
        for seed in SEEDS:
            data = make_data(data_kind, seed)
            w = make_windows(hold, hold.split.test, train_scaler(data, 64))
            paths = {"archive": CKPT / f"{prefix}_seed{seed}_n64.pt"}
            if arm == "excited" and seed <= 3:
                paths["ga2_local"] = LOCAL / f"lstm_ga2_seed{seed}_n64.pt"
            entry: dict = {"arm": arm, "seed": seed, "checks": {}}
            for source, path in paths.items():
                pred, meta = net_predict(path, w)
                score = float(rmse_run_median(pred, w["ref"], w["run"], 36)[MOIST])
                rec = records[arm][seed]["metrics"][key]
                entry["checks"][source] = {
                    "checkpoint": path.name,
                    "rmse_run_median_h36": score,
                    "record": rec,
                    "rel_diff": abs(score - rec) / rec,
                }
                if source == "archive":
                    main_pred = pred[..., MOIST]
            name = f"{arm}_seed{seed}"
            path_npz = RAW_DIR / f"{name}.npz"
            np.savez_compressed(
                path_npz,
                run=w["run"].astype(np.int16),
                start=w["start"].astype(np.int16),
                truth=w["truth"][..., MOIST],
                measured=w["ref"][..., MOIST],
                pred=main_pred,
            )
            entry["file"] = path_npz.name
            entry["bytes"] = path_npz.stat().st_size
            entry["sha256"] = hashlib.sha256(path_npz.read_bytes()).hexdigest()
            entry["n_windows"] = int(len(w["run"]))
            sets.append(entry)
            print(name, entry["checks"], flush=True)
    n_win = sets[0]["n_windows"]
    write_artifact(
        "verify_zw1_raw_predictions",
        {
            "_provenance": provenance(),
            "columns": "run, start, truth, measured, pred (float64, W x 36; step k = column k-1)",
            "holdout_test_run_ids": test_ids,
            "n_windows": n_win,
            "window_steps_per_set": n_win * 36,
            "total_bytes_npz": int(sum(s["bytes"] for s in sets)),
            "sets": sets,
        },
    )


if __name__ == "__main__":
    main()
