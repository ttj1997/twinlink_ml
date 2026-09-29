"""PR4 -- retrain the excited-trained LSTM (n_train=64, 200 epochs) for one seed and score it.

Closes the gap for the 0.00054 kg/kg figure: the archive holds no 200-epoch LSTM checkpoint at
n_train=64 (overwritten by the 600-epoch probe); GA2 retrained seeds 1-3 locally. This retrains one
seed (protocol of Befund 51: composite LSTM, FIXED_POOL validation, model seed from the seed
bundle, one thread) and scores it with the audit's own metric on both test sets.

    python scripts/verify/pr4_retrain_lstm.py --seed 4
"""

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import time  # noqa: E402

import numpy as np  # noqa: E402
from _common import (  # noqa: E402
    TARGETS,
    provenance,
    read_jsonl,
    require_clean_tree,
    write_artifact,
)
from _data import HOLDOUT_SEED, make_data  # noqa: E402
from pr3_metrics import make_windows, rmse_run_median, train_scaler  # noqa: E402


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    seed = parser.parse_args().seed
    require_clean_tree()
    torch.set_num_threads(1)
    from twinlink_ml.data.windows import WindowBatch
    from twinlink_ml.evaluation.experiments import build_datasets
    from twinlink_ml.models.composite import CompositeLstmSequenceModel

    data = make_data("excited", seed)
    hold = make_data("holdout", HOLDOUT_SEED)
    train, val, _ = build_datasets(data, 64)
    model = CompositeLstmSequenceModel()
    start = time.perf_counter()
    model.fit(train, seed=data.seeds.model_seed, validation=val)
    elapsed = time.perf_counter() - start
    sc = train_scaler(data, 64)
    records = {
        (r["params"]["master_seed"], r["tags"]["test_set"]): r
        for r in read_jsonl("arch_learning_curve_projected")
        if r["tags"]["model"] == "lstm" and r["params"]["n_train"] == 64
    }
    result = {}
    for test_set, src in (("excited", data), ("constant_holdout", hold)):
        w = make_windows(src, src.split.test, sc)
        preds = []
        for a in range(0, len(w["run"]), 1024):
            s = slice(a, a + 1024)
            preds.append(
                model.predict(
                    WindowBatch(
                        history_inputs=w["hist"][s],
                        future_exogenous=w["fut_exo"][s],
                        future_targets=(w["ref"][s] - w["tmean"]) / w["tscale"],
                        run_index=w["run"][s].astype(np.float64),
                        window_start=w["start"][s].astype(np.float64),
                    )
                )
            )
        pred = np.concatenate(preds) * w["tscale"] + w["tmean"]
        v = rmse_run_median(pred, w["ref"], w["run"], 36)
        rec = records[(seed, test_set)]["metrics"]
        result[test_set] = {
            ch: {
                "retrained": float(v[i]),
                "record_linux": rec[f"rmse_run_median_physical/{ch}/h36"],
                "rel_diff": float(
                    abs(v[i] - rec[f"rmse_run_median_physical/{ch}/h36"])
                    / rec[f"rmse_run_median_physical/{ch}/h36"]
                ),
            }
            for i, ch in enumerate(TARGETS)
        }
    rec_train = records[(seed, "excited")]["params"]
    write_artifact(
        f"verify_pr4_retrain_lstm_seed{seed}",
        {
            "_provenance": provenance(),
            "seed": seed,
            "elapsed_s": elapsed,
            "record_elapsed_s_linux": rec_train["training/elapsed_s"],
            "epochs_run_retrained": {
                k: v for k, v in model.training_diagnostics().items() if k.startswith("epochs_run/")
            },
            "epochs_run_record": {ch: rec_train[f"training/epochs_run/{ch}"] for ch in TARGETS},
            "rmse_run_median_physical_h36_n64": result,
        },
    )


if __name__ == "__main__":
    main()
