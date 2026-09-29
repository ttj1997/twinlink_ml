"""PR4 -- inference and training cost of LSTM, GRU and Transformer on ONE machine.

The recorded numbers (183 us LSTM, 323 us GRU, ~3.1 ms Transformer; training ~10x) come from
different machines (Linux for GRU/LSTM, Windows for the Transformer) and, on Linux, from four
jobs running in parallel. Here: this machine, one thread (as the records' ``torch/num_threads``),
2 warm-up calls, batch size 4096 (the evaluator's), 10 repeats of the full 4352-window excited
test set of seed 1; networks loaded read-only from the archive (weights do not change the cost).
Training cost: 2 epochs (patience large) per architecture on the n_train=64 data, seconds per
epoch, no early stopping.

    python scripts/verify/pr4_timing.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import dataclasses  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from _common import provenance, require_clean_tree, write_artifact  # noqa: E402
from _data import make_data  # noqa: E402
from pr3_metrics import CKPT, CKPT_EXT, make_windows, train_scaler  # noqa: E402

REPEATS, WARMUP, BATCH = 10, 2, 4096


def time_inference(path: Any, w: dict[str, Any]) -> dict[str, float]:
    from twinlink_ml.data.windows import WindowBatch
    from twinlink_ml.utils.checkpoints import load_checkpoint

    model, _ = load_checkpoint(path)
    batches = [
        WindowBatch(
            history_inputs=w["hist"][a : a + BATCH],
            future_exogenous=w["fut_exo"][a : a + BATCH],
            future_targets=(w["ref"][a : a + BATCH] - w["tmean"]) / w["tscale"],
            run_index=w["run"][a : a + BATCH].astype(np.float64),
            window_start=w["start"][a : a + BATCH].astype(np.float64),
        )
        for a in range(0, len(w["run"]), BATCH)
    ]
    samples = []
    for i in range(WARMUP + REPEATS):
        start = time.perf_counter()
        for b in batches:
            model.predict(b)
        elapsed = time.perf_counter() - start
        if i >= WARMUP:
            samples.append(1e6 * elapsed / len(w["run"]))
    s = np.asarray(samples)
    return {
        "mean_us": float(s.mean()),
        "sd_us": float(s.std(ddof=1)),
        "min_us": float(s.min()),
        "median_us": float(np.median(s)),
        "max_us": float(s.max()),
        "relative_sd": float(s.std(ddof=1) / s.mean()),
    }


def time_training(core: str, train: Any, val: Any) -> dict[str, float]:
    from twinlink_ml.models.composite import (
        CompositeGruSequenceModel,
        CompositeLstmSequenceModel,
        CompositeSequenceModel,
        CompositeTransformerSequenceModel,
    )

    factory = {
        "gru": CompositeGruSequenceModel,
        "lstm": CompositeLstmSequenceModel,
        "transformer": CompositeTransformerSequenceModel,
    }[core]
    base = factory()
    model = CompositeSequenceModel(dataclasses.replace(base.config, max_epochs=2, patience=50))
    start = time.perf_counter()
    model.fit(train, seed=2509151119, validation=val)
    elapsed = time.perf_counter() - start
    diagnostics = model.training_diagnostics()
    epochs = [v for k, v in diagnostics.items() if k.startswith("epochs_run/")]
    return {
        "elapsed_s": elapsed,
        "epochs_per_head": float(np.mean(epochs)),
        "seconds_per_epoch_all_three_heads": elapsed / float(np.mean(epochs)),
    }


def main() -> None:
    import torch

    require_clean_tree()
    torch.set_num_threads(1)
    from twinlink_ml.evaluation.experiments import build_datasets

    data = make_data("excited", 1)
    w = make_windows(data, data.split.test, train_scaler(data, 64))
    paths = {
        "gru": CKPT / "gru_seed1_n64.pt",
        "lstm": CKPT / "lstm_seed1_n64.pt",  # 600-epoch probe weights, same architecture
        "transformer": CKPT_EXT / "transformer_seed1_n64.pt",
    }
    inference = {k: time_inference(p, w) for k, p in paths.items()}
    for k, v in inference.items():
        print("inference", k, round(v["median_us"], 1), "us", flush=True)
    train, val, _ = build_datasets(data, 64)
    training = {}
    for core in ("lstm", "gru", "transformer"):
        training[core] = time_training(core, train, val)
        print("training", core, training[core], flush=True)
    write_artifact(
        "verify_pr4_timing",
        {
            "_provenance": provenance(),
            "machine": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
            },
            "protocol": {
                "repeats": REPEATS,
                "warmup": WARMUP,
                "batch": BATCH,
                "windows": len(w["run"]),
                "test_set": "excited, seed 1",
            },
            "inference": inference,
            "inference_ratios_median": {
                "gru_over_lstm": inference["gru"]["median_us"] / inference["lstm"]["median_us"],
                "transformer_over_lstm": inference["transformer"]["median_us"]
                / inference["lstm"]["median_us"],
            },
            "training_2_epochs_n64": training,
            "training_epoch_cost_ratio_transformer_over_lstm": training["transformer"][
                "seconds_per_epoch_all_three_heads"
            ]
            / training["lstm"]["seconds_per_epoch_all_three_heads"],
        },
    )


if __name__ == "__main__":
    main()
