"""TE3a -- inference time and training time per epoch, all architectures, one machine.

One thread (environment and torch), no parallel load (run with nothing else active), batch size
4096 (the one of the evaluator), the 4352 excited test windows of seed 1. Warm-up passes are
excluded for EVERY model (3 full passes before timing), 15 timed repeats, and the whole
measurement is repeated in reversed model order so that a residual order or warm-up effect would
show as a difference between the two rounds. Networks are loaded read-only from the archive
checkpoints (weights do not change the cost). Training: one discarded warm-up epoch per
architecture, then 3 timed epochs (patience large so that no early stopping interferes), all three
heads, n_train=64 data.

    python scripts/verify/te3a_timing_all.py
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

REPEATS, WARMUP_PASSES, BATCH, TIMED_EPOCHS = 15, 3, 4096, 3
ARCHS = ("lstm", "gru", "transformer", "xlstm", "mamba")


def ckpt_path(arch: str) -> Any:
    return {
        "lstm": CKPT / "lstm_seed1_n64.pt",
        "gru": CKPT / "gru_seed1_n64.pt",
        "transformer": CKPT_EXT / "transformer_seed1_n64.pt",
        "xlstm": CKPT_EXT / "xlstm_seed1_n64.pt",
        "mamba": CKPT_EXT / "mamba_seed1_n64.pt",
    }[arch]


def stats(samples: list[float]) -> dict[str, float]:
    s = np.asarray(samples)
    return {
        "median_us": float(np.median(s)),
        "min_us": float(s.min()),
        "max_us": float(s.max()),
        "relative_range": float((s.max() - s.min()) / np.median(s)),
        "n": len(samples),
    }


def main() -> None:
    import torch

    require_clean_tree()
    torch.set_num_threads(1)
    from twinlink_ml.data.windows import WindowBatch
    from twinlink_ml.evaluation.experiments import build_datasets
    from twinlink_ml.models.composite import (
        CompositeGruSequenceModel,
        CompositeLstmSequenceModel,
        CompositeMambaSequenceModel,
        CompositeSequenceModel,
        CompositeTransformerSequenceModel,
        CompositeXLSTMSequenceModel,
    )
    from twinlink_ml.utils.checkpoints import load_checkpoint

    data = make_data("excited", 1)
    w = make_windows(data, data.split.test, train_scaler(data, 64))
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
    models = {a: load_checkpoint(ckpt_path(a))[0] for a in ARCHS}

    def one_pass(model: Any) -> float:
        start = time.perf_counter()
        for b in batches:
            model.predict(b)
        return 1e6 * (time.perf_counter() - start) / len(w["run"])

    for arch in ARCHS:  # warm-up for every model, excluded
        for _ in range(WARMUP_PASSES):
            one_pass(models[arch])
    rounds: dict[str, dict[str, list[float]]] = {"forward": {}, "reversed": {}}
    for name, order in (("forward", ARCHS), ("reversed", ARCHS[::-1])):
        for arch in order:
            rounds[name][arch] = [one_pass(models[arch]) for _ in range(REPEATS)]
            print(
                "inference", name, arch, round(float(np.median(rounds[name][arch])), 1), flush=True
            )
    inference = {
        arch: {
            "forward_round": stats(rounds["forward"][arch]),
            "reversed_round": stats(rounds["reversed"][arch]),
            "both_rounds": stats(rounds["forward"][arch] + rounds["reversed"][arch]),
        }
        for arch in ARCHS
    }

    train, val, _ = build_datasets(data, 64)
    factories = {
        "lstm": CompositeLstmSequenceModel,
        "gru": CompositeGruSequenceModel,
        "transformer": CompositeTransformerSequenceModel,
        "xlstm": CompositeXLSTMSequenceModel,
        "mamba": CompositeMambaSequenceModel,
    }
    training: dict[str, Any] = {}
    for arch, factory in factories.items():
        base = factory()

        def fit(epochs: int, base: Any = base) -> float:
            m = CompositeSequenceModel(
                dataclasses.replace(base.config, max_epochs=epochs, patience=50)
            )
            start = time.perf_counter()
            m.fit(train, seed=2509151119, validation=val)
            return time.perf_counter() - start

        fit(1)  # warm-up, discarded
        elapsed = fit(TIMED_EPOCHS)
        training[arch] = {
            "elapsed_s": elapsed,
            "epochs": TIMED_EPOCHS,
            "seconds_per_epoch_three_heads": elapsed / TIMED_EPOCHS,
        }
        print("training", arch, training[arch], flush=True)
    write_artifact(
        "verify_te3a_timing_all",
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
                "repeats_per_round": REPEATS,
                "warmup_passes_all_models": WARMUP_PASSES,
                "batch": BATCH,
                "windows": len(w["run"]),
                "test_set": "excited, seed 1",
                "training_timed_epochs": TIMED_EPOCHS,
                "training_warmup_epoch_discarded": True,
            },
            "inference": inference,
            "inference_ratio_to_lstm_median_both_rounds": {
                a: inference[a]["both_rounds"]["median_us"]
                / inference["lstm"]["both_rounds"]["median_us"]
                for a in ARCHS
            },
            "training_seconds_per_epoch_ratio_to_lstm": {
                a: training[a]["seconds_per_epoch_three_heads"]
                / training["lstm"]["seconds_per_epoch_three_heads"]
                for a in ARCHS
            },
            "training_timed_epochs_n64": training,
        },
    )


if __name__ == "__main__":
    main()
