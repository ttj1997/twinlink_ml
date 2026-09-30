"""SX3 -- run-time probe before the capacity-matched comparison (no training, synthetic tensors).

Times one forward+backward pass and one forward pass of a single head network (7 exogenous inputs
+ 1 target, history 32, horizon 36, batch 256, one thread -- the shapes and thread setting of the
Befund 51/58 protocol) for the reference configurations of Befund 51/58 and the capacity-matched
configurations of ``twinlink_ml.models.capacity_matched``. The reference timings are compared with
the per-epoch cost recorded in the committed Befund 51/58 records (n_train=64, seconds per epoch and
head), which gives a calibration factor between the probe and a real epoch (which also contains data
handling, the optimiser, validation and early-stopping bookkeeping). The estimate for the new
configurations is the probe time times that factor, times an assumed number of epochs; the
assumed epochs are a range taken from the recorded epochs of the reference models.

    python scripts/verify/sx3_cost_probe.py
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from _common import provenance, read_jsonl, require_clean_tree, write_artifact

from twinlink_ml.models.capacity_matched import CAPACITY_MATCHED_CONFIGS
from twinlink_ml.models.torch_sequence import TorchSequenceConfig, _build_network

N_EXOGENOUS = 7
N_TARGETS = 1
HISTORY = 32
HORIZON = 36
BATCH = 256
TRAIN_WINDOWS_PER_RUN = 136  # (1152 - 68) / 8 + 1, DataSpec of Befund 51
N_TRAIN = 64
N_VAL_RUNS = 32
REPEATS = 5

REFERENCES = {
    "lstm": TorchSequenceConfig(core="lstm", hidden_size=64),
    "gru": TorchSequenceConfig(core="gru", hidden_size=64),
    "xlstm": TorchSequenceConfig(core="xlstm", hidden_size=64),
    "mamba": TorchSequenceConfig(core="mamba", hidden_size=16, state_size=16),
}


def _time_config(config: TorchSequenceConfig) -> dict[str, Any]:
    torch.manual_seed(0)
    network = _build_network(torch, n_exogenous=N_EXOGENOUS, n_targets=N_TARGETS, config=config)
    history = torch.randn(BATCH, HISTORY, N_EXOGENOUS + N_TARGETS)
    future = torch.randn(BATCH, HORIZON, N_EXOGENOUS)
    target = torch.randn(BATCH, HORIZON, N_TARGETS)
    optimiser = torch.optim.AdamW(network.parameters(), lr=1e-3)
    train_times, forward_times = [], []
    for repeat in range(REPEATS + 2):  # two warm-up passes are discarded
        network.train()
        start = time.perf_counter()
        optimiser.zero_grad()
        loss = (network(history, future) - target).square().mean()
        loss.backward()
        optimiser.step()
        train_time = time.perf_counter() - start
        network.eval()
        with torch.no_grad():
            start = time.perf_counter()
            network(history, future)
            forward_time = time.perf_counter() - start
        if repeat >= 2:
            train_times.append(train_time)
            forward_times.append(forward_time)
    return {
        "parameters_per_head": sum(p.numel() for p in network.parameters()),
        "train_step_s_median": statistics.median(train_times),
        "forward_s_median": statistics.median(forward_times),
    }


def main() -> None:
    require_clean_tree()
    torch.set_num_threads(1)
    train_batches = -(-N_TRAIN * TRAIN_WINDOWS_PER_RUN // BATCH)
    val_batches = -(-N_VAL_RUNS * TRAIN_WINDOWS_PER_RUN // BATCH)

    probes: dict[str, Any] = {}
    for label, config in {**REFERENCES, **CAPACITY_MATCHED_CONFIGS}.items():
        probes[label] = _time_config(config)
        probes[label]["probe_epoch_s"] = (
            train_batches * probes[label]["train_step_s_median"]
            + val_batches * probes[label]["forward_s_median"]
        )
        print(label, probes[label])

    # Recorded epoch cost of the reference models at n_train=64 (seconds per epoch and head).
    records = read_jsonl("arch_learning_curve_projected") + read_jsonl(
        "arch_learning_curve_projected_extended"
    )
    recorded: dict[str, Any] = {}
    for label in REFERENCES:
        seconds, epochs = [], []
        for record in records:
            if (
                record["tags"]["model"] == label
                and record["tags"]["test_set"] == "excited"
                and record["tags"]["n_train"] == "64"
            ):
                total_epochs = sum(
                    value
                    for key, value in record["params"].items()
                    if key.startswith("training/epochs_run/")
                )
                seconds.append(record["params"]["training/elapsed_s"] / total_epochs)
                epochs.append(total_epochs / 3)
        recorded[label] = {
            "seconds_per_epoch_and_head_median": statistics.median(seconds),
            "epochs_per_head_median": statistics.median(epochs),
            "epochs_per_head_min": min(epochs),
            "epochs_per_head_max": max(epochs),
        }
    calibration = {
        label: recorded[label]["seconds_per_epoch_and_head_median"] / probes[label]["probe_epoch_s"]
        for label in REFERENCES
    }
    scale = statistics.median(calibration.values())
    all_epochs = [recorded[label]["epochs_per_head_median"] for label in REFERENCES]
    epochs_low, epochs_high = min(all_epochs), max(all_epochs)

    estimates: dict[str, Any] = {}
    for label in CAPACITY_MATCHED_CONFIGS:
        per_epoch_head = probes[label]["probe_epoch_s"] * scale
        estimates[label] = {
            "seconds_per_epoch_and_head": per_epoch_head,
            "seed_job_hours_low": per_epoch_head * epochs_low * 3 / 3600,
            "seed_job_hours_high": per_epoch_head * epochs_high * 3 / 3600,
        }
    write_artifact(
        "sx3_cost_probe",
        {
            "_provenance": provenance(),
            "shapes": {
                "batch": BATCH,
                "history": HISTORY,
                "horizon": HORIZON,
                "train_batches_per_epoch": train_batches,
                "validation_batches_per_epoch": val_batches,
            },
            "probes": probes,
            "recorded_reference_epoch_cost": recorded,
            "calibration_factor_recorded_over_probe": calibration,
            "calibration_scale_used_median": scale,
            "assumed_epochs_per_head_range": [epochs_low, epochs_high],
            "estimates_one_seed_job_n_train_64_three_heads": estimates,
        },
    )


if __name__ == "__main__":
    main()
