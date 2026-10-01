r"""SX3 (Befund 67) diagnostics: early stopping / epoch cap, divergence per variant and seed.

Reads the committed records of the capacity-matched runs and the Befund 51/58 references
(n_train = 64, excited records carry the training diagnostics). Reports, per variant and head
(seed 1-5): epochs run, whether the fit ended at ``max_epochs`` (budget) or by early stopping, the
``train_loss_ratio_last10`` (training-loss ratio of the last 10 epochs to the 10 before; near 1 =
plateau), and per cell/seed the ratio ``rmse / median rmse of the LSTM`` (h = 36) with a flag for
ratios above 10 and a check for non-finite values. The validation curves are not stored in the
records or checkpoints, so a learning-rate diagnosis from them is not possible here.

    python scripts/analyze_sx3_diagnostics.py
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, read_jsonl, require_clean_tree, write_artifact

LABELS = (
    "lstm", "gru", "xlstm", "mamba", "mlstm_cm", "slstm_cm", "slstm_l1_cm", "mixed_cm",
    "mamba_cm", "lstm_small",
)  # fmt: skip
HEADS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
TEST_SETS = ("excited", "constant_holdout")
THRESHOLD = 10.0


def main() -> None:
    require_clean_tree()
    records = (
        read_jsonl("arch_learning_curve_projected")
        + read_jsonl("arch_learning_curve_projected_extended")
        + read_jsonl("sx3_capacity_matched_a")
        + read_jsonl("sx3_capacity_matched_b")
    )
    rows = [r for r in records if r["tags"]["n_train"] == "64" and r["tags"]["model"] in LABELS]
    training: dict[str, Any] = {}
    for label in LABELS:
        per_head: dict[str, Any] = {}
        for head in HEADS:
            entries = []
            for r in sorted(rows, key=lambda x: x["params"]["master_seed"]):
                if r["tags"]["model"] != label or r["tags"]["test_set"] != "excited":
                    continue
                p = r["params"]
                epochs = p[f"training/epochs_run/{head}"]
                entries.append(
                    {
                        "seed": p["master_seed"],
                        "epochs_run": epochs,
                        "at_epoch_cap": bool(epochs >= p["model/net_max_epochs"]),
                        "train_loss_ratio_last10": p[f"training/train_loss_ratio_last10/{head}"],
                    }
                )
            per_head[head] = {
                "fits": entries,
                "n_at_cap": sum(e["at_epoch_cap"] for e in entries),
                "n_early_stopped": sum(not e["at_epoch_cap"] for e in entries),
            }
        training[label] = per_head

    divergence: dict[str, Any] = {}
    finite = True
    for head in HEADS:
        for test_set in TEST_SETS:
            key = f"rmse_run_median_physical/{head}/h36"
            sel = [r for r in rows if r["tags"]["test_set"] == test_set and key in r["metrics"]]
            lstm_median = statistics.median(
                r["metrics"][key] for r in sel if r["tags"]["model"] == "lstm"
            )
            for label in LABELS:
                ratios = {}
                for r in sel:
                    if r["tags"]["model"] == label:
                        value = r["metrics"][key]
                        finite &= math.isfinite(value)
                        ratios[r["params"]["master_seed"]] = value / lstm_median
                divergence.setdefault(label, {})[f"{head}/{test_set}"] = {
                    "ratio_to_lstm_median_by_seed": dict(sorted(ratios.items())),
                    "seeds_above_threshold": sorted(s for s, v in ratios.items() if v > THRESHOLD),
                }
    write_artifact(
        "sx3_diagnostics",
        {
            "_provenance": provenance(),
            "divergence_threshold_ratio": THRESHOLD,
            "all_metrics_finite": finite,
            "training": training,
            "divergence": divergence,
        },
    )
    print("all metrics finite:", finite)
    for label in LABELS:
        caps = {h: training[label][h]["n_at_cap"] for h in HEADS}
        flagged = {
            c: v["seeds_above_threshold"]
            for c, v in divergence[label].items()
            if v["seeds_above_threshold"]
        }
        print(f"{label:12s} heads at cap/5 {caps}  seeds >10x LSTM: {flagged}")


if __name__ == "__main__":
    main()
