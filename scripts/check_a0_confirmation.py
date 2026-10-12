r"""Confirmation that the old records serve as arm A0 (decision of the project lead, 2026-10-12).

The pilot (Befund 84) reproduced the old records of arm A0 bit for bit for Transformer, xLSTM and
Mamba (Windows) and within 4e-5 for GRU and LSTM (Linux), at seed 1 and ``n_train`` 8. The old records
are therefore used as A0, with ONE confirmation where the platform differs: GRU and LSTM, seed 1,
``n_train`` 64, legacy data specification, re-run on this machine with the worker of the re-run.

Rule fixed before the run: for each of GRU and LSTM and each test set (excited, constant holdout), the
relative deviation of EVERY error metric (all common numeric metrics of the primary variant, except the
speed metrics) from the old record is below 1e-3 (relative, absolute floor 1e-12). If so, the old records
count as A0; otherwise the deviation is reported and the project lead decides.

    tools/run_here.sh scripts/check_a0_confirmation.py --records rerun_a0confirm_A0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from check_rerun_gate import max_relative_deviation

THRESHOLD = 1e-3
LABELS = ("gru", "lstm")
TEST_SETS = ("excited", "constant_holdout")
SEED, N_TRAIN = 1, 64


def load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="merged file artifacts/runs/<name>.jsonl")
    arguments = parser.parse_args()
    require_clean_tree()
    old = load(RUNS_DIR / "arch_learning_curve_projected.jsonl")
    new = [r for r in load(RUNS_DIR / f"{arguments.records}.jsonl") if r["params"]["eval/primary"]]
    cells = []
    for label in LABELS:
        for test_set in TEST_SETS:

            def pick(records: list[dict]) -> dict:
                return next(
                    r
                    for r in records
                    if r["params"]["model/label"] == label
                    and r["params"]["test_set"] == test_set
                    and int(r["params"]["n_train"]) == N_TRAIN
                    and int(r["params"]["master_seed"]) == SEED
                )

            o, n = pick(old), pick(new)
            worst, where, count = max_relative_deviation(n["metrics"], o["metrics"])
            epochs = {
                k: (o["params"].get(k), n["params"].get(k))
                for k in o["params"]
                if k.startswith("training/epochs_run")
            }
            cells.append(
                {
                    "model": label,
                    "test_set": test_set,
                    "metrics_compared": count,
                    "max_relative_deviation": worst,
                    "at": where,
                    "epochs_run_old_new": epochs,
                    "platform_old": o["params"].get("platform"),
                    "torch_old": o["params"].get("version/torch"),
                }
            )
    worst_all = max(c["max_relative_deviation"] for c in cells)
    passed = worst_all < THRESHOLD
    write_artifact(
        arguments.records + "_check",
        {
            "_provenance": provenance(),
            "threshold": THRESHOLD,
            "cells": cells,
            "worst": worst_all,
            "old_records_count_as_A0": passed,
        },
    )
    print("OLD RECORDS COUNT AS A0" if passed else "DEVIATION ABOVE THRESHOLD -- report")
    for c in cells:
        print(
            f"  {c['model']} {c['test_set']}: {c['metrics_compared']} metrics, max {c['max_relative_deviation']:.3e} at {c['at']}"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
