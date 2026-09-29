"""TE3b -- test-set shift of persistence and static regression at the SAME horizon.

Befund 66 quoted the persistence shift at h=1 (6.3) next to the static shift, and persistence at
h=36 (18) separately. Here both baselines (and, for reference, GRU and LSTM) are compared between
the excited test set and the constant-run holdout at h=1 and h=36, for every budget, in physical
units (median over 5 seeds; per-seed ratios give the spread), from the committed records.

    python scripts/verify/te3b_static_shift.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _common import TARGETS, provenance, read_jsonl, require_clean_tree, write_artifact

BUDGETS = (8, 16, 32, 64)
MODELS = ("persistence", "static", "gru", "lstm")


def main() -> None:
    require_clean_tree()
    recs = read_jsonl("arch_learning_curve_projected")
    out: dict[str, Any] = {"_provenance": provenance(), "cells": {}}

    def series(model: str, test_set: str, n: int, channel: str, h: int) -> list[float]:
        rows = sorted(
            (
                r
                for r in recs
                if r["tags"]["model"] == model
                and r["tags"]["test_set"] == test_set
                and r["params"]["n_train"] == n
            ),
            key=lambda r: r["params"]["master_seed"],
        )
        return [r["metrics"][f"rmse_run_median_physical/{channel}/h{h}"] for r in rows]

    for channel in TARGETS:
        for h in (1, 36):
            for model in MODELS:
                for n in BUDGETS:
                    ex = series(model, "excited", n, channel, h)
                    co = series(model, "constant_holdout", n, channel, h)
                    per_seed = [a / b for a, b in zip(ex, co, strict=True)]
                    out["cells"][f"{channel}/h{h}/{model}/n{n}"] = {
                        "excited_median": float(np.median(ex)),
                        "constant_median": float(np.median(co)),
                        "ratio_of_medians": float(np.median(ex) / np.median(co)),
                        "per_seed_ratio_min_max": [min(per_seed), max(per_seed)],
                    }
    energy = {
        f"h{h}/{m}": out["cells"][f"energy_target/h{h}/{m}/n64"]["ratio_of_medians"]
        for h in (1, 36)
        for m in MODELS
    }
    out["energy_target_n64_ratio_excited_over_constant"] = energy
    out["static_ratio_range_over_budgets_energy"] = {
        f"h{h}": [
            min(
                out["cells"][f"energy_target/h{h}/static/n{n}"]["ratio_of_medians"] for n in BUDGETS
            ),
            max(
                out["cells"][f"energy_target/h{h}/static/n{n}"]["ratio_of_medians"] for n in BUDGETS
            ),
        ]
        for h in (1, 36)
    }
    write_artifact("verify_te3b_static_shift", out)
    print(energy, out["static_ratio_range_over_budgets_energy"])


if __name__ == "__main__":
    main()
