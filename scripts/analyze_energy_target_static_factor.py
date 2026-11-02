r"""SA1: On which test set does Befund 58's factor 6.4-7.8 (energy_target, static
regression against GRU/LSTM/Transformer) hold? Loads the same two logs as OA5
(``scripts/analyze_arch_learning_curve_extended.py``, commit ``bc6b097``), WITHOUT training
again, and computes the factor per test set, horizon and budget point --
``rmse_run_median_physical(static) / rmse_run_median_physical(Modell)``, median over the 5
seeds per cell.

Background: Befund 51 (K5) showed on `product_moisture_kg_per_kg` that a factor measured on
the excited test set (19-24) can collapse to 1.31 on the operationally relevant
constant-run holdout, because the baseline used there (persistence) differs structurally
in strength between the test sets. This script checks whether the same effect is present
for `energy_target`/static regression.

Usage::

    python scripts/analyze_energy_target_static_factor.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_LOG = REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_projected.jsonl"
EXT_LOG = REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_projected_extended.jsonl"

TEST_SETS = ("excited", "constant_holdout")
HORIZONS = (1, 36)
BUDGET_POINTS = (8, 16, 32, 64)
COMPARISON_MODELS = ("gru", "lstm", "transformer")
TARGET = "energy_target"


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _rmse_values(
    records: list[dict], *, label: str, test_set: str, n_train: int, horizon: int
) -> list[float]:
    key = f"rmse_run_median_physical/{TARGET}/h{horizon}"
    return [
        record["metrics"][key]
        for record in records
        if record["params"]["model/label"] == label
        and record["params"]["test_set"] == test_set
        and record["params"]["n_train"] == n_train
    ]


def main() -> None:
    records = _load(BASE_LOG) + _load(EXT_LOG)
    print(
        f"{'Testmenge':>17} h  n_train   static[kW]  "
        + "".join(f"{m:>14}" for m in COMPARISON_MODELS)
    )
    for test_set in TEST_SETS:
        for horizon in HORIZONS:
            for n_train in BUDGET_POINTS:
                static_vals = _rmse_values(
                    records, label="static", test_set=test_set, n_train=n_train, horizon=horizon
                )
                static_med = statistics.median(static_vals)
                row = f"{test_set:>17} {horizon:<2} {n_train:<7}   {static_med:.5f}    "
                for model in COMPARISON_MODELS:
                    model_vals = _rmse_values(
                        records, label=model, test_set=test_set, n_train=n_train, horizon=horizon
                    )
                    model_med = statistics.median(model_vals)
                    row += f"  {model_med:.5f}(x{static_med / model_med:.2f})"
                print(row)


if __name__ == "__main__":
    main()
