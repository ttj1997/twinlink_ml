r"""SA1: Auf welcher Testmenge gilt Befund 58s Faktor 6.4-7.8 (energy_target, statische
Regression gegen GRU/LSTM/Transformer)? Lädt dieselben zwei Logs wie OA5
(``scripts/analyze_arch_learning_curve_extended.py``, Commit ``bc6b097``), OHNE erneut zu
trainieren, und rechnet den Faktor je Testmenge, Horizont und Budgetpunkt aus --
``rmse_run_median_physical(static) / rmse_run_median_physical(Modell)``, Median über die 5
Seeds je Zelle.

Hintergrund: Befund 51 (K5) zeigte an `product_moisture_kg_per_kg`, dass ein auf der
angeregten Testmenge gemessener Faktor (19-24) auf dem betriebsrelevanten
Konstantlauf-Holdout auf 1.31 einbrechen kann, weil die dort verwendete Baseline
(Persistenz) zwischen den Testmengen strukturell unterschiedlich stark ist. Dieses Skript
prüft, ob derselbe Effekt bei `energy_target`/statischer Regression vorliegt.

Aufruf::

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
