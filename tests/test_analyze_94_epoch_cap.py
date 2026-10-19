"""Epoch-cap counts (addendum to Befund 94) on synthetic records with known epochs."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import analyze_94_epoch_cap as cap  # noqa: E402


def _record(label: str, seed: int, n_train: int, epochs: dict[str, int], best: dict[str, int]):
    params = {
        "model/label": label,
        "master_seed": seed,
        "n_train": n_train,
        "model/net_max_epochs": 200,
        "test_set": "excited",
    }
    params.update({f"training/epochs_run/{h}": e for h, e in epochs.items()})
    params.update({f"training/best_epoch/{h}": b for h, b in best.items()})
    return {"params": params, "metrics": {}}


def _records():
    rows = []
    for seed in (1, 2):
        # two evaluation variants of the same training must be counted once
        for test_set in ("excited", "constant_holdout"):
            rec = _record("transformer", seed, 64, {"a": 200, "b": 120}, {"a": 195, "b": 100})
            rec["params"]["test_set"] = test_set
            rows.append(rec)
    rows.append(_record("lstm", 1, 8, {"a": 200, "b": 200}, {"a": 100, "b": 190}))
    rows.append(_record("persistence", 1, 8, {}, {}))
    return rows


def test_counts_are_per_training_not_per_record() -> None:
    result = cap.analyse(_records())
    assert result["transformer"]["total"]["trainings"] == 4  # 2 seeds x 2 heads, not 8
    assert result["transformer"]["total"]["at_cap"] == 2
    assert result["transformer"]["by_head"]["a"] == {
        "trainings": 2,
        "at_cap": 2,
        "best_in_last": 2,
        "at_cap_and_best_in_last": 2,
    }
    assert result["transformer"]["by_head"]["b"]["at_cap"] == 0


def test_best_epoch_in_the_last_epochs_and_networks_only() -> None:
    result = cap.analyse(_records())
    assert result["lstm"]["total"] == {
        "trainings": 2,
        "at_cap": 2,
        "best_in_last": 1,
        "at_cap_and_best_in_last": 1,
    }
    assert "persistence" not in result
    assert result["lstm"]["by_n_train"]["8"]["trainings"] == 2
