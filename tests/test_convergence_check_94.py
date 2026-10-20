"""Convergence check of Befund 94 (600 epochs): driver jobs, gate and readings (synthetic input)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import analyze_convergence_check_94 as conv  # noqa: E402
import run_convergence_check_94 as driver  # noqa: E402


def test_jobs_cover_networks_sizes_seeds_and_only_change_the_epoch_budget():
    jobs = driver.build_jobs([16, 64], [1, 2, 3, 4, 5])
    assert len(jobs) == 30
    assert jobs[0].architecture == "transformer" and jobs[0].n_train == 64
    command = jobs[0].command(Path("out.jsonl"), Path("ckpt"))
    assert command[command.index("--net-max-epochs") + 1] == "600"
    assert command[command.index("--arm") + 1] == "N"
    assert "--skip-baselines" in command
    assert jobs[0].expected_records == 8  # 1 method x 1 budget x 4 variants x 2 test sets


def _params(label: str, n: int, seed: int, max_epochs: int, epochs: int, history: list[float]):
    p = {
        "model/label": label,
        "n_train": n,
        "master_seed": seed,
        "test_set": "excited",
        "eval/pool": "own",
        "eval/window": "planned",
        "eval/primary": True,
        "model/net_max_epochs": max_epochs,
        "git/dirty": False,
        "pool/projected_steps": 0,
        "holdout_pool/projected_steps": 0,
        "training/elapsed_s": 1.0,
    }
    for head in conv.HEADS:
        p[f"training/epochs_run/{head}"] = epochs
        p[f"training/best_epoch/{head}"] = epochs - 1
        p[f"training/loss_history_val/{head}"] = history[:epochs]
    return {"params": p, "metrics": {"rmse_run_median_physical/energy_target/h1": 1.0}}


def _pair(epochs_old: int, epochs_new: int, history_new: list[float] | None = None):
    history = [1.0 / (i + 1) for i in range(600)]
    old, new = [], []
    for label in conv.NETWORKS:
        for n in conv.SIZES:
            for seed in conv.SEEDS:
                old.append(_params(label, n, seed, 200, epochs_old, history))
                new.append(_params(label, n, seed, 600, epochs_new, history_new or history))
    return old, new


def test_gate_passes_for_a_continuation_and_fails_for_a_changed_history():
    old, new = _pair(200, 450)
    assert conv.gate(old, new)["passed"]
    changed = [1.0 / (i + 1) for i in range(600)]
    changed[5] *= 1.01
    old, new = _pair(200, 450, changed)
    result = conv.gate(old, new)
    assert not result["passed"] and "epoch 5" in result["failures"][0]


def test_gate_fails_if_an_early_stopped_head_runs_longer():
    old, new = _pair(150, 170)
    assert not conv.gate(old, new)["passed"]


def test_gate_fails_on_dirty_tree_or_wrong_budget():
    old, new = _pair(200, 450)
    bad = copy.deepcopy(new)
    bad[0]["params"]["git/dirty"] = True
    assert not conv.gate(old, bad)["passed"]
    bad = copy.deepcopy(new)
    bad[0]["params"]["model/net_max_epochs"] = 200
    assert not conv.gate(old, bad)["passed"]


def _rows(transformer_worse_200: int, transformer_worse_600: int, flip: bool = False):
    rows = []
    for candidate, reference in conv.PAIRS:
        for i, cell in enumerate(conv.cell_list(control=False)):
            v200 = v600 = conv.UNRESOLVED
            if candidate == "transformer":
                v200 = conv.WORSE if i < transformer_worse_200 else conv.UNRESOLVED
                v600 = conv.WORSE if i < transformer_worse_600 else conv.UNRESOLVED
            if flip and candidate == "gru" and i == 0:
                v600 = conv.WORSE
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "n_train": cell[0],
                    "test_set": cell[1],
                    "channel": cell[2],
                    "horizon": cell[3],
                    "primary": True,
                    "verdict_200": v200,
                    "verdict_600": v600,
                }
            )
    return rows


def _family(changed: bool = False):
    key = ("transformer", 64, "excited", "energy_target", 36)
    return {
        "cells": [
            {
                "key": key,
                "holm_200": conv.WORSE,
                "holm_600": conv.UNRESOLVED if changed else conv.WORSE,
            }
        ]
    }


def test_readings():
    assert conv.reading(_rows(13, 13), _family())["reading"] == "K1"
    assert conv.reading(_rows(13, 6), _family())["reading"] == "K2"
    assert conv.reading(_rows(13, 12), _family(changed=True))["reading"] == "none"
    assert conv.reading(_rows(13, 13, flip=True), _family())["reading"] == "K3"
    # K2 boundary: at most half (13 -> 6 is K2, 13 -> 7 is not)
    assert conv.reading(_rows(13, 7), _family())["reading"] == "K1"


def test_gate_ignores_the_wall_clock_metric_only():
    old, new = _pair(150, 150)
    for r in old:
        r["metrics"]["inference_us_per_window"] = 90.0
    for r in new:
        r["metrics"]["inference_us_per_window"] = 80.0
    assert conv.gate(old, new)["passed"]
    new[0]["metrics"]["rmse_run_median_physical/energy_target/h1"] = 1.1
    assert not conv.gate(old, new)["passed"]
