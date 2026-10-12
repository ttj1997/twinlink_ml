r"""Evaluation of the re-run of the architecture comparison (docs/15, section 5). Committed BEFORE the full run.

Arms (decisions of the project lead, 2026-10-12): **A0 = the old records of Befunde 51/58** (the pilot
reproduced them bit for bit for Transformer, xLSTM, Mamba and within 4e-5 for GRU/LSTM; confirmation for
GRU/LSTM at ``n_train`` 64 in ``rerun_a0confirm_A0_check``), **A1** (legacy trajectories, planned future
inputs; ``n_train`` 8 and 64) and **N** (``DataSpec.new_study()``; ``n_train`` 8, 16, 32, 64). Seeds 1-5,
paired by ``master_seed``; metric ``rmse_run_median_physical``, h = 1 and 36, three channels, excited test
set and constant-run holdout (every result names its test set).

What is computed (rules fixed here, before any new record exists):

1. CHANGE OF THE ERRORS. For every method, channel, test set, h and budget: A1 against A0 (n_train 8, 64) and
   N against A0 (all budgets), primary variant of the arm (own pool, own window). Primary verdict: paired t
   test on the differences of the physical metric (``paired_resolution``; as Befund 83); next to it the mean of
   ``log(arm / A0)`` over the seeds and the paired t test on the log values (sensitivity, not a verdict), the
   medians, and in how many seeds the arm is worse.
2. SHARE DUE TO D1. ``share = mean log(A1/A0) / mean log(N/A0)`` per method, channel, test set, h, budget 8 and
   64, **only where the denominator is resolved** (paired t test on the log ratio of N against A0, p < 0.05) and
   has the same sign as the numerator's; otherwise ``null``. The pool of A1 is the pool of A0; N has another
   pool (E3), so the share is a share of the combined D1 + E3 effect.
3. TRAINING PART AND TEST PART OF D1 (cross-evaluation of arm A1): ``log(A1 model on planned windows / A0)`` =
   test part ``log(A1 model on planned / A1 model on realised windows)`` + training part
   ``log(A1 model on realised windows / A0)``; mean over the seeds per cell. For arm N the variants
   (own pool / legacy pool) x (planned / realised windows) are tabulated as log ratios against A0.
4. RANK STATEMENTS of Befund 83 repeated on A0 (old records), A1 and N: every method against LSTM (GRU,
   Transformer, xLSTM, Mamba, static, ARX, persistence), paired verdicts per cell; the number of cells in
   which the verdict changes between arms.
5. MULTIPLICITY, sensitivity only: Holm (5 %) and Benjamini-Hochberg (5 %) within the families
   ``a1_vs_a0``, ``n_vs_a0`` and the rank cells of each arm; the number of tests is reported.
6. SEEDS NEEDED (80 % power, 12.5 % and 10 %, from the observed standard deviation of the paired
   differences) for GRU and Transformer against LSTM in each arm.

Not claimed: a change of the ranking is a change in this model and these data. No pooling across channels or
test sets. Platform: A0 consists of records from two platforms; the confirmation covers GRU/LSTM.

    tools/run_here.sh scripts/analyze_rerun.py --prefix rerun_full --a1-sizes 8 64
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from analyze_83_addendum import benjamini_hochberg, holm, seeds_row
from analyze_paired_reevaluation import compare

from twinlink_ml.evaluation.paired_resolution import paired_comparison

METRIC = "rmse_run_median_physical"
CHANNELS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
TEST_SETS = ("excited", "constant_holdout")
HORIZONS = (1, 36)
SEEDS = (1, 2, 3, 4, 5)
METHODS = ("persistence", "static", "arx", "gru", "lstm", "transformer", "xlstm", "mamba")
RANK_CANDIDATES = ("gru", "transformer", "xlstm", "mamba", "static", "arx", "persistence")
ALPHA = 0.05
Records = list[dict[str, Any]]


def load(path: Path) -> Records:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def series(
    records: Records,
    *,
    label: str,
    n_train: int,
    test_set: str,
    channel: str,
    horizon: int,
    pool: str = "own",
    window: str | None = None,
) -> dict[int, float]:
    """Metric per seed. ``window=None``: the primary variant (old records have no variant fields)."""
    key = f"{METRIC}/{channel}/h{horizon}"
    out: dict[int, float] = {}
    for record in records:
        p = record["params"]
        if (
            p["model/label"] != label
            or int(p["n_train"]) != n_train
            or p["test_set"] != test_set
            or key not in record["metrics"]
        ):
            continue
        if "eval/primary" in p:  # new records
            if window is None:
                if not p["eval/primary"]:
                    continue
            elif p["eval/pool"] != pool or p["eval/window"] != window:
                continue
        elif window is not None and (window != "realised" or pool != "own"):
            continue  # old record: only the realised window of the legacy pool exists
        seed = int(p["master_seed"])
        if seed in out:
            raise SystemExit(f"duplicate record {label} n={n_train} {test_set} seed {seed}")
        out[seed] = float(record["metrics"][key])
    if tuple(sorted(out)) != SEEDS:
        raise SystemExit(
            f"{label} n={n_train} {test_set} {channel} h{horizon} ({pool}/{window}): seeds "
            f"{sorted(out)}, expected {SEEDS}"
        )
    return out


def mean_log_ratio(candidate: dict[int, float], reference: dict[int, float]) -> float:
    return float(np.mean([math.log(candidate[s] / reference[s]) for s in SEEDS]))


def cells(sizes: list[int]) -> list[tuple[int, str, str, int]]:
    return [
        (n, test_set, channel, h)
        for n in sizes
        for test_set in TEST_SETS
        for channel in CHANNELS
        for h in HORIZONS
    ]


def change_table(
    arm: Records, a0: Records, sizes: list[int], arm_name: str
) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for n, test_set, channel, h in cells(sizes):
            kw = {
                "label": method,
                "n_train": n,
                "test_set": test_set,
                "channel": channel,
                "horizon": h,
            }
            candidate = series(arm, **kw)
            reference = series(a0, **kw)
            row = compare(candidate, reference, f"{method}[{arm_name}]", f"{method}[A0]")
            logged = paired_comparison(
                {s: math.log(v) for s, v in candidate.items()},
                {s: math.log(v) for s, v in reference.items()},
            )
            row.update(
                {
                    "method": method,
                    "n_train": n,
                    "test_set": test_set,
                    "channel": channel,
                    "horizon": h,
                    "arm": arm_name,
                    "mean_log_ratio_arm_over_a0": mean_log_ratio(candidate, reference),
                    "log_ratio_p_value": logged.p_value,
                    "log_ratio_resolved": bool(logged.is_resolved),
                }
            )
            rows.append(row)
    return rows


def d1_share(a1: Records, n_arm: Records, a0: Records, sizes: list[int]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for n, test_set, channel, h in cells(sizes):
            kw = {
                "label": method,
                "n_train": n,
                "test_set": test_set,
                "channel": channel,
                "horizon": h,
            }
            ref = series(a0, **kw)
            to_a1 = mean_log_ratio(series(a1, **kw), ref)
            n_values = series(n_arm, **kw)
            total = mean_log_ratio(n_values, ref)
            logged = paired_comparison(
                {s: math.log(v) for s, v in n_values.items()},
                {s: math.log(v) for s, v in ref.items()},
            )
            usable = bool(logged.is_resolved) and to_a1 * total > 0.0
            rows.append(
                {
                    "method": method,
                    "n_train": n,
                    "test_set": test_set,
                    "channel": channel,
                    "horizon": h,
                    "mean_log_a1_over_a0": to_a1,
                    "mean_log_n_over_a0": total,
                    "denominator_resolved": bool(logged.is_resolved),
                    "share_d1": to_a1 / total if usable else None,
                }
            )
    return rows


def cross_parts(a1: Records, a0: Records, sizes: list[int]) -> list[dict[str, Any]]:
    """Test part and training part of the A1 effect (cross-evaluation)."""
    rows = []
    for method in METHODS:
        for n, test_set, channel, h in cells(sizes):
            kw = {
                "label": method,
                "n_train": n,
                "test_set": test_set,
                "channel": channel,
                "horizon": h,
            }
            ref = series(a0, **kw)
            planned = series(a1, **kw, window="planned")
            realised = series(a1, **kw, window="realised")
            rows.append(
                {
                    "method": method,
                    "n_train": n,
                    "test_set": test_set,
                    "channel": channel,
                    "horizon": h,
                    "total": mean_log_ratio(planned, ref),
                    "test_part": mean_log_ratio(planned, realised),
                    "training_part": mean_log_ratio(realised, ref),
                }
            )
    return rows


def n_variants(n_arm: Records, a0: Records, sizes: list[int]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for n, test_set, channel, h in cells(sizes):
            kw = {
                "label": method,
                "n_train": n,
                "test_set": test_set,
                "channel": channel,
                "horizon": h,
            }
            ref = series(a0, **kw)
            entry: dict[str, Any] = {
                "method": method,
                "n_train": n,
                "test_set": test_set,
                "channel": channel,
                "horizon": h,
            }
            for pool in ("own", "legacy"):
                for window in ("realised", "planned"):
                    entry[f"{pool}_pool_{window}_windows"] = mean_log_ratio(
                        series(n_arm, **kw, pool=pool, window=window), ref
                    )
            rows.append(entry)
    return rows


def rank_cells(arm: Records, sizes: list[int]) -> list[dict[str, Any]]:
    rows = []
    for candidate in RANK_CANDIDATES:
        for n, test_set, channel, h in cells(sizes):
            kw = {"n_train": n, "test_set": test_set, "channel": channel, "horizon": h}
            c = series(arm, label=candidate, **kw)
            r = series(arm, label="lstm", **kw)
            row = compare(c, r, candidate, "lstm")
            row.update({"n_train": n, "test_set": test_set, "channel": channel, "horizon": h})
            rows.append(row)
    return rows


def correction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    p = np.array([1.0 if math.isnan(r["p_value"]) else r["p_value"] for r in rows])
    resolved = np.array([r["verdict"] in ("candidate better", "candidate worse") for r in rows])
    holm_p, bh_p = holm(p), benjamini_hochberg(p)
    return {
        "tests": len(rows),
        "resolved_uncorrected": int(resolved.sum()),
        "resolved_holm": int((resolved & (holm_p < ALPHA)).sum()),
        "resolved_bh": int((resolved & (bh_p < ALPHA)).sum()),
    }


def evaluate(
    a0: Records, a1: Records, n_arm: Records, a1_sizes: list[int], n_sizes: list[int]
) -> dict[str, Any]:
    a1_vs = change_table(a1, a0, a1_sizes, "A1")
    n_vs = change_table(n_arm, a0, n_sizes, "N")
    ranks = {
        "A0": rank_cells(a0, n_sizes),
        "A1": rank_cells(a1, a1_sizes),
        "N": rank_cells(n_arm, n_sizes),
    }
    changed = []
    index_a0 = {
        (r["candidate"], r["n_train"], r["test_set"], r["channel"], r["horizon"]): r
        for r in ranks["A0"]
    }
    for arm in ("A1", "N"):
        for r in ranks[arm]:
            base = index_a0[
                (r["candidate"], r["n_train"], r["test_set"], r["channel"], r["horizon"])
            ]
            if base["verdict"] != r["verdict"]:
                changed.append(
                    {
                        "arm": arm,
                        "candidate": r["candidate"],
                        "n_train": r["n_train"],
                        "test_set": r["test_set"],
                        "channel": r["channel"],
                        "horizon": r["horizon"],
                        "a0": base["verdict"],
                        "arm_verdict": r["verdict"],
                    }
                )
    shared = sorted(set(a1_sizes) & set(n_sizes))
    seeds = {}
    for arm_name in ("A0", "A1", "N"):
        rows = []
        for cell in ranks[arm_name]:
            if cell["candidate"] in ("gru", "transformer"):
                rows.append(
                    seeds_row(
                        {
                            "channel": cell["channel"],
                            "test_set": cell["test_set"],
                            "n_train": cell["n_train"],
                            "candidate": cell["candidate"],
                            "reference": "lstm",
                            "per_seed_candidate": cell["per_seed_candidate"],
                            "per_seed_reference": cell["per_seed_reference"],
                        }
                    )
                )
        seeds[arm_name] = rows
    return {
        "a1_vs_a0": a1_vs,
        "n_vs_a0": n_vs,
        "d1_share": d1_share(a1, n_arm, a0, shared),
        "a1_cross_parts": cross_parts(a1, a0, a1_sizes),
        "n_variants_log_ratio_vs_a0": n_variants(n_arm, a0, n_sizes),
        "rank_cells": ranks,
        "rank_verdicts_changed_against_a0": changed,
        "multiplicity": {
            "a1_vs_a0": correction(a1_vs),
            "n_vs_a0": correction(n_vs),
            **{f"rank_{arm}": correction(rows) for arm, rows in ranks.items()},
        },
        "seeds_needed": seeds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix", required=True, help="merged files <prefix>_A1.jsonl and _N.jsonl"
    )
    parser.add_argument("--a1-sizes", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--n-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    arguments = parser.parse_args()
    require_clean_tree()
    a0 = load(RUNS_DIR / "arch_learning_curve_projected.jsonl") + load(
        RUNS_DIR / "arch_learning_curve_projected_extended.jsonl"
    )
    a1 = load(RUNS_DIR / f"{arguments.prefix}_A1.jsonl")
    n_arm = load(RUNS_DIR / f"{arguments.prefix}_N.jsonl")
    result = evaluate(a0, a1, n_arm, arguments.a1_sizes, arguments.n_sizes)
    write_artifact(
        f"rerun_evaluation_{arguments.prefix}",
        {"_provenance": provenance(), "prefix": arguments.prefix, **result},
    )
    for name, value in result["multiplicity"].items():
        print(f"{name}: {value}")
    print(f"rank verdicts that differ from A0: {len(result['rank_verdicts_changed_against_a0'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
