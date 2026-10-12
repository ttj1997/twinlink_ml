r"""Addendum to Befund 83: multiplicity correction as a sensitivity, and seeds needed (paired design).

Pure evaluation of the artifact ``paired_reevaluation_58_63_67.json`` (no training, no new data).
Written and committed before the run. Rules fixed here:

(a) MULTIPLICITY. The p values are those of the primary paired test of Befund 83 (two-sided, mean of the
differences). Holm (family-wise error 5 %) and Benjamini-Hochberg (false discovery rate 5 %) are applied
to families that are stated, not chosen afterwards:
  * ``all``       every unique test of the artifact;
  * ``b63_67``    the cells of Befund 63 / 67 (SX1 cells, epoch-budget cells, SX3 cells);
  * ``b58``       the budget-range comparisons and the pairwise comparisons of Befund 58;
  * ``gru_small`` the 18 GRU-vs-LSTM cells at n_train 8, 16, 32 (the four cells that are resolved);
  * ``gru_all``   the 24 GRU-vs-LSTM cells at n_train 8, 16, 32, 64.
A test is "unique" if (unordered pair of methods, channel, test set, n_train, epoch budget) is new:
the repeated 200-epoch cells and the mirrored pairwise comparisons count once. Comparisons made but not
stored in the artifact (the 72 Transformer/xLSTM/Mamba-vs-GRU comparisons, none of them resolved) are not
in any family, which makes the families slightly smaller than the 632 tests of Befund 83 and the
correction slightly lenient; stated here. A resolved verdict "survives" if the adjusted p value is below
0.05. Nothing is re-tested; no verdict is changed in the lab book by this.

(b) SEEDS NEEDED. For every cell of Befund 58 (GRU and Transformer against LSTM, n_train 8-64, 3 channels,
2 test sets) the standard deviation s_d of the observed paired differences (five seeds) gives the number of
seeds n at which a paired t test (two-sided, 5 %) has power >= 0.80 for a true difference of delta times
the mean of the reference, delta = 12.5 % and 10 % (the M2 target and the lower end of its 10-15 % range),
from the noncentral t distribution. The estimate of s_d rests on five seeds: the interval is that for
s_d from its chi-square distribution (4 degrees of freedom, 80 % two-sided bounds). A multiplicative
version (differences of log values, delta as log(1 + delta)) is reported next to it. Both are
planning figures, not tests; a cell with s_d = 0 gets no figure. Replaces the "41 seeds" of Befund 63 h
(unpaired, 50 % power, LSTM spread only).

    tools/run_here.sh scripts/analyze_83_addendum.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

ALPHA = 0.05
POWER = 0.80
DELTAS = (0.125, 0.10)
MAX_SEEDS = 5000
OUTPUT_NAME = "paired_reevaluation_83_addendum"


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p values."""
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH adjusted p values (q values)."""
    m = len(p)
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m)
    running = 1.0
    for position, index in enumerate(order):
        rank = m - position
        running = min(running, p[index] * m / rank)
        adjusted[index] = running
    return adjusted


def _unique(cells: list[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cell in cells:
        epochs = 600 if "600" in str(cell.get("source", "")) else 200
        key = (
            tag,
            frozenset((cell["candidate"], cell["reference"])),
            cell["channel"],
            cell["test_set"],
            cell["n_train"],
            epochs,
            cell.get("source", "")[:20] if tag == "b67" else "",
        )
        seen.setdefault(key, {**cell, "epochs": epochs})
    return list(seen.values())


def family_report(name: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    p = np.array([1.0 if math.isnan(c["p_value"]) else c["p_value"] for c in cells])
    holm_p = holm(p)
    bh_p = benjamini_hochberg(p)
    resolved = np.array([c["verdict"] in ("candidate better", "candidate worse") for c in cells])
    rows = []
    for cell, raw, h, b, res in zip(cells, p, holm_p, bh_p, resolved, strict=True):
        rows.append(
            {
                "channel": cell["channel"],
                "test_set": cell["test_set"],
                "n_train": cell["n_train"],
                "candidate": cell["candidate"],
                "reference": cell["reference"],
                "source": cell.get("source", ""),
                "verdict": cell["verdict"],
                "p": float(raw),
                "p_holm": float(h),
                "p_bh": float(b),
                "survives_holm": bool(res and h < ALPHA),
                "survives_bh": bool(res and b < ALPHA),
            }
        )
    return {
        "family": name,
        "tests": len(cells),
        "resolved_uncorrected": int(resolved.sum()),
        "resolved_holm": int(sum(r["survives_holm"] for r in rows)),
        "resolved_bh": int(sum(r["survives_bh"] for r in rows)),
        "cells": rows,
    }


def power_paired(effect_over_sd: float, n: int) -> float:
    """Power of the two-sided paired t test (noncentral t).

    ``scipy.stats.nct`` returns nan for large noncentrality (about 8 and above); there the power is
    above 0.9999 for every n >= 3 and is set to 1.0 (a first run without this rule skipped those
    cases and gave wrong seed numbers; it was discarded, not committed).
    """
    df = n - 1
    critical = stats.t.ppf(1.0 - ALPHA / 2.0, df)
    noncentrality = effect_over_sd * math.sqrt(n)
    power = float(
        stats.nct.sf(critical, df, noncentrality) + stats.nct.cdf(-critical, df, noncentrality)
    )
    if math.isnan(power):
        return 1.0 if noncentrality > 8.0 and n >= 3 else 0.0
    return power


def seeds_needed(effect: float, sd: float) -> int | None:
    if sd <= 0.0 or effect <= 0.0:
        return None
    ratio = effect / sd
    for n in range(2, MAX_SEEDS + 1):
        if power_paired(ratio, n) >= POWER:
            return n
    return MAX_SEEDS  # lower bound: not reached


def seeds_row(cell: dict[str, Any]) -> dict[str, Any]:
    cand = np.array(cell["per_seed_candidate"])
    ref = np.array(cell["per_seed_reference"])
    diff = ref - cand
    sd = float(diff.std(ddof=1))
    mean_ref = float(ref.mean())
    log_sd = float(np.log(cand / ref).std(ddof=1))
    df = len(diff) - 1
    lower = sd * math.sqrt(df / stats.chi2.ppf(0.90, df))
    upper = sd * math.sqrt(df / stats.chi2.ppf(0.10, df))
    row: dict[str, Any] = {
        "channel": cell["channel"],
        "test_set": cell["test_set"],
        "n_train": cell["n_train"],
        "candidate": cell["candidate"],
        "reference": cell["reference"],
        "sd_of_differences": sd,
        "mean_reference": mean_ref,
        "sd_relative_to_reference_mean": sd / abs(mean_ref),
        "sd_of_log_ratio": log_sd,
    }
    for delta in DELTAS:
        tag = f"{delta:.3f}"
        row[f"seeds_{tag}"] = seeds_needed(delta * abs(mean_ref), sd)
        row[f"seeds_{tag}_sd_low"] = seeds_needed(delta * abs(mean_ref), lower)
        row[f"seeds_{tag}_sd_high"] = seeds_needed(delta * abs(mean_ref), upper)
        row[f"seeds_{tag}_log"] = seeds_needed(math.log1p(delta), log_sd)
    return row


def main() -> int:
    require_clean_tree()
    art = json.loads((RUNS_DIR / "paired_reevaluation_58_63_67.json").read_text(encoding="utf-8"))
    for part in (
        "part_63_67_sx1_cells",
        "part_67_sx3_cells",
        "part_58_cells",
        "part_58_pairwise_cells",
    ):
        if not art[part]:
            raise SystemExit(f"empty part {part}")
    b63 = _unique(art["part_63_67_sx1_cells"], "b63")
    b67 = _unique(art["part_67_sx3_cells"], "b67")
    b58 = _unique(art["part_58_cells"] + art["part_58_pairwise_cells"], "b58")
    everything = _unique(b63 + b67 + b58, "all")
    gru = [c for c in art["part_58_cells"] if c["candidate"] == "gru" and c["reference"] == "lstm"]
    families = [
        family_report("all", everything),
        family_report("b63_67", b63 + b67),
        family_report("b58", b58),
        family_report("gru_small", [c for c in gru if c["n_train"] in (8, 16, 32)]),
        family_report("gru_all", gru),
    ]
    seed_rows = [
        seeds_row(c)
        for c in art["part_58_cells"]
        if c["reference"] == "lstm" and c["candidate"] in ("gru", "transformer")
    ]
    payload = {
        "_provenance": provenance(),
        "rule": "see the module docstring of scripts/analyze_83_addendum.py",
        "alpha": ALPHA,
        "power": POWER,
        "deltas": list(DELTAS),
        "families": families,
        "seeds_needed": seed_rows,
    }
    write_artifact(OUTPUT_NAME, payload)
    for fam in families:
        print(
            f"{fam['family']:10s} tests {fam['tests']:4d}  resolved {fam['resolved_uncorrected']:4d}"
            f"  Holm {fam['resolved_holm']:4d}  BH {fam['resolved_bh']:4d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
