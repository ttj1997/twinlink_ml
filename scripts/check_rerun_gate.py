r"""Gate of the re-run (docs/15, section 4): arm A0 must reproduce the old records before any arm counts.

Compares the records of the pilot (or of the full run) with the records of Befunde 51/58
(``arch_learning_curve_projected.jsonl`` and ``..._extended.jsonl``) for the same ``master_seed``, ``n_train`` and test set:

1. Fingerprints of the data: ``process/fingerprint`` (equal or equivalent in the fingerprint history:
   the old records carry ``2d2679eeed716577``, the same configuration after the pseudonymisation cut of
   2026-09-25 is ``bddca37f716f5cfc``), ``excitation/fingerprint``, ``split/digest``, ``seed/*`` of arm A0
   must equal the old ones; ``measurement/digest`` only if the old record has one (it has not).
2. The baselines that do not depend on future inputs or on training randomness: persistence and the two
   fitted linear baselines (static, ARX): every common metric of the PRIMARY variant of arm A0 within
   ``--tolerance`` (relative, with an absolute floor of 1e-12; proposal of the plan: 1e-6).
3. Arm A1 has the same trajectories as A0: identical fingerprints (except ``window/future_actions``) and an
   identical persistence record (persistence reads the history, not the future inputs). Arm N must carry the
   new-study options and a different ``excitation/fingerprint``.

Exit status 1 and a clear message if a check fails; the plan then stops and the project lead is informed.

    tools/run_here.sh scripts/check_rerun_gate.py --prefix rerun_pilot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent

FINGERPRINTS = (
    "process/fingerprint",
    "excitation/fingerprint",
    "split/digest",
    "measurement/digest",
    "seed/data",
    "seed/noise",
    "seed/model",
)
GATED_MODELS = ("persistence", "static", "arx")
TIMING_KEYS = ("inference_us", "elapsed", "_time", "seconds")
"""Metric names that measure speed, not error: never compared."""


def load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def index(records: list[dict[str, Any]], *, primary_only: bool) -> dict[tuple, dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    for record in records:
        p = record["params"]
        if primary_only and not p.get("eval/primary", True):
            continue
        key = (p["model/label"], int(p["master_seed"]), int(p["n_train"]), p["test_set"])
        out[key] = record
    return out


def max_relative_deviation(a: dict[str, float], b: dict[str, float]) -> tuple[float, str, int]:
    worst, where, count = 0.0, "", 0
    for key in sorted(set(a) & set(b)):
        if any(token in key for token in TIMING_KEYS):
            continue
        x, y = a[key], b[key]
        if not (isinstance(x, int | float) and isinstance(y, int | float)):
            continue
        if math.isnan(x) and math.isnan(y):
            continue
        count += 1
        deviation = abs(x - y) / max(abs(y), 1e-12)
        if deviation > worst:
            worst, where = deviation, key
    return worst, where, count


def run_checks(
    old: list[dict[str, Any]], arms: dict[str, list[dict[str, Any]]], tolerance: float
) -> dict[str, Any]:
    failures: list[str] = []
    report: dict[str, Any] = {"tolerance": tolerance}
    old_index = index(old, primary_only=False)
    a0 = index(arms["A0"], primary_only=True)
    worst_overall = 0.0
    comparisons = []
    for key, record in sorted(a0.items()):
        label, seed, n_train, test_set = key
        old_record = old_index.get(key)
        if old_record is None:
            failures.append(f"no old record for {key}")
            continue
        for name in FINGERPRINTS:
            new_value, old_value = record["params"].get(name), old_record["params"].get(name)
            if old_value is None and name == "measurement/digest":
                continue  # not recorded before Befund 51 (the noise seed and the errors cover it)
            if name == "process/fingerprint":
                same = fingerprints_equivalent(str(new_value), str(old_value))
            else:
                same = new_value == old_value
            if not same:
                failures.append(f"{key}: {name} {new_value!r} != old {old_value!r}")
        if label in GATED_MODELS:
            worst, where, count = max_relative_deviation(record["metrics"], old_record["metrics"])
            comparisons.append(
                {"key": list(key), "max_relative_deviation": worst, "at": where, "metrics": count}
            )
            worst_overall = max(worst_overall, worst)
            if worst > tolerance:
                failures.append(f"{key}: {where} deviates by {worst:.3e} (> {tolerance:g})")
    report["a0_vs_old"] = {
        "cells": len(a0),
        "baseline_comparisons": comparisons,
        "worst": worst_overall,
    }

    if "A1" in arms:
        a1 = index(arms["A1"], primary_only=True)
        for key, record in sorted(a1.items()):
            twin = a0.get(key)
            if twin is None:
                failures.append(f"A1 cell {key} has no A0 counterpart")
                continue
            for name in FINGERPRINTS:
                if record["params"].get(name) != twin["params"].get(name):
                    failures.append(f"{key}: A1 {name} differs from A0")
            if key[0] == "persistence":
                worst, where, _ = max_relative_deviation(record["metrics"], twin["metrics"])
                if worst > tolerance:
                    failures.append(f"{key}: A1 persistence differs from A0 ({where}, {worst:.3e})")
            if record["params"].get("window/future_actions") != "planned":
                failures.append(f"{key}: A1 record lacks window/future_actions=planned")
    if "N" in arms:
        n_arm = index(arms["N"], primary_only=True)
        for key, record in sorted(n_arm.items()):
            twin = a0.get(key)
            p = record["params"]
            if (
                p.get("data/future_actions") != "planned"
                or p.get("data/solid_fraction_process") != "independent"
            ):
                failures.append(f"{key}: arm N lacks the new-study options")
            if twin is not None and p.get("excitation/fingerprint") == twin["params"].get(
                "excitation/fingerprint"
            ):
                failures.append(f"{key}: arm N has the excitation fingerprint of A0")
    report["failures"] = failures
    report["passed"] = not failures
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="merged files <prefix>_A0.jsonl, _A1, _N")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--artifact", default=None, help="artifact name (default <prefix>_gate)")
    arguments = parser.parse_args()
    require_clean_tree()
    old = load(RUNS_DIR / "arch_learning_curve_projected.jsonl") + load(
        RUNS_DIR / "arch_learning_curve_projected_extended.jsonl"
    )
    arms = {
        arm: load(RUNS_DIR / f"{arguments.prefix}_{arm}.jsonl")
        for arm in ("A0", "A1", "N")
        if (RUNS_DIR / f"{arguments.prefix}_{arm}.jsonl").exists()
    }
    if "A0" not in arms:
        raise SystemExit("arm A0 missing")
    report = run_checks(old, arms, arguments.tolerance)
    payload = {"_provenance": provenance(), "prefix": arguments.prefix, **report}
    write_artifact(arguments.artifact or f"{arguments.prefix}_gate", payload)
    print("GATE PASSED" if report["passed"] else "GATE FAILED")
    for failure in report["failures"][:40]:
        print("  ", failure)
    print(f"worst relative deviation of the gated baselines: {report['a0_vs_old']['worst']:.3e}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
