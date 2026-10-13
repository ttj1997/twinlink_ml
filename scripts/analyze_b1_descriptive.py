r"""Descriptive analysis of the B1 main run (no new hypotheses), see ``evaluation/b1_descriptive.py``.

    tools/run_here.sh scripts/analyze_b1_descriptive.py --log artifacts/runs/b1_main.jsonl \
        --name b1_descriptive_analysis

Writes ``artifacts/runs/<name>.json`` (never overwritten): (1) rank correlations of the information
measures with the test errors, (2) paired 95 % intervals of the differences with the "differences larger
than X % are excluded" wording, (3) run equivalents against B1 with bootstrap intervals (primary target
and horizon; point estimates for the secondary ones), (4) the same for energy target, bed temperature and
horizon 1. Run from a clean tree.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

from twinlink_ml.evaluation.b1_analysis import RecordTable, load_records
from twinlink_ml.evaluation.b1_descriptive import (
    DEFAULT_COMBOS,
    INFO_EXPECTED_SIGN,
    bootstrap_run_equivalent,
    information_by_arm,
    pair_interval,
    rank_correlations,
    run_equivalent,
)

SIZES = (2, 4, 8, 16, 32)
TEST_SETS = ("T1", "T2", "T3")
MODELS = ("lstm", "arx")
PLAN_ARMS = ("b0", "b1", "b2", "b3", "b4")
ALL_ARMS = (*PLAN_ARMS, "b5", "b5r")
EXCITED_ARMS = ("b1", "b2", "b3", "b4", "b5", "b5r")  # without the constant operation B0
ARM_SETS = (("b0-b4", PLAN_ARMS), ("all", ALL_ARMS), ("excited", EXCITED_ARMS))
PAIRS = (("b2", "b1"), ("b3", "b1"), ("b4", "b1"), ("b5", "b1"), ("b5", "b5r"))
EQUIVALENT_ARMS = ("b0", "b2", "b3", "b4", "b5", "b5r")
PRIMARY = ("product_moisture_kg_per_kg", 24)


def _key(*parts: object) -> str:
    return "/".join(str(p) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    arguments = parser.parse_args()
    require_clean_tree()
    if (RUNS_DIR / f"{arguments.name}.json").exists():
        raise SystemExit("artifact exists; use a new --name")
    start = time.perf_counter()
    records = load_records(arguments.log)
    table = RecordTable(records)
    information = information_by_arm(records)

    correlations: dict = {}
    intervals: dict = {}
    equivalents: dict = {}
    for target, horizon in DEFAULT_COMBOS:
        combo = _key(target, f"h{horizon}")
        for model in MODELS:
            for test_set in TEST_SETS:
                for n in SIZES:
                    for label, arms in ARM_SETS:
                        correlations[_key(combo, model, test_set, f"N{n}", label)] = (
                            rank_correlations(
                                table,
                                information,
                                arms=arms,
                                n_train=n,
                                model=model,
                                test_set=test_set,
                                target=target,
                                horizon=horizon,
                            )
                        )
                    for candidate, reference in PAIRS:
                        if candidate == "b5" and reference == "b5r" and n == 2:
                            continue  # identical start pool by construction
                        intervals[_key(combo, model, test_set, f"N{n}", candidate, reference)] = (
                            pair_interval(
                                table,
                                candidate,
                                reference,
                                n,
                                model,
                                test_set,
                                target=target,
                                horizon=horizon,
                            )
                        )
                    for arm in EQUIVALENT_ARMS:
                        if arm in ("b5", "b5r") and n == 2:
                            continue
                        entry = run_equivalent(
                            table, arm, "b1", n, model, test_set, target=target, horizon=horizon
                        )
                        if (target, horizon) == PRIMARY:
                            entry["bootstrap"] = bootstrap_run_equivalent(
                                table,
                                arm,
                                "b1",
                                n,
                                model,
                                test_set,
                                target=target,
                                horizon=horizon,
                                n_boot=arguments.n_boot,
                            )
                        equivalents[_key(combo, model, test_set, f"N{n}", arm)] = entry
        print(f"{combo} done ({time.perf_counter() - start:.0f} s)", flush=True)

    # ---- compact printout of the primary quantity, LSTM
    combo = _key(*PRIMARY[:1], f"h{PRIMARY[1]}")
    print(
        "\n(1) seed-level Spearman of measure vs error (moisture h24, LSTM); expected sign in brackets"
    )
    for key, sign in INFO_EXPECTED_SIGN.items():
        cells = []
        for test_set in TEST_SETS:
            row = [
                correlations[_key(combo, "lstm", test_set, f"N{n}", "all")][key]["seed_spearman"]
                for n in SIZES
            ]
            cells.append(f"{test_set}: " + " ".join(f"{v:+.2f}" for v in row))
        print(f"  {key:<28}[{sign:+d}]  " + "   ".join(cells))
    print("\n(2) what the data exclude (moisture h24, LSTM): excluded above X % of the B1 error")
    for test_set in TEST_SETS:
        for candidate, reference in PAIRS:
            cells = []
            for n in SIZES:
                entry = intervals.get(_key(combo, "lstm", test_set, f"N{n}", candidate, reference))
                cells.append(
                    "   -   "
                    if not entry or entry.get("degenerate")
                    else f"{entry['relative_low_percent']:+.0f}..{entry['relative_high_percent']:+.0f}%"
                )
            print(
                f"  {test_set} {candidate} vs {reference}: " + "  ".join(f"{c:>13}" for c in cells)
            )
    print("\n(3) run equivalent against B1 (moisture h24, LSTM): extra runs [95 % bootstrap]")
    for test_set in TEST_SETS:
        for arm in EQUIVALENT_ARMS:
            cells = []
            for n in SIZES:
                entry = equivalents.get(_key(combo, "lstm", test_set, f"N{n}", arm))
                if not entry:
                    cells.append("-")
                    continue
                boot = entry["bootstrap"]
                extra = entry["extra_runs"]
                text = f">{SIZES[-1] - n}" if math.isinf(extra) else f"{extra:+.1f}"
                cells.append(f"{text} [{boot['extra_low']:+.0f},{boot['extra_high']:+.0f}]")
            print(f"  {test_set} {arm}: " + "  ".join(f"{c:>20}" for c in cells))

    write_artifact(
        arguments.name,
        {
            "_provenance": provenance(),
            "log": arguments.log.name,
            "n_records": len(records),
            "seeds": sorted({r["master_seed"] for r in records}),
            "sizes": list(SIZES),
            "n_boot": arguments.n_boot,
            "primary": {"target": PRIMARY[0], "horizon": PRIMARY[1]},
            "information_measures": INFO_EXPECTED_SIGN,
            "rank_correlations": correlations,
            "paired_intervals": intervals,
            "run_equivalents": equivalents,
            "wall_s": time.perf_counter() - start,
        },
    )


if __name__ == "__main__":
    main()
