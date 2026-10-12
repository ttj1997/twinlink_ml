r"""Analysis of the B1 main run: tables, HB1-HB3 by the pre-registered criteria, information measures.

    tools/run_here.sh scripts/analyze_b1_main.py --log artifacts/runs/b1_main.jsonl --name b1_main_analysis

Writes ``artifacts/runs/<name>.json`` (never overwritten) and prints the tables: for every test set the
median primary metric per arm and N (LSTM and ARX), the value relative to persistence on that test
set, and the verdicts of HB1-HB3 with the number of tests behind each.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

from twinlink_ml.evaluation.b1_analysis import (
    PRIMARY_HORIZON,
    PRIMARY_TARGET,
    RecordTable,
    hb1,
    hb2,
    hb3,
    load_records,
    relative_to_persistence,
)

TEST_SETS = ("T1", "T2", "T3")
TARGETS = (PRIMARY_TARGET, "energy_target", "bed_temperature_celsius")


def _tables(table: RecordTable) -> tuple[dict, list[str]]:
    out: dict = {}
    lines: list[str] = []
    for target in TARGETS:
        for model in ("lstm", "arx"):
            for test_set in TEST_SETS:
                title = f"{target}, h{PRIMARY_HORIZON}, {model}, test set {test_set}"
                lines.append(f"\n{title}: median RMSE (physical) [x persistence]")
                header = "arm  " + "".join(f"{f'N={n}':>22}" for n in (2, 4, 8, 16, 32))
                lines.append(header)
                for arm in table.arms():
                    cells = []
                    for n in (2, 4, 8, 16, 32):
                        value = table.median(arm, n, model, test_set, target=target)
                        ratio = relative_to_persistence(
                            table, arm, n, model, test_set, target=target
                        )
                        out[f"{target}/{model}/{test_set}/{arm}/N{n}"] = {
                            "median": value,
                            "relative_to_persistence": ratio,
                        }
                        cells.append(f"{value:>12.4g} [{ratio:>6.2f}x]")
                    lines.append(f"{arm:<5}" + "".join(f"{c:>22}" for c in cells))
    return out, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--name", required=True)
    arguments = parser.parse_args()
    require_clean_tree()
    if (RUNS_DIR / f"{arguments.name}.json").exists():
        raise SystemExit("artifact exists; use a new --name")
    records = load_records(arguments.log)
    table = RecordTable(records)
    tables, lines = _tables(table)
    verdicts = {"HB1": hb1(table), "HB2": hb2(table), "HB3": hb3(table)}
    print("\n".join(lines))
    for name, verdict in verdicts.items():
        print(f"\n{name}: {verdict['verdict']} (tests: {verdict['n_tests']})")
    seeds = sorted({r["master_seed"] for r in records})
    write_artifact(
        arguments.name,
        {
            "_provenance": provenance(),
            "log": arguments.log.name,
            "n_records": len(records),
            "seeds": seeds,
            "primary": {"target": PRIMARY_TARGET, "horizon": PRIMARY_HORIZON},
            "verdicts": verdicts,
            "tables": tables,
        },
    )


if __name__ == "__main__":
    main()
