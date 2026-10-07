"""Three-way comparison (equation document / workbook / rebuild) for ``Model_v4.1.xlsx``.

No simulation, no optimisation: reads the workbook (``Model v4.1.xlsx`` in ``local/``, or
``--workbook``; the directory can be named with ``TWINLINK_LOCAL_DIR``; the file is confidential
and never committed), rebuilds it with :mod:`twinlink_ml.partner_v4_1` and writes one JSON
artifact with

* ``cells``: every numeric formula cell of ``Scenarios``, ``Balances`` and ``Balance``: Excel's
  stored value, the as-built reproduction, the value following the equation document, cause,
* ``summary``: counts, the largest reproduction error, which rows differ from the document,
* ``gas``: the gas formula decomposed (hand calculation),
* ``document``: typos of the equation document and which cells realise which equation,
* ``workbook``: references (Inputs column K, the sheets ``Balance``/``Balances``), the inputs
  against the previous version ``Model_v4.xlsx``,
* ``cases``: the six cases from the document's equations (circuit flows as in the sheet),
* ``equal_outlet``: the assumption of equal outlet temperatures -- counter-current bound,
  case 6 in reversed stage order, scan over the circuit flow (negative recovery).

The artifact is never overwritten (new name per run). Run from a clean working tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from twinlink_ml.partner_v4_1 import as_built as ab
from twinlink_ml.partner_v4_1 import cascade as k
from twinlink_ml.partner_v4_1 import deviations as dev
from twinlink_ml.partner_v4_1 import layout as lay
from twinlink_ml.partner_v4_1.inputs import load_excel_inputs_v41

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", ROOT / "local"))
FLOWS_KG_PER_H = (5.0, 20.0, 50.0, 100.0, 150.0, 200.0, 500.0, 1000.0)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def _find(*names: str) -> Path:
    for name in names:
        if (LOCAL / name).exists():
            return LOCAL / name
    raise FileNotFoundError(f"none of {names} in {LOCAL}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--previous", type=Path, default=None, help="Model_v4.xlsx")
    parser.add_argument("--name", default="partner_v4_1_comparison")
    args = parser.parse_args()
    target = ROOT / "artifacts" / "runs" / f"{args.name}.json"
    if target.exists():
        print(f"{target} exists; choose a new --name (artifacts are never overwritten)")
        return 1
    workbook = args.workbook or _find("Model v4.1.xlsx", "Model_v4_1.xlsx")
    previous = args.previous or _find("Model_v4.xlsx", "Model v4.xlsx")

    inputs = load_excel_inputs_v41(workbook)
    si = inputs.to_si()
    cases = {n: k.to_case_cells(k.evaluate_case(si, n)) for n in range(1, 7)}
    document_scenarios = lay.scenario_cells(k.plant_cells(si), cases)
    document_balances = k.balances_cells(si)
    built_balances = ab.balances_cells_as_built(inputs)

    comparisons = []
    comparisons += dev.compare_sheet(
        "Scenarios",
        dev.read_formula_cells(workbook, "Scenarios"),
        ab.scenario_cells_as_built(inputs),
        document_scenarios,
    )
    comparisons += dev.compare_sheet(
        "Balances",
        dev.read_formula_cells(workbook, "Balances"),
        built_balances,
        document_balances,
    )
    comparisons += dev.compare_sheet(
        "Balance",
        dev.read_formula_cells(workbook, "Balance"),
        {a: v for a, v in built_balances.items() if re.fullmatch(r"[A-F]\d+", a)},
        {a: v for a, v in document_balances.items() if re.fullmatch(r"[A-F]\d+", a)},
    )
    worst = max((c.as_built_vs_excel or 0.0) for c in comparisons if c.as_built is not None)
    differing = [
        c for c in comparisons if c.document_vs_excel and abs(c.document_vs_excel - 1) > 1e-9
    ]
    summary = {
        "cells_compared": len(comparisons),
        "cells_without_as_built": [
            f"{c.sheet}!{c.cell}" for c in comparisons if c.as_built is None
        ],
        "largest_as_built_error_relative": worst,
        "cells_where_document_differs": len(differing),
        "rows_where_document_differs": sorted(
            {int(re.sub(r"[A-Z]+", "", c.cell)) for c in differing}
        ),
        "document_vs_excel_factors": sorted(
            {round(c.document_vs_excel or 0.0, 6) for c in differing}
        ),
    }

    from twinlink_ml.partner_v4.workbook import read_cells

    k_cells = read_cells(workbook, [f"Inputs!K{r}" for r in range(3, 9)])
    case_rows = {}
    for n in range(0, 7):
        result = k.evaluate_case(si, n)
        case_rows[str(n)] = {
            "gas_document_kg_per_h": result.total_gas_kg_per_h,
            "recovered_kw": result.total_recovered_w / 1e3,
            "stages": [
                {
                    "source": s.source,
                    "sink_in_celsius": s.sink_in_celsius,
                    "sink_out_celsius": s.sink_out_celsius,
                    "exhaust_out_celsius": s.exhaust_out_celsius,
                    "recovered_kw": s.recovered_w / 1e3,
                    "dew_point_celsius": s.exhaust_dew_point_celsius,
                    "condenses": s.condenses,
                }
                for s in result.stages
            ],
            "heater_in_celsius": [result.sink_1.heater_in_celsius, result.sink_2.heater_in_celsius],
            "workbook_total_kg_per_h": (
                dev.read_formula_cells(workbook, "Scenarios")[f"{lay.CASE_COLUMNS[n][0]}41"][1]
                if n
                else None
            ),
        }

    reversed_6 = k.evaluate_case(si, 6, order=((), (2, 1)))
    drawn_6 = k.evaluate_case(si, 6)
    record = {
        "provenance": {
            "git/commit": _git("rev-parse", "HEAD").strip(),
            "git/dirty": bool(_git("status", "--porcelain").strip()),
            "python": sys.version.split()[0],
        },
        "note": "document = values following the partner's equation document (typos corrected); "
        "as_built = the sheet's formulas as they are; excel = the value stored in the workbook",
        "summary": summary,
        "cells": [c.as_dict() for c in comparisons],
        "gas": {
            "case_100_kg_per_h_15_to_300": dev.gas_hand_calculation(inputs, 100.0, 15.0, 300.0),
            "case_1_heater_1_after_recovery": dev.gas_hand_calculation(
                inputs, 100.0, cases[1].sink_1.heater_in_celsius, 150.0
            ),
        },
        "document": {
            "typos": dev.DOCUMENT_TYPOS,
            "equation_to_cells": dev.EQUATION_TO_CELLS,
        },
        "workbook": {
            "references": dev.reference_scan(workbook),
            "inputs_column_k": {c.address: c.value for c in k_cells.values()},
            "inputs_diff_previous_to_new": dev.inputs_diff(previous, workbook),
            "circuit_flows_kg_per_h": list(inputs.htm_flow_kg_per_h),
        },
        "cases": case_rows,
        "equal_outlet": {
            "assumption": "both fluids leave an exchanger at the same temperature = parallel flow "
            "with unlimited area (document, 2.3/2.4 and 4.1/4.2); Q = (T_c,in - T_h,in) / "
            "(1/A + 1/B + 1/C) per stage",
            "counter_current_bound": dev.equal_outlet_vs_counter_current(inputs),
            "case_6": {
                "as_drawn_order_1_2": {
                    "heater_in_celsius": drawn_6.sink_2.heater_in_celsius,
                    "gas_kg_per_h": drawn_6.total_gas_kg_per_h,
                    "stage_kw": [s.recovered_w / 1e3 for s in drawn_6.stages],
                },
                "reversed_order_2_1": {
                    "heater_in_celsius": reversed_6.sink_2.heater_in_celsius,
                    "gas_kg_per_h": reversed_6.total_gas_kg_per_h,
                    "stage_kw": [s.recovered_w / 1e3 for s in reversed_6.stages],
                },
            },
            "break_even_htm_flow_kg_per_h": {
                "case_3": dev.break_even_htm_flow(inputs, 3),
                "case_4": dev.break_even_htm_flow(inputs, 4),
                "case_5": dev.break_even_htm_flow(inputs, 5),
                "case_6": dev.break_even_htm_flow(inputs, 6),
                "case_5_reversed": dev.break_even_htm_flow(inputs, 5, ((2, 1), ())),
                "case_6_reversed": dev.break_even_htm_flow(inputs, 6, ((), (2, 1))),
            },
            "scan_over_htm_flow": dev.negative_recovery_scan(inputs, FLOWS_KG_PER_H),
        },
    }
    target.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
