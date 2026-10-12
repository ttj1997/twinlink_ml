"""Deviation list and proposed reference values for the partner workbook ``Model_v4.xlsx``.

No simulation, no optimisation: reads the workbook (path argument, default
``local/Model_v4.xlsx``; the file is confidential and never committed), rebuilds it with
:mod:`twinlink_ml.partner_v4` and writes one JSON artifact:

* ``deviations``: workbook cell, workbook value, rebuilt value, ratio, cause, hand calculation,
* ``proposal``: values for the open cells (``#REF!`` or empty) of the sheet ``Scenarios`` --
  OUR RECONSTRUCTION (assumptions A1-A5 in :mod:`twinlink_ml.partner_v4.cascade`), in the
  variants sheet structure / counter-current dT_min 0 K / counter-current dT_min 20 K.

The artifact is never overwritten (new name per run). Run from a clean working tree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from twinlink_ml.partner_v4.cascade import CASE_COLUMNS, evaluate_case, proposed_cell_values
from twinlink_ml.partner_v4.deviations import CELLS_TO_READ, build_deviations
from twinlink_ml.partner_v4.inputs import load_excel_inputs
from twinlink_ml.partner_v4.workbook import read_cells
from twinlink_ml.utils.code_location import require_code_location

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=ROOT / "local" / "Model_v4.xlsx")
    parser.add_argument("--name", default="partner_v4_deviations")
    args = parser.parse_args()
    target = ROOT / "artifacts" / "runs" / f"{args.name}.json"
    if target.exists():
        print(f"{target} exists; choose a new --name (artifacts are never overwritten)")
        return 1

    excel = load_excel_inputs(args.workbook)
    si = excel.to_si()
    cells = read_cells(args.workbook, CELLS_TO_READ)
    deviations = build_deviations(excel, cells)

    variants = {
        "sheet_structure": {"mode": "sheet", "dt_min_k": 0.0},
        "counter_current_dt0": {"mode": "counter_current", "dt_min_k": 0.0},
        "counter_current_dt20": {"mode": "counter_current", "dt_min_k": 20.0},
    }
    proposal = {}
    for name, kwargs in variants.items():
        proposal[name] = {
            "cells": proposed_cell_values(si, **kwargs),  # type: ignore[arg-type]
            "total_gas_kg_per_h": {
                str(case): evaluate_case(si, case, **kwargs).total_gas_kg_per_h  # type: ignore[arg-type]
                for case in (0, *CASE_COLUMNS)
            },
        }
    record = {
        "provenance": {
            "git/commit": _git("rev-parse", "HEAD").strip(),
            "git/dirty": bool(_git("status", "--porcelain").strip()),
            "python": sys.version.split()[0],
            **require_code_location(ROOT),
        },
        "note": "proposal = OUR RECONSTRUCTION (assumptions A1-A5), not the partner's numbers; "
        "T_cold = ambient 15 degC is a placeholder; case 0 = no recovery",
        "deviations": [d.as_dict() for d in deviations],
        "proposal": proposal,
    }
    target.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    for d in deviations:
        ratio = "" if d.ratio is None else f"{d.ratio:.4g}"
        print(
            f"{d.item:8s} {d.cell:42s} {d.workbook_value!s:>14.14} {d.rebuilt_value!s:>14.14} {ratio}"
        )
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
