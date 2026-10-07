"""Stage order of the pre-heating cascade: where does a stage cool the gas (v4.1, intermediate).

No simulation, no optimisation. Two input sets, document model of
:mod:`twinlink_ml.partner_v4_1` (equal outlet temperatures):

* ``v4_1``: read from ``Model v4.1.xlsx`` (``local/`` or ``TWINLINK_LOCAL_DIR``; confidential),
* ``intermediate``: the intermediate version of the partner workbook, which is not available
  as a file; its inputs are set directly here (as communicated: dryer 1 100 kg/h, 6 g/kg,
  300 degC, spray 12 kg/h; dryer 2 50 kg/h, 6 g/kg, 150 degC, spray 3 kg/h; solid fraction 0.4,
  spray 15 degC; ambient 15 degC; circuits 40 / 20 kg/h; material properties as in v4.1).

For both: per case and stage the check whether the circuit arrives colder than the gas
(``stage_order_check``) and the proposed rules (``rule_comparison``; proposals, not part of the
workbook). The artifact is never overwritten (new name per run). Run from a clean working tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from twinlink_ml.partner_v4.inputs import ExcelDryerInputs, ExcelInputs, ExcelMaterialInputs
from twinlink_ml.partner_v4_1 import cascade as k
from twinlink_ml.partner_v4_1 import deviations as dev
from twinlink_ml.partner_v4_1.inputs import ExcelInputsV41, load_excel_inputs_v41

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", ROOT / "local"))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def intermediate_inputs(material: ExcelMaterialInputs) -> ExcelInputsV41:
    """Inputs of the intermediate version, set directly (no file)."""
    plant = ExcelInputs(
        dryer_1=ExcelDryerInputs(100.0, 6.0, 300.0, 12.0, 0.4, 15.0),
        dryer_2=ExcelDryerInputs(50.0, 6.0, 150.0, 3.0, 0.4, 15.0),
        material=material,
        ambient_temperature_celsius=15.0,
        ambient_pressure_atm=1.0,
    )
    return ExcelInputsV41(plant, (40.0, 20.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--name", default="partner_v4_1_stage_order")
    args = parser.parse_args()
    target = ROOT / "artifacts" / "runs" / f"{args.name}.json"
    if target.exists():
        print(f"{target} exists; choose a new --name (artifacts are never overwritten)")
        return 1
    workbook = args.workbook
    if workbook is None:
        workbook = next(
            p for p in (LOCAL / "Model v4.1.xlsx", LOCAL / "Model_v4_1.xlsx") if p.exists()
        )
    v41 = load_excel_inputs_v41(workbook)
    sets = {"v4_1": v41, "intermediate": intermediate_inputs(v41.plant.material)}
    record = {
        "provenance": {
            "git/commit": _git("rev-parse", "HEAD").strip(),
            "git/dirty": bool(_git("status", "--porcelain").strip()),
            "python": sys.version.split()[0],
        },
        "note": "intermediate = inputs set directly (the file is not available); rules are "
        "proposals, not part of the workbook",
        "inputs": {
            name: {
                "dryer_1": vars(i.plant.dryer_1),
                "dryer_2": vars(i.plant.dryer_2),
                "circuit_flows_kg_per_h": list(i.htm_flow_kg_per_h),
            }
            for name, i in sets.items()
        },
        "stage_order_check": {name: dev.stage_order_check(i) for name, i in sets.items()},
        "rule_comparison": {name: dev.rule_comparison(i) for name, i in sets.items()},
        "exhausts": {
            name: [
                {
                    "temperature_celsius": s.temperature_celsius,
                    "humidity_g_per_kg": s.humidity_kg_per_kg * 1e3,
                }
                for s in k._exhausts(i.to_si()).values()
            ]
            for name, i in sets.items()
        },
    }
    target.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    for name, rows in record["stage_order_check"].items():
        flagged = [
            (r["case"], r["source"], round(r["recovered_kw"], 4)) for r in rows if r["reverses"]
        ]
        print(name, "reversing stages:", flagged)
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
