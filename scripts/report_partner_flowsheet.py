r"""Partner flowsheet: reproduction checks and the switch-by-switch decomposition (docs/11, section 11).

The units of the partner model (``Model_v4.1.xlsx``) as exchangeable units with one switch per
assumption (``twinlink_ml.flowsheet``). This script writes one artifact with

* ``excel_reproduction`` -- all switches ``"partner"`` against every stored number of the sheet
  ``Scenarios`` (needs the workbook from ``local/``; never copied),
* ``befund_90_reproduction`` -- all switches ``"physical"`` (and ``"partner"``) with the exhausts
  pinned to the source variants of Befund 90, against its artifact,
* ``walks`` -- from ``"partner"`` to ``"physical"``, one switch per step, in the order of the task
  and in the reverse order, in both conventions of ``dT_min`` (main: per exchanger),
* ``single_switch`` -- each switch alone from all-partner and each switch alone back from
  all-physical (order-free view),
* ``physical_with_dryer_2_pinned_to_the_bed`` -- sensitivity: all physical, dryer 2 pinned to the
  laboratory model's state (the remaining dryer constants of docs/11, 9.6 are not switches).

The artifact is never overwritten (new name per run). Run from a clean working tree, started with
``tools/run_here.sh``; the workbook comes from ``$TWINLINK_LOCAL_DIR`` or ``<root>/local``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

from twinlink_ml.flowsheet import (
    SWITCH_NAMES,
    Flowsheet,
    FlowsheetConfig,
    PhysicalParameters,
    Switches,
)
from twinlink_ml.flowsheet.bed_state import BedState, bed_state_at_partner_inputs
from twinlink_ml.flowsheet.decomposition import Step, rank_groups, ranking_text, walk
from twinlink_ml.flowsheet.flowsheet import SCENARIOS
from twinlink_ml.flowsheet.sheet import sheet_scenario_cells
from twinlink_ml.partner_v4_1 import deviations as dev
from twinlink_ml.partner_v4_1.inputs import ExcelInputsV41, load_excel_inputs_v41
from twinlink_ml.utils.code_location import require_code_location

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", ROOT / "local"))
BEFUND_90 = ROOT / "artifacts" / "runs" / "heat_recovery_comparison_r2.json"
CONVENTIONS = ("per_exchanger", "through_circuit")
DT_MIN_K = (20.0, 10.0, 5.0)
VARIANTS = ("i_partner_exhausts", "ii_laboratory_model_dryer_2")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def _find(*names: str) -> Path:
    for name in names:
        if (LOCAL / name).exists():
            return LOCAL / name
    raise FileNotFoundError(f"none of {names} in {LOCAL}")


def _config(
    inputs: ExcelInputsV41,
    switches: Switches,
    convention: str,
    bed: BedState | None,
    dt_min_k: float = 10.0,
    pinned: dict[int, tuple[float, float]] | None = None,
) -> FlowsheetConfig:
    return FlowsheetConfig(
        inputs.to_si(),
        switches,
        PhysicalParameters(dt_min_k=dt_min_k, convention=convention),  # type: ignore[arg-type]
        bed=bed,
        pinned_exhausts=pinned,
    )


def excel_reproduction(inputs: ExcelInputsV41, workbook: Path) -> dict[str, Any]:
    """All ``"partner"`` against the stored numbers of ``Scenarios`` (as built, gas formula included)."""
    cells = sheet_scenario_cells(
        Flowsheet(_config(inputs, Switches.all_partner(), "per_exchanger", None))
    )
    excel = dev.read_formula_cells(workbook, "Scenarios")
    worst_rel, worst_cell = 0.0, ""
    worst_abs = 0.0
    for address, (_, stored) in excel.items():
        diff = abs(cells[address] - stored)
        rel = diff / max(abs(stored), 1e-300)
        worst_abs = max(worst_abs, diff)
        if rel > worst_rel:
            worst_rel, worst_cell = rel, address
    corrected = sheet_scenario_cells(
        Flowsheet(_config(inputs, Switches(gas_formula="physical"), "per_exchanger", None))
    )
    gas_rows = {a for a in excel if int("".join(ch for ch in a if ch.isdigit())) in (26, 39, 41)}
    return {
        "workbook": workbook.name,
        "cells": len(excel),
        "same_cell_set": set(cells) == set(excel),
        "largest_relative_difference": worst_rel,
        "largest_relative_difference_cell": worst_cell,
        "largest_absolute_difference": worst_abs,
        "tolerance_of_the_workbook_tests": {"rel": 1e-12, "abs": 1e-12},
        "gas_formula_corrected": {
            "cells_that_differ_from_excel": sorted(
                a
                for a, (_, v) in excel.items()
                if abs(corrected[a] - v) > 1e-9 * max(abs(v), 1e-12)
            ),
            "ratio_excel_to_corrected_in_gas_rows": {
                a: excel[a][1] / corrected[a] for a in sorted(gas_rows)
            },
        },
    }


def _pins(artifact: dict[str, Any], variant: str) -> dict[int, tuple[float, float]]:
    ex = artifact["source_variants"][variant]["exhausts"]
    return {int(k): (v["temperature_celsius"], v["humidity_g_per_kg"] / 1e3) for k, v in ex.items()}


def befund_90_reproduction(inputs: ExcelInputsV41) -> dict[str, Any]:
    """Largest deviation from the stored columns a, b, c of Befund 90 (pinned exhausts)."""
    ref = json.loads(BEFUND_90.read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "artifact": BEFUND_90.name,
        "git/commit_of_the_reference": ref["provenance"]["git/commit"],
    }
    worst: dict[str, float] = {"recovered_kw": 0.0, "gas_kg_per_h": 0.0, "stage_kw": 0.0}
    rows = 0

    def compare(fs: Flowsheet, stored: dict[str, Any]) -> None:
        nonlocal rows
        for case in SCENARIOS:
            r = fs.evaluate(case)
            s = stored[str(case)]
            worst["recovered_kw"] = max(
                worst["recovered_kw"], abs(r.recovered_kw - s["recovered_kw"])
            )
            worst["gas_kg_per_h"] = max(
                worst["gas_kg_per_h"], abs(r.fuel_kg_per_h - s["gas_kg_per_h"])
            )
            for mine, theirs in zip(r.stages, s["stages"], strict=True):
                worst["stage_kw"] = max(
                    worst["stage_kw"], abs(mine.q_w / 1e3 - theirs["recovered_kw"])
                )
            rows += 1

    for variant in VARIANTS:
        pins = _pins(ref, variant)
        wu1 = ref["source_variants"][variant]["wu1"]
        compare(
            Flowsheet(
                _config(
                    inputs, Switches(gas_formula="physical"), "per_exchanger", None, pinned=pins
                )
            ),
            wu1["a"],
        )
        for dt in DT_MIN_K:
            for convention in CONVENTIONS:
                key = f"dt{dt:g}" + ("_per_exchanger" if convention == "per_exchanger" else "")
                for model in ("b", "c"):
                    sw = Switches(
                        exchanger="physical",
                        cp_gas="physical",
                        condensation="physical" if model == "c" else "partner",
                        gas_formula="physical",
                    )
                    compare(
                        Flowsheet(_config(inputs, sw, convention, None, dt, pins)),
                        wu1[key][model],
                    )
    out["scenario_rows_compared"] = rows
    out["largest_abs_difference"] = worst
    out["mapping"] = (
        "a: all partner, gas_formula physical; b: exchanger, cp_gas, gas_formula physical; "
        "c: b + condensation physical; Befund 90 main convention = through_circuit"
    )
    return out


def _case_row(step: Step) -> dict[str, Any]:
    return step.as_dict()


def single_switch(inputs: ExcelInputsV41, bed: BedState, convention: str) -> dict[str, Any]:
    """Each switch alone from all-partner, and each alone back to partner from all-physical."""
    out: dict[str, Any] = {"from_partner": {}, "from_physical": {}}
    base_p = Flowsheet(_config(inputs, Switches.all_partner(), convention, bed)).evaluate_all()
    base_q = Flowsheet(_config(inputs, Switches.all_physical(), convention, bed)).evaluate_all()
    for name in SWITCH_NAMES:
        for key, start, value, base in (
            ("from_partner", Switches.all_partner(), "physical", base_p),
            ("from_physical", Switches.all_physical(), "partner", base_q),
        ):
            sw = start.with_(name, value)  # type: ignore[arg-type]
            res = Flowsheet(_config(inputs, sw, convention, bed)).evaluate_all()
            out[key][name] = {
                str(case): {
                    "recovered_kw": res[case].recovered_kw,
                    "delta_recovered_kw": res[case].recovered_kw - base[case].recovered_kw,
                    "fuel_corrected_kg_per_h": res[case].fuel_corrected_kg_per_h,
                    "delta_fuel_corrected_kg_per_h": res[case].fuel_corrected_kg_per_h
                    - base[case].fuel_corrected_kg_per_h,
                    "co2_kg_per_h": res[case].co2_kg_per_h,
                    "delta_co2_kg_per_h": res[case].co2_kg_per_h - base[case].co2_kg_per_h,
                    "htm_boiling_stages": res[case].htm_boiling_stages,
                }
                for case in SCENARIOS
            }
    return out


def ranking_summary(steps: list[Step]) -> dict[str, Any]:
    """State whether the ranking of the scenarios holds in every intermediate step."""
    first, last = steps[0], steps[-1]
    return {
        "ranking_partner": ranking_text(first.ranking_co2),
        "ranking_physical": ranking_text(last.ranking_co2),
        "steps": [
            {
                "step": s.index,
                "changed": s.changed,
                "ranking_co2": ranking_text(s.ranking_co2),
                "same_as_partner": s.ranking_co2 == first.ranking_co2,
                "same_as_physical": s.ranking_co2 == last.ranking_co2,
                "ranking_fuel_corrected": ranking_text(s.ranking_fuel),
                "ranking_recovery": ranking_text(s.ranking_recovery),
                "gap_between_groups_co2_kg_per_h": _gaps(s),
            }
            for s in steps
        ],
        "holds_in_every_step": all(s.ranking_co2 == first.ranking_co2 for s in steps),
    }


def _gaps(step: Step) -> list[float]:
    """CO2 differences between neighbouring rank groups (how far apart the scenarios are)."""
    values = step.metric("co2_kg_per_h")
    groups = rank_groups(values)
    return [values[b[0]] - values[a[0]] for a, b in pairwise(groups)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--name", default="partner_flowsheet_decomposition")
    args = parser.parse_args()
    target = ROOT / "artifacts" / "runs" / f"{args.name}.json"
    if target.exists():
        print(f"{target} exists; choose a new --name (artifacts are never overwritten)")
        return 1
    workbook = args.workbook or _find("Model v4.1.xlsx", "Model_v4_1.xlsx")
    inputs = load_excel_inputs_v41(workbook)
    bed = bed_state_at_partner_inputs()
    ref90 = json.loads(BEFUND_90.read_text(encoding="utf-8"))
    pins_bed = {2: (bed.exhaust_temperature_celsius, bed.exhaust_humidity_kg_per_kg)}

    record: dict[str, Any] = {
        "provenance": {
            "git/commit": _git("rev-parse", "HEAD").strip(),
            "git/dirty": bool(_git("status", "--porcelain").strip()),
            "python": sys.version.split()[0],
            **require_code_location(ROOT),
        },
        "parameters": {
            "switch_order_forward": list(SWITCH_NAMES),
            "switch_order_reverse": list(reversed(SWITCH_NAMES)),
            "main_convention": "per_exchanger (dT_min per exchanger)",
            "sensitivity_convention": "through_circuit (dT_min through the circuit, Befund 90)",
            "physical_parameters": dataclasses.asdict(PhysicalParameters()),
            "auxiliary_placeholders": dataclasses.asdict(
                Flowsheet(_config(inputs, Switches(), "per_exchanger", None)).config.auxiliary
            ),
            "ranking_metric": "CO2 of the corrected fuel + placeholder electricity, lowest first",
            "workbook": workbook.name,
            "bed_state_label": bed.label,
        },
        "inputs_sheet_units": {
            "plant": dataclasses.asdict(inputs.plant),
            "htm_flow_kg_per_h": list(inputs.htm_flow_kg_per_h),
        },
        "bed_state": bed.as_dict(),
        "excel_reproduction": excel_reproduction(inputs, workbook),
        "befund_90_reproduction": befund_90_reproduction(inputs),
        "walks": {},
        "single_switch": {},
        "physical_with_dryer_2_pinned_to_the_bed": {},
    }
    for convention in CONVENTIONS:
        base = _config(inputs, Switches(), convention, bed)
        record["walks"][convention] = {}
        for label, order in (("forward", SWITCH_NAMES), ("reverse", tuple(reversed(SWITCH_NAMES)))):
            steps = walk(base, order)
            record["walks"][convention][label] = {
                "order": list(order),
                "steps": [s.as_dict() for s in steps],
                "ranking": ranking_summary(steps),
            }
        record["single_switch"][convention] = single_switch(inputs, bed, convention)
        pinned = Flowsheet(
            _config(inputs, Switches.all_physical(), convention, bed, pinned=pins_bed)
        ).evaluate_all()
        free = Flowsheet(_config(inputs, Switches.all_physical(), convention, bed)).evaluate_all()
        record["physical_with_dryer_2_pinned_to_the_bed"][convention] = {
            "pinned_exhaust_dryer_2": {
                "temperature_celsius": bed.exhaust_temperature_celsius,
                "humidity_kg_per_kg": bed.exhaust_humidity_kg_per_kg,
            },
            "block_exhaust_dryer_2_celsius": free[1].exhausts[2].temperature_celsius,
            "cases": {
                str(case): {
                    "recovered_kw": pinned[case].recovered_kw,
                    "recovered_kw_block": free[case].recovered_kw,
                    "fuel_corrected_kg_per_h": pinned[case].fuel_corrected_kg_per_h,
                    "co2_kg_per_h": pinned[case].co2_kg_per_h,
                }
                for case in range(7)
            },
            "ranking_co2": ranking_text(
                rank_groups({c: pinned[c].co2_kg_per_h for c in SCENARIOS})
            ),
        }
    record["befund_90_reference_exhausts"] = {
        v: ref90["source_variants"][v]["exhausts"] for v in VARIANTS
    }
    target.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
