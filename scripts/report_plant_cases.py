r"""The six heat-recovery cases at the plant inputs of the partner deliveries.

Design: ``docs/NN_plant_cases.md``. Lab book: Befund NN (number open).

**Question.** Befund 110 found case 1 the best of the six cases at the workbook point; Befund 121
found, with the dry-air flows and circuit flows of the workbook, the rule "case 1 if exhaust 1 is
at least as warm as exhaust 2". The partner deliveries of 2026-11-19 and 2026-11-24 hold plant
inputs. (1) Is the delivery reproduced, including the four cascades? (2) At admissible plant-near
points, is case 1 the best case? (3) Does the worded rule hold there? (4) If not, is it the size
of the two lines that decides?

Inputs (confidential, never copied into the repository): the sweep sheet, the comparison
workbook and the seven flowsheets from ``$TWINLINK_LOCAL_DIR`` (default ``<root>/local``); the
two exports of plant A through ``TWINLINK_PLANT_DATA_PROD1`` / ``TWINLINK_PLANT_DATA_PROD2``; the
channel roles from ``--roles``. Dyssol from ``$DYSSOL_HOME``, the compiled units from
``$TWINLINK_DYSSOL_UNITS``, the patched materials database and the scratch folders in
``$TWINLINK_RUNS_DIR`` (default ``C:/twinlink_runs/<branch>``).

**Lines and cases.** Line 1 is the line numbered 1 in the delivered case flowsheets, line 2 the
line that both deliveries label as plant A (the sweep sheet numbers its processes the other way
round; the processes are assigned by the rule of Befund 130). Cases by their wiring
(``dyssol_steady.network.CASES``); circuit *i* cools exhaust *i*.

**Models.** U: the delivered units (rebuild extended to stages in series, Dyssol control with the
seven delivered flowsheets). W: the workbook rebuild in its three modes, with the properties of
the workbook and the equilibrium moisture, heater efficiency and calorific value of the lines.

**Points.** P0: the plant sheet of 2026-11-19 as delivered (U: the stored inputs of the
flowsheets; W: the inlet block of the sheet). S: both lines as the sweep sheet states them. R: S
with the reference solid fraction (0.65) in line 2. M: line 2 at the measured inputs of every
stationary segment of plant A (mapping, reference solid fraction and dried supply air as Befund
132), line 1 as in S, the fresh-gas temperature of both lines the measured intake temperature.
X: R with the sizes of the two lines exchanged. Circuit flows: those stored in the delivered
case flowsheets. **Read points: R and the segments.**

**Reading, fixed in this docstring before the first run of this script.**

* **Step 0, controls.** (a) For cases 1 and 2 the extended rebuild equals
  ``reference.evaluate`` at every point (relative 1e-12). (b) The segments are those of Befund
  132: labels, digests of both exports and of the role file equal those of
  ``sheet_plant_a_v1.json``, and the saturation ratio of line 2 at every segment equals the
  stored one (relative 1e-9). (c) Dyssol against the rebuild at every point, case and flow
  setting within the bounds of docs/45, no Dyssol run failed. (d) W in the partner mode at P0
  reproduces every Excel total of the plant sheet within rounding (two halves of the last
  decimal), with the circuit flows the sheet states or with those the flowsheets store. If (a),
  (b) or (c) fails, U is not read; if (d) fails, W is not read.
* **Reading F, feasibility first** (per model). A cell (read point, case) is admissible if both
  dryer exhausts are admissible and, in U, no exhaust is over-saturated after its recovery
  stage. U: saturation ratio at or below 1; W: at or below 90 % relative humidity (Befund 121).
  A stage with negative recovery is counted, not excluded. **F0** all cells admissible; **F1**
  otherwise.
* **Reading C, case 1 against the best case** (per model, at each read point over the cases
  admissible there). Practical tie: a gap below 1 % of the saving of the best case (Befund 121,
  F4; 0.5 % and 2 % reported). **C0** case 1 best or practically equal at every read point;
  **C1** at none; **C2** otherwise. The worded rule of Befund 121 is evaluated at the same
  points by its scale-free first split; where it predicts case 1, C1 means that it fails.
  Headline: U; the physical unpressurised mode of W is named next to it.
* **Reading A, attribution** (point X; not applicable if case 1 is best or practically equal at
  R). **A0** at X case 1 is best or practically equal in U and in the physical unpressurised
  mode of W; **A1** otherwise.

**Predictions, fixed with the reading** (arithmetic by hand and the totals the delivery stores,
disclosed in the document): all controls hold; F0; C1 in U and in all three modes of W; A0.

**Reported without a reading:** P0 and S (inadmissible as delivered); the plant sheet and the
stored results against the models; ranking, savings and regrets per read point; negative
stages; both circuits at each flow of the sweep sheet; the band of the solid fraction; X at the
segments and with scaled circuits.

**Generalised rule (addition of the project lead of 2026-11-25; descriptive, no reading, a
post-hoc hypothesis):** "the warmer exhaust preheats the sink with the larger heat-capacity
rate". Hit rate for the choice between cases 1 and 2 at the read points and X, per model; and at
the points of Befund 121 from its stored arrays (nothing recomputed), next to the worded rule.

**What is committed** (decision of the project lead, 2026-11-25): ranks, savings and regrets in
%, saturation ratios, counts, flags, reading codes, differences Dyssol minus rebuild, digests.
Everything else goes to the local record ``<root>/local/runs/plant_cases/<name>_local.json``
(git-ignored), which is copied to ``$TWINLINK_RUNS_DIR``.

Artifacts are never overwritten (``--name`` for a new one). Run from a clean working tree with
``tools/run_here.sh scripts/report_plant_cases.py``. With ``--dry-run`` every point goes through
the rebuild and W, Dyssol runs every flowsheet at R only; the readings are printed, nothing is
written and the tree may be dirty.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))
sys.path.insert(0, str(SCRIPTS))

from _common import (  # noqa: E402
    NO_HEAT_NETWORK,
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)
from report_sheet_plant_a import (  # noqa: E402
    DEFAULT_ROLES,
    DELIVERY,
    DYSSOL_HOME,
    GAS_FLOW,
    WORKBOOK,
    dyssol_installation,
    load_days,
    runs_dir,
)

from twinlink_ml.case_ranking import load_config as load_case_ranking_config  # noqa: E402
from twinlink_ml.case_ranking.evaluator import MODES  # noqa: E402
from twinlink_ml.case_ranking.space import ambient_humidity_kg_per_kg  # noqa: E402
from twinlink_ml.case_ranking.workbook import WORKBOOK as WORKBOOK_CONSTANTS  # noqa: E402
from twinlink_ml.dyssol_steady import plant_cases as pcs  # noqa: E402
from twinlink_ml.dyssol_steady import plant_check as pc  # noqa: E402
from twinlink_ml.dyssol_steady import study  # noqa: E402
from twinlink_ml.dyssol_steady.flowsheet import (  # noqa: E402
    DyssolFlowsheet,
    discover_network,
    extract_outputs,
    nominal_inputs,
)
from twinlink_ml.dyssol_steady.materials import (  # noqa: E402
    compound_names,
    patch_materials_database,
    read_properties,
    sha256_of,
)
from twinlink_ml.dyssol_steady.network import CASES, evaluate_network  # noqa: E402
from twinlink_ml.dyssol_steady.reference import LineInputs, LineOutputs, evaluate  # noqa: E402
from twinlink_ml.dyssol_steady.sweep_sheet import line_inputs, load_sweep_sheet  # noqa: E402
from twinlink_ml.partner_v4_1.comparison import load_dyssol_steady_state  # noqa: E402
from twinlink_ml.plant_scale import load_plant_scale_config  # noqa: E402

COMPARISON = DELIVERY / "Dyssol_Excel simulation_results (2).xlsx"
PLANT_SHEET = "Simulation-Plant data"
FLOWSHEETS = (
    "Case 1.1.dflw",
    "Case 1.2.dflw",
    "Case 2.dflw",
    "Case 3.dflw",
    "Case 4.dflw",
    "Case 5.dflw",
    "Case 6.dflw",
)
BEFUND_132 = RUNS_DIR / "sheet_plant_a_v1.json"
BEFUND_121_ARRAYS = RUNS_DIR / "case_ranking_v1.npz"
BEFUND_121_ANALYSIS = RUNS_DIR / "case_ranking_v1_analysis.json"
LOCAL_RUNS = REPO_ROOT / "local" / "runs" / "plant_cases"
DEFAULT_NAME = "plant_cases_v1"

PREDICTIONS: dict[str, Any] = {
    "F": dict.fromkeys(pcs.MODELS, "F0"),
    "C": dict.fromkeys(pcs.MODELS, "C1"),
    "A": "A0",
    "A_models": (pcs.MODEL_U, "W:physical_unpressurised_95"),
}
REBUILD_TOLERANCE = 1e-12
SATURATION_TOLERANCE = 1e-9
SHEET_DECIMALS = 2
"""Decimals of the fuel and heat values of the plant sheet."""
DRY_RUN_POINT = "R"


def relative_difference(a: LineOutputs, b: LineOutputs) -> float:
    """Largest relative difference over the fields of two line results."""
    worst = 0.0
    for field in fields(LineOutputs):
        x, y = getattr(a, field.name), getattr(b, field.name)
        worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1.0))
    return worst


def load_flowsheets(names: dict[str, str]) -> dict[str, Any]:
    """The delivered flowsheets: file, wiring, stored inputs and results; which file is which case.

    One-line files stand for the lines of case 1 (the hotter stored dryer inlet is line 1); a
    two-line file is the case whose wiring it has. In every two-line file the first line (order
    of the dryer names) must be the one with the hotter stored dryer inlet.
    """
    found: dict[str, Any] = {}
    for name in FLOWSHEETS:
        file = load_dyssol_steady_state(DELIVERY / name)
        topology = discover_network(file)
        found[name] = {
            "topology": topology,
            "stored_inputs": nominal_inputs(file, topology, names),
            "stored_outputs": extract_outputs(file, topology, names),
        }
    single = [n for n in FLOWSHEETS if len(found[n]["topology"].lines) == 1]
    if len(single) != 2:
        raise SystemExit("expected two one-line flowsheets (the lines of case 1)")
    single.sort(key=lambda n: -found[n]["stored_inputs"][0].dryer_inlet_celsius)
    by_case: dict[int, str] = {}
    for name in FLOWSHEETS:
        entry = found[name]
        if name in single:
            continue
        stored = entry["stored_inputs"]
        if not stored[0].dryer_inlet_celsius > stored[1].dryer_inlet_celsius:
            raise SystemExit(f"{name}: the first line is not the one with the hotter dryer inlet")
        cases = [k for k, wiring in CASES.items() if wiring == entry["topology"].sources_of_sink]
        if len(cases) != 1 or cases[0] in by_case:
            raise SystemExit(f"{name}: wiring is none of the six cases, or a case appears twice")
        by_case[cases[0]] = name
    if sorted(by_case) != [2, 3, 4, 5, 6]:
        raise SystemExit(f"flowsheets of the cases 2 to 6 expected, found {sorted(by_case)}")
    return {"files": found, "lines_of_case_1": tuple(single), "by_case": by_case}


def control_segments(
    segments: dict[str, list[pc.MeasuredPoint]],
    found: dict[str, dict[str, Any]],
    roles: Path,
    ratios: dict[str, float],
) -> dict[str, Any]:
    """Control (b): the segments are those of Befund 132."""
    stored = json.loads(BEFUND_132.read_text(encoding="utf-8"))
    labels = [p.label for pts in segments.values() for p in pts]
    same_labels = stored["test_set"]["segments"] == labels
    same_exports = stored["inputs"]["exports_sha256"] == {
        day: entry["export_sha256"] for day, entry in found.items()
    }
    same_roles = stored["inputs"]["role_file_sha256"] == sha256_of(roles)
    reference = stored["without_reading"]["variants"]["0.65|dried_supply_air"]["saturation_ratio"]
    largest = (
        max(abs(ratios[label] - reference[label]) / reference[label] for label in labels)
        if same_labels
        else float("inf")
    )
    return {
        "same_segments": same_labels,
        "same_exports": same_exports,
        "same_role_file": same_roles,
        "saturation_ratio_max_relative_deviation": largest,
        "tolerance": SATURATION_TOLERANCE,
        "held": same_labels and same_exports and same_roles and largest <= SATURATION_TOLERANCE,
    }


def befund_121_hit_rates(tie_exact: float, tie_fraction: float, max_rh: float) -> dict[str, Any]:
    """The generalised rule and the worded rule at the points of Befund 121 (stored arrays only)."""
    arrays = np.load(BEFUND_121_ARRAYS, allow_pickle=False)
    analysis = json.loads(BEFUND_121_ANALYSIS.read_text(encoding="utf-8"))
    material = WORKBOOK_CONSTANTS.material
    air = np.array(WORKBOOK_CONSTANTS.air_kg_per_h) / 3600.0
    out: dict[str, Any] = {"source": BEFUND_121_ARRAYS.name, "recomputed": False, "sets": {}}
    for label in ("full", "rule_test"):
        points = arrays[f"fixed/{label}/points"]
        admissible = np.ones(len(points), dtype=bool)
        for mode in MODES:
            admissible &= (arrays[f"fixed/{label}/{mode}/exh_rh"] <= max_rh).all(axis=1)
            admissible &= ~arrays[f"fixed/{label}/{mode}/error"]
        humidity = np.array([ambient_humidity_kg_per_kg(t, rh) for t, rh in points[:, :2]])
        capacity = air[None, :] * (
            material.cp_gas_kj_per_kg_k + humidity[:, None] * material.cp_vapour_kj_per_kg_k
        )
        per_mode = {}
        for mode in MODES:
            per_mode[mode] = pcs.pair_rule_hit_rates(
                arrays[f"fixed/{label}/{mode}/fuel"][admissible],
                arrays[f"fixed/{label}/{mode}/exh_T"][admissible],
                capacity[admissible],
                exact=tie_exact,
                fraction=tie_fraction,
                worded_threshold_k=pcs.WORDED_RULE_THRESHOLD_K[f"W:{mode}"],
            )
        out["sets"][label] = {"admissible_points": int(admissible.sum()), "modes": per_mode}
    out["stored_tree_test_accuracy_strictly_best"] = {
        mode: analysis["rule"][f"fixed/{mode}"]["strict"]["test_accuracy"] for mode in MODES
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--name", default=DEFAULT_NAME, help="name of the artifact")
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES, help="channel roles (local)")
    parser.add_argument("--workers", type=int, default=4, help="parallel Dyssol processes")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    args = parser.parse_args()
    local_record = LOCAL_RUNS / f"{args.name}_local.json"
    if not args.dry_run:
        require_clean_tree()
        for path in (RUNS_DIR / f"{args.name}.json", local_record):
            if path.exists():
                raise SystemExit(f"{path} exists -- use a new --name")
    record = provenance(NO_HEAT_NETWORK)
    installed = DYSSOL_HOME / "Materials.dmdb"
    needed = [WORKBOOK, COMPARISON, installed, BEFUND_132, BEFUND_121_ARRAYS, BEFUND_121_ANALYSIS]
    for path in (*needed, args.roles, *(DELIVERY / n for n in FLOWSHEETS)):
        if not path.exists():
            raise SystemExit(f"missing file: {path}")
    steady = study.load_config()
    plant = load_plant_scale_config()
    ranking = load_case_ranking_config()
    exact, fraction = ranking.tie_exact_kg_per_h, ranking.tie_practical_fraction
    max_rh = ranking.max_exhaust_relative_humidity

    patched = runs_dir() / "materials" / f"Materials_{sha256_of(installed)[:12]}_boiling_point.dmdb"
    if not patched.exists():
        patch_materials_database(installed, patched)
    props = read_properties(patched)
    names = compound_names(patched)

    # ---- the delivery: flowsheets, stored inputs, stored circuit flows
    delivery = load_flowsheets(names)
    files = delivery["files"]
    plant_files = [delivery["by_case"][k] for k in (2, 3, 4)]
    stored_lines: tuple[LineInputs, LineInputs] = files[delivery["by_case"][2]]["stored_inputs"]
    for name in plant_files:
        if files[name]["stored_inputs"] != stored_lines:
            raise SystemExit(f"{name}: stored inputs differ from those of the file of case 2")
    for i, name in enumerate(delivery["lines_of_case_1"]):
        if files[name]["stored_inputs"][0] != stored_lines[i]:
            raise SystemExit(f"{name}: stored inputs differ from line {i + 1} of case 2")
    stored_flows = (stored_lines[0].medium_kg_per_h, stored_lines[1].medium_kg_per_h)

    # ---- the sheets and the measured segments
    sheet = load_sweep_sheet(WORKBOOK)
    candidates = [line_inputs(sheet, p, GAS_FLOW) for p in range(len(sheet.lines))]
    order = pcs.assign_lines(candidates, stored_lines)
    sheet_lines = (candidates[order[0]], candidates[order[1]])
    grid_flows = [flow for flow in sheet.medium_kg_per_h if flow > 0.0]

    import openpyxl

    book = openpyxl.load_workbook(COMPARISON, data_only=True)
    plant_sheet = pcs.parse_plant_sheet(list(book[PLANT_SHEET].iter_rows(values_only=True)))
    inlet = [
        plant_sheet.line_inputs(p, stored_lines[0].equilibrium_moisture_g_per_kg)
        for p in range(len(plant_sheet.lines))
    ]
    inlet_order = pcs.assign_lines(inlet, stored_lines)
    inlet_lines = (inlet[inlet_order[0]], inlet[inlet_order[1]])
    stated_flows = (inlet_lines[0].medium_kg_per_h, inlet_lines[1].medium_kg_per_h)

    segments, found = load_days(plant, args.roles)
    placeholders = plant.placeholders
    labels = [p.label for pts in segments.values() for p in pts]
    band = [f"{value:g}" for value in placeholders.solid_fraction_band]
    points = pcs.build_points(
        sheet_lines,
        segments,
        solid_reference=placeholders.solid_fraction_reference,
        solid_band=placeholders.solid_fraction_band,
        humidity_reference_g_per_kg=1000.0 * placeholders.inlet_absolute_humidity_kg_per_kg,
        normal_density_kg_per_m3=plant.data.normal_density_dry_air_kg_per_m3,
    )
    read_points = ["R", *(f"M|{label}" for label in labels)]
    groups = {"R": ["R"], **{day: [f"M|{p.label}" for p in pts] for day, pts in segments.items()}}
    grid_settings = [f"grid|{i}" for i in range(len(grid_flows))]

    # ---- every (tag, setting) with its inputs
    cells: dict[tuple[str, str], pcs.Point] = {}
    for tag, point in points.items():
        cells[tag, pcs.READ_SETTING] = pcs.with_flows(point, stored_flows)
        if tag in ("S", "X", *read_points):
            for setting, flow in zip(grid_settings, grid_flows, strict=True):
                cells[tag, setting] = pcs.with_flows(point, (flow, flow))
    cells["X|scaled", pcs.READ_SETTING] = pcs.exchange_sizes(
        pcs.with_flows(points["R"], stored_flows), scale_circuits=True
    )
    cells["P0", pcs.READ_SETTING] = stored_lines
    workbook_only = {
        ("P0", "stated"): pcs.with_flows(inlet_lines, stated_flows),
        ("P0", pcs.READ_SETTING): pcs.with_flows(inlet_lines, stored_flows),
    }

    # ---- models
    t0 = time.perf_counter()
    results: dict[str, dict[str, dict[str, pcs.ModelResult]]] = {}
    for (tag, setting), point in cells.items():
        entry = results.setdefault(tag, {}).setdefault(setting, {})
        entry[pcs.MODEL_U] = pcs.evaluate_units(props, point)
        if tag != "P0":
            for mode in MODES:
                entry[f"W:{mode}"] = pcs.evaluate_workbook(point, mode, max_rh)
    print(f"{len(cells)} cells through U and W in {time.perf_counter() - t0:.1f} s", flush=True)

    # ---- step 0 (a): the extended rebuild against reference.evaluate
    worst_a = 0.0
    for point in cells.values():
        for case, preheated_by in ((1, (0, 1)), (2, (1, 0))):
            ours = evaluate_network(props, point, CASES[case]).lines
            theirs = evaluate(props, point, preheated_by)
            worst_a = max(
                worst_a, *(relative_difference(a, b) for a, b in zip(ours, theirs, strict=True))
            )
    control_a = {
        "cells": len(cells),
        "max_relative_difference": worst_a,
        "tolerance": REBUILD_TOLERANCE,
        "held": worst_a <= REBUILD_TOLERANCE,
    }

    # ---- step 0 (b): the segments of Befund 132
    ratios = {
        label: results[f"M|{label}"][pcs.READ_SETTING][pcs.MODEL_U].saturation[1]
        for label in labels
    }
    control_b = control_segments(segments, found, args.roles, ratios)

    # ---- step 0 (c): Dyssol against the rebuild
    installation, installation_record = dyssol_installation(patched)
    work = runs_dir() / "work" / args.name
    flowsheets = {
        name: DyssolFlowsheet(
            DELIVERY / name,
            installation,
            work / Path(name).stem.replace(" ", "_"),
            names,
            discover=discover_network,
        )
        for name in FLOWSHEETS
    }
    for flowsheet in flowsheets.values():
        flowsheet.local_source()
    jobs: list[tuple[str, str, list[LineInputs], list[str]]] = []
    reference: dict[str, LineOutputs] = {}
    for (tag, setting), point in cells.items():
        if args.dry_run and (tag, setting) != (DRY_RUN_POINT, pcs.READ_SETTING):
            continue
        for case in pcs.CASE_NUMBERS:
            ours = evaluate_network(props, point, CASES[case]).lines
            keys = [f"{tag}|{setting}|case{case}|line{i + 1}" for i in (0, 1)]
            reference.update(zip(keys, ours, strict=True))
            if case == 1:
                for i, name in enumerate(delivery["lines_of_case_1"]):
                    jobs.append((name, f"j{len(jobs):05d}", [point[i]], [keys[i]]))
            else:
                jobs.append((delivery["by_case"][case], f"j{len(jobs):05d}", list(point), keys))

    dyssol: dict[str, LineOutputs | None] = {}
    failures: list[dict[str, Any]] = []
    seconds: list[float] = []
    peaks: list[int] = []

    def run(job: tuple[str, str, list[LineInputs], list[str]]) -> None:
        name, scratch, lines, keys = job
        try:
            evaluation = flowsheets[name].evaluate(lines, scratch)
        except Exception as exc:  # a failing run must not stop the others
            failures.append({"job": scratch, "flowsheet_of": keys[0], "error": type(exc).__name__})
            print(f"{scratch}: FAILED {type(exc).__name__}: {exc}", flush=True)
            dyssol.update(dict.fromkeys(keys))
            return
        dyssol.update(zip(keys, evaluation.outputs, strict=True))
        seconds.append(evaluation.wall_seconds)
        if evaluation.peak_memory_bytes:
            peaks.append(evaluation.peak_memory_bytes)
        print(f"{scratch} of {len(jobs)}: done in {evaluation.wall_seconds:.2f} s", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(run, jobs))
    print(f"{len(jobs)} Dyssol runs in {time.perf_counter() - t0:.1f} s", flush=True)
    deviation = pc.control_deviation(dyssol, reference)
    control_c = {
        **deviation,
        "dyssol_runs": len(jobs),
        "bounds": {
            "temperature_k": steady.control_temperature_k,
            "relative_percent": steady.control_relative_percent,
        },
        "failures": failures,
        "dyssol_wall_seconds": [min(seconds), max(seconds)] if seconds else None,
        "dyssol_peak_memory_mb": max(peaks) / 2**20 if peaks else None,
        "workers": args.workers,
        "held": not failures
        and deviation["points_compared"] == deviation["points"]
        and deviation["max_temperature_k"] <= steady.control_temperature_k
        and deviation["max_relative_percent"] <= steady.control_relative_percent,
    }

    # ---- step 0 (d): W in the partner mode against the Excel columns of the plant sheet
    order_of_sheet = inlet_order  # sheet process -> line
    excel: dict[str, Any] = {}
    for (_, setting), point in workbook_only.items():
        totals_within = fuel_within = heat_within = fuel_n = heat_n = 0
        worst_total = 0.0
        for case in pcs.CASE_NUMBERS:
            ours = pcs.workbook_case_details(point, "partner", case)
            total = plant_sheet.total[case]["excel"]
            if total is None:
                raise SystemExit(f"the plant sheet has no Excel total of case {case}")
            ours_total = sum(ours["fuel_kg_per_h"])
            worst_total = max(worst_total, abs(ours_total - total))
            totals_within += pcs.within_rounding(ours_total, total, SHEET_DECIMALS, halves=2)
            for line, process in enumerate(order_of_sheet):
                reported = plant_sheet.fuel[case]["excel"][process]
                if reported is not None:
                    fuel_n += 1
                    fuel_within += pcs.within_rounding(
                        ours["fuel_kg_per_h"][line], reported, SHEET_DECIMALS
                    )
                reported = plant_sheet.recovered[case]["excel"][process]
                if reported is not None:
                    heat_n += 1
                    heat_within += pcs.within_rounding(
                        ours["recovered_kw"][line], reported, SHEET_DECIMALS
                    )
        excel[setting] = {
            "totals_within_rounding": totals_within,
            "totals": len(pcs.CASE_NUMBERS),
            "largest_total_difference_in_units_of_the_last_decimal": worst_total
            * 10**SHEET_DECIMALS,
            "fuel_values_within_rounding": [fuel_within, fuel_n],
            "recovered_heats_within_rounding": [heat_within, heat_n],
            "held": totals_within == len(pcs.CASE_NUMBERS),
        }
    control_d = {"flow_settings": excel, "held": any(e["held"] for e in excel.values())}

    read_u = bool(control_a["held"] and control_b["held"] and control_c["held"])
    read_w = bool(control_d["held"])

    # ---- reported without a reading: the delivery against the models
    p0 = results["P0"][pcs.READ_SETTING][pcs.MODEL_U]
    dyssol_columns = {"fuel": [0, 0], "total": [0, 0], "recovered": [0, 0]}
    for case in pcs.CASE_NUMBERS:
        if plant_sheet.total[case].get("dyssol") is None:
            continue
        ours = evaluate_network(props, stored_lines, CASES[case])
        dyssol_columns["total"][1] += 1
        dyssol_columns["total"][0] += pcs.within_rounding(
            ours.fuel_kg_per_h, plant_sheet.total[case]["dyssol"], SHEET_DECIMALS, halves=2
        )
        for line, process in enumerate(order_of_sheet):
            for key, value in (
                ("fuel", ours.lines[line].fuel_kg_per_h),
                ("recovered", ours.lines[line].heat_recovered_kw),
            ):
                table = plant_sheet.fuel if key == "fuel" else plant_sheet.recovered
                reported = table[case]["dyssol"][process]
                if reported is not None:
                    dyssol_columns[key][1] += 1
                    dyssol_columns[key][0] += pcs.within_rounding(value, reported, SHEET_DECIMALS)
    stored_against_rebuild = {}
    for name in FLOWSHEETS:
        entry = files[name]
        ours = evaluate_network(props, entry["stored_inputs"], entry["topology"].sources_of_sink)
        keys = [f"line{i + 1}" for i in range(len(ours.lines))]
        dev = pc.control_deviation(
            dict(zip(keys, entry["stored_outputs"], strict=True)),
            dict(zip(keys, ours.lines, strict=True)),
        )
        label = (
            f"case 1, line {delivery['lines_of_case_1'].index(name) + 1}"
            if name in delivery["lines_of_case_1"]
            else f"case {next(k for k, n in delivery['by_case'].items() if n == name)}"
        )
        stored_against_rebuild[label] = {
            "max_temperature_k": dev["max_temperature_k"],
            "max_relative_percent": dev["max_relative_percent"],
        }
    p0_entry = pcs.case_one(p0, exact, fraction)
    delivery_record = {
        "plant_sheet_dyssol_columns_within_rounding": dyssol_columns,
        "stored_results_against_rebuild": stored_against_rebuild,
        "P0_units": {
            "saturation": list(p0.saturation),
            "dryers_admissible": bool(p0_entry["admissible_cases"]),
            "order_of_the_formula": sorted(pcs.CASE_NUMBERS, key=lambda k: p0.fuel()[k]),
            "saving_percent": {
                str(k): 100.0 * (p0.fuel()[0] - p0.fuel()[k]) / p0.fuel()[0]
                for k in pcs.CASE_NUMBERS
            },
            "negative_stages": {str(c.case): c.negative_stages for c in p0.cases[1:]},
        },
        "sheet_numbering_reversed_between_the_deliveries": order != inlet_order,
        "stated_and_stored_circuit_flows_equal": stated_flows == stored_flows,
    }

    # ---- readings and records
    repo, local = pcs.summarise(
        results={tag: by for tag, by in results.items() if tag != "P0"},
        read_points=read_points,
        groups=groups,
        grid_settings=grid_settings,
        band=band,
        exact=exact,
        fraction=fraction,
        sensitivity=ranking.tie_practical_sensitivity,
        predictions=PREDICTIONS,
        points={
            tag: point for (tag, setting), point in cells.items() if setting == pcs.READ_SETTING
        },
    )
    rule_121 = befund_121_hit_rates(exact, fraction, max_rh)
    local["P0"] = {
        "units": asdict(p0),
        "stored_inputs": [asdict(line) for line in stored_lines],
        "inlet_block": [asdict(line) for line in inlet_lines],
        "stored_flows_kg_per_h": list(stored_flows),
        "stated_flows_kg_per_h": list(stated_flows),
        "grid_flows_kg_per_h": grid_flows,
        "plant_sheet": {
            "fuel": plant_sheet.fuel,
            "recovered": plant_sheet.recovered,
            "total": plant_sheet.total,
        },
        "stored_outputs": {
            name: [asdict(o) for o in files[name]["stored_outputs"]] for name in FLOWSHEETS
        },
        "workbook_partner_mode": {
            setting: {
                str(case): pcs.workbook_case_details(point, "partner", case)
                for case in pcs.CASE_NUMBERS
            }
            for (_, setting), point in workbook_only.items()
        },
    }
    payload: dict[str, Any] = {
        "_provenance": record,
        "name": args.name,
        "inputs": {
            "sweep_sheet_sha256": sha256_of(WORKBOOK),
            "comparison_workbook_sha256": sha256_of(COMPARISON),
            "flowsheets_sha256": {
                f"case 1, line {i + 1}": sha256_of(DELIVERY / n)
                for i, n in enumerate(delivery["lines_of_case_1"])
            }
            | {
                f"case {k}": sha256_of(DELIVERY / n) for k, n in sorted(delivery["by_case"].items())
            },
            "materials": {
                "installed_sha256": sha256_of(installed),
                "patched_sha256": sha256_of(patched),
            },
            "exports_sha256": {day: entry["export_sha256"] for day, entry in found.items()},
            "role_file_sha256": sha256_of(args.roles),
            "dyssol_steady_config_fingerprint": steady.fingerprint,
            "case_ranking_config_fingerprint": ranking.fingerprint,
            "plant_scale_config_sha256": sha256_of(
                REPO_ROOT / "conf" / "plant_scale" / "plant_scale.yaml"
            ),
            "gas_flow_reading": GAS_FLOW,
            "max_exhaust_relative_humidity": max_rh,
            "cells": len(cells),
            "grid_settings": len(grid_settings),
            **installation_record,
        },
        "control": {
            "a_extended_rebuild": control_a,
            "b_segments": control_b,
            "c_dyssol": control_c,
            "d_workbook_against_excel": control_d,
            "read_units": read_u,
            "read_workbook": read_w,
        },
        **repo,
        "delivery": delivery_record,
        "generalised_rule_at_befund_121": rule_121,
    }

    print(f"control a (extended rebuild): {control_a}")
    print(f"control b (segments of Befund 132): {control_b}")
    print(f"control c (Dyssol against the rebuild): {control_c}")
    print(f"control d (W against the Excel columns): {control_d}")
    print(f"U is read: {read_u}; W is read: {read_w}")
    for model in pcs.MODELS:
        entry = repo["models"][model]
        print(f"{model}: {entry['reading_f']}")
        print(f"{model}: reading C {entry['reading_c']}")
        print(f"{model}: worded rule {entry['worded_rule']}")
        rule = entry["generalised_rule"]
        print(f"{model}: generalised rule {rule['hits']} of {rule['decided_points']}")
        print(f"{model}: X {entry['exchanged_sizes']['X']['best_case']}, band {entry['band']}")
        for tag in read_points:
            point = entry["points"][tag]
            print(
                f"   {tag}: order {point.get('order')}, regret "
                f"{point.get('regret_case_1_percent_of_best_saving')} % of the best saving"
            )
    print(f"reading A: {repo['reading_a']}")
    print(f"predictions: {repo['predictions']}")
    print(f"delivery: {delivery_record}")
    for label, entry in rule_121["sets"].items():
        for mode, rates in entry["modes"].items():
            print(
                f"Befund 121 {label} {mode}: generalised {rates['generalised']['decided']}, "
                f"worded {rates['worded']['decided']}, differ {rates['rules_differ']}"
            )
    if args.dry_run:
        print("dry run: nothing written")
        return
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    local_record.write_text(
        json.dumps({"name": args.name, "git/commit": record["git/commit"], **local}, indent=1),
        encoding="utf-8",
    )
    copy = runs_dir() / local_record.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local_record, copy)
    if sha256_of(copy) != sha256_of(local_record):
        raise SystemExit(f"copy {copy} differs from {local_record}")
    print(f"wrote {local_record} (copy verified: {copy})")
    write_artifact(args.name, payload)


if __name__ == "__main__":
    main()
