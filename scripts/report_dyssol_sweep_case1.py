r"""Sweep of the medium flow in the two one-line flowsheets ("case 1"): partner sheet against the
rebuild of the stationary units and against Dyssol.

Lab book: Befund 130 (number proposed on the branch); docs/45 section 17.

**Question.** The partner sent a sheet (2026-11-24) with results of the two lines of the delivery
of 2026-11-19, each with its own heat recovery, over eleven flows of the heat transfer medium
(none, and ten steps), at a fresh-gas state other than the one stored in the flowsheets. Do the
Python rebuild of the units (``twinlink_ml.dyssol_steady.reference``, Befund 128) and our own
Dyssol runs give the reported numbers?

Inputs (confidential, never copied into the repository) from ``$TWINLINK_LOCAL_DIR`` (default
``<root>/local``): ``partner/dyssol_2026-11-24/Simulation results_flow rates.xlsx`` (the sheet),
``partner/dyssol_2026-11-19/Case 1.1.dflw`` and ``Case 1.2.dflw`` (the two one-line flowsheets),
``dyssol_steady/box_local.yaml`` (the box of docs/45). Dyssol from ``$DYSSOL_HOME``; the compiled
units from ``$TWINLINK_DYSSOL_UNITS`` (default: the folder built for Befund 128, checked against
its manifest); the patched materials database and the scratch folders of the Dyssol runs in
``$TWINLINK_RUNS_DIR`` (default ``C:/twinlink_runs/<branch>``).

**Reading, fixed in this docstring before the first run of this script.**

* "Within rounding" = ``|ours - reported|`` at most half a unit of the last reported decimal (as
  in docs/11 section 13.3); the sheet reports one decimal. Reported next to it: within one unit
  of that decimal, and whether ours rounded or cut off to one decimal is the reported value.
* The sheet does not say whether its gas mass flow is the dry gas or the humid total. Both readings
  are computed with the rebuild; the one with more values within rounding over both processes is
  taken (a tie goes to "dry", listed first). Both are in the artifact.
* **Verdict** on the rebuild with that reading, over all reported values of both processes:
  **S1** every value within rounding; **S2** every value within one unit of the last decimal;
  **S3** otherwise (counts and largest deviation per quantity are the result).
* **Control.** Every point is also run with Dyssol (one process per source flowsheet; a process
  takes the flowsheet whose stored dryer inlet temperature is nearest to its own, the assignment
  must be one-to-one). The control holds when every Dyssol run succeeded and Dyssol and the rebuild
  agree within the bounds of docs/45 (``control`` of the YAML). If it holds, the verdict on the
  rebuild is the verdict on Dyssol. If it does not, the comparison of Dyssol with the sheet is the
  one to read, and the deviation between Dyssol and the rebuild is reported. A failing Dyssol run
  is recorded and does not stop the others.

**Reported without a reading:** the saturation ratio of the dryer exhaust of each process (the
dryer unit has no saturation limit; open point of Befund 106 and docs/45); how many points of the
sweep lie in the box of the ``one_line`` case of docs/45, and by which input they leave it; whether
the fresh-gas state is the stored one, and by how much the rebuild's exhaust temperature and fuel
move when the stored fresh-gas state is used instead (the surrogates of docs/45 have no input for
the fresh-gas state).

**Seen before this script was written (disclosure).** Probes without an artifact on 2026-11-24, in
the session that wrote this script: the rebuild at all points of the sheet with both readings of
the gas flow (the dry reading fitted, most values within one unit of the last decimal, not all
within rounding; the exhaust of one process above saturation), and the rebuild with the stored
fresh-gas state at three points. The reading above was written after these probes; it uses the
rule of docs/11 for "within rounding" and fixes no threshold that the probes suggested. No Dyssol
run at any point of the sheet had been made.

The artifact holds counts, deviations, ratios and digests -- no input value and no reported value
of the sheet, no name of a process. It is never overwritten (``--name`` for a new one). Run from
a clean working tree with ``tools/run_here.sh scripts/report_dyssol_sweep_case1.py``; with
``--dry-run`` everything is computed and printed, nothing is written and the tree may be dirty.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import (  # noqa: E402
    NO_HEAT_NETWORK,
    RUNS_DIR,
    git,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.dyssol_steady import study  # noqa: E402
from twinlink_ml.dyssol_steady.design import VARIED, load_box  # noqa: E402
from twinlink_ml.dyssol_steady.flowsheet import (  # noqa: E402
    DyssolFlowsheet,
    DyssolInstallation,
    discover_topology,
    nominal_inputs,
)
from twinlink_ml.dyssol_steady.materials import (  # noqa: E402
    DyssolProperties,
    compound_names,
    patch_materials_database,
    read_properties,
    sha256_of,
)
from twinlink_ml.dyssol_steady.reference import LineInputs, LineOutputs, evaluate  # noqa: E402
from twinlink_ml.dyssol_steady.sweep_sheet import (  # noqa: E402
    GAS_FLOW_READINGS,
    REPORTED_DECIMALS,
    RESULT_KEYS,
    SweepSheet,
    compare,
    line_inputs,
    load_sweep_sheet,
    sweep_table,
    totals,
    verdict,
)
from twinlink_ml.partner_v4_1.comparison import load_dyssol_steady_state  # noqa: E402

ROOT = SCRIPTS.parent
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", ROOT / "local"))
DYSSOL_HOME = Path(os.environ.get("DYSSOL_HOME", "C:/Program Files/Dyssol"))
UNITS = Path(os.environ.get("TWINLINK_DYSSOL_UNITS", "C:/twinlink_runs/dyssol-steady-state/units"))
WORKBOOK = LOCAL / "partner" / "dyssol_2026-11-24" / "Simulation results_flow rates.xlsx"
DELIVERY = LOCAL / "partner" / "dyssol_2026-11-19"
SOURCE_FILES = ("Case 1.1.dflw", "Case 1.2.dflw")
BOX_PATH = LOCAL / "dyssol_steady" / "box_local.yaml"
BOX_CASE, BOX_SOURCE = "one_line", "Case 1.1.dflw"
REFERENCE_ARTIFACT = RUNS_DIR / "dyssol_steady_v1_one_line_training.json.gz"
DEFAULT_NAME = "dyssol_sweep_case1_comparison"
CONTROL_TEMPERATURES = ("dryer_outlet_celsius", "exhaust_celsius", "preheated_celsius")
CONTROL_RELATIVE = ("heat_recovered_kw", "heater_duty_kw", "fuel_kg_per_h")


def runs_dir() -> Path:
    default = Path("C:/twinlink_runs") / (git("rev-parse", "--abbrev-ref", "HEAD") or "detached")
    return Path(os.environ.get("TWINLINK_RUNS_DIR", default))


def sweep(line: LineInputs, flows: Sequence[float]) -> list[LineInputs]:
    return [replace(line, medium_kg_per_h=m) for m in flows]


def rebuild(props: DyssolProperties, points: Sequence[LineInputs]) -> list[LineOutputs]:
    return [evaluate(props, [point], (0,))[0] for point in points]


def against_sheet(
    outputs: Sequence[LineOutputs], sheet: SweepSheet, process: int
) -> dict[str, dict[str, float | int]]:
    table = sweep_table(outputs)
    return {key: compare(table[key], sheet.results[key][process]) for key in RESULT_KEYS}


def summed(per_process: Sequence[Mapping[str, Mapping[str, float | int]]]) -> dict[str, int]:
    parts = [totals(entry) for entry in per_process]
    return {name: sum(part[name] for part in parts) for name in parts[0]}


def assign_sources(lines: Sequence[LineInputs], stored: Mapping[str, LineInputs]) -> list[str]:
    """Per process the source flowsheet with the nearest stored dryer inlet temperature."""
    chosen = [
        min(stored, key=lambda name: abs(stored[name].dryer_inlet_celsius - ln.dryer_inlet_celsius))
        for ln in lines
    ]
    if len(set(chosen)) != len(lines):
        raise SystemExit("the processes of the sheet do not map one-to-one onto the flowsheets")
    return chosen


def dyssol_installation(materials_db: Path) -> tuple[DyssolInstallation, dict[str, Any]]:
    manifest_path = UNITS / "manifest.json"
    executable = DYSSOL_HOME / "DyssolC.exe"
    if not manifest_path.exists() or not executable.exists():
        raise SystemExit(f"need {executable} and {manifest_path} (scripts/build_dyssol_units.py)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for dll, digest in manifest["libraries"].items():
        if sha256_of(UNITS / dll) != digest:
            raise SystemExit(f"{UNITS / dll} differs from its manifest")
    version = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return DyssolInstallation(executable, UNITS, materials_db), {
        "dyssol_version": version,
        "units_libraries": manifest["libraries"],
        "units_sources": manifest["sources"],
    }


def control(
    dyssol: Sequence[LineOutputs | None], reference: Sequence[LineOutputs]
) -> dict[str, float | int]:
    """Largest deviation of Dyssol from the rebuild over the points that ran."""
    temperature = relative = 0.0
    n = 0
    for d, r in zip(dyssol, reference, strict=True):
        if d is None:
            continue
        n += 1
        for name in CONTROL_TEMPERATURES:
            temperature = max(temperature, abs(getattr(d, name) - getattr(r, name)))
        for name in CONTROL_RELATIVE:
            ours = getattr(r, name)
            relative = max(relative, 100.0 * abs(getattr(d, name) - ours) / max(abs(ours), 1.0))
    return {
        "points_compared": n,
        "max_temperature_k": temperature,
        "max_relative_percent": relative,
    }


def span(values: Sequence[float]) -> dict[str, float] | None:
    return {"min": min(values), "max": max(values)} if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--name", default=DEFAULT_NAME, help="name of the artifact (never overwritten)"
    )
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    args = parser.parse_args()
    if not args.dry_run:
        require_clean_tree()
        if (RUNS_DIR / f"{args.name}.json").exists():
            raise SystemExit(f"artifact {args.name} exists -- use a new --name")
    record = provenance(NO_HEAT_NETWORK)
    installed = DYSSOL_HOME / "Materials.dmdb"
    needed = [
        WORKBOOK,
        BOX_PATH,
        installed,
        REFERENCE_ARTIFACT,
        *(DELIVERY / n for n in SOURCE_FILES),
    ]
    for path in needed:
        if not path.exists():
            raise SystemExit(f"missing file: {path}")
    config = study.load_config()
    sheet = load_sweep_sheet(WORKBOOK)
    flows = sheet.medium_kg_per_h
    processes = range(len(sheet.lines))

    patched = runs_dir() / "materials" / f"Materials_{sha256_of(installed)[:12]}_boiling_point.dmdb"
    if not patched.exists():
        patch_materials_database(installed, patched)
    props = read_properties(patched)
    names = compound_names(patched)
    stored: dict[str, LineInputs] = {}
    for name in SOURCE_FILES:
        file = load_dyssol_steady_state(DELIVERY / name)
        topology = discover_topology(file)
        if len(topology.lines) != 1 or tuple(topology.preheated_by) != (0,):
            raise SystemExit(f"{name} is not one line with its own heat recovery")
        stored[name] = nominal_inputs(file, topology, names)[0]

    # ---- both readings of the gas flow with the rebuild
    readings: dict[str, Any] = {}
    for reading in GAS_FLOW_READINGS:
        per_process = [
            against_sheet(rebuild(props, sweep(line_inputs(sheet, p, reading), flows)), sheet, p)
            for p in processes
        ]
        readings[reading] = {"processes": per_process, "total": summed(per_process)}
    chosen = max(GAS_FLOW_READINGS, key=lambda r: readings[r]["total"]["within_half"])
    lines = [line_inputs(sheet, p, chosen) for p in processes]
    points = [sweep(line, flows) for line in lines]
    reference = [rebuild(props, pts) for pts in points]
    total = readings[chosen]["total"]

    # ---- control: Dyssol at every point
    sources = assign_sources(lines, stored)
    installation, installation_record = dyssol_installation(patched)
    work = runs_dir() / "work" / args.name
    dyssol: list[list[LineOutputs | None]] = []
    failures: list[dict[str, Any]] = []
    seconds: list[float] = []
    for p in processes:
        flowsheet = DyssolFlowsheet(DELIVERY / sources[p], installation, work, names)
        outputs: list[LineOutputs | None] = []
        for j, point in enumerate(points[p]):
            tag = f"process{p + 1}_flow{j:02d}"
            try:
                evaluation = flowsheet.evaluate([point], tag)
            except Exception as exc:  # a failing run must not stop the others
                failures.append({"process": p + 1, "flow_index": j, "error": type(exc).__name__})
                print(f"{tag}: FAILED {type(exc).__name__}: {exc}", flush=True)
                outputs.append(None)
                continue
            outputs.append(evaluation.outputs[0])
            seconds.append(evaluation.wall_seconds)
            print(f"{tag}: done in {evaluation.wall_seconds:.2f} s", flush=True)
        dyssol.append(outputs)
    control_per_process = [control(dyssol[p], reference[p]) for p in processes]
    held = (
        not failures
        and max(c["max_temperature_k"] for c in control_per_process) <= config.control_temperature_k
        and max(c["max_relative_percent"] for c in control_per_process)
        <= config.control_relative_percent
    )
    dyssol_against_sheet: list[dict[str, Any] | None] = []
    for p in processes:
        complete = [o for o in dyssol[p] if o is not None]
        dyssol_against_sheet.append(
            against_sheet(complete, sheet, p) if len(complete) == len(flows) else None
        )
    dyssol_total = (
        summed([d for d in dyssol_against_sheet if d is not None])
        if all(d is not None for d in dyssol_against_sheet)
        else None
    )

    # ---- without a reading
    box = load_box(BOX_PATH, BOX_CASE, [stored[BOX_SOURCE]])
    without_reading: list[dict[str, Any]] = []
    for p in processes:
        inside_by_input = {
            name: sum(lo <= getattr(pt, name) <= hi for pt in points[p])
            for name, lo, hi in zip(VARIED, box.lower, box.upper, strict=True)
        }
        inside = [
            all(
                lo <= getattr(pt, n) <= hi
                for n, lo, hi in zip(VARIED, box.lower, box.upper, strict=True)
            )
            for pt in points[p]
        ]
        source = stored[sources[p]]
        same_state = (
            abs(source.gas_celsius - lines[p].gas_celsius) <= 1e-9
            and abs(source.gas_humidity_g_per_kg - lines[p].gas_humidity_g_per_kg) <= 1e-9
        )
        with_stored_state = rebuild(
            props,
            [
                replace(
                    pt,
                    gas_celsius=source.gas_celsius,
                    gas_humidity_g_per_kg=source.gas_humidity_g_per_kg,
                )
                for pt in points[p]
            ],
        )
        exhaust_shift = [
            s.exhaust_celsius - r.exhaust_celsius
            for s, r in zip(with_stored_state, reference[p], strict=True)
        ]
        fuel_shift = [
            100.0 * (s.fuel_kg_per_h - r.fuel_kg_per_h) / r.fuel_kg_per_h
            for s, r in zip(with_stored_state, reference[p], strict=True)
        ]
        dyssol_ratio = [o.saturation_ratio for o in dyssol[p] if o is not None]
        without_reading.append(
            {
                "process": p + 1,
                "source_flowsheet": sources[p],
                "saturation_ratio_rebuild": span([o.saturation_ratio for o in reference[p]]),
                "saturation_ratio_dyssol": span(dyssol_ratio),
                "oversaturated": any(o.oversaturated for o in reference[p]),
                "box_one_line": {
                    "points": len(flows),
                    "points_inside": sum(inside),
                    "points_inside_by_input": inside_by_input,
                },
                "fresh_gas_state_is_the_stored_one": same_state,
                "shift_with_stored_fresh_gas_state": {
                    "exhaust_k_all_points": span(exhaust_shift),
                    "fuel_percent_all_points": span(fuel_shift),
                    "exhaust_k_points_inside_box": span(
                        [v for v, ok in zip(exhaust_shift, inside, strict=True) if ok]
                    ),
                    "fuel_percent_points_inside_box": span(
                        [v for v, ok in zip(fuel_shift, inside, strict=True) if ok]
                    ),
                },
            }
        )

    with gzip.open(REFERENCE_ARTIFACT, "rt", encoding="utf-8") as fh:
        befund_128 = json.load(fh)
    same_setup = {
        "units_libraries": befund_128["data"]["units_manifest"]["libraries"]
        == installation_record["units_libraries"],
        "patched_materials": befund_128["data"]["materials"]["patched_sha256"]
        == sha256_of(patched),
        "box": befund_128["header"]["box_sha256"] == box.sha256(),
        "source_of_one_line": befund_128["header"]["source_sha256"]
        == sha256_of(DELIVERY / BOX_SOURCE),
    }

    code = verdict(total)
    payload: dict[str, Any] = {
        "_provenance": record,
        "name": args.name,
        "inputs": {
            "workbook_sha256": sha256_of(WORKBOOK),
            "sources_sha256": {name: sha256_of(DELIVERY / name) for name in SOURCE_FILES},
            "box_sha256": box.sha256(),
            "materials": {
                "installed_sha256": sha256_of(installed),
                "patched_sha256": sha256_of(patched),
            },
            "config_fingerprint": config.fingerprint(),
            **installation_record,
            "same_as_dyssol_steady_v1_one_line_training": same_setup,
        },
        "sheet": {
            "processes": len(sheet.lines),
            "medium_flows": len(flows),
            "reported_decimals": REPORTED_DECIMALS,
            "reported_values": total["n"],
            "condensation": sheet.condensation,
        },
        "readings_of_the_gas_flow": readings,
        "chosen_reading": chosen,
        "control": {
            "bounds": {
                "temperature_k": config.control_temperature_k,
                "relative_percent": config.control_relative_percent,
            },
            "processes": control_per_process,
            "failures": failures,
            "held": held,
            "dyssol_wall_seconds": span(seconds),
        },
        "dyssol_against_sheet": {"processes": dyssol_against_sheet, "total": dyssol_total},
        "verdict": {
            "code": code,
            "on": "rebuild" if held else "see dyssol_against_sheet (control not held)",
            "total": total,
            "dyssol_code": verdict(dyssol_total) if dyssol_total is not None else None,
        },
        "without_reading": without_reading,
    }

    print(f"chosen reading of the gas flow: {chosen}")
    for reading in GAS_FLOW_READINGS:
        print(f"  {reading}: {readings[reading]['total']}")
    print(f"verdict on the rebuild: {code} {total}")
    for p in processes:
        for key, entry in readings[chosen]["processes"][p].items():
            print(
                f"  process {p + 1} {key:20s} n {entry['n']:2d} half {entry['within_half']:2d} "
                f"one {entry['within_one']:2d} rounded {entry['equal_if_rounded']:2d} "
                f"truncated {entry['equal_if_truncated']:2d} d {entry['min_signed']:+.3f}.."
                f"{entry['max_signed']:+.3f} rel {entry['max_relative_percent']:.2f} %"
            )
    print(f"control held: {held}; failures {len(failures)}; {control_per_process}")
    print(f"Dyssol against the sheet: {dyssol_total}")
    for entry in without_reading:
        print(f"  without reading: {entry}")
    print(f"same setup as Befund 128: {same_setup}")
    if args.dry_run:
        print("dry run: nothing written")
        return
    write_artifact(args.name, payload)


if __name__ == "__main__":
    main()
