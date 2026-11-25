r"""Process 1 of the partner sweep sheet against the measured stationary operation of plant A.

Design: ``docs/51_sheet_plant_a.md``. Lab book: Befund 132.

**Question.** Befund 130 found the dryer exhaust of process 1 of the sheet of 2026-11-24
over-saturated at every point (saturation ratio 2.015). By its label in the sheet, process 1 is
the line of which the repository has two measured days (plant A; Befund 117). (1) Does the dryer
unit over-saturate at the measured operation? (2) Does it reproduce the measured exhaust
temperature when it is given the measured inputs? (3) Which input of process 1 carries the
over-saturation?

Inputs (confidential, never copied into the repository): the sheet and the one-line flowsheets
from ``$TWINLINK_LOCAL_DIR`` (default ``<root>/local``); the two exports of plant A through
``TWINLINK_PLANT_DATA_PROD1`` (day 1) and ``TWINLINK_PLANT_DATA_PROD2`` (day 2); the channel
roles from ``--roles`` (default ``conf/plant/plant_local.yaml``, git-ignored). Stationary segments
with the unchanged settings of ``conf/plant_scale/plant_scale.yaml``. Dyssol from
``$DYSSOL_HOME``, the compiled units from ``$TWINLINK_DYSSOL_UNITS`` (default: the folder built
for Befund 128, checked against its manifest), the patched materials database and the scratch
folders in ``$TWINLINK_RUNS_DIR`` (default ``C:/twinlink_runs/<branch>``).

**Points.** S: process 1 of the sheet without a medium flow (gas flow read as dry gas, Befund
130). M: every stationary segment of both days as inputs of the unit (process air, supply-air
temperature, spray rate, intake temperature measured; spray temperature, equilibrium moisture,
heater efficiency and calorific value of the sheet), for the solid fractions 0.65 (reference),
0.60, 0.634, 0.70 and the sheet's, and for two humidities of the gas: the dried-supply-air
placeholder of Befund 117 (reference) and the sheet's. R: S with ONE input replaced by the value
of a day (day aggregate = mean over its segments; reference solid fraction and humidity): dry-gas
flow, dryer inlet temperature, spray flow, solid fraction, gas humidity. B: from S, the spray
flow, the solid fraction and the dryer inlet temperature at which the saturation ratio is 1.

**Reading, fixed in this docstring before the first run of this script.**

* **Step 0, controls.** (a) S reproduces Befund 130: its saturation ratio equals the one stored
  in ``dyssol_sweep_case1_comparison.json`` to a relative 1e-12. (b) The segments are those of
  Befund 117: start and stop minute of all segments, the digests of both exports and of the role
  file equal those of ``plant_scale_v1.json``, and the day aggregates of the six quantities used
  here equal the stored ones to a relative 1e-9. (c) Dyssol against the rebuild at every point
  of S, M, R and B within the bounds of docs/45 (``control`` of
  ``conf/dyssol_steady/dyssol_steady.yaml``), no Dyssol run failed. If a control fails, the
  readings are not read (``read`` is false in the artifact) and the deviation is reported.
* **Reading F, feasibility first** (reference variant, all segments): number of segments at
  which the dryer exhaust of the unit is over-saturated. **F0** none; **F2** all; **F1**
  otherwise.
* **Reading T** (reference variant, segments that are not over-saturated): dryer outlet
  temperature of the unit minus the measured exhaust temperature at the granulator outlet.
  **T0** within 2 K at every segment (threshold of reading P1 of Befund 117); **T1** otherwise.
  Without such a segment "not applicable".
* **Reading A** (points R; a replacement "removes" the over-saturation if the saturation ratio is
  at or below 1). **A0** exactly one input removes it on both days (that input is named); **A1**
  more than one input removes it on both days, or none does on both days but at least one does
  on one day; **A2** no single replacement removes it on either day.

**Predictions, fixed with the reading** (from the day aggregates of docs/34 and arithmetic by
hand, disclosed in the document): F0; T1 with the unit warmer than the measured exhaust at every
read segment; A0 with the solid fraction. Each is reported as confirmed or not confirmed.
Expected in addition, not part of the check: the spray flow removes the over-saturation on day 1
only.

**Reported without a reading:** F and T in every variant; the unit's outlet against the measured
bed temperature; per segment the smallest solid fraction at which the unit is admissible (rebuild
only); the distance of S to admissibility (B).

**What is committed** (decision of the project lead, 2026-11-25): saturation ratios, counts,
temperature differences in K, relative changes, flags, digests. Whatever depends on an input of
the sheet is reduced to counts, flags and reading codes, except the saturation ratio of S
(already in the artifact of Befund 130) and the relative change of the spray flow and the change
of the dryer inlet temperature of B. Everything else goes to the local record
``<root>/local/runs/sheet_plant_a/<name>_local.json`` (git-ignored), which is copied to
``$TWINLINK_RUNS_DIR``.

Artifacts are never overwritten (``--name`` for a new one). Run from a clean working tree with
``tools/run_here.sh scripts/report_sheet_plant_a.py``. With ``--dry-run`` every point goes
through the rebuild and Dyssol runs at S and at the first measured point only; the readings are
printed, nothing is written and the tree may be dirty.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import (  # noqa: E402
    NO_HEAT_NETWORK,
    REPO_ROOT,
    RUNS_DIR,
    git,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.dyssol_steady import plant_check as pc  # noqa: E402
from twinlink_ml.dyssol_steady import study  # noqa: E402
from twinlink_ml.dyssol_steady.flowsheet import (  # noqa: E402
    DyssolFlowsheet,
    DyssolInstallation,
    discover_topology,
    nominal_inputs,
)
from twinlink_ml.dyssol_steady.materials import (  # noqa: E402
    compound_names,
    patch_materials_database,
    read_properties,
    sha256_of,
)
from twinlink_ml.dyssol_steady.reference import LineInputs, LineOutputs  # noqa: E402
from twinlink_ml.dyssol_steady.sweep_sheet import line_inputs, load_sweep_sheet  # noqa: E402
from twinlink_ml.partner_v4_1.comparison import load_dyssol_steady_state  # noqa: E402
from twinlink_ml.plant_scale import (  # noqa: E402
    load_day_series,
    load_plant_scale_config,
    segment_statistics,
    stationary_segments,
)

LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", REPO_ROOT / "local"))
DYSSOL_HOME = Path(os.environ.get("DYSSOL_HOME", "C:/Program Files/Dyssol"))
UNITS = Path(os.environ.get("TWINLINK_DYSSOL_UNITS", "C:/twinlink_runs/dyssol-steady-state/units"))
WORKBOOK = LOCAL / "partner" / "dyssol_2026-11-24" / "Simulation results_flow rates.xlsx"
DELIVERY = LOCAL / "partner" / "dyssol_2026-11-19"
SOURCE_FILES = ("Case 1.1.dflw", "Case 1.2.dflw")
DEFAULT_ROLES = REPO_ROOT / "conf" / "plant" / "plant_local.yaml"
BEFUND_130 = RUNS_DIR / "dyssol_sweep_case1_comparison.json"
BEFUND_117 = RUNS_DIR / "plant_scale_v1.json"
LOCAL_RUNS = REPO_ROOT / "local" / "runs" / "sheet_plant_a"
DEFAULT_NAME = "sheet_plant_a_v1"

PROCESS = 0
"""Process 1 of the sheet (0-based), the over-saturated one of Befund 130."""
GAS_FLOW = "dry"
"""Reading of the sheet's gas mass flow (inferred in Befund 130, not confirmed by the sender)."""
EXHAUST_THRESHOLD_K = 2.0
PREDICTIONS = {"F": "F0", "T": "T1:unit warmer", "A": "A0:solid_fraction"}
SATURATION_TOLERANCE = 1e-12
AGGREGATE_TOLERANCE = 1e-9
AGGREGATE_FIELDS = (
    "process_air_nm3h",
    "supply_celsius",
    "spray_kg_h",
    "ambient_celsius",
    "exhaust_celsius",
    "bed_celsius",
)
SOLID_BRACKET = (0.01, 0.99)
INLET_SPAN_K = 150.0


def runs_dir() -> Path:
    default = Path("C:/twinlink_runs") / (git("rev-parse", "--abbrev-ref", "HEAD") or "detached")
    return Path(os.environ.get("TWINLINK_RUNS_DIR", default))


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
    }


def source_of_process(sheet: Any, stored: dict[str, LineInputs]) -> str:
    """Source flowsheet of :data:`PROCESS` by the rule of Befund 130.

    Each process takes the flowsheet whose stored dryer inlet temperature is nearest to its own;
    the assignment must be one-to-one.
    """
    lines = [line_inputs(sheet, p, GAS_FLOW) for p in range(len(sheet.lines))]
    chosen = [
        min(stored, key=lambda name: abs(stored[name].dryer_inlet_celsius - ln.dryer_inlet_celsius))
        for ln in lines
    ]
    if len(set(chosen)) != len(lines):
        raise SystemExit("the processes of the sheet do not map one-to-one onto the flowsheets")
    return chosen[PROCESS]


def load_days(
    config: Any, roles: Path
) -> tuple[dict[str, list[pc.MeasuredPoint]], dict[str, dict[str, Any]]]:
    """Stationary segments of both days as measured points, and what control (b) compares."""
    segments: dict[str, list[pc.MeasuredPoint]] = {}
    found: dict[str, dict[str, Any]] = {}
    for day, env in config.data.days.items():
        path = os.environ.get(env)
        if not path:
            raise SystemExit(f"environment variable {env} (export of {day}) is not set")
        series = load_day_series(path, roles, day, config.data.role_key)
        windows = stationary_segments(series, config.stationarity)
        segments[day] = [
            pc.MeasuredPoint.from_statistics(f"{day}_s{i + 1}", segment_statistics(series, w))
            for i, w in enumerate(windows)
        ]
        found[day] = {
            "export_sha256": sha256_of(path),
            "windows": [(w.start, w.stop) for w in windows],
        }
    return segments, found


def control_segments(
    segments: dict[str, list[pc.MeasuredPoint]], found: dict[str, dict[str, Any]], roles: Path
) -> dict[str, Any]:
    """Control (b): the segments and the day aggregates are those of ``plant_scale_v1``."""
    stored = json.loads(BEFUND_117.read_text(encoding="utf-8"))
    days = stored["survey"]["days"]
    same_windows = same_exports = set(days) == set(segments)
    largest = 0.0
    if same_windows:
        for day, entry in days.items():
            windows = [(s["start_minute"], s["stop_minute"]) for s in entry["segments"]]
            same_windows = same_windows and windows == found[day]["windows"]
            same_exports = same_exports and entry["export_sha256"] == found[day]["export_sha256"]
            ours = pc.mean_point(day, segments[day])
            for name in AGGREGATE_FIELDS:
                reference = entry["aggregate"][name]
                largest = max(largest, abs(getattr(ours, name) - reference) / abs(reference))
    same_roles = stored["role_file_sha256"] == sha256_of(roles)
    return {
        "same_windows": same_windows,
        "same_exports": same_exports,
        "same_role_file": same_roles,
        "aggregate_max_relative_deviation": largest,
        "aggregate_tolerance": AGGREGATE_TOLERANCE,
        "held": same_windows and same_exports and same_roles and largest <= AGGREGATE_TOLERANCE,
    }


def control_sheet_point(ratio: float) -> dict[str, Any]:
    """Control (a): the saturation ratio of S is the one of Befund 130."""
    stored = json.loads(BEFUND_130.read_text(encoding="utf-8"))
    entry = next(e for e in stored["without_reading"] if e["process"] == PROCESS + 1)
    reference = entry["saturation_ratio_rebuild"]["min"]
    deviation = abs(ratio - reference) / reference
    return {
        "relative_deviation": deviation,
        "tolerance": SATURATION_TOLERANCE,
        "held": deviation <= SATURATION_TOLERANCE
        and entry["saturation_ratio_rebuild"]["max"] == reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--name", default=DEFAULT_NAME, help="name of the artifact")
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES, help="channel roles (local)")
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
    needed = [WORKBOOK, installed, BEFUND_130, BEFUND_117, args.roles]
    for path in (*needed, *(DELIVERY / n for n in SOURCE_FILES)):
        if not path.exists():
            raise SystemExit(f"missing file: {path}")
    steady = study.load_config()
    plant = load_plant_scale_config()
    placeholders = plant.placeholders
    density = plant.data.normal_density_dry_air_kg_per_m3

    sheet = load_sweep_sheet(WORKBOOK)
    sheet_line = line_inputs(sheet, PROCESS, GAS_FLOW)
    segments, found = load_days(plant, args.roles)
    solids = {
        f"{value:g}": value
        for value in (placeholders.solid_fraction_reference, *placeholders.solid_fraction_band)
    }
    solids[pc.SHEET] = sheet_line.solid_fraction
    humidities = {
        pc.HUMIDITY_REFERENCE: 1000.0 * placeholders.inlet_absolute_humidity_kg_per_kg,
        pc.SHEET: sheet_line.gas_humidity_g_per_kg,
    }
    reference = (f"{placeholders.solid_fraction_reference:g}", pc.HUMIDITY_REFERENCE)

    patched = runs_dir() / "materials" / f"Materials_{sha256_of(installed)[:12]}_boiling_point.dmdb"
    if not patched.exists():
        patch_materials_database(installed, patched)
    props = read_properties(patched)
    names = compound_names(patched)

    # ---- points and rebuild
    points = pc.build_points(sheet_line, segments, solids, humidities, reference, density)
    start = points[pc.SHEET_POINT]
    roots = {
        "spray_kg_per_h": pc.admissibility_root(
            props, start, "spray_kg_per_h", 1e-6 * start.spray_kg_per_h, start.spray_kg_per_h
        ),
        "solid_fraction": pc.admissibility_root(
            props, start, "solid_fraction", start.solid_fraction, SOLID_BRACKET[1]
        ),
        "dryer_inlet_celsius": pc.admissibility_root(
            props,
            start,
            "dryer_inlet_celsius",
            start.dryer_inlet_celsius,
            start.dryer_inlet_celsius + INLET_SPAN_K,
        ),
    }
    for name, value in roots.items():
        if value is not None:
            points[f"B|{name}"] = replace(start, **{name: value})
    outputs = {tag: pc.rebuild(props, line) for tag, line in points.items()}
    minimum_solid = {
        humidity: {
            p.label: pc.admissibility_root(
                props,
                points[pc.point_tag(p.label, reference[0], humidity)],
                "solid_fraction",
                *SOLID_BRACKET,
            )
            for pts in segments.values()
            for p in pts
        }
        for humidity in humidities
    }

    # ---- step 0: controls
    control_a = control_sheet_point(outputs[pc.SHEET_POINT].saturation_ratio)
    control_b = control_segments(segments, found, args.roles)
    stored: dict[str, LineInputs] = {}
    for name in SOURCE_FILES:
        file = load_dyssol_steady_state(DELIVERY / name)
        topology = discover_topology(file)
        if len(topology.lines) != 1 or tuple(topology.preheated_by) != (0,):
            raise SystemExit(f"{name} is not one line with its own heat recovery")
        stored[name] = nominal_inputs(file, topology, names)[0]
    source = source_of_process(sheet, stored)
    installation, installation_record = dyssol_installation(patched)
    flowsheet = DyssolFlowsheet(
        DELIVERY / source, installation, runs_dir() / "work" / args.name, names
    )
    first_measured = next(tag for tag in points if tag.startswith("M|"))
    to_run = [pc.SHEET_POINT, first_measured] if args.dry_run else list(points)
    dyssol: dict[str, LineOutputs | None] = {}
    failures: list[dict[str, Any]] = []
    seconds: list[float] = []
    for i, tag in enumerate(to_run):
        try:
            evaluation = flowsheet.evaluate([points[tag]], f"point{i:03d}")
        except Exception as exc:  # a failing run must not stop the others
            failures.append({"point": i, "kind": tag.split("|")[0], "error": type(exc).__name__})
            print(f"point {i:03d}: FAILED {type(exc).__name__}: {exc}", flush=True)
            dyssol[tag] = None
            continue
        dyssol[tag] = evaluation.outputs[0]
        seconds.append(evaluation.wall_seconds)
        print(
            f"point {i:03d} of {len(to_run)}: done in {evaluation.wall_seconds:.2f} s", flush=True
        )
    deviation = pc.control_deviation(dyssol, outputs)
    control_c = {
        **deviation,
        "bounds": {
            "temperature_k": steady.control_temperature_k,
            "relative_percent": steady.control_relative_percent,
        },
        "failures": failures,
        "dyssol_wall_seconds": [min(seconds), max(seconds)] if seconds else None,
        "held": not failures
        and deviation["points_compared"] == deviation["points"]
        and deviation["max_temperature_k"] <= steady.control_temperature_k
        and deviation["max_relative_percent"] <= steady.control_relative_percent,
    }
    read = bool(control_a["held"] and control_b["held"] and control_c["held"])

    repo, local = pc.summarise(
        points=points,
        outputs=outputs,
        segments=segments,
        solids=solids,
        humidities=humidities,
        reference=reference,
        threshold_k=EXHAUST_THRESHOLD_K,
        roots=roots,
        minimum_solid=minimum_solid,
        predictions=PREDICTIONS,
    )
    befund_130 = json.loads(BEFUND_130.read_text(encoding="utf-8"))["inputs"]
    same_setup = {
        "workbook": befund_130["workbook_sha256"] == sha256_of(WORKBOOK),
        "source": befund_130["sources_sha256"][source] == sha256_of(DELIVERY / source),
        "units_libraries": befund_130["units_libraries"] == installation_record["units_libraries"],
        "patched_materials": befund_130["materials"]["patched_sha256"] == sha256_of(patched),
    }
    payload: dict[str, Any] = {
        "_provenance": record,
        "name": args.name,
        "inputs": {
            "workbook_sha256": sha256_of(WORKBOOK),
            "source_sha256": sha256_of(DELIVERY / source),
            "materials": {
                "installed_sha256": sha256_of(installed),
                "patched_sha256": sha256_of(patched),
            },
            "exports_sha256": {day: entry["export_sha256"] for day, entry in found.items()},
            "role_file_sha256": sha256_of(args.roles),
            "dyssol_steady_config_fingerprint": steady.fingerprint,
            "plant_scale_config_sha256": sha256_of(
                REPO_ROOT / "conf" / "plant_scale" / "plant_scale.yaml"
            ),
            "gas_flow_reading": GAS_FLOW,
            "normal_density_dry_air_kg_per_m3": density,
            **installation_record,
            "same_as_dyssol_sweep_case1_comparison": same_setup,
        },
        "control": {
            "a_sheet_point": control_a,
            "b_segments": control_b,
            "c_dyssol": control_c,
            "read": read,
        },
        **repo,
    }

    print(f"control a (S as Befund 130): {control_a}")
    print(f"control b (segments of Befund 117): {control_b}")
    print(f"control c (Dyssol against the rebuild): {control_c}")
    print(f"readings are read: {read}")
    print(f"reading F: {repo['reading_f']}")
    print(f"reading T: {repo['reading_t']}")
    print(f"reading A: {repo['reading_a']}")
    print(f"predictions: {repo['predictions']}")
    for key, entry in repo["without_reading"]["variants"].items():
        print(
            f"  variant {key}: over-saturated {entry['oversaturated_segments']} of "
            f"{entry['segments']}, {entry['reading_f']}, {entry['reading_t']}"
        )
    print(
        f"  minimum solid fraction: {repo['without_reading']['minimum_solid_fraction_of_the_unit']}"
    )
    print(f"  distance to admissibility: {repo['without_reading']['distance_to_admissibility']}")
    print(f"same setup as Befund 130: {same_setup}")
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
