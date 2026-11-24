"""Stationary Dyssol flowsheets as data for surrogates: pilot, data generation, training.

Design and pre-registered reading: ``docs/45_dyssol_steady_state.md`` and the docstring of
``scripts/analyze_dyssol_steady.py``; settings: ``conf/dyssol_steady/dyssol_steady.yaml``; package
``twinlink_ml.dyssol_steady``.

Stages (``--case`` is a key of ``cases`` in the YAML)
------
``--stage pilot --name <pilot>``
    Selects the optimiser of the MLP on the PYTHON REBUILD of the flowsheet (no Dyssol run): every
    setting of ``pilot.grid`` at the pilot budgets and seeds, on the pilot's own pool and test
    points. Output ``artifacts/runs/<pilot>.json.gz`` (errors and the selection; no plant value).
    The selection is copied into ``mlp.selected.<case>`` of the YAML (own commit) before the data
    are generated.
``--stage data --name <data>``
    One Dyssol run per point of the training pool and of the test set (the units of this stage),
    with the rebuild evaluated at the same point. Every finished point is appended to
    ``<out>/_<data>_cache.jsonl`` at once; ``--resume`` continues it (same commit, config and box
    required); a failing point is recorded with its message and does not stop the others. At the
    end the points are packed into ``<out>/<data>.json.gz``. LOCAL ONLY: the file holds the box and
    the inputs.
``--stage data --dry-run``
    Every point through the rebuild (extreme draws, over-saturation counts), Dyssol only at the
    points with the extreme rebuild results plus the first points of pool and test set: surfaces
    edge cases, records time and peak memory per Dyssol process.
``--stage training --name <training> --data <data>``
    Control first (Dyssol against the rebuild at every point; above the bounds of the YAML the
    stage stops). Then every cell (model x N x seed) is fitted and scored on the test set; cells
    are cached in ``<out>/_<training>_cache.jsonl`` (``--resume``). Output
    ``artifacts/runs/<training>.json.gz``: errors in physical units, counts, digests -- no input
    values. With ``--dry-run`` (on a dry-run data set) every fit takes 5 optimiser steps, points
    without a Dyssol result take the rebuild's, and the output stays in ``<out>``.

Local folders
-------------
``$TWINLINK_LOCAL_DIR`` (default ``<root>/local``): ``partner/<delivery>/`` with the source
flowsheets, ``dyssol_steady/box_local.yaml`` with the box. ``<out>`` =
``$TWINLINK_DYSSOL_OUT`` (default ``<root>/local/runs/dyssol_steady``). ``$TWINLINK_RUNS_DIR``
(default ``C:/twinlink_runs/<branch>``): compiled units (``scripts/build_dyssol_units.py``), the
patched materials database, scratch folders of the Dyssol runs. ``$DYSSOL_HOME`` (default
``C:/Program Files/Dyssol``).

Other settings (``--config``)
-----------------------------
``--config <file>`` names another YAML next to the default one. ``dyssol_freshgas.yaml``
(``docs/NN_dyssol_freshgas.md``) varies seven inputs (the fresh-gas temperature and its relative
humidity as well), reads its own box file, and adds

* a pilot grid given by its axes, extended once automatically where the selection lies at an edge
  (rule in the YAML, fixed before the pilot);
* a third set of points, the points of one process of a partner sheet (``sheet_test``): they get
  a Dyssol run and the control like every other point, are never part of the pool, of the Sobol
  test set or of the pilot, and are scored per cell as a second, named test set;
* the control tabulated over the fresh-gas temperature, and the count of points whose heater
  inlet is colder than the lower bound of the vapour-enthalpy correlation;
* in the dry run of the training stage a probe: the largest cell of every model with the full
  optimiser settings (time and peak memory per worker).

``$TWINLINK_DYSSOL_UNITS`` overrides the folder of the compiled units (default
``$TWINLINK_RUNS_DIR/units``).

Clean tree required; start with ``tools/run_here.sh scripts/run_dyssol_steady.py ...``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, _json_default, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.dyssol_steady import study  # noqa: E402
from twinlink_ml.dyssol_steady.carrier_sweep import saturation_ratios  # noqa: E402
from twinlink_ml.dyssol_steady.design import (  # noqa: E402
    RELATIVE_HUMIDITY,
    Box,
    apply_point,
    load_box,
    point_row,
    sobol_points,
    training_subset,
)
from twinlink_ml.dyssol_steady.flowsheet import (  # noqa: E402
    DyssolFlowsheet,
    DyssolInstallation,
    ReferenceFlowsheet,
    Topology,
    discover_topology,
    nominal_inputs,
    peak_working_set,
)
from twinlink_ml.dyssol_steady.materials import (  # noqa: E402
    DyssolProperties,
    compound_names,
    patch_materials_database,
    read_properties,
    sha256_of,
)
from twinlink_ml.dyssol_steady.models import MlpSettings  # noqa: E402
from twinlink_ml.dyssol_steady.reference import KELVIN, LineInputs, LineOutputs  # noqa: E402
from twinlink_ml.dyssol_steady.sweep_sheet import (  # noqa: E402
    RESULT_KEYS,
    compare,
    load_sweep_sheet,
    sweep_points,
    totals,
)
from twinlink_ml.partner_v4_1.comparison import load_dyssol_steady_state  # noqa: E402

REPO_ROOT = SCRIPTS.parent
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", REPO_ROOT / "local"))
OUT = Path(os.environ.get("TWINLINK_DYSSOL_OUT", REPO_ROOT / "local" / "runs" / "dyssol_steady"))
DYSSOL_HOME = Path(os.environ.get("DYSSOL_HOME", "C:/Program Files/Dyssol"))
DRY_STEPS = 5
CONFIG_DIR = REPO_ROOT / "conf" / "dyssol_steady"
SHEET_SET = "sheet"
_WORKER: dict[str, Any] = {}


def runs_dir() -> Path:
    default = Path("C:/twinlink_runs") / (git("rev-parse", "--abbrev-ref", "HEAD") or "detached")
    return Path(os.environ.get("TWINLINK_RUNS_DIR", default))


def units_dir() -> Path:
    return Path(os.environ.get("TWINLINK_DYSSOL_UNITS", runs_dir() / "units"))


# ----------------------------------------------------------------------------- context
@dataclass(frozen=True)
class SheetSet:
    """Points of one process of a partner sheet as a second test set (local values)."""

    name: str
    rows: np.ndarray
    """Design rows of the points, in the order of the sheet."""
    reported: dict[str, tuple[float | None, ...]]
    """The sheet's own values per quantity (descriptive comparison only)."""
    workbook_sha256: str
    gas_flow_reading: str


@dataclass(frozen=True)
class Context:
    """Everything a stage needs about one case."""

    config: study.StudyConfig
    case: str
    source: Path
    names: dict[str, str]
    properties: DyssolProperties
    topology: Topology
    nominal: tuple[LineInputs, ...]
    box: Box
    materials_db: Path
    materials: dict[str, str]
    sheet: SheetSet | None = None

    def lines_at(self, row: np.ndarray) -> tuple[LineInputs, ...]:
        """Line inputs of one design row."""
        return apply_point(self.nominal, np.asarray(row), self.config.varied)


def load_sheet_set(
    config: study.StudyConfig, nominal: tuple[LineInputs, ...], box: Box
) -> SheetSet:
    """The sheet's points as design rows; aborts unless the surrogate's inputs can express them."""
    spec = config.sheet_test
    if spec is None or len(nominal) != 1:
        raise SystemExit("sheet_test needs a config entry and a flowsheet with one line")
    workbook = LOCAL / "partner" / str(spec["delivery"]) / str(spec["workbook"])
    if not workbook.exists():
        raise SystemExit(f"missing local file: {workbook}")
    sheet = load_sweep_sheet(workbook)
    process = int(spec["process"]) - 1
    points = sweep_points(sheet, process, str(spec["gas_flow_reading"]))
    if len(points) != int(spec["points"]):
        raise SystemExit(f"the sheet has {len(points)} points, the YAML says {spec['points']}")
    varied = config.varied
    fixed = [
        name
        for name in LineInputs.__dataclass_fields__
        if name not in varied
        and not (name == "gas_humidity_g_per_kg" and RELATIVE_HUMIDITY in varied)
    ]
    rows = []
    for point in points:
        differing = [
            name
            for name in fixed
            if not math.isclose(getattr(point, name), getattr(nominal[0], name), rel_tol=1e-12)
        ]
        if differing:
            raise SystemExit(f"sheet point differs from the stored flowsheet in {differing}")
        row = point_row([point], varied)
        (back,) = apply_point(nominal, row, varied)
        for name in LineInputs.__dataclass_fields__:
            if not math.isclose(
                getattr(back, name), getattr(point, name), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise SystemExit(f"sheet point is not reproduced by its design row ({name})")
        rows.append(row)
    array = np.asarray(rows)
    if not bool(box.contains(array).all()):
        raise SystemExit("a point of the sheet lies outside the box")
    return SheetSet(
        name=str(spec["name"]),
        rows=array,
        reported={key: tuple(sheet.results[key][process]) for key in RESULT_KEYS},
        workbook_sha256=sha256_of(workbook),
        gas_flow_reading=str(spec["gas_flow_reading"]),
    )


def build_context(
    case: str, config_file: str = "dyssol_steady.yaml", sheet: bool = True
) -> Context:
    config = study.load_config(CONFIG_DIR / config_file)
    if case not in config.cases:
        raise SystemExit(f"unknown case {case!r}; known: {sorted(config.cases)}")
    source = (
        LOCAL / "partner" / str(config.raw["delivery"]) / str(config.cases[case]["source_file"])
    )
    box_path = LOCAL / "dyssol_steady" / config.box_file
    for path in (source, box_path, DYSSOL_HOME / "Materials.dmdb"):
        if not path.exists():
            raise SystemExit(f"missing local file: {path}")
    installed = DYSSOL_HOME / "Materials.dmdb"
    patched = runs_dir() / "materials" / f"Materials_{sha256_of(installed)[:12]}_boiling_point.dmdb"
    if not patched.exists():
        patch_materials_database(installed, patched)
    materials = {
        "installed_sha256": sha256_of(installed),
        "patched_sha256": sha256_of(patched),
        "patched_file": patched.name,
    }
    names = compound_names(patched)
    file = load_dyssol_steady_state(source)
    topology = discover_topology(file)
    if len(topology.lines) != int(config.cases[case]["lines"]):
        raise SystemExit(f"{source.name} has {len(topology.lines)} lines, the YAML says otherwise")
    nominal = nominal_inputs(file, topology, names)
    box = load_box(box_path, case, nominal, config.varied)
    return Context(
        config=config,
        case=case,
        source=source,
        names=names,
        properties=read_properties(patched),
        topology=topology,
        nominal=nominal,
        box=box,
        materials_db=patched,
        materials=materials,
        sheet=(
            load_sheet_set(config, nominal, box)
            if sheet and config.sheet_test is not None
            else None
        ),
    )


def header(ctx: Context, stage: str, name: str, dry_run: bool) -> dict[str, Any]:
    return {
        "stage": stage,
        "name": name,
        "case": ctx.case,
        "dry_run": dry_run,
        "git/commit": git("rev-parse", "HEAD"),
        "config_fingerprint": ctx.config.fingerprint,
        "box_sha256": ctx.box.sha256(),
        "source_sha256": sha256_of(ctx.source),
    }


class Cache:
    """Append-only JSONL of finished units; the first line is the header."""

    def __init__(self, path: Path, head: dict[str, Any], resume: bool) -> None:
        self.path = path
        self.done: dict[str, dict[str, Any]] = {}
        if path.exists():
            if not resume:
                raise SystemExit(f"{path} exists -- pass --resume or use a new name")
            with path.open(encoding="utf-8") as fh:
                first = json.loads(fh.readline())
                if first.get("header") != head:
                    raise SystemExit(f"{path} was written by another commit/config/box: {first}")
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("ok", True):
                        self.done[str(rec["id"])] = rec
            print(f"resume: {len(self.done)} cached units", flush=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"header": head}) + "\n", encoding="utf-8")

    def add(self, rec: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=_json_default) + "\n")
        if rec.get("ok", True):
            self.done[str(rec["id"])] = rec


def write_gz(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists -- artifacts are never overwritten; use a new name")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, default=_json_default)
    print(f"wrote {path} ({path.stat().st_size} bytes, sha256 {sha256_of(path)[:16]})", flush=True)


def outputs_to_json(outputs: tuple[LineOutputs, ...]) -> list[dict[str, float]]:
    return [asdict(o) for o in outputs]


def outputs_from_json(rows: list[dict[str, float]]) -> tuple[LineOutputs, ...]:
    return tuple(LineOutputs(**r) for r in rows)


# ----------------------------------------------------------------------------- fits
def _init_fit(payload: dict[str, Any]) -> None:
    import torch

    torch.set_num_threads(1)
    _WORKER.update(payload)


def _fit_cell(task: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        w = _WORKER
        idx = w["available"][training_subset(len(w["available"]), task["n"], task["seed"])]
        settings = None if task["settings"] is None else MlpSettings(**task["settings"])
        from twinlink_ml.dyssol_steady.models import MlpRegressor

        model = study.make_model(task["model"], settings, task["seed"]).fit(
            w["x_pool"][idx], w["y_pool"][idx]
        )
        prediction = model.predict(w["x_test"])
        result: dict[str, Any] = {
            "metrics": study.cell_metrics(w["y_test"], prediction, w["names"]),
        }
        mask = w["admissible"]
        if 0 < int(mask.sum()) < len(mask):
            result["metrics_admissible"] = study.cell_metrics(
                w["y_test"][mask], prediction[mask], w["names"]
            )
        strict = w.get("admissible_strict")
        if strict is not None and 0 < int(strict.sum()) < len(strict):
            result["metrics_admissible_strict"] = study.cell_metrics(
                w["y_test"][strict], prediction[strict], w["names"]
            )
        if w.get("x_sheet") is not None:
            # the points of the sheet are scored only; they never enter a fit
            result["metrics_sheet"] = study.sheet_errors(
                w["y_sheet"], model.predict(w["x_sheet"]), w["names"], w.get("reported")
            )
        if isinstance(model, MlpRegressor):
            result["final_loss"] = model.final_loss
        return {
            **task,
            "ok": True,
            **result,
            "seconds": time.perf_counter() - t0,
            "peak_memory_mb": (peak_working_set() or 0) / 2**20,
            "pid": os.getpid(),
        }
    except Exception:
        return {**task, "ok": False, "error": traceback.format_exc(), "pid": os.getpid()}


def run_fits(
    tasks: list[dict[str, Any]], payload: dict[str, Any], cache: Cache, workers: int
) -> list[dict[str, Any]]:
    todo = [t for t in tasks if t["id"] not in cache.done]
    print(f"{len(tasks)} cells, {len(todo)} to run, {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init_fit, initargs=(payload,)) as pool:
        futures = [pool.submit(_fit_cell, t) for t in todo]
        for k, future in enumerate(as_completed(futures), start=1):
            rec = future.result()
            cache.add(rec)
            state = "ok" if rec["ok"] else "FAILED"
            print(
                f"[{k}/{len(todo)}] {rec['id']} {state} {rec.get('seconds', 0.0):.1f} s "
                f"(elapsed {time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
            if not rec["ok"]:
                print(rec["error"], flush=True)
    missing = [t["id"] for t in tasks if t["id"] not in cache.done]
    if missing:
        raise SystemExit(f"{len(missing)} cells failed: {missing[:5]} -- fix and --resume")
    return [cache.done[t["id"]] for t in tasks]


def scaled(ctx: Context, x: np.ndarray) -> np.ndarray:
    return ctx.box.to_unit_symmetric(x)


# ----------------------------------------------------------------------------- pilot
def stage_pilot(ctx: Context, name: str, workers: int, resume: bool) -> None:
    config = ctx.config
    pilot = config.pilot
    reference = ReferenceFlowsheet(ctx.properties, ctx.topology.preheated_by)

    if ctx.sheet is not None:
        raise SystemExit("the pilot must not see the points of the sheet")

    def evaluated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = sobol_points(ctx.box, n, seed)
        outputs = [reference.evaluate(ctx.lines_at(row)).outputs for row in x]
        ok = np.asarray([not any(o.oversaturated for o in point) for point in outputs])
        return x, study.target_matrix(outputs), ok

    x_pool, y_pool, _ = evaluated(config.pool_size, int(pilot["pool_seed"]))
    x_test, y_test, admissible = evaluated(config.test_size, int(pilot["test_seed"]))
    names = study.target_names(len(ctx.nominal))

    def tasks_of(grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tasks = [
            {
                "id": f"s{k}-n{n}-seed{seed}",
                "setting": k,
                "model": "mlp",
                "n": int(n),
                "seed": int(seed),
                "settings": {"hidden": list(config.hidden), **setting},
            }
            for k, setting in enumerate(grid)
            for n in pilot["budgets"]
            for seed in pilot["seeds"]
        ]
        for t in tasks:
            t["settings"]["hidden"] = tuple(t["settings"]["hidden"])
            t["settings"]["steps"] = int(t["settings"]["steps"])
        return tasks

    grid = study.pilot_grid(pilot)
    payload = {
        "x_pool": scaled(ctx, x_pool),
        "y_pool": y_pool,
        "x_test": scaled(ctx, x_test),
        "y_test": y_test,
        "admissible": admissible,
        "available": np.arange(len(x_pool)),
        "names": names,
    }
    head = header(ctx, "pilot", name, False)
    cache = Cache(OUT / f"_{name}_cache.jsonl", head, resume)
    cells = run_fits(tasks_of(grid), payload, cache, workers)
    selection = study.pilot_selection(cells, config)
    chosen = grid[selection["selected_setting"]]
    extension: dict[str, Any] | None = None
    if "axes" in pilot:
        # one automatic extension where the selection lies at an edge (rule of the YAML)
        first = {"settings": len(grid), "selected_setting": selection["selected_setting"]}
        step = study.pilot_extension(pilot, chosen)
        print("round 1 selected:", json.dumps(chosen), "extension:", json.dumps(step), flush=True)
        if step["extended"]:
            grid = study.pilot_grid(pilot, step["axes"])
            cells = run_fits(tasks_of(grid), payload, cache, workers)
            selection = study.pilot_selection(cells, config)
            chosen = grid[selection["selected_setting"]]
        final = study.pilot_extension({"axes": step["axes"]}, chosen)
        extension = {
            "round_1": {**first, "settings_selected": grid[first["selected_setting"]]},
            "rule": pilot.get("extension"),
            "extended": step["extended"],
            "at_edge_without_extension_round_1": step["at_edge_without_extension"],
            "axes_after_extension": step["axes"],
            "settings_after_extension": len(grid),
            "final_selection_at_edge_of": final["at_edge_without_extension"],
        }
        print("extension record:", json.dumps(extension))
    print("criterion per setting:", json.dumps(selection["criterion"]))
    print("selected:", json.dumps(chosen))
    extra = {} if extension is None else {"grid": grid, "extension": extension}
    write_gz(
        RUNS_DIR / f"{name}.json.gz",
        {
            "provenance": provenance(ignore_paths=[RUNS_DIR / f"{name}.json.gz"]),
            "header": head,
            "source": "python rebuild (no Dyssol run)",
            "materials": ctx.materials,
            "target_names": names,
            "selection": {**selection, "settings": chosen},
            **extra,
            "cells": [
                {
                    k: c[k]
                    for k in ("id", "setting", "n", "seed", "metrics", "final_loss", "seconds")
                }
                for c in cells
            ],
        },
    )


# ----------------------------------------------------------------------------- data
def dyssol_installation(ctx: Context) -> tuple[DyssolInstallation, dict[str, Any]]:
    units = units_dir()
    manifest_path = units / "manifest.json"
    executable = DYSSOL_HOME / "DyssolC.exe"
    if not manifest_path.exists() or not executable.exists():
        raise SystemExit(f"need {executable} and {manifest_path} (scripts/build_dyssol_units.py)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for dll, digest in manifest["libraries"].items():
        if sha256_of(units / dll) != digest:
            raise SystemExit(
                f"{units / dll} differs from its manifest -- rebuild into a new folder"
            )
    version = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return DyssolInstallation(executable, units, ctx.materials_db), {
        "dyssol_version": version,
        "units_manifest": manifest,
    }


def point_deviation(p: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    """Deviation of Dyssol from the rebuild at one point: temperatures [K], the rest [%]."""
    temperature = relative = 0.0
    worst: dict[str, float] = {}
    for d, r in zip(p["dyssol"], p["reference"], strict=True):
        for key, value in d.items():
            if key == "saturation_ratio":
                continue
            diff = abs(value - r[key])
            if key.endswith("_celsius"):
                temperature = max(temperature, diff)
            else:
                diff = 100.0 * diff / max(abs(value), 1.0)
                relative = max(relative, diff)
            worst[key] = max(worst.get(key, 0.0), diff)
    return temperature, relative, worst


def control_summary(
    points: list[dict[str, Any]], table: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Largest deviation of Dyssol from the rebuild over the points that have both.

    ``table`` (settings with the fresh-gas temperature as an input): ``column`` of that input,
    bin ``edges`` [degC] and ``vapour_lower_k``, the lower bound of the vapour-enthalpy
    correlation. Then the deviation is also given per bin, per set, and over the points whose
    heater inlet (rebuild) is colder than that bound.
    """
    temperature = relative = 0.0
    worst: dict[str, float] = {}
    compared = [p for p in points if "dyssol" in p]
    deviations = [point_deviation(p) for p in compared]
    for kelvin, percent, by_key in deviations:
        temperature, relative = max(temperature, kelvin), max(relative, percent)
        for key, diff in by_key.items():
            worst[key] = max(worst.get(key, 0.0), diff)
    out: dict[str, Any] = {
        "points_compared": len(compared),
        "max_temperature_k": temperature,
        "max_relative_percent": relative,
        "max_by_quantity": worst,
    }
    if table is not None:

        def largest(rows: list[tuple[float, float, dict[str, float]]]) -> dict[str, Any]:
            return {
                "points": len(rows),
                "max_temperature_k": max((r[0] for r in rows), default=0.0),
                "max_relative_percent": max((r[1] for r in rows), default=0.0),
            }

        out["by_fresh_gas_celsius"] = study.control_by_bins(
            [p["x"][table["column"]] for p in compared],
            [d[0] for d in deviations],
            [d[1] for d in deviations],
            table["edges"],
        )
        out["by_set"] = {
            label: largest(
                [d for p, d in zip(compared, deviations, strict=True) if p["set"] == label]
            )
            for label in dict.fromkeys(p["set"] for p in compared)
        }
        cold = [
            d
            for p, d in zip(compared, deviations, strict=True)
            if min(r["preheated_celsius"] for r in p["reference"]) + KELVIN
            < table["vapour_lower_k"]
        ]
        out["heater_inlet_below_vapour_correlation"] = {
            "lower_bound_k": table["vapour_lower_k"],
            **largest(cold),
        }
    return out


def oversaturated_after_recovery(rows: list[dict[str, float]]) -> bool:
    """True if any line's gas is over-saturated at the dryer outlet or after the recovery."""
    return any(max(saturation_ratios(o)) > 1.0 for o in outputs_from_json(rows))


def data_summary(
    points: list[dict[str, Any]], table: dict[str, Any] | None = None, strict: bool = False
) -> dict[str, Any]:
    def over(p: dict[str, Any]) -> bool:
        rows = p.get("dyssol", p["reference"])
        return any(r["saturation_ratio"] > 1.0 for r in rows)

    by_set: dict[str, dict[str, int]] = {}
    for p in points:
        s = by_set.setdefault(
            p["set"], {"points": 0, "dyssol_ok": 0, "failed": 0, "oversaturated": 0}
        )
        s["points"] += 1
        s["dyssol_ok"] += int("dyssol" in p)
        s["failed"] += int(not p.get("ok", True))
        s["oversaturated"] += int(over(p))
        if strict:
            s["oversaturated_also_after_recovery"] = s.get(
                "oversaturated_also_after_recovery", 0
            ) + int(oversaturated_after_recovery(p.get("dyssol", p["reference"])))
    walls = [p["wall_seconds"] for p in points if "wall_seconds" in p]
    peaks = [p["peak_memory_bytes"] for p in points if p.get("peak_memory_bytes")]
    return {
        "by_set": by_set,
        "dyssol_wall_seconds": {
            "median": statistics.median(walls) if walls else None,
            "max": max(walls) if walls else None,
            "sum": sum(walls),
        },
        "dyssol_peak_memory_mb": max(peaks) / 2**20 if peaks else None,
        "control": control_summary(points, table),
    }


def control_table(ctx: Context) -> dict[str, Any] | None:
    """Settings of the control table over the fresh-gas temperature, if the YAML asks for one."""
    edges = ctx.config.raw["control"].get("fresh_gas_bins_celsius")
    if edges is None:
        return None
    if len(ctx.nominal) != 1 or "gas_celsius" not in ctx.config.varied:
        raise SystemExit("control.fresh_gas_bins_celsius needs one line with gas_celsius varied")
    return {
        "column": ctx.config.varied.index("gas_celsius"),
        "edges": [float(e) for e in edges],
        "vapour_lower_k": float(ctx.properties.vapour_enthalpy_range_k[0]),
    }


def strict_flags(ctx: Context) -> bool:
    return bool(ctx.config.raw.get("oversaturation_also_after_recovery", False))


def stage_data(ctx: Context, name: str, workers: int, resume: bool, dry_run: bool) -> None:
    config = ctx.config
    if not dry_run and config.selected.get(ctx.case) is None:
        raise SystemExit(f"mlp.selected.{ctx.case} is empty: pilot selection first (own commit)")
    installation, software = dyssol_installation(ctx)
    flowsheet = DyssolFlowsheet(ctx.source, installation, runs_dir() / "work" / name, ctx.names)
    reference = ReferenceFlowsheet(ctx.properties, ctx.topology.preheated_by)
    design = [
        ("train", sobol_points(ctx.box, config.pool_size, config.pool_seed)),
        ("test", sobol_points(ctx.box, config.test_size, config.test_seed)),
    ]
    if ctx.sheet is not None:
        # the points of the sheet: their own set, never part of the pool or of the Sobol test set
        for _, x in design:
            gap = np.abs(x[:, None, :] - ctx.sheet.rows[None, :, :]).max(axis=2).min()
            if not gap > 0.0:
                raise SystemExit("a design point coincides with a point of the sheet")
        design.append((SHEET_SET, ctx.sheet.rows))
    points: list[dict[str, Any]] = []
    for label, x in design:
        if not bool(ctx.box.contains(x).all()):
            raise SystemExit("design point outside the box")
        for i, row in enumerate(x):
            lines = ctx.lines_at(row)
            points.append(
                {
                    "id": f"{label}-{i:04d}",
                    "set": label,
                    "index": i,
                    "x": row.tolist(),
                    "reference": outputs_to_json(reference.evaluate(lines).outputs),
                }
            )
    selected = set(range(len(points)))
    if dry_run:
        columns = np.asarray(
            [[v for line in p["reference"] for v in line.values()] for p in points], dtype=float
        )
        finite = np.where(np.isfinite(columns), columns, 0.0)
        selected = set(finite.argmin(axis=0)) | set(finite.argmax(axis=0))
        selected |= set(range(8)) | set(range(config.pool_size, config.pool_size + 4))
        if ctx.sheet is not None:
            # also the extreme draws of every input and every point of the sheet
            inputs = np.asarray([p["x"] for p in points], dtype=float)
            selected |= set(inputs.argmin(axis=0)) | set(inputs.argmax(axis=0))
            selected |= {k for k, p in enumerate(points) if p["set"] == SHEET_SET}
        selected = {int(k) for k in selected}
    head = header(ctx, "data", name, dry_run)
    cache = Cache(OUT / f"_{name}_cache.jsonl", head, resume)
    todo = [k for k in sorted(selected) if points[k]["id"] not in cache.done]
    print(
        f"{len(points)} points, Dyssol at {len(selected)}, {len(todo)} to run, {workers} workers",
        flush=True,
    )

    def unit(k: int) -> dict[str, Any]:
        p = points[k]
        try:
            ev = flowsheet.evaluate(ctx.lines_at(np.asarray(p["x"])), p["id"])
            return {
                "id": p["id"],
                "ok": True,
                "dyssol": outputs_to_json(ev.outputs),
                "wall_seconds": ev.wall_seconds,
                "peak_memory_bytes": ev.peak_memory_bytes,
            }
        except Exception:
            return {"id": p["id"], "ok": False, "error": traceback.format_exc()}

    t0 = time.perf_counter()
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(unit, k) for k in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            rec = future.result()
            cache.add(rec)
            if not rec["ok"]:
                failures[rec["id"]] = rec["error"]
                print(f"FAILED {rec['id']}:\n{rec['error']}", flush=True)
            if done % 25 == 0 or done == len(todo) or not rec["ok"]:
                print(
                    f"[{done}/{len(todo)}] last {rec['id']} "
                    f"{rec.get('wall_seconds', float('nan')):.2f} s, "
                    f"elapsed {time.perf_counter() - t0:.0f} s, failed {len(failures)}",
                    flush=True,
                )
    for p in points:
        rec = cache.done.get(p["id"])
        if rec is not None:
            p.update({k: v for k, v in rec.items() if k != "id"})
        elif p["id"] in failures:
            p.update({"ok": False, "error": failures[p["id"]]})
    summary = data_summary(points, control_table(ctx), strict_flags(ctx))
    print(json.dumps(summary, indent=1))
    target = OUT / f"{name}.json.gz"
    write_gz(
        target,
        {
            "provenance": provenance(),
            "header": head,
            **software,
            "materials": ctx.materials,
            "properties": ctx.properties.as_dict(),
            "topology": asdict(ctx.topology),
            "nominal": [asdict(ln) for ln in ctx.nominal],
            "box": asdict(ctx.box),
            "elapsed_seconds": time.perf_counter() - t0,
            "workers": workers,
            "summary": summary,
            "points": points,
        },
    )


# ----------------------------------------------------------------------------- training
def stage_training(
    ctx: Context, name: str, data: str, workers: int, resume: bool, dry_run: bool
) -> None:
    config = ctx.config
    data_path = OUT / f"{data}.json.gz"
    with gzip.open(data_path, "rt", encoding="utf-8") as fh:
        dataset = json.load(fh)
    head = header(ctx, "training", name, dry_run)
    for key in ("case", "config_fingerprint", "box_sha256", "source_sha256"):
        if dataset["header"][key] != head[key]:
            raise SystemExit(f"data set and current {key} differ: {dataset['header'][key]}")
    if bool(dataset["header"]["dry_run"]) != dry_run:
        raise SystemExit("a dry-run data set needs --dry-run, a main data set must not use it")
    summary = dataset["summary"]
    control = summary["control"]
    exceeded = (
        control["max_temperature_k"] > config.control_temperature_k
        or control["max_relative_percent"] > config.control_relative_percent
    )
    print("control:", json.dumps(control))
    if exceeded:
        raise SystemExit("control exceeded: Dyssol and the rebuild disagree -- no fit; report")

    strict_by_label: dict[str, np.ndarray] = {}
    raw_x: dict[str, np.ndarray] = {}

    def arrays(label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        rows = [p for p in dataset["points"] if p["set"] == label]
        usable = [p for p in rows if "dyssol" in p or dry_run]
        outputs = [outputs_from_json(p.get("dyssol", p["reference"])) for p in usable]
        x = np.asarray([p["x"] for p in usable], dtype=float)
        ok = np.asarray([not any(o.oversaturated for o in point) for point in outputs])
        strict_by_label[label] = np.asarray(
            [not any(max(saturation_ratios(o)) > 1.0 for o in point) for point in outputs]
        )
        raw_x[label] = x
        return scaled(ctx, x), study.target_matrix(outputs), ok, len(rows) - len(usable)

    x_pool, y_pool, _, pool_missing = arrays("train")
    x_test, y_test, admissible, test_missing = arrays("test")
    if len(x_pool) < config.budgets[-1]:
        raise SystemExit("fewer usable pool points than the largest budget")
    names = study.target_names(len(ctx.nominal))
    settings = None
    if dry_run:
        first = study.pilot_grid(config.pilot)[0]
        settings = {
            "hidden": tuple(first.get("hidden", config.hidden)),
            "learning_rate": float(first["learning_rate"]),
            "steps": DRY_STEPS,
            "weight_decay": float(first["weight_decay"]),
        }
    else:
        selected = config.mlp_settings(ctx.case)
        settings = asdict(selected)
    tasks = [
        {
            "id": f"{model}-n{n}-seed{seed}",
            "model": model,
            "n": int(n),
            "seed": int(seed),
            "settings": settings if model == "mlp" else None,
        }
        for model in study.MODELS
        for n in config.budgets
        for seed in config.seeds
    ]
    payload = {
        "x_pool": x_pool,
        "y_pool": y_pool,
        "x_test": x_test,
        "y_test": y_test,
        "admissible": admissible,
        "available": np.arange(len(x_pool)),
        "names": names,
    }
    extra: dict[str, Any] = {}
    if strict_flags(ctx):
        payload["admissible_strict"] = strict_by_label["test"]
        extra["test_set_admissible_also_after_recovery"] = int(strict_by_label["test"].sum())
    if ctx.sheet is not None:
        x_sheet, y_sheet, sheet_ok, sheet_missing = arrays(SHEET_SET)
        if sheet_missing or not np.allclose(raw_x[SHEET_SET], ctx.sheet.rows, rtol=1e-12, atol=0.0):
            raise SystemExit("the data set does not hold a result at every point of the sheet")
        payload.update(x_sheet=x_sheet, y_sheet=y_sheet, reported=ctx.sheet.reported)
        lower, upper = np.asarray(ctx.box.lower), np.asarray(ctx.box.upper)
        rows = ctx.sheet.rows
        fresh = [k for k, v in enumerate(config.varied) if v in ("gas_celsius", RELATIVE_HUMIDITY)]
        truth = {
            key: compare(
                y_sheet[:, names.index(f"line1/{study.SHEET_TARGETS[key]}")].tolist(),
                ctx.sheet.reported[key],
            )
            for key in RESULT_KEYS
        }
        extra["sheet_test"] = {
            "name": ctx.sheet.name,
            "points": len(rows),
            "workbook_sha256": ctx.sheet.workbook_sha256,
            "gas_flow_reading": ctx.sheet.gas_flow_reading,
            "gas_flow_reading_status": "inferred from the agreement (Befund 130), "
            "not confirmed by the sender",
            "truth": "our own Dyssol run at each point of the sheet",
            "points_inside_box": int(ctx.box.contains(rows).sum()),
            "points_strictly_inside_in_fresh_gas_state": int(
                np.all(
                    (rows[:, fresh] > lower[fresh]) & (rows[:, fresh] < upper[fresh]), axis=1
                ).sum()
            ),
            "points_on_a_face_of_the_box": {
                v: int(((rows[:, k] <= lower[k]) | (rows[:, k] >= upper[k])).sum())
                for k, v in enumerate(config.varied)
            },
            "oversaturated": int((~sheet_ok).sum()),
            "oversaturated_also_after_recovery": int((~strict_by_label[SHEET_SET]).sum()),
            "dyssol_against_sheet": {"per_quantity": truth, "total": totals(truth)},
        }
    t0 = time.perf_counter()
    cache = Cache(OUT / f"_{name}_cache.jsonl", head, resume)
    cells = run_fits(tasks, payload, cache, workers)
    if dry_run and config.selected.get(ctx.case) is not None:
        # probe with full-size fits: the largest cell of every model, full optimiser settings
        probe = [
            {
                "id": f"probe-{model}-n{config.budgets[-1]}-seed{config.seeds[0]}",
                "model": model,
                "n": int(config.budgets[-1]),
                "seed": int(config.seeds[0]),
                "settings": asdict(config.mlp_settings(ctx.case)) if model == "mlp" else None,
            }
            for model in study.MODELS
        ]
        probed = run_fits(probe, payload, cache, workers)
        extra["probe_full_size"] = [
            {k: c[k] for k in ("id", "seconds", "peak_memory_mb")} for c in probed
        ]
        print("probe (full size):", json.dumps(extra["probe_full_size"]), flush=True)
    target = (OUT if dry_run else RUNS_DIR) / f"{name}.json.gz"
    keep = (
        "id",
        "model",
        "n",
        "seed",
        "metrics",
        "metrics_admissible",
        "metrics_admissible_strict",
        "metrics_sheet",
        "final_loss",
        "seconds",
    )
    write_gz(
        target,
        {
            "provenance": provenance(ignore_paths=[target]),
            "header": head,
            "data": {
                "name": data,
                "sha256": sha256_of(data_path),
                "git/commit": dataset["provenance"]["git/commit"],
                "dyssol_version": dataset.get("dyssol_version"),
                "units_manifest": dataset.get("units_manifest"),
                "materials": dataset["materials"],
                "summary": summary,
                "pool_points_without_result": pool_missing,
                "test_points_without_result": test_missing,
            },
            "test_set": {
                "name": config.test_set_name,
                "points": len(x_test),
                "admissible_points": int(admissible.sum()),
            },
            **extra,
            "mlp_settings": {**settings, "hidden": list(settings["hidden"])},
            "target_names": names,
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - t0,
            "peak_memory_mb_per_worker": max(c["peak_memory_mb"] for c in cells),
            "cells": [{k: c[k] for k in keep if k in c} for c in cells],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("pilot", "data", "training"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", help="name of the data set (stage training)")
    parser.add_argument(
        "--config", default="dyssol_steady.yaml", help="settings file in conf/dyssol_steady/"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cores = (os.cpu_count() or 2) // 2
    if not 1 <= args.workers <= max(cores, 1):
        raise SystemExit(f"--workers must be between 1 and the physical cores ({cores})")
    require_clean_tree()
    ctx = build_context(args.case, args.config, sheet=args.stage != "pilot")
    print(
        f"stage {args.stage}, case {ctx.case}, lines {len(ctx.nominal)}, "
        f"box {ctx.box.sha256()[:12]}, config {ctx.config.fingerprint[:12]}",
        flush=True,
    )
    if args.stage == "pilot":
        if args.dry_run:
            raise SystemExit("the pilot has no dry run")
        stage_pilot(ctx, args.name, args.workers, args.resume)
    elif args.stage == "data":
        stage_data(ctx, args.name, args.workers, args.resume, args.dry_run)
    else:
        if not args.data:
            raise SystemExit("--stage training needs --data")
        stage_training(ctx, args.name, args.data, args.workers, args.resume, args.dry_run)


if __name__ == "__main__":
    main()
