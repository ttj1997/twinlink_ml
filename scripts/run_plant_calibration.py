r"""Driver of the plant calibration study (docs/40_plant_calibration.md). Reading: analyze_plant_calibration.py.

Identifiability first, then a stationary calibration of at most three identifiable parameters with
a cross-check between the two measured days of plant A. Basis: data, stationary segments, roles
and the plant-near parametrisation of Befund 117 (``conf/plant_scale/plant_scale.yaml``, read
unchanged); study settings ``conf/plant_calibration/plant_calibration.yaml``. Raw exports only
through ``TWINLINK_PLANT_DATA_PROD1`` (day 1 = product 1) and ``TWINLINK_PLANT_DATA_PROD2`` (day 2 =
product 2); channel roles from the git-ignored ``conf/plant/plant_local.yaml`` (``--roles``). The
grey-box model is parametrised, not changed. Reference: solid fraction 0.65 on both days, nominal
kinetics. Step 3 of the request (transients) is not part of this study.

Call (Git Bash, from the worktree; main runs detached, output to a log)::

    TWINLINK_PLANT_DATA_PROD1=... TWINLINK_PLANT_DATA_PROD2=... \
        tools/run_here.sh scripts/run_plant_calibration.py --name plant_calibration_v1 \
        --roles <role file>

Dry run over ALL units at minimal cost (profile grid = the two bounds, fits with ``max_nfev`` 2;
record only under ``local/runs/<name>/``)::

    ... scripts/run_plant_calibration.py --name plant_calibration_dry1 --dry --roles <role file>

Stages (each unit saved at once as ``local/runs/<name>/<unit>.json.gz``; ``--resume`` skips units
already saved; a failing unit is logged and recorded, the others go on):

A. **Sensitivity at the nominal parameters** (uncalibrated model, switch off), one unit per
   stationary segment: base point and central differences (+- 0.01 x scale, clipped to the bounds: one-sided at a
   bound such as UA = 0) for all seven
   candidates; rows bed (sigma 0.2/sqrt(6) K) and exhaust (0.2 K) per segment.
B. **Selection** (no model runs): per data set (day 1, day 2, both) delta_msqr, SVD, condition
   number, collinearity index and determinant measure of every subset (<= 3) of the "steady"
   candidates; selected set = largest identifiable subset (rules in the YAML). Kinetics and the
   heat capacity are never calibrated (request; the heat capacity has no steady effect). Main
   variant: one distance per product; the parameter of the other product is not identifiable
   from one day and stays at its nominal 0 (switch off) in the cross-check (decision of the
   project lead, 2026-11-13). Descriptive variant without reading: one shared distance; its S
   column is the sum of the two per-product columns (exact: the distance is subtracted after the
   steady state).
C. **Fits** (bounded WLS, weights 1/sigma): main and shared variant on day 1 and on day 2, and the
   main variant on both days (basis of the profiles); every fit predicts all segments of both
   days. **Profiles**: the three candidates with the largest delta_msqr on the joint matrix of the
   main variant, each on 11 grid points over its bounds, the joint selected set (minus the
   profiled parameter) re-fitted.
D. **Sensitivity at the estimates** of the main variant (day 1, day 2, both days): as A.

Records: ``artifacts/runs/<name>.json.gz`` (committed after the run): segment windows and
stationarity ratios, normalised sensitivities, SVD and subset scores, fitted parameters, model
errors (model - measured) per segment, metrics, bias offsets, profiles (relative SSR). NO absolute
measured or modelled temperatures, flows or rates. ``local/reports/<name>_local.json.gz``
(git-ignored): segment points (absolute means), model outputs. The export file names never enter
any record (SHA-256 and the variable name only).
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import NO_HEAT_NETWORK, REPO_ROOT, RUNS_DIR, provenance  # noqa: E402

from twinlink_ml.plant_calibration import (  # noqa: E402
    ModelContext,
    apply_bias,
    bias_offsets,
    delta_msqr,
    error_metrics,
    evaluate_rows,
    fd_steps,
    finite_difference_thetas,
    fit_parameters,
    identifiable_subsets,
    load_plant_calibration_config,
    select_subset,
    sensitivity_from_outputs,
    svd_summary,
)
from twinlink_ml.plant_calibration.config import PlantCalibrationConfig  # noqa: E402
from twinlink_ml.plant_scale import (  # noqa: E402
    PlantPoint,
    load_day_series,
    load_plant_scale_config,
    segment_statistics,
    stationary_segments,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "conf" / "plant_calibration" / "plant_calibration.yaml"
DEFAULT_ROLES = REPO_ROOT / "conf" / "plant" / "plant_local.yaml"
LOCAL_RUNS = REPO_ROOT / "local" / "runs"
LOCAL_REPORTS = REPO_ROOT / "local" / "reports"
SHARED = "offset_shared_k"
PER_PRODUCT = ("offset_product_1_k", "offset_product_2_k")


# ------------------------------------------------------------------------------------ helpers


def clean(obj: Any) -> Any:
    """JSON-safe copy (NaN/inf -> None, numpy -> Python)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_gz(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(clean(obj), f, indent=1)
    tmp.replace(path)


def read_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_memory_mb() -> float | None:
    """Peak working set of this process (Windows), else None."""
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_: ClassVar = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    getinfo = kernel32.K32GetProcessMemoryInfo
    getinfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    getinfo.restype = ctypes.c_int
    ok = getinfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20 if ok else None


def log(message: str, started: float) -> None:
    print(f"[{time.perf_counter() - started:8.0f} s] {message}", flush=True)


# ------------------------------------------------------------------------------------ survey


def point_from_stats(label: str, stats: dict[str, Any]) -> PlantPoint:
    """As Befund 117 (scripts/run_plant_scale.py)."""
    sensors = stats["bed_sensors_mean"]
    downstream = stats.get("downstream_exhaust_temperature", {}).get("mean", math.nan)
    return PlantPoint(
        label=label,
        ambient_celsius=stats["ambient_temperature"]["mean"],
        process_air_nm3h=stats["process_air_flow"]["mean"],
        atomising_air_nm3h=stats["atomising_air_flow"]["mean"],
        supply_celsius=stats["supply_air_temperature"]["mean"],
        spray_kg_h=stats["spray_rate"]["mean"],
        bed_celsius=stats["bed_temperature"]["mean"],
        exhaust_celsius=stats["exhaust_outlet_temperature"]["mean"],
        bed_group_a_celsius=stats["bed_temperature_group_a"]["mean"],
        bed_group_b_celsius=stats["bed_temperature_group_b"]["mean"],
        bed_sensor_min_celsius=min(sensors),
        bed_sensor_max_celsius=max(sensors),
        downstream_celsius=downstream,
    )


def survey(ps: Any, roles: Path) -> tuple[dict[str, Any], list[PlantPoint], list[str]]:
    repo: dict[str, Any] = {}
    points: list[PlantPoint] = []
    days: list[str] = []
    for day, env in ps.data.days.items():
        path = os.environ.get(env)
        if not path:
            raise SystemExit(f"environment variable {env} (export of {day}) is not set")
        series = load_day_series(path, roles, day, ps.data.role_key)
        segments = stationary_segments(series, ps.stationarity)
        for i, seg in enumerate(segments):
            points.append(point_from_stats(f"{day}_s{i + 1}", segment_statistics(series, seg)))
            days.append(day)
        repo[day] = {
            "environment_variable": env,
            "export_sha256": sha256(Path(path)),
            "minutes": series.n_minutes,
            "segments": [s.as_dict() for s in segments],
            "segments_admissible": sum(s.admissible for s in segments),
        }
    return repo, points, days


# ------------------------------------------------------------------------------------ worker

_CTX: ModelContext | None = None


def _init_worker(ctx: ModelContext) -> None:
    global _CTX
    _CTX = ctx
    warnings.simplefilter("ignore", PlaceholderConfigWarning)


def _rows(
    points: Sequence[dict[str, Any]], products: Sequence[str]
) -> list[tuple[PlantPoint, str]]:
    return [(PlantPoint(**p), pr) for p, pr in zip(points, products, strict=True)]


def unit_sensitivity(payload: dict[str, Any]) -> dict[str, Any]:
    """Outputs of the finite-difference thetas at ONE segment (fixed velocity start)."""
    assert _CTX is not None
    started = time.perf_counter()
    rows = _rows([payload["point"]], [payload["product"]])
    thetas = finite_difference_thetas(
        payload["theta"],
        payload["names"],
        payload["scales"],
        payload["relative_step"],
        {k: tuple(v) for k, v in payload["bounds"].items()},
    )
    guesses: dict[str, float] = {}
    base = evaluate_rows(_CTX, rows, thetas[0], guesses)
    fixed = dict(guesses)
    outputs = [base]
    for t in thetas[1:]:
        outputs.append(evaluate_rows(_CTX, rows, t, dict(fixed)))
    return {
        "outputs": np.vstack(outputs).tolist(),
        "steps": fd_steps(thetas, payload["names"], payload["scales"]).tolist(),
        "seconds": time.perf_counter() - started,
        "peak_memory_mb": peak_memory_mb(),
    }


def unit_fit(payload: dict[str, Any]) -> dict[str, Any]:
    """Fit ``free`` on the rows ``fit_rows``; predict every row at the result."""
    assert _CTX is not None
    started = time.perf_counter()
    rows_all = _rows(payload["points"], payload["products"])
    idx = payload["fit_rows"]
    rows = [rows_all[i] for i in idx]
    measured = np.array(payload["measured"], dtype=np.float64)
    sigma = np.array(payload["sigma"], dtype=np.float64)
    sel = np.array([2 * i + k for i in idx for k in (0, 1)])
    guesses: dict[str, float] = {}

    def outputs_for(theta: Mapping[str, float]) -> np.ndarray:
        return evaluate_rows(_CTX, rows, theta, guesses)

    fit = fit_parameters(
        outputs_for,
        measured[sel],
        sigma[sel],
        payload["theta_fixed"],
        payload["free"],
        scales=payload["scales"],
        bounds={k: tuple(v) for k, v in payload["bounds"].items()},
        start=payload["start"],
        max_nfev=payload["max_nfev"],
    )
    out: dict[str, Any] = {"fit": fit.as_dict()}
    if payload.get("predict_all", True):
        out["outputs_all"] = evaluate_rows(_CTX, rows_all, fit.theta, {}).tolist()
    out["seconds"] = time.perf_counter() - started
    out["peak_memory_mb"] = peak_memory_mb()
    return out


UNIT_FUNCTIONS = {"sensitivity": unit_sensitivity, "fit": unit_fit}


def run_units(
    units: Mapping[str, tuple[str, dict[str, Any]]],
    stage_dir: Path,
    pool: ProcessPoolExecutor,
    started: float,
    resume: bool,
) -> dict[str, Any]:
    """Run units in the pool, save each result at once; return unit -> result."""
    results: dict[str, Any] = {}
    pending: dict[Future[Any], str] = {}
    for name, (kind, payload) in units.items():
        path = stage_dir / f"{name}.json.gz"
        if resume and path.exists():
            results[name] = read_gz(path)
            log(f"unit {name}: resumed", started)
            continue
        pending[pool.submit(UNIT_FUNCTIONS[kind], payload)] = name
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            name = pending.pop(fut)
            try:
                result = fut.result()
            except Exception as error:  # a failing unit must not stop the others
                result = {
                    "failure": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                log(f"unit {name}: FAILED {result['failure']}", started)
            else:
                log(
                    f"unit {name}: done ({result.get('seconds', 0):.0f} s, "
                    f"peak {result.get('peak_memory_mb') or 0:.0f} MB)",
                    started,
                )
            write_gz(stage_dir / f"{name}.json.gz", result)
            results[name] = result
    return results


# ------------------------------------------------------------------------------------ analysis


def shared_matrix(s: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Columns of the shared variant: the two per-product distances replaced by their sum."""
    i1, i2 = names.index(PER_PRODUCT[0]), names.index(PER_PRODUCT[1])
    keep = [j for j in range(len(names)) if j not in (i1, i2)]
    cols = [s[:, j] for j in keep]
    new_names = [names[j] for j in keep]
    pos = min(i1, i2)
    cols.insert(pos, s[:, i1] + s[:, i2])
    new_names.insert(pos, SHARED)
    return np.column_stack(cols), new_names


def identifiability_block(
    s: np.ndarray,
    names: Sequence[str],
    eligible: Sequence[str],
    pc: PlantCalibrationConfig,
    row_labels: Sequence[str],
) -> dict[str, Any]:
    st = pc.sensitivity
    scores = identifiable_subsets(
        s,
        names,
        eligible,
        delta_min=st.delta_min,
        gamma_max=st.gamma_max,
        max_size=st.max_parameters,
    )
    all_scores = identifiable_subsets(
        s, names, names, delta_min=st.delta_min, gamma_max=st.gamma_max, max_size=st.max_parameters
    )
    return {
        "rows": list(row_labels),
        "parameters": list(names),
        "S": s.tolist(),
        "delta_msqr": {n: float(v) for n, v in zip(names, delta_msqr(s), strict=True)},
        "svd": svd_summary(s, names),
        "subsets_eligible": [sc.as_dict() for sc in scores],
        "subsets_all": [sc.as_dict() for sc in all_scores],
        "eligible": list(eligible),
        "selected": list(select_subset(scores, names)),
        "largest_identifiable_any": list(select_subset(all_scores, names)),
    }


def assemble_s(
    results: Mapping[str, Any],
    prefix: str,
    n_points: int,
    pc: PlantCalibrationConfig,
    sigma: np.ndarray,
) -> np.ndarray:
    blocks = []
    for i in range(n_points):
        r = results[f"{prefix}_seg{i}"]
        if "failure" in r:
            blocks.append(np.full((2, len(pc.names)), np.nan))
            continue
        blocks.append(
            sensitivity_from_outputs(
                np.array(r["outputs"], dtype=np.float64),
                sigma[2 * i : 2 * i + 2],
                np.array(r["steps"], dtype=np.float64),
            )
        )
    return np.vstack(blocks)


# ------------------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--roles", default=str(DEFAULT_ROLES))
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    stage_dir = LOCAL_RUNS / args.name
    record_path = (
        (stage_dir / "record.json.gz") if args.dry else (RUNS_DIR / f"{args.name}.json.gz")
    )
    local_path = (
        (stage_dir / "local.json.gz")
        if args.dry
        else (LOCAL_REPORTS / f"{args.name}_local.json.gz")
    )
    if record_path.exists():
        raise SystemExit(f"{record_path} exists; artifacts are never overwritten")
    if stage_dir.exists() and not args.resume:
        raise SystemExit(f"{stage_dir} exists; use --resume or a new name")
    stage_dir.mkdir(parents=True, exist_ok=True)
    LOCAL_REPORTS.mkdir(parents=True, exist_ok=True)

    prov = provenance(NO_HEAT_NETWORK)
    started = time.perf_counter()
    pc = load_plant_calibration_config(args.config)
    ps = load_plant_scale_config(pc.plant_scale_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(ps.process_config)
    steady_kwargs = {
        "normal_density_kg_per_m3": ps.data.normal_density_dry_air_kg_per_m3,
        "inlet_humidity_kg_per_kg": ps.placeholders.inlet_absolute_humidity_kg_per_kg,
        "feed_celsius": ps.placeholders.feed_temperature_celsius,
        "iterations": ps.calibration.velocity_iterations,
        "relative_tolerance": ps.calibration.velocity_relative_tolerance,
    }
    ctx = ModelContext(base, ps.placeholders.solid_fraction_reference, steady_kwargs)
    workers = min(pc.workers, args.workers or pc.workers)
    max_nfev = 2 if args.dry else pc.calibration.max_nfev
    grid_points = 2 if args.dry else pc.profile.grid_points

    # ---- survey
    survey_repo, points, point_days = survey(ps, Path(args.roles))
    products = [pc.products[d] for d in point_days]
    n = len(points)
    measured = np.array([v for p in points for v in (p.bed_celsius, p.exhaust_celsius)])
    sigma = np.array([v for _ in points for v in (pc.sigma_bed_k, pc.sigma_exhaust_k)])
    row_labels = [f"{p.label}/{q}" for p in points for q in ("bed", "exhaust")]
    point_dicts = [dataclasses.asdict(p) for p in points]
    days = list(ps.data.days)
    day_rows = {d: [i for i in range(n) if point_days[i] == d] for d in days}
    log(
        f"survey: {n} segments ({', '.join(f'{d}: {len(v)}' for d, v in day_rows.items())})",
        started,
    )

    names = list(pc.names)
    scales = {p.name: p.scale for p in pc.parameters} | {SHARED: pc.shared_offset.scale}
    bounds = {p.name: p.bounds for p in pc.parameters} | {SHARED: pc.shared_offset.bounds}
    nominal = pc.nominal()
    steady_names = [p.name for p in pc.parameters if p.kind == "steady"]

    def s_units(prefix: str, theta: Mapping[str, float]) -> dict[str, tuple[str, dict[str, Any]]]:
        return {
            f"{prefix}_seg{i}": (
                "sensitivity",
                {
                    "point": point_dicts[i],
                    "product": products[i],
                    "theta": dict(theta),
                    "names": names,
                    "scales": scales,
                    "relative_step": pc.sensitivity.relative_step,
                    "bounds": {k: list(v) for k, v in bounds.items()},
                },
            )
            for i in range(n)
        }

    def fit_unit(
        fit_rows: Sequence[int],
        free: Sequence[str],
        theta_fixed: Mapping[str, float],
        predict_all: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        return (
            "fit",
            {
                "points": point_dicts,
                "products": products,
                "fit_rows": list(fit_rows),
                "measured": measured.tolist(),
                "sigma": sigma.tolist(),
                "theta_fixed": dict(theta_fixed),
                "free": list(free),
                "scales": scales,
                "bounds": {k: list(v) for k, v in bounds.items()},
                "start": dict(pc.calibration.start),
                "max_nfev": max_nfev,
                "predict_all": predict_all,
            },
        )

    record: dict[str, Any] = {
        "name": args.name,
        "dry": args.dry,
        "provenance": prov,
        "config": json.loads(json.dumps(dataclasses.asdict(pc), default=str)),
        "reference": {"solid_fraction": ctx.solid_fraction, "kinetics": "nominal"},
        "survey": survey_repo,
        "rows": row_labels,
        "sigma_k": {"bed": pc.sigma_bed_k, "exhaust": pc.sigma_exhaust_k},
        "workers": workers,
    }

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(ctx,)
    ) as pool:
        # ---- A: sensitivity at nominal
        res_a = run_units(s_units("S_nominal", nominal), stage_dir, pool, started, args.resume)
        s_nom = assemble_s(res_a, "S_nominal", n, pc, sigma)
        y0 = np.concatenate(
            [
                np.array(res_a[f"S_nominal_seg{i}"]["outputs"][0])
                if "failure" not in res_a[f"S_nominal_seg{i}"]
                else np.full(2, np.nan)
                for i in range(n)
            ]
        )
        log("stage A done", started)

        # ---- B: selection
        def rows_of(idx: Sequence[int]) -> np.ndarray:
            return np.array([2 * i + k for i in idx for k in (0, 1)])

        data_sets = {**{d: day_rows[d] for d in days}, "both": list(range(n))}
        s_shared, names_shared = shared_matrix(s_nom, names)
        steady_shared = [m for m in names_shared if m in steady_names or m == SHARED]
        ident: dict[str, Any] = {"nominal": {"main": {}, "shared": {}}}
        for label, idx in data_sets.items():
            r = rows_of(idx)
            labels = [row_labels[i] for i in r]
            ident["nominal"]["main"][label] = identifiability_block(
                s_nom[r], names, steady_names, pc, labels
            )
            ident["nominal"]["shared"][label] = identifiability_block(
                s_shared[r], names_shared, steady_shared, pc, labels
            )
        sel = {
            v: {d: ident["nominal"][v][d]["selected"] for d in data_sets} for v in ident["nominal"]
        }
        joint_delta = ident["nominal"]["main"]["both"]["delta_msqr"]
        order = {m: i for i, m in enumerate(names)}
        top3 = sorted(names, key=lambda m: (-(joint_delta[m] or 0.0), order[m]))[:3]
        log(f"stage B: selected {sel}; profiles for {top3}", started)

        # ---- C: fits and profiles
        theta_shared0 = dict(nominal) | {SHARED: pc.shared_offset.nominal}
        units: dict[str, tuple[str, dict[str, Any]]] = {}
        for d in days:
            units[f"fit_main_{d}"] = fit_unit(day_rows[d], sel["main"][d], nominal)
            units[f"fit_shared_{d}"] = fit_unit(day_rows[d], sel["shared"][d], theta_shared0)
        units["fit_main_both"] = fit_unit(list(range(n)), sel["main"]["both"], nominal)
        grids: dict[str, list[float]] = {}
        for m in top3:
            lo, hi = bounds[m]
            grids[m] = [float(v) for v in np.linspace(lo, hi, grid_points)]
            free = [f for f in sel["main"]["both"] if f != m]
            for k, v in enumerate(grids[m]):
                units[f"profile_{m}_{k}"] = fit_unit(
                    list(range(n)), free, dict(nominal) | {m: v}, predict_all=False
                )
        res_c = run_units(units, stage_dir, pool, started, args.resume)
        log("stage C done", started)

        # ---- D: sensitivity at the estimates (main variant)
        units_d: dict[str, tuple[str, dict[str, Any]]] = {}
        for label in (*days, "both"):
            r = res_c[f"fit_main_{label}"]
            if "failure" not in r:
                units_d |= s_units(f"S_est_{label}", r["fit"]["theta"])
        res_d = run_units(units_d, stage_dir, pool, started, args.resume)
        log("stage D done", started)

    # ---- assemble
    for label in (*days, "both"):
        prefix = f"S_est_{label}"
        if f"{prefix}_seg0" not in res_d:
            continue
        s_est = assemble_s(res_d, prefix, n, pc, sigma)
        ident[f"estimate_{label}"] = {
            "main": {
                d: identifiability_block(
                    s_est[rows_of(idx)],
                    names,
                    steady_names,
                    pc,
                    [row_labels[i] for i in rows_of(idx)],
                )
                for d, idx in data_sets.items()
            }
        }
    record["identifiability"] = ident
    record["profile_parameters"] = top3

    # uncalibrated model and bias per calibration day
    local_outputs: dict[str, Any] = {"uncalibrated": y0.tolist()}
    calib: dict[str, Any] = {"uncalibrated": {}, "variants": {}}
    for d in days:
        r = rows_of(day_rows[d])
        calib["uncalibrated"][d] = error_metrics(y0[r], measured[r])
    other = {days[0]: days[1], days[1]: days[0]}
    bias_block: dict[str, Any] = {}
    for d in days:
        rc, ro = rows_of(day_rows[d]), rows_of(day_rows[other[d]])
        b = bias_offsets(y0[rc], measured[rc])
        corrected = apply_bias(y0, b)
        bias_block[d] = {
            "offsets_k": b,
            "own_day": error_metrics(corrected[rc], measured[rc]),
            "other_day": error_metrics(corrected[ro], measured[ro]),
        }
    calib["bias"] = bias_block
    for variant in ("main", "shared"):
        vblock: dict[str, Any] = {}
        for d in days:
            r = res_c[f"fit_{variant}_{d}"]
            if "failure" in r:
                vblock[d] = {"failure": r["failure"]}
                continue
            out = np.array(r["outputs_all"], dtype=np.float64)
            local_outputs[f"fit_{variant}_{d}"] = out.tolist()
            rc, ro = rows_of(day_rows[d]), rows_of(day_rows[other[d]])
            vblock[d] = {
                "selected": sel[variant][d],
                "fit": r["fit"],
                "own_day": error_metrics(out[rc], measured[rc]),
                "other_day": error_metrics(out[ro], measured[ro]),
                "seconds": r.get("seconds"),
            }
        calib["variants"][variant] = vblock
    rj = res_c["fit_main_both"]
    if "failure" not in rj:
        out = np.array(rj["outputs_all"], dtype=np.float64)
        local_outputs["fit_main_both"] = out.tolist()
        calib["joint_main"] = {
            "selected": sel["main"]["both"],
            "fit": rj["fit"],
            "per_day": {
                d: error_metrics(out[rows_of(day_rows[d])], measured[rows_of(day_rows[d])])
                for d in days
            },
        }
    else:
        calib["joint_main"] = {"failure": rj["failure"]}
    record["calibration"] = calib

    profiles: dict[str, Any] = {}
    for m in top3:
        rows_p = []
        for k, v in enumerate(grids[m]):
            r = res_c[f"profile_{m}_{k}"]
            rows_p.append(
                {"value": v, "failure": r.get("failure")}
                if "failure" in r
                else {
                    "value": v,
                    "ssr": r["fit"]["ssr"],
                    "theta": r["fit"]["theta"],
                    "status": r["fit"]["status"],
                    "evaluations": r["fit"]["evaluations"],
                }
            )
        profiles[m] = {
            "free": [f for f in sel["main"]["both"] if f != m],
            "bounds": list(bounds[m]),
            "grid": rows_p,
        }
    record["profiles"] = profiles
    memory = [
        r.get("peak_memory_mb")
        for res in (res_a, res_c, res_d)
        for r in res.values()
        if isinstance(r, dict) and r.get("peak_memory_mb")
    ]
    record["peak_memory_mb_per_worker_max"] = max(memory) if memory else None
    record["failures"] = sorted(
        k for res in (res_a, res_c, res_d) for k, r in res.items() if "failure" in r
    )
    record["seconds"] = time.perf_counter() - started
    write_gz(record_path, record)
    write_gz(
        local_path,
        {
            "name": args.name,
            "points": point_dicts,
            "rows": row_labels,
            "measured": measured.tolist(),
            "model_outputs": local_outputs,
        },
    )
    log(f"record {record_path}; local {local_path}; failures {record['failures']}", started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
