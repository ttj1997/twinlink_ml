r"""Driver of the plant-near study of plant A (docs/34_plant_scale.md). Reading: analyze_plant_scale.py.

Reads ``conf/plant_scale/plant_scale.yaml`` and the plant-scale process config. Raw exports only
through the environment variables named in the YAML (``TWINLINK_PLANT_DATA_P1`` = day 1 /
product 1, ``TWINLINK_PLANT_DATA_P2`` = day 2 / product 2); channel roles from the git-ignored
``conf/plant/plant_local.yaml`` (key ``plant_scale_roles``; ``--roles`` to override). Model
evaluation only; the grey-box model is parametrised, not changed.

Call (Git Bash, from the worktree)::

    TWINLINK_PLANT_DATA_P1=... TWINLINK_PLANT_DATA_P2=... \
        tools/run_here.sh scripts/run_plant_scale.py --name plant_scale_v1

**What goes where.** ``artifacts/runs/<name>.json`` (committed): segment windows and their
stationarity ratios, DAY-LEVEL aggregates (mean over the day's stationary segments) of the model
inputs and the measured temperatures (the level of the day means of docs/23), relative spreads,
temperature differences, balance numbers, model errors, calibrated parameters and every model
result. ``local/reports/<name>_local.json`` (git-ignored): per segment and role the absolute mean
and standard deviation, the individual bed sensors, the 15-minute block statistics, the
product-specific minimum admissible solid fractions. The export file names never enter any record
(only their SHA-256 and the environment variable).

Steps
-----
1. **Survey.** Per day the stationary segments (criterion in the YAML, fixed before the first run:
   60-minute windows, rolling standard deviations of bed, exhaust and supply temperature <= 0.3 K,
   process air <= 0.5 %, spray rate <= 3 %; 2-4 non-overlapping windows at least 60 minutes apart,
   best score first; fewer admissible windows: the best ones anyway, flagged). One
   :class:`~twinlink_ml.plant_scale.model.PlantPoint` per segment (segment means).
2. **Balances** (:mod:`twinlink_ml.plant_scale.balance`) per segment at the reference solid fraction
   0.65 and the band: the gap bed - exhaust and the number each candidate explanation needs;
   the docs/23 dryer balance with the outlet at the exhaust and at the bed temperature; the solid
   fraction that is consistent for both days with each outlet choice (day aggregates).
3. **Calibration and cross-check.** Candidates C0 (switch off), C3 (offset), C1 (heat loss), C2
   (evaporation above the bed), each with the bed heat-loss UA: fitted on all stationary segments
   of one day, predicted on every segment of the other day, both directions; at the reference
   solid fraction (reading) and at every value of the band (sensitivity), kinetics nominal; and at
   the reference with each of the 20 parameter draws of Befund 111 (kinetics band of Befund 108;
   sensitivity, no reading).
   **Selection of the plant-near model for step 4 (fixed before the run):** among the candidates
   whose largest absolute cross error (both temperatures, every segment, both directions, reference
   solid fraction, nominal kinetics) is <= 2 K, the one with the smallest such error; if none
   passes (reading P2), the candidate with the smallest largest error, used under reservation.
   Ties: the order of ``candidate_order``. Step 4 uses per day the selected candidate calibrated on
   the SAME day (own-day calibration), at the day aggregate.
4. **(a) Observability** with the code of docs/33 (:func:`~twinlink_ml.observability.
   steady_measurements`, ``linear_fit``, ``sensor_sigma``, ``snr``): per day the plant-near model at
   the day aggregate with the nominal kinetics and the 20 draws; SNR = ``|dm/dX| 0.01 / (sigma /
   sqrt(n))``, slope over the draws, sigma at the median over the draws (sensor model of the
   process config = that of the lab line). Measurements ``Y_out``, ``Y_out_balance``, ``T_out``
   (after the switch), ``T_bed``, ``T_bed_x6``. Lab line at P1-P7 of the docs/33 record twice: as
   in docs/33 (ambient 20 degC / 0.50; a reproduction check against the Befund 116 analysis) and
   with the dried supply air of the plant placeholder ("supply air dry").
   **(b) Steady headroom** (:mod:`twinlink_ml.plant_scale.headroom`): R0 fixed command robust over
   the 8 corners of the disturbance box against R1 tracked optimum, savings at the 9 points;
   disturbance amplitudes from the plant data (spray rate 2 x the relative std of the 15-minute
   block means, ambient half the block range, larger day each) and the solid-fraction placeholder
   +-7.69 %. Plant: per day and draw (nominal and 20), limit = the model moisture at today's
   measured point, commands 70-100 degC and 0.8-1.1 x today's gas velocity. Lab: nominal and the 20
   draws, limit 0.15 kg/kg, 60-120 degC, 0.8-2.5 m/s, same relative amplitudes and the same ambient
   amplitude in K. Plus the Befund 108 definition on the lab nominal (margins 0.04355 / 0.00664) as
   a check of the optimiser against the 9.67 %.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import NO_HEAT_NETWORK, REPO_ROOT, RUNS_DIR, provenance  # noqa: E402

from twinlink_ml.headroom.config import load_headroom_config, study_process_config  # noqa: E402
from twinlink_ml.observability import (  # noqa: E402
    linear_fit,
    operating_point_config,
    sensor_sigma,
    snr,
    steady_measurements,
)
from twinlink_ml.plant_a_op.config import DryerPlaceholders  # noqa: E402
from twinlink_ml.plant_a_op.data import DayState  # noqa: E402
from twinlink_ml.plant_a_op.dryer import (  # noqa: E402
    consistent_solid_fraction,
    minimum_admissible_solid_fraction,
)
from twinlink_ml.plant_scale import (  # noqa: E402
    Disturbance,
    Freeboard,
    LabFactory,
    PlantFactory,
    PlantParameters,
    PlantPoint,
    SearchBox,
    SteadySearch,
    block_statistics,
    box_points,
    calibrate,
    gap_balance,
    load_day_series,
    load_plant_scale_config,
    overall_balance,
    predict,
    segment_statistics,
    stationary_segments,
    steady_plant,
)
from twinlink_ml.plant_scale.model import relative_humidity_for  # noqa: E402
from twinlink_ml.sensitivity.config import load_sensitivity_config  # noqa: E402
from twinlink_ml.sensor_value import load_sensor_value_config, parameter_draws  # noqa: E402
from twinlink_ml.sensor_value.draws import truth_config  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402

LOCAL_REPORTS = REPO_ROOT / "local" / "reports"
DEFAULT_ROLES = REPO_ROOT / "conf" / "plant" / "plant_local.yaml"
MEASUREMENTS = {
    # name: (key in the row, temperature channel, redundant bed sensors)
    "Y_out": ("outlet_humidity_kg_per_kg", False, False),
    "Y_out_balance": ("outlet_humidity_balance_kg_per_kg", False, False),
    "T_out": ("exhaust_temperature_celsius", True, False),
    "T_bed": ("bed_temperature_celsius", True, False),
    "T_bed_x6": ("bed_temperature_celsius", True, True),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(obj: Any) -> Any:
    """JSON-safe copy (NaN/inf -> None, numpy scalars -> Python)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ------------------------------------------------------------------------------------ survey


def point_from_stats(label: str, stats: dict[str, Any]) -> PlantPoint:
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


def aggregate(label: str, points: list[PlantPoint]) -> PlantPoint:
    """Day aggregate: mean of every field over the day's segment points."""
    fields = [f.name for f in dataclasses.fields(PlantPoint) if f.name != "label"]
    return PlantPoint(label, **{f: float(np.mean([getattr(p, f) for p in points])) for f in fields})


def survey(
    config: Any, roles: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[PlantPoint]]]:
    repo: dict[str, Any] = {"days": {}}
    local: dict[str, Any] = {"days": {}}
    points: dict[str, list[PlantPoint]] = {}
    for day, env in config.data.days.items():
        path = os.environ.get(env)
        if not path:
            raise SystemExit(f"environment variable {env} (export of {day}) is not set")
        series = load_day_series(path, roles, day, config.data.role_key)
        segments = stationary_segments(series, config.stationarity)
        stats = [segment_statistics(series, s) for s in segments]
        pts = [point_from_stats(f"{day}_s{i + 1}", st) for i, st in enumerate(stats)]
        points[day] = pts
        blocks = {
            name: block_statistics(series[name], config.data.block_minutes)
            for name in (
                "spray_rate",
                "ambient_temperature",
                "process_air_flow",
                "supply_air_temperature",
                "bed_temperature",
                "exhaust_outlet_temperature",
            )
        }
        agg = aggregate(day, pts)
        repo["days"][day] = {
            "environment_variable": env,
            "export_sha256": sha256(Path(path)),
            "minutes": series.n_minutes,
            "segments": [s.as_dict() for s in segments],
            "segments_admissible": sum(s.admissible for s in segments),
            "aggregate": agg.as_dict(),
            "segment_gaps_k": [p.bed_celsius - p.exhaust_celsius for p in pts],
            "segment_bed_group_b_minus_group_a_k": [
                p.bed_group_b_celsius - p.bed_group_a_celsius for p in pts
            ],
            "segment_bed_sensor_spread_k": [
                p.bed_sensor_max_celsius - p.bed_sensor_min_celsius for p in pts
            ],
            "segment_exhaust_minus_downstream_k": [
                p.exhaust_celsius - p.downstream_celsius for p in pts
            ],
            "segment_relative_std": [
                {
                    k: st[k]["std"] / abs(st[k]["mean"])
                    for k in ("process_air_flow", "spray_rate", "atomising_air_flow")
                }
                for st in stats
            ],
            "segment_temperature_std_k": [
                {
                    k: st[k]["std"]
                    for k in (
                        "bed_temperature",
                        "exhaust_outlet_temperature",
                        "supply_air_temperature",
                        "ambient_temperature",
                    )
                }
                for st in stats
            ],
            "block_relative_std": {k: v["relative_std"] for k, v in blocks.items()},
            "block_half_range": {k: v["half_range"] for k, v in blocks.items()},
        }
        local["days"][day] = {
            "segments": [
                {**s.as_dict(), "statistics": st} for s, st in zip(segments, stats, strict=True)
            ],
            "points": [p.as_dict() for p in pts],
            "blocks": blocks,
        }
    return repo, local, points


def balances(
    config: Any, points: dict[str, list[PlantPoint]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    ph = config.placeholders
    rho = config.data.normal_density_dry_air_kg_per_m3
    fractions = [ph.solid_fraction_reference, *ph.solid_fraction_band]
    repo: dict[str, Any] = {"per_segment": {}, "overall": {}, "consistent_solid_fraction": {}}
    local: dict[str, Any] = {"minimum_admissible_solid_fraction": {}}
    for day, pts in points.items():
        repo["per_segment"][day] = [
            {
                f"{x:g}": gap_balance(p, ph, solid_fraction=x, normal_density_kg_per_m3=rho)
                for x in fractions
            }
            for p in pts
        ]
        repo["overall"][day] = [
            {
                f"{x:g}": overall_balance(p, ph, solid_fraction=x, normal_density_kg_per_m3=rho)
                for x in fractions
            }
            for p in pts
        ]
    dryer = DryerPlaceholders(
        solid_fraction=ph.solid_fraction_reference,
        intake_relative_humidity=0.0,
        feed_temperature_celsius=ph.feed_temperature_celsius,
        atomising_air_temperature_celsius=ph.atomising_air_temperature_celsius,
        room_temperature_celsius=ph.room_temperature_celsius,
        normal_density_kg_per_nm3=rho,
    )

    def state(p: PlantPoint, outlet: float) -> DayState:
        return DayState(
            p.label, p.ambient_celsius, p.process_air_nm3h, p.atomising_air_nm3h,
            p.supply_celsius, outlet, p.spray_kg_h, 0.0, 0.0, 0.0, 0.0, 0.0,
        )  # fmt: skip

    aggs = {day: aggregate(day, pts) for day, pts in points.items()}
    d1, d2 = (aggs[d] for d in config.data.days)
    for outlet in ("exhaust", "bed"):
        s1 = state(d1, d1.exhaust_celsius if outlet == "exhaust" else d1.bed_celsius)
        s2 = state(d2, d2.exhaust_celsius if outlet == "exhaust" else d2.bed_celsius)
        x, ua = consistent_solid_fraction(dryer, s1, s2)
        repo["consistent_solid_fraction"][outlet] = {"solid_fraction": x, "ua_kw_per_k": ua}
        local["minimum_admissible_solid_fraction"][outlet] = {
            d1.label: minimum_admissible_solid_fraction(dryer, s1),
            d2.label: minimum_admissible_solid_fraction(dryer, s2),
        }
    return repo, local


# ------------------------------------------------------------------------------- calibration


def steady_kwargs(config: Any) -> dict[str, Any]:
    return {
        "normal_density_kg_per_m3": config.data.normal_density_dry_air_kg_per_m3,
        "inlet_humidity_kg_per_kg": config.placeholders.inlet_absolute_humidity_kg_per_kg,
        "feed_celsius": config.placeholders.feed_temperature_celsius,
        "iterations": config.calibration.velocity_iterations,
        "relative_tolerance": config.calibration.velocity_relative_tolerance,
    }


def calibration_task(args: tuple[Any, ...]) -> dict[str, Any]:
    """One fit on the calibration day and its prediction on both days."""
    config_path, candidate, cal_day, check_day, x_s, draw, points = args
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    config = load_plant_scale_config(config_path)
    base = load_process_config(config.process_config)
    mode = config.calibration.candidates[candidate]
    material = draw["values"] if draw is not None else None
    kw = steady_kwargs(config)
    started = time.perf_counter()
    fit = calibrate(
        base,
        points[cal_day],
        candidate,
        mode,
        solid_fraction=x_s,
        bounds=config.calibration.bounds,
        start=config.calibration.start,
        steady_kwargs=kw,
        material_values=material,
    )
    return {
        "candidate": candidate,
        "calibration_day": cal_day,
        "check_day": check_day,
        "solid_fraction": x_s,
        "draw": None if draw is None else draw["index"],
        "fit": fit.as_dict(),
        "in_sample": predict(
            base, points[cal_day], fit.parameters, steady_kwargs=kw, material_values=material
        ),
        "cross": predict(
            base, points[check_day], fit.parameters, steady_kwargs=kw, material_values=material
        ),
        "runtime_s": time.perf_counter() - started,
    }


def max_abs_error(rows: list[dict[str, Any]]) -> dict[str, float]:
    if any("failure" in r for r in rows):
        return {"bed": math.inf, "exhaust": math.inf, "both": math.inf}
    bed = max(abs(r["error_bed_k"]) for r in rows)
    exh = max(abs(r["error_exhaust_k"]) for r in rows)
    return {"bed": bed, "exhaust": exh, "both": max(bed, exh)}


def select_candidate(config: Any, reference: list[dict[str, Any]]) -> dict[str, Any]:
    """Selection rule of the module docstring on the reference fits (nominal kinetics)."""
    worst = {}
    for c in config.calibration.candidate_order:
        rows = [r for r in reference if r["candidate"] == c]
        worst[c] = max(max_abs_error(r["cross"])["both"] for r in rows)
    tol = config.calibration.tolerance_k
    passing = [c for c in config.calibration.candidate_order if worst[c] <= tol]
    pool = passing or list(config.calibration.candidate_order)
    best = min(pool, key=lambda c: (worst[c], config.calibration.candidate_order.index(c)))
    return {
        "worst_cross_error_k": worst,
        "passing": passing,
        "selected": best,
        "reading": "P1" if passing else "P2",
    }


# ------------------------------------------------------------------------------ observability


def snr_table(
    rows: list[dict[str, Any]], measurement: Any, delta_x: float, redundancy: int
) -> dict[str, Any]:
    """SNR per measurement over the draws (definition of docs/33)."""
    solved = [r for r in rows if "failure" not in r]
    x = np.array([r["moisture_kg_per_kg"] for r in solved])
    out: dict[str, Any] = {
        "n_draws": len(rows),
        "n_failed": len(rows) - len(solved),
        "n_infeasible": sum(not r["feasible"] for r in solved),
        "moisture_range_kg_per_kg": [float(x.min()), float(x.max())] if len(x) else None,
        "variants": {},
    }
    if len(solved) < 2:
        return out
    for name, (key, temperature, redundant) in MEASUREMENTS.items():
        y = np.array([r[key] for r in solved])
        fit = linear_fit(x, y)
        sigma = float(sensor_sigma(float(np.median(y)), measurement, temperature=temperature))
        n = redundancy if redundant else 1
        out["variants"][name] = {
            "fit": fit.as_dict(),
            "median_value": float(np.median(y)),
            "sigma_one_sensor": sigma,
            "redundancy": n,
            "signal_for_delta_x": abs(fit.slope) * delta_x,
            "snr": snr(fit.slope, delta_x, sigma, n),
        }
    return out


def plant_row(
    base: Any, point: PlantPoint, params: PlantParameters, kw: dict[str, Any], values: Any
) -> dict[str, Any]:
    try:
        s = steady_plant(base, point, params, material_values=values, **kw)
    except (RuntimeError, ValueError) as error:
        return {"failure": f"{type(error).__name__}: {error}"}
    return {
        "moisture_kg_per_kg": s.moisture_kg_per_kg,
        "bed_temperature_celsius": s.bed_temperature_celsius,
        "exhaust_temperature_celsius": s.exhaust_temperature_celsius,
        "outlet_humidity_kg_per_kg": s.outlet_humidity_kg_per_kg,
        "outlet_humidity_balance_kg_per_kg": s.outlet_humidity_balance_kg_per_kg,
        "feasible": s.bed.feasible,
        "infeasibility_reasons": list(s.bed.infeasibility_reasons),
        "gas_velocity_m_per_s": s.gas_velocity_m_per_s,
        "solid_to_air_ratio": s.bed.solid_inflow_kg_per_s / s.bed.dry_air_flow_kg_per_s,
    }


def lab_row(config: Any) -> dict[str, Any]:
    try:
        s = steady_measurements(config)
    except (RuntimeError, ValueError) as error:
        return {"failure": f"{type(error).__name__}: {error}"}
    return {
        "moisture_kg_per_kg": s.moisture_kg_per_kg,
        "bed_temperature_celsius": s.bed_temperature_celsius,
        "exhaust_temperature_celsius": s.exhaust_temperature_celsius,
        "outlet_humidity_kg_per_kg": s.outlet_humidity_kg_per_kg,
        "outlet_humidity_balance_kg_per_kg": s.outlet_humidity_balance_kg_per_kg,
        "feasible": s.feasible,
        "infeasibility_reasons": list(s.infeasibility_reasons),
        "solid_to_air_ratio": s.solid_inflow_kg_per_s / s.dry_air_flow_kg_per_s,
    }


def lab_dry(lab_point: Any, humidity: float) -> Any:
    t = lab_point.disturbances.ambient_temperature_celsius.nominal
    rh = relative_humidity_for(lab_point, t, humidity)
    d = lab_point.disturbances
    return dataclasses.replace(
        lab_point,
        disturbances=dataclasses.replace(
            d,
            ambient_relative_humidity=dataclasses.replace(
                d.ambient_relative_humidity,
                nominal=rh,
                minimum=min(d.ambient_relative_humidity.minimum, rh),
            ),
        ),
    )


def observability_task(args: tuple[Any, ...]) -> dict[str, Any]:
    kind, label, payload, draws = args
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    rows: list[dict[str, Any]] = []
    if kind == "plant":
        config_path, point, params = payload
        config = load_plant_scale_config(config_path)
        base = load_process_config(config.process_config)
        kw = steady_kwargs(config)
        nominal = plant_row(base, point, params, kw, None)
        rows = [plant_row(base, point, params, kw, d["values"]) for d in draws]
    else:
        lab_config = payload
        nominal = lab_row(lab_config)
        rows = [lab_row(truth_config(lab_config, d["values"])) for d in draws]
    return {"kind": kind, "label": label, "nominal": nominal, "draws": rows}


# ---------------------------------------------------------------------------------- headroom


def headroom_task(args: tuple[Any, ...]) -> dict[str, Any]:
    kind, label, payload = args
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    started = time.perf_counter()
    if kind == "plant":
        factory, box, nominal, corners, points, today = payload
        search = SteadySearch(factory, box)
        e = search.evaluate(today["inlet_celsius"], today["velocity_m_per_s"], nominal)
        limit = e.moisture_kg_per_kg
        result = search.fixed_against_tracked(nominal, corners, points, limit)
        result["today"] = {**today, "moisture_kg_per_kg": e.moisture_kg_per_kg, "power_kw": e.power_kw,
                           "bed_temperature_celsius": e.bed_temperature_celsius}  # fmt: skip
    elif kind == "lab":
        factory, box, nominal, corners, points, limit = payload
        result = SteadySearch(factory, box).fixed_against_tracked(nominal, corners, points, limit)
    else:  # lab reference of Befund 108
        factory, box, nominal, limit, margins = payload
        result = SteadySearch(factory, box).margin_reference(nominal, limit, margins)
    result["runtime_s"] = time.perf_counter() - started
    return {"kind": kind, "label": label, "result": result}


# --------------------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "conf" / "plant_scale" / "plant_scale.yaml")
    )
    parser.add_argument("--roles", default=str(DEFAULT_ROLES))
    parser.add_argument("--name", default="plant_scale_v1")
    parser.add_argument("--output", default=None, help="smoke runs only: path outside artifacts/")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 draws, reference fits only, the lab-reference headroom task",
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if args.smoke and not args.output:
        parser.error("--smoke needs --output outside artifacts/")
    output = Path(args.output) if args.output else RUNS_DIR / f"{args.name}.json"
    local_output = (output.parent if args.smoke else LOCAL_REPORTS) / f"{args.name}_local.json"
    for path in (output, local_output):
        if path.exists():
            raise SystemExit(f"{path} exists; artifacts are never overwritten")

    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    prov = provenance(NO_HEAT_NETWORK, ignore_paths=[output] if not args.smoke else ())
    started = time.perf_counter()
    config_path = Path(args.config)
    config = load_plant_scale_config(config_path)
    workers = args.workers or config.workers
    base = load_process_config(config.process_config)
    roles = Path(args.roles)

    # 1-2 survey and balances
    survey_repo, survey_local, points = survey(config, roles)
    balance_repo, balance_local = balances(config, points)
    days = list(config.data.days)
    aggs = {day: aggregate(day, pts) for day, pts in points.items()}
    print(f"survey done ({time.perf_counter() - started:.0f} s)", flush=True)

    # draws of Befund 111 (validated on the lab line, as there)
    sv = load_sensor_value_config(config.observability.sensor_value_config)
    study = load_headroom_config(sv.headroom_config)
    lab_nominal = study_process_config(study)
    draws_obj, rejected = parameter_draws(
        sv.draws, load_sensitivity_config(sv.sensitivity_config), lab_nominal
    )
    draws = [{"index": d.index, "values": dict(d.values)} for d in draws_obj]
    if args.smoke:
        draws = draws[:2]

    # 3 calibration
    ph = config.placeholders
    fractions = [ph.solid_fraction_reference, *ph.solid_fraction_band]
    directions = [(days[0], days[1]), (days[1], days[0])]
    tasks = []
    for cand in config.calibration.candidate_order:
        for cal, chk in directions:
            for x in fractions if not args.smoke else fractions[:1]:
                tasks.append((config_path, cand, cal, chk, x, None, points))
            for d in draws if not args.smoke else []:
                tasks.append((config_path, cand, cal, chk, ph.solid_fraction_reference, d, points))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fits = list(pool.map(calibration_task, tasks))
    reference = [
        f for f in fits if f["draw"] is None and f["solid_fraction"] == ph.solid_fraction_reference
    ]
    selection = select_candidate(config, reference)
    print(f"calibration done: {selection} ({time.perf_counter() - started:.0f} s)", flush=True)

    # own-day parameters of the selected candidate (reference, nominal kinetics)
    own: dict[str, PlantParameters] = {}
    for day in days:
        f = next(
            r
            for r in reference
            if r["candidate"] == selection["selected"] and r["calibration_day"] == day
        )
        p = f["fit"]["parameters"]
        freeboard = (
            Freeboard()
            if p["freeboard_mode"] == "none"
            else Freeboard(p["freeboard_mode"], p["freeboard_value"])
        )
        own[day] = PlantParameters(p["heat_loss_ua_w_per_k"], p["solid_fraction"], freeboard)

    # 4a observability
    measurement = base.measurement
    with gzip.open(config.observability.lab_record, "rt", encoding="utf-8") as f:
        lab_record = json.load(f)
    lab_points = lab_record["operating_points"]
    obs_tasks = [("plant", day, (config_path, aggs[day], own[day]), draws) for day in days]
    for p in lab_points:
        cfg = operating_point_config(
            lab_nominal, inlet_temperature_celsius=p["inlet_temperature_celsius"],
            gas_velocity_m_per_s=p["gas_velocity_m_per_s"],
        )  # fmt: skip
        obs_tasks.append(("lab_as_docs33", p["name"], cfg, draws))
        obs_tasks.append(
            ("lab_dry", p["name"], lab_dry(cfg, ph.inlet_absolute_humidity_kg_per_kg), draws)
        )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        obs_raw = list(pool.map(observability_task, obs_tasks))
    delta_x = config.observability.delta_x_kg_per_kg
    n_bed = config.observability.redundant_bed_sensors
    observability = {
        f"{r['kind']}:{r['label']}": {
            "nominal": r["nominal"],
            "draws": r["draws"],
            "snr": snr_table(r["draws"], measurement, delta_x, n_bed),
        }
        for r in obs_raw
    }
    print(f"observability done ({time.perf_counter() - started:.0f} s)", flush=True)

    # 4b headroom
    h = config.headroom
    spray_rel = h.spray_sigma_factor * max(
        survey_repo["days"][d]["block_relative_std"]["spray_rate"] for d in days
    )
    ambient_k = max(survey_repo["days"][d]["block_half_range"]["ambient_temperature"] for d in days)
    solid_rel = h.solid_fraction_relative_half_band
    kw = steady_kwargs(config)
    head_tasks: list[tuple[Any, ...]] = []
    today_velocity: dict[str, float] = {}
    for day in days:
        agg = aggs[day]
        s = steady_plant(base, agg, own[day], **kw)
        u0 = s.gas_velocity_m_per_s
        today_velocity[day] = u0
        nominal = Disturbance(agg.spray_kg_h, ph.solid_fraction_reference, agg.ambient_celsius)
        corners, pts = box_points(nominal, spray_rel, solid_rel, ambient_k)
        box = SearchBox(
            h.inlet_temperature_celsius, (h.air_factor[0] * u0, h.air_factor[1] * u0),
            h.bed_temperature_limit_celsius, h.velocity_grid_points, h.inlet_tolerance_k,
            h.velocity_relative_tolerance,
        )  # fmt: skip
        today = {"inlet_celsius": agg.supply_celsius, "velocity_m_per_s": u0}
        for d in [None, *draws]:
            values = () if d is None else tuple(sorted(d["values"].items()))
            factory = PlantFactory(
                base, agg, own[day], ph.inlet_absolute_humidity_kg_per_kg, values
            )
            label = f"{day}:{'nominal' if d is None else d['index']}"
            head_tasks.append(("plant", label, (factory, box, nominal, corners, pts, today)))
    lab_d0 = Disturbance(
        study.line.nominal_spray_rate_kg_per_h, study.line.nominal_solid_fraction,
        lab_nominal.disturbances.ambient_temperature_celsius.nominal,
    )  # fmt: skip
    lab_corners, lab_pts = box_points(lab_d0, spray_rel, solid_rel, ambient_k)
    lab_box = SearchBox(
        h.lab_inlet_temperature_celsius, h.lab_gas_velocity_m_per_s, h.bed_temperature_limit_celsius,
        h.velocity_grid_points, h.inlet_tolerance_k, h.velocity_relative_tolerance,
    )  # fmt: skip
    for d in [None, *draws]:
        values = () if d is None else tuple(sorted(d["values"].items()))
        label = f"lab:{'nominal' if d is None else d['index']}"
        payload = (
            LabFactory(lab_nominal, values),
            lab_box,
            lab_d0,
            lab_corners,
            lab_pts,
            h.lab_limit_kg_per_kg,
        )
        head_tasks.append(("lab", label, payload))
    head_tasks.append(
        ("lab_reference", "lab:befund108", (LabFactory(lab_nominal), lab_box, lab_d0, h.lab_limit_kg_per_kg,
                                            h.lab_reference_margins_kg_per_kg))
    )  # fmt: skip
    if args.smoke:
        head_tasks = [head_tasks[-1]]  # the lab reference (shortest); plant tasks ran in smoke1
    with ProcessPoolExecutor(max_workers=workers) as pool:
        head_raw = list(pool.map(headroom_task, head_tasks))
    headroom = {r["label"]: {"kind": r["kind"], **r["result"]} for r in head_raw}
    print(f"headroom done ({time.perf_counter() - started:.0f} s)", flush=True)

    record = {
        "provenance": prov,
        "study": "plant_scale",
        "smoke": bool(args.smoke),
        "config": str(config_path.relative_to(REPO_ROOT))
        if config_path.is_relative_to(REPO_ROOT)
        else str(config_path),
        "config_fingerprint": config.fingerprint,
        "process_config": str(config.process_config.relative_to(REPO_ROOT)),
        "process_config_fingerprint": base.fingerprint,
        "role_file_sha256": sha256(roles),
        "sensor_value_fingerprint": sv.fingerprint,
        "lab_record_sha256": sha256(config.observability.lab_record),
        "draws": draws,
        "rejected_sequence_positions": rejected,
        "survey": survey_repo,
        "balances": balance_repo,
        "calibration": {
            "fits": fits,
            "selection": selection,
        },
        "own_day_parameters": {d: p.as_dict() for d, p in own.items()},
        "observability": observability,
        "lab_operating_points": lab_points,
        "headroom": {
            "amplitudes": {
                "spray_relative": spray_rel,
                "ambient_k": ambient_k,
                "solid_fraction_relative": solid_rel,
            },
            "today_velocity_m_per_s": today_velocity,
            "results": headroom,
        },
        "runtime_s": time.perf_counter() - started,
    }
    local = {
        "study": "plant_scale",
        "name": args.name,
        "git_commit": prov["git/commit"],
        "survey": survey_local,
        "balances": balance_local,
        "note": "confidential: absolute plant values per segment; never commit",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    local_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean(record), indent=1), encoding="utf-8")
    local_output.write_text(json.dumps(clean(local), indent=1), encoding="utf-8")
    print(f"wrote {output} and {local_output} ({time.perf_counter() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
