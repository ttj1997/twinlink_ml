r"""Site pinch: largest usable waste heat, pinch, source/sink pairing and network temperature level.

Question: how much waste heat can the site use at most (minimum fossil heat against today), where
is the pinch, which sources and sinks belong together, and at which temperature level should the
planned waste-heat network run? Exploration on placeholders, no gate; every number carries its
scenario. Fixed in this docstring before any computation of this script.

Method (``twinlink_ml.site``, reusing ``heatnetwork.pinch`` for the condensing exhaust and
``storage_timeslice`` for the measured days of plant A, plant C, compressor heat, hall heating and
process water): 12 representative days (the measured day shifted to the monthly mean of a
placeholder climate), 96 steps of 15 min each, problem table per step (no storage; heat usable
only when it occurs, ``E_direct`` of Befund 99), annual means weighted by days per month.

* target = minimum hot utility of ALL streams (``dT_min``, sources down to ``T_kalt`` with
  condensation); today = recovery inside today's groups only (compressor -> plant A capped at
  ``compressor_used_today_kw``; plant C own exhaust/compressor -> own air plus at most
  ``c_low_temperature_today_kw`` to hall heating/process water; generic lines nothing);
  saving = fossil today - target.
* network level (the study's network model): zonal LP -- direct recovery inside each zone
  (compressor station | plant A | plant C with hall and process water | generic lines | hall and
  water without C | boiler flue gas), between zones only through one loop at (supply, return)
  with ``dT_min/2`` per exchanger, parallel branches, return cooler allowed; levels
  ``DEFAULT_ZONAL_LEVELS``. Diagnostic: the composite loop (closed form, source/sink side).
* heat pump hints (handover to study 2): evaporator/condenser limits from the grand composite
  curve of each step, COP = ``heat_pump_carnot_efficiency`` x Carnot, best pair per electricity
  CO2 factor (0.366 and the "2030" variant 0.20 kg/kWh).

Grid (all placeholders else at their main value, ``conf/site/site_placeholder.yaml``):
day of plant A {product_1, product_2} x plant C reading {volume, annual, none} x compressor heat
{400, 800, 1200} kW x generic lines {0; n = 1..5 x flow {10 000, 20 000, 30 000} Nm3/h x day
{product_1 (exhaust about 61 degC), product_2 (about 34 degC)}} x ``dT_min`` {10, 20} K x hall
heating / process water {0/0, 200/200} kW = 2 x 3 x 3 x 31 x 2 x 2 = 2232 scenarios.

Reference scenarios (detail: time-averaged composite and grand composite curves, per-stream heat
above/below the pinch, single source x sink pairs, network potential per stream and level, two
loops at once): R1 = product_1, C volume, 800 kW, no generic lines, hall/water 200/200,
``dT_min`` 20; R2 = R1 with product_2; R3 = R1 with C annual; R4 = R1 without C; R5 = R1 + 2
generic lines of 20 000 Nm3/h on product_1; R6 = the same on product_2.

Sensitivity (one at a time around R1 and R5, each placeholder to both band ends, plus the
structural switches; the plant-C operating fraction 0.5 is combined exactly from the runs with and
without C): ranked by the change of the saving (target vs today) in MWh/a, also reported for the
best network level. The ranking is the list of questions for the partner.

Checks (recorded): energy balance per step; fossil today >= target <= every network level;
Befund 99 reproduced (``direct_kwh_per_day`` of every day-horizon scenario of its artifact
recomputed with this package's problem table).

Variant "Befund 99 assumption" (revision of 2026-10-16, before the first complete run; a first
start with ``solid_fraction`` 0.3 was stopped after 100 of 2340 runs, no artifact, no result
used): the main value of ``solid_fraction`` is 0.634 (band 0.60-0.70; the energy balance of the
measured days of plant A requires at least 0.575 / 0.625, parallel study 4, ``docs/23`` section 5).
The 0.3 of Befund 99 is run as a PAIRED variant on the sub-grid ``dT_min`` 20 K, hall/water
200/200, generic flow 20 000 Nm3/h (2 x 3 x 3 x 11 = 198 scenarios) and on R1-R6; the artifact
reports the difference per scenario (``variant_befund_99``). In the sensitivity the 0.3 is an extra
entry ``solid_fraction_befund_99`` next to the band ends 0.60 / 0.70.

Call::

    TWINLINK_PLANT_DATA_P1=... TWINLINK_PLANT_DATA_P2=... \
        tools/run_here.sh scripts/run_site_pinch.py --name run_site_pinch

Paired variant alone against an existing main artifact (added 2026-10-16 after the main run, whose
field ``variant_befund_99`` was overwritten by a shadowed loop variable; nothing else changed)::

    tools/run_here.sh scripts/run_site_pinch.py --name <new> --variant-of run_site_pinch_sf0634
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (
    NO_HEAT_NETWORK,
    REPO_ROOT,
    _json_default,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.site import (
    DEFAULT_TWO_LEVEL_PAIRS,
    Ramps,
    SiteConfig,
    SiteScenario,
    SiteStreamBuilder,
    annual_summary,
    evaluate_year,
    load_site_config,
    problem_table,
)
from twinlink_ml.storage_timeslice import (
    MeasuredDay,
    Placeholders,
    Scenario,
    StreamFactory,
    load_measured_day,
    site_streams,
)

PRODUCT_ENV = {"product_1": "TWINLINK_PLANT_DATA_P1", "product_2": "TWINLINK_PLANT_DATA_P2"}
ROLE_FILE = REPO_ROOT / "conf" / "plant" / "plant_local.yaml"
DAYS = ("product_1", "product_2")
C_BASES = ("volume", "annual", "none")
COMPRESSOR_KW = (400.0, 800.0, 1200.0)
GENERIC_COUNTS = (1, 2, 3, 4, 5)
GENERIC_FLOWS = (10000.0, 20000.0, 30000.0)
DT_MIN = (10.0, 20.0)
HALL_WATER = ((0.0, 0.0), (200.0, 200.0))

BEFUND_99_SOLID_FRACTION = 0.3
"""Solid fraction of Befund 99 (inadmissible by the energy balance of plant A; paired variant)."""

_STATE: dict[str, Any] = {}


def variant_subgrid(scenarios: list[SiteScenario]) -> list[tuple[int, SiteScenario]]:
    """Indices and scenarios of the sub-grid of the paired variant "Befund 99 assumption"."""
    out = []
    for i, s in enumerate(scenarios):
        ph = s.placeholders
        if ph.minimum_pinch_k != 20.0 or ph.hall_kw != 200.0:
            continue
        if s.n_generic and ph.generic_flow_nm3_per_h != 20000.0:
            continue
        out.append((i, s.with_placeholders(solid_fraction=BEFUND_99_SOLID_FRACTION)))
    return out


def paired_variant(
    main: dict[str, dict[str, Any]], variant: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Differences variant (0.3) minus main (0.634) per scenario, same label."""
    fields = ("saving_kw", "fossil_today_kw", "fossil_target_kw", "rec_target_kw")
    rows = []
    for key, v in variant.items():
        m = main[key]
        best = m["network_best_level"]
        rows.append(
            {
                "label": m["label"],
                **{f"main_{f}": m[f] for f in fields},
                **{f"delta_{f}": v[f] - m[f] for f in fields},
                "main_network_best_level": best,
                "variant_network_best_level": v["network_best_level"],
                "delta_network_fossil_at_main_best_kw": v["network"][best]["fossil_kw"]
                - m["network"][best]["fossil_kw"],
                "main_pinch_dominant": max(
                    m["pinch_labels_weighted"], key=m["pinch_labels_weighted"].get
                ),
                "variant_pinch_dominant": max(
                    v["pinch_labels_weighted"], key=v["pinch_labels_weighted"].get
                ),
                "delta_supply_kw": v["mean_kw"]["supply_kw"] - m["mean_kw"]["supply_kw"],
            }
        )

    def spread(name: str) -> dict[str, float]:
        a = np.asarray([row[name] for row in rows], dtype=float)
        if a.size == 0:
            return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
        return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}

    return {
        "solid_fraction_main": None,
        "solid_fraction_variant": BEFUND_99_SOLID_FRACTION,
        "n": len(rows),
        "rows": rows,
        "delta_saving_kw": spread("delta_saving_kw"),
        "delta_fossil_target_kw": spread("delta_fossil_target_kw"),
        "delta_supply_kw": spread("delta_supply_kw"),
        "delta_network_fossil_at_main_best_kw": spread("delta_network_fossil_at_main_best_kw"),
        "best_level_changed": sum(
            row["main_network_best_level"] != row["variant_network_best_level"] for row in rows
        ),
        "pinch_dominant_changed": sum(
            row["main_pinch_dominant"] != row["variant_pinch_dominant"] for row in rows
        ),
    }


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_days() -> tuple[dict[str, MeasuredDay], dict[str, str]]:
    missing = [env for env in PRODUCT_ENV.values() if not os.environ.get(env)]
    if missing:
        raise SystemExit(f"set the environment variables {', '.join(missing)}")
    days, hashes = {}, {}
    for label, env in PRODUCT_ENV.items():
        hashes[label] = sha256_of(os.environ[env])
        days[label] = load_measured_day(os.environ[env], ROLE_FILE, label)
    return days, hashes


def _init(days: dict[str, MeasuredDay]) -> None:
    config = load_site_config()
    _STATE["config"] = config
    _STATE["builder"] = SiteStreamBuilder(config, days)


def grid(config: SiteConfig) -> list[SiteScenario]:
    out: list[SiteScenario] = []
    generic: list[tuple[int, float, str]] = [(0, 20000.0, "product_1")]
    generic += [(n, f, d) for n in GENERIC_COUNTS for f in GENERIC_FLOWS for d in DAYS]
    for day in DAYS:
        for c in C_BASES:
            for comp in COMPRESSOR_KW:
                for n, flow, gday in generic:
                    for dt in DT_MIN:
                        for hall, water in HALL_WATER:
                            base = SiteScenario(
                                config.placeholders,
                                day_a=day,
                                c_basis=c,  # type: ignore[arg-type]
                                n_generic=n,
                                generic_day=gday,
                            )
                            out.append(
                                base.with_placeholders(
                                    compressor_station_kw=comp,
                                    generic_flow_nm3_per_h=flow,
                                    minimum_pinch_k=dt,
                                    hall_kw=hall,
                                    water_kw=water,
                                )
                            )
    return out


def references(config: SiteConfig) -> dict[str, SiteScenario]:
    r1 = SiteScenario(config.placeholders)
    return {
        "R1": r1,
        "R2": replace(r1, day_a="product_2"),
        "R3": replace(r1, c_basis="annual"),
        "R4": replace(r1, c_basis="none"),
        "R5": replace(r1, n_generic=2, generic_day="product_1"),
        "R6": replace(r1, n_generic=2, generic_day="product_2"),
    }


SENSITIVITY_PLACEHOLDERS = (
    "minimum_pinch_k",
    "cold_temperature_celsius",
    "compressor_station_kw",
    "compressor_hot_in_celsius",
    "compressor_hot_out_celsius",
    "compressor_used_today_kw",
    "compressor_c_gwh_per_a",
    "c_line_nm3_per_h",
    "c_low_temperature_today_kw",
    "generic_flow_nm3_per_h",
    "hall_kw",
    "hall_supply_celsius",
    "hall_return_celsius",
    "hall_limit_celsius",
    "hall_design_celsius",
    "hall_day_factor",
    "water_kw",
    "water_supply_celsius",
    "water_return_celsius",
    "solid_fraction",
    "relative_humidity",
    "climate_shift_k",
    "gas_efficiency",
)


def sensitivity_variants(
    config: SiteConfig, base: SiteScenario
) -> list[tuple[str, str, SiteScenario]]:
    out: list[tuple[str, str, SiteScenario]] = []
    for name in SENSITIVITY_PLACEHOLDERS:
        low, high = config.band(name)
        main = getattr(base.placeholders, name)
        for side, value in (("low", low), ("high", high)):
            if value != main:
                out.append((name, f"{side}={value:g}", base.with_placeholders(**{name: value})))
    other_day = "product_2" if base.day_a == "product_1" else "product_1"
    out.append(("day_a", other_day, replace(base, day_a=other_day)))
    out.append(("c_basis", "annual", replace(base, c_basis="annual")))
    out.append(("c_basis", "none", replace(base, c_basis="none")))
    if base.n_generic == 0:
        out.append(
            ("n_generic", "2@product_1", replace(base, n_generic=2, generic_day="product_1"))
        )
        out.append(
            ("n_generic", "2@product_2", replace(base, n_generic=2, generic_day="product_2"))
        )
    else:
        out.append(("n_generic", "0", replace(base, n_generic=0)))
        out.append(("n_generic", "5", replace(base, n_generic=5)))
        other = "product_2" if base.generic_day == "product_1" else "product_1"
        out.append(("generic_day", other, replace(base, generic_day=other)))
        out.append(("generic_today", "own", replace(base, generic_today="own")))
    out.append(("boiler", "on", replace(base, boiler=True)))
    out.append(
        (
            "solid_fraction_befund_99",
            f"{BEFUND_99_SOLID_FRACTION:g}",
            base.with_placeholders(solid_fraction=BEFUND_99_SOLID_FRACTION),
        )
    )
    return out


def compact(summary: dict[str, Any], scenario: SiteScenario) -> dict[str, Any]:
    ph = scenario.placeholders
    return {
        "axes": {
            "day_a": scenario.day_a,
            "c_basis": scenario.c_basis,
            "compressor_station_kw": ph.compressor_station_kw,
            "n_generic": scenario.n_generic,
            "generic_flow_nm3_per_h": ph.generic_flow_nm3_per_h if scenario.n_generic else 0.0,
            "generic_day": scenario.generic_day if scenario.n_generic else "",
            "minimum_pinch_k": ph.minimum_pinch_k,
            "hall_kw": ph.hall_kw,
            "water_kw": ph.water_kw,
            "generic_today": scenario.generic_today,
            "boiler": scenario.boiler,
        },
        **summary,
    }


def run_one(task: tuple[str, SiteScenario, bool]) -> tuple[str, dict[str, Any]]:
    key, scenario, detail = task
    config: SiteConfig = _STATE["config"]
    builder: SiteStreamBuilder = _STATE["builder"]
    year = evaluate_year(
        builder,
        config,
        scenario,
        detail=detail,
        two_level_pairs=DEFAULT_TWO_LEVEL_PAIRS if detail else (),
    )
    out = compact(annual_summary(year, config), scenario)
    out["checks"] = checks_of(out)
    if not detail:  # keep grid entries small: loop diagnostic and heat-pump limits only for refs
        out["loop_composite"] = {k: v["q_kw"] for k, v in out["loop_composite"].items()}
        out["heat_pump"] = {k: v for k, v in out["heat_pump"].items() if k.startswith("el_")}
    if detail:
        out["detail"] = year.detail
        out["scenario"] = asdict(scenario)
    return key, out


def checks_of(summary: dict[str, Any]) -> dict[str, Any]:
    target = summary["fossil_target_kw"]
    return {
        "balance_residual_max_kw": summary["balance_residual_max_kw"],
        "today_not_below_target": summary["fossil_today_kw"] >= target - 1e-6,
        "network_not_below_target": all(
            v["fossil_kw"] >= target - 1e-6 for v in summary["network"].values()
        ),
    }


def reproduce_befund_99(days: dict[str, MeasuredDay]) -> dict[str, Any]:
    """``direct_kwh_per_day`` of every day-horizon scenario of the Befund 99 artifact."""
    path = REPO_ROOT / "artifacts" / "runs" / "gate_storage_timeslice.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    factory = StreamFactory(days)
    ph = Placeholders()
    deviations = []
    for entry in data["grid"]:
        if entry["horizon"] != "day":
            continue
        scenario = Scenario(
            site=entry["site"],
            day=entry["day"],
            horizon="day",
            mode=entry["mode"],
            ambient_shift_k=entry["ambient_shift_k"],
            compressor_a_kw=entry["compressor_a_kw"],
            hall_kw=entry["hall_kw"],
            water_kw=entry["water_kw"],
            loss_per_h=entry["loss_per_h"],
            placeholders=ph,
        )
        bank = factory.build(scenario)
        hot, cold = bank.select(*site_streams(entry["site"]))
        total = 0.0
        for i in range(bank.n_steps):
            total += problem_table(
                Ramps.of_segments(hot[i]), Ramps.of_segments(cold[i]), ph.minimum_pinch_k
            ).hot_utility_kw
        ours = total * bank.dt_h
        deviations.append(abs(ours - entry["direct_kwh_per_day"]) / entry["direct_kwh_per_day"])
    return {
        "scenarios": len(deviations),
        "max_relative_deviation": float(max(deviations)),
        "reference_artifact": "gate_storage_timeslice",
    }


def combine_c_fraction(
    with_c: dict[str, Any], without_c: dict[str, Any], f: float
) -> dict[str, float]:
    """Exact time combination of the runs with and without plant C (per-step results additive)."""
    keys = ("fossil_today_kw", "fossil_target_kw", "saving_kw", "rec_target_kw")
    out = {k: f * with_c[k] + (1 - f) * without_c[k] for k in keys}
    best = with_c["network_best_level"]
    out["network_best_fossil_kw"] = (
        f * with_c["network"][best]["fossil_kw"] + (1 - f) * without_c["network"][best]["fossil_kw"]
    )
    return out


def summarise(results: dict[str, dict[str, Any]], config: SiteConfig) -> dict[str, Any]:
    """Tables of the document (every number traceable to a grid entry)."""
    rows = list(results.values())
    to_mwh = 8.76

    def stats(values: list[float]) -> dict[str, float]:
        a = np.asarray(values, dtype=float)
        return {
            "min": float(a.min()),
            "median": float(np.median(a)),
            "max": float(a.max()),
            "n": int(a.size),
        }

    by_c: dict[str, Any] = {}
    for c in C_BASES:
        sub = [r for r in rows if r["axes"]["c_basis"] == c]
        by_c[c] = {
            "saving_kw": stats([r["saving_kw"] for r in sub]),
            "rec_target_kw": stats([r["rec_target_kw"] for r in sub]),
            "fossil_today_kw": stats([r["fossil_today_kw"] for r in sub]),
        }
    # main table: hall/water on, dT 20, generic flow 20 000, per (day, C, compressor) x generic
    main_table = []
    for r in rows:
        ax = r["axes"]
        if ax["hall_kw"] != 200.0 or ax["minimum_pinch_k"] != 20.0:
            continue
        if ax["n_generic"] and ax["generic_flow_nm3_per_h"] != 20000.0:
            continue
        main_table.append(
            {
                "day_a": ax["day_a"],
                "c_basis": ax["c_basis"],
                "compressor_kw": ax["compressor_station_kw"],
                "n_generic": ax["n_generic"],
                "generic_day": ax["generic_day"],
                "fossil_today_kw": r["fossil_today_kw"],
                "fossil_target_kw": r["fossil_target_kw"],
                "saving_kw": r["saving_kw"],
                "saving_mwh_per_a": r["saving_mwh_per_a"],
                "saving_co2_t_per_a": r["saving_co2_t_per_a"],
                "rec_today_kw": r["rec_today_kw"],
                "rec_target_kw": r["rec_target_kw"],
                "pinch_hot_celsius_median": r["pinch_hot_celsius_median"],
                "pinch_dominant": max(
                    r["pinch_labels_weighted"], key=r["pinch_labels_weighted"].get
                ),
                "network_best_level": r["network_best_level"],
                "network_best_fraction": r["network_best_fraction"],
                "network_fraction_lt_best": max(
                    (v["fraction_of_target_saving"], k)
                    for k, v in r["network"].items()
                    if float(k.split("/")[0]) <= 60.0
                ),
                "network_fraction_ht_best": max(
                    (v["fraction_of_target_saving"], k)
                    for k, v in r["network"].items()
                    if float(k.split("/")[0]) >= 70.0
                ),
            }
        )
    # network level choice over the whole grid
    best_counter = Counter(r["network_best_level"] for r in rows)
    level_fraction: dict[str, Any] = {}
    for level in rows[0]["network"]:
        level_fraction[level] = stats(
            [r["network"][level]["fraction_of_target_saving"] for r in rows if r["saving_kw"] > 1.0]
        )
    lt = [
        max(
            v["fraction_of_target_saving"]
            for k, v in r["network"].items()
            if float(k.split("/")[0]) <= 60.0
        )
        for r in rows
        if r["saving_kw"] > 1.0
    ]
    ht = [
        max(
            v["fraction_of_target_saving"]
            for k, v in r["network"].items()
            if float(k.split("/")[0]) >= 70.0
        )
        for r in rows
        if r["saving_kw"] > 1.0
    ]
    pinch_counter = Counter(
        max(r["pinch_labels_weighted"], key=r["pinch_labels_weighted"].get).split(":")[0]
        for r in rows
    )
    gas_ref = float(config.site_reference["gas_gwh_per_a"])
    existing = []
    for r in rows:
        if r["axes"]["c_basis"] != "none":
            continue
        m = r["mean_kw"]
        fossil_existing = r["fossil_today_kw"]  # without C: A, generic lines, hall, water
        gas_gwh = fossil_existing / r_eta(config) * to_mwh / 1000.0
        existing.append(
            {
                "day_a": r["axes"]["day_a"],
                "compressor_kw": r["axes"]["compressor_station_kw"],
                "n_generic": r["axes"]["n_generic"],
                "generic_flow_nm3_per_h": r["axes"]["generic_flow_nm3_per_h"],
                "hall_kw": r["axes"]["hall_kw"],
                "dT_min": r["axes"]["minimum_pinch_k"],
                "implied_gas_gwh_per_a": gas_gwh,
                "share_of_site_gas": gas_gwh / gas_ref,
                "demand_air_a_kw": m["demand_air_a_kw"],
            }
        )
    consistent = [e for e in existing if e["share_of_site_gas"] <= 1.0]
    return {
        "by_c_basis": by_c,
        "main_table": main_table,
        "network_best_level_counts": dict(best_counter.most_common()),
        "network_level_fraction_stats": level_fraction,
        "network_low_temperature_best_fraction": stats(lt),
        "network_high_temperature_best_fraction": stats(ht),
        "pinch_dominant_stream_counts": dict(pinch_counter.most_common()),
        "existing_site_gas_check": {
            "rows": existing,
            "n_consistent_with_site_gas": len(consistent),
            "max_generic_lines_consistent_at_main_values": {
                f"{day}|{flow:g}": max(
                    (
                        e["n_generic"]
                        for e in consistent
                        if e["generic_flow_nm3_per_h"] in (flow, 0.0)
                        and e["day_a"] == day
                        and e["compressor_kw"] == 800.0
                        and e["hall_kw"] == 200.0
                        and e["dT_min"] == 20.0
                    ),
                    default=-1,
                )
                for day in DAYS
                for flow in GENERIC_FLOWS
            },
        },
        "saving_kw_all": stats([r["saving_kw"] for r in rows]),
        "saving_co2_t_per_a_all": stats([r["saving_co2_t_per_a"] for r in rows]),
    }


def variant_only(
    name: str,
    main_name: str,
    workers: int,
    config: SiteConfig,
    days: dict[str, MeasuredDay],
    hashes: dict[str, str],
    t0: float,
) -> int:
    """Paired variant "Befund 99 assumption" against an existing main artifact (matched by label).

    Follow-up of ``run_site_pinch_sf0634``, whose field ``variant_befund_99`` was overwritten by a
    shadowed loop variable (it holds the string "0.3"); the same 198 + 6 runs are repeated here.
    """
    main_path = REPO_ROOT / "artifacts" / "runs" / f"{main_name}.json"
    main = json.loads(main_path.read_text(encoding="utf-8"))
    main_grid = {r["label"]: r for r in main["grid"]}
    if len(main_grid) != len(main["grid"]):
        raise SystemExit("grid labels of the main artifact are not unique")
    tasks = [(f"grid:{s.label}", s, False) for _, s in variant_subgrid(grid(config))]
    tasks += [
        (f"ref:{k}", s.with_placeholders(solid_fraction=BEFUND_99_SOLID_FRACTION), False)
        for k, s in references(config).items()
    ]
    print(f"{len(tasks)} variant runs on {workers} processes", flush=True)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(days,)) as pool:
        for key, out in pool.map(run_one, tasks, chunksize=1):
            results[key] = out
    var_grid = {k.removeprefix("grid:"): v for k, v in results.items() if k.startswith("grid:")}
    var_refs = {k.removeprefix("ref:"): v for k, v in results.items() if k.startswith("ref:")}
    paired = paired_variant(main_grid, var_grid)
    paired["solid_fraction_main"] = main["placeholders"]["solid_fraction"]["value"]
    paired["references"] = paired_variant(main["references"], var_refs)
    payload: dict[str, Any] = {
        "_provenance": provenance(NO_HEAT_NETWORK),
        "name": name,
        "main_artifact": main_name,
        "main_artifact_git_commit": main["provenance"]["git/commit"],
        "raw_file_sha256": hashes,
        "raw_files_match_main": hashes == main["raw_file_sha256"],
        "variant_befund_99": paired,
        "checks": {
            "balance_residual_max_kw": max(
                r["checks"]["balance_residual_max_kw"] for r in results.values()
            ),
            "today_not_below_target_all": all(
                r["checks"]["today_not_below_target"] for r in results.values()
            ),
            "network_not_below_target_all": all(
                r["checks"]["network_not_below_target"] for r in results.values()
            ),
        },
        "runtime_seconds": time.time() - t0,
    }
    print(json.dumps(payload["checks"], indent=1))
    write_artifact(name, payload)
    return 0


def r_eta(config: SiteConfig) -> float:
    return config.placeholders.gas_efficiency


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="artifact name (never overwritten)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="few scenarios, no artifact")
    parser.add_argument(
        "--variant-of",
        default="",
        help="only the paired variant 'Befund 99 assumption', paired with this main artifact",
    )
    args = parser.parse_args()
    if not args.smoke:
        require_clean_tree()
        if (REPO_ROOT / "artifacts" / "runs" / f"{args.name}.json").exists():
            raise SystemExit("artifact exists -- use a new name")
    t0 = time.time()
    config = load_site_config()
    days, hashes = load_days()
    if args.variant_of:
        return variant_only(args.name, args.variant_of, args.workers, config, days, hashes, t0)
    scenarios = grid(config)
    refs = references(config)
    tasks: list[tuple[str, SiteScenario, bool]] = []
    if args.smoke:
        scenarios = scenarios[:: max(1, len(scenarios) // 6)]
        refs = {"R1": refs["R1"]}
    tasks += [(f"grid:{i}", s, False) for i, s in enumerate(scenarios)]
    tasks += [(f"ref:{k}", s, True) for k, s in refs.items()]
    sub = variant_subgrid(scenarios)
    tasks += [(f"var99:grid:{i}", s, False) for i, s in sub]
    tasks += [
        (f"var99:ref:{k}", s.with_placeholders(solid_fraction=BEFUND_99_SOLID_FRACTION), False)
        for k, s in refs.items()
    ]
    sens_bases = {"R1": references(config)["R1"], "R5": references(config)["R5"]}
    sens_meta: dict[str, tuple[str, str, str]] = {}
    if not args.smoke:
        for base_key, base in sens_bases.items():
            for j, (name, label, scenario) in enumerate(sensitivity_variants(config, base)):
                key = f"sens:{base_key}:{j}"
                sens_meta[key] = (base_key, name, label)
                tasks.append((key, scenario, False))
    print(f"{len(tasks)} runs on {args.workers} processes", flush=True)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")  # one thread per worker (spawned workers inherit this)
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(days,)) as pool:
        for n, (key, out) in enumerate(pool.map(run_one, tasks, chunksize=1), 1):
            results[key] = out
            if n % 100 == 0:
                print(f"  {n}/{len(tasks)} after {time.time() - t0:.0f} s", flush=True)

    grid_results = {k: v for k, v in results.items() if k.startswith("grid:")}
    ref_results = {k.split(":")[1]: v for k, v in results.items() if k.startswith("ref:")}
    var_grid = {
        k.removeprefix("var99:"): v for k, v in results.items() if k.startswith("var99:grid:")
    }
    var_refs = {k.split(":")[2]: v for k, v in results.items() if k.startswith("var99:ref:")}
    paired = paired_variant(grid_results, var_grid)
    paired["solid_fraction_main"] = config.placeholders.solid_fraction
    paired["references"] = paired_variant(ref_results, var_refs)
    sensitivity: dict[str, Any] = {}
    for base_key in sens_bases if not args.smoke else ():
        base = ref_results[base_key]
        entries = []
        for key, (bk, name, label) in sens_meta.items():
            if bk != base_key:
                continue
            r = results[key]
            best = base["network_best_level"]
            entries.append(
                {
                    "placeholder": name,
                    "variant": label,
                    "saving_kw": r["saving_kw"],
                    "delta_saving_mwh_per_a": r["saving_mwh_per_a"] - base["saving_mwh_per_a"],
                    "delta_saving_co2_t_per_a": r["saving_co2_t_per_a"]
                    - base["saving_co2_t_per_a"],
                    "delta_fossil_target_kw": r["fossil_target_kw"] - base["fossil_target_kw"],
                    "network_best_level": r["network_best_level"],
                    "network_best_fraction": r["network_best_fraction"],
                    "delta_network_fossil_at_base_level_kw": r["network"][best]["fossil_kw"]
                    - base["network"][best]["fossil_kw"],
                }
            )
        none_key = next(k for k, m in sens_meta.items() if m == (base_key, "c_basis", "none"))
        for f in (0.5, 0.7):
            combo = combine_c_fraction(base, results[none_key], f)
            entries.append(
                {
                    "placeholder": "c_operating_fraction",
                    "variant": f"{f:g}",
                    "saving_kw": combo["saving_kw"],
                    "delta_saving_mwh_per_a": (combo["saving_kw"] - base["saving_kw"]) * 8.76,
                    "delta_saving_co2_t_per_a": (combo["saving_kw"] - base["saving_kw"])
                    * 8.76
                    * config.placeholders.gas_co2_kg_per_kwh
                    / config.placeholders.gas_efficiency,
                    "delta_fossil_target_kw": combo["fossil_target_kw"] - base["fossil_target_kw"],
                    "network_best_level": base["network_best_level"],
                    "network_best_fraction": float("nan"),
                    "delta_network_fossil_at_base_level_kw": combo["network_best_fossil_kw"]
                    - base["network"][base["network_best_level"]]["fossil_kw"],
                }
            )
        ranking: dict[str, float] = {}
        for e in entries:
            ranking[e["placeholder"]] = max(
                ranking.get(e["placeholder"], 0.0), abs(e["delta_saving_mwh_per_a"])
            )
        sensitivity[base_key] = {
            "entries": entries,
            "ranking_by_max_abs_delta_saving_mwh_per_a": sorted(
                ranking.items(), key=lambda kv: -kv[1]
            ),
        }

    checks = {
        "balance_residual_max_kw": max(
            r["checks"]["balance_residual_max_kw"] for r in results.values()
        ),
        "today_not_below_target_all": all(
            r["checks"]["today_not_below_target"] for r in results.values()
        ),
        "network_not_below_target_all": all(
            r["checks"]["network_not_below_target"] for r in results.values()
        ),
        "befund_99_reproduction": reproduce_befund_99(days),
    }
    payload: dict[str, Any] = {
        "_provenance": provenance(NO_HEAT_NETWORK),
        "name": args.name,
        "raw_file_sha256": hashes,
        "config_file": "conf/site/site_placeholder.yaml",
        "placeholders": {k: asdict(v) for k, v in config.info.items()},
        "climate_monthly_mean_celsius": config.climate_monthly_mean_celsius,
        "site_reference": dict(config.site_reference),
        "references": ref_results,
        "variant_befund_99": paired,
        "sensitivity": sensitivity,
        "summary": summarise(grid_results, config),
        "grid": list(grid_results.values()),
        "checks": checks,
        "runtime_seconds": time.time() - t0,
    }
    print(json.dumps(checks, indent=1, default=str))
    if args.smoke:
        size = len(json.dumps(payload, default=_json_default))
        print(f"payload serialises to {size / 1e6:.1f} MB")
        print(f"smoke run done in {time.time() - t0:.0f} s, no artifact written")
        return 0
    write_artifact(args.name, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
