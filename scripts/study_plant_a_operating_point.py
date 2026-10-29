r"""Operating point of plant A: CO2 lever of another air flow and supply-air temperature.

Question: what would another combination of process-air flow and supply-air temperature at plant A
save in CO2 against today's operation, counting the steam AND the electricity of both fans?

THIS DOCSTRING IS THE PRE-REGISTRATION of the computation. It was committed before any
optimisation of this study was run; nothing in it is changed after seeing results (corrections
only as dated addenda in the lab book). Before writing it, only the day means of the two exports,
the closure of the dryer energy balance over the solid fraction and the two-point fan exponents
were looked at (needed to set the grid of the solid fraction, see below); no operating point was
evaluated. Every number that is not a measurement is a PLACEHOLDER
(``twinlink_ml.plant_a_op.config.PLACEHOLDERS``: value, band, source, status). The proxy
constraints do NOT replace a product-moisture measurement: every result is an UPPER BOUND of the
lever, not an operating recommendation.

Model (code: ``twinlink_ml.plant_a_op``)
    Stationary, on the day means of each measured day (within a day the air flow and the supply
    temperature move by less than 0.5 %). Dryer: energy balance of process air, atomising air,
    evaporating water (spray rate x (1 - solid fraction)), solids and a wall loss UA (T_out -
    T_room); linear in T_out, so T_out is explicit. Exhaust humidity from the water balance,
    relative humidity at T_out (not clipped). Steam via the air balance as in Befund 91:
    Q_steam = C_air (T_in - T_amb) - f_fan P_process - Q_waste; the preheater duty Q_waste is the
    measured water-side duty, the preheater conductance fitted per day (effectiveness-NTU).
    Production (spray rate) held at the measured day mean.
    Calibration: UA fitted on one day (own day = in-sample; other day = the transfer check);
    a negative UA makes the case inadmissible (recorded, not clipped). Check on the other day:
    T_out of the other day predicted from its inputs, error of the day mean and block RMSE
    against the within-day spread; the calibration counts as TRANSFERRING at a solid fraction if
    |error of the day mean| <= 2 K in both directions (threshold fixed here, a choice).
    Fans: P = P_meas (V / V_meas)^n per fan; n from the two days for the process fan
    (``data_fit``, applied to both fans because the exhaust fan's two-point exponent is outside
    0-3), ``affinity`` n = 3 as placeholder variant; n = 0 and n = 2 as sensitivity.
    CO2: gas = Q_steam / eta, CO2 = 0.201 gas + 0.366 (P_process + P_exhaust) kg/h; variant 2030:
    electricity 0.20 kg/kWh (placeholder).

Search (per product and weather day)
    air factor V / V_meas in [0.6, 1.1], supply temperature in [70, 100] degC, grid 0.005 x 0.1 K
    plus one 10x refinement; the measured point is always a candidate (saving >= 0).
    Constraints, each variant separately: (a) T_out >= observed minimum of the product's day
    (15-min blocks); (b) relative exhaust humidity <= observed maximum (15-min blocks,
    RECONSTRUCTED); (a and b) both. Minimum air fraction 0.6 / 0.7 / 0.8 / 0.9 (placeholder for
    the fluidisation limit). Thresholds are moved to the model baseline where it lies beyond
    them (another weather, other-day calibration), so today's operation is always admissible.

Main grid (every combination, both products)
    weather shift of the intake temperature -10 / -5 / 0 / +5 K (as Befund 99);
    solid fraction 0.60 / 0.65 / 0.70 / 0.75 (same for both products; lower end from the energy
    balance: UA >= 0 needs >= 0.57 / 0.63 at the reference placeholders);
    calibration own day / other day; waste heat fixed_water_inlet / duty_held;
    fan law data_fit / affinity; boiler efficiency 0.85 / 0.88 / 0.92;
    electricity 0.366 / 0.20; constraint a / b / a_and_b; minimum air 0.6 / 0.7 / 0.8 / 0.9.
    Reference case (for tables and the sensitivity): weather 0, solid fraction 0.65, own day,
    fixed_water_inlet, data_fit, eta 0.88, electricity 0.366, constraint a, minimum air 0.8, all
    other placeholders at their reference values.

Readings (fixed beforehand)
    1. Saving in kg CO2/d and in % of today's CO2 (steam + fans) per product, at the reference
       and as ranges over the grid; never a single number without its labels.
    2. Decomposition of the saving: steam (air heating, lost fan heat, lost waste heat) against
       fan electricity (process / exhaust fan).
    3. Which constraint binds (outlet_min, rh_max, air_factor_min, box edges, curtailed waste heat),
       counted over the grid.
    4. Sensitivity ranking: one placeholder at a time around the reference case, swing of the
       saving (kg CO2/d), ranked by the largest absolute change over the two products. Placeholders
       outside the main grid enter here only: intake humidity 0.5 / 0.9, feed temperature 20 / 50,
       atomising-air temperature 15 / 40, room temperature 10 / 30, cp of the solids 800 / 1600,
       fan heat fraction 0.8 / 1.0, maximum preheater water 70 / 90 (with duty_held), fan law
       constant / quadratic, solid fraction at the value consistent between both days.
    5. Annualised order of magnitude: kg/d x 365 (24/7 operation as stated by the partner;
       product mix unknown: each product all year, and 50/50 as placeholder), at 55 and 300 EUR/t.
       Labelled as an upper bound under placeholders.
    Plant C does not enter (the inconsistency of its flow and exhaust-heat figures, Befund 99, is
    not used here).

Call (Git Bash, from the worktree; raw data never enter the repository)::

    TWINLINK_PLANT_DATA_P1=... TWINLINK_PLANT_DATA_P2=... \
        tools/run_here.sh scripts/study_plant_a_operating_point.py

Artifact: ``artifacts/runs/study_plant_a_operating_point.json`` (never overwritten; ``--name``).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import NO_HEAT_NETWORK, REPO_ROOT, provenance, require_clean_tree, write_artifact

from twinlink_ml.plant_a_op.config import (
    AIR_FRACTION_MIN_GRID,
    AMBIENT_SHIFTS_K,
    BOILER_EFFICIENCY_GRID,
    ELECTRICITY_CO2_GRID,
    PLACEHOLDERS,
    SOLID_FRACTION_GRID,
    ConstraintVariant,
    DryerPlaceholders,
    EnergyPlaceholders,
    SearchSpace,
    fingerprint_of,
)
from twinlink_ml.plant_a_op.data import day_state, load_measured_day
from twinlink_ml.plant_a_op.dryer import (
    consistent_solid_fraction,
    loss_conductance_kw_per_k,
    minimum_admissible_solid_fraction,
    observed_outlet_extremes,
)
from twinlink_ml.plant_a_op.fans import within_day_exponent
from twinlink_ml.plant_a_op.study import (
    Case,
    StudyData,
    build_model,
    calibration_check,
    run_case,
    with_dryer,
    with_energy,
)

PRODUCT_ENV = {"product_1": "TWINLINK_PLANT_DATA_P1", "product_2": "TWINLINK_PLANT_DATA_P2"}
ROLE_FILE = REPO_ROOT / "conf" / "plant" / "plant_local.yaml"
PRODUCTS = ("product_1", "product_2")
CONSTRAINTS: tuple[ConstraintVariant, ...] = ("a", "b", "a_and_b")
VARIANTS: list[tuple[ConstraintVariant, float]] = [
    (c, a) for c in CONSTRAINTS for a in AIR_FRACTION_MIN_GRID
]
REFERENCE_CONSTRAINT: ConstraintVariant = "a"
REFERENCE_AIR_MIN = 0.8
TRANSFER_THRESHOLD_K = 2.0
CO2_PRICES_EUR_PER_T = (55.0, 300.0)
SITE_CO2_T_PER_A = 11500.0  # partner statement, order of magnitude
SPACE = SearchSpace()


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_data() -> tuple[StudyData, dict[str, str]]:
    missing = [env for env in PRODUCT_ENV.values() if not os.environ.get(env)]
    if missing:
        raise SystemExit(f"set the environment variables {', '.join(missing)}")
    days, hashes = {}, {}
    for label, env in PRODUCT_ENV.items():
        hashes[label] = sha256_of(os.environ[env])
        days[label] = load_measured_day(os.environ[env], ROLE_FILE, label)
    return StudyData(days=days, other={"product_1": "product_2", "product_2": "product_1"}), hashes


def reference_case(product_label: str) -> Case:
    return Case(product=product_label)


# ----------------------------------------------------------------------------- data diagnostics


def data_section(data: StudyData) -> dict[str, Any]:
    ph = DryerPlaceholders()
    out: dict[str, Any] = {"day_states": {}, "fans_within_day": {}, "befund_91_consistency": {}}
    for label in PRODUCTS:
        day = data.days[label]
        s = day_state(day)
        out["day_states"][label] = s.to_dict()
        out["fans_within_day"][label] = {
            "process_fan": within_day_exponent(day["process_fan_power"], day["process_air_flow"]),
            "exhaust_fan": within_day_exponent(day["exhaust_fan_power"], day["process_air_flow"]),
            "supply_temperature_std_k": float(np.std(day["supply_air_temperature"])),
            "outlet_temperature_std_k": float(np.std(day["exhaust_outlet_temperature"])),
        }
        model, _ = build_model(data, reference_case(label))
        assert model is not None
        b = model.baseline().at(0)
        demand = b["air_heating_kw"] - b["fan_heat_kw"]
        out["befund_91_consistency"][label] = {
            "air_heating_without_fan_credit_kw": b["air_heating_kw"],
            "heating_demand_with_fan_credit_kw": demand,
            "waste_heat_kw": b["waste_heat_kw"],
            "waste_heat_share": b["waste_heat_kw"] / demand,
            "note": "Befund 91 (fan heat fraction 0.9): demand 852 / 697 kW, waste-heat share "
            "32.5 % / 55.6 %; Befund 99: 957 / 798 kW without fan credit",
        }
    out["fan_two_point_exponents"] = {
        "process_fan": data.process_fan_exponent,
        "exhaust_fan": data.exhaust_fan_exponent,
        "exhaust_fan_used": False,
    }
    out["solid_fraction_admissibility"] = {
        label: minimum_admissible_solid_fraction(ph, day_state(data.days[label]))
        for label in PRODUCTS
    }
    x, ua = consistent_solid_fraction(
        ph, day_state(data.days["product_1"]), day_state(data.days["product_2"])
    )
    out["solid_fraction_consistent_between_days"] = {"solid_fraction": x, "ua_kw_per_k": ua}
    out["loss_conductance_grid"] = {
        label: {
            str(xs): loss_conductance_kw_per_k(
                replace(ph, solid_fraction=xs), day_state(data.days[label])
            )
            for xs in SOLID_FRACTION_GRID
        }
        for label in PRODUCTS
    }
    out["observed_extremes"] = {
        label: {
            str(xs): observed_outlet_extremes(data.days[label], replace(ph, solid_fraction=xs))
            for xs in (*SOLID_FRACTION_GRID, x)
        }
        for label in PRODUCTS
    }
    return out


def calibration_section(data: StudyData) -> dict[str, Any]:
    ph = DryerPlaceholders()
    x_cons, _ = consistent_solid_fraction(
        ph, day_state(data.days["product_1"]), day_state(data.days["product_2"])
    )
    rows = []
    for xs in (*SOLID_FRACTION_GRID, x_cons):
        p = replace(ph, solid_fraction=xs)
        for cal, chk in (("product_1", "product_2"), ("product_2", "product_1")):
            rows.append(calibration_check(data, p, cal, chk))
    transferring = sorted(
        {
            r["solid_fraction"]
            for r in rows
            if all(
                abs(q["outlet_mean_error_k"]) <= TRANSFER_THRESHOLD_K
                and q["ua_calibration_kw_per_k"] >= 0.0
                for q in rows
                if q["solid_fraction"] == r["solid_fraction"]
            )
        }
    )
    return {
        "threshold_k": TRANSFER_THRESHOLD_K,
        "rows": rows,
        "solid_fractions_transferring": transferring,
    }


def baseline_section(data: StudyData) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in PRODUCTS:
        rows = []
        for eta, f_el in product(BOILER_EFFICIENCY_GRID, ELECTRICITY_CO2_GRID):
            case = with_energy(
                reference_case(label), boiler_efficiency=eta, electricity_co2_kg_per_kwh=f_el
            )
            model, _ = build_model(data, case)
            assert model is not None
            b = model.baseline().at(0)
            rows.append(
                {
                    "boiler_efficiency": eta,
                    "electricity_co2_kg_per_kwh": f_el,
                    "co2_kg_per_day": 24.0 * b["co2_kg_h"],
                    "co2_steam_kg_per_day": 24.0 * b["co2_steam_kg_h"],
                    "co2_fans_kg_per_day": 24.0 * b["co2_fans_kg_h"],
                    "fan_share": b["co2_fans_kg_h"] / b["co2_kg_h"],
                    "point": b,
                }
            )
        out[label] = rows
    return out


# ------------------------------------------------------------------------------- main grid

GRID_FIELDS = (
    "product",
    "ambient_shift_k",
    "solid_fraction",
    "calibration",
    "waste_heat",
    "fan_law",
    "boiler_efficiency",
    "electricity_co2",
    "constraint",
    "air_factor_min",
    "admissible",
    "saving_kg_per_day",
    "saving_fraction",
    "baseline_co2_kg_per_day",
    "best_air_factor",
    "best_supply_celsius",
    "best_outlet_celsius",
    "best_outlet_rh",
    "best_waste_heat_kw",
    "baseline_waste_heat_kw",
    "saving_steam_kg_per_day",
    "saving_fans_kg_per_day",
    "binding",
)


def grid_cases() -> list[Case]:
    cases = []
    for label, shift, xs, cal, waste, fan, eta, f_el in product(
        PRODUCTS,
        AMBIENT_SHIFTS_K,
        SOLID_FRACTION_GRID,
        ("own_day", "other_day"),
        ("fixed_water_inlet", "duty_held"),
        ("data_fit", "affinity"),
        BOILER_EFFICIENCY_GRID,
        ELECTRICITY_CO2_GRID,
    ):
        cases.append(
            Case(
                product=label,
                ambient_shift_k=shift,
                dryer=DryerPlaceholders(solid_fraction=xs),
                energy=EnergyPlaceholders(boiler_efficiency=eta, electricity_co2_kg_per_kwh=f_el),
                calibration=cal,  # type: ignore[arg-type]
                waste_heat=waste,  # type: ignore[arg-type]
                fan_law=fan,  # type: ignore[arg-type]
            )
        )
    return cases


def compact(row: dict[str, Any]) -> dict[str, Any]:
    c = row["case"]
    out: dict[str, Any] = {
        "product": c["product"],
        "ambient_shift_k": c["ambient_shift_k"],
        "solid_fraction": c["dryer.solid_fraction"],
        "calibration": c["calibration"],
        "waste_heat": c["waste_heat"],
        "fan_law": c["fan_law"],
        "boiler_efficiency": c["energy.boiler_efficiency"],
        "electricity_co2": c["energy.electricity_co2_kg_per_kwh"],
        "constraint": row["constraint"],
        "air_factor_min": row["air_factor_min"],
        "admissible": row["admissible"],
    }
    if not row["admissible"]:
        return {k: out.get(k) for k in GRID_FIELDS}
    b, best, d = row["baseline"], row["best"], row["decomposition_kg_per_day"]
    out.update(
        {
            "saving_kg_per_day": row["saving_kg_per_day"],
            "saving_fraction": row["saving_fraction"],
            "baseline_co2_kg_per_day": 24.0 * b["co2_kg_h"],
            "best_air_factor": best["air_factor"],
            "best_supply_celsius": best["supply_celsius"],
            "best_outlet_celsius": best["outlet_celsius"],
            "best_outlet_rh": best["outlet_relative_humidity"],
            "best_waste_heat_kw": best["waste_heat_kw"],
            "baseline_waste_heat_kw": b["waste_heat_kw"],
            "saving_steam_kg_per_day": d["steam"],
            "saving_fans_kg_per_day": d["fan_electricity"],
            "binding": "+".join(row["binding"]),
        }
    )
    return {k: out.get(k) for k in GRID_FIELDS}


def run_grid(data: StudyData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = grid_cases()
    t0 = time.perf_counter()
    for k, case in enumerate(cases):
        rows += [compact(r) for r in run_case(data, case, VARIANTS, SPACE)]
        if (k + 1) % 128 == 0:
            print(
                f"  grid {k + 1}/{len(cases)} cases, {time.perf_counter() - t0:.0f} s", flush=True
            )
    return rows


def summarise_grid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Tables of the readings 1 and 3 (reference row and ranges over the other placeholders)."""

    def sel(**kw: Any) -> list[dict[str, Any]]:
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    def stats(rs: list[dict[str, Any]], key: str) -> dict[str, Any]:
        ok = [r[key] for r in rs if r["admissible"]]
        if not ok:
            return {"n": 0, "n_inadmissible": len(rs)}
        return {
            "n": len(ok),
            "n_inadmissible": len(rs) - len(ok),
            "min": float(np.min(ok)),
            "median": float(np.median(ok)),
            "max": float(np.max(ok)),
        }

    ref = {
        "solid_fraction": 0.65,
        "calibration": "own_day",
        "waste_heat": "fixed_water_inlet",
        "boiler_efficiency": 0.88,
        "electricity_co2": ELECTRICITY_CO2_GRID[0],
    }
    table: dict[str, Any] = {}
    for label, shift, cons, amin, fan in product(
        PRODUCTS, AMBIENT_SHIFTS_K, CONSTRAINTS, AIR_FRACTION_MIN_GRID, ("data_fit", "affinity")
    ):
        key = f"{label}|{shift:+.0f}K|{cons}|{amin}|{fan}"
        cell = sel(
            product=label, ambient_shift_k=shift, constraint=cons, air_factor_min=amin, fan_law=fan
        )
        r0 = [r for r in cell if all(r[k] == v for k, v in ref.items())]
        table[key] = {
            "reference": r0[0] if r0 else None,
            "range_saving_kg_per_day": stats(cell, "saving_kg_per_day"),
            "range_saving_fraction": stats(cell, "saving_fraction"),
            "range_steam_share": stats(
                [
                    {
                        **r,
                        "steam_share": (r["saving_steam_kg_per_day"] / r["saving_kg_per_day"])
                        if r["admissible"] and r["saving_kg_per_day"] > 1e-9
                        else None,
                        "admissible": r["admissible"] and r["saving_kg_per_day"] > 1e-9,
                    }
                    for r in cell
                ],
                "steam_share",
            ),
        }
    binding: dict[str, dict[str, int]] = {}
    for r in rows:
        if not r["admissible"]:
            continue
        key = f"{r['product']}|{r['constraint']}"
        binding.setdefault(key, {})
        binding[key][r["binding"] or "none"] = binding[key].get(r["binding"] or "none", 0) + 1
    per_product = {
        label: {
            "all": stats(sel(product=label), "saving_kg_per_day"),
            "all_fraction": stats(sel(product=label), "saving_fraction"),
            "weather_0": stats(sel(product=label, ambient_shift_k=0.0), "saving_kg_per_day"),
            "weather_0_fraction": stats(sel(product=label, ambient_shift_k=0.0), "saving_fraction"),
        }
        for label in PRODUCTS
    }
    return {
        "cells": table,
        "binding_counts": binding,
        "per_product": per_product,
        "reference_selection": ref,
    }


# ------------------------------------------------------------------------------- sensitivity


def one_record(
    data: StudyData, case: Case, variant: ConstraintVariant, air_min: float
) -> dict[str, Any]:
    return run_case(data, case, [(variant, air_min)], SPACE)[0]


def sensitivity(data: StudyData, x_consistent: float) -> dict[str, Any]:
    """Reading 4: one placeholder at a time around the reference case."""
    changes: dict[str, list[tuple[str, Any]]] = {
        "solid_fraction": [
            (f"{v}", ("dryer", {"solid_fraction": v})) for v in (0.60, 0.70, 0.75, x_consistent)
        ],
        "calibration_day": [("other_day", ("case", {"calibration": "other_day"}))],
        "waste_heat_variant": [("duty_held", ("case", {"waste_heat": "duty_held"}))],
        "fan_law": [(n, ("case", {"fan_law": n})) for n in ("affinity", "constant", "quadratic")],
        "boiler_efficiency": [(f"{v}", ("energy", {"boiler_efficiency": v})) for v in (0.85, 0.92)],
        "electricity_co2": [("0.20", ("energy", {"electricity_co2_kg_per_kwh": 0.20}))],
        "constraint_variant": [(v, ("constraint", v)) for v in ("b", "a_and_b")],
        "air_factor_min": [(f"{v}", ("air_min", v)) for v in (0.6, 0.7, 0.9)],
        "weather_shift_k": [
            (f"{v:+.0f}", ("case", {"ambient_shift_k": v})) for v in (-10.0, -5.0, 5.0)
        ],
        "intake_relative_humidity": [
            (f"{v}", ("dryer", {"intake_relative_humidity": v})) for v in (0.5, 0.9)
        ],
        "feed_temperature": [
            (f"{v}", ("dryer", {"feed_temperature_celsius": v})) for v in (20.0, 50.0)
        ],
        "atomising_air_temperature": [
            (f"{v}", ("dryer", {"atomising_air_temperature_celsius": v})) for v in (15.0, 40.0)
        ],
        "room_temperature": [
            (f"{v}", ("dryer", {"room_temperature_celsius": v})) for v in (10.0, 30.0)
        ],
        "cp_solid": [(f"{v}", ("dryer", {"cp_solid_j_per_kg_k": v})) for v in (800.0, 1600.0)],
        "fan_heat_fraction": [(f"{v}", ("energy", {"fan_heat_fraction": v})) for v in (0.8, 1.0)],
        "max_preheater_water_with_duty_held": [
            (f"{v}", ("duty_held_energy", {"max_preheater_water_celsius": v}))
            for v in (70.0, 80.0, 90.0)
        ],
    }
    out: dict[str, Any] = {"reference": {}, "swings": {}}
    for label in PRODUCTS:
        r = one_record(data, reference_case(label), REFERENCE_CONSTRAINT, REFERENCE_AIR_MIN)
        out["reference"][label] = r
    for name, options in changes.items():
        out["swings"][name] = {}
        for opt_label, (kind, value) in options:
            per: dict[str, Any] = {}
            for label in PRODUCTS:
                case, variant, air_min = (
                    reference_case(label),
                    REFERENCE_CONSTRAINT,
                    REFERENCE_AIR_MIN,
                )
                if kind == "dryer":
                    case = with_dryer(case, **value)
                elif kind == "energy":
                    case = with_energy(case, **value)
                elif kind == "case":
                    case = replace(case, **value)
                elif kind == "duty_held_energy":
                    case = with_energy(replace(case, waste_heat="duty_held"), **value)
                elif kind == "constraint":
                    variant = value
                elif kind == "air_min":
                    air_min = value
                rec = one_record(data, case, variant, air_min)
                ref = out["reference"][label]
                per[label] = {
                    "admissible": rec["admissible"],
                    "saving_kg_per_day": rec.get("saving_kg_per_day"),
                    "delta_kg_per_day": (rec["saving_kg_per_day"] - ref["saving_kg_per_day"])
                    if rec["admissible"]
                    else None,
                    "saving_fraction": rec.get("saving_fraction"),
                    "binding": rec.get("binding"),
                    "best_air_factor": rec.get("best", {}).get("air_factor"),
                    "best_supply_celsius": rec.get("best", {}).get("supply_celsius"),
                }
            out["swings"][name][opt_label] = per
    ranking = []
    for name, opts in out["swings"].items():
        deltas = [
            abs(p["delta_kg_per_day"])
            for o in opts.values()
            for p in o.values()
            if p["delta_kg_per_day"] is not None
        ]
        inadmissible = sum(1 for o in opts.values() for p in o.values() if not p["admissible"])
        ranking.append(
            {
                "placeholder": name,
                "max_abs_delta_kg_per_day": max(deltas, default=0.0),
                "n_inadmissible": inadmissible,
            }
        )
    ranking.sort(key=lambda d: -d["max_abs_delta_kg_per_day"])
    out["ranking"] = ranking
    return out


def annualise(summary: dict[str, Any], sens: dict[str, Any]) -> dict[str, Any]:
    """Reading 5: order of magnitude per year, upper bound under placeholders."""
    out: dict[str, Any] = {"basis": "kg/d x 365 (24/7 as stated by the partner)"}
    ref = {label: sens["reference"][label]["saving_kg_per_day"] for label in PRODUCTS}
    rng = {label: summary["per_product"][label]["weather_0"] for label in PRODUCTS}
    for label in PRODUCTS:
        t_ref = ref[label] * 365.0 / 1000.0
        out[label] = {
            "reference_t_per_a": t_ref,
            "range_t_per_a_weather_0": [rng[label]["min"] * 0.365, rng[label]["max"] * 0.365],
            "reference_share_of_site_co2": t_ref / SITE_CO2_T_PER_A,
            "reference_eur_per_a": {f"{p:.0f}": t_ref * p for p in CO2_PRICES_EUR_PER_T},
        }
    mix = 0.5 * (ref["product_1"] + ref["product_2"]) * 365.0 / 1000.0
    out["mix_50_50_reference_t_per_a"] = mix
    out["mix_50_50_reference_eur_per_a"] = {f"{p:.0f}": mix * p for p in CO2_PRICES_EUR_PER_T}
    out["site_co2_t_per_a"] = SITE_CO2_T_PER_A
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="study_plant_a_operating_point")
    args = parser.parse_args(argv)
    require_clean_tree()
    t0 = time.perf_counter()
    data, hashes = load_data()
    ph = DryerPlaceholders()
    x_cons, _ = consistent_solid_fraction(
        ph, day_state(data.days["product_1"]), day_state(data.days["product_2"])
    )
    print("data and calibration ...", flush=True)
    data_out = data_section(data)
    calib = calibration_section(data)
    base = baseline_section(data)
    print("main grid ...", flush=True)
    rows = run_grid(data)
    summary = summarise_grid(rows)
    print("sensitivity ...", flush=True)
    sens = sensitivity(data, x_cons)
    payload = {
        "_provenance": provenance(NO_HEAT_NETWORK),
        "question": "CO2 lever of another air flow / supply temperature at plant A (steam + fans)",
        "caveat": "proxy constraints do not replace a product-moisture measurement; the result is "
        "an upper bound of the lever, not an operating recommendation; all non-measured inputs "
        "are placeholders",
        "data_sha256": hashes,
        "placeholders": {k: asdict(v) for k, v in PLACEHOLDERS.items()},
        "search_space": asdict(SPACE),
        "search_space_fingerprint": fingerprint_of(SPACE),
        "reference_case": {
            label: {
                **reference_case(label).to_dict(),
                "fingerprint": reference_case(label).fingerprint,
                "constraint": REFERENCE_CONSTRAINT,
                "air_factor_min": REFERENCE_AIR_MIN,
            }
            for label in PRODUCTS
        },
        "data": data_out,
        "calibration_check": calib,
        "baseline": base,
        "grid": {
            "fields": list(GRID_FIELDS),
            "columns": {f: [r[f] for r in rows] for f in GRID_FIELDS},
            "n_rows": len(rows),
        },
        "summary": summary,
        "sensitivity": sens,
        "annualised": annualise(summary, sens),
        "wall_time_s": time.perf_counter() - t0,
    }
    write_artifact(args.name, payload)


if __name__ == "__main__":
    main()
