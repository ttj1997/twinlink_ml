r"""Storage test 1: time-slice pinch -- upper bound for the benefit of a heat storage.

Question: how much fossil residual heat can an IDEAL heat storage save at most, compared with
using heat only at the moment it occurs, and does a simple rule without foresight reach it?
This is the gate for any RL or MPC set-up on the heat network. No reinforcement learning here.

THIS DOCSTRING IS THE PRE-REGISTRATION. It was committed before any number of this study was
computed; nothing in it may be changed after seeing results (corrections only as dated addenda in
the lab book). Every number below that is not a measurement is a PLACEHOLDER (source and band are
stated) and every result carries its scenario and says so. Nothing here is a plant statement.

Method (code: ``twinlink_ml.storage_timeslice``; reuses ``heatnetwork.pinch`` for the stream
segments and the condensation curve of the exhaust)
    Time grid 15 min (block means of the one-minute plant data); horizons 1 day and 1 week.
    Per step t the problem-table algorithm (``heatnetwork.pinch``, minimum approach ``dT_min``)
    gives the minimum hot utility U_t of all hot streams (sources, cooled down to ``T_kalt``,
    condensation allowed) and cold streams (sinks) that exist at that step:
    E_direct = sum_t U_t dt, "fossil residual heat" without storage (heat usable only when it
    occurs). U is the useful heat of the heater, not fuel; fuel = U / efficiency, which does not
    change any ratio.
    Ideal storage of capacity C [kWh] at ONE temperature level T_S: charging is an additional
    isothermal cold stream at T_S, discharging an additional isothermal hot stream at T_S in the
    same problem table, i.e. charging takes heat only from source parts above T_S + a and
    discharging serves sink parts only below T_S - a, ``a`` = approach on the storage side
    (default ``a = dT_min``, as specified; the variant ``a = dT_min/2`` is the sensitivity that
    treats the storage as part of the existing water circuit). Because the stream temperatures
    are fixed, U_t is piecewise linear and convex in (q_charge, q_discharge) with only three
    active pieces per step (see the module); E_ideal(C) is therefore a linear programme over the
    horizon: minimise sum_t U_t dt with s_{t+1} = (1 - lambda dt) s_t + (q_c - q_d) dt,
    0 <= s_t <= C, no power limits, CYCLIC (s at the end = s at the start, so the storage
    cannot be emptied for free). E_ideal(C) = the best T_S of the grid (30 .. 80 degC, 5 K).
    As a check of the upper-bound character the same programme is solved with ALL levels of the
    grid at once sharing the capacity (multi-level ideal storage) on the reference scenarios.
    Rule E_rule(C): no foresight, same storage, same T_S grid (best T_S reported), per step:
    first discharge as much as reduces the hot utility (not more), then charge with exactly the
    surplus that does not raise it ("charge surplus, discharge on demand", greedy), run cyclically
    to its periodic steady state. This is the natural rule; no deliberately weak rule is chosen.
    Checks (tests and recorded in the artifact): C = 0 gives E_direct exactly; storage energy
    balance closed (without losses sum q_c = sum q_d; with losses the difference equals the
    loss); E_ideal non-increasing in C; E_rule >= E_ideal; U of the programme equals the
    problem table with the isothermal streams added explicitly.
    Losses: fraction of the stored energy per hour, 0 and the placeholder 0.5 %/h.

Scenarios (all placeholders as grids; band = what counts as REALISTIC)
    Sources (hot): compressor heat of plant A: level 400 / 800 / 1200 kW (partner: about 1200 kW
    in total, about 400 kW used today; band 400-1200), as a stream from 85 to 45 degC (water
    loop; placeholder, band top 70-90, bottom 40-55). Compressor heat of plant C: mean 2.4 GWh/a /
    8760 h = 274 kW (partner annual figure, placeholder), same temperatures. Exhaust of plant A:
    the two measured days (product 1 / 2): measured outlet temperature, humidity RECONSTRUCTED
    (not measured) from the spray rate, solid fraction 0.3 (placeholder, band 0.2-0.5) and the
    intake humidity (relative humidity 0.75, placeholder, band 0.5-0.9), capped at saturation
    at the outlet temperature; condensation down to ``T_kalt`` = 15 degC (repository default,
    band 10-25). Exhaust of plant C, lines C1 and C2: about 20 000 Nm3/h per line (partner,
    band 15 000-25 000), state of the measured exhaust of plant A (placeholder: same process);
    variant: scaled so that the flat-profile heat down to ``T_kalt`` equals the partner annual
    figure 1.8 GWh/a (reported as a check, not forced).
    Sinks (cold): process air of plant A (measured days: flow, intake temperature, supply
    temperature as logged; no credit for fan heat); process air of plant C, lines C1/C2 (20 000
    Nm3/h each, supply temperature profile of the measured day, outdoor-air profile of the
    measured day); hall heating up to 200 kW (band 0-200) as a stream from 35 to 50 degC with a
    demand proportional to the heating degree (15 degC heating limit, -10 degC design point)
    times a night factor (1.0 from 18:00 to 06:00, 0.4 otherwise; placeholder); process water
    0 / 200 kW (band 0-200) from 15 to 60 degC (placeholder), constant or tied to the shift.
    ``dT_min`` 20 K (repository default, band 10-30).
    Operating modes (profiles; plant A runs 24/7 in every mode, as the partner states):
    ``flat``    compressors constant, C1 and C2 24/7, process water constant;
    ``shift``   compressors 100 % from 06:00 to 22:00 and 30 % otherwise (band 10-50 %), C1 from
                06:00 to 22:00, C2 from 06:00 to 14:00, process water in 06:00-22:00 only;
    ``weekly``  (week horizon only) compressors 100 % Monday-Friday and 30 % Saturday-Sunday,
                C1 and C2 Monday-Friday 24 h and off at the weekend, process water Monday-Friday.
    Day horizon: modes flat and shift; week horizon: flat, shift, weekly (the measured day is
    repeated seven times; weather constant over the week).
    Weather: both measured days, ambient temperature shifted by -10 / -5 / 0 / +5 K (the
    exhaust state is NOT shifted). Capacity C: 0, 0.5, 1, 2, 5, 10, 20 MWh (for comparison:
    storage A1 holds about 0.7 MWh).
    Main grid: site {A, C, A+C} x product day {1, 2} x weather {4 values} x mode x compressor
    level {400, 800, 1200 kW; plant C alone: n/a} x hall {0, 200 kW} x process water {0,
    200 kW} x loss {0, 0.5 %/h} x horizon {day, week}. Plus the sensitivity variants below (one
    placeholder at a time around the two reference scenarios); all of them are inside the
    bands and enter the gate. C = 20 MWh is outside the realistic range and is reported only.

Gate (pre-registered). A CELL is (scenario, capacity C <= 10 MWh); the saving is
    G = (E_direct - E_ideal) / E_direct and the rule gap D = (E_rule - E_ideal) / E_direct
    (both on the same best-T_S convention, per day: week values divided by 7 for kWh/d; the
    fractions are unaffected). Scenarios with E_direct < 100 kWh/d are excluded from the
    gate (the percentages have no meaning there; their number is reported).
    L1  RL / MPC set-up justified:  some cell has G >= 5 % AND D >= 2 percentage points.
    L2  storage pays, a rule suffices:  some cell has G >= 5 %, but no cell has both.
    L3  no relevant potential:  no cell has G >= 5 %.
    The letter reading is a statement about the BEST case over the placeholder bands and must
    not be quoted alone. Always reported with it: the number and share of cells with G >= 5 %
    and with both conditions, the readings per site subset, per horizon and per product day, and
    the reading restricted to the conditions closest to the measured days (site A, mode flat,
    no hall heating, no process water).

Evaluation (all labelled with the scenario)
    Per scenario: saving over C in kWh/d and in % of E_direct, C at 90 % of the maximum saving
    (over the C grid up to 20 MWh), gap rule - ideal.
    Decomposition on three scenarios fixed by rule (the reference scenario; the realistic cell
    with the largest saving fraction; the one with the largest saving in kWh/d): leave one source
    or one sink out at a time (saving recomputed), and the matrix of single source x single
    sink pairs (all other streams removed). The split is order-dependent and says so.
    Sensitivity: reference scenarios (site A+C, product 1, 0 K, compressor 800 kW, hall 200 kW,
    process water 200 kW, loss 0): (R1) day horizon, mode shift; (R2) week horizon, mode weekly.
    One placeholder at a time over its band (``dT_min`` 10/30, ``T_kalt`` 10/25, compressor
    level 400/1200, compressor temperatures 75-45 and 90-50, low fraction 0.1/0.5, hall 0/100 kW,
    hall temperatures, process water 0/100 kW, line flow 15 000/25 000, exhaust C by annual figure,
    plant C compressor 1.2/3.6 GWh/a, solid fraction 0.2/0.5, humidity 0.5/0.9, loss 0.5/1 %/h,
    storage approach dT_min/2, shift window 07-19 / 05-23, weather -10/+5 K, product day 2),
    ranked by the change of the saving at C = 5 MWh in kWh/d and in percentage points of
    E_direct. The ranking becomes the list of questions to the partner (which time series are
    really needed).

Call::

    TWINLINK_PLANT_DATA_ACE=... TWINLINK_PLANT_DATA_RID=... TWINLINK_LOCAL_DIR=... \
        tools/run_here.sh scripts/gate_storage_timeslice.py --name gate_storage_timeslice
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import NO_HEAT_NETWORK, REPO_ROOT, provenance, require_clean_tree, write_artifact

from twinlink_ml.storage_timeslice import (
    CAPACITIES_KWH,
    LEVELS_CELSIUS,
    PLACEHOLDER_BANDS,
    MeasuredDay,
    Placeholders,
    Scenario,
    ScenarioResult,
    StreamFactory,
    evaluate_scenario,
    load_measured_day,
    site_streams,
)
from twinlink_ml.storage_timeslice.config import is_realistic
from twinlink_ml.storage_timeslice.gate import (
    capacity_at_fraction_of_maximum,
    gate_cells,
    gate_verdict,
    rank_sensitivity,
    subset_verdicts,
)

PRODUCT_ENV = {"product_1": "TWINLINK_PLANT_DATA_ACE", "product_2": "TWINLINK_PLANT_DATA_RID"}
ROLE_FILE = REPO_ROOT / "conf" / "plant" / "plant_local.yaml"
SITES = ("A", "C", "A+C")
DAYS = ("product_1", "product_2")
SHIFTS_K = (-10.0, -5.0, 0.0, 5.0)
HORIZON_MODES = (
    ("day", "flat"),
    ("day", "shift"),
    ("week", "flat"),
    ("week", "shift"),
    ("week", "weekly"),
)
COMPRESSOR_LEVELS_KW = (400.0, 800.0, 1200.0)
HALL_KW = (0.0, 200.0)
WATER_KW = (0.0, 200.0)
LOSSES_PER_H = (0.0, 0.005)
SENSITIVITY_CAPACITY_KWH = 5000.0
DECOMPOSITION_CAPACITIES_KWH = (0.0, 5000.0, 10000.0)
PARTNER_EXHAUST_C_KW = 1.8e6 / 8760.0


def main_grid() -> list[Scenario]:
    """The main grid of the pre-registration (plant C alone has no compressor-A level axis)."""
    out: list[Scenario] = []
    for site, day, shift, (horizon, mode), hall, water, loss in product(
        SITES, DAYS, SHIFTS_K, HORIZON_MODES, HALL_KW, WATER_KW, LOSSES_PER_H
    ):
        levels = (800.0,) if site == "C" else COMPRESSOR_LEVELS_KW
        for comp in levels:
            out.append(
                Scenario(
                    site=site,  # type: ignore[arg-type]
                    day=day,
                    horizon=horizon,  # type: ignore[arg-type]
                    mode=mode,  # type: ignore[arg-type]
                    ambient_shift_k=shift,
                    compressor_a_kw=comp,
                    hall_kw=hall,
                    water_kw=water,
                    loss_per_h=loss,
                )
            )
    return out


def reference_scenarios() -> dict[str, Scenario]:
    """R1: day horizon, mode shift; R2: week horizon, mode weekly (site A+C, product 1, 0 K)."""
    base = {
        "site": "A+C",
        "day": "product_1",
        "ambient_shift_k": 0.0,
        "compressor_a_kw": 800.0,
        "hall_kw": 200.0,
        "water_kw": 200.0,
        "loss_per_h": 0.0,
    }
    return {
        "R1": Scenario(horizon="day", mode="shift", **base),  # type: ignore[arg-type]
        "R2": Scenario(horizon="week", mode="weekly", **base),  # type: ignore[arg-type]
    }


def sensitivity_variants(base: Scenario) -> dict[str, Scenario]:
    """One placeholder at a time over its band (pre-registered list)."""
    v: dict[str, Scenario] = {}

    def ph(name: str, *values: Any, **extra: Any) -> None:
        for value in values:
            v[f"{name}={value}"] = base.with_placeholders(**{name: value}, **extra)

    ph("minimum_pinch_k", 10.0, 30.0)
    ph("cold_temperature_celsius", 10.0, 25.0)
    for level in (400.0, 1200.0):
        v[f"compressor_a_kw={level:g}"] = replace(base, compressor_a_kw=level)
    v["compressor_temperatures=75-45"] = base.with_placeholders(
        compressor_hot_in_celsius=75.0, compressor_hot_out_celsius=45.0
    )
    v["compressor_temperatures=90-50"] = base.with_placeholders(
        compressor_hot_in_celsius=90.0, compressor_hot_out_celsius=50.0
    )
    ph("low_fraction", 0.1, 0.5)
    for kw in (0.0, 100.0):
        v[f"hall_kw={kw:g}"] = replace(base, hall_kw=kw)
    v["hall_temperatures=40-30"] = base.with_placeholders(
        hall_supply_celsius=40.0, hall_return_celsius=30.0
    )
    v["hall_temperatures=60-45"] = base.with_placeholders(
        hall_supply_celsius=60.0, hall_return_celsius=45.0
    )
    for kw in (0.0, 100.0):
        v[f"water_kw={kw:g}"] = replace(base, water_kw=kw)
    ph("c_line_nm3_per_h", 15000.0, 25000.0)
    v["exhaust_c_basis=annual"] = base.with_placeholders(exhaust_c_basis="annual")
    ph("compressor_c_gwh_per_a", 1.2, 3.6)
    ph("solid_fraction", 0.2, 0.5)
    ph("relative_humidity", 0.5, 0.9)
    for loss in (0.005, 0.01):
        v[f"loss_per_h={loss:g}"] = replace(base, loss_per_h=loss)
    v["storage_approach=dT_min/2"] = base.with_placeholders(
        storage_approach_k=base.placeholders.minimum_pinch_k / 2.0
    )
    v["shift_window=07-19"] = base.with_placeholders(shift_start_h=7.0, shift_end_h=19.0)
    v["shift_window=05-23"] = base.with_placeholders(shift_start_h=5.0, shift_end_h=23.0)
    for shift in (-10.0, 5.0):
        v[f"ambient_shift_k={shift:+g}"] = replace(base, ambient_shift_k=shift)
    v["day=product_2"] = replace(base, day="product_2")
    return v


# ---------------------------------------------------------------------------------- workers

_FACTORY: StreamFactory | None = None


def _init_worker(days: dict[str, MeasuredDay]) -> None:
    global _FACTORY
    _FACTORY = StreamFactory(days)


def _run_task(task: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, float]]:
    assert _FACTORY is not None
    scenario: Scenario = task["scenario"]
    bank = _FACTORY.build(scenario)
    result = evaluate_scenario(
        bank,
        scenario,
        hot_names=task.get("hot_names"),
        cold_names=task.get("cold_names"),
        capacities_kwh=task.get("capacities", CAPACITIES_KWH),
        levels_celsius=LEVELS_CELSIUS,
        multi_level=task.get("multi_level", False),
    )
    return task["key"], {"result": result}, dict(bank.notes)


def run_tasks(
    tasks: Sequence[dict[str, Any]], days: dict[str, MeasuredDay], workers: int
) -> dict[str, tuple[ScenarioResult, dict[str, float]]]:
    """Run tasks in a process pool (or in-process for one worker); keys must be unique."""
    keys = [t["key"] for t in tasks]
    if len(set(keys)) != len(keys):
        raise ValueError("task keys must be unique")
    out: dict[str, tuple[ScenarioResult, dict[str, float]]] = {}
    started = time.time()
    if workers <= 1:
        _init_worker(days)
        for i, task in enumerate(tasks, 1):
            key, payload, notes = _run_task(task)
            out[key] = (payload["result"], notes)
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)} ({time.time() - started:.0f}s)", flush=True)
        return out
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(days,)
    ) as pool:
        for i, (key, payload, notes) in enumerate(pool.map(_run_task, tasks, chunksize=4), 1):
            out[key] = (payload["result"], notes)
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    return out


# ------------------------------------------------------------------------------- data, records


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_days() -> tuple[dict[str, MeasuredDay], dict[str, str]]:
    days: dict[str, MeasuredDay] = {}
    hashes: dict[str, str] = {}
    missing = [env for env in PRODUCT_ENV.values() if not os.environ.get(env)]
    if missing:
        raise SystemExit(f"set the environment variables {', '.join(missing)}")
    for label, env in PRODUCT_ENV.items():
        path = os.environ[env]
        hashes[label] = sha256_of(path)
        days[label] = load_measured_day(path, ROLE_FILE, label)
    return days, hashes


def day_consistency(days: dict[str, MeasuredDay]) -> dict[str, Any]:
    """Consistency of the reconstruction with the earlier balance of plant A (Befund 91)."""
    from twinlink_ml.storage_timeslice.plant import exhaust_humidity, intake_humidity

    ph = Placeholders()
    out: dict[str, Any] = {}
    for label, day in days.items():
        y_in = intake_humidity(day.ambient_c, ph.relative_humidity, ph.gas)
        cp = (ph.gas.cp_dry_air_j_per_kg_k + y_in * ph.gas.cp_vapour_j_per_kg_k) / 1000.0
        m_dry = day.process_air_nm3h * ph.normal_density_kg_per_nm3 / 3600.0
        demand = m_dry * cp * (day.supply_air_c - day.ambient_c)
        y_out, m_exh, capped = exhaust_humidity(
            day,
            solid_fraction=ph.solid_fraction,
            relative_humidity=ph.relative_humidity,
            normal_density_kg_per_nm3=ph.normal_density_kg_per_nm3,
            gas=ph.gas,
        )
        out[label] = {
            "process_air_heating_demand_mean_kw_without_fan_credit": float(np.mean(demand)),
            "ambient_mean_celsius": float(np.mean(day.ambient_c)),
            "supply_air_mean_celsius": float(np.mean(day.supply_air_c)),
            "exhaust_outlet_mean_celsius": float(np.mean(day.exhaust_out_c)),
            "exhaust_humidity_reconstructed_mean_kg_per_kg": float(np.mean(y_out)),
            "exhaust_capped_at_saturation_fraction": float(np.mean(capped)),
            "exhaust_dry_air_mean_kg_per_s": float(np.mean(m_exh)),
        }
    return out


def record_of(scenario: Scenario, result: ScenarioResult) -> dict[str, Any]:
    d = result.to_dict()
    rows = d.pop("rows")
    cols = {
        key: [r[key] for r in rows]
        for key in (
            "ideal_kwh_per_day",
            "ideal_level_celsius",
            "rule_kwh_per_day",
            "rule_level_celsius",
            "rule_converged",
            "saving_kwh_per_day",
            "saving_fraction",
            "gap_fraction",
        )
    }
    return {
        "label": d["label"],
        "site": scenario.site,
        "day": scenario.day,
        "horizon": scenario.horizon,
        "mode": scenario.mode,
        "ambient_shift_k": scenario.ambient_shift_k,
        "compressor_a_kw": scenario.compressor_a_kw,
        "hall_kw": scenario.hall_kw,
        "water_kw": scenario.water_kw,
        "loss_per_h": scenario.loss_per_h,
        "realistic": is_realistic(scenario),
        "direct_kwh_per_day": d["direct_kwh_per_day"],
        "capacity_at_90_percent_of_max_saving_kwh": capacity_at_fraction_of_maximum(result),
        **cols,
        "checks": d["checks"],
    }


def check_summary(results: dict[str, ScenarioResult]) -> dict[str, Any]:
    failing: dict[str, list[str]] = {}
    for key, res in results.items():
        for name, value in res.checks.items():
            if isinstance(value, bool) and not value:
                failing.setdefault(name, []).append(key)
    worst = max(
        (r.checks["max_storage_balance_residual_kwh"] for r in results.values()), default=0.0
    )
    return {
        "n_results": len(results),
        "failing": {k: v[:20] for k, v in failing.items()},
        "n_failing": {k: len(v) for k, v in failing.items()},
        "max_storage_balance_residual_kwh": worst,
    }


# ------------------------------------------------------------------------------------- main


def decomposition_tasks(tag: str, scenario: Scenario) -> list[dict[str, Any]]:
    hot, cold = site_streams(scenario.site)
    tasks: list[dict[str, Any]] = []
    for h in hot:
        tasks.append(
            {
                "key": f"{tag}|without|{h}",
                "scenario": scenario,
                "hot_names": tuple(x for x in hot if x != h),
                "cold_names": cold,
                "capacities": DECOMPOSITION_CAPACITIES_KWH,
            }
        )
    for c in cold:
        tasks.append(
            {
                "key": f"{tag}|without|{c}",
                "scenario": scenario,
                "hot_names": hot,
                "cold_names": tuple(x for x in cold if x != c),
                "capacities": DECOMPOSITION_CAPACITIES_KWH,
            }
        )
    for h, c in product(hot, cold):
        tasks.append(
            {
                "key": f"{tag}|pair|{h}|{c}",
                "scenario": scenario,
                "hot_names": (h,),
                "cold_names": (c,),
                "capacities": DECOMPOSITION_CAPACITIES_KWH,
            }
        )
    tasks.append(
        {
            "key": f"{tag}|full",
            "scenario": scenario,
            "capacities": DECOMPOSITION_CAPACITIES_KWH,
            "multi_level": True,
        }
    )
    return tasks


def summarise_decomposition(
    tag: str, scenario: Scenario, results: dict[str, tuple[ScenarioResult, dict[str, float]]]
) -> dict[str, Any]:
    hot, cold = site_streams(scenario.site)

    def numbers(res: ScenarioResult) -> dict[str, Any]:
        row = next(r for r in res.rows if r.capacity_kwh == SENSITIVITY_CAPACITY_KWH)
        top = next(r for r in res.rows if r.capacity_kwh == DECOMPOSITION_CAPACITIES_KWH[-1])
        return {
            "direct_kwh_per_day": res.direct_kwh_per_day,
            "saving_5MWh_kwh_per_day": res.direct_kwh_per_day - row.ideal_kwh_per_day,
            "saving_5MWh_fraction": res.saving_fraction(row),
            "saving_10MWh_kwh_per_day": res.direct_kwh_per_day - top.ideal_kwh_per_day,
            "rule_gap_5MWh_fraction": res.gap_fraction(row),
        }

    full = results[f"{tag}|full"][0]
    out: dict[str, Any] = {
        "scenario": scenario.label,
        "full": numbers(full),
        "multi_level_5MWh_kwh_per_day": next(
            r.multi_level_kwh_per_day
            for r in full.rows
            if r.capacity_kwh == SENSITIVITY_CAPACITY_KWH
        ),
        "ideal_single_level_5MWh_kwh_per_day": next(
            r.ideal_kwh_per_day for r in full.rows if r.capacity_kwh == SENSITIVITY_CAPACITY_KWH
        ),
        "leave_one_out": {},
        "pairs": {},
        "note": "order-dependent split; leave-one-out and single pairs are two readings, not parts of one sum",
    }
    for name in (*hot, *cold):
        out["leave_one_out"][name] = numbers(results[f"{tag}|without|{name}"][0])
    for h, c in product(hot, cold):
        out["pairs"][f"{h}|{c}"] = numbers(results[f"{tag}|pair|{h}|{c}"][0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="artifact name (never overwritten)")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument(
        "--limit", type=int, default=0, help="debug: only the first N grid cases, no artifact"
    )
    args = parser.parse_args()
    if not args.limit:
        require_clean_tree()
    started = time.time()
    days, hashes = load_days()
    print("days loaded", flush=True)

    grid = main_grid()
    refs = reference_scenarios()
    sens = {rid: sensitivity_variants(base) for rid, base in refs.items()}
    if args.limit:
        grid = grid[: args.limit]
    tasks: list[dict[str, Any]] = [{"key": f"grid|{i}", "scenario": s} for i, s in enumerate(grid)]
    for rid, base in refs.items():
        tasks.append({"key": f"{rid}|base", "scenario": base})
        for name, s in sens[rid].items():
            tasks.append({"key": f"{rid}|var|{name}", "scenario": s})
    print(
        f"{len(grid)} grid scenarios, {len(tasks) - len(grid)} reference/sensitivity runs",
        flush=True,
    )
    done = run_tasks(tasks, days, args.workers)
    print(f"batch 1 done ({time.time() - started:.0f}s)", flush=True)

    scenario_of = {t["key"]: t["scenario"] for t in tasks}
    pairs = [(scenario_of[k], done[k][0]) for k in done]
    cells, counts = gate_cells(pairs)
    groupings = {
        "site": lambda s: s.site,
        "horizon": lambda s: s.horizon,
        "day": lambda s: s.day,
        "mode": lambda s: s.mode,
    }
    verdict = gate_verdict(cells, counts)
    closest = [
        (scenario_of[k], done[k][0])
        for k in done
        if scenario_of[k].site == "A"
        and scenario_of[k].mode == "flat"
        and scenario_of[k].hall_kw == 0.0
        and scenario_of[k].water_kw == 0.0
        and scenario_of[k].ambient_shift_k == 0.0
        and scenario_of[k].loss_per_h == 0.0
        and k.startswith("grid|")
    ]
    closest_cells, closest_counts = gate_cells(closest)
    gate: dict[str, Any] = {
        "verdict": verdict,
        "per_group": subset_verdicts(cells, groupings),
        "closest_to_measured_days_site_A_flat_no_hall_no_water": gate_verdict(
            closest_cells, closest_counts
        ),
    }
    print("gate:", gate["verdict"]["reading"], flush=True)

    # decomposition and multi-level check on scenarios fixed by rule
    best_frac = max(cells, key=lambda c: c.saving_fraction)
    best_kwh = max(cells, key=lambda c: c.saving_kwh_per_day)
    chosen = {
        "R1": refs["R1"],
        "R2": refs["R2"],
        "max_fraction": best_frac.scenario,
        "max_kwh": best_kwh.scenario,
    }
    tasks2: list[dict[str, Any]] = []
    for tag, scen in chosen.items():
        tasks2.extend(decomposition_tasks(tag, scen))
    done2 = run_tasks(tasks2, days, args.workers)
    decomposition = {tag: summarise_decomposition(tag, scen, done2) for tag, scen in chosen.items()}

    sensitivity: dict[str, Any] = {}
    for rid in refs:
        base_res = done[f"{rid}|base"][0]
        variants = {name: done[f"{rid}|var|{name}"][0] for name in sens[rid]}
        sensitivity[rid] = {
            "base": refs[rid].label,
            "capacity_kwh": SENSITIVITY_CAPACITY_KWH,
            "base_direct_kwh_per_day": base_res.direct_kwh_per_day,
            "ranking": rank_sensitivity(base_res, variants, SENSITIVITY_CAPACITY_KWH),
            "notes": done[f"{rid}|base"][1],
        }

    all_results = {k: v[0] for k, v in done.items()}
    all_results.update({f"decomp|{k}": v[0] for k, v in done2.items()})
    exhaust_c_notes = {rid: done[f"{rid}|base"][1] for rid in refs}
    payload: dict[str, Any] = {
        "_provenance": provenance(NO_HEAT_NETWORK),
        "name": args.name,
        "raw_file_sha256": hashes,
        "days": day_consistency(days),
        "partner_exhaust_c_mean_kw_from_annual_figure": PARTNER_EXHAUST_C_KW,
        "exhaust_c_check_reference_scenarios": exhaust_c_notes,
        "placeholders_default": asdict(Placeholders()),
        "placeholder_bands": {k: list(v) for k, v in PLACEHOLDER_BANDS.items()},
        "capacities_kwh": list(CAPACITIES_KWH),
        "levels_celsius": list(LEVELS_CELSIUS),
        "gate": gate,
        "decomposition": decomposition,
        "sensitivity": sensitivity,
        "checks": check_summary(all_results),
        "grid": [record_of(scenario_of[k], done[k][0]) for k in done if k.startswith("grid|")],
        "reference_runs": {
            k: record_of(scenario_of[k], done[k][0]) for k in done if not k.startswith("grid|")
        },
        "runtime_seconds": time.time() - started,
    }
    if args.limit:
        print("debug run (--limit): artifact not written")
        print(gate["verdict"])
        return 0
    write_artifact(args.name, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
