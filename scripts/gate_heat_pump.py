r"""Heat-pump study: Stage A (heat pump on waste heat), Stage B (plus storage) and the gate.

Question: does a heat pump on low-temperature waste heat pay, by CO2 and by cost, today and 2030
(Stage A), and does heat pump plus storage under a time-varying power signal need foresight
(Stage B: the gate for any RL or MPC set-up)?

THIS DOCSTRING IS THE PRE-REGISTRATION. It was committed before any number of this study was
computed on the plant data (the package ``twinlink_ml.heatpump`` and its tests had been written and
run on synthetic streams only); nothing in it may be changed after seeing results (corrections
only as dated addenda in the lab book). Every number below that is not a measurement or a partner
statement is a PLACEHOLDER (central value, band, source status in the docstrings of
``twinlink_ml.heatpump.config`` and ``twinlink_ml.storage_timeslice.config``); every result
carries its scenario and says so. Nothing here is a plant statement. No reinforcement learning.

Streams and site (unchanged from storage test 1, Befund 99; ``twinlink_ml.storage_timeslice``)
    Time grid 15 min, the two measured days of plant A (product 1 / product 2, block means),
    horizons 1 day and 1 week (the day repeated). Sources: compressor heat of plant A (level
    400 / 800 / 1200 kW, 85 -> 45 degC; the user's statement "cooling water 50-80 degC" is the
    sensitivity variant 80 -> 50), compressor heat of plant C (2.4 GWh/a), exhaust of plant A
    (measured outlet temperature, humidity reconstructed, condensation down to ``T_kalt`` =
    15 degC), exhaust of plant C (two lines of 20 000 Nm3/h, state of the exhaust of plant A).
    Sinks: process air of plant A (measured), process air of plant C (lines C1/C2), hall heating
    0 / 200 kW, process water 0 / 200 kW. Operating modes flat / shift / weekly and weather
    shifts as in Befund 99. ``dT_min`` 20 K. The CONTRADICTION of Befund 99 (exhaust of plant C by
    volume flow = 4.9 x the partner's annual figure) is carried as two variants: ``volume`` and
    ``annual`` (scaled to 1.8 GWh/a) for every scenario that contains plant C.

Heat-pump model (``twinlink_ml.heatpump``)
    COP = min(quality x COP_Carnot(T_e, T_k), cop_max), COP_Carnot = T_k / (T_k - T_e) in K;
    quality 0.50 (grid 0.45 / 0.50 / 0.55, band = user 0.45-0.55); admissible designs: T_e in
    5 ... 70 degC, T_k in 60 ... 90 degC (5 K grid), 20 K <= lift <= 60 K, T_k <= 90 degC, cop_max
    8. Evaporator and condenser are ISOTHERMAL PSEUDO-STREAMS in the same problem table as all
    process streams (approach dT_min/2 on the process side, refrigerant contribution 0): the
    evaporator is a cold stream at T_e (takes heat from hot parts above T_e + 10 K; condensation
    of the exhaust is allowed because the exhaust stream is sampled with its condensation curve),
    the condenser a hot stream at T_k (serves cold parts below T_k - 10 K). The hot utility per
    step is then exact and piecewise linear in the condenser heat Q (Q_e = Q (1 - 1/COP),
    W = Q / COP), so the PINCH RULE is not imposed but follows from the table: across the pinch
    dU = Q, entirely above dU = W, entirely below dU = 0.
    BASELINE: the pinch target of the site with maximum DIRECT heat recovery (as in Befund 99);
    the heat pump is credited only with what it saves on top of direct recovery.
    Gas heat: heater efficiency 0.85 (band 0.75-0.95; steam heater of plant A and direct burner
    of plant C, one value). Criteria (value of a kWh of gas heat c_U and of a kWh of electricity
    c_el): CO2 with gas 0.201 kg/kWh and electricity 0.366 (repository factors, "today") or 0.20
    (variant "2030", placeholder); cost with gas price 0.030 / 0.040 / 0.060 EUR/kWh (without
    CO2), electricity 0.12 / 0.16 / 0.20 EUR/kWh, CO2 price 55 / 150 / 300 EUR/t on gas
    (placeholders; central: 0.040 / 0.16; "today" = 55, "2030" = 150 and 300). Capital (only for
    the payback column): 600 EUR/kW (band 400-1000), 15 a, 6 %, 2 %/a maintenance.

Stage A (stationary: the heat pump has no storage; it runs at the best output per step)
    Per step and criterion the operator chooses 0 <= Q <= Q_max minimising c_U U(Q) + c_el Q/COP
    (exact: ends and kinks of a convex piecewise-linear function). Per scenario, size and
    criterion the best admissible design pair (T_e, T_k) is reported with its heat, the gas heat
    replaced, the work, the saving of the criterion, the share of its heat delivered in steps in
    which the pair straddles the pinch, and annual figures (day x 365, a scaling of the measured
    day, not an annual simulation). Sizes Q_max: 0, 100, 250, 500, 750, 1000, 1250, 1500 kW.
    Main grid A: site {A, C, A+C} x day {1, 2} x mode {flat, shift} (day horizon) x weather
    {-10, 0, +5 K} x compressor level of plant A {400, 800, 1200 kW; not for site C} x hall
    {0, 200 kW} x process water {0, 200 kW} x exhaust-C basis {volume, annual; not for site A}
    = 528 scenarios, each for the three qualities and the 29 criteria (2 CO2 + 27 cost).
    Reference scenarios: RA1 = A+C, product 1, flat, 0 K, compressor 800, hall 200, water 200,
    exhaust C by volume; RA2 = RA1 with exhaust C by annual figure; RA3 = A, product 1, flat,
    0 K, compressor 800, hall 0, water 0 (closest to the measured days); RA4 = RA3 on product 2.
    For them: own pinch calculation (utilities, pinch temperatures, mean grand composite curve on
    -10 ... 100 degC shifted), the pinch-rule check (every admissible pair at every size: placement
    per step and the utility saving at forced full output against Q, W and 0), break-even sweeps
    at 500 and 1000 kW, quality 0.50: electricity CO2 factor 0 ... 1 kg/kWh (step 0.02) and
    electricity value / gas-heat value 0.5 ... 8 (step 0.1); quality sweep 0.30 ... 0.70 (step
    0.05) under CO2 today and cost today ("from which COP does it pay").
    Sensitivity (one placeholder at a time around RA1 and RA3, size 500 kW, criteria CO2 today,
    CO2 2030, cost today, cost 300): quality 0.45 / 0.55; max lift 50 / 70 K; min lift 15 / 30 K;
    max condensing 80 / 100 degC; cop_max 6 / 10; heat-pump approach 5 / 20 K; dT_min 10 / 30 K;
    T_kalt 10 / 25 degC; compressor level 400 / 1200 kW; compressor temperatures 80-50 / 75-45 /
    90-50; solid fraction 0.2 / 0.5; intake humidity 0.5 / 0.9; exhaust C by annual figure; line
    flow 15 000 / 25 000 Nm3/h; compressor C 1.2 / 3.6 GWh/a; hall 0 / 100 kW; water 0 / 100 kW;
    weather -10 / +5 K; product day 2; mode shift; heater efficiency 0.75 / 0.95; electricity
    price 0.12 / 0.20; gas price 0.030 / 0.060 (cost criteria only). Ranked by the absolute change
    of the annual saving (t CO2/a, kEUR/a); the ranking becomes the list of questions to the
    partner. No gate in Stage A: it answers the question with grids.

Stage B (heat pump plus storage plus time-varying signal; the GATE)
    Storage as in storage test 1 (one level T_S in {40, 50, 60, 70 degC}, approach dT_min, i.e.
    charging from hot parts above T_S + 20 K -- among them the condenser -- and discharging to
    cold parts below T_S - 20 K; capacity C in {0, 0.5, 1, 2, 5 MWh}; loss 0; no power limit;
    cyclic). Electricity value per step c_el,t = mean x (1 + A shape(h)) x week(d), shape = daily
    pattern of mean 0 and minimum -1 (morning peak, solar trough at midday, evening peak;
    placeholder), A = depth of the trough in fractions of the mean, weekend 10 % below weekdays
    (week horizon; band 0-20 %). Realistic amplitude bands: price 0-0.6, CO2 factor 0-0.4.
    Objective J = sum dt (c_U U_t + c_el,t Q_t / COP); J_none = without pump and storage;
    J_pump = pump alone (C = 0, best output per step).
    Methods: FORESIGHT = linear programme with perfect foresight over the whole horizon (24 h for
    the day horizon, 1 week for the week horizon), cyclic (HiGHS); in the week horizon also the
    24 h foresight variant "daily state-neutral" (content equal at every midnight). RULES without
    foresight (same storage, same levels, cyclic steady state, at most 30 cycles): per step the
    exact optimum of c_U U + p_t Q - v_t (q_c - q_d) with a value v_t of stored heat from the
    present and past only: ``myopic`` (v = 0+, stores only free heat, the rule of Befund 99),
    ``gas_value`` (v = c_U-, fills whenever pump heat is cheaper than gas heat), ``trailing``
    (v_t = min(c_U, mean c_el of the past 24 h / COP)). The RULE value of a cell is the best of
    the three (the most favourable no-foresight reading, conservative against L1). Best level per
    method and capacity.
    Design pair: the Stage-A best pair for (scenario, size, criterion) at quality 0.50 under the
    constant (mean) signal; if no pair saves anything there, the Stage-A best pair at the trough
    value mean x (1 - A); if none either, the run is recorded with the pump idle.
    Main grid B: site {A, A+C} x day {1, 2} x (horizon, mode) {(day, flat), (day, shift),
    (week, flat), (week, shift), (week, weekly)} x size {500, 1000 kW} x criterion {CO2 today,
    CO2 2030, cost today (0.16 / 0.040 / 55), cost 150, cost 300} x amplitude A {0, 0.1, 0.2,
    0.3, 0.4, 0.6, 0.8, 1.0}; weather 0 K, compressor 800 kW, hall 200 kW, water 200 kW, exhaust C
    by volume = 1600 runs.
    GATE (pre-registered). A CELL is (run, capacity C in {0.5, 1, 2, 5 MWh}); the rule gap
    D = (J_rule - J_foresight) / J_none and the storage gain G = (J_pump - J_foresight) / J_none.
    Only REALISTIC runs count: all stream placeholders inside the bands of Befund 99, all
    heat-pump, price and factor placeholders inside the bands of ``heatpump.config`` and the
    amplitude inside its band (price <= 0.6, CO2 <= 0.4); runs with E_direct < 100 kWh/d are
    excluded (their number is reported).
    L1  RL / MPC set-up justified: some cell has D >= 2 percentage points (CO2 or cost).
    L2  storage with the pump pays, a rule suffices: no cell with D >= 2 points, but some cell
        has G >= 5 %.
    L3  no relevant potential: no cell with D >= 2 points and no cell with G >= 5 %.
    The letter is a statement about the BEST case over the bands and must not be quoted alone.
    Always reported with it: numbers and shares of cells with D >= 2 points and with G >= 5 %;
    readings per criterion, per horizon, per site and per product day; the map max D over
    (amplitude, capacity) for every amplitude of the grid (the ones outside the band marked) and
    the smallest amplitude at which D reaches 2 points per capacity ("from which signal swing
    and storage size foresight becomes valuable"); the value of weekly over daily foresight
    (daily state-neutral minus week programme); the share of non-converged rule runs.
    Sensitivity (one at a time around RB1 = A+C, product 1, day, shift, 1000 kW, cost today,
    A = 0.4 and RB2 = A+C, product 1, week, weekly, 1000 kW, cost today, A = 0.4): loss 0.5 %/h;
    storage approach dT_min/2; quality 0.45 / 0.55; size 500 / 1500 kW; electricity price 0.12 /
    0.20; gas price 0.030 / 0.060; heater efficiency 0.75 / 0.95; product day 2; amplitude 0.2 /
    0.6; weekend reduction 0 / 0.2 (RB2 only); criterion cost 300; site A; ranked by the change of
    D and G at C = 2 MWh (points).

Checks (tests and recorded): the pseudo-streams equal the problem table with explicit isothermal
streams; the pinch rule dU = Q / W / 0 on a hand case; the step dispatch equals a brute-force
search; the exact step optimum of the rules equals HiGHS; the programme with C = 0 equals the step
dispatch; J_foresight non-increasing in C; rules never below the programme; daily state-neutral
never below the week programme; storage balance closed.

Call::

    TWINLINK_PLANT_DATA_ACE=... TWINLINK_PLANT_DATA_RID=... \
        tools/run_here.sh scripts/gate_heat_pump.py --name gate_heat_pump
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (  # type: ignore[import-untyped]
    NO_HEAT_NETWORK,
    REPO_ROOT,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.heatpump import (
    ELECTRICITY_CO2_2030_KG_PER_KWH,
    HEAT_PUMP_SIZES_KW,
    PLACEHOLDER_BANDS,
    SITE_CO2_T_PER_A,
    Capital,
    Criterion,
    DesignPair,
    Economics,
    HeatPumpSpec,
    Signal,
    co2_criterion,
    cost_criterion,
    design_pairs,
    evaluate_stage_a,
    evaluate_stage_b,
    grand_composite_kw,
    is_realistic,
    pinch_summary,
    pseudo_stream_points,
    site_cascade,
)
from twinlink_ml.heatpump.gate import (
    cells_of,
    foresight_map,
    gate_verdict,
    smallest_amplitude,
    subset_verdicts,
)
from twinlink_ml.storage_timeslice import PLACEHOLDER_BANDS as STREAM_BANDS
from twinlink_ml.storage_timeslice import (
    MeasuredDay,
    Placeholders,
    Scenario,
    StreamFactory,
    load_measured_day,
)
from twinlink_ml.storage_timeslice.config import is_realistic as streams_realistic

PRODUCT_ENV = {"product_1": "TWINLINK_PLANT_DATA_ACE", "product_2": "TWINLINK_PLANT_DATA_RID"}
ROLE_FILE = REPO_ROOT / "conf" / "plant" / "plant_local.yaml"
DAYS_PER_YEAR = 365.0
QUALITIES = (0.45, 0.50, 0.55)
SIZES_A = HEAT_PUMP_SIZES_KW
SIZES_B = (500.0, 1000.0)
AMPLITUDES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
CAPACITIES_B = (0.0, 500.0, 1000.0, 2000.0, 5000.0)
LEVELS_B = (40.0, 50.0, 60.0, 70.0)
SENS_SIZE_KW = 500.0
SENS_CAPACITY_B_KWH = 2000.0
GCC_GRID = tuple(float(t) for t in range(-10, 101))
CENTRAL_A = ("co2_today", "co2_2030", "cost_today", "cost_150", "cost_300")
SENS_CRITERIA = ("co2_today", "co2_2030", "cost_today", "cost_300")


# --------------------------------------------------------------------------------- criteria


def economics_of(name: str, base: Economics | None = None) -> Economics:
    """Economics behind a named criterion (central prices unless the name says otherwise)."""
    eco = Economics() if base is None else base
    if name == "co2_today":
        return eco
    if name == "co2_2030":
        return replace(eco, electricity_co2_kg_per_kwh=ELECTRICITY_CO2_2030_KG_PER_KWH)
    if name == "cost_today":
        return replace(eco, co2_eur_per_t=55.0)
    if name == "cost_150":
        return replace(eco, co2_eur_per_t=150.0)
    if name == "cost_300":
        return replace(eco, co2_eur_per_t=300.0)
    if name.startswith("cost|"):
        _, pel, pgas, co2 = name.split("|")
        return replace(
            eco,
            electricity_eur_per_kwh=float(pel),
            gas_eur_per_kwh=float(pgas),
            co2_eur_per_t=float(co2),
        )
    raise ValueError(f"unknown criterion {name!r}")


def criterion_of(name: str, base: Economics | None = None) -> Criterion:
    eco = economics_of(name, base)
    if name.startswith("co2"):
        return co2_criterion(name, eco)
    return cost_criterion(name, eco)


def criteria_names_a() -> list[str]:
    names = list(CENTRAL_A)
    for pel, pgas, co2 in product((0.12, 0.16, 0.20), (0.030, 0.040, 0.060), (55, 150, 300)):
        names.append(f"cost|{pel:g}|{pgas:g}|{co2:g}")
    return names


def kind_of(name: str) -> str:
    return "co2" if name.startswith("co2") else "cost"


# -------------------------------------------------------------------------------- scenarios


def scenario(
    site: str,
    day: str,
    mode: str,
    *,
    horizon: str = "day",
    shift: float = 0.0,
    comp: float = 800.0,
    hall: float = 200.0,
    water: float = 200.0,
    basis: str = "volume",
    loss: float = 0.0,
    **ph: Any,
) -> Scenario:
    return Scenario(
        site=site,  # type: ignore[arg-type]
        day=day,
        horizon=horizon,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        ambient_shift_k=shift,
        compressor_a_kw=comp,
        hall_kw=hall,
        water_kw=water,
        loss_per_h=loss,
        placeholders=replace(Placeholders(exhaust_c_basis=basis), **ph),  # type: ignore[arg-type]
    )


def main_grid_a() -> list[tuple[str, Scenario]]:
    out: list[tuple[str, Scenario]] = []
    for site, day, mode, shift, hall, water in product(
        ("A", "C", "A+C"),
        ("product_1", "product_2"),
        ("flat", "shift"),
        (-10.0, 0.0, 5.0),
        (0.0, 200.0),
        (0.0, 200.0),
    ):
        comps = (800.0,) if site == "C" else (400.0, 800.0, 1200.0)
        bases = ("volume",) if site == "A" else ("volume", "annual")
        for comp, basis in product(comps, bases):
            s = scenario(
                site, day, mode, shift=shift, comp=comp, hall=hall, water=water, basis=basis
            )
            out.append((f"A|{s.label}|{basis}", s))
    return out


def references_a() -> dict[str, Scenario]:
    ra1 = scenario("A+C", "product_1", "flat")
    return {
        "RA1": ra1,
        "RA2": scenario("A+C", "product_1", "flat", basis="annual"),
        "RA3": scenario("A", "product_1", "flat", hall=0.0, water=0.0),
        "RA4": scenario("A", "product_2", "flat", hall=0.0, water=0.0),
    }


def sensitivity_a(base: Scenario) -> dict[str, tuple[Scenario, HeatPumpSpec, Economics]]:
    """One placeholder at a time (pre-registered list)."""
    spec, eco = HeatPumpSpec(), Economics()
    v: dict[str, tuple[Scenario, HeatPumpSpec, Economics]] = {}

    def sp(name: str, **kw: Any) -> None:
        v[name] = (base, replace(spec, **kw), eco)

    def ec(name: str, **kw: Any) -> None:
        v[name] = (base, spec, replace(eco, **kw))

    def ph(name: str, **kw: Any) -> None:
        v[name] = (base.with_placeholders(**kw), spec, eco)

    def sc(name: str, **kw: Any) -> None:
        v[name] = (replace(base, **kw), spec, eco)

    sp("quality=0.45", quality=0.45)
    sp("quality=0.55", quality=0.55)
    sp("max_lift=50", max_lift_k=50.0)
    sp("max_lift=70", max_lift_k=70.0)
    sp("min_lift=15", min_lift_k=15.0)
    sp("min_lift=30", min_lift_k=30.0)
    sp("max_condensing=80", max_condensing_celsius=80.0)
    sp("max_condensing=100", max_condensing_celsius=100.0)
    sp("cop_max=6", cop_max=6.0)
    sp("cop_max=10", cop_max=10.0)
    sp("hp_approach=5", approach_k=5.0)
    sp("hp_approach=20", approach_k=20.0)
    ph("dT_min=10", minimum_pinch_k=10.0)
    ph("dT_min=30", minimum_pinch_k=30.0)
    ph("T_kalt=10", cold_temperature_celsius=10.0)
    ph("T_kalt=25", cold_temperature_celsius=25.0)
    if base.site != "C":
        sc("compressor_a=400", compressor_a_kw=400.0)
        sc("compressor_a=1200", compressor_a_kw=1200.0)
    ph("compressor_T=80-50", compressor_hot_in_celsius=80.0, compressor_hot_out_celsius=50.0)
    ph("compressor_T=75-45", compressor_hot_in_celsius=75.0, compressor_hot_out_celsius=45.0)
    ph("compressor_T=90-50", compressor_hot_in_celsius=90.0, compressor_hot_out_celsius=50.0)
    ph("solid_fraction=0.2", solid_fraction=0.2)
    ph("solid_fraction=0.5", solid_fraction=0.5)
    ph("humidity=0.5", relative_humidity=0.5)
    ph("humidity=0.9", relative_humidity=0.9)
    if base.site != "A":
        ph("exhaust_c=annual", exhaust_c_basis="annual")
        ph("line_flow=15000", c_line_nm3_per_h=15000.0)
        ph("line_flow=25000", c_line_nm3_per_h=25000.0)
        ph("compressor_c=1.2", compressor_c_gwh_per_a=1.2)
        ph("compressor_c=3.6", compressor_c_gwh_per_a=3.6)
    for kw in (0.0, 100.0):
        if kw != base.hall_kw:
            sc(f"hall={kw:g}", hall_kw=kw)
        if kw != base.water_kw:
            sc(f"water={kw:g}", water_kw=kw)
    sc("weather=-10", ambient_shift_k=-10.0)
    sc("weather=+5", ambient_shift_k=5.0)
    sc(
        "day=product_2" if base.day == "product_1" else "day=product_1",
        day="product_2" if base.day == "product_1" else "product_1",
    )
    sc("mode=shift", mode="shift")
    ec("heater_efficiency=0.75", heater_efficiency=0.75)
    ec("heater_efficiency=0.95", heater_efficiency=0.95)
    ec("electricity_price=0.12", electricity_eur_per_kwh=0.12)
    ec("electricity_price=0.20", electricity_eur_per_kwh=0.20)
    ec("gas_price=0.030", gas_eur_per_kwh=0.030)
    ec("gas_price=0.060", gas_eur_per_kwh=0.060)
    return v


def main_grid_b() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for site, day, (horizon, mode), size, crit, amp in product(
        ("A", "A+C"),
        ("product_1", "product_2"),
        (
            ("day", "flat"),
            ("day", "shift"),
            ("week", "flat"),
            ("week", "shift"),
            ("week", "weekly"),
        ),
        SIZES_B,
        CENTRAL_A,
        AMPLITUDES,
    ):
        s = scenario(site, day, mode, horizon=horizon)
        out.append(
            {
                "key": f"B|{s.label}|{size:g}|{crit}|A{amp:g}",
                "scenario": s,
                "size": size,
                "criterion": crit,
                "signal": Signal(amplitude=amp),
                "spec": HeatPumpSpec(),
                "economics": None,
            }
        )
    return out


def references_b() -> dict[str, dict[str, Any]]:
    common = {
        "size": 1000.0,
        "criterion": "cost_today",
        "signal": Signal(amplitude=0.4),
        "spec": HeatPumpSpec(),
        "economics": None,
    }
    return {
        "RB1": {"scenario": scenario("A+C", "product_1", "shift"), **common},
        "RB2": {"scenario": scenario("A+C", "product_1", "weekly", horizon="week"), **common},
    }


def sensitivity_b(rid: str, base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    v: dict[str, dict[str, Any]] = {}
    s: Scenario = base["scenario"]

    def put(name: str, **changes: Any) -> None:
        v[name] = {**base, **changes}

    put("loss=0.005", scenario=replace(s, loss_per_h=0.005))
    put(
        "storage_approach=dT_min/2",
        scenario=s.with_placeholders(storage_approach_k=s.placeholders.minimum_pinch_k / 2.0),
    )
    put("quality=0.45", spec=replace(base["spec"], quality=0.45))
    put("quality=0.55", spec=replace(base["spec"], quality=0.55))
    put("size=500", size=500.0)
    put("size=1500", size=1500.0)
    put("electricity_price=0.12", economics=replace(Economics(), electricity_eur_per_kwh=0.12))
    put("electricity_price=0.20", economics=replace(Economics(), electricity_eur_per_kwh=0.20))
    put("gas_price=0.030", economics=replace(Economics(), gas_eur_per_kwh=0.030))
    put("gas_price=0.060", economics=replace(Economics(), gas_eur_per_kwh=0.060))
    put("heater_efficiency=0.75", economics=replace(Economics(), heater_efficiency=0.75))
    put("heater_efficiency=0.95", economics=replace(Economics(), heater_efficiency=0.95))
    put("day=product_2", scenario=replace(s, day="product_2"))
    put("amplitude=0.2", signal=replace(base["signal"], amplitude=0.2))
    put("amplitude=0.6", signal=replace(base["signal"], amplitude=0.6))
    if rid == "RB2":
        put("weekend=0", signal=replace(base["signal"], weekend_reduction=0.0))
        put("weekend=0.2", signal=replace(base["signal"], weekend_reduction=0.2))
    put("criterion=cost_300", criterion="cost_300")
    put("site=A", scenario=replace(s, site="A"))
    return v


# ---------------------------------------------------------------------------------- workers

_FACTORY: StreamFactory | None = None


def _init_worker(days: dict[str, MeasuredDay]) -> None:
    global _FACTORY
    _FACTORY = StreamFactory(days)


def _factory() -> StreamFactory:
    assert _FACTORY is not None
    return _FACTORY


def _rounded(x: Any) -> Any:
    """Round floats to 7 significant digits for compact records."""
    if isinstance(x, float):
        if not math.isfinite(x) or x == 0.0:
            return x
        return float(f"{x:.7g}")
    if isinstance(x, dict):
        return {k: _rounded(val) for k, val in x.items()}
    if isinstance(x, list | tuple):
        return [_rounded(val) for val in x]
    if isinstance(x, np.ndarray):
        return _rounded(x.tolist())
    if isinstance(x, np.generic):
        return _rounded(x.item())
    return x


def _label(pair: DesignPair | None) -> str | None:
    return None if pair is None else pair.label


def stage_a_record(
    scen: Scenario,
    spec: HeatPumpSpec,
    names: Sequence[str],
    sizes: Sequence[float],
    full: Sequence[str],
    keep_pair_table: bool = False,
    base_eco: Economics | None = None,
) -> dict[str, Any]:
    bank = _factory().build(scen)
    crits = [criterion_of(n, base_eco) for n in names]
    res = evaluate_stage_a(
        bank, scen, spec=spec, criteria=crits, sizes_kw=sizes, keep_pair_table=keep_pair_table
    )
    idx_full = [names.index(n) for n in full]
    rec: dict[str, Any] = {
        "direct_kwh_per_day": res.direct_kwh_per_day,
        "pinch": res.pinch,
        "sizes_kw": list(res.sizes_kw),
        "criteria": list(names),
        "objective_direct_per_day": res.objective_direct_per_day[0].tolist(),
        "objective_saving_per_day": res.objective_saving_per_day.tolist(),
        "full": {},
        "notes": dict(bank.notes),
    }
    for j in idx_full:
        name = names[j]
        rec["full"][name] = {
            "pair": [_label(p[j]) for p in res.best_pair],
            "cop": [None if (pj := p[j]) is None else pj.cop for p in res.best_pair],
            "heat_kwh_per_day": res.heat_kwh_per_day[:, j].tolist(),
            "utility_saving_kwh_per_day": res.utility_saving_kwh_per_day[:, j].tolist(),
            "work_kwh_per_day": res.work_kwh_per_day[:, j].tolist(),
            "across_share": res.across_share[:, j].tolist(),
        }
    if keep_pair_table:
        rec["pair_table"] = res.pair_table
    return rec


def _task_a(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    kind = task["kind"]
    scen: Scenario = task["scenario"]
    if kind == "grid":
        out = {
            "meta": task["meta"],
            "realistic_streams": streams_realistic(scen),
            "qualities": {},
        }
        for q in QUALITIES:
            out["qualities"][f"{q:g}"] = stage_a_record(
                scen, HeatPumpSpec(quality=q), task["names"], SIZES_A, CENTRAL_A
            )
        return task["key"], _rounded(out)
    if kind == "reference":
        return task["key"], _rounded(reference_a(scen))
    if kind == "sens":
        spec, eco = task["spec"], task["economics"]
        rec = stage_a_record(
            scen, spec, list(SENS_CRITERIA), (SENS_SIZE_KW,), SENS_CRITERIA, base_eco=eco
        )
        rec["realistic"] = bool(
            streams_realistic(scen)
            and is_realistic(
                spec=spec, economics=eco, minimum_pinch_k=scen.placeholders.minimum_pinch_k
            )
        )
        return task["key"], _rounded(rec)
    raise ValueError(kind)


def reference_a(scen: Scenario) -> dict[str, Any]:
    """Pinch calculation, pinch-rule table, break-even and quality sweeps of a reference."""
    bank = _factory().build(scen)
    spec = HeatPumpSpec()
    pairs = design_pairs(spec, minimum_pinch_k=scen.placeholders.minimum_pinch_k)
    cascade = site_cascade(bank, scen, pseudo_stream_points(pairs))
    summ = pinch_summary(cascade)
    gcc = grand_composite_kw(cascade, np.asarray(GCC_GRID))
    out: dict[str, Any] = {"label": scen.label}
    out["pinch"] = {
        "hot_utility_kw_mean": float(np.mean(summ.hot_utility_kw)),
        "cold_utility_kw_mean": float(np.mean(summ.cold_utility_kw)),
        "pinch_high_shifted_per_step": summ.pinch_high.tolist(),
        "pinch_low_shifted_per_step": summ.pinch_low.tolist(),
        "gcc_shifted_celsius": list(GCC_GRID),
        "gcc_mean_kw": np.mean(gcc, axis=0).tolist(),
        "gcc_min_kw": np.min(gcc, axis=0).tolist(),
        "gcc_max_kw": np.max(gcc, axis=0).tolist(),
    }
    names = list(CENTRAL_A)
    out["central"] = stage_a_record(scen, spec, names, SIZES_A, CENTRAL_A, keep_pair_table=True)
    # break-even sweeps (quality 0.50, sizes 500 and 1000 kW)
    eco = Economics()
    fel = [round(0.02 * i, 2) for i in range(51)]
    co2_crits = [Criterion(f"fel={f:g}", "co2", eco.gas_co2_per_kwh_heat, f) for f in fel]
    ratios = [round(0.5 + 0.1 * i, 2) for i in range(76)]
    ratio_crits = [Criterion(f"ratio={r:g}", "cost", 1.0, r) for r in ratios]
    sweep: dict[str, Any] = {}
    for tag, crits, axis in (("co2_factor", co2_crits, fel), ("value_ratio", ratio_crits, ratios)):
        res = evaluate_stage_a(bank, scen, spec=spec, criteria=crits, sizes_kw=(500.0, 1000.0))
        sweep[tag] = {
            "axis": axis,
            "saving_per_day": res.objective_saving_per_day.tolist(),
            "utility_saving_kwh_per_day": res.utility_saving_kwh_per_day.tolist(),
            "work_kwh_per_day": res.work_kwh_per_day.tolist(),
            "breakeven": [
                max((a for a, s in zip(axis, row, strict=True) if s > 1e-6), default=None)
                for row in res.objective_saving_per_day
            ],
        }
    q_axis = [round(0.30 + 0.05 * i, 2) for i in range(9)]
    q_rows: list[dict[str, Any]] = []
    for q in q_axis:
        res = evaluate_stage_a(
            bank,
            scen,
            spec=HeatPumpSpec(quality=q),
            criteria=[criterion_of("co2_today"), criterion_of("cost_today")],
            sizes_kw=(500.0, 1000.0),
        )
        q_rows.append(
            {
                "quality": q,
                "saving_per_day": res.objective_saving_per_day.tolist(),
                "cop": [[None if p is None else p.cop for p in row] for row in res.best_pair],
                "pair": [[None if p is None else p.label for p in row] for row in res.best_pair],
            }
        )
    sweep["quality"] = q_rows
    out["sweeps"] = sweep
    return out


def _stage_b_pair(
    scen: Scenario, spec: HeatPumpSpec, crit: Criterion, size: float, signal: Signal
) -> tuple[DesignPair | None, str]:
    bank = _factory().build(scen)
    flat = evaluate_stage_a(bank, scen, spec=spec, criteria=[crit], sizes_kw=(size,))
    if flat.best_pair[0][0] is not None:
        return flat.best_pair[0][0], "mean"
    trough = Criterion(
        crit.name,
        crit.kind,
        crit.heat_value_per_kwh,
        crit.electricity_value_per_kwh * (1.0 - signal.amplitude),
    )
    low = evaluate_stage_a(bank, scen, spec=spec, criteria=[trough], sizes_kw=(size,))
    return low.best_pair[0][0], "trough" if low.best_pair[0][0] is not None else "idle"


def _task_b(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    scen: Scenario = task["scenario"]
    spec: HeatPumpSpec = task["spec"]
    eco = task["economics"]
    crit = criterion_of(task["criterion"], eco)
    eco_used = economics_of(task["criterion"], eco)
    signal: Signal = task["signal"]
    size = float(task["size"])
    pair, how = _stage_b_pair(scen, spec, crit, size, signal)
    realistic = bool(
        streams_realistic(scen)
        and is_realistic(
            spec=spec,
            economics=eco_used,
            minimum_pinch_k=scen.placeholders.minimum_pinch_k,
            criterion_kind=crit.kind,
            signal=signal,
        )
    )
    meta = {
        "site": scen.site,
        "day": scen.day,
        "horizon": scen.horizon,
        "mode": scen.mode,
        "size_kw": size,
        "criterion": crit.name,
        "kind": crit.kind,
        "amplitude": signal.amplitude,
        "weekend_reduction": signal.weekend_reduction,
    }
    if pair is None:
        # pump idle: storage alone (heat pump at zero size, any admissible pair)
        pair = design_pairs(spec, minimum_pinch_k=scen.placeholders.minimum_pinch_k)[0]
        size = 0.0
    res = evaluate_stage_b(
        _factory().build(scen),
        scen,
        criterion=crit,
        signal=signal,
        size_kw=size,
        pair=pair,
        levels_celsius=LEVELS_B,
        capacities_kwh=CAPACITIES_B,
    )
    rec = res.to_dict()
    weekly = None
    if res.objective_daily_neutral_per_day is not None:
        weekly = (
            (res.objective_daily_neutral_per_day - res.objective_foresight_per_day)
            / res.objective_none_per_day
        ).tolist()
    rec.update(
        {
            "key": task["key"],
            "meta": meta,
            "realistic": realistic,
            "pair_chosen_by": how,
            "weekly_value_fraction": weekly,
        }
    )
    return task["key"], _rounded(rec)


def _run(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if task["stage"] == "A":
        return _task_a(task)
    return _task_b(task)


def run_tasks(
    tasks: Sequence[dict[str, Any]], days: dict[str, MeasuredDay], workers: int
) -> dict[str, dict[str, Any]]:
    keys = [t["key"] for t in tasks]
    if len(set(keys)) != len(keys):
        raise ValueError("task keys must be unique")
    out: dict[str, dict[str, Any]] = {}
    started = time.time()
    if workers <= 1:
        _init_worker(days)
        for i, task in enumerate(tasks, 1):
            k, rec = _run(task)
            out[k] = rec
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} ({time.time() - started:.0f}s)", flush=True)
        return out
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(days,)
    ) as pool:
        for i, (k, rec) in enumerate(pool.map(_run, tasks, chunksize=2), 1):
            out[k] = rec
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    return out


# ------------------------------------------------------------------------------- data, summary


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
    days: dict[str, MeasuredDay] = {}
    hashes: dict[str, str] = {}
    for label, env in PRODUCT_ENV.items():
        path = os.environ[env]
        hashes[label] = sha256_of(path)
        days[label] = load_measured_day(path, ROLE_FILE, label)
    return days, hashes


def annual(per_day: float) -> float:
    """kg/d or EUR/d -> t/a or kEUR/a."""
    return per_day * DAYS_PER_YEAR / 1000.0


def summarise_a(grid: dict[str, dict[str, Any]], names: list[str]) -> dict[str, Any]:
    """Distribution of the best saving over the main grid per criterion, quality and size."""
    out: dict[str, Any] = {}
    for q in QUALITIES:
        qk = f"{q:g}"
        per_crit: dict[str, Any] = {}
        for j, name in enumerate(CENTRAL_A):
            jj = names.index(name)
            rows = []
            for i, size in enumerate(SIZES_A):
                vals = np.array(
                    [
                        rec["qualities"][qk]["objective_saving_per_day"][i][jj]
                        for rec in grid.values()
                    ]
                )
                pos = vals > 1e-6
                rows.append(
                    {
                        "size_kw": size,
                        "share_scenarios_saving_positive": float(np.mean(pos)),
                        "median_annual": annual(float(np.median(vals))),
                        "max_annual": annual(float(np.max(vals))),
                        "min_annual": annual(float(np.min(vals))),
                    }
                )
            per_crit[name] = rows
            del j
        out[qk] = per_crit
    # cost grid: share of scenarios with a positive saving at 1000 kW, quality 0.50
    i1000 = SIZES_A.index(1000.0)
    cost_grid = {}
    for name in names:
        if not name.startswith("cost|"):
            continue
        jj = names.index(name)
        vals = np.array(
            [
                rec["qualities"]["0.5"]["objective_saving_per_day"][i1000][jj]
                for rec in grid.values()
            ]
        )
        cost_grid[name] = {
            "share_positive": float(np.mean(vals > 1e-6)),
            "median_annual_keur": annual(float(np.median(vals))),
            "max_annual_keur": annual(float(np.max(vals))),
        }
    out["cost_grid_1000kW_quality_0.5"] = cost_grid
    return out


def capital_table(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Payback of the central cost criteria for a reference record (quality 0.50)."""
    rows = []
    for name in ("cost_today", "cost_150", "cost_300"):
        j = rec["criteria"].index(name)
        for i, size in enumerate(rec["sizes_kw"]):
            if size <= 0.0:
                continue
            saving_a = rec["objective_saving_per_day"][i][j] * DAYS_PER_YEAR
            row = {"criterion": name, "size_kw": size, "operating_saving_eur_per_a": saving_a}
            for capex in (400.0, 600.0, 1000.0):
                cap = Capital(capex_eur_per_kw=capex)
                row[f"net_eur_per_a|capex{capex:g}"] = saving_a - cap.annual_cost_eur(size)
                row[f"simple_payback_a|capex{capex:g}"] = (
                    size * capex / saving_a if saving_a > 1e-6 else None
                )
            rows.append(row)
    return rows


def rank(rows: dict[str, dict[str, Any]], base: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Sensitivity ranking per criterion by |change of the annual saving|."""
    out: dict[str, list[dict[str, Any]]] = {}
    for name in SENS_CRITERIA:
        j = base["criteria"].index(name)
        b = annual(base["objective_saving_per_day"][0][j])
        lst = []
        for var, rec in rows.items():
            val = annual(rec["objective_saving_per_day"][0][j])
            lst.append(
                {
                    "variant": var,
                    "annual_saving": val,
                    "delta_annual_saving": val - b,
                    "pair": rec["full"][name]["pair"][0],
                    "cop": rec["full"][name]["cop"][0],
                    "realistic": rec["realistic"],
                }
            )
        lst.sort(key=lambda r: -abs(r["delta_annual_saving"]))
        out[name] = lst
    return out


def rank_b(rows: dict[str, dict[str, Any]], base: dict[str, Any]) -> list[dict[str, Any]]:
    i = base["capacities_kwh"].index(SENS_CAPACITY_B_KWH)
    lst = []
    for var, rec in rows.items():
        ii = rec["capacities_kwh"].index(SENS_CAPACITY_B_KWH)
        lst.append(
            {
                "variant": var,
                "gap_points": 100.0 * rec["gap_fraction"][ii],
                "delta_gap_points": 100.0 * (rec["gap_fraction"][ii] - base["gap_fraction"][i]),
                "gain_points": 100.0 * rec["storage_gain_fraction"][ii],
                "delta_gain_points": 100.0
                * (rec["storage_gain_fraction"][ii] - base["storage_gain_fraction"][i]),
                "pair": rec["pair"],
                "realistic": rec["realistic"],
            }
        )
    lst.sort(key=lambda r: -max(abs(r["delta_gap_points"]), abs(r["delta_gain_points"])))
    return lst


def check_summary_b(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failing: dict[str, list[str]] = {}
    for key, rec in records.items():
        for name, value in rec["checks"].items():
            if isinstance(value, bool) and not value:
                failing.setdefault(name, []).append(key)
    return {
        "n_runs": len(records),
        "n_failing": {k: len(v) for k, v in failing.items()},
        "failing_examples": {k: v[:10] for k, v in failing.items()},
        "max_storage_balance_residual_kwh": max(
            (r["checks"]["max_storage_balance_residual_kwh"] for r in records.values()), default=0.0
        ),
        "pair_chosen_by": {
            how: sum(1 for r in records.values() if r["pair_chosen_by"] == how)
            for how in ("mean", "trough", "idle")
        },
    }


def gate_b(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    recs = list(records.values())
    cells, counts = cells_of(recs)
    groupings: dict[str, Callable[[Any], object]] = {
        "criterion": lambda c: c.meta["criterion"],
        "kind": lambda c: c.meta["kind"],
        "horizon": lambda c: c.meta["horizon"],
        "mode": lambda c: c.meta["mode"],
        "site": lambda c: c.meta["site"],
        "day": lambda c: c.meta["day"],
        "size_kw": lambda c: c.meta["size_kw"],
        "capacity_kwh": lambda c: c.capacity_kwh,
    }
    by_kind = {
        "co2": lambda c: str(c.meta["kind"]),
        "criterion": lambda c: str(c.meta["criterion"]),
    }
    return {
        "verdict": gate_verdict(cells, counts),
        "per_group": subset_verdicts(cells, groupings),
        "foresight_map_by_kind": foresight_map(cells, by_kind["co2"]),
        "foresight_map_by_criterion": foresight_map(cells, by_kind["criterion"]),
        "smallest_amplitude_gap_2_points_by_kind": smallest_amplitude(cells, by_kind["co2"]),
        "smallest_amplitude_gap_2_points_by_criterion": smallest_amplitude(
            cells, by_kind["criterion"]
        ),
        "smallest_amplitude_gap_1_point_by_kind": smallest_amplitude(
            cells, by_kind["co2"], threshold=0.01
        ),
        "amplitude_bands": {
            "price": PLACEHOLDER_BANDS["price_amplitude"],
            "co2": PLACEHOLDER_BANDS["co2_amplitude"],
        },
    }


# ------------------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--name", required=True, help="artifact name (never overwritten)")
    parser.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0, help="debug: N tasks per part, no artifact")
    args = parser.parse_args()
    if not args.limit:
        require_clean_tree()
    started = time.time()
    days, hashes = load_days()
    print("days loaded", flush=True)
    names = criteria_names_a()

    tasks: list[dict[str, Any]] = []
    grid_a = main_grid_a()
    for key, s in grid_a[: args.limit or None]:
        meta = {
            "site": s.site,
            "day": s.day,
            "mode": s.mode,
            "ambient_shift_k": s.ambient_shift_k,
            "compressor_a_kw": s.compressor_a_kw,
            "hall_kw": s.hall_kw,
            "water_kw": s.water_kw,
            "exhaust_c_basis": s.placeholders.exhaust_c_basis,
        }
        tasks.append(
            {"stage": "A", "kind": "grid", "key": key, "scenario": s, "names": names, "meta": meta}
        )
    refs_a = references_a()
    for rid, s in refs_a.items():
        tasks.append({"stage": "A", "kind": "reference", "key": f"ref|{rid}", "scenario": s})
    sens_a_keys: dict[str, list[str]] = {}
    for rid in ("RA1", "RA3"):
        base = refs_a[rid]
        tasks.append(
            {
                "stage": "A",
                "kind": "sens",
                "key": f"sensA|{rid}|base",
                "scenario": base,
                "spec": HeatPumpSpec(),
                "economics": Economics(),
            }
        )
        sens_a_keys[rid] = []
        for var, (s, spec, eco) in sensitivity_a(base).items():
            k = f"sensA|{rid}|{var}"
            sens_a_keys[rid].append(k)
            tasks.append(
                {
                    "stage": "A",
                    "kind": "sens",
                    "key": k,
                    "scenario": s,
                    "spec": spec,
                    "economics": eco,
                }
            )
    grid_b = main_grid_b()[: args.limit or None]
    for t in grid_b:
        tasks.append({"stage": "B", **t})
    refs_b = references_b()
    sens_b_keys: dict[str, list[str]] = {}
    for rid, base_b in refs_b.items():
        tasks.append({"stage": "B", "key": f"sensB|{rid}|base", **base_b})
        sens_b_keys[rid] = []
        for var, t_b in sensitivity_b(rid, base_b).items():
            k = f"sensB|{rid}|{var}"
            sens_b_keys[rid].append(k)
            tasks.append({"stage": "B", "key": k, **t_b})
    # longest first (week horizon of Stage B), for the pool
    tasks.sort(key=lambda t: 0 if t["stage"] == "B" and t["scenario"].horizon == "week" else 1)
    print(f"{len(tasks)} tasks", flush=True)
    done = run_tasks(tasks, days, args.workers)
    print(f"all tasks done ({time.time() - started:.0f}s)", flush=True)

    grid_a_done = {k: done[k] for k, _ in grid_a if k in done}
    stage_b = {t["key"]: done[t["key"]] for t in grid_b}
    gate = gate_b(stage_b)
    print("gate:", gate["verdict"]["reading"], flush=True)

    references = {rid: done[f"ref|{rid}"] for rid in refs_a}
    payload: dict[str, Any] = {
        "_provenance": provenance(NO_HEAT_NETWORK),
        "name": args.name,
        "raw_file_sha256": hashes,
        "placeholders_streams_default": asdict(Placeholders()),
        "placeholder_bands_streams": {k: list(v) for k, v in STREAM_BANDS.items()},
        "placeholders_heat_pump_default": asdict(HeatPumpSpec()),
        "placeholders_economics_default": asdict(Economics()),
        "placeholders_capital_default": asdict(Capital()),
        "placeholder_bands_heat_pump": {k: list(v) for k, v in PLACEHOLDER_BANDS.items()},
        "site_co2_t_per_a": SITE_CO2_T_PER_A,
        "criteria_values": {
            n: {
                "heat_value_per_kwh": criterion_of(n).heat_value_per_kwh,
                "electricity_value_per_kwh": criterion_of(n).electricity_value_per_kwh,
                "breakeven_cop": criterion_of(n).breakeven_cop,
            }
            for n in names
        },
        "stage_a": {
            "summary": summarise_a(grid_a_done, names) if grid_a_done else None,
            "references": references,
            "capital_references": {
                rid: capital_table(rec["central"]) for rid, rec in references.items()
            },
            "sensitivity": {
                rid: {
                    "base": done[f"sensA|{rid}|base"],
                    "ranking": rank(
                        {k.split("|", 2)[2]: done[k] for k in sens_a_keys[rid]},
                        done[f"sensA|{rid}|base"],
                    ),
                }
                for rid in sens_a_keys
            },
            "grid": grid_a_done,
        },
        "stage_b": {
            "gate": gate,
            "checks": check_summary_b(stage_b),
            "sensitivity": {
                rid: {
                    "base": done[f"sensB|{rid}|base"],
                    "ranking": rank_b(
                        {k.split("|", 2)[2]: done[k] for k in sens_b_keys[rid]},
                        done[f"sensB|{rid}|base"],
                    ),
                }
                for rid in sens_b_keys
            },
            "grid": stage_b,
        },
        "runtime_seconds": time.time() - started,
    }
    if args.limit:
        print("debug run (--limit): artifact not written")
        print({k: gate["verdict"][k] for k in ("reading", "max_gap", "max_gain")})
        return 0
    write_artifact(args.name, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
