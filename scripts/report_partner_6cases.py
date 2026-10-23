r"""Six-case network of the partner model with new, asymmetric inputs: Excel, Dyssol, physics, optimum.

docs/11 section 13. Inputs (confidential, never copied, never committed) come from
``$TWINLINK_LOCAL_DIR`` (or ``<root>/local``):

* ``partner/TwinLink Excel.xlsx``: the workbook of the six cases, formulas as ``Model v4.1_2.xlsx``,
  new inputs (dryer 1: 120 kg/h at 300 degC, spray 12 kg/h; dryer 2: 100 kg/h at 150 degC,
  spray 3 kg/h; circuits 40 and 30 kg/h; ambient 20 degC),
* ``partner/Dyssol_Excel simulation_results.xlsx`` (also with a space instead of the underscore):
  the four sheets of ``Comparison.xlsx`` and the sheet ``Simulation`` (six cases, Excel against
  Dyssol),
* ``partner/Model v4.1_2.xlsx`` and ``partner/Comparison.xlsx`` (Befund 106) for the diffs.

The artifact holds

* ``workbook`` -- inputs and formula texts against 4.1_2 and the reproduction of every stored
  number of ``Scenarios``, ``Balances``, ``Balance`` with the switch set of 4.1_2,
* ``results_workbook`` -- whether its four comparison sheets equal ``Comparison.xlsx``, and the
  inputs of the sheet ``Simulation`` against the workbook,
* ``cases`` -- every result of the sheet ``Simulation`` against the partner mode; the index 1/2 of
  the source-indexed rows is assigned to a source by the Excel values (identity or swapped, the one
  within rounding), and the same assignment is used for the Dyssol column,
* ``case_2_hypothesis`` -- the Dyssol column against three assignments of the medium flows (as in
  the workbook, i.e. per exhaust; swapped; per sink), the internal balances of the reported values,
  and the offset of the exhaust of dryer 1 that alone would give Dyssol's exhaust temperature,
* ``validators`` -- checks (a), (b), (c) of :mod:`twinlink_ml.flowsheet.validators` on every case,
* ``physical_mode`` -- counter-current exchangers (10 K per exchanger), saturation limit on, medium
  (i) unpressurised up to 95 degC and (ii) pressurised: gas per case, ranking, distances,
* ``pinch`` -- the least burner duty of any network of circuits with the same exchanger
  assumptions (:mod:`twinlink_ml.flowsheet.pinch_target`, problem table of
  :mod:`twinlink_ml.heatnetwork.pinch` unchanged) and the distance of every case to it.

Readings fixed before the run (so that the numbers cannot choose them):

* reproduction: "exact" = every stored number of the three sheets within ``rel`` 1e-9 (as 106);
* "within rounding" = ``|ours - reported| <= half a unit of the reported last decimal``; the total
  fuel row (a sum of two rounded values) gets the sum of the two halves;
* case-2 hypothesis "confirmed" when, with the medium flows swapped, every Dyssol value of case 2 is
  within rounding or nearer to ours than with the workbook's flows **and** the medium-side balance of
  Dyssol's own reported numbers closes with the swapped flows (within the rounding of the two
  temperatures) and not with the workbook's. Computed twice: over all twelve values (strict) and
  over the ten that depend on the medium flows (without the two product temperatures of the
  dryers); this split was added after a dry run of the script (no artifact), see ``verdict.note``;
* ranking: by total corrected fuel, lowest first; two cases are tied when they differ by less than
  1e-9 kg/h. A "rank change" is any pair whose order differs from the partner mode, including a
  tie that was an order or the reverse.

The artifact is never overwritten (new name per run). Run from a clean working tree, started with
``tools/run_here.sh``.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from report_partner_v41_2 import (  # same directory; the helpers of Befund 106, unchanged
    _compare,
    _deviation,
    _findings,
    _formula_diff,
    _stage_dict,
)

from twinlink_ml.flowsheet import Flowsheet, FlowsheetConfig, ScenarioResult, Switches
from twinlink_ml.flowsheet.dryer import DryerInputs
from twinlink_ml.flowsheet.loop import StageOut
from twinlink_ml.flowsheet.pinch_target import SinkDemand, network_pinch_target
from twinlink_ml.flowsheet.sheet import sheet_scenario_cells
from twinlink_ml.flowsheet.thermo import capacity, partner_props
from twinlink_ml.flowsheet.validators import (
    Finding,
    SaturationProps,
    check_exhaust_above_adiabatic_saturation,
    check_exhaust_saturation,
    check_medium_boiling,
    check_product_mass,
)
from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4_1 import as_built
from twinlink_ml.partner_v4_1 import deviations as dev
from twinlink_ml.partner_v4_1.comparison import Reported, load_comparison, reported_decimals
from twinlink_ml.partner_v4_1.inputs import ExcelInputsV41, SiInputsV41, load_excel_inputs_v41
from twinlink_ml.partner_v4_1.layout import CASES as CASE_SOURCES
from twinlink_ml.partner_v4_1.simulation_sheet import (
    CASES,
    SimulationSheet,
    load_simulation_sheet,
)
from twinlink_ml.utils.code_location import require_code_location

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", ROOT / "local"))
UNPRESSURISED_MAX_CELSIUS = 95.0
DT_MIN_K = 10.0
TIE_KG_PER_H = 1e-9
SWITCHES_412 = Switches(gas_formula="physical")
"""The switch set of 4.1_2 (Befund 106): the corrected gas formula, everything else as built."""
PHYSICAL_SWITCHES = Switches(exchanger="physical", condensation="physical", gas_formula="physical")
"""Counter-current loop with the condensing release curve; ``cp_gas`` stays the workbook's."""
MODES: dict[str, dict[str, Any]] = {
    "partner": {"switches": SWITCHES_412, "saturation_limit": False, "htm_max_celsius": None},
    "physical_unpressurised_95": {
        "switches": PHYSICAL_SWITCHES,
        "saturation_limit": True,
        "htm_max_celsius": UNPRESSURISED_MAX_CELSIUS,
    },
    "physical_pressurised": {
        "switches": PHYSICAL_SWITCHES,
        "saturation_limit": True,
        "htm_max_celsius": None,
    },
}
Assignments = dict[int, dict[str, dict[int, int]]]
"""``case -> quantity -> index of the sheet -> source (or sink, dryer)``."""
SOURCE_KEYS = ("exhaust_out_celsius", "recovered_kw", "medium_hot_celsius", "medium_cold_celsius")
"""Rows of ``Simulation`` that belong to an exhaust (its circuit); the index is assigned by Excel."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def _find(*names: str) -> Path:
    for name in names:
        if (LOCAL / name).exists():
            return LOCAL / name
    raise FileNotFoundError(f"none of {names} in {LOCAL}")


def _flowsheet(si: SiInputsV41, mode: str) -> Flowsheet:
    m = MODES[mode]
    return Flowsheet(
        FlowsheetConfig(
            si,
            m["switches"],
            saturation_limit=m["saturation_limit"],
            htm_max_celsius=m["htm_max_celsius"],
        )
    )


# ---------------------------------------------------------------------- 1. workbook
def workbook_section(previous: Path, new: Path) -> dict[str, Any]:
    prev_in, new_in = load_excel_inputs_v41(previous), load_excel_inputs_v41(new)
    reproduction: dict[str, Any] = {}
    excel = dev.read_formula_cells(new, "Scenarios")
    for label, switches in (
        ("v41_1_all_partner", Switches.all_partner()),
        ("v41_2_gas_formula_corrected", SWITCHES_412),
    ):
        cells = sheet_scenario_cells(Flowsheet(FlowsheetConfig(new_in.to_si(), switches)))
        reproduction[f"scenarios__{label}_switches"] = _compare(cells, excel)
    balances = as_built.balances_cells_as_built(new_in)
    for sheet in ("Balances", "Balance"):
        stored = dev.read_formula_cells(new, sheet)
        reproduction[f"as_built_{sheet}"] = _compare({a: balances[a] for a in stored}, stored)
    return {
        "previous_workbook": previous.name,
        "workbook": new.name,
        "inputs_diff": dev.inputs_diff(previous, new),
        "formula_diff": _formula_diff(previous, new),
        "reference_scan": dev.reference_scan(new),
        "reproduction": reproduction,
        "inputs_sheet_units": {
            "plant": dataclasses.asdict(new_in.plant),
            "htm_flow_kg_per_h": list(new_in.htm_flow_kg_per_h),
        },
        "inputs_sheet_units_previous": {
            "plant": dataclasses.asdict(prev_in.plant),
            "htm_flow_kg_per_h": list(prev_in.htm_flow_kg_per_h),
        },
    }


# ---------------------------------------------------------------------- 2. results workbook
def _expected_simulation_inputs(ex: ExcelInputsV41) -> dict[tuple[int, str, str, str], float]:
    p = ex.plant
    out: dict[tuple[int, str, str, str], float] = {}
    for i, d in ((1, p.dryer_1), (2, p.dryer_2)):
        out |= {
            (i, "Heat recovery", "Fresh gas", "Mass flow rate"): d.air_mass_flow_kg_per_h,
            (i, "Heat recovery", "Fresh gas", "Moisture content"): d.air_moisture_g_per_kg,
            (i, "Heat recovery", "Fresh gas", "Temperature"): p.ambient_temperature_celsius,
            (i, "Heating transfer medium", "", "Flow rate"): ex.htm_flow_kg_per_h[i - 1],
            (i, "Dryer", "Process air", "Mass flow rate"): d.air_mass_flow_kg_per_h,
            (i, "Dryer", "Process air", "Moisture content"): d.air_moisture_g_per_kg,
            (i, "Dryer", "Process air", "Temperature"): d.air_temperature_celsius,
            (i, "Dryer", "Spray rate", "Mass flow rate"): d.spray_mass_flow_kg_per_h,
            (i, "Dryer", "Spray rate", "Solid fraction"): d.spray_solid_fraction_kg_per_kg,
            (i, "Dryer", "Spray rate", "Temperature"): d.spray_temperature_celsius,
            (i, "Heater", "Efficiency", ""): p.material.heater_efficiency_percent,
            (i, "Heater", "Calorific value", ""): p.material.calorific_value_kj_per_kg,
        }
    return out


def results_workbook_section(
    results: Path, comparison: Path, sim: SimulationSheet, ex: ExcelInputsV41
) -> dict[str, Any]:
    ours, theirs = load_comparison(results), load_comparison(comparison)
    expected = _expected_simulation_inputs(ex)
    keys = sorted(set(expected) | set(sim.inputs))
    differing = [
        {"key": "/".join(map(str, k)), "simulation": sim.inputs.get(k), "workbook": expected.get(k)}
        for k in keys
        if sim.inputs.get(k) != expected.get(k)
    ]
    return {
        "workbook": results.name,
        "comparison_sheets_equal_to": comparison.name,
        "comparison_sheets_equal": {name: ours[name] == theirs[name] for name in theirs},
        "simulation_inputs": {"/".join(map(str, k)): v for k, v in sim.inputs.items()},
        "simulation_inputs_differing_from_workbook": differing,
    }


# ---------------------------------------------------------------------- 3. cases
def _ours(fs: Flowsheet, r: ScenarioResult) -> dict[str, dict[int, float]]:
    """Partner-mode values in the shape of the sheet: per dryer, per source, per sink."""
    ex = fs.exhausts()
    by_source = {st.source: st for st in r.stages}
    out: dict[str, dict[int, float]] = {
        "product_celsius": {i: ex[i].temperature_celsius for i in (1, 2)},
        "fuel_kg_per_h": {s.sink: s.burner.fuel_corrected_kg_per_s * 3600.0 for s in r.sinks},
    }
    getters: dict[str, Callable[[StageOut], float | None]] = {
        "exhaust_out_celsius": lambda st: st.exhaust_out_celsius,
        "recovered_kw": lambda st: st.q_w / 1e3,
        "medium_hot_celsius": lambda st: st.htm_hot_celsius,
        "medium_cold_celsius": lambda st: st.htm_cold_celsius,
    }
    for key, get in getters.items():
        out[key] = {s: float(get(by_source[s]) or math.nan) for s in (1, 2)}
    return out


def _half(value: float) -> float:
    dec = reported_decimals(value)
    return 0.5 * 10.0 ** (-dec) if dec is not None else 0.0


def _excess(ours: float, reported: float) -> float:
    """``|ours - reported|`` in units of half the reported last digit (<= 1: within rounding)."""
    half = _half(reported)
    d = abs(ours - reported)
    return d / half if half > 0.0 else (0.0 if d <= 1e-9 * max(1.0, abs(reported)) else math.inf)


def _assignment(
    ours: Mapping[int, float], reported: Mapping[int, Reported]
) -> tuple[dict[int, int], bool]:
    """Index of the sheet -> source; identity unless the swap fits Excel and the identity does not."""
    options = ({1: 1, 2: 2}, {1: 2, 2: 1})
    worst = [max(_excess(ours[o[k]], reported[k].excel) for k in (1, 2)) for o in options]
    within = [w <= 1.0 for w in worst]
    swap_fits_better = within[1] and not within[0]
    same_fit = within[1] == within[0]
    choice = 1 if swap_fits_better or (same_fit and worst[1] < worst[0]) else 0
    return dict(options[choice]), within[choice]


def _value(ours: float, r: Reported) -> dict[str, Any]:
    return {
        "ours": ours,
        "excel": _deviation(ours, r.excel),
        "dyssol": _deviation(ours, r.dyssol),
        "excel_minus_dyssol": r.excel - r.dyssol,
    }


def compare_case(
    sim: SimulationSheet,
    n: int,
    values: Mapping[str, Mapping[int, float]],
    total_fuel: float,
    assignment: Mapping[str, Mapping[int, int]] | None = None,
) -> dict[str, Any]:
    """Every result of case ``n`` against ``values``; ``assignment`` None: chosen by Excel."""
    out: dict[str, Any] = {"assignment": {}, "values": {}}
    for key, per_index in sim.results.items():
        reported = per_index[n]
        if key in SOURCE_KEYS:
            if assignment is None:
                amap, ok = _assignment(values[key], reported)
            else:
                amap, ok = dict(assignment[key]), True
        else:
            amap, ok = {1: 1, 2: 2}, True
        out["assignment"][key] = {
            "index_to": {str(k): v for k, v in amap.items()},
            "of": "source"
            if key in SOURCE_KEYS
            else ("sink" if key == "fuel_kg_per_h" else "dryer"),
            "excel_within_rounding_with_it": ok,
        }
        out["values"][key] = {
            str(k): {"maps_to": amap[k], **_value(values[key][amap[k]], reported[k])}
            for k in (1, 2)
        }
    t = sim.total_fuel[n]
    halves = sum(_half(sim.results["fuel_kg_per_h"][n][k].excel) for k in (1, 2))
    halves_d = sum(_half(sim.results["fuel_kg_per_h"][n][k].dyssol) for k in (1, 2))
    out["total_fuel_kg_per_h"] = {
        "ours": total_fuel,
        "excel": t.excel,
        "dyssol": t.dyssol,
        "ours_minus_excel": total_fuel - t.excel,
        "ours_minus_dyssol": total_fuel - t.dyssol,
        "within_summed_rounding_excel": abs(total_fuel - t.excel) <= halves + 1e-9,
        "within_summed_rounding_dyssol": abs(total_fuel - t.dyssol) <= halves_d + 1e-9,
    }
    return out


def _outside(case: Mapping[str, Any], column: str) -> list[dict[str, Any]]:
    rows = []
    for key, per in case["values"].items():
        for k, v in per.items():
            if not v[column]["within_rounding"]:
                rows.append(
                    {
                        "quantity": key,
                        "index": int(k),
                        "maps_to": v["maps_to"],
                        "ours": v["ours"],
                        "reported": v[column]["reported"],
                        "deviation": v[column]["deviation"],
                    }
                )
    return rows


def cases_section(sim: SimulationSheet, fs: Flowsheet) -> tuple[dict[str, Any], Assignments]:
    out: dict[str, Any] = {}
    assignments: Assignments = {}
    for n in CASES:
        r = fs.evaluate(n)
        cmp_ = compare_case(sim, n, _ours(fs, r), r.fuel_corrected_kg_per_h)
        assignments[n] = {
            key: {int(k): v for k, v in a["index_to"].items()}
            for key, a in cmp_["assignment"].items()
        }
        cmp_["outside_rounding_excel"] = _outside(cmp_, "excel")
        cmp_["outside_rounding_dyssol"] = _outside(cmp_, "dyssol")
        cmp_["stages"] = [_stage_dict(s) for s in r.stages]
        out[str(n)] = cmp_
    summary = {
        "values_per_column": sum(2 * len(sim.results) for _ in CASES),
        "excel_within_rounding": sum(
            v["excel"]["within_rounding"]
            for case in out.values()
            for per in case["values"].values()
            for v in per.values()
        ),
        "dyssol_within_rounding": sum(
            v["dyssol"]["within_rounding"]
            for case in out.values()
            for per in case["values"].values()
            for v in per.values()
        ),
        "swapped_index_assignments": [
            {"case": n, "quantity": key}
            for n in CASES
            for key, a in out[str(n)]["assignment"].items()
            if a["index_to"]["1"] != 1
        ],
    }
    return {"summary": summary, "per_case": out}, assignments


# ---------------------------------------------------------------------- 4. case-2 hypothesis
def _flows_for(variant: str, case: int, flows: tuple[float, float]) -> tuple[float, float]:
    """Medium flow of the circuit of exhaust 1 and 2 (kg/h) for an assignment rule."""
    if variant == "per_exhaust_as_workbook":
        return flows
    if variant == "swapped":
        return flows[1], flows[0]
    if variant == "per_sink":
        sink_of = {s: j for j, srcs in enumerate(CASE_SOURCES[case], 1) for s in srcs}
        return flows[sink_of[1] - 1], flows[sink_of[2] - 1]
    raise KeyError(variant)


def _dyssol_fit(case_cmp: Mapping[str, Any]) -> dict[str, Any]:
    excess = [
        _excess(v["ours"], v["dyssol"]["reported"])
        for per in case_cmp["values"].values()
        for v in per.values()
    ]
    devs = [
        abs(v["dyssol"]["deviation"]) for per in case_cmp["values"].values() for v in per.values()
    ]
    return {
        "dyssol_within_rounding": sum(e <= 1.0 for e in excess),
        "values": len(excess),
        "largest_excess_over_half_unit": max(excess),
        "sum_abs_deviation": sum(devs),
    }


def _reported_internal(
    sim: SimulationSheet,
    n: int,
    column: str,
    assignment: Mapping[str, Mapping[int, int]],
    si: SiInputsV41,
    humidity: Mapping[int, float],
) -> dict[str, Any]:
    """Whether the reported numbers of one column close the equal-outlet balances of their stages."""
    m = si.plant.material
    props = partner_props(m)
    dryers = {1: si.plant.dryer_1, 2: si.plant.dryer_2}
    flows = {1: si.htm_flow_kg_per_s[0], 2: si.htm_flow_kg_per_s[1]}

    def get(key: str, source: int) -> float:
        idx = next(i for i, s in assignment[key].items() if s == source)
        r = sim.results[key][n][idx]
        return float(getattr(r, column))

    stages = {}
    for s in (1, 2):
        q = get("recovered_kw", s)
        th, tc = get("medium_hot_celsius", s), get("medium_cold_celsius", s)
        t_in = sim.results["product_celsius"][n][s]
        t_out = get("exhaust_out_celsius", s)
        ex = c.Stream(dryers[s].air_mass_flow_kg_per_s, humidity[s], float(getattr(t_in, column)))
        a = capacity(ex, props)
        stages[str(s)] = {
            "recovered_reported_kw": q,
            "medium_side_kw_with_own_flow": flows[s] * si.cp_htm_j_per_kg_k * (th - tc) / 1e3,
            "medium_side_kw_with_other_flow": flows[3 - s] * si.cp_htm_j_per_kg_k * (th - tc) / 1e3,
            "medium_side_rounding_kw": {
                "own_flow": flows[s] * si.cp_htm_j_per_kg_k * (_half(th) + _half(tc)) / 1e3,
                "other_flow": flows[3 - s] * si.cp_htm_j_per_kg_k * (_half(th) + _half(tc)) / 1e3,
            },
            "exhaust_side_kw": a * (ex.temperature_celsius - t_out) / 1e3,
            "exhaust_side_rounding_kw": a * (_half(ex.temperature_celsius) + _half(t_out)) / 1e3,
            "exhaust_out_equals_medium_hot": t_out - th,
        }
    sinks = {}
    for j, srcs in enumerate(CASE_SOURCES[n], 1):
        if not srcs:
            continue
        d = dryers[j]
        gas = c.Stream(
            d.air_mass_flow_kg_per_s, d.air_humidity_kg_per_kg, si.plant.ambient.temperature_celsius
        )
        cg = capacity(gas, props)
        t_prev = gas.temperature_celsius
        gas_side = []
        for s in srcs:
            tc = get("medium_cold_celsius", s)
            gas_side.append({"source": s, "gas_side_kw": cg * (tc - t_prev) / 1e3})
            t_prev = tc
        fuel = sim.results["fuel_kg_per_h"][n][j]
        sinks[str(j)] = {
            "stages_gas_side": gas_side,
            "fuel_from_last_cold_side_kg_per_h": cg
            * (d.air_temperature_celsius - t_prev)
            / (m.heater_efficiency * m.lower_heating_value_j_per_kg)
            * 3600.0,
            "fuel_reported_kg_per_h": float(getattr(fuel, column)),
        }
    return {"stages_by_source": stages, "sinks": sinks}


def _offset_for_exhaust_out(
    ex_in: ExcelInputsV41, case: int, flows: tuple[float, float], source: int, target: float
) -> float:
    """Shift of one dryer's exhaust temperature (K) that gives ``target`` after its stage."""
    fs0 = Flowsheet(FlowsheetConfig(ex_in.with_htm_flow(flows).to_si(), SWITCHES_412))
    base = fs0.exhausts()

    def t_out(delta: float) -> float:
        pinned = {
            i: (
                base[i].temperature_celsius + (delta if i == source else 0.0),
                base[i].humidity_kg_per_kg,
            )
            for i in (1, 2)
        }
        fs = Flowsheet(
            FlowsheetConfig(
                ex_in.with_htm_flow(flows).to_si(), SWITCHES_412, pinned_exhausts=pinned
            )
        )
        st = next(s for s in fs.evaluate(case).stages if s.source == source)
        return st.exhaust_out_celsius

    lo, hi = -5.0, 5.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t_out(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


FLOW_INDEPENDENT_KEYS = ("product_celsius",)
"""Results of the sheet that do not depend on the medium flows (the dryers)."""


def _hypothesis_verdict(detail: Mapping[str, Any], internal: Mapping[str, Any]) -> dict[str, Any]:
    """The reading of the module doc, computed; strict (all values) and flow-dependent values."""
    base, swap = detail["per_exhaust_as_workbook"]["values"], detail["swapped"]["values"]
    per_value = []
    for key, per in swap.items():
        for k, v in per.items():
            d_swap = abs(v["dyssol"]["deviation"])
            d_base = abs(base[key][k]["dyssol"]["deviation"])
            per_value.append(
                {
                    "quantity": key,
                    "index": int(k),
                    "within_rounding_swapped": v["dyssol"]["within_rounding"],
                    "nearer_swapped": d_swap < d_base,
                    "ok": v["dyssol"]["within_rounding"] or d_swap < d_base,
                    "flow_dependent": key not in FLOW_INDEPENDENT_KEYS,
                }
            )
    balance = {}
    for s, st in internal["stages_by_source"].items():
        q, tol = st["recovered_reported_kw"], st["medium_side_rounding_kw"]
        balance[s] = {
            "closes_with_other_flow": abs(st["medium_side_kw_with_other_flow"] - q)
            <= tol["other_flow"],
            "closes_with_own_flow": abs(st["medium_side_kw_with_own_flow"] - q) <= tol["own_flow"],
        }
    balance_ok = all(
        b["closes_with_other_flow"] and not b["closes_with_own_flow"] for b in balance.values()
    )
    strict = all(v["ok"] for v in per_value)
    flow_only = all(v["ok"] for v in per_value if v["flow_dependent"])
    return {
        "values": per_value,
        "medium_side_balance_of_dyssol": balance,
        "medium_side_balance_supports_swap": balance_ok,
        "confirmed_strict_all_values": strict and balance_ok,
        "confirmed_flow_dependent_values": flow_only and balance_ok,
        "note": "the split into flow-dependent values was written after a dry run of this script "
        "(no artifact) had shown the dryer-2 product temperature of Dyssol's case 2; both readings "
        "are kept",
    }


def case_2_hypothesis(
    sim: SimulationSheet,
    ex: ExcelInputsV41,
    assignments: Assignments,
) -> dict[str, Any]:
    flows = ex.htm_flow_kg_per_h
    variants: dict[str, Any] = {}
    detail: dict[str, Any] = {}
    for variant in ("per_exhaust_as_workbook", "swapped", "per_sink"):
        per_case = {}
        for n in CASES:
            f = _flows_for(variant, n, flows)
            fs = Flowsheet(FlowsheetConfig(ex.with_htm_flow(f).to_si(), SWITCHES_412))
            r = fs.evaluate(n)
            cmp_ = compare_case(sim, n, _ours(fs, r), r.fuel_corrected_kg_per_h, assignments[n])
            per_case[str(n)] = {"flows_kg_per_h": list(f), **_dyssol_fit(cmp_)}
            if n == 2:
                cmp_["outside_rounding_dyssol"] = _outside(cmp_, "dyssol")
                cmp_["stages"] = [_stage_dict(s) for s in r.stages]
                detail[variant] = cmp_
        variants[variant] = per_case
    fs_workbook = Flowsheet(FlowsheetConfig(ex.to_si(), SWITCHES_412))
    humidity = {i: s.humidity_kg_per_kg for i, s in fs_workbook.exhausts().items()}
    internal = {
        column: _reported_internal(sim, 2, column, assignments[2], ex.to_si(), humidity)
        for column in ("excel", "dyssol")
    }
    dyssol_t = sim.results["exhaust_out_celsius"][2]
    idx_source_1 = next(i for i, s in assignments[2]["exhaust_out_celsius"].items() if s == 1)
    offset = _offset_for_exhaust_out(ex, 2, (flows[1], flows[0]), 1, dyssol_t[idx_source_1].dyssol)
    pinned_fs_base = Flowsheet(
        FlowsheetConfig(ex.with_htm_flow((flows[1], flows[0])).to_si(), SWITCHES_412)
    )
    base = pinned_fs_base.exhausts()
    pinned = {
        1: (base[1].temperature_celsius + offset, base[1].humidity_kg_per_kg),
        2: (base[2].temperature_celsius, base[2].humidity_kg_per_kg),
    }
    fs_off = Flowsheet(
        FlowsheetConfig(
            ex.with_htm_flow((flows[1], flows[0])).to_si(), SWITCHES_412, pinned_exhausts=pinned
        )
    )
    r_off = fs_off.evaluate(2)
    cmp_off = compare_case(
        sim, 2, _ours(fs_off, r_off), r_off.fuel_corrected_kg_per_h, assignments[2]
    )
    cmp_off["outside_rounding_dyssol"] = _outside(cmp_off, "dyssol")
    return {
        "verdict": _hypothesis_verdict(detail, internal["dyssol"]),
        "variants": variants,
        "case_2": detail,
        "reported_internal_balances_case_2": internal,
        "swapped_with_exhaust_1_offset": {
            "offset_k": offset,
            "reason": "the shift of the exhaust of dryer 1 that alone gives Dyssol's 103.6 degC "
            "after its stage (an equivalent, not a cause)",
            "exhaust_1_celsius": base[1].temperature_celsius + offset,
            **cmp_off,
        },
    }


# ---------------------------------------------------------------------- 5. validators
def validators_section(
    sim: SimulationSheet,
    fs_by_mode: Mapping[str, Flowsheet],
    fs_swapped: Flowsheet,
    si: SiInputsV41,
    assignments: Assignments,
) -> dict[str, Any]:
    m = si.plant.material
    sat = SaturationProps.from_material(m)
    dryers = {1: si.plant.dryer_1, 2: si.plant.dryer_2}
    out: dict[str, Any] = {"dryers": {}, "medium": {}}
    for mode, fs in fs_by_mode.items():
        exs = fs.exhausts()
        per = {}
        for i in (1, 2):
            d = dryers[i]
            res = fs.dryer_unit(i).solve(DryerInputs(d))
            per[str(i)] = {
                "exhaust_celsius": exs[i].temperature_celsius,
                "exhaust_humidity_g_per_kg": exs[i].humidity_kg_per_kg * 1e3,
                "product_kg_per_h": res.product_kg_per_s * 3600.0,
                "saturation_limited": res.saturation_limited,
                "findings": _findings(
                    [
                        check_exhaust_saturation(
                            exs[i].temperature_celsius, exs[i].humidity_kg_per_kg
                        ),
                        check_exhaust_above_adiabatic_saturation(
                            exs[i].temperature_celsius,
                            d.air_temperature_celsius,
                            d.air_humidity_kg_per_kg,
                            sat,
                        ),
                        check_product_mass(
                            res.product_kg_per_s,
                            d.spray_mass_flow_kg_per_s,
                            d.spray_solid_fraction_kg_per_kg,
                        ),
                    ]
                ),
            }
        out["dryers"][mode] = per
    partner_ex = fs_by_mode["partner"].exhausts()
    reported_dryers: dict[str, Any] = {}
    for n in CASES:
        per_column: dict[str, Any] = {}
        for i in (1, 2):
            d = dryers[i]
            for column in ("excel", "dyssol"):
                t = float(getattr(sim.results["product_celsius"][n][i], column))
                y = partner_ex[i].humidity_kg_per_kg
                per_column[f"{i}/{column}"] = {
                    "exhaust_celsius": t,
                    "humidity_from_our_mass_balance_g_per_kg": y * 1e3,
                    "findings": _findings(
                        [
                            check_exhaust_saturation(t, y),
                            check_exhaust_above_adiabatic_saturation(
                                t, d.air_temperature_celsius, d.air_humidity_kg_per_kg, sat
                            ),
                        ]
                    ),
                }
        reported_dryers[str(n)] = per_column
    out["dryers_reported"] = reported_dryers
    out["dryers_reported_note"] = (
        "the sheet reports no product mass and no exhaust humidity: (a) uses the reported "
        "temperature with the humidity of our mass balance, (b) is checked on ours only"
    )
    medium: dict[str, Any] = {}
    for n in CASES:
        per_circuit: dict[str, Any] = {}
        for s in (1, 2):
            idx = next(i for i, src in assignments[n]["medium_hot_celsius"].items() if src == s)
            rep = sim.results["medium_hot_celsius"][n][idx]
            items: dict[str, Finding] = {
                "excel": check_medium_boiling(rep.excel),
                "dyssol": check_medium_boiling(rep.dyssol),
            }
            for mode, fs in (*fs_by_mode.items(), ("partner_swapped_flows", fs_swapped)):
                st = next(x for x in fs.evaluate(n).stages if x.source == s)
                if st.htm_hot_celsius is not None:
                    items[mode] = check_medium_boiling(st.htm_hot_celsius)
            per_circuit[f"circuit_of_exhaust_{s}"] = {k: f.as_dict() for k, f in items.items()}
        medium[str(n)] = per_circuit
    out["medium"] = medium
    hits: list[dict[str, Any]] = []
    for case_key, circuits in medium.items():
        for circuit, by_source in circuits.items():
            for source, found in by_source.items():
                if found["severity"] != "ok":
                    hits.append(
                        {
                            "case": int(case_key),
                            "circuit": circuit,
                            "source": source,
                            "medium_celsius": found["values"]["medium_celsius"],
                            "required_pressure_bar": found["values"].get(
                                "required_pressure_pa", 0.0
                            )
                            / 1e5,
                        }
                    )
    for mode, by_dryer in out["dryers"].items():
        for i_key, v in by_dryer.items():
            for found in v["findings"]:
                if found["severity"] != "ok":
                    hits.append({"dryer": int(i_key), "source": mode, "check": found["check"]})
    for case_key, by_column in reported_dryers.items():
        for key, v in by_column.items():
            for found in v["findings"]:
                if found["severity"] != "ok":
                    hits.append(
                        {"case": int(case_key), "dryer_column": key, "check": found["check"]}
                    )
    out["hits"] = hits
    return out


# ---------------------------------------------------------------------- 6. ranking, physics, pinch
def ranking(fuel: Mapping[int, float]) -> dict[str, Any]:
    order = sorted(fuel, key=lambda n: (fuel[n], n))
    text = str(order[0])
    for a, b in itertools.pairwise(order):
        text += (" = " if fuel[b] - fuel[a] < TIE_KG_PER_H else " < ") + str(b)
    gaps = [
        {"from": a, "to": b, "kg_per_h": fuel[b] - fuel[a]} for a, b in itertools.pairwise(order)
    ]
    return {"order": order, "text": text, "neighbour_gaps": gaps}


def _relation(fuel: Mapping[int, float], a: int, b: int) -> str:
    d = fuel[a] - fuel[b]
    return "=" if abs(d) < TIE_KG_PER_H else ("<" if d < 0 else ">")


def rank_changes(
    reference: Mapping[int, float], other: Mapping[int, float]
) -> list[dict[str, Any]]:
    out = []
    for a, b in itertools.combinations(CASES, 2):
        ra, rb = _relation(reference, a, b), _relation(other, a, b)
        if ra != rb:
            out.append({"pair": [a, b], "partner": ra, "this_mode": rb})
    return out


def mode_section(fs: Flowsheet) -> dict[str, Any]:
    res = {n: fs.evaluate(n) for n in (0, *CASES)}
    fuel = {n: res[n].fuel_corrected_kg_per_h for n in CASES}
    f0 = res[0].fuel_corrected_kg_per_h
    cases = {}
    for n in (0, *CASES):
        r = res[n]
        cases[str(n)] = {
            "fuel_kg_per_h": r.fuel_corrected_kg_per_h,
            "fuel_per_sink_kg_per_h": [s.burner.fuel_corrected_kg_per_s * 3600.0 for s in r.sinks],
            "burner_duty_kw": r.burner_duty_kw,
            "recovered_kw": r.recovered_kw,
            "saving_against_no_recovery_kg_per_h": f0 - r.fuel_corrected_kg_per_h,
            "saving_against_no_recovery_percent": 100.0 * (f0 - r.fuel_corrected_kg_per_h) / f0,
            "minus_case_1_kg_per_h": r.fuel_corrected_kg_per_h - fuel[1],
            "medium_hot_max_celsius": r.htm_hot_max_celsius if r.stages else None,
            "heater_in_celsius": [s.heater_in_celsius for s in r.sinks],
            "stages": [_stage_dict(s) for s in r.stages],
        }
    return {"cases": cases, "ranking": ranking(fuel), "fuel": fuel}


def pinch_section(
    fs: Flowsheet, si: SiInputsV41, mode: Mapping[str, Any], cases: Mapping[str, Any]
) -> dict[str, Any]:
    plant = si.plant
    m = plant.material
    exs = fs.exhausts()
    sinks = [
        SinkDemand(
            f"sink {j}",
            c.Stream(
                d.air_mass_flow_kg_per_s,
                d.air_humidity_kg_per_kg,
                plant.ambient.temperature_celsius,
            ),
            d.air_temperature_celsius,
        )
        for j, d in ((1, plant.dryer_1), (2, plant.dryer_2))
    ]
    target = network_pinch_target(
        [("exhaust 1", exs[1]), ("exhaust 2", exs[2])],
        sinks,
        partner_props(m),
        dt_min_k_per_exchanger=DT_MIN_K,
        htm_max_celsius=mode["htm_max_celsius"],
    )
    to_fuel: Callable[[float], float] = lambda kw: (  # noqa: E731
        kw * 1e3 / (m.heater_efficiency * m.lower_heating_value_j_per_kg) * 3600.0
    )
    fuel_min = to_fuel(target.min_burner_kw)
    f0 = cases["0"]["fuel_kg_per_h"]
    dist = {}
    for n in CASES:
        f = cases[str(n)]["fuel_kg_per_h"]
        dist[str(n)] = {
            "fuel_minus_optimum_kg_per_h": f - fuel_min,
            "fuel_minus_optimum_percent_of_optimum": 100.0 * (f - fuel_min) / fuel_min,
            "recovered_gap_kw": target.max_recovered_kw - cases[str(n)]["recovered_kw"],
            "share_of_possible_saving_percent": 100.0 * (f0 - f) / (f0 - fuel_min),
        }
    p = target.pinch
    return {
        "approach_of_problem_table_k": target.approach_k,
        "htm_max_celsius": mode["htm_max_celsius"],
        "total_demand_kw": target.total_demand_kw,
        "fixed_burner_kw_above_medium_limit": target.fixed_burner_kw,
        "problem_table_hot_utility_kw": p.minimum_hot_utility_kw,
        "min_burner_kw": target.min_burner_kw,
        "max_recovered_kw": target.max_recovered_kw,
        "min_fuel_kg_per_h": fuel_min,
        "fuel_no_recovery_kg_per_h": f0,
        "cold_utility_kw": p.minimum_cold_utility_kw,
        "pinch_hot_side_celsius": p.pinch_temperature_hot_side_celsius,
        "pinch_cold_side_celsius": p.pinch_temperature_cold_side_celsius,
        "bound_not_above_best_case": fuel_min
        <= min(cases[str(n)]["fuel_kg_per_h"] for n in CASES) + 1e-12,
        "distance_per_case": dist,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="partner_6cases_comparison")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "runs")
    args = parser.parse_args()
    target = args.out_dir / f"{args.name}.json"
    if target.exists():
        print(f"{target} exists; choose a new --name (artifacts are never overwritten)")
        return 1
    workbook = _find("partner/TwinLink Excel.xlsx", "partner/TwinLink_Excel.xlsx")
    results = _find(
        "partner/Dyssol_Excel simulation_results.xlsx",
        "partner/Dyssol_Excel simulation results.xlsx",
    )
    previous = _find("partner/Model v4.1_2.xlsx", "partner/Model_v4.1_2.xlsx")
    comparison = _find("partner/Comparison.xlsx")
    ex = load_excel_inputs_v41(workbook)
    si = ex.to_si()
    sim = load_simulation_sheet(results)
    fs_by_mode = {mode: _flowsheet(si, mode) for mode in MODES}
    fs_swapped = Flowsheet(
        FlowsheetConfig(
            ex.with_htm_flow((ex.htm_flow_kg_per_h[1], ex.htm_flow_kg_per_h[0])).to_si(),
            SWITCHES_412,
        )
    )
    cases, assignments = cases_section(sim, fs_by_mode["partner"])
    modes = {mode: mode_section(fs) for mode, fs in fs_by_mode.items()}
    for mode in modes:
        modes[mode]["rank_changes_against_partner"] = rank_changes(
            modes["partner"]["fuel"], modes[mode]["fuel"]
        )
    pinch = {
        mode: pinch_section(fs_by_mode[mode], si, MODES[mode], modes[mode]["cases"])
        for mode in MODES
        if mode != "partner"
    }
    record: dict[str, Any] = {
        "provenance": {
            "git/commit": _git("rev-parse", "HEAD").strip(),
            "git/dirty": bool(_git("status", "--porcelain").strip()),
            "python": sys.version.split()[0],
            **require_code_location(ROOT),
        },
        "parameters": {
            "files": [p.name for p in (workbook, results, previous, comparison)],
            "material_si": dataclasses.asdict(si.plant.material),
            "htm_flow_kg_per_h": list(ex.htm_flow_kg_per_h),
            "cp_htm_j_per_kg_k": si.cp_htm_j_per_kg_k,
            "modes": {
                k: {
                    "switches": v["switches"].as_dict(),
                    "saturation_limit": v["saturation_limit"],
                    "htm_max_celsius": v["htm_max_celsius"],
                }
                for k, v in MODES.items()
            },
            "dt_min_k_per_exchanger": DT_MIN_K,
            "dt_min_convention": "per exchanger; the problem table uses 2 dT (two exchangers in series)",
            "tie_kg_per_h": TIE_KG_PER_H,
            "within_rounding": "|ours - reported| <= half a unit of the reported last decimal",
            "case_sources": {str(k): [list(a), list(b)] for k, (a, b) in CASE_SOURCES.items()},
        },
        "workbook": workbook_section(previous, workbook),
        "results_workbook": results_workbook_section(results, comparison, sim, ex),
        "cases": cases,
        "case_2_hypothesis": case_2_hypothesis(sim, ex, assignments),
        "validators": validators_section(sim, fs_by_mode, fs_swapped, si, assignments),
        "physical_mode": {
            k: {kk: vv for kk, vv in v.items() if kk != "fuel"} for k, v in modes.items()
        },
        "pinch": pinch,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
