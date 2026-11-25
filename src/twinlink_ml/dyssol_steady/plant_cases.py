"""The six heat-recovery cases at the plant inputs of the partner deliveries.

Design and pre-registered reading: ``docs/NN_plant_cases.md`` (docstring of
``scripts/report_plant_cases.py``).

Two lines: line 1 is the line numbered 1 in the delivered case flowsheets, line 2 the line that
both deliveries label as plant A. A *point* is the pair of their inputs. Two model families
evaluate the six cases (and case 0, no recovery) at a point:

* **U**, the delivered units (:func:`evaluate_units`, through :mod:`.network`);
* **W**, the workbook rebuild :mod:`twinlink_ml.flowsheet` in the three modes of docs/11
  section 13.6 (:func:`evaluate_workbook`), with the properties of the workbook.

The module builds the points (:func:`build_points`), applies the readings (:func:`reading_f`,
:func:`reading_c`, :func:`reading_a`), evaluates the worded rule of Befund 121 and the
generalised rule added by the project lead (:func:`generalised_rule_case`,
:func:`pair_rule_hit_rates`), reads the plant sheet of the delivery of 2026-11-19
(:func:`parse_plant_sheet`) and assembles two records (:func:`summarise`): one for the repository
and one that stays local.

**What the repository record may hold** (decision of the project lead, 2026-11-25): ranks,
savings and regrets in %, saturation ratios, counts, flags, reading codes, differences Dyssol
minus rebuild. No input or result of a partner sheet or flowsheet, no circuit flow, no fuel or
heat in absolute units, no temperature, no measured value, no label of a line.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from twinlink_ml.case_ranking.evaluator import MODES, exhaust_relative_humidity
from twinlink_ml.case_ranking.workbook import WORKBOOK
from twinlink_ml.flowsheet import Flowsheet, FlowsheetConfig
from twinlink_ml.partner_v4.inputs import ExcelDryerInputs, ExcelInputs
from twinlink_ml.partner_v4_1.inputs import ExcelInputsV41

from .materials import DyssolProperties
from .network import CASES, evaluate_network, sink_capacity_kw_per_k
from .plant_check import MeasuredPoint, unit_inputs
from .reference import LineInputs

__all__ = [
    "CASE_NUMBERS",
    "MODELS",
    "MODEL_U",
    "READ_SETTING",
    "CaseResult",
    "ModelResult",
    "PlantSheet",
    "Point",
    "assign_lines",
    "build_points",
    "case_one",
    "evaluate_units",
    "evaluate_workbook",
    "exchange_sizes",
    "generalised_rule_case",
    "pair_choice",
    "pair_rule_hit_rates",
    "parse_plant_sheet",
    "reading_a",
    "reading_c",
    "reading_f",
    "summarise",
    "with_flows",
    "within_rounding",
    "worded_rule_predicts_case_1",
    "workbook_case_details",
    "workbook_inputs",
]

Point = tuple[LineInputs, LineInputs]
"""Inputs of line 1 and line 2; ``medium_kg_per_h`` is the circuit of the line's own exhaust."""

CASE_NUMBERS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
MODEL_U = "U"
MODELS: tuple[str, ...] = (MODEL_U, *(f"W:{name}" for name in MODES))
"""The delivered units and the three modes of the workbook rebuild."""
READ_SETTING = "stored"
"""Circuit flows stored in the delivered case flowsheets: the setting the readings use."""
WORDED_RULE_THRESHOLD_K: dict[str, float] = {
    MODEL_U: -15.5,
    "W:partner": -15.5,
    "W:physical_unpressurised_95": -0.54,
    "W:physical_pressurised": -0.54,
}
"""First split of the rule of Befund 121 (docs/38, section 6.4): case 1 where exhaust 1 minus
exhaust 2 lies above the threshold. U has the exchanger of the partner mode (equal outlet
temperatures) and gets its threshold."""


@dataclass(frozen=True)
class CaseResult:
    """One case at one point in one model."""

    case: int
    fuel_kg_per_h: float
    recovered_kw: float
    admissible: bool
    negative_stages: int


@dataclass(frozen=True)
class ModelResult:
    """One model at one point: the two dryer exhausts, the two sinks, cases 0 to 6."""

    model: str
    exhaust_celsius: tuple[float, float]
    saturation: tuple[float, float]
    """U: saturation ratio of each dryer exhaust; W: its relative humidity."""
    sink_capacity_kw_per_k: tuple[float, float]
    cases: tuple[CaseResult, ...]

    def fuel(self) -> np.ndarray:
        """Fuel of cases 0 to 6 in kg/h (index = case)."""
        return np.array([c.fuel_kg_per_h for c in self.cases])


# ------------------------------------------------------------------------------------ models


def evaluate_units(props: DyssolProperties, point: Point) -> ModelResult:
    """Model U: the delivered units at ``point``.

    A case is admissible if no dryer exhaust is over-saturated, at the dryer outlet or after its
    recovery stage (the unit has no saturation limit and cannot condense).
    """
    cases = []
    for case in (0, *CASE_NUMBERS):
        out = evaluate_network(props, point, CASES[case])
        cases.append(
            CaseResult(
                case, out.fuel_kg_per_h, out.recovered_kw, out.admissible, out.negative_stages
            )
        )
    base = evaluate_network(props, point, CASES[0]).lines
    return ModelResult(
        model=MODEL_U,
        exhaust_celsius=(base[0].dryer_outlet_celsius, base[1].dryer_outlet_celsius),
        saturation=(base[0].saturation_ratio, base[1].saturation_ratio),
        sink_capacity_kw_per_k=(
            sink_capacity_kw_per_k(props, point[0]),
            sink_capacity_kw_per_k(props, point[1]),
        ),
        cases=tuple(cases),
    )


def workbook_inputs(point: Point) -> ExcelInputsV41:
    """Inputs of the workbook rebuild at ``point``.

    Properties of the workbook (docs/11 section 13.1); equilibrium moisture, heater efficiency
    and calorific value of the lines. The workbook has one ambient temperature and one set of
    these three values: both lines must agree in them.
    """
    one, two = point
    for name in (
        "gas_celsius",
        "equilibrium_moisture_g_per_kg",
        "heater_efficiency_percent",
        "calorific_value_kj_per_kg",
    ):
        if not math.isclose(getattr(one, name), getattr(two, name), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"the workbook rebuild needs one {name} for both lines")
    material = replace(
        WORKBOOK.material,
        equilibrium_moisture_g_per_kg=one.equilibrium_moisture_g_per_kg,
        heater_efficiency_percent=one.heater_efficiency_percent,
        calorific_value_kj_per_kg=one.calorific_value_kj_per_kg,
    )

    def dryer(line: LineInputs) -> ExcelDryerInputs:
        return ExcelDryerInputs(
            air_mass_flow_kg_per_h=line.dry_gas_kg_per_h,
            air_moisture_g_per_kg=line.gas_humidity_g_per_kg,
            air_temperature_celsius=line.dryer_inlet_celsius,
            spray_mass_flow_kg_per_h=line.spray_kg_per_h,
            spray_solid_fraction_kg_per_kg=line.solid_fraction,
            spray_temperature_celsius=line.spray_celsius,
        )

    plant = ExcelInputs(
        dryer(one), dryer(two), material, one.gas_celsius, WORKBOOK.ambient_pressure_atm
    )
    return ExcelInputsV41(plant, (one.medium_kg_per_h, two.medium_kg_per_h))


def evaluate_workbook(point: Point, mode: str, max_relative_humidity: float) -> ModelResult:
    """Model W in one mode of docs/11 section 13.6.

    Admissible (rule of Befund 121): both dryer exhausts at or below ``max_relative_humidity``;
    the same for every case of the point.
    """
    settings = MODES[mode]
    inputs = workbook_inputs(point)
    sheet = Flowsheet(
        FlowsheetConfig(
            inputs.to_si(),
            settings.switches,
            saturation_limit=settings.saturation_limit,
            htm_max_celsius=settings.htm_max_celsius,
        )
    )
    exhausts = sheet.exhausts()
    humidity = tuple(
        exhaust_relative_humidity(exhausts[i].temperature_celsius, exhausts[i].humidity_kg_per_kg)
        for i in (1, 2)
    )
    admissible = all(h <= max_relative_humidity for h in humidity)
    cases = []
    for case in (0, *CASE_NUMBERS):
        r = sheet.evaluate(case)
        cases.append(
            CaseResult(
                case,
                r.fuel_corrected_kg_per_h,
                r.recovered_kw,
                admissible,
                sum(s.q_w < 0.0 for s in r.stages),
            )
        )
    material = inputs.plant.material
    capacity = tuple(
        d.air_mass_flow_kg_per_h
        / 3600.0
        * (
            material.cp_gas_kj_per_kg_k
            + d.air_moisture_g_per_kg / 1000.0 * material.cp_vapour_kj_per_kg_k
        )
        for d in (inputs.plant.dryer_1, inputs.plant.dryer_2)
    )
    return ModelResult(
        model=f"W:{mode}",
        exhaust_celsius=(exhausts[1].temperature_celsius, exhausts[2].temperature_celsius),
        saturation=(humidity[0], humidity[1]),
        sink_capacity_kw_per_k=(capacity[0], capacity[1]),
        cases=tuple(cases),
    )


def workbook_case_details(point: Point, mode: str, case: int) -> dict[str, tuple[float, float]]:
    """Model W, one case: fuel per heater [kg/h] and heat taken from each exhaust [kW]."""
    settings = MODES[mode]
    sheet = Flowsheet(
        FlowsheetConfig(
            workbook_inputs(point).to_si(),
            settings.switches,
            saturation_limit=settings.saturation_limit,
            htm_max_celsius=settings.htm_max_celsius,
        )
    )
    r = sheet.evaluate(case)
    fuel = [sink.burner.fuel_corrected_kg_per_s * 3600.0 for sink in r.sinks]
    heat = [sum(s.q_w for s in r.stages if s.source == source) / 1000.0 for source in (1, 2)]
    return {"fuel_kg_per_h": (fuel[0], fuel[1]), "recovered_kw": (heat[0], heat[1])}


# ------------------------------------------------------------------------------------ points


def assign_lines(candidates: Sequence[LineInputs], stored: Sequence[LineInputs]) -> tuple[int, ...]:
    """Index of the candidate that is line 1, line 2, ... of the stored flowsheets.

    Rule of Befund 130: a candidate belongs to the stored line whose dryer inlet temperature is
    nearest to its own; the assignment must be one-to-one.
    """
    chosen = tuple(
        min(
            range(len(candidates)),
            key=lambda i: abs(candidates[i].dryer_inlet_celsius - line.dryer_inlet_celsius),
        )
        for line in stored
    )
    if len(set(chosen)) != len(stored) or len(candidates) != len(stored):
        raise ValueError("the processes do not map one-to-one onto the lines of the flowsheets")
    return chosen


def with_flows(point: Point, flows_kg_per_h: tuple[float, float]) -> Point:
    """``point`` with the circuit flows of exhaust 1 and exhaust 2."""
    return (
        replace(point[0], medium_kg_per_h=flows_kg_per_h[0]),
        replace(point[1], medium_kg_per_h=flows_kg_per_h[1]),
    )


def exchange_sizes(point: Point, *, scale_circuits: bool = False) -> Point:
    """``point`` with the sizes of the two lines exchanged.

    Dry gas and spray of a line are scaled by one factor per line so that the two dry-gas flows
    are exchanged; humidity, temperatures and solid fraction stay, and with them both dryer
    exhausts. The circuit flows stay with their exhaust unless ``scale_circuits``.
    """
    one, two = point
    f1 = two.dry_gas_kg_per_h / one.dry_gas_kg_per_h
    f2 = one.dry_gas_kg_per_h / two.dry_gas_kg_per_h

    def scaled(line: LineInputs, factor: float) -> LineInputs:
        return replace(
            line,
            dry_gas_kg_per_h=line.dry_gas_kg_per_h * factor,
            spray_kg_per_h=line.spray_kg_per_h * factor,
            medium_kg_per_h=line.medium_kg_per_h * (factor if scale_circuits else 1.0),
        )

    return scaled(one, f1), scaled(two, f2)


def build_points(
    sheet_lines: Point,
    segments: Mapping[str, Sequence[MeasuredPoint]],
    *,
    solid_reference: float,
    solid_band: Sequence[float],
    humidity_reference_g_per_kg: float,
    normal_density_kg_per_m3: float,
) -> dict[str, Point]:
    """All points of the study by tag (without circuit flows).

    ``S`` both lines as the sweep sheet states them; ``R`` S with the reference solid fraction in
    line 2; ``M|<segment>`` line 2 at the measured inputs of a segment (reference solid fraction,
    reference humidity), line 1 as in S with the fresh-gas temperature of the segment; ``X`` R
    with the sizes of the lines exchanged; ``X|scaled`` the same with the circuit flows scaled
    (set by the caller, the points here carry no flow); ``X|M|<segment>``; and for every solid
    fraction of the band ``R|sf=<value>`` and ``M|<segment>|sf=<value>``.
    """
    one, two = sheet_lines
    points: dict[str, Point] = {"S": (one, two)}

    def reference(solid: float) -> Point:
        return one, replace(two, solid_fraction=solid)

    def measured(segment: MeasuredPoint, solid: float) -> Point:
        line_two = unit_inputs(
            segment,
            two,
            solid_fraction=solid,
            humidity_g_per_kg=humidity_reference_g_per_kg,
            normal_density_kg_per_m3=normal_density_kg_per_m3,
        )
        return replace(one, gas_celsius=segment.ambient_celsius), line_two

    labels = [p for pts in segments.values() for p in pts]
    points["R"] = reference(solid_reference)
    for segment in labels:
        points[f"M|{segment.label}"] = measured(segment, solid_reference)
    points["X"] = exchange_sizes(points["R"])
    for segment in labels:
        points[f"X|M|{segment.label}"] = exchange_sizes(points[f"M|{segment.label}"])
    for solid in solid_band:
        points[f"R|sf={solid:g}"] = reference(solid)
        for segment in labels:
            points[f"M|{segment.label}|sf={solid:g}"] = measured(segment, solid)
    return points


# ------------------------------------------------------------------------------------ readings


def case_one(result: ModelResult, exact: float, fraction: float) -> dict[str, Any]:
    """Case 1 against the best admissible case at one point.

    Practical tie (Befund 121, F4): the gap of case 1 to the best case is below ``fraction`` of
    the saving of the best case against case 0, never below ``exact``. A point at which case 1
    is inadmissible counts as "not best".
    """
    fuel = [float(v) for v in result.fuel()]
    admissible = [c.case for c in result.cases[1:] if c.admissible]
    if not admissible:
        return {"admissible_cases": [], "best_case": None, "best_or_practically_equal": False}
    best = min(admissible, key=lambda k: fuel[k])
    saving = fuel[0] - fuel[best]
    gap = fuel[1] - fuel[best]
    one_admissible = 1 in admissible
    return {
        "admissible_cases": admissible,
        "best_case": best,
        "order": sorted(admissible, key=lambda k: fuel[k]),
        "best_or_practically_equal": bool(one_admissible and gap < max(fraction * saving, exact)),
        "saving_percent": {str(k): 100.0 * (fuel[0] - fuel[k]) / fuel[0] for k in CASE_NUMBERS},
        "regret_case_1_percent_of_best_saving": 100.0 * gap / saving if saving > 0.0 else None,
        "regret_case_1_percent_of_best_fuel": 100.0 * gap / fuel[best],
        "negative_stages": {str(c.case): c.negative_stages for c in result.cases[1:]},
    }


def reading_f(cells: Sequence[bool]) -> str:
    """F0 every cell (point, case) admissible, F1 otherwise."""
    if not cells:
        raise ValueError("reading F needs at least one cell")
    return "F0" if all(cells) else "F1"


def reading_c(best_or_equal: Sequence[bool]) -> str:
    """C0 case 1 best or practically equal at every read point, C1 at none, C2 otherwise."""
    if not best_or_equal:
        raise ValueError("reading C needs at least one point")
    if all(best_or_equal):
        return "C0"
    return "C2" if any(best_or_equal) else "C1"


def reading_a(at_reference: Mapping[str, bool], at_exchanged: Mapping[str, bool]) -> str:
    """Attribution: whether exchanging the sizes of the lines turns the order.

    ``at_reference`` / ``at_exchanged``: model -> case 1 best or practically equal at R / at X.
    Not applicable if case 1 already is at R in a model; A0 if it is at X in every model given;
    A1 otherwise.
    """
    if any(at_reference.values()):
        return "not applicable"
    return "A0" if all(at_exchanged.values()) else "A1"


def worded_rule_predicts_case_1(result: ModelResult) -> bool:
    """The scale-free part of the rule of Befund 121: exhaust 1 minus exhaust 2 above the split."""
    difference = result.exhaust_celsius[0] - result.exhaust_celsius[1]
    return difference > WORDED_RULE_THRESHOLD_K[result.model]


def generalised_rule_case(
    exhaust_celsius: Sequence[float], sink_capacity: Sequence[float]
) -> int | None:
    """Case 1 or 2 by the generalised rule; ``None`` if either pair ties.

    Post-hoc hypothesis of the project lead (2026-11-25): the warmer exhaust preheats the sink
    with the larger heat-capacity rate.
    """
    if exhaust_celsius[0] == exhaust_celsius[1] or sink_capacity[0] == sink_capacity[1]:
        return None
    warmer = 0 if exhaust_celsius[0] > exhaust_celsius[1] else 1
    larger = 0 if sink_capacity[0] > sink_capacity[1] else 1
    return 1 if warmer == larger else 2


def pair_choice(result: ModelResult, exact: float) -> int | None:
    """The better of cases 1 and 2 (lower fuel), ``None`` within ``exact`` of each other."""
    fuel = result.fuel()
    if abs(fuel[1] - fuel[2]) < exact:
        return None
    return 1 if fuel[1] < fuel[2] else 2


def pair_rule_hit_rates(
    fuel: np.ndarray,
    exhaust_celsius: np.ndarray,
    sink_capacity: np.ndarray,
    *,
    exact: float,
    fraction: float,
    worded_threshold_k: float,
) -> dict[str, Any]:
    """Hit rates of two rules for the choice between cases 1 and 2 over many points.

    ``fuel`` (n, 7) with column = case, ``exhaust_celsius`` and ``sink_capacity`` (n, 2).
    Generalised rule: :func:`generalised_rule_case`. Worded rule of Befund 121: case 1 where
    exhaust 1 minus exhaust 2 lies above ``worded_threshold_k``. *Decided* points: the two fuels
    differ by at least ``exact``; *clear* points: by at least the practical tie (``fraction`` of
    the saving of the better of the two against case 0).
    """
    fuel = np.asarray(fuel, dtype=np.float64)
    t = np.asarray(exhaust_celsius, dtype=np.float64)
    c = np.asarray(sink_capacity, dtype=np.float64)
    gap = fuel[:, 2] - fuel[:, 1]
    actual = np.where(gap > 0.0, 1, 2)
    decided = np.abs(gap) >= exact
    saving = fuel[:, 0] - np.minimum(fuel[:, 1], fuel[:, 2])
    clear = np.abs(gap) >= np.maximum(fraction * saving, exact)
    generalised = np.where((t[:, 0] > t[:, 1]) == (c[:, 0] > c[:, 1]), 1, 2)
    worded = np.where(t[:, 0] - t[:, 1] > worded_threshold_k, 1, 2)

    def rate(prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        n = int(mask.sum())
        hits = int((prediction[mask] == actual[mask]).sum())
        return {"n": n, "hits": hits, "rate": hits / n if n else None}

    differ = generalised != worded
    return {
        "points": len(fuel),
        "case_1_better": int((actual[decided] == 1).sum()),
        "generalised": {"decided": rate(generalised, decided), "clear": rate(generalised, clear)},
        "worded": {"decided": rate(worded, decided), "clear": rate(worded, clear)},
        "rules_differ": {
            "decided_points": int((differ & decided).sum()),
            "generalised_right": int((differ & decided & (generalised == actual)).sum()),
            "worded_right": int((differ & decided & (worded == actual)).sum()),
        },
        "sink_1_larger_share": float((c[:, 0] > c[:, 1]).mean()) if len(c) else None,
        "worded_threshold_k": worded_threshold_k,
    }


# ------------------------------------------------------------------------------------ plant sheet


@dataclass(frozen=True)
class PlantSheet:
    """Inlet block and case table of the plant sheet of the delivery of 2026-11-19.

    ``lines``: the processes in the order of their numbers; ``medium_kg_per_h``: the circuit flow
    the sheet states per process; ``fuel`` / ``recovered``: case -> source (``"excel"`` or
    ``"dyssol"``) -> one value per process (``None`` where the sheet has none); ``total``: case ->
    source -> the sheet's total fuel.
    """

    lines: tuple[dict[str, float], ...]
    fuel: Mapping[int, Mapping[str, tuple[float | None, ...]]]
    recovered: Mapping[int, Mapping[str, tuple[float | None, ...]]]
    total: Mapping[int, Mapping[str, float | None]]

    def line_inputs(self, process: int, equilibrium_moisture_g_per_kg: float) -> LineInputs:
        """Inputs of one process (0-based); the sheet's gas flow is the dry gas."""
        raw = self.lines[process]
        return LineInputs(
            dry_gas_kg_per_h=raw["gas_kg_per_h"],
            gas_humidity_g_per_kg=raw["gas_humidity_g_per_kg"],
            gas_celsius=raw["gas_celsius"],
            dryer_inlet_celsius=raw["dryer_inlet_celsius"],
            spray_kg_per_h=raw["spray_kg_per_h"],
            solid_fraction=raw["solid_fraction"],
            spray_celsius=raw["spray_celsius"],
            medium_kg_per_h=raw["medium_kg_per_h"],
            equilibrium_moisture_g_per_kg=equilibrium_moisture_g_per_kg,
            heater_efficiency_percent=raw["heater_efficiency_percent"],
            calorific_value_kj_per_kg=raw["calorific_value_kj_per_kg"],
        )


_INLET = {
    "gas_kg_per_h": ("heat recovery", "fresh gas", "mass flow rate", "kg/hr"),
    "gas_humidity_g_per_kg": ("heat recovery", "fresh gas", "moisture content", "gm/kg"),
    "gas_celsius": ("heat recovery", "fresh gas", "temperature", "c"),
    "medium_kg_per_h": ("heating transfer medium", "", "flow rate", "kg/hr"),
    "dryer_gas_kg_per_h": ("dryer", "process air", "mass flow rate", "kg/hr"),
    "dryer_humidity_g_per_kg": ("dryer", "process air", "moisture content", "gm/kg"),
    "dryer_inlet_celsius": ("dryer", "process air", "temperature", "c"),
    "spray_kg_per_h": ("dryer", "spray rate", "mass flow rate", "kg/hr"),
    "solid_fraction": ("dryer", "spray rate", "solid fraction", "-"),
    "spray_celsius": ("dryer", "spray rate", "temperature", "c"),
    "heater_efficiency_percent": ("heater", "efficiency", "", "%"),
    "calorific_value_kj_per_kg": ("heater", "calorific value", "", "kj/kg"),
}
_CASE = re.compile(r"^case\s+(\d)$", re.IGNORECASE)
_RESULT_COLUMN = 10
_NAME_COLUMN = 11


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def parse_plant_sheet(rows: Sequence[Sequence[Any]]) -> PlantSheet:
    """Parse the stored values of the plant sheet (rows as ``openpyxl`` returns them, from ``A1``).

    Layout: ``Process n`` in column ``A`` and the name of the process in ``B``; component,
    stream, parameter, unit and value of the inlet block in ``C`` to ``G``; the case table from
    column ``K``: the quantity in ``K``, the name of the process in ``L``, one column per case
    and source, headed by ``Case n`` and, one row below, ``Excel`` / ``Dyssol``. Rows are found
    by their labels; a missing label or an unexpected unit raises.
    """
    width = max(len(row) for row in rows)
    padded = [list(row) + [None] * (width - len(row)) for row in rows]

    names: dict[str, int] = {}
    raw: dict[int, dict[tuple[str, str, str], tuple[float, str]]] = {}
    process = 0
    component = stream = ""
    for cells in padded:
        head = _text(cells[0]).lower().split()
        if len(head) == 2 and head[0] == "process" and head[1].isdigit():
            process, component, stream = int(head[1]), "", ""
            name = _text(cells[1])
            if not name or name in names:
                raise ValueError(f"process {process}: missing or repeated name in column B")
            names[name] = process
            raw[process] = {}
        elif _text(cells[0]):
            process = 0
        if not process:
            continue
        if _text(cells[2]):
            component, stream = _text(cells[2]).lower(), ""
        if _text(cells[3]):
            stream = _text(cells[3]).lower()
        unit, value = _text(cells[5]), _number(cells[6])
        if unit and value is not None:
            raw[process][(component, stream, _text(cells[4]).lower())] = (value, unit)
    processes = sorted(raw)
    if processes != list(range(1, len(processes) + 1)) or not processes:
        raise ValueError("processes are not numbered 1..n")

    lines = []
    for p in processes:
        values: dict[str, float] = {}
        for field, (comp, strm, parameter, unit) in _INLET.items():
            key = (comp, strm, parameter)
            if key not in raw[p]:
                raise ValueError(f"process {p}: no input {key}")
            value, found = raw[p][key]
            found = found.replace(" ", "").lower()
            if (unit == "c" and not found.endswith("c")) or (unit != "c" and found != unit):
                raise ValueError(f"process {p}: {key} in {found!r}, expected {unit!r}")
            values[field] = value
        if (
            values.pop("dryer_gas_kg_per_h") != values["gas_kg_per_h"]
            or values.pop("dryer_humidity_g_per_kg") != values["gas_humidity_g_per_kg"]
        ):
            raise ValueError(f"process {p}: fresh gas and process air differ in flow or moisture")
        lines.append(values)

    columns: dict[int, tuple[int, str]] = {}
    for index, cells in enumerate(padded[:-1]):
        found = {j: _CASE.match(_text(c)) for j, c in enumerate(cells)}
        heads = {j: int(m.group(1)) for j, m in found.items() if m}
        if not heads:
            continue
        below = padded[index + 1]
        case = 0
        for j in range(min(heads), width):
            case = heads.get(j, case)
            source = _text(below[j]).lower()
            if source in ("excel", "dyssol"):
                columns[j] = (case, source)
        break
    if not columns:
        raise ValueError("no header row of the cases")

    table: dict[str, dict[int, dict[str, list[float | None]]]] = {"fuel": {}, "recovered": {}}
    total: dict[int, dict[str, float | None]] = {}
    current: str | None = None
    for cells in padded:
        label = _text(cells[_RESULT_COLUMN]).lower()
        if label:
            if "total fuel" in label:
                current = "total"
            elif "fuel consumption" in label:
                current = "fuel"
            elif "recovery" in label:
                current = "recovered"
            else:
                current = None
        if current is None:
            continue
        name = _text(cells[_NAME_COLUMN])
        if current == "total":
            if label:
                for j, (case, source) in columns.items():
                    total.setdefault(case, {})[source] = _number(cells[j])
            continue
        if name not in names:
            continue
        for j, (case, source) in columns.items():
            slot = table[current].setdefault(case, {}).setdefault(source, [None] * len(processes))
            slot[names[name] - 1] = _number(cells[j])
    for key in ("fuel", "recovered"):
        if not table[key]:
            raise ValueError(f"no rows of {key}")
    for case, sources in total.items():
        for source in sources:  # a total without both heater values is a sum over empty cells
            if any(v is None for v in table["fuel"][case][source]):
                sources[source] = None
    return PlantSheet(
        lines=tuple(lines),
        fuel={c: {s: tuple(v) for s, v in d.items()} for c, d in table["fuel"].items()},
        recovered={c: {s: tuple(v) for s, v in d.items()} for c, d in table["recovered"].items()},
        total=total,
    )


def within_rounding(ours: float, reported: float, decimals: int, halves: int = 1) -> bool:
    """``ours`` equals ``reported`` within ``halves`` half units of its last reported decimal."""
    return abs(ours - reported) <= halves * 0.5 * 10.0**-decimals + 1e-12


# ------------------------------------------------------------------------------------ records


def _rule_entry(result: ModelResult, exact: float) -> dict[str, Any]:
    predicted = generalised_rule_case(result.exhaust_celsius, result.sink_capacity_kw_per_k)
    actual = pair_choice(result, exact)
    return {
        "exhaust_1_warmer": result.exhaust_celsius[0] > result.exhaust_celsius[1],
        "sink_1_larger": result.sink_capacity_kw_per_k[0] > result.sink_capacity_kw_per_k[1],
        "generalised_rule_case": predicted,
        "better_of_cases_1_and_2": actual,
        "generalised_rule_hit": None
        if predicted is None or actual is None
        else predicted == actual,
        "worded_rule_predicts_case_1": worded_rule_predicts_case_1(result),
    }


def summarise(
    *,
    results: Mapping[str, Mapping[str, Mapping[str, ModelResult]]],
    read_points: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    grid_settings: Sequence[str],
    band: Sequence[str],
    exact: float,
    fraction: float,
    sensitivity: Sequence[float],
    predictions: Mapping[str, Any],
    points: Mapping[str, Point],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Readings and reported numbers: the record for the repository and the local record.

    ``results[tag][setting][model]``; ``read_points``: the tags the readings use (setting
    :data:`READ_SETTING`); ``groups``: label -> read points (R, each day); ``grid_settings``: the
    settings with both circuits at one flow of the sweep sheet; ``band``: the labels of the
    solid fractions of the band; ``predictions``: ``{"F": {model: code}, "C": {model: code},
    "A": code}``.
    """

    def at(tag: str, model: str, setting: str = READ_SETTING) -> ModelResult:
        return results[tag][setting][model]

    per_model: dict[str, Any] = {}
    found_f: dict[str, str] = {}
    found_c: dict[str, str] = {}
    for model in MODELS:
        entries = {tag: case_one(at(tag, model), exact, fraction) for tag in read_points}
        cells = [c.admissible for tag in read_points for c in at(tag, model).cases[1:]]
        flags = [entries[tag]["best_or_practically_equal"] for tag in read_points]
        found_f[model] = reading_f(cells)
        found_c[model] = reading_c(flags)
        rules = {tag: _rule_entry(at(tag, model), exact) for tag in (*read_points, "X")}
        hits = [r["generalised_rule_hit"] for r in rules.values()]
        per_model[model] = {
            "reading_f": {
                "code": found_f[model],
                "cells": len(cells),
                "admissible_cells": sum(cells),
                "admissible_cells_per_case": {
                    str(k): sum(at(tag, model).cases[k].admissible for tag in read_points)
                    for k in CASE_NUMBERS
                },
            },
            "reading_c": {
                "code": found_c[model],
                "points": len(flags),
                "case_1_best_or_practically_equal": sum(flags),
                "per_group": {
                    label: reading_c([entries[tag]["best_or_practically_equal"] for tag in tags])
                    for label, tags in groups.items()
                },
                "sensitivity": {
                    f"{s:g}": reading_c(
                        [
                            case_one(at(tag, model), exact, s)["best_or_practically_equal"]
                            for tag in read_points
                        ]
                    )
                    for s in sensitivity
                },
                "best_case_per_point": {tag: entries[tag]["best_case"] for tag in read_points},
            },
            "points": {
                tag: {**entries[tag], "saturation": list(at(tag, model).saturation)}
                for tag in read_points
            },
            "worded_rule": {
                "threshold_k": WORDED_RULE_THRESHOLD_K[model],
                "points_where_it_predicts_case_1": sum(
                    rules[tag]["worded_rule_predicts_case_1"] for tag in read_points
                ),
                "points_where_it_predicts_case_1_and_case_1_is_best_or_equal": sum(
                    rules[tag]["worded_rule_predicts_case_1"]
                    and entries[tag]["best_or_practically_equal"]
                    for tag in read_points
                ),
            },
            "generalised_rule": {
                "status": "post-hoc hypothesis of the project lead (2026-11-25), no reading",
                "points": len(hits),
                "decided_points": sum(h is not None for h in hits),
                "hits": sum(h is True for h in hits),
                "per_point": rules,
            },
            "exchanged_sizes": {
                "X": case_one(at("X", model), exact, fraction),
                "X_scaled_circuits": case_one(at("X|scaled", model), exact, fraction),
                "measured_points_with_case_1_best_or_practically_equal": sum(
                    case_one(at(f"X|{tag}", model), exact, fraction)["best_or_practically_equal"]
                    for tag in read_points
                    if tag.startswith("M|")
                ),
            },
            "band": {
                label: {
                    "reading_f": reading_f(
                        [
                            c.admissible
                            for tag in read_points
                            for c in at(f"{tag}|sf={label}", model).cases[1:]
                        ]
                    ),
                    "reading_c": reading_c(
                        [
                            case_one(at(f"{tag}|sf={label}", model), exact, fraction)[
                                "best_or_practically_equal"
                            ]
                            for tag in read_points
                        ]
                    ),
                    "best_cases": sorted(
                        {
                            str(
                                case_one(at(f"{tag}|sf={label}", model), exact, fraction)[
                                    "best_case"
                                ]
                            )
                            for tag in read_points
                        }
                    ),
                }
                for label in band
            },
            "grid": {
                tag: {
                    "best_case_per_flow": [
                        case_one(at(tag, model, setting), exact, fraction)["best_case"]
                        for setting in grid_settings
                    ],
                    "case_1_best_or_practically_equal_per_flow": [
                        case_one(at(tag, model, setting), exact, fraction)[
                            "best_or_practically_equal"
                        ]
                        for setting in grid_settings
                    ],
                }
                for tag in (*read_points, "X")
            },
            "as_delivered": {
                tag: {
                    "saturation": list(at(tag, model).saturation),
                    "dryers_admissible": any(c.admissible for c in at(tag, model).cases[1:]),
                    "order_of_the_formula": sorted(
                        CASE_NUMBERS, key=lambda k, t=tag: at(t, model).fuel()[k]
                    ),
                    "saving_percent": {
                        str(k): 100.0
                        * (at(tag, model).fuel()[0] - at(tag, model).fuel()[k])
                        / at(tag, model).fuel()[0]
                        for k in CASE_NUMBERS
                    },
                    "negative_stages": {
                        str(c.case): c.negative_stages for c in at(tag, model).cases[1:]
                    },
                }
                for tag in ("S",)
            },
        }

    reference = {
        m: per_model[m]["points"]["R"]["best_or_practically_equal"] for m in predictions["A_models"]
    }
    exchanged = {
        m: per_model[m]["exchanged_sizes"]["X"]["best_or_practically_equal"]
        for m in predictions["A_models"]
    }
    found_a = reading_a(reference, exchanged)
    checks = {
        "F": {
            m: {
                "predicted": predictions["F"][m],
                "found": found_f[m],
                "confirmed": predictions["F"][m] == found_f[m],
            }
            for m in MODELS
        },
        "C": {
            m: {
                "predicted": predictions["C"][m],
                "found": found_c[m],
                "confirmed": predictions["C"][m] == found_c[m],
            }
            for m in MODELS
        },
        "A": {
            "predicted": predictions["A"],
            "found": found_a,
            "confirmed": predictions["A"] == found_a,
        },
    }
    repo: dict[str, Any] = {
        "test_set": {
            "read_points": list(read_points),
            "groups": {k: list(v) for k, v in groups.items()},
        },
        "tie": {
            "exact_kg_per_h": exact,
            "practical_fraction": fraction,
            "sensitivity": list(sensitivity),
        },
        "headline_model": MODEL_U,
        "models": per_model,
        "reading_a": {
            "code": found_a,
            "models": list(predictions["A_models"]),
            "at_R": reference,
            "at_X": exchanged,
        },
        "predictions": checks,
    }
    local: dict[str, Any] = {
        "points": {tag: [asdict(line) for line in point] for tag, point in points.items()},
        "results": {
            tag: {
                setting: {model: asdict(result) for model, result in by_model.items()}
                for setting, by_model in by_setting.items()
            }
            for tag, by_setting in results.items()
        },
    }
    return repo, local
