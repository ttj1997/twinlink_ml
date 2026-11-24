"""Reader of a sweep sheet of the stationary flowsheets, and its comparison with a source.

The sheet (one sheet of a results workbook; delivery of 2026-11-24) varies the flow of the heat
transfer medium of the heat recovery for two processes, each one line whose exhaust preheats its
own fresh gas ("case 1" of the six-case network, docs/11 section 13). It has

* an input block, columns ``A``-``G``: ``Process 1`` / ``Process 2`` in ``A``, the name of the
  process in ``B``, the component in ``C`` (``Heat recovery``, ``Dryer``, ``Heater``), the stream in
  ``D``, the parameter in ``E``, the unit in ``F``, the value in ``G``;
* a result block, columns ``I``-``V``: the quantity in ``I``, the name of the process in ``J``, the
  side of the medium (``Hot side`` / ``Cold side``) in ``K``, and one column per medium flow from
  ``L`` on; the header row holds the flows in kg/h, the first one as the text ``No flow``;
* two further inputs below the blocks: the equilibrium moisture and whether condensation is on.

Rows are found by their labels; a missing label, a unit that is not the expected one or an
incomplete result raises. The names of the processes are used only to assign the result rows to
``Process 1`` / ``Process 2``; they are not kept. Every result is stored with
:data:`REPORTED_DECIMALS` decimals; a value with more raises (the comparison depends on it).

**The sheet does not say whether the gas mass flow is the dry gas or the humid total**; both
readings are offered (:data:`GAS_FLOW_READINGS`) and the comparison decides.

Confidential partner material, never in the repository; the reader takes a path.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reference import KELVIN, LineInputs, LineOutputs

__all__ = [
    "GAS_FLOW_READINGS",
    "REPORTED_DECIMALS",
    "RESULT_KEYS",
    "SheetLine",
    "SweepSheet",
    "compare",
    "line_inputs",
    "load_sweep_sheet",
    "parse_sweep_rows",
    "sweep_table",
    "totals",
    "verdict",
]

RESULT_KEYS: tuple[str, ...] = (
    "product_celsius",
    "exhaust_celsius",
    "recovered_kw",
    "medium_hot_celsius",
    "medium_cold_celsius",
    "fuel_kg_per_h",
)
REPORTED_DECIMALS = 1
"""Decimals of every result of the sheet."""
GAS_FLOW_READINGS: tuple[str, ...] = ("dry", "humid_total")
"""Readings of the sheet's gas mass flow: the dry gas, or dry gas plus the stated moisture."""

_FLOW_HEADER = "medium flow rate"
_RESULT_LABELS = (
    ("product stream temperature", "product_celsius", "c)"),
    ("exhaust gas temperature", "exhaust_celsius", "c)"),
    ("recovery", "recovered_kw", "kw"),
    ("heat transfer media temperature", "medium", "c)"),
    ("fuel consumption", "fuel_kg_per_h", "kg/h"),
)
_FIRST_FLOW_COLUMN = 11
_MOISTURE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*gm?/kg\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class SheetLine:
    """Inputs of one process as the sheet states them (temperatures converted to degC)."""

    gas_kg_per_s: float
    gas_water_kg_per_s: float
    gas_celsius: float
    dryer_inlet_celsius: float
    spray_kg_per_s: float
    spray_solid_kg_per_s: float
    spray_celsius: float
    heater_efficiency_percent: float
    calorific_value_kj_per_kg: float


@dataclass(frozen=True)
class SweepSheet:
    """Inputs of the processes, the medium flows and the reported results."""

    lines: tuple[SheetLine, ...]
    medium_kg_per_h: tuple[float, ...]
    results: Mapping[str, tuple[tuple[float | None, ...], ...]]
    """``key -> process (0-based) -> medium flow``; ``None`` where the sheet has no value."""
    equilibrium_moisture_g_per_kg: float
    condensation: bool


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _reported(value: object) -> float | None:
    if value is None or _text(value) == "-":
        return None
    number = _number(value)
    if number is None:
        raise ValueError(f"not a reported number: {value!r}")
    scale = 10**REPORTED_DECIMALS
    if abs(number * scale - round(number * scale)) > 1e-6:
        raise ValueError(f"{number!r} has more than {REPORTED_DECIMALS} decimal(s)")
    return number


def _result_key(label: str) -> str | None:
    low = label.lower()
    for text, key, unit in _RESULT_LABELS:
        if text in low:
            if unit not in low.replace(" ", ""):
                raise ValueError(f"result label {label!r} without the unit {unit!r}")
            return key
    return None


def _flows(cells: Sequence[object]) -> tuple[float, ...]:
    flows: list[float] = []
    for cell in cells[_FIRST_FLOW_COLUMN:]:
        if cell is None:
            break
        number = 0.0 if _text(cell).lower() == "no flow" else _number(cell)
        if number is None:
            raise ValueError(f"not a medium flow: {cell!r}")
        flows.append(number)
    if not flows:
        raise ValueError("the header row holds no medium flow")
    return tuple(flows)


def _line(raw: Mapping[tuple[str, str, str], tuple[float, str]], process: int) -> SheetLine:
    def get(component: str, stream: str, parameter: str, unit: str) -> float:
        key = (component, stream, parameter)
        if key not in raw:
            raise ValueError(f"process {process}: no input {key}")
        value, found = raw[key]
        found = found.replace(" ", "")
        if unit == "temperature":
            if found == "K":
                return value - KELVIN
            if found.endswith("C"):
                return value
            raise ValueError(f"process {process}: {key} is not a temperature ({found!r})")
        if found.lower() != unit:
            raise ValueError(f"process {process}: {key} in {found!r}, expected {unit!r}")
        return value

    gas = get("heat recovery", "fresh gas", "mass flow rate", "kg/s")
    water = get("heat recovery", "fresh gas", "moisture content", "kg/s")
    same = (
        get("dryer", "process air", "mass flow rate", "kg/s"),
        get("dryer", "process air", "moisture content", "kg/s"),
    )
    if not (
        math.isclose(gas, same[0], rel_tol=1e-12) and math.isclose(water, same[1], rel_tol=1e-12)
    ):
        raise ValueError(f"process {process}: fresh gas and process air differ in flow or moisture")
    return SheetLine(
        gas_kg_per_s=gas,
        gas_water_kg_per_s=water,
        gas_celsius=get("heat recovery", "fresh gas", "temperature", "temperature"),
        dryer_inlet_celsius=get("dryer", "process air", "temperature", "temperature"),
        spray_kg_per_s=get("dryer", "spray rate", "mass flow rate", "kg/s"),
        spray_solid_kg_per_s=get("dryer", "spray rate", "solid fraction", "kg/s"),
        spray_celsius=get("dryer", "spray rate", "temperature", "temperature"),
        heater_efficiency_percent=get("heater", "efficiency", "", "%"),
        calorific_value_kj_per_kg=get("heater", "calorific value", "", "kj/kg"),
    )


def parse_sweep_rows(rows: Sequence[Sequence[Any]]) -> SweepSheet:
    """Parse the stored values of the sheet (rows as ``openpyxl`` returns them, from ``A1``)."""
    width = _FIRST_FLOW_COLUMN + 1
    padded = [list(row) + [None] * max(0, width - len(row)) for row in rows]

    names: dict[str, int] = {}
    for cells in padded:
        head = _text(cells[0]).lower().split()
        if len(head) == 2 and head[0] == "process" and head[1].isdigit():
            name = _text(cells[1])
            if not name or name in names:
                raise ValueError(f"process {head[1]}: missing or repeated name in column B")
            names[name] = int(head[1])
    if sorted(names.values()) != list(range(1, len(names) + 1)) or not names:
        raise ValueError("processes are not numbered 1..n")

    flows: tuple[float, ...] | None = None
    raw: dict[int, dict[tuple[str, str, str], tuple[float, str]]] = {p: {} for p in names.values()}
    results: dict[str, dict[int, tuple[float | None, ...]]] = {}
    other: dict[str, object] = {}
    process = 0
    component = stream = ""
    current: str | None = None
    last_name = ""
    for cells in padded:
        # ---- input block
        a = _text(cells[0])
        head = a.lower().split()
        if len(head) == 2 and head[0] == "process" and head[1].isdigit():
            process, component, stream = int(head[1]), "", ""
        elif a:
            process = 0
            other[a.lower()] = cells[3]
        if process:
            if _text(cells[2]):
                component, stream = _text(cells[2]), ""
            if _text(cells[3]):
                stream = _text(cells[3])
            unit, value = _text(cells[5]), _number(cells[6])
            if unit and value is not None:
                entry = (component.lower(), stream.lower(), _text(cells[4]).lower())
                if entry in raw[process]:
                    raise ValueError(f"process {process}: input {entry} twice")
                raw[process][entry] = (value, unit)
        # ---- result block
        label = _text(cells[8])
        if _FLOW_HEADER in label.lower():
            if "kg/h" not in label.lower():
                raise ValueError(f"medium flows not in kg/h: {label!r}")
            flows = _flows(cells)
            continue
        if label:
            current = _result_key(label)
        name, side = _text(cells[9]), _text(cells[10]).lower()
        if name:
            last_name = name
        if current is None or not (name or side):
            continue
        if flows is None:
            raise ValueError("result row above the header row of the medium flows")
        if last_name not in names:
            raise ValueError("result row of a process that the input block does not name")
        key_name = current
        if current == "medium":
            if not side.startswith(("hot", "cold")):
                raise ValueError(f"medium row without a side: {side!r}")
            key_name = "medium_hot_celsius" if side.startswith("hot") else "medium_cold_celsius"
        per_process = results.setdefault(key_name, {})
        if names[last_name] in per_process:
            raise ValueError(f"result {key_name} of process {names[last_name]} twice")
        values = tuple(
            _reported(v) for v in cells[_FIRST_FLOW_COLUMN : _FIRST_FLOW_COLUMN + len(flows)]
        )
        if len(values) != len(flows):
            raise ValueError(f"result {key_name}: fewer values than medium flows")
        per_process[names[last_name]] = values

    if flows is None:
        raise ValueError("no header row of the medium flows")
    processes = sorted(names.values())
    for key in RESULT_KEYS:
        if sorted(results.get(key, {})) != processes:
            raise ValueError(f"result {key}: processes incomplete")
    moisture = next((v for k, v in other.items() if k.startswith("equilibrium moisture")), None)
    match = _MOISTURE.match(moisture) if isinstance(moisture, str) else None
    if match is None:
        raise ValueError(f"equilibrium moisture not as '<number> g/kg': {moisture!r}")
    condensation = _text(next((v for k, v in other.items() if k.startswith("condensation")), ""))
    if condensation.lower() not in ("yes", "no"):
        raise ValueError(f"condensation is neither yes nor no: {condensation!r}")
    return SweepSheet(
        lines=tuple(_line(raw[p], p) for p in processes),
        medium_kg_per_h=flows,
        results={key: tuple(results[key][p] for p in processes) for key in RESULT_KEYS},
        equilibrium_moisture_g_per_kg=float(match.group(1)),
        condensation=condensation.lower() == "yes",
    )


def load_sweep_sheet(path: str | Path, sheet: str | None = None) -> SweepSheet:
    """Read the sweep sheet (needs ``openpyxl``; stored values; the first sheet by default)."""
    import openpyxl

    book = openpyxl.load_workbook(Path(path), data_only=True)
    if sheet is not None and sheet not in book.sheetnames:
        raise ValueError(f"sheet {sheet!r} missing")
    worksheet = book[sheet] if sheet is not None else book.worksheets[0]
    return parse_sweep_rows(list(worksheet.iter_rows(values_only=True)))


def line_inputs(sheet: SweepSheet, process: int, gas_flow: str) -> LineInputs:
    """Inputs of one process (0-based) without a medium flow, for one reading of the gas flow."""
    if gas_flow not in GAS_FLOW_READINGS:
        raise ValueError(f"unknown reading {gas_flow!r}; known: {GAS_FLOW_READINGS}")
    line = sheet.lines[process]
    dry = line.gas_kg_per_s - (line.gas_water_kg_per_s if gas_flow == "humid_total" else 0.0)
    return LineInputs(
        dry_gas_kg_per_h=dry * 3600.0,
        gas_humidity_g_per_kg=line.gas_water_kg_per_s / dry * 1000.0,
        gas_celsius=line.gas_celsius,
        dryer_inlet_celsius=line.dryer_inlet_celsius,
        spray_kg_per_h=line.spray_kg_per_s * 3600.0,
        solid_fraction=line.spray_solid_kg_per_s / line.spray_kg_per_s,
        spray_celsius=line.spray_celsius,
        medium_kg_per_h=0.0,
        equilibrium_moisture_g_per_kg=sheet.equilibrium_moisture_g_per_kg,
        heater_efficiency_percent=line.heater_efficiency_percent,
        calorific_value_kj_per_kg=line.calorific_value_kj_per_kg,
    )


def sweep_table(outputs: Sequence[LineOutputs]) -> dict[str, tuple[float, ...]]:
    """The sheet's quantities from the results of one line over the medium flows.

    The units give no medium temperature of their own: by equations 4.1 / 4.2 the medium leaves
    the recovery exchanger at the outlet temperature of the exhaust (hot side) and the preheater
    at that of the fresh gas (cold side), so these two stream temperatures stand for it.
    """
    pick: dict[str, Callable[[LineOutputs], float]] = {
        "product_celsius": lambda o: o.dryer_outlet_celsius,
        "exhaust_celsius": lambda o: o.exhaust_celsius,
        "recovered_kw": lambda o: o.heat_recovered_kw,
        "medium_hot_celsius": lambda o: o.exhaust_celsius,
        "medium_cold_celsius": lambda o: o.preheated_celsius,
        "fuel_kg_per_h": lambda o: o.fuel_kg_per_h,
    }
    return {key: tuple(pick[key](o) for o in outputs) for key in RESULT_KEYS}


def compare(
    ours: Sequence[float], reported: Sequence[float | None], decimals: int = REPORTED_DECIMALS
) -> dict[str, float | int]:
    """Counts and deviations ``ours - reported`` over the values the sheet reports.

    ``within_half`` / ``within_one``: at most half a unit / one unit of the last reported decimal.
    ``equal_if_rounded`` / ``equal_if_truncated``: ours rounded (half up) / cut off (towards minus
    infinity) to the reported decimals is the reported value.
    """
    if len(ours) != len(reported):
        raise ValueError("ours and reported differ in length")
    unit = 10.0**-decimals
    pairs = [(o, r) for o, r in zip(ours, reported, strict=True) if r is not None]
    if not pairs:
        raise ValueError("no reported value to compare")
    diff = [o - r for o, r in pairs]
    return {
        "n": len(pairs),
        "within_half": sum(abs(d) <= 0.5 * unit + 1e-9 for d in diff),
        "within_one": sum(abs(d) <= unit + 1e-9 for d in diff),
        "equal_if_rounded": sum(
            abs(math.floor(o / unit + 0.5) * unit - r) < 1e-9 for o, r in pairs
        ),
        "equal_if_truncated": sum(
            abs(math.floor(o / unit + 1e-9) * unit - r) < 1e-9 for o, r in pairs
        ),
        "min_signed": min(diff),
        "max_signed": max(diff),
        "max_abs": max(abs(d) for d in diff),
        "max_relative_percent": max(
            100.0 * abs(d) / max(abs(r), 1.0) for d, (_, r) in zip(diff, pairs, strict=True)
        ),
    }


def totals(per_key: Mapping[str, Mapping[str, float | int]]) -> dict[str, int]:
    """Sums of ``n``, ``within_half`` and ``within_one`` over the quantities."""
    return {
        name: int(sum(entry[name] for entry in per_key.values()))
        for name in ("n", "within_half", "within_one")
    }


def verdict(total: Mapping[str, int]) -> str:
    """``S1`` every value within half a unit, ``S2`` within one unit, ``S3`` otherwise."""
    if total["within_half"] == total["n"]:
        return "S1"
    if total["within_one"] == total["n"]:
        return "S2"
    return "S3"
