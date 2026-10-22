"""Readers for the delivery 4.1_2: the comparison workbook and the stationary Dyssol flowsheet.

``Comparison.xlsx`` has four sheets (``Dryer``, ``Heater``, ``Heat recovery``, ``Combined``),
each with five data sets: an input block (rows 3 onward; data set *k* has its value in column
``E + 4k``, the unit in ``D + 4k``) and result tables (a header in column ``A`` such as
``Heat recovered (kW)``, then ``Data set 1`` .. ``Data set 5`` with the Excel value in column
``C`` and the Dyssol value in column ``D``). Rows are found by their labels, not by position;
a missing label raises.

``Simulation.dflw`` is the HDF5 file of the Dyssol flowsheet (``h5py``, imported only when the
reader is called). :func:`load_dyssol_steady_state` returns the last time point of every stream
(mass flow, temperature, phase fractions, compound fractions per phase) and the parameters and
state variables of every unit.

Both files are confidential partner material, never in the repository; the readers take a path.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DATA_SETS",
    "SHEETS",
    "ComparisonSheet",
    "DyssolStream",
    "DyssolUnit",
    "Reported",
    "load_comparison",
    "load_dyssol_steady_state",
    "reported_decimals",
]

SHEETS = ("Dryer", "Heater", "Heat recovery", "Combined")
DATA_SETS = (1, 2, 3, 4, 5)

_RESULT_HEADERS = {
    "Product flow rate": "product_kg_per_h",
    "Exhaust temperature": "exhaust_celsius",
    "Fuel consumption": "fuel_kg_per_h",
    "Heat recovered": "recovered_kw",
    "Hot side temperature": "medium_hot_celsius",
    "Cold side temperature": "medium_cold_celsius",
    "Product temperature": "product_celsius",
}
_UNITS = {"kg/hr", "gm/kg", "-", "%", "kJ/kg"}
"""Units of the input rows; a temperature is ``degree sign + C`` (the sign is stored as it is)."""


@dataclass(frozen=True)
class Reported:
    """One result of the partner's table: the Excel and the Dyssol value."""

    excel: float
    dyssol: float


@dataclass(frozen=True)
class ComparisonSheet:
    """One sheet: inputs per data set (``(stream, parameter) -> value``) and the results."""

    name: str
    inputs: Mapping[int, Mapping[tuple[str, str], float]]
    """Sheet units: kg/h, g/kg, degC, fraction, percent, kJ/kg."""
    results: Mapping[str, Mapping[int, Reported]] = field(default_factory=dict)

    def value(self, data_set: int, stream: str, parameter: str) -> float:
        try:
            return self.inputs[data_set][(stream, parameter)]
        except KeyError:
            raise KeyError(f"{self.name}: no input {stream!r}/{parameter!r}") from None


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"not a number: {value!r}")
    return float(value)


def _parse_sheet(name: str, rows: list[tuple[Any, ...]]) -> ComparisonSheet:
    inputs: dict[int, dict[tuple[str, str], float]] = {k: {} for k in DATA_SETS}
    results: dict[str, dict[int, Reported]] = {}
    component = stream = None
    current: str | None = None
    for row in rows:
        cells = list(row) + [None] * (23 - len(row))
        a, b, c = _text(cells[0]), _text(cells[1]), _text(cells[2])
        if a is not None and a.startswith("Data set") and current is not None:
            k = int(a.split()[-1])
            results[current][k] = Reported(_number(cells[2]), _number(cells[3]))
            continue
        header = next(
            (key for prefix, key in _RESULT_HEADERS.items() if a and a.startswith(prefix)), None
        )
        if header is not None:
            current = header
            results[current] = {}
            continue
        unit = _text(cells[3])
        is_unit = unit in _UNITS or (unit is not None and len(unit) == 2 and unit.endswith("C"))
        if not is_unit or not isinstance(cells[4], int | float):
            continue
        if a is not None:
            component, stream = a, None
        if c is None:
            key = (component or "", b or "")
        else:
            stream = b or stream or component
            key = (stream or "", c)
        for k in DATA_SETS:
            col = 4 * k
            if _text(cells[col - 1]) != unit:
                raise ValueError(f"{name}: unit of data set {k} for {key} differs from {unit!r}")
            inputs[k][key] = _number(cells[col])
    for key, per_set in results.items():
        if set(per_set) != set(DATA_SETS):
            raise ValueError(f"{name}: result {key} has data sets {sorted(per_set)}")
    return ComparisonSheet(name, inputs, results)


def load_comparison(path: str | Path) -> dict[str, ComparisonSheet]:
    """Read the four sheets of ``Comparison.xlsx`` (needs ``openpyxl``; stored values)."""
    import openpyxl

    book = openpyxl.load_workbook(Path(path), data_only=True)
    missing = [s for s in SHEETS if s not in book.sheetnames]
    if missing:
        raise ValueError(f"sheets missing: {missing}")
    return {s: _parse_sheet(s, list(book[s].iter_rows(values_only=True))) for s in SHEETS}


def reported_decimals(value: float) -> int | None:
    """Decimals a reported value was rounded to (``None``: not rounded, more than 4 decimals)."""
    text = repr(float(value))
    if "e" in text:
        return None
    decimals = len(text.split(".")[1].rstrip("0")) if "." in text else 0
    return decimals if decimals <= 4 else None


@dataclass(frozen=True)
class DyssolStream:
    """Last time point of a Dyssol stream (SI; temperature converted to degC)."""

    name: str
    key: str
    time_s: float
    mass_flow_kg_per_s: float
    temperature_celsius: float
    phase_fractions: Mapping[str, float]
    compound_fractions: Mapping[str, tuple[float, ...]]
    """Per phase, mass fractions in the order of :attr:`DyssolFile.compounds` (-1: undefined)."""


@dataclass(frozen=True)
class DyssolUnit:
    """A Dyssol unit: ports, parameters and the state variables at the end."""

    name: str
    ports: Mapping[str, str]
    """Port name -> stream key."""
    parameters: Mapping[str, tuple[float, ...]]
    state: Mapping[str, float]


@dataclass(frozen=True)
class DyssolFile:
    """Compounds, streams and units of a ``.dflw`` file at its last time point."""

    compounds: tuple[str, ...]
    """Compound keys of the flowsheet (names are not in the file)."""
    streams: Mapping[str, DyssolStream]
    units: Mapping[str, DyssolUnit]

    def vapour_index(self) -> int:
        """The water vapour among the gas compounds.

        Dyssol keeps liquid water and vapour as two compounds; the gas phase of every stream
        holds exactly two compounds, dry air (the larger share) and vapour.
        """
        gases = [
            s.compound_fractions["Gas"]
            for s in self.streams.values()
            if s.phase_fractions.get("Gas", 0.0) > 0.0
        ]
        present = {i for f in gases for i, x in enumerate(f) if x > 0.0}
        if len(present) != 2:
            raise ValueError(f"gas phase holds compounds {sorted(present)}, expected two")
        air = max(present, key=lambda i: sum(f[i] for f in gases))
        return (present - {air}).pop()

    def gas_split(self, stream: str) -> tuple[float, float]:
        """``(dry gas kg/s, humidity kg/kg)`` of the gas phase of a stream."""
        s = self.streams[stream]
        w = self.vapour_index()
        gas = s.mass_flow_kg_per_s * s.phase_fractions["Gas"]
        x_w = s.compound_fractions["Gas"][w]
        dry = gas * (1.0 - x_w)
        return dry, gas * x_w / dry


def _last(data: Any) -> tuple[float, float]:
    times, values = data[0], data[1]
    if len(times) == 0:
        return math.nan, math.nan
    return float(times[-1]), float(values[-1])


def load_dyssol_steady_state(path: str | Path) -> DyssolFile:
    """Last time point of every stream and the units of a ``.dflw`` file (needs ``h5py``)."""
    import h5py

    def decode(x: Any) -> str:
        return x.decode() if isinstance(x, bytes) else str(x)

    def ordered(group: Any, prefix: str) -> list[str]:
        names = [n for n in group if re.fullmatch(rf"{prefix}\d+", n)]
        return sorted(names, key=lambda n: int(n[len(prefix) :]))

    with h5py.File(Path(path), "r") as f:
        fs = f["Flowsheet"]
        compounds = tuple(decode(k) for k in fs["DistrGrid/Grid0/StrGrid"][()])
        streams: dict[str, DyssolStream] = {}
        for g in ordered(fs["Streams"], "Stream"):
            s = fs["Streams"][g]
            t, mass = _last(s["Overalls/Overall0/Data"][()])
            _, kelvin = _last(s["Overalls/Overall1/Data"][()])
            phases: dict[str, float] = {}
            fractions: dict[str, tuple[float, ...]] = {}
            for p in ordered(s["Phases"], "Phase"):
                ph = s["Phases"][p]
                name = decode(ph["Name"][0])
                phases[name] = _last(ph["Fractions/Data"][()])[1]
                dist = ph["Distribution"]
                data = ordered(dist, "Data")
                fractions[name] = tuple(float(x[-1]) for x in dist[data[-1]][()]) if data else ()
            name = decode(s["StreamName"][0])
            streams[name] = DyssolStream(
                name, decode(s["StreamKey"][0]), t, mass, kelvin - 273.15, phases, fractions
            )
        units: dict[str, DyssolUnit] = {}
        for g in ordered(fs["Models"], "Model"):
            m = fs["Models"][g]
            u = m["Unit"]
            ports = {
                decode(u[f"Ports/{p}/Name"][0]): decode(u[f"Ports/{p}/StreamKey"][0])
                for p in ordered(u["Ports"], "Port")
            }
            params: dict[str, tuple[float, ...]] = {}
            if "UnitParameters" in u and "ParamsNames" in u["UnitParameters"]:
                for i, n in enumerate(u["UnitParameters/ParamsNames"][()]):
                    values = u["UnitParameters"].get(f"UnitParameter{i}/Values")
                    params[decode(n)] = tuple(float(v) for v in values[()]) if values else ()
            state: dict[str, float] = {}
            if "StateVariables" in u:
                for v in ordered(u["StateVariables"], "StateVariable"):
                    sv = u["StateVariables"][v]
                    state[decode(sv["Name"][0])] = float(sv["Value"][0])
            name = decode(m["UnitName"][0])
            units[name] = DyssolUnit(name, ports, params, state)
    return DyssolFile(compounds, streams, units)
