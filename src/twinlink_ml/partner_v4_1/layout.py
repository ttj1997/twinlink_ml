"""Where each quantity sits in the sheets ``Scenarios`` and ``Balances`` of ``Model_v4.1.xlsx``.

Both the as-built reproduction (:mod:`.as_built`) and the document-based model
(:mod:`.cascade`) fill the same neutral containers (:class:`CaseCells`, :class:`PlantCells`,
sheet units: degC, kg/h, g/kg, kW); this module turns them into ``{address: value}``, so that
the two can be compared cell by cell with each other and with the values Excel stored.

The layout is read from the workbook (cell formulas), not guessed:

* ``Scenarios`` rows 3-7 / 9-13: exhaust 1 / 2 and its circuit (``T2`` in the outlet column,
  ``T1`` in row 7 / 13), rows 15-26: sink 1 (stage "low" rows 15-20, stage "high" rows 21-24,
  heater rows 25-26), rows 28-39: sink 2 (low 28-33, high 34-37, heater 38-39), row 41: total.
* Case ``n`` occupies the column pair ``(inlet, outlet)`` of :data:`CASE_COLUMNS`; its two
  circuits sit in the ``Balances`` columns of :data:`BALANCES_COLUMNS` (recovery 1, 2).
* Case structure (order of the sources into each sink): the six case drawings of the equation
  document, identical to assumption A1 of :mod:`twinlink_ml.partner_v4.cascade`.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinlink_ml.partner_v4.cascade import CASE_COLUMNS, CASES

__all__ = [
    "BALANCES_COLUMNS",
    "CASES",
    "CASE_COLUMNS",
    "CaseCells",
    "PlantCells",
    "SinkCells",
    "StageCells",
    "scenario_cells",
]

BALANCES_COLUMNS: dict[int, tuple[str, str]] = {
    1: ("I", "J"),
    2: ("K", "L"),
    3: ("M", "N"),
    4: ("O", "P"),
    5: ("Q", "R"),
    6: ("S", "T"),
}
"""Case -> ``Balances`` columns of (recovery 1, recovery 2), the intermediate variables K1..K6."""

# (air, moisture, temperature) rows of the sink's own air; stage rows (flow, T2/T1, energy) for
# the low and the high stage; heater rows (temperatures, gas).
_SINK_ROWS = {
    1: {
        "air": 15,
        "y": 16,
        "t": 17,
        "low": (18, 19, 20),
        "high": (21, 22, 23, 24),
        "heater": 25,
        "gas": 26,
    },
    2: {
        "air": 28,
        "y": 29,
        "t": 30,
        "low": (31, 32, 33),
        "high": (34, 35, 36, 37),
        "heater": 38,
        "gas": 39,
    },
}
_SOURCE_ROWS = {1: (3, 4, 5, 6, 7), 2: (9, 10, 11, 12, 13)}


@dataclass(frozen=True)
class StageCells:
    """One pre-heating stage in sheet units."""

    circuit: int
    """Exhaust (and circuit) number, 1 or 2."""
    sink_in_celsius: float
    sink_out_celsius: float
    """``T1``."""
    htm_hot_celsius: float
    """``T2``."""
    htm_flow_kg_per_h: float
    energy_kw: float


@dataclass(frozen=True)
class SinkCells:
    """One sink: its stages in order, the heater."""

    stages: tuple[StageCells, ...]
    heater_in_celsius: float
    heater_out_celsius: float
    gas_kg_per_h: float


@dataclass(frozen=True)
class CaseCells:
    """One case: sink 1 and sink 2."""

    sink_1: SinkCells
    sink_2: SinkCells


@dataclass(frozen=True)
class PlantCells:
    """Case-independent numbers that the sheet repeats in every case column (sheet units)."""

    air_kg_per_h: tuple[float, float]
    inlet_moisture_g_per_kg: tuple[float, float]
    exhaust_moisture_g_per_kg: tuple[float, float]
    exhaust_celsius: tuple[float, float]
    ambient_celsius: float


def scenario_cells(plant: PlantCells, cases: dict[int, CaseCells]) -> dict[str, float]:
    """All numeric cells of the sheet ``Scenarios`` for the given cases (``{address: value}``)."""
    cells: dict[str, float] = {}
    for case, cc in cases.items():
        ci, co = CASE_COLUMNS[case]
        stages = {s.circuit: s for sink in (cc.sink_1, cc.sink_2) for s in sink.stages}
        for i, (r_air, r_y, r_t, r_flow, r_cold) in _SOURCE_ROWS.items():
            stage = stages[i]
            cells[f"{ci}{r_air}"] = plant.air_kg_per_h[i - 1]
            cells[f"{ci}{r_y}"] = plant.exhaust_moisture_g_per_kg[i - 1]
            cells[f"{ci}{r_t}"] = plant.exhaust_celsius[i - 1]
            cells[f"{co}{r_t}"] = stage.htm_hot_celsius
            cells[f"{ci}{r_flow}"] = stage.htm_flow_kg_per_h
            cells[f"{ci}{r_cold}"] = stage.sink_out_celsius
            cells[f"{co}{r_cold}"] = stage.htm_hot_celsius
        for sink_no, sink in ((1, cc.sink_1), (2, cc.sink_2)):
            rows = _SINK_ROWS[sink_no]
            cells[f"{ci}{rows['air']}"] = plant.air_kg_per_h[sink_no - 1]
            cells[f"{ci}{rows['y']}"] = plant.inlet_moisture_g_per_kg[sink_no - 1]
            cells[f"{ci}{rows['t']}"] = plant.ambient_celsius
            first_out = sink.stages[0].sink_out_celsius if sink.stages else plant.ambient_celsius
            cells[f"{co}{rows['t']}"] = first_out
            for key, stage in zip(("low", "high"), sink.stages, strict=False):
                if key == "low":
                    r_flow, r_t, r_e = rows["low"]
                    cells[f"{ci}{r_flow}"] = stage.htm_flow_kg_per_h
                    cells[f"{ci}{r_t}"] = stage.htm_hot_celsius
                    cells[f"{co}{r_t}"] = stage.sink_out_celsius
                    cells[f"{ci}{r_e}"] = stage.energy_kw
                else:
                    r_in, r_flow, r_t, r_e = rows["high"]
                    cells[f"{ci}{r_in}"] = stage.sink_in_celsius
                    cells[f"{co}{r_in}"] = stage.sink_out_celsius
                    cells[f"{ci}{r_flow}"] = stage.htm_flow_kg_per_h
                    cells[f"{ci}{r_t}"] = stage.htm_hot_celsius
                    cells[f"{co}{r_t}"] = stage.sink_out_celsius
                    cells[f"{ci}{r_e}"] = stage.energy_kw
            cells[f"{ci}{rows['heater']}"] = sink.heater_in_celsius
            cells[f"{co}{rows['heater']}"] = sink.heater_out_celsius
            cells[f"{ci}{rows['gas']}"] = sink.gas_kg_per_h
        cells[f"{ci}41"] = cc.sink_1.gas_kg_per_h + cc.sink_2.gas_kg_per_h
    return cells
