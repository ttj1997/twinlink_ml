"""The six cases of ``Model_v4.1.xlsx`` computed from the equations of the document.

Every pre-heating stage is a circuit of a heat transfer medium between an exhaust and the
sink's process gas, solved with (4.1)/(4.2) (:func:`~.loops.equal_outlet_loop`); the heater
follows from (3.2). Everything is SI inside; the mass flows of the circuits are inputs
(``Inputs!E20``, ``E21``) and can be changed (:meth:`.ExcelInputsV41.with_htm_flow`).

Case structure (document drawings, blue = exhaust 1, red = exhaust 2; "low" = first stage):

====  ======================================  ======================================
case  sink 1 (inlet air of dryer 1)            sink 2 (inlet air of dryer 2)
====  ======================================  ======================================
1     exhaust 1                                exhaust 2
2     exhaust 2                                exhaust 1
3     low: exhaust 2, high: exhaust 1          none
4     none                                     low: exhaust 2, high: exhaust 1
5     low: exhaust 1, high: exhaust 2          none
6     none                                     low: exhaust 1, high: exhaust 2
====  ======================================  ======================================

``evaluate_case`` accepts an ``order`` to try other stage orders (used for the reversed order of
case 6); ``mode="counter_current"`` evaluates the same structure with the counter-current
bound of :mod:`.loops` (a comparison, not the document's model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from twinlink_ml.partner_v4 import corrected as c

from . import loops
from .inputs import SiInputsV41
from .layout import BALANCES_COLUMNS, CASES, CaseCells, PlantCells, SinkCells, StageCells

__all__ = [
    "CASES",
    "CaseResult",
    "Mode",
    "Order",
    "Rule",
    "SinkResult",
    "StageResult",
    "balances_cells",
    "evaluate_case",
    "plant_cells",
    "to_case_cells",
]

Mode = Literal["equal_outlet", "counter_current"]
Rule = Literal["drawn", "by_temperature", "bypass", "by_temperature_and_bypass"]
Order = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class StageResult:
    """One pre-heating stage."""

    source: int
    sink_in_celsius: float
    sink_out_celsius: float
    exhaust_in_celsius: float
    exhaust_out_celsius: float
    htm_flow_kg_per_s: float
    recovered_w: float
    exhaust_dew_point_celsius: float

    @property
    def condenses(self) -> bool:
        """Exhaust leaves below its dew point (the document's (4.1) would need condensation)."""
        return self.exhaust_out_celsius < self.exhaust_dew_point_celsius


@dataclass(frozen=True)
class SinkResult:
    """One sink: stages, heater, gas."""

    stages: tuple[StageResult, ...]
    heater_in_celsius: float
    heater_out_celsius: float
    heater_power_w: float
    gas_kg_per_s: float


@dataclass(frozen=True)
class CaseResult:
    """One case."""

    case: int
    sink_1: SinkResult
    sink_2: SinkResult

    @property
    def stages(self) -> tuple[StageResult, ...]:
        """All stages of both sinks."""
        return self.sink_1.stages + self.sink_2.stages

    @property
    def total_gas_kg_per_h(self) -> float:
        """Gas of both heaters [kg/h]."""
        return (self.sink_1.gas_kg_per_s + self.sink_2.gas_kg_per_s) * 3600.0

    @property
    def total_recovered_w(self) -> float:
        """Sum of the heat recovered in all stages [W]."""
        return sum(s.recovered_w for s in self.stages)


def _exhausts(inputs: SiInputsV41) -> dict[int, c.Stream]:
    m = inputs.plant.material
    return {
        1: c.exhaust_stream(inputs.plant.dryer_1, c.dryer_balance(inputs.plant.dryer_1, m)),
        2: c.exhaust_stream(inputs.plant.dryer_2, c.dryer_balance(inputs.plant.dryer_2, m)),
    }


def _sink(
    inputs: SiInputsV41,
    sink_no: int,
    sources: tuple[int, ...],
    exhausts: dict[int, c.Stream],
    mode: Mode,
    rule: Rule = "drawn",
) -> SinkResult:
    plant = inputs.plant
    m = plant.material
    dryer = plant.dryer_1 if sink_no == 1 else plant.dryer_2
    gas = c.Stream(
        dryer.air_mass_flow_kg_per_s,
        dryer.air_humidity_kg_per_kg,
        plant.ambient.temperature_celsius,
    )
    solve = (
        loops.equal_outlet_loop
        if mode == "equal_outlet"
        else loops.counter_current_infinite_area_loop
    )
    if rule in ("by_temperature", "by_temperature_and_bypass"):
        sources = tuple(sorted(sources, key=lambda i: exhausts[i].temperature_celsius))
    stages = []
    for source in sources:
        exhaust = exhausts[source]
        flow = inputs.htm_flow_kg_per_s[source - 1]
        if rule in ("bypass", "by_temperature_and_bypass") and (
            exhaust.temperature_celsius <= gas.temperature_celsius
        ):
            result = loops.LoopResult(gas.temperature_celsius, exhaust.temperature_celsius, 0.0)
        else:
            result = solve(exhaust, gas, flow, inputs.cp_htm_j_per_kg_k, m)
        stages.append(
            StageResult(
                source=source,
                sink_in_celsius=gas.temperature_celsius,
                sink_out_celsius=result.sink_out_celsius,
                exhaust_in_celsius=exhaust.temperature_celsius,
                exhaust_out_celsius=result.exhaust_out_celsius,
                htm_flow_kg_per_s=flow,
                recovered_w=result.recovered_w,
                exhaust_dew_point_celsius=loops.exhaust_dew_point_celsius(
                    exhaust, plant.ambient.pressure_pa
                ),
            )
        )
        gas = c.Stream(gas.dry_air_kg_per_s, gas.humidity_kg_per_kg, result.sink_out_celsius)
    target = dryer.air_temperature_celsius
    return SinkResult(
        stages=tuple(stages),
        heater_in_celsius=gas.temperature_celsius,
        heater_out_celsius=target,
        heater_power_w=c.heater_power_w(gas, gas.temperature_celsius, target, m),
        gas_kg_per_s=loops.heater_gas_kg_per_s(gas, target, m),
    )


def evaluate_case(
    inputs: SiInputsV41,
    case: int,
    *,
    order: Order | None = None,
    mode: Mode = "equal_outlet",
    rule: Rule = "drawn",
) -> CaseResult:
    """Evaluate one case; ``case=0`` is no recovery (both heaters from ambient).

    ``order`` overrides the sources of each sink (default: the case drawing).

    ``rule`` is a PROPOSAL, not part of the workbook (``"drawn"`` reproduces the workbook):
    ``"by_temperature"`` orders the stages of a sink by the actual exhaust temperature (coldest
    first), ``"bypass"`` skips a stage whose exhaust is not warmer than the gas (it would cool
    it), both together combine them. A bypassed stage is kept with 0 W and unchanged temperatures.
    """
    order_1, order_2 = order if order is not None else ((), ()) if case == 0 else CASES[case]
    exhausts = _exhausts(inputs)
    return CaseResult(
        case,
        _sink(inputs, 1, order_1, exhausts, mode, rule),
        _sink(inputs, 2, order_2, exhausts, mode, rule),
    )


def plant_cells(inputs: SiInputsV41) -> PlantCells:
    """The case-independent sheet numbers (sheet units) from the dryer balances."""
    m = inputs.plant.material
    d = (inputs.plant.dryer_1, inputs.plant.dryer_2)
    r = tuple(c.dryer_balance(x, m) for x in d)
    return PlantCells(
        air_kg_per_h=(d[0].air_mass_flow_kg_per_s * 3600.0, d[1].air_mass_flow_kg_per_s * 3600.0),
        inlet_moisture_g_per_kg=(
            d[0].air_humidity_kg_per_kg * 1e3,
            d[1].air_humidity_kg_per_kg * 1e3,
        ),
        exhaust_moisture_g_per_kg=(
            r[0].outlet_humidity_kg_per_kg * 1e3,
            r[1].outlet_humidity_kg_per_kg * 1e3,
        ),
        exhaust_celsius=(r[0].outlet_temperature_celsius, r[1].outlet_temperature_celsius),
        ambient_celsius=inputs.plant.ambient.temperature_celsius,
    )


def to_case_cells(result: CaseResult) -> CaseCells:
    """Sheet units (kg/h, kW) for :func:`~.layout.scenario_cells`."""

    def sink(s: SinkResult) -> SinkCells:
        return SinkCells(
            stages=tuple(
                StageCells(
                    circuit=st.source,
                    sink_in_celsius=st.sink_in_celsius,
                    sink_out_celsius=st.sink_out_celsius,
                    htm_hot_celsius=st.exhaust_out_celsius,
                    htm_flow_kg_per_h=st.htm_flow_kg_per_s * 3600.0,
                    energy_kw=st.recovered_w / 1e3,
                )
                for st in s.stages
            ),
            heater_in_celsius=s.heater_in_celsius,
            heater_out_celsius=s.heater_out_celsius,
            gas_kg_per_h=s.gas_kg_per_s * 3600.0,
        )

    return CaseCells(sink(result.sink_1), sink(result.sink_2))


def balances_cells(inputs: SiInputsV41) -> dict[str, float]:
    """Document-based values for the numeric cells of the sheet ``Balances`` (sheet units).

    Columns ``A:F``: the dryer balances (1.3), (1.5). Columns ``H:T``, rows 3-8: the intermediate
    variables ``K1``..``K6``, i.e. (4.1)/(4.2) written as ``(A+B) T2 - B T1 = A T_c,in`` and
    ``-B T2 + (B+C) T1 = C T_h,in`` with the capacity rates in kJ/(h K).
    """
    plant = inputs.plant
    m = plant.material
    cells: dict[str, float] = {}
    for first, dryer in ((3, plant.dryer_1), (10, plant.dryer_2)):
        result = c.dryer_balance(dryer, m)
        cells[f"E{first}"] = dryer.air_mass_flow_kg_per_s * 3600.0
        cells[f"E{first + 1}"] = dryer.air_humidity_kg_per_kg * 1e3
        cells[f"F{first + 1}"] = result.outlet_humidity_kg_per_kg * 1e3
        cells[f"E{first + 2}"] = dryer.air_temperature_celsius
        cells[f"F{first + 2}"] = result.outlet_temperature_celsius
        cells[f"E{first + 3}"] = result.product_solids_kg_per_s * 3600.0
        cells[f"E{first + 4}"] = m.equilibrium_moisture_kg_per_kg * 1e3
        cells[f"E{first + 5}"] = result.outlet_temperature_celsius
    exhausts = _exhausts(inputs)
    to_kj_per_h = 3.6  # W/K -> kJ/(h K)
    for case, columns in BALANCES_COLUMNS.items():
        result = evaluate_case(inputs, case)
        for sink_no, sink in ((1, result.sink_1), (2, result.sink_2)):
            dryer = plant.dryer_1 if sink_no == 1 else plant.dryer_2
            for stage in sink.stages:
                gas = c.Stream(
                    dryer.air_mass_flow_kg_per_s,
                    dryer.air_humidity_kg_per_kg,
                    stage.sink_in_celsius,
                )
                a, b, cc = loops.loop_capacity_rates(
                    exhausts[stage.source],
                    gas,
                    stage.htm_flow_kg_per_s,
                    inputs.cp_htm_j_per_kg_k,
                    m,
                )
                constants = (
                    -b,
                    a + b,
                    a * stage.exhaust_in_celsius,
                    b + cc,
                    -b,
                    cc * stage.sink_in_celsius,
                )
                column = columns[stage.source - 1]
                for row, value in zip(range(3, 9), constants, strict=True):
                    cells[f"{column}{row}"] = value * to_kj_per_h
    return cells
