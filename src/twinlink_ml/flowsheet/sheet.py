"""The flowsheet in the cell layout of ``Model_v4.1.xlsx`` (for the comparison with the sheet).

Turns the scenarios of a :class:`~twinlink_ml.flowsheet.flowsheet.Flowsheet` into the neutral
containers of :mod:`twinlink_ml.partner_v4_1.layout` (sheet units: degC, kg/h, g/kg, kW), so that
``{address: value}`` of the sheet ``Scenarios`` can be compared cell by cell with what Excel stored.
Only meaningful for the partner side of ``exchanger`` (the sheet has no counter-current stage):
the "medium hot" column is the medium temperature ``T2`` of the stage, which is what the sheet
shows.
"""

from __future__ import annotations

from twinlink_ml.partner_v4_1.layout import (
    CaseCells,
    PlantCells,
    SinkCells,
    StageCells,
    scenario_cells,
)

from .flowsheet import SCENARIOS, Flowsheet, ScenarioResult, SinkResult

__all__ = ["plant_cells", "sheet_scenario_cells", "to_case_cells"]


def plant_cells(flowsheet: Flowsheet) -> PlantCells:
    """The case-independent cells (dryer balances, sheet units)."""
    plant = flowsheet.config.inputs.plant
    exhausts = flowsheet.exhausts()
    dryers = (plant.dryer_1, plant.dryer_2)
    return PlantCells(
        air_kg_per_h=(
            dryers[0].air_mass_flow_kg_per_s * 3600.0,
            dryers[1].air_mass_flow_kg_per_s * 3600.0,
        ),
        inlet_moisture_g_per_kg=(
            dryers[0].air_humidity_kg_per_kg * 1e3,
            dryers[1].air_humidity_kg_per_kg * 1e3,
        ),
        exhaust_moisture_g_per_kg=(
            exhausts[1].humidity_kg_per_kg * 1e3,
            exhausts[2].humidity_kg_per_kg * 1e3,
        ),
        exhaust_celsius=(exhausts[1].temperature_celsius, exhausts[2].temperature_celsius),
        ambient_celsius=plant.ambient.temperature_celsius,
    )


def to_case_cells(flowsheet: Flowsheet, result: ScenarioResult) -> CaseCells:
    """Sheet units (kg/h, kW) for :func:`~twinlink_ml.partner_v4_1.layout.scenario_cells`."""
    plant = flowsheet.config.inputs.plant
    dryers = (plant.dryer_1, plant.dryer_2)

    def sink(s: SinkResult) -> SinkCells:
        stages = []
        for st in s.stages:
            if st.htm_hot_celsius is None:
                raise ValueError("a direct stage has no medium; the sheet has only circuits")
            stages.append(
                StageCells(
                    circuit=st.source,
                    sink_in_celsius=st.gas_in_celsius,
                    sink_out_celsius=st.gas_out_celsius,
                    htm_hot_celsius=st.htm_hot_celsius,
                    htm_flow_kg_per_h=flowsheet.config.inputs.htm_flow_kg_per_s[st.source - 1]
                    * 3600.0,
                    energy_kw=st.q_w / 1e3,
                )
            )
        return SinkCells(
            stages=tuple(stages),
            heater_in_celsius=s.heater_in_celsius,
            heater_out_celsius=dryers[s.sink - 1].air_temperature_celsius,
            gas_kg_per_h=s.burner.fuel_kg_per_s * 3600.0,
        )

    return CaseCells(sink(result.sinks[0]), sink(result.sinks[1]))


def sheet_scenario_cells(flowsheet: Flowsheet) -> dict[str, float]:
    """Every numeric cell of the sheet ``Scenarios`` as the flowsheet computes it (cases 1-6)."""
    cases = {n: to_case_cells(flowsheet, flowsheet.evaluate(n)) for n in SCENARIOS}
    return scenario_cells(plant_cells(flowsheet), cases)
