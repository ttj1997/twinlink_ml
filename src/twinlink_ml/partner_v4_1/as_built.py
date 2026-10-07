"""Formula-for-formula reproduction of ``Model_v4.1.xlsx`` as it is -- bugs included.

Works in the sheet's units (kg/h, g/kg, kJ, percent) and follows the formulas of the cells:

* ``Balances`` ``F4``, ``F5``, ``F11``, ``F12`` -- dryer balances (unchanged from v4),
* ``Balances`` rows 3-8, columns ``I:T`` -- the intermediate variables ``K1``..``K6`` of the
  two circuits per case (the coefficients of the 2x2 system of (4.1)/(4.2)),
* ``Scenarios`` ``F5``, ``E7`` etc. -- the closed solution of that system for ``T2`` and ``T1``,
* ``Scenarios`` row 20/24/33/37 -- recovered energy ``Mw cp_w (T2 - T1) / 3600`` in kW,
* ``Scenarios`` rows 26 and 39 -- gas, **as built** (see :func:`gas_kg_per_h_as_built`),
* ``Scenarios`` row 41 -- total fuel (sum of the two heaters).

Nothing here is physically corrected; the document-based model is :mod:`.cascade`. Whether
the cell references of the sheet realise the case structure of the document is not assumed
but tested: the generated cells are compared with every number Excel stored
(``tests/test_partner_v4_1.py``).
"""

from __future__ import annotations

from twinlink_ml.partner_v4.as_built import dryer_as_built
from twinlink_ml.partner_v4.inputs import ExcelMaterialInputs

from .inputs import ExcelInputsV41
from .layout import (
    BALANCES_COLUMNS,
    CASES,
    CaseCells,
    PlantCells,
    SinkCells,
    StageCells,
    scenario_cells,
)

__all__ = [
    "balances_cells_as_built",
    "gas_kg_per_h_as_built",
    "loop_constants_as_built",
    "loop_solution_as_built",
    "plant_cells_as_built",
    "scenario_cells_as_built",
]

LoopConstants = tuple[float, float, float, float, float, float]


def gas_kg_per_h_as_built(
    air_kg_per_h: float,
    t_in_celsius: float,
    t_out_celsius: float,
    moisture_g_per_kg: float,
    m: ExcelMaterialInputs,
) -> float:
    """``Scenarios!E26`` (``G26 .. O26``, ``E39 .. O39``).

    ``= air * (T_out - T_in) * (cp_g + Y * cp_v) / (H_u * eta / 1000)`` with ``eta`` the percent
    value of ``Inputs!I6`` (80) and ``Y`` in g/kg (``Inputs!E4``, 6).

    As built: the ``1000`` sits at the efficiency (``eta / 1000 = 0.08`` instead of ``0.8``,
    factor 10), and ``Y`` enters in g/kg (6 instead of 0.006: ``cp = 13.2`` instead of
    ``1.212`` kJ/(kg K), factor 10.89). The numerator is kJ/h and ``H_u`` kJ/kg, so the quotient
    is already kg/h. Total factor 108.9 too high.
    """
    return (
        air_kg_per_h
        * (t_out_celsius - t_in_celsius)
        * (m.cp_gas_kj_per_kg_k + moisture_g_per_kg * m.cp_vapour_kj_per_kg_k)
    ) / (m.calorific_value_kj_per_kg * m.heater_efficiency_percent / 1000.0)


def loop_constants_as_built(
    htm_flow_kg_per_h: float,
    source_air_kg_per_h: float,
    source_moisture_g_per_kg: float,
    source_temperature_celsius: float,
    sink_air_kg_per_h: float,
    sink_moisture_g_per_kg: float,
    sink_in_celsius: float,
    m: ExcelMaterialInputs,
) -> LoopConstants:
    """``Balances!I3:I8`` (and the same rows of ``J:T``): ``K1`` .. ``K6``.

    ``K1 = K5 = -Mw cp_w``, ``K2 = A + B``, ``K3 = A T_c,in``, ``K4 = B + C``,
    ``K6 = C T_h,in`` with ``A = air_src (cp_g + cp_v Y_src/1000)``, ``B = Mw cp_w``,
    ``C = air_snk (cp_g + cp_v Y_snk/1000)``, all in kJ/(h K).
    """
    b = htm_flow_kg_per_h * m.cp_heating_medium_kj_per_kg_k
    a = source_air_kg_per_h * (
        m.cp_gas_kj_per_kg_k + m.cp_vapour_kj_per_kg_k * source_moisture_g_per_kg / 1000.0
    )
    cc = sink_air_kg_per_h * (
        m.cp_gas_kj_per_kg_k + m.cp_vapour_kj_per_kg_k * sink_moisture_g_per_kg / 1000.0
    )
    return (-b, b + a, a * source_temperature_celsius, b + cc, -b, cc * sink_in_celsius)


def loop_solution_as_built(k: LoopConstants) -> tuple[float, float]:
    """``Scenarios!F5`` (exhaust and medium leave the recovery exchanger) and ``E7`` (gas leaves).

    ``T2 = (K1 K6 - K3 K4) / (K1 K5 - K2 K4)``, ``T1 = (K3 K5 - K2 K6) / (K1 K5 - K2 K4)``.
    Returns ``(T1, T2)``.
    """
    k1, k2, k3, k4, k5, k6 = k
    det = k1 * k5 - k2 * k4
    return (k3 * k5 - k2 * k6) / det, (k1 * k6 - k3 * k4) / det


def plant_cells_as_built(inputs: ExcelInputsV41) -> PlantCells:
    """The case-independent numbers: dryer balances ``Balances!F4, F5, F11, F12``."""
    p = inputs.plant
    d1 = dryer_as_built(p.dryer_1, p.material)
    d2 = dryer_as_built(p.dryer_2, p.material)
    return PlantCells(
        air_kg_per_h=(p.dryer_1.air_mass_flow_kg_per_h, p.dryer_2.air_mass_flow_kg_per_h),
        inlet_moisture_g_per_kg=(p.dryer_1.air_moisture_g_per_kg, p.dryer_2.air_moisture_g_per_kg),
        exhaust_moisture_g_per_kg=(d1.outlet_moisture_g_per_kg, d2.outlet_moisture_g_per_kg),
        exhaust_celsius=(d1.outlet_temperature_celsius, d2.outlet_temperature_celsius),
        ambient_celsius=p.ambient_temperature_celsius,
    )


def _evaluate(
    inputs: ExcelInputsV41,
) -> tuple[dict[int, CaseCells], dict[int, dict[int, LoopConstants]]]:
    """All six cases as the sheet computes them, and the ``K`` constants per case and circuit."""
    p = inputs.plant
    m = p.material
    plant = plant_cells_as_built(inputs)
    dryers = (p.dryer_1, p.dryer_2)
    cases: dict[int, CaseCells] = {}
    constants: dict[int, dict[int, LoopConstants]] = {}
    for case, orders in CASES.items():
        constants[case] = {}
        sinks = []
        for sink_no, sources in ((1, orders[0]), (2, orders[1])):
            dryer = dryers[sink_no - 1]
            t_gas = p.ambient_temperature_celsius
            stages = []
            for source in sources:
                k = loop_constants_as_built(
                    inputs.htm_flow_kg_per_h[source - 1],
                    plant.air_kg_per_h[source - 1],
                    plant.exhaust_moisture_g_per_kg[source - 1],
                    plant.exhaust_celsius[source - 1],
                    dryer.air_mass_flow_kg_per_h,
                    dryer.air_moisture_g_per_kg,
                    t_gas,
                    m,
                )
                constants[case][source] = k
                t1, t2 = loop_solution_as_built(k)
                flow = inputs.htm_flow_kg_per_h[source - 1]
                stages.append(
                    StageCells(
                        circuit=source,
                        sink_in_celsius=t_gas,
                        sink_out_celsius=t1,
                        htm_hot_celsius=t2,
                        htm_flow_kg_per_h=flow,
                        energy_kw=flow * m.cp_heating_medium_kj_per_kg_k * (t2 - t1) / 3600.0,
                    )
                )
                t_gas = t1
            target = dryer.air_temperature_celsius
            sinks.append(
                SinkCells(
                    stages=tuple(stages),
                    heater_in_celsius=t_gas,
                    heater_out_celsius=target,
                    gas_kg_per_h=gas_kg_per_h_as_built(
                        dryer.air_mass_flow_kg_per_h,
                        t_gas,
                        target,
                        dryer.air_moisture_g_per_kg,
                        m,
                    ),
                )
            )
        cases[case] = CaseCells(sinks[0], sinks[1])
    return cases, constants


def scenario_cells_as_built(inputs: ExcelInputsV41) -> dict[str, float]:
    """Every numeric cell of the sheet ``Scenarios`` as the sheet computes it."""
    cases, _ = _evaluate(inputs)
    return scenario_cells(plant_cells_as_built(inputs), cases)


def balances_cells_as_built(inputs: ExcelInputsV41) -> dict[str, float]:
    """Every numeric cell of the sheet ``Balances`` (and of the duplicate sheet ``Balance``).

    Columns ``A:F`` (the two dryer blocks) and ``H:T`` rows 3-8 (``K1``..``K6`` per case and
    recovery). ``E7``/``E14`` are the equilibrium moisture ``Inputs!I10``.
    """
    p = inputs.plant
    d1 = dryer_as_built(p.dryer_1, p.material)
    d2 = dryer_as_built(p.dryer_2, p.material)
    cells: dict[str, float] = {}
    for first, balance, dryer in ((3, d1, p.dryer_1), (10, d2, p.dryer_2)):
        cells[f"E{first}"] = dryer.air_mass_flow_kg_per_h
        cells[f"E{first + 1}"] = dryer.air_moisture_g_per_kg
        cells[f"F{first + 1}"] = balance.outlet_moisture_g_per_kg
        cells[f"E{first + 2}"] = dryer.air_temperature_celsius
        cells[f"F{first + 2}"] = balance.outlet_temperature_celsius
        cells[f"E{first + 3}"] = balance.product_mass_flow_kg_per_h
        cells[f"E{first + 4}"] = p.material.equilibrium_moisture_g_per_kg
        cells[f"E{first + 5}"] = balance.outlet_temperature_celsius
    _, constants = _evaluate(inputs)
    for case, columns in BALANCES_COLUMNS.items():
        for circuit, column in ((1, columns[0]), (2, columns[1])):
            for row, value in zip(range(3, 9), constants[case][circuit], strict=True):
                cells[f"{column}{row}"] = value
    return cells
