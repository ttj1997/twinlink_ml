"""The six scenario cases of the sheet ``Scenarios`` as a corrected cascade.

**OUR RECONSTRUCTION, not the partner's numbers.** The workbook leaves the heat sources of
the pre-heater rows as ``#REF!`` (the referenced cells were deleted), and rows 13-17 of the
sheet (pre-heater 1, low/high) are empty. What follows fills the gap with the most
economical reading of the sheet's own structure; every assumption is listed and is a
question to the partner.

Assumptions (A1-A5):

* **A1 -- case structure.** Read from which blocks exist per column and from the six cases of
  ``Model_v3`` (Befund 57): case 1 separate (source 1 -> sink 1, source 2 -> sink 2), case 2
  crossed (source 2 -> sink 1, source 1 -> sink 2), cases 3/5 both sources into sink 1
  (stage order 2,1 / 1,2), cases 4/6 both sources into sink 2 (order 2,1 / 1,2). Source 1 =
  exhaust of dryer 1 (``Heat recovery 1``), sink i = inlet air of dryer i (heater i).
* **A2 -- what a source offers.** Heat released when the exhaust is cooled to ``T_cold``
  including condensation (:func:`~.corrected.stream_heat_release_w`). ``T_cold`` is not in
  the sheet; placeholder: ambient temperature (15 degC), the sink's inlet.
* **A3 -- limits of a stage.** The sink leaves a stage at ``min(T_in + Q/C, T_target,
  T_source_in - dT_min)``; if ``T_in`` already is at or above that limit the stage transfers
  nothing. This is the structure of the sheet's ``MIN(..., target, #REF!)`` and
  ``IF(#REF! <= T_sink, no transfer, ...)`` (reading: the ``#REF!`` there is the source's
  inlet temperature). Mode ``"sheet"``.
* **A4 -- feasibility check (additional, not in the sheet).** Mode ``"counter_current"``
  limits the transfer to what a counter-current exchanger with minimum approach ``dT_min``
  can do between the exhaust's enthalpy curve (condensation included) and the sink.
* **A5 -- the stage splits low/high.** Stage 1 is the "low-temperature" pre-heater row, stage
  2 the "high-temperature" row (sink 1: rows 10-13 / 14-17; sink 2: rows 23-26 / 27-30).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import corrected as c
from .inputs import PartnerV4Inputs, SiDryer, SiMaterial

__all__ = [
    "CASES",
    "CASE_COLUMNS",
    "CaseResult",
    "Mode",
    "SinkResult",
    "StageResult",
    "evaluate_case",
    "max_counter_current_transfer_w",
    "proposed_cell_values",
]

Mode = Literal["sheet", "counter_current"]

CASES: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {
    1: ((1,), (2,)),
    2: ((2,), (1,)),
    3: ((2, 1), ()),
    4: ((), (2, 1)),
    5: ((1, 2), ()),
    6: ((), (1, 2)),
}
"""Case -> (source order into sink 1, source order into sink 2); assumption A1."""

CASE_COLUMNS: dict[int, tuple[str, str]] = {
    1: ("E", "F"),
    2: ("G", "H"),
    3: ("I", "J"),
    4: ("K", "L"),
    5: ("M", "N"),
    6: ("O", "P"),
}
"""Case -> (inlet column, outlet column) in the sheet ``Scenarios``."""


@dataclass(frozen=True)
class StageResult:
    """One pre-heater stage of one sink."""

    source: int
    sink_in_celsius: float
    sink_out_celsius: float
    power_w: float
    offered_w: float
    limited_by: str


@dataclass(frozen=True)
class SinkResult:
    """One sink: pre-heater stages, heater, gas."""

    stages: tuple[StageResult, ...]
    heater_in_celsius: float
    heater_out_celsius: float
    heater_power_w: float
    gas_kg_per_s: float


@dataclass(frozen=True)
class CaseResult:
    """One scenario case."""

    case: int
    sink_1: SinkResult
    sink_2: SinkResult

    @property
    def total_gas_kg_per_h(self) -> float:
        """Total gas of both heaters [kg/h]."""
        return (self.sink_1.gas_kg_per_s + self.sink_2.gas_kg_per_s) * 3600.0


def max_counter_current_transfer_w(
    source: c.Stream,
    sink_capacity_w_per_k: float,
    sink_in_celsius: float,
    sink_limit_celsius: float,
    t_cold_celsius: float,
    dt_min_k: float,
    pressure_pa: float,
    m: SiMaterial,
    *,
    grid: int = 400,
) -> float:
    """Largest ``Q`` [W] such that a counter-current exchanger is feasible (assumption A4).

    Feasible means ``T_exhaust(Q - x) >= T_sink(x) + dT_min`` for all heat ``x`` absorbed
    from the sink's cold end (the exhaust's temperature after it has released ``Q - x`` from
    its hot end). Bounded by the heat the exhaust offers down to ``t_cold`` and by the sink
    limit. Feasibility is monotone in ``Q``, so bisection.
    """
    q_offered = c.stream_heat_release_w(source, t_cold_celsius, pressure_pa, m)
    q_hi = min(q_offered, sink_capacity_w_per_k * max(sink_limit_celsius - sink_in_celsius, 0.0))

    def feasible(q: float) -> bool:
        for i in range(grid + 1):
            x = q * i / grid
            t_ex = c.temperature_at_released_heat(source, q - x, t_cold_celsius, pressure_pa, m)
            if t_ex < sink_in_celsius + x / sink_capacity_w_per_k + dt_min_k - 1e-9:
                return False
        return True

    if q_hi <= 0.0 or feasible(q_hi):
        return max(q_hi, 0.0)
    lo, hi = 0.0, q_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return lo


def _sink(
    dryer: SiDryer,
    sources: tuple[int, ...],
    exhausts: dict[int, c.Stream],
    inputs: PartnerV4Inputs,
    t_cold: float,
    dt_min_k: float,
    mode: Mode,
) -> SinkResult:
    m = inputs.material
    p = inputs.ambient.pressure_pa
    air = c.Stream(
        dryer.air_mass_flow_kg_per_s,
        dryer.air_humidity_kg_per_kg,
        inputs.ambient.temperature_celsius,
    )
    capacity = c.heat_capacity_rate_w_per_k(air, m)
    target = dryer.air_temperature_celsius
    t = air.temperature_celsius
    stages = []
    for index in sources:
        src = exhausts[index]
        offered = c.stream_heat_release_w(src, t_cold, p, m)
        cap = min(target, src.temperature_celsius - dt_min_k)
        if mode == "sheet":
            limit_energy = t + offered / capacity
            t_out = min(limit_energy, cap)
            t_out = max(t_out, t)
            limited = (
                "none"
                if t_out == limit_energy
                else ("target" if cap == target else "source temperature")
            )
            power = capacity * (t_out - t)
        else:
            power = max_counter_current_transfer_w(src, capacity, t, target, t_cold, dt_min_k, p, m)
            t_out = t + power / capacity
            limited = "none" if power >= offered * (1 - 1e-9) else "exchanger limit"
        stages.append(StageResult(index, t, t_out, power, offered, limited))
        t = t_out
    heater_power = capacity * (target - t)
    return SinkResult(
        stages=tuple(stages),
        heater_in_celsius=t,
        heater_out_celsius=target,
        heater_power_w=heater_power,
        gas_kg_per_s=c.gas_mass_flow_kg_per_s(heater_power, m),
    )


def evaluate_case(
    inputs: PartnerV4Inputs,
    case: int,
    *,
    t_cold_celsius: float | None = None,
    dt_min_k: float = 0.0,
    mode: Mode = "sheet",
) -> CaseResult:
    """Evaluate one of the six cases; ``case=0`` is no recovery (both heaters 15 -> target).

    ``t_cold_celsius`` defaults to the ambient temperature (assumption A2).
    """
    t_cold = inputs.ambient.temperature_celsius if t_cold_celsius is None else t_cold_celsius
    order_1, order_2 = ((), ()) if case == 0 else CASES[case]
    m = inputs.material
    exhausts = {
        1: c.exhaust_stream(inputs.dryer_1, c.dryer_balance(inputs.dryer_1, m)),
        2: c.exhaust_stream(inputs.dryer_2, c.dryer_balance(inputs.dryer_2, m)),
    }
    return CaseResult(
        case,
        _sink(inputs.dryer_1, order_1, exhausts, inputs, t_cold, dt_min_k, mode),
        _sink(inputs.dryer_2, order_2, exhausts, inputs, t_cold, dt_min_k, mode),
    )


def proposed_cell_values(
    inputs: PartnerV4Inputs,
    *,
    t_cold_celsius: float | None = None,
    dt_min_k: float = 0.0,
    mode: Mode = "sheet",
) -> dict[str, float]:
    """Proposed values for the cells of the sheet ``Scenarios`` that are ``#REF!`` or empty.

    **Our reconstruction (assumptions A1-A5), not the partner's numbers.** Keys are cell
    addresses of ``Scenarios``; values are in the sheet's units (degC, kW, kg/h). Energy cells
    (rows 13, 17, 26, 30) hold the heat transferred in the stage; ``E36`` etc. the total gas.
    """
    cells: dict[str, float] = {}
    for case, (ci, co) in CASE_COLUMNS.items():
        result = evaluate_case(
            inputs, case, t_cold_celsius=t_cold_celsius, dt_min_k=dt_min_k, mode=mode
        )
        # (stage rows: in-temperature, out-temperature, energy) per sink and stage number
        rows = {1: ((12, 13), (16, 17)), 2: ((25, 26), (29, 30))}
        for sink_no, sink in ((1, result.sink_1), (2, result.sink_2)):
            for stage, (t_row, e_row) in zip(sink.stages, rows[sink_no], strict=False):
                cells[f"{ci}{t_row}"] = stage.sink_in_celsius
                cells[f"{co}{t_row}"] = stage.sink_out_celsius
                cells[f"{ci}{e_row}"] = stage.power_w / 1e3
            heater_row = 20 if sink_no == 1 else 33
            gas_row = 21 if sink_no == 1 else 34
            cells[f"{ci}{heater_row}"] = sink.heater_in_celsius
            cells[f"{co}{heater_row}"] = sink.heater_out_celsius
            cells[f"{ci}{gas_row}"] = sink.gas_kg_per_s * 3600.0
        cells[f"{ci}36"] = result.total_gas_kg_per_h
    return cells
