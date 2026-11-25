"""The stationary units in a network: every exhaust preheats one fresh gas, stages may be in series.

:func:`twinlink_ml.dyssol_steady.reference.evaluate` covers flowsheets in which every exhaust
preheats exactly one fresh-gas feed and every feed is preheated exactly once (the cases 1 and 2
of the six-case network, docs/11 section 13). The four cascades (cases 3 to 6) put two heat
recoveries in series on one fresh-gas stream and leave the other one without recovery.
:func:`evaluate_network` evaluates any such wiring with the same three unit functions; for the
cases 1 and 2 it calls them with the same arguments as ``evaluate`` and returns the same numbers.

A wiring names, per sink (the fresh gas and heater of line ``i``, 0-based), the lines whose
dryer exhausts preheat it, in the order the fresh gas passes them. :data:`CASES` holds the six
cases of the workbook (``partner_v4.cascade.CASES``, there 1-based) and case 0, no recovery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from twinlink_ml.flowsheet.validators import saturation_humidity_kg_per_kg
from twinlink_ml.partner_v4.cascade import CASES as _WORKBOOK_CASES

from .materials import DyssolProperties
from .reference import (
    KELVIN,
    EnthalpyTable,
    LineInputs,
    LineOutputs,
    dryer,
    heat_recovery,
    heater,
)

__all__ = [
    "CASES",
    "NetworkOutputs",
    "StageOutputs",
    "Wiring",
    "evaluate_network",
    "exhaust_capacity_kw_per_k",
    "sink_capacity_kw_per_k",
]

Wiring = tuple[tuple[int, ...], ...]
"""Per sink (0-based line) the lines whose exhausts preheat it, in the order of the fresh gas."""

CASES: dict[int, Wiring] = {
    0: ((), ()),
    **{
        case: tuple(tuple(source - 1 for source in sink) for sink in sinks)
        for case, sinks in _WORKBOOK_CASES.items()
    },
}
"""Case 0 (no recovery) and the six cases of the workbook for two lines, 0-based."""


@dataclass(frozen=True)
class StageOutputs:
    """One heat recovery: exhaust of line ``exhaust`` preheats the fresh gas of line ``sink``."""

    exhaust: int
    sink: int
    heat_kw: float
    """Heat taken from the exhaust; negative if the fresh gas arrives warmer than the exhaust
    (equations 4.1 / 4.2 have no kink)."""
    cold_in_celsius: float
    cold_out_celsius: float
    hot_out_celsius: float
    saturation_ratio_after: float
    """Humidity of the exhaust over the saturation humidity at its temperature after the stage;
    above 1 the unit's result is inadmissible (the unit cannot condense)."""


@dataclass(frozen=True)
class NetworkOutputs:
    """Results of every line (as :class:`~.reference.LineOutputs`) and of every stage.

    For a line whose exhaust preheats nothing, ``exhaust_celsius`` is the dryer outlet temperature
    and ``heat_recovered_kw`` is 0; for a sink without a stage, ``preheated_celsius`` is the
    temperature of its fresh gas.
    """

    lines: tuple[LineOutputs, ...]
    stages: tuple[StageOutputs, ...]

    @property
    def fuel_kg_per_h(self) -> float:
        """Total fuel of all heaters."""
        return sum(line.fuel_kg_per_h for line in self.lines)

    @property
    def recovered_kw(self) -> float:
        return sum(stage.heat_kw for stage in self.stages)

    @property
    def negative_stages(self) -> int:
        """Stages that cool the fresh gas."""
        return sum(stage.heat_kw < 0.0 for stage in self.stages)

    @property
    def dryers_admissible(self) -> bool:
        """No dryer exhaust is over-saturated."""
        return not any(line.oversaturated for line in self.lines)

    @property
    def admissible(self) -> bool:
        """No dryer exhaust is over-saturated, at the dryer outlet or after its recovery stage."""
        return self.dryers_admissible and all(s.saturation_ratio_after <= 1.0 for s in self.stages)


def exhaust_capacity_kw_per_k(props: DyssolProperties, line: LineInputs) -> float:
    """Capacity rate of the dryer exhaust of ``line`` (dry gas plus all vapour) [kW/K]."""
    gas = line.dry_gas_kg_per_h / 3600.0
    solid = line.spray_kg_per_h / 3600.0 * line.solid_fraction
    liquid = line.spray_kg_per_h / 3600.0 - solid
    evaporated = max(liquid - solid * line.equilibrium_moisture_g_per_kg / 1000.0, 0.0)
    vapour = gas * line.gas_humidity_g_per_kg / 1000.0 + evaporated
    return (gas * props.cp_gas + vapour * props.cp_vapour) / 1000.0


def sink_capacity_kw_per_k(props: DyssolProperties, line: LineInputs) -> float:
    """Capacity rate of the fresh gas of ``line`` (the cold side of a heat recovery) [kW/K]."""
    gas = line.dry_gas_kg_per_h / 3600.0
    vapour = gas * line.gas_humidity_g_per_kg / 1000.0
    return (gas * props.cp_gas + vapour * props.cp_vapour) / 1000.0


def evaluate_network(
    props: DyssolProperties, lines: Sequence[LineInputs], wiring: Wiring
) -> NetworkOutputs:
    """All lines of a flowsheet with the given wiring.

    Every exhaust may preheat at most one fresh gas. The medium flow of a stage is the one of the
    line whose exhaust it cools (``LineInputs.medium_kg_per_h``).
    """
    n = len(lines)
    used = [source for sink in wiring for source in sink]
    if len(wiring) != n or len(set(used)) != len(used) or any(not 0 <= s < n for s in used):
        raise ValueError(
            "wiring must name, per line, exhausts of existing lines, each at most once"
        )
    table = EnthalpyTable(props)
    gas = [ln.dry_gas_kg_per_h / 3600.0 for ln in lines]
    vapour = [g * ln.gas_humidity_g_per_kg / 1000.0 for g, ln in zip(gas, lines, strict=True)]
    dried = []
    for ln, g, v in zip(lines, gas, vapour, strict=True):
        spray = ln.spray_kg_per_h / 3600.0
        solid = spray * ln.solid_fraction
        t_out, vapour_out, liquid_out = dryer(
            props,
            g,
            v,
            ln.dryer_inlet_celsius + KELVIN,
            solid,
            spray - solid,
            ln.spray_celsius + KELVIN,
            ln.equilibrium_moisture_g_per_kg / 1000.0,
        )
        dried.append((t_out, vapour_out, solid + liquid_out))

    stages: list[StageOutputs] = []
    hot_out = [t_out for t_out, _, _ in dried]
    heat = [0.0] * n
    heater_in = []
    for sink, sources in enumerate(wiring):
        cold = lines[sink].gas_celsius + KELVIN
        for source in sources:
            t_out, vapour_out, _ = dried[source]
            t1, t2, q = heat_recovery(
                props,
                (gas[source], vapour_out, t_out),
                (gas[sink], vapour[sink], cold),
                lines[source].medium_kg_per_h / 3600.0,
            )
            saturation = saturation_humidity_kg_per_kg(t2 - KELVIN)
            stages.append(
                StageOutputs(
                    exhaust=source,
                    sink=sink,
                    heat_kw=q / 1000.0,
                    cold_in_celsius=cold - KELVIN,
                    cold_out_celsius=t1 - KELVIN,
                    hot_out_celsius=t2 - KELVIN,
                    saturation_ratio_after=vapour_out / gas[source] / saturation
                    if saturation > 0.0
                    else float("inf"),
                )
            )
            hot_out[source], heat[source], cold = t2, q, t1
        heater_in.append(cold)

    outputs = []
    for i, ln in enumerate(lines):
        t_out, vapour_out, product = dried[i]
        duty, fuel = heater(
            table,
            gas[i],
            vapour[i],
            heater_in[i],
            ln.dryer_inlet_celsius + KELVIN,
            ln.heater_efficiency_percent / 100.0,
            ln.calorific_value_kj_per_kg,
        )
        humidity = vapour_out / gas[i]
        saturation = saturation_humidity_kg_per_kg(t_out - KELVIN)
        outputs.append(
            LineOutputs(
                dryer_outlet_celsius=t_out - KELVIN,
                dryer_outlet_humidity_g_per_kg=humidity * 1000.0,
                exhaust_celsius=hot_out[i] - KELVIN,
                preheated_celsius=heater_in[i] - KELVIN,
                heat_recovered_kw=heat[i] / 1000.0,
                heater_duty_kw=duty,
                fuel_kg_per_h=fuel,
                product_kg_per_h=product * 3600.0,
                saturation_ratio=humidity / saturation if saturation > 0.0 else float("inf"),
            )
        )
    return NetworkOutputs(tuple(outputs), tuple(stages))
