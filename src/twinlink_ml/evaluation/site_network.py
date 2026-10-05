r"""Site CO2 of fluidised-bed lines coupled by the heat network (M2).

The lab lines stay laboratory models; their stationary flows are scaled linearly with ONE constant
factor ``k`` (D4): a scaling computation, never a plant statement. The site consists of the lines,
one fixed external source (Q1) and one fixed external sink (S1) of the partner data case (Befund
60), and the cascade of :mod:`twinlink_ml.heatnetwork` (condensing recovery).

Network of a scenario (lines ``A`` and optionally ``B``): sources ``Q1`` and one exhaust source per
line (``LA``, ``LB``: scaled dry-air flow, exhaust temperature and humidity of the line's steady
state); sinks ``S1`` and one inlet-air sink per line (``SA``, ``SB``: scaled dry-air flow, ambient
intake temperature and humidity, set point = the line's inlet temperature). The heater of each line
is the gas-fired air heater (D5 main case); its fuel is what the sink needs after recovery.

Objective per scenario and allocation (a set of ``(source, sink)`` connections)::

    CO2 = 0.366 * E_el + 0.201 * E_gas        [kg CO2e/h]
    kWh sum = E_el + E_gas                    [kW], from the same solution

with ``E_el`` = fan + pump + atomiser of the lines, scaled by ``k``, and ``E_gas`` = the fuel power
of ALL sinks (including S1) after recovery. The cascade order is the network's own rule
(low-temperature sources first). An allocation that the cascade rejects is counted and left out;
nothing is replaced by a sentinel. The mechanics are those of the M0 script
``analyze_m0_line_b_network.py`` (LB2), which this module generalises to any number of lines; the
test pins that the two-line results are identical.

Allocations (D7): every source feeds at least one sink. For one line this gives 3 x 3 = 9 sets, for
two lines 7 x 7 x 7 = 343. Named routings of the arms of M2: ``fixed`` ("Fall 1", each source to its
own sink), ``all_to_all`` (every source feeds every sink), ``best`` (the allocation of minimal CO2).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from twinlink_ml.heatnetwork.balance import SinkState, SourceState, evaluate_cascade
from twinlink_ml.heatnetwork.config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig
from twinlink_ml.heatnetwork.routing import FixedRouting
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.config import ProcessConfig
from twinlink_ml.simulation.steady_state import SteadyStateSolution
from twinlink_ml.simulation.thermo import absolute_humidity_kg_per_kg

from .net_energy import ELECTRICITY_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH

__all__ = [
    "EXTERNAL_SINK_S1",
    "EXTERNAL_SOURCE_Q1",
    "Allocation",
    "LineState",
    "SiteResult",
    "SiteSpec",
    "all_allocations",
    "all_to_all_allocation",
    "evaluate_site",
    "fixed_allocation",
    "line_state",
    "scale_factor",
]

Allocation = tuple[tuple[str, str], ...]

EXTERNAL_SOURCE_Q1 = HeatSourceConfig("Q1", 20.0, 125.233, 0.06482)
"""Exhaust of the partner spray tower (Befund 60, data case PA2; mapping to be confirmed, D4)."""
EXTERNAL_SINK_S1 = HeatSinkConfig("S1", 20.0, 0.005, 15.0, 300.0)
"""Inlet air of the partner spray tower (Befund 60)."""
PARTNER_LINE_EXHAUST_FLOW_KG_PER_S = 8.0
"""Dry-air flow of the fluidised-bed source of the partner model (Befund 60), the scale basis."""

_BACKEND = NumpyBackend()


@dataclass(frozen=True, slots=True)
class SiteSpec:
    """Network settings of a site evaluation (placeholders of the partner data case, Befund 60)."""

    scale_factor: float
    t_kalt_celsius: float = 29.1
    dt_min_k: float = 20.0
    heater_efficiency: float = 0.80
    fuel_lower_heating_value_j_per_kg: float = 45.0e6
    external_source: HeatSourceConfig = EXTERNAL_SOURCE_Q1
    external_sink: HeatSinkConfig = EXTERNAL_SINK_S1


@dataclass(frozen=True, slots=True)
class LineState:
    """What the network needs from one line's steady state (lab scale, not scaled)."""

    name: str
    ambient_temperature_celsius: float
    inlet_temperature_celsius: float
    inlet_humidity_kg_per_kg: float
    outlet_temperature_celsius: float
    outlet_humidity_kg_per_kg: float
    dry_air_mass_flow_kg_per_s: float
    electric_kw: float


@dataclass(frozen=True, slots=True)
class SiteResult:
    """One allocation of one scenario."""

    connections: Allocation
    electric_kw: float
    gas_kw: float
    gas_kw_lines: float
    recovered_kw_lines: float
    recovered_kw_total: float

    @property
    def co2_kg_per_h(self) -> float:
        return ELECTRICITY_CO2_KG_PER_KWH * self.electric_kw + GAS_CO2_KG_PER_KWH * self.gas_kw

    @property
    def kwh_sum_kw(self) -> float:
        return self.electric_kw + self.gas_kw

    @property
    def co2_lines_kg_per_h(self) -> float:
        """CO2 attributable to the lines: their electricity and the fuel of their own sinks."""
        return (
            ELECTRICITY_CO2_KG_PER_KWH * self.electric_kw + GAS_CO2_KG_PER_KWH * self.gas_kw_lines
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "connections": [list(c) for c in self.connections],
            "electric_kw": self.electric_kw,
            "gas_kw": self.gas_kw,
            "gas_kw_lines": self.gas_kw_lines,
            "recovered_kw_lines": self.recovered_kw_lines,
            "recovered_kw_total": self.recovered_kw_total,
            "co2_kg_per_h": self.co2_kg_per_h,
            "kwh_sum_kw": self.kwh_sum_kw,
            "co2_lines_kg_per_h": self.co2_lines_kg_per_h,
        }


def scale_factor(nominal_dry_air_flow_kg_per_s: float) -> float:
    """``k``: the exhaust flow of one line at the config nominal point then equals the partner's."""
    return PARTNER_LINE_EXHAUST_FLOW_KG_PER_S / nominal_dry_air_flow_kg_per_s


def line_state(
    name: str,
    solution: SteadyStateSolution,
    config: ProcessConfig,
    *,
    t_amb: float,
    phi_amb: float,
    t_in: float,
) -> LineState:
    """Line state from a steady-state solution (the exhaust leaves through the duct, with loss)."""
    d = solution.diagnostics
    y_in = float(
        _BACKEND.to_numpy(
            absolute_humidity_kg_per_kg(
                _BACKEND, config.gas, _BACKEND.asarray(t_amb), _BACKEND.asarray(phi_amb)
            )
        )
    )
    return LineState(
        name=name,
        ambient_temperature_celsius=t_amb,
        inlet_temperature_celsius=t_in,
        inlet_humidity_kg_per_kg=y_in,
        outlet_temperature_celsius=solution.bed_temperature_celsius
        - config.exhaust.outlet_duct_heat_loss_k,
        outlet_humidity_kg_per_kg=solution.outlet_humidity_kg_per_kg,
        dry_air_mass_flow_kg_per_s=solution.dry_air_mass_flow_kg_per_s,
        electric_kw=float(d["power_fan_kw"] + d["power_pump_kw"] + d["power_atomiser_kw"]),
    )


def all_allocations(n_lines: int) -> list[Allocation]:
    """Every set of connections in which each source feeds at least one sink (D7)."""
    letters = "AB"[:n_lines]
    sources = ["Q1", *(f"L{x}" for x in letters)]
    sinks = ["S1", *(f"S{x}" for x in letters)]
    nonempty = [s for size in range(1, len(sinks) + 1) for s in itertools.combinations(sinks, size)]
    out: list[Allocation] = []
    for choice in itertools.product(nonempty, repeat=len(sources)):
        out.append(
            tuple((src, snk) for src, subset in zip(sources, choice, strict=True) for snk in subset)
        )
    return out


def fixed_allocation(n_lines: int) -> Allocation:
    """Partner "Fall 1": each source feeds its own sink (Q1 to S1, each exhaust to its inlet)."""
    letters = "AB"[:n_lines]
    return (("Q1", "S1"), *((f"L{x}", f"S{x}") for x in letters))


def all_to_all_allocation(n_lines: int) -> Allocation:
    """Every source feeds every sink."""
    letters = "AB"[:n_lines]
    sources = ["Q1", *(f"L{x}" for x in letters)]
    sinks = ["S1", *(f"S{x}" for x in letters)]
    return tuple((src, snk) for src in sources for snk in sinks)


def _network(
    lines: tuple[LineState, ...], spec: SiteSpec
) -> tuple[HeatNetworkConfig, tuple[SourceState, ...], tuple[SinkState, ...]]:
    k = spec.scale_factor
    sources = (
        spec.external_source,
        *(
            HeatSourceConfig(
                f"L{x.name}",
                k * x.dry_air_mass_flow_kg_per_s,
                x.outlet_temperature_celsius,
                x.outlet_humidity_kg_per_kg,
            )
            for x in lines
        ),
    )
    sinks = (
        spec.external_sink,
        *(
            HeatSinkConfig(
                f"S{x.name}",
                k * x.dry_air_mass_flow_kg_per_s,
                x.inlet_humidity_kg_per_kg,
                x.ambient_temperature_celsius,
                x.inlet_temperature_celsius,
            )
            for x in lines
        ),
    )
    network = HeatNetworkConfig(
        sources=sources,
        sinks=sinks,
        minimum_pinch_k=spec.dt_min_k,
        recovery_mode="condensing",
        cold_temperature_celsius=spec.t_kalt_celsius,
        fuel_lower_heating_value_j_per_kg=spec.fuel_lower_heating_value_j_per_kg,
        heater_efficiency=spec.heater_efficiency,
    )
    return (
        network,
        tuple(SourceState.from_config(c) for c in sources),
        tuple(SinkState.from_config(c) for c in sinks),
    )


def evaluate_site(
    lines: tuple[LineState, ...],
    spec: SiteSpec,
    allocations: list[Allocation],
) -> tuple[list[SiteResult], int]:
    """Cascade for every allocation; returns the results and the number rejected by the cascade."""
    network, source_states, sink_states = _network(lines, spec)
    electric_kw = spec.scale_factor * sum(x.electric_kw for x in lines)
    h_u = spec.fuel_lower_heating_value_j_per_kg
    results: list[SiteResult] = []
    rejected = 0
    for connections in allocations:
        try:
            stages = FixedRouting(connections).route(network)
            cascade = evaluate_cascade(network, stages, source_states, sink_states)
        except (ValueError, RuntimeError):
            rejected += 1
            continue
        gas_kw_per_sink = cascade.fuel_mass_flow_kg_per_s[:, 0] * h_u / 1000.0
        stage_kw = np.asarray(cascade.stage_recovered_power_kw)[:, 0]
        to_lines = np.array([name != spec.external_sink.name for name in cascade.stage_sink_names])
        results.append(
            SiteResult(
                connections=connections,
                electric_kw=float(electric_kw),
                gas_kw=float(gas_kw_per_sink.sum()),
                gas_kw_lines=float(gas_kw_per_sink[1:].sum()),
                recovered_kw_lines=float(np.sum(stage_kw[to_lines])),
                recovered_kw_total=float(np.sum(stage_kw)),
            )
        )
    return results, rejected
