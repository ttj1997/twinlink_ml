"""The flowsheet: the six scenarios of the partner model built from exchangeable units.

For a scenario of the sheet (``partner_v4_1.layout.CASES``; case 0 is "no recovery") the
flowsheet wires, per sink ``j`` (the inlet air of dryer ``j``)::

    ambient air --> Preheater (a HeatTransferLoop per source, drawn order) --> Burner --> Dryer j
                                                                                          |
    exhaust j  <---------------------------------------------------------------------------+

The dryers do not depend on the recovery (their inlet temperature is the set point of the burner),
so the exhausts are computed first and then fed into the stages of the scenario. Every assumption
of the partner model that the independent calculation replaces is one value of
:class:`~twinlink_ml.flowsheet.base.Switches`; the flowsheet reads the switches only here, when it
builds the units, so the units themselves stay free of the switch logic.

``pinned_exhausts`` replaces the computed exhaust of a dryer by a given state ``(T degC, Y kg/kg)``.
It exists for the reproduction of Befund 90, whose source variants fix the exhausts independently
of the property set of the exchangers (partner exhaust of dryer 1, laboratory-model exhaust of
dryer 2); it is not a modelling option.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from twinlink_ml.evaluation.carrier_energy import CarrierEnergy
from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import SiDryer
from twinlink_ml.partner_v4_1.inputs import SiInputsV41
from twinlink_ml.partner_v4_1.layout import CASES

from .auxiliary import (
    AuxiliaryConfig,
    AuxiliaryPower,
    fan_power_w,
    pump_power_w,
    volume_flow_m3_per_s,
)
from .base import PhysicalParameters, Switches
from .bed_state import BedState
from .burner import Burner, BurnerConfig, BurnerInputs, BurnerOut
from .dryer import DryerBlock, DryerConfig, DryerInputs
from .loop import HeatTransferLoop, StageModel, StageOut
from .preheater import Preheater, PreheaterInputs, SourceFeed
from .thermo import REPO_PROPS, GasProps, ProfileKind, partner_props

__all__ = [
    "Flowsheet",
    "FlowsheetConfig",
    "ScenarioResult",
    "SinkResult",
    "Topology",
]

SCENARIOS = (1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class FlowsheetConfig:
    """Inputs, switches and the placeholders behind them."""

    inputs: SiInputsV41
    switches: Switches = field(default_factory=Switches)
    physical: PhysicalParameters = field(default_factory=PhysicalParameters)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
    bed: BedState | None = None
    """State of the laboratory model; required when ``product_moisture`` is ``"physical"``."""
    pinned_exhausts: Mapping[int, tuple[float, float]] | None = None
    """``{dryer: (T degC, Y kg/kg)}`` replaces that dryer's computed exhaust (see module doc)."""


@dataclass(frozen=True)
class Topology:
    """Who feeds whom in one scenario: the sources of each sink in the order of the drawing."""

    case: int
    sources_of_sink: tuple[tuple[int, ...], tuple[int, ...]]

    @property
    def stage_count(self) -> int:
        return sum(len(s) for s in self.sources_of_sink)


@dataclass(frozen=True)
class SinkResult:
    """One sink: its stages, the temperature entering the burner, the burner."""

    sink: int
    stages: tuple[StageOut, ...]
    heater_in_celsius: float
    burner: BurnerOut


@dataclass(frozen=True)
class ScenarioResult:
    """One scenario: stages, burners, totals, auxiliary power and CO2."""

    case: int
    exhausts: Mapping[int, c.Stream]
    sinks: tuple[SinkResult, SinkResult]
    auxiliary: AuxiliaryPower
    heating_value_j_per_kg: float
    htm_boiling_stages: int | None
    """Stages whose medium leaves the recovery exchanger above the boiling point (only with
    ``boiling_check="physical"``; ``None`` when the check is off). A flag, never a cap."""

    @property
    def stages(self) -> tuple[StageOut, ...]:
        return self.sinks[0].stages + self.sinks[1].stages

    @property
    def recovered_kw(self) -> float:
        return sum(s.q_w for s in self.stages) / 1e3

    @property
    def burner_duty_kw(self) -> float:
        return sum(s.burner.duty_w for s in self.sinks) / 1e3

    @property
    def fuel_kg_per_h(self) -> float:
        """With the gas formula the switch selects (as built: 108.9 times too high)."""
        return sum(s.burner.fuel_kg_per_s for s in self.sinks) * 3600.0

    @property
    def fuel_corrected_kg_per_h(self) -> float:
        """Always the corrected formula; the basis of ranking and CO2."""
        return sum(s.burner.fuel_corrected_kg_per_s for s in self.sinks) * 3600.0

    @property
    def carrier_energy(self) -> CarrierEnergy:
        return self.auxiliary.carrier_energy(
            sum(s.burner.fuel_corrected_kg_per_s for s in self.sinks), self.heating_value_j_per_kg
        )

    @property
    def co2_kg_per_h(self) -> float:
        """``0.366 E_el + 0.201 E_gas`` with the corrected fuel and the placeholder electricity."""
        return self.carrier_energy.co2_kg_per_h

    @property
    def co2_fuel_only_kg_per_h(self) -> float:
        return CarrierEnergy(0.0, self.carrier_energy.gas_kw).co2_kg_per_h

    @property
    def htm_hot_max_celsius(self) -> float:
        hot = [s.htm_hot_celsius for s in self.stages if s.htm_hot_celsius is not None]
        return max(hot) if hot else float("nan")


class Flowsheet:
    """Builds the units from the switches and evaluates the scenarios."""

    def __init__(self, config: FlowsheetConfig) -> None:
        self.config = config
        sw = config.switches
        if sw.product_moisture == "physical" and config.bed is None:
            raise ValueError("product_moisture='physical' needs the laboratory-model state (bed)")
        material = config.inputs.plant.material
        self.props: GasProps = REPO_PROPS if sw.cp_gas == "physical" else partner_props(material)
        self.stage_model = self._stage_model()
        self.preheater = Preheater(
            HeatTransferLoop(self.stage_model, assumption=sw.exchanger),
        )
        self.burner = Burner(BurnerConfig(material, self.props, sw.gas_formula))
        self._exhausts: dict[int, c.Stream] | None = None

    # ------------------------------------------------------------------ building the units
    def _stage_model(self) -> StageModel:
        sw, ph = self.config.switches, self.config.physical
        condensing = sw.condensation == "physical"
        if sw.exchanger == "partner":
            kind = "equal_outlet"
            profile: ProfileKind = "condensing" if condensing else "linear"
        else:
            kind = "counter_current"
            profile = "condensing" if condensing else "sensible"
        name = f"{kind}/{profile}/{self.props.name}"
        return StageModel(
            name,
            self.props,
            kind,  # type: ignore[arg-type]
            profile,
            ph.dt_min_k if kind == "counter_current" else 0.0,
            "each" if ph.convention == "per_exchanger" else "half",
        )

    def dryer_unit(self, dryer_no: int) -> DryerBlock:
        sw, ph = self.config.switches, self.config.physical
        plant = self.config.inputs.plant
        if sw.product_moisture == "physical" and dryer_no == 2:
            assert self.config.bed is not None
            x_p = self.config.bed.product_moisture_kg_per_kg
        else:
            x_p = plant.material.equilibrium_moisture_kg_per_kg
        return DryerBlock(
            DryerConfig(
                material=plant.material,
                ambient=plant.ambient,
                props=self.props,
                heat_loss_ua_w_per_k=ph.heat_loss_ua_w_per_k if sw.heat_loss == "physical" else 0.0,
                product_moisture_kg_per_kg=x_p,
                assumptions=(
                    ("cp_gas", sw.cp_gas),
                    ("heat_loss", sw.heat_loss),
                    ("product_moisture", sw.product_moisture),
                ),
                name=f"dryer {dryer_no}",
            )
        )

    def _dryer_inputs(self, dryer_no: int) -> SiDryer:
        plant = self.config.inputs.plant
        return plant.dryer_1 if dryer_no == 1 else plant.dryer_2

    # ------------------------------------------------------------------ evaluation
    def exhausts(self) -> dict[int, c.Stream]:
        """The exhaust of both dryers (computed once; pinned states replace computed ones)."""
        if self._exhausts is None:
            out: dict[int, c.Stream] = {}
            pinned = self.config.pinned_exhausts or {}
            for i in (1, 2):
                d = self._dryer_inputs(i)
                if i in pinned:
                    t, y = pinned[i]
                    out[i] = c.Stream(d.air_mass_flow_kg_per_s, y, t)
                else:
                    out[i] = self.dryer_unit(i).solve(DryerInputs(d)).exhaust
            self._exhausts = out
        return self._exhausts

    @staticmethod
    def topology(case: int) -> Topology:
        if case == 0:
            return Topology(0, ((), ()))
        return Topology(case, CASES[case])

    def evaluate(self, case: int) -> ScenarioResult:
        """One scenario; ``case`` 0 is no recovery (both burners from ambient)."""
        cfg = self.config
        plant = cfg.inputs.plant
        exhausts = self.exhausts()
        topo = self.topology(case)
        aux = cfg.auxiliary
        sinks: list[SinkResult] = []
        process_fans = exhaust_fans = pumps = 0.0
        boiling = 0
        for sink_no, sources in ((1, topo.sources_of_sink[0]), (2, topo.sources_of_sink[1])):
            d = self._dryer_inputs(sink_no)
            ambient_air = c.Stream(
                d.air_mass_flow_kg_per_s,
                d.air_humidity_kg_per_kg,
                plant.ambient.temperature_celsius,
            )
            train = self.preheater.solve(
                PreheaterInputs(
                    sink=sink_no,
                    gas=ambient_air,
                    feeds=tuple(
                        SourceFeed(s, exhausts[s], cfg.inputs.htm_flow_kg_per_s[s - 1])
                        for s in sources
                    ),
                    htm_cp_j_per_kg_k=cfg.inputs.cp_htm_j_per_kg_k,
                    target_celsius=d.air_temperature_celsius,
                )
            )
            burn = self.burner.solve(BurnerInputs(train.gas, d.air_temperature_celsius))
            sinks.append(SinkResult(sink_no, train.stages, train.gas.temperature_celsius, burn))
            process_fans += fan_power_w(
                volume_flow_m3_per_s(
                    ambient_air, plant.ambient.temperature_celsius, plant.ambient.pressure_pa
                ),
                aux.dryer_pressure_drop_pa + aux.exchanger_pressure_drop_pa * len(train.stages),
                aux.fan_efficiency,
            )
            for stage, feed in zip(train.stages, topo.sources_of_sink[sink_no - 1], strict=True):
                exhaust_fans += fan_power_w(
                    volume_flow_m3_per_s(
                        exhausts[feed], stage.exhaust_out_celsius, plant.ambient.pressure_pa
                    ),
                    aux.exchanger_pressure_drop_pa,
                    aux.fan_efficiency,
                )
                pumps += pump_power_w(
                    cfg.inputs.htm_flow_kg_per_s[feed - 1],
                    aux.loop_pressure_drop_pa,
                    aux.htm_density_kg_per_m3,
                    aux.pump_efficiency,
                )
                if cfg.switches.boiling_check == "physical" and stage.above_boiling(
                    cfg.physical.boiling_celsius
                ):
                    boiling += 1
        return ScenarioResult(
            case=case,
            exhausts=dict(exhausts),
            sinks=(sinks[0], sinks[1]),
            auxiliary=AuxiliaryPower(process_fans, exhaust_fans, pumps),
            heating_value_j_per_kg=plant.material.lower_heating_value_j_per_kg,
            htm_boiling_stages=boiling if cfg.switches.boiling_check == "physical" else None,
        )

    def evaluate_all(self) -> dict[int, ScenarioResult]:
        """Cases 0 (no recovery) and 1-6."""
        return {case: self.evaluate(case) for case in (0, *SCENARIOS)}
