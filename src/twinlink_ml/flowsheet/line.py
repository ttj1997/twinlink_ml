"""One dryer line from the units of this package: the stationary flowsheet of ``Model v4.1_2``.

The partner's Dyssol flowsheet and the comparison tables of the delivery 4.1_2 describe one line,
not the two dryers and six scenarios of the workbook (:mod:`.flowsheet`)::

    fresh gas --> HeatTransferLoop (pre-heating side) --> Burner --> DryerBlock --> product
                         ^                                              |
                         +---------- HeatTransferLoop (recovery side) <-+ exhaust --> out

The dryer does not depend on the recovery (its inlet temperature is the set point of the burner),
so the dryer is solved first and its exhaust then feeds the loop. The product leaves at the exhaust
temperature (equation 1.5 of the equation document, ``T_g,out = T_p``).

:class:`LineMode` holds what the partner model and the physical calculation do differently:

``partner`` (the delivery 4.1_2 and the Dyssol units)
    equal outlet temperatures in both exchangers of the loop, no saturation limit anywhere, the
    corrected gas formula (``/1000`` for g/kg, ``/100`` for the efficiency: the formula of 4.1_2).
``physical``
    counter-current exchangers, unlimited area, minimum approach ``dT_min`` per exchanger (10 K,
    the main convention of docs/11 section 11), the saturation limit on (the dryer caps the exhaust
    at saturation, :class:`~twinlink_ml.flowsheet.dryer.DryerConfig`; the exhaust follows the
    condensing release curve in the recovery exchanger), and optionally an upper limit of the
    medium temperature (an unpressurised liquid circuit).

In both modes the gas properties are those of the workbook (``Inputs!I4``, ``I9``; in 4.1_2
1.006 and 1.906 kJ/(kg K)); the ``cp_gas`` switch of the six-scenario flowsheet is not used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import ATM_PA, SiAmbient, SiDryer, SiMaterial

from .base import Assumption
from .burner import Burner, BurnerConfig, BurnerInputs, BurnerOut
from .dryer import DryerBlock, DryerConfig, DryerInputs, DryerOut
from .loop import LoopInputs, StageModel, StageOut, run_stage
from .thermo import ProfileKind, partner_props
from .validators import (
    Finding,
    SaturationProps,
    check_exhaust_above_adiabatic_saturation,
    check_exhaust_saturation,
    check_medium_boiling,
    check_product_mass,
)

__all__ = [
    "Line",
    "LineInputs",
    "LineMode",
    "LineResult",
    "with_heater",
]


@dataclass(frozen=True)
class LineMode:
    """What the partner model and the physical calculation do differently (see module doc)."""

    name: str
    exchanger: Assumption = "partner"
    saturation_limit: bool = False
    dt_min_k: float = 10.0
    """Minimum approach per exchanger (counter-current only)."""
    htm_max_celsius: float | None = None
    """Highest medium temperature (counter-current only); ``None``: pressurised, no limit."""

    def __post_init__(self) -> None:
        if self.exchanger not in ("partner", "physical"):
            raise ValueError("exchanger must be 'partner' or 'physical'")
        if self.htm_max_celsius is not None and self.exchanger == "partner":
            raise ValueError("a medium limit needs the counter-current exchanger")

    @classmethod
    def partner(cls) -> LineMode:
        return cls("partner")

    @classmethod
    def physical(cls, *, htm_max_celsius: float | None = None, dt_min_k: float = 10.0) -> LineMode:
        name = "physical/pressurised" if htm_max_celsius is None else "physical/unpressurised"
        return cls(name, "physical", True, dt_min_k, htm_max_celsius)

    def stage_model(self, material: SiMaterial) -> StageModel:
        props = partner_props(material)
        profile: ProfileKind
        if self.exchanger == "partner":
            profile = "condensing" if self.saturation_limit else "linear"
            return StageModel(f"{self.name}/equal_outlet", props, "equal_outlet", profile)
        profile = "condensing" if self.saturation_limit else "sensible"
        return StageModel(
            f"{self.name}/counter_current",
            props,
            "counter_current",
            profile,
            self.dt_min_k,
            "each",
            htm_max_celsius=self.htm_max_celsius,
        )


@dataclass(frozen=True)
class LineInputs:
    """One line in SI: the dryer block, the fresh gas temperature, material, medium."""

    dryer: SiDryer
    """Process air of the dryer (the fresh gas has its flow and humidity) and the spray."""
    fresh_gas_celsius: float
    """Temperature of the fresh gas entering the pre-heating side (also the ambient)."""
    material: SiMaterial
    """Properties, heater efficiency and heating value of the line."""
    htm_kg_per_s: float
    htm_cp_j_per_kg_k: float
    pressure_pa: float = ATM_PA

    @property
    def fresh_gas(self) -> c.Stream:
        d = self.dryer
        return c.Stream(d.air_mass_flow_kg_per_s, d.air_humidity_kg_per_kg, self.fresh_gas_celsius)


def with_heater(
    material: SiMaterial, efficiency: float, heating_value_j_per_kg: float
) -> SiMaterial:
    """The material with another heater efficiency (fraction) and heating value."""
    return replace(
        material, heater_efficiency=efficiency, lower_heating_value_j_per_kg=heating_value_j_per_kg
    )


@dataclass(frozen=True)
class LineResult:
    """Dryer, loop stage and burner of one line."""

    mode: LineMode
    inputs: LineInputs
    dryer: DryerOut
    stage: StageOut
    burner: BurnerOut

    @property
    def recovered_kw(self) -> float:
        return self.stage.q_w / 1e3

    @property
    def heater_duty_kw(self) -> float:
        return self.burner.duty_w / 1e3

    @property
    def fuel_kg_per_h(self) -> float:
        return self.burner.fuel_corrected_kg_per_s * 3600.0

    @property
    def product_kg_per_h(self) -> float:
        return self.dryer.product_kg_per_s * 3600.0

    @property
    def product_celsius(self) -> float:
        return self.dryer.exhaust.temperature_celsius

    @property
    def medium_hot_celsius(self) -> float:
        return math.nan if self.stage.htm_hot_celsius is None else self.stage.htm_hot_celsius

    @property
    def medium_cold_celsius(self) -> float:
        return math.nan if self.stage.htm_cold_celsius is None else self.stage.htm_cold_celsius

    def findings(self) -> tuple[Finding, ...]:
        """Checks (a), (b), (c) of :mod:`.validators` on this result."""
        d, ex = self.inputs.dryer, self.dryer.exhaust
        p = self.inputs.pressure_pa
        sat = SaturationProps.from_material(self.inputs.material)
        out = [
            check_exhaust_saturation(ex.temperature_celsius, ex.humidity_kg_per_kg, pressure_pa=p),
            check_exhaust_above_adiabatic_saturation(
                ex.temperature_celsius,
                d.air_temperature_celsius,
                d.air_humidity_kg_per_kg,
                sat,
                pressure_pa=p,
            ),
            check_product_mass(
                self.dryer.product_kg_per_s,
                d.spray_mass_flow_kg_per_s,
                d.spray_solid_fraction_kg_per_kg,
            ),
        ]
        if self.stage.htm_hot_celsius is not None:
            out.append(check_medium_boiling(self.stage.htm_hot_celsius, pressure_pa=p))
        return tuple(out)


class Line:
    """Builds the units of one line for a :class:`LineMode` and solves it."""

    def __init__(self, mode: LineMode) -> None:
        self.mode = mode

    def dryer_unit(self, inputs: LineInputs) -> DryerBlock:
        m = inputs.material
        return DryerBlock(
            DryerConfig(
                material=m,
                ambient=SiAmbient(inputs.fresh_gas_celsius, inputs.pressure_pa),
                props=partner_props(m),
                heat_loss_ua_w_per_k=0.0,
                product_moisture_kg_per_kg=m.equilibrium_moisture_kg_per_kg,
                name="dryer",
                saturation_limit=self.mode.saturation_limit,
            )
        )

    def burner_unit(self, material: SiMaterial) -> Burner:
        return Burner(BurnerConfig(material, partner_props(material), "physical"))

    def recovery(
        self,
        material: SiMaterial,
        exhaust: c.Stream,
        gas: c.Stream,
        htm_kg_per_s: float,
        htm_cp_j_per_kg_k: float,
        target_celsius: float = math.inf,
    ) -> StageOut:
        """The loop alone (the sheet ``Heat recovery``: the exhaust is given, no set point)."""
        return run_stage(
            self.mode.stage_model(material),
            LoopInputs(1, 1, exhaust, gas, htm_kg_per_s, htm_cp_j_per_kg_k, target_celsius),
        )

    def solve(self, inputs: LineInputs) -> LineResult:
        dryer = self.dryer_unit(inputs).solve(DryerInputs(inputs.dryer))
        target = inputs.dryer.air_temperature_celsius
        stage = self.recovery(
            inputs.material,
            dryer.exhaust,
            inputs.fresh_gas,
            inputs.htm_kg_per_s,
            inputs.htm_cp_j_per_kg_k,
            target,
        )
        heater_in = c.Stream(
            inputs.dryer.air_mass_flow_kg_per_s,
            inputs.dryer.air_humidity_kg_per_kg,
            stage.gas_out_celsius,
        )
        burner = self.burner_unit(inputs.material).solve(BurnerInputs(heater_in, target))
        return LineResult(self.mode, inputs, dryer, stage, burner)
