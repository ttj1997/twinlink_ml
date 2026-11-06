r"""The partner balance in the intensive formulation, and the synthetic plants built on it.

One class, :class:`DryerPlantModel`, computes the three targets for inputs ``x (n, 5)``
(:data:`~.config.INPUT_NAMES`): exhaust temperature as measured [degC], exhaust humidity as
measured [g/kg dry air] and heater power per unit of process dry-air flow [kW per kg/s, i.e.
kJ per kg]. The bed itself is ALWAYS the rebuild of the partner block,
:func:`twinlink_ml.partner_v4_1.dryer_block.dryer_block` (equations 1.1-1.5 of the equation
document), called point by point; nothing of the balance is re-derived here. What a plant adds is
injected (dependency injection), each part off in the partner balance:

``heat_loss_w_per_k_per_kg_h``
    heat loss ``UA (T_out - T_amb)`` of the bed with ``UA`` per kg/h of process air; the plant's
    dry-air flow is the source's (fixed), so this is the source's UA at that flow.
``atomising_air_share``
    a share ``a`` of the process air enters the bed unheated at ``atomising_air_celsius`` with the
    inlet humidity; adiabatic mixing before the block (same humidity, so the mixed temperature is
    the capacity-weighted mean, exact).
``false_air_share``
    a share ``f`` of the process air at the ambient state mixes into the exhaust before the
    sensors (enthalpy and water conserved; the evaporation-enthalpy terms cancel, exact).
:class:`EfficiencyModel`
    heater efficiency, constant (the balance) or load dependent; the load ``q`` is the specific
    duty relative to the source's nominal specific duty.
:class:`MoistureModel`
    product moisture ``X_p``: the equilibrium moisture of the sheet (the balance), a constant
    excess above it (calibration L1), or an excess that grows with the relative humidity of the
    bed exhaust (plant P-moist), solved implicitly by bisection.

The heater heats only the process air from the ambient temperature; its duty is the formula of
``dryer_block.heater_power_w`` (vectorised here because that function uses the scalar ``max``;
a test pins both together).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from twinlink_ml.flowsheet.validators import MOLAR_MASS_RATIO, water_saturation_pressure_pa
from twinlink_ml.partner_v4_1.dryer_block import BlockInputs, BlockProperties, dryer_block

from .config import EfficiencySpec, MoistureSpec, PlantSpec, SourceSpec

__all__ = [
    "BedState",
    "ConstantEfficiency",
    "ConstantMoisture",
    "DryerPlantModel",
    "EfficiencyModel",
    "LoadDependentEfficiency",
    "MoistureModel",
    "RhDependentMoisture",
    "plant_model",
    "relative_humidity",
    "source_model",
]

_BISECTION_STEPS = 60


def _psat_pa(temperature_celsius: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(temperature_celsius, dtype=np.float64), 0.0, 370.0)
    return np.array([water_saturation_pressure_pa(float(v)) for v in t.ravel()]).reshape(t.shape)


def relative_humidity(
    humidity_kg_per_kg: np.ndarray, temperature_celsius: np.ndarray, pressure_pa: float
) -> np.ndarray:
    """Unclipped relative humidity (IAPWS-IF97 saturation; below 0 degC the 0 degC value)."""
    y = np.asarray(humidity_kg_per_kg, dtype=np.float64)
    p_v = y * pressure_pa / (MOLAR_MASS_RATIO + y)
    return np.asarray(p_v / _psat_pa(temperature_celsius), dtype=np.float64)


class EfficiencyModel(ABC):
    """Heater efficiency as a function of the load ``q`` (specific duty / nominal one)."""

    @abstractmethod
    def efficiency(self, load: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ConstantEfficiency(EfficiencyModel):
    """The balance's heater efficiency: one value at every load."""

    value: float

    def efficiency(self, load: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(load, dtype=np.float64), self.value)


@dataclass(frozen=True)
class LoadDependentEfficiency(EfficiencyModel):
    """Quadratic part-load curve of :class:`~.config.EfficiencySpec` (plant P-eta)."""

    spec: EfficiencySpec

    def efficiency(self, load: np.ndarray) -> np.ndarray:
        q = np.asarray(load, dtype=np.float64)
        eta = self.spec.peak - self.spec.curvature * (q - self.spec.load_at_peak) ** 2
        if np.any(eta <= 0.05):
            raise ValueError("load-dependent efficiency left its range (eta <= 0.05)")
        return np.asarray(eta, dtype=np.float64)


@dataclass(frozen=True)
class BedState:
    """State of the bed exhaust (before any false air) for one product moisture per point."""

    temperature_celsius: np.ndarray
    humidity_kg_per_kg: np.ndarray
    relative_humidity: np.ndarray
    evaporation_kg_per_kg_air: np.ndarray


BedFunction = Callable[[np.ndarray], BedState]


class MoistureModel(ABC):
    """Product moisture ``X_p`` (dry basis) per point."""

    @abstractmethod
    def solve(self, bed: BedFunction, equilibrium: float, n: int) -> np.ndarray:
        """``X_p`` for ``n`` points; ``bed(X_p)`` gives the bed state at a trial moisture."""


@dataclass(frozen=True)
class ConstantMoisture(MoistureModel):
    """``X_p = X_eq + excess`` (excess 0: the balance of the sheet)."""

    excess: float = 0.0

    def solve(
        self,
        bed: BedFunction,  # noqa: ARG002 - shared interface
        equilibrium: float,
        n: int,
    ) -> np.ndarray:
        return np.full(n, equilibrium + self.excess)


@dataclass(frozen=True)
class RhDependentMoisture(MoistureModel):
    """``X_p = X_eq + excess_max RH / (RH + K)`` at the bed exhaust, solved by bisection.

    A larger ``X_p`` leaves more water in the product, so less evaporates, the exhaust gets warmer
    and drier and its RH falls: ``g(X) = X - X_eq - excess(RH(X))`` rises monotonically and has
    exactly one root in ``[X_eq, X_eq + excess_max]``.
    """

    spec: MoistureSpec
    half_saturation_rh: float
    """``K``; fixed by :func:`plant_model` from the excess at the source's nominal point."""

    def excess(self, rh: np.ndarray) -> np.ndarray:
        rh = np.maximum(np.asarray(rh, dtype=np.float64), 0.0)
        return np.asarray(self.spec.excess_max * rh / (rh + self.half_saturation_rh))

    def solve(self, bed: BedFunction, equilibrium: float, n: int) -> np.ndarray:
        lo = np.full(n, equilibrium)
        hi = np.full(n, equilibrium + self.spec.excess_max)
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            g = mid - equilibrium - self.excess(bed(mid).relative_humidity)
            lo, hi = np.where(g < 0.0, mid, lo), np.where(g < 0.0, hi, mid)
        return np.asarray(0.5 * (lo + hi))


@dataclass(frozen=True)
class DryerPlantModel:
    """The balance (all extras off) or a synthetic plant (see the module doc)."""

    source: SourceSpec
    efficiency: EfficiencyModel
    moisture: MoistureModel
    heat_loss_w_per_k_per_kg_h: float = 0.0
    atomising_air_share: float = 0.0
    atomising_air_celsius: float = 25.0
    false_air_share: float = 0.0

    # ---------------------------------------------------------------- pieces
    @property
    def _air_kg_per_s(self) -> float:
        return self.source.air_kg_per_h / 3600.0

    def _properties(self) -> BlockProperties:
        s = self.source
        return BlockProperties(
            cp_gas_j_per_kg_k=s.cp_gas_j_per_kg_k,
            cp_vapour_j_per_kg_k=s.cp_vapour_j_per_kg_k,
            cp_liquid_j_per_kg_k=s.cp_liquid_j_per_kg_k,
            cp_solid_j_per_kg_k=s.cp_solid_j_per_kg_k,
            evaporation_enthalpy_0c_j_per_kg=s.evaporation_enthalpy_0c_j_per_kg,
            heat_loss_ua_w_per_k=self.heat_loss_w_per_k_per_kg_h * s.air_kg_per_h,
        )

    def specific_duty_kj_per_kg(self, x: np.ndarray) -> np.ndarray:
        """Heater duty per kg of process air: ``(cp_g + Y_in cp_v) max(T_in - T_amb, 0)``."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        s = self.source
        y_in = x[:, 3] / 1e3
        humid = s.cp_gas_j_per_kg_k + y_in * s.cp_vapour_j_per_kg_k
        return np.asarray(humid * np.maximum(x[:, 0] - x[:, 2], 0.0) / 1e3)

    def nominal_specific_duty(self) -> float:
        return float(self.specific_duty_kj_per_kg(np.asarray([self.source.nominal]))[0])

    def bed(self, x: np.ndarray, product_moisture: np.ndarray) -> BedState:
        """Bed exhaust per point via :func:`dryer_block` (point by point)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        props = self._properties()
        m_g = self._air_kg_per_s
        a = self.atomising_air_share
        t_out = np.empty(len(x))
        y_out = np.empty(len(x))
        evap = np.empty(len(x))
        for i, (t_in, load, t_amb, y_in_g, xs) in enumerate(x):
            result = dryer_block(
                BlockInputs(
                    dry_air_mass_flow_kg_per_s=m_g * (1.0 + a),
                    air_humidity_kg_per_kg=y_in_g / 1e3,
                    air_temperature_celsius=(t_in + a * self.atomising_air_celsius) / (1.0 + a),
                    spray_mass_flow_kg_per_s=load * m_g,
                    spray_solid_fraction_kg_per_kg=xs,
                    spray_temperature_celsius=self.source.spray_temperature_celsius,
                    product_moisture_kg_per_kg=float(product_moisture[i]),
                    ambient_temperature_celsius=t_amb,
                ),
                props,
            )
            t_out[i] = result.outlet_temperature_celsius
            y_out[i] = result.outlet_humidity_kg_per_kg
            evap[i] = result.evaporation_kg_per_s / m_g
        rh = relative_humidity(y_out, t_out, self.source.pressure_pa)
        return BedState(t_out, y_out, rh, evap)

    # ---------------------------------------------------------------- public
    def solve(self, x: np.ndarray) -> tuple[np.ndarray, BedState, np.ndarray]:
        """Targets ``(n, 3)``, the bed state and the product moisture per point."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        n = len(x)
        x_p = self.moisture.solve(
            lambda moist: self.bed(x, moist), self.source.equilibrium_moisture_kg_per_kg, n
        )
        bed = self.bed(x, x_p)
        s = self.source
        a, f = self.atomising_air_share, self.false_air_share
        y_in = x[:, 3] / 1e3
        c_bed = (1.0 + a) * (s.cp_gas_j_per_kg_k + bed.humidity_kg_per_kg * s.cp_vapour_j_per_kg_k)
        c_false = f * (s.cp_gas_j_per_kg_k + y_in * s.cp_vapour_j_per_kg_k)
        t_meas = (c_bed * bed.temperature_celsius + c_false * x[:, 2]) / (c_bed + c_false)
        y_meas = ((1.0 + a) * bed.humidity_kg_per_kg + f * y_in) / (1.0 + a + f)
        duty = self.specific_duty_kj_per_kg(x)
        eta = self.efficiency.efficiency(duty / self.nominal_specific_duty())
        y = np.column_stack([t_meas, 1e3 * y_meas, duty / eta])
        return y, bed, x_p

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.solve(x)[0]


def source_model(source: SourceSpec) -> DryerPlantModel:
    """L0: the partner balance (constant efficiency, equilibrium moisture, no extras)."""
    return DryerPlantModel(source, ConstantEfficiency(source.heater_efficiency), ConstantMoisture())


def plant_model(source: SourceSpec, spec: PlantSpec) -> DryerPlantModel:
    """A synthetic plant; ``K`` of the moisture law from the excess at the nominal point."""
    efficiency: EfficiencyModel = (
        ConstantEfficiency(source.heater_efficiency)
        if spec.efficiency is None
        else LoadDependentEfficiency(spec.efficiency)
    )
    moisture: MoistureModel = ConstantMoisture()
    if spec.moisture is not None:
        # RH of the plain balance at the nominal point when the product carries the nominal excess
        base = source_model(source)
        nominal = np.asarray([source.nominal])
        x_p = np.asarray([source.equilibrium_moisture_kg_per_kg + spec.moisture.excess_nominal])
        rh_nominal = float(base.bed(nominal, x_p).relative_humidity[0])
        ratio = spec.moisture.excess_max / spec.moisture.excess_nominal
        moisture = RhDependentMoisture(spec.moisture, rh_nominal * (ratio - 1.0))
    return DryerPlantModel(
        source=source,
        efficiency=efficiency,
        moisture=moisture,
        heat_loss_w_per_k_per_kg_h=spec.heat_loss_w_per_k_per_kg_h,
        atomising_air_share=spec.atomising_air_share,
        atomising_air_celsius=spec.atomising_air_celsius,
        false_air_share=spec.false_air_share,
    )
