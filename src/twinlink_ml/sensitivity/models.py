"""The two steady-state models of the Sobol study as functions of the uncertain inputs.

Neither model is changed; both wrappers only set the inputs and read the results.

P1 -- :class:`LineModel`
    One line heater -> dryer -> heat recovery (:class:`twinlink_ml.flowsheet.line.Line`) in the
    physical mode, at data set 5 of delivery 4.1_2 (docs/11 section 12). The dryer's heat loss is
    fixed at 0 inside ``Line.dryer_unit``; :class:`_LineWithHeatLoss` builds the same dryer unit
    with the sampled ``UA`` (the only composition repeated here). The fan is not modelled in P1.
P2 -- :class:`GreyBoxModel`
    The lab line of the dynamic headroom study (docs/24). Fixed operating point: the exact
    steady state (:func:`twinlink_ml.simulation.steady_state.solve_steady_state`) at the command
    that is the steady optimum at the nominal disturbance and nominal parameters with the R0
    margin. Optimum: the RTO solver of that study
    (:class:`twinlink_ml.headroom.controllers.SteadyStateOptimizer`) with the R1 margin, rebuilt
    per sample (the parameters are constants of its symbolic model), and checked with the exact
    steady state at the optimal command (validity flags and consistency of the moisture).

Each evaluation returns every output with its own validity and the reasons for invalid parts,
so that invalid samples are counted, never dropped silently.
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.flowsheet.dryer import DryerBlock
from twinlink_ml.flowsheet.line import Line, LineInputs, LineMode
from twinlink_ml.flowsheet.validators import MOLAR_MASS_RATIO, water_saturation_pressure_pa
from twinlink_ml.partner_v4.inputs import ATM_PA, SiDryer, SiMaterial

from .config import REPO_ROOT, InputSpec

__all__ = [
    "Evaluation",
    "GreyBoxModel",
    "LineModel",
    "OutputSpec",
    "SteadyModel",
    "classify_steady_state_reason",
]


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """One model output: label of the request ((a)-(e)), unit, description."""

    name: str
    label: str
    unit: str
    description: str


@dataclass(frozen=True)
class Evaluation:
    """Outputs of one model run, each with its validity, and the reasons for invalid parts."""

    outputs: dict[str, float]
    valid: dict[str, bool]
    reasons: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


class SteadyModel(ABC):
    """A steady-state model as a function of named inputs."""

    name: str
    inputs: tuple[InputSpec, ...]
    outputs: tuple[OutputSpec, ...]

    def nominal_values(self) -> dict[str, float]:
        return {s.name: s.nominal for s in self.inputs}

    @abstractmethod
    def evaluate(self, values: Mapping[str, float]) -> Evaluation:
        """Run the model with ``values`` (missing inputs at their nominal values)."""

    def _complete(self, values: Mapping[str, float]) -> dict[str, float]:
        unknown = set(values) - {s.name for s in self.inputs}
        if unknown:
            raise KeyError(f"unknown inputs {sorted(unknown)}")
        return {**self.nominal_values(), **values}


# ---------------------------------------------------------------------------------------- P1
def _humidity_from_relative(
    relative: float, temperature_celsius: float, pressure_pa: float
) -> float:
    p_v = relative * water_saturation_pressure_pa(temperature_celsius)
    return MOLAR_MASS_RATIO * p_v / (pressure_pa - p_v)


def _relative_from_humidity(
    humidity: float, temperature_celsius: float, pressure_pa: float
) -> float:
    return (
        humidity
        * pressure_pa
        / ((MOLAR_MASS_RATIO + humidity) * water_saturation_pressure_pa(temperature_celsius))
    )


P1_OUTPUTS = (
    OutputSpec("heater_duty_kw", "(a)", "kW", "heater duty at the fixed operating point"),
    OutputSpec("fuel_kg_per_h", "(a)", "kg/h", "fuel at the fixed operating point"),
    OutputSpec(
        "co2_kg_per_h", "(a)", "kg/h", "CO2 of the fuel (proportional to the fuel: same indices)"
    ),
    OutputSpec("recovered_kw", "(e)", "kW", "heat recovered from the exhaust"),
    OutputSpec(
        "dew_point_distance_k",
        "(e)",
        "K",
        "exhaust temperature after the recovery minus its dew point",
    ),
)


class LineModel(SteadyModel):
    """P1: one line in the physical mode at data set 5 (see the module docstring)."""

    outputs = P1_OUTPUTS

    def __init__(
        self, settings: Mapping[str, Any], inputs: tuple[InputSpec, ...], mode: str
    ) -> None:
        self.name = f"p1/{mode}"
        self.settings = dict(settings)
        limit = settings["modes"][mode]["htm_max_celsius"]
        self.mode = LineMode.physical(
            htm_max_celsius=None if limit is None else float(limit),
            dt_min_k=next(s.nominal for s in inputs if s.name == "dt_min_k"),
        )
        record = json.loads((REPO_ROOT / settings["material_from"]).read_text(encoding="utf-8"))
        self.material = SiMaterial(**record["parameters"]["material_si"])
        self.pressure_pa = ATM_PA
        nominal_t = next(s.nominal for s in inputs if s.name == "ambient_temperature_celsius")
        nominal_rh = _relative_from_humidity(
            float(settings["nominal_air_humidity_kg_per_kg"]), nominal_t, ATM_PA
        )
        self.inputs = tuple(
            dataclasses.replace(s, nominal=nominal_rh) if math.isnan(s.nominal) else s
            for s in inputs
        )

    def evaluate(self, values: Mapping[str, float]) -> Evaluation:
        v = self._complete(values)
        s = self.settings
        ambient = v["ambient_temperature_celsius"]
        humidity = _humidity_from_relative(
            v["ambient_relative_humidity"], ambient, self.pressure_pa
        )
        material = dataclasses.replace(
            self.material,
            heater_efficiency=v["heater_efficiency"],
            lower_heating_value_j_per_kg=float(s["lower_heating_value_j_per_kg"]),
            cp_vapour_j_per_kg_k=v["cp_vapour_j_per_kg_k"],
            equilibrium_moisture_kg_per_kg=v["equilibrium_moisture_kg_per_kg"],
        )
        dryer = SiDryer(
            float(s["air_kg_per_h"]) / 3600.0,
            humidity,
            float(s["air_temperature_celsius"]),
            v["spray_kg_per_h"] / 3600.0,
            v["solid_fraction"],
            float(s["spray_temperature_celsius"]),
        )
        inputs = LineInputs(
            dryer,
            ambient,
            material,
            float(s["medium_kg_per_h"]) / 3600.0,
            float(s["medium_cp_j_per_kg_k"]),
            self.pressure_pa,
        )
        mode: LineMode = dataclasses.replace(self.mode, dt_min_k=v["dt_min_k"])
        nan = dict.fromkeys((o.name for o in self.outputs), math.nan)
        try:
            result = _LineWithHeatLoss(mode, v["heat_loss_ua_w_per_k"]).solve(inputs)
        except (ValueError, RuntimeError, ArithmeticError) as error:
            return Evaluation(
                nan, dict.fromkeys(nan, False), (f"line_error:{type(error).__name__}",)
            )
        reasons = []
        if result.dryer.saturation_limited:
            reasons.append("dryer_saturation_limited")
        for finding in result.findings():
            if finding.severity == "violation":
                reasons.append(f"check_violation:{finding.check}")
        stage = result.stage
        fuel = result.fuel_kg_per_h
        outputs = {
            "heater_duty_kw": result.heater_duty_kw,
            "fuel_kg_per_h": fuel,
            "co2_kg_per_h": fuel
            * float(s["lower_heating_value_j_per_kg"])
            / 3.6e6
            * float(s["gas_co2_kg_per_kwh"]),
            "recovered_kw": result.recovered_kw,
            "dew_point_distance_k": stage.exhaust_out_celsius - stage.exhaust_dew_point_celsius,
        }
        ok = not reasons and all(math.isfinite(x) for x in outputs.values())
        return Evaluation(
            outputs,
            dict.fromkeys(outputs, ok),
            tuple(reasons),
            {"binding": stage.binding, "medium_hot_celsius": result.medium_hot_celsius},
        )


class _LineWithHeatLoss(Line):
    """``Line`` whose dryer unit carries the heat loss ``UA (T_out - T_amb)``."""

    def __init__(self, mode: LineMode, heat_loss_ua_w_per_k: float) -> None:
        super().__init__(mode)
        self.heat_loss_ua_w_per_k = heat_loss_ua_w_per_k

    def dryer_unit(self, inputs: LineInputs) -> DryerBlock:
        unit = super().dryer_unit(inputs)
        return DryerBlock(
            dataclasses.replace(unit.config, heat_loss_ua_w_per_k=self.heat_loss_ua_w_per_k)
        )


# ---------------------------------------------------------------------------------------- P2
P2_OUTPUTS = (
    OutputSpec("energy_fixed_kw", "(a)", "kW", "model total power at the fixed operating point"),
    OutputSpec(
        "co2_fixed_kg_per_h", "(a)", "kg/h", "CO2 at the fixed operating point (gas heater)"
    ),
    OutputSpec(
        "moisture_fixed_kg_per_kg", "(a)", "kg/kg", "product moisture at the fixed operating point"
    ),
    OutputSpec(
        "fixed_violates_limit",
        "(a)",
        "-",
        "indicator: the fixed operating point exceeds the quality limit (+ tolerance)",
    ),
    OutputSpec("energy_optimum_kw", "(b)", "kW", "model total power at the steady optimum"),
    OutputSpec("co2_optimum_kg_per_h", "(b)", "kg/h", "CO2 at the steady optimum (gas heater)"),
    OutputSpec(
        "inlet_temperature_optimum_celsius", "(c)", "degC", "optimal inlet temperature command"
    ),
    OutputSpec("gas_velocity_optimum_m_per_s", "(c)", "m/s", "optimal gas velocity command"),
    OutputSpec(
        "savings_all",
        "(d)",
        "-",
        "(E_fixed - E_optimum) / E_fixed on every sample with both parts valid",
    ),
    OutputSpec(
        "savings_fixed_feasible",
        "(d)",
        "-",
        "the same, only where the fixed operating point holds the quality limit",
    ),
)

_REASON_KEYS = (
    ("u_mf", "fluidisation_u_mf"),
    ("u_t", "fluidisation_u_t"),
    ("Trocknungsregime", "no_drying_regime"),
    ("Suchgrenze", "bed_temperature_at_search_bound"),
    ("Sättigungsgrad", "saturation_degree_above_1"),
    ("Partikelfeuchte", "moisture_outside_eq_crit"),
)


def classify_steady_state_reason(reason: str) -> str:
    """Short category of an infeasibility reason of ``solve_steady_state``."""
    for key, label in _REASON_KEYS:
        if key in reason:
            return label
    return "other"


class GreyBoxModel(SteadyModel):
    """P2: fixed operating point and steady optimum of the lab line (see the module docstring)."""

    outputs = P2_OUTPUTS

    def __init__(self, settings: Mapping[str, Any], inputs: tuple[InputSpec, ...]) -> None:
        from twinlink_ml.headroom.config import load_headroom_config, study_process_config
        from twinlink_ml.headroom.controllers import SteadyStateOptimizer, nominal_disturbance

        self.name = "p2"
        self.settings = dict(settings)
        self.inputs = inputs
        self.study = load_headroom_config(Path(REPO_ROOT / settings["headroom_config"]))
        self.base = study_process_config(self.study)
        self.limit = self.study.line.product_moisture_limit_kg_per_kg
        self.margin_fixed = float(settings["margin_fixed_kg_per_kg"])
        self.margin_optimum = float(settings["margin_optimum_kg_per_kg"])
        self.tolerance = float(settings["violation_tolerance_kg_per_kg"])
        nominal = self.nominal_values()
        self._check_nominal_matches_base(nominal)
        point = SteadyStateOptimizer(self.study, self.base).solve(
            nominal_disturbance(self.study, self.base), self.limit - self.margin_fixed
        )
        self.fixed_command = (float(point.command[0]), float(point.command[1]))
        self.fixed_nominal_power_kw = float(point.power_kw)

    def _check_nominal_matches_base(self, nominal: Mapping[str, float]) -> None:
        """The nominal inputs must be the model's own values (the YAML only names them)."""
        b = self.base
        own = {
            "gab_monolayer_moisture_kg_per_kg": b.material.gab_monolayer_moisture_kg_per_kg,
            "gab_c_constant_at_reference_temperature": (
                b.material.gab_c_constant_at_reference_temperature
            ),
            "gab_k_constant_at_reference_temperature": (
                b.material.gab_k_constant_at_reference_temperature
            ),
            "gab_c_enthalpy_j_per_mol": b.material.gab_c_enthalpy_j_per_mol,
            "gab_k_enthalpy_j_per_mol": b.material.gab_k_enthalpy_j_per_mol,
            "max_saturation_degree": b.material.max_saturation_degree,
            "critical_moisture_kg_per_kg": b.material.critical_moisture_kg_per_kg,
            "heat_loss_ua_w_per_k": b.geometry.heat_loss_ua_w_per_k,
            "gas_heater_efficiency": self.study.control.gas_heater_efficiency,
            "cp_vapour_j_per_kg_k": b.gas.cp_vapour_j_per_kg_k,
            "ambient_temperature_celsius": b.disturbances.ambient_temperature_celsius.nominal,
            "ambient_relative_humidity": b.disturbances.ambient_relative_humidity.nominal,
            "spray_rate_kg_per_h": self.study.line.nominal_spray_rate_kg_per_h,
            "suspension_solid_fraction": self.study.line.nominal_solid_fraction,
        }
        for name, value in own.items():
            if value is None or abs(float(value) - nominal[name]) > 1e-12 * max(1.0, abs(value)):
                raise ValueError(f"nominal of {name} ({nominal[name]}) is not the model's {value}")

    def process_config(self, v: Mapping[str, float]) -> Any:
        """Process config with the epistemic inputs of ``v`` (raises on invalid values)."""
        b = self.base
        return dataclasses.replace(
            b,
            material=dataclasses.replace(
                b.material,
                gab_monolayer_moisture_kg_per_kg=v["gab_monolayer_moisture_kg_per_kg"],
                gab_c_constant_at_reference_temperature=v[
                    "gab_c_constant_at_reference_temperature"
                ],
                gab_k_constant_at_reference_temperature=v[
                    "gab_k_constant_at_reference_temperature"
                ],
                gab_c_enthalpy_j_per_mol=v["gab_c_enthalpy_j_per_mol"],
                gab_k_enthalpy_j_per_mol=v["gab_k_enthalpy_j_per_mol"],
                max_saturation_degree=v["max_saturation_degree"],
                critical_moisture_kg_per_kg=v["critical_moisture_kg_per_kg"],
            ),
            geometry=dataclasses.replace(
                b.geometry, heat_loss_ua_w_per_k=v["heat_loss_ua_w_per_k"]
            ),
            gas=dataclasses.replace(b.gas, cp_vapour_j_per_kg_k=v["cp_vapour_j_per_kg_k"]),
        )

    @staticmethod
    def disturbance(v: Mapping[str, float]) -> np.ndarray:
        """Disturbance vector in the order of ``headroom.symbolic.DISTURBANCE_CHANNELS``."""
        return np.array(
            [
                v["ambient_temperature_celsius"],
                v["ambient_relative_humidity"],
                v["suspension_solid_fraction"],
                v["spray_rate_kg_per_h"],
                0.0,
            ]
        )

    def steady_state(self, config: Any, command: tuple[float, float], d: np.ndarray) -> Any:
        """Exact steady state at a fixed command and disturbance (the project's solver)."""
        from twinlink_ml.simulation import GreyBoxFluidBedGranulator
        from twinlink_ml.simulation.sorption import isotherm_from_material, sorption_safety_grid
        from twinlink_ml.simulation.steady_state import solve_steady_state

        a, dist = config.actions, config.disturbances
        cfg = dataclasses.replace(
            config,
            actions=dataclasses.replace(
                a,
                inlet_temperature_celsius=dataclasses.replace(
                    a.inlet_temperature_celsius, nominal=float(command[0])
                ),
                gas_velocity_m_per_s=dataclasses.replace(
                    a.gas_velocity_m_per_s, nominal=float(command[1])
                ),
                spray_rate_kg_per_h=dataclasses.replace(a.spray_rate_kg_per_h, nominal=float(d[3])),
            ),
            disturbances=dataclasses.replace(
                dist,
                ambient_temperature_celsius=dataclasses.replace(
                    dist.ambient_temperature_celsius, nominal=float(d[0])
                ),
                ambient_relative_humidity=dataclasses.replace(
                    dist.ambient_relative_humidity, nominal=float(d[1])
                ),
                suspension_solid_fraction=dataclasses.replace(
                    dist.suspension_solid_fraction, nominal=float(d[2])
                ),
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulator = GreyBoxFluidBedGranulator(cfg)
        # inner bracket: the solver's own lower end, a wider upper end (see the YAML)
        backend = simulator.backend
        phi_min, _, _, t_max = sorption_safety_grid(backend, cfg.gas, cfg.actions, cfg.disturbances)
        low = float(
            backend.to_numpy(
                isotherm_from_material(cfg.material).equilibrium_moisture(
                    backend, backend.asarray(phi_min), backend.asarray(t_max)
                )
            )
        )
        return solve_steady_state(
            simulator,
            moisture_bracket_kg_per_kg=(
                low,
                float(self.settings["steady_state_moisture_bracket_high_kg_per_kg"]),
            ),
            max_iterations=int(self.settings["steady_state_max_iterations"]),
            tolerance=float(self.settings["steady_state_tolerance"]),
        )

    def _co2(self, diagnostics: Mapping[str, float], gas_eta: float) -> float:
        from twinlink_ml.evaluation.carrier_energy import carrier_energy_from_diagnostics

        return carrier_energy_from_diagnostics(
            dict(diagnostics),
            heater="gas",
            process_heater_efficiency=self.base.energy.heater_efficiency,
            gas_heater_efficiency=gas_eta,
        ).co2_kg_per_h

    def evaluate(self, values: Mapping[str, float]) -> Evaluation:
        from twinlink_ml.headroom.controllers import SteadyStateOptimizer
        from twinlink_ml.simulation.sorption import validate_isotherm_against_critical_moisture

        v = self._complete(values)
        out = dict.fromkeys((o.name for o in self.outputs), math.nan)
        reasons: list[str] = []
        try:
            config = self.process_config(v)
            validate_isotherm_against_critical_moisture(config)
        except ValueError:
            reasons.append("config_rejected:isotherm_or_material_check")
            return Evaluation(out, dict.fromkeys(out, False), tuple(reasons))
        d = self.disturbance(v)
        gas_eta = v["gas_heater_efficiency"]

        # fixed operating point
        fixed_ok = False
        try:
            fixed = self.steady_state(config, self.fixed_command, d)
        except (RuntimeError, ValueError):
            reasons.append("fixed:solver_error")
        else:
            if fixed.feasible:
                fixed_ok = True
            else:
                reasons.extend(
                    f"fixed:{classify_steady_state_reason(r)}" for r in fixed.infeasibility_reasons
                )
            out["energy_fixed_kw"] = fixed.diagnostics["power_total_kw"]
            out["co2_fixed_kg_per_h"] = self._co2(fixed.diagnostics, gas_eta)
            out["moisture_fixed_kg_per_kg"] = fixed.particle_moisture_kg_per_kg
            out["fixed_violates_limit"] = float(
                fixed.particle_moisture_kg_per_kg > self.limit + self.tolerance
            )

        # steady optimum (RTO solver), checked with the exact steady state
        optimum_ok = False
        try:
            point = SteadyStateOptimizer(self.study, config).solve(
                d, self.limit - self.margin_optimum
            )
        except RuntimeError:
            reasons.append("optimum:ipopt_failed")
        else:
            command = (float(point.command[0]), float(point.command[1]))
            out["inlet_temperature_optimum_celsius"] = command[0]
            out["gas_velocity_optimum_m_per_s"] = command[1]
            try:
                check = self.steady_state(config, command, d)
            except (RuntimeError, ValueError):
                reasons.append("optimum:steady_state_solver_error")
            else:
                mismatch = abs(
                    check.particle_moisture_kg_per_kg - float(point.state[1] / point.state[0])
                )
                if not check.feasible:
                    reasons.extend(
                        f"optimum:{classify_steady_state_reason(r)}"
                        for r in check.infeasibility_reasons
                    )
                elif mismatch > float(self.settings["consistency_tolerance_kg_per_kg"]):
                    reasons.append("optimum:ipopt_exact_mismatch")
                else:
                    optimum_ok = True
                out["energy_optimum_kw"] = check.diagnostics["power_total_kw"]
                out["co2_optimum_kg_per_h"] = self._co2(check.diagnostics, gas_eta)

        both = fixed_ok and optimum_ok
        if both:
            out["savings_all"] = (out["energy_fixed_kw"] - out["energy_optimum_kw"]) / out[
                "energy_fixed_kw"
            ]
            out["savings_fixed_feasible"] = out["savings_all"]
        violates = bool(out["fixed_violates_limit"] == 1.0)
        valid = {
            "energy_fixed_kw": fixed_ok,
            "co2_fixed_kg_per_h": fixed_ok,
            "moisture_fixed_kg_per_kg": fixed_ok,
            "fixed_violates_limit": fixed_ok,
            "energy_optimum_kw": optimum_ok,
            "co2_optimum_kg_per_h": optimum_ok,
            "inlet_temperature_optimum_celsius": optimum_ok,
            "gas_velocity_optimum_m_per_s": optimum_ok,
            "savings_all": both,
            "savings_fixed_feasible": both and not violates,
        }
        return Evaluation(out, valid, tuple(reasons), {"fixed_violates": violates})
