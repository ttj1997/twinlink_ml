r"""Operating-point constraints, penalty constants and variant configs (Befund 52, 62).

Moved unchanged from ``scripts/study_operating_point_regret.py`` (S0.1) so that
``rl/env.py`` no longer has to load a script by ``importlib``. The script re-imports every
name from here, so existing probe scripts that read them off the loaded script module keep
working. Behaviour is bit-identical; the VA2 oracle test pins this.

Constants
---------
``PENALTY_WEIGHT_KW`` scales the normalised violation into the order of magnitude of the
objective (about 0.5-3 kW at the nominal point). ``SOLVE_FAILURE_PENALTY_KW`` is the value
returned when the bisection in ``solve_steady_state`` finds no sign change. It is only safe
for an ordinal optimiser (``differential_evolution``); a regression-based critic must not
be given it (Befund 62 EA3).

The three tolerances are one shared set for feasibility AND binding checks (Befund 52 e):
two different tolerances for the same quantity caused 8 of 14 false infeasibilities in the
first grid run. ``BED_TEMPERATURE_LIMIT_C`` is a freely set placeholder (Befund 46).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from .config import ProcessConfig
from .steady_state import SteadyStateSolution

__all__ = [
    "BED_TEMPERATURE_LIMIT_C",
    "DEFAULT_SEARCH_MAX_ITERATIONS",
    "DEFAULT_SEARCH_TOLERANCE",
    "PENALTY_WEIGHT_KW",
    "SOLVE_FAILURE_PENALTY_KW",
    "TOLERANCE_MOISTURE_KG_PER_KG",
    "TOLERANCE_TEMPERATURE_K",
    "TOLERANCE_VELOCITY_M_PER_S",
    "ConstraintCheck",
    "build_variant_config",
    "check_constraints",
    "specific_energy_kwh_per_kg",
]

BED_TEMPERATURE_LIMIT_C = 80.0
PENALTY_WEIGHT_KW = 50.0
SOLVE_FAILURE_PENALTY_KW = 1.0e6
TOLERANCE_MOISTURE_KG_PER_KG = 1.0e-4
TOLERANCE_VELOCITY_M_PER_S = 1.0e-3
TOLERANCE_TEMPERATURE_K = 0.05

DEFAULT_SEARCH_MAX_ITERATIONS = 30
DEFAULT_SEARCH_TOLERANCE = 1.0e-7
"""Search precision of Befund 52. The solver defaults (100 / 1e-13) do NOT reproduce it:
6 of the 25 grid points fail with a bracket sign error (Befund 62 b)."""


def build_variant_config(
    base: ProcessConfig, *, t_in: float, u_gas: float, p_nozzle: float, t_amb: float, phi_amb: float
) -> ProcessConfig:
    """Config with exactly the five varied nominal values overridden.

    ``solve_steady_state`` reads actions and disturbances only through the ``nominal``
    fields of the config, so one variant per evaluation is the only way to evaluate the
    solver at an arbitrary operating point. All other fields (including bounds and
    ``max_rate_per_s``) stay unchanged.
    """
    actions = dataclasses.replace(
        base.actions,
        inlet_temperature_celsius=dataclasses.replace(
            base.actions.inlet_temperature_celsius, nominal=t_in
        ),
        gas_velocity_m_per_s=dataclasses.replace(base.actions.gas_velocity_m_per_s, nominal=u_gas),
        nozzle_pressure_bar=dataclasses.replace(base.actions.nozzle_pressure_bar, nominal=p_nozzle),
    )
    disturbances = dataclasses.replace(
        base.disturbances,
        ambient_temperature_celsius=dataclasses.replace(
            base.disturbances.ambient_temperature_celsius, nominal=t_amb
        ),
        ambient_relative_humidity=dataclasses.replace(
            base.disturbances.ambient_relative_humidity, nominal=phi_amb
        ),
    )
    return dataclasses.replace(base, actions=actions, disturbances=disturbances)


@dataclass(frozen=True, slots=True)
class ConstraintCheck:
    """The three hard constraints plus solver validity, evaluated on one solution."""

    moisture_kg_per_kg: float
    moisture_violation_kg_per_kg: float
    gas_velocity_m_per_s: float
    minimum_fluidization_velocity_m_per_s: float
    terminal_velocity_m_per_s: float
    bed_temperature_celsius: float
    bed_temperature_violation_k: float
    solver_feasible: bool
    feasible: bool
    binding_moisture: bool
    binding_gas_lower: bool
    binding_gas_upper: bool
    binding_bed_temperature: bool
    normalized_violation: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def check_constraints(
    solution: SteadyStateSolution,
    *,
    critical_moisture_kg_per_kg: float,
    gas_velocity_m_per_s: float,
) -> ConstraintCheck:
    """``critical_moisture_kg_per_kg`` is the moisture THRESHOLD of the constraint.

    The argument keeps its historical name; callers pass the quality limit (S0.2), which
    equals the critical moisture unless ``MaterialConfig.product_moisture_limit_kg_per_kg``
    is set.
    """
    moisture = solution.particle_moisture_kg_per_kg
    moisture_violation = max(0.0, moisture - critical_moisture_kg_per_kg)

    u_mf = solution.diagnostics["minimum_fluidization_velocity_m_per_s"]
    u_terminal = solution.diagnostics["terminal_velocity_m_per_s"]
    gas_lower_violation = max(0.0, u_mf - gas_velocity_m_per_s)
    gas_upper_violation = max(0.0, gas_velocity_m_per_s - u_terminal)

    bed_temperature_violation = max(0.0, solution.bed_temperature_celsius - BED_TEMPERATURE_LIMIT_C)

    feasible = (
        moisture_violation <= TOLERANCE_MOISTURE_KG_PER_KG
        and gas_lower_violation <= TOLERANCE_VELOCITY_M_PER_S
        and gas_upper_violation <= TOLERANCE_VELOCITY_M_PER_S
        and bed_temperature_violation <= TOLERANCE_TEMPERATURE_K
        and solution.feasible
    )

    window = max(u_terminal - u_mf, 1e-3)
    normalized_violation = (
        moisture_violation / max(critical_moisture_kg_per_kg, 1e-6)
        + gas_lower_violation / window
        + gas_upper_violation / window
        + bed_temperature_violation / 10.0
        + (0.0 if solution.feasible else 1.0)
    )

    return ConstraintCheck(
        moisture_kg_per_kg=moisture,
        moisture_violation_kg_per_kg=moisture_violation,
        gas_velocity_m_per_s=gas_velocity_m_per_s,
        minimum_fluidization_velocity_m_per_s=u_mf,
        terminal_velocity_m_per_s=u_terminal,
        bed_temperature_celsius=solution.bed_temperature_celsius,
        bed_temperature_violation_k=bed_temperature_violation,
        solver_feasible=solution.feasible,
        feasible=feasible,
        binding_moisture=abs(moisture - critical_moisture_kg_per_kg)
        <= TOLERANCE_MOISTURE_KG_PER_KG,
        binding_gas_lower=abs(gas_velocity_m_per_s - u_mf) <= TOLERANCE_VELOCITY_M_PER_S,
        binding_gas_upper=abs(u_terminal - gas_velocity_m_per_s) <= TOLERANCE_VELOCITY_M_PER_S,
        binding_bed_temperature=abs(BED_TEMPERATURE_LIMIT_C - solution.bed_temperature_celsius)
        <= TOLERANCE_TEMPERATURE_K,
        normalized_violation=normalized_violation,
    )


def specific_energy_kwh_per_kg(solution: SteadyStateSolution) -> float:
    """Total electrical-equivalent power per kg of product mass flow."""
    product_flow = solution.diagnostics["product_mass_flow_kg_per_s"]
    total_power_kw = solution.diagnostics["power_total_kw"]
    return total_power_kw / (3600.0 * max(product_flow, 1e-12))
