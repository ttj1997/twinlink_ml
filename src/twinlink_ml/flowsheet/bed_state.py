"""Dryer 2 of the partner workbook in the laboratory grey-box model (an extrapolation).

The "physical" side of the ``product_moisture`` switch: the moisture the product leaves the
fluidised bed with, instead of the sheet's equilibrium moisture of 2 g/kg. Lifted unchanged from
``scripts/report_heat_recovery_comparison.py`` (``bed_exhaust_at_partner_inputs``, Befund 90),
with the path of the process configuration as an argument.

Inputs of dryer 2 (300 degC, 12 kg/h spray, 100 kg/h dry air of 6 g/kg at 15 degC); the gas
velocity is the root of the dry-air flow. **Every one of them is outside the laboratory model's
range** (docs/11, 9.6); the bounds of the process configuration are widened in memory for this
call only. The state is what the model computes outside its range, not a prediction.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

from twinlink_ml.simulation.backend import NumpyBackend

__all__ = ["BedState", "bed_state_at_partner_inputs", "default_config_path"]

SOLID_FRACTION = 0.40
"""Suspension solid fraction of this comparison (run parameter; the config default stays 0.3)."""


def default_config_path() -> Path:
    """``conf/process/greybox_gab_placeholder.yaml`` of the checkout this code is loaded from."""
    return Path(__file__).resolve().parents[3] / "conf" / "process" / "greybox_gab_placeholder.yaml"


@dataclass(frozen=True)
class BedState:
    """Steady state of the laboratory model at the partner's inputs of dryer 2."""

    label: str
    solid_fraction: float
    gas_velocity_m_per_s: float
    dry_air_kg_per_h: float
    exhaust_temperature_celsius: float
    exhaust_humidity_kg_per_kg: float
    product_moisture_kg_per_kg: float
    feasible: bool
    residual_inf_norm: float
    config_fingerprint_base: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def bed_state_at_partner_inputs(
    config_path: Path | None = None, solid_fraction: float = SOLID_FRACTION
) -> BedState:
    """Solve the laboratory model for dryer 2 of the partner workbook.

    ``config_path`` defaults to :func:`default_config_path`.
    """
    config_path = config_path or default_config_path()
    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
    from twinlink_ml.simulation.greybox import GreyBoxFluidBedGranulator
    from twinlink_ml.simulation.operating_point import (
        DEFAULT_SEARCH_MAX_ITERATIONS,
        DEFAULT_SEARCH_TOLERANCE,
        build_variant_config,
    )
    from twinlink_ml.simulation.steady_state import solve_steady_state
    from twinlink_ml.simulation.thermo import relative_humidity_from_absolute_humidity

    backend = NumpyBackend()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(config_path)
    t_in, spray, air_kg_h, y_in, t_s, t_amb, nozzle_bar = 300.0, 12.0, 100.0, 0.006, 15.0, 15.0, 1.0
    phi = float(
        relative_humidity_from_absolute_humidity(
            backend, config.gas, np.array(y_in), np.array(t_amb)
        )
    )

    def variant_for(u: float) -> Any:
        actions = dataclasses.replace(
            config.actions,
            inlet_temperature_celsius=dataclasses.replace(
                config.actions.inlet_temperature_celsius, maximum=t_in, nominal=t_in
            ),
            gas_velocity_m_per_s=dataclasses.replace(
                config.actions.gas_velocity_m_per_s, nominal=u
            ),
            spray_rate_kg_per_h=dataclasses.replace(
                config.actions.spray_rate_kg_per_h, maximum=spray, nominal=spray
            ),
            nozzle_pressure_bar=dataclasses.replace(
                config.actions.nozzle_pressure_bar, nominal=nozzle_bar
            ),
        )
        disturbances = dataclasses.replace(
            config.disturbances,
            ambient_temperature_celsius=dataclasses.replace(
                config.disturbances.ambient_temperature_celsius, nominal=t_amb
            ),
            ambient_relative_humidity=dataclasses.replace(
                config.disturbances.ambient_relative_humidity, nominal=phi
            ),
            suspension_solid_fraction=dataclasses.replace(
                config.disturbances.suspension_solid_fraction, nominal=solid_fraction
            ),
        )
        operation = dataclasses.replace(config.operation, suspension_temperature_celsius=t_s)
        return dataclasses.replace(
            config, actions=actions, disturbances=disturbances, operation=operation
        )

    def solve(u: float) -> Any:
        variant = build_variant_config(
            variant_for(u), t_in=t_in, u_gas=u, p_nozzle=nozzle_bar, t_amb=t_amb, phi_amb=phi
        )
        return solve_steady_state(
            GreyBoxFluidBedGranulator(variant),
            max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=DEFAULT_SEARCH_TOLERANCE,
        )

    target = air_kg_h / 3600.0
    u_star = brentq(lambda u: solve(u).dry_air_mass_flow_kg_per_s - target, 1.3, 1.9, xtol=1e-6)
    solution = solve(u_star)
    return BedState(
        label="extrapolation: every input of dryer 2 is outside the laboratory model's range",
        solid_fraction=solid_fraction,
        gas_velocity_m_per_s=u_star,
        dry_air_kg_per_h=solution.dry_air_mass_flow_kg_per_s * 3600.0,
        exhaust_temperature_celsius=solution.bed_temperature_celsius,
        exhaust_humidity_kg_per_kg=solution.outlet_humidity_kg_per_kg,
        product_moisture_kg_per_kg=solution.particle_moisture_kg_per_kg,
        feasible=bool(solution.feasible),
        residual_inf_norm=solution.residual_inf_norm,
        config_fingerprint_base=config.fingerprint,
    )
