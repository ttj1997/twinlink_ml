"""Tests for ``simulation/exhaust.py`` -- back-calculation of the exhaust air outlet quantities.

What is checked is the wiring against the existing reference calculation from
``steady_state.py``/Befund 31 (``docs/04_stand_ap1.md``) -- with the TRUE, simulated
values as input, not with model predictions. Whether a trained sequence model predicts
T_bed/``product_moisture_kg_per_kg`` well enough is a different, later question
(rollout quality) that this test does not answer.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from twinlink_ml.simulation import GreyBoxFluidBedGranulator, ProcessConfig
from twinlink_ml.simulation.backend import Array, ArrayBackend
from twinlink_ml.simulation.exhaust import reconstruct_exhaust_state
from twinlink_ml.simulation.sorption import isotherm_from_material
from twinlink_ml.simulation.steady_state import SteadyStateSolution, solve_steady_state


def _reconstruct_at_solution(
    config: ProcessConfig,
    simulator: GreyBoxFluidBedGranulator,
    backend: ArrayBackend,
) -> tuple[float, float, float, SteadyStateSolution]:
    """Back-calculation with the true values of the steady-state solution, plus the solution."""
    solution = solve_steady_state(simulator)
    assert solution.feasible, solution.infeasibility_reasons

    isotherm = isotherm_from_material(config.material)

    def as_array(value: float) -> Array:
        return backend.asarray(np.array([value], dtype=np.float64))

    t_gas_out, y_gas_out, outlet_temperature = reconstruct_exhaust_state(
        backend,
        config.gas,
        config.material,
        isotherm,
        config.exhaust,
        bed_temperature_celsius=as_array(solution.bed_temperature_celsius),
        particle_moisture_kg_per_kg=as_array(solution.particle_moisture_kg_per_kg),
        dry_air_mass_flow_kg_per_s=as_array(solution.dry_air_mass_flow_kg_per_s),
        inlet_absolute_humidity_kg_per_kg=as_array(
            solution.diagnostics["inlet_humidity_kg_per_kg"]
        ),
    )
    return (
        float(backend.to_numpy(t_gas_out)[0]),
        float(backend.to_numpy(y_gas_out)[0]),
        float(backend.to_numpy(outlet_temperature)[0]),
        solution,
    )


def test_reconstruction_matches_steady_state_reference_bitgenau(
    config: ProcessConfig, simulator: GreyBoxFluidBedGranulator, backend: ArrayBackend
) -> None:
    """At the nominal point (Befund 31) against ``solve_steady_state()`` -- bit-identical expected.

    Both paths call ``thermo.evaporation_rate_kg_per_s`` with the same inputs;
    a deviation would be a wiring error in ``exhaust.py``, not a physical
    question -- hence ``rel=1e-12``, not a generous tolerance.
    """
    t_gas_out, y_gas_out, _outlet_temperature, solution = _reconstruct_at_solution(
        config, simulator, backend
    )

    assert t_gas_out == solution.bed_temperature_celsius
    assert y_gas_out == pytest.approx(solution.outlet_humidity_kg_per_kg, rel=1e-12)


def test_reconstruction_matches_documented_befund_31_values(
    config: ProcessConfig, simulator: GreyBoxFluidBedGranulator, backend: ArrayBackend
) -> None:
    """Counter-check against the documented table value (``docs/04_stand_ap1.md``, Befund 31)."""
    _t_gas_out, y_gas_out, _outlet_temperature, _solution = _reconstruct_at_solution(
        config, simulator, backend
    )
    assert y_gas_out == pytest.approx(0.0171, abs=5e-5)


def test_outlet_temperature_matches_bed_temperature_at_zero_loss(
    config: ProcessConfig, simulator: GreyBoxFluidBedGranulator, backend: ArrayBackend
) -> None:
    """``outlet_duct_heat_loss_k`` default 0.0: bit-identical identity.

    ``outlet_temperature_celsius`` == ``bed_temperature_celsius`` -- no numerical
    change compared with the previous (wrongly named) behaviour.
    """
    assert config.exhaust.outlet_duct_heat_loss_k == 0.0
    t_gas_out, _y_gas_out, outlet_temperature, _solution = _reconstruct_at_solution(
        config, simulator, backend
    )
    assert outlet_temperature == t_gas_out


def test_outlet_temperature_subtracts_duct_loss(
    config: ProcessConfig, simulator: GreyBoxFluidBedGranulator, backend: ArrayBackend
) -> None:
    """With a set duct loss: cooler by exactly this amount.

    ``outlet_temperature_celsius`` is lower than ``bed_temperature_celsius`` by ``loss_k``.
    """
    loss_k = 3.5
    lossy_config = replace(config, exhaust=replace(config.exhaust, outlet_duct_heat_loss_k=loss_k))
    t_gas_out, _y_gas_out, outlet_temperature, _solution = _reconstruct_at_solution(
        lossy_config, simulator, backend
    )
    assert outlet_temperature == pytest.approx(t_gas_out - loss_k, rel=1e-12)
