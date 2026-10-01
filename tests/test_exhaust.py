"""Tests für ``simulation/exhaust.py`` -- Rückrechnung der Abluft-Austrittsgrößen.

Geprüft wird die Verdrahtung gegen die bestehende Referenzrechnung aus
``steady_state.py``/Befund 31 (``docs/04_stand_ap1.md``) -- mit den WAHREN, simulierten
Werten als Eingabe, nicht mit Modellvorhersagen. Ob ein trainiertes Sequenzmodell
T_bed/``product_moisture_kg_per_kg`` gut genug vorhersagt, ist eine andere, spätere Frage
(Rollout-Güte), die dieser Test nicht beantwortet.
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
    """Rückrechnung mit den wahren Werten der Beharrungszustandslösung, plus die Lösung selbst."""
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
    """Am Nominalpunkt (Befund 31) gegen ``solve_steady_state()`` -- bitgleich erwartet.

    Beide Wege rufen ``thermo.evaporation_rate_kg_per_s`` mit denselben Eingaben auf;
    eine Abweichung wäre ein Verdrahtungsfehler in ``exhaust.py``, keine physikalische
    Frage -- deshalb ``rel=1e-12``, nicht eine großzügige Tolerenz.
    """
    t_gas_out, y_gas_out, _outlet_temperature, solution = _reconstruct_at_solution(
        config, simulator, backend
    )

    assert t_gas_out == solution.bed_temperature_celsius
    assert y_gas_out == pytest.approx(solution.outlet_humidity_kg_per_kg, rel=1e-12)


def test_reconstruction_matches_documented_befund_31_values(
    config: ProcessConfig, simulator: GreyBoxFluidBedGranulator, backend: ArrayBackend
) -> None:
    """Gegenprobe gegen den dokumentierten Tabellenwert (``docs/04_stand_ap1.md``, Befund 31)."""
    _t_gas_out, y_gas_out, _outlet_temperature, _solution = _reconstruct_at_solution(
        config, simulator, backend
    )
    assert y_gas_out == pytest.approx(0.0171, abs=5e-5)


def test_outlet_temperature_matches_bed_temperature_at_zero_loss(
    config: ProcessConfig, simulator: GreyBoxFluidBedGranulator, backend: ArrayBackend
) -> None:
    """``outlet_duct_heat_loss_k`` Default 0.0: bitgleiche Identität.

    ``outlet_temperature_celsius`` == ``bed_temperature_celsius`` -- keine numerische
    Änderung gegenüber dem bisherigen (fehlerhaft benannten) Verhalten.
    """
    assert config.exhaust.outlet_duct_heat_loss_k == 0.0
    t_gas_out, _y_gas_out, outlet_temperature, _solution = _reconstruct_at_solution(
        config, simulator, backend
    )
    assert outlet_temperature == t_gas_out


def test_outlet_temperature_subtracts_duct_loss(
    config: ProcessConfig, simulator: GreyBoxFluidBedGranulator, backend: ArrayBackend
) -> None:
    """Bei gesetztem Rohrleitungsverlust: um genau diesen Betrag kühler.

    ``outlet_temperature_celsius`` ist um ``loss_k`` niedriger als ``bed_temperature_celsius``.
    """
    loss_k = 3.5
    lossy_config = replace(config, exhaust=replace(config.exhaust, outlet_duct_heat_loss_k=loss_k))
    t_gas_out, _y_gas_out, outlet_temperature, _solution = _reconstruct_at_solution(
        lossy_config, simulator, backend
    )
    assert outlet_temperature == pytest.approx(t_gas_out - loss_k, rel=1e-12)
