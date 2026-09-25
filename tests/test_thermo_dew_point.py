"""Tests für ``simulation.thermo.dew_point_celsius`` -- vektorisierte Taupunktbestimmung.

Gebraucht für X3 (``docs/04_stand_ap1.md``): Q_rec-Berechnung auf ganzen
Vorhersage-Arrays. ``simulation.steady_state._solve_dew_point`` löst dieselbe Bisektion
bereits skalar für den Beharrungszustands-Löser -- die Tests hier prüfen, dass die
array-fähige, öffentliche Variante dasselbe Ergebnis liefert, nicht eine zweite Physik.
"""

from __future__ import annotations

import numpy as np
import pytest

from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.config import GasConfig
from twinlink_ml.simulation.steady_state import _solve_dew_point
from twinlink_ml.simulation.thermo import dew_point_celsius, saturation_pressure_pa


def test_dew_point_matches_scalar_steady_state_solver() -> None:
    """Bitnahe Übereinstimmung mit ``_solve_dew_point`` über mehrere Feuchtewerte."""
    backend = NumpyBackend()
    gas = GasConfig()
    humidity = np.array([0.005, 0.0171, 0.03, 0.05])

    vectorised = backend.to_numpy(
        dew_point_celsius(backend, gas, backend.asarray(humidity))
    )
    for y, expected in zip(humidity, vectorised, strict=True):
        scalar = _solve_dew_point(
            backend=backend,
            outlet_humidity_kg_per_kg=float(y),
            pressure_pa=gas.pressure_pa,
            max_iterations=100,
            tolerance=1e-13,
        )
        assert expected == pytest.approx(scalar, abs=1e-8)


def test_dew_point_inverts_saturation_pressure() -> None:
    """``p_s(T_dp)`` muss den aus ``Y`` berechneten Partialdruck reproduzieren."""
    backend = NumpyBackend()
    gas = GasConfig()
    humidity = backend.asarray(np.array([0.01, 0.02, 0.04]))

    dew_point = dew_point_celsius(backend, gas, humidity)
    p_partial = backend.to_numpy(humidity) * gas.pressure_pa / (0.622 + backend.to_numpy(humidity))
    p_at_dew_point = backend.to_numpy(saturation_pressure_pa(backend, dew_point))

    assert p_at_dew_point == pytest.approx(p_partial, rel=1e-8)


def test_dew_point_increases_with_humidity() -> None:
    """Monoton: mehr absolute Feuchte -> höherer Taupunkt (physikalisch notwendig)."""
    backend = NumpyBackend()
    gas = GasConfig()
    humidity = backend.asarray(np.array([0.005, 0.015, 0.03, 0.05]))

    dew_point = backend.to_numpy(dew_point_celsius(backend, gas, humidity))
    assert np.all(np.diff(dew_point) > 0.0)
