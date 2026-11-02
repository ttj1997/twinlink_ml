"""Tests of the property and fluidisation correlations against literature values.

These tests are not an end in themselves: they are the only place where a wrongly
typed correlation is noticed before it wanders into training data over 18 months.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from twinlink_ml.simulation.backend import ArrayBackend, get_backend
from twinlink_ml.simulation.config import GasConfig, MaterialConfig
from twinlink_ml.simulation.fluidization import (
    minimum_fluidization_velocity_m_per_s,
    terminal_velocity_m_per_s,
)
from twinlink_ml.simulation.sorption import ConstantIsotherm, GabIsotherm
from twinlink_ml.simulation.thermo import (
    absolute_humidity_kg_per_kg,
    evaporation_rate_kg_per_s,
    gas_density_kg_per_m3,
    gas_viscosity_pa_s,
    normalised_moisture,
    saturation_humidity_kg_per_kg,
    saturation_pressure_pa,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.parametrize(
    ("temperature_celsius", "expected_pa", "tolerance_pa"),
    [(0.0, 611.0, 10.0), (20.0, 2339.0, 30.0), (60.0, 19_932.0, 300.0), (100.0, 101_325.0, 1500.0)],
)
def test_saturation_pressure_matches_steam_table(
    backend: ArrayBackend, temperature_celsius: float, expected_pa: float, tolerance_pa: float
) -> None:
    value = saturation_pressure_pa(backend, np.array(temperature_celsius))
    assert abs(float(value) - expected_pa) < tolerance_pa


def test_saturation_humidity_increases_strongly_with_temperature(backend: ArrayBackend) -> None:
    gas = GasConfig()
    y_40 = float(saturation_humidity_kg_per_kg(backend, gas, np.array(40.0)))
    y_80 = float(saturation_humidity_kg_per_kg(backend, gas, np.array(80.0)))
    assert 0.045 < y_40 < 0.055
    assert y_80 > 8.0 * y_40, "Sättigungsfeuchte muss überproportional wachsen"


def test_gas_properties_at_ambient(backend: ArrayBackend) -> None:
    gas = GasConfig()
    rho = float(gas_density_kg_per_m3(backend, gas, np.array(20.0), np.array(0.0)))
    mu = float(gas_viscosity_pa_s(backend, gas, np.array(20.0)))
    assert 1.15 < rho < 1.25
    assert 1.7e-5 < mu < 1.9e-5


def test_minimum_fluidization_velocity_sand_reference(backend: ArrayBackend) -> None:
    """200 µm sand, 20 °C: literature value u_mf ~ 0.03-0.04 m/s."""
    material = MaterialConfig(particle_density_kg_per_m3=2600.0)
    u_mf = float(
        minimum_fluidization_velocity_m_per_s(
            backend, GasConfig(), material, np.array(200e-6), np.array(20.0), np.array(0.0)
        )
    )
    assert 0.025 < u_mf < 0.045


def test_terminal_velocity_sand_reference(backend: ArrayBackend) -> None:
    """200 µm sand: u_t of the order of 1-2 m/s."""
    material = MaterialConfig(particle_density_kg_per_m3=2600.0, sphericity=1.0)
    u_t = float(
        terminal_velocity_m_per_s(
            backend, GasConfig(), material, np.array(200e-6), np.array(20.0), np.array(0.0)
        )
    )
    assert 0.9 < u_t < 2.2


def test_fluidization_window_is_ordered_and_widens_with_size(backend: ArrayBackend) -> None:
    gas, material = GasConfig(), MaterialConfig()
    diameters = np.array([150e-6, 400e-6, 750e-6, 1500e-6])
    temperature = np.full_like(diameters, 60.0)
    humidity = np.full_like(diameters, 0.01)
    u_mf = minimum_fluidization_velocity_m_per_s(
        backend, gas, material, diameters, temperature, humidity
    )
    u_t = terminal_velocity_m_per_s(backend, gas, material, diameters, temperature, humidity)
    assert np.all(u_mf < u_t)
    assert np.all(np.diff(u_mf) > 0.0), "u_mf muss mit der Partikelgröße steigen"
    assert np.all(np.diff(u_t) > 0.0), "u_t muss mit der Partikelgröße steigen"


def test_minimum_fluidization_velocity_grows_by_factor_three(backend: ArrayBackend) -> None:
    """Justifies why the lower limit of u_gas must be state-dependent."""
    gas, material = GasConfig(), MaterialConfig()
    small, large = np.array(400e-6), np.array(750e-6)
    args = (np.array(60.0), np.array(0.01))
    ratio = float(
        minimum_fluidization_velocity_m_per_s(backend, gas, material, large, *args)
        / minimum_fluidization_velocity_m_per_s(backend, gas, material, small, *args)
    )
    assert ratio > 2.5


def test_normalised_moisture_is_clipped(backend: ArrayBackend) -> None:
    """Signature extended (step 2, isotherm injected) -- the result stays identical.

    With ConstantIsotherm the result stays identical to the previous scalar behaviour,
    relative_humidity/temperature are arbitrary here (ConstantIsotherm ignores them).
    """
    material = MaterialConfig()
    isotherm = ConstantIsotherm(material.equilibrium_moisture_kg_per_kg)
    values = np.array([0.0, material.equilibrium_moisture_kg_per_kg, 0.01, 0.05])
    relative_humidity = np.full_like(values, 0.4)
    temperature_celsius = np.full_like(values, 62.0)
    eta = normalised_moisture(
        backend, material, isotherm, values, relative_humidity, temperature_celsius
    )
    assert np.all((eta >= 0.0) & (eta <= 1.0))
    assert float(eta[0]) == 0.0
    assert float(eta[1]) == 0.0
    assert float(eta[3]) == 1.0


def test_outlet_humidity_never_exceeds_saturation(backend: ArrayBackend) -> None:
    """Hard limit from 02_prozess_setup.md -- here structural, not via projection.

    Deliberately tested with absurdly wet particles and minimal air flow: precisely then
    a limit applied downstream via projection would have to engage.
    """
    gas, material = GasConfig(), MaterialConfig()
    rng = np.random.default_rng(0)
    n = 2000
    air_flow = rng.uniform(1e-4, 0.1, n)
    bed_temperature = rng.uniform(20.0, 110.0, n)
    ambient_humidity = absolute_humidity_kg_per_kg(
        backend, gas, rng.uniform(0.0, 40.0, n), rng.uniform(0.0, 1.0, n)
    )
    moisture = rng.uniform(0.0, 0.5, n)
    isotherm = ConstantIsotherm(material.equilibrium_moisture_kg_per_kg)

    _, outlet, saturation_degree = evaporation_rate_kg_per_s(
        backend, gas, material, isotherm, air_flow, bed_temperature, ambient_humidity, moisture
    )
    y_saturation = saturation_humidity_kg_per_kg(backend, gas, bed_temperature)
    assert np.all(outlet <= y_saturation + 1e-12)
    assert np.all(saturation_degree <= material.max_saturation_degree + 1e-12)


# -------------------------------------------------- Picard iteration: dual backend


_GAB_ISOTHERM_PLACEHOLDER = GabIsotherm(
    monolayer_moisture_kg_per_kg=0.020,
    c_constant_at_reference_temperature=25.0,
    k_constant_at_reference_temperature=0.75,
    c_enthalpy_j_per_mol=6000.0,
    k_enthalpy_j_per_mol=500.0,
    reference_temperature_celsius=62.0,
)

# Three fixed test points coupled to earlier operating points (docs/04_stand_ap1.md,
# Befund 34): synthetic kink scenario, Befund 31 edge point (infeasible, X_p=0.397),
# real Befund 31 edge point (T_in=115°C/spray=0.6kg/h, feasible).
_PICARD_TEST_POINTS = {
    "knick": {"y_in": 0.007261, "t_bed": 90.0, "dry_air_flow": 0.042514, "x_p": 0.004445},
    "befund31_unzulaessig": {
        "y_in": 0.007261,
        "t_bed": 31.4396,
        "dry_air_flow": 0.032440,
        "x_p": 0.397368,
    },
    "befund31_real": {
        "y_in": 0.007261,
        "t_bed": 97.8529,
        "dry_air_flow": 0.026633,
        "x_p": 0.003358,
    },
}


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
@pytest.mark.parametrize("label", list(_PICARD_TEST_POINTS))
def test_evaporation_rate_picard_matches_across_numpy_and_torch_backends(label: str) -> None:
    """The damped Picard iteration (alpha=0.7, n=5) must converge independently of the backend.

    For Y_out it must converge identically on the NumPy and Torch backends -- both
    compute float64 (TorchBackend casts explicitly), no float32 precision difference to
    be expected, but checked independently.
    """
    gas, material = GasConfig(), MaterialConfig(critical_moisture_kg_per_kg=0.200)
    point = _PICARD_TEST_POINTS[label]

    results = {}
    for backend_name in ("numpy", "torch"):
        backend = get_backend(backend_name)
        _, outlet_humidity, _ = evaporation_rate_kg_per_s(
            backend,
            gas,
            material,
            _GAB_ISOTHERM_PLACEHOLDER,
            backend.asarray(point["dry_air_flow"]),
            backend.asarray(point["t_bed"]),
            backend.asarray(point["y_in"]),
            backend.asarray(point["x_p"]),
        )
        results[backend_name] = float(backend.to_numpy(outlet_humidity))

    assert results["numpy"] == pytest.approx(results["torch"], abs=1e-12)


@pytest.mark.parametrize("label", list(_PICARD_TEST_POINTS))
def test_evaporation_rate_picard_converges_under_one_percent(
    label: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Convergence of iteration 4 against 5 -- to check separately from the backend parity above.

    The backend parity above only shows that both backends agree, not that
    the iteration converges. Re-measured after the branch-selective damping correction (damping
    only in the drying branch, see the evaporation_rate_kg_per_s docstring):
    all three points stay unchanged below 1 %, because none of the three passes through the
    condensation branch (surplus > 0 for all three) -- the correction
    could therefore not change their values at all.
    """
    import twinlink_ml.simulation.thermo as thermo_module

    gas, material = GasConfig(), MaterialConfig(critical_moisture_kg_per_kg=0.200)
    point = _PICARD_TEST_POINTS[label]
    backend = get_backend("numpy")

    results = {}
    for n_iterations in (4, 5):
        monkeypatch.setattr(thermo_module, "_PICARD_ITERATIONS", n_iterations)
        _, outlet_humidity, _ = evaporation_rate_kg_per_s(
            backend,
            gas,
            material,
            _GAB_ISOTHERM_PLACEHOLDER,
            backend.asarray(point["dry_air_flow"]),
            backend.asarray(point["t_bed"]),
            backend.asarray(point["y_in"]),
            backend.asarray(point["x_p"]),
        )
        results[n_iterations] = float(backend.to_numpy(outlet_humidity))

    relative_difference = abs(results[5] - results[4]) / max(abs(results[5]), 1e-12)
    assert relative_difference < 0.01
