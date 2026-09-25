"""Tests der Stoffwert- und Fluidisierungskorrelationen gegen Literaturwerte.

Diese Tests sind kein Selbstzweck: sie sind die einzige Stelle, an der eine falsch
abgetippte Korrelation auffällt, bevor sie über 18 Monate in Trainingsdaten wandert.
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
    """200 µm Sand, 20 °C: Literaturwert u_mf ~ 0.03-0.04 m/s."""
    material = MaterialConfig(particle_density_kg_per_m3=2600.0)
    u_mf = float(
        minimum_fluidization_velocity_m_per_s(
            backend, GasConfig(), material, np.array(200e-6), np.array(20.0), np.array(0.0)
        )
    )
    assert 0.025 < u_mf < 0.045


def test_terminal_velocity_sand_reference(backend: ArrayBackend) -> None:
    """200 µm Sand: u_t in der Größenordnung 1-2 m/s."""
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
    """Begründet, warum die untere Grenze von u_gas zustandsabhängig sein muss."""
    gas, material = GasConfig(), MaterialConfig()
    small, large = np.array(400e-6), np.array(750e-6)
    args = (np.array(60.0), np.array(0.01))
    ratio = float(
        minimum_fluidization_velocity_m_per_s(backend, gas, material, large, *args)
        / minimum_fluidization_velocity_m_per_s(backend, gas, material, small, *args)
    )
    assert ratio > 2.5


def test_normalised_moisture_is_clipped(backend: ArrayBackend) -> None:
    """Signatur erweitert (Schritt 2, Isotherme injiziert) -- Ergebnis bleibt identisch.

    Mit ConstantIsotherm bleibt das Ergebnis identisch zum vorherigen Skalarverhalten,
    relative_humidity/temperature sind hier beliebig (ConstantIsotherm ignoriert sie).
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
    """Harte Grenze aus 02_prozess_setup.md -- hier strukturell, nicht per Projektion.

    Bewusst mit absurd nassen Partikeln und minimalem Luftstrom getestet: gerade dann
    müsste eine per Projektion nachgeschaltete Begrenzung greifen.
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


# ------------------------------------------------ Picard-Iteration: Doppel-Backend


_GAB_ISOTHERM_PLACEHOLDER = GabIsotherm(
    monolayer_moisture_kg_per_kg=0.020,
    c_constant_at_reference_temperature=25.0,
    k_constant_at_reference_temperature=0.75,
    c_enthalpy_j_per_mol=6000.0,
    k_enthalpy_j_per_mol=500.0,
    reference_temperature_celsius=62.0,
)

# Drei an frühere Betriebspunkte gekoppelte, feste Testpunkte (docs/04_stand_ap1.md,
# Befund 34): synthetisches Knick-Szenario, Befund-31-Randpunkt (unzulässig, X_p=0.397),
# echter Befund-31-Randpunkt (T_in=115°C/Spray=0.6kg/h, zulässig).
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
    """Die gedämpfte Picard-Iteration (alpha=0.7, n=5) muss backendunabhängig konvergieren.

    Für Y_out muss sie auf NumPy- und Torch-Backend identisch konvergieren -- beide
    rechnen float64 (TorchBackend castet explizit), keine Float32-Präzisionsdifferenz zu
    erwarten, aber unabhängig geprüft.
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
    """Konvergenz Iteration 4 gegen 5 -- separat von der Backend-Parität oben zu prüfen.

    Die Backend-Parität oben zeigt nur, dass beide Backends übereinstimmen, nicht dass
    die Iteration konvergiert. Nach der branch-selektiven Dämpfungskorrektur (Dämpfung
    nur im Trocknungszweig, siehe evaporation_rate_kg_per_s-Docstring) neu vermessen:
    alle drei Punkte liegen unverändert unter 1 %, weil keiner der drei den
    Kondensationszweig durchläuft (surplus > 0 bei allen dreien) -- die Korrektur
    konnte ihre Werte also gar nicht verändern.
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
