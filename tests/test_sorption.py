"""Tests der Sorptionsisotherme (``simulation.sorption``).

``ConstantIsotherm`` muss das bisherige Verhalten exakt reproduzieren (Regression).
``GabIsotherm`` wird gegen die in ``docs/04_stand_ap1.md`` (Befund 34) vorgelegte und
gegengelesene Plausibilitätstabelle sowie die dort abgestimmten Sicherheitseigenschaften
geprüft: Monotonie, kein Polstellen-Zugriff im Sicherheitsraster, Ankerwert exakt.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from twinlink_ml.simulation.backend import ArrayBackend, get_backend
from twinlink_ml.simulation.config import MaterialConfig, ProcessConfig
from twinlink_ml.simulation.sorption import (
    ConstantIsotherm,
    GabIsotherm,
    isotherm_from_material,
    sorption_safety_grid,
    validate_isotherm_against_critical_moisture,
)
from twinlink_ml.simulation.thermo import (
    absolute_humidity_kg_per_kg,
    relative_humidity_from_absolute_humidity,
    saturation_pressure_pa,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None

# Placeholder-Parametrisierung aus Schritt 1 (docs/04_stand_ap1.md, Befund 34), gegengelesen
# und bestätigt: verankert bei phi=0.5, T=62°C -> 0.030 kg/kg (Ankerwert aus
# greybox_lab_moisture.yaml).
_GAB_KWARGS = {
    "monolayer_moisture_kg_per_kg": 0.020,
    "c_constant_at_reference_temperature": 25.0,
    "k_constant_at_reference_temperature": 0.75,
    "c_enthalpy_j_per_mol": 6000.0,
    "k_enthalpy_j_per_mol": 500.0,
    "reference_temperature_celsius": 62.0,
}


def _gab_material(*, critical_moisture_kg_per_kg: float = 0.200) -> MaterialConfig:
    return MaterialConfig(
        critical_moisture_kg_per_kg=critical_moisture_kg_per_kg,
        equilibrium_moisture_kg_per_kg=0.030,
        sorption_isotherm_kind="gab",
        gab_monolayer_moisture_kg_per_kg=_GAB_KWARGS["monolayer_moisture_kg_per_kg"],
        gab_c_constant_at_reference_temperature=_GAB_KWARGS["c_constant_at_reference_temperature"],
        gab_k_constant_at_reference_temperature=_GAB_KWARGS["k_constant_at_reference_temperature"],
        gab_c_enthalpy_j_per_mol=_GAB_KWARGS["c_enthalpy_j_per_mol"],
        gab_k_enthalpy_j_per_mol=_GAB_KWARGS["k_enthalpy_j_per_mol"],
        gab_reference_temperature_celsius=_GAB_KWARGS["reference_temperature_celsius"],
    )


# --------------------------------------------------------------------- ConstantIsotherm


def test_constant_isotherm_reproduces_scalar_exactly(backend: ArrayBackend) -> None:
    """Regression: identisch zum bisherigen Skalarverhalten, für beliebige phi/T."""
    isotherm = ConstantIsotherm(equilibrium_moisture_kg_per_kg=0.003)
    phi = np.array([0.0, 0.2, 0.5, 0.8, 0.99])
    temperature = np.array([20.0, 40.0, 62.0, 90.0, 130.0])
    result = isotherm.equilibrium_moisture(backend, phi, temperature)
    assert np.allclose(backend.to_numpy(result), 0.003)


def test_constant_isotherm_ignores_relative_humidity_and_temperature(backend: ArrayBackend) -> None:
    isotherm = ConstantIsotherm(equilibrium_moisture_kg_per_kg=0.030)
    low = isotherm.equilibrium_moisture(backend, np.array(0.01), np.array(20.0))
    high = isotherm.equilibrium_moisture(backend, np.array(0.95), np.array(120.0))
    assert float(backend.to_numpy(low)) == float(backend.to_numpy(high)) == 0.030


# -------------------------------------------------------------------------- GabIsotherm


def test_gab_isotherm_rejects_k_greater_equal_one() -> None:
    kwargs = dict(_GAB_KWARGS)
    kwargs["k_constant_at_reference_temperature"] = 1.0
    with pytest.raises(ValueError, match="k_constant_at_reference_temperature"):
        GabIsotherm(**kwargs)  # type: ignore[arg-type]

    kwargs["k_constant_at_reference_temperature"] = 1.2
    with pytest.raises(ValueError, match="k_constant_at_reference_temperature"):
        GabIsotherm(**kwargs)  # type: ignore[arg-type]


def test_gab_isotherm_rejects_non_positive_monolayer_or_c() -> None:
    kwargs = dict(_GAB_KWARGS)
    kwargs["monolayer_moisture_kg_per_kg"] = 0.0
    with pytest.raises(ValueError, match="monolayer_moisture_kg_per_kg"):
        GabIsotherm(**kwargs)  # type: ignore[arg-type]

    kwargs = dict(_GAB_KWARGS)
    kwargs["c_constant_at_reference_temperature"] = -1.0
    with pytest.raises(ValueError, match="c_constant_at_reference_temperature"):
        GabIsotherm(**kwargs)  # type: ignore[arg-type]


def test_gab_isotherm_hits_anchor_value_exactly(backend: ArrayBackend) -> None:
    isotherm = GabIsotherm(**_GAB_KWARGS)  # type: ignore[arg-type]
    value = isotherm.equilibrium_moisture(backend, np.array(0.5), np.array(62.0))
    assert float(backend.to_numpy(value)) == pytest.approx(0.030, rel=1e-9)


def test_gab_isotherm_status_is_always_placeholder() -> None:
    isotherm = GabIsotherm(**_GAB_KWARGS)  # type: ignore[arg-type]
    assert isotherm.status == "placeholder"


def test_gab_isotherm_monotonic_increasing_in_relative_humidity(backend: ArrayBackend) -> None:
    isotherm = GabIsotherm(**_GAB_KWARGS)  # type: ignore[arg-type]
    phi = np.linspace(0.001, 0.99, 200)
    for temperature in (40.0, 62.0, 90.0):
        values = backend.to_numpy(
            isotherm.equilibrium_moisture(backend, phi, np.full_like(phi, temperature))
        )
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)
        assert np.all(np.diff(values) > 0.0), f"nicht monoton steigend bei T={temperature}"


def test_gab_isotherm_monotonic_falling_in_temperature(backend: ArrayBackend) -> None:
    """Desorption: X_eq fällt mit steigender Temperatur bei fester relativer Feuchte."""
    isotherm = GabIsotherm(**_GAB_KWARGS)  # type: ignore[arg-type]
    temperature = np.linspace(-10.0, 130.0, 200)
    for phi in (0.2, 0.5, 0.8):
        values = backend.to_numpy(
            isotherm.equilibrium_moisture(backend, np.full_like(temperature, phi), temperature)
        )
        assert np.all(np.isfinite(values))
        assert np.all(np.diff(values) < 0.0), f"nicht monoton fallend bei phi={phi}"


def test_gab_isotherm_low_phi_limit_well_below_critical_moisture(backend: ArrayBackend) -> None:
    """phi->0 liegt deutlich unter X_cr.

    Sonst wäre die Trocknungskurve von Anfang an im zweiten Abschnitt (eta<1 bereits am
    unteren Rand des Feuchtebereichs).
    """
    isotherm = GabIsotherm(**_GAB_KWARGS)  # type: ignore[arg-type]
    material = _gab_material()
    value = isotherm.equilibrium_moisture(backend, np.array(0.001), np.array(62.0))
    assert float(backend.to_numpy(value)) < 0.1 * material.critical_moisture_kg_per_kg


def test_gab_isotherm_plausibility_table(backend: ArrayBackend) -> None:
    """Kontrolltabelle aus Schritt 1 (gegengelesen), auf zwei Nachkommastellen relativ."""
    isotherm = GabIsotherm(**_GAB_KWARGS)  # type: ignore[arg-type]
    expected = {
        (0.2, 40.0): 0.01978,
        (0.2, 62.0): 0.01918,
        (0.2, 90.0): 0.01845,
        (0.5, 40.0): 0.03053,
        (0.5, 62.0): 0.03000,
        (0.5, 90.0): 0.02937,
        (0.8, 40.0): 0.04986,
        (0.8, 62.0): 0.04870,
        (0.8, 90.0): 0.04744,
    }
    for (phi, temperature), value in expected.items():
        result = isotherm.equilibrium_moisture(backend, np.array(phi), np.array(temperature))
        assert float(backend.to_numpy(result)) == pytest.approx(value, rel=1e-3)


def test_gab_isotherm_finite_positive_over_safety_grid(
    config: ProcessConfig, backend: ArrayBackend
) -> None:
    """Sicherheitsraster (nicht das Plausibilitätsraster oben): keine Polstelle.

    phi bis 0.99, T-Rand an der Aktions-/Störgrößenhüllkurve -- keine Polstelle im
    numerisch erreichbaren Bereich.
    """
    isotherm = GabIsotherm(**_GAB_KWARGS)  # type: ignore[arg-type]
    phi_min, phi_max, t_min, t_max = sorption_safety_grid(
        backend, config.gas, config.actions, config.disturbances
    )
    phi_grid, t_grid = np.meshgrid(
        np.linspace(phi_min, phi_max, 300), np.linspace(t_min, t_max, 150)
    )
    values = backend.to_numpy(
        isotherm.equilibrium_moisture(backend, backend.asarray(phi_grid), backend.asarray(t_grid))
    )
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_sorption_safety_grid_phi_min_is_envelope_transformed_not_zero(
    config: ProcessConfig, backend: ArrayBackend
) -> None:
    """phi_min folgt aus der Störgrößenhülle, nicht aus einer mathematischen Grenze.

    T_amb.min, ambient_relative_humidity.min, erwärmt auf t_max -- nicht aus einer
    mathematischen phi->0-Grenze.
    """
    phi_min, phi_max, t_min, t_max = sorption_safety_grid(
        backend, config.gas, config.actions, config.disturbances
    )
    assert 0.0 < phi_min < 0.01
    assert phi_max == pytest.approx(0.99)
    assert t_min == pytest.approx(config.disturbances.ambient_temperature_celsius.minimum - 10.0)
    assert t_max == pytest.approx(config.actions.inlet_temperature_celsius.maximum + 10.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_gab_isotherm_matches_across_numpy_and_torch_backends() -> None:
    numpy_backend = get_backend("numpy")
    torch_backend = get_backend("torch")
    isotherm = GabIsotherm(**_GAB_KWARGS)  # type: ignore[arg-type]

    phi_np = np.array([0.05, 0.3, 0.6, 0.9])
    temperature_np = np.array([35.0, 62.0, 97.85, 120.0])

    result_numpy = numpy_backend.to_numpy(
        isotherm.equilibrium_moisture(
            numpy_backend, numpy_backend.asarray(phi_np), numpy_backend.asarray(temperature_np)
        )
    )
    result_torch = torch_backend.to_numpy(
        isotherm.equilibrium_moisture(
            torch_backend, torch_backend.asarray(phi_np), torch_backend.asarray(temperature_np)
        )
    )
    assert np.allclose(result_numpy, result_torch, atol=1e-12)


# --------------------------------------------------------------- isotherm_from_material


def test_isotherm_from_material_defaults_to_constant(config: ProcessConfig) -> None:
    isotherm = isotherm_from_material(config.material)
    assert isinstance(isotherm, ConstantIsotherm)
    assert isotherm.equilibrium_moisture_kg_per_kg == config.material.equilibrium_moisture_kg_per_kg


def test_isotherm_from_material_builds_gab() -> None:
    material = _gab_material()
    isotherm = isotherm_from_material(material)
    assert isinstance(isotherm, GabIsotherm)
    assert isotherm.monolayer_moisture_kg_per_kg == _GAB_KWARGS["monolayer_moisture_kg_per_kg"]
    assert (
        isotherm.k_constant_at_reference_temperature
        == _GAB_KWARGS["k_constant_at_reference_temperature"]
    )


# ------------------------------------------------------- MaterialConfig-Feldvalidierung


def test_material_config_gab_requires_all_fields() -> None:
    with pytest.raises(ValueError, match="gab_monolayer_moisture_kg_per_kg"):
        MaterialConfig(sorption_isotherm_kind="gab")


def test_material_config_gab_validates_k_bounds() -> None:
    kwargs = dict(_GAB_KWARGS)
    kwargs["k_constant_at_reference_temperature"] = 1.5
    with pytest.raises(ValueError, match="gab_k_constant_at_reference_temperature"):
        MaterialConfig(
            critical_moisture_kg_per_kg=0.200,
            equilibrium_moisture_kg_per_kg=0.030,
            sorption_isotherm_kind="gab",
            gab_monolayer_moisture_kg_per_kg=kwargs["monolayer_moisture_kg_per_kg"],
            gab_c_constant_at_reference_temperature=kwargs["c_constant_at_reference_temperature"],
            gab_k_constant_at_reference_temperature=kwargs["k_constant_at_reference_temperature"],
            gab_c_enthalpy_j_per_mol=kwargs["c_enthalpy_j_per_mol"],
            gab_k_enthalpy_j_per_mol=kwargs["k_enthalpy_j_per_mol"],
            gab_reference_temperature_celsius=kwargs["reference_temperature_celsius"],
        )


def test_material_config_constant_kind_ignores_missing_gab_fields() -> None:
    """Default-Fall: kein gab_* Feld gesetzt, sorption_isotherm_kind bleibt 'constant'.

    Lädt unverändert, keine Validierung greift.
    """
    material = MaterialConfig()
    assert material.sorption_isotherm_kind == "constant"
    assert material.gab_monolayer_moisture_kg_per_kg is None


# --------------------------------------------------- validate_isotherm_against_critical_moisture


def test_validate_isotherm_skips_constant_isotherm(config: ProcessConfig) -> None:
    """Default-Config (ConstantIsotherm) -- die Prüfung greift gar nicht erst."""
    validate_isotherm_against_critical_moisture(config)  # darf nicht raisen


def test_validate_isotherm_passes_for_lab_level_critical_moisture(config: ProcessConfig) -> None:
    import dataclasses

    gab_config = dataclasses.replace(
        config, material=_gab_material(critical_moisture_kg_per_kg=0.200)
    )
    validate_isotherm_against_critical_moisture(gab_config)  # darf nicht raisen


def test_validate_isotherm_raises_when_critical_moisture_below_grid_maximum(
    config: ProcessConfig,
) -> None:
    """X_cr=0.05 liegt über dem Ankerwert, aber unter dem Isothermen-Maximum.

    Besteht die lokale MaterialConfig-Prüfung (Ankerwert 0.030), aber liegt unter dem
    tatsächlichen Isothermen-Maximum im Sicherheitsraster (~0.090, siehe
    docs/04_stand_ap1.md, Befund 34) -- genau der Fall, den die generische,
    ProcessConfig-weite Prüfung zusätzlich zur lokalen abdeckt.
    """
    import dataclasses

    gab_config = dataclasses.replace(
        config, material=_gab_material(critical_moisture_kg_per_kg=0.05)
    )
    with pytest.raises(ValueError, match="critical_moisture_kg_per_kg"):
        validate_isotherm_against_critical_moisture(gab_config)


# -------------------------------------- relative_humidity_from_absolute_humidity


def test_relative_humidity_from_absolute_humidity_is_exact_inverse_within_safety_grid(
    config: ProcessConfig, backend: ArrayBackend
) -> None:
    """Exakte algebraische Inversion von absolute_humidity_kg_per_kg.

    Bestätigt bis auf Float64-Rundung, aber nur außerhalb von deren eigenem
    p_partial-Clip (T ab ca. 90-100°C mit hohem phi, siehe Docstring). Im
    Sicherheitsraster-Bereich greift der Clip bei einem Teil der Stichproben; dort ist
    eine Abweichung erwartet und wird getrennt geprüft.
    """
    phi_min, phi_max, t_min, t_max = sorption_safety_grid(
        backend, config.gas, config.actions, config.disturbances
    )
    rng = np.random.default_rng(0)
    phi = rng.uniform(phi_min, phi_max, 5000)
    temperature = rng.uniform(t_min, t_max, 5000)

    p_saturation = backend.to_numpy(saturation_pressure_pa(backend, temperature))
    clip_active = phi * p_saturation > 0.95 * config.gas.pressure_pa

    y = absolute_humidity_kg_per_kg(backend, config.gas, temperature, phi)
    phi_roundtrip = backend.to_numpy(
        relative_humidity_from_absolute_humidity(backend, config.gas, y, temperature)
    )
    error = np.abs(phi_roundtrip - phi)

    assert error[~clip_active].max() < 1e-9, "keine exakte Inversion außerhalb des Clips"
    assert clip_active.sum() > 0, "Testraster deckt den Clip-Fall nicht ab -- Test unscharf"
