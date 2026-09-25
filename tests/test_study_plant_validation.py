"""R1: Regressionstest gegen den Luftstrom-Fehler in ``study_plant_validation.py``.

Relative statt absolute Feuchte in der Nm3/h-Umrechnung, Faktor ~2.2, Befund 56 Korrektur.
``scripts/`` ist kein Package -- Laden per ``importlib.util.spec_from_file_location``,
wie in ``test_arch_learning_curve_resume.py``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from twinlink_ml.simulation import GreyBoxFluidBedGranulator, NumpyBackend

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str) -> types.ModuleType:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_under_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gas_velocity_from_normal_flow_matches_mass_conservation() -> None:
    """Der Trockenluftmassenstrom darf nicht um Faktor 2 vom Normvolumenstrom abweichen.

    Ein Normkubikmeter ist per Definition eine Massenangabe: der Trockenluft-
    massenstrom, den das Modell aus einem gegebenen Normvolumenstrom ableitet, muss
    Normvolumenstrom mal Normdichte entsprechen -- bis auf die Feuchtekorrektur von
    wenigen Prozent (moist/dry-Umrechnung, temperaturabhaengige Gasdichte). Weder die
    angenommene Umgebungsfeuchte (0.75 relativ) noch die Betttemperatur duerfen diesen
    Massenstrom um einen Faktor wie 2 verschieben -- genau das tat der Fehler, den
    dieser Test verhindert (relative statt absolute Feuchte in der Nm3/h-Umrechnung).

    Prueft den echten Produktionspfad: ``_gas_velocity_m_per_s`` liefert die
    Modellgeschwindigkeit, ``GreyBoxFluidBedGranulator.evaluate_derivative`` liest daraus
    -- wie im echten Lauf -- ``dry_air_mass_flow_kg_per_s`` zurueck. Keine der beiden
    Formeln wird hier erneut abgetippt.
    """
    module = _load_script("study_plant_validation")
    backend = NumpyBackend()

    point = module.OperatingPoint(
        label="SYNTHETIC",
        ambient_temperature_c=10.0,
        inlet_temperature_c=85.0,
        bed_temperature_measured_c=60.0,
        upper_position_measured_c=61.0,
        outlet_temperature_measured_c=54.0,
        process_air_flow_nm3_per_h=30000.0,
        spray_rate_kg_per_h=1000.0,
    )
    area_m2 = 4.0
    bed_mass_kg = 800.0
    config = module.build_plant_config(
        point, cross_section_area_m2=area_m2, bed_mass_solid_setpoint_kg=bed_mass_kg
    )

    normal_density_kg_per_m3 = config.gas.pressure_pa / (
        config.gas.r_specific_dry_air_j_per_kg_k * 273.15
    )
    expected_dry_air_flow_kg_per_s = (
        point.process_air_flow_nm3_per_h / 3600.0 * normal_density_kg_per_m3
    )

    simulator = GreyBoxFluidBedGranulator(config)
    state = backend.asarray(
        np.array(
            [
                [
                    bed_mass_kg,
                    bed_mass_kg * config.operation.initial_moisture_kg_per_kg,
                    1.0e12,
                    point.bed_temperature_measured_c,
                ]
            ]
        )
    )
    action = backend.asarray(
        np.array(
            [
                [
                    config.actions.inlet_temperature_celsius.nominal,
                    config.actions.gas_velocity_m_per_s.nominal,
                    config.actions.spray_rate_kg_per_h.nominal,
                    config.actions.nozzle_pressure_bar.nominal,
                ]
            ]
        )
    )
    disturbance = backend.asarray(
        np.array(
            [
                [
                    config.disturbances.ambient_temperature_celsius.nominal,
                    config.disturbances.ambient_relative_humidity.nominal,
                    config.disturbances.suspension_solid_fraction.nominal,
                ]
            ]
        )
    )
    _, diagnostics = simulator.evaluate_derivative(state, action, disturbance)
    actual_dry_air_flow_kg_per_s = float(
        backend.to_numpy(diagnostics["dry_air_mass_flow_kg_per_s"])[0]
    )

    relative_error = abs(actual_dry_air_flow_kg_per_s - expected_dry_air_flow_kg_per_s) / (
        expected_dry_air_flow_kg_per_s
    )
    assert relative_error < 0.05, (
        f"Trockenluftmassenstrom {actual_dry_air_flow_kg_per_s:.4f} kg/s weicht um "
        f"{relative_error:.1%} von der Normvolumenstrom-Erwartung "
        f"{expected_dry_air_flow_kg_per_s:.4f} kg/s ab -- mehr als die paar Prozent, die "
        "Feuchte-/Dichtekorrekturen erklaeren (frueherer Fehler: Faktor ~2.2, siehe "
        "Befund 56 Korrektur)."
    )
