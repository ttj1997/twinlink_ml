"""R1: regression test against the air-flow error in ``study_plant_validation.py``.

Relative instead of absolute humidity in the Nm3/h conversion, factor ~2.2, Befund 56 correction.
``scripts/`` is not a package -- loading via ``importlib.util.spec_from_file_location``,
as in ``test_arch_learning_curve_resume.py``.
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
    """The dry-air mass flow must not deviate from the normal volume flow by a factor of 2.

    A normal cubic metre is by definition a mass quantity: the dry-air
    mass flow that the model derives from a given normal volume flow must correspond to
    normal volume flow times normal density -- up to the moisture correction of
    a few percent (moist/dry conversion, temperature-dependent gas density). Neither the
    assumed ambient humidity (0.75 relative) nor the bed temperature may shift this
    mass flow by a factor like 2 -- exactly what the error did that
    this test prevents (relative instead of absolute humidity in the Nm3/h conversion).

    Checks the real production path: ``_gas_velocity_m_per_s`` yields the
    model velocity, ``GreyBoxFluidBedGranulator.evaluate_derivative`` reads
    ``dry_air_mass_flow_kg_per_s`` back from it -- as in the real run. Neither of the two
    formulas is typed out again here.
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
