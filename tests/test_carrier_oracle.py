"""Carrier-separated oracle (M1): objective definitions, penalty scaling, legacy anchor."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import carrier_oracle as oracle
from twinlink_ml.evaluation.net_energy import GAS_CO2_KG_PER_KWH
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.operating_point import (
    PENALTY_WEIGHT_KW,
    SOLVE_FAILURE_PENALTY_KW,
    check_constraints,
)
from twinlink_ml.simulation.oracle_search import evaluate_point

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml")


def test_penalty_and_failure_value_are_the_befund_52_constants_in_the_objective_unit() -> None:
    legacy = oracle.OracleSpec(0.15, "legacy_power_kw")
    co2 = oracle.OracleSpec(0.15, "co2_gas_heater")
    assert oracle.penalty_weight(legacy) == PENALTY_WEIGHT_KW
    assert oracle.solve_failure_value(legacy) == SOLVE_FAILURE_PENALTY_KW
    assert oracle.penalty_weight(co2) == pytest.approx(PENALTY_WEIGHT_KW * GAS_CO2_KG_PER_KWH)
    assert oracle.solve_failure_value(co2) == pytest.approx(
        SOLVE_FAILURE_PENALTY_KW * GAS_CO2_KG_PER_KWH
    )


def test_legacy_objective_is_the_befund_52_penalised_energy(config) -> None:
    """Anchor: same set point, same threshold -> energy_target + 50 * violation, bit for bit."""
    oracle.init_worker(config)
    x = np.array([88.0, 0.9, 1.0])
    t_amb, phi, threshold = 20.0, 0.55, 0.144
    spec = oracle.OracleSpec(threshold, "legacy_power_kw")
    solution = evaluate_point(*x, t_amb, phi, max_iterations=30, tolerance=1e-7, base_config=config)
    check = check_constraints(
        solution, critical_moisture_kg_per_kg=threshold, gas_velocity_m_per_s=float(x[1])
    )
    expected = float(
        solution.diagnostics["energy_target"] + PENALTY_WEIGHT_KW * check.normalized_violation
    )
    assert oracle.penalized_objective(x, t_amb, phi, spec) == expected


def test_co2_objective_follows_the_carrier_booking(config) -> None:
    oracle.init_worker(config)
    x = np.array([88.0, 0.9, 1.0])
    solution = evaluate_point(*x, 20.0, 0.55, max_iterations=30, tolerance=1e-7, base_config=config)
    d = solution.diagnostics
    duty = d["power_heater_kw"] * config.energy.heater_efficiency
    electric = d["power_fan_kw"] + d["power_pump_kw"] + d["power_atomiser_kw"]
    gas_spec = oracle.OracleSpec(0.15, "co2_gas_heater", gas_heater_efficiency=0.8)
    expected = 0.366 * electric + GAS_CO2_KG_PER_KWH * duty / 0.8
    assert oracle.objective_value(solution, config, gas_spec) == pytest.approx(expected)
    electric_spec = oracle.OracleSpec(0.15, "co2_electric_heater")
    assert oracle.objective_value(solution, config, electric_spec) == pytest.approx(
        0.366 * d["power_total_kw"]
    )
