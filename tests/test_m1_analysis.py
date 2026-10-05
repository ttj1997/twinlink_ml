"""M1: the analysis of the margin cost on synthetic arm results (no solver)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    scripts = REPO_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(scripts / "verify"))
    spec = importlib.util.spec_from_file_location(
        "run_m1_margin_cost_for_test", scripts / "run_m1_margin_cost.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _point(co2: float, kwh: float, *, binding: bool = True, feasible: bool = True) -> dict:
    return {
        "ambient_temperature_celsius": 10.0,
        "ambient_relative_humidity": 0.5,
        "feasible": feasible,
        "solve_failed": False,
        "gas_heater": {"co2_kg_per_h": co2, "kwh_sum_kw": kwh},
        "constraint_check": {"binding_moisture": binding},
        "distance_to_validity_boundary_kg_per_kg": 0.05,
        "particle_moisture_kg_per_kg": 0.144,
        "inlet_temperature_celsius": 90.0,
        "gas_velocity_m_per_s": 0.8,
        "nozzle_pressure_bar": 1.0,
    }


def test_remaining_fraction_and_cost() -> None:
    module = _load()
    f = [_point(1.0, 5.0), _point(1.0, 5.0)]
    m0 = [_point(0.5, 3.0), _point(0.9, 4.0, binding=False)]
    m = [_point(0.6, 3.5), _point(0.9, 4.0, binding=False)]
    result = module.analyse(f, m0, m)
    first, second = result["per_point"]
    assert first["co2_kg_per_h"]["remaining_fraction"] == pytest.approx((1.0 - 0.6) / (1.0 - 0.5))
    assert first["co2_kg_per_h"]["cost_abs"] == pytest.approx(0.1)
    assert first["kwh_sum_kw"]["cost_rel"] == pytest.approx(0.5 / 3.0)
    assert second["co2_kg_per_h"]["remaining_fraction"] == pytest.approx(1.0)
    summary = result["summary"]
    assert summary["n_feasible_in_all_arms"] == 2
    assert summary["n_m0_binding"] == 1
    pooled = summary["co2_kg_per_h"]["all"]["remaining_fraction_pooled"]
    assert pooled == pytest.approx((0.4 + 0.1) / (0.5 + 0.1))


def test_infeasible_points_are_excluded() -> None:
    module = _load()
    f = [_point(1.0, 5.0)]
    m0 = [_point(0.5, 3.0)]
    m = [_point(0.6, 3.5, feasible=False)]
    result = module.analyse(f, m0, m)
    assert result["summary"]["n_feasible_in_all_arms"] == 0
    assert result["per_point"][0]["feasible_in_all_arms"] is False
