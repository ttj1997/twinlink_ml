"""M0: the oracle mechanics moved to ``simulation/oracle_search.py`` behave as before.

The functions were moved verbatim from ``scripts/study_operating_point_regret.py``; the
script re-imports them. These tests pin that the script still exposes the same objects
(the probe scripts read them off the loaded script module) and that the DE + Nelder-Mead
wrapper is exactly the two scipy calls it was.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation import oracle_search as oracle

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"


def _load_script():
    path = REPO_ROOT / "scripts" / "study_operating_point_regret.py"
    spec = importlib.util.spec_from_file_location("regret_script_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_reexports_the_moved_objects() -> None:
    script = _load_script()
    assert script.OptimizationBudget is oracle.OptimizationBudget
    assert script.evaluate_point is oracle.evaluate_point
    assert script._optimize is oracle.run_de_nm
    assert script.build_grid is oracle.build_grid
    assert script.action_bounds is oracle.action_bounds
    assert script.compute_regret is oracle.compute_regret


def test_run_de_nm_is_the_two_scipy_calls() -> None:
    budget = oracle.OptimizationBudget(
        seed=0,
        popsize=4,
        maxiter=5,
        de_tol=1.0e-4,
        search_max_iterations=30,
        search_tolerance=1.0e-7,
        workers=1,
    )
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]

    def objective(x: np.ndarray, shift: float) -> float:
        return float(np.sum((x - shift) ** 2))

    de_result, refine_result, _, _ = oracle.run_de_nm(
        objective, bounds, (0.5,), budget, de_workers=1
    )
    direct_de = differential_evolution(
        objective,
        bounds,
        args=(0.5,),
        seed=0,
        polish=False,
        workers=1,
        updating="immediate",
        popsize=4,
        maxiter=5,
        tol=1.0e-4,
    )
    direct_refine = minimize(
        objective,
        direct_de.x,
        args=(0.5,),
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 200, "adaptive": True},
    )
    assert np.array_equal(de_result.x, direct_de.x)
    assert np.array_equal(refine_result.x, direct_refine.x)


def test_grid_and_bounds_follow_the_config() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    grid = oracle.build_grid(config, 3)
    assert len(grid) == 9
    bounds = oracle.action_bounds(config)
    assert len(bounds) == 3
    assert all(low < high for low, high in bounds)


def test_gas_velocity_minimum_is_a_run_parameter_not_a_default() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    default = oracle.action_bounds(config)
    shifted = oracle.with_gas_velocity_minimum(default, 1.26)
    assert shifted[0] == default[0] and shifted[2] == default[2]
    assert shifted[1] == (1.26, default[1][1])
    # the config and the input list are untouched
    assert config.actions.gas_velocity_m_per_s.minimum == 0.8
    assert default[1][0] == 0.8
    # the config default value reproduces the default bounds exactly
    assert oracle.with_gas_velocity_minimum(default, default[1][0]) == default


def test_gas_velocity_minimum_must_lie_below_the_upper_bound() -> None:
    import pytest

    bounds = [(60.0, 120.0), (0.8, 2.5), (1.0, 4.0)]
    with pytest.raises(ValueError):
        oracle.with_gas_velocity_minimum(bounds, 2.5)
