"""Tests of the hybrid-model study (docs/NN_hybrid_model.md).

Required by the request: M3 with a residual of exactly zero equals M0 exactly; M2 with the true
parameters hits the truth. In addition: the batched steady solver against the bandit evaluation of
Befund 52/96, the replay of the truth with its own config, the margin bisection, the grid search,
the case rotation of the committed config.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter

import numpy as np
import pytest

from twinlink_ml.evaluation.experiments import DataSpec, generate_data
from twinlink_ml.hybrid_model.coverage import chebyshev_distance
from twinlink_ml.hybrid_model.grid_rto import (
    BisectionSettings,
    CaseEvaluator,
    GridSettings,
    grid_zoom,
    search_parameter,
)
from twinlink_ml.hybrid_model.maps import GreyBoxMap, HybridMap, MapPrediction, SteadyMap
from twinlink_ml.hybrid_model.refit import RefitSettings, fit_kinetics, refit_config
from twinlink_ml.hybrid_model.residual import replay, residual_data
from twinlink_ml.hybrid_model.steady_batch import solve_steady_states
from twinlink_ml.hybrid_model.study import load_hybrid_config, widen
from twinlink_ml.rl.env import FIXED_NOZZLE_PRESSURE_BAR, BanditEnvConfig, OperatingPointBanditEnv
from twinlink_ml.sensor_value.draws import truth_config
from twinlink_ml.simulation import PlaceholderConfigWarning

FITTED = ("critical_moisture_kg_per_kg", "max_saturation_degree")
TRUE_VALUES = {"critical_moisture_kg_per_kg": 0.24, "max_saturation_degree": 0.48}


@pytest.fixture(scope="module")
def env() -> OperatingPointBanditEnv:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return OperatingPointBanditEnv(BanditEnvConfig(product_moisture_limit_kg_per_kg=0.15))


def _points(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(60, 120, n),
            rng.uniform(0.8, 2.5, n),
            rng.uniform(5, 35, n),
            rng.uniform(0.25, 0.85, n),
        ]
    )


def _map(config) -> GreyBoxMap:  # type: ignore[no-untyped-def]
    return GreyBoxMap(
        config,
        widen(config, (0.0, 40.0), (0.10, 0.95)),
        nozzle_pressure_bar=FIXED_NOZZLE_PRESSURE_BAR,
        quality_limit_kg_per_kg=0.15,
    )


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def test_batch_steady_solver_equals_bandit_evaluation(env: OperatingPointBanditEnv) -> None:
    p = _points(12, 3)
    result = solve_steady_states(
        env._base_config, p[:, 0], p[:, 1], p[:, 2], p[:, 3],
        nozzle_pressure_bar=FIXED_NOZZLE_PRESSURE_BAR, quality_limit_kg_per_kg=0.15,
    )  # fmt: skip
    for i, row in enumerate(p):
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=row[0], gas_velocity_m_per_s=row[1],
            ambient_temperature_celsius=row[2], ambient_relative_humidity=row[3],
        )  # fmt: skip
        assert bool(info.get("solve_failed", False)) == bool(result.solve_failed[i])
        if info.get("solve_failed"):
            continue
        check = info["constraint_check"]
        assert info["energy_target"] == result.energy_kw[i]
        assert check["moisture_kg_per_kg"] == result.moisture_kg_per_kg[i]
        assert check["bed_temperature_celsius"] == result.bed_temperature_celsius[i]
        assert check["feasible"] == bool(result.feasible[i])
        assert check["solver_feasible"] == bool(result.solution_feasible[i])
        assert check["normalized_violation"] == pytest.approx(
            result.normalized_violation[i], abs=1e-12
        )


class _ZeroResidual:
    """A residual network whose steady residual is exactly zero everywhere."""

    def steady_physical(self, actions, contexts):  # type: ignore[no-untyped-def]  # noqa: ARG002
        n = np.atleast_2d(actions).shape[0]
        return np.zeros((n, 3)), np.ones((n, 3), dtype=bool), np.full((n, 3), "ok", dtype=object)


@pytest.mark.parametrize("wide", [False, True])
def test_m3_with_zero_residual_equals_m0_exactly(env: OperatingPointBanditEnv, wide: bool) -> None:
    p = _points(20, 5)
    m0 = _map(env._base_config)
    m3 = HybridMap(m0, [_ZeroResidual()])
    a, c = p[:, :2], p[:, 2:]
    if wide:
        c = c.copy()
        c[:5, 0] = [0.0, 40.0, 0.0, 40.0, 2.0]
    base = m0.predict(a, c, wide=wide)
    hybrid = m3.predict(a, c, wide=wide)
    for field in ("energy_kw", "moisture_kg_per_kg", "bed_temperature_celsius", "valid"):
        assert _same(getattr(base, field), getattr(hybrid, field)), field
    ensemble = HybridMap(m0, [_ZeroResidual(), _ZeroResidual()]).predict(a, c, wide=wide)
    assert _same(ensemble.moisture_kg_per_kg, base.moisture_kg_per_kg)
    assert np.all(ensemble.moisture_std_kg_per_kg[ensemble.valid] == 0.0)


def test_m2_with_true_parameters_hits_the_truth_steady(env: OperatingPointBanditEnv) -> None:
    nominal = env._base_config
    truth = truth_config(nominal, TRUE_VALUES)
    m2 = refit_config(nominal, FITTED, [TRUE_VALUES[k] for k in FITTED])
    p = _points(15, 7)
    a, c = p[:, :2], p[:, 2:]
    expected = _map(truth).predict(a, c, wide=False)
    got = _map(m2).predict(a, c, wide=False)
    for field in ("energy_kw", "moisture_kg_per_kg", "bed_temperature_celsius", "valid"):
        assert _same(getattr(expected, field), getattr(got, field)), field
    # the nominal model does not (otherwise the test would be empty)
    nominal_prediction = _map(nominal).predict(a, c, wide=False)
    assert not _same(nominal_prediction.moisture_kg_per_kg, expected.moisture_kg_per_kg)


@pytest.fixture(scope="module")
def small_pool():  # type: ignore[no-untyped-def]
    from twinlink_ml.hybrid_model.study import DEFAULT_CONFIG_PATH, load_hybrid_config
    from twinlink_ml.simulation import load_process_config

    cfg = load_hybrid_config(DEFAULT_CONFIG_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        nominal = load_process_config(cfg.process_config)
        nominal = dataclasses.replace(
            nominal,
            numerics=dataclasses.replace(nominal.numerics, enforce_gas_velocity_projection=True),
        )
        spec = DataSpec.new_study(
            sample_time_s=25.0, run_duration_s=6000.0, n_train_max=4, n_val=2, n_test=2,
            history_length=8, horizon=6, stride=4,
            target_names=("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius"),
        )  # fmt: skip
        truth = generate_data(truth_config(nominal, TRUE_VALUES), spec, master_seed=1)
    return nominal, truth


def test_replay_with_the_truth_config_reproduces_the_truth(small_pool) -> None:  # type: ignore[no-untyped-def]
    _, truth = small_pool
    again = replay(truth, truth.process_config)
    assert np.array_equal(again.outputs, truth.trajectory.outputs)
    residual = residual_data(truth, again)
    assert np.all(residual.trajectory.outputs == 0.0)
    assert residual.split.digest == truth.split.digest


def test_m2_with_true_parameters_hits_the_truth_dynamic(small_pool) -> None:  # type: ignore[no-untyped-def]
    nominal, truth = small_pool
    m2 = refit_config(nominal, FITTED, [TRUE_VALUES[k] for k in FITTED])
    assert np.array_equal(replay(truth, m2).outputs, truth.trajectory.outputs)


def test_refit_recovers_the_true_parameters_from_noise_free_runs(small_pool) -> None:  # type: ignore[no-untyped-def]
    nominal, truth = small_pool
    rows = list(range(4))
    settings = RefitSettings(FITTED, (0.15, 0.2), (0.45, 1.0), max_nfev=60, diff_step=1e-4)
    fit = fit_kinetics(
        nominal, truth, rows,
        ["energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius"],
        settings, use_noise_free=True,
    )  # fmt: skip
    assert fit["theta"] == pytest.approx([TRUE_VALUES[k] for k in FITTED], rel=2e-3)


class _Bowl(SteadyMap):
    """Energy (T - 90)^2 / 100 + 10 u, moisture falls with T and u; known constrained optimum."""

    def predict(self, actions, contexts, *, wide):  # type: ignore[no-untyped-def]  # noqa: ARG002
        t, u = actions[:, 0], actions[:, 1]
        n = t.size
        return MapPrediction(
            energy_kw=(t - 90.0) ** 2 / 100.0 + 10.0 * u,
            moisture_kg_per_kg=0.30 - 0.002 * t - 0.02 * u,
            moisture_std_kg_per_kg=np.full(n, 0.01),
            bed_temperature_celsius=np.full(n, 50.0),
            valid=np.ones(n, dtype=bool),
        )


def test_grid_zoom_finds_the_constrained_optimum() -> None:
    ev = CaseEvaluator(_Bowl(), np.array([[20.0, 0.5]]), np.array([False]))
    bounds = ((60.0, 120.0), (0.8, 2.5))
    action, _ = grid_zoom(ev, np.array([0]), np.array([0.10]), np.zeros(1), bounds, GridSettings())
    # optimum on 0.30 - 0.002 T - 0.02 u = 0.10 at u = 0.8: T = 92 (energy minimum 90 infeasible)
    assert action[0, 1] == pytest.approx(0.8)
    assert action[0, 0] == pytest.approx(92.0, abs=60.0 / 20 / 16 + 1e-9)
    # with k = 1 the constraint moisture is 0.01 higher: T = 97
    action_k, _ = grid_zoom(ev, np.array([0]), np.array([0.10]), np.ones(1), bounds, GridSettings())
    assert action_k[0, 0] == pytest.approx(97.0, abs=60.0 / 20 / 16 + 1e-9)


def test_margin_bisection() -> None:
    need = np.array([0.0, 0.0123, 0.2, 0.05])

    def hold_at(rows: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p >= need[rows]

    result = search_parameter(hold_at, 4, BisectionSettings(upper=0.12, tolerance=1e-4))
    assert result.needed[0] == 0.0
    assert result.needed[2] == np.inf
    for i in (1, 3):
        assert need[i] <= result.needed[i] <= need[i] + 1e-4


def test_case_rotation_of_the_committed_config() -> None:
    cfg = load_hybrid_config()
    counts = Counter(i for row in cfg.interior_grid_indices for i in row)
    assert len(cfg.interior_grid_indices) == 20
    assert set(counts) == set(range(25)) and set(counts.values()) == {4}
    for row in cfg.interior_grid_indices:
        assert sorted(i // 5 for i in row) == list(range(5))
        assert sorted(i % 5 for i in row) == list(range(5))
    ext = Counter(i for row in cfg.extrapolation_per_draw for i in row)
    assert set(ext.values()) == {10}
    for t, phi in cfg.extrapolation_contexts:
        assert not (5.0 <= t <= 35.0 and 0.25 <= phi <= 0.85)


def test_chebyshev_distance() -> None:
    lower, upper = np.zeros(2), np.array([10.0, 1.0])
    samples = np.array([[0.0, 0.0], [10.0, 1.0]])
    d = chebyshev_distance(np.array([[1.0, 0.5], [9.0, 0.9]]), samples, lower, upper)
    assert d == pytest.approx([0.5, 0.1])
