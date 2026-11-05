"""Tests of the hybrid-model study (docs/NN_hybrid_model.md).

Required by the request: M3 with a residual of exactly zero equals M0 exactly; M2 with the true
parameters hits the truth. In addition: the batched steady solver against the bandit evaluation of
Befund 52/96, the replay of the truth with its own config, the margin bisection, the grid search,
the case rotation of the committed config.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from types import ModuleType

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
from twinlink_ml.hybrid_model.margin_tasks import (
    RoundDecision,
    cell_parameter,
    evaluate_draw,
    next_round,
    search_draw,
)
from twinlink_ml.hybrid_model.refit import RefitSettings, fit_kinetics, refit_config
from twinlink_ml.hybrid_model.residual import replay, residual_data
from twinlink_ml.hybrid_model.steady_batch import solve_steady_states
from twinlink_ml.hybrid_model.study import Case, load_hybrid_config, widen
from twinlink_ml.rl.env import FIXED_NOZZLE_PRESSURE_BAR, BanditEnvConfig, OperatingPointBanditEnv
from twinlink_ml.sensor_value.draws import truth_config
from twinlink_ml.simulation import PlaceholderConfigWarning
from twinlink_ml.utils.tracking import status_is_dirty

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


# --- margins split per draw (after the failed run of a343840, 2026-11-05) ----------------------


class _ShiftedBowl(_Bowl):
    """The bowl with a moisture underestimated by ``shift`` (a model that needs a margin)."""

    def __init__(self, shift: float) -> None:
        self.shift = shift

    def predict(self, actions, contexts, *, wide):  # type: ignore[no-untyped-def]
        p = super().predict(actions, contexts, wide=wide)
        return dataclasses.replace(p, moisture_kg_per_kg=p.moisture_kg_per_kg - self.shift)


def _cases(draw: int, n: int) -> list[Case]:
    return [Case(draw, slot, "interior", 20.0, 0.5, slot) for slot in range(n)]


def test_empty_case_set_gives_empty_columns() -> None:
    ev = CaseEvaluator(_Bowl(), np.array([[20.0, 0.5]]), np.array([False]))
    values = ev.values(np.array([], dtype=int), np.zeros((0, 2)))
    assert all(v.shape == (0,) for v in values.values())
    assert ev.n_predicted == 0


def test_draw_without_inside_case_has_no_search_and_is_evaluated_at_the_cell_parameter() -> None:
    bounds = ((60.0, 120.0), (0.8, 2.5))
    grid = GridSettings(coarse=21, zoom=9, zoom_levels=3, max_moves=50, starts=4)
    bisection = BisectionSettings(upper=0.12, tolerance=1e-4)
    limit, shift = 0.10, 0.01  # at 0.10 the moisture constraint is active (see above)
    contexts = np.array([[20.0, 0.5]] * 3)
    wide = np.zeros(3, dtype=bool)
    common = {"limit": limit, "is_k": False, "bounds": bounds}
    # draw A: three inside cases; draw B: two cases, none inside
    model_a = CaseEvaluator(_ShiftedBowl(shift), contexts, wide)
    truth_a = CaseEvaluator(_Bowl(), contexts, wide)
    model_b = CaseEvaluator(_ShiftedBowl(shift), contexts[:2], wide[:2])
    truth_b = CaseEvaluator(_Bowl(), contexts[:2], wide[:2])
    inside_b = np.zeros(2, dtype=bool)
    search_b = search_draw(model_b, truth_b, inside_b, grid=grid, settings=bisection, **common)
    assert search_b.needed == {} and search_b.n_solves == {}
    assert model_b.n_predicted == 0 and truth_b.n_predicted == 0  # no search at all
    search_a = search_draw(
        model_a, truth_a, np.ones(3, dtype=bool), grid=grid, settings=bisection, **common
    )
    assert sorted(search_a.needed) == [0, 1, 2]
    for v in search_a.needed.values():
        assert shift - 1e-3 <= v <= shift + 2e-4  # the underestimate, up to grid and bracket
    cell = cell_parameter([*search_a.needed.values(), *search_b.needed.values()], 0.95)
    assert cell.n_inside == 3 and cell.rank == 3 and cell.reached
    oracle = [{"energy_kw": 1.0}] * 2
    rows = evaluate_draw(
        model_b, truth_b, _cases(1, 2), inside_b, oracle, cell.quantile, draw=1, grid=grid, **common
    )
    assert [r["slot"] for r in rows] == [0, 1]
    assert all(not r["inside"] and r["holds"] for r in rows)
    assert all(r["effective_margin"] == pytest.approx(cell.quantile) for r in rows)


def test_raise_rule_over_rounds() -> None:
    cell = cell_parameter([0.0, 0.01, 0.02, 0.03, np.inf], 0.6)  # rank ceil(0.6 * 5) = 3
    assert cell.quantile == 0.02 and cell.candidates == (0.02, 0.03)
    assert next_round(cell, [], 0.6, 0.12) == RoundDecision(False, 0.02, 0)
    assert next_round(cell, [0.6], 0.6, 0.12) == RoundDecision(True, 0.02, 0)
    assert next_round(cell, [0.4], 0.6, 0.12) == RoundDecision(False, 0.03, 1)
    assert next_round(cell, [0.4, 0.8], 0.6, 0.12) == RoundDecision(True, 0.03, 1)
    assert next_round(cell, [0.4, 0.5], 0.6, 0.12) == RoundDecision(True, 0.03, 2)
    none = cell_parameter([], 0.95)
    assert none.n_inside == 0 and not none.reached
    assert next_round(none, [], 0.95, 0.12) == RoundDecision(False, 0.12, 0)
    assert next_round(none, [math.nan], 0.95, 0.12) == RoundDecision(True, 0.12, 0)
    inf = cell_parameter([np.inf, np.inf], 0.95)
    assert not inf.reached and next_round(inf, [], 0.95, 10.0).parameter == 10.0


def _evaluation_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_hybrid_evaluation.py"
    spec = importlib.util.spec_from_file_location("run_hybrid_evaluation_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses of the script look up their module
    spec.loader.exec_module(module)
    return module


def test_clean_tree_exception_is_exactly_the_output_and_the_two_inputs(tmp_path: Path) -> None:
    script = _evaluation_script()
    runs = tmp_path / "artifacts" / "runs"
    out = runs / "hybrid_model_v1_margins_per_draw.jsonl"
    training, oracle = "hybrid_model_training_v1", "hybrid_model_v1_oracle"
    exempt = script.clean_tree_exceptions(runs, out, training, oracle)
    assert exempt == [out, runs / f"{training}.jsonl", runs / f"{oracle}.jsonl"]
    assert script.clean_tree_exceptions(runs, out, training, None) == [
        out,
        runs / f"{training}.jsonl",
    ]
    untracked = (
        "?? artifacts/runs/hybrid_model_v1_margins_per_draw.jsonl\n"
        "?? artifacts/runs/hybrid_model_training_v1.jsonl\n"
        "?? artifacts/runs/hybrid_model_v1_oracle.jsonl\n"
    )
    assert not status_is_dirty(untracked, exempt, tmp_path)
    # any other untracked file, or a modified tracked input, still counts
    assert status_is_dirty(untracked + "?? artifacts/runs/other.jsonl\n", exempt, tmp_path)
    assert status_is_dirty(untracked + "?? scripts/new.py\n", exempt, tmp_path)
    assert status_is_dirty(" M artifacts/runs/hybrid_model_v1_oracle.jsonl\n", exempt, tmp_path)
    assert status_is_dirty(" M scripts/run_hybrid_evaluation.py\n", exempt, tmp_path)


class _CountingBowl(_Bowl):
    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict(self, actions, contexts, *, wide):  # type: ignore[no-untyped-def]
        self.calls.append(len(actions))
        return super().predict(actions, contexts, wide=wide)


def test_chunked_predictions_equal_one_call() -> None:
    contexts = np.array([[20.0, 0.5], [10.0, 0.3], [30.0, 0.8]])
    wide = np.array([False, False, True])
    rng = np.random.default_rng(3)
    actions = np.column_stack([rng.uniform(60, 120, 50), rng.uniform(0.8, 2.5, 50)])
    cases = rng.integers(0, 3, 50)
    whole, chunked = _CountingBowl(), _CountingBowl()
    a = CaseEvaluator(whole, contexts, wide).values(cases, actions)
    b = CaseEvaluator(chunked, contexts, wide, max_batch=7).values(cases, actions)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert max(chunked.calls) <= 7 and sum(chunked.calls) == sum(whole.calls)
    with pytest.raises(ValueError):
        CaseEvaluator(_Bowl(), contexts, wide, max_batch=0)
