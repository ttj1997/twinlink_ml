"""Tests of the learned-environment study (docs/30): fixed point, models, environment, RTO."""

from __future__ import annotations

import numpy as np
import pytest

from twinlink_ml.surrogate_env import (
    EnsembleFixedPointModel,
    FixedInputs,
    FixedPointCriterion,
    MultistartConfig,
    PointEvaluation,
    Prediction,
    learned_info,
    multistart_minimize,
    solve_fixed_points,
    start_points,
)
from twinlink_ml.surrogate_env.reading import classify, criterion, energy_excess, point_metrics

FIXED = FixedInputs(
    nozzle_pressure_bar=1.002634656115166, spray_rate_kg_per_h=1.5, suspension_solid_fraction=0.3
)


def _solve(f, n_heads=1):
    def residual(candidates: np.ndarray) -> np.ndarray:
        return f(candidates) - candidates

    return solve_fixed_points(residual, 1, n_heads)


def test_unique_stable_fixed_point_is_found() -> None:
    result = _solve(lambda x: 0.3 * x + 0.7)  # x* = 1, slope of r = -0.7
    assert result.status[0, 0] == "ok"
    assert result.root[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert abs(result.residual[0, 0]) <= 1e-3


def test_nonlinear_relaxation_is_found() -> None:
    result = _solve(lambda x: x + 0.4 * (np.tanh(2.0 - x)))  # r = 0.4 tanh(2 - x), root x = 2
    assert result.status[0, 0] == "ok"
    assert result.root[0, 0] == pytest.approx(2.0, abs=1e-4)


def test_unstable_fixed_point_is_flagged() -> None:
    result = _solve(lambda x: 1.5 * x - 0.5)  # r = 0.5 x - 0.5, increasing through x = 1
    assert result.status[0, 0] == "unstable"
    assert np.isnan(result.root[0, 0])


def test_no_root_and_multiple_roots_are_flagged() -> None:
    assert _solve(lambda x: x + 1.0).status[0, 0] == "no_root"
    # r = -(x - 1)(x + 1)(x - 3): three roots in [-5, 5]
    assert _solve(lambda x: x - (x - 1.0) * (x + 1.0) * (x - 3.0)).status[0, 0] == "multiple_roots"


def test_root_outside_the_scan_range_is_no_root() -> None:
    assert _solve(lambda x: 0.5 * x + 4.0).status[0, 0] == "no_root"  # x* = 8 > 5


def test_not_converged_when_residual_is_discontinuous() -> None:
    # r jumps from +1 to -1 at x = 0.37 (no root): the interpolated point keeps |r| = 1
    result = _solve(lambda x: x + np.where(x < 0.37, 1.0, -1.0))
    assert result.status[0, 0] == "not_converged"


def test_heads_are_independent() -> None:
    def f(c: np.ndarray) -> np.ndarray:
        out = c.copy()
        out[..., 0] = 0.5 * c[..., 0] + 1.0  # root 2
        out[..., 1] = c[..., 1] + 1.0  # no root
        return out

    result = _solve(f, n_heads=2)
    assert list(result.status[0]) == ["ok", "no_root"]


def test_criterion_validation() -> None:
    with pytest.raises(ValueError):
        FixedPointCriterion(lower=1.0, upper=0.0)
    with pytest.raises(ValueError):
        FixedPointCriterion(residual_tolerance=0.0)


class _Member:
    def __init__(self, values: np.ndarray, ok: np.ndarray) -> None:
        self.values, self.ok = values, ok

    def steady_physical(self, actions, contexts):  # noqa: ARG002
        status = np.where(self.ok, "ok", "no_root").astype(object)
        return self.values, self.ok, status


def test_ensemble_pessimistic_moisture_and_validity() -> None:
    n = 2
    members = []
    for moisture in (0.10, 0.11, 0.12):
        values = np.tile(np.array([1.0, moisture, 40.0]), (n, 1))
        ok = np.ones((n, 3), dtype=bool)
        members.append(_Member(values, ok))
    members[2].ok[1, 2] = False
    ensemble = EnsembleFixedPointModel(members, k_sigma=2.0)  # type: ignore[arg-type]
    prediction = ensemble.predict(np.zeros((n, 2)), np.zeros((n, 2)))
    assert prediction.moisture_kg_per_kg[0] == pytest.approx(0.11)
    assert prediction.moisture_std_kg_per_kg[0] == pytest.approx(0.01)
    assert prediction.moisture_for_constraint_kg_per_kg[0] == pytest.approx(0.13)
    assert list(prediction.valid) == [True, False]
    assert np.isnan(prediction.energy_kw[1])
    assert prediction.status_counts["bed_temperature_celsius/no_root"] == 1


def _prediction(valid: bool, moisture: float = 0.14, bed: float = 50.0) -> dict:
    return Prediction(
        energy_kw=np.array([1.2]),
        moisture_kg_per_kg=np.array([moisture]),
        bed_temperature_celsius=np.array([bed]),
        moisture_std_kg_per_kg=np.array([0.0]),
        moisture_for_constraint_kg_per_kg=np.array([moisture]),
        valid=np.array([valid]),
        status_counts={},
    ).point(0)


def test_learned_info_layout_and_violation() -> None:
    kwargs = {
        "quality_limit_kg_per_kg": 0.15,
        "inlet_temperature_celsius": 80.0,
        "gas_velocity_m_per_s": 0.9,
        "ambient_temperature_celsius": 20.0,
        "ambient_relative_humidity": 0.5,
        "fixed": FIXED,
        "energy_unit": "power_kw",
    }
    ok = learned_info(_prediction(True, moisture=0.16, bed=85.0), **kwargs)
    check = ok["constraint_check"]
    assert not ok["solve_failed"] and not ok["feasible"]
    assert check["solver_feasible"]
    assert check["moisture_violation_kg_per_kg"] == pytest.approx(0.01)
    assert check["normalized_violation"] == pytest.approx(0.01 / 0.15 + 0.5)
    invalid = learned_info(_prediction(False), **kwargs)
    assert invalid["solve_failed"] and not invalid["feasible"]


def test_learned_env_masks_invalid_and_rewards_valid_point() -> None:
    from twinlink_ml.rl.env import BanditEnvConfig
    from twinlink_ml.rl.lagrange import LagrangeConfig
    from twinlink_ml.surrogate_env import LearnedSeparatedLagrangeBanditEnv, OperatingPointModel

    class _Model(OperatingPointModel):
        def __init__(self, valid: bool) -> None:
            self.valid = valid

        def predict(self, actions, contexts):  # noqa: ARG002
            return Prediction(
                energy_kw=np.array([1.3]),
                moisture_kg_per_kg=np.array([0.13]),
                bed_temperature_celsius=np.array([45.0]),
                moisture_std_kg_per_kg=np.array([0.0]),
                moisture_for_constraint_kg_per_kg=np.array([0.13]),
                valid=np.array([self.valid]),
                status_counts={},
            )

    lagrange = LagrangeConfig(
        margin_kg_per_kg=0.03,
        initial_multiplier_kw=0.5,
        step_size_kw=0.2,
        update_interval_steps=250,
    )
    config = BanditEnvConfig(product_moisture_limit_kg_per_kg=0.15)
    env = LearnedSeparatedLagrangeBanditEnv(_Model(True), FIXED, lagrange, config)
    env.reset(seed=0)
    _, reward, *_, info = env.step(np.zeros(2))
    g = (0.13 - (0.15 - 0.03)) / 0.15
    assert not info["masked"]
    assert reward == pytest.approx(-(1.3 + 0.5 * g))
    env_bad = LearnedSeparatedLagrangeBanditEnv(_Model(False), FIXED, lagrange, config)
    env_bad.reset(seed=0)
    _, _, *_, info_bad = env_bad.step(np.zeros(2))
    assert info_bad["masked"] and info_bad["mask_reason"] == "solve_failed"
    assert env_bad.n_invalid == 1


def test_multistart_finds_constrained_minimum() -> None:
    # energy falls with x, the "moisture" rises with x; constraint m <= 0.5 at x = 0.5
    def evaluate(x: float, y: float) -> PointEvaluation:
        energy = 2.0 - x + (y - 0.3) ** 2
        moisture = x
        objective = energy + 50.0 * max(0.0, moisture - 0.5) / 0.5
        return PointEvaluation(objective, energy, moisture, 40.0, True)

    result = multistart_minimize(evaluate, [(0.0, 1.0), (0.0, 1.0)], (0.2, 0.8), MultistartConfig())
    assert result["inlet_temperature_celsius"] == pytest.approx(0.5, abs=1e-3)
    assert result["gas_velocity_m_per_s"] == pytest.approx(0.3, abs=1e-2)
    assert len(result["starts"]) == 8


def test_start_points_include_fixed_point_and_lie_in_box() -> None:
    points = start_points([(60.0, 120.0), (0.8, 2.5)], (93.0, 0.8), MultistartConfig())
    assert points.shape == (8, 2)
    assert tuple(points[0]) == (93.0, 0.8)
    assert np.all(points[:, 0] >= 60.0) and np.all(points[:, 0] <= 120.0)


def _rows(energies, feasible, predicted=None):
    rows = []
    for i, (e, f) in enumerate(zip(energies, feasible, strict=True)):
        row = {
            "true_feasible": f,
            "true_solve_failed": False,
            "true_energy_kw": e,
            "true_moisture_kg_per_kg": 0.14 if f else 0.16,
            "energy_oracle_kw": 1.0,
            "energy_oracle_m_kw": 1.01,
        }
        if predicted is not None:
            row["predicted"] = predicted[i]
        rows.append(row)
    return rows


def test_point_metrics_gap_and_violation() -> None:
    predicted = [
        {
            "valid": True,
            "energy_kw": 0.9,
            "moisture_kg_per_kg": 0.13,
            "moisture_for_constraint_kg_per_kg": 0.14,
        },
        {
            "valid": True,
            "energy_kw": 1.0,
            "moisture_kg_per_kg": 0.14,
            "moisture_for_constraint_kg_per_kg": 0.145,
        },
    ]
    m = point_metrics(_rows([1.0, 1.1], [True, False], predicted), 0.15)
    assert m["violation_share"] == pytest.approx(0.5)
    assert m["max_moisture_violation_kg_per_kg"] == pytest.approx(0.01)
    assert m["energy_over_optimum_pct"] == pytest.approx(5.0)
    assert m["gap_energy_mean_kw"] == pytest.approx(-0.1)
    assert m["gap_moisture_mean_kg_per_kg"] == pytest.approx((-0.01 - 0.02) / 2)
    assert m["n_model_feasible_truly_infeasible"] == 1.0


def test_reading_classes() -> None:
    ref = [np.full(25, 1.0), np.full(25, 1.0)]
    good = criterion([(25, 25)] * 3, [energy_excess(np.full(25, 1.01), ref)] * 3)
    infeasible = criterion([(20, 25)] * 3, [0.0] * 3)
    costly = criterion([(25, 25)] * 3, [energy_excess(np.full(25, 1.03), ref)] * 3)
    assert good.passes and not infeasible.passes and not costly.passes
    assert classify(good, infeasible) == "G0"
    assert classify(infeasible, good) == "G1"
    assert classify(costly, infeasible) == "G2"


def test_energy_excess_uses_points_solved_on_both_sides() -> None:
    energy = np.array([1.0, np.nan, 1.0])
    ref = [np.array([1.0, 5.0, np.nan])]
    assert energy_excess(energy, ref) == pytest.approx(0.0)


def test_feasibility_share_boundary() -> None:
    # 72 of 75 = 0.96 passes, 71 of 75 = 0.9467 does not
    assert criterion([(24, 25), (24, 25), (24, 25)], [0.0] * 3).passes
    assert not criterion([(23, 25), (24, 25), (24, 25)], [0.0] * 3).passes
