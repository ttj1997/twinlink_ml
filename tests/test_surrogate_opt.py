"""Surrogate-RTO study (docs/25): models, optimiser, designs, budget, and the two true models.

The quadratic test problem (:mod:`twinlink_ml.surrogate_opt.quadratic`) has a known optimum that
every model must hit with a large design; the P1/P2 tests pin the study's true models against the
read-only models they wrap (flowsheet line, R1 solver).
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from twinlink_ml.surrogate_opt.config import (
    DEFAULT_CONFIG_PATH,
    SurrogateStudyConfig,
    build_problems,
    load_study_config,
)
from twinlink_ml.surrogate_opt.optimize import minimize_box
from twinlink_ml.surrogate_opt.quadratic import QuadraticFamily, QuadraticProblem
from twinlink_ml.surrogate_opt.study import (
    BudgetedTruth,
    BudgetExceededError,
    adaptive_arm,
    adaptive_batches,
    blackbox_arm,
    check_disjoint,
    greybox_arm,
    joint_box,
    make_test_states,
    surrogate_batch,
    training_design,
)
from twinlink_ml.surrogate_opt.surrogates import QuadraticSurface, make_surrogate

REPO = Path(__file__).resolve().parents[1]


def _config(**changes: object) -> SurrogateStudyConfig:
    """The study config with small settings for tests."""
    base = load_study_config(DEFAULT_CONFIG_PATH)
    raw = copy.deepcopy(base.raw)
    raw["models"]["m2"]["restarts"] = 0
    raw["optimizer"]["surrogate"] = {"candidates": 128, "polish": 2}
    raw["optimizer"]["model"] = {"candidates": 128, "polish": 2}
    defaults = {"budgets": (8, 128), "kappas": (0.0,), "test_count": 6, "raw": raw}
    return dataclasses.replace(base, **{**defaults, **changes})  # type: ignore[arg-type]


def _regret(
    family: QuadraticFamily, rows: list[dict], test_w: np.ndarray, n: int
) -> tuple[float, int]:
    truth = family.truth()
    e_star = np.array([truth.evaluate(truth.optimum(w), w).energy_kw for w in test_w])
    row = next(r for r in rows if r["N"] == n)
    e = np.array(row["score"]["energy"])
    return float(np.max((e - e_star) / e_star)), int(sum(not f for f in row["score"]["feasible"]))


# ---------------------------------------------------------------------- the optimiser
def test_minimize_box_finds_the_known_optimum_of_the_true_quadratic() -> None:
    p = QuadraticProblem()
    for w in (np.array([0.1, 0.9]), np.array([0.9, 0.1]), np.array([0.5, 0.5])):

        def model(u: np.ndarray, w: np.ndarray = w) -> tuple[np.ndarray, np.ndarray]:
            y = p.outputs(u, w)
            return y[:, 0], y[:, 1:]

        r = minimize_box(model, p.decisions, candidates=64, polish=3)
        assert r.feasible
        np.testing.assert_allclose(r.u, p.optimum(w), atol=1e-5)


# ---------------------------------------------------------------------- the models
@pytest.mark.parametrize(("key", "tol"), [("M1", 1e-6), ("M2", 2e-4), ("M3", 5e-3)])
def test_every_black_box_model_hits_the_known_optimum_with_a_large_design(
    key: str, tol: float
) -> None:
    family, cfg = QuadraticFamily(), _config()
    tw = make_test_states(family, cfg)
    rows = blackbox_arm(family, key, 0, cfg, tw, log=lambda _: None)
    regret, infeasible = _regret(family, rows, tw, 128)
    assert regret < tol
    if key != "M3":
        assert infeasible == 0


def test_greybox_fit_is_exact_when_nothing_is_hidden_and_biased_when_something_is() -> None:
    cfg = _config()
    exact = QuadraticFamily()
    tw = make_test_states(exact, cfg)
    rows = greybox_arm(exact, 0, cfg, tw, log=lambda _: None)
    np.testing.assert_allclose(rows[0]["fit"]["theta"], [0.35, 0.45], atol=1e-6)
    regret, infeasible = _regret(exact, rows, tw, 8)
    assert regret < 1e-8 and infeasible == 0
    hidden = QuadraticFamily({"h0": 0.55})
    rows = greybox_arm(hidden, 0, cfg, tw, log=lambda _: None)
    assert sum(not f for f in rows[-1]["score"]["feasible"]) > 0  # the model's h0 is wrong


def test_quadratic_surface_loo_equals_brute_force_loo() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.random((20, 2)), rng.random((20, 1))
    alpha = np.array([0.3])
    m = QuadraticSurface(alpha).fit(x, y)
    loo = [
        y[i, 0]
        - QuadraticSurface(alpha)
        .fit(np.delete(x, i, 0), np.delete(y, i, 0))
        .predict(x[i : i + 1])[0][0, 0]
        for i in range(20)
    ]
    # the brute force re-standardises the features without point i; agreement to that order
    assert m.sigma[0] == pytest.approx(np.sqrt(np.mean(np.square(loo))), rel=0.05)


def test_conservative_margin_tightens_the_model_constraint() -> None:
    family = QuadraticFamily()
    truth = family.truth()
    u, w = training_design(truth, 0, 16)
    y = truth.outputs(u, w)
    gp = make_surrogate("M2", _config().model("m2"), 0).fit(
        joint_box(truth).to_unit(np.hstack((u, w))), y
    )
    uu = np.array([[0.5, 0.5]])
    g0 = surrogate_batch(gp, truth, np.array([0.3, 0.3]), 0.0)(uu)[1]
    g2 = surrogate_batch(gp, truth, np.array([0.3, 0.3]), 2.0)(uu)[1]
    assert np.all(g2 > g0)


# ---------------------------------------------------------------------- designs and budget
def test_test_states_are_disjoint_from_every_training_design() -> None:
    cfg = load_study_config()
    for family in (QuadraticFamily(),):
        tw = make_test_states(family, cfg)
        truth = family.truth()
        for seed in cfg.seeds:
            _, w = training_design(truth, seed, max(cfg.budgets))
            assert check_disjoint(truth.disturbances, tw, w) > 0.0
        with pytest.raises(ValueError, match="coincides"):
            check_disjoint(truth.disturbances, tw, tw[:3])


def test_training_designs_are_nested_and_differ_between_seeds() -> None:
    truth = QuadraticProblem()
    u8, w8 = training_design(truth, 3, 8)
    u64, w64 = training_design(truth, 3, 64)
    np.testing.assert_array_equal(u8, u64[:8])
    np.testing.assert_array_equal(w8, w64[:8])
    assert not np.allclose(training_design(truth, 4, 8)[0], u8)


def test_budgeted_truth_counts_and_refuses_beyond_the_budget() -> None:
    b = BudgetedTruth(QuadraticProblem(), 2)
    b(np.zeros(2), np.zeros(2))
    b(np.zeros(2), np.zeros(2))
    assert b.calls == 2
    with pytest.raises(BudgetExceededError):
        b(np.zeros(2), np.zeros(2))


def test_adaptive_batches_hit_every_budget() -> None:
    sizes = adaptive_batches(8, (8, 16, 32, 64, 128, 256), 8)
    totals = 8 + np.cumsum(sizes)
    assert totals[-1] == 256
    assert {16, 32, 64, 128, 256} <= set(totals.tolist())
    assert max(sizes) <= 8


def test_adaptive_arm_counts_every_true_evaluation() -> None:
    family = QuadraticFamily()
    cfg = _config(budgets=(8, 16, 32))
    tw = make_test_states(family, cfg)
    rows = adaptive_arm(family, 0.0, 0, cfg, tw, log=lambda _: None)
    assert [r["true_evaluations"] for r in rows] == [8, 16, 32]
    assert [r["N"] for r in rows] == [8, 16, 32]
    assert len(rows[-1]["design_u"]) == 32
    # the sequential points sit at the predicted optima of new training disturbances
    w_seq = rows[-1]["design_w"][8:]
    assert check_disjoint(family.truth().disturbances, tw, w_seq) > 0.0


# ---------------------------------------------------------------------- P1 and P2
def test_line_with_zero_heat_loss_is_the_flowsheet_line_unchanged() -> None:
    from twinlink_ml.surrogate_opt.line_problem import dryer_heat_loss_check

    line = build_problems(load_study_config())["P1"]
    for u, w in (
        ([90.0, 60.0, 20.0], [20.0, 0.5, 1.5, 0.3]),
        ([120.0, 50.0, 12.0], [5.0, 0.25, 1.2, 0.4]),
    ):
        assert dryer_heat_loss_check(line.settings, np.array(u), np.array(w)) == 0.0  # type: ignore[attr-defined]


def test_p1_hidden_shift_is_the_upper_edge_of_the_befund_90_band() -> None:
    tree = ast.parse(
        (REPO / "scripts" / "report_heat_recovery_comparison.py").read_text(encoding="utf-8")
    )
    band = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "DT_MIN_K"
    )
    spec = {p.name: p for p in build_problems(load_study_config())["P1"].parameters}["dt_min_k"]
    assert spec.truth == max(band) and not spec.fitted


def test_p2_hidden_shift_is_read_from_the_headroom_robustness_band() -> None:
    pytest.importorskip("casadi")
    from twinlink_ml.headroom.config import load_headroom_config

    spec = {p.name: p for p in build_problems(load_study_config())["P2"].parameters}
    assert (
        spec["gab_monolayer_factor"].truth == load_headroom_config().robustness.monolayer_factor[1]
    )
    assert not spec["gab_monolayer_factor"].fitted


def test_p2_nominal_optimum_is_the_r1_optimum() -> None:
    pytest.importorskip("casadi")
    from twinlink_ml.headroom.controllers import SteadyStateOptimizer, nominal_disturbance

    family = build_problems(load_study_config())["P2"]
    nominal = family.build({})
    w = family.nominal_disturbance  # type: ignore[attr-defined]
    ref = SteadyStateOptimizer(nominal.study, family.process).solve(  # type: ignore[attr-defined]
        nominal_disturbance(nominal.study, family.process),
        nominal.moisture_limit,  # type: ignore[attr-defined]
    )
    ours = nominal.optimum(w, np.array([[90.0, 0.8], [60.0, 2.5]]))  # type: ignore[attr-defined]
    np.testing.assert_allclose(ours.u, ref.command, rtol=1e-6)
    assert ours.energy_kw == pytest.approx(ref.power_kw, rel=1e-8)
    ev = nominal.evaluate(ours.u, w)
    assert ev.energy_kw == pytest.approx(ref.power_kw, rel=1e-7)
    assert ev.constraints[0] == pytest.approx(0.0, abs=1e-6)


def test_p2_steady_state_matches_the_exact_numpy_solver() -> None:
    pytest.importorskip("casadi")
    import warnings

    from twinlink_ml.simulation import GreyBoxFluidBedGranulator, PlaceholderConfigWarning
    from twinlink_ml.simulation.steady_state import solve_steady_state

    family = build_problems(load_study_config())["P2"]
    truth = family.truth()
    u, w = np.array([95.0, 1.2]), np.array([12.0, 0.6, 1.6, 0.25])
    z, power = truth.steady_state(u, w)  # type: ignore[attr-defined]
    proc = truth.process  # type: ignore[attr-defined]
    proc = dataclasses.replace(
        proc,
        actions=dataclasses.replace(
            proc.actions,
            inlet_temperature_celsius=dataclasses.replace(
                proc.actions.inlet_temperature_celsius, nominal=u[0]
            ),
            gas_velocity_m_per_s=dataclasses.replace(
                proc.actions.gas_velocity_m_per_s, nominal=u[1]
            ),
            spray_rate_kg_per_h=dataclasses.replace(proc.actions.spray_rate_kg_per_h, nominal=w[2]),
        ),
        disturbances=dataclasses.replace(
            proc.disturbances,
            ambient_temperature_celsius=dataclasses.replace(
                proc.disturbances.ambient_temperature_celsius, nominal=w[0]
            ),
            ambient_relative_humidity=dataclasses.replace(
                proc.disturbances.ambient_relative_humidity, nominal=w[1]
            ),
            suspension_solid_fraction=dataclasses.replace(
                proc.disturbances.suspension_solid_fraction, nominal=w[3]
            ),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        sol = solve_steady_state(GreyBoxFluidBedGranulator(proc))
    assert z[0] / truth.solid_mass == pytest.approx(sol.particle_moisture_kg_per_kg, rel=1e-5)  # type: ignore[attr-defined]
    assert z[1] == pytest.approx(sol.bed_temperature_celsius, abs=1e-4)
    assert power == pytest.approx(sol.diagnostics["power_total_kw"], rel=1e-6)


def test_p2_bisection_fallback_finds_the_newton_root() -> None:
    pytest.importorskip("casadi")
    truth = build_problems(load_study_config())["P2"].truth()
    u, w = np.array([90.0, 1.0]), np.array([20.0, 0.5, 1.5, 0.3])
    z, _ = truth.steady_state(u, w)  # type: ignore[attr-defined]
    zb = truth._bisect(np.concatenate((u, w)))  # type: ignore[attr-defined]
    np.testing.assert_allclose(zb, z, rtol=1e-8, atol=1e-9)
    # a hot, fast point where Newton fails from every start: the fallback still closes both balances
    hot = truth.evaluate(np.array([118.2, 2.43]), np.array([23.8, 0.285, 1.245, 0.275]))
    assert hot.side["bed_temperature_celsius"] > 80.0  # infeasible, but evaluated
