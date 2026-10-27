"""Joint-recovery study (docs/NN_joint_recovery.md): bypass, medium variant, strategies S0-S3.

Pins :class:`~twinlink_ml.joint_recovery.problem.JointLineProblem` against the read-only P1 of
Befund 107 (medium flow > 0: unchanged), the bypass (no duty, no pump), the independence of S0
from ``dT_min`` and the medium, and the optimality relations of the strategies: ``S2 <= S1 <= S0``
and ``S2 <= S3``, and ``S1 = S2 = S0`` for a very large ``dT_min``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from twinlink_ml.joint_recovery import (
    BINDINGS,
    JointLineFamily,
    RestrictedProblem,
    build_families,
    joint_test_states,
    load_joint_config,
    medium_optimum,
    strategy_s0,
    strategy_s2,
)
from twinlink_ml.surrogate_opt.config import build_problems
from twinlink_ml.surrogate_opt.optimize import batch_from_problem, minimize_box
from twinlink_ml.surrogate_opt.study import make_test_states

REPO = Path(__file__).resolve().parents[1]
U = np.array([95.0, 60.0, 40.0])
W = np.array([15.0, 0.5, 1.5, 0.3])


@pytest.fixture(scope="module")
def config():  # type: ignore[no-untyped-def]
    base = load_joint_config()
    raw = dict(base.base.raw)
    raw["optimizer"] = {
        **raw["optimizer"],
        "truth": {"candidates": 256, "polish": 4},
        "truth_check": {"candidates": 256, "polish": 4, "seed": 7},
    }
    small = dataclasses.replace(base.base, raw=raw)
    return dataclasses.replace(base, base=small)


@pytest.fixture(scope="module")
def families(config):  # type: ignore[no-untyped-def]
    return build_families(config, dt_values=(5.0, 10.0, 20.0, 200.0))


def _family(config, dt: float, pressurised: bool) -> JointLineFamily:  # type: ignore[no-untyped-def]
    p1 = build_problems(config.base)["P1"]
    return JointLineFamily(p1.settings, p1.parameters, dt_min_k=dt, pressurised=pressurised)


def test_medium_flow_positive_is_p1_unchanged(config) -> None:  # type: ignore[no-untyped-def]
    p1 = build_problems(config.base)["P1"]
    for dt in (5.0, 10.0, 20.0):
        ours = _family(config, dt, False).truth()
        theirs = p1.build({**p1.truth_values, "dt_min_k": dt})
        for u in (U, np.array([110.0, 45.0, 12.0]), np.array([70.0, 150.0, 140.0])):
            a, b = ours.evaluate(u, W), theirs.evaluate(u, W)
            assert a.energy_kw == b.energy_kw
            np.testing.assert_array_equal(a.constraints, b.constraints)
            for key, value in b.side.items():
                assert a.side[key] == value


def test_bypass_carries_no_heat_and_no_pump(config) -> None:  # type: ignore[no-untyped-def]
    truth = _family(config, 5.0, False).truth()
    off = truth.evaluate(np.array([95.0, 60.0, 0.0]), W)
    on = truth.evaluate(U, W)
    assert off.side["recovered_kw"] == 0.0
    assert off.side["condensate_kg_per_h"] == 0.0
    assert off.side["exhaust_after_recovery_celsius"] == off.side["exhaust_celsius"]
    assert BINDINGS[int(off.side["binding_index"])] == "bypass"
    assert off.side["bypass"] == 1.0 and on.side["bypass"] == 0.0
    assert off.constraints[3] == W[0] - truth.settings.medium_max_celsius
    # same dryer: the first three constraints do not depend on the recovery
    np.testing.assert_array_equal(off.constraints[:3], on.constraints[:3])
    assert on.side["recovered_kw"] > 0.0  # at 5 K this point recovers heat
    assert math.isnan(off.side["medium_saturation_pressure_bar"])


def test_pressurised_drops_the_medium_constraint(config) -> None:  # type: ignore[no-untyped-def]
    un, pr = _family(config, 5.0, False).truth(), _family(config, 5.0, True).truth()
    assert [c.name for c in pr.constraints] == [c.name for c in un.constraints][:3]
    assert pr.mode.htm_max_celsius is None and un.mode.htm_max_celsius == 95.0
    a, b = un.evaluate(U, W), pr.evaluate(U, W)
    # the medium stays far below 95 degC on this line, so the limit cannot change anything
    assert a.side["medium_hot_celsius"] < 80.0 - 5.0
    assert a.energy_kw == b.energy_kw
    assert b.side["medium_saturation_pressure_bar"] > 0.0


def test_s0_does_not_depend_on_dt_or_medium(families) -> None:  # type: ignore[no-untyped-def]
    u = np.array([95.0, 60.0, 0.0])
    energies = {k: f.truth().evaluate(u, W).energy_kw for k, f in families.items()}
    assert len(set(energies.values())) == 1


def test_test_states_are_those_of_befund_107(config) -> None:  # type: ignore[no-untyped-def]
    p1 = build_problems(config.base)["P1"]
    np.testing.assert_array_equal(joint_test_states(config), make_test_states(p1, config.base))
    assert len(joint_test_states(config)) == 50


def test_restricted_problem_maps_to_full_decisions(families) -> None:  # type: ignore[no-untyped-def]
    truth = families[(5.0, "unpressurised")].truth()
    r = RestrictedProblem(truth, (2,), [95.0, 60.0, np.nan])
    assert r.decisions.names == ("medium_kg_per_h",)
    np.testing.assert_array_equal(r.full(np.array([40.0])), U)
    assert r.evaluate([40.0], W).energy_kw == truth.evaluate(U, W).energy_kw


def _joint_on(family: JointLineFamily, w: np.ndarray) -> np.ndarray:
    truth = family.truth()
    r = minimize_box(batch_from_problem(truth, w), truth.decisions, candidates=512, polish=6)
    return r.u


def test_optimality_relations(config, families) -> None:  # type: ignore[no-untyped-def]
    w = joint_test_states(config)[1]  # a cold state in which the recovery pays at 5 K and 10 K
    for dt in (5.0, 10.0):
        fam = families[(dt, "unpressurised")]
        s0 = strategy_s0(fam, w, config)
        s1 = medium_optimum(fam, w, s0["best"]["u"], config)
        s2 = strategy_s2(fam, w, _joint_on(fam, w), s0["best"]["u"], config)
        u_fixed = [s0["best"]["u"][0] + 5.0, s0["best"]["u"][1] + 5.0]
        s3 = medium_optimum(fam, w, u_fixed, config)
        e0, e1 = s0["best"]["energy"], s1["best"]["energy"]
        e2, e3 = s2["best"]["energy"], s3["best"]["energy"]
        tol = 1e-6
        assert e2 <= e1 * (1 + tol) and e1 <= e0 * (1 + tol) and e2 <= e3 * (1 + tol)
        assert s1["best"]["side"]["recovered_kw"] > 0.0
        assert all(r["best"]["feasible"] for r in (s0, s1, s2, s3))


def test_very_large_dt_gives_s1_equal_s2_equal_s0(config, families) -> None:  # type: ignore[no-untyped-def]
    fam = families[(200.0, "unpressurised")]
    for w in joint_test_states(config)[:3]:
        s0 = strategy_s0(fam, w, config)
        s1 = medium_optimum(fam, w, s0["best"]["u"], config)
        s2 = strategy_s2(fam, w, _joint_on(fam, w), s0["best"]["u"], config)
        assert s1["on"]["side"]["recovered_kw"] == 0.0
        assert s2["on"]["side"]["recovered_kw"] == 0.0
        assert s1["best"]["energy"] == s0["best"]["energy"]
        assert s2["best"]["energy"] == s0["best"]["energy"]
        assert s2["best"]["u"] == s0["best"]["u"]


def _analysis_module():  # type: ignore[no-untyped-def]
    path = REPO / "scripts" / "analyze_joint_recovery.py"
    sys.path.insert(0, str(REPO / "scripts" / "verify"))
    spec = importlib.util.spec_from_file_location("analyze_joint_recovery", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reading_boundaries() -> None:
    reading = _analysis_module().reading
    assert reading(0.0) == "J0" and reading(0.00999) == "J0"
    assert reading(0.01) == "J1" and reading(0.05) == "J1"
    assert reading(0.0501) == "J2"
    assert reading(-1e-9) == "J0"
