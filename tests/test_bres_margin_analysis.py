"""Pure functions of the margin study T1b (``analyze_bres_margin.py``) and the margin option."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    for extra in (SCRIPTS_DIR, SCRIPTS_DIR / "verify"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location(
        f"_script_under_test_{name}", SCRIPTS_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def margin():
    return _load("analyze_bres_margin")


def test_closed_fractions_variants(margin) -> None:
    e_fixed = np.array([10.0, 10.0, 10.0, 10.0])
    e_oracle = np.array([8.0, 8.0, 8.0, 8.0])
    e_policy = np.array([9.0, 8.0, np.nan, 9.0])
    feasible = np.array([True, True, False, True])
    out = margin.closed_fractions(e_fixed, e_oracle, e_policy, feasible)
    assert out["A"] == pytest.approx((1 + 2 + 1) / 6.0)  # feasible contexts only
    assert out["B"] == pytest.approx((1 + 2 + 1) / 8.0)  # infeasible counts as 0 in the denominator


def test_closed_fractions_all_feasible_variants_coincide(margin) -> None:
    e_fixed = np.array([10.0, 12.0])
    e_oracle = np.array([8.0, 9.0])
    e_policy = np.array([8.5, 9.5])
    out = margin.closed_fractions(e_fixed, e_oracle, e_policy, np.array([True, True]))
    assert out["A"] == pytest.approx(out["B"])


def test_decomposition_adds_up(margin) -> None:
    out = margin.decompose(e_policy=12.0, e_oracle_m=10.0, e_oracle_km=10.5)
    assert out["margin_pct"] == pytest.approx(5.0)
    assert out["policy_pct"] == pytest.approx(15.0)
    assert out["total_pct"] == pytest.approx(out["margin_pct"] + out["policy_pct"])


def test_gate_is_smallest_k_with_strict_feasibility(margin) -> None:
    ones = [1.0] * 5
    almost = [1.0, 1.0, 1.0, 1.0, 0.96]
    assert margin.smallest_k_meeting_gate({1: ones, 2: almost, 3: ones, 5: ones}) == 3
    assert margin.smallest_k_meeting_gate({2: almost, 3: almost, 5: almost}) is None
    # k = 1 (Befund 93) never counts as a gate margin of this study
    assert margin.smallest_k_meeting_gate({1: ones, 2: almost}) is None


def test_above_ceiling_rule(margin) -> None:
    assert margin.above_ceiling(0.906, 0.818)
    assert not margin.above_ceiling(0.818, 0.818)
    assert not margin.above_ceiling(float("nan"), 0.818)


def test_margin_option_defaults_and_sweep_exemption() -> None:
    rr = _load("run_rl_bandit_residual")
    own = {p.name for p in rr.sweep_output_paths("rl_bandit_residual_b_k2")}
    assert "rl_bandit_residual_b_k5_ressep_seed4.jsonl" in own  # siblings exempt each other
    assert "rl_bandit_residual_b_k2_ressep_trace_seed0.jsonl" in own
    base = {p.name for p in rr.sweep_output_paths(rr.PREFIX)}
    assert all(name.startswith("rl_bandit_residual_") for name in base)
    assert not any("_b_k" in name for name in base)  # Befund 93 sweep unchanged
