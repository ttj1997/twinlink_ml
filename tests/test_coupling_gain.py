"""Verdict rule of the coupling gain (fixed before any run of the stage)."""

from __future__ import annotations

import pytest

from twinlink_ml.evaluation import coupling_gain as cg


def test_constants_are_those_fixed_by_the_project_lead() -> None:
    assert cg.INTERACTION_RESOLUTION_KG_PER_H == 0.5
    assert cg.GAIN_MEDIAN_MIN_KG_PER_H == 1.0
    assert cg.GAIN_SEED_FACTOR == 3.0
    assert cg.GAIN_PER_SCENARIO_MIN_KG_PER_H == 0.2
    assert cg.GAIN_SCENARIO_SHARE_MIN == 0.6


def test_interaction_below_the_resolution_stops_the_joint_runs() -> None:
    assert not cg.interaction_verdict([0.0, 0.1, 0.49])["joint_runs_justified"]
    assert cg.interaction_verdict([0.0, 0.5])["joint_runs_justified"]
    assert not cg.interaction_verdict([])["joint_runs_justified"]


def _verdict(gains, spread=0.1):
    return cg.coupling_gain_verdict(gains, spread, [90.0] * len(gains))


def test_a_clear_gain_is_demonstrated() -> None:
    v = _verdict([2.0] * 25)
    assert v["demonstrated"] and v["gain_relative_to_line_co2_median"] == pytest.approx(2.0 / 90.0)


def test_each_criterion_can_fail_alone() -> None:
    assert not _verdict([0.9] * 25)["criterion_i_median"]
    assert not _verdict([2.0] * 25, spread=0.8)["criterion_ii_seed_spread"]  # 2.0 < 3 * 0.8
    mostly_small = [3.0] * 10 + [0.1] * 15  # median 0.1, share 0.4
    assert not _verdict(mostly_small)["demonstrated"]
    share_low = [1.5] * 12 + [0.1] * 13  # median 0.1 fails (i); craft (iii) only:
    assert _verdict(share_low)["share_above_per_scenario_threshold"] == pytest.approx(0.48)


def test_without_a_seed_spread_nothing_is_demonstrated() -> None:
    v = cg.coupling_gain_verdict([5.0] * 25, None, [90.0] * 25)
    assert not v["criterion_ii_seed_spread"] and not v["demonstrated"]


def test_input_shapes_are_checked() -> None:
    with pytest.raises(ValueError):
        cg.coupling_gain_verdict([], 0.1, [])
    with pytest.raises(ValueError):
        cg.coupling_gain_verdict([1.0], 0.1, [1.0, 2.0])
