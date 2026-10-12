"""B1.1: equal-cost excitation plans B0-B4."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from twinlink_ml.simulation import ExcitationConfig, MixedExcitationDesign
from twinlink_ml.simulation.excitation import DiurnalDisturbanceDesign
from twinlink_ml.simulation.excitation_plans import (
    PLAN_NAMES,
    MeasuredTimeConstants,
    MultiLevelPrbsPlan,
    PlanTimings,
    SpaceFillingHoldPlan,
    check_equal_cost,
    make_plan,
)

DT = 25.0
DURATION = 28_800.0


@pytest.fixture
def cfg(config):
    return dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, sample_time_s=DT)
    )


def _plan(name, cfg, **kwargs: object):
    return make_plan(name, cfg, sample_time_s=DT, run_duration_s=DURATION, **kwargs)


def test_timings_follow_the_measured_time_constants():
    timings = PlanTimings.from_time_constants(
        MeasuredTimeConstants(), sample_time_s=DT, run_duration_s=DURATION
    )
    assert timings.fast_clock_s == 75.0  # 2 * 30.23 s rounded up to a multiple of dt
    assert timings.slow_clock_s == 900.0  # 0.75 * 1194.88 s rounded to a multiple of dt
    assert timings.n_holds == 12 and timings.hold_s == 2400.0
    assert timings.hold_s >= 2.0 * 1194.88
    assert timings.band_low_hz == pytest.approx(1.0 / (2 * np.pi * 1194.88))
    assert timings.band_high_hz == pytest.approx(1.0 / (2 * np.pi * 30.23))


def test_all_plans_are_equal_cost(cfg):
    schedules = {name: _plan(name, cfg).build(4, seed=5) for name in PLAN_NAMES}
    assert check_equal_cost(schedules, cfg, reference="b1") == []
    assert {s.actions.shape for s in schedules.values()} == {(4, 1152, 4)}


def test_equal_cost_check_detects_violations(cfg):
    schedules = {name: _plan(name, cfg).build(2, seed=5) for name in ("b1", "b2")}
    bound = cfg.actions.bounds[0]
    bad = dataclasses.replace(schedules["b2"], actions=schedules["b2"].actions + 10.0 * bound.span)
    assert check_equal_cost({"b1": schedules["b1"], "b2": bad}, cfg, reference="b1")
    shifted = dataclasses.replace(schedules["b2"], disturbances=schedules["b2"].disturbances + 1.0)
    assert any(
        "disturbances" in v
        for v in check_equal_cost({"b1": schedules["b1"], "b2": shifted}, cfg, reference="b1")
    )


@pytest.mark.parametrize("name", PLAN_NAMES)
def test_plans_are_deterministic_and_seed_dependent(cfg, name):
    plan = _plan(name, cfg)
    first, again, other = plan.build(4, seed=1), plan.build(4, seed=1), plan.build(4, seed=2)
    assert np.array_equal(first.actions, again.actions)
    assert first.run_id == again.run_id
    assert not np.array_equal(first.disturbances, other.disturbances)
    assert all(run_id.startswith(f"{name}-") for run_id in first.run_id)


def test_b0_is_constant_within_runs_and_varies_between_runs(cfg):
    actions = _plan("b0", cfg).build(8, seed=1).actions
    assert np.all(np.ptp(actions, axis=1) < 1e-9)
    assert np.all(actions[:, 0, :].std(axis=0) > 0.0)


def test_b1_is_the_previous_design_without_constant_runs(cfg):
    plan = _plan("b1", cfg)
    schedule = plan.build(16, seed=4)
    reference = MixedExcitationDesign(
        ExcitationConfig(run_duration_s=DURATION, constant_run_fraction=0.0),
        cfg.actions,
        cfg.disturbances,
        sample_time_s=DT,
        geometry=cfg.geometry,
        material=cfg.material,
        operation=cfg.operation,
        disturbance_design=DiurnalDisturbanceDesign(solid_fraction_process="independent"),
        name="b1",
    ).build(16, seed=4)
    assert np.array_equal(schedule.actions, reference.actions)
    assert not any("constant" in kinds for kinds in schedule.signal_kinds)


def test_b2_has_four_levels_and_clocks_below_the_rate_limits(cfg):
    plan = _plan("b2", cfg)
    assert isinstance(plan, MultiLevelPrbsPlan)
    rng = np.random.default_rng(0)
    normalised, _ = plan._build_normalised_actions(3, plan.n_steps, rng, 0)
    for channel in range(4):
        assert len(np.unique(np.round(normalised[:, :, channel], 9))) == 4
    # a fast step of fast_weight * span within one fast clock stays inside every rate limit
    for bound in cfg.actions.bounds:
        step_rate = plan.fast_weight * bound.span / plan.timings.fast_clock_s
        assert step_rate <= bound.max_rate_per_s
    # the m-sequence period exceeds the run (no repetition within a run)
    register = plan._register_length(plan.n_steps, round(plan.timings.fast_clock_s / DT))
    assert (2**register - 1) * plan.timings.fast_clock_s >= DURATION


def test_b2_slow_component_switches_on_the_slow_clock(cfg):
    plan = _plan("b2", cfg)
    rng = np.random.default_rng(1)
    normalised, _ = plan._build_normalised_actions(1, plan.n_steps, rng, 0)
    signal = normalised[0, :, 0]
    block = round(plan.timings.slow_clock_s / DT)
    means = signal[: (len(signal) // block) * block].reshape(-1, block).mean(axis=1)
    assert means.max() - means.min() > 0.4


def test_b3_power_lies_inside_the_measured_band(cfg):
    plan = _plan("b3", cfg)
    schedule = plan.build(2, seed=3)
    low, high = plan.timings.band_low_hz, plan.timings.band_high_hz
    for channel, bound in enumerate(cfg.actions.bounds):
        signal = schedule.actions[0, :, channel]
        signal = signal - signal.mean()
        spectrum = np.abs(np.fft.rfft(signal)) ** 2
        frequencies = np.fft.rfftfreq(len(signal), d=DT)
        inside = spectrum[(frequencies >= 0.7 * low) & (frequencies <= 1.3 * high)].sum()
        assert inside / spectrum.sum() > 0.95
        peak_rate = np.abs(np.diff(schedule.actions[0, :, channel])).max() / DT
        assert peak_rate <= bound.max_rate_per_s * (1 + 1e-6)


def test_b3_channels_do_not_share_frequencies(cfg):
    plan = _plan("b3", cfg)
    grids = [plan._unit_signal(c, 4)[0] for c in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not set(np.round(grids[i], 12)) & set(np.round(grids[j], 12))


def test_b4_holds_and_nested_operating_points(cfg):
    plan = _plan("b4", cfg)
    assert isinstance(plan, SpaceFillingHoldPlan)
    n_holds = plan.timings.n_holds
    rng = np.random.default_rng(0)
    small, _ = plan._build_normalised_actions(4, plan.n_steps, rng, 7)
    large, _ = plan._build_normalised_actions(8, plan.n_steps, rng, 7)
    hold_steps = round(plan.timings.hold_s / DT)
    blocks = small[0].reshape(n_holds, hold_steps, 4)
    assert np.all(blocks.std(axis=1) == 0.0)
    # the set of operating points of run r does not depend on the number of runs
    for run in range(4):
        a = small[run, ::hold_steps]
        b = large[run, ::hold_steps]
        assert sorted(map(tuple, np.round(a, 12))) == sorted(map(tuple, np.round(b, 12)))
    points = small[:, ::hold_steps].reshape(-1, 4)
    assert points.min() < 0.1 and points.max() > 0.9


def test_unknown_plan_raises(cfg):
    with pytest.raises(KeyError):
        _plan("b9", cfg)
