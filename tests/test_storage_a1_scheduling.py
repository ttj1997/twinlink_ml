"""Tests of ``storage_a1.scheduling``: replay, rule, DP optimum (brute-force check), accounting."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from twinlink_ml.storage_a1.config import StorageConfig, StorageSystemConfig
from twinlink_ml.storage_a1.roles import PlantDay
from twinlink_ml.storage_a1.scheduling import (
    ShiftProblem,
    build_problem,
    co2_kg,
    corrected_steam_kwh,
    dp_policy,
    replay,
    rule_policy,
    simulate,
)


def _problem(
    n: int = 12, q_dump: float = 0.0, volume: float = 8.0, t_max: float = 80.0
) -> ShiftProblem:
    cfg = StorageSystemConfig(
        storage=StorageConfig(volume_m3=volume, max_temperature_celsius=t_max)
    )
    ramp = 300.0 + 150.0 * np.sin(np.linspace(0.0, 3.0, n))
    return ShiftProblem(
        cfg=cfg,
        dt_s=900.0,
        t_ambient=np.full(n, 5.0),
        v_storage_loop=np.full(n, 37.0),
        air_flow=np.full(n, 33000.0),
        q_air_kw=np.full(n, 800.0),
        q_in_base_kw=ramp,
        q_dump_kw=np.full(n, q_dump),
        v2_logged=np.full(n, 18.5),
        t_s0=52.0,
        t_min=40.0,
        t_max=t_max,
        v2_min=5.0,
        v2_max=28.0,
    )


def test_replay_conserves_energy() -> None:
    p = _problem()
    r = replay(p)
    loss = p.chain.storage.loss_kw(r.t_s[:-1])
    stored = r.capacity_kj_per_k * (r.t_s[-1] - r.t_s[0])
    assert stored == pytest.approx(np.sum(r.q_in_kw - r.q_out_kw - loss) * p.dt_s)
    assert r.steam_kwh == pytest.approx(np.sum(p.q_air_kw - r.q_out_kw) * p.dt_s / 3600.0)


def test_dp_matches_brute_force_on_a_tiny_problem() -> None:
    p = _problem(n=5)
    ref = -1.0e3  # no terminal condition
    best = np.inf
    for seq in itertools.product((p.v2_min, p.v2_max), repeat=p.n_steps):
        it = iter(seq)
        pol = lambda _k, _t, it=it: (next(it), 0.0)  # noqa: E731
        res = simulate(p, pol)
        if res.feasible:
            best = min(best, res.steam_kwh)
    dp = simulate(p, dp_policy(p, allow_dump=False, t_end_reference=ref, grid_k=0.05, v2_levels=2))
    assert dp.feasible
    assert dp.steam_kwh == pytest.approx(best, abs=0.5)


def test_optimum_is_not_worse_than_rule_and_respects_terminal_condition() -> None:
    p = _problem(n=24, q_dump=200.0)
    r0 = replay(p)
    r1 = simulate(p, rule_policy(p, use_dump=True))
    dp = simulate(p, dp_policy(p, allow_dump=True, t_end_reference=r0.t_end, grid_k=0.25))
    assert r1.feasible and dp.feasible
    e1 = corrected_steam_kwh(r1, r0.t_end)
    ed = corrected_steam_kwh(dp, r0.t_end)
    assert ed <= e1 + 1.0
    assert dp.t_end >= r0.t_end - 0.5
    assert np.all(dp.t_s <= p.t_max + 1e-6)


def test_dump_heat_is_only_taken_below_t_max_and_helps() -> None:
    p = _problem(n=24, q_dump=600.0, t_max=70.0)
    with_dump = simulate(p, rule_policy(p, use_dump=True))
    without = simulate(p, rule_policy(p, use_dump=False))
    assert np.max(with_dump.t_s) <= p.t_max + 1e-6
    assert with_dump.dump_used_kwh > 0.0
    assert with_dump.steam_kwh < without.steam_kwh


def test_no_dump_available_means_dump_flag_changes_nothing() -> None:
    p = _problem(n=12, q_dump=0.0)
    a = simulate(p, rule_policy(p, use_dump=True))
    b = simulate(p, rule_policy(p, use_dump=False))
    assert a.steam_kwh == pytest.approx(b.steam_kwh)


def test_accounting_helpers() -> None:
    p = _problem()
    r = replay(p)
    assert corrected_steam_kwh(r, r.t_end) == pytest.approx(r.steam_kwh)
    assert corrected_steam_kwh(r, r.t_end - 1.0) < r.steam_kwh  # ended hotter than reference
    assert co2_kg(100.0, p.cfg) == pytest.approx(100.0 / 0.9 * 0.201)


def test_build_problem_shapes_and_nonnegative_base() -> None:
    n = 1441
    t = np.arange(n) * 60.0
    roles = {
        "storage_temperature": 55.0 - 3.0 * t / t[-1],
        "storage_loop_flow": np.full(n, 37.0),
        "preheater_loop_flow": np.full(n, 18.5),
        "preheater_supply_temperature": np.full(n, 45.0),
        "preheater_return_temperature": np.full(n, 32.0),
        "supply_air_temperature": np.full(n, 84.0),
        "ambient_temperature": np.full(n, 5.0),
        "process_air_flow": np.full(n, 33000.0),
        "steam_valve_position": np.full(n, 45.0),
        "process_fan_power": np.full(n, 115.0),
    }
    day = PlantDay("d", t, roles)
    cfg = StorageSystemConfig()
    p = build_problem(day, cfg, reference_volume_m3=10.0, q_dump_kw=np.full(288, 100.0))
    assert p.n_steps == 288 and p.dt_s == 300.0
    assert np.all(p.q_in_base_kw >= 0.0)
    assert p.v2_max == pytest.approx(1.5 * 18.5)
    warm = build_problem(
        day, cfg, reference_volume_m3=10.0, q_dump_kw=np.zeros(288), ambient_shift_k=5.0
    )
    assert np.all(warm.q_air_kw < p.q_air_kw)  # warmer intake air, less heating needed
    with pytest.raises(ValueError):
        build_problem(day, cfg, reference_volume_m3=10.0, q_dump_kw=np.zeros(10))
