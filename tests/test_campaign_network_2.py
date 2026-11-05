"""Tests of the campaign-network follow-up study (K0b, K1d, rolling planning P1 / P2)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from twinlink_ml.campaign_network import (
    build_consumers,
    key_figures,
    load_campaign_network_config,
    plant_schedules,
    run_k0,
    run_k1,
    run_k2,
    run_k3,
)
from twinlink_ml.campaign_network.config import from_raw
from twinlink_ml.campaign_network.dynamic import fixed_shares, run_k0b, run_k1d
from twinlink_ml.campaign_network.planning import (
    PlanForecast,
    TrueForecast,
    cut_data,
    planned_transitions,
    rolling_plan,
    use_of,
)

TOL = 1e-3


def _cfg(hours: int = 24 * 28, water: float | None = None, hall: float | None = None):
    cfg = load_campaign_network_config()
    raw = copy.deepcopy(cfg.raw)
    raw["campaigns"]["hours"] = hours
    if water is not None:
        raw["low_temperature"]["water"]["power_kw"] = water
    if hall is not None:
        raw["low_temperature"]["hall"]["peak_kw"] = hall
    return from_raw(raw, "test")


def _use(res) -> float:
    return float(sum(v.sum() for v in res.duty_kw.values()))


def test_fixed_shares_follow_air_flow() -> None:
    shares = fixed_shares(load_campaign_network_config())
    assert shares == pytest.approx({"A": 20 / 45, "B": 15 / 45, "C": 10 / 45})


def test_k0b_equals_air_stage_of_k1_when_all_run_without_low_temperature_sinks() -> None:
    cfg = _cfg(water=0.0, hall=0.0)
    c = build_consumers(cfg, plant_schedules(cfg, 1.0, "independent", 0))
    for p in (400.0, 1200.0):
        r0b, r1 = run_k0b(c, p, cfg), run_k1(c, p, cfg)
        for name in ("A", "B", "C"):
            np.testing.assert_allclose(r0b.duty_kw[name], r1.duty_kw[name], rtol=1e-12)


def test_reduced_cut_model_equals_k3_without_storage() -> None:
    cfg = _cfg()
    on = plant_schedules(cfg, 0.57, "independent", 1)
    c = build_consumers(cfg, on)
    cuts = cut_data(cfg)
    for p in (400.0, 800.0, 1200.0):
        reduced = use_of(cuts.f, cuts.demand_below(on, slice(None)), p).sum()
        assert reduced == pytest.approx(_use(run_k3(c, p, 0.0, cfg)), rel=1e-9)


def test_cut_data_refuses_unequal_efficiencies() -> None:
    cfg = load_campaign_network_config()
    raw = copy.deepcopy(cfg.raw)
    raw["low_temperature"]["hall"]["gas_efficiency"] = 0.8
    with pytest.raises(ValueError):
        cut_data(from_raw(raw, "test"))


@pytest.mark.parametrize("every", [1, 24])
def test_p1_with_horizon_to_the_end_reproduces_k3(every) -> None:
    cfg = _cfg(hours=24 * 21)
    on = plant_schedules(cfg, 0.57, "independent", 0)
    c = build_consumers(cfg, on)
    cuts = cut_data(cfg)
    for s in (8.0, 24.0):
        k3 = _use(run_k3(c, 800.0, s, cfg))
        r = rolling_plan(cfg, cuts, on, TrueForecast(on), 800.0, s, 24 * 21, every)
        assert r.use_kw.sum() == pytest.approx(k3, rel=1e-7)
        assert r.lp_failures == 0


def test_p1_plans_once_a_day_with_exact_plan() -> None:
    cfg = _cfg(hours=24 * 10)
    on = plant_schedules(cfg, 0.57, "independent", 0)
    r = rolling_plan(cfg, cut_data(cfg), on, TrueForecast(on), 800.0, 8.0, 168, 24)
    assert r.plans == 10


def test_plan_shifts_and_naive_correction() -> None:
    cfg = load_campaign_network_config()
    on = plant_schedules(cfg, 0.57, "independent", 2)
    planned = planned_transitions(cfg, 0.57, "independent", 2)
    for name, p in planned.items():
        tau = np.flatnonzero(on[name][1:] != on[name][:-1]) + 1
        assert p.size == tau.size
        assert np.all(np.diff(p) >= 1)
        # shifted by at most 24 h, except where the order had to be restored
        assert np.mean(np.abs(p - tau) <= 24) > 0.95
    joint = planned_transitions(cfg, 0.57, "joint", 2)
    assert np.array_equal(joint["A"], joint["B"]) and np.array_equal(joint["A"], joint["C"])
    fc = PlanForecast(on, planned)
    name = "A"
    tau = np.flatnonzero(on[name][1:] != on[name][:-1]) + 1
    for t in (0, 500, 3000, 8000):
        states = fc.states(t, 168)
        # the hour t itself is observed: the forecast equals the truth there
        assert bool(states[name][0]) == bool(on[name][t])
    # an overdue planned change is expected in the next hour
    p = planned[name]
    k = int(np.flatnonzero(p < tau)[0])  # planned earlier than it happens
    t = int(p[k])  # at the planned hour the change has not happened yet
    if t < tau[k]:
        s = fc.states(t, 3)[name]
        assert s[0] == on[name][t] and s[1] != on[name][t]


@pytest.mark.parametrize(("a", "mode"), [(0.57, "independent"), (0.4, "joint")])
@pytest.mark.parametrize("water", [0.0, 200.0])
def test_k3_bounds_every_configuration_in_every_draw(a, mode, water) -> None:
    cfg = _cfg(hours=24 * 42, water=water)
    cuts = cut_data(cfg)
    for draw in range(2):
        on = plant_schedules(cfg, a, mode, draw)
        c = build_consumers(cfg, on)
        planned = planned_transitions(cfg, a, mode, draw)
        for p in (400.0, 800.0, 1200.0):
            base = {
                "K0": run_k0(c, p, cfg),
                "K0b": run_k0b(c, p, cfg),
                "K1": run_k1(c, p, cfg),
                "K1d0": run_k1d(c, p, 0.0, cfg),
            }
            k30 = key_figures(run_k3(c, p, 0.0, cfg), c, p, cfg)
            for r in base.values():
                f = key_figures(r, c, p, cfg)
                for key in ("use_kwh", "co2_net_kg"):
                    assert f[key] <= k30[key] + TOL
            for s in (8.0, 24.0):
                k3 = _use(run_k3(c, p, s, cfg))
                assert _use(run_k1d(c, p, s, cfg)) <= k3 + TOL
                assert _use(run_k2(c, p, s, cfg, rule="k2b")) <= k3 + TOL
                r1 = rolling_plan(cfg, cuts, on, TrueForecast(on), p, s, 168, 24)
                r2 = rolling_plan(cfg, cuts, on, PlanForecast(on, planned), p, s, 168, 24)
                assert r1.use_kw.sum() <= k3 + TOL
                assert r2.use_kw.sum() <= k3 + TOL
                assert r1.lp_failures == 0 and r2.lp_failures == 0


def test_k1d_storage_respects_balance() -> None:
    cfg = _cfg(hours=24 * 21)
    c = build_consumers(cfg, plant_schedules(cfg, 0.57, "independent", 0))
    r = run_k1d(c, 800.0, 8.0, cfg)
    keep = 1.0 - cfg.storage.loss_per_h
    e = r.content_kwh
    np.testing.assert_allclose(e[1:], keep * e[:-1] + r.charge_kw - r.discharge_kw, atol=1e-9)
    assert e.min() >= -1e-9 and e.max() <= 8.0 * 800.0 + 1e-9
