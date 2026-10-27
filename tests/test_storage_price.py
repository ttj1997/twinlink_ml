"""Tests of the storage-price study (``twinlink_ml.storage_price``).

No test needs the price file in ``local/``: the prices are synthetic, plant and clock come from the
committed config.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from twinlink_ml.heatpump.storage import solve_foresight
from twinlink_ml.storage_price import (
    MPC,
    Action,
    Hindsight,
    NoPump,
    build_market,
    build_plant,
    known_price_forecast,
    load_storage_price_config,
    month_hours,
    project,
    r0_rule,
    r1_rule,
    run_period,
    solve_window,
    year_clock,
)
from twinlink_ml.storage_price.controllers import pump_only, r1_values
from twinlink_ml.storage_price.env import StoragePriceEnv, action_to_duties
from twinlink_ml.storage_price.prices import PriceYear


def _prices(seed: int = 0) -> PriceYear:
    clock = year_clock(2024)
    rng = np.random.default_rng(seed)
    h = clock.hour
    spot = (
        0.08
        + 0.04 * np.cos(2 * np.pi * (h - 19) / 24)
        - 0.03 * np.exp(-((h - 13) ** 2) / 8.0)
        + 0.02 * rng.standard_normal(clock.n_hours)
    )
    return PriceYear(spot_eur_per_kwh=spot, clock=clock, raw_sha256="synthetic", license_info="")


@pytest.fixture(scope="module")
def case():  # type: ignore[no-untyped-def]
    cfg = load_storage_price_config()
    prices = _prices()
    plant = build_plant(cfg, prices.clock)
    market = build_market(cfg, prices, cop=plant.cop, horizon_h=36)
    return cfg, prices, plant, market


def test_clock_2024() -> None:
    clock = year_clock(2024)
    assert clock.n_hours == 8784
    assert clock.weekday[0] == 0  # 2024-01-01 was a Monday
    assert np.sum(clock.month == 2) == 29 * 24
    assert clock.hour[25] == 1 and clock.day[25] == 1


def test_known_forecast_is_true_until_midnight_then_today() -> None:
    clock = year_clock(2024)
    price = np.arange(clock.n_hours, dtype=float)
    f = known_price_forecast(price, clock, 36)
    t = 24 * 10 + 15  # 15:00 of day 10
    assert np.array_equal(f[t, :9], price[t : t + 9])  # 15:00 .. 23:00 known
    # 00:00 .. 23:00 of day 11 and 00:00 .. 02:00 of day 12: same hour of day 10
    assert np.array_equal(f[t, 9:33], price[24 * 10 : 24 * 11])
    assert np.array_equal(f[t, 33:36], price[24 * 10 : 24 * 10 + 3])
    assert np.array_equal(f[24 * 10, :24], price[24 * 10 : 24 * 11])


def test_r1_uses_no_price_after_midnight(case) -> None:  # type: ignore[no-untyped-def]
    cfg, prices, plant, _ = case
    changed = dataclasses.replace(prices, spot_eur_per_kwh=prices.spot_eur_per_kwh.copy())
    changed.spot_eur_per_kwh[24 * 41 :] += 5.0  # every price from day 41 on
    m0 = build_market(cfg, prices, cop=plant.cop, horizon_h=36)
    m1 = build_market(cfg, changed, cop=plant.cop, horizon_h=36)
    day40 = slice(24 * 40, 24 * 41)
    assert np.array_equal(m0.known[day40], m1.known[day40])
    v0 = r1_values(m0, 0.3, 24, 1e-6)
    v1 = r1_values(m1, 0.3, 24, 1e-6)
    assert np.array_equal(v0[day40], v1[day40])
    assert not np.array_equal(m0.known[24 * 41], m1.known[24 * 41])


def test_pieces_at_zero_are_the_direct_target(case) -> None:  # type: ignore[no-untyped-def]
    _, _, plant, _ = case
    for t in (0, 100, 24 * 5 + 3, 24 * 200 + 12):
        assert plant.utility_kw(t, 0.0, 0.0, 0.0) == pytest.approx(plant.direct_kw[t])
    # weekday 500 kW above the waste-heat range, weekend 0.6 x 500 (main-case placeholders)
    assert plant.direct_kw[12] == pytest.approx(500.0)
    assert plant.direct_kw[24 * 5 + 12] == pytest.approx(300.0)
    assert plant.cop == pytest.approx(0.5 * 363.15 / 60.0)


def test_window_cyclic_equals_befund_101_foresight(case) -> None:  # type: ignore[no-untyped-def]
    _, _, plant, market = case
    rows = np.arange(24 * 3, 24 * 5)
    pc = plant.pieces
    sub = dataclasses.replace(pc, offset_kw=pc.offset_kw[rows])
    el = market.electricity[rows]
    ref = solve_foresight(
        sub,
        q_max_kw=plant.size_kw,
        capacity_kwh=plant.capacity_kwh,
        loss_per_h=plant.loss_per_h,
        dt_h=1.0,
        heat_value=market.heat_value,
        electricity=el,
    )
    big = 1e7
    mine = solve_window(
        sub.offset_kw,
        pc.alpha,
        pc.beta,
        pc.gamma,
        electricity=el,
        heat_value=market.heat_value,
        cop=plant.cop,
        q_max_kw=plant.size_kw,
        charge_max_kw=big,
        discharge_max_kw=big,
        capacity_kwh=plant.capacity_kwh,
        keep=plant.keep,
        dt_h=1.0,
        cyclic=True,
    )
    assert mine.objective == pytest.approx(ref.objective, rel=1e-9)


@pytest.mark.parametrize("kappa", [0.0, 0.5, 0.85, 1.0])
def test_step_rule_equals_one_step_programme(case, kappa: float) -> None:  # type: ignore[no-untyped-def]
    _, _, plant, market = case
    rule = r0_rule(plant, market, kappa)
    hours = np.arange(24 * 7, 24 * 9)
    rule.start(hours)
    pc = plant.pieces
    v = kappa * market.heat_value
    rng = np.random.default_rng(1)
    for k in range(0, len(hours), 5):
        s = float(rng.uniform(0.0, plant.capacity_kwh))
        a = rule.decide(k, s)
        t = int(hours[k])

        def f(act: Action, t: int = t) -> float:
            u = plant.utility_kw(t, act.heat_kw, act.charge_kw, act.discharge_kw)
            return (
                market.heat_value * u
                + market.electricity[t] * act.heat_kw / plant.cop
                - v * (act.charge_kw - act.discharge_kw)
            )

        lp = solve_window(
            pc.offset_kw[[t]],
            pc.alpha,
            pc.beta,
            pc.gamma,
            electricity=market.electricity[[t]],
            heat_value=market.heat_value,
            cop=plant.cop,
            q_max_kw=plant.size_kw,
            charge_max_kw=plant.charge_max_kw,
            discharge_max_kw=plant.discharge_max_kw,
            capacity_kwh=plant.capacity_kwh,
            keep=plant.keep,
            dt_h=1.0,
            initial_kwh=s,
            terminal_value=v,
        )
        best = Action(float(lp.heat_kw[0]), float(lp.charge_kw[0]), float(lp.discharge_kw[0]))
        assert f(a) == pytest.approx(f(best), abs=1e-6)
        assert project(plant, a, s).cut_content_kw <= 1e-6
        assert a.charge_kw <= plant.charge_max_kw + 1e-9
        assert a.discharge_kw <= plant.discharge_max_kw + 1e-9


def test_hindsight_is_the_optimum_of_the_protocol(case) -> None:  # type: ignore[no-untyped-def]
    cfg, prices, plant, market = case
    hours = np.arange(24 * 14, 24 * 21)
    tol = cfg.controllers.violation_tolerance_relative
    eps = cfg.controllers.stored_value_epsilon
    results = {
        c.name: run_period(c, plant, market, hours, month=0, tolerance=tol)
        for c in (
            NoPump(plant, market),
            pump_only(plant, market, eps),
            r0_rule(plant, market, 0.85),
            r1_rule(plant, market, 0.4, 24, eps),
            MPC(plant, market, horizon_h=24, information="perfect"),
            MPC(plant, market, horizon_h=24, information="known"),
            Hindsight(plant, market),
        )
    }
    star = results["P*"].cost_eur
    for name, r in results.items():
        assert r.cost_eur >= star - 1e-6 * abs(star), name
        assert r.violations_power == 0 and r.violations_content == 0, name
        assert abs(r.balance_residual_kwh) < 1e-6, name
    assert results["P0"].cost_eur < results["N0"].cost_eur


def test_projection_counts_and_clips(case) -> None:  # type: ignore[no-untyped-def]
    _, _, plant, _ = case
    pr = project(plant, Action(900.0, 400.0, 0.0), 0.0)
    assert pr.action.heat_kw == plant.size_kw
    assert pr.action.charge_kw == plant.charge_max_kw
    assert pr.cut_power_kw == pytest.approx(100.0 + 100.0)
    full = plant.capacity_kwh / plant.keep
    pr = project(plant, Action(0.0, 100.0, 0.0), full)
    assert pr.action.charge_kw == pytest.approx(0.0, abs=1e-9)
    assert pr.cut_content_kw == pytest.approx(100.0)
    pr = project(plant, Action(0.0, 0.0, 200.0), 50.0)
    assert pr.action.discharge_kw == pytest.approx(50.0 * plant.keep)


def test_env_reproduces_the_protocol_cost(case) -> None:  # type: ignore[no-untyped-def]
    cfg, prices, plant, market = case
    env = StoragePriceEnv(
        plant, market, prices.clock, cfg.training_months, episode_h=48, reward_scale=1.0
    )
    env.reset(seed=3)
    env.t, env.k, env.s = int(month_hours(prices.clock, 3)[0]), 0, 0.0
    rng = np.random.default_rng(5)
    acts = rng.uniform(-1, 1, size=(48, 2)).astype(np.float32)
    total = 0.0
    for a in acts:
        _, r, term, trunc, info = env.step(a)
        total += info["cost_eur"]
        assert not term
    assert trunc

    class Replay(NoPump):
        name = "replay"

        def decide(self, k: int, stored_kwh: float) -> Action:  # noqa: ARG002
            return action_to_duties(plant, acts[k])

    hours = month_hours(prices.clock, 3)[:48]
    res = run_period(
        Replay(plant, market),
        plant,
        market,
        hours,
        month=3,
        tolerance=cfg.controllers.violation_tolerance_relative,
    )
    assert res.cost_operating_eur == pytest.approx(total, rel=1e-12)


def test_env_episodes_stay_in_training_months(case) -> None:  # type: ignore[no-untyped-def]
    cfg, prices, plant, market = case
    env = StoragePriceEnv(
        plant, market, prices.clock, cfg.training_months, episode_h=168, reward_scale=30.0
    )
    test = set(np.nonzero(np.isin(prices.clock.month, cfg.test_months))[0].tolist())
    for start in env.starts:
        assert int(start) not in test
        assert int(start) + 167 not in test
        assert prices.clock.month[start] == prices.clock.month[start + 167]


def test_variants_override_the_main_case() -> None:
    main = load_storage_price_config()
    var = load_storage_price_config(variant="capacity_4000")
    assert var.storage.capacity_kwh == 4000.0 and main.storage.capacity_kwh == 2000.0
    assert var.fingerprint != main.fingerprint
    assert set(main.test_months).isdisjoint(main.training_months)
    assert len(main.training_months) == 8 and len(main.test_months) == 4
    with pytest.raises(KeyError):
        load_storage_price_config(variant="nope")
