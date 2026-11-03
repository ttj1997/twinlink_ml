"""Tests of the campaign-network study (``twinlink_ml.campaign_network``)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from twinlink_ml.campaign_network import (
    build_consumers,
    campaign_schedule,
    cascade_stages,
    key_figures,
    load_campaign_network_config,
    plant_schedules,
    run_k0,
    run_k1,
    run_k2,
    run_k3,
    stage_duty,
    variant_names,
)
from twinlink_ml.campaign_network.config import from_raw
from twinlink_ml.campaign_network.profiles import ColdStream, Consumers

TOL_KWH = 1e-3


@pytest.fixture(scope="module")
def cfg():
    return load_campaign_network_config()


def _with(cfg, **changes: object):
    raw = copy.deepcopy(cfg.raw)
    for path, value in changes.items():
        section, field = path.split("__")
        raw[section][field] = value
    return from_raw(raw, "test")


def test_variants_load_and_differ(cfg) -> None:
    names = variant_names()
    assert names == ["approach_5", "supply_70"]
    prints = {cfg.fingerprint}
    for name in names:
        prints.add(load_campaign_network_config(variant=name).fingerprint)
    assert len(prints) == 3
    assert load_campaign_network_config(variant="approach_5").approach_k == 5.0
    assert load_campaign_network_config(variant="supply_70").source.supply_celsius == 70.0


def test_stage_duty_respects_approach_floor_and_balance() -> None:
    rng = np.random.default_rng(0)
    for _ in range(2000):
        t_w = rng.uniform(55, 90)
        cp_w = rng.uniform(0.1, 40)
        t_in = rng.uniform(-10, 50)
        t_out = t_in + rng.uniform(0, 60)
        cp_c = rng.uniform(0.1, 40)
        dt = rng.choice([5.0, 10.0])
        floor = 50.0
        q = float(stage_duty(t_w, cp_w, t_in, t_out, cp_c, dt, floor))
        assert q >= 0.0
        if q == 0.0:
            continue
        w_out = t_w - q / cp_w
        c_out = t_in + q / cp_c
        assert c_out <= t_out + 1e-9
        assert t_w - c_out >= dt - 1e-9  # hot end
        assert w_out - t_in >= dt - 1e-9  # cold end
        assert w_out >= floor - 1e-9


def test_schedule_availability_and_durations() -> None:
    rng = np.random.default_rng(1)
    assert campaign_schedule(1.0, (1.0, 14.0), 100, 50, rng).all()
    for a in (0.8, 0.57, 0.4):
        s = campaign_schedule(a, (1.0, 14.0), 200 * 8760, 8760, np.random.default_rng(2))
        assert abs(s.mean() - a) < 0.01
        # inner on-runs lie in 1-14 days
        edges = np.flatnonzero(np.diff(s.astype(np.int8)))
        runs = np.diff(edges)
        on_runs = runs[s[edges[:-1] + 1]]
        assert on_runs.min() >= 24 and on_runs.max() <= 336


def test_joint_mode_shares_plant_a(cfg) -> None:
    ind = plant_schedules(cfg, 0.57, "independent", 3)
    joint = plant_schedules(cfg, 0.57, "joint", 3)
    assert np.array_equal(ind["A"], joint["A"])
    assert np.array_equal(joint["A"], joint["B"]) and np.array_equal(joint["A"], joint["C"])
    assert not np.array_equal(ind["A"], ind["B"])


def test_cascade_order(cfg) -> None:
    c = build_consumers(cfg, plant_schedules(cfg, 1.0, "independent", 0))
    assert [s.name for s in cascade_stages(c)] == ["process_air", "water", "hall"]


@pytest.mark.parametrize("variant", ["main", "approach_5", "supply_70"])
@pytest.mark.parametrize(("a", "mode"), [(0.57, "independent"), (0.4, "joint"), (1.0, "joint")])
def test_k3_ge_k2_ge_k1_ge_k0_every_draw(variant, a, mode) -> None:
    cfg = load_campaign_network_config(variant=variant)
    cfg = _with(cfg, campaigns__hours=24 * 42)
    for draw in range(2):
        c = build_consumers(cfg, plant_schedules(cfg, a, mode, draw))
        for p in (400.0, 1200.0):
            k0 = key_figures(run_k0(c, p, cfg), c, p, cfg)
            k1 = key_figures(run_k1(c, p, cfg), c, p, cfg)
            k30 = key_figures(run_k3(c, p, 0.0, cfg), c, p, cfg)
            for key in ("use_kwh", "co2_net_kg"):
                assert k1[key] >= k0[key] - TOL_KWH
                assert k30[key] >= k1[key] - TOL_KWH
            for s in (2.0, 8.0, 24.0):
                k2 = key_figures(run_k2(c, p, s, cfg), c, p, cfg)
                k3 = key_figures(run_k3(c, p, s, cfg), c, p, cfg)
                assert k3["lp_status"] == 0
                for key in ("use_kwh", "co2_net_kg"):
                    assert k2[key] >= k1[key] - TOL_KWH
                    assert k3[key] >= k2[key] - TOL_KWH
                assert k3["use_kwh"] <= k3["source_kwh"] + TOL_KWH


def test_k2_without_storage_equals_k1(cfg) -> None:
    c = build_consumers(cfg, plant_schedules(cfg, 0.57, "independent", 0))
    r1 = run_k1(c, 800.0, cfg)
    r2 = run_k2(c, 800.0, 0.0, cfg)
    for name in r1.duty_kw:
        np.testing.assert_allclose(r2.duty_kw[name], r1.duty_kw[name], atol=1e-9)


def test_v_n_zero_at_full_availability_when_source_within_own_demand(cfg) -> None:
    c = build_consumers(cfg, plant_schedules(cfg, 1.0, "independent", 0))
    p = 100.0
    r0 = run_k0(c, p, cfg)
    # premise: plant A alone takes the whole source in every hour
    np.testing.assert_allclose(r0.duty_kw["A"], p, rtol=0, atol=1e-9)
    k0 = key_figures(r0, c, p, cfg)
    k1 = key_figures(run_k1(c, p, cfg), c, p, cfg)
    assert (k1["use_kwh"] - k0["use_kwh"]) / k0["source_kwh"] == pytest.approx(0.0, abs=1e-12)


def _toy(n_hours: int, cp_by_hour: list[float]) -> Consumers:
    ones = np.ones(n_hours)
    stream = ColdStream(
        name="A",
        t_in=15.0 * ones,
        t_out=60.0 * ones,
        cp=np.asarray(cp_by_hour, dtype=np.float64),
        gas_efficiency=0.9,
        stage_celsius=150.0,
        group="process_air",
    )
    return Consumers(streams=(stream,), ambient=15.0 * ones)


@pytest.mark.parametrize("loss", [0.0, 0.1])
def test_storage_by_hand(cfg, loss) -> None:
    # hour 0: consumer off, hour 1: demand far above the source; the source heat of hour 0 is
    # stored and delivered in hour 1 (after the loss of one hour)
    cfg = _with(cfg, storage__loss_per_h=loss)
    p = 300.0
    c = _toy(2, [0.0, 40.0])
    expected = p + (1.0 - loss) * p
    k1 = key_figures(run_k1(c, p, cfg), c, p, cfg)
    k2 = key_figures(run_k2(c, p, 24.0, cfg), c, p, cfg)
    k3 = key_figures(run_k3(c, p, 24.0, cfg), c, p, cfg)
    assert k1["use_kwh"] == pytest.approx(p)
    assert k2["use_kwh"] == pytest.approx(expected, abs=1e-6)
    assert k3["use_kwh"] == pytest.approx(expected, abs=1e-6)


def test_k3_without_storage_by_hand(cfg) -> None:
    # source 80 -> 50 degC, 300 kW (10 kW/K); process water 15 -> 60 degC with 8 kW/K (360 kW):
    # the floor limits the duty to the source heat, 300 kW
    c = _toy(1, [8.0])
    k3 = key_figures(run_k3(c, 300.0, 0.0, cfg), c, 300.0, cfg)
    k1 = key_figures(run_k1(c, 300.0, cfg), c, 300.0, cfg)
    assert k3["use_kwh"] == pytest.approx(300.0)
    assert k1["use_kwh"] == pytest.approx(300.0)
    # 2 kW/K (90 kW): the demand limits both
    c = _toy(1, [2.0])
    assert key_figures(run_k3(c, 300.0, 0.0, cfg), c, 300.0, cfg)["use_kwh"] == pytest.approx(90.0)
    assert key_figures(run_k1(c, 300.0, cfg), c, 300.0, cfg)["use_kwh"] == pytest.approx(90.0)


def test_key_figures_pump_and_co2(cfg) -> None:
    c = _toy(1, [2.0])
    k = key_figures(run_k1(c, 300.0, cfg), c, 300.0, cfg)
    assert k["gas_kwh"] == pytest.approx(90.0 / 0.9)
    assert k["pump_electricity_kwh"] == pytest.approx(0.015 * 90.0)
    assert k["co2_net_kg"] == pytest.approx(0.201 * 100.0 - 0.366 * 0.015 * 90.0)
