"""Tests of ``twinlink_ml.storage_a1``: config, model, identifiability, calibration, balance, roles.

The plant days are replaced by a synthetic day generated from the model itself with KNOWN
parameters (volume, conductances, ``Q_in(t)``); the identifiability and calibration code must
recover what the generator put in -- and the bounds must contain the true volume.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.plant_data.loader import PlantTimeSeries
from twinlink_ml.simulation.config import PlaceholderConfigWarning
from twinlink_ml.storage_a1.balance import air_heating_kw, daily_balance
from twinlink_ml.storage_a1.calibration import (
    arx_free_run,
    error_metrics,
    fit_arx,
    fit_conductances,
    mean_q_in_kw,
    oracle_q_in_kw,
    persistence,
    predict_extraction,
    repair_glitches,
    require_different_days,
    rollout_storage,
)
from twinlink_ml.storage_a1.config import (
    CompressorHeatConfig,
    ExchangerConfig,
    SteamHeaterConfig,
    StorageConfig,
    StorageSystemConfig,
)
from twinlink_ml.storage_a1.identifiability import (
    implied_conductances,
    q_in_curve,
    smoothed_derivative_k_per_s,
    volume_bounds,
)
from twinlink_ml.storage_a1.model import (
    CounterflowExchanger,
    ExtractionChain,
    MixedNodeStorage,
    air_capacity_rate_kw_per_k,
    effectiveness_counterflow,
    min_approach_k,
    water_capacity_rate_kw_per_k,
)
from twinlink_ml.storage_a1.roles import (
    REQUIRED_ROLES,
    PlantDay,
    day_from_series,
    load_role_map,
)

REPO = Path(__file__).resolve().parents[1]
TRUE_VOLUME = 12.0


def _cfg(topology: str = "storage_then_exchanger", **kw: object) -> StorageSystemConfig:
    base = StorageSystemConfig(
        topology=topology,  # type: ignore[arg-type]
        storage=StorageConfig(volume_m3=TRUE_VOLUME),
        storage_exchanger=ExchangerConfig(ua_kw_per_k=15.0),
        preheater=ExchangerConfig(ua_kw_per_k=12.0),
        steam=SteamHeaterConfig(),
        compressor=CompressorHeatConfig(),
    )
    return dataclasses.replace(base, **kw)  # type: ignore[arg-type]


def _synthetic_day(cfg: StorageSystemConfig, label: str = "synthetic_1", seed: int = 0) -> PlantDay:
    n = 1441
    t = np.arange(n) * 60.0
    rng = np.random.default_rng(seed)
    q_in = 420.0 + 260.0 * np.sin(2 * np.pi * t / (7 * 3600.0) + seed) + 0 * t
    base = {
        "storage_temperature": np.full(n, 55.0),
        "storage_loop_flow": 37.0 + 0.5 * np.sin(t / 3000.0),
        "preheater_loop_flow": 18.5 + 1.0 * np.sin(t / 5000.0 + 1.0),
        "preheater_supply_temperature": np.zeros(n),
        "preheater_return_temperature": np.zeros(n),
        "supply_air_temperature": np.zeros(n),
        "ambient_temperature": 6.0 + 2.0 * np.sin(t / 9000.0),
        "process_air_flow": 33000.0 + 50.0 * rng.standard_normal(n),
        "steam_valve_position": np.full(n, 45.0),
        "process_fan_power": np.full(n, 115.0),
    }
    t_s, _ = rollout_storage(cfg, PlantDay(label, t, base), q_in)
    chain = ExtractionChain.from_config(cfg)
    res = chain.solve_given_q_in(
        t_s,
        q_in,
        base["storage_loop_flow"],
        base["preheater_loop_flow"],
        base["process_air_flow"],
        base["ambient_temperature"],
    )
    c_air = air_capacity_rate_kw_per_k(
        base["process_air_flow"], base["ambient_temperature"], cfg.air
    )
    q_steam = 480.0
    fan = cfg.steam.fan_heat_fraction * base["process_fan_power"]
    t_supply_air = base["ambient_temperature"] + (res.q_out_kw + q_steam + fan) / c_air
    roles = {
        **base,
        "storage_temperature": t_s,
        "preheater_supply_temperature": res.t_supply_celsius,
        "preheater_return_temperature": res.t_return_celsius,
        "supply_air_temperature": t_supply_air,
    }
    day = PlantDay(label, t, roles)
    day.roles["_true_q_in"] = q_in
    return day


# -- config ------------------------------------------------------------------------------------
def test_fingerprint_is_stable_and_sees_every_placeholder() -> None:
    a, b = StorageSystemConfig(), StorageSystemConfig()
    assert a.fingerprint == b.fingerprint
    changed = dataclasses.replace(a, storage=StorageConfig(volume_m3=11.0))
    assert changed.fingerprint != a.fingerprint
    assert changed.section_fingerprints["storage"] != a.section_fingerprints["storage"]
    assert a.section_fingerprints["water"] == changed.section_fingerprints["water"]
    flipped = dataclasses.replace(a, topology="exchanger_then_storage")
    assert flipped.fingerprint != a.fingerprint


def test_placeholder_warning_and_validation() -> None:
    with pytest.warns(PlaceholderConfigWarning):
        StorageSystemConfig().warn_if_placeholder()
    with pytest.raises(ValueError):
        StorageConfig(volume_m3=0.0)
    with pytest.raises(ValueError):
        CompressorHeatConfig(rejected_fraction=1.5)
    with pytest.raises(ValueError):
        StorageSystemConfig(topology="nonsense")  # type: ignore[arg-type]
    assert "storage.volume_m3" in StorageSystemConfig().provenance.open_todos


# -- exchanger ---------------------------------------------------------------------------------
def test_effectiveness_limits_and_continuity_at_balanced_flows() -> None:
    assert effectiveness_counterflow(np.array([1e-9]), np.array([0.5]))[0] < 1e-8
    assert effectiveness_counterflow(np.array([60.0]), np.array([0.5]))[0] == pytest.approx(1.0)
    ntu = np.array([0.7, 2.0])
    near = effectiveness_counterflow(ntu, np.array([1.0 - 1e-6, 1.0 - 1e-6]))
    assert near == pytest.approx(ntu / (1 + ntu), rel=1e-5)
    assert effectiveness_counterflow(ntu, np.array([1.0, 1.0])) == pytest.approx(ntu / (1 + ntu))


def test_min_approach_equals_dense_profile_minimum() -> None:
    c_hot, c_cold = 30.0, 20.0
    t_hi, t_ci = 70.0, 20.0
    eps = float(effectiveness_counterflow(np.array([1.4]), np.array([c_cold / c_hot]))[0])
    q = eps * c_cold * (t_hi - t_ci)
    t_ho, t_co = t_hi - q / c_hot, t_ci + q / c_cold
    frac = np.linspace(0.0, q, 2001)
    hot = t_hi - frac / c_hot  # hot temperature after giving ``frac``
    cold = t_co - frac / c_cold  # cold temperature facing it (counter-current)
    assert min_approach_k(t_hi, t_ho, t_ci, t_co) == pytest.approx(np.min(hot - cold))


def test_exchanger_duty_energy_balance_both_sides() -> None:
    hx = CounterflowExchanger(ExchangerConfig(ua_kw_per_k=15.0))
    c_hot, c_cold = np.array([40.0]), np.array([21.0])
    q = hx.duty_kw(np.array([60.0]), c_hot, np.array([25.0]), c_cold)
    assert 0.0 < q[0] < 21.0 * 35.0
    assert hx.effectiveness(c_hot, c_cold)[0] == pytest.approx(q[0] / (21.0 * 35.0))


# -- chain and storage -------------------------------------------------------------------------
@pytest.mark.parametrize("topology", ["storage_then_exchanger", "exchanger_then_storage"])
def test_chain_satisfies_both_exchanger_equations(topology: str) -> None:
    cfg = _cfg(topology)
    chain = ExtractionChain.from_config(cfg)
    arr = lambda x: np.array([x], dtype=float)  # noqa: E731
    r = chain.solve_given_q_in(arr(55.0), arr(300.0), arr(37.0), arr(18.5), arr(33000.0), arr(5.0))
    c1 = water_capacity_rate_kw_per_k(37.0, cfg.water)
    c2 = water_capacity_rate_kw_per_k(18.5, cfg.water)
    c_air = air_capacity_rate_kw_per_k(33000.0, 5.0, cfg.air)
    a = chain.storage_exchanger.conductance_kw_per_k(c1, c2)
    b = chain.preheater.conductance_kw_per_k(c2, c_air)
    q = r.q_out_kw[0]
    assert q == pytest.approx(a * (r.t_hot_in_celsius[0] - r.t_return_celsius[0]))
    assert q == pytest.approx(b * (r.t_supply_celsius[0] - 5.0))
    assert q == pytest.approx(c2 * (r.t_supply_celsius[0] - r.t_return_celsius[0]))
    assert q == pytest.approx(c1 * (r.t_hot_in_celsius[0] - r.t_hot_out_celsius[0]))


def test_topology_a_reduces_to_b_without_compressor_heat() -> None:
    args = [np.array([x]) for x in (55.0, 0.0, 37.0, 18.5, 33000.0, 5.0)]
    qb = (
        ExtractionChain.from_config(_cfg("storage_then_exchanger")).solve_given_q_in(*args).q_out_kw
    )
    qa = (
        ExtractionChain.from_config(_cfg("exchanger_then_storage")).solve_given_q_in(*args).q_out_kw
    )
    assert qa == pytest.approx(qb)


@pytest.mark.parametrize("topology", ["storage_then_exchanger", "exchanger_then_storage"])
def test_storage_rate_and_q_in_forms_agree(topology: str) -> None:
    chain = ExtractionChain.from_config(_cfg(topology))
    arr = lambda x: np.array([x], dtype=float)  # noqa: E731
    d_ts = arr(2.0e-4)
    r1 = chain.solve_given_storage_rate(
        arr(55.0), d_ts, arr(37.0), arr(18.5), arr(33000.0), arr(5.0)
    )
    r2 = chain.solve_given_q_in(arr(55.0), r1.q_in_kw, arr(37.0), arr(18.5), arr(33000.0), arr(5.0))
    assert r2.q_out_kw == pytest.approx(r1.q_out_kw)
    back = chain.storage.derivative_k_per_s(arr(55.0), r1.q_in_kw, r1.q_out_kw)
    assert back == pytest.approx(d_ts)


def test_storage_balance_and_exponential_decay() -> None:
    cfg = _cfg()
    store = MixedNodeStorage(cfg.storage, cfg.water)
    t = np.array([60.0])
    loss = store.loss_kw(t)
    assert store.derivative_k_per_s(t, loss, np.zeros(1)) == pytest.approx(0.0)
    # pure loss: T - T_room decays with tau = C / UA
    ua, cap = cfg.storage.ua_kw_per_k, store.heat_capacity_kj_per_k()
    temp, dt_s = 60.0, 10.0
    for _ in range(int(3600 / dt_s)):
        temp += dt_s * float(
            store.derivative_k_per_s(np.array([temp]), np.zeros(1), np.zeros(1))[0]
        )
    expected = 20.0 + 40.0 * np.exp(-3600.0 * ua / cap)
    assert temp == pytest.approx(expected, abs=1e-3)


def test_rollout_conserves_energy() -> None:
    cfg = _cfg()
    day = _synthetic_day(cfg)
    q_in = day["_true_q_in"]
    t_s, q_out = rollout_storage(cfg, day, q_in)
    store = MixedNodeStorage(cfg.storage, cfg.water)
    stored = store.heat_capacity_kj_per_k() * (t_s[-1] - t_s[0])
    net = np.sum((q_in - q_out - store.loss_kw(t_s))[:-1]) * 60.0
    assert stored == pytest.approx(net, rel=2e-3)


# -- identifiability ---------------------------------------------------------------------------
def test_volume_bounds_contain_true_volume_and_q_in_is_recovered() -> None:
    cfg = _cfg()
    day = _synthetic_day(cfg)
    for window in (1, 15, 30):
        strict = volume_bounds(day, cfg, window)
        assert strict.lower_m3 == 0.0
        assert TRUE_VOLUME <= strict.upper_m3 < np.inf
    index, q_rec = q_in_curve(day, cfg, TRUE_VOLUME, 15)
    assert q_rec == pytest.approx(day["_true_q_in"][index], abs=8.0)
    # a wrong volume distorts the curve (not identified by the curve itself)
    _, q_wrong = q_in_curve(day, cfg, 3.0 * TRUE_VOLUME, 15)
    assert np.max(np.abs(q_wrong - day["_true_q_in"][index])) > 100.0


def test_volume_bounds_tolerance_and_q_max_monotone() -> None:
    cfg = _cfg()
    day = _synthetic_day(cfg)
    strict = volume_bounds(day, cfg, 15, 0.0)
    loose = volume_bounds(day, cfg, 15, 0.05)
    assert loose.upper_from_nonnegative_m3 >= strict.upper_from_nonnegative_m3
    tight_max = volume_bounds(day, cfg, 15, 0.0, q_max_kw=700.0)
    assert tight_max.upper_from_q_max_m3 <= strict.upper_from_q_max_m3


def test_implied_conductances_topology_b_independent_of_volume_and_exact() -> None:
    cfg = _cfg()
    day = _synthetic_day(cfg)
    ua1, ua2 = implied_conductances(day, cfg, "storage_then_exchanger", 1.0, 15)
    ua1b, _ = implied_conductances(day, cfg, "storage_then_exchanger", 99.0, 15)
    assert np.nanmedian(ua1) == pytest.approx(15.0, rel=1e-3)
    assert np.nanmedian(ua2) == pytest.approx(12.0, rel=1e-3)
    assert np.allclose(ua1, ua1b, equal_nan=True)
    ua1_a, _ = implied_conductances(day, cfg, "exchanger_then_storage", 1.0, 15)
    assert not np.allclose(ua1, ua1_a, equal_nan=True)


def test_smoothed_derivative_of_a_ramp() -> None:
    index, d = smoothed_derivative_k_per_s(np.arange(100.0) * 0.5, 60.0, 9)
    assert d == pytest.approx(0.5 / 60.0)
    assert index.stop - index.start == len(d)


# -- calibration -------------------------------------------------------------------------------
def test_fit_recovers_conductances_and_day_split_rule() -> None:
    truth = _cfg()
    day = _synthetic_day(truth)
    start = dataclasses.replace(
        truth, storage_exchanger=ExchangerConfig(5.0), preheater=ExchangerConfig(30.0)
    )
    fit = fit_conductances(start, day)
    assert fit.success
    assert fit.ua_exchanger_kw_per_k == pytest.approx(15.0, rel=1e-3)
    assert fit.ua_preheater_kw_per_k == pytest.approx(12.0, rel=1e-3)
    other = _synthetic_day(truth, label="synthetic_2", seed=3)
    require_different_days(day, other)
    with pytest.raises(ValueError):
        require_different_days(day, day)
    index, res = predict_extraction(fit.config, other)
    assert (
        error_metrics(res.t_supply_celsius, other["preheater_supply_temperature"][index])["max_abs"]
        < 0.05
    )


def test_storage_rollout_with_oracle_q_in_tracks_and_constant_q_in_is_labelled_weaker() -> None:
    cfg = _cfg()
    day = _synthetic_day(cfg)
    t_oracle, _ = rollout_storage(cfg, day, oracle_q_in_kw(cfg, day))
    t_const, _ = rollout_storage(cfg, day, mean_q_in_kw(cfg, day))
    ref = day["storage_temperature"]
    assert error_metrics(t_oracle, ref)["rmse"] < 0.3
    assert error_metrics(t_const, ref)["rmse"] > error_metrics(t_oracle, ref)["rmse"]
    assert error_metrics(persistence(ref), ref)["max_abs"] > 0.0


def test_arx_recovers_an_ar_process_and_free_run_starts_at_y0() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 2))
    y = np.zeros(400)
    for k in range(1, 400):
        y[k] = 0.9 * y[k - 1] + 0.5 * x[k, 0] - 0.2 * x[k, 1] + 1.0
    model = fit_arx(y, x)
    assert model.a == pytest.approx(0.9, abs=1e-3)
    free = arx_free_run(model, y[0], x)
    assert free[0] == y[0]
    assert np.max(np.abs(free - y)) < 1e-2


def test_repair_glitches_fixes_a_single_dropout_only() -> None:
    day = _synthetic_day(_cfg())
    spoiled = dict(day.roles)
    flow = spoiled["preheater_loop_flow"].copy()
    original = flow[500]
    flow[500] = 0.5 * original
    spoiled["preheater_loop_flow"] = flow
    fixed, counts = repair_glitches(PlantDay(day.label, day.time_s, spoiled))
    assert counts["preheater_loop_flow"] == 1
    assert counts["storage_loop_flow"] == 0
    assert fixed["preheater_loop_flow"][500] == pytest.approx(original, rel=0.01)
    assert np.array_equal(np.delete(fixed["preheater_loop_flow"], 500), np.delete(flow, 500))


# -- balance -----------------------------------------------------------------------------------
def test_steam_is_air_heating_minus_waste_heat_and_co2_follows() -> None:
    cfg = _cfg()
    day = _synthetic_day(cfg)
    q_out = np.asarray(
        ExtractionChain.from_config(cfg)
        .solve_given_q_in(
            day["storage_temperature"],
            day["_true_q_in"],
            day["storage_loop_flow"],
            day["preheater_loop_flow"],
            day["process_air_flow"],
            day["ambient_temperature"],
        )
        .q_out_kw
    )
    bal = daily_balance(day, cfg, q_out)
    assert bal.steam_kwh == pytest.approx(bal.heating_demand_kwh - bal.waste_heat_kwh)
    assert bal.waste_heat_share + bal.steam_share == pytest.approx(1.0)
    assert np.mean(air_heating_kw(day, cfg) - q_out) == pytest.approx(480.0, rel=1e-6)
    assert bal.co2_without_recovery_kg > bal.co2_kg > 0.0
    assert bal.steam_negative_minutes == 0


# -- roles -------------------------------------------------------------------------------------
def test_example_role_map_has_only_placeholder_tags() -> None:
    role_map = load_role_map(REPO / "conf" / "plant" / "plant_local.yaml.example")
    assert set(REQUIRED_ROLES) <= set(role_map.used)
    assert all(tag.startswith("TAG_") for tags in role_map.used.values() for tag in tags)


def test_day_from_series_averages_multi_channel_roles_and_rejects_missing_ones() -> None:
    role_map = load_role_map(REPO / "conf" / "plant" / "plant_local.yaml.example")
    stamps = tuple(dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i) for i in range(3))
    columns = {tag: (1.0, 2.0, 3.0) for tags in role_map.used.values() for tag in tags}
    columns["TAG_SUPPLY_AIR_T_1"] = (3.0, 4.0, 5.0)
    series = PlantTimeSeries(stamps, dict.fromkeys(columns, ""), columns)
    day = day_from_series(series, role_map, "d")
    assert day["supply_air_temperature"][0] == pytest.approx((3.0 + 1.0 + 1.0) / 3.0)
    assert day.step_s == 60.0
    del columns["TAG_STORAGE_T"]
    with pytest.raises(KeyError):
        day_from_series(PlantTimeSeries(stamps, dict.fromkeys(columns, ""), columns), role_map, "d")


# -- confidentiality ---------------------------------------------------------------------------
def test_new_files_name_no_measurement_points_or_equipment_codes() -> None:
    """No tags / equipment codes in code, docs and tests of this study (roles only)."""
    out = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--cached",
            "--exclude-standard",
            "--",
            "src/twinlink_ml/storage_a1",
            "scripts/study_storage_a1.py",
            "docs/18_storage_plant_a.md",
            "tests/test_storage_a1.py",
            "conf/plant",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    ).stdout.split()
    tag = re.compile(r"\b[A-Z]{2,7}-\d{5,6}\b|\b[WB]-? ?\d{4}[A-Z]?\b|\bCV-\d")
    hits = []
    for name in out:
        path = REPO / name
        if path.is_file():
            hits += [f"{name}: {m}" for m in tag.findall(path.read_text(encoding="utf-8"))]
    assert not hits


def test_no_numpy_warnings_in_default_chain() -> None:
    cfg = _cfg()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _synthetic_day(cfg)
