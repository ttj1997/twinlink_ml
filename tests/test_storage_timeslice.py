"""Tests of the time-slice pinch study (storage test 1): table, programme, rule, streams."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from twinlink_ml.heatnetwork.config import HeatSinkConfig, HeatSourceConfig
from twinlink_ml.heatnetwork.pinch import (
    StreamSegment,
    cold_segment_for_sink,
    compute_pinch_target,
    hot_segments_for_source,
)
from twinlink_ml.simulation.config import GasConfig
from twinlink_ml.storage_timeslice import (
    PLACEHOLDER_BANDS,
    MeasuredDay,
    Placeholders,
    Scenario,
    StreamFactory,
    cascade_profile,
    direct_hot_utility_kw,
    evaluate_scenario,
    is_realistic,
    level_groups,
    simulate_rule,
    site_streams,
    solve_ideal_storage,
    storage_stream_temperatures,
)
from twinlink_ml.storage_timeslice.plant import (
    block_mean,
    exhaust_humidity,
    intake_humidity,
)
from twinlink_ml.storage_timeslice.profiles import (
    Clock,
    compressor_profile,
    line_c_profile,
    night_profile,
    water_profile,
)

DT_H = 0.25


def _pad(rows: list[list[tuple[float, float, float]]]) -> np.ndarray:
    width = max(len(r) for r in rows)
    out = np.zeros((len(rows), max(width, 1), 3))
    for i, r in enumerate(rows):
        if r:
            out[i, : len(r)] = np.asarray(r)
    return out


def _random_streams(
    rng: np.random.Generator,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    hot = []
    for _ in range(rng.integers(1, 5)):
        high = rng.uniform(40, 110)
        hot.append((high, high - rng.uniform(5, 60), rng.uniform(0.5, 12)))
    cold = []
    for _ in range(rng.integers(1, 5)):
        low = rng.uniform(-5, 60)
        cold.append((low + rng.uniform(5, 60), low, rng.uniform(0.5, 12)))
    return hot, cold


# ---------------------------------------------------------------- problem table


def test_vectorised_table_equals_pinch_module() -> None:
    rng = np.random.default_rng(1)
    for _ in range(40):
        hot_rows, cold_rows = _random_streams(rng)
        dt_min = float(rng.choice([5.0, 10.0, 20.0, 30.0]))
        extra = rng.uniform(20, 90, size=4)
        reference = compute_pinch_target(
            [StreamSegment("h", True, a, b, c) for a, b, c in hot_rows],
            [StreamSegment("c", False, a, b, c) for a, b, c in cold_rows],
            minimum_pinch_k=dt_min,
        ).minimum_hot_utility_kw
        cascade = cascade_profile(
            _pad([hot_rows]), _pad([cold_rows]), minimum_pinch_k=dt_min, extra_points=extra
        )
        assert direct_hot_utility_kw(cascade)[0] == pytest.approx(reference, abs=1e-8)


def test_padding_segments_are_ignored() -> None:
    hot = _pad([[(80.0, 40.0, 5.0)], [(80.0, 40.0, 5.0), (60.0, 30.0, 3.0)]])
    cold = _pad([[(70.0, 20.0, 4.0)], [(70.0, 20.0, 4.0)]])
    both = cascade_profile(hot, cold, minimum_pinch_k=10.0, extra_points=np.array([]))
    single = cascade_profile(hot[:1, :1], cold[:1], minimum_pinch_k=10.0, extra_points=np.array([]))
    assert direct_hot_utility_kw(both)[0] == pytest.approx(direct_hot_utility_kw(single)[0])


def test_storage_stream_temperatures_follow_the_specification() -> None:
    tc, td = storage_stream_temperatures(np.array([50.0]), minimum_pinch_k=20.0, approach_k=20.0)
    # charging: hot real temperature above T_S + a  <=>  shifted (hot - dT/2) above T*_c
    assert tc[0] + 10.0 == pytest.approx(50.0 + 20.0)
    # discharging: cold real temperature below T_S - a  <=>  shifted (cold + dT/2) below T*_d
    assert td[0] - 10.0 == pytest.approx(50.0 - 20.0)
    tc2, td2 = storage_stream_temperatures(np.array([50.0]), minimum_pinch_k=20.0, approach_k=10.0)
    assert tc2[0] == pytest.approx(td2[0])
    with pytest.raises(ValueError):
        storage_stream_temperatures(np.array([50.0]), minimum_pinch_k=20.0, approach_k=5.0)


def test_group_utility_equals_explicit_isothermal_streams() -> None:
    """U of the pieces equals the problem table with the storage streams added explicitly."""
    rng = np.random.default_rng(7)
    width = 1e-4  # narrow stand-in for an isothermal stream (StreamSegment needs high > low)
    for _ in range(25):
        hot_rows, cold_rows = _random_streams(rng)
        dt_min = float(rng.choice([10.0, 20.0]))
        level = float(rng.uniform(35, 75))
        approach = dt_min
        tc, td = storage_stream_temperatures(
            np.array([level]), minimum_pinch_k=dt_min, approach_k=approach
        )
        cascade = cascade_profile(
            _pad([hot_rows]),
            _pad([cold_rows]),
            minimum_pinch_k=dt_min,
            extra_points=np.concatenate([tc, td]),
        )
        groups = level_groups(cascade, tc, td)
        q_c, q_d = float(rng.uniform(0, 80)), float(rng.uniform(0, 80))
        ours = groups.utility_kw(np.array([[q_c]]), np.array([[q_d]]))[0]
        # explicit streams in real temperatures: the shifted temperature of the charging stream
        # is T*_c, so its real (cold) temperature is T*_c - dT/2; discharging: T*_d + dT/2
        hot_seg = [StreamSegment("h", True, a, b, c) for a, b, c in hot_rows]
        cold_seg = [StreamSegment("c", False, a, b, c) for a, b, c in cold_rows]
        real_c = float(tc[0] - dt_min / 2.0)
        real_d = float(td[0] + dt_min / 2.0)
        if q_c > 0:
            cold_seg.append(StreamSegment("charge", False, real_c, real_c - width, q_c / width))
        if q_d > 0:
            hot_seg.append(StreamSegment("discharge", True, real_d + width, real_d, q_d / width))
        reference = compute_pinch_target(hot_seg, cold_seg, minimum_pinch_k=dt_min)
        assert ours == pytest.approx(reference.minimum_hot_utility_kw, abs=0.05, rel=1e-3)


# ------------------------------------------------------------------- programme


def _toy(n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Toy case: 100 kW of hot water in the first half, 100 kW demand in the second half.

    Hot 80 -> 78 degC, cold 20 -> 22 degC; nothing is usable at the same time (direct demand
    1200 kWh/d).
    """
    hot = np.zeros((n, 1, 3))
    cold = np.zeros((n, 1, 3))
    hot[: n // 2, 0] = (80.0, 78.0, 50.0)
    cold[n // 2 :, 0] = (22.0, 20.0, 50.0)
    return hot, cold


def _toy_groups(level: float = 50.0, approach: float = 20.0):  # type: ignore[no-untyped-def]
    hot, cold = _toy()
    tc, td = storage_stream_temperatures(
        np.array([level]), minimum_pinch_k=20.0, approach_k=approach
    )
    cascade = cascade_profile(
        hot, cold, minimum_pinch_k=20.0, extra_points=np.concatenate([tc, td])
    )
    return cascade, level_groups(cascade, tc, td)


def test_toy_direct_energy_and_hand_solved_storage() -> None:
    cascade, groups = _toy_groups()
    assert float(np.sum(direct_hot_utility_kw(cascade))) * DT_H == pytest.approx(1200.0)
    for cap in (0.0, 300.0, 600.0, 1200.0, 5000.0):
        ideal = solve_ideal_storage(groups, dt_h=DT_H, capacity_kwh=cap, loss_per_h=0.0)
        assert ideal.energy_kwh == pytest.approx(1200.0 - min(cap, 1200.0), abs=1e-5)
        rule = simulate_rule(groups, dt_h=DT_H, capacities_kwh=np.array([cap]), loss_per_h=0.0)
        assert rule.energy_kwh[0] == pytest.approx(1200.0 - min(cap, 1200.0), abs=1e-5)


def test_toy_level_outside_the_streams_gives_no_saving() -> None:
    _, groups = _toy_groups(level=76.0)  # charging needs hot above 96 degC: impossible
    ideal = solve_ideal_storage(groups, dt_h=DT_H, capacity_kwh=5000.0, loss_per_h=0.0)
    assert ideal.energy_kwh == pytest.approx(1200.0)


def test_capacity_zero_is_exactly_direct() -> None:
    cascade, groups = _toy_groups()
    direct = float(np.sum(direct_hot_utility_kw(cascade))) * DT_H
    assert (
        solve_ideal_storage(groups, dt_h=DT_H, capacity_kwh=0.0, loss_per_h=0.01).energy_kwh
        == direct
    )
    rule = simulate_rule(groups, dt_h=DT_H, capacities_kwh=np.array([0.0]), loss_per_h=0.01)
    assert rule.energy_kwh[0] == direct


def test_energy_balance_closed_without_and_with_losses() -> None:
    _, groups = _toy_groups()
    lossless = solve_ideal_storage(groups, dt_h=DT_H, capacity_kwh=800.0, loss_per_h=0.0)
    assert float(
        np.sum(lossless.charge_kw) - np.sum(lossless.discharge_kw)
    ) * DT_H == pytest.approx(0.0, abs=1e-6)
    lossy = solve_ideal_storage(groups, dt_h=DT_H, capacity_kwh=800.0, loss_per_h=0.02)
    assert lossy.loss_kwh > 0.0
    assert lossy.balance_residual_kwh(DT_H) == pytest.approx(0.0, abs=1e-6)
    assert float(np.max(lossy.stored_kwh)) <= 800.0 + 1e-6
    assert lossy.energy_kwh >= lossless.energy_kwh - 1e-9  # losses never help


def test_ideal_is_monotone_and_rule_not_better_on_random_problems() -> None:
    rng = np.random.default_rng(3)
    caps = np.array([0.0, 200.0, 800.0, 3000.0])
    for _ in range(6):
        n = 24
        hot_rows = [_random_streams(rng)[0] for _ in range(n)]
        cold_rows = [_random_streams(rng)[1] for _ in range(n)]
        # random on/off pattern: heat and demand at different times
        hot = _pad(hot_rows) * rng.integers(0, 2, size=(n, 1, 1))
        cold = _pad(cold_rows) * rng.integers(0, 2, size=(n, 1, 1))
        level = float(rng.uniform(40, 70))
        tc, td = storage_stream_temperatures(
            np.array([level]), minimum_pinch_k=10.0, approach_k=10.0
        )
        cascade = cascade_profile(
            hot, cold, minimum_pinch_k=10.0, extra_points=np.concatenate([tc, td])
        )
        groups = level_groups(cascade, tc, td)
        for loss in (0.0, 0.01):
            ideal = np.array(
                [
                    solve_ideal_storage(
                        groups, dt_h=1.0, capacity_kwh=float(c), loss_per_h=loss
                    ).energy_kwh
                    for c in caps
                ]
            )
            rule = simulate_rule(groups, dt_h=1.0, capacities_kwh=caps, loss_per_h=loss)
            assert np.all(np.diff(ideal) <= 1e-6)
            assert np.all(rule.energy_kwh >= ideal - 1e-6)
            assert np.all(rule.energy_kwh <= ideal[0] + 1e-9)
            assert rule.converged.all()


def test_multi_level_with_one_level_equals_single_level_and_never_exceeds_best_single() -> None:
    rng = np.random.default_rng(11)
    n = 24
    hot = _pad([_random_streams(rng)[0] for _ in range(n)]) * rng.integers(0, 2, size=(n, 1, 1))
    cold = _pad([_random_streams(rng)[1] for _ in range(n)]) * rng.integers(0, 2, size=(n, 1, 1))
    levels = np.array([40.0, 55.0, 70.0])
    tc, td = storage_stream_temperatures(levels, minimum_pinch_k=10.0, approach_k=10.0)
    cascade = cascade_profile(
        hot, cold, minimum_pinch_k=10.0, extra_points=np.concatenate([tc, td])
    )
    singles = []
    for j in range(3):
        g = level_groups(cascade, tc[j : j + 1], td[j : j + 1])
        singles.append(
            solve_ideal_storage(g, dt_h=1.0, capacity_kwh=1000.0, loss_per_h=0.0).energy_kwh
        )
    multi = solve_ideal_storage(
        level_groups(cascade, tc, td), dt_h=1.0, capacity_kwh=1000.0, loss_per_h=0.0
    ).energy_kwh
    assert multi <= min(singles) + 1e-6


# ------------------------------------------------------------------ streams


def _synthetic_day(label: str = "product_1") -> MeasuredDay:
    t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    return MeasuredDay(
        label=label,
        ambient_c=6.0 + 3.0 * np.sin(t),
        process_air_nm3h=np.full(96, 30000.0),
        supply_air_c=np.full(96, 84.0),
        exhaust_out_c=np.full(96, 55.0) + np.sin(t),
        spray_kg_h=np.full(96, 900.0),
        atomising_nm3h=np.full(96, 560.0),
    )


def _fast(**changes: object) -> Placeholders:
    return replace(Placeholders(n_segments_below_dew_point=6), **changes)


def test_exhaust_unit_scaling_equals_direct_sampling() -> None:
    factory = StreamFactory({"product_1": _synthetic_day()})
    ph = _fast()
    unit, m_dry, _ = factory.exhaust_unit("product_1", ph)
    day = _synthetic_day()
    y_out, _, _ = exhaust_humidity(
        day,
        solid_fraction=ph.solid_fraction,
        relative_humidity=ph.relative_humidity,
        normal_density_kg_per_nm3=ph.normal_density_kg_per_nm3,
        gas=ph.gas,
    )
    for i in (0, 40):
        source = HeatSourceConfig(
            "x", float(m_dry[i]), float(day.exhaust_out_c[i]), float(y_out[i])
        )
        direct = hot_segments_for_source(
            source, ph.gas, ph.cold_temperature_celsius, n_segments_below_dew_point=6
        )
        scaled = unit[i][unit[i][:, 2] > 0] * np.array([1.0, 1.0, m_dry[i]])
        assert len(direct) == len(scaled)
        for seg, row in zip(direct, scaled, strict=True):
            assert row[0] == pytest.approx(seg.high_temperature_celsius)
            assert row[1] == pytest.approx(seg.low_temperature_celsius)
            assert row[2] == pytest.approx(seg.capacity_rate_kw_per_k, rel=1e-9)


def test_air_sink_equals_cold_segment_of_the_heat_network_module() -> None:
    factory = StreamFactory({"product_1": _synthetic_day()})
    scenario = Scenario("A", "product_1", "day", "flat", placeholders=_fast())
    bank = factory.build(scenario)
    day = _synthetic_day()
    ph = scenario.placeholders
    y = intake_humidity(day.ambient_c, ph.relative_humidity, ph.gas)
    i = 10
    sink = HeatSinkConfig(
        "air",
        float(day.process_air_nm3h[i] * ph.normal_density_kg_per_nm3 / 3600.0),
        float(y[i]),
        float(day.ambient_c[i]),
        float(day.supply_air_c[i]),
    )
    ref = cold_segment_for_sink(sink, GasConfig())
    row = bank.cold["air_a"][i, 0]
    assert row[0] == pytest.approx(ref.high_temperature_celsius)
    assert row[1] == pytest.approx(ref.low_temperature_celsius)
    assert row[2] == pytest.approx(ref.capacity_rate_kw_per_k)


def test_exhaust_humidity_is_capped_at_saturation() -> None:
    day = replace(_synthetic_day(), exhaust_out_c=np.full(96, 30.0), spray_kg_h=np.full(96, 3000.0))
    ph = Placeholders()
    y, _, capped = exhaust_humidity(
        day,
        solid_fraction=0.3,
        relative_humidity=0.75,
        normal_density_kg_per_nm3=1.293,
        gas=ph.gas,
    )
    assert capped.all()
    assert np.all(y < 0.03)  # saturation humidity at 30 degC is about 0.027 kg/kg


def test_site_membership_and_scenario_validation() -> None:
    assert site_streams("A")[0] == ("compressor_a", "exhaust_a")
    assert "air_c1" in site_streams("C")[1] and "air_a" not in site_streams("C")[1]
    with pytest.raises(ValueError):
        Scenario("A", "product_1", "day", "weekly")
    with pytest.raises(ValueError):
        Scenario("B", "product_1", "day", "flat")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Placeholders(compressor_hot_in_celsius=40.0, compressor_hot_out_celsius=45.0)


def test_realistic_band() -> None:
    base = Scenario("A+C", "product_1", "day", "shift", compressor_a_kw=800.0)
    assert is_realistic(base)
    assert not is_realistic(replace(base, hall_kw=300.0))
    assert not is_realistic(base.with_placeholders(minimum_pinch_k=40.0))
    assert not is_realistic(replace(base, ambient_shift_k=-20.0))
    assert set(PLACEHOLDER_BANDS) >= {"minimum_pinch_k", "loss_per_h"}


def test_profiles() -> None:
    ph = Placeholders()
    clock = Clock.of(7)
    flat = compressor_profile("flat", clock, ph)
    assert np.all(flat == 1.0)
    shift = compressor_profile("shift", clock, ph)
    assert shift.min() == ph.low_fraction and shift.max() == 1.0
    weekly = compressor_profile("weekly", clock, ph)
    assert weekly[: 5 * 96].min() == 1.0 and weekly[5 * 96 :].max() == ph.low_fraction
    c1, c2 = line_c_profile("shift", clock, ph, 1), line_c_profile("shift", clock, ph, 2)
    assert c1.sum() > c2.sum() > 0
    assert line_c_profile("weekly", clock, ph, 1)[5 * 96 :].sum() == 0.0
    assert water_profile("weekly", clock, ph)[5 * 96 :].sum() == 0.0
    night = night_profile(clock, 0.4)
    assert night[0] == 1.0 and night[12 * 4] == 0.4


def test_block_mean() -> None:
    values = np.arange(1441, dtype=float)
    out = block_mean(values)
    assert len(out) == 96 and out[0] == pytest.approx(7.0)
    with pytest.raises(ValueError):
        block_mean(np.arange(100.0))


# ------------------------------------------------------------------ scenario


def _eval(scenario: Scenario, **kw: object):  # type: ignore[no-untyped-def]
    factory = StreamFactory({"product_1": _synthetic_day()})
    return evaluate_scenario(
        factory.build(scenario),
        scenario,
        capacities_kwh=(0.0, 1000.0, 5000.0),
        levels_celsius=(35.0, 45.0, 55.0),
        **kw,  # type: ignore[arg-type]
    )


def test_scenario_checks_pass_and_flat_has_no_shifting_potential() -> None:
    flat = _eval(
        Scenario("A", "product_1", "day", "flat", hall_kw=0.0, water_kw=0.0, placeholders=_fast())
    )
    assert all(v for k, v in flat.checks.items() if isinstance(v, bool))
    assert flat.direct_kwh_per_day > 0.0
    # everything runs all day at constant power: storing cannot help beyond solver noise
    assert abs(flat.saving_fraction(flat.rows[-1])) < 0.01


def test_shift_mode_has_potential_and_rule_is_close() -> None:
    scenario = Scenario(
        "A+C",
        "product_1",
        "week",
        "weekly",
        compressor_a_kw=1200.0,
        hall_kw=0.0,
        water_kw=200.0,
        placeholders=_fast(),
    )
    result = _eval(scenario, multi_level=True)
    assert all(v for k, v in result.checks.items() if isinstance(v, bool))
    big = result.rows[-1]
    assert result.saving_fraction(big) >= 0.0
    assert big.multi_level_kwh_per_day is not None
    assert big.multi_level_kwh_per_day <= big.ideal_kwh_per_day + 1e-6
    assert 0.0 <= result.gap_fraction(big) <= result.saving_fraction(big) + 1e-9


def test_leaving_streams_out_changes_the_stream_set_only() -> None:
    scenario = Scenario("A+C", "product_1", "day", "shift", placeholders=_fast())
    factory = StreamFactory({"product_1": _synthetic_day()})
    bank = factory.build(scenario)
    full = evaluate_scenario(bank, scenario, capacities_kwh=(0.0,), levels_celsius=(45.0,))
    no_hall = evaluate_scenario(
        bank,
        scenario,
        cold_names=("air_a", "air_c1", "air_c2", "water"),
        capacities_kwh=(0.0,),
        levels_celsius=(45.0,),
    )
    assert no_hall.direct_kwh_per_day <= full.direct_kwh_per_day + 1e-9
    assert "hall" not in no_hall.cold_streams
