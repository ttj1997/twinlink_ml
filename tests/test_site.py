"""Site pinch study (``twinlink_ml.site``, ``docs/20_site_pinch.md``).

Checks: the YAML data set and the docstring table agree; the ramp problem table equals
``heatnetwork.pinch.compute_pinch_target`` and reproduces Befund 60; energy balance; a single line
reproduces the heat-network pinch target; network <= direct target; the network-capacity formula
equals the LP and, along the exchanger, ``heatnetwork.approach.max_duty_along``; heat-pump limits;
the stream builder (generic scaling, months, plant C off, flue gas).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from twinlink_ml.heatnetwork.approach import build_source_profile, max_duty_along
from twinlink_ml.heatnetwork.config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig
from twinlink_ml.heatnetwork.pinch import (
    compute_pinch_target,
    evaluate_pinch_target,
    hot_segments_for_source,
)
from twinlink_ml.simulation.config import GasConfig
from twinlink_ml.site import (
    Ramps,
    SiteScenario,
    SiteStreamBuilder,
    annual_summary,
    evaluate_year,
    flue_gas_state,
    heat_pump_limits,
    load_site_config,
    multi_loop_capacity,
    network_capacity,
    problem_table,
)
from twinlink_ml.site import config as site_config_module
from twinlink_ml.site.curves import zonal_network_target
from twinlink_ml.storage_timeslice.plant import MeasuredDay, exhaust_humidity

GAS = GasConfig()
RNG_SEED = 20261016


def _segments_array(segments: tuple) -> np.ndarray:  # type: ignore[type-arg]
    return np.array(
        [
            (s.high_temperature_celsius, s.low_temperature_celsius, s.capacity_rate_kw_per_k)
            for s in segments
        ]
    ).reshape(-1, 3)


def _random_streams(
    rng: np.random.Generator, n_hot: int, n_cold: int
) -> tuple[np.ndarray, np.ndarray]:
    def draw(n: int, lo: float, hi: float) -> np.ndarray:
        a = rng.uniform(lo, hi, (n, 2))
        return np.column_stack([a.max(axis=1) + 0.5, a.min(axis=1), rng.uniform(0.5, 30.0, n)])

    return draw(n_hot, 15.0, 130.0), draw(n_cold, 0.0, 110.0)


# --- data set ---------------------------------------------------------------------------------


def test_yaml_and_docstring_table_agree() -> None:
    config = load_site_config()
    doc = site_config_module.__doc__ or ""
    rows = {}
    for line in doc.splitlines():
        m = re.match(r"^(\w+)\s+(-?[\d.]+)\s+(-?[\d.]+) - (-?[\d.]+)\s", line)
        if m:
            rows[m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4))
    assert set(rows) == set(config.info)
    for name, entry in config.info.items():
        assert rows[name] == pytest.approx((entry.value, entry.low, entry.high)), name


def test_every_placeholder_is_inside_its_band_and_streams_are_named() -> None:
    config = load_site_config()
    for name, entry in config.info.items():
        assert entry.low <= getattr(config.placeholders, name) <= entry.high
        assert entry.source
    assert config.provenance.status == "placeholder"
    assert set(config.streams) == {
        "exhaust_a", "compressor_a", "compressor_c", "exhaust_c1", "exhaust_c2",
        "exhaust_generic", "boiler_flue", "air_a", "air_c1", "air_c2", "air_generic", "hall",
        "water",
    }  # fmt: skip
    assert sum(config.climate_days) == 365


# --- problem table ----------------------------------------------------------------------------


def test_ramps_total_and_average() -> None:
    seg = np.array([[80.0, 50.0, 2.0], [60.0, 20.0, 1.0]])
    r = Ramps.of_segments(seg)
    assert r.total == pytest.approx(2.0 * 30.0 + 1.0 * 40.0)
    assert float(r(70.0)) == pytest.approx(20.0)
    assert float(r(55.0)) == pytest.approx(2.0 * 25.0 + 5.0)
    assert float(r.below(55.0)) == pytest.approx(r.total - 55.0)
    avg = Ramps.combine([(r, 0.25), (Ramps.of_segments(seg * [1, 1, 3]), 0.75)])
    assert float(avg(55.0)) == pytest.approx(0.25 * 55.0 + 0.75 * 165.0)


def test_problem_table_equals_heatnetwork_pinch_on_random_cases() -> None:
    rng = np.random.default_rng(RNG_SEED)
    from twinlink_ml.heatnetwork.pinch import StreamSegment

    for _ in range(25):
        hot, cold = _random_streams(rng, 6, 4)
        dt = float(rng.uniform(0.0, 25.0))
        reference = compute_pinch_target(
            [StreamSegment("h", True, *row) for row in hot],
            [StreamSegment("c", False, *row) for row in cold],
            minimum_pinch_k=dt,
        )
        table = problem_table(Ramps.of_segments(hot), Ramps.of_segments(cold), dt)
        assert table.hot_utility_kw == pytest.approx(reference.minimum_hot_utility_kw, abs=1e-6)
        assert table.cold_utility_kw == pytest.approx(reference.minimum_cold_utility_kw, abs=1e-6)
        if reference.minimum_hot_utility_kw > 1e-6:
            assert table.pinch_shifted + dt / 2 == pytest.approx(
                reference.pinch_temperature_hot_side_celsius
            )


def test_reproduces_befund_60_pinch_target() -> None:
    """Befund 60: Q1/Q2 -> S1/S2, dT_min 10 K, T_kalt 15 degC: 4048.99 kW, pinch 81.42 degC."""
    sources = (
        HeatSourceConfig("Q1", 20.0, 125.233, 0.06482),
        HeatSourceConfig("Q2", 8.0, 81.416, 0.01998),
    )
    sinks = (
        HeatSinkConfig("S1", 20.0, 0.005, 15.0, 300.0),
        HeatSinkConfig("S2", 8.0, 0.005, 15.0, 120.0),
    )
    result, hot, cold = evaluate_pinch_target(
        sources, sinks, GAS, cold_temperature_celsius=15.0, minimum_pinch_k=10.0
    )
    table = problem_table(
        Ramps.of_segments(_segments_array(hot)), Ramps.of_segments(_segments_array(cold)), 10.0
    )
    assert table.hot_utility_kw == pytest.approx(result.minimum_hot_utility_kw, rel=1e-12)
    assert table.hot_utility_kw == pytest.approx(4048.99, abs=0.01)
    assert table.pinch_shifted + 5.0 == pytest.approx(81.416, abs=1e-6)
    # fuel of Befund 60 (H_u 45 MJ/kg, eta 0.80): 404.90 kg/h
    assert table.hot_utility_kw * 1000.0 / (45.0e6 * 0.80) * 3600.0 == pytest.approx(
        404.90, abs=0.01
    )


def test_energy_balance_closes() -> None:
    rng = np.random.default_rng(RNG_SEED + 1)
    for _ in range(20):
        hot, cold = _random_streams(rng, 5, 5)
        h, c = Ramps.of_segments(hot), Ramps.of_segments(cold)
        table = problem_table(h, c, 20.0)
        assert table.hot_utility_kw - table.cold_utility_kw == pytest.approx(
            c.total - h.total, abs=1e-6
        )
        assert np.min(table.gcc_kw) == pytest.approx(0.0, abs=1e-6)


def test_time_average_never_needs_more_utility_than_the_mean_of_the_steps() -> None:
    rng = np.random.default_rng(RNG_SEED + 2)
    steps = [_random_streams(rng, 4, 3) for _ in range(8)]
    per_step = np.mean(
        [
            problem_table(Ramps.of_segments(h), Ramps.of_segments(c), 20.0).hot_utility_kw
            for h, c in steps
        ]
    )
    tam = problem_table(
        Ramps.combine([(Ramps.of_segments(h), 1 / 8) for h, _ in steps]),
        Ramps.combine([(Ramps.of_segments(c), 1 / 8) for _, c in steps]),
        20.0,
    )
    assert tam.hot_utility_kw <= per_step + 1e-9


# --- network levels ---------------------------------------------------------------------------


def test_network_never_beats_the_direct_target_and_equals_the_lp() -> None:
    rng = np.random.default_rng(RNG_SEED + 3)
    levels = [(s, s - d) for s in (40.0, 55.0, 70.0, 90.0) for d in (10.0, 30.0)]
    for _ in range(15):
        hot, cold = _random_streams(rng, 5, 4)
        h, c = Ramps.of_segments(hot), Ramps.of_segments(cold)
        target_recovery = c.total - problem_table(h, c, 20.0).hot_utility_kw
        q_src, q_sink = network_capacity(h, c, [s for s, _ in levels], [r for _, r in levels], 10.0)
        q_n = np.minimum(q_src, q_sink)
        assert np.all(q_n <= target_recovery + 1e-6)
        for j, level in enumerate(levels):
            assert multi_loop_capacity(h, c, (level,), 10.0)[0] == pytest.approx(q_n[j], abs=1e-5)


def test_two_loops_are_never_worse_than_the_better_single_loop() -> None:
    rng = np.random.default_rng(RNG_SEED + 4)
    for _ in range(10):
        hot, cold = _random_streams(rng, 5, 4)
        h, c = Ramps.of_segments(hot), Ramps.of_segments(cold)
        low, high = (45.0, 30.0), (80.0, 60.0)
        both = multi_loop_capacity(h, c, (low, high), 10.0).sum()
        singles = [multi_loop_capacity(h, c, (lv,), 10.0)[0] for lv in (low, high)]
        assert both >= max(singles) - 1e-6
        assert both <= c.total - problem_table(h, c, 20.0).hot_utility_kw + 1e-6


def test_zonal_lp_limits() -> None:
    """Orderings of the zonal LP.

    One zone = direct target. A zone of all sources and one of all sinks never needs more than the
    composite loop of :func:`network_capacity` (a zone may reject loop heat to a return cooler,
    which the closed form does not allow) and never less than the direct target. Every stream in a
    zone of its own (each on its own branch between supply and return, parallel) needs at least as
    much as the two-zone case (where the sources may heat the loop in series).
    """
    rng = np.random.default_rng(RNG_SEED + 5)
    for _ in range(8):
        hot, cold = _random_streams(rng, 3, 3)
        h, c = Ramps.of_segments(hot), Ramps.of_segments(cold)
        direct = problem_table(h, c, 20.0).hot_utility_kw
        assert zonal_network_target([(h, c)], 70.0, 40.0, 20.0)[0] == pytest.approx(
            direct, abs=1e-6
        )
        empty = Ramps.of_segments(np.zeros((0, 3)))
        singles = [(Ramps.of_segments(row[None, :]), empty) for row in hot] + [
            (empty, Ramps.of_segments(row[None, :])) for row in cold
        ]
        q_src, q_sink = network_capacity(h, c, [70.0], [40.0], 10.0)
        expected = c.total - min(float(q_src[0]), float(q_sink[0]))
        two = zonal_network_target([(h, empty), (empty, c)], 70.0, 40.0, 20.0)[0]
        assert direct - 1e-6 <= two <= expected + 1e-6
        assert zonal_network_target(singles, 70.0, 40.0, 20.0)[0] >= two - 1e-6


def test_loop_capacity_matches_check_along_the_exchanger() -> None:
    """A condensing exhaust heating the loop: the closed form agrees with ``max_duty_along``."""
    source = HeatSourceConfig("exhaust", 12.0, 61.0, 0.019)
    floor = 15.0
    seg = _segments_array(
        hot_segments_for_source(source, GAS, floor, n_segments_below_dew_point=400)
    )
    hot = Ramps.of_segments(seg)
    t_s, t_r, a = 50.0, 20.0, 10.0
    q_src, _ = network_capacity(hot, Ramps.of_segments(np.zeros((0, 3))), [t_s], [t_r], a)
    q = float(q_src[0])
    assert q > 0.0
    config = HeatNetworkConfig(
        sources=(source,),
        sinks=(HeatSinkConfig("loop", 1.0, 0.0, t_r, t_s),),
        recovery_mode="condensing",
        cold_temperature_celsius=floor,
    )
    profile = build_source_profile(
        config,
        dry_air_mass_flow_kg_per_s=np.array([12.0]),
        outlet_temperature_celsius=np.array([61.0]),
        outlet_humidity_kg_per_kg=np.array([0.019]),
    )
    for factor, feasible in ((0.98, True), (1.03, False)):
        duty = factor * q * 1000.0
        cap = np.array([duty / (t_s - t_r)])
        got = max_duty_along(profile, np.zeros(1), np.array([t_r]), cap, a, np.array([duty]))
        if feasible:
            assert got[0] == pytest.approx(duty, rel=1e-6)
        else:
            assert got[0] < duty * (1 - 1e-4)


# --- heat pump ----------------------------------------------------------------------------------


def test_heat_pump_limits() -> None:
    rng = np.random.default_rng(RNG_SEED + 6)
    for _ in range(10):
        hot, cold = _random_streams(rng, 5, 5)
        h, c = Ramps.of_segments(hot), Ramps.of_segments(cold)
        table = problem_table(h, c, 20.0)
        if table.threshold:
            continue
        pinch = table.pinch_shifted
        e, u = heat_pump_limits(
            table, [pinch + 5.0 - 10.0, -100.0], [pinch - 5.0 + 10.0, 500.0], 20.0
        )
        assert e[1] == pytest.approx(table.cold_utility_kw)
        assert u[1] == pytest.approx(table.hot_utility_kw)
        # evaporator above / condenser below the pinch: nothing
        e2, u2 = heat_pump_limits(table, [pinch + 1.0 - 10.0], [pinch - 1.0 + 10.0], 20.0)
        assert e2[0] == pytest.approx(0.0, abs=1e-9)
        assert u2[0] == pytest.approx(0.0, abs=1e-9)
        # a condenser of duty U at T_c lowers the hot utility by exactly U (narrow stand-in)
        t_c = pinch + 10.0 + 12.0
        _, u3 = heat_pump_limits(table, [0.0], [t_c], 20.0)
        width = 1e-4
        extra = np.array([[t_c, t_c - width, u3[0] / width]])
        with_hp = problem_table(Ramps.of_segments(np.vstack([hot, extra])), c, 20.0)
        assert with_hp.hot_utility_kw == pytest.approx(table.hot_utility_kw - u3[0], abs=0.05)


# --- stream builder and the single-line limit -------------------------------------------------


def _day(label: str, exhaust: float) -> MeasuredDay:
    t = np.arange(96)
    return MeasuredDay(
        label=label,
        ambient_c=5.0 + 2.0 * np.sin(2 * np.pi * t / 96),
        process_air_nm3h=np.full(96, 30000.0) + 500.0 * np.cos(2 * np.pi * t / 96),
        supply_air_c=np.full(96, 84.0),
        exhaust_out_c=np.full(96, exhaust),
        spray_kg_h=np.full(96, 900.0),
        atomising_nm3h=np.full(96, 2000.0),
    )


@pytest.fixture(scope="module")
def builder() -> SiteStreamBuilder:
    days = {"product_1": _day("product_1", 60.0), "product_2": _day("product_2", 35.0)}
    return SiteStreamBuilder(load_site_config(), days)


def test_builder_generic_scaling_months_and_plant_c(builder: SiteStreamBuilder) -> None:
    config = load_site_config()
    base = SiteScenario(config.placeholders, n_generic=1, generic_day="product_2")
    two = SiteScenario(config.placeholders, n_generic=2, generic_day="product_2")
    s1, s2 = builder.build(base, 0), builder.build(two, 0)
    np.testing.assert_allclose(
        s2.hot["exhaust_generic"][:, :, 2], 2 * s1.hot["exhaust_generic"][:, :, 2]
    )
    np.testing.assert_allclose(s2.cold["air_generic"][:, :, 2], 2 * s1.cold["air_generic"][:, :, 2])
    assert s1.ambient_mean_celsius == pytest.approx(config.climate_monthly_mean_celsius[0])
    july = builder.build(base, 6)
    assert july.ambient_mean_celsius == pytest.approx(config.climate_monthly_mean_celsius[6])
    assert "hall" not in july.cold or float(np.max(july.cold["hall"][:, :, 2])) == 0.0
    none = builder.build(SiteScenario(config.placeholders, c_basis="none"), 0)
    assert not {"exhaust_c1", "exhaust_c2", "compressor_c"} & set(none.hot)
    assert not {"air_c1", "air_c2"} & set(none.cold)


def test_flue_gas_state() -> None:
    dry, y = flue_gas_state(1.15, 0.005)
    assert dry == pytest.approx(1.0 + 17.2 * 1.15 - 2.25)
    assert y == pytest.approx((2.25 + 17.2 * 1.15 * 0.005) / dry)
    assert 0.11 < y < 0.14


def test_single_line_reproduces_the_heat_network_pinch(builder: SiteStreamBuilder) -> None:
    """Limit case: plant A alone (no compressor, C, hall, water) = heatnetwork pinch per step."""
    config = load_site_config()
    scenario = SiteScenario(config.placeholders, c_basis="none").with_placeholders(
        compressor_station_kw=0.0, hall_kw=0.0, water_kw=0.0
    )
    snap = builder.build(scenario, 3)
    assert set(snap.hot) == {"exhaust_a"} and set(snap.cold) == {"air_a"}
    day = builder.days["product_1"]
    ph = config.placeholders
    y_out, m_dry, _ = exhaust_humidity(
        day,
        solid_fraction=ph.solid_fraction,
        relative_humidity=ph.relative_humidity,
        normal_density_kg_per_nm3=ph.normal_density_kg_per_nm3,
        gas=GAS,
    )
    for step in (0, 40, 77):
        hot = Ramps.of_segments(snap.segments(("exhaust_a",), step, hot=True))
        cold_seg = snap.segments(("air_a",), step, hot=False)
        cold = Ramps.of_segments(cold_seg)
        ours = problem_table(hot, cold, ph.minimum_pinch_k)
        source = HeatSourceConfig(
            "A", float(m_dry[step]), float(day.exhaust_out_c[step]), float(y_out[step])
        )
        cp_sink = cold_seg[0, 2]
        y_in = (
            cp_sink
            * 1000.0
            / (float(day.process_air_nm3h[step]) * ph.normal_density_kg_per_nm3 / 3600.0)
            - GAS.cp_dry_air_j_per_kg_k
        ) / GAS.cp_vapour_j_per_kg_k
        sink = HeatSinkConfig(
            "air", float(day.process_air_nm3h[step]) * ph.normal_density_kg_per_nm3 / 3600.0,
            y_in, float(cold_seg[0, 1]), float(cold_seg[0, 0]),
        )  # fmt: skip
        reference, _, _ = evaluate_pinch_target(
            (source,), (sink,), GAS,
            cold_temperature_celsius=ph.cold_temperature_celsius,
            minimum_pinch_k=ph.minimum_pinch_k,
        )  # fmt: skip
        assert ours.hot_utility_kw == pytest.approx(reference.minimum_hot_utility_kw, rel=1e-9)


def test_year_orderings_and_balance(builder: SiteStreamBuilder) -> None:
    config = load_site_config()
    scenario = SiteScenario(config.placeholders, n_generic=1, generic_day="product_2")
    year = evaluate_year(builder, config, scenario, zonal_levels=((70.0, 40.0), (50.0, 30.0)))
    summary = annual_summary(year, config)
    assert summary["balance_residual_max_kw"] < 1e-6
    assert summary["saving_kw"] >= -1e-9
    fossil_target = summary["fossil_target_kw"]
    for key, entry in summary["network"].items():
        assert entry["fossil_kw"] >= fossil_target - 1e-6, key
    assert summary["mean_kw"]["demand_kw"] == pytest.approx(
        summary["mean_kw"]["demand_air_a_kw"]
        + summary["mean_kw"]["demand_air_c_kw"]
        + summary["mean_kw"]["demand_air_generic_kw"]
        + summary["mean_kw"]["demand_low_kw"]
    )
