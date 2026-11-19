"""Tests of the load-following study (``twinlink_ml.load_following``, docs/43_load_following.md).

Fixed beforehand in the study document (section 12): with the spray rate pinned to the nominal
value, C0 reproduces R1 and C3 the R3 commands of Befund 105; at a constant price the daily
target is met and the schedule is flat where the cost is convex; no test-month information in
tuning; with a constant price the end of the day gives no advantage (tail, decision 10). Further:
the interval function and the programme are those of the headroom study when the spray rate is
held; the truth with three commands is the headroom truth at the nominal spray rate; the schedule
is exact; the headroom configuration is unchanged.
"""

from __future__ import annotations

import dataclasses
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")

from twinlink_ml.headroom.controllers import (  # noqa: E402
    DynamicOptimizer,
    QuasiStationaryController,
    SteadyStateOptimizer,
    nominal_disturbance,
)
from twinlink_ml.headroom.scenarios import Scenario, build_scenarios  # noqa: E402
from twinlink_ml.headroom.study import Solvers as HeadroomSolvers  # noqa: E402
from twinlink_ml.headroom.study import run_controller  # noqa: E402
from twinlink_ml.headroom.symbolic import interval_function  # noqa: E402
from twinlink_ml.headroom.truth import TruthPlant  # noqa: E402
from twinlink_ml.load_following.config import (  # noqa: E402
    LoadFollowingConfig,
    load_load_following_config,
)
from twinlink_ml.load_following.controllers import (  # noqa: E402
    ConstantThroughput,
    SetPointLaw,
)
from twinlink_ml.load_following.dynamic import LoadOptimizer, load_interval_function  # noqa: E402
from twinlink_ml.load_following.market import (  # noqa: E402
    DayScenario,
    Market,
    build_days,
    month_days,
    stratified_days,
)
from twinlink_ml.load_following.steady import (  # noqa: E402
    SteadyMap,
    cost_rate,
    optimal_levels,
    rule_levels,
)
from twinlink_ml.load_following.study import (  # noqa: E402
    RunSpec,
    Solvers,
    initial_conditions,
    run_days,
)
from twinlink_ml.load_following.truth import LoadTruthPlant  # noqa: E402
from twinlink_ml.storage_price.prices import year_clock  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BAND = (1.2, 1.8)


@pytest.fixture(scope="module")
def config() -> LoadFollowingConfig:
    return load_load_following_config()


@pytest.fixture(scope="module")
def short() -> LoadFollowingConfig:
    """Days of one hour plus the tail."""
    return load_load_following_config(day_hours=1)


@pytest.fixture(scope="module")
def market(config: LoadFollowingConfig) -> Market:
    """A synthetic tariff (the raw price file is not in the repository)."""
    clock = year_clock(2024)
    rng = np.random.default_rng(7)
    hour = clock.hour.astype(np.float64)
    tariff = 0.16 + 0.05 * np.sin(2.0 * np.pi * hour / 24.0) + 0.02 * rng.normal(size=clock.n_hours)
    tariff = np.maximum(tariff, config.tariff_floor_eur_per_kwh)
    return Market(
        tariff=tariff,
        clock=clock,
        reference_tariff=float(tariff.mean()),
        gas_cost_eur_per_kwh_fuel=config.gas_cost_eur_per_kwh_fuel,
        floored_hours=0,
        raw_sha256="synthetic",
    )


def _constant_day(config: LoadFollowingConfig, prices: np.ndarray) -> DayScenario:
    """A day with constant nominal ambient air and the given tariff per hour (day and tail)."""
    headroom = config.headroom
    hours, tail = config.day.hours, config.day.tail_hours
    n = round((hours + tail) * 3600.0 / headroom.scenarios.sample_time_s)
    d = nominal_disturbance(headroom, config.process)
    return DayScenario(
        name="test/constant",
        day=0,
        month=1,
        hours=hours,
        tail_hours=tail,
        sample_time_s=headroom.scenarios.sample_time_s,
        disturbances=np.repeat(d[:, None], n, axis=1),
        moisture_limit=np.full(n, headroom.line.product_moisture_limit_kg_per_kg),
        price_true=np.asarray(prices, dtype=np.float64),
        price_known=np.asarray(prices, dtype=np.float64),
        price_mode="test",
    )


# ------------------------------------------------------------------------------ same model


def test_interval_function_is_the_headroom_one_when_the_spray_rate_is_held(
    config: LoadFollowingConfig,
) -> None:
    headroom, process = config.headroom, config.process
    kwargs = {
        "interval_s": headroom.control.interval_s,
        "step_s": headroom.control.optimizer_step_s,
        "nozzle_pressure_bar": headroom.line.nozzle_pressure_bar,
        "sample_time_s": headroom.scenarios.sample_time_s,
    }
    reference = interval_function(process, **kwargs)
    ours = load_interval_function(process, **kwargs)
    rng = np.random.default_rng(3)
    steps = headroom.control.steps_per_interval
    for _ in range(5):
        x0 = np.array([2.0, rng.uniform(0.2, 0.3), rng.uniform(35.0, 50.0)])
        u, u_prev = rng.uniform([70.0, 0.8], [110.0, 1.5], (2, 2))
        spray = rng.uniform(1.2, 1.8)
        dist = np.tile(nominal_disturbance(headroom, process)[:, None], (1, steps))
        dist[0] += rng.normal(0.0, 2.0, steps)
        dist[3] = spray
        expected = reference(x0, u, u_prev, dist)
        got = ours(x0, np.append(u, spray), np.append(u_prev, spray), dist)
        for a, b in zip(expected, got[:4], strict=True):
            np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12, atol=1e-14)
        assert float(got[4]) == pytest.approx(spray * headroom.control.interval_s / 3600.0)


def test_programme_is_the_headroom_one_with_pinned_spray_rate_and_constant_price(
    config: LoadFollowingConfig,
) -> None:
    headroom, process = config.headroom, config.process
    n = 6
    limit = headroom.line.product_moisture_limit_kg_per_kg
    d = nominal_disturbance(headroom, process)
    start = SteadyStateOptimizer(headroom, process).solve(d, limit - 0.01)
    reference = DynamicOptimizer(headroom, process, n, warm_start=False, guess_start=True)
    ours = LoadOptimizer(headroom, process, n, expression="SX")
    nodes = n * reference.n_steps
    dist = np.tile(d[:, None], (1, nodes))
    dist[0] += np.linspace(0.0, 4.0, nodes)
    bound = np.full(nodes, limit - 0.004)
    guess = np.tile(start.command[:, None], (1, n))
    expected = reference.solve(start.state, start.command, dist, bound, guess_commands=guess)
    nominal = config.nominal_spray_rate
    box_low = np.tile(np.array([60.0, 0.8, nominal])[:, None], (1, n))
    box_high = np.tile(np.array([120.0, 2.5, nominal])[:, None], (1, n))
    got = ours.solve(
        start.state,
        np.append(start.command, nominal),
        dist,
        bound,
        np.ones(n),
        command_low=box_low,
        command_high=box_high,
        guess_commands=np.vstack([guess, np.full(n, nominal)]),
    )
    np.testing.assert_allclose(got[:2], expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(got[2], nominal)
    assert ours.last_objective == pytest.approx(reference.last_objective, rel=1e-6)


def test_programme_meets_the_production_target_and_shifts_to_the_cheap_half(
    config: LoadFollowingConfig,
) -> None:
    headroom, process = config.headroom, config.process
    n = 12
    limit = headroom.line.product_moisture_limit_kg_per_kg
    nominal = config.nominal_spray_rate
    d = nominal_disturbance(headroom, process)
    start = SteadyStateOptimizer(headroom, process).solve(d, limit - 0.01)
    opt = LoadOptimizer(headroom, process, n, expression="SX")
    nodes = n * opt.n_steps
    dist = np.tile(d[:, None], (1, nodes))
    bound = np.full(nodes, limit - 0.004)
    price = np.repeat([1.4, 0.6], n // 2)
    box_low = np.tile(np.array([60.0, 0.8, BAND[0]])[:, None], (1, n))
    box_high = np.tile(np.array([120.0, 2.5, BAND[1]])[:, None], (1, n))
    u_prev = np.append(start.command, nominal)
    guess = np.tile(u_prev[:, None], (1, n))
    target = nominal * n * headroom.control.interval_s / 3600.0
    plan = opt.solve(
        start.state,
        u_prev,
        dist,
        bound,
        price,
        command_low=box_low,
        command_high=box_high,
        guess_commands=guess,
        counted=np.ones(n),
        target_kg=target,
    )
    _, moisture, _, sprayed = opt.predict(start.state, u_prev, plan, dist)
    assert sprayed.sum() == pytest.approx(target, rel=1e-6)
    assert plan[2, : n // 2].mean() < nominal < plan[2, n // 2 :].mean()
    assert moisture.max() <= limit - 0.004 + 1e-6
    assert opt.last_objective < opt.penalised_objective(
        start.state, u_prev, guess, dist, bound, price
    )


def test_truth_with_three_commands_is_the_headroom_truth_at_the_nominal_spray_rate(
    short: LoadFollowingConfig,
) -> None:
    headroom, process = short.headroom, short.process
    day = _constant_day(short, np.full(2, 0.16))
    scenario = Scenario(
        name="test/constant/00",
        family="test",
        kind="constant",
        index=0,
        sample_time_s=day.sample_time_s,
        disturbances=day.disturbances,
        moisture_limit=day.moisture_limit,
    )
    solvers = Solvers(short, 0.16)
    states, commands = initial_conditions(solvers, [day])
    ours = LoadTruthPlant(process, [day], states, commands, interval_s=headroom.control.interval_s)
    reference = TruthPlant(
        process,
        [scenario],
        states,
        commands[:, :2],
        interval_s=headroom.control.interval_s,
        gas_heater_efficiency=headroom.control.gas_heater_efficiency,
    )
    rng = np.random.default_rng(5)
    for _ in range(ours.n_intervals):
        step = commands[:, :2] + rng.normal(0.0, [3.0, 0.02], (1, 2))
        step[:, 1] = np.maximum(step[:, 1], 0.8)
        ours.advance(np.hstack([step, commands[:, 2:]]))
        reference.advance(step)
    a, b = ours.result(), reference.result()
    np.testing.assert_array_equal(a.moisture, b.moisture)
    np.testing.assert_array_equal(a.energy_kwh_per_interval, b.energy_kwh_per_interval)
    rebook = process.energy.heater_efficiency / headroom.control.gas_heater_efficiency
    np.testing.assert_allclose(a.heater_kwh_per_interval * rebook, b.gas_kwh_per_interval)
    np.testing.assert_allclose(
        a.sprayed_kg_per_interval, short.nominal_spray_rate * headroom.control.interval_s / 3600.0
    )


# ------------------------------------------------------------------------------ reproduction


def test_c0_reproduces_r1_of_befund_105(config: LoadFollowingConfig) -> None:
    headroom, process = config.headroom, config.process
    limit = headroom.line.product_moisture_limit_kg_per_kg
    margin = 0.00664  # the R1 margin of the family "observed" (docs/24 section 8.1)
    scenario = build_scenarios(headroom, process, "observed")[3]
    spi = round(headroom.control.interval_s / headroom.scenarios.sample_time_s)
    r1 = QuasiStationaryController(SteadyStateOptimizer(headroom, process), margin)
    c0 = ConstantThroughput(
        SetPointLaw(SteadyMap(config), limit, margin), config.nominal_spray_rate
    )
    last = np.array([90.0, 1.0, config.nominal_spray_rate])
    state = np.zeros(4)
    for k in range(0, scenario.n_samples // spi, 7):
        d = scenario.disturbances[:, k * spi]
        expected = r1.command(k, state, last[:2], d, limit)
        got = c0.command(k, state, last, d, 0.0)
        np.testing.assert_allclose(got[:2], expected, rtol=1e-7, atol=1e-7)
        assert got[2] == config.nominal_spray_rate
    assert c0.info["failures"] == 0


def test_c3_reproduces_r3_of_befund_105_with_the_spray_rate_pinned(
    short: LoadFollowingConfig, market: Market
) -> None:
    headroom, process = short.headroom, short.process
    margin = 0.0027  # the R3 margin of the family "observed" (docs/24 section 8.1)
    day = build_days(short, market, [40], price_mode="flat")[0]
    nominal = short.nominal_spray_rate
    spec = RunSpec("C3", margin, margin, (nominal, nominal))
    record = run_days(spec, [day], Solvers(short, market.reference_tariff))[0]
    n_day = day.hours * short.intervals_per_hour
    samples = n_day * short.samples_per_interval
    scenario = Scenario(
        name="test/day/00",
        family="test",
        kind="day",
        index=0,
        sample_time_s=day.sample_time_s,
        disturbances=day.disturbances[:, :samples],
        moisture_limit=day.moisture_limit[:samples],
    )
    reference = run_controller("R3", margin, [scenario], HeadroomSolvers(headroom, process))[0]
    # same model, same optimiser, same inputs: the energies of the hour of the day agree
    assert record["energy_kwh_per_hour"][0] == pytest.approx(reference.energy_kwh, rel=1e-6)
    assert record["solver"]["failures"] == 0
    assert record["solver"]["plan_failures"] == 0
    assert record["sprayed_day_kg"] == pytest.approx(nominal * day.hours, rel=1e-9)


def test_headroom_configuration_is_the_one_of_befund_105(config: LoadFollowingConfig) -> None:
    artifact = REPO_ROOT / "artifacts" / "runs" / "dynamic_headroom_v2.json.gz"
    if not artifact.exists():
        pytest.skip("artifact of Befund 105 not present")
    with gzip.open(artifact, "rt", encoding="utf-8") as handle:
        recorded = json.load(handle)
    assert config.headroom.fingerprint == recorded["study_fingerprint"]
    assert config.process.fingerprint == recorded["process_config_fingerprint"]


# ------------------------------------------------------------------------------ map, schedule, rule


def test_steady_map_power_is_the_optimiser_power_and_the_gas_booking(
    config: LoadFollowingConfig,
) -> None:
    headroom, process = config.headroom, config.process
    limit = headroom.line.product_moisture_limit_kg_per_kg
    d = nominal_disturbance(headroom, process)
    steady_map = SteadyMap(config)
    optimizer = SteadyStateOptimizer(headroom, process)
    for spray in (1.2, 1.5, 1.8):
        point = steady_map.point(d, spray, limit - 0.005)
        assert point is not None
        dd = d.copy()
        dd[3] = spray
        assert point.total_kw == pytest.approx(
            optimizer.solve(dd, limit - 0.005).power_kw, rel=1e-9
        )
        assert 0.0 < point.heater_kw < point.total_kw
    total, heater = steady_map.curve(d, np.array([1.2, 1.5, 1.8]), limit - 0.005)
    assert np.all(np.diff(total) > 0.0)
    electric = cost_rate(total, heater, 0.2, "E", config)
    np.testing.assert_allclose(electric, 0.2 * total)
    rebook = process.energy.heater_efficiency / headroom.control.gas_heater_efficiency
    gas = cost_rate(total, heater, 0.2, "G", config)
    np.testing.assert_allclose(
        gas, 0.2 * (total - heater) + config.gas_cost_eur_per_kwh_fuel * heater * rebook
    )


def test_schedule_is_exact_against_brute_force() -> None:
    rng = np.random.default_rng(11)
    for _ in range(20):
        hours, levels = 4, 5
        cost = rng.uniform(0.0, 1.0, (hours, levels))
        cost[rng.uniform(size=cost.shape) < 0.1] = np.nan
        index_sum = int(rng.integers(0, hours * (levels - 1) + 1))
        best = np.inf
        for choice in itertools.product(range(levels), repeat=hours):
            if sum(choice) == index_sum:
                value = sum(cost[h, j] for h, j in enumerate(choice))
                if np.isfinite(value):
                    best = min(best, value)
        if not np.isfinite(best):
            with pytest.raises(ValueError, match="no admissible schedule"):
                optimal_levels(cost, index_sum)
            continue
        chosen = optimal_levels(cost, index_sum)
        assert chosen.sum() == index_sum
        assert cost[np.arange(hours), chosen].sum() == pytest.approx(best)


def test_schedule_is_flat_at_a_constant_price_where_the_cost_is_convex() -> None:
    levels = np.linspace(1.2, 1.8, 41)
    convex = 0.5 + 0.7 * levels + 0.9 * (levels - 1.2) ** 2
    cost = np.tile(0.16 * convex, (24, 1))
    chosen = optimal_levels(cost, 24 * 20)
    assert np.all(chosen == 20)
    concave = 0.5 + 0.7 * levels - 0.9 * (levels - 1.2) ** 2
    chosen = optimal_levels(np.tile(0.16 * concave, (24, 1)), 24 * 20)
    assert set(chosen.tolist()) == {0, 40}


def test_rule_levels_keep_the_mean_and_the_band() -> None:
    rng = np.random.default_rng(13)
    for _ in range(50):
        prices = rng.uniform(0.05, 0.4, 24)
        quantile = rng.uniform(0.05, 0.95)
        spread = rng.uniform(0.0, 0.6)
        levels = rule_levels(prices, quantile, spread, nominal=1.5, low=1.2, high=1.8)
        assert levels.mean() == pytest.approx(1.5)
        assert levels.min() >= 1.2 - 1e-12
        assert levels.max() <= 1.8 + 1e-12
        if levels.max() > levels.min():
            assert prices[levels == levels.max()].max() <= prices[levels == levels.min()].min()


# ------------------------------------------------------------------------------ prices and days


def test_price_information_and_day_sets(config: LoadFollowingConfig, market: Market) -> None:
    split = config.prices
    assert not set(split.test_months) & set(split.training_months)
    training = stratified_days(
        market, split.training_months, config.day.training_per_month_by_tercile, config.day.seed
    )
    test = stratified_days(
        market, split.test_months, config.day.test_per_month_by_tercile, config.day.seed
    )
    assert len(training) == 5 * len(split.training_months)
    assert len(test) == 10 * len(split.test_months)
    for days, months in ((training, split.training_months), (test, split.test_months)):
        allowed = {d for m in months for d in month_days(market.clock, m)}
        assert set(days) <= allowed
    again = stratified_days(
        market, split.test_months, config.day.test_per_month_by_tercile, config.day.seed
    )
    assert test == again

    day = build_days(config, market, [40])[0]
    start = 40 * 24
    np.testing.assert_array_equal(day.price_true[:24], market.tariff[start : start + 24])
    np.testing.assert_array_equal(day.price_known[:24], day.price_true[:24])
    assert day.price_true[24] == market.tariff[start + 24]  # the tail is booked at its true price
    assert day.price_known[24] == market.tariff[start]  # ... and forecast from the day itself
    assert day.n_samples == 25 * 720
    assert np.all(day.disturbances[3] == config.nominal_spray_rate)
    flat = build_days(config, market, [40], price_mode="flat")[0]
    np.testing.assert_array_equal(flat.price_true, np.full(25, market.reference_tariff))
    np.testing.assert_array_equal(flat.disturbances, day.disturbances)


def test_real_price_file_split_if_present(config: LoadFollowingConfig) -> None:
    from twinlink_ml.load_following.market import build_market

    if not (REPO_ROOT / config.prices.prices.raw_file).exists():
        pytest.skip("raw price file not present (local/)")
    real = build_market(config, REPO_ROOT)
    assert real.tariff.min() >= config.tariff_floor_eur_per_kwh
    train = np.isin(real.clock.month, config.prices.training_months)
    assert real.reference_tariff == pytest.approx(float(real.tariff[train].mean()))


# ------------------------------------------------------------------------------ whole runs


def test_cheap_controllers_meet_the_target_and_the_limit(
    short: LoadFollowingConfig, market: Market
) -> None:
    two_hours = load_load_following_config(day_hours=2)
    day = _constant_day(two_hours, np.array([0.25, 0.10, 0.16]))
    solvers = Solvers(two_hours, market.reference_tariff)
    target = two_hours.nominal_spray_rate * 2
    costs = {}
    for kind in ("C0", "C1", "C2"):
        spec = RunSpec(kind, 0.01, 0.01, BAND, rule_quantile=0.5, rule_spread_kg_per_h=0.6)
        record = run_days(spec, [day], solvers)[0]
        assert record["sprayed_day_kg"] == pytest.approx(target, rel=1e-3)
        assert record["max_violation"] <= two_hours.headroom.control.violation_tolerance_kg_per_kg
        assert record["sprayed_kg_per_hour"][2] == pytest.approx(
            two_hours.nominal_spray_rate, rel=1e-3
        )
        costs[kind] = record["cost_e"]
    # the expensive hour first: both the rule and the schedule shift to the cheap hour
    assert costs["C1"] < costs["C0"]
    assert costs["C2"] < costs["C0"]
    assert short.day.hours == 1


def test_no_advantage_from_the_end_of_the_day_at_a_constant_price(
    market: Market,
) -> None:
    """Decision 10: with the tail a programme cannot gain by how it leaves the day.

    Constant price, constant ambient air, spray rate pinned: what is left to a perfect preview is
    the start (every run starts at the common state with the start margin, above its own margin)
    and the end of the day. Without the tail the programme saves energy in the LAST hour of the
    day (it stops drying before the end); with the tail the last hour and the tail cost the same
    as with C0.
    """
    two_hours = load_load_following_config(day_hours=2)
    no_tail = dataclasses.replace(two_hours, day=dataclasses.replace(two_hours.day, tail_hours=0))
    nominal = two_hours.nominal_spray_rate
    margin = 0.005
    pinned = (nominal, nominal)
    last = {}
    for name, cfg in (("tail", two_hours), ("no tail", no_tail)):
        day = _constant_day(cfg, np.full(cfg.day.hours + cfg.day.tail_hours, 0.16))
        solvers = Solvers(cfg, market.reference_tariff)
        c0 = run_days(RunSpec("C0", margin, margin, pinned), [day], solvers)[0]
        c4 = run_days(RunSpec("C4", margin, margin, pinned), [day], solvers)[0]
        assert c4["max_violation"] <= cfg.headroom.control.violation_tolerance_kg_per_kg
        last[name] = (c0["energy_kwh_per_hour"], c4["energy_kwh_per_hour"])
    c0_hours, c4_hours = last["tail"]
    assert c4_hours[1] == pytest.approx(c0_hours[1], rel=1e-3)  # last hour of the day
    assert c4_hours[2] == pytest.approx(c0_hours[2], rel=1e-3)  # the tail
    c0_hours, c4_hours = last["no tail"]
    assert (c0_hours[1] - c4_hours[1]) / c0_hours[1] > 3e-3  # the end effect the tail removes
