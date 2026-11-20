"""Tests of the switch ``cold_starts`` and of the order-dependence measurement.

Study document ``docs/NN_order_dependence.md``. Required by the task: with the switch off the
earlier results are bit-equal (the default path of ``SteadyStateOptimizer.solve`` is compared
with a copy of the algorithm of Befund 105 kept in this file, and the caches keep their rounded
keys); with the switch on a steady optimum is a function of its query alone (independent of the
order, of the history of the optimiser and of the composition of a run).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")

from twinlink_ml.headroom.config import (  # noqa: E402
    HeadroomStudyConfig,
    load_headroom_config,
    study_process_config,
)
from twinlink_ml.headroom.controllers import (  # noqa: E402
    COLD_START_IMPROVEMENT,
    QuasiStationaryController,
    SteadyPoint,
    SteadyStateOptimizer,
    nominal_disturbance,
)
from twinlink_ml.headroom.scenarios import Scenario  # noqa: E402
from twinlink_ml.headroom.study import Solvers as HeadroomSolvers  # noqa: E402
from twinlink_ml.headroom.study import run_controller  # noqa: E402
from twinlink_ml.load_following.config import (  # noqa: E402
    LoadFollowingConfig,
    load_load_following_config,
)
from twinlink_ml.load_following.market import DayScenario  # noqa: E402
from twinlink_ml.load_following.steady import SteadyMap  # noqa: E402
from twinlink_ml.load_following.study import RunSpec, run_days  # noqa: E402
from twinlink_ml.load_following.study import Solvers as LoadSolvers  # noqa: E402
from twinlink_ml.order_dependence.keynumbers import (  # noqa: E402
    cost_per_kg,
    g10_median,
    gain,
    median_gap_to_twin,
    reading,
    relative_change,
)
from twinlink_ml.order_dependence.queries import (  # noqa: E402
    Query,
    RecordingOptimizer,
    measure,
    order_indices,
    summarise,
    warm_pass,
)
from twinlink_ml.order_dependence.runs import chunks_of, compare_records  # noqa: E402
from twinlink_ml.order_dependence.settings import OrderDependenceSettings  # noqa: E402
from twinlink_ml.order_dependence.statistics import pool_units, query_statistics  # noqa: E402
from twinlink_ml.simulation import ProcessConfig  # noqa: E402

BAND = (1.2, 1.8)
LIMIT = 0.15


@pytest.fixture(scope="module")
def study() -> HeadroomStudyConfig:
    return load_headroom_config()


@pytest.fixture(scope="module")
def process(study: HeadroomStudyConfig) -> ProcessConfig:
    return study_process_config(study)


@pytest.fixture(scope="module")
def short() -> LoadFollowingConfig:
    """Days of one hour plus the tail."""
    return load_load_following_config(day_hours=1)


def _queries(study: HeadroomStudyConfig, process: ProcessConfig, n: int, seed: int) -> list[Query]:
    """Queries spread over ambient air, solid fraction, spray rate and bound."""
    rng = np.random.default_rng(seed)
    nominal = nominal_disturbance(study, process)
    out = []
    for _ in range(n):
        d = nominal.copy()
        d[0] += rng.uniform(-8.0, 8.0)
        d[1] = rng.uniform(0.3, 0.7)
        d[2] = rng.uniform(0.27, 0.33)
        d[3] = rng.uniform(*BAND)
        d[4] = rng.normal(0.0, 0.5)
        out.append(Query(d, LIMIT - rng.uniform(0.0, 0.02)))
    return out


def _same(a: SteadyPoint, b: SteadyPoint) -> bool:
    return bool(
        a.power_kw == b.power_kw
        and np.array_equal(a.command, b.command)
        and np.array_equal(a.state, b.state)
    )


class _Befund105Chain:
    """The solve of Befund 105, copied: warm start from the previous solution, one retry."""

    def __init__(self, optimizer: SteadyStateOptimizer) -> None:
        self._opt = optimizer
        self._guess = np.array([0.25, 45.0, 90.0, optimizer._lbx[3]])

    def solve(self, disturbance: np.ndarray, moisture_bound: float) -> SteadyPoint:
        opt = self._opt
        params = np.append(np.asarray(disturbance, dtype=np.float64), moisture_bound)
        box = {"lbx": opt._lbx, "ubx": opt._ubx, "lbg": opt._lbg, "ubg": opt._ubg}
        solution = opt._solver(x0=self._guess, p=params, **box)
        if not opt._solver.stats()["success"]:
            solution = opt._solver(x0=np.array([0.25, 45.0, 90.0, opt._lbx[3]]), p=params, **box)
            if not opt._solver.stats()["success"]:
                raise RuntimeError("steady optimum failed")
        w = np.asarray(solution["x"]).ravel()
        self._guess = w
        return SteadyPoint(
            command=w[2:4].copy(),
            state=np.array([opt._solid_mass, w[0], w[1]]),
            power_kw=float(solution["f"]),
        )


# ------------------------------------------------------------------------------ switch off


def test_default_is_the_warm_start_of_befund_105_bit_for_bit(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    default = SteadyStateOptimizer(study, process)
    assert not default.order_independent
    assert default.start_points.shape == (0, 4)
    reference = _Befund105Chain(SteadyStateOptimizer(study, process))
    for query in _queries(study, process, 60, seed=1):
        assert _same(
            default.solve(query.disturbance, query.bound),
            reference.solve(query.disturbance, query.bound),
        )


def test_solve_from_leaves_the_warm_start_alone(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    queries = _queries(study, process, 12, seed=2)
    plain = SteadyStateOptimizer(study, process)
    probed = SteadyStateOptimizer(study, process)
    for query in queries:
        point, status = probed.solve_from(
            query.disturbance, query.bound, np.array([0.5, 60.0, 110.0, 2.0])
        )
        assert point is None or status
        assert _same(
            plain.solve(query.disturbance, query.bound),
            probed.solve(query.disturbance, query.bound),
        )


def test_caches_keep_their_rounded_keys_without_the_switch(
    study: HeadroomStudyConfig, process: ProcessConfig, short: LoadFollowingConfig
) -> None:
    d = nominal_disturbance(study, process)
    near = d.copy()
    near[0] += 1.0e-13
    steady_map = SteadyMap(short)
    first = steady_map.point(d, 1.5, LIMIT)
    assert steady_map.point(near, 1.5, LIMIT) is first
    assert len(steady_map._cache) == 1
    controller = QuasiStationaryController(SteadyStateOptimizer(study, process), 0.005)
    state, last = np.zeros(4), np.zeros(2)
    command = controller.command(0, state, last, d, LIMIT)
    assert controller.command(1, state, last, near, LIMIT) is command
    assert len(controller._cache) == 1


def test_solvers_are_unchanged_by_default_and_pass_the_switch(
    study: HeadroomStudyConfig, process: ProcessConfig, short: LoadFollowingConfig
) -> None:
    assert not HeadroomSolvers(study, process).steady.order_independent
    assert not LoadSolvers(short, 0.16).steady._optimizer.order_independent
    cold = HeadroomSolvers(study, process, steady_cold_starts=3).steady
    assert cold.order_independent and cold.start_points.shape == (3, 4)
    cold_map = LoadSolvers(short, 0.16, steady_cold_starts=3).steady
    assert cold_map._optimizer.start_points.shape == (3, 4)
    with pytest.raises(ValueError, match="cold_starts"):
        SteadyStateOptimizer(study, process, cold_starts=-1)


# ------------------------------------------------------------------------------ switch on


def test_cold_multi_start_is_a_function_of_the_query_alone(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    queries = _queries(study, process, 30, seed=3)
    forward = SteadyStateOptimizer(study, process, cold_starts=4)
    backward = SteadyStateOptimizer(study, process, cold_starts=4)
    for query in _queries(study, process, 10, seed=4):  # another history
        backward.solve(query.disturbance, query.bound)
    a = warm_pass(forward, queries, range(len(queries)))
    b = warm_pass(backward, queries, range(len(queries) - 1, -1, -1))
    for x, y in zip(a, b, strict=True):
        assert x is not None and y is not None and _same(x, y)
    again = warm_pass(forward, queries, range(len(queries)))
    for x, y in zip(a, again, strict=True):
        assert x is not None and y is not None and _same(x, y)


def test_cold_multi_start_applies_the_incumbent_rule(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    optimizer = SteadyStateOptimizer(study, process, cold_starts=6)
    starts = optimizer.start_points
    assert np.array_equal(starts[0], [0.25, 45.0, 90.0, optimizer._lbx[3]])
    assert np.all(starts >= optimizer._lbx) and np.all(starts <= optimizer._ubx)
    same_seed = SteadyStateOptimizer(study, process, cold_starts=6).start_points
    assert np.array_equal(starts, same_seed)
    other_seed = SteadyStateOptimizer(study, process, cold_starts=6, start_seed=1).start_points
    assert not np.array_equal(starts[1:], other_seed[1:])
    for query in _queries(study, process, 15, seed=5):
        best: SteadyPoint | None = None
        best_strict = False
        for guess in starts:
            point, status = optimizer.solve_from(query.disturbance, query.bound, guess)
            if point is None:
                continue
            strict = status == "Solve_Succeeded"
            if best is None or (strict and not best_strict):
                best, best_strict = point, strict
            elif strict == best_strict and point.power_kw < best.power_kw * (
                1.0 - COLD_START_IMPROVEMENT
            ):
                best = point
        assert best is not None
        chosen = optimizer.solve(query.disturbance, query.bound)
        assert _same(chosen, best)
        first, _ = optimizer.solve_from(query.disturbance, query.bound, starts[0])
        if first is not None:
            assert chosen.power_kw <= first.power_kw


def test_one_cold_start_is_the_default_guess(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    optimizer = SteadyStateOptimizer(study, process, cold_starts=1)
    plain = SteadyStateOptimizer(study, process)
    query = _queries(study, process, 1, seed=6)[0]
    assert _same(
        optimizer.solve(query.disturbance, query.bound),
        plain.solve(query.disturbance, query.bound),  # the first solve of a fresh optimiser
    )


def test_caches_use_the_unrounded_query_with_the_switch(
    study: HeadroomStudyConfig, process: ProcessConfig, short: LoadFollowingConfig
) -> None:
    d = nominal_disturbance(study, process)
    near = d.copy()
    near[0] += 1.0e-13
    optimizer = SteadyStateOptimizer(study, process, cold_starts=2)
    steady_map = SteadyMap(short, optimizer=optimizer)
    first = steady_map.point(d, 1.5, LIMIT)
    assert steady_map.point(d, 1.5, LIMIT) is first
    assert steady_map.point(near, 1.5, LIMIT) is not first
    assert len(steady_map._cache) == 2
    controller = QuasiStationaryController(optimizer, 0.005)
    state, last = np.zeros(4), np.zeros(2)
    command = controller.command(0, state, last, d, LIMIT)
    assert controller.command(1, state, last, d, LIMIT) is command
    assert controller.command(2, state, last, near, LIMIT) is not command
    assert len(controller._cache) == 2


def _scenario(study: HeadroomStudyConfig, process: ProcessConfig, index: int) -> Scenario:
    """Half an hour with a step of the ambient temperature that differs per scenario."""
    n = round(1800.0 / study.scenarios.sample_time_s)
    d = np.repeat(nominal_disturbance(study, process)[:, None], n, axis=1)
    d[0, n // 2 :] += 2.0 + index
    d[3] *= 1.0 + 0.01 * index
    return Scenario(
        name=f"test/step/{index:02d}",
        family="test",
        kind="step",
        index=index,
        sample_time_s=study.scenarios.sample_time_s,
        disturbances=d,
        moisture_limit=np.full(n, study.line.product_moisture_limit_kg_per_kg),
    )


def test_r1_with_the_switch_does_not_depend_on_what_was_run_before(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    scenarios = [_scenario(study, process, i) for i in range(3)]
    together = run_controller(
        "R1", 0.005, scenarios, HeadroomSolvers(study, process, steady_cold_starts=3)
    )
    used = HeadroomSolvers(study, process, steady_cold_starts=3)
    run_controller("R1", 0.01, scenarios[::-1], used)
    alone = [run_controller("R1", 0.005, [s], used)[0] for s in scenarios[::-1]][::-1]
    for a, b in zip(together, alone, strict=True):
        assert a.commands == b.commands
        assert a.energy_kwh == b.energy_kwh
        assert a.max_violation == b.max_violation


def _day(config: LoadFollowingConfig, index: int) -> DayScenario:
    """A day with a step of the ambient temperature and two price levels."""
    headroom = config.headroom
    hours, tail = config.day.hours, config.day.tail_hours
    n = round((hours + tail) * 3600.0 / headroom.scenarios.sample_time_s)
    d = np.repeat(nominal_disturbance(headroom, config.process)[:, None], n, axis=1)
    d[0, n // 3 :] += 1.0 + index
    prices = np.where(np.arange(hours + tail) % 2 == 0, 0.10, 0.20 + 0.01 * index)
    return DayScenario(
        name=f"test/day{index}",
        day=index,
        month=1,
        hours=hours,
        tail_hours=tail,
        sample_time_s=headroom.scenarios.sample_time_s,
        disturbances=d,
        moisture_limit=np.full(n, headroom.line.product_moisture_limit_kg_per_kg),
        price_true=prices,
        price_known=prices,
        price_mode="test",
    )


def test_c2_with_the_switch_does_not_depend_on_what_was_run_before(
    short: LoadFollowingConfig,
) -> None:
    days = [_day(short, i) for i in range(2)]
    spec = RunSpec(kind="C2", margin=0.006, tail_margin=0.005, band=BAND)
    together = run_days(spec, days, LoadSolvers(short, 0.16, steady_cold_starts=3))
    used = LoadSolvers(short, 0.16, steady_cold_starts=3)
    run_days(RunSpec(kind="C0", margin=0.01, tail_margin=0.01, band=BAND), days[::-1], used)
    alone = [run_days(spec, [day], used)[0] for day in days[::-1]][::-1]
    assert compare_records(
        [{**r, "unit": r["day_of_year"]} for r in together],
        [{**r, "unit": r["day_of_year"]} for r in alone],
    )["equal_units"] == len(days)


# ------------------------------------------------------------------------------ measurement


def test_recorded_queries_are_reproduced_and_measured(
    study: HeadroomStudyConfig, process: ProcessConfig
) -> None:
    settings = OrderDependenceSettings(cold_starts=3, reference_starts=5, permutation_seeds=(1,))
    recorder = RecordingOptimizer(study, process)
    scenario = _scenario(study, process, 1)
    run_controller("R1", 0.005, [scenario], HeadroomSolvers(study, process, _steady=recorder))
    queries = recorder.queries
    assert len(queries) == len(recorder.results) >= 2  # initial condition and two levels

    def factory(cold_starts: int, seed: int) -> SteadyStateOptimizer:
        return SteadyStateOptimizer(study, process, cold_starts=cold_starts, start_seed=seed)

    measured = measure(queries, factory, settings)
    n = len(queries)
    assert measured["warm"].shape == (3, n, 3)
    assert measured["multi"].shape == measured["cold"].shape == (n, 3)
    summary = summarise(measured, recorder.results, settings)
    assert summary["natural_reproduced"]
    assert summary["orders"] == ["natural", "reversed", "permutation_1"]
    assert summary["n_queries"] == n
    assert all(len(column) == n for column in summary["rel_warm"])
    assert all(1 <= v <= 3 for v in summary["distinct_bits"])
    assert all(v >= 1 for v in summary["reference_optima"])
    # the reference contains better or equal points than any single start within the tolerance
    pooled = pool_units([summary])
    stats = query_statistics(pooled, settings)
    assert stats["n_queries"] == n and stats["natural_pass_reproduced_in_all_units"]
    assert stats["failed"]["warm_natural"] == 0


def test_order_indices_are_permutations_and_seeded() -> None:
    settings = OrderDependenceSettings()
    orders = order_indices(17, settings)
    assert list(orders) == list(settings.orders)
    assert all(sorted(order.tolist()) == list(range(17)) for order in orders.values())
    assert np.array_equal(orders["reversed"], np.arange(17)[::-1])
    again = order_indices(17, settings)
    assert all(np.array_equal(orders[name], again[name]) for name in orders)


def test_query_statistics_count_the_shares() -> None:
    settings = OrderDependenceSettings(permutation_seeds=(1,), thresholds=(1.0e-6, 1.0e-3))
    unit = {
        "orders": list(settings.orders),
        "natural_reproduced": True,
        "rel_warm": [
            [0.0, 2.0e-3, -5.0e-4, None],
            [0.0, 0.0, -5.0e-4, 1.0e-5],
            [0.0, 0.0, 0.0, 1.0e-5],
        ],
        "rel_cold": [0.0, 2.0e-3, 0.0, 0.0],
        "rel_reference": [0.0, 0.0, -2.0e-3, 0.0],
        "d_inlet_k": [0.0, 12.0, -1.0, None],
        "d_gas_m_per_s": [0.0, -0.4, 0.0, None],
        "d_inlet_orders_k": [0.0, 12.0, 1.0, 0.0],
        "d_gas_orders_m_per_s": [0.0, 0.4, 0.0, 0.0],
        "distinct_bits": [1, 2, 2, 1],
        "reference_optima": [1, 2, 2, 1],
        "reference_failed": [0, 0, 1, 0],
        "reference_spread": [0.0, 2.0e-3, 2.0e-3, 0.0],
    }
    failed_unit = {"error": "Traceback", "kind": "C0", "unit": 3}
    stats = query_statistics(pool_units([unit, failed_unit]), settings)
    assert stats["n_queries"] == 4 and stats["n_units"] == 1
    assert stats["failed"]["warm_natural"] == 1 and stats["failed"]["warm_any_order"] == 1
    assert stats["share_bit_distinct_between_orders"] == pytest.approx(0.5)
    fine, coarse = stats["by_threshold"]["1e-06"], stats["by_threshold"]["0.001"]
    assert fine["share_warm_natural_worse"] == pytest.approx(1 / 3)
    assert fine["share_warm_natural_better"] == pytest.approx(1 / 3)
    assert fine["share_differing_between_orders"] == pytest.approx(2 / 3)
    assert coarse["share_differing_between_orders"] == pytest.approx(1 / 3)
    assert fine["share_cold_single_start_worse"] == pytest.approx(0.25)
    assert fine["share_reference_better_than_multi_start"] == pytest.approx(0.25)
    assert fine["worse_warm_natural_relative_power"]["max"] == pytest.approx(2.0e-3)
    assert fine["worse_warm_natural_abs_d_inlet_k"]["max"] == pytest.approx(12.0)
    assert coarse["share_warm_natural_better"] == 0.0
    assert stats["local_optima"]["share_of_queries_with_several"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no unit"):
        pool_units([failed_unit])


def test_key_numbers_and_reading() -> None:
    r0 = [{"scenario": "a", "energy_kwh": 100.0}, {"scenario": "b", "energy_kwh": 200.0}]
    r1 = [{"scenario": "b", "energy_kwh": 150.0}, {"scenario": "a", "energy_kwh": 90.0}]
    assert g10_median(r0, r1) == pytest.approx(0.175)
    with pytest.raises(ValueError, match="different scenarios"):
        g10_median(r0, r1[:1])
    days: list[dict[str, Any]] = [
        {"cost_e": 10.0, "product_dry_kg": 20.0, "max_violation": 0.0},
        {"cost_e": 14.0, "product_dry_kg": 28.0, "max_violation": 1.0e-5},
        {"error": "x", "max_violation": None},
    ]
    assert cost_per_kg(days) == pytest.approx(0.5)
    assert gain(0.5, 0.49) == pytest.approx(0.02)
    assert median_gap_to_twin({"1": 0.412, "2": 0.408, "3": 0.42}, 0.4) == pytest.approx(0.03)
    assert relative_change(0.0893, 0.0900) == pytest.approx(0.7 / 89.3)
    assert reading({"K1/held": 0.10, "K2/held": 0.03}, 0.10) == "O0"  # the boundary belongs to O0
    assert reading({"K1/held": 0.02, "K2/searched": 0.1001}, 0.10) == "O1"
    with pytest.raises(ValueError, match="no key number"):
        reading({}, 0.10)


def test_chunks_and_record_comparison() -> None:
    ids = list(range(10))
    assert chunks_of(ids, 4, "a") == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert chunks_of(ids, 4, "b") == [[9, 8], [7, 6], [5, 4], [3, 2], [1, 0]]
    assert chunks_of([], 4, "a") == []
    with pytest.raises(ValueError, match="layout"):
        chunks_of(ids, 4, "c")
    a = [{"unit": 1, "cost_e": 2.0, "decide_s": 0.1}, {"unit": 2, "cost_e": 3.0, "decide_s": 0.2}]
    b = [{"unit": 1, "cost_e": 2.0, "decide_s": 0.3}, {"unit": 2, "cost_e": 3.5, "decide_s": 0.2}]
    out = compare_records(a, b)
    assert out["equal_units"] == 1 and out["differing_units"] == [2]
    assert out["largest_abs_difference"] == {"cost_e": 0.5}
    with pytest.raises(ValueError, match="different units"):
        compare_records(a, b[::-1])
