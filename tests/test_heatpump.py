"""Tests of the heat-pump study: COP, pseudo-streams, pinch rule, dispatch, storage LP, rules."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.optimize import linprog

from twinlink_ml.heatnetwork.pinch import StreamSegment, compute_pinch_target
from twinlink_ml.heatpump import (
    Capital,
    Economics,
    HeatPumpSpec,
    Signal,
    carnot_cop,
    classify_placement,
    co2_criterion,
    cost_criterion,
    coupled_pieces,
    daily_shape,
    design_pairs,
    dispatch_steps,
    evaluate_stage_a,
    evaluate_stage_b,
    grand_composite_kw,
    heat_pump_cop,
    heat_pump_lines,
    is_realistic,
    pinch_summary,
    signal_series,
    simulate_rule,
    solve_foresight,
    step_optimum,
    trailing_mean,
)
from twinlink_ml.storage_timeslice import MeasuredDay, Placeholders, Scenario, StreamFactory
from twinlink_ml.storage_timeslice.profiles import Clock
from twinlink_ml.storage_timeslice.table import cascade_profile, direct_hot_utility_kw

DT_H = 0.25
WIDTH = 1e-4  # narrow stand-in for an isothermal stream (StreamSegment needs high > low)


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
        high = rng.uniform(30, 100)
        hot.append((high, high - rng.uniform(5, 50), rng.uniform(0.5, 12)))
    cold = []
    for _ in range(rng.integers(1, 5)):
        low = rng.uniform(0, 50)
        cold.append((low + rng.uniform(10, 70), low, rng.uniform(0.5, 12)))
    return hot, cold


def _explicit_utility(
    hot_rows: list[tuple[float, float, float]],
    cold_rows: list[tuple[float, float, float]],
    dt_min: float,
    cold_iso: list[tuple[float, float]],
    hot_iso: list[tuple[float, float]],
) -> float:
    """Problem table with isothermal streams (shifted temperature, duty) added explicitly."""
    hot_seg = [StreamSegment("h", True, a, b, c) for a, b, c in hot_rows]
    cold_seg = [StreamSegment("c", False, a, b, c) for a, b, c in cold_rows]
    for t_shift, duty in cold_iso:
        if duty > 0:
            real = t_shift - dt_min / 2.0
            cold_seg.append(StreamSegment("ci", False, real, real - WIDTH, duty / WIDTH))
    for t_shift, duty in hot_iso:
        if duty > 0:
            real = t_shift + dt_min / 2.0
            hot_seg.append(StreamSegment("hi", True, real + WIDTH, real, duty / WIDTH))
    return compute_pinch_target(hot_seg, cold_seg, minimum_pinch_k=dt_min).minimum_hot_utility_kw


# ------------------------------------------------------------------------------------- COP


def test_cop_model_and_design_grid() -> None:
    assert carnot_cop(20.0, 80.0) == pytest.approx(353.15 / 60.0)
    spec = HeatPumpSpec(quality=0.5, cop_max=8.0)
    assert heat_pump_cop(20.0, 80.0, spec) == pytest.approx(0.5 * 353.15 / 60.0)
    assert heat_pump_cop(70.0, 90.0, spec) == pytest.approx(8.0)  # capped
    with pytest.raises(ValueError):
        carnot_cop(50.0, 50.0)
    pairs = design_pairs(spec, minimum_pinch_k=20.0)
    assert pairs
    for p in pairs:
        assert 20.0 - 1e-9 <= p.lift_k <= 60.0 + 1e-9
        assert p.condensing_celsius <= 90.0
        # approach dT_min/2: pseudo-temperatures equal the refrigerant temperatures
        assert p.evaporator_shifted == pytest.approx(p.evaporating_celsius)
        assert p.condenser_shifted == pytest.approx(p.condensing_celsius)
    wide = design_pairs(replace(spec, approach_k=20.0), minimum_pinch_k=20.0)
    assert wide[0].evaporator_shifted == pytest.approx(wide[0].evaporating_celsius + 10.0)
    assert wide[0].condenser_shifted == pytest.approx(wide[0].condensing_celsius - 10.0)


def test_criteria_and_breakeven() -> None:
    eco = Economics()
    co2 = co2_criterion("co2", eco)
    assert co2.heat_value_per_kwh == pytest.approx(0.201 / 0.85)
    assert co2.breakeven_cop == pytest.approx(0.366 * 0.85 / 0.201)
    cost = cost_criterion("cost", eco)
    assert cost.heat_value_per_kwh == pytest.approx((0.040 + 55.0 * 0.201 / 1000.0) / 0.85)
    assert cost.breakeven_cop == pytest.approx(0.16 / cost.heat_value_per_kwh)
    assert Capital().annuity_factor == pytest.approx(0.06 / (1 - 1.06**-15))
    assert is_realistic(spec=HeatPumpSpec(), economics=eco, minimum_pinch_k=20.0)
    assert not is_realistic(spec=HeatPumpSpec(quality=0.6), economics=eco, minimum_pinch_k=20.0)
    assert is_realistic(
        spec=HeatPumpSpec(),
        economics=eco,
        minimum_pinch_k=20.0,
        criterion_kind="cost",
        signal=Signal(amplitude=0.6),
    )
    assert not is_realistic(
        spec=HeatPumpSpec(),
        economics=eco,
        minimum_pinch_k=20.0,
        criterion_kind="co2",
        signal=Signal(amplitude=0.6),
    )


# ------------------------------------------------------------------ pseudo-streams, pinch rule


def test_heat_pump_lines_equal_explicit_isothermal_streams() -> None:
    rng = np.random.default_rng(3)
    for _ in range(30):
        hot_rows, cold_rows = _random_streams(rng)
        dt_min = float(rng.choice([10.0, 20.0]))
        te = float(rng.uniform(5, 50))
        tk = te + float(rng.uniform(20, 60))
        cop = float(rng.uniform(1.5, 6))
        cascade = cascade_profile(
            _pad([hot_rows]),
            _pad([cold_rows]),
            minimum_pinch_k=dt_min,
            extra_points=np.array([te, tk]),
        )
        lines = heat_pump_lines(cascade, te, tk, cop)
        q = float(rng.uniform(0, 300))
        ours = lines.utility_kw(np.array([q]))[0]
        ref = _explicit_utility(hot_rows, cold_rows, dt_min, [(te, q * (1 - 1 / cop))], [(tk, q)])
        assert ours == pytest.approx(ref, abs=0.05, rel=1e-3)


def _pinch_case() -> tuple[np.ndarray, np.ndarray]:
    """Hot 70 -> 20 (CP 10), cold 10 -> 40 (CP 5) and 40 -> 80 (CP 15), dT_min 10.

    Shifted: hot 65 -> 15, cold 15 -> 45 and 45 -> 85; cascade 0, -300, -400, -250 at 85, 65, 45,
    15: Q_H = 400, Q_C = 150, single pinch at shifted 45.
    """
    hot = _pad([[(70.0, 20.0, 10.0)]])
    cold = _pad([[(40.0, 10.0, 5.0), (80.0, 40.0, 15.0)]])
    return hot, cold


def test_pinch_summary_equals_pinch_module_and_gcc() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        hot_rows, cold_rows = _random_streams(rng)
        dt_min = float(rng.choice([10.0, 20.0]))
        cascade = cascade_profile(
            _pad([hot_rows]), _pad([cold_rows]), minimum_pinch_k=dt_min, extra_points=np.array([])
        )
        ref = compute_pinch_target(
            [StreamSegment("h", True, a, b, c) for a, b, c in hot_rows],
            [StreamSegment("c", False, a, b, c) for a, b, c in cold_rows],
            minimum_pinch_k=dt_min,
        )
        summ = pinch_summary(cascade)
        assert summ.hot_utility_kw[0] == pytest.approx(ref.minimum_hot_utility_kw, abs=1e-8)
        assert summ.cold_utility_kw[0] == pytest.approx(ref.minimum_cold_utility_kw, abs=1e-8)
        gcc = grand_composite_kw(cascade, np.array([-100.0, 500.0]))
        assert gcc[0, 1] == pytest.approx(ref.minimum_hot_utility_kw, abs=1e-8)
        assert gcc[0, 0] == pytest.approx(ref.minimum_cold_utility_kw, abs=1e-8)
    hot, cold = _pinch_case()
    summ = pinch_summary(
        cascade_profile(hot, cold, minimum_pinch_k=10.0, extra_points=np.array([]))
    )
    assert summ.hot_utility_kw[0] == pytest.approx(400.0)
    assert summ.cold_utility_kw[0] == pytest.approx(150.0)
    assert summ.pinch_low[0] == pytest.approx(45.0)
    assert summ.pinch_high[0] == pytest.approx(45.0)


def test_placement_rule_across_above_below() -> None:
    """Across the pinch dU = Q; above it dU = W; below it dU = 0 (for small Q)."""
    hot, cold = _pinch_case()
    cases = {"across": (20.0, 75.0), "above": (50.0, 80.0), "below": (10.0, 40.0)}
    cascade = cascade_profile(
        hot,
        cold,
        minimum_pinch_k=10.0,
        extra_points=np.array([t for p in cases.values() for t in p]),
    )
    summ = pinch_summary(cascade)
    u0 = direct_hot_utility_kw(cascade)[0]
    cop = 3.0
    q = 30.0
    for name, (te, tk) in cases.items():
        assert classify_placement(te, tk, summ)[0] == name
        u = heat_pump_lines(cascade, te, tk, cop).utility_kw(np.array([q]))[0]
        expected = {"across": q, "above": q / cop, "below": 0.0}[name]
        assert u0 - u == pytest.approx(expected, abs=1e-9)


# ------------------------------------------------------------------------------- dispatch


def test_dispatch_equals_brute_force() -> None:
    rng = np.random.default_rng(11)
    for _ in range(25):
        hot_rows, cold_rows = _random_streams(rng)
        te, tk = float(rng.uniform(5, 40)), float(rng.uniform(60, 90))
        cascade = cascade_profile(
            _pad([hot_rows]),
            _pad([cold_rows]),
            minimum_pinch_k=20.0,
            extra_points=np.array([te, tk]),
        )
        lines = heat_pump_lines(cascade, te, tk, float(rng.uniform(1.5, 5)))
        q_max, c_u, c_el = float(rng.uniform(10, 400)), 1.0, float(rng.uniform(0.2, 3))
        d = dispatch_steps(lines, q_max, c_u, c_el)
        grid = np.linspace(0, q_max, 4001)
        cost = c_u * lines.utility_kw(grid[:, None])[:, 0] + c_el * grid / lines.cop
        best = c_u * d.utility_kw[0] + c_el * d.work_kw[0]
        assert best <= np.min(cost) + 1e-9
        assert 0.0 <= d.heat_kw[0] <= q_max + 1e-12
        assert d.utility_kw[0] == pytest.approx(lines.utility_kw(d.heat_kw)[0])


def test_dispatch_broadcasts_over_sizes_and_criteria() -> None:
    hot, cold = _pinch_case()
    cascade = cascade_profile(hot, cold, minimum_pinch_k=10.0, extra_points=np.array([5.0, 75.0]))
    lines = heat_pump_lines(cascade, 5.0, 75.0, 3.0)
    d = dispatch_steps(
        lines,
        np.array([0.0, 30.0, 500.0])[:, None, None],
        np.array([1.0, 1.0])[None, :, None],
        np.array([0.5, 5.0])[None, :, None],
    )
    assert d.heat_kw.shape == (3, 2, 1)
    assert d.heat_kw[0, 0, 0] == 0.0
    assert d.heat_kw[1, 0, 0] == pytest.approx(30.0)  # cheap power: full output
    assert d.heat_kw[1, 1, 0] == 0.0  # COP 3 < price ratio 5: off


# ------------------------------------------------------------------- storage: pieces, LP, rules


def test_coupled_pieces_equal_explicit_streams() -> None:
    rng = np.random.default_rng(13)
    for _ in range(25):
        hot_rows, cold_rows = _random_streams(rng)
        dt_min = 20.0
        te, tk = float(rng.uniform(5, 40)), float(rng.uniform(60, 90))
        ts = float(rng.uniform(35, 70))
        tc, td = ts + 10.0, ts - 10.0
        cascade = cascade_profile(
            _pad([hot_rows]),
            _pad([cold_rows]),
            minimum_pinch_k=dt_min,
            extra_points=np.array([te, tk, tc, td]),
        )
        cop = float(rng.uniform(1.5, 5))
        pieces = coupled_pieces(
            cascade,
            evaporator_shifted=te,
            condenser_shifted=tk,
            cop=cop,
            charge_shifted=tc,
            discharge_shifted=td,
        )
        q, qc, qd = (float(x) for x in rng.uniform(0, 150, size=3))
        ours = pieces.utility_kw(np.array([q]), np.array([qc]), np.array([qd]))[0]
        ref = _explicit_utility(
            hot_rows, cold_rows, dt_min, [(te, q * (1 - 1 / cop)), (tc, qc)], [(tk, q), (td, qd)]
        )
        assert ours == pytest.approx(ref, abs=0.05, rel=1e-3)


def _random_pieces(rng: np.random.Generator, n_steps: int = 1):  # type: ignore[no-untyped-def]
    rows_h, rows_c = [], []
    for _ in range(n_steps):
        h, c = _random_streams(rng)
        rows_h.append(h)
        rows_c.append(c)
    te, tk = float(rng.uniform(5, 40)), float(rng.uniform(60, 90))
    ts = float(rng.uniform(35, 70))
    cascade = cascade_profile(
        _pad(rows_h),
        _pad(rows_c),
        minimum_pinch_k=20.0,
        extra_points=np.array([te, tk, ts + 10.0, ts - 10.0]),
    )
    return coupled_pieces(
        cascade,
        evaporator_shifted=te,
        condenser_shifted=tk,
        cop=float(rng.uniform(1.5, 5)),
        charge_shifted=ts + 10.0,
        discharge_shifted=ts - 10.0,
    )


def test_step_optimum_equals_linprog() -> None:
    rng = np.random.default_rng(17)
    for _ in range(60):
        pieces = _random_pieces(rng)
        q_max = float(rng.uniform(10, 300))
        cmax, dmax = float(rng.uniform(0, 300)), float(rng.uniform(0, 300))
        c_u, p, v = 1.0, float(rng.uniform(0.05, 1.5)), float(rng.uniform(0, 1.2))
        f, q, qc, qd = step_optimum(
            pieces,
            t=0,
            q_max_kw=q_max,
            charge_max_kw=cmax,
            discharge_max_kw=dmax,
            heat_value=c_u,
            electricity_per_heat=p,
            stored_value=v,
        )
        # reference: variables Q, qc, qd, U
        m = pieces.offset_kw[0]
        a_ub, b_ub = [], []
        for g in range(len(m)):
            if np.isfinite(m[g]):
                a_ub.append([pieces.alpha[g], pieces.beta[g], -pieces.gamma[g], -1.0])
                b_ub.append(-m[g])
        res = linprog(
            [p, -v, v, c_u],
            A_ub=np.array(a_ub),
            b_ub=np.array(b_ub),
            bounds=[(0, q_max), (0, cmax), (0, dmax), (0, None)],
            method="highs",
        )
        assert res.success
        assert f == pytest.approx(res.fun, abs=1e-6 * max(1.0, abs(res.fun)))
        u = pieces.utility_kw(np.array([q]), np.array([qc]), np.array([qd]))[0]
        assert c_u * u + p * q - v * (qc - qd) == pytest.approx(f, abs=1e-6 * max(1.0, abs(f)))
        assert min(qc, qd) == 0.0


def _toy_shift() -> tuple[np.ndarray, np.ndarray]:
    """Constant demand: air 10 -> 60 degC (CP 2, 100 kW); waste heat 40 -> 20 degC (CP 10)."""
    n = 96
    hot = np.zeros((n, 1, 3))
    cold = np.zeros((n, 1, 3))
    hot[:, 0] = (40.0, 20.0, 10.0)
    cold[:, 0] = (60.0, 10.0, 2.0)
    return hot, cold


def test_foresight_programme_properties_and_rules() -> None:
    hot, cold = _toy_shift()
    te, tk, ts = 10.0, 75.0, 60.0
    cascade = cascade_profile(
        hot, cold, minimum_pinch_k=20.0, extra_points=np.array([te, tk, ts + 10, ts - 10])
    )
    pieces = coupled_pieces(
        cascade,
        evaporator_shifted=te,
        condenser_shifted=tk,
        cop=3.0,
        charge_shifted=ts + 10,
        discharge_shifted=ts - 10,
    )
    clock = Clock.of(1)
    elec = signal_series(0.3, Signal(amplitude=0.8), clock)  # cheap at midday
    lines = heat_pump_lines(cascade, te, tk, 3.0)
    d0 = dispatch_steps(lines, 200.0, 0.25, elec)
    j0 = float(np.sum(0.25 * d0.utility_kw + elec * d0.work_kw) * DT_H)
    zero = solve_foresight(
        pieces,
        q_max_kw=200.0,
        capacity_kwh=0.0,
        loss_per_h=0.0,
        dt_h=DT_H,
        heat_value=0.25,
        electricity=elec,
    )
    assert zero.objective == pytest.approx(j0, rel=1e-7)
    prev = zero.objective
    for cap in (100.0, 500.0, 2000.0):
        res = solve_foresight(
            pieces,
            q_max_kw=200.0,
            capacity_kwh=cap,
            loss_per_h=0.005,
            dt_h=DT_H,
            heat_value=0.25,
            electricity=elec,
        )
        assert res.objective <= prev + 1e-7
        assert abs(res.balance_residual_kwh(DT_H)) < 1e-5
        prev = res.objective
        for rule in ("myopic", "gas_value", "trailing"):
            run = simulate_rule(
                pieces,
                rule,
                q_max_kw=200.0,
                capacities_kwh=np.array([cap]),
                loss_per_h=0.005,
                dt_h=DT_H,
                heat_value=0.25,
                electricity=elec,
            )
            assert run.objective[0] >= res.objective - 1e-6
    # a big storage with foresight beats the pump alone when power is cheap at midday only
    assert prev < zero.objective - 1.0


def test_daily_neutral_programme_is_between() -> None:
    hot, cold = _toy_shift()
    hot = np.tile(hot, (7, 1, 1))
    cold = np.tile(cold, (7, 1, 1))
    te, tk, ts = 10.0, 75.0, 60.0
    cascade = cascade_profile(
        hot, cold, minimum_pinch_k=20.0, extra_points=np.array([te, tk, ts + 10, ts - 10])
    )
    pieces = coupled_pieces(
        cascade,
        evaporator_shifted=te,
        condenser_shifted=tk,
        cop=3.0,
        charge_shifted=ts + 10,
        discharge_shifted=ts - 10,
    )
    elec = signal_series(0.3, Signal(amplitude=0.5, weekend_reduction=0.2), Clock.of(7))
    kw = {
        "q_max_kw": 150.0,
        "capacity_kwh": 3000.0,
        "loss_per_h": 0.0,
        "dt_h": DT_H,
        "heat_value": 0.25,
        "electricity": elec,
    }
    week = solve_foresight(pieces, **kw)  # type: ignore[arg-type]
    neutral = solve_foresight(pieces, steps_per_day=96, **kw)  # type: ignore[arg-type]
    assert neutral.objective >= week.objective - 1e-7
    mids = neutral.stored_kwh[::96]
    assert np.allclose(mids, mids[0], atol=1e-6)


def test_rule_without_storage_need_equals_programme_on_random_problems() -> None:
    rng = np.random.default_rng(19)
    for _ in range(6):
        pieces = _random_pieces(rng, n_steps=12)
        elec = rng.uniform(0.1, 1.0, size=12)
        caps = np.array([0.0, 50.0, 400.0])
        run = simulate_rule(
            pieces,
            "trailing",
            q_max_kw=100.0,
            capacities_kwh=caps,
            loss_per_h=0.0,
            dt_h=DT_H,
            heat_value=1.0,
            electricity=elec,
            window_steps=4,
        )
        for i, cap in enumerate(caps):
            res = solve_foresight(
                pieces,
                q_max_kw=100.0,
                capacity_kwh=float(cap),
                loss_per_h=0.0,
                dt_h=DT_H,
                heat_value=1.0,
                electricity=elec,
            )
            assert run.objective[i] >= res.objective - 1e-6
        assert np.all(run.stored_kwh <= caps[:, None] + 1e-9)


# ---------------------------------------------------------------------------------- signal


def test_signal_shape_and_week() -> None:
    hours = (np.arange(96) + 0.5) * 0.25
    s = daily_shape(hours)
    assert np.mean(s) == pytest.approx(0.0, abs=1e-12)
    assert np.min(s) == pytest.approx(-1.0)
    assert 12.0 <= hours[np.argmin(s)] <= 14.0  # solar trough
    day = signal_series(0.2, Signal(amplitude=0.5), Clock.of(1))
    assert np.mean(day) == pytest.approx(0.2)
    assert np.min(day) == pytest.approx(0.1)
    week = signal_series(0.2, Signal(amplitude=0.0, weekend_reduction=0.2), Clock.of(7))
    assert np.mean(week) == pytest.approx(0.2)
    assert week[-1] == pytest.approx(week[0] * 0.8)
    tm = trailing_mean(np.arange(8, dtype=float), 2)
    assert tm[0] == pytest.approx((6 + 7) / 2)
    assert tm[3] == pytest.approx((1 + 2) / 2)


# -------------------------------------------------------------------------- scenario level


def _synthetic_day(label: str = "product_1") -> MeasuredDay:
    t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    return MeasuredDay(
        label=label,
        ambient_c=6.0 + 3.0 * np.sin(t),
        process_air_nm3h=np.full(96, 30000.0),
        supply_air_c=np.full(96, 84.0),
        exhaust_out_c=np.full(96, 45.0) + np.sin(t),
        spray_kg_h=np.full(96, 900.0),
        atomising_nm3h=np.full(96, 560.0),
    )


def _fast(**changes: Any) -> Placeholders:
    return replace(Placeholders(n_segments_below_dew_point=6), **changes)


def test_stage_a_on_synthetic_day() -> None:
    factory = StreamFactory({"product_1": _synthetic_day()})
    scenario = Scenario(
        "A", "product_1", "day", "flat", hall_kw=0.0, water_kw=0.0, placeholders=_fast()
    )
    eco = Economics()
    crits = [co2_criterion("co2", eco), cost_criterion("cost", eco)]
    res = evaluate_stage_a(
        factory.build(scenario),
        scenario,
        spec=HeatPumpSpec(),
        criteria=crits,
        sizes_kw=(0.0, 250.0, 1000.0),
        keep_pair_table=True,
    )
    assert res.direct_kwh_per_day > 0.0
    assert np.all(res.objective_saving_per_day >= 0.0)
    assert np.all(res.objective_saving_per_day[0] == 0.0)  # size 0
    # saving non-decreasing in size (the operator may always run less)
    assert np.all(np.diff(res.objective_saving_per_day, axis=0) >= -1e-9)
    # heat replaced never exceeds the heat delivered
    assert np.all(res.utility_saving_kwh_per_day <= res.heat_kwh_per_day + 1e-6)
    assert res.pair_table
    d = res.to_dict()
    assert len(d["rows"]) == 6


def test_stage_b_on_synthetic_day_checks_pass() -> None:
    factory = StreamFactory({"product_1": _synthetic_day()})
    scenario = Scenario(
        "A", "product_1", "day", "shift", hall_kw=0.0, water_kw=0.0, placeholders=_fast()
    )
    eco = Economics()
    crit = cost_criterion("cost", eco)
    a = evaluate_stage_a(
        factory.build(scenario), scenario, spec=HeatPumpSpec(), criteria=[crit], sizes_kw=(500.0,)
    )
    pair = a.best_pair[0][0]
    if pair is None:
        pair = design_pairs(HeatPumpSpec(), minimum_pinch_k=20.0)[0]
    res = evaluate_stage_b(
        factory.build(scenario),
        scenario,
        criterion=crit,
        signal=Signal(amplitude=0.5),
        size_kw=500.0,
        pair=pair,
        levels_celsius=(50.0, 60.0),
        capacities_kwh=(0.0, 500.0, 2000.0),
    )
    bools = {k: v for k, v in res.checks.items() if isinstance(v, bool)}
    assert all(v for k, v in bools.items() if k != "rules_converged_all"), res.checks
    assert np.all(res.gap_fraction() >= -1e-9)
    assert np.all(res.storage_gain_fraction() >= -1e-9)
