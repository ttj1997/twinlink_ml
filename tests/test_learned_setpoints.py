"""Tests of the learned-set-points study (``twinlink_ml.learned_setpoints``).

Fixed beforehand in the study document (``docs/NN_learned_setpoints.md``, section 11): with the
true map behind the learned-map interface an M arm reproduces C2 on a day bit for bit; with the
truth behind the rollout interface the horizon problem of D-gru reproduces the decision of C3-15;
a policy that looks its labels up reproduces its expert; with the margin input moved the policy's
moisture moves in the right direction; training and test months are disjoint for labels and
tuning; the guard meets the daily target; a search that does not reach a margin gives a not
admissible cell and no number; the fingerprints of the load-following and headroom configs equal
those of the load-following artifacts. Further: the runner equals ``run_days`` of the
load-following study, a failing row does not touch the others, the true steady response equals
the steady optimiser, the table map finds the optimum of a known response, the NumPy GRU equals
the PyTorch model, the reading and the budget rule on synthetic inputs.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")

from twinlink_ml.headroom.controllers import DynamicOptimizer  # noqa: E402
from twinlink_ml.learned_setpoints.config import (  # noqa: E402
    LearnedSetpointsConfig,
    load_learned_setpoints_config,
)
from twinlink_ml.learned_setpoints.feedback import BiasedSetPointLaw  # noqa: E402
from twinlink_ml.learned_setpoints.gru_mpc import (  # noqa: E402
    HorizonOptimizer,
    NumpyGruRollout,
    TruthRollout,
)
from twinlink_ml.learned_setpoints.labels import (  # noqa: E402
    FEATURE_NAMES,
    LabelSet,
    nested_label_days,
)
from twinlink_ml.learned_setpoints.maps import (  # noqa: E402
    SteadyResponse,
    TableMap,
    TruthResponse,
    table_axes,
    tabulate,
)
from twinlink_ml.learned_setpoints.policy import (  # noqa: E402
    LookupPolicy,
    PolicyController,
    TargetGuard,
    fit_policy,
    load_policy,
    save_policy,
)
from twinlink_ml.learned_setpoints.study import (  # noqa: E402
    Cell,
    MapController,
    run_rows,
)
from twinlink_ml.load_following.controllers import (  # noqa: E402
    LoadController,
    Schedule,
    SetPointLaw,
)
from twinlink_ml.load_following.market import (  # noqa: E402
    DayScenario,
    Market,
    build_days,
    build_market,
)
from twinlink_ml.load_following.study import (  # noqa: E402
    RunSpec,
    Solvers,
    initial_conditions,
    run_days,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
BAND = (1.2, 1.8)
MARGIN, TAIL_MARGIN = 0.0074, 0.0055
PRICES = REPO_ROOT / "local" / "prices" / "energy_charts_price_DE-LU_2024.json"
needs_prices = pytest.mark.skipif(not PRICES.exists(), reason="price file not in local/")


@pytest.fixture(scope="module")
def config() -> LearnedSetpointsConfig:
    return load_learned_setpoints_config(day_hours=1)


@pytest.fixture(scope="module")
def market(config: LearnedSetpointsConfig) -> Market:
    return build_market(config.load_following, REPO_ROOT)


@pytest.fixture(scope="module")
def solvers(config: LearnedSetpointsConfig, market: Market) -> Solvers:
    return Solvers(config.load_following, market.reference_tariff)


@pytest.fixture(scope="module")
def days(config: LearnedSetpointsConfig, market: Market) -> list[DayScenario]:
    return build_days(config.load_following, market, [131, 320])


@pytest.fixture(scope="module")
def labels(
    config: LearnedSetpointsConfig, solvers: Solvers, days: list[DayScenario]
) -> tuple[LabelSet, list[dict[str, Any]]]:
    """The expert C2 on two days at two margins, with every decision recorded."""
    rows = [(Cell("C2"), m, d) for d in days for m in (0.0, 0.04)]
    records = run_rows(rows, TAIL_MARGIN, solvers, None, config, BAND, log_decisions=True)
    parts = []
    for (_, margin, day), record in zip(rows, records, strict=True):
        n = len(record["decisions"]["features"])
        parts.append(
            LabelSet(
                record["decisions"]["features"],
                record["decisions"]["commands"],
                np.full(n, day.day),
                np.full(n, margin),
                np.arange(n),
            )
        )
    return LabelSet.concatenate(parts), records


# ------------------------------------------------------------------------------ runner


@needs_prices
def test_runner_equals_run_days_and_true_map_behind_family_m_is_c2(
    config: LearnedSetpointsConfig, solvers: Solvers, days: list[DayScenario]
) -> None:
    lf = config.load_following
    reference = run_days(
        RunSpec(kind="C2", margin=MARGIN, tail_margin=TAIL_MARGIN, band=BAND), days, solvers
    )
    behind = [
        TargetGuard(MapController(solvers.steady, config, BAND, MARGIN), lf, BAND) for _ in days
    ]
    rows = [(Cell("C2"), MARGIN, d) for d in days] + [(Cell("M-gru"), MARGIN, d) for d in days]
    controllers = [
        *[
            Schedule(SetPointLaw(solvers.steady, 0.15, MARGIN), solvers.steady, lf, BAND, "E")
            for _ in days
        ],
        *behind,
    ]
    assert lf.headroom.line.product_moisture_limit_kg_per_kg == 0.15
    mine = run_rows(rows, TAIL_MARGIN, solvers, None, config, BAND, controllers=controllers)
    keys = (
        "cost_e", "cost_g", "energy_kwh", "heater_kwh", "sprayed_day_kg", "sprayed_total_kg",
        "product_dry_kg", "max_violation", "max_violation_tail", "max_bed_temperature",
        "energy_kwh_per_hour", "sprayed_kg_per_hour", "moisture_max_per_hour",
    )  # fmt: skip
    for i, ref in enumerate(reference):
        for record in (mine[i], mine[len(days) + i]):
            for key in keys:
                assert record[key] == ref[key], key
    for record in mine[len(days) :]:
        guard = {k: v for k, v in record["solver"].items() if k.startswith("guard_")}
        assert set(guard.values()) == {0}


class _Raises(LoadController):
    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
        pass

    def command(self, k: int, *args: Any) -> np.ndarray:  # noqa: ARG002
        if k == 3:
            raise RuntimeError("deliberate failure")
        return np.array([90.0, 1.0, 1.5])


@needs_prices
def test_lookup_policy_reproduces_its_expert_and_a_failing_row_stays_alone(
    config: LearnedSetpointsConfig,
    solvers: Solvers,
    days: list[DayScenario],
    labels: tuple[LabelSet, list[dict[str, Any]]],
) -> None:
    label_set, expert_records = labels
    lf = config.load_following
    policy = LookupPolicy(label_set.subset([days[0].day]))
    controllers = [
        TargetGuard(PolicyController(policy, lf, 0.0, solvers.reference_tariff), lf, BAND),
        _Raises(),
    ]
    rows = [(Cell("P-mlp"), 0.0, days[0]), (Cell("Cfix"), 0.0, days[0])]
    records = run_rows(rows, TAIL_MARGIN, solvers, None, config, BAND, controllers=controllers)
    assert policy.largest_distance == 0.0
    expert = expert_records[0]  # day 0 at margin 0
    for key in ("cost_e", "energy_kwh", "sprayed_total_kg", "max_violation"):
        assert records[0][key] == expert[key], key
    assert records[0]["solver"]["outside_label_range"] == 0
    assert "deliberate failure" in records[1]["error"]
    assert records[1]["max_violation"] is None


@needs_prices
def test_margin_input_moves_the_moisture_of_a_policy(
    config: LearnedSetpointsConfig,
    solvers: Solvers,
    days: list[DayScenario],
    labels: tuple[LabelSet, list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    label_set, _ = labels
    lf = config.load_following
    policy, info = fit_policy("P-lin", label_set, [d.day for d in days], config.policy, seed=1)
    assert info["rows"] == label_set.features.shape[0]
    path = tmp_path / "policy.npz"
    save_policy(policy, path)
    loaded = load_policy(path)
    features = label_set.features[5]
    np.testing.assert_allclose(loaded.act(features), policy.act(features))
    rows = [(Cell("P-lin"), m, days[0]) for m in (0.0, 0.04)]
    controllers = [
        TargetGuard(PolicyController(load_policy(path), lf, m, solvers.reference_tariff), lf, BAND)
        for m in (0.0, 0.04)
    ]
    low, high = run_rows(rows, TAIL_MARGIN, solvers, None, config, BAND, controllers=controllers)
    assert high["moisture_mean"] < low["moisture_mean"]
    assert high["energy_kwh"] > low["energy_kwh"]
    with pytest.raises(FileExistsError):
        save_policy(policy, path)


def test_policy_arms_fit_and_range_check(config: LearnedSetpointsConfig) -> None:
    torch = pytest.importorskip("torch")
    del torch
    rng = np.random.default_rng(0)
    n_days, steps = 4, 12
    features = rng.normal(size=(n_days * steps, len(FEATURE_NAMES)))
    commands = np.stack(
        [80 + 5 * features[:, 28], 1 + 0.1 * features[:, 29], 1.5 + 0.1 * features[:, 3]], axis=1
    )
    label_set = LabelSet(
        features,
        commands,
        np.repeat(np.arange(n_days), steps),
        np.zeros(n_days * steps),
        np.tile(np.arange(steps), n_days),
    )
    settings = dataclasses.replace(config.policy, max_epochs=5)
    for arm in ("P-lin", "P-mlp", "P-gru"):
        policy, info = fit_policy(arm, label_set, list(range(n_days)), settings, seed=3)
        policy.reset()
        assert policy.act(features[0]).shape == (3,)
        assert not policy.outside(features[0])
        assert policy.outside(features[0] + 100.0)
        if arm != "P-lin":
            assert len(info["validation_days"]) == 1
            assert info["fit_rows"] == (n_days - 1) * steps


# ------------------------------------------------------------------------------ guard


def test_guard_meets_the_daily_target(config: LearnedSetpointsConfig, market: Market) -> None:
    lf = load_learned_setpoints_config().load_following
    day = build_days(lf, market, [131])[0] if PRICES.exists() else None
    if day is None:
        pytest.skip("price file not in local/")
    per_hour = lf.intervals_per_hour
    target = lf.nominal_spray_rate * day.hours

    class Constant(LoadController):
        def __init__(self, spray: float) -> None:
            super().__init__()
            self._spray = spray

        def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
            pass

        def command(self, *args: Any) -> np.ndarray:  # noqa: ARG002
            return np.array([130.0, 0.5, self._spray])

    for spray in (0.9, 1.25, 1.5, 1.79, 2.4):
        guard = TargetGuard(Constant(spray), lf, BAND)
        guard.reset(day, np.zeros(4), np.zeros(3))
        sprayed = 0.0
        for k in range(day.hours * per_hour):
            out = guard.command(k, np.zeros(4), np.zeros(3), np.zeros(5), sprayed)
            assert 60.0 <= out[0] <= 120.0 and 0.8 <= out[1] <= 2.5
            assert BAND[0] - 1e-12 <= out[2] <= BAND[1] + 1e-12
            sprayed += out[2] / per_hour
        assert sprayed == pytest.approx(target, abs=0.02 + 1e-9)
        assert guard.info["guard_box"] == day.hours * per_hour
    # a controller that keeps the target itself is not touched
    guard = TargetGuard(Constant(1.5), lf, BAND)
    guard.reset(day, np.zeros(4), np.zeros(3))
    for k in range(day.hours * per_hour):
        guard.command(k, np.zeros(4), np.zeros(3), np.zeros(5), 1.5 * k / per_hour)
    assert (
        guard.info["guard_target"] == guard.info["guard_band"] == guard.info["guard_last_hour"] == 0
    )


# ------------------------------------------------------------------------------ maps


def test_true_steady_response_equals_the_steady_optimiser(
    config: LearnedSetpointsConfig, solvers: Solvers
) -> None:
    process = config.load_following.process
    nozzle = process.actions.nozzle_pressure_bar.nominal
    solid_mass = process.geometry.bed_mass_solid_setpoint_kg
    rows, expected = [], []
    for d in ([20.0, 0.5, 0.3, 1.5, 0.0], [5.0, 0.25, 0.3, 1.2, 0.0], [33.0, 0.8, 0.3, 1.8, 0.0]):
        point = solvers.steady.point(np.array(d), d[3], 0.14)
        assert point is not None
        rows.append([point.command[0], point.command[1], d[3], nozzle, d[0], d[1], d[2]])
        expected.append([point.total_kw, point.state[1] / solid_mass, point.state[2]])
    values, valid = TruthResponse(process).predict(np.array(rows))
    assert valid.all()
    np.testing.assert_allclose(values, np.array(expected), rtol=1e-6, atol=1e-7)


class _Synthetic(SteadyResponse):
    """Known response: power rises with both commands, moisture falls with them."""

    def predict(self, exogenous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inlet, gas, spray = exogenous[:, 0], exogenous[:, 1], exogenous[:, 2]
        power = 0.5 + inlet / 100.0 + 0.4 * gas
        moisture = 0.2 * spray * np.exp(-(inlet - 60.0) / 40.0) / gas
        bed = 0.5 * inlet + 5.0
        valid = ~((inlet > 119.0) & (gas > 2.4))
        return np.stack([power, moisture, bed], axis=1), valid


def test_table_map_finds_the_optimum_of_a_known_response(config: LearnedSetpointsConfig) -> None:
    table = tabulate(
        _Synthetic(),
        table_axes(config, BAND),
        nozzle_pressure_bar=1.0,
        solid_fraction=0.3,
        source="t",
    )
    assert not table.valid.all()
    steady = TableMap(table, config)
    d = np.array([17.0, 0.6, 0.3, 1.45, 0.0])
    bound = 0.12
    point = steady.point(d, 1.45, bound)
    assert point is not None
    inlet = np.linspace(60.0, 120.0, 1201)[:, None]
    gas = np.linspace(0.8, 2.5, 1701)[None, :]
    power = 0.5 + inlet / 100.0 + 0.4 * gas
    moisture = 0.2 * 1.45 * np.exp(-(inlet - 60.0) / 40.0) / gas
    best = float(np.min(np.where(moisture <= bound, power, np.inf)))
    assert point.total_kw == pytest.approx(best, abs=2e-3)
    true_moisture = 0.2 * 1.45 * math.exp(-(point.command[0] - 60.0) / 40.0) / point.command[1]
    assert true_moisture == pytest.approx(bound, abs=1e-3)
    values, ok = steady.response(point.command[0], point.command[1], 1.45, 17.0, 0.6)
    assert ok and values[1] == pytest.approx(bound, abs=1e-6)
    assert steady.point(d, 1.45, 1e-4) is None and steady.failures == 1
    total, _ = steady.curve(d, np.array([1.2, 1.5, 1.8]), bound)
    assert np.all(np.diff(total) > 0.0)
    # the law with a bias of zero is the unchanged law of the load-following study
    plain = SetPointLaw(steady, 0.15, 0.03)
    biased = BiasedSetPointLaw(steady, 0.15, 0.03)
    assert biased.bound == plain.bound
    np.testing.assert_array_equal(biased(d, 1.45, np.zeros(3)), plain(d, 1.45, np.zeros(3)))
    biased.bias = 0.01
    assert biased(d, 1.45, np.zeros(3))[0] > plain(d, 1.45, np.zeros(3))[0]


# ------------------------------------------------------------------------------ family D


def test_numpy_gru_equals_the_torch_model() -> None:
    torch = pytest.importorskip("torch")
    from twinlink_ml.data.windows import Standardizer, WindowBatch
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig
    from twinlink_ml.surrogate_env.model import EXOGENOUS_NAMES, TARGET_NAMES

    hidden, length, horizon = 8, 5, 7
    generator = torch.Generator().manual_seed(0)

    def state() -> dict[str, Any]:
        shapes = {
            "core.weight_ih_l0": (3 * hidden, 8),
            "core.weight_hh_l0": (3 * hidden, hidden),
            "core.bias_ih_l0": (3 * hidden,),
            "core.bias_hh_l0": (3 * hidden,),
            "head.weight": (1, hidden),
            "head.bias": (1,),
        }
        return {k: 0.5 * torch.randn(*shape, generator=generator) for k, shape in shapes.items()}

    model_config = TorchSequenceConfig(hidden_size=hidden)
    model = CompositeSequenceModel(model_config)
    model.load_checkpoint(
        submodel_state_dicts={t: state() for t in TARGET_NAMES},
        target_names=TARGET_NAMES,
        n_exogenous=len(EXOGENOUS_NAMES),
    )
    rng = np.random.default_rng(1)
    scaler = Standardizer(mean=rng.normal(size=10), scale=rng.uniform(0.5, 2.0, size=10))
    target_scaler = scaler.select(range(7, 10))
    metadata = {
        "exogenous_names": EXOGENOUS_NAMES,
        "target_names": TARGET_NAMES,
        "config": dataclasses.asdict(model_config),
        "input_scaler": scaler,
        "target_scaler": target_scaler,
        "data_spec": {"data/history_length": length, "data/horizon": horizon},
    }
    rollout = NumpyGruRollout(model.state_dict(), metadata)
    history = rng.normal(size=(length, 10))
    future = rng.normal(size=(3, horizon, 7))
    mine = rollout.rollout(rollout.prepare(history), future)
    batch = WindowBatch(
        history_inputs=np.repeat(scaler.transform(history)[None], 3, axis=0),
        future_exogenous=scaler.select(range(7)).transform(future),
        future_targets=np.zeros((3, horizon, 3)),
        run_index=np.zeros(3),
        window_start=np.zeros(3),
    )
    reference = target_scaler.inverse(np.asarray(model.predict(batch), dtype=np.float64))
    np.testing.assert_allclose(mine, reference, rtol=1e-4, atol=1e-4)


@needs_prices
def test_horizon_problem_on_the_truth_reproduces_the_decision_of_c3_15(
    config: LearnedSetpointsConfig, solvers: Solvers, days: list[DayScenario]
) -> None:
    """Tolerance 0.5 K and 0.01 m/s: the two programmes discretise the same model differently."""
    lf = config.load_following
    block_s = config.feedback.measurement_interval_s
    coarse = dataclasses.replace(
        lf.process, numerics=dataclasses.replace(lf.process.numerics, sample_time_s=block_s)
    )
    model = TruthRollout(coarse, block_s=block_s, history_length=32, horizon=36)
    optimizer = HorizonOptimizer(
        model,
        lf.process,
        n_intervals=config.short_horizon_intervals,
        blocks_per_interval=round(lf.headroom.control.interval_s / block_s),
        block_s=block_s,
        bed_limit=lf.headroom.line.bed_temperature_limit_celsius,
        slack_weight_kwh=lf.headroom.control.soft_constraint_weight_kwh,
        fd_step=config.gru_mpc_fd_step,
        max_iterations=config.gru_mpc_max_iterations,
    )
    fast = DynamicOptimizer(
        lf.headroom, lf.process, config.short_horizon_intervals, expression="MX"
    )
    states, commands = initial_conditions(solvers, days)
    nozzle = lf.process.actions.nozzle_pressure_bar.nominal
    bound = 0.15 - MARGIN
    for i, day in enumerate(days):
        d = day.disturbances[:, 0]
        nodes = fast.n_intervals * fast.n_steps
        fast.reset()
        reference = fast.solve(
            states[i][[0, 1, 3]],
            commands[i][:2],
            np.tile(d[:, None], (1, nodes)),
            np.full(nodes, bound),
        )
        model.set_state(states[i])
        base = np.empty((36, 7))
        base[:, 0:2], base[:, 2], base[:, 3], base[:, 4:7] = commands[i][:2], d[3], nozzle, d[0:3]
        guess = np.repeat(commands[i][:2][:, None], optimizer.n_intervals, axis=1)
        mine = optimizer.solve(None, base, bound, guess)
        assert abs(mine[0, 0] - reference[0, 0]) <= 0.5
        assert abs(mine[1, 0] - reference[1, 0]) <= 0.01


# ------------------------------------------------------------------------------ driver and reading


def test_label_days_are_nested_and_stratified() -> None:
    days = list(range(100, 140))
    spread = list(np.random.default_rng(0).uniform(size=40))
    nested = nested_label_days(days, spread, [5, 10, 20, 40], 5, seed=2)
    assert [len(nested[k]) for k in (5, 10, 20, 40)] == [5, 10, 20, 40]
    assert set(nested[5]) < set(nested[10]) < set(nested[20]) < set(nested[40]) == set(days)
    order = np.argsort(spread)
    strata = [set(np.array(days)[order[8 * g : 8 * g + 8]]) for g in range(5)]
    for budget in (5, 10, 20):
        assert [len(set(nested[budget]) & s) for s in strata] == [budget // 5] * 5
    assert nested_label_days(days, spread, [5], 5, seed=3)[5] != nested[5]
    with pytest.raises(ValueError, match="does not fit"):
        nested_label_days(days, spread, [7], 5, seed=2)


@needs_prices
def test_months_are_disjoint_and_fingerprints_equal_the_load_following_artifacts(
    market: Market,
) -> None:
    import run_learned_setpoints as driver

    config = load_learned_setpoints_config()
    lf = config.load_following
    task = driver.reference_task(config, market)  # aborts if a fingerprint differs
    for name in (config.stage0_artifact, config.main_artifact):
        with gzip.open(REPO_ROOT / "artifacts" / "runs" / name, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        assert artifact["study_fingerprint"] == lf.fingerprint
        assert artifact["headroom_fingerprint"] == lf.headroom.fingerprint
        assert artifact["process_config_fingerprint"] == lf.process.fingerprint
    month = {
        name: {int(market.clock.month[d * 24]) for d in ids}
        for name, ids in task["day_sets"].items()
    }
    assert month["test"] <= set(lf.prices.test_months)
    assert month["label"] <= set(lf.prices.training_months)
    assert month["tuning"] <= set(lf.prices.training_months)
    assert not set(lf.prices.test_months) & set(lf.prices.training_months)
    assert not set(task["day_sets"]["label"]) & set(task["day_sets"]["tuning"])
    assert len(task["day_sets"]["test"]) == len(task["day_sets"]["tuning"]) == 40
    assert task["twin"] == "C2" and task["load_following_reading"] == "T1"
    cells = driver.study_cells(config)
    assert [len(cells[p]) for p in ("M", "D", "P")] == [100, 10, 60]
    cut = driver.study_cells(config, config.cut_order)
    assert [len(cut[p]) for p in ("M", "D", "P")] == [60, 6, 40]
    # the cuts never touch what carries the reading at the largest budget
    kept = {c.key for cs in cut.values() for c in cs}
    for part in ("M", "D", "P"):
        for cell in cells[part]:
            carries = driver.largest_budget(cell, config) and cell.level != "lab"
            if carries and cell.arm not in ("D-gru", "P-gru"):
                assert cell.key in kept


class _NeverHolds:
    """Evaluator stub: every day violates the limit at every margin."""

    def run(self, cell: Cell, margin: float, days: list[int], day_set: str) -> list[dict]:  # noqa: ARG002
        return [{"max_violation": 0.01, "day_of_year": d, "margin": margin} for d in days]


def test_search_without_a_margin_is_not_admissible_and_has_no_number() -> None:
    import run_learned_setpoints as driver

    config = load_learned_setpoints_config()
    task = {"day_sets": {"test": list(range(12))}}
    study = driver.Study(_NeverHolds(), config, task, margin_tolerance=2.5e-4)  # type: ignore[arg-type]
    block = study.search(Cell("M-gru", "none", 64, 1))
    assert block["admissible"] is False
    assert block["margin"] is None
    assert block["largest_margin_tried"] == config.load_following.margin_maximum

    import analyze_learned_setpoints as analysis

    summary = analysis.cell_summary(block, 0.4, 0.42, 0.41)
    assert summary["admissible"] is False and "J" not in summary and "gap" not in summary


def test_budget_rule() -> None:
    import run_learned_setpoints as driver

    config = load_learned_setpoints_config()

    def rule(uncut: float, saving: float) -> dict[str, Any]:
        return driver.apply_budget_rule(config, lambda cuts: uncut - saving * len(cuts), 1.0)

    assert rule(20.0, 0.0)["budget_h"] == 30.0 and rule(20.0, 0.0)["cuts"] == []
    extended = rule(40.0, 0.0)
    assert extended["budget_h"] == 48.0 and extended["cuts"] == []
    cut = rule(60.0, 7.0)
    assert cut["cuts"] == ["lab", "n_train_32"] and cut["wall_h"] == pytest.approx(47.0)
    assert rule(200.0, 1.0)["cuts"] == list(config.cut_order)


def _table(gaps: dict[int, list[float | None]], j_twin: float = 1.0) -> dict[str, Any]:
    out = {}
    for budget, values in gaps.items():
        good = {s: g for s, g in enumerate(values, start=1) if g is not None}
        rng = (
            {"median": float(np.median(list(good.values()))), "max": max(good.values())}
            if good
            else None
        )
        out[str(budget)] = {
            "seeds": list(range(1, len(values) + 1)),
            "admissible_seeds": sorted(good),
            "gap": rng,
            "J": None if not good else {"median": j_twin * (1 + rng["median"])},  # type: ignore[index]
            "errored_units": 0,
        }
    return out


def test_reading_on_synthetic_tables() -> None:
    import analyze_learned_setpoints as analysis

    thresholds = {
        "within_twin": 0.02,
        "within_twin_every_seed": 0.05,
        "admissible_seeds_minimum": 4,
    }

    def read(
        tables: dict[str, Any], j_fix: float | None = 1.2, family: str = "M"
    ) -> dict[str, Any]:
        return analysis.read_family(family, tables, [8, 64], 1.0, j_fix, thresholds, 5)

    good = [0.01, 0.012, 0.015, 0.018, 0.019]
    bad = [0.06, 0.07, 0.08, 0.09, 0.10]
    u1 = read({"M-quad": _table({8: bad, 64: good}), "M-gru": _table({8: good, 64: good})})
    assert (u1["reading"], u1["named_arm"], u1["data_need"]) == ("U1", "M-quad", 64)
    u2 = read({"M-quad": _table({8: bad, 64: bad}), "M-gru": _table({8: good, 64: good})})
    assert (u2["reading"], u2["named_arm"], u2["data_need"]) == ("U2", "M-gru", 8)
    u3 = read({"M-quad": _table({8: bad, 64: bad}), "M-gru": _table({8: bad, 64: bad})})
    assert (u3["reading"], u3["named_arm"], u3["data_need"]) == ("U3", None, None)
    u0 = read({"M-quad": _table({8: bad, 64: bad}), "M-gru": _table({8: bad, 64: bad})}, j_fix=1.05)
    assert u0["reading"] == "U0"
    few = [0.01, None, None, 0.01, 0.01]
    none = read({"M-quad": _table({8: bad, 64: few}), "M-gru": _table({8: bad, 64: few})})
    assert none["reading"] == "not admissible"
    partly = read({"M-quad": _table({8: bad, 64: few}), "M-gru": _table({8: bad, 64: good})})
    assert (partly["reading"], partly["named_arm"]) == ("U2", "M-gru")
    assert partly["arms_admissible_for_reading"] == ["M-gru"]
    # within 2 % in the median but one seed above 5 %: the data need is not determined
    wide = [0.01, 0.012, 0.015, 0.018, 0.06]
    need = read({"M-quad": _table({8: bad, 64: wide}), "M-gru": _table({8: bad, 64: bad})})
    assert (need["reading"], need["data_need"]) == ("U1", "not determined")
    d = read({"D-gru": _table({8: bad, 64: good})}, family="D")
    assert (d["reading"], d["named_arm"]) == ("U2", "D-gru")
    assert read({"D-gru": _table({8: bad, 64: bad})}, family="D")["reading"] == "U3"
    no_fix = read(
        {"M-quad": _table({8: bad, 64: bad}), "M-gru": _table({8: bad, 64: bad})}, j_fix=None
    )
    assert no_fix["reading"] == "U3" and "U0_note" in no_fix
    # three seeds after the cut: all three must be admissible
    three = analysis.read_family(
        "D", {"D-gru": _table({64: [0.01, None, 0.01]})}, [64], 1.0, 1.2, thresholds, 5
    )
    assert three["reading"] == "not admissible"

    assert analysis.holm([0.01, 0.04, 0.03, math.nan]) == pytest.approx(
        [0.03, 0.06, 0.06, math.nan], nan_ok=True
    )
    test = analysis.rank_test(
        {"1": 1.0, "2": 1.1, "3": 1.2}, {"1": 1.1, "2": 1.3, "3": 1.25}, "a vs b", 1.0
    )
    assert test["mean_difference"] > 0 and test["sign_consistency"] == 1.0
    assert math.isnan(analysis.rank_test({"1": 1.0}, {"2": 1.0}, "x", 1.0)["p_value"])
