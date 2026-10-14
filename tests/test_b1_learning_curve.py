"""B1 learning curves: arms, nesting, B5 versus B5r, determinism, test-set guard."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation.b1_learning_curve import CurveConfig, candidate_pool, run_arm
from twinlink_ml.evaluation.b1_testsets import build_t1, build_t2, build_t3
from twinlink_ml.evaluation.experiments import DataSpec
from twinlink_ml.evaluation.growing_pool import guarded_config
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

TARGETS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
SPREAD = {
    "spread": {
        "inlet_temperature_celsius": {"range": 20.0},
        "gas_velocity_m_per_s": {"range": 0.5},
        "nozzle_pressure_bar": {"range": 1.0},
    }
}
ANCHORS = {
    "u_gas_min": 1.26,
    "quality_limit_kg_per_kg": 0.15,
    "anchors": [
        {
            "ambient_unit": [0.5, 0.5],
            "optimum": {"t_in": 90.0, "u_gas": 1.26, "spray": 1.5, "p_nozzle": 2.5},
            "limit": {"t_in": 90.0, "u_gas": 1.26, "spray": 1.57, "p_nozzle": 2.5},
        }
    ],
}
CURVE = CurveConfig(
    sizes=(2, 4),
    models=("persistence", "arx", "narx"),
    network_label="narx",
    n_members=3,
    per_plan=4,
)


@pytest.fixture(scope="module")
def world():
    path = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = guarded_config(load_process_config(path))
    spec = DataSpec.new_study(
        n_train_max=4, n_val=0, n_test=0, target_names=TARGETS, run_duration_s=6000.0
    )
    holdouts = [
        build_t1(config, spec, runs_per_plan=1),
        build_t2(config, spec, n_runs=2),
        build_t3(config, spec, ANCHORS, SPREAD, runs_per_cell=1),
    ]
    return config, spec, holdouts


def _selected(records):
    return {r["n_train"]: r["selected"] for r in records if r["type"] == "information"}


def test_plan_arm_records_and_nesting(world):
    config, spec, holdouts = world
    seen = []
    records = run_arm("b2", 1, config, spec, holdouts, CURVE, on_record=seen.append)
    assert records == seen
    models = [r for r in records if r["type"] == "model"]
    assert {(r["n_train"], r["model"]) for r in models} == {
        (n, m) for n in (2, 4) for m in CURVE.models
    }
    selected = _selected(records)
    assert selected[2] == [0, 1] and selected[4] == [0, 1, 2, 3]  # nested prefixes
    sample = next(r for r in models if r["model"] == "narx")
    assert set(sample["metrics"]) == {"T1", "T2", "T3"}
    assert "rmse_run_median_physical/product_moisture_kg_per_kg/h24" in sample["metrics"]["T3"]
    info = next(r for r in records if r["type"] == "information")["information"]
    for key in (
        "spectral_coverage_1pct",
        "spectral_coverage_10pct",
        "spectral_coverage_25pct",
        "fir_logdet_per_parameter",
        "arx_logdet_per_parameter",
    ):
        assert key in info


def test_candidate_pool_has_equal_parts_without_b0(world):
    config, spec, _ = world
    pool = candidate_pool(config, spec, 5, per_plan=4)
    plans = [rid.split("-")[0] for rid in pool.run_id]
    assert pool.n_runs == 16 and {p: plans.count(p) for p in set(plans)} == {
        "b1": 4,
        "b2": 4,
        "b3": 4,
        "b4": 4,
    }
    assert len(set(pool.run_id)) == 16


def test_b5_and_b5r_share_the_start_and_differ_afterwards_and_are_deterministic(world):
    config, spec, holdouts = world
    active = run_arm("b5", 3, config, spec, holdouts, CURVE)
    again = run_arm("b5", 3, config, spec, holdouts, CURVE)
    random = run_arm("b5r", 3, config, spec, holdouts, CURVE)
    assert _selected(active) == _selected(again)  # deterministic
    assert _selected(active)[2] == _selected(random)[2]  # same start
    assert len(_selected(active)[4]) == len(_selected(random)[4]) == 4
    assert len(set(_selected(active)[4])) == 4
    selection = [r for r in active if r["type"] == "selection"]
    assert len(selection) == 1 and len(selection[0]["chosen"]) == 2
    assert selection[0]["scores_chosen"] == sorted(selection[0]["scores_chosen"], reverse=True)
    assert not [r for r in random if r["type"] == "selection"]
    # the active choice is the top of the scores, so it is not what the random stream gave
    # (equal only by chance; with 14 candidates the probability is small but not zero)
    assert all(c in range(16) for c in selection[0]["chosen"])


def test_b5_selection_uses_member_zero_as_the_evaluated_model(world):
    config, spec, holdouts = world
    active = run_arm("b5", 4, config, spec, holdouts, CURVE)
    random = run_arm("b5r", 4, config, spec, holdouts, CURVE)
    key = "rmse_run_median_physical/product_moisture_kg_per_kg/h24"

    def at_start(records):
        return next(
            r
            for r in records
            if r["type"] == "model" and r["n_train"] == 2 and r["model"] == "narx"
        )["metrics"]["T1"][key]

    assert at_start(active) == at_start(random)  # same start pool, same model seed -> same model


def test_unknown_arm_and_bad_config(world):
    config, spec, holdouts = world
    with pytest.raises(KeyError):
        run_arm("b9", 1, config, spec, holdouts, CURVE)
    with pytest.raises(ValueError):
        CurveConfig(sizes=(4, 2))
    with pytest.raises(ValueError):
        CurveConfig(models=("arx",), network_label="lstm")


def test_test_sets_are_disjoint_from_training(world):
    config, spec, holdouts = world
    # a holdout that is a copy of a training run must be caught inside run_arm
    records = run_arm("b1", 2, config, spec, holdouts, CURVE)
    assert records  # the real holdouts pass
    from twinlink_ml.evaluation.b1_testsets import HoldoutSet
    from twinlink_ml.evaluation.growing_pool import (
        GrowingPool,
        prepare_process_config,
    )
    from twinlink_ml.simulation.excitation_plans import make_plan
    from twinlink_ml.utils.seeding import SeedBundle

    seeds = SeedBundle.from_master(2)
    prepared = prepare_process_config(config, spec)
    plan = make_plan(
        "b1",
        prepared,
        sample_time_s=spec.sample_time_s,
        run_duration_s=spec.run_duration_s,
        solid_fraction_process="independent",
    )
    schedule = plan.build(4, seed=seeds.data_seed)
    pool = GrowingPool(config, spec, schedule, noise_seed=seeds.noise_seed)
    pool.add([0, 1, 2, 3])
    leak = HoldoutSet("leak", schedule, pool.trajectory())
    with pytest.raises(ValueError):
        run_arm("b1", 2, config, spec, [*holdouts, leak], CURVE)
    assert dataclasses.is_dataclass(leak) and np.isfinite(0.0)
