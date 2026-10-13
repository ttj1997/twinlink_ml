"""B1.2/B1.3: growing pool (bit-identity to generate_data) and test sets T1-T3."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from twinlink_ml.evaluation.b1_testsets import (
    HoldoutSet,
    assert_disjoint,
    build_t1,
    build_t2,
    build_t3,
    find_limit_spray,
    holdout_seed,
    steady_state_at,
)
from twinlink_ml.evaluation.experiments import DataSpec, generate_data
from twinlink_ml.evaluation.growing_pool import GrowingPool, guarded_config, windows_for_test_set
from twinlink_ml.simulation.excitation_plans import PLAN_NAMES

SPEC = DataSpec.new_study(n_train_max=4, n_val=2, n_test=2)
SPREAD = {
    "spread": {
        "inlet_temperature_celsius": {"range": 20.0},
        "gas_velocity_m_per_s": {"range": 0.5},
        "nozzle_pressure_bar": {"range": 1.0},
    }
}


@pytest.fixture(scope="module")
def data_and_config():
    import warnings
    from pathlib import Path

    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

    path = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = guarded_config(load_process_config(path))
    return generate_data(config, SPEC, master_seed=1), config


def test_full_pool_equals_generate_data(data_and_config):
    data, config = data_and_config
    pool = GrowingPool(config, SPEC, data.schedule, noise_seed=data.seeds.noise_seed)
    pool.add([0, 1, 2])
    pool.add([3, 4])
    pool.add([5, 6, 7])
    full, reference = pool.legacy_trajectory(), data.trajectory
    assert full.run_id == reference.run_id
    for name in ("actions", "disturbances", "states", "outputs", "measurements"):
        assert np.array_equal(getattr(full, name), getattr(reference, name)), name


def test_candidates_are_inputs_only_and_added_once(data_and_config):
    data, config = data_and_config
    pool = GrowingPool(config, SPEC, data.schedule, noise_seed=1)
    inputs = pool.candidate_inputs([2, 5])
    assert not hasattr(inputs, "outputs") and not hasattr(inputs, "states")
    assert inputs.run_id == (data.schedule.run_id[2], data.schedule.run_id[5])
    with pytest.raises(ValueError):
        pool.trajectory()  # nothing simulated yet: no truth exists for any candidate
    pool.add([5])
    with pytest.raises(ValueError):
        pool.add([5])
    assert pool.selected == (5,) and 5 not in pool.remaining


def test_per_run_noise_does_not_depend_on_how_the_pool_grew(data_and_config):
    data, config = data_and_config
    first = GrowingPool(config, SPEC, data.schedule, noise_seed=9)
    first.add([0, 1])
    first.add([2])
    second = GrowingPool(config, SPEC, data.schedule, noise_seed=9)
    second.add([2])
    second.add([1, 0])
    a, b = first.trajectory(), second.trajectory()
    assert np.array_equal(a.measurements[2], b.measurements[0])
    assert np.array_equal(a.measurements[0], b.measurements[2])
    assert np.array_equal(a.outputs[1], b.outputs[1])


def test_pool_windows_and_prefix(data_and_config):
    data, config = data_and_config
    pool = GrowingPool(config, SPEC, data.schedule, noise_seed=1)
    pool.add([0, 1, 2, 3])
    assert pool.windows(2).n_windows < pool.windows(4).n_windows
    assert pool.trajectory(2).run_id == data.schedule.run_id[:2]


def test_t1_t2_and_disjointness(data_and_config):
    data, config = data_and_config
    spec = dataclasses.replace(SPEC, run_duration_s=6000.0)
    t1 = build_t1(config, spec, runs_per_plan=2)
    t2 = build_t2(config, spec, n_runs=4)
    assert t1.trajectory.n_runs == 2 * len(PLAN_NAMES)
    assert {rid.split("-")[0] for rid in t1.trajectory.run_id} == set(PLAN_NAMES)
    assert all(set(kinds) == {"constant"} for kinds in t2.schedule.signal_kinds)
    train = generate_data(
        config, dataclasses.replace(spec, n_train_max=4), master_seed=3
    ).trajectory
    assert_disjoint(train, [t1, t2])
    with pytest.raises(ValueError):
        assert_disjoint(train, [HoldoutSet("same", t1.schedule, train)])
    assert holdout_seed("t1", 0, "b0") != holdout_seed("t1", 0, "b1")


def _anchors(config):
    base = {"t_in": 90.0, "u_gas": 1.26, "spray": 1.5, "p_nozzle": 2.5}
    disturbance = (20.0, 0.55, 0.30)
    spray, _ = find_limit_spray(
        config,
        (base["t_in"], base["u_gas"], base["spray"], base["p_nozzle"]),
        disturbance,
        target_moisture_kg_per_kg=0.15 - 0.003,
    )
    return {
        "u_gas_min": 1.26,
        "quality_limit_kg_per_kg": 0.15,
        "anchors": [
            {"ambient_unit": [0.5, 0.5], "optimum": base, "limit": {**base, "spray": spray}}
        ],
    }


def test_t3_profiles(data_and_config):
    _, config = data_and_config
    spec = dataclasses.replace(SPEC, run_duration_s=28_800.0)
    anchors = _anchors(config)
    t3 = build_t3(config, spec, anchors, SPREAD, runs_per_cell=1)
    assert t3.trajectory.n_runs == 2
    dt = spec.sample_time_s
    hold = round(3600.0 / dt)
    for run, family in enumerate(("optimum", "limit")):
        base = anchors["anchors"][0][family]
        actions = t3.schedule.actions[run]
        # one hour at the anchor, then set-point changes
        assert np.allclose(actions[:hold, 0], base["t_in"])
        assert np.ptp(actions[hold:, :], axis=0).max() > 0.0
        # the gas velocity never goes below the study bound
        assert actions[:, 1].min() >= 1.26 - 1e-9
        # starts in the steady state of its own first disturbance (no start-up transient)
        steady = steady_state_at(
            config,
            [base[c] for c in ("t_in", "u_gas", "spray", "p_nozzle")],
            t3.schedule.disturbances[run, 0],
        )
        assert t3.trajectory.outputs[run, 0, 2] == pytest.approx(
            steady.particle_moisture_kg_per_kg, abs=2e-3
        )
    assert t3.trajectory.outputs[1, 0, 2] > t3.trajectory.outputs[0, 0, 2]  # limit run is wetter


def test_t3_family_two_alternates_over_and_inside_the_limit(data_and_config):
    _, config = data_and_config
    spec = dataclasses.replace(SPEC, run_duration_s=28_800.0)
    t3 = build_t3(config, spec, _anchors(config), SPREAD, runs_per_cell=2)
    assert t3.schedule.metadata["across_limit_requirement_failures"] == 0
    assert all(k == ("t3-optimum",) * 4 for k in t3.schedule.signal_kinds[:2])
    assert all(k == ("t3-across_limit",) * 4 for k in t3.schedule.signal_kinds[2:])
    dt = spec.sample_time_s
    n_steps = round(spec.run_duration_s / dt)
    edges = np.linspace(round(3600.0 / dt), n_steps, 5).astype(int)
    scale = t3.schedule.metadata["step_scale"]
    assert scale[0] == 20.0 and scale[1] == 0.5 and scale[3] == 1.0
    for run in (2, 3):
        d0 = t3.schedule.disturbances[run, 0]
        for change in range(4):
            # end of the segment: the rate-limited transition has long settled
            action = t3.schedule.actions[run, edges[change + 1] - 1]
            moisture = steady_state_at(config, action, d0).particle_moisture_kg_per_kg
            assert (moisture > 0.15) == (change % 2 == 0), (run, change, moisture)
        assert t3.schedule.actions[run, :, 1].min() >= 1.26 - 1e-9


def test_find_limit_spray_hits_the_target(data_and_config):
    _, config = data_and_config
    target = 0.147
    spray, moisture = find_limit_spray(
        config, (90.0, 1.26, 1.5, 2.5), (20.0, 0.55, 0.30), target_moisture_kg_per_kg=target
    )
    assert abs(moisture - target) < 1e-6
    again = steady_state_at(config, (90.0, 1.26, spray, 2.5), (20.0, 0.55, 0.30))
    assert again.particle_moisture_kg_per_kg == pytest.approx(target, abs=1e-6)


def test_windows_for_test_set_uses_training_scalers(data_and_config):
    data, config = data_and_config
    pool = GrowingPool(config, SPEC, data.schedule, noise_seed=1)
    pool.add([0, 1])
    train = pool.windows()
    t2 = build_t2(config, dataclasses.replace(SPEC, run_duration_s=6000.0), n_runs=2)
    test = windows_for_test_set(t2.trajectory, SPEC.window, train.scalers)
    assert test.n_windows > 0


def test_a_capped_spec_needs_the_guard(data_and_config):
    from twinlink_ml.evaluation.growing_pool import prepare_process_config

    _, config = data_and_config
    unguarded = dataclasses.replace(
        config,
        numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=False),
    )
    with pytest.raises(ValueError, match="projection"):
        prepare_process_config(unguarded, SPEC)
    assert SPEC.gas_velocity_cap == "initial_diameter" and SPEC.new_study_version == 2
    prepare_process_config(config, SPEC)  # guarded: fine
    legacy = dataclasses.replace(SPEC, gas_velocity_cap="none", new_study_version=None)
    prepare_process_config(unguarded, legacy)  # no cap, no guard needed


def test_projection_counts_zero_with_the_cap_and_positive_without(data_and_config):
    from twinlink_ml.evaluation.growing_pool import projection_counts
    from twinlink_ml.simulation.excitation_plans import make_plan

    _, config = data_and_config
    counts = {}
    for cap in ("initial_diameter", "none"):
        spec = dataclasses.replace(SPEC, gas_velocity_cap=cap)
        plan = make_plan(
            "b2",
            config,
            sample_time_s=spec.sample_time_s,
            run_duration_s=spec.run_duration_s,
            solid_fraction_process="independent",
            gas_velocity_cap=cap,
        )
        schedule = plan.build(8, seed=1)
        pool = GrowingPool(config, spec, schedule, noise_seed=1)
        pool.add(list(range(8)))
        counts[cap] = projection_counts(pool.trajectory())
    assert counts["initial_diameter"]["projected_steps"] == 0
    assert counts["initial_diameter"]["projection_active"] == 1
    assert counts["initial_diameter"]["total_steps"] == 8 * 1152
    # the guard is meaningful: without the cap the same plan is projected
    assert counts["none"]["projected_steps"] > 0 and counts["none"]["runs_with_projection"] > 0
