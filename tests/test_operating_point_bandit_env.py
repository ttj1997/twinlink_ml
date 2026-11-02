r"""VA2: oracle check -- the environment must reproduce Befund 52 bit-identically.

Most important test of this task (see the task text): an environment that does not reproduce
the known optimum from ``artifacts/runs/bb1_bb3_operating_point_regret.json`` is
worthless for RL. Three checks:

a) Oracle actions at all 25 grid points -> reward exactly ``-energy_target``.
b) Fixed operating point at all 25 grid points -> regret statistics (median/P95/max,
   absolute and relative) reproduce the documented values.
c) Feasibility at all 25 points for both action sets.

Question of scale (nozzle pressure fixed vs. freely optimised per point, see the task "Caution:
scale"): empirically bit-identical (see ``test_fixed_nozzle_pressure_scale_is_negligible``)
-- the nozzle-pressure degree of freedom of the Befund 52 optima moves throughout in
``[1.0002, 1.0111]`` bar, below the clamp threshold ``gas.pressure_pa / 1e5 = 1.01325``
bar in ``energy.power_breakdown_kw`` (``max(pressure_ratio, 1 + 1e-12)``) -- the
atomiser share of ``energy_target`` is exactly constant (~0) in this entire window,
independent of the exact nozzle pressure.

Solver precision: all calls here go through :meth:`OperatingPointBanditEnv.
evaluate_operating_point`, which internally uses the SEARCH precision from Befund 52
(``DEFAULT_SEARCH_MAX_ITERATIONS``/``DEFAULT_SEARCH_TOLERANCE``), not the solver defaults
-- see the ``env.py`` module docstring for the reasoning (bracket sign error at 6 of
25 points with the solver defaults, checked empirically).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from gymnasium import spaces

from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv
from twinlink_ml.rl.env import (
    ANCHOR_ARTIFACT_PATH,
    CONFIG_PATH,
    EXPECTED_CONFIG_FINGERPRINT,
    FIXED_NOZZLE_PRESSURE_BAR,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent

REL_TOLERANCE = 1e-6
"""Grosszuegig ueber dem beobachteten Gleitkommarauschen (~1e-9 relativ, siehe Auftrag/
Analyse) -- eng genug, echte Regressionen von Rauschen zu unterscheiden."""


@pytest.fixture(scope="module")
def anchor() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(ANCHOR_ARTIFACT_PATH).read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def env() -> OperatingPointBanditEnv:
    return OperatingPointBanditEnv()


def test_config_fingerprint_matches_anchor_artifact(anchor: dict[str, Any]) -> None:
    """Precondition for every comparison below: the same starting point as Befund 52.

    ``anchor["config_fingerprint"]`` carries the frozen value from before the
    pseudonymisation cut (see the ``EXPECTED_CONFIG_FINGERPRINT`` docstring in
    ``env.py``) -- :func:`fingerprints_equivalent` recognises both as the same numerical
    configuration.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    assert config.fingerprint == EXPECTED_CONFIG_FINGERPRINT
    assert fingerprints_equivalent(EXPECTED_CONFIG_FINGERPRINT, anchor["config_fingerprint"])


def test_va2a_oracle_actions_reproduce_energy_target(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    for point in anchor["grid_results"]:
        reward, info = env.evaluate_operating_point(
            inlet_temperature_celsius=point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        assert not info["solve_failed"], (
            f"Gitterpunkt {point['_grid_index']}: Loeser schlaegt an der Orakelaktion fehl"
        )
        assert reward == pytest.approx(-point["energy_target"], rel=REL_TOLERANCE)
        assert info["energy_target"] == pytest.approx(point["energy_target"], rel=REL_TOLERANCE)


def test_va2c_oracle_actions_are_feasible(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    for point in anchor["grid_results"]:
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        assert info["feasible"] == point["feasible"]


def test_va2c_fixed_point_actions_are_feasible(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    fixed = anchor["fixed_point"]
    for point in fixed["per_grid_point"]:
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=fixed["inlet_temperature_celsius"],
            gas_velocity_m_per_s=fixed["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        assert info["feasible"] == point["feasible"]


def test_va2b_fixed_point_regret_matches_befund_52(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    fixed = anchor["fixed_point"]
    absolute_kw: list[float] = []
    relative_pct: list[float] = []
    for grid_point, _fixed_point_at_grid in zip(
        anchor["grid_results"], fixed["per_grid_point"], strict=True
    ):
        t_amb = grid_point["ambient_temperature_celsius"]
        phi_amb = grid_point["ambient_relative_humidity"]
        _, oracle_info = env.evaluate_operating_point(
            inlet_temperature_celsius=grid_point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=grid_point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=t_amb,
            ambient_relative_humidity=phi_amb,
        )
        _, fixed_info = env.evaluate_operating_point(
            inlet_temperature_celsius=fixed["inlet_temperature_celsius"],
            gas_velocity_m_per_s=fixed["gas_velocity_m_per_s"],
            ambient_temperature_celsius=t_amb,
            ambient_relative_humidity=phi_amb,
        )
        assert oracle_info["feasible"] and fixed_info["feasible"]
        regret = fixed_info["energy_target"] - oracle_info["energy_target"]
        absolute_kw.append(regret)
        relative_pct.append(100.0 * regret / oracle_info["energy_target"])

    absolute_arr = np.asarray(absolute_kw)
    relative_arr = np.asarray(relative_pct)
    documented = anchor["regret"]

    assert float(np.median(absolute_arr)) == pytest.approx(
        documented["absolute_kw"]["median"], rel=REL_TOLERANCE
    )
    assert float(np.percentile(absolute_arr, 95)) == pytest.approx(
        documented["absolute_kw"]["p95"], rel=REL_TOLERANCE
    )
    assert float(np.max(absolute_arr)) == pytest.approx(
        documented["absolute_kw"]["max"], rel=REL_TOLERANCE
    )
    assert float(np.median(relative_arr)) == pytest.approx(
        documented["relative_pct"]["median"], rel=REL_TOLERANCE
    )
    assert float(np.percentile(relative_arr, 95)) == pytest.approx(
        documented["relative_pct"]["p95"], rel=REL_TOLERANCE
    )
    assert float(np.max(relative_arr)) == pytest.approx(
        documented["relative_pct"]["max"], rel=REL_TOLERANCE
    )
    # Median 16.03 %, P95 22.2 %, max 25.80 % -- numbers from the task, only as an
    # additional, readable regression bound (the pytest.approx comparisons above
    # against the artifact are the actual check).
    assert float(np.median(relative_arr)) == pytest.approx(16.03, abs=0.01)
    assert float(np.percentile(relative_arr, 95)) == pytest.approx(22.2, abs=0.1)
    assert float(np.max(relative_arr)) == pytest.approx(25.80, abs=0.01)


def test_fixed_nozzle_pressure_scale_is_negligible(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    """The task, "Caution: scale".

    Nozzle pressure is fixed in the environment, free in the Befund 52 optima -- quantifies the
    resulting deviation. Compares ``energy_target`` at ``FIXED_NOZZLE_PRESSURE_BAR``
    (as in the environment) against the respective POINT-SPECIFIC, freely optimised nozzle
    pressure -- both otherwise at the identical operating point. The 0.4 % spread from Befund 52
    (``setpoint_spread.nozzle_pressure_bar``) lies entirely below the
    clamp threshold in ``power_breakdown_kw`` -- the deviation is therefore not merely
    "within a tolerance", but numerically indistinguishable from zero.
    """
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator, operating_point
    from twinlink_ml.simulation.steady_state import solve_steady_state

    deltas_kw: list[float] = []
    for point in anchor["grid_results"]:
        variant_own_nozzle = operating_point.build_variant_config(
            env._base_config,
            t_in=point["inlet_temperature_celsius"],
            u_gas=point["gas_velocity_m_per_s"],
            p_nozzle=point["nozzle_pressure_bar"],
            t_amb=point["ambient_temperature_celsius"],
            phi_amb=point["ambient_relative_humidity"],
        )
        solution_own_nozzle = solve_steady_state(
            GreyBoxFluidBedGranulator(variant_own_nozzle),
            max_iterations=operating_point.DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=operating_point.DEFAULT_SEARCH_TOLERANCE,
        )
        _, fixed_nozzle_info = env.evaluate_operating_point(
            inlet_temperature_celsius=point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        deltas_kw.append(
            fixed_nozzle_info["energy_target"] - solution_own_nozzle.diagnostics["energy_target"]
        )

    max_abs_delta_kw = max(abs(delta) for delta in deltas_kw)
    # Observed: ~3.3e-16 kW (floating-point noise). The 1e-9 kW bound leaves ample margin,
    # but stays far below any physically meaningful order of magnitude.
    assert max_abs_delta_kw < 1.0e-9
    assert env._base_config.gas.pressure_pa / 1.0e5 > FIXED_NOZZLE_PRESSURE_BAR


def test_reset_with_explicit_disturbance_is_deterministic(
    env: OperatingPointBanditEnv,
) -> None:
    obs1, info1 = env.reset(
        options={"ambient_temperature_celsius": 12.5, "ambient_relative_humidity": 0.6}
    )
    obs2, info2 = env.reset(
        options={"ambient_temperature_celsius": 12.5, "ambient_relative_humidity": 0.6}
    )
    np.testing.assert_array_equal(obs1, obs2)
    assert info1 == info2


def test_reset_without_options_samples_within_disturbance_bounds() -> None:
    env = OperatingPointBanditEnv()
    obs, _ = env.reset(seed=0)
    observation_space = env.observation_space
    assert isinstance(observation_space, spaces.Box)
    assert np.all(obs >= observation_space.low) and np.all(obs <= observation_space.high)


def test_action_space_box_matches_normalisation_convention() -> None:
    env = OperatingPointBanditEnv()
    action_space = env.action_space
    assert isinstance(action_space, spaces.Box)
    assert action_space.shape == (2,)
    np.testing.assert_array_equal(action_space.low, np.array([-1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(action_space.high, np.array([1.0, 1.0], dtype=np.float32))


def test_step_before_reset_raises() -> None:
    env = OperatingPointBanditEnv()
    with pytest.raises(RuntimeError):
        env.step(np.array([0.0, 0.0], dtype=np.float32))


def test_episode_always_terminates_after_one_step() -> None:
    env = OperatingPointBanditEnv()
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated is True
    assert truncated is False


def test_config_override_uses_custom_penalty_weight() -> None:
    default_env = OperatingPointBanditEnv()
    custom_env = OperatingPointBanditEnv(BanditEnvConfig(penalty_weight_kw=5.0))
    assert custom_env._penalty_weight_kw == 5.0
    assert default_env._penalty_weight_kw == pytest.approx(50.0)


def test_gymnasium_check_env_passes() -> None:
    gymnasium_utils = pytest.importorskip("gymnasium.utils.env_checker")
    env = OperatingPointBanditEnv()
    gymnasium_utils.check_env(env, skip_render_check=True)
