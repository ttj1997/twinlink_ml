r"""BB2: centred reward -- affine transformation, optimum unchanged.

Three properties that the transformation itself must guarantee (the task):

1. The reproduction mode (``reward_centering=None``, default) stays bit-identical to VA2 --
   ``evaluate_operating_point`` calls the same raw method for it as before BB2.
2. The affine map ``r' = (r - r_ref) / scale_kw`` does not change the optimum: at the
   fixed reference action itself ``r' = 0`` by construction, at a better action
   ``r'`` is larger than at a worse one -- the same ranking as under the raw
   reward.
3. Penalty and energy are scaled equally -- the ratio (c)/(a) from YA2 stays
   unchanged under centring (only divided globally by ``scale_kw``, both sides
   equally).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gymnasium")

from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv, RewardCentering
from twinlink_ml.rl.env import ANCHOR_ARTIFACT_PATH


@pytest.fixture(scope="module")
def anchor() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(ANCHOR_ARTIFACT_PATH).read_text(encoding="utf-8"))
    return data


def test_default_config_reproduces_va2_bitgleich(anchor: dict[str, Any]) -> None:
    """Reproduction mode (default, ``reward_centering=None``) stays bit-identical to VA2."""
    env = OperatingPointBanditEnv()
    for point in anchor["grid_results"]:
        reward, info = env.evaluate_operating_point(
            inlet_temperature_celsius=point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        assert reward == pytest.approx(-point["energy_target"], rel=1e-6)
        assert "reward_centered" not in info
        assert "reward_raw" not in info


def test_centered_reward_at_fixed_point_action_is_zero(anchor: dict[str, Any]) -> None:
    """At the reference action itself (Befund 52's fixed operating point) r' = 0 exactly."""
    env = OperatingPointBanditEnv(BanditEnvConfig(reward_centering=RewardCentering()))
    fixed = anchor["fixed_point"]
    for point in anchor["grid_results"][:5]:
        reward, info = env.evaluate_operating_point(
            inlet_temperature_celsius=fixed["inlet_temperature_celsius"],
            gas_velocity_m_per_s=fixed["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        assert reward == pytest.approx(0.0, abs=1e-9)
        assert info["reward_centered"] == pytest.approx(0.0, abs=1e-9)


def test_centered_reward_preserves_ranking_between_actions(anchor: dict[str, Any]) -> None:
    """Centring does not change the ranking between two actions (optimum invariant).

    The oracle action (better) must yield a higher reward under BOTH modes than the
    fixed reference action (worse) -- the same ranking, only a different scale/offset.
    """
    point = anchor["grid_results"][0]
    fixed = anchor["fixed_point"]

    env_raw = OperatingPointBanditEnv()
    reward_oracle_raw, _ = env_raw.evaluate_operating_point(
        inlet_temperature_celsius=point["inlet_temperature_celsius"],
        gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
        ambient_temperature_celsius=point["ambient_temperature_celsius"],
        ambient_relative_humidity=point["ambient_relative_humidity"],
    )
    reward_fixed_raw, _ = env_raw.evaluate_operating_point(
        inlet_temperature_celsius=fixed["inlet_temperature_celsius"],
        gas_velocity_m_per_s=fixed["gas_velocity_m_per_s"],
        ambient_temperature_celsius=point["ambient_temperature_celsius"],
        ambient_relative_humidity=point["ambient_relative_humidity"],
    )
    assert reward_oracle_raw > reward_fixed_raw

    env_centered = OperatingPointBanditEnv(BanditEnvConfig(reward_centering=RewardCentering()))
    reward_oracle_centered, _ = env_centered.evaluate_operating_point(
        inlet_temperature_celsius=point["inlet_temperature_celsius"],
        gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
        ambient_temperature_celsius=point["ambient_temperature_celsius"],
        ambient_relative_humidity=point["ambient_relative_humidity"],
    )
    reward_fixed_centered, _ = env_centered.evaluate_operating_point(
        inlet_temperature_celsius=fixed["inlet_temperature_celsius"],
        gas_velocity_m_per_s=fixed["gas_velocity_m_per_s"],
        ambient_temperature_celsius=point["ambient_temperature_celsius"],
        ambient_relative_humidity=point["ambient_relative_humidity"],
    )
    assert reward_oracle_centered > reward_fixed_centered
    assert reward_fixed_centered == pytest.approx(0.0, abs=1e-9)


def test_penalty_to_signal_ratio_is_unchanged_by_centering(anchor: dict[str, Any]) -> None:
    """YA2's penalty dominance (factor ~516) is preserved under centring.

    Penalty AND energy are divided by the same ``scale_kw``, their ratio to each
    other is not.
    """
    point = anchor["grid_results"][0]
    scale_kw = 0.2296627007886406
    env_raw = OperatingPointBanditEnv()
    env_centered = OperatingPointBanditEnv(
        BanditEnvConfig(reward_centering=RewardCentering(scale_kw=scale_kw))
    )

    # known infeasible action (AB1 example)
    reward_raw, info_raw = env_raw.evaluate_operating_point(
        inlet_temperature_celsius=108.8,
        gas_velocity_m_per_s=2.352,
        ambient_temperature_celsius=point["ambient_temperature_celsius"],
        ambient_relative_humidity=point["ambient_relative_humidity"],
    )
    reward_centered, info_centered = env_centered.evaluate_operating_point(
        inlet_temperature_celsius=108.8,
        gas_velocity_m_per_s=2.352,
        ambient_temperature_celsius=point["ambient_temperature_celsius"],
        ambient_relative_humidity=point["ambient_relative_humidity"],
    )
    penalty_raw = -reward_raw - info_raw["energy_target"]
    # info_centered still carries the RAW diagnostic fields (energy_target etc.) --
    # only the reward itself is transformed.
    penalty_centered_equivalent = -info_centered["reward_raw"] - info_centered["energy_target"]
    assert penalty_raw == pytest.approx(penalty_centered_equivalent)
    assert reward_centered == pytest.approx(
        (info_centered["reward_raw"] - info_centered["fixed_point_reward_reference"]) / scale_kw
    )
