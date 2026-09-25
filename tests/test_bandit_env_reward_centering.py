r"""BB2: zentrierte Belohnung -- affine Transformation, Optimum unveraendert.

Drei Eigenschaften, die die Transformation selbst garantieren muss (Auftrag):

1. Der Reproduktionsmodus (``reward_centering=None``, Default) bleibt bitgleich zu VA2 --
   ``evaluate_operating_point`` ruft dafuer dieselbe rohe Methode wie vor BB2 auf.
2. Die affine Abbildung ``r' = (r - r_ref) / scale_kw`` aendert das Optimum nicht: an der
   festen Referenzaktion selbst ist ``r' = 0`` per Konstruktion, an einer besseren Aktion
   ist ``r'`` groesser als an einer schlechteren -- dieselbe Rangfolge wie unter der rohen
   Belohnung.
3. Strafe und Energie werden gleich skaliert -- das Verhaeltnis (c)/(a) aus YA2 bleibt
   unter Zentrierung unveraendert (nur global durch ``scale_kw`` geteilt, beide Seiten
   gleichermassen).
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
    """Reproduktionsmodus (Default, ``reward_centering=None``) bleibt bitgleich zu VA2."""
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
    """An der Referenzaktion selbst (Befund 52s fester Betriebspunkt) ist r' = 0 exakt."""
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
    """Zentrierung aendert die Rangfolge zwischen zwei Aktionen nicht (Optimum invariant).

    Oracle-Aktion (besser) muss unter BEIDEN Modi eine hoehere Belohnung liefern als die
    feste Referenzaktion (schlechter) -- dieselbe Rangfolge, nur andere Skala/Offset.
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
    """YA2s Strafdominanz (Faktor ~516) bleibt unter Zentrierung erhalten.

    Strafe UND Energie werden durch denselben ``scale_kw`` geteilt, ihr Verhaeltnis
    zueinander nicht.
    """
    point = anchor["grid_results"][0]
    scale_kw = 0.2296627007886406
    env_raw = OperatingPointBanditEnv()
    env_centered = OperatingPointBanditEnv(
        BanditEnvConfig(reward_centering=RewardCentering(scale_kw=scale_kw))
    )

    # bekannt unzulaessige Aktion (AB1-Beispiel)
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
    # info_centered traegt weiterhin die ROHEN Diagnosefelder (energy_target etc.) --
    # nur die Belohnung selbst ist transformiert.
    penalty_centered_equivalent = -info_centered["reward_raw"] - info_centered["energy_target"]
    assert penalty_raw == pytest.approx(penalty_centered_equivalent)
    assert reward_centered == pytest.approx(
        (info_centered["reward_raw"] - info_centered["fixed_point_reward_reference"]) / scale_kw
    )
