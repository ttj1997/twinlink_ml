r"""AB1/AB2: zwei Pfade, die VA2 NICHT geprueft hat.

VA2 hat nur zulaessige Orakelaktionen eingespeist (immer physikalisch, nie durch die
Normierung). Zwei Fragen blieben offen, ausgeloest durch den Befund, dass der Agent bei
der Zulaessigkeit UNTER dem Zufallsniveau liegt (YA1) -- eine Erklaerung ("strafdominierter
Gradient", YA2) haette einen Agenten erwarten lassen, der die Strafe als staerkstes Signal
zuerst vermeidet, also UEBER Zufall bei der Zulaessigkeit liegt. Das passt nicht zusammen,
also: ist ein Pfad kaputt, den VA2 nicht abdeckt?

AB1 -- Vorzeichen der Strafe: eine bekannt unzulaessige Aktion muss eine DEUTLICH
negativere Belohnung ergeben als eine bekannt zulaessige Aktion am selben Kontext.

AB2 -- der normierte Aktionspfad ``[-1, 1] -> physikalisch``, ueber den ein Agent
tatsaechlich handelt (VA2 hat physikalische Aktionen direkt eingespeist, nie durch die
Normierung). Drei Pruefungen: Orakelaktionen durch den normierten Pfad -> bitgleich zu VA2;
die vier Ecken -> plausible physikalische Extremwerte, insbesondere ``[1, 1]`` bei
maximaler Zulufttemperatur UND maximaler Gasgeschwindigkeit (der "heissen Ecke").

Befund (siehe Auftragstext/Befund-Entwurf): beide Pruefungen finden HIER keinen Fehler --
Vorzeichen und Normierung sind korrekt. Die Erklaerung fuer "unter Zufall" liegt woanders
(AB3: Politikkollaps auf die heisse Ecke, siehe ``scripts/probe_bandit_env_policy_
inspection.py``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from twinlink_ml.rl import OperatingPointBanditEnv
from twinlink_ml.rl.env import ANCHOR_ARTIFACT_PATH, _normalise
from twinlink_ml.simulation import PlaceholderConfigWarning


@pytest.fixture(scope="module")
def anchor() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(ANCHOR_ARTIFACT_PATH).read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def env() -> OperatingPointBanditEnv:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return OperatingPointBanditEnv()


# --------------------------------------------------------------------------------- AB1


def test_infeasible_action_reward_is_much_more_negative_than_feasible(
    env: OperatingPointBanditEnv,
) -> None:
    """AB1: bekannt unzulaessige gegen bekannt zulaessige Aktion, gleicher Kontext.

    Unzulaessig: XA3s Beispiel (Zulufttemperatur 108.8 °C, Gasgeschwindigkeit 2.352 m/s,
    Nominalstoergroesse) -- Bettemperatur 86.97 °C, 6.97 K ueber der Grenze. Zulaessig:
    die nominelle Aktion (90 °C, 1.6 m/s) am selben Kontext.
    """
    reward_infeasible, info_infeasible = env.evaluate_operating_point(
        inlet_temperature_celsius=108.8,
        gas_velocity_m_per_s=2.352,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    reward_feasible, info_feasible = env.evaluate_operating_point(
        inlet_temperature_celsius=90.0,
        gas_velocity_m_per_s=1.6,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    assert not info_infeasible["feasible"]
    assert info_feasible["feasible"]

    # Deutlich negativer, nicht nur numerisch kleiner -- Groessenordnungspruefung, nicht
    # nur eine Vorzeichenpruefung.
    assert reward_feasible - reward_infeasible > 10.0

    # Zerlegung: Belohnung = -(Energie + Strafe) -- Strafe muss positiv und dominant sein.
    penalty_infeasible = -reward_infeasible - info_infeasible["energy_target"]
    assert penalty_infeasible > 0.0
    assert penalty_infeasible == pytest.approx(34.85, abs=0.01)
    assert info_feasible["constraint_check"]["normalized_violation"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------- AB2


def test_action_space_corners_map_to_expected_physical_extremes(
    env: OperatingPointBanditEnv,
) -> None:
    """AB2: die vier Ecken von ``[-1, 1]^2`` muessen plausible Extreme ergeben.

    ``[1, 1]`` muss die "heisse Ecke" sein (maximale Zulufttemperatur UND maximale
    Gasgeschwindigkeit), nicht irgendeine vertauschte Kombination.
    """
    env.reset(options={"ambient_temperature_celsius": 20.0, "ambient_relative_humidity": 0.5})
    _, _, _, _, info_hot = env.step(np.array([1.0, 1.0], dtype=np.float64))
    assert info_hot["action_physical_applied"]["inlet_temperature_celsius"] == pytest.approx(
        env._inlet_bound.maximum
    )
    assert info_hot["action_physical_applied"]["gas_velocity_m_per_s"] == pytest.approx(
        env._gas_bound.maximum
    )

    env.reset(options={"ambient_temperature_celsius": 20.0, "ambient_relative_humidity": 0.5})
    _, _, _, _, info_cold = env.step(np.array([-1.0, -1.0], dtype=np.float64))
    assert info_cold["action_physical_applied"]["inlet_temperature_celsius"] == pytest.approx(
        env._inlet_bound.minimum
    )
    assert info_cold["action_physical_applied"]["gas_velocity_m_per_s"] == pytest.approx(
        env._gas_bound.minimum
    )

    env.reset(options={"ambient_temperature_celsius": 20.0, "ambient_relative_humidity": 0.5})
    _, _, _, _, info_mixed_a = env.step(np.array([-1.0, 1.0], dtype=np.float64))
    assert info_mixed_a["action_physical_applied"]["inlet_temperature_celsius"] == pytest.approx(
        env._inlet_bound.minimum
    )
    assert info_mixed_a["action_physical_applied"]["gas_velocity_m_per_s"] == pytest.approx(
        env._gas_bound.maximum
    )

    env.reset(options={"ambient_temperature_celsius": 20.0, "ambient_relative_humidity": 0.5})
    _, _, _, _, info_mixed_b = env.step(np.array([1.0, -1.0], dtype=np.float64))
    assert info_mixed_b["action_physical_applied"]["inlet_temperature_celsius"] == pytest.approx(
        env._inlet_bound.maximum
    )
    assert info_mixed_b["action_physical_applied"]["gas_velocity_m_per_s"] == pytest.approx(
        env._gas_bound.minimum
    )


def test_oracle_actions_through_normalised_step_path_reproduce_va2(
    env: OperatingPointBanditEnv, anchor: dict[str, Any]
) -> None:
    """AB2: dieselben 25 Orakelaktionen wie VA2, aber DURCH den normierten Pfad.

    VA2 selbst ruft ``evaluate_operating_point`` mit physikalischen Werten direkt auf, nie
    durch ``_normalise``/``_denormalise`` -- hier ueber ``step()``, muss bitgleich bleiben.
    """
    for point in anchor["grid_results"]:
        env.reset(
            options={
                "ambient_temperature_celsius": point["ambient_temperature_celsius"],
                "ambient_relative_humidity": point["ambient_relative_humidity"],
            }
        )
        normalised_action = np.array(
            [
                _normalise(
                    point["inlet_temperature_celsius"],
                    env._inlet_bound.minimum,
                    env._inlet_bound.maximum,
                ),
                _normalise(
                    point["gas_velocity_m_per_s"], env._gas_bound.minimum, env._gas_bound.maximum
                ),
            ],
            dtype=np.float64,
        )
        _, reward, _, _, _ = env.step(normalised_action)
        assert reward == pytest.approx(-point["energy_target"], rel=1e-6)
