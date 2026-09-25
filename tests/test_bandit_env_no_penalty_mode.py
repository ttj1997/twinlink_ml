r"""DA1/EA2: Belohnung ohne Strafe, begrenzter Loeserfehlschlagwert.

Nebenbedingungen bleiben geprueft, nur nicht in der Belohnung. Reproduktionsmodus
(``include_penalty=True``, ``solve_failure_penalty_kw=None``, beides Default) bleibt
bitgleich zu VA2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gymnasium")

from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv
from twinlink_ml.rl.env import ANCHOR_ARTIFACT_PATH


@pytest.fixture(scope="module")
def anchor() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(ANCHOR_ARTIFACT_PATH).read_text(encoding="utf-8"))
    return data


def test_default_config_still_reproduces_va2_bitgleich(anchor: dict[str, Any]) -> None:
    """``include_penalty=True`` (Default) bleibt bitgleich zu VA2 -- unveraendert."""
    env = OperatingPointBanditEnv()
    for point in anchor["grid_results"]:
        reward, _ = env.evaluate_operating_point(
            inlet_temperature_celsius=point["inlet_temperature_celsius"],
            gas_velocity_m_per_s=point["gas_velocity_m_per_s"],
            ambient_temperature_celsius=point["ambient_temperature_celsius"],
            ambient_relative_humidity=point["ambient_relative_humidity"],
        )
        assert reward == pytest.approx(-point["energy_target"], rel=1e-6)


def test_no_penalty_reward_equals_negative_energy_regardless_of_feasibility() -> None:
    """``include_penalty=False``: Belohnung ist immer ``-energy_target``.

    Auch bei unzulaessigen Aktionen -- die Zulaessigkeit wird weiterhin berichtet.
    """
    env = OperatingPointBanditEnv(BanditEnvConfig(include_penalty=False))

    # bekannt unzulaessige Aktion (heisse Ecke, AB3/DA0-Beispiel)
    reward, info = env.evaluate_operating_point(
        inlet_temperature_celsius=120.0,
        gas_velocity_m_per_s=2.5,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    assert not info["feasible"]
    assert reward == pytest.approx(-info["energy_target"])
    assert info["constraint_check"]["bed_temperature_violation_k"] > 0.0

    # bekannt zulaessige Aktion
    reward_feasible, info_feasible = env.evaluate_operating_point(
        inlet_temperature_celsius=90.0,
        gas_velocity_m_per_s=1.6,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    assert info_feasible["feasible"]
    assert reward_feasible == pytest.approx(-info_feasible["energy_target"])


def test_no_penalty_reward_differs_from_penalized_reward_when_infeasible() -> None:
    """Bei identischer unzulaessiger Aktion muss der Strafmodus deutlich negativer sein.

    Sonst wirkt der Schalter nicht.
    """
    env_penalized = OperatingPointBanditEnv()
    env_no_penalty = OperatingPointBanditEnv(BanditEnvConfig(include_penalty=False))

    kwargs = {
        "inlet_temperature_celsius": 120.0,
        "gas_velocity_m_per_s": 2.5,
        "ambient_temperature_celsius": 20.0,
        "ambient_relative_humidity": 0.5,
    }
    reward_penalized, _ = env_penalized.evaluate_operating_point(**kwargs)
    reward_no_penalty, _ = env_no_penalty.evaluate_operating_point(**kwargs)
    assert reward_penalized < reward_no_penalty


def test_default_solve_failure_penalty_still_uses_va2_sentinel() -> None:
    """``solve_failure_penalty_kw=None`` bleibt bitgleich zu VA2.

    Der Sentinel aus Befund 52 (``_SOLVE_FAILURE_PENALTY_KW`` = 1.000.000), unveraendert.
    """
    env = OperatingPointBanditEnv()
    # bekannter Loeserfehlschlagpunkt (EA1(b)/DA0(a))
    reward, info = env.evaluate_operating_point(
        inlet_temperature_celsius=60.0,
        gas_velocity_m_per_s=0.8,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    assert info["solve_failed"]
    assert reward == pytest.approx(-1_000_000.0)
    assert info["solve_failure_penalty_kw"] == pytest.approx(1_000_000.0)


def test_bounded_solve_failure_penalty_overrides_sentinel() -> None:
    """EA2: ein gesetzter ``solve_failure_penalty_kw`` ersetzt den Sentinel exakt."""
    env = OperatingPointBanditEnv(BanditEnvConfig(solve_failure_penalty_kw=4.5))
    reward, info = env.evaluate_operating_point(
        inlet_temperature_celsius=60.0,
        gas_velocity_m_per_s=0.8,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    assert info["solve_failed"]
    assert reward == pytest.approx(-4.5)
    assert info["solve_failure_penalty_kw"] == pytest.approx(4.5)


def test_fa1_bounded_solve_failure_penalty_with_penalty_mode() -> None:
    """Pinnt den fuer FA2 verwendeten Grenzwert im DEFAULT-Modus "mit Strafe".

    Aus ``probe_bandit_env_solve_failure_bound.py`` (``with_penalty``, 155.57896052362634
    kW -- schlechteste Summe aus Energie und Strafe ueber die loesbaren 13x9-Gitterpunkte,
    erreicht bei t_in=65 °C/u_gas=0.8 m/s, siehe Befund 62 DA0(a)). Ersetzt den Sentinel
    bitgleich, auch im DEFAULT-Modus "mit Strafe" (``include_penalty=True``) -- derselbe
    generische Override-Mechanismus wie EA2s ohne-Strafe-Grenzwert, hier mit dem
    tatsaechlich fuer FA2 verwendeten Wert gepinnt, damit eine spaetere Aenderung an diesem
    Wert hier auffaellt statt still im Trainingslauf zu verschwinden. ``include_penalty``
    bleibt Default (True) -- kein zweiter Modus-Schalter noetig, der Sentinel-Override ist
    von ``include_penalty`` unabhaengig (siehe env.py).
    """
    env = OperatingPointBanditEnv(BanditEnvConfig(solve_failure_penalty_kw=155.57896052362634))
    reward, info = env.evaluate_operating_point(
        inlet_temperature_celsius=60.0,
        gas_velocity_m_per_s=0.8,
        ambient_temperature_celsius=20.0,
        ambient_relative_humidity=0.5,
    )
    assert info["solve_failed"]
    assert reward == pytest.approx(-155.57896052362634)
    assert info["solve_failure_penalty_kw"] == pytest.approx(155.57896052362634)
