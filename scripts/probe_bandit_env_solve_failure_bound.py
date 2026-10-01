r"""EA2: modusabhaengiger Loeserfehlschlag-Grenzwert -- reine Auswertung, kein Training.

Befund 52s Sentinel (`_SOLVE_FAILURE_PENALTY_KW = 1.000.000`) ist vier bis fuenf
Groessenordnungen ueber jeder anderen Belohnung im Aktionsraum (siehe EA1) -- ein moeglicher
Kritiker-Vergiftungsmechanismus. Statt eines neu erfundenen festen Ersatzwerts (Auftrag,
Nutzerentscheidung nach Rueckfrage): der Loeserfehlschlag-Grenzwert ist die SCHLECHTESTE
Belohnung, die eine LOESBARE Aktion im jeweils AKTIVEN Modus ueber das bestehende
13x9-Gitter (Nominalstoergroesse, dasselbe Gitter wie DA0(a)) tatsaechlich erreicht -- ein
Loeserfehlschlag ist damit "so schlecht wie der schlechteste loesbare Betriebspunkt", in
jedem Modus, ohne Ausreisser.

Zwei Werte, je Modus:

* OHNE Strafe: minus die maximale rohe Energie ueber die loesbaren Gitterpunkte.
* MIT Strafe: die schlechteste Summe aus Energie und Strafe ueber die loesbaren
  Gitterpunkte (nicht notwendig derselbe Gitterpunkt wie ohne Strafe -- die Gutsfeuchte-
  Verletzung am kalten/langsamen Rand kann schlechter sein als die Bettemperaturverletzung
  an der heissen Ecke, siehe DA0(a)).

Aufruf::

    python scripts/probe_bandit_env_solve_failure_bound.py \
        --output artifacts/runs/rl_bandit_solve_failure_bound.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv
from twinlink_ml.simulation import PlaceholderConfigWarning

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_solve_failure_bound.json"
GRID_T_IN = np.linspace(60.0, 120.0, 13)
GRID_U_GAS = np.linspace(0.8, 2.5, 9)
NOMINAL_AMBIENT_TEMPERATURE_CELSIUS = 20.0
NOMINAL_AMBIENT_RELATIVE_HUMIDITY = 0.5


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def worst_solvable_reward(env: OperatingPointBanditEnv) -> dict[str, Any]:
    worst_t_in: float | None = None
    worst_u_gas: float | None = None
    worst_reward = float("inf")
    n_solved = 0
    n_solve_failed = 0
    for t_in in GRID_T_IN:
        for u_gas in GRID_U_GAS:
            reward, info = env.evaluate_operating_point(
                inlet_temperature_celsius=float(t_in),
                gas_velocity_m_per_s=float(u_gas),
                ambient_temperature_celsius=NOMINAL_AMBIENT_TEMPERATURE_CELSIUS,
                ambient_relative_humidity=NOMINAL_AMBIENT_RELATIVE_HUMIDITY,
            )
            if info.get("solve_failed"):
                n_solve_failed += 1
                continue
            n_solved += 1
            if reward < worst_reward:
                worst_reward = reward
                worst_t_in = float(t_in)
                worst_u_gas = float(u_gas)
    return {
        "worst_solvable_reward_kw": worst_reward,
        "solve_failure_penalty_kw": -worst_reward,
        "at_inlet_temperature_celsius": worst_t_in,
        "at_gas_velocity_m_per_s": worst_u_gas,
        "n_grid_points": len(GRID_T_IN) * len(GRID_U_GAS),
        "n_solved": n_solved,
        "n_solve_failed": n_solve_failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--quality-limit-kg-per-kg",
        type=float,
        default=None,
        help="S0.6: quality limit of the product moisture (kinetics unchanged); default: "
        "critical moisture as before.",
    )
    arguments = parser.parse_args()
    _suppress_placeholder_warning()

    limit = arguments.quality_limit_kg_per_kg
    env_no_penalty = OperatingPointBanditEnv(
        BanditEnvConfig(include_penalty=False, product_moisture_limit_kg_per_kg=limit)
    )
    env_with_penalty = OperatingPointBanditEnv(
        BanditEnvConfig(product_moisture_limit_kg_per_kg=limit)
    )

    print("=" * 78)
    print("EA2: schlechteste loesbare Belohnung je Modus, 13x9-Gitter, Nominalstoergroesse")
    print("=" * 78)

    result_no_penalty = worst_solvable_reward(env_no_penalty)
    print(
        f"\nOHNE Strafe: reward={result_no_penalty['worst_solvable_reward_kw']:.6f} kW "
        f"(Grenzwert {result_no_penalty['solve_failure_penalty_kw']:.6f} kW) bei "
        f"t_in={result_no_penalty['at_inlet_temperature_celsius']} °C, "
        f"u_gas={result_no_penalty['at_gas_velocity_m_per_s']} m/s "
        f"({result_no_penalty['n_solved']}/{result_no_penalty['n_grid_points']} loesbar)"
    )

    result_with_penalty = worst_solvable_reward(env_with_penalty)
    print(
        f"MIT Strafe:  reward={result_with_penalty['worst_solvable_reward_kw']:.6f} kW "
        f"(Grenzwert {result_with_penalty['solve_failure_penalty_kw']:.6f} kW) bei "
        f"t_in={result_with_penalty['at_inlet_temperature_celsius']} °C, "
        f"u_gas={result_with_penalty['at_gas_velocity_m_per_s']} m/s "
        f"({result_with_penalty['n_solved']}/{result_with_penalty['n_grid_points']} loesbar)"
    )

    output = {
        "quality_limit_kg_per_kg": limit,
        "without_penalty": result_no_penalty,
        "with_penalty": result_with_penalty,
        "grid": {
            "t_in_celsius": GRID_T_IN.tolist(),
            "u_gas_m_per_s": GRID_U_GAS.tolist(),
            "ambient_temperature_celsius": NOMINAL_AMBIENT_TEMPERATURE_CELSIUS,
            "ambient_relative_humidity": NOMINAL_AMBIENT_RELATIVE_HUMIDITY,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
