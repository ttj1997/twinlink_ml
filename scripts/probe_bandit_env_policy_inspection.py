r"""AB3: was die XA3-Politik tatsaechlich tut -- reine Auswertung, KEIN Training.

Laedt die in XA2 gesicherten SAC-Checkpoints (``artifacts/checkpoints/rl_bandit_sac_
learning_curve/``, per ``.gitignore`` nicht committet -- dieses Skript braucht einen
vorherigen Lauf von ``scripts/run_rl_bandit_learning_curve.py``) bei zwei Schrittzahlen
(Default 1000 und 10000) und berichtet fuer alle 25 Befund-52-Testkontexte die
DETERMINISTISCHE Aktion (``model.predict(obs, deterministic=True)`` -- Mittelwert der
tanh-gequetschten Gauss-Politik, KEINE stochastische Ziehung; dieselbe Einstellung wie in
``scripts/run_rl_bandit_learning_curve.py``/``twinlink_ml.rl.evaluation.evaluate_policy``,
siehe Auftrag "Zusaetzlich: wird deterministisch oder stochastisch gezogen").

Kernfragen (Auftrag):
  - Gibt die Politik fuer alle 25 Kontexte (nahezu) dieselbe Aktion aus (Kollaps, den
    Kontext ignorierend)?
  - Liegt diese Aktion an einer Ecke (normierte Werte nahe +-1, tanh-Saettigung)?
  - Welche Ecke -- insbesondere die "heisse" (hohe Zulufttemperatur, hohe
    Gasgeschwindigkeit), die mit einem Energieminimierer nicht vereinbar waere?

Aufruf::

    python scripts/probe_bandit_env_policy_inspection.py \
        --output artifacts/runs/rl_bandit_policy_inspection.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.rl import OperatingPointBanditEnv, load_test_points
from twinlink_ml.simulation import PlaceholderConfigWarning

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "rl_bandit_sac_learning_curve"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_policy_inspection.json"
DEFAULT_CHECKPOINT_STEPS = (1_000, 10_000)
COLLAPSE_STD_THRESHOLD = 0.05
"""Normierte Standardabweichung ueber die 25 Kontexte, unterhalb derer die Politik als
"kollabiert" (kontextunabhaengig) gilt -- 0.05 entspricht < 3 % der Aktionsspanne
``[-1, 1]``, deutlich unter der beobachteten Kontextvariation einer nicht-kollabierten
Achse (siehe (b) in ``twinlink_ml.rl.evaluation`` fuer die Groessenordnung realer
Energieunterschiede zwischen Kontexten)."""
CORNER_THRESHOLD = 0.9
"""Normierter Aktionswert, ab dessen Betrag eine Achse als "an der Schranke gesaettigt"
gilt (tanh-Saettigung) -- 0.9 entspricht 95 % des Wegs von der Mitte zur Schranke."""


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def _find_checkpoint(step: int, run_uid: str | None) -> Path:
    """Checkpoint fuer ``step`` im Verzeichnis (mehrere Laeufe moeglich, je mit eigenem
    ``run_uid``-Praefix).

    Mit ``run_uid``: exakter Treffer, NICHT mtime-basiert -- noetig, sobald mehrere
    Trainingslaeufe GLEICHZEITIG laufen (DA1s 2x2-Matrix teils parallelisiert), weil dann
    "zuletzt geschrieben" die Laeufe nicht mehr zuverlaessig unterscheidet. Ohne ``run_uid``
    (Default): zuletzt geschriebene Datei, wie zuvor -- nur korrekt, wenn kein anderer Lauf
    gleichzeitig schreibt.
    """
    if run_uid is not None:
        path = CHECKPOINT_DIR / f"{run_uid}_step{step:06d}.zip"
        if not path.exists():
            raise FileNotFoundError(f"{path} nicht gefunden.")
        return path
    candidates = sorted(
        CHECKPOINT_DIR.glob(f"*_step{step:06d}.zip"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(
            f"Kein Checkpoint fuer Schritt {step} unter {CHECKPOINT_DIR} gefunden -- "
            "scripts/run_rl_bandit_learning_curve.py muss vorher gelaufen sein."
        )
    return candidates[-1]


def inspect_checkpoint(path: Path, env: OperatingPointBanditEnv) -> dict[str, Any]:
    from stable_baselines3 import SAC

    model = SAC.load(path)
    points = load_test_points()

    per_point: list[dict[str, Any]] = []
    normalised_actions = np.zeros((len(points), 2), dtype=np.float64)
    for index, point in enumerate(points):
        observation = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity],
            dtype=np.float32,
        )
        action, _ = model.predict(observation, deterministic=True)
        normalised_actions[index] = action
        t_in = env._inlet_bound.minimum + 0.5 * (action[0] + 1.0) * env._inlet_bound.span
        u_gas = env._gas_bound.minimum + 0.5 * (action[1] + 1.0) * env._gas_bound.span
        per_point.append(
            {
                "ambient_temperature_celsius": point.ambient_temperature_celsius,
                "ambient_relative_humidity": point.ambient_relative_humidity,
                "action_normalised": [float(action[0]), float(action[1])],
                "inlet_temperature_celsius": float(t_in),
                "gas_velocity_m_per_s": float(u_gas),
            }
        )

    mean = normalised_actions.mean(axis=0)
    std = normalised_actions.std(axis=0)
    collapsed = bool(np.all(std < COLLAPSE_STD_THRESHOLD))
    at_corner = [bool(abs(mean[dim]) > CORNER_THRESHOLD) for dim in range(2)]
    corner_sign = [
        ("hoch" if mean[dim] > 0 else "niedrig") if at_corner[dim] else "nicht gesaettigt"
        for dim in range(2)
    ]

    return {
        "checkpoint_path": str(path.relative_to(REPO_ROOT)),
        "n_contexts": len(points),
        "normalised_action_mean": mean.tolist(),
        "normalised_action_std": std.tolist(),
        "normalised_action_min": normalised_actions.min(axis=0).tolist(),
        "normalised_action_max": normalised_actions.max(axis=0).tolist(),
        "collapsed_ignoring_context": collapsed,
        "collapse_std_threshold": COLLAPSE_STD_THRESHOLD,
        "at_corner_inlet_temperature": at_corner[0],
        "at_corner_gas_velocity": at_corner[1],
        "corner_direction_inlet_temperature": corner_sign[0],
        "corner_direction_gas_velocity": corner_sign[1],
        "is_hot_corner": at_corner[0]
        and at_corner[1]
        and corner_sign[0] == "hoch"
        and corner_sign[1] == "hoch",
        "per_point": per_point,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, nargs="+", default=list(DEFAULT_CHECKPOINT_STEPS)
    )
    parser.add_argument(
        "--run-uid",
        type=str,
        default=None,
        help="exakter run_uid statt mtime-basierter Auswahl -- noetig bei parallel "
        "laufenden Trainingslaeufen (siehe _find_checkpoint-Docstring).",
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()
    env = OperatingPointBanditEnv()

    print("=" * 78)
    print("Auswertung: DETERMINISTISCH (model.predict(obs, deterministic=True), Mittelwert")
    print("der tanh-gequetschten Gauss-Politik, KEINE stochastische Ziehung).")
    print("=" * 78)

    results: dict[str, Any] = {}
    for step in arguments.checkpoint_steps:
        checkpoint_path = _find_checkpoint(step, arguments.run_uid)
        print(f"\n--- Schritt {step} ({checkpoint_path.name}) " + "-" * 10)
        result = inspect_checkpoint(checkpoint_path, env)
        results[str(step)] = result

        mean = result["normalised_action_mean"]
        std = result["normalised_action_std"]
        print(f"normierte Aktion, Mittelwert ueber 25 Kontexte: t_in={mean[0]:+.4f}  "
              f"u_gas={mean[1]:+.4f}")
        print(f"normierte Aktion, Streuung ueber 25 Kontexte:   t_in={std[0]:.4f}  "
              f"u_gas={std[1]:.4f}  (Kollaps-Schwelle {COLLAPSE_STD_THRESHOLD})")
        print(f"Kollaps (kontextunabhaengig)? {result['collapsed_ignoring_context']}")
        print(f"an Schranke gesaettigt? t_in={result['at_corner_inlet_temperature']} "
              f"({result['corner_direction_inlet_temperature']})  "
              f"u_gas={result['at_corner_gas_velocity']} "
              f"({result['corner_direction_gas_velocity']})")
        print(f"HEISSE ECKE (hoch/hoch)? {result['is_hot_corner']}")
        print("erste 3 physikalische Aktionen:")
        for entry in result["per_point"][:3]:
            print(f"  t_amb={entry['ambient_temperature_celsius']:5.1f} "
                  f"phi={entry['ambient_relative_humidity']:.2f} -> "
                  f"t_in={entry['inlet_temperature_celsius']:6.2f} °C  "
                  f"u_gas={entry['gas_velocity_m_per_s']:.4f} m/s")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
