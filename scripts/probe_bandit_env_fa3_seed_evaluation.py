r"""FA3: Auswertung des FA2-Abschlusslaufs -- Median ueber drei Seeds.

Liest die drei FA2-Lernkurven-JSONLs (``scripts/run_rl_bandit_learning_curve.py --seed
{0,1,2} --solve-failure-penalty-kw 155.57896052362634``, MIT Strafe, siehe dessen
Docstring fuer die exakten Aufrufe) und berichtet an jedem Auswertungspunkt
(1000/2500/5000/10000), je Seed UND als Median ueber die drei Seeds:

  - die XA1-korrigierte Metrik (Zulaessigkeitsrate, gepoolter Anteil geschlossenen
    Regrets ueber zulaessige Aktionen, Verletzungen nach Nebenbedingung) -- direkt aus
    den bereits geloggten RunRecords, KEINE Neuberechnung.
  - denselben Anteil getrennt fuer Punkte mit Orakel-Optimum im Inneren/an einer
    Aktionsschranke (XA3) -- ebenfalls bereits geloggt.
  - die deterministischen Aktionen an den 25 Testkontexten, geladen aus den XA2-
    Checkpoints (Pfad steht im RunRecord unter ``checkpoint/path``): Streuung ueber den
    Kontext (Kollaps-Diagnostik wie AB3) und Abstand zur Orakelaktion, physikalisch UND
    normiert.

Zusaetzlich, an denselben 25 Testpunkten, zum Vergleich:
  - Zufallsbaseline (YA1, wiederverwendet aus ``artifacts/runs/rl_bandit_random_
    baseline.jsonl`` -- unveraendertes Environment/Testgitter, KEINE Neuberechnung).
  - fester Betriebspunkt (Regret-Anteil per Konstruktion 0) und Orakel (per Konstruktion
    1), hier frisch mit :func:`~twinlink_ml.rl.evaluate_policy` berechnet, keine
    Behauptung aus Erinnerung.

Kernfrage (Auftrag, ein Satz): findet der Agent das beschraenkte Optimum?

Aufruf::

    python scripts/probe_bandit_env_fa3_seed_evaluation.py \
        --output artifacts/runs/rl_bandit_fa3_seed_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.rl import OperatingPointBanditEnv, evaluate_policy, load_test_points
from twinlink_ml.rl.env import (
    FIXED_OPERATING_POINT_GAS_VELOCITY_M_PER_S,
    FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS,
    _normalise,
)
from twinlink_ml.rl.evaluation import TestPoint
from twinlink_ml.simulation import PlaceholderConfigWarning
from twinlink_ml.utils.tracking import RunRecord, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_JSONLS = tuple(
    REPO_ROOT / "artifacts" / "runs" / f"rl_bandit_sac_learning_curve_fa2_seed{seed}.jsonl"
    for seed in (0, 1, 2)
)
DEFAULT_RANDOM_BASELINE_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_random_baseline.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_fa3_seed_evaluation.json"
DEFAULT_CHECKPOINTS = (1_000, 2_500, 5_000, 10_000)
_SCALAR_METRIC_KEYS = (
    "feasibility_rate",
    "pooled_closed_regret_fraction",
    "feasibility_rate_interior",
    "pooled_closed_regret_fraction_interior",
    "feasibility_rate_boundary",
    "pooled_closed_regret_fraction_boundary",
    "n_violations_bed_temperature",
    "n_violations_moisture",
    "n_violations_gas_velocity_window",
    "n_violations_solve_failed",
)


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def _median_or_nan(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def _load_seed_records(paths: tuple[Path, ...]) -> dict[int, dict[int, RunRecord]]:
    """``seed -> {n_env_steps: RunRecord}``, ueber alle uebergebenen JSONL-Dateien."""
    by_seed: dict[int, dict[int, RunRecord]] = {}
    for path in paths:
        for record in read_jsonl(path):
            seed = int(record.params["seed"])
            step = int(record.params["n_env_steps"])
            by_seed.setdefault(seed, {})[step] = record
    return by_seed


def _inspect_checkpoint_actions(
    checkpoint_path: Path, env: OperatingPointBanditEnv, points: tuple[TestPoint, ...]
) -> dict[str, Any]:
    """Deterministische Aktionen an den 25 Testkontexten -- Streuung und Orakelabstand.

    Gleiches Vorgehen wie ``probe_bandit_env_policy_inspection.py``s ``inspect_checkpoint``
    (nicht importiert -- ``scripts/`` ist kein Package, siehe dortiger fehlender
    ``__init__.py``), erweitert um den Abstand zur je Kontext EIGENEN Orakelaktion
    (physikalisch UND normiert), das FEHLT dort und ist hier die eigentliche Frage.
    """
    from stable_baselines3 import SAC

    model = SAC.load(checkpoint_path)
    normalised = np.zeros((len(points), 2), dtype=np.float64)
    physical_distance_inlet = np.zeros(len(points), dtype=np.float64)
    physical_distance_gas = np.zeros(len(points), dtype=np.float64)
    normalised_distance = np.zeros(len(points), dtype=np.float64)

    for index, point in enumerate(points):
        observation = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity],
            dtype=np.float32,
        )
        action, _ = model.predict(observation, deterministic=True)
        normalised[index] = action
        t_in = env._inlet_bound.minimum + 0.5 * (float(action[0]) + 1.0) * env._inlet_bound.span
        u_gas = env._gas_bound.minimum + 0.5 * (float(action[1]) + 1.0) * env._gas_bound.span
        physical_distance_inlet[index] = abs(t_in - point.oracle_inlet_temperature_celsius)
        physical_distance_gas[index] = abs(u_gas - point.oracle_gas_velocity_m_per_s)
        oracle_inlet_norm = _normalise(
            point.oracle_inlet_temperature_celsius,
            env._inlet_bound.minimum,
            env._inlet_bound.maximum,
        )
        oracle_gas_norm = _normalise(
            point.oracle_gas_velocity_m_per_s, env._gas_bound.minimum, env._gas_bound.maximum
        )
        normalised_distance[index] = float(
            np.hypot(float(action[0]) - oracle_inlet_norm, float(action[1]) - oracle_gas_norm)
        )

    return {
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
        "normalised_action_mean": normalised.mean(axis=0).tolist(),
        "normalised_action_std": normalised.std(axis=0).tolist(),
        "mean_absolute_distance_to_oracle_inlet_temperature_celsius": float(
            physical_distance_inlet.mean()
        ),
        "mean_absolute_distance_to_oracle_gas_velocity_m_per_s": float(physical_distance_gas.mean()),
        "mean_normalised_distance_to_oracle": float(normalised_distance.mean()),
        "median_normalised_distance_to_oracle": float(np.median(normalised_distance)),
    }


def _fixed_policy(env: OperatingPointBanditEnv) -> Any:
    inlet_norm = _normalise(
        FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS,
        env._inlet_bound.minimum,
        env._inlet_bound.maximum,
    )
    gas_norm = _normalise(
        FIXED_OPERATING_POINT_GAS_VELOCITY_M_PER_S, env._gas_bound.minimum, env._gas_bound.maximum
    )
    fixed_action = np.array([inlet_norm, gas_norm], dtype=np.float64)

    def policy(_observation: np.ndarray) -> np.ndarray:
        return fixed_action

    return policy


def _oracle_policy(env: OperatingPointBanditEnv, points: tuple[TestPoint, ...]) -> Any:
    def policy(observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        point = min(
            points,
            key=lambda p: abs(p.ambient_temperature_celsius - obs[0])
            + abs(p.ambient_relative_humidity - obs[1]),
        )
        inlet_norm = _normalise(
            point.oracle_inlet_temperature_celsius,
            env._inlet_bound.minimum,
            env._inlet_bound.maximum,
        )
        gas_norm = _normalise(
            point.oracle_gas_velocity_m_per_s, env._gas_bound.minimum, env._gas_bound.maximum
        )
        return np.array([inlet_norm, gas_norm], dtype=np.float64)

    return policy


def _random_baseline_summary(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    feasibility = [r.metrics["feasibility_rate"] for r in records]
    pooled = [r.metrics["pooled_closed_regret_fraction"] for r in records]
    return {
        "n_repetitions": len(records),
        "feasibility_rate_median": _median_or_nan(feasibility),
        "feasibility_rate_min": float(np.min(feasibility)),
        "feasibility_rate_max": float(np.max(feasibility)),
        "pooled_closed_regret_fraction_median": _median_or_nan(pooled),
        "pooled_closed_regret_fraction_min": float(np.nanmin(pooled)),
        "pooled_closed_regret_fraction_max": float(np.nanmax(pooled)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed-jsonls", type=Path, nargs=3, default=list(DEFAULT_SEED_JSONLS))
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--random-baseline", type=Path, default=DEFAULT_RANDOM_BASELINE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    _suppress_placeholder_warning()

    env = OperatingPointBanditEnv()
    points = load_test_points()

    by_seed = _load_seed_records(tuple(arguments.seed_jsonls))
    seeds = sorted(by_seed)

    print("=" * 88)
    print(f"FA3: Median ueber Seeds {seeds} -- MIT Strafe, begrenzter Sentinel (FA1)")
    print("=" * 88)

    per_checkpoint: dict[str, Any] = {}
    for checkpoint in arguments.checkpoints:
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            record = by_seed[seed][checkpoint]
            scalar_metrics = {key: record.metrics[key] for key in _SCALAR_METRIC_KEYS}
            checkpoint_path = REPO_ROOT / record.params["checkpoint/path"]
            action_stats = _inspect_checkpoint_actions(checkpoint_path, env, points)
            per_seed[str(seed)] = {**scalar_metrics, **action_stats}

        median_metrics = {
            key: _median_or_nan([per_seed[str(s)][key] for s in seeds]) for key in _SCALAR_METRIC_KEYS
        }
        median_distance = _median_or_nan(
            [per_seed[str(s)]["mean_normalised_distance_to_oracle"] for s in seeds]
        )
        per_checkpoint[str(checkpoint)] = {
            "per_seed": per_seed,
            "median": {**median_metrics, "median_mean_normalised_distance_to_oracle": median_distance},
        }

        print(f"\n--- Schritt {checkpoint} " + "-" * 20)
        print(
            "  Median: (a) feas={feasibility_rate:.3f}  (b) pooled={pooled_closed_regret_fraction:.4f}  "
            "[innen feas={feasibility_rate_interior:.3f} pooled={pooled_closed_regret_fraction_interior:.4f} | "
            "rand feas={feasibility_rate_boundary:.3f} pooled={pooled_closed_regret_fraction_boundary:.4f}]".format(
                **median_metrics
            )
        )
        print(
            "  Median: (c) bett={n_violations_bed_temperature:.1f}  feuchte={n_violations_moisture:.1f}  "
            "gasfenster={n_violations_gas_velocity_window:.1f}  "
            "loeserfehler={n_violations_solve_failed:.1f}".format(**median_metrics)
        )
        print(f"  Median: mittlerer normierter Abstand zur Orakelaktion = {median_distance:.4f}")

    fixed_result = evaluate_policy(env, _fixed_policy(env), points)
    oracle_result = evaluate_policy(env, _oracle_policy(env, points), points)
    random_baseline = _random_baseline_summary(arguments.random_baseline)

    print("\n" + "=" * 88)
    print("Referenzen (an denselben 25 Testpunkten):")
    print(
        f"  fester Betriebspunkt : feas={fixed_result.feasibility_rate:.3f}  "
        f"pooled={fixed_result.pooled_closed_regret_fraction:.4f}"
    )
    print(
        f"  Orakel               : feas={oracle_result.feasibility_rate:.3f}  "
        f"pooled={oracle_result.pooled_closed_regret_fraction:.4f}"
    )
    print(
        "  Zufall (YA1, n={n_repetitions}) : feas_median={feasibility_rate_median:.3f} "
        "[{feasibility_rate_min:.3f}; {feasibility_rate_max:.3f}]  "
        "pooled_median={pooled_closed_regret_fraction_median:.4f} "
        "[{pooled_closed_regret_fraction_min:.4f}; {pooled_closed_regret_fraction_max:.4f}]".format(
            **random_baseline
        )
    )

    output = {
        "seeds": seeds,
        "per_checkpoint": per_checkpoint,
        "reference": {
            "fixed_operating_point": {
                "feasibility_rate": fixed_result.feasibility_rate,
                "pooled_closed_regret_fraction": fixed_result.pooled_closed_regret_fraction,
            },
            "oracle": {
                "feasibility_rate": oracle_result.feasibility_rate,
                "pooled_closed_regret_fraction": oracle_result.pooled_closed_regret_fraction,
            },
            "random_baseline_ya1": random_baseline,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
