r"""YA1: Zufallsbaseline -- reine Auswertung, KEIN Training.

Gleichverteilte Aktionen im normierten Aktionsraum ``[-1, 1]^2`` (also gleichverteilt ueber
die ``ActionConfig``-Grenzen, siehe ``OperatingPointBanditEnv``s Normierung), ausgewertet an
denselben 25 Befund-52-Testpunkten wie XA3. ``--n-repetitions`` Wiederholungen (Default 20)
mit verschiedenen Seeds, damit die Streuung sichtbar wird -- Median und Spannweite
(Min/Max) ueber die Wiederholungen, neben XA3s vier Checkpoints gestellt.

Ausgewertet mit der XA1-korrigierten Metrik (siehe ``twinlink_ml.rl.evaluation``):
(a) Zulaessigkeitsrate, (b) gepoolter Anteil geschlossenen Regrets NUR ueber zulaessige
Aktionen (rohe Energie ohne Strafe), (c) Verletzungen nach Nebenbedingung. Skala fuer (b):
fester Betriebspunkt = 0, Orakel = 1 (Definition der Metrik).

Kernfrage (Auftrag), getrennt beantwortet:
  - Zulaessigkeit: liegt XA3 (10^4 Schritte) ueber, auf oder unter Zufallsniveau?
  - Anteil ueber zulaessige Aktionen: dasselbe.

Erwartete Lesart, zu pruefen und nicht vorauszusetzen (Auftrag): der Agent liegt bei der
Zulaessigkeit ueber dem Zufall, beim Energieanteil etwa auf Zufallsniveau -- er haette
gelernt, der Strafe auszuweichen, aber noch nichts ueber die Energie.

Aufruf::

    python scripts/probe_bandit_env_random_baseline.py \
        --output artifacts/runs/rl_bandit_random_baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.rl import OperatingPointBanditEnv, evaluate_policy, flat_metrics
from twinlink_ml.rl.evaluation import PolicyFn
from twinlink_ml.simulation import PlaceholderConfigWarning
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_random_baseline.jsonl"
XA3_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_sac_learning_curve_xa3.jsonl"
DEFAULT_N_REPETITIONS = 20

_SPREAD_KEYS = (
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


def _make_random_policy(seed: int) -> PolicyFn:
    """Gleichverteilte Aktion in ``[-1, 1]^2``, EIN ``Generator`` je Wiederholung.

    Der Generator wird bei Konstruktion einmal geseedet und dann ueber alle 25 Testpunkte
    einer Wiederholung weiterverwendet (nicht je Aufruf neu geseedet) -- eine Wiederholung
    ist damit EIN reproduzierbarer Ziehungsstrom ueber die 25 Punkte, keine 25 unabhaengig
    geseedete Einzelziehungen.
    """
    rng = np.random.default_rng(seed)

    def policy(observation: np.ndarray) -> np.ndarray:
        del observation
        return rng.uniform(-1.0, 1.0, size=2)

    return policy


def run_random_baseline(seeds: tuple[int, ...], output: Path) -> list[dict[str, float]]:
    env = OperatingPointBanditEnv()
    env_info = environment_fingerprint()
    logger = JsonlExperimentLogger(path=output)
    all_metrics: list[dict[str, float]] = []
    for seed in seeds:
        result = evaluate_policy(env, _make_random_policy(seed))
        metrics = flat_metrics(result)
        all_metrics.append(metrics)
        logger.log(
            RunRecord(
                group="rl_bandit_random_baseline",
                tags={"policy": "uniform_random", "seed": str(seed)},
                params={**env_info, "seed": seed, "n_repetitions": len(seeds)},
                metrics=metrics,
            )
        )
    return all_metrics


def summarise(all_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Median/Min/Max je Groesse ueber die Wiederholungen -- ``nan`` (z. B. ``pooled_
    closed_regret_fraction`` bei 0 zulaessigen Aktionen in einer Wiederholung) wird
    ausgeschlossen, ``*_n_defined`` zaehlt, wie viele der Wiederholungen einen definierten
    Wert beigetragen haben.
    """
    summary: dict[str, dict[str, float]] = {}
    for key in _SPREAD_KEYS:
        values = np.asarray([m[key] for m in all_metrics], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = {
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
            "n_defined": float(finite.size),
        }
    return summary


def _load_xa3_checkpoints(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return sorted(records, key=lambda record: int(record["tags"]["checkpoint"]))


def print_comparison(
    summary: dict[str, dict[str, float]], xa3_path: Path, n_repetitions: int
) -> None:
    print("\n--- Zufall (Median [Min, Max] ueber die Wiederholungen) vs. XA3 " + "-" * 10)
    feasibility = summary["feasibility_rate"]
    pooled = summary["pooled_closed_regret_fraction"]
    print(
        f"Zufall   (a)={feasibility['median']:.2f} "
        f"[{feasibility['min']:.2f}, {feasibility['max']:.2f}]   "
        f"(b)={pooled['median']:.4f} [{pooled['min']:.4f}, {pooled['max']:.4f}] "
        f"(b definiert in {int(pooled['n_defined'])}/{n_repetitions} Wiederholungen)"
    )
    if not xa3_path.exists():
        print(f"(XA3-Artefakt {xa3_path} nicht gefunden -- kein direkter Vergleich gedruckt)")
        return
    for record in _load_xa3_checkpoints(xa3_path):
        m = record["metrics"]
        checkpoint = record["tags"]["checkpoint"]
        pooled_value = m["pooled_closed_regret_fraction"]
        pooled_text = f"{pooled_value:.4f}" if np.isfinite(pooled_value) else "n. def."
        print(
            f"XA3 {checkpoint:>6s}  (a)={m['feasibility_rate']:.2f}  (b)={pooled_text}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-repetitions", type=int, default=DEFAULT_N_REPETITIONS)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--xa3-artifact", type=Path, default=XA3_ARTIFACT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()
    seeds = tuple(range(arguments.n_repetitions))

    print("=" * 78)
    print(f"Zufallsbaseline      {arguments.n_repetitions} Wiederholungen, Seeds {seeds[0]}..{seeds[-1]}")
    print("Aktion               gleichverteilt in [-1, 1]^2, keine Politik, kein Training")
    print("=" * 78)

    all_metrics = run_random_baseline(seeds, arguments.output)
    summary = summarise(all_metrics)

    for key in _SPREAD_KEYS:
        s = summary[key]
        print(
            f"{key:45s} median={s['median']:9.4f}  "
            f"[{s['min']:9.4f}, {s['max']:9.4f}]  n_def={int(s['n_defined'])}/{len(seeds)}"
        )

    print_comparison(summary, arguments.xa3_artifact, arguments.n_repetitions)
    print(f"\nZufallsbaseline geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
