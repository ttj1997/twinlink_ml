r"""VA4: Rauchtest der RL-Bandit-Pipeline -- KEIN volles Training.

Der Hauptlauf des Projekts belegt die Kerne (siehe Auftrag) -- hier wird nur geprueft,
dass die Kette (``OperatingPointBanditEnv`` -> SB3-Algorithmus -> VA3-Metrik) steht, mit
sehr kleinem Budget und einem Seed. OB der Agent im Rauchtest schon etwas lernt, ist NICHT
die Frage (siehe Auftrag) -- die Frage ist, was ein ernsthaftes Training (Groessenordnung
10^4-10^5 Schritte) kosten wuerde und ob ``solve_steady_state`` dafuer als Uebergang reicht.

Drei Messungen:

1. Reine Umgebungsschrittkosten (``env.step()``, ohne SB3-Overhead), Hochrechnung auf
   10^4/10^5 Schritte.
2. Disjunktheit der gezogenen Trainingspunkte von den 25 VA3-Testpunkten (Laufzeitpruefung,
   siehe ``twinlink_ml.rl.evaluation.assert_disjoint_from_test_grid``).
3. Ein winziger SAC- oder TD3-Lauf (Standard: SAC) ueber ``stable_baselines3`` --
   ``total_timesteps`` per ``--total-timesteps`` (Default 64), EIN Seed. Die VA3-Metrik
   wird am Ende einmal berechnet, um die Pipeline zu pruefen -- NICHT als Trainingsergebnis
   (siehe Auftrag).

Aufruf::

    python scripts/probe_bandit_env_sac_smoke.py \
        --output artifacts/runs/probe_bandit_env_sac_smoke.json

Threading: MKL_NUM_THREADS=1/OMP_NUM_THREADS=1 als Prozessvariablen setzen, bevor dieses
Skript laeuft (siehe ``environment_fingerprint``s ``env/mkl_num_threads``/``env/omp_num_
threads`` -- Befund zur Faktor-5-Streuung durch Thread-/Prozess-Ueberzeichnung, ``docs/
04_stand_ap1.md``).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.rl import (
    OperatingPointBanditEnv,
    assert_disjoint_from_test_grid,
    evaluate_policy,
)
from twinlink_ml.simulation import PlaceholderConfigWarning
from twinlink_ml.utils.seeding import seed_everything
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_bandit_env_sac_smoke.json"
EXTRAPOLATION_STEP_COUNTS = (10_000, 100_000)


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def measure_env_step_cost(n_steps: int, *, seed: int) -> dict[str, Any]:
    """Reine ``env.step()``-Kosten, ohne SB3-Overhead -- getrennter Datenpunkt von der
    SB3-Trainingsmessung unten, damit Umgebungs- und Algorithmus-Overhead unterscheidbar
    bleiben.
    """
    env = OperatingPointBanditEnv()
    env.reset(seed=seed)
    elapsed_s: list[float] = []
    for _ in range(n_steps):
        action = env.action_space.sample()
        start = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(action)
        elapsed_s.append(time.perf_counter() - start)
        if terminated or truncated:
            env.reset()
    elapsed_arr = np.asarray(elapsed_s)
    return {
        "n_steps": n_steps,
        "median_s": float(np.median(elapsed_arr)),
        "mean_s": float(np.mean(elapsed_arr)),
        "p95_s": float(np.percentile(elapsed_arr, 95)),
        "min_s": float(np.min(elapsed_arr)),
        "max_s": float(np.max(elapsed_arr)),
        "total_s": float(np.sum(elapsed_arr)),
    }


def check_training_disjointness(n_draws: int, *, seed: int) -> dict[str, Any]:
    """VA3: belegt, dass ``reset()``-Ziehungen die 25 Testpunkte nicht treffen (siehe
    ``assert_disjoint_from_test_grid``-Docstring fuer die Mass-Null-Begruendung).
    """
    env = OperatingPointBanditEnv()
    draws = []
    for i in range(n_draws):
        obs, _ = env.reset(seed=seed * 1_000_003 + i)
        draws.append(obs)
    draws_arr = np.asarray(draws, dtype=np.float64)
    assert_disjoint_from_test_grid(draws_arr)
    return {"n_draws_checked": n_draws, "collision_found": False}


def run_smoke_training(
    algorithm: str, total_timesteps: int, *, seed: int
) -> tuple[dict[str, Any], Any]:
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.monitor import Monitor

    algorithm_cls = {"SAC": SAC, "TD3": TD3}[algorithm]
    train_env = Monitor(OperatingPointBanditEnv())
    learning_starts = min(16, max(total_timesteps - 1, 0))
    model = algorithm_cls(
        "MlpPolicy",
        train_env,
        seed=seed,
        learning_starts=learning_starts,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
    )
    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    wall_s = time.perf_counter() - start
    return (
        {
            "algorithm": algorithm,
            "total_timesteps": total_timesteps,
            "learning_starts": learning_starts,
            "wall_s": wall_s,
            "wall_s_per_step": wall_s / total_timesteps,
        },
        model,
    )


def evaluate_smoke_policy(model: Any) -> dict[str, Any]:
    """VA3-Metrik einmalig berechnet -- Pipeline-Pruefung, KEIN Trainingsergebnis (siehe
    Auftrag VA4: "Ob der Agent im Rauchtest schon etwas lernt, ist NICHT die Frage").
    """

    def policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    eval_env = OperatingPointBanditEnv()
    result = evaluate_policy(eval_env, policy)
    return {
        "n_points": result.n_points,
        "n_solve_failed": result.n_solve_failed,
        "feasibility_rate": result.feasibility_rate,
        "closed_fraction_median": result.closed_fraction_median,
        "closed_fraction_p95": result.closed_fraction_p95,
        "closed_fraction_min": result.closed_fraction_min,
        "mean_moisture_violation_of_infeasible_kg_per_kg": (
            result.mean_moisture_violation_of_infeasible_kg_per_kg
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algorithm", choices=["SAC", "TD3"], default="SAC")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=64,
        help="SB3-Budget, sehr klein (Rauchtest, kein Training, siehe Moduldocstring)",
    )
    parser.add_argument("--timing-steps", type=int, default=30)
    parser.add_argument("--disjointness-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()

    env_info = environment_fingerprint()
    if env_info["git/dirty"]:
        print(
            "WARNUNG: Arbeitsbaum ist dirty -- git/dirty wird in den Record geschrieben.",
            flush=True,
        )
    if env_info.get("env/mkl_num_threads") != "1" or env_info.get("env/omp_num_threads") != "1":
        print(
            "WARNUNG: MKL_NUM_THREADS/OMP_NUM_THREADS sind nicht beide auf '1' gesetzt "
            f"(env/mkl_num_threads={env_info.get('env/mkl_num_threads')!r}, "
            f"env/omp_num_threads={env_info.get('env/omp_num_threads')!r}) -- Zeitmessungen "
            "unten sind damit nicht mit einem Lauf mit gepinnten Threads vergleichbar.",
            flush=True,
        )

    seed_everything(arguments.seed)

    print("=" * 78)
    print(f"Algorithmus         {arguments.algorithm}")
    print(f"total_timesteps     {arguments.total_timesteps}")
    print(f"Seed                {arguments.seed}")
    print(f"git/commit={env_info['git/commit']}  git/dirty={env_info['git/dirty']}")
    print("=" * 78)

    print(f"\n--- Reine env.step()-Kosten ({arguments.timing_steps} Schritte) " + "-" * 10)
    timing = measure_env_step_cost(arguments.timing_steps, seed=arguments.seed)
    print(
        f"median={timing['median_s'] * 1000:.1f} ms  mean={timing['mean_s'] * 1000:.1f} ms  "
        f"p95={timing['p95_s'] * 1000:.1f} ms"
    )

    extrapolation: dict[str, Any] = {}
    for n_steps in EXTRAPOLATION_STEP_COUNTS:
        hours = timing["median_s"] * n_steps / 3600.0
        extrapolation[str(n_steps)] = {"estimated_wall_hours_single_process": hours}
        print(f"Hochrechnung {n_steps:>7d} Schritte: {hours:.2f} h (einstraengig, Median-Kosten)")

    print(f"\n--- VA3: Disjunktheit Trainings-/Testpunkte ({arguments.disjointness_draws} Ziehungen) " + "-" * 5)
    disjointness = check_training_disjointness(arguments.disjointness_draws, seed=arguments.seed)
    print(f"{disjointness['n_draws_checked']} Ziehungen geprueft, keine Kollision mit den 25 Testpunkten.")

    print(f"\n--- {arguments.algorithm}-Rauchtest (stable-baselines3, {arguments.total_timesteps} Schritte) " + "-" * 5)
    training_info, model = run_smoke_training(
        arguments.algorithm, arguments.total_timesteps, seed=arguments.seed
    )
    print(
        f"wall={training_info['wall_s']:.2f} s "
        f"({training_info['wall_s_per_step'] * 1000:.1f} ms/Schritt inkl. SB3-Overhead)"
    )

    print("\n--- VA3-Metrik (Pipeline-Pruefung, KEIN Trainingsergebnis) " + "-" * 10)
    smoke_policy_metric = evaluate_smoke_policy(model)
    print(
        f"feasibility_rate={smoke_policy_metric['feasibility_rate']:.2f}  "
        f"closed_fraction_median={smoke_policy_metric['closed_fraction_median']:.4f}"
    )

    output = {
        "environment": env_info,
        "arguments": vars(arguments) | {"output": str(arguments.output)},
        "env_step_timing": timing,
        "extrapolation": extrapolation,
        "training_disjointness_check": disjointness,
        "sb3_smoke_training": training_info,
        "smoke_policy_metric": smoke_policy_metric,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
