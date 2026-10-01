r"""WA3/XA1-XA3: RL-Lernkurve -- Anteil geschlossenen Regrets ueber Umgebungsschritten.

EIN SAC-Lauf (stable-baselines3), ``total_timesteps=10_000``, EIN Seed, einsträngig --
laeuft NEBEN dem Hauptlauf (siehe Auftrag: keine Seed-Sweeps, kein 10^5-Lauf in dieser
Session). Hyperparameter durchgehend SB3-DEFAULT, keine Abstimmung.

Zwischenauswertung an festen Schrittzahlen (``--eval-checkpoints``, Default 1000, 2500,
5000, 10000): die XA1-korrigierte VA3-Metrik (siehe ``twinlink_ml.rl.evaluation``) auf den
25 Befund-52-Testpunkten -- IMMER Zulaessigkeitsrate (a) und gepoolter, nur-zulaessiger
Anteil (b) zusammen berichtet, nie (b) allein (siehe dortiger Moduldocstring fuer die
Begruendung). Jede Zwischenauswertung EIN ``RunRecord`` (``twinlink_ml.utils.tracking``),
angehaengt an ein JSONL-Log -- dasselbe Format wie die Architektur-Lernkurven
(``artifacts/runs/arch_gru.jsonl`` u. ae., ``scripts/run_arch_learning_curve_worker.py``).

XA2: an jedem Auswertungspunkt wird die Politik gesichert (SB3-natives ``model.save()``,
``.zip``) unter ``artifacts/checkpoints/rl_bandit_sac_learning_curve/`` -- per
``.gitignore`` (``artifacts/checkpoints/``) ausgeschlossen, NICHT committen. Pfad steht im
RunRecord unter ``checkpoint/path`` (Provenienz-Konvention aus ``utils/checkpoints.py``,
dort fuer Torch-Sequenzmodelle, hier fuer SB3-Policies analog angewendet). Anlass: der
WA3-Lauf liess sich nicht weiter diagnostizieren, weil keine Politik gespeichert war.

XA3: Befund-52-Optima werden teils an einer Aktionsschranke gefunden (im warmen Regime:
Gasgeschwindigkeit an der unteren Schranke, 23/25 Punkte). Eine tanh-gequetschte Politik
erreicht eine Schranke nur asymptotisch -- ein geschlossener Anteil von 1.0 ist dort fuer
SAC/TD3 STRUKTURELL nicht erreichbar. Metriken werden deshalb zusaetzlich getrennt fuer
Punkte mit Orakel-Optimum im Inneren und an einer Schranke berichtet.

Kernfrage (Auftrag, ein Satz): wie viel des negativen WA3-Ergebnisses war die Metrik und
wie viel der Agent?

BB2: ``--reward-centering`` trainiert gegen ``r' = (r - r_ref) / scale_kw`` statt der
rohen Belohnung (siehe ``twinlink_ml.rl.env.RewardCentering``) -- die ZWISCHENAUSWERTUNG
bleibt in JEDEM Fall die rohe VA3-Metrik (eigener, stets roher ``eval_env`` in
:func:`_evaluate`), unabhaengig davon, womit trainiert wurde. Kernfrage (Auftrag):
verschwindet der in AB3 gefundene Eckenkollaps?

Aufruf::

    python scripts/run_rl_bandit_learning_curve.py \
        --output artifacts/runs/rl_bandit_sac_learning_curve.jsonl

    python scripts/run_rl_bandit_learning_curve.py --reward-centering \
        --output artifacts/runs/rl_bandit_sac_learning_curve_bb2_centered.jsonl

CB2: ``--gamma`` setzt den Diskontfaktor explizit (SB3-Default sonst 0.99). Fuer einen
kontextuellen Bandit ist ``gamma=0`` der korrekte Wert -- es gibt keine Zukunft. Ist das
Episodenende korrekt gesetzt (siehe CB1), darf ``gamma`` keinen Einfluss auf das Ergebnis
haben.

    python scripts/run_rl_bandit_learning_curve.py --gamma 0.0 \
        --output artifacts/runs/rl_bandit_sac_learning_curve_cb2_gamma0.jsonl

DA1: ``--no-penalty`` trainiert gegen ``-energy_target`` ohne Strafterm (siehe
``twinlink_ml.rl.env.BanditEnvConfig.include_penalty``), ``--learning-starts`` setzt SB3s
Zufalls-Aufwaermphase explizit (SB3-Default sonst 100). 2x2-Matrix mit XA3 als Zelle
(Strafe, learning_starts=100): trennt die Strafklippe (YA2, Faktor 516) von frueher
Kritiker-Ausnutzung als Erklaerung fuer den Eckenkollaps. Jede Zwischenauswertung
protokolliert zusaetzlich die mittlere rohe Energie der deterministischen Politik an den 25
Testkontexten gegen den festen Betriebspunkt (``mean_energy_agent_kw``/``mean_energy_fixed_
kw``/``mean_energy_delta_kw`` in ``RunRecord.metrics``) -- das primaere Kriterium OHNE
Strafe, wo die XA1-Zulaessigkeitsmetrik laut Auftrag nicht aussagefaehig ist.

    python scripts/run_rl_bandit_learning_curve.py --no-penalty \
        --output artifacts/runs/rl_bandit_sac_learning_curve_da1_nopenalty_ls100.jsonl
    python scripts/run_rl_bandit_learning_curve.py --learning-starts 2000 \
        --output artifacts/runs/rl_bandit_sac_learning_curve_da1_penalty_ls2000.jsonl
    python scripts/run_rl_bandit_learning_curve.py --no-penalty --learning-starts 2000 \
        --output artifacts/runs/rl_bandit_sac_learning_curve_da1_nopenalty_ls2000.jsonl

EA2: ``--solve-failure-penalty-kw`` ersetzt Befund 52s Loeserfehlschlag-Sentinel
(``_SOLVE_FAILURE_PENALTY_KW`` = 1.000.000, vier bis fuenf Groessenordnungen ueber jeder
anderen Belohnung, siehe EA1) durch einen MODUSABHAENGIGEN Grenzwert -- die schlechteste
Belohnung, die eine LOESBARE Aktion im aktiven Modus tatsaechlich erreicht (siehe
``scripts/probe_bandit_env_solve_failure_bound.py``). Wiederholt DA1s Zelle "ohne Strafe,
learning_starts=100", nur mit begrenztem Loeserfehlschlagwert statt des Sentinels -- ohne
Strafe, damit ausschliesslich der Sentinel-Effekt gemessen wird.

    python scripts/run_rl_bandit_learning_curve.py --no-penalty \
        --solve-failure-penalty-kw 4.500066942725786 \
        --output artifacts/runs/rl_bandit_sac_learning_curve_ea2_nopenalty_bounded.jsonl

FA1/FA2: Abschlusslauf der RL-Voruntersuchung -- MIT Strafe UND begrenztem Sentinel.
``--solve-failure-penalty-kw`` ist modusunabhaengig (siehe ``env.py``,
``_evaluate_operating_point_raw``) und wurde bereits in EA2 fuer den Modus OHNE Strafe
verwendet; FA1 bestimmt denselben Grenzwert fuer den Modus MIT Strafe ueber dasselbe
13x9-Gitter (``probe_bandit_env_solve_failure_bound.py``, ``with_penalty``):
155.57896052362634 kW, schlechteste Summe aus Energie und Strafe unter den loesbaren
Gitterpunkten, bei t_in=65 °C/u_gas=0.8 m/s (Befund 62 DA0(a)-Punkt, dort bereits als
schlechteste Belohnung im Gitter dokumentiert). Kein Code-Fehler und kein neuer Modus
noetig -- derselbe Mechanismus wie EA2, hier ohne ``--no-penalty`` (Strafe bleibt aktiv,
Default). Drei Seeds, je einstraengig, parallel:

    python scripts/run_rl_bandit_learning_curve.py --seed 0 \
        --solve-failure-penalty-kw 155.57896052362634 \
        --output artifacts/runs/rl_bandit_sac_learning_curve_fa2_seed0.jsonl
    python scripts/run_rl_bandit_learning_curve.py --seed 1 \
        --solve-failure-penalty-kw 155.57896052362634 \
        --output artifacts/runs/rl_bandit_sac_learning_curve_fa2_seed1.jsonl
    python scripts/run_rl_bandit_learning_curve.py --seed 2 \
        --solve-failure-penalty-kw 155.57896052362634 \
        --output artifacts/runs/rl_bandit_sac_learning_curve_fa2_seed2.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.rl import (
    BanditEnvConfig,
    OperatingPointBanditEnv,
    RewardCentering,
    evaluate_policy,
    flat_metrics,
    load_test_points,
)
from twinlink_ml.rl.evaluation import ClosedRegretResult
from twinlink_ml.simulation import PlaceholderConfigWarning
from twinlink_ml.utils.seeding import seed_everything
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_sac_learning_curve.jsonl"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "rl_bandit_sac_learning_curve"
DEFAULT_CHECKPOINTS = (1_000, 2_500, 5_000, 10_000)
DEFAULT_TOTAL_TIMESTEPS = 10_000
DEFAULT_SEED = 0


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def _evaluate(model: Any) -> ClosedRegretResult:
    def policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    eval_env = OperatingPointBanditEnv()
    return evaluate_policy(eval_env, policy)


def _mean_energy_vs_fixed(model: Any) -> dict[str, float]:
    """DA1: mittlere rohe Energie der deterministischen Politik an den 25 Testkontexten
    gegen den festen Befund-52-Betriebspunkt -- UNABHAENGIG von Zulaessigkeit.

    Primaeres Kriterium fuer die ``--no-penalty``-Zellen (Auftrag: die XA1-Metrik ueber
    zulaessige Aktionen ist dort nicht aussagefaehig, weil das energieoptimale Setzen ohne
    Feuchtegrenze vermutlich unzulaessig ist) -- deshalb hier IMMER mitberechnet, auch mit
    Strafe, fuer die direkte Vergleichbarkeit ueber die 2x2-Matrix.
    """
    eval_env = OperatingPointBanditEnv()
    points = load_test_points()
    energies_agent: list[float] = []
    n_solve_failed = 0
    for point in points:
        observation = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity],
            dtype=np.float32,
        )
        action, _ = model.predict(observation, deterministic=True)
        t_in = (
            eval_env._inlet_bound.minimum
            + 0.5 * (float(action[0]) + 1.0) * eval_env._inlet_bound.span
        )
        u_gas = (
            eval_env._gas_bound.minimum + 0.5 * (float(action[1]) + 1.0) * eval_env._gas_bound.span
        )
        _, info = eval_env.evaluate_operating_point(
            inlet_temperature_celsius=t_in,
            gas_velocity_m_per_s=u_gas,
            ambient_temperature_celsius=point.ambient_temperature_celsius,
            ambient_relative_humidity=point.ambient_relative_humidity,
        )
        if info.get("solve_failed"):
            n_solve_failed += 1
            continue
        energies_agent.append(float(info["energy_target"]))

    mean_energy_fixed = float(np.mean([p.energy_fixed_kw for p in points]))
    mean_energy_agent = float(np.mean(energies_agent)) if energies_agent else float("nan")
    return {
        "mean_energy_agent_kw": mean_energy_agent,
        "mean_energy_fixed_kw": mean_energy_fixed,
        "mean_energy_delta_kw": mean_energy_agent - mean_energy_fixed,
        "n_energy_eval_solve_failed": float(n_solve_failed),
    }


def _make_checkpoint_callback(
    checkpoints: tuple[int, ...],
    logger: JsonlExperimentLogger,
    env_info: dict[str, Any],
    seed: int,
    algorithm: str,
    run_uid: str,
    run_start: float,
    reward_centering_scale_kw: float | None,
    gamma: float | None,
    include_penalty: bool,
    learning_starts: int | None,
    solve_failure_penalty_kw: float | None,
) -> Any:
    from stable_baselines3.common.callbacks import BaseCallback

    remaining = list(checkpoints)

    class _CheckpointEvalCallback(BaseCallback):
        def _on_step(self) -> bool:
            nonlocal remaining
            if remaining and self.num_timesteps >= remaining[0]:
                checkpoint = remaining.pop(0)
                wall_s_so_far = time.perf_counter() - run_start
                print(
                    f"\n--- Zwischenauswertung bei {checkpoint} Umgebungsschritten "
                    f"({wall_s_so_far:.1f} s seit Trainingsstart) " + "-" * 10
                )
                result = _evaluate(self.model)

                # XA2: Politik sichern -- SB3-natives Format, Pfad geht ins RunRecord.
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                checkpoint_file = CHECKPOINT_DIR / f"{run_uid}_step{checkpoint:06d}.zip"
                self.model.save(checkpoint_file)

                metrics = flat_metrics(result)
                metrics.update(_mean_energy_vs_fixed(self.model))
                print(
                    f"(DA1) mean_energy_agent={metrics['mean_energy_agent_kw']:.3f} kW  "
                    f"mean_energy_fixed={metrics['mean_energy_fixed_kw']:.3f} kW  "
                    f"delta={metrics['mean_energy_delta_kw']:+.3f} kW"
                )
                print(
                    "(a) feasibility_rate={:.2f}  (b) pooled_closed={:.4f}  "
                    "[interior n={} feas={:.2f} pooled={:.4f} | "
                    "boundary n={} feas={:.2f} pooled={:.4f}]".format(
                        metrics["feasibility_rate"],
                        metrics["pooled_closed_regret_fraction"],
                        result.n_interior_points,
                        metrics["feasibility_rate_interior"],
                        metrics["pooled_closed_regret_fraction_interior"],
                        result.n_boundary_points,
                        metrics["feasibility_rate_boundary"],
                        metrics["pooled_closed_regret_fraction_boundary"],
                    )
                )
                print(
                    f"(nennerempfindlich) median={metrics['closed_regret_fraction_median']:.4f}  "
                    f"p95={metrics['closed_regret_fraction_p95']:.4f}  "
                    f"min={metrics['closed_regret_fraction_min']:.4f}"
                )
                print(
                    f"(c) Verletzungen: bett={int(metrics['n_violations_bed_temperature'])}  "
                    f"feuchte={int(metrics['n_violations_moisture'])}  "
                    f"gasfenster={int(metrics['n_violations_gas_velocity_window'])}  "
                    f"loeserfehler={int(metrics['n_violations_solve_failed'])}"
                )

                record = RunRecord(
                    run_uid=f"{run_uid}_step{checkpoint:06d}",
                    group="rl_bandit_learning_curve",
                    tags={"algorithm": algorithm, "checkpoint": str(checkpoint)},
                    params={
                        **env_info,
                        "seed": seed,
                        "n_env_steps": checkpoint,
                        "total_timesteps_planned": max(checkpoints),
                        "wall_s_since_training_start": wall_s_so_far,
                        "hyperparameters": "stable-baselines3 default, keine Abstimmung",
                        "reward_centering_scale_kw": reward_centering_scale_kw,
                        "gamma": gamma if gamma is not None else self.model.gamma,
                        "gamma_explicitly_set": gamma is not None,
                        "include_penalty": include_penalty,
                        "learning_starts": (
                            learning_starts
                            if learning_starts is not None
                            else self.model.learning_starts
                        ),
                        "learning_starts_explicitly_set": learning_starts is not None,
                        "solve_failure_penalty_kw": solve_failure_penalty_kw,
                        "checkpoint/path": str(checkpoint_file.relative_to(REPO_ROOT)),
                        "infeasible_violations": [
                            dataclasses.asdict(v) for v in result.infeasible_violations
                        ],
                    },
                    metrics=metrics,
                )
                logger.log(record)
            return True

    return _CheckpointEvalCallback()


def run_training(
    algorithm: str,
    total_timesteps: int,
    checkpoints: tuple[int, ...],
    seed: int,
    output: Path,
    reward_centering_scale_kw: float | None,
    gamma: float | None,
    include_penalty: bool,
    learning_starts: int | None,
    solve_failure_penalty_kw: float | None,
) -> None:
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.monitor import Monitor

    algorithm_cls = {"SAC": SAC, "TD3": TD3}[algorithm]
    env_info = environment_fingerprint()
    if env_info["git/dirty"]:
        print("WARNUNG: Arbeitsbaum ist dirty -- git/dirty wird in jeden Record geschrieben.")
    if env_info.get("env/mkl_num_threads") != "1" or env_info.get("env/omp_num_threads") != "1":
        print(
            "WARNUNG: MKL_NUM_THREADS/OMP_NUM_THREADS sind nicht beide '1' "
            f"(env/mkl_num_threads={env_info.get('env/mkl_num_threads')!r}, "
            f"env/omp_num_threads={env_info.get('env/omp_num_threads')!r})."
        )

    seed_everything(seed)
    run_uid = uuid.uuid4().hex[:12]
    train_env_config = BanditEnvConfig(
        reward_centering=(
            RewardCentering(scale_kw=reward_centering_scale_kw)
            if reward_centering_scale_kw is not None
            else None
        ),
        include_penalty=include_penalty,
        solve_failure_penalty_kw=solve_failure_penalty_kw,
    )
    train_env = Monitor(OperatingPointBanditEnv(train_env_config))
    model_kwargs: dict[str, Any] = {"seed": seed, "verbose": 0}
    if gamma is not None:
        model_kwargs["gamma"] = gamma
    if learning_starts is not None:
        model_kwargs["learning_starts"] = learning_starts
    model = algorithm_cls("MlpPolicy", train_env, **model_kwargs)

    logger = JsonlExperimentLogger(path=output)
    run_start = time.perf_counter()
    callback = _make_checkpoint_callback(
        checkpoints,
        logger,
        env_info,
        seed,
        algorithm,
        run_uid,
        run_start,
        reward_centering_scale_kw,
        gamma,
        include_penalty,
        learning_starts,
        solve_failure_penalty_kw,
    )

    print("=" * 78)
    print(
        f"Algorithmus         {algorithm} (stable-baselines3, "
        f"Hyperparameter-Default ausser evtl. gamma/learning_starts)"
    )
    print(f"total_timesteps     {total_timesteps}")
    print(f"Zwischenauswertungen {checkpoints}")
    print(f"Seed                {seed}")
    print(f"run_uid             {run_uid}")
    print(
        f"Belohnung           "
        f"{'zentriert (scale_kw=' + str(reward_centering_scale_kw) + ')' if reward_centering_scale_kw is not None else 'roh (VA2-Modus)'}"
        f"{' -- OHNE Strafe (DA1)' if not include_penalty else ' -- MIT Strafe'}"
    )
    print(
        f"gamma               {model.gamma} "
        f"({'CB2: explizit gesetzt' if gamma is not None else 'SB3-Default'})"
    )
    print(
        f"learning_starts     {model.learning_starts} "
        f"({'DA1: explizit gesetzt' if learning_starts is not None else 'SB3-Default'})"
    )
    print(
        f"Loeserfehlschlag    "
        f"{f'begrenzt auf {solve_failure_penalty_kw:.6f} kW (EA2)' if solve_failure_penalty_kw is not None else 'Befund-52-Sentinel (1000000 kW)'}"
    )
    print(f"Checkpoints unter   {CHECKPOINT_DIR}")
    print(f"git/commit={env_info['git/commit']}  git/dirty={env_info['git/dirty']}")
    print("=" * 78)

    model.learn(total_timesteps=total_timesteps, callback=callback)
    wall_s_total = time.perf_counter() - run_start
    print(
        f"\nTraining fertig: {wall_s_total:.1f} s fuer {total_timesteps} Schritte "
        f"({1000.0 * wall_s_total / total_timesteps:.1f} ms/Schritt inkl. SB3-Overhead)."
    )
    print(f"Lernkurve geschrieben: {output}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algorithm", choices=["SAC", "TD3"], default="SAC")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument(
        "--eval-checkpoints",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
        help="Umgebungsschrittzahlen, an denen die Metrik berechnet und die Politik gesichert wird",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--reward-centering",
        action="store_true",
        help="BB2: trainiert gegen r' = (r - r_ref) / scale_kw statt der rohen Belohnung "
        "(siehe twinlink_ml.rl.env.RewardCentering) -- die Auswertung bleibt IMMER die "
        "rohe VA3-Metrik, unabhaengig von dieser Option.",
    )
    parser.add_argument(
        "--centering-scale-kw",
        type=float,
        default=None,
        help="nur mit --reward-centering: scale_kw fuer RewardCentering, Default "
        "twinlink_ml.rl.env.DEFAULT_REWARD_CENTERING_SCALE_KW",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="CB2: explizit gesetzter Diskontfaktor (z. B. 0.0 fuer den Bandit-Test) -- "
        "ohne diese Option gilt SB3s Algorithmus-Default (0.99 fuer SAC/TD3).",
    )
    parser.add_argument(
        "--no-penalty",
        action="store_true",
        help="DA1: trainiert gegen -energy_target ohne Strafterm (siehe "
        "twinlink_ml.rl.env.BanditEnvConfig.include_penalty) -- Nebenbedingungen bleiben "
        "geprueft/protokolliert, gehen nur nicht in die Belohnung ein.",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="DA1: explizit gesetzte Laenge der SB3-Zufalls-Aufwaermphase (SB3-Default sonst 100).",
    )
    parser.add_argument(
        "--solve-failure-penalty-kw",
        type=float,
        default=None,
        help="EA2: begrenzter Loeserfehlschlagwert statt Befund-52-Sentinel (1000000) -- "
        "siehe scripts/probe_bandit_env_solve_failure_bound.py fuer die modusabhaengige "
        "Berechnung.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()
    checkpoints = tuple(sorted(arguments.eval_checkpoints))
    if arguments.reward_centering:
        from twinlink_ml.rl.env import DEFAULT_REWARD_CENTERING_SCALE_KW

        scale_kw = (
            arguments.centering_scale_kw
            if arguments.centering_scale_kw is not None
            else DEFAULT_REWARD_CENTERING_SCALE_KW
        )
    else:
        scale_kw = None
    run_training(
        arguments.algorithm,
        arguments.total_timesteps,
        checkpoints,
        arguments.seed,
        arguments.output,
        scale_kw,
        arguments.gamma,
        not arguments.no_penalty,
        arguments.learning_starts,
        arguments.solve_failure_penalty_kw,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
