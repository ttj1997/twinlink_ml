r"""S0.3: SAC with a Lagrangian moisture constraint on the one-step bandit (Befund 68 draft).

One run = one seed. SAC, stable-baselines3 DEFAULT hyperparameters (as FA2), 10 000 steps,
checkpoints at 1000 / 2500 / 5000 / 10000 (as FA2). Only the treatment of the moisture
constraint differs from the penalty arm (``run_rl_bandit_learning_curve.py`` with
``--solve-failure-penalty-kw 155.57896052362634``): see ``twinlink_ml.rl.lagrange``.

The multiplier settings have NO defaults on purpose. They are set on the command line after
the user has confirmed them (the S0.3 instruction: propose, then wait)::

    python scripts/run_rl_bandit_lagrange.py --seed 0 \
        --margin-kg-per-kg 0.005961819615175404 --initial-multiplier-kw ... \
        --step-size-kw ... --update-interval-steps ... --context-bins 5 5

Evaluation always runs on the PLAIN ``OperatingPointBanditEnv`` (penalty reward, TRUE quality
limit), on the 25 Befund 52 grid points (test set: ambient temperature x relative humidity
grid, disjoint from the continuous training draws). Logged per run: checkpoint records
(``RunRecord``), a per-step trace (context, bin, lambda, E, g_tilde, moisture, losses) and the
lambda history of every dual update. The gate criteria are fixed in
``scripts/analyze_s03_gate.py``, which was committed before any run.

The replay buffer is relabelled after every dual update (``relabel_rewards``): the bandit has
no bootstrapping, so a reward stored under an old multiplier would be a wrong regression
target. The transition of the very step that triggers an update is relabelled one step
later (it is stored after the callback runs); SAC takes at most one gradient step on it with
the old label.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.rl import (
    BanditEnvConfig,
    OperatingPointBanditEnv,
    evaluate_policy,
    flat_metrics,
    load_test_points,
)
from twinlink_ml.rl.evaluation import moisture_margin_at_oracle_binding_points
from twinlink_ml.rl.lagrange import (
    LagrangeBanditEnv,
    LagrangeConfig,
    dual_ascent,
    relabel_rewards,
)
from twinlink_ml.rl.shield import ShieldConfig, ShieldedLagrangeBanditEnv
from twinlink_ml.simulation import PlaceholderConfigWarning
from twinlink_ml.utils.seeding import seed_everything
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "artifacts" / "runs"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "rl_bandit_lagrange_s03"
CHECKPOINTS = (1_000, 2_500, 5_000, 10_000)
MARGIN_SOURCE_ARTIFACT = RUNS / "verify_fa2_spread_calibration.json"


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def margin_from_fa2_artifact() -> float:
    """Median half-width of the calibrated band, excited-trained arm, EV3 settled binding."""
    data = json.loads(MARGIN_SOURCE_ARTIFACT.read_text(encoding="utf-8"))
    value = data["excited"]["ev3_settled_binding"]["primary_leave_one_out"][
        "median_band_half_width_2k_spread"
    ]
    return float(value)


_QUALITY_LIMIT_KG_PER_KG: float | None = None
_TEST_POINTS_ARTIFACT: Path | None = None


def _eval_env() -> OperatingPointBanditEnv:
    return OperatingPointBanditEnv(
        BanditEnvConfig(product_moisture_limit_kg_per_kg=_QUALITY_LIMIT_KG_PER_KG)
    )


def _test_points() -> Any:
    if _TEST_POINTS_ARTIFACT is None:
        return load_test_points()
    return load_test_points(_TEST_POINTS_ARTIFACT)


def mean_energy_vs_fixed(model: Any) -> dict[str, float]:
    env = _eval_env()
    points = _test_points()
    energies: list[float] = []
    for point in points:
        observation = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity], dtype=np.float32
        )
        action, _ = model.predict(observation, deterministic=True)
        t_in = env._inlet_bound.minimum + 0.5 * (float(action[0]) + 1.0) * env._inlet_bound.span
        u_gas = env._gas_bound.minimum + 0.5 * (float(action[1]) + 1.0) * env._gas_bound.span
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=t_in,
            gas_velocity_m_per_s=u_gas,
            ambient_temperature_celsius=point.ambient_temperature_celsius,
            ambient_relative_humidity=point.ambient_relative_humidity,
        )
        if not info.get("solve_failed"):
            energies.append(float(info["energy_target"]))
    fixed = float(np.mean([p.energy_fixed_kw for p in points]))
    agent = float(np.mean(energies)) if energies else float("nan")
    return {
        "mean_energy_agent_kw": agent,
        "mean_energy_fixed_kw": fixed,
        "mean_energy_delta_kw": agent - fixed,
    }


def make_callback(
    *,
    algorithm: str,
    target_entropy: float | None,
    action_noise_sigma: float | None,
    lagrange: LagrangeConfig,
    logger: JsonlExperimentLogger,
    trace_path: Path,
    env_info: dict[str, Any],
    seed: int,
    run_uid: str,
    run_start: float,
    shield: ShieldConfig | None = None,
) -> Any:
    from stable_baselines3.common.callbacks import BaseCallback

    remaining = list(CHECKPOINTS)

    class _LagrangeCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.energy: list[float] = []
            self.rest: list[float] = []
            self.g_tilde: list[float] = []
            self.failed: list[bool] = []
            self.bin: list[int] = []
            self.trace: list[dict[str, Any]] = []
            self.lambda_history: list[dict[str, Any]] = []
            self.window_start = 0
            self.pending_relabel = False

        @property
        def lag_env(self) -> LagrangeBanditEnv:
            env = self.training_env.envs[0].unwrapped  # type: ignore[attr-defined]
            assert isinstance(env, LagrangeBanditEnv)
            return env

        def _relabel(self) -> None:
            buffer = self.model.replay_buffer
            n = len(self.energy)
            rewards = relabel_rewards(
                energy_kw=np.array(self.energy),
                rest_penalty_kw=np.array(self.rest),
                g_tilde=np.nan_to_num(np.array(self.g_tilde)),
                solve_failed=np.array(self.failed),
                bin_index=np.array(self.bin),
                multipliers=self.lag_env.multipliers,
                table=self.lag_env.reference_table,
            )
            stored = min(n, buffer.buffer_size if buffer.full else buffer.pos)
            buffer.rewards[:stored, 0] = rewards[:stored].astype(buffer.rewards.dtype)

        def _on_step(self) -> bool:
            if self.pending_relabel:
                self._relabel()
                self.pending_relabel = False
            info = self.locals["infos"][0]
            failed = bool(info.get("solve_failed"))
            self.failed.append(failed)
            self.bin.append(int(info["lagrange_bin"]))
            self.energy.append(float(info.get("lagrange_energy_kw", 0.0)) if not failed else 0.0)
            self.rest.append(
                float(info.get("lagrange_rest_penalty_kw", 0.0)) if not failed else 0.0
            )
            self.g_tilde.append(float(info["lagrange_g_tilde"]))
            values = self.model.logger.name_to_value
            self.trace.append(
                {
                    "step": self.num_timesteps,
                    "bin": self.bin[-1],
                    "lambda_kw": float(info["lagrange_multiplier_kw"]),
                    "energy_kw": self.energy[-1],
                    "g_tilde": self.g_tilde[-1],
                    "solve_failed": failed,
                    "feasible": bool(info["feasible"]),
                    "moisture_kg_per_kg": float(
                        info.get("constraint_check", {}).get("moisture_kg_per_kg", float("nan"))
                    ),
                    "reward": float(self.locals["rewards"][0]),
                    "critic_loss": float(values.get("train/critic_loss", float("nan"))),
                    "actor_loss": float(values.get("train/actor_loss", float("nan"))),
                    "ent_coef": float(values.get("train/ent_coef", float("nan"))),
                }
            )
            if shield is not None:
                self.trace[-1].update(
                    {
                        "shield_active": bool(info["shield_active"]),
                        "shield_saturated": bool(info["shield_saturated"]),
                        "proposed_t_in_celsius": float(
                            info["shield_proposed_inlet_temperature_celsius"]
                        ),
                        "executed_t_in_celsius": float(
                            info["shield_executed_inlet_temperature_celsius"]
                        ),
                        "proposed_moisture_kg_per_kg": float(
                            info["shield_proposed_moisture_kg_per_kg"]
                        ),
                        "shield_n_solves": int(info["shield_n_solves"]),
                        "env_step_seconds": float(info["shield_step_seconds"]),
                    }
                )

            if self.num_timesteps % lagrange.update_interval_steps == 0:
                self._dual_update()
            if remaining and self.num_timesteps >= remaining[0]:
                self._checkpoint(remaining.pop(0))
            return True

        def _dual_update(self) -> None:
            lam = self.lag_env.multipliers
            flat = lam.reshape(-1).copy()
            window = slice(self.window_start, len(self.g_tilde))
            bins = np.array(self.bin[window])
            gs = np.array(self.g_tilde[window])
            failed = np.array(self.failed[window])
            n_solved: list[int] = []
            for b in range(flat.size):
                mask = (bins == b) & ~failed
                n_solved.append(int(mask.sum()))
                if mask.any():
                    flat[b] = dual_ascent(
                        float(flat[b]), float(gs[mask].mean()), lagrange.step_size_kw
                    )
            self.lag_env.set_multipliers(flat.reshape(lam.shape))
            self.lambda_history.append(
                {
                    "step": self.num_timesteps,
                    "multipliers_kw": flat.tolist(),
                    "n_solved_per_bin": n_solved,
                    "n_failed_in_window": int(failed.sum()),
                }
            )
            self.window_start = len(self.g_tilde)
            self._relabel()
            self.pending_relabel = True

        def _checkpoint(self, checkpoint: int) -> None:
            wall = time.perf_counter() - run_start
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_file = CHECKPOINT_DIR / f"{run_uid}_step{checkpoint:06d}.zip"
            self.model.save(checkpoint_file)

            def policy(observation: np.ndarray) -> np.ndarray:
                action, _ = self.model.predict(observation, deterministic=True)
                return np.asarray(action, dtype=np.float64)

            result = evaluate_policy(_eval_env(), policy, _test_points())
            metrics = flat_metrics(result)
            metrics.update(mean_energy_vs_fixed(self.model))
            margin_value, margin_n = moisture_margin_at_oracle_binding_points(
                _eval_env(), policy, _test_points()
            )
            metrics["moisture_margin_binding_median_kg_per_kg"] = margin_value
            metrics["moisture_margin_binding_n"] = float(margin_n)
            lam = self.lag_env.multipliers
            metrics["lambda_mean_kw"] = float(lam.mean())
            metrics["lambda_max_kw"] = float(lam.max())
            metrics["lambda_min_kw"] = float(lam.min())
            print(
                f"[seed {seed}] step {checkpoint}: feasibility={metrics['feasibility_rate']:.2f} "
                f"pooled={metrics['pooled_closed_regret_fraction']:.4f} "
                f"margin={metrics['moisture_margin_binding_median_kg_per_kg']:.4f} "
                f"lambda mean={metrics['lambda_mean_kw']:.3f} ({wall:.0f} s)",
                flush=True,
            )
            logger.log(
                RunRecord(
                    run_uid=f"{run_uid}_step{checkpoint:06d}",
                    group="rl_bandit_lagrange_s03",
                    tags={"algorithm": algorithm, "checkpoint": str(checkpoint), "arm": "lagrange"},
                    params={
                        **env_info,
                        "seed": seed,
                        "n_env_steps": checkpoint,
                        "total_timesteps_planned": max(CHECKPOINTS),
                        "wall_s_since_training_start": wall,
                        "hyperparameters": "stable-baselines3 default, keine Abstimmung",
                        "lagrange": dataclasses.asdict(lagrange),
                        "algorithm": algorithm,
                        "target_entropy": target_entropy,
                        "action_noise_sigma": action_noise_sigma,
                        "shield": dataclasses.asdict(shield) if shield is not None else None,
                        "test_set": "25 Befund-52 grid points (ambient T x relative humidity)",
                        "checkpoint/path": str(checkpoint_file.relative_to(REPO_ROOT)),
                        "infeasible_violations": [
                            dataclasses.asdict(v) for v in result.infeasible_violations
                        ],
                        "multipliers_kw": lam.reshape(-1).tolist(),
                        "quality_limit_kg_per_kg": _QUALITY_LIMIT_KG_PER_KG,
                        "test_points_artifact": (
                            str(_TEST_POINTS_ARTIFACT) if _TEST_POINTS_ARTIFACT else None
                        ),
                    },
                    metrics=metrics,
                )
            )

        def _on_training_end(self) -> None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("w", encoding="utf-8") as handle:
                for row in self.trace:
                    handle.write(json.dumps(row) + "\n")
            history_path = trace_path.with_name(trace_path.name.replace("trace", "lambda"))
            with history_path.open("w", encoding="utf-8") as handle:
                for row in self.lambda_history:
                    handle.write(json.dumps(row) + "\n")

    return _LagrangeCallback()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--margin-kg-per-kg", type=float, required=True)
    parser.add_argument("--initial-multiplier-kw", type=float, required=True)
    parser.add_argument("--step-size-kw", type=float, required=True)
    parser.add_argument("--update-interval-steps", type=int, required=True)
    parser.add_argument("--context-bins", type=int, nargs=2, default=(5, 5))
    parser.add_argument("--output-prefix", type=str, default="rl_bandit_lagrange_s03")
    parser.add_argument(
        "--algorithm",
        choices=["SAC", "TD3"],
        default="SAC",
        help="S0.8: TD3 = deterministic policy (stable-baselines3 default, no action noise).",
    )
    parser.add_argument(
        "--action-noise-sigma",
        type=float,
        default=None,
        help="S0.8: TD3 Gaussian action noise (normalised action units); default: none.",
    )
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help="S0.8: SAC target entropy; default: stable-baselines3 auto (-dim(action) = -2).",
    )
    parser.add_argument(
        "--shield",
        action="store_true",
        help="SH1: execute the moisture-shielded action (see twinlink_ml.rl.shield).",
    )
    parser.add_argument(
        "--quality-limit-kg-per-kg",
        type=float,
        default=None,
        help="S0.6: quality limit of the product moisture (kinetics unchanged).",
    )
    parser.add_argument(
        "--test-points-artifact",
        type=Path,
        default=None,
        help="S0.6: oracle/fixed reference artifact (load_test_points format).",
    )
    arguments = parser.parse_args()
    global _QUALITY_LIMIT_KG_PER_KG, _TEST_POINTS_ARTIFACT
    _QUALITY_LIMIT_KG_PER_KG = arguments.quality_limit_kg_per_kg
    _TEST_POINTS_ARTIFACT = arguments.test_points_artifact
    _suppress_placeholder_warning()

    margin = arguments.margin_kg_per_kg
    stored_margin = margin_from_fa2_artifact()
    if abs(margin - stored_margin) > 1e-15:
        print(
            f"HINWEIS: margin {margin!r} differs from the FA2 artifact value {stored_margin!r}",
            flush=True,
        )

    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.monitor import Monitor

    lagrange = LagrangeConfig(
        margin_kg_per_kg=margin,
        initial_multiplier_kw=arguments.initial_multiplier_kw,
        step_size_kw=arguments.step_size_kw,
        update_interval_steps=arguments.update_interval_steps,
        context_bins=(int(arguments.context_bins[0]), int(arguments.context_bins[1])),
    )
    env_info = environment_fingerprint()
    if env_info["git/dirty"]:
        print("WARNUNG: Arbeitsbaum ist dirty -- git/dirty wird in jeden Record geschrieben.")
    seed_everything(arguments.seed)
    run_uid = uuid.uuid4().hex[:12]
    output = RUNS / f"{arguments.output_prefix}_seed{arguments.seed}.jsonl"
    trace = RUNS / f"{arguments.output_prefix}_trace_seed{arguments.seed}.jsonl"
    if output.exists() or trace.exists():
        raise SystemExit(f"{output} / {trace} exist already; use another --output-prefix")
    logger = JsonlExperimentLogger(output)

    env_config = BanditEnvConfig(product_moisture_limit_kg_per_kg=_QUALITY_LIMIT_KG_PER_KG)
    shield = ShieldConfig() if arguments.shield else None
    env: LagrangeBanditEnv = (
        ShieldedLagrangeBanditEnv(lagrange, env_config, shield=shield)
        if shield is not None
        else LagrangeBanditEnv(lagrange, env_config)
    )
    env.reference_table  # noqa: B018  (build the failure-bound table before training)
    train_env = Monitor(env)
    algorithm_cls = {"SAC": SAC, "TD3": TD3}[arguments.algorithm]
    model_kwargs: dict[str, Any] = {"seed": arguments.seed, "verbose": 0}
    if arguments.target_entropy is not None:
        if arguments.algorithm != "SAC":
            raise SystemExit("--target-entropy applies to SAC only")
        model_kwargs["target_entropy"] = arguments.target_entropy
    if arguments.action_noise_sigma is not None:
        if arguments.algorithm != "TD3":
            raise SystemExit("--action-noise-sigma applies to TD3 only")
        from stable_baselines3.common.noise import NormalActionNoise

        model_kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(2), sigma=arguments.action_noise_sigma * np.ones(2)
        )
    model = algorithm_cls("MlpPolicy", train_env, **model_kwargs)
    run_start = time.perf_counter()
    callback = make_callback(
        algorithm=arguments.algorithm,
        target_entropy=arguments.target_entropy,
        action_noise_sigma=arguments.action_noise_sigma,
        lagrange=lagrange,
        logger=logger,
        trace_path=trace,
        env_info=env_info,
        seed=arguments.seed,
        run_uid=run_uid,
        run_start=run_start,
        shield=shield,
    )
    model.learn(total_timesteps=max(CHECKPOINTS), callback=callback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
