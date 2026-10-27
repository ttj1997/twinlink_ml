r"""Controller C1 (RL) of the study docs/30_surrogate_env.md: cell ``ressep`` of Befund 93/96 on U0-U3.

Pre-registration of the study (environments, controllers, metrics, reading): docstring of
``scripts/analyze_surrogate_env.py``, committed before the first run.

Everything as Befund 96 (cell ``ressep``, margin k = 5: ``5 m``, residual radius 0.5, separated
moisture cost, Lagrange per context cell 5 x 5, lambda_0 0.5 kW, eta 0.2 kW, update every 250 steps,
30 000 steps, checkpoints 1000 / 2500 / 5000 / 10 000 / 20 000 / 30 000, one torch thread; the
constants are imported from ``run_rl_bandit_residual.py``), but the transition is the environment of
the study:

* ``U0``  grey-box steady state (``SeparatedLagrangeBanditEnv``), i.e. Befund 96 itself;
* ``U1``  composite GRU of Befund 94, ``n_train`` 16, steady state = fixed point (``surrogate_env``);
* ``U2``  the same with ``n_train`` 64;
* ``U3``  ensemble of five GRU on the data of U2 (model seeds differ), moisture of the constraint
          ``mean + 2 std``.

Replicate ``r`` = RL seed ``r`` (0, 1, 2) with the GRU of master seed ``r + 1``.

Algorithms: ``SAC`` (stable-baselines3 defaults, as Befund 96) and ``TD3`` (Befund 96 used only SAC;
stable-baselines3 defaults plus Gaussian exploration noise sigma 0.1 on the residual action, the
default of the TD3 paper -- stable-baselines3 itself has no noise by default, with which TD3 collapsed
to a box corner in Befund 68 S0.8). For both: masked replay buffer and the output layer of the actor
zero-initialised (TD3: also the target actor), so the policy starts exactly at the fixed point.

SAC on U0 is run by the unchanged ``scripts/run_rl_bandit_residual.py --cell ressep --margin-factor 5``
(prefix ``surrogate_env_u0_sac_k5``), so that its records can be compared bit for bit with those of
Befund 96; this driver refuses that combination.

Evaluation at every checkpoint: deterministic policy on the 25 Befund-52 grid points (test set,
always named) against the TRUE limit 0.15 on the grey-box model; per point the action, the true
energy / moisture / bed temperature and, for U1-U3, the prediction of the training environment at
the same action (exploitation gap), plus the metrics of Befund 93/96.

    tools/run_here.sh scripts/run_surrogate_env_rl.py --env U2 --algo SAC --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import run_rl_bandit_residual as rr

from twinlink_ml.rl import evaluate_policy, flat_metrics, load_test_points
from twinlink_ml.rl.env import FIXED_NOZZLE_PRESSURE_BAR, _denormalise
from twinlink_ml.rl.evaluation import moisture_margin_at_oracle_binding_points
from twinlink_ml.rl.lagrange import LagrangeBanditEnv, LagrangeConfig, dual_ascent, relabel_rewards
from twinlink_ml.rl.residual import (
    MaskedReplayBuffer,
    ResidualActionWrapper,
    ResidualConfig,
    SeparatedLagrangeBanditEnv,
)
from twinlink_ml.simulation import PlaceholderConfigWarning
from twinlink_ml.surrogate_env import (
    EnsembleFixedPointModel,
    FixedInputs,
    GruFixedPointModel,
    LearnedSeparatedLagrangeBanditEnv,
    OperatingPointModel,
)
from twinlink_ml.utils.code_location import require_code_location
from twinlink_ml.utils.seeding import seed_everything
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = rr.REPO_ROOT
RUNS = rr.RUNS
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "surrogate_env_rl"
TRACE_DIR = Path("C:/twinlink_runs/surrogate-env/traces")
MODELS_RECORD = RUNS / "surrogate_env_models_v1.json"

# --- fixed before the runs ---------------------------------------------------------------------
MARGIN_FACTOR = 5.0  # Befund 96: smallest k with feasibility 1.0 in every seed
ENVIRONMENTS = ("U0", "U1", "U2", "U3")
ALGORITHMS = ("SAC", "TD3")
SEEDS = (0, 1, 2)
TD3_NOISE_SIGMA = 0.1
ENSEMBLE_K_SIGMA = 2.0
ENSEMBLE_MEMBERS = (0, 1, 2, 3, 4)
TRACE_EVERY = 10
PREFIX = "surrogate_env_rl"
U0_SAC_PREFIX = "surrogate_env_u0_sac_k5"
# -----------------------------------------------------------------------------------------------


def fixed_inputs() -> FixedInputs:
    env = rr.eval_env()
    return FixedInputs(
        nozzle_pressure_bar=FIXED_NOZZLE_PRESSURE_BAR,
        spray_rate_kg_per_h=env._base_config.actions.spray_rate_kg_per_h.nominal,
        suspension_solid_fraction=env._base_config.disturbances.suspension_solid_fraction.nominal,
    )


def _model_entry(record: dict[str, Any], seed: int, n_train: int, member: int) -> dict[str, Any]:
    for model in record["models"]:
        if (model["master_seed"], model["n_train"], model["member"]) == (seed, n_train, member):
            return model
    raise KeyError((seed, n_train, member))


def _load_checked(entry: dict[str, Any], fixed: FixedInputs) -> GruFixedPointModel:
    path = REPO_ROOT / entry["checkpoint"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["checkpoint_sha256"]:
        raise SystemExit(f"{path}: SHA-256 differs from the training record")
    return GruFixedPointModel.load(path, fixed)


def build_model(env_name: str, replicate_seed: int, fixed: FixedInputs) -> OperatingPointModel:
    """Learned operating-point model of U1-U3 for master seed ``replicate_seed``."""
    record = json.loads(MODELS_RECORD.read_text(encoding="utf-8"))
    if env_name == "U1":
        return _load_checked(_model_entry(record, replicate_seed, 16, 0), fixed)
    if env_name == "U2":
        return _load_checked(_model_entry(record, replicate_seed, 64, 0), fixed)
    if env_name == "U3":
        members = [
            _load_checked(_model_entry(record, replicate_seed, 64, j), fixed)
            for j in ENSEMBLE_MEMBERS
        ]
        return EnsembleFixedPointModel(members, ENSEMBLE_K_SIGMA)
    raise ValueError(env_name)


def output_paths(env_name: str, algo: str, seed: int, prefix: str = PREFIX) -> dict[str, Path]:
    stem = f"{prefix}_{env_name}_{algo}"
    return {
        "records": RUNS / f"{stem}_seed{seed}.jsonl",
        "lambda": RUNS / f"{stem}_lambda_seed{seed}.jsonl",
        "trace": TRACE_DIR / f"{stem}_trace_seed{seed}.jsonl",
    }


def study_output_paths() -> list[Path]:
    """Every file a run of the study writes into the repository (exempt from the dirty check)."""
    paths = [
        p
        for env_name in ENVIRONMENTS
        for algo in ALGORITHMS
        for seed in SEEDS
        for key, p in output_paths(env_name, algo, seed).items()
        if key != "trace"
    ]
    paths += rr.sweep_output_paths(U0_SAC_PREFIX)
    paths += [RUNS / "surrogate_env_rto_v1.json", RUNS / "surrogate_env_oracle_check_v1.json"]
    return paths


def zero_init_actor_output(model: Any) -> None:
    """Zero the output layer of the actor mean (SAC: ``mu``; TD3: last Linear of ``mu``)."""
    import torch

    mu = model.actor.mu
    layer = (
        mu
        if isinstance(mu, torch.nn.Linear)
        else [m for m in mu if isinstance(m, torch.nn.Linear)][-1]
    )
    layer.weight.data.zero_()
    layer.bias.data.zero_()
    if hasattr(model, "actor_target"):
        model.actor_target.load_state_dict(model.actor.state_dict())


def evaluate_points(
    policy: Any,
    truth_env: Any,
    learned: OperatingPointModel | None,
    points: Any,
) -> list[dict[str, Any]]:
    """Per test point: action, truth on the grey-box model, prediction of the training environment."""
    rows: list[dict[str, Any]] = []
    for point in points:
        observation = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity], dtype=np.float32
        )
        action = np.clip(policy(observation), -1.0, 1.0)
        t_in = _denormalise(
            float(action[0]), truth_env._inlet_bound.minimum, truth_env._inlet_bound.maximum
        )
        u_gas = _denormalise(
            float(action[1]), truth_env._gas_bound.minimum, truth_env._gas_bound.maximum
        )
        _, info = truth_env.evaluate_operating_point(
            inlet_temperature_celsius=t_in,
            gas_velocity_m_per_s=u_gas,
            ambient_temperature_celsius=point.ambient_temperature_celsius,
            ambient_relative_humidity=point.ambient_relative_humidity,
        )
        row: dict[str, Any] = {
            "ambient_temperature_celsius": point.ambient_temperature_celsius,
            "ambient_relative_humidity": point.ambient_relative_humidity,
            "inlet_temperature_celsius": t_in,
            "gas_velocity_m_per_s": u_gas,
            "true_solve_failed": bool(info.get("solve_failed")),
            "true_feasible": bool(info["feasible"]),
            "true_energy_kw": float(info.get("energy_target", np.nan)),
            "true_moisture_kg_per_kg": float(
                info.get("constraint_check", {}).get("moisture_kg_per_kg", np.nan)
            ),
            "true_bed_temperature_celsius": float(
                info.get("constraint_check", {}).get("bed_temperature_celsius", np.nan)
            ),
            "energy_oracle_limit015_kw": point.energy_oracle_kw,
            "energy_fixed_kw": point.energy_fixed_kw,
        }
        if learned is not None:
            prediction = learned.predict(
                np.array([[t_in, u_gas]]),
                np.array([[point.ambient_temperature_celsius, point.ambient_relative_humidity]]),
            )
            row["predicted"] = prediction.point(0)
            row["predicted_status_counts"] = prediction.status_counts
        rows.append(row)
    return rows


def make_callback(
    *,
    env_name: str,
    algo: str,
    lagrange: LagrangeConfig,
    logger: JsonlExperimentLogger,
    paths: dict[str, Path],
    env_info: dict[str, Any],
    seed: int,
    run_uid: str,
    run_start: float,
    anchor: np.ndarray,
    fixed_point: tuple[float, float],
    learned: OperatingPointModel | None,
    checkpoints: tuple[int, ...],
) -> Any:
    from stable_baselines3.common.callbacks import BaseCallback

    remaining = list(checkpoints)
    points = load_test_points(rr.REFERENCE_015)
    truth_env = rr.eval_env()

    class _Callback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.energy: list[float] = []
            self.rest: list[float] = []
            self.g_tilde: list[float] = []
            self.failed: list[bool] = []
            self.bin: list[int] = []
            self.n_masked: dict[str, int] = {}
            self.status_counts: dict[str, int] = {}
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
            assert not buffer.full
            rewards = relabel_rewards(
                energy_kw=np.array(self.energy),
                rest_penalty_kw=np.array(self.rest),
                g_tilde=np.nan_to_num(np.array(self.g_tilde)),
                solve_failed=np.array(self.failed, dtype=bool),
                bin_index=np.array(self.bin, dtype=int),
                multipliers=self.lag_env.multipliers,
                table=rr._EMPTY_TABLE,
            )
            stored = min(len(self.energy), buffer.pos)
            buffer.rewards[:stored, 0] = rewards[:stored].astype(buffer.rewards.dtype)

        def _on_step(self) -> bool:
            if self.pending_relabel:
                self._relabel()
                self.pending_relabel = False
            info = self.locals["infos"][0]
            masked = bool(info.get("masked", False))
            failed = bool(info.get("solve_failed"))
            prediction = info.get("learned_prediction")
            if prediction is not None:
                for key, value in prediction.get("status_counts", {}).items():
                    self.status_counts[key] = self.status_counts.get(key, 0) + int(value)
            if masked:
                reason = str(info["mask_reason"])
                self.n_masked[reason] = self.n_masked.get(reason, 0) + 1
            else:
                self.failed.append(failed)
                self.bin.append(int(info["lagrange_bin"]))
                self.energy.append(
                    float(info.get("lagrange_energy_kw", 0.0)) if not failed else 0.0
                )
                self.rest.append(
                    float(info.get("lagrange_rest_penalty_kw", 0.0)) if not failed else 0.0
                )
                self.g_tilde.append(float(info["lagrange_g_tilde"]))
            if self.num_timesteps % TRACE_EVERY == 0 or masked:
                values = self.model.logger.name_to_value
                self.trace.append(
                    {
                        "step": self.num_timesteps,
                        "bin": int(info["lagrange_bin"]),
                        "lambda_kw": float(info["lagrange_multiplier_kw"]),
                        "energy_kw": float(info.get("lagrange_energy_kw", 0.0)),
                        "g_tilde": float(info["lagrange_g_tilde"]),
                        "masked": masked,
                        "mask_reason": info.get("mask_reason"),
                        "moisture_kg_per_kg": float(
                            info.get("constraint_check", {}).get("moisture_kg_per_kg", np.nan)
                        ),
                        "action_applied": info.get("action_normalised_applied"),
                        "critic_loss": float(values.get("train/critic_loss", np.nan)),
                        "actor_loss": float(values.get("train/actor_loss", np.nan)),
                    }
                )
            if self.num_timesteps % lagrange.update_interval_steps == 0:
                self._dual_update()
            if remaining and self.num_timesteps >= remaining[0]:
                self.log_checkpoint(remaining.pop(0))
            return True

        def _dual_update(self) -> None:
            lam = self.lag_env.multipliers
            flat = lam.reshape(-1).copy()
            window = slice(self.window_start, len(self.g_tilde))
            bins = np.array(self.bin[window], dtype=int)
            gs = np.array(self.g_tilde[window])
            failed = np.array(self.failed[window], dtype=bool)
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
                    "n_masked_total": dict(self.n_masked),
                }
            )
            self.window_start = len(self.g_tilde)
            self._relabel()
            self.pending_relabel = True

        def log_checkpoint(self, checkpoint: int) -> None:
            import torch

            wall = time.perf_counter() - run_start
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_file = CHECKPOINT_DIR / f"{run_uid}_step{checkpoint:06d}.zip"
            self.model.save(checkpoint_file)
            policy = rr.make_policy(self.model, anchor)
            result = evaluate_policy(truth_env, policy, points)
            metrics = flat_metrics(result)
            metrics.update(rr.mean_energy_delta(policy, truth_env, points))
            margin_value, margin_n = moisture_margin_at_oracle_binding_points(
                truth_env, policy, points
            )
            metrics["moisture_margin_binding_median_kg_per_kg"] = margin_value
            metrics["distance_in_units_of_m"] = margin_value / rr.MARGIN_M
            lam = self.lag_env.multipliers
            metrics["lambda_mean_kw"] = float(lam.mean())
            metrics["lambda_max_kw"] = float(lam.max())
            metrics["n_masked_total"] = float(sum(self.n_masked.values()))
            per_point = evaluate_points(policy, truth_env, learned, points)
            print(
                f"[{env_name} {algo} seed {seed}] step {checkpoint}: "
                f"feas={metrics['feasibility_rate']:.2f} "
                f"pooled={metrics['pooled_closed_regret_fraction']:.4f} "
                f"dist/m={metrics['distance_in_units_of_m']:.2f} masked={self.n_masked} ({wall:.0f} s)",
                flush=True,
            )
            logger.log(
                RunRecord(
                    run_uid=f"{run_uid}_step{checkpoint:06d}",
                    group="surrogate_env_rl",
                    tags={"algorithm": algo, "checkpoint": str(checkpoint), "env": env_name},
                    params={
                        **env_info,
                        "seed": seed,
                        "env": env_name,
                        "algorithm": algo,
                        "replicate_master_seed": seed + 1,
                        "cell": "ressep",
                        "n_env_steps": checkpoint,
                        "total_timesteps_planned": max(checkpoints),
                        "wall_s_since_training_start": wall,
                        "hyperparameters": (
                            "stable-baselines3 default, no tuning"
                            if algo == "SAC"
                            else f"stable-baselines3 default + NormalActionNoise(0, {TD3_NOISE_SIGMA})"
                        ),
                        "lagrange": dataclasses.asdict(lagrange),
                        "residual_scale": rr.RESIDUAL_SCALE,
                        "fixed_point_inlet_gas": list(fixed_point),
                        "n_masked": dict(self.n_masked),
                        "fixed_point_status_counts_training": dict(self.status_counts),
                        "placeholders": "quality limit 0.15 kg/kg; gas velocity lower bound 0.80 m/s",
                        "test_set": "25 Befund-52 grid points (ambient T x relative humidity), "
                        "truth = grey-box model against the limit 0.15",
                        "checkpoint/path": str(checkpoint_file.relative_to(REPO_ROOT)),
                        "multipliers_kw": lam.reshape(-1).tolist(),
                        "quality_limit_kg_per_kg": rr.LIMIT_KG_PER_KG,
                        "margin_factor": MARGIN_FACTOR,
                        "margin_kg_per_kg": MARGIN_FACTOR * rr.MARGIN_M,
                        "torch_num_threads": torch.get_num_threads(),
                        "eval/points": per_point,
                    },
                    metrics=metrics,
                )
            )

        def _on_training_end(self) -> None:
            paths["trace"].parent.mkdir(parents=True, exist_ok=True)
            with paths["trace"].open("w", encoding="utf-8") as handle:
                for row in self.trace:
                    handle.write(json.dumps(row) + "\n")
            with paths["lambda"].open("w", encoding="utf-8") as handle:
                for row in self.lambda_history:
                    handle.write(json.dumps(row) + "\n")

    return _Callback()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--algo", choices=ALGORITHMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--prefix", type=str, default=PREFIX)
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help="mechanics test only: one checkpoint at this step; prefix must start with 'smoke'",
    )
    arguments = parser.parse_args()
    if arguments.env == "U0" and arguments.algo == "SAC":
        raise SystemExit(
            "SAC on U0 runs through scripts/run_rl_bandit_residual.py --cell ressep "
            f"--margin-factor 5 --prefix {U0_SAC_PREFIX} (bit-for-bit check against Befund 96)"
        )
    checkpoints = rr.CHECKPOINTS
    if arguments.smoke_steps is not None:
        if not arguments.prefix.startswith("smoke"):
            raise SystemExit("--smoke-steps needs a prefix that starts with 'smoke'")
        checkpoints = (arguments.smoke_steps,)
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    require_code_location(REPO_ROOT)

    import torch
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise

    torch.set_num_threads(1)
    paths = output_paths(arguments.env, arguments.algo, arguments.seed, arguments.prefix)
    for path in paths.values():
        if path.exists():
            raise SystemExit(f"{path} exists already; use another --prefix")
    env_info = environment_fingerprint(ignore_paths=study_output_paths())
    if env_info["git/dirty"] and arguments.smoke_steps is None:
        raise SystemExit(
            "working tree is dirty -- commit first (CLAUDE.md: clean tree before runs)"
        )
    seed_everything(arguments.seed)
    run_uid = f"{arguments.env}_{arguments.algo}_s{arguments.seed}_{uuid.uuid4().hex[:8]}"
    logger = JsonlExperimentLogger(paths["records"])

    lagrange = LagrangeConfig(
        margin_kg_per_kg=MARGIN_FACTOR * rr.MARGIN_M,
        initial_multiplier_kw=rr.INITIAL_MULTIPLIER_KW,
        step_size_kw=rr.STEP_SIZE_KW,
        update_interval_steps=rr.UPDATE_INTERVAL_STEPS,
        context_bins=rr.CONTEXT_BINS,
    )
    env_config = rr.BanditEnvConfig(product_moisture_limit_kg_per_kg=rr.LIMIT_KG_PER_KG)
    fixed = fixed_inputs()
    learned: OperatingPointModel | None = None
    base_env: LagrangeBanditEnv
    if arguments.env == "U0":
        base_env = SeparatedLagrangeBanditEnv(lagrange, env_config)
    else:
        learned = build_model(arguments.env, arguments.seed + 1, fixed)
        base_env = LearnedSeparatedLagrangeBanditEnv(learned, fixed, lagrange, env_config)
    fixed_point = rr.fixed_point_from_reference()
    train_env: Any = ResidualActionWrapper(
        base_env, ResidualConfig(fixed_point[0], fixed_point[1], rr.RESIDUAL_SCALE)
    )
    anchor = train_env.anchor_normalised
    train_env = Monitor(train_env)
    if arguments.algo == "SAC":
        model: Any = SAC(
            "MlpPolicy",
            train_env,
            seed=arguments.seed,
            verbose=0,
            replay_buffer_class=MaskedReplayBuffer,
        )
    else:
        model = TD3(
            "MlpPolicy",
            train_env,
            seed=arguments.seed,
            verbose=0,
            replay_buffer_class=MaskedReplayBuffer,
            action_noise=NormalActionNoise(np.zeros(2), TD3_NOISE_SIGMA * np.ones(2)),
        )
    zero_init_actor_output(model)
    run_start = time.perf_counter()
    callback = make_callback(
        env_name=arguments.env,
        algo=arguments.algo,
        lagrange=lagrange,
        logger=logger,
        paths=paths,
        env_info=env_info,
        seed=arguments.seed,
        run_uid=run_uid,
        run_start=run_start,
        anchor=anchor,
        fixed_point=fixed_point,
        learned=learned,
        checkpoints=checkpoints,
    )
    callback.init_callback(model)
    callback.log_checkpoint(0)
    model.learn(total_timesteps=max(checkpoints), callback=callback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
