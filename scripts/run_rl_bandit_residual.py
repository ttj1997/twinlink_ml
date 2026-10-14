r"""Closure of the one-step bandit: residual action x separated moisture cost (Befund 68, 2 x 2 ablation).

PRE-REGISTRATION. Everything below is fixed before the first run of this script; the gate is also
fixed, in ``scripts/analyze_bres_gate.py``, committed before the runs. Budget: at most one day of
compute, no new line of work, one 2 x 2 table and one sentence of classification at the end.

Context. Up to Befund 68 / SH no agent got past the fixed operating point on the bandit of one line
(Befund 62/68, SH). Plausible causes: (i) the 50 kW step that ``check_constraints`` adds when the
solver marks a solution infeasible (for moisture > critical moisture: S0.4), so the Lagrange
multiplier never got a signal; (ii) a critic that smears this cliff (S0.9); (iii) a policy that
starts far from the boundary. This study tests (i) and (iii) with two measures and their
combination; (ii) is not tested separately.

Measures (module ``twinlink_ml.rl.residual``)
  R  residual action: executed normalised action = clip(a_fixed + s * delta, -1, 1); the last layer
     of the actor mean is zero-initialised, so the deterministic policy is EXACTLY the fixed point
     at step 0 (checked and logged: record ``checkpoint`` "0", closed fraction 0 by construction).
     a_fixed = fixed operating point of the quality limit 0.15 (Befund 68 S0.5: 92.989 degC,
     0.8005 m/s), read from ``s05_limit015_oracle_and_fixed.json``. s = 0.5 in normalised units
     (+-15 K inlet temperature, +-0.425 m/s gas velocity). s was chosen from the S0.5 reference only
     (before any run): the oracle inlet temperature lies 0-33 K below the fixed point, 24 of 25 points
     within 15 K; the one point beyond (20 degC, 0.25) is not reachable and is counted as such.
  S  separated moisture cost: a solver failure (exception) and a solved point with any
     infeasibility reason OTHER than "moisture above the critical moisture" are MASKED (transition
     not stored in the replay buffer, counted per run); a solved point whose only reason is the
     moisture above the critical moisture gets no 50 kW step, only the smooth cost lambda_b * g_tilde
     (g_tilde linear in the moisture, divided by the quality limit). Lagrange multiplier per context
     cell as in S0.3 (5 x 5 cells, lambda_0 0.5, eta 0.2 kW, update every 250 steps, tightened limit
     0.15 - m, m = 0.005961819615175404 kg/kg). The failure bound of S0.3 is not used in S cells.
     This is my reading of "smooth, normalised cost": no additional penalty term was added.

Design. 2 x 2: {residual yes/no} x {separated yes/no}; cells ``base`` (no, no; = arm C of S0.6 run
by this script), ``sep`` (no, yes), ``res`` (yes, no), ``ressep`` (yes, yes). SAC, stable-baselines3
default hyperparameters, seeds 0-4, 30 000 steps with checkpoints at 1000, 2500, 5000, 10000, 20000,
30000 (the budgets "1e4" and "3e4" of the instruction are the checkpoints 10 000 and 30 000 of the
same run: SAC has no schedule that depends on the planned length, so the first 10 000 steps of a
30 000-step run are those of a 10 000-step run). Evaluation: always the plain environment against the
TRUE limit, deterministic policy, the 25 Befund 52 grid points with the oracle and the fixed
operating point of the limit 0.15 (test set, always named).

PLACEHOLDERS. The quality limit 0.15 kg/kg (main case) and the lower bound 0.80 m/s of the gas
velocity (``ActionBound``, unchanged) are placeholders without source; the critical moisture 0.20
of the kinetics, the penalty weight 50 kW and the margin m are likewise placeholder / earlier
settings, not results. Every table and artifact of this study inherits that label.

Gate (details and decision rule: ``analyze_bres_gate.py``): feasibility 1.0; median closed fraction
> 0 (stage 1) or > 0.5 (stage 2); distance to the limit in units of m.

    tools/run_here.sh scripts/run_rl_bandit_residual.py --seed 0 --cell ressep
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
from twinlink_ml.rl.env import _denormalise
from twinlink_ml.rl.evaluation import moisture_margin_at_oracle_binding_points
from twinlink_ml.rl.lagrange import (
    LagrangeBanditEnv,
    LagrangeConfig,
    ReferenceTable,
    dual_ascent,
    relabel_rewards,
)
from twinlink_ml.rl.residual import (
    MaskedReplayBuffer,
    ResidualActionWrapper,
    ResidualConfig,
    SeparatedLagrangeBanditEnv,
    residual_to_action,
    zero_init_actor_mean,
)
from twinlink_ml.simulation import PlaceholderConfigWarning
from twinlink_ml.utils.code_location import require_code_location
from twinlink_ml.utils.seeding import seed_everything
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "artifacts" / "runs"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "rl_bandit_residual"
REFERENCE_015 = RUNS / "s05_limit015_oracle_and_fixed.json"

# --- fixed before the runs ---------------------------------------------------------------------
LIMIT_KG_PER_KG = 0.15  # PLACEHOLDER (main case)
MARGIN_M = 0.005961819615175404
INITIAL_MULTIPLIER_KW = 0.5
STEP_SIZE_KW = 0.2
UPDATE_INTERVAL_STEPS = 250
CONTEXT_BINS = (5, 5)
RESIDUAL_SCALE = 0.5
SEEDS = (0, 1, 2, 3, 4)
CHECKPOINTS = (1_000, 2_500, 5_000, 10_000, 20_000, 30_000)
CELLS: dict[str, dict[str, bool]] = {
    "base": {"residual": False, "separated": False},
    "sep": {"residual": False, "separated": True},
    "res": {"residual": True, "separated": False},
    "ressep": {"residual": True, "separated": True},
}
PREFIX = "rl_bandit_residual"
# -----------------------------------------------------------------------------------------------


def output_paths(cell: str, seed: int, prefix: str = PREFIX) -> dict[str, Path]:
    stem = f"{prefix}_{cell}"
    return {
        "records": RUNS / f"{stem}_seed{seed}.jsonl",
        "trace": RUNS / f"{stem}_trace_seed{seed}.jsonl",
        "lambda": RUNS / f"{stem}_lambda_seed{seed}.jsonl",
    }


def sweep_output_paths(prefix: str = PREFIX) -> list[Path]:
    """Every file any run of the sweep writes (exempt from the dirty check of the siblings)."""
    return [
        p for cell in CELLS for seed in SEEDS for p in output_paths(cell, seed, prefix).values()
    ]


def fixed_point_from_reference() -> tuple[float, float]:
    data = json.loads(REFERENCE_015.read_text(encoding="utf-8"))
    fixed = data["fixed_point"]
    assert data["quality_limit_kg_per_kg"] == LIMIT_KG_PER_KG
    return float(fixed["inlet_temperature_celsius"]), float(fixed["gas_velocity_m_per_s"])


def eval_env() -> OperatingPointBanditEnv:
    return OperatingPointBanditEnv(
        BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT_KG_PER_KG)
    )


def make_policy(model: Any, anchor: np.ndarray | None) -> Any:
    """Deterministic policy as a map observation -> normalised ABSOLUTE action."""

    def policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float64)
        if anchor is None:
            return action
        return residual_to_action(action, anchor, RESIDUAL_SCALE)

    return policy


def mean_energy_delta(policy: Any, env: OperatingPointBanditEnv, points: Any) -> dict[str, float]:
    energies: list[float] = []
    for point in points:
        observation = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity], dtype=np.float32
        )
        action = np.clip(policy(observation), -1.0, 1.0)
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=_denormalise(
                float(action[0]), env._inlet_bound.minimum, env._inlet_bound.maximum
            ),
            gas_velocity_m_per_s=_denormalise(
                float(action[1]), env._gas_bound.minimum, env._gas_bound.maximum
            ),
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
    cell: str,
    lagrange: LagrangeConfig,
    logger: JsonlExperimentLogger,
    paths: dict[str, Path],
    env_info: dict[str, Any],
    seed: int,
    run_uid: str,
    run_start: float,
    anchor: np.ndarray | None,
    table: ReferenceTable | None,
    fixed_point: tuple[float, float],
    checkpoints: tuple[int, ...],
) -> Any:
    from stable_baselines3.common.callbacks import BaseCallback

    remaining = list(checkpoints)
    points = load_test_points(REFERENCE_015)

    class _Callback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            # arrays of STORED transitions only (masked steps are not in the replay buffer)
            self.energy: list[float] = []
            self.rest: list[float] = []
            self.g_tilde: list[float] = []
            self.failed: list[bool] = []
            self.bin: list[int] = []
            self.n_masked: dict[str, int] = {}
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
            n = len(self.energy)
            rewards = relabel_rewards(
                energy_kw=np.array(self.energy),
                rest_penalty_kw=np.array(self.rest),
                g_tilde=np.nan_to_num(np.array(self.g_tilde)),
                solve_failed=np.array(self.failed, dtype=bool),
                bin_index=np.array(self.bin, dtype=int),
                multipliers=self.lag_env.multipliers,
                table=table if table is not None else _EMPTY_TABLE,
            )
            stored = min(n, buffer.pos)
            buffer.rewards[:stored, 0] = rewards[:stored].astype(buffer.rewards.dtype)

        def _on_step(self) -> bool:
            if self.pending_relabel:
                self._relabel()
                self.pending_relabel = False
            info = self.locals["infos"][0]
            masked = bool(info.get("masked", False))
            failed = bool(info.get("solve_failed"))
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
            values = self.model.logger.name_to_value
            self.trace.append(
                {
                    "step": self.num_timesteps,
                    "bin": int(info["lagrange_bin"]),
                    "lambda_kw": float(info["lagrange_multiplier_kw"]),
                    "energy_kw": float(info.get("lagrange_energy_kw", 0.0)),
                    "g_tilde": float(info["lagrange_g_tilde"]),
                    "solve_failed": failed,
                    "masked": masked,
                    "mask_reason": info.get("mask_reason"),
                    "feasible": bool(info["feasible"]),
                    "moisture_kg_per_kg": float(
                        info.get("constraint_check", {}).get("moisture_kg_per_kg", float("nan"))
                    ),
                    "reward": float(self.locals["rewards"][0]),
                    "delta": info.get("residual_delta"),
                    "action_applied": info.get("action_normalised_applied"),
                    "critic_loss": float(values.get("train/critic_loss", float("nan"))),
                    "actor_loss": float(values.get("train/actor_loss", float("nan"))),
                    "ent_coef": float(values.get("train/ent_coef", float("nan"))),
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
                    "n_failed_in_window": int(failed.sum()),
                    "n_masked_total": dict(self.n_masked),
                }
            )
            self.window_start = len(self.g_tilde)
            self._relabel()
            self.pending_relabel = True

        def log_checkpoint(self, checkpoint: int) -> None:
            wall = time.perf_counter() - run_start
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_file = CHECKPOINT_DIR / f"{run_uid}_step{checkpoint:06d}.zip"
            self.model.save(checkpoint_file)
            policy = make_policy(self.model, anchor)
            env = eval_env()
            result = evaluate_policy(env, policy, points)
            metrics = flat_metrics(result)
            metrics.update(mean_energy_delta(policy, env, points))
            margin_value, margin_n = moisture_margin_at_oracle_binding_points(env, policy, points)
            metrics["moisture_margin_binding_median_kg_per_kg"] = margin_value
            metrics["moisture_margin_binding_n"] = float(margin_n)
            metrics["distance_in_units_of_m"] = margin_value / MARGIN_M
            lam = self.lag_env.multipliers
            metrics["lambda_mean_kw"] = float(lam.mean())
            metrics["lambda_max_kw"] = float(lam.max())
            metrics["n_masked_total"] = float(sum(self.n_masked.values()))
            print(
                f"[{cell} seed {seed}] step {checkpoint}: feas={metrics['feasibility_rate']:.2f} "
                f"pooled={metrics['pooled_closed_regret_fraction']:.4f} "
                f"dist/m={metrics['distance_in_units_of_m']:.2f} "
                f"lambda mean={metrics['lambda_mean_kw']:.3f} masked={self.n_masked} ({wall:.0f} s)",
                flush=True,
            )
            logger.log(
                RunRecord(
                    run_uid=f"{run_uid}_step{checkpoint:06d}",
                    group="rl_bandit_residual",
                    tags={"algorithm": "SAC", "checkpoint": str(checkpoint), "cell": cell},
                    params={
                        **env_info,
                        "seed": seed,
                        "cell": cell,
                        **CELLS[cell],
                        "n_env_steps": checkpoint,
                        "total_timesteps_planned": max(checkpoints),
                        "wall_s_since_training_start": wall,
                        "hyperparameters": "stable-baselines3 default, no tuning",
                        "lagrange": dataclasses.asdict(lagrange),
                        "residual_scale": RESIDUAL_SCALE if anchor is not None else None,
                        "fixed_point_inlet_gas": list(fixed_point),
                        "n_masked": dict(self.n_masked),
                        "placeholders": "quality limit 0.15 kg/kg; gas velocity lower bound 0.80 m/s",
                        "test_set": "25 Befund-52 grid points (ambient T x relative humidity), "
                        "oracle and fixed point of the limit 0.15",
                        "checkpoint/path": str(checkpoint_file.relative_to(REPO_ROOT)),
                        "multipliers_kw": lam.reshape(-1).tolist(),
                        "quality_limit_kg_per_kg": LIMIT_KG_PER_KG,
                    },
                    metrics=metrics,
                )
            )

        def _on_training_end(self) -> None:
            with paths["trace"].open("w", encoding="utf-8") as handle:
                for row in self.trace:
                    handle.write(json.dumps(row) + "\n")
            with paths["lambda"].open("w", encoding="utf-8") as handle:
                for row in self.lambda_history:
                    handle.write(json.dumps(row) + "\n")

    return _Callback()


_EMPTY_TABLE = ReferenceTable(np.zeros(0), np.zeros(0), np.zeros(0))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--prefix", type=str, default=PREFIX)
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help="mechanics test only: a single checkpoint at this step; the prefix must start "
        "with 'smoke'. Never used for a result.",
    )
    arguments = parser.parse_args()
    checkpoints = CHECKPOINTS
    if arguments.smoke_steps is not None:
        if not arguments.prefix.startswith("smoke"):
            raise SystemExit("--smoke-steps needs a prefix that starts with 'smoke'")
        checkpoints = (arguments.smoke_steps,)
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    require_code_location(REPO_ROOT)

    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)  # up to 20 runs share the machine (one solver call per step)
    spec = CELLS[arguments.cell]
    paths = output_paths(arguments.cell, arguments.seed, arguments.prefix)
    for path in paths.values():
        if path.exists():
            raise SystemExit(f"{path} exists already; use another --prefix")
    env_info = environment_fingerprint(ignore_paths=sweep_output_paths(arguments.prefix))
    if env_info["git/dirty"]:
        raise SystemExit(
            "working tree is dirty -- commit first (CLAUDE.md: clean tree before runs)"
        )
    seed_everything(arguments.seed)
    run_uid = uuid.uuid4().hex[:12]
    logger = JsonlExperimentLogger(paths["records"])

    lagrange = LagrangeConfig(
        margin_kg_per_kg=MARGIN_M,
        initial_multiplier_kw=INITIAL_MULTIPLIER_KW,
        step_size_kw=STEP_SIZE_KW,
        update_interval_steps=UPDATE_INTERVAL_STEPS,
        context_bins=CONTEXT_BINS,
    )
    env_config = BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT_KG_PER_KG)
    base_env: LagrangeBanditEnv = (
        SeparatedLagrangeBanditEnv(lagrange, env_config)
        if spec["separated"]
        else LagrangeBanditEnv(lagrange, env_config)
    )
    table = None if spec["separated"] else base_env.reference_table
    fixed_point = fixed_point_from_reference()
    anchor: np.ndarray | None = None
    train_env: Any = base_env
    if spec["residual"]:
        train_env = ResidualActionWrapper(
            base_env, ResidualConfig(fixed_point[0], fixed_point[1], RESIDUAL_SCALE)
        )
        anchor = train_env.anchor_normalised
    train_env = Monitor(train_env)
    model = SAC(
        "MlpPolicy",
        train_env,
        seed=arguments.seed,
        verbose=0,
        replay_buffer_class=MaskedReplayBuffer,
    )
    if spec["residual"]:
        zero_init_actor_mean(model)
    run_start = time.perf_counter()
    callback = make_callback(
        cell=arguments.cell,
        lagrange=lagrange,
        logger=logger,
        paths=paths,
        env_info=env_info,
        seed=arguments.seed,
        run_uid=run_uid,
        run_start=run_start,
        anchor=anchor,
        table=table,
        fixed_point=fixed_point,
        checkpoints=checkpoints,
    )
    callback.init_callback(model)
    callback.log_checkpoint(0)  # step 0: residual cells = the fixed point exactly
    model.learn(total_timesteps=max(checkpoints), callback=callback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
