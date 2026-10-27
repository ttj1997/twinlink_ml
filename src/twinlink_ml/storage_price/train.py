"""Training of the SAC / TD3 agents (stable-baselines3) with the learning curve of the study.

Selection rule (fixed before the run): the policy after ``total_steps`` steps is evaluated; the
intermediate checkpoints only give the learning curve. Every ``eval_every_steps`` steps the
deterministic policy runs the evaluation protocol on the training months (in-sample) and on the
test months; both totals are recorded, neither is used for any choice.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .env import PolicyController, StoragePriceEnv
from .evaluate import run_months
from .study import Case

__all__ = ["evaluate_policy", "make_agent", "train_agent"]


def make_agent(case: Case, algorithm: str, seed: int, env: StoragePriceEnv) -> Any:
    """SAC or TD3 with the settings of the config (other hyperparameters: library defaults)."""
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise

    rl = case.config.rl
    common: dict[str, Any] = {
        "learning_rate": rl.learning_rate,
        "buffer_size": rl.buffer_size,
        "batch_size": rl.batch_size,
        "gamma": rl.gamma,
        "learning_starts": rl.learning_starts,
        "policy_kwargs": {"net_arch": list(rl.net_arch)},
        "seed": seed,
        "device": "cpu",
        "verbose": 0,
    }
    if algorithm == "SAC":
        return SAC("MlpPolicy", env, **common)
    if algorithm == "TD3":
        noise = NormalActionNoise(np.zeros(2), np.full(2, rl.td3_action_noise_std))
        return TD3("MlpPolicy", env, action_noise=noise, **common)
    raise ValueError(f"unknown algorithm {algorithm!r}")


def evaluate_policy(case: Case, policy: Any, name: str, months: tuple[int, ...]) -> list[dict]:
    """Month runs of a policy (evaluation protocol)."""
    ctrl = PolicyController(case.plant, case.market, case.prices.clock, policy, name)
    tol = case.config.controllers.violation_tolerance_relative
    return [
        r.as_dict()
        for r in run_months(ctrl, case.plant, case.market, case.prices.clock, months, tolerance=tol)
    ]


def train_agent(
    case: Case,
    algorithm: str,
    seed: int,
    *,
    total_steps: int,
    checkpoint_dir: Path,
    log: Any = print,
) -> dict[str, Any]:
    """Train one agent; learning curve, checkpoints and the final evaluation."""
    import torch

    torch.set_num_threads(1)
    cfg = case.config
    rl = cfg.rl
    env = StoragePriceEnv(
        case.plant,
        case.market,
        case.prices.clock,
        cfg.training_months,
        episode_h=rl.episode_h,
        reward_scale=rl.reward_scale_eur,
    )
    env.reset(seed=seed)
    agent = make_agent(case, algorithm, seed, env)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    label = f"{algorithm}_s{seed}"
    curve: list[dict[str, Any]] = []
    done = 0
    t_train = 0.0
    t_eval = 0.0
    while done < total_steps:
        chunk = min(rl.eval_every_steps, total_steps - done)
        t0 = time.perf_counter()
        agent.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        t_train += time.perf_counter() - t0
        done += chunk
        t0 = time.perf_counter()
        train_res = evaluate_policy(case, agent, label, cfg.training_months)
        test_res = evaluate_policy(case, agent, label, cfg.test_months)
        t_eval += time.perf_counter() - t0
        path = checkpoint_dir / f"{label}_step{done}.zip"
        agent.save(path)
        curve.append(
            {
                "steps": done,
                "train_cost_eur": float(sum(r["cost_eur"] for r in train_res)),
                "test_cost_eur": float(sum(r["cost_eur"] for r in test_res)),
                "test_cost_by_month_eur": {str(r["month"]): r["cost_eur"] for r in test_res},
                "content_cuts_test": int(sum(r["violations_content"] for r in test_res)),
                "checkpoint": str(path.name),
            }
        )
        log(
            f"{label} {done}/{total_steps}: train {curve[-1]['train_cost_eur']:.0f} "
            f"test {curve[-1]['test_cost_eur']:.0f} EUR ({t_train / 60:.1f} min train)"
        )
    final_path = checkpoint_dir / f"{label}_final.zip"
    agent.save(final_path)
    t0 = time.perf_counter()
    final_test = evaluate_policy(case, agent, label, cfg.test_months)
    final_train = evaluate_policy(case, agent, label, cfg.training_months)
    t_eval += time.perf_counter() - t0
    return {
        "algorithm": algorithm,
        "seed": seed,
        "total_steps": total_steps,
        "learning_curve": curve,
        "test": final_test,
        "train": final_train,
        "checkpoint_final": str(final_path.name),
        "train_seconds": t_train,
        "eval_seconds": t_eval,
        "training_steps_with_content_cut": int(env.cuts),
    }
