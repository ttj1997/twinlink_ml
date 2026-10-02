r"""S0.8 follow-up: width of the SAC policies and actions of the TD3 policies.

Report only, from the step-10 000 checkpoints of the arms C (SAC default), G (SAC with target
entropy -6), F (TD3, no action noise) and H (TD3, noise 0.1), seeds 0-4, at the 25 test contexts of
the limit-0.15 reference. No solver, no training.

SAC arms (C, G): per seed the median over the contexts of the pre-tanh ``log_std`` per action
dimension and of the standard deviation of the squashed, denormalised action in physical units
(10 000 draws per context, fixed seed): inlet temperature [K], gas velocity [m/s]. This is the
"actual width" check of the confirmed S0.8 plan.

TD3 arms (F, H): the deterministic action in normalised units at the 25 contexts: per seed the mean
and the standard deviation across the contexts per dimension, the minimum and maximum, and the
share of contexts with ``|a| >= 0.99`` per dimension (an action pinned to the box edge).

    python scripts/probe_s08_width_and_td3_actions.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "artifacts" / "runs"
SEEDS = (0, 1, 2, 3, 4)
N_DRAWS = 10_000
SATURATION = 0.99
ARMS = {
    "C_SAC_default": ("rl_bandit_lagrange_s06_l015_lam05", "SAC"),
    "G_SAC_low_entropy": ("rl_bandit_lagrange_s08_sac_lowent", "SAC"),
    "F_TD3_default": ("rl_bandit_lagrange_s08_td3", "TD3"),
    "H_TD3_noise": ("rl_bandit_lagrange_s08_td3_noise", "TD3"),
}


def _checkpoint(prefix: str, seed: int) -> Path:
    path = RUNS / f"{prefix}_seed{seed}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["params"]["n_env_steps"] == 10_000:
            return REPO_ROOT / row["params"]["checkpoint/path"]
    raise SystemExit(f"no step-10000 record in {path}")


def main() -> int:
    require_clean_tree()
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import SAC, TD3

    from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv, load_test_points

    env = OperatingPointBanditEnv(BanditEnvConfig(product_moisture_limit_kg_per_kg=0.15))
    inlet, gas = env._inlet_bound, env._gas_bound
    points = load_test_points(RUNS / "s05_limit015_oracle_and_fixed.json")
    out: dict[str, Any] = {}
    for arm, (prefix, algorithm) in ARMS.items():
        cls = {"SAC": SAC, "TD3": TD3}[algorithm]
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            model = cls.load(_checkpoint(prefix, seed), device="cpu")
            log_stds, sigma_t, sigma_u, actions = [], [], [], []
            rng = np.random.default_rng(seed)
            for p in points:
                obs = np.array(
                    [p.ambient_temperature_celsius, p.ambient_relative_humidity], dtype=np.float32
                )
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                if algorithm == "SAC":
                    with torch.no_grad():
                        mean, log_std, _ = model.actor.get_action_dist_params(obs_tensor)
                    mean_np = mean.cpu().numpy().reshape(-1).astype(np.float64)
                    log_std_np = log_std.cpu().numpy().reshape(-1).astype(np.float64)
                    draws = mean_np + np.exp(log_std_np) * rng.standard_normal((N_DRAWS, 2))
                    a = np.tanh(draws)
                    log_stds.append(log_std_np)
                    sigma_t.append(float((0.5 * (a[:, 0] + 1.0) * inlet.span).std()))
                    sigma_u.append(float((0.5 * (a[:, 1] + 1.0) * gas.span).std()))
                    actions.append(np.tanh(mean_np))
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    actions.append(np.asarray(action, dtype=np.float64))
            acts = np.array(actions)
            entry: dict[str, Any] = {
                "action_mean": acts.mean(axis=0).tolist(),
                "action_std_across_contexts": acts.std(axis=0).tolist(),
                "action_min": acts.min(axis=0).tolist(),
                "action_max": acts.max(axis=0).tolist(),
                "share_saturated_per_dim": (np.abs(acts) >= SATURATION).mean(axis=0).tolist(),
            }
            if algorithm == "SAC":
                ls = np.array(log_stds)
                entry.update(
                    {
                        "log_std_median": np.median(ls, axis=0).tolist(),
                        "sigma_t_k_median": float(np.median(sigma_t)),
                        "sigma_u_m_per_s_median": float(np.median(sigma_u)),
                    }
                )
            per_seed[str(seed)] = entry
        out[arm] = per_seed
    write_artifact(
        "s08_width_and_td3_actions",
        {"_provenance": provenance(), "saturation_threshold": SATURATION, "arms": out},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
