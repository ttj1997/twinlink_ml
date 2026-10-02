r"""S0.7: test whether the spread of the policy sets the distance to the validity boundary.

Hypothesis (to be tested, not assumed): the distance of the agent to the solver validity boundary
0.20 kg/kg arises from the spread of the stochastic SAC policy and the height of the +1.0 (50 kW)
step behind the boundary; for a spreading policy the best mean lies several spreads before the
step, and the evaluated point is the mean. If so, the ratio  distance / moisture spread  should be
about the same in all arms.

No training: existing step-10 000 checkpoints of the arms A-E of S0.6 and of the penalty arm of
S0.3 (seeds 0-4), at the 25 test contexts of the arm's own reference (limit 0.20: Befund 52,
limit 0.15: ``s05_limit015_oracle_and_fixed``). Per context:

a) spread of the policy per action dimension: the pre-tanh ``log_std`` of the SAC actor
   (``actor.get_action_dist_params``), converted to physical units as the standard deviation of
   the squashed, denormalised action over 10 000 draws (fixed seed);
b) sensitivity of the steady-state product moisture to each action at the evaluated (deterministic,
   ``tanh(mean)``) action: central finite differences with the steady-state solver (steps 0.5 K
   and 0.01 m/s), combined with a) as ``sigma_X = sqrt((dX/dT sigma_T)^2 + (dX/du sigma_u)^2)``;
   cross-check ``sigma_X_direct``: standard deviation of the moisture over 32 policy draws solved
   directly;
c) ratio = (0.20 - X) / sigma_X with X the moisture at the evaluated action.

Aggregates per (arm, seed): median over the contexts where the oracle of the arm's reference binds
the limit (17 at limit 0.20, 22 at 0.15; same set as the gate G3), and over all 25 contexts.

Decision rule, fixed before the run, on the 30 (arm, seed) pairs of binding-context medians:
  SUPPORTED   iff Spearman(distance, sigma_X) >= 0.5 AND CV(ratio) < CV(distance)
  WEAKENED    iff Spearman(distance, sigma_X) < 0.2 OR  CV(ratio) >= CV(distance)
  INCONCLUSIVE otherwise.
CV = standard deviation / mean over the 30 pairs. Also reported: the median ratio per arm, its
relative range across arms ((max - min) / median of the arm medians), and whether the median ratio
is at least 2 ("several spreads").

    python scripts/probe_s07_policy_spread.py
"""

from __future__ import annotations

import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "artifacts" / "runs"
SEEDS = (0, 1, 2, 3, 4)
EVALUATION_STEP = 10_000
VALIDITY_BOUNDARY = 0.20
N_DRAWS = 10_000
N_DIRECT = 32
STEP_T_K = 0.5
STEP_U_M_PER_S = 0.01
WORKERS = 12

# --- decision rule, fixed before the run -------------------------------------------------------
SUPPORTED_MIN_SPEARMAN = 0.5
WEAKENED_MAX_SPEARMAN = 0.2
SEVERAL_SPREADS = 2.0
# -------------------------------------------------------------------------------------------------

ARMS: dict[str, dict[str, Any]] = {
    "A_L020_lam05": {"prefix": "rl_bandit_lagrange_s03", "limit": None},
    "B_L020_lam0": {"prefix": "rl_bandit_lagrange_s06_l020_lam0", "limit": None},
    "C_L015_lam05": {"prefix": "rl_bandit_lagrange_s06_l015_lam05", "limit": 0.15},
    "D_L015_lam0": {"prefix": "rl_bandit_lagrange_s06_l015_lam0", "limit": 0.15},
    "E_penalty_L015": {
        "prefix": "rl_bandit_sac_learning_curve_s06_penalty_l015",
        "limit": 0.15,
    },
    "P_penalty_L020_s03": {"prefix": "rl_bandit_sac_learning_curve_s03_penalty", "limit": None},
}
REFERENCE_015 = RUNS / "s05_limit015_oracle_and_fixed.json"

_CACHE: dict[str, Any] = {}


def _final_checkpoint(prefix: str, seed: int) -> Path:
    path = RUNS / f"{prefix}_seed{seed}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["params"]["n_env_steps"] == EVALUATION_STEP:
            return REPO_ROOT / row["params"]["checkpoint/path"]
    raise SystemExit(f"no step-{EVALUATION_STEP} record in {path}")


def _env_and_points(limit: float | None) -> tuple[Any, Any]:
    key = f"env_{limit}"
    if key not in _CACHE:
        from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv, load_test_points

        warnings.filterwarnings("ignore")
        env = OperatingPointBanditEnv(BanditEnvConfig(product_moisture_limit_kg_per_kg=limit))
        points = load_test_points() if limit is None else load_test_points(REFERENCE_015)
        _CACHE[key] = (env, points)
    return _CACHE[key]


def _moisture(env: Any, t_amb: float, phi: float, t_in: float, u_gas: float) -> float:
    _, info = env.evaluate_operating_point(
        inlet_temperature_celsius=float(t_in),
        gas_velocity_m_per_s=float(u_gas),
        ambient_temperature_celsius=t_amb,
        ambient_relative_humidity=phi,
    )
    if info.get("solve_failed"):
        return float("nan")
    return float(info["constraint_check"]["moisture_kg_per_kg"])


def _context_task(task: tuple[str, int, int]) -> dict[str, Any]:
    arm, seed, index = task
    import torch
    from stable_baselines3 import SAC

    torch.set_num_threads(1)
    spec = ARMS[arm]
    env, points = _env_and_points(spec["limit"])
    path = _final_checkpoint(spec["prefix"], seed)
    if _CACHE.get("model_path") != str(path):
        _CACHE["model"] = SAC.load(path, device="cpu")
        _CACHE["model_path"] = str(path)
    model = _CACHE["model"]
    point = points[index]
    t_amb, phi = point.ambient_temperature_celsius, point.ambient_relative_humidity

    observation = np.array([t_amb, phi], dtype=np.float32)
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        mean, log_std, _ = model.actor.get_action_dist_params(obs_tensor)
    mean_np = mean.cpu().numpy().reshape(-1).astype(np.float64)
    log_std_np = log_std.cpu().numpy().reshape(-1).astype(np.float64)

    inlet, gas = env._inlet_bound, env._gas_bound

    def physical(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.tanh(u)
        return (
            inlet.minimum + 0.5 * (a[..., 0] + 1.0) * inlet.span,
            gas.minimum + 0.5 * (a[..., 1] + 1.0) * gas.span,
        )

    rng = np.random.default_rng(1000 * seed + index)
    draws = mean_np + np.exp(log_std_np) * rng.standard_normal((N_DRAWS, 2))
    t_draws, u_draws = physical(draws)
    sigma_t, sigma_u = float(t_draws.std()), float(u_draws.std())

    t_mid, u_mid = physical(mean_np[None, :])
    t0, u0 = float(t_mid[0]), float(u_mid[0])
    x0 = _moisture(env, t_amb, phi, t0, u0)

    def derivative(plus: float, minus: float, step_plus: float, step_minus: float) -> float:
        return (plus - minus) / (step_plus + step_minus)

    t_hi = min(t0 + STEP_T_K, inlet.maximum)
    t_lo = max(t0 - STEP_T_K, inlet.minimum)
    u_hi = min(u0 + STEP_U_M_PER_S, gas.maximum)
    u_lo = max(u0 - STEP_U_M_PER_S, gas.minimum)
    d_t = derivative(
        _moisture(env, t_amb, phi, t_hi, u0),
        _moisture(env, t_amb, phi, t_lo, u0),
        t_hi - t0,
        t0 - t_lo,
    )
    d_u = derivative(
        _moisture(env, t_amb, phi, t0, u_hi),
        _moisture(env, t_amb, phi, t0, u_lo),
        u_hi - u0,
        u0 - u_lo,
    )
    sigma_x = float(np.hypot(d_t * sigma_t, d_u * sigma_u))

    direct = [
        _moisture(env, t_amb, phi, float(t_draws[i]), float(u_draws[i])) for i in range(N_DIRECT)
    ]
    direct_array = np.array(direct)
    finite = direct_array[np.isfinite(direct_array)]
    sigma_x_direct = float(finite.std()) if finite.size > 1 else float("nan")

    return {
        "arm": arm,
        "seed": seed,
        "context_index": index,
        "t_amb": t_amb,
        "phi": phi,
        "log_std": log_std_np.tolist(),
        "sigma_t_k": sigma_t,
        "sigma_u_m_per_s": sigma_u,
        "t_in_celsius": t0,
        "u_gas_m_per_s": u0,
        "moisture_kg_per_kg": x0,
        "dX_dT_per_k": d_t,
        "dX_du_per_m_per_s": d_u,
        "sigma_x_kg_per_kg": sigma_x,
        "sigma_x_direct_kg_per_kg": sigma_x_direct,
        "n_direct_failed_or_nan": int(N_DIRECT - finite.size),
        "distance_to_validity_boundary": VALIDITY_BOUNDARY - x0,
        "ratio": (VALIDITY_BOUNDARY - x0) / sigma_x if sigma_x > 0 else float("nan"),
        "ratio_direct": (VALIDITY_BOUNDARY - x0) / sigma_x_direct
        if sigma_x_direct > 0
        else float("nan"),
    }


def _binding_indices(limit: float | None) -> list[int]:
    env, points = _env_and_points(limit)
    indices = []
    for i, p in enumerate(points):
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=p.oracle_inlet_temperature_celsius,
            gas_velocity_m_per_s=p.oracle_gas_velocity_m_per_s,
            ambient_temperature_celsius=p.ambient_temperature_celsius,
            ambient_relative_humidity=p.ambient_relative_humidity,
        )
        if not info.get("solve_failed") and info["constraint_check"]["binding_moisture"]:
            indices.append(i)
    return indices


def _median(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def main() -> int:
    require_clean_tree()
    binding = {name: _binding_indices(spec["limit"]) for name, spec in ARMS.items()}
    tasks = [(arm, seed, i) for arm in ARMS for seed in SEEDS for i in range(25)]
    with Pool(WORKERS) as pool:
        rows = pool.map(_context_task, tasks, chunksize=5)

    per_pair: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            mine = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            on = [r for r in mine if r["context_index"] in binding[arm]]
            per_pair[f"{arm}|{seed}"] = {
                "arm": arm,
                "seed": seed,
                "n_binding": len(on),
                "binding": {
                    "distance": _median([r["distance_to_validity_boundary"] for r in on]),
                    "sigma_t_k": _median([r["sigma_t_k"] for r in on]),
                    "sigma_u_m_per_s": _median([r["sigma_u_m_per_s"] for r in on]),
                    "sigma_x": _median([r["sigma_x_kg_per_kg"] for r in on]),
                    "sigma_x_direct": _median([r["sigma_x_direct_kg_per_kg"] for r in on]),
                    "ratio": _median([r["ratio"] for r in on]),
                    "ratio_direct": _median([r["ratio_direct"] for r in on]),
                    "log_std_median": [
                        _median([r["log_std"][0] for r in on]),
                        _median([r["log_std"][1] for r in on]),
                    ],
                },
                "all_25": {
                    "distance": _median([r["distance_to_validity_boundary"] for r in mine]),
                    "sigma_x": _median([r["sigma_x_kg_per_kg"] for r in mine]),
                    "ratio": _median([r["ratio"] for r in mine]),
                },
            }

    pairs = list(per_pair.values())
    distance = np.array([p["binding"]["distance"] for p in pairs])
    sigma_x = np.array([p["binding"]["sigma_x"] for p in pairs])
    ratio = np.array([p["binding"]["ratio"] for p in pairs])
    ratio_direct = np.array([p["binding"]["ratio_direct"] for p in pairs])
    sigma_direct = np.array([p["binding"]["sigma_x_direct"] for p in pairs])
    rho = float(stats.spearmanr(distance, sigma_x).statistic)
    rho_direct = float(stats.spearmanr(distance, sigma_direct, nan_policy="omit").statistic)
    cv = lambda v: float(np.nanstd(v) / np.nanmean(v))  # noqa: E731
    cv_ratio, cv_distance = cv(ratio), cv(distance)
    if rho >= SUPPORTED_MIN_SPEARMAN and cv_ratio < cv_distance:
        verdict = "SUPPORTED"
    elif rho < WEAKENED_MAX_SPEARMAN or cv_ratio >= cv_distance:
        verdict = "WEAKENED"
    else:
        verdict = "INCONCLUSIVE"

    arm_summary = {}
    for arm in ARMS:
        mine = [p for p in pairs if p["arm"] == arm]
        arm_summary[arm] = {
            "distance_median": float(np.median([p["binding"]["distance"] for p in mine])),
            "sigma_x_median": float(np.median([p["binding"]["sigma_x"] for p in mine])),
            "ratio_median": float(np.median([p["binding"]["ratio"] for p in mine])),
            "ratio_per_seed": [p["binding"]["ratio"] for p in mine],
            "ratio_direct_median": float(
                np.nanmedian([p["binding"]["ratio_direct"] for p in mine])
            ),
            "sigma_t_k_median": float(np.median([p["binding"]["sigma_t_k"] for p in mine])),
            "sigma_u_median": float(np.median([p["binding"]["sigma_u_m_per_s"] for p in mine])),
        }
    medians = np.array([a["ratio_median"] for a in arm_summary.values()])
    payload = {
        "_provenance": provenance(),
        "decision_rule_constants": {
            "supported_min_spearman": SUPPORTED_MIN_SPEARMAN,
            "weakened_max_spearman": WEAKENED_MAX_SPEARMAN,
            "several_spreads": SEVERAL_SPREADS,
        },
        "n_pairs": len(pairs),
        "spearman_distance_vs_sigma_x": rho,
        "spearman_distance_vs_sigma_x_direct": rho_direct,
        "cv_ratio": cv_ratio,
        "cv_distance": cv_distance,
        "cv_ratio_direct": cv(ratio_direct),
        "verdict": verdict,
        "arm_summary": arm_summary,
        "relative_range_of_arm_median_ratio": float(
            (medians.max() - medians.min()) / np.median(medians)
        ),
        "median_ratio_at_least_several_spreads": bool(np.median(medians) >= SEVERAL_SPREADS),
        "per_pair": per_pair,
        "binding_context_indices": binding,
        "per_context": rows,
    }
    write_artifact("s07_policy_spread", payload)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in ("verdict", "spearman_distance_vs_sigma_x", "cv_ratio", "cv_distance")
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
