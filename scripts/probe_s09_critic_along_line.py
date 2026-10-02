r"""S0.9: test whether the learned Q value falls well before the cliff while the true reward rises.

Evaluation of existing step-10 000 checkpoints, no training. Arms: A (limit 0.20, Lagrange term),
C (limit 0.15, Lagrange term), E (penalty arm at limit 0.15), P (penalty arm of S0.3, limit 0.20),
seeds 0-4. At the test contexts where the Befund 52 oracle of the arm's reference binds the limit
(17 at limit 0.20, 22 at 0.15):

  * the policy mean (the evaluated deterministic action ``tanh(mean)``) in normalised action space
    is the start of a line in the direction of increasing product moisture: the unit gradient of the
    moisture with respect to the two normalised actions (central differences, step 0.01), i.e. the
    direction towards the validity boundary 0.20 kg/kg;
  * the line is followed until the moisture reaches 0.20 (coarse scan of 40 points, then 10
    bisection steps for the crossing, ``s_cliff``) and a little beyond (to 1.15 * s_cliff), but not
    outside the action box; 20 equally spaced points on [0, s_end];
  * at every point: the true reward from the simulator (the reward the arm was trained on: the
    Lagrangian reward with the FINAL multipliers for the Lagrange arms, the penalty reward with the
    arm's solver-failure value for the penalty arms), both critics separately and their minimum, and
    the product moisture.

Criterion, fixed before the run. For a line, let ``X_Q`` be the moisture at the point (among the
points with moisture <= 0.20) where the minimum of the two critics is largest, i.e. where it starts
to fall. The line SUPPORTS the smearing by the critic iff the cliff is reached, the gap
``0.20 - X_Q`` is at least 0.05 kg/kg, and the true reward at the last point before the cliff exceeds
the true reward at ``X_Q`` by more than 0.001 kW (the true reward still rises). An arm is supported
iff the median over seeds of the share of its lines that support it is at least 0.5. The smearing is
regarded as supported overall iff at least three of the four arms are supported.

    python scripts/probe_s09_critic_along_line.py
"""

from __future__ import annotations

import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "artifacts" / "runs"
SEEDS = (0, 1, 2, 3, 4)
EVALUATION_STEP = 10_000
VALIDITY_BOUNDARY = 0.20
N_POINTS = 20
N_COARSE = 40
N_BISECT = 10
BEYOND = 1.15
GRADIENT_STEP = 0.01
WORKERS = 12
REFERENCE_015 = RUNS / "s05_limit015_oracle_and_fixed.json"

# --- criterion, fixed before the run --------------------------------------------------------
MIN_GAP_KG_PER_KG = 0.05
MIN_REWARD_RISE_KW = 0.001
MIN_SHARE_PER_ARM = 0.5
MIN_ARMS_SUPPORTED = 3
# --------------------------------------------------------------------------------------------

ARMS: dict[str, dict[str, Any]] = {
    "A_L020_lam05": {
        "prefix": "rl_bandit_lagrange_s03",
        "limit": None,
        "lagrange": True,
    },
    "C_L015_lam05": {
        "prefix": "rl_bandit_lagrange_s06_l015_lam05",
        "limit": 0.15,
        "lagrange": True,
    },
    "E_penalty_L015": {
        "prefix": "rl_bandit_sac_learning_curve_s06_penalty_l015",
        "limit": 0.15,
        "lagrange": False,
        "failure_kw": 207.16872605855167,
    },
    "P_penalty_L020_s03": {
        "prefix": "rl_bandit_sac_learning_curve_s03_penalty",
        "limit": None,
        "lagrange": False,
        "failure_kw": 155.57896052362634,
    },
}

_CACHE: dict[str, Any] = {}


def _record(prefix: str, seed: int) -> dict[str, Any]:
    path = RUNS / f"{prefix}_seed{seed}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["params"]["n_env_steps"] == EVALUATION_STEP:
            return row
    raise SystemExit(f"no step-{EVALUATION_STEP} record in {path}")


def _plain_env(limit: float | None, failure_kw: float | None = None) -> Any:
    from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv

    warnings.filterwarnings("ignore")
    return OperatingPointBanditEnv(
        BanditEnvConfig(product_moisture_limit_kg_per_kg=limit, solve_failure_penalty_kw=failure_kw)
    )


def _points(limit: float | None) -> Any:
    from twinlink_ml.rl import load_test_points

    return load_test_points() if limit is None else load_test_points(REFERENCE_015)


def _reward_env(arm: str, seed: int) -> Any:
    """The environment whose reward the arm was trained on (final multipliers for Lagrange arms)."""
    spec = ARMS[arm]
    key = f"reward_env|{arm}|{seed}"
    if key not in _CACHE:
        if spec["lagrange"]:
            from twinlink_ml.rl import BanditEnvConfig
            from twinlink_ml.rl.lagrange import LagrangeBanditEnv, LagrangeConfig

            params = _record(spec["prefix"], seed)["params"]
            lagrange = LagrangeConfig(
                margin_kg_per_kg=params["lagrange"]["margin_kg_per_kg"],
                initial_multiplier_kw=params["lagrange"]["initial_multiplier_kw"],
                step_size_kw=params["lagrange"]["step_size_kw"],
                update_interval_steps=params["lagrange"]["update_interval_steps"],
                context_bins=tuple(params["lagrange"]["context_bins"]),
            )
            warnings.filterwarnings("ignore")
            env = LagrangeBanditEnv(
                lagrange, BanditEnvConfig(product_moisture_limit_kg_per_kg=spec["limit"])
            )
            env.set_multipliers(np.array(params["multipliers_kw"]).reshape(lagrange.context_bins))
        else:
            env = _plain_env(spec["limit"], spec["failure_kw"])
        _CACHE[key] = env
    return _CACHE[key]


def _binding_indices(limit: float | None) -> list[int]:
    env = _plain_env(limit)
    out = []
    for i, p in enumerate(_points(limit)):
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=p.oracle_inlet_temperature_celsius,
            gas_velocity_m_per_s=p.oracle_gas_velocity_m_per_s,
            ambient_temperature_celsius=p.ambient_temperature_celsius,
            ambient_relative_humidity=p.ambient_relative_humidity,
        )
        if not info.get("solve_failed") and info["constraint_check"]["binding_moisture"]:
            out.append(i)
    return out


def _evaluate(env: Any, t_amb: float, phi: float, a: np.ndarray) -> tuple[float, float]:
    """``(moisture, reward)`` at the normalised action ``a``; moisture NaN on solver failure."""
    inlet, gas = env._inlet_bound, env._gas_bound
    reward, info = env.evaluate_operating_point(
        inlet_temperature_celsius=inlet.minimum + 0.5 * (a[0] + 1.0) * inlet.span,
        gas_velocity_m_per_s=gas.minimum + 0.5 * (a[1] + 1.0) * gas.span,
        ambient_temperature_celsius=t_amb,
        ambient_relative_humidity=phi,
    )
    if info.get("solve_failed"):
        return float("nan"), float(reward)
    return float(info["constraint_check"]["moisture_kg_per_kg"]), float(reward)


def _line_task(task: tuple[str, int, int]) -> dict[str, Any]:
    arm, seed, index = task
    import torch
    from stable_baselines3 import SAC

    torch.set_num_threads(1)
    spec = ARMS[arm]
    env = _reward_env(arm, seed)
    plain = _plain_env(spec["limit"])
    point = _points(spec["limit"])[index]
    t_amb, phi = point.ambient_temperature_celsius, point.ambient_relative_humidity
    path = REPO_ROOT / _record(spec["prefix"], seed)["params"]["checkpoint/path"]
    if _CACHE.get("model_path") != str(path):
        _CACHE["model"] = SAC.load(path, device="cpu")
        _CACHE["model_path"] = str(path)
    model = _CACHE["model"]

    observation = np.array([t_amb, phi], dtype=np.float32)
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        mean, _, _ = model.actor.get_action_dist_params(obs_tensor)
    a0 = np.tanh(mean.cpu().numpy().reshape(-1).astype(np.float64))

    # direction of increasing moisture: unit gradient in normalised action space
    gradient = np.zeros(2)
    for i in range(2):
        hi, lo = a0.copy(), a0.copy()
        hi[i] = min(a0[i] + GRADIENT_STEP, 1.0)
        lo[i] = max(a0[i] - GRADIENT_STEP, -1.0)
        x_hi = _evaluate(plain, t_amb, phi, hi)[0]
        x_lo = _evaluate(plain, t_amb, phi, lo)[0]
        gradient[i] = (x_hi - x_lo) / (hi[i] - lo[i])
    if not np.all(np.isfinite(gradient)) or np.linalg.norm(gradient) == 0.0:
        return {"arm": arm, "seed": seed, "context_index": index, "skipped": "no gradient"}
    direction = gradient / np.linalg.norm(gradient)

    with np.errstate(divide="ignore", invalid="ignore"):
        limits = np.where(
            direction > 0,
            (1.0 - a0) / direction,
            np.where(direction < 0, (-1.0 - a0) / direction, np.inf),
        )
    s_bound = float(np.min(limits))

    def moisture(s: float) -> float:
        return _evaluate(plain, t_amb, phi, np.clip(a0 + s * direction, -1.0, 1.0))[0]

    def beyond(s: float) -> bool:
        x = moisture(s)
        return bool(np.isnan(x) or x >= VALIDITY_BOUNDARY)

    cliff_reached = False
    s_cliff = s_bound
    previous = 0.0
    for k in range(1, N_COARSE + 1):
        s = s_bound * k / N_COARSE
        if beyond(s):
            lo_s, hi_s = previous, s
            for _ in range(N_BISECT):
                mid = 0.5 * (lo_s + hi_s)
                if beyond(mid):
                    hi_s = mid
                else:
                    lo_s = mid
            s_cliff = hi_s
            cliff_reached = True
            break
        previous = s
    s_end = min(s_bound, BEYOND * s_cliff) if cliff_reached else s_bound

    points = []
    for j in range(N_POINTS):
        s = s_end * j / (N_POINTS - 1)
        a = np.clip(a0 + s * direction, -1.0, 1.0)
        x, reward = _evaluate(env, t_amb, phi, a)
        x_plain = _evaluate(plain, t_amb, phi, a)[0] if np.isnan(x) else x
        with torch.no_grad():
            obs_t, _ = model.policy.obs_to_tensor(observation)
            act_t = torch.as_tensor(a.astype(np.float32)).reshape(1, 2)
            q1, q2 = (v.item() for v in model.critic(obs_t, act_t))
        points.append(
            {
                "s": s,
                "moisture": x_plain,
                "true_reward_kw": reward,
                "q1": q1,
                "q2": q2,
                "q_min": min(q1, q2),
            }
        )

    pre = [p for p in points if np.isfinite(p["moisture"]) and p["moisture"] <= VALIDITY_BOUNDARY]
    result: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "context_index": index,
        "t_amb": t_amb,
        "phi": phi,
        "a0": a0.tolist(),
        "direction": direction.tolist(),
        "s_bound": s_bound,
        "s_cliff": s_cliff,
        "cliff_reached": cliff_reached,
        "points": points,
    }
    if cliff_reached and len(pre) >= 3:
        j_q = int(np.argmax([p["q_min"] for p in pre]))
        x_q = pre[j_q]["moisture"]
        reward_rise = pre[-1]["true_reward_kw"] - pre[j_q]["true_reward_kw"]
        result.update(
            {
                "moisture_at_q_min_peak": x_q,
                "gap_to_boundary": VALIDITY_BOUNDARY - x_q,
                "q_min_drop_to_cliff_edge": pre[-1]["q_min"] - pre[j_q]["q_min"],
                "true_reward_rise_to_cliff_edge_kw": reward_rise,
                "supports_smearing": bool(
                    VALIDITY_BOUNDARY - x_q >= MIN_GAP_KG_PER_KG
                    and reward_rise > MIN_REWARD_RISE_KW
                ),
            }
        )
    else:
        result["supports_smearing"] = False
        result["note"] = "cliff not reached inside the action box or fewer than 3 points before it"
    return result


def main() -> int:
    require_clean_tree()
    binding = {arm: _binding_indices(spec["limit"]) for arm, spec in ARMS.items()}
    tasks = [(arm, seed, i) for arm in ARMS for seed in SEEDS for i in binding[arm]]
    with Pool(WORKERS) as pool:
        lines = pool.map(_line_task, tasks, chunksize=4)

    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        shares = []
        for seed in SEEDS:
            mine = [r for r in lines if r["arm"] == arm and r["seed"] == seed]
            shares.append(float(np.mean([bool(r.get("supports_smearing")) for r in mine])))
        mine_all = [r for r in lines if r["arm"] == arm and "gap_to_boundary" in r]
        per_arm[arm] = {
            "n_lines": sum(1 for r in lines if r["arm"] == arm),
            "n_cliff_reached": sum(1 for r in lines if r["arm"] == arm and r.get("cliff_reached")),
            "share_supporting_per_seed": shares,
            "median_share": float(np.median(shares)),
            "arm_supported": bool(np.median(shares) >= MIN_SHARE_PER_ARM),
            "median_gap_to_boundary": float(np.median([r["gap_to_boundary"] for r in mine_all]))
            if mine_all
            else None,
            "median_moisture_at_q_min_peak": float(
                np.median([r["moisture_at_q_min_peak"] for r in mine_all])
            )
            if mine_all
            else None,
            "median_true_reward_rise_kw": float(
                np.median([r["true_reward_rise_to_cliff_edge_kw"] for r in mine_all])
            )
            if mine_all
            else None,
            "median_q_min_drop_kw": float(
                np.median([r["q_min_drop_to_cliff_edge"] for r in mine_all])
            )
            if mine_all
            else None,
        }
    n_supported = sum(1 for v in per_arm.values() if v["arm_supported"])
    payload = {
        "_provenance": provenance(),
        "criterion_constants": {
            "min_gap_kg_per_kg": MIN_GAP_KG_PER_KG,
            "min_reward_rise_kw": MIN_REWARD_RISE_KW,
            "min_share_per_arm": MIN_SHARE_PER_ARM,
            "min_arms_supported": MIN_ARMS_SUPPORTED,
        },
        "binding_context_indices": binding,
        "per_arm": per_arm,
        "n_arms_supported": n_supported,
        "smearing_supported": bool(n_supported >= MIN_ARMS_SUPPORTED),
        "lines": lines,
    }
    write_artifact("s09_critic_along_line", payload)
    print(json.dumps({k: payload[k] for k in ("n_arms_supported", "smearing_supported")}, indent=1))
    print(json.dumps(per_arm, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
