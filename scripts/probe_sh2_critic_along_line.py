r"""SH2: S0.9 lines for the critic of the shielded arm (existing checkpoints, no training).

Same method and the same criterion constants as ``probe_s09_critic_along_line.py`` (imported, not
copied): at the oracle-binding contexts of the limit 0.15 (22 of 25), the line from the policy mean in
the direction of increasing moisture (unit moisture gradient in normalised action space, clipped to the
action box) up to the validity boundary 0.20 and a little beyond; 20 points; both critics and their
minimum; the TRUE reward of the unshielded Lagrangian environment with the final multipliers (as S0.9);
the criterion "the line supports smearing" (gap of the critic peak to 0.20 at least 0.05 kg/kg and the
true reward still rising by more than 0.001 kW) is unchanged and applied per line. The arm is supported iff the
median over seeds of the share of supporting lines is at least 0.5 (as in S0.9).

Arm: SH, prefix ``rl_bandit_lagrange_sh2_shield`` (limit 0.15, Lagrange term, shield during training).
Reference: arm C of S0.9 (artifact ``s09_critic_along_line``, not recomputed).

Added, because the critic of this arm was trained on the SHIELDED reward (flat beyond the tightened
limit), and reported separately, not as a criterion: per line the reward the learner actually received
for a proposal at that point (shield applied: the Lagrangian reward of the executed action) next to the
unshielded true reward, and the change of the minimum critic from the point where the moisture crosses
the tightened limit (0.15 - m) to the last point before the cliff: a critic that has learned the flat
reward changes little there, a smeared cliff falls by 8-16 kW (S0.9, arm C: -11.25 kW median over all
the points from the peak).

    python scripts/probe_sh2_critic_along_line.py
"""

from __future__ import annotations

import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

import probe_s09_critic_along_line as p9
from _common import provenance, require_clean_tree, write_artifact

ARM = "SH_L015_shield"
MARGIN_M = 0.005961819615175404
LIMIT = 0.15
p9.ARMS[ARM] = {"prefix": "rl_bandit_lagrange_sh2_shield", "limit": LIMIT, "lagrange": True}

_SHIELD_ENV: Any = None


def _shield_env(seed: int) -> Any:
    """Shielded environment with the final multipliers of the run (reward the learner received)."""
    global _SHIELD_ENV
    key = f"shield_env|{seed}"
    if p9._CACHE.get("shield_key") != key:
        from twinlink_ml.rl import BanditEnvConfig
        from twinlink_ml.rl.lagrange import LagrangeConfig
        from twinlink_ml.rl.shield import ShieldedLagrangeBanditEnv

        params = p9._record(p9.ARMS[ARM]["prefix"], seed)["params"]
        lagrange = LagrangeConfig(
            margin_kg_per_kg=params["lagrange"]["margin_kg_per_kg"],
            initial_multiplier_kw=params["lagrange"]["initial_multiplier_kw"],
            step_size_kw=params["lagrange"]["step_size_kw"],
            update_interval_steps=params["lagrange"]["update_interval_steps"],
            context_bins=tuple(params["lagrange"]["context_bins"]),
        )
        warnings.filterwarnings("ignore")
        env = ShieldedLagrangeBanditEnv(
            lagrange, BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT)
        )
        env.set_multipliers(np.array(params["multipliers_kw"]).reshape(lagrange.context_bins))
        _SHIELD_ENV = env
        p9._CACHE["shield_key"] = key
    return _SHIELD_ENV


def _task(task: tuple[str, int, int]) -> dict[str, Any]:
    arm, seed, index = task
    result = p9._line_task(task)
    if "points" not in result:
        return result
    env = _shield_env(seed)
    t_amb, phi = result["t_amb"], result["phi"]
    inlet, gas = env._inlet_bound, env._gas_bound
    a0, direction = np.array(result["a0"]), np.array(result["direction"])
    for point in result["points"]:
        a = np.clip(a0 + point["s"] * direction, -1.0, 1.0)
        record, info = env.shield_action(
            inlet.minimum + 0.5 * (a[0] + 1.0) * inlet.span,
            gas.minimum + 0.5 * (a[1] + 1.0) * gas.span,
            t_amb,
            phi,
        )
        reward, _ = env.lagrangian_reward(
            info, ambient_temperature_celsius=t_amb, ambient_relative_humidity=phi
        )
        point["shielded_reward_kw"] = float(reward)
        point["shield_active"] = bool(record["active"])
        point["shield_saturated"] = bool(record["saturated"])
    target = LIMIT - MARGIN_M
    pre = [
        p
        for p in result["points"]
        if np.isfinite(p["moisture"]) and p["moisture"] <= p9.VALIDITY_BOUNDARY
    ]
    crossing = [j for j, p in enumerate(pre) if p["moisture"] > target]
    if crossing and result.get("cliff_reached"):
        j = crossing[0]
        result["q_min_change_from_tightened_limit_to_cliff_edge"] = (
            pre[-1]["q_min"] - pre[max(j - 1, 0)]["q_min"]
        )
        result["shielded_reward_change_from_tightened_limit_to_cliff_edge_kw"] = (
            pre[-1]["shielded_reward_kw"] - pre[max(j - 1, 0)]["shielded_reward_kw"]
        )
        result["true_reward_change_from_tightened_limit_to_cliff_edge_kw"] = (
            pre[-1]["true_reward_kw"] - pre[max(j - 1, 0)]["true_reward_kw"]
        )
    return result


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def main() -> int:
    require_clean_tree()
    spec = p9.ARMS[ARM]
    binding = p9._binding_indices(spec["limit"])
    tasks = [(ARM, seed, i) for seed in p9.SEEDS for i in binding]
    with Pool(p9.WORKERS) as pool:
        lines = pool.map(_task, tasks, chunksize=2)

    shares = []
    for seed in p9.SEEDS:
        mine = [r for r in lines if r["seed"] == seed]
        shares.append(float(np.mean([bool(r.get("supports_smearing")) for r in mine])))
    with_gap = [r for r in lines if "gap_to_boundary" in r]
    crossing = [r for r in lines if "q_min_change_from_tightened_limit_to_cliff_edge" in r]
    arm = {
        "n_lines": len(lines),
        "n_cliff_reached": sum(1 for r in lines if r.get("cliff_reached")),
        "share_supporting_per_seed": shares,
        "median_share": float(np.median(shares)),
        "arm_supported_as_in_s09": bool(np.median(shares) >= p9.MIN_SHARE_PER_ARM),
        "median_gap_to_boundary": _median([r["gap_to_boundary"] for r in with_gap]),
        "median_moisture_at_q_min_peak": _median([r["moisture_at_q_min_peak"] for r in with_gap]),
        "median_true_reward_rise_kw": _median(
            [r["true_reward_rise_to_cliff_edge_kw"] for r in with_gap]
        ),
        "median_q_min_drop_kw": _median([r["q_min_drop_to_cliff_edge"] for r in with_gap]),
        "n_lines_crossing_tightened_limit_before_cliff": len(crossing),
        "median_q_min_change_tightened_limit_to_cliff_edge_kw": _median(
            [r["q_min_change_from_tightened_limit_to_cliff_edge"] for r in crossing]
        ),
        "median_shielded_reward_change_tightened_limit_to_cliff_edge_kw": _median(
            [r["shielded_reward_change_from_tightened_limit_to_cliff_edge_kw"] for r in crossing]
        ),
        "median_true_reward_change_tightened_limit_to_cliff_edge_kw": _median(
            [r["true_reward_change_from_tightened_limit_to_cliff_edge_kw"] for r in crossing]
        ),
    }
    payload = {
        "_provenance": provenance(),
        "criterion_constants": {
            "min_gap_kg_per_kg": p9.MIN_GAP_KG_PER_KG,
            "min_reward_rise_kw": p9.MIN_REWARD_RISE_KW,
            "min_share_per_arm": p9.MIN_SHARE_PER_ARM,
        },
        "binding_context_indices": binding,
        "arm": arm,
        "lines": lines,
    }
    write_artifact("sh2_critic_along_line", payload)
    print(json.dumps(arm, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
