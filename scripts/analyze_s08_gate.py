r"""S0.8: two counter-tests of the policy-spread hypothesis (environment of arm C of S0.6).

Arms, all in the environment of arm C (quality limit 0.15, Lagrange term, lambda_0 0.5, m =
0.005961819615175404, 5 x 5 cells, eta 0.2 kW, update every 250 steps), seeds 0-4, 10 000 steps:

    C  SAC, stable-baselines3 defaults (S0.6, exists)
    F  TD3, defaults (deterministic policy; stable-baselines3 TD3 has NO action noise by default,
       so it explores only during the first 100 random steps -- a test WITHOUT exploration after
       step 100, to be labelled so in the finding)
    G  SAC with target_entropy = -6 (default -2; confirmed with the user before training)
    H  TD3 with Gaussian action noise sigma = 0.1 (added after the first S0.8 plan; the readings
       below were extended BEFORE any S0.8 evaluation, only the F runs were in progress)

Pure evaluation of records and checkpoints at step 10 000 (deterministic policy), on the 25
Befund 52 grid points with the oracle and fixed operating point of the limit 0.15
(``s05_limit015_oracle_and_fixed``). Gate G1-G3 per arm as in S0.6 (constants of S0.3), against the
limit 0.15 and the penalty arm E of S0.6. THE READINGS ARE FIXED HERE before any S0.8 run:

  An arm X in {F, G, H} QUALIFIES iff
    (i)  it is clearly closer to the limit than C: the median distance to the true limit at the
         binding contexts is smaller than that of C by more than the SX1 two-sample threshold
         (``t(nu_W, 0.975) * sqrt((s_C^2 + s_X^2) / n)``, nu_eff = 3.7623), AND
    (ii) its pooled closed-regret fraction is positive and above the one-sample t threshold (G2).
  L1  G qualifies AND at least one of F, H qualifies: the spread of the policy was the cause; RL
      is viable with a deterministic or narrow policy.
  L2  at least one of F, G, H qualifies but not as in L1: report which; it narrows down the
      mechanism.
  L3  none qualifies: the cause lies in the critic or the learning budget, not in the spread.

Also reported (not part of the readings): per arm the lambda history per cell (categories of S0.3),
the number of active lambda series, the number of training steps with g_tilde > 0, the distance in
units of m, and for SAC arms the median pre-tanh log_std of the actor at the binding contexts.

    python scripts/analyze_s08_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

import analyze_s03_gate as g3
import analyze_s06_gate as g6
from _common import provenance, require_clean_tree, write_artifact

from twinlink_ml.rl import (
    BanditEnvConfig,
    OperatingPointBanditEnv,
    evaluate_policy,
    flat_metrics,
    load_test_points,
)
from twinlink_ml.rl.evaluation import moisture_margin_at_oracle_binding_points

RUNS = g3.RUNS
REPO_ROOT = g3.REPO_ROOT
SEEDS = g3.SEEDS
LIMIT = 0.15
MARGIN_M = g6.MARGIN_M

ARMS: dict[str, dict[str, str]] = {
    "C_SAC_default": {"prefix": "rl_bandit_lagrange_s06_l015_lam05", "algorithm": "SAC"},
    "F_TD3_default": {"prefix": "rl_bandit_lagrange_s08_td3", "algorithm": "TD3"},
    "G_SAC_low_entropy": {"prefix": "rl_bandit_lagrange_s08_sac_lowent", "algorithm": "SAC"},
    "H_TD3_noise": {"prefix": "rl_bandit_lagrange_s08_td3_noise", "algorithm": "TD3"},
}
PENALTY_E = "rl_bandit_sac_learning_curve_s06_penalty_l015"


def _evaluate(prefix: str, algorithm: str) -> dict[int, dict[str, float]]:
    from stable_baselines3 import SAC, TD3

    cls = {"SAC": SAC, "TD3": TD3}[algorithm]
    env = OperatingPointBanditEnv(BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT))
    points = load_test_points(g6.REFERENCE_015)
    out: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        record = g3._final_record(prefix, seed)
        model = cls.load(REPO_ROOT / record["params"]["checkpoint/path"], device="cpu")

        def policy(observation: np.ndarray, model: Any = model) -> np.ndarray:
            action, _ = model.predict(observation, deterministic=True)
            return np.asarray(action, dtype=np.float64)

        metrics = flat_metrics(evaluate_policy(env, policy, points))
        margin, n_binding = moisture_margin_at_oracle_binding_points(env, policy, points)
        metrics["moisture_margin_binding_median_kg_per_kg"] = margin
        metrics["moisture_margin_binding_n"] = float(n_binding)
        if algorithm == "SAC":
            import torch

            log_stds = []
            for p in points:
                obs = np.array(
                    [p.ambient_temperature_celsius, p.ambient_relative_humidity], dtype=np.float32
                )
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    _, log_std, _ = model.actor.get_action_dist_params(obs_tensor)
                log_stds.append(log_std.cpu().numpy().reshape(-1))
            metrics["log_std_median_dim0"] = float(np.median([v[0] for v in log_stds]))
            metrics["log_std_median_dim1"] = float(np.median([v[1] for v in log_stds]))
        out[seed] = metrics
    return out


def _col(arm: dict[int, dict[str, float]], key: str) -> list[float]:
    return [arm[s][key] for s in SEEDS]


def main() -> int:
    require_clean_tree()
    nu = g3._nu_eff()
    penalty = _evaluate(PENALTY_E, "SAC")
    evaluated = {name: _evaluate(spec["prefix"], spec["algorithm"]) for name, spec in ARMS.items()}

    results: dict[str, Any] = {}
    for name, arm in evaluated.items():
        prefix = ARMS[name]["prefix"]
        results[name] = {
            "algorithm": ARMS[name]["algorithm"],
            "feasibility_rate": g3._summary(_col(arm, "feasibility_rate")),
            "pooled_closed_regret_fraction": g3._summary(
                _col(arm, "pooled_closed_regret_fraction")
            ),
            "distance_to_limit_binding_kg_per_kg": g3._summary(
                _col(arm, "moisture_margin_binding_median_kg_per_kg")
            ),
            "distance_in_units_of_m": g3._summary(
                [v / MARGIN_M for v in _col(arm, "moisture_margin_binding_median_kg_per_kg")]
            ),
            "n_binding_contexts": _col(arm, "moisture_margin_binding_n"),
            "gate": g6._gate(arm, penalty, nu),
            "n_active_lambda_series_of_125": g6._active_series(prefix),
            "steps_with_g_tilde_positive_per_seed": g6._positive_g_steps(prefix),
            "lambda_per_cell": g3.lambda_cells(prefix),
            "trace_summary": g3._trace_summary(prefix),
        }
        if ARMS[name]["algorithm"] == "SAC":
            results[name]["log_std_median_dim0_per_seed"] = _col(arm, "log_std_median_dim0")
            results[name]["log_std_median_dim1_per_seed"] = _col(arm, "log_std_median_dim1")

    dist_c = _col(evaluated["C_SAC_default"], "moisture_margin_binding_median_kg_per_kg")
    qualify: dict[str, Any] = {}
    for name in ("F_TD3_default", "G_SAC_low_entropy", "H_TD3_noise"):
        dist_x = _col(evaluated[name], "moisture_margin_binding_median_kg_per_kg")
        pooled_x = _col(evaluated[name], "pooled_closed_regret_fraction")
        two = g3.two_sample_threshold(dist_c, dist_x, nu)
        closer = float(np.median(dist_c) - np.median(dist_x))
        g2_threshold = g3.one_sample_threshold(pooled_x, nu)
        positive = float(np.median(pooled_x)) > 0.0 and float(np.median(pooled_x)) > g2_threshold
        qualify[name] = {
            "closer_than_C_by_kg_per_kg": closer,
            "two_sample_threshold": two,
            "closer_resolved": bool(closer > two["threshold"]),
            "pooled_median": float(np.median(pooled_x)),
            "pooled_one_sample_threshold": g2_threshold,
            "pooled_positive_and_resolved": bool(positive),
            "qualifies": bool(closer > two["threshold"] and positive),
        }
    q = {k: v["qualifies"] for k, v in qualify.items()}
    l1 = q["G_SAC_low_entropy"] and (q["F_TD3_default"] or q["H_TD3_noise"])
    any_q = any(q.values())
    readings = {
        "L1_G_and_a_TD3_arm_qualify_spread_was_the_cause": l1,
        "L2_some_arm_qualifies_not_L1": any_q and not l1,
        "L2_which": [k for k, v in q.items() if v] if any_q and not l1 else [],
        "L3_none_critic_or_budget": not any_q,
    }
    payload = {
        "_provenance": provenance(),
        "test_set": "25 Befund-52 grid points; oracle and fixed point of the limit 0.15",
        "evaluation_step": g3.EVALUATION_STEP,
        "seeds": list(SEEDS),
        "m_kg_per_kg": MARGIN_M,
        "arms": results,
        "qualification": qualify,
        "readings": readings,
    }
    write_artifact("s08_gate", payload)
    print(json.dumps({k: v["gate"] for k, v in results.items()}, indent=1))
    print(json.dumps(qualify, indent=1))
    print(json.dumps(readings, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
