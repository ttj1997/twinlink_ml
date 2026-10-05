r"""SH2: SAC with the moisture shield in the environment of arm C (Befund 68 addendum, SH1/SH2).

Arm SH: SAC, stable-baselines3 defaults, seeds 0-4, 10 000 steps, environment of arm C of S0.6 (quality
limit 0.15, Lagrange term, lambda_0 0.5, m = 0.005961819615175404, 5 x 5 cells, eta 0.2 kW, update every
250 steps) with the shield of ``twinlink_ml.rl.shield`` (inlet temperature pushed up at constant gas
velocity onto the tightened limit 0.15 - m, Brent tolerance 0.05 K; replay buffer holds the proposed
action with the reward of the executed one). Run prefix ``rl_bandit_lagrange_sh2_shield``.

Pure evaluation of the records and checkpoints at step 10 000, deterministic policy, on the 25
Befund 52 grid points with the oracle and fixed operating point of the limit 0.15
(``s05_limit015_oracle_and_fixed``). THE GATE AND THE CONSEQUENCE ARE FIXED HERE before any SH2 run.

GATE (verdict). The policy ALONE (no shield at evaluation, as for every earlier arm), evaluated in the
plain environment against the TRUE limit 0.15: G1-G3 exactly as in S0.6 (``analyze_s06_gate._gate``,
constants of S0.3), against the penalty arm E of S0.6. PASS = G1 and G2 and G3.

CONSEQUENCE, fixed beforehand by the project lead: PASS -> RL goes into the multi-plant stage as a
comparison; no PASS -> RL stays a diagnostic finding. No further RL attempt without consultation.

ALSO REPORTED, NOT PART OF THE VERDICT (labelled as such):
  * the same metrics for the SHIELDED deterministic policy (the policy mean action pushed by the
    shield before evaluation) -- the shield is part of what would be deployed, but the gate was
    specified for the policy alone;
  * how often the shield acts: per seed the share of training steps with a shifted action, per window
    of 1 000 steps, saturations, solver-failed proposals (not shieldable: moisture undefined, they keep
    the Lagrange failure value), steps with g_tilde > 0 at the EXECUTED action, median shift in K;
  * run-time per step against the estimate below, and the lambda series (categories of S0.3);
  * arm C (S0.6) evaluated the same way, and the SH - C differences with their SX1 thresholds.

RUN-TIME ESTIMATE, made before the runs (probe ``sh1_shield_probe``: 8-9 solver calls and 2.4 s per
active shield step with 12 workers, 0.3 s per plain step; ad-hoc single-process timing 0.17 s per solver
call; arm C needed 4600-4800 s per run, 0.46 s per step): per run about 4700 s + p * 10 000 * (1.2 to 2.1) s
for a share p of shifted steps; p unknown. p = 0.1: 1.8 h, p = 0.4: 3.2 h, p = 1: 5.8 h (upper bound).
Five runs in parallel on 16 cores; expected 2-3.5 h.

    python scripts/analyze_sh2_gate.py
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
from twinlink_ml.rl.env import _denormalise, _normalise
from twinlink_ml.rl.evaluation import moisture_margin_at_oracle_binding_points
from twinlink_ml.rl.lagrange import LagrangeConfig
from twinlink_ml.rl.shield import ShieldedLagrangeBanditEnv

RUNS = g3.RUNS
REPO_ROOT = g3.REPO_ROOT
SEEDS = g3.SEEDS
LIMIT = 0.15
MARGIN_M = g6.MARGIN_M
PREFIX_SH = "rl_bandit_lagrange_sh2_shield"
PREFIX_C = "rl_bandit_lagrange_s06_l015_lam05"
PENALTY_E = g6.PENALTY[0.15]
WINDOW = 1_000


def _model(prefix: str, seed: int) -> Any:
    from stable_baselines3 import SAC

    record = g3._final_record(prefix, seed)
    return SAC.load(REPO_ROOT / record["params"]["checkpoint/path"], device="cpu")


def _evaluate(prefix: str, *, shielded: bool) -> dict[int, dict[str, float]]:
    env = OperatingPointBanditEnv(BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT))
    points = load_test_points(g6.REFERENCE_015)
    shield_env = None
    if shielded:
        shield_env = ShieldedLagrangeBanditEnv(
            LagrangeConfig(
                margin_kg_per_kg=MARGIN_M,
                initial_multiplier_kw=0.5,
                step_size_kw=0.2,
                update_interval_steps=250,
            ),
            BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT),
        )
    out: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        model = _model(prefix, seed)

        def policy(observation: np.ndarray, model: Any = model) -> np.ndarray:
            action, _ = model.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float64)
            if shield_env is None:
                return action
            inlet, gas = shield_env._inlet_bound, shield_env._gas_bound
            # the observation is float32: take the exact context of the nearest test point
            exact = min(
                points,
                key=lambda p: abs(p.ambient_temperature_celsius - float(observation[0]))
                + abs(p.ambient_relative_humidity - float(observation[1])),
            )
            record, _ = shield_env.shield_action(
                _denormalise(float(action[0]), inlet.minimum, inlet.maximum),
                _denormalise(float(action[1]), gas.minimum, gas.maximum),
                exact.ambient_temperature_celsius,
                exact.ambient_relative_humidity,
            )
            return np.array([_normalise(record["t_exec"], inlet.minimum, inlet.maximum), action[1]])

        metrics = flat_metrics(evaluate_policy(env, policy, points))
        margin, n_binding = moisture_margin_at_oracle_binding_points(env, policy, points)
        metrics["moisture_margin_binding_median_kg_per_kg"] = margin
        metrics["moisture_margin_binding_n"] = float(n_binding)
        out[seed] = metrics
    return out


def _col(arm: dict[int, dict[str, float]], key: str) -> list[float]:
    return [arm[s][key] for s in SEEDS]


def _arm_summary(arm: dict[int, dict[str, float]]) -> dict[str, Any]:
    return {
        "feasibility_rate": g3._summary(_col(arm, "feasibility_rate")),
        "pooled_closed_regret_fraction": g3._summary(_col(arm, "pooled_closed_regret_fraction")),
        "distance_to_limit_binding_kg_per_kg": g3._summary(
            _col(arm, "moisture_margin_binding_median_kg_per_kg")
        ),
        "distance_in_units_of_m": g3._summary(
            [v / MARGIN_M for v in _col(arm, "moisture_margin_binding_median_kg_per_kg")]
        ),
        "n_binding_contexts": _col(arm, "moisture_margin_binding_n"),
    }


def _shield_engagement(prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in SEEDS:
        path = RUNS / f"{prefix}_trace_seed{seed}.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        active = np.array([r["shield_active"] for r in rows], dtype=bool)
        failed = np.array([r["solve_failed"] for r in rows], dtype=bool)
        saturated = np.array([r["shield_saturated"] for r in rows], dtype=bool)
        seconds = np.array([r["env_step_seconds"] for r in rows])
        n_solves = np.array([r["shield_n_solves"] for r in rows])
        shift = np.array([r["executed_t_in_celsius"] - r["proposed_t_in_celsius"] for r in rows])[
            active
        ]
        g = np.array([r["g_tilde"] for r in rows])
        windows = [float(active[i : i + WINDOW].mean()) for i in range(0, len(rows), WINDOW)]
        records = [
            json.loads(x)
            for x in (RUNS / f"{prefix}_seed{seed}.jsonl").read_text(encoding="utf-8").splitlines()
            if x
        ]
        out[str(seed)] = {
            "n_steps": len(rows),
            "share_shifted_steps": float(active.mean()),
            "share_shifted_per_window_of_1000": windows,
            "n_saturated": int(saturated.sum()),
            "n_solver_failed_proposals": int(failed.sum()),
            "n_steps_g_tilde_positive_at_executed": int(((g > 0.0) & ~failed).sum()),
            "median_shift_k_when_active": float(np.median(shift)) if shift.size else None,
            "p90_shift_k_when_active": float(np.percentile(shift, 90)) if shift.size else None,
            "env_step_seconds_median": float(np.median(seconds)),
            "env_step_seconds_mean": float(seconds.mean()),
            "env_step_seconds_p90": float(np.percentile(seconds, 90)),
            "solver_calls_per_step_mean": float(n_solves.mean()),
            "wall_s_10000": float(
                next(r for r in records if r["params"]["n_env_steps"] == 10_000)["params"][
                    "wall_s_since_training_start"
                ]
            ),
        }
    return out


def main() -> int:
    require_clean_tree()
    nu = g3._nu_eff()
    penalty = g6._evaluate_arm(PENALTY_E, 0.15)
    sh = _evaluate(PREFIX_SH, shielded=False)
    sh_shielded = _evaluate(PREFIX_SH, shielded=True)
    c_arm = _evaluate(PREFIX_C, shielded=False)
    c_shielded = _evaluate(PREFIX_C, shielded=True)

    gate = g6._gate(sh, penalty, nu)
    dist_sh = _col(sh, "moisture_margin_binding_median_kg_per_kg")
    dist_c = _col(c_arm, "moisture_margin_binding_median_kg_per_kg")
    pooled_sh = _col(sh, "pooled_closed_regret_fraction")
    pooled_c = _col(c_arm, "pooled_closed_regret_fraction")
    payload = {
        "_provenance": provenance(),
        "test_set": "25 Befund-52 grid points; oracle and fixed point of the limit 0.15",
        "evaluation_step": g3.EVALUATION_STEP,
        "seeds": list(SEEDS),
        "m_kg_per_kg": MARGIN_M,
        "verdict_evaluates": "the policy alone (no shield at evaluation), plain env, true limit 0.15",
        "SH_policy_alone": _arm_summary(sh),
        "gate": gate,
        "consequence": (
            "PASS: RL enters the multi-plant stage as a comparison"
            if gate["PASS"]
            else "no PASS: RL stays a diagnostic finding; no further RL attempt without consultation"
        ),
        "not_part_of_verdict": {
            "SH_shielded_policy": _arm_summary(sh_shielded),
            "SH_shielded_policy_gate_like": g6._gate(sh_shielded, penalty, nu),
            "C_policy_alone": _arm_summary(c_arm),
            "C_shielded_policy": _arm_summary(c_shielded),
            "SH_minus_C_distance": {
                "difference_C_minus_SH": float(np.median(dist_c) - np.median(dist_sh)),
                "two_sample_threshold": g3.two_sample_threshold(dist_c, dist_sh, nu),
            },
            "SH_minus_C_pooled": {
                "difference_SH_minus_C": float(np.median(pooled_sh) - np.median(pooled_c)),
                "two_sample_threshold": g3.two_sample_threshold(pooled_sh, pooled_c, nu),
            },
            "penalty_E_policy_alone": _arm_summary(penalty),
            "shield_engagement": _shield_engagement(PREFIX_SH),
            "n_active_lambda_series_of_125": g6._active_series(PREFIX_SH),
            "steps_with_g_tilde_positive_per_seed": g6._positive_g_steps(PREFIX_SH),
            "lambda_per_cell": g3.lambda_cells(PREFIX_SH),
            "trace_summary": g3._trace_summary(PREFIX_SH),
        },
    }
    write_artifact("sh2_gate", payload)
    print(json.dumps(payload["SH_policy_alone"], indent=1))
    print(json.dumps(gate, indent=1))
    print(payload["consequence"])
    print(json.dumps(payload["not_part_of_verdict"]["SH_shielded_policy"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
