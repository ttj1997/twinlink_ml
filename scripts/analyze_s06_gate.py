r"""S0.6: 2 x 2 design (quality limit x initial multiplier) plus the penalty arm at limit 0.15.

Pure evaluation of the records and checkpoints of the runs listed in ``ARMS`` (seeds 0-4,
step 10 000, deterministic policy). THE GATE AND THE READINGS ARE FIXED HERE before any S0.6
run; they are not to be changed after the runs have started.

Arms (all SAC default hyperparameters, 10 000 steps, m = 0.005961819615175404 kg/kg, 5 x 5
context cells, eta 0.2 kW, update every 250 steps; the Lagrange arms differ only in the
quality limit and in lambda_0):

    A  limit 0.20, lambda_0 0.5   (S0.3, exists)
    B  limit 0.20, lambda_0 0   (new)
    C  limit 0.15, lambda_0 0.5   (new)
    D  limit 0.15, lambda_0 0   (new)
    E  penalty arm at limit 0.15 (new; solver-failure value by the Befund 62 rule for this limit)

The drying kinetics keep ``critical_moisture = 0.20`` in every arm (S0.4: solver validity
boundary). Test set (always named): the 25 Befund 52 grid points, with the oracle and the fixed
operating point of the arm's own limit: Befund 52 for limit 0.20, ``s05_limit015_oracle_and_fixed``
for limit 0.15.

Gate G1-G3 per Lagrange arm, constants as in S0.3 (``analyze_s03_gate``), judged against the
arm's own limit, its own oracle/fixed reference and the penalty arm of the same limit (A, B
against the S0.3 penalty arm; C, D against E):

  G1 feasibility: median >= 0.96, no seed below 0.92.
  G2 pooled closed-regret fraction: median > 0 and above the one-sample t threshold
     (nu_eff 3.7623, sigma_hat = R / d2(5)).
  G3 distance of the moisture to the true limit at the oracle-binding contexts: median
     <= 0.02 kg/kg and the difference to the penalty arm above the SX1 two-sample threshold.

Readings, fixed before the runs (they may hold together):

  R1  "the limit 0.15 activates lambda and brings the agent close to the tightened limit":
      for C or D, at least 25 of the 125 cell-seed lambda series are not inactive (lambda > 0
      somewhere in the last 10 of 40 updates) AND the median distance at the binding contexts
      is <= 0.02 kg/kg.
  R2  "lambda_0 = 0 changes the behaviour clearly": for limit 0.20 (A vs B) or limit 0.15 (C vs D)
      the difference of the arms in the distance at the binding contexts, or in the pooled
      closed-regret fraction, exceeds its SX1 two-sample threshold.
  R3  "neither has an effect": neither R1 nor R2.

Also reported: lambda history per cell (categories of S0.3), the distance in units of m, the number
of training steps with g_tilde > 0 per arm, the differences A-C and B-D (effect of the limit) with
thresholds.

    python scripts/analyze_s06_gate.py
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
MARGIN_M = 0.005961819615175404
REFERENCE_015 = RUNS / "s05_limit015_oracle_and_fixed.json"

# --- fixed before the runs -------------------------------------------------------------------
R1_MIN_ACTIVE_SERIES = 25  # of 125 cell-seed series
R1_MAX_DISTANCE_KG_PER_KG = 0.02
# -------------------------------------------------------------------------------------------

ARMS: dict[str, dict[str, Any]] = {
    "A_L020_lam05": {"prefix": "rl_bandit_lagrange_s03", "limit": None, "lagrange": True},
    "B_L020_lam0": {"prefix": "rl_bandit_lagrange_s06_l020_lam0", "limit": None, "lagrange": True},
    "C_L015_lam05": {
        "prefix": "rl_bandit_lagrange_s06_l015_lam05",
        "limit": 0.15,
        "lagrange": True,
    },
    "D_L015_lam0": {"prefix": "rl_bandit_lagrange_s06_l015_lam0", "limit": 0.15, "lagrange": True},
}
PENALTY = {
    0.20: "rl_bandit_sac_learning_curve_s03_penalty",
    0.15: "rl_bandit_sac_learning_curve_s06_penalty_l015",
}


def _env_and_points(limit: float | None) -> tuple[OperatingPointBanditEnv, Any]:
    env = OperatingPointBanditEnv(BanditEnvConfig(product_moisture_limit_kg_per_kg=limit))
    points = load_test_points() if limit is None else load_test_points(REFERENCE_015)
    return env, points


def _evaluate_arm(prefix: str, limit: float | None) -> dict[int, dict[str, float]]:
    from stable_baselines3 import SAC

    env, points = _env_and_points(limit)
    out: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        record = g3._final_record(prefix, seed)
        model = SAC.load(REPO_ROOT / record["params"]["checkpoint/path"])

        def policy(observation: np.ndarray, model: Any = model) -> np.ndarray:
            action, _ = model.predict(observation, deterministic=True)
            return np.asarray(action, dtype=np.float64)

        metrics = flat_metrics(evaluate_policy(env, policy, points))
        margin, n_binding = moisture_margin_at_oracle_binding_points(env, policy, points)
        metrics["moisture_margin_binding_median_kg_per_kg"] = margin
        metrics["moisture_margin_binding_n"] = float(n_binding)
        out[seed] = metrics
    return out


def _col(arm: dict[int, dict[str, float]], key: str) -> list[float]:
    return [arm[s][key] for s in SEEDS]


def _gate(
    lagr: dict[int, dict[str, float]], pen: dict[int, dict[str, float]], nu: float
) -> dict[str, Any]:
    feas = _col(lagr, "feasibility_rate")
    pooled = _col(lagr, "pooled_closed_regret_fraction")
    dist_l = _col(lagr, "moisture_margin_binding_median_kg_per_kg")
    dist_p = _col(pen, "moisture_margin_binding_median_kg_per_kg")
    g1 = (
        float(np.median(feas)) >= g3.G1_FEASIBILITY_MEDIAN_MIN
        and min(feas) >= g3.G1_FEASIBILITY_SEED_MIN
    )
    g2_thr = g3.one_sample_threshold(pooled, nu)
    g2 = float(np.median(pooled)) > 0.0 and float(np.median(pooled)) > g2_thr
    two = g3.two_sample_threshold(dist_l, dist_p, nu)
    diff = float(np.median(dist_p)) - float(np.median(dist_l))
    g3_ok = (
        float(np.median(dist_l)) <= g3.G3_MARGIN_MEDIAN_MAX_KG_PER_KG and diff > two["threshold"]
    )
    return {
        "G1": g1,
        "G2": g2,
        "G2_threshold": g2_thr,
        "G3": g3_ok,
        "G3_difference_penalty_minus_lagrange": diff,
        "G3_two_sample_threshold": two,
        "PASS": bool(g1 and g2 and g3_ok),
    }


def _active_series(prefix: str) -> int:
    cells = g3.lambda_cells(prefix)
    return sum(1 for seed in cells.values() for c in seed["cells"] if c["category"] != "inactive")


def _positive_g_steps(prefix: str) -> list[int]:
    counts = []
    for seed in SEEDS:
        path = RUNS / f"{prefix}_trace_seed{seed}.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        counts.append(sum(1 for r in rows if not r["solve_failed"] and r["g_tilde"] > 0.0))
    return counts


def main() -> int:
    require_clean_tree()
    nu = g3._nu_eff()
    penalty = {
        limit: _evaluate_arm(prefix, None if limit == 0.20 else limit)
        for limit, prefix in PENALTY.items()
    }
    results: dict[str, Any] = {}
    evaluated: dict[str, dict[int, dict[str, float]]] = {}
    for name, spec in ARMS.items():
        arm = _evaluate_arm(spec["prefix"], spec["limit"])
        evaluated[name] = arm
        limit = 0.20 if spec["limit"] is None else spec["limit"]
        results[name] = {
            "limit_kg_per_kg": limit,
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
            "gate": _gate(arm, penalty[limit], nu),
            "n_active_lambda_series_of_125": _active_series(spec["prefix"]),
            "steps_with_g_tilde_positive_per_seed": _positive_g_steps(spec["prefix"]),
            "lambda_per_cell": g3.lambda_cells(spec["prefix"]),
            "trace_summary": g3._trace_summary(spec["prefix"]),
        }
    results["E_penalty_L015"] = {
        "limit_kg_per_kg": 0.15,
        "feasibility_rate": g3._summary(_col(penalty[0.15], "feasibility_rate")),
        "pooled_closed_regret_fraction": g3._summary(
            _col(penalty[0.15], "pooled_closed_regret_fraction")
        ),
        "distance_to_limit_binding_kg_per_kg": g3._summary(
            _col(penalty[0.15], "moisture_margin_binding_median_kg_per_kg")
        ),
        "distance_in_units_of_m": g3._summary(
            [v / MARGIN_M for v in _col(penalty[0.15], "moisture_margin_binding_median_kg_per_kg")]
        ),
        "n_binding_contexts": _col(penalty[0.15], "moisture_margin_binding_n"),
    }

    def dist(name: str) -> list[float]:
        return _col(evaluated[name], "moisture_margin_binding_median_kg_per_kg")

    def pooled(name: str) -> list[float]:
        return _col(evaluated[name], "pooled_closed_regret_fraction")

    def compare(a: str, b: str) -> dict[str, Any]:
        return {
            "distance_difference_a_minus_b": float(np.median(dist(a)) - np.median(dist(b))),
            "distance_threshold": g3.two_sample_threshold(dist(a), dist(b), nu),
            "pooled_difference_a_minus_b": float(np.median(pooled(a)) - np.median(pooled(b))),
            "pooled_threshold": g3.two_sample_threshold(pooled(a), pooled(b), nu),
        }

    comparisons = {
        "lambda0_effect_limit020_B_vs_A": compare("B_L020_lam0", "A_L020_lam05"),
        "lambda0_effect_limit015_D_vs_C": compare("D_L015_lam0", "C_L015_lam05"),
        "limit_effect_lambda05_C_vs_A": compare("C_L015_lam05", "A_L020_lam05"),
        "limit_effect_lambda0_D_vs_B": compare("D_L015_lam0", "B_L020_lam0"),
    }

    def resolved(c: dict[str, Any]) -> bool:
        return bool(
            abs(c["distance_difference_a_minus_b"]) > c["distance_threshold"]["threshold"]
            or abs(c["pooled_difference_a_minus_b"]) > c["pooled_threshold"]["threshold"]
        )

    r1 = any(
        results[n]["n_active_lambda_series_of_125"] >= R1_MIN_ACTIVE_SERIES
        and float(np.median(dist(n))) <= R1_MAX_DISTANCE_KG_PER_KG
        for n in ("C_L015_lam05", "D_L015_lam0")
    )
    r2 = resolved(comparisons["lambda0_effect_limit020_B_vs_A"]) or resolved(
        comparisons["lambda0_effect_limit015_D_vs_C"]
    )
    readings = {
        "R1_limit_activates_lambda_and_reaches_tightened_limit": r1,
        "R2_lambda0_changes_behaviour": r2,
        "R3_neither": not (r1 or r2),
    }
    payload = {
        "_provenance": provenance(),
        "test_set": "25 Befund-52 grid points; oracle and fixed point of the arm's own limit",
        "evaluation_step": g3.EVALUATION_STEP,
        "seeds": list(SEEDS),
        "m_kg_per_kg": MARGIN_M,
        "constants_fixed_before_runs": {
            "G1_median_min": g3.G1_FEASIBILITY_MEDIAN_MIN,
            "G1_seed_min": g3.G1_FEASIBILITY_SEED_MIN,
            "G3_distance_max_kg_per_kg": g3.G3_MARGIN_MEDIAN_MAX_KG_PER_KG,
            "R1_min_active_series": R1_MIN_ACTIVE_SERIES,
            "R1_max_distance_kg_per_kg": R1_MAX_DISTANCE_KG_PER_KG,
        },
        "arms": results,
        "comparisons": comparisons,
        "readings": readings,
    }
    write_artifact("s06_gate", payload)
    print(json.dumps({k: v["gate"] for k, v in results.items() if "gate" in v}, indent=1))
    print(json.dumps(readings, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
