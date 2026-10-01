r"""S0.3 gate: Lagrangian arm against the penalty arm (FA2 configuration), same five seeds.

Pure evaluation of the records and checkpoints written by ``run_rl_bandit_lagrange.py`` and
``run_rl_bandit_learning_curve.py`` (penalty arm, ``--solve-failure-penalty-kw
155.57896052362634``). THE GATE IS FIXED HERE, before any training run. Nothing below may be
changed after the runs have started; a different reading is reported as an addendum.

Test set (always named): the 25 Befund 52 grid points (ambient temperature x relative
humidity), disjoint from the continuous training draws. Evaluation point: step 10 000, the
last checkpoint, deterministic policy. Seeds 0-4 in both arms.

Gate criteria (all three must hold; each is reported separately)
------------------------------------------------------------------
G1 Feasibility against the TRUE limit and all other constraints: median over the five seeds
   >= 0.96 (24 of 25 points) AND no seed below 0.92 (23 of 25).
G2 Pooled closed-regret fraction (XA1 b, feasible points only, raw energy): median over the
   seeds > 0, i.e. better than the fixed operating point, AND resolved: the median exceeds
   ``t(nu_eff, 0.975) * sigma_hat / sqrt(n)`` with ``sigma_hat = R / d2(5)`` of the five seed
   values, ``nu_eff = 3.7623`` (SX1, Befund 67, artifact ``sx1_resolution_t.json``). The
   reference value 0 is the fixed point by construction and has no spread.
G3 Distance of the product moisture to the true limit at the contexts where the Befund 52
   oracle binds the moisture limit (17 of 25 in Befund 62): median over the seeds of the
   per-seed median <= 0.02 kg/kg (at 0.14 kg/kg in the penalty arm, GA1), AND the difference
   to the penalty arm is resolved with the SX1 two-sample threshold
   ``t(nu_W, 0.975) * sqrt((s_c^2 + s_r^2) / n)``, Welch-Satterthwaite ``nu_W`` with
   ``nu_c = nu_r = nu_eff``.

Also reported, not part of the gate: differences to the penalty arm in feasibility and pooled
fraction with the same two-sample threshold; the reproduction of the committed FA2 values
(seeds 0-2) by the rerun penalty arm; lambda history and loss traces summary.

    python scripts/analyze_s03_gate.py --lagrange-prefix rl_bandit_lagrange_s03 \
        --penalty-prefix rl_bandit_sac_learning_curve_s03_penalty
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact

from twinlink_ml.evaluation.resolution import _HARTLEY_D2
from twinlink_ml.rl import OperatingPointBanditEnv, evaluate_policy, flat_metrics
from twinlink_ml.rl.evaluation import moisture_margin_at_oracle_binding_points

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "artifacts" / "runs"
SEEDS = (0, 1, 2, 3, 4)
EVALUATION_STEP = 10_000
ALPHA_TWO_SIDED = 0.05

# --- gate constants, fixed before any run -------------------------------------------------
G1_FEASIBILITY_MEDIAN_MIN = 0.96
G1_FEASIBILITY_SEED_MIN = 0.92
G2_POOLED_MEDIAN_MIN_EXCLUSIVE = 0.0
G3_MARGIN_MEDIAN_MAX_KG_PER_KG = 0.02
# -------------------------------------------------------------------------------------------

COMMITTED_FA2 = {  # Befund 62 FA3: step 10000, pooled closed-regret fraction (committed)
    0: -1.9984602459667704,
    1: -2.7973488622832794,
    2: -3.012366063063362,
}
COMMITTED_GA1_MARGIN = {0: 0.1299, 1: 0.1454, 2: 0.1395}  # Befund 62 GA1, rounded


def _nu_eff() -> float:
    data = json.loads((RUNS / "sx1_resolution_t.json").read_text(encoding="utf-8"))
    assert data["dof_of_range_estimator"]["n"] == 5
    return float(data["dof_of_range_estimator"]["nu_eff_range"])


def _sigma(values: list[float]) -> float:
    return (max(values) - min(values)) / _HARTLEY_D2[len(values)]


def one_sample_threshold(values: list[float], nu: float) -> float:
    n = len(values)
    return float(stats.t.ppf(1.0 - ALPHA_TWO_SIDED / 2.0, nu) * _sigma(values) / math.sqrt(n))


def two_sample_threshold(a: list[float], b: list[float], nu: float) -> dict[str, float]:
    n = len(a)
    va, vb = _sigma(a) ** 2 / n, _sigma(b) ** 2 / n
    nu_welch = (va + vb) ** 2 / (va**2 / nu + vb**2 / nu) if va + vb > 0 else math.inf
    t_crit = float(stats.t.ppf(1.0 - ALPHA_TWO_SIDED / 2.0, nu_welch))
    return {"nu_welch": nu_welch, "t": t_crit, "threshold": t_crit * math.sqrt(va + vb)}


def _final_record(prefix: str, seed: int) -> dict[str, Any]:
    path = RUNS / f"{prefix}_seed{seed}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row["params"]["n_env_steps"] == EVALUATION_STEP:
            return row
    raise SystemExit(f"no step-{EVALUATION_STEP} record in {path}")


def _evaluate_checkpoint(record: dict[str, Any]) -> dict[str, float]:
    from stable_baselines3 import SAC

    model = SAC.load(REPO_ROOT / record["params"]["checkpoint/path"])

    def policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    env = OperatingPointBanditEnv()
    metrics = flat_metrics(evaluate_policy(env, policy))
    margin, n_binding = moisture_margin_at_oracle_binding_points(env, policy)
    metrics["moisture_margin_binding_median_kg_per_kg"] = margin
    metrics["moisture_margin_binding_n"] = float(n_binding)
    return metrics


def _arm(prefix: str) -> dict[int, dict[str, float]]:
    return {seed: _evaluate_checkpoint(_final_record(prefix, seed)) for seed in SEEDS}


def _column(arm: dict[int, dict[str, float]], key: str) -> list[float]:
    return [arm[seed][key] for seed in SEEDS]


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "per_seed": values,
        "median": float(np.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _trace_summary(prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in SEEDS:
        lam_path = RUNS / f"{prefix}_lambda_seed{seed}.jsonl"
        trace_path = RUNS / f"{prefix}_trace_seed{seed}.jsonl"
        lam = [json.loads(x) for x in lam_path.read_text(encoding="utf-8").splitlines() if x]
        trace = [json.loads(x) for x in trace_path.read_text(encoding="utf-8").splitlines() if x]
        final = np.array(lam[-1]["multipliers_kw"])
        losses = [r["critic_loss"] for r in trace if np.isfinite(r["critic_loss"])]
        out[str(seed)] = {
            "n_updates": len(lam),
            "lambda_final_min_median_max_kw": [
                float(final.min()),
                float(np.median(final)),
                float(final.max()),
            ],
            "n_solve_failed_steps": int(sum(r["solve_failed"] for r in trace)),
            "critic_loss_first_last_median_of_100": [
                float(np.median(losses[:100])),
                float(np.median(losses[-100:])),
            ]
            if len(losses) >= 200
            else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lagrange-prefix", required=True)
    parser.add_argument("--penalty-prefix", required=True)
    parser.add_argument("--output-name", default="s03_gate")
    arguments = parser.parse_args()
    require_clean_tree()

    nu = _nu_eff()
    lagrange = _arm(arguments.lagrange_prefix)
    penalty = _arm(arguments.penalty_prefix)

    feas_l = _column(lagrange, "feasibility_rate")
    pooled_l = _column(lagrange, "pooled_closed_regret_fraction")
    margin_l = _column(lagrange, "moisture_margin_binding_median_kg_per_kg")
    feas_p = _column(penalty, "feasibility_rate")
    pooled_p = _column(penalty, "pooled_closed_regret_fraction")
    margin_p = _column(penalty, "moisture_margin_binding_median_kg_per_kg")

    g1 = (
        float(np.median(feas_l)) >= G1_FEASIBILITY_MEDIAN_MIN
        and min(feas_l) >= G1_FEASIBILITY_SEED_MIN
    )
    g2_threshold = one_sample_threshold(pooled_l, nu)
    g2 = (
        float(np.median(pooled_l)) > G2_POOLED_MEDIAN_MIN_EXCLUSIVE
        and float(np.median(pooled_l)) > g2_threshold
    )
    g3_two = two_sample_threshold(margin_l, margin_p, nu)
    g3_diff = float(np.median(margin_p)) - float(np.median(margin_l))
    g3 = (
        float(np.median(margin_l)) <= G3_MARGIN_MEDIAN_MAX_KG_PER_KG
        and g3_diff > g3_two["threshold"]
    )

    reproduction = {
        str(seed): {
            "pooled_rerun": pooled_p[SEEDS.index(seed)],
            "pooled_committed_fa2": COMMITTED_FA2[seed],
            "margin_rerun": margin_p[SEEDS.index(seed)],
            "margin_committed_ga1_rounded": COMMITTED_GA1_MARGIN[seed],
        }
        for seed in COMMITTED_FA2
    }
    payload: dict[str, Any] = {
        "provenance": provenance(),
        "test_set": "25 Befund-52 grid points (ambient temperature x relative humidity)",
        "evaluation_step": EVALUATION_STEP,
        "seeds": list(SEEDS),
        "nu_eff_range_n5": nu,
        "gate_constants": {
            "g1_feasibility_median_min": G1_FEASIBILITY_MEDIAN_MIN,
            "g1_feasibility_seed_min": G1_FEASIBILITY_SEED_MIN,
            "g2_pooled_median_min_exclusive": G2_POOLED_MEDIAN_MIN_EXCLUSIVE,
            "g3_margin_median_max_kg_per_kg": G3_MARGIN_MEDIAN_MAX_KG_PER_KG,
        },
        "lagrange": {
            "feasibility_rate": _summary(feas_l),
            "pooled_closed_regret_fraction": _summary(pooled_l),
            "moisture_margin_binding_median_kg_per_kg": _summary(margin_l),
            "n_binding_contexts": _column(lagrange, "moisture_margin_binding_n"),
        },
        "penalty": {
            "feasibility_rate": _summary(feas_p),
            "pooled_closed_regret_fraction": _summary(pooled_p),
            "moisture_margin_binding_median_kg_per_kg": _summary(margin_p),
        },
        "gate": {
            "G1_feasibility": g1,
            "G2_pooled_better_than_fixed": g2,
            "G2_threshold_one_sample": g2_threshold,
            "G3_margin": g3,
            "G3_difference_penalty_minus_lagrange_kg_per_kg": g3_diff,
            "G3_two_sample_threshold": g3_two,
            "PASS": bool(g1 and g2 and g3),
        },
        "vs_penalty_not_gate": {
            "feasibility_difference_lagrange_minus_penalty": float(np.median(feas_l))
            - float(np.median(feas_p)),
            "feasibility_threshold": two_sample_threshold(feas_l, feas_p, nu),
            "pooled_difference_lagrange_minus_penalty": float(np.median(pooled_l))
            - float(np.median(pooled_p)),
            "pooled_threshold": two_sample_threshold(pooled_l, pooled_p, nu),
        },
        "reproduction_of_committed_fa2": reproduction,
        "lagrange_trace_summary": _trace_summary(arguments.lagrange_prefix),
    }
    path = write_artifact(arguments.output_name, payload)
    print(json.dumps(payload["gate"], indent=1))
    print("written:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
