r"""SH3: is the lower bound of the gas velocity the main cause of the negative pooled closed regret?

Evaluation of existing step-10 000 checkpoints, NO training. Arms (all SAC, environment of arm C:
quality limit 0.15, Lagrange term, m = 0.005961819615175404): C (``rl_bandit_lagrange_s06_l015_lam05``),
G (``rl_bandit_lagrange_s08_sac_lowent``, target entropy -6), SH (``rl_bandit_lagrange_sh2_shield``,
policy alone, shield not applied). Seeds 0-4, the 25 Befund 52 grid points (test set), oracle and
fixed operating point of the limit 0.15 (``s05_limit015_oracle_and_fixed``), deterministic policy,
plain environment, true limit 0.15.

Hypothesis (project lead): the oracle and the fixed operating point lie at the lower bound of the gas
velocity (0.80 m/s) at many points; a SAC policy reaches a bound only asymptotically (tanh squashing),
and even a small excess above the bound costs energy of the order of the whole regret gap (0.105 kW
at the limit 0.15).

  a) gas velocity of the agent per test point against the oracle and the fixed point;
  b) the energy at the agent's action and at the SAME action with the gas velocity set to the lower
     bound (inlet temperature unchanged), if that action is feasible (solved, all constraints met);
     otherwise the original action is kept for that point (and counted);
  c) the pooled closed-regret fraction with this correction (same pooled definition as the gate:
     points that are infeasible or unsolved contribute neither to numerator nor denominator).

READING, fixed before the run. Per arm, with the median over seeds of the pooled fraction before
(P0) and after (P1) the correction:
  the bound is SUPPORTED as main cause for the arm iff  P1 > 0   OR   P1 - P0 > 0.5 * (0 - P0)
  (the correction closes more than half of the gap from P0 to the fixed point, which is 0)  [P0 < 0].
Overall: SUPPORTED iff all three arms are supported, PARTLY iff one or two, NOT SUPPORTED iff none.
Also reported, not part of the rule: whether P1 exceeds the one-sample t threshold of the SX1 procedure
(nu_eff 3.7623), per point the energy change by the correction, the sum of these changes against the
sum of the regret gaps (E_fixed - E_oracle), the number of points where the correction was not
applicable, and the number of test points where the oracle / the fixed point sit at the bound.

    python scripts/probe_sh3_gas_velocity_bound.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

import analyze_s03_gate as g3
import analyze_s06_gate as g6
from _common import provenance, require_clean_tree, write_artifact

from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv, load_test_points
from twinlink_ml.rl.env import _denormalise
from twinlink_ml.rl.evaluation import _group_metrics

REPO_ROOT = g3.REPO_ROOT
SEEDS = g3.SEEDS
LIMIT = 0.15
ARMS = {
    "C_SAC_default": "rl_bandit_lagrange_s06_l015_lam05",
    "G_SAC_low_entropy": "rl_bandit_lagrange_s08_sac_lowent",
    "SH_SAC_shield_policy_alone": "rl_bandit_lagrange_sh2_shield",
}
AT_BOUND_TOLERANCE_M_PER_S = 0.01  # 1 % of the span, as is_boundary_optimum
HALF = 0.5


def _solved(info: dict[str, Any]) -> tuple[bool, float | None]:
    if info.get("solve_failed"):
        return False, None
    return bool(info["feasible"]), float(info["energy_target"])


def _evaluate_seed(model: Any, env: OperatingPointBanditEnv, points: Any) -> dict[str, Any]:
    gas_min = env._gas_bound.minimum
    rows = []
    entries_before: list[tuple[bool, float | None, float, float]] = []
    entries_after: list[tuple[bool, float | None, float, float]] = []
    for point in points:
        observation = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity], dtype=np.float32
        )
        action, _ = model.predict(observation, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        t_in = _denormalise(float(action[0]), env._inlet_bound.minimum, env._inlet_bound.maximum)
        u_gas = _denormalise(float(action[1]), gas_min, env._gas_bound.maximum)

        def evaluate(u: float, point: Any = point, t_in: float = t_in) -> dict[str, Any]:
            return env.evaluate_operating_point(
                inlet_temperature_celsius=t_in,
                gas_velocity_m_per_s=u,
                ambient_temperature_celsius=point.ambient_temperature_celsius,
                ambient_relative_humidity=point.ambient_relative_humidity,
            )[1]

        feasible0, energy0 = _solved(evaluate(u_gas))
        info1 = evaluate(gas_min)
        feasible1, energy1 = _solved(info1)
        applicable = bool(feasible1 and energy1 is not None)
        entries_before.append((feasible0, energy0, point.energy_fixed_kw, point.energy_oracle_kw))
        if applicable:
            entries_after.append(
                (feasible1, energy1, point.energy_fixed_kw, point.energy_oracle_kw)
            )
        else:
            entries_after.append(
                (feasible0, energy0, point.energy_fixed_kw, point.energy_oracle_kw)
            )
        rows.append(
            {
                "t_amb": point.ambient_temperature_celsius,
                "phi": point.ambient_relative_humidity,
                "agent_t_in": t_in,
                "agent_u_gas": u_gas,
                "agent_excess_over_bound_m_per_s": u_gas - gas_min,
                "oracle_u_gas": point.oracle_gas_velocity_m_per_s,
                "oracle_t_in": point.oracle_inlet_temperature_celsius,
                "energy_oracle_kw": point.energy_oracle_kw,
                "energy_fixed_kw": point.energy_fixed_kw,
                "agent_feasible": feasible0,
                "energy_agent_kw": energy0,
                "correction_applicable": applicable,
                "energy_corrected_kw": energy1 if applicable else None,
                "energy_change_by_correction_kw": (energy1 - energy0)
                if applicable and energy0 is not None
                else None,
            }
        )
    feas0, pooled0 = _group_metrics(entries_before)
    feas1, pooled1 = _group_metrics(entries_after)
    return {
        "feasibility_before": feas0,
        "feasibility_after": feas1,
        "pooled_before": pooled0,
        "pooled_after": pooled1,
        "n_correction_not_applicable": sum(not r["correction_applicable"] for r in rows),
        "rows": rows,
    }


def main() -> int:
    require_clean_tree()
    warnings.filterwarnings("ignore")
    from stable_baselines3 import SAC

    nu = g3._nu_eff()
    env = OperatingPointBanditEnv(BanditEnvConfig(product_moisture_limit_kg_per_kg=LIMIT))
    points = load_test_points(g6.REFERENCE_015)
    gas_min = env._gas_bound.minimum
    fixed_u = json.loads(g6.REFERENCE_015.read_text(encoding="utf-8"))["fixed_point"][
        "gas_velocity_m_per_s"
    ]
    reference = {
        "gas_velocity_lower_bound_m_per_s": gas_min,
        "fixed_point_gas_velocity_m_per_s": fixed_u,
        "n_points": len(points),
        "n_oracle_at_gas_lower_bound": sum(
            abs(p.oracle_gas_velocity_m_per_s - gas_min) < AT_BOUND_TOLERANCE_M_PER_S
            for p in points
        ),
        "regret_gap_sum_kw": float(sum(p.energy_fixed_kw - p.energy_oracle_kw for p in points)),
        "regret_gap_median_kw": float(
            np.median([p.energy_fixed_kw - p.energy_oracle_kw for p in points])
        ),
    }
    arms: dict[str, Any] = {}
    for name, prefix in ARMS.items():
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            record = g3._final_record(prefix, seed)
            model = SAC.load(REPO_ROOT / record["params"]["checkpoint/path"], device="cpu")
            per_seed[str(seed)] = _evaluate_seed(model, env, points)
            print(
                name,
                seed,
                round(per_seed[str(seed)]["pooled_before"], 3),
                round(per_seed[str(seed)]["pooled_after"], 3),
                flush=True,
            )
        before = [per_seed[str(s)]["pooled_before"] for s in SEEDS]
        after = [per_seed[str(s)]["pooled_after"] for s in SEEDS]
        p0, p1 = float(np.median(before)), float(np.median(after))
        changes = [
            r["energy_change_by_correction_kw"]
            for s in SEEDS
            for r in per_seed[str(s)]["rows"]
            if r["energy_change_by_correction_kw"] is not None
        ]
        excess = [
            r["agent_excess_over_bound_m_per_s"] for s in SEEDS for r in per_seed[str(s)]["rows"]
        ]
        gap_sum = reference["regret_gap_sum_kw"]
        per_seed_change_sum = [
            float(
                sum(
                    r["energy_change_by_correction_kw"]
                    for r in per_seed[str(s)]["rows"]
                    if r["energy_change_by_correction_kw"] is not None
                )
            )
            for s in SEEDS
        ]
        supported = bool(p1 > 0.0 or (p0 < 0.0 and (p1 - p0) > HALF * (0.0 - p0)))
        arms[name] = {
            "pooled_before": g3._summary(before),
            "pooled_after": g3._summary(after),
            "feasibility_before": g3._summary(
                [per_seed[str(s)]["feasibility_before"] for s in SEEDS]
            ),
            "feasibility_after": g3._summary(
                [per_seed[str(s)]["feasibility_after"] for s in SEEDS]
            ),
            "pooled_after_one_sample_threshold": g3.one_sample_threshold(after, nu),
            "pooled_after_exceeds_threshold": bool(p1 > g3.one_sample_threshold(after, nu)),
            "gap_closed_by_correction_share": (p1 - p0) / (0.0 - p0) if p0 < 0.0 else None,
            "supported": supported,
            "n_correction_not_applicable_per_seed": [
                per_seed[str(s)]["n_correction_not_applicable"] for s in SEEDS
            ],
            "agent_excess_over_bound_m_per_s": {
                "median": float(np.median(excess)),
                "p90": float(np.percentile(excess, 90)),
                "share_above_0.01": float(np.mean(np.array(excess) > AT_BOUND_TOLERANCE_M_PER_S)),
            },
            "energy_change_by_correction_kw": {
                "median_per_point": float(np.median(changes)),
                "mean_per_point": float(np.mean(changes)),
                "sum_per_seed": per_seed_change_sum,
                "sum_median_over_seeds_against_regret_gap_sum": float(
                    np.median(per_seed_change_sum) / gap_sum
                ),
            },
            "per_seed": per_seed,
        }
    n_sup = sum(a["supported"] for a in arms.values())
    verdict = "SUPPORTED" if n_sup == len(arms) else ("PARTLY" if n_sup else "NOT SUPPORTED")
    payload = {
        "_provenance": provenance(),
        "test_set": "25 Befund-52 grid points; oracle and fixed point of the limit 0.15",
        "evaluation_step": g3.EVALUATION_STEP,
        "seeds": list(SEEDS),
        "reference": reference,
        "arms": arms,
        "n_arms_supported": n_sup,
        "gas_velocity_bound_as_main_cause": verdict,
    }
    write_artifact("sh3_gas_velocity_bound", payload)
    for name, a in arms.items():
        print(
            name,
            "pooled",
            round(a["pooled_before"]["median"], 3),
            "->",
            round(a["pooled_after"]["median"], 3),
            "supported",
            a["supported"],
        )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
