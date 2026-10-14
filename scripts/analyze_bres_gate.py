r"""Gate and readings of the bandit closure (residual action x separated moisture cost).

Pure evaluation of the run records of ``scripts/run_rl_bandit_residual.py`` (cells ``base``, ``sep``,
``res``, ``ressep``; SAC, seeds 0-4, checkpoints 1000 ... 30 000; deterministic policy, plain
environment, TRUE limit 0.15 = PLACEHOLDER, lower gas-velocity bound 0.80 m/s = PLACEHOLDER; test
set: the 25 Befund 52 grid points with the oracle and fixed operating point of the limit 0.15).
THE GATE AND THE READINGS ARE FIXED HERE before any run; they are not to be changed afterwards.

Gate, per cell and per budget (10 000 steps and 30 000 steps = checkpoints of the same run):

  F       feasibility 1.0: ``feasibility_rate`` equals 1.0 in EVERY seed (strict reading; the
          median alone is also reported).
  STAGE 1 F and the median over seeds of ``pooled_closed_regret_fraction`` > 0.
  STAGE 2 F and that median > 0.5.

  The distance of the product moisture to the true limit at the oracle-binding contexts is reported
  in units of the margin m = 0.005961819615175404 kg/kg (``distance_in_units_of_m``); no threshold,
  it is a reading (the tightened limit of the Lagrange term lies at 1 m).

  NOT part of the verdict, reported next to every stage: whether the median exceeds the one-sample
  t threshold of SX1 (``analyze_s03_gate.one_sample_threshold``, nu_eff from ``sx1_resolution_t``,
  sigma_hat = R / d2(5)), i.e. whether it is resolved at five seeds. The residual cells start at the
  fixed point with closed fraction 0 by construction, so a median "> 0" alone is cheap; the
  resolved flag says whether it is more than that. The headline names both.

Which measure carries the effect (pre-registered rule, on the median pooled closed fraction, per
budget; a contrast counts as "resolved better" if it is positive and exceeds the SX1 two-sample
threshold ``analyze_s03_gate.two_sample_threshold``):

  contrasts  R|S0 = res - base,   R|S1 = ressep - sep,   S|R0 = sep - base,   S|R1 = ressep - res
  R carries         both R contrasts resolved better, neither S contrast
  S carries         both S contrasts resolved better, neither R contrast
  both, independent all four resolved better
  both, combined    R|S1 and S|R1 resolved better, R|S0 and S|R0 not (the effect needs both)
  none              no contrast resolved better
  mixed             anything else (the four contrasts are reported)

Also reported: learning curves (median and range per checkpoint, primary result), lambda history per
context cell (categories of S0.3), masked steps per run and reason, training steps with g_tilde > 0,
steps with an infeasible stored action, the start value of the residual cells at step 0 (must be 0 up
to 1e-9; otherwise the run is flagged), the reachable ceiling of the residual box (the oracle action
clipped into +-s around the fixed point: feasibility and pooled closed fraction), the cell ``base``
at step 10 000 against arm C of S0.6 (same seeds and settings; reported, not a verdict) and the
wall time of the runs.

    tools/run_here.sh scripts/analyze_bres_gate.py
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
import run_rl_bandit_residual as rr
from _common import provenance, require_clean_tree, write_artifact

from twinlink_ml.rl import evaluate_policy, flat_metrics, load_test_points
from twinlink_ml.rl.residual import residual_to_action

RUNS = g3.RUNS
SEEDS = rr.SEEDS
CELLS = tuple(rr.CELLS)
BUDGETS = (10_000, 30_000)
MARGIN_M = rr.MARGIN_M
ARM_C_PREFIX = "rl_bandit_lagrange_s06_l015_lam05"
STAGE1_MIN_EXCLUSIVE = 0.0
STAGE2_MIN_EXCLUSIVE = 0.5
START_VALUE_ATOL = 1e-9


def _records(prefix: str, cell: str, seed: int) -> dict[int, dict[str, Any]]:
    path = rr.output_paths(cell, seed, prefix)["records"]
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return {int(r["params"]["n_env_steps"]): r for r in rows}


def _metric(records: dict[int, dict[str, Any]], step: int, key: str) -> float:
    return float(records[step]["metrics"][key])


def _cell_table(prefix: str, cell: str, step: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"feasibility": [], "pooled": [], "distance_m": []}
    for seed in SEEDS:
        records = _records(prefix, cell, seed)
        out["feasibility"].append(_metric(records, step, "feasibility_rate"))
        out["pooled"].append(_metric(records, step, "pooled_closed_regret_fraction"))
        out["distance_m"].append(_metric(records, step, "distance_in_units_of_m"))
    return out


def _verdict(table: dict[str, list[float]], nu: float) -> dict[str, Any]:
    pooled = table["pooled"]
    feasibility = table["feasibility"]
    median = float(np.median(pooled)) if not any(np.isnan(pooled)) else float("nan")
    feasible_all = bool(min(feasibility) == 1.0)
    threshold = g3.one_sample_threshold(pooled, nu) if not any(np.isnan(pooled)) else float("nan")
    stage1 = feasible_all and median > STAGE1_MIN_EXCLUSIVE
    stage2 = feasible_all and median > STAGE2_MIN_EXCLUSIVE
    return {
        "feasibility": g3._summary(feasibility),
        "feasible_in_every_seed": feasible_all,
        "pooled_closed_regret_fraction": g3._summary(pooled),
        "distance_in_units_of_m": g3._summary(table["distance_m"]),
        "stage1": stage1,
        "stage2": stage2,
        "one_sample_threshold": threshold,
        "stage1_resolved": bool(stage1 and median > threshold),
        "stage2_resolved": bool(stage2 and median - STAGE2_MIN_EXCLUSIVE > threshold),
    }


def _contrast(a: list[float], b: list[float], nu: float) -> dict[str, Any]:
    """``a - b`` on the median pooled fraction with the SX1 two-sample threshold."""
    difference = float(np.median(a) - np.median(b))
    threshold = g3.two_sample_threshold(a, b, nu)
    return {
        "difference": difference,
        "threshold": threshold["threshold"],
        "resolved_better": bool(difference > threshold["threshold"]),
    }


def classify_effect(contrasts: dict[str, dict[str, Any]]) -> str:
    r0, r1 = contrasts["R|S0"]["resolved_better"], contrasts["R|S1"]["resolved_better"]
    s0, s1 = contrasts["S|R0"]["resolved_better"], contrasts["S|R1"]["resolved_better"]
    if r0 and r1 and s0 and s1:
        return "both, independent"
    if r0 and r1 and not (s0 or s1):
        return "R carries"
    if s0 and s1 and not (r0 or r1):
        return "S carries"
    if r1 and s1 and not (r0 or s0):
        return "both, combined"
    if not (r0 or r1 or s0 or s1):
        return "none"
    return "mixed"


def _trace_counts(prefix: str, cell: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in SEEDS:
        path = rr.output_paths(cell, seed, prefix)["trace"]
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        masked: dict[str, int] = {}
        for row in rows:
            if row["masked"]:
                masked[row["mask_reason"]] = masked.get(row["mask_reason"], 0) + 1
        stored = [r for r in rows if not r["masked"]]
        out[str(seed)] = {
            "n_steps": len(rows),
            "masked_by_reason": masked,
            "n_solve_failed_stored": int(sum(r["solve_failed"] for r in stored)),
            "n_g_tilde_positive": int(
                sum(1 for r in stored if not r["solve_failed"] and r["g_tilde"] > 0.0)
            ),
            "n_stored_infeasible": int(sum(1 for r in stored if not r["feasible"])),
        }
    return out


def _lambda_cells(prefix: str, cell: str) -> dict[str, Any]:
    stem = f"{prefix}_{cell}"
    cells = g3.lambda_cells(stem)
    n_active = sum(1 for s in cells.values() for c in s["cells"] if c["category"] != "inactive")
    return {"n_not_inactive_of_125": n_active, "per_seed": cells}


def _ceiling() -> dict[str, Any]:
    """Oracle action clipped into the residual box: what the box can reach at best."""
    env = rr.eval_env()
    points = load_test_points(rr.REFERENCE_015)
    fixed = rr.fixed_point_from_reference()
    from twinlink_ml.rl.env import _normalise

    anchor = np.array(
        [
            _normalise(fixed[0], env._inlet_bound.minimum, env._inlet_bound.maximum),
            _normalise(fixed[1], env._gas_bound.minimum, env._gas_bound.maximum),
        ]
    )
    by_point = {
        (p.ambient_temperature_celsius, p.ambient_relative_humidity): np.array(
            [
                _normalise(
                    p.oracle_inlet_temperature_celsius,
                    env._inlet_bound.minimum,
                    env._inlet_bound.maximum,
                ),
                _normalise(
                    p.oracle_gas_velocity_m_per_s, env._gas_bound.minimum, env._gas_bound.maximum
                ),
            ]
        )
        for p in points
    }

    def policy(observation: np.ndarray) -> np.ndarray:
        key = min(
            by_point,
            key=lambda k: abs(k[0] - float(observation[0])) + abs(k[1] - float(observation[1])),
        )
        delta = (by_point[key] - anchor) / rr.RESIDUAL_SCALE
        return residual_to_action(delta, anchor, rr.RESIDUAL_SCALE)

    metrics = flat_metrics(evaluate_policy(env, policy, points))
    return {
        "residual_scale": rr.RESIDUAL_SCALE,
        "feasibility_rate": metrics["feasibility_rate"],
        "pooled_closed_regret_fraction": metrics["pooled_closed_regret_fraction"],
    }


def main() -> int:
    require_clean_tree()
    nu = g3._nu_eff()
    result: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    steps = (0, *rr.CHECKPOINTS)
    for cell in CELLS:
        curves[cell] = {
            str(step): {
                key: g3._summary(_cell_table(rr.PREFIX, cell, step)[key])
                for key in ("feasibility", "pooled", "distance_m")
            }
            for step in steps
        }
        result[cell] = {
            str(budget): _verdict(_cell_table(rr.PREFIX, cell, budget), nu) for budget in BUDGETS
        }

    effects: dict[str, Any] = {}
    for budget in BUDGETS:
        pooled = {c: _cell_table(rr.PREFIX, c, budget)["pooled"] for c in CELLS}
        contrasts = {
            "R|S0": _contrast(pooled["res"], pooled["base"], nu),
            "R|S1": _contrast(pooled["ressep"], pooled["sep"], nu),
            "S|R0": _contrast(pooled["sep"], pooled["base"], nu),
            "S|R1": _contrast(pooled["ressep"], pooled["res"], nu),
        }
        # interaction = (R|S1 - R|S0), identical to (S|R1 - S|R0)
        interaction = contrasts["R|S1"]["difference"] - contrasts["R|S0"]["difference"]
        effects[str(budget)] = {
            "contrasts": contrasts,
            "interaction_median_difference": float(interaction),
            "which_measure_carries_the_effect": classify_effect(contrasts),
        }

    start = {}
    for cell in ("res", "ressep"):
        values = [
            _metric(_records(rr.PREFIX, cell, s), 0, "pooled_closed_regret_fraction") for s in SEEDS
        ]
        start[cell] = {
            "per_seed": values,
            "ok": bool(all(abs(v) < START_VALUE_ATOL for v in values)),
        }

    arm_c = {
        "base_minus_armC_pooled_at_10000_per_seed": [
            _metric(_records(rr.PREFIX, "base", s), 10_000, "pooled_closed_regret_fraction")
            - float(
                json.loads(
                    next(
                        line
                        for line in (RUNS / f"{ARM_C_PREFIX}_seed{s}.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line and json.loads(line)["params"]["n_env_steps"] == 10_000
                    )
                )["metrics"]["pooled_closed_regret_fraction"]
            )
            for s in SEEDS
        ]
    }
    masked_total = {
        cell: [
            _metric(_records(rr.PREFIX, cell, s), rr.CHECKPOINTS[-1], "n_masked_total")
            for s in SEEDS
        ]
        for cell in CELLS
    }

    payload = {
        "_provenance": provenance(),
        "test_set": "25 Befund-52 grid points; oracle and fixed point of the limit 0.15",
        "placeholders": "quality limit 0.15 kg/kg, lower gas-velocity bound 0.80 m/s",
        "seeds": list(SEEDS),
        "m_kg_per_kg": MARGIN_M,
        "cells": result,
        "effects": effects,
        "learning_curves": curves,
        "start_value_residual_cells_step0": start,
        "reachable_ceiling_of_residual_box": _ceiling(),
        "base_vs_arm_C_S06": arm_c,
        "masked_total_at_30000_per_seed": masked_total,
        "trace_counts": {c: _trace_counts(rr.PREFIX, c) for c in CELLS},
        "lambda": {c: _lambda_cells(rr.PREFIX, c) for c in CELLS},
        "wall_s_30000_per_seed": {
            c: [
                float(
                    _records(rr.PREFIX, c, s)[rr.CHECKPOINTS[-1]]["params"][
                        "wall_s_since_training_start"
                    ]
                )
                for s in SEEDS
            ]
            for c in CELLS
        },
    }
    write_artifact("bres_gate", payload)
    for budget in BUDGETS:
        print(f"--- budget {budget}")
        for cell in CELLS:
            v = result[cell][str(budget)]
            print(
                cell,
                "feas_all",
                v["feasible_in_every_seed"],
                "pooled median",
                round(v["pooled_closed_regret_fraction"]["median"], 4),
                "stage1",
                v["stage1"],
                "(resolved",
                v["stage1_resolved"],
                ") stage2",
                v["stage2"],
                "dist/m median",
                round(v["distance_in_units_of_m"]["median"], 2),
            )
        print("carries:", effects[str(budget)]["which_measure_carries_the_effect"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
