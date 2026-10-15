r"""Margin study T1b (Befund 95, addendum to Befund 93): margin the RL policy needs, and its cost.

PRE-REGISTRATION. Everything below is fixed BEFORE the first run of the study (this docstring is
committed before any run); the readings are not to be changed afterwards. If no margin meets the gate
that is reported as such; the set of margins is not extended afterwards.

Question. In Befund 93 the cell ``ressep`` (residual action, separated moisture cost) did not meet
the feasibility gate: it settles at the tightened limit (distance about 1 m) and violates the true
limit at 0-40 % of the contexts per seed, because the margin m is small against the policy error.
Which margin ``k * m`` makes the policy feasible in every seed, and what does that cost against the
optimiser?

PLACEHOLDERS. The quality limit 0.15 kg/kg (main case) and the lower gas-velocity bound 0.80 m/s
(``ActionBound``, unchanged) are placeholders without source; the critical moisture 0.20 of the
kinetics, the penalty weight 50 kW and the margin m = 0.005961819615175404 kg/kg are settings or
earlier choices, not results. Every table and artifact of this study inherits that label. Test set
(always named): the 25 Befund 52 grid points (ambient temperature x relative humidity), all
references at the 25 contexts of that grid.

Design (only the cell ``ressep``; otherwise exactly as Befund 93: residual radius s = 0.5 around the
fixed operating point of the limit 0.15, anchor unchanged; separated moisture cost; Lagrange
multiplier per context cell 5 x 5, lambda_0 0.5, eta 0.2 kW, update every 250 steps; SAC with
stable-baselines3 defaults). Margins ``k * m`` with k in {2, 3, 5} (k = 1 is the cell ``ressep`` of
Befund 93, taken from its records; not re-run). Seeds 0-4, 30 000 steps, checkpoints as in Befund 93.
Torch threads fixed to 1 (``torch.set_num_threads(1)``, recorded as ``torch_num_threads``; the base
repetition in Befund 93 was not bit-identical). The margin enters only through the tightened limit of
the Lagrange term (``LagrangeConfig.margin_kg_per_kg``); evaluation is always against the TRUE limit
0.15. Runs: ``scripts/run_rl_bandit_residual.py --cell ressep --margin-factor k``, prefixes
``rl_bandit_residual_b_k{k}``.

References (``scripts/run_bres_margin_references.py``; legacy objective ``energy_target`` of the RL
environment, existing optimisers, Befund 52 budget, 25 contexts): for each k the oracle and the fixed
operating point against the limit ``0.15 - k * m`` (same margin as the RL cell); in addition the
oracle with margin m ("Oracle(m)") as the common reference. The test-set references of the closed
fraction (``E_fixed``, ``E_oracle``) are those of Befund 93 (``s05_limit015_oracle_and_fixed``, limit
0.15, margin 0). CO2 is the gas-heater booking (eta_gas 0.80, factors of ``net_energy.py``) of the
same solution; the nozzle pressure of the RL environment is fixed (``FIXED_NOZZLE_PRESSURE_BAR``),
the optimisers choose it freely (as in Befund 93, part of every policy-to-oracle difference).

Metrics per k and seed (checkpoint 30 000, deterministic policy, plain environment, true limit; the
checkpoints are re-evaluated here and the feasibility and the pooled fraction are cross-checked
against the run record):

  feasibility      share of the 25 contexts feasible against the true limit 0.15;
  closed A         ``sum_feasible (E_fixed - E_agent) / sum_feasible (E_fixed - E_oracle)`` -- the
                   metric of Befund 93, over the contexts at which the policy is feasible only;
  closed B         ``sum_feasible (E_fixed - E_agent) / sum_all (E_fixed - E_oracle)`` -- an
                   infeasible context counts as 0 closed regret (denominator over all 25 contexts);
  ratio to the ceiling   A / C and B / C, C = reachable admissible ceiling of the residual box
                   (oracle action of the limit 0.15 clipped into the box; Befund 93: feasibility
                   1.0, pooled fraction 0.818; recomputed here by ``analyze_bres_gate._ceiling``).
                   CITATION RULE of Befund 93: a closed fraction above C is carried by infeasible
                   points and is NOT reported as a performance figure (it is flagged ``above_ceiling``
                   and printed with the flag only). As extra information the ceiling with the margin
                   of the reference is computed too (oracle of ``0.15 - k m`` clipped into the box);
  cost over Oracle(m)   on the context set S of the seed = contexts at which the policy is feasible
                   and Oracle(m) and Oracle(k m) are feasible: sums over S of the energy (kW) and of
                   the CO2 (kg/h), in per cent of the Oracle(m) sum. Total = (RL(k m) - Oracle(m)) /
                   Oracle(m); decomposition: margin part = (Oracle(k m) - Oracle(m)) / Oracle(m),
                   policy part = (RL(k m) - Oracle(k m)) / Oracle(m); the two add up to the total. The
                   fixed point with margin k m is given on S intersected with its feasible contexts
                   (``n_fixed``) in the same unit. ``n_S`` is reported per seed; for the gate margin
                   (feasibility 1.0 everywhere) S is the full grid unless a reference is infeasible.
  distance         median distance of the product moisture to the true limit at the oracle-binding
                   contexts in units of m (from the run records, as in Befund 93).

Aggregation: median over the five seeds with minimum and maximum; no rank claim is made (descriptive
study; differences between margins are read against the seed range, no resolution threshold is
computed). Learning curves (feasibility and pooled fraction per checkpoint, median and range) are
stored in the artifact.

GATE (fixed here): the smallest k in {2, 3, 5} for which the feasibility is 1.0 in EVERY seed at step
30 000 (strict reading, as in Befund 93). One-sentence reading, as pre-agreed: "RL needs k*m margin,
additional cost X % over the optimiser with m." with X the median over seeds of the total energy cost
(the CO2 value and the seed ranges next to it), the decomposition into margin part and policy part
below it. If no k meets the gate this is reported ("no k in {2, 3, 5} reaches the gate") and k is not
extended afterwards. k = 1 (Befund 93: minimum feasibility 0.60 over seeds) is listed for reference.

    tools/run_here.sh scripts/analyze_bres_margin.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

import run_bres_margin_references as refs
import run_rl_bandit_residual as rr

CELL = "ressep"
FACTORS = (1, 2, 3, 5)  # k = 1 from the records of Befund 93
NEW_FACTORS = (2, 3, 5)
SEEDS = rr.SEEDS
FINAL_STEP = rr.CHECKPOINTS[-1]
MARGIN_M = rr.MARGIN_M
CEILING_TOLERANCE = 1.0e-9
CROSSCHECK_TOLERANCE = 1.0e-9
GAS_HEATER_EFFICIENCY = 0.80


def prefix_for(k: int) -> str:
    return rr.PREFIX if k == 1 else f"{rr.MARGIN_STUDY_PREFIX}{k}"


# ----------------------------------------------------------------------------- pure functions


def closed_fractions(
    e_fixed: np.ndarray, e_oracle: np.ndarray, e_policy: np.ndarray, feasible: np.ndarray
) -> dict[str, float]:
    """Closed fraction A (feasible contexts only) and B (infeasible counts as 0).

    ``e_policy`` entries at infeasible contexts are ignored (they may be NaN).
    """
    feasible = np.asarray(feasible, dtype=bool)
    gain = np.where(feasible, e_fixed - np.where(feasible, e_policy, 0.0), 0.0)
    denominator_feasible = float(np.sum((e_fixed - e_oracle)[feasible]))
    denominator_all = float(np.sum(e_fixed - e_oracle))
    numerator = float(np.sum(gain))
    return {
        "A": numerator / denominator_feasible
        if abs(denominator_feasible) > 1e-12
        else float("nan"),
        "B": numerator / denominator_all if abs(denominator_all) > 1e-12 else float("nan"),
    }


def decompose(e_policy: float, e_oracle_m: float, e_oracle_km: float) -> dict[str, float]:
    """Total cost over Oracle(m) and its split, each in per cent of Oracle(m)."""
    return {
        "total_pct": 100.0 * (e_policy - e_oracle_m) / e_oracle_m,
        "margin_pct": 100.0 * (e_oracle_km - e_oracle_m) / e_oracle_m,
        "policy_pct": 100.0 * (e_policy - e_oracle_km) / e_oracle_m,
    }


def above_ceiling(value: float, ceiling: float) -> bool:
    return bool(np.isfinite(value) and value > ceiling + CEILING_TOLERANCE)


def smallest_k_meeting_gate(feasibility_by_k: dict[int, list[float]]) -> int | None:
    """Smallest k in {2, 3, 5} with feasibility 1.0 in every seed (strict), else None."""
    for k in sorted(feasibility_by_k):
        if (
            k in NEW_FACTORS
            and len(feasibility_by_k[k]) == len(SEEDS)
            and min(feasibility_by_k[k]) == 1.0
        ):
            return k
    return None


def summary(values: list[float]) -> dict[str, Any]:
    array = np.array(values, dtype=float)
    return {
        "per_seed": [float(v) for v in array],
        "median": float(np.nanmedian(array)) if np.isfinite(array).any() else float("nan"),
        "min": float(np.nanmin(array)) if np.isfinite(array).any() else float("nan"),
        "max": float(np.nanmax(array)) if np.isfinite(array).any() else float("nan"),
    }


# ----------------------------------------------------------------------------- data access


def _key(t_amb: float, phi: float) -> tuple[float, float]:
    return (round(float(t_amb), 9), round(float(phi), 9))


def load_reference(kind: str, factor: int, runs: Path) -> dict[tuple[float, float], dict[str, Any]]:
    """Per context: feasible, energy_kw, co2_kg_per_h (None where infeasible or failed)."""
    data = json.loads((runs / f"{refs.artifact_name(kind, factor)}.json").read_text("utf-8"))
    rows = data["results"] if kind == "oracle" else data["fixed"]["per_grid_point"]
    out: dict[tuple[float, float], dict[str, Any]] = {}
    for row in rows:
        ok = (not row.get("solve_failed")) and bool(row.get("feasible", False))
        out[_key(row["ambient_temperature_celsius"], row["ambient_relative_humidity"])] = {
            "feasible": ok,
            "energy_kw": float(row["legacy_energy_target_kw"]) if ok else None,
            "co2_kg_per_h": float(row["gas_heater"]["co2_kg_per_h"]) if ok else None,
            "inlet_temperature_celsius": row.get("inlet_temperature_celsius"),
            "gas_velocity_m_per_s": row.get("gas_velocity_m_per_s"),
        }
    return out


def load_records(prefix: str, seed: int) -> dict[int, dict[str, Any]]:
    path = rr.output_paths(CELL, seed, prefix)["records"]
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return {int(r["params"]["n_env_steps"]): r for r in rows}


# ----------------------------------------------------------------------------- policy evaluation


def solve_action(env: Any, t_in: float, u_gas: float, t_amb: float, phi: float) -> dict[str, Any]:
    """Energy, CO2 (gas booking), moisture and feasibility of one action, as the environment does."""
    from twinlink_ml.evaluation.carrier_energy import carrier_energy_from_diagnostics
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator
    from twinlink_ml.simulation.operating_point import (
        DEFAULT_SEARCH_MAX_ITERATIONS,
        DEFAULT_SEARCH_TOLERANCE,
        build_variant_config,
        check_constraints,
    )
    from twinlink_ml.simulation.steady_state import solve_steady_state

    config = build_variant_config(
        env._base_config,
        t_in=t_in,
        u_gas=u_gas,
        p_nozzle=env._env_config.fixed_nozzle_pressure_bar,
        t_amb=t_amb,
        phi_amb=phi,
    )
    try:
        solution = solve_steady_state(
            GreyBoxFluidBedGranulator(config),
            max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=DEFAULT_SEARCH_TOLERANCE,
        )
    except (RuntimeError, ValueError):
        return {"solved": False, "feasible": False, "energy_kw": None, "co2_kg_per_h": None}
    check = check_constraints(
        solution,
        critical_moisture_kg_per_kg=env._base_config.material.quality_moisture_limit_kg_per_kg,
        gas_velocity_m_per_s=u_gas,
    )
    gas = carrier_energy_from_diagnostics(
        solution.diagnostics,
        heater="gas",
        process_heater_efficiency=env._base_config.energy.heater_efficiency,
        gas_heater_efficiency=GAS_HEATER_EFFICIENCY,
    )
    return {
        "solved": True,
        "feasible": bool(check.feasible),
        "energy_kw": float(solution.diagnostics["energy_target"]),
        "co2_kg_per_h": float(gas.co2_kg_per_h),
        "moisture_kg_per_kg": float(solution.particle_moisture_kg_per_kg),
    }


def evaluate_actions(env: Any, points: Any, policy: Any) -> list[dict[str, Any]]:
    from twinlink_ml.rl.env import _denormalise

    rows = []
    for point in points:
        observation = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity], dtype=np.float32
        )
        action = np.clip(np.asarray(policy(observation), dtype=np.float64), -1.0, 1.0)
        t_in = _denormalise(float(action[0]), env._inlet_bound.minimum, env._inlet_bound.maximum)
        u_gas = _denormalise(float(action[1]), env._gas_bound.minimum, env._gas_bound.maximum)
        rows.append(
            {
                "context": _key(point.ambient_temperature_celsius, point.ambient_relative_humidity),
                "inlet_temperature_celsius": t_in,
                "gas_velocity_m_per_s": u_gas,
                **solve_action(
                    env,
                    t_in,
                    u_gas,
                    point.ambient_temperature_celsius,
                    point.ambient_relative_humidity,
                ),
            }
        )
    return rows


def anchor_normalised(env: Any) -> np.ndarray:
    from twinlink_ml.rl.env import _normalise

    inlet, gas = rr.fixed_point_from_reference()
    return np.array(
        [
            _normalise(inlet, env._inlet_bound.minimum, env._inlet_bound.maximum),
            _normalise(gas, env._gas_bound.minimum, env._gas_bound.maximum),
        ]
    )


def clipped_oracle_policy(env: Any, reference: dict[tuple[float, float], dict[str, Any]]) -> Any:
    """Policy that plays the oracle action of ``reference`` clipped into the residual box."""
    from twinlink_ml.rl.env import _normalise
    from twinlink_ml.rl.residual import residual_to_action

    anchor = anchor_normalised(env)
    targets = {
        key: np.array(
            [
                _normalise(
                    row["inlet_temperature_celsius"],
                    env._inlet_bound.minimum,
                    env._inlet_bound.maximum,
                ),
                _normalise(
                    row["gas_velocity_m_per_s"], env._gas_bound.minimum, env._gas_bound.maximum
                ),
            ]
        )
        for key, row in reference.items()
        if row["feasible"]
    }

    def policy(observation: np.ndarray) -> np.ndarray:
        key = _key(float(observation[0]), float(observation[1]))
        match = min(targets, key=lambda k: abs(k[0] - key[0]) + abs(k[1] - key[1]))
        delta = (targets[match] - anchor) / rr.RESIDUAL_SCALE
        return residual_to_action(delta, anchor, rr.RESIDUAL_SCALE)

    return policy


def metrics_of(
    rows: list[dict[str, Any]],
    points: Any,
    oracle_m: dict[Any, dict[str, Any]],
    oracle_km: dict[Any, dict[str, Any]],
    fixed_km: dict[Any, dict[str, Any]] | None,
) -> dict[str, Any]:
    e_fixed = np.array([p.energy_fixed_kw for p in points])
    e_oracle0 = np.array([p.energy_oracle_kw for p in points])
    feasible = np.array([bool(r["feasible"]) for r in rows])
    e_policy = np.array([r["energy_kw"] if r["feasible"] else np.nan for r in rows], dtype=float)
    out: dict[str, Any] = {
        "feasibility": float(feasible.mean()),
        "n_feasible": int(feasible.sum()),
        **{
            f"closed_{key}": v
            for key, v in closed_fractions(e_fixed, e_oracle0, e_policy, feasible).items()
        },
    }
    keys = [r["context"] for r in rows]
    in_set = np.array(
        [
            bool(r["feasible"]) and oracle_m[k]["feasible"] and oracle_km[k]["feasible"]
            for r, k in zip(rows, keys, strict=True)
        ]
    )
    out["n_S"] = int(in_set.sum())
    for unit, label in (("energy_kw", "energy"), ("co2_kg_per_h", "co2")):
        if not in_set.any():
            out[label] = None
            continue
        pol = sum(r[unit] for r, s in zip(rows, in_set, strict=True) if s)
        om = sum(oracle_m[k][unit] for k, s in zip(keys, in_set, strict=True) if s)
        okm = sum(oracle_km[k][unit] for k, s in zip(keys, in_set, strict=True) if s)
        entry: dict[str, Any] = {
            "policy_sum": pol,
            "oracle_m_sum": om,
            "oracle_km_sum": okm,
            **decompose(pol, om, okm),
        }
        if fixed_km is not None:
            sub = [
                (k, s) for k, s in zip(keys, in_set, strict=True) if s and fixed_km[k]["feasible"]
            ]
            if sub:
                entry["n_fixed"] = len(sub)
                entry["fixed_km_over_oracle_m_pct"] = 100.0 * (
                    sum(fixed_km[k][unit] for k, _ in sub) / sum(oracle_m[k][unit] for k, _ in sub)
                    - 1.0
                )
                entry["oracle_km_over_oracle_m_pct_same_set"] = 100.0 * (
                    sum(oracle_km[k][unit] for k, _ in sub) / sum(oracle_m[k][unit] for k, _ in sub)
                    - 1.0
                )
        out[label] = entry
    return out


# ----------------------------------------------------------------------------- main


def main() -> int:
    import analyze_bres_gate as gate
    from _common import NO_HEAT_NETWORK, RUNS_DIR, provenance, require_clean_tree, write_artifact

    from twinlink_ml.rl import load_test_points

    require_clean_tree()
    import torch
    from stable_baselines3 import SAC

    torch.set_num_threads(1)
    env = rr.eval_env()
    points = load_test_points(rr.REFERENCE_015)
    anchor = anchor_normalised(env)
    ceiling = gate._ceiling()
    ceiling_value = float(ceiling["pooled_closed_regret_fraction"])

    oracle_m = load_reference("oracle", 1, RUNS_DIR)
    result: dict[str, Any] = {}
    feasibility_by_k: dict[int, list[float]] = {}
    for k in FACTORS:
        oracle_km = load_reference("oracle", k, RUNS_DIR)
        fixed_km = load_reference("fixed", k, RUNS_DIR) if k in NEW_FACTORS else None
        margin_rows = evaluate_actions(env, points, clipped_oracle_policy(env, oracle_km))
        margin_ceiling = closed_fractions(
            np.array([p.energy_fixed_kw for p in points]),
            np.array([p.energy_oracle_kw for p in points]),
            np.array([r["energy_kw"] if r["feasible"] else np.nan for r in margin_rows]),
            np.array([r["feasible"] for r in margin_rows]),
        )
        per_seed: dict[str, Any] = {}
        curves: dict[str, Any] = {}
        for seed in SEEDS:
            records = load_records(prefix_for(k), seed)
            record = records[FINAL_STEP]
            path = rr.REPO_ROOT / record["params"]["checkpoint/path"]
            model = SAC.load(str(path), device="cpu")
            rows = evaluate_actions(env, points, rr.make_policy(model, anchor))
            metrics = metrics_of(rows, points, oracle_m, oracle_km, fixed_km)
            m = record["metrics"]
            metrics["crosscheck_feasibility_ok"] = bool(
                abs(metrics["feasibility"] - m["feasibility_rate"]) < CROSSCHECK_TOLERANCE
            )
            metrics["crosscheck_pooled_ok"] = bool(
                abs(metrics["closed_A"] - m["pooled_closed_regret_fraction"]) < 1e-6
            )
            metrics["distance_in_units_of_m"] = float(m["distance_in_units_of_m"])
            metrics["above_ceiling_A"] = above_ceiling(metrics["closed_A"], ceiling_value)
            metrics["above_ceiling_B"] = above_ceiling(metrics["closed_B"], ceiling_value)
            metrics["A_over_ceiling"] = metrics["closed_A"] / ceiling_value
            metrics["B_over_ceiling"] = metrics["closed_B"] / ceiling_value
            metrics["checkpoint"] = record["params"]["checkpoint/path"]
            metrics["margin_factor_in_record"] = record["params"].get("margin_factor", 1.0)
            metrics["torch_num_threads_in_record"] = record["params"].get("torch_num_threads")
            metrics["git_dirty_in_record"] = record["params"].get("git/dirty")
            metrics["points"] = rows
            per_seed[str(seed)] = metrics
            curves[str(seed)] = {
                str(step): {
                    "feasibility": float(r["metrics"]["feasibility_rate"]),
                    "pooled": float(r["metrics"]["pooled_closed_regret_fraction"]),
                }
                for step, r in sorted(records.items())
            }
            print(
                f"k={k} seed {seed}: feas {metrics['feasibility']:.2f} A {metrics['closed_A']:.3f}"
                f" B {metrics['closed_B']:.3f} crosscheck {metrics['crosscheck_feasibility_ok']}"
                f"/{metrics['crosscheck_pooled_ok']}",
                flush=True,
            )
        feas = [per_seed[str(s)]["feasibility"] for s in SEEDS]
        feasibility_by_k[k] = feas

        def agg(label: str, key: str, per_seed: dict[str, Any] = per_seed) -> dict[str, Any]:
            return summary(
                [
                    per_seed[str(s)][label][key] if per_seed[str(s)][label] else float("nan")
                    for s in SEEDS
                ]
            )

        result[str(k)] = {
            "margin_kg_per_kg": k * MARGIN_M,
            "source": "records of Befund 93" if k == 1 else "this study",
            "feasibility": summary(feas),
            "feasible_in_every_seed": bool(min(feas) == 1.0),
            "closed_A": summary([per_seed[str(s)]["closed_A"] for s in SEEDS]),
            "closed_B": summary([per_seed[str(s)]["closed_B"] for s in SEEDS]),
            "A_over_ceiling": summary([per_seed[str(s)]["A_over_ceiling"] for s in SEEDS]),
            "B_over_ceiling": summary([per_seed[str(s)]["B_over_ceiling"] for s in SEEDS]),
            "above_ceiling_A_per_seed": [per_seed[str(s)]["above_ceiling_A"] for s in SEEDS],
            "above_ceiling_B_per_seed": [per_seed[str(s)]["above_ceiling_B"] for s in SEEDS],
            "distance_in_units_of_m": summary(
                [per_seed[str(s)]["distance_in_units_of_m"] for s in SEEDS]
            ),
            "n_S_per_seed": [per_seed[str(s)]["n_S"] for s in SEEDS],
            "cost_energy_total_pct": agg("energy", "total_pct"),
            "cost_energy_margin_part_pct": agg("energy", "margin_pct"),
            "cost_energy_policy_part_pct": agg("energy", "policy_pct"),
            "cost_co2_total_pct": agg("co2", "total_pct"),
            "cost_co2_margin_part_pct": agg("co2", "margin_pct"),
            "cost_co2_policy_part_pct": agg("co2", "policy_pct"),
            "fixed_km_over_oracle_m_energy_pct": (
                agg("energy", "fixed_km_over_oracle_m_pct") if fixed_km is not None else None
            ),
            "ceiling_with_margin_of_reference": {
                "feasibility": float(np.mean([r["feasible"] for r in margin_rows])),
                **margin_ceiling,
            },
            "crosschecks_ok": all(
                per_seed[str(s)][c]
                for s in SEEDS
                for c in ("crosscheck_feasibility_ok", "crosscheck_pooled_ok")
            ),
            "learning_curves_per_seed": curves,
            "per_seed": per_seed,
        }
    gate_k = smallest_k_meeting_gate(feasibility_by_k)
    if gate_k is None:
        reading = (
            "No k in {2, 3, 5} reaches the gate (feasibility 1.0 in every seed at 30 000 steps); "
            "k is not extended afterwards."
        )
    else:
        r = result[str(gate_k)]
        reading = (
            f"RL needs {gate_k}*m margin, additional cost {r['cost_energy_total_pct']['median']:.2f} % "
            f"(energy; CO2 {r['cost_co2_total_pct']['median']:.2f} %) over the optimiser with m "
            f"(median over 5 seeds; margin part {r['cost_energy_margin_part_pct']['median']:.2f} %, "
            f"policy part {r['cost_energy_policy_part_pct']['median']:.2f} %)."
        )
    payload = {
        "_provenance": provenance(NO_HEAT_NETWORK),
        "test_set": "25 Befund-52 grid points; references against the limit 0.15 - k*m",
        "placeholders": "quality limit 0.15 kg/kg; gas velocity lower bound 0.80 m/s; margin m",
        "m_kg_per_kg": MARGIN_M,
        "seeds": list(SEEDS),
        "step": FINAL_STEP,
        "ceiling_of_residual_box_limit015": ceiling,
        "gate_k": gate_k,
        "reading": reading,
        "cells": result,
    }
    write_artifact("bres_margin", payload)
    print(reading)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
