r"""Analysis of the study docs/30_surrogate_env.md -- PRE-REGISTRATION (fixed before the first run).

Question. How strongly do RL and an optimiser exploit the errors of a learned sequence model used as
the environment for control? Problem: the one-step operating-point problem of Befund 96 unchanged
(context = ambient temperature and relative humidity; action = inlet temperature and gas velocity;
nozzle pressure fixed, spray rate and solid fraction nominal; moisture limit 0.15 kg/kg, bed
temperature <= 80 degC; objective ``energy_target``). Truth and evaluation always on the grey-box
model, steady state, against the TRUE limit 0.15. Test set (always named): the 25 Befund-52 grid
points.

Environments (for training / optimisation)
  U0   grey-box steady state (reference). SAC: Befund 96 (cell ``ressep``, k = 5) re-run for seeds
       0-2 by the unchanged script -- the checkpoints of Befund 96 are lost -- and compared with the
       records of Befund 96 bit for bit (reproduction finding); oracle k = 5 of Befund 96 recomputed
       at 10 of 25 points (control).
  U0'  the mapping of the learned environments applied to the grey-box model. The mapping is the
       FIXED POINT of the h36 horizon map with constant inputs (decision of the project lead,
       2026-10-27: the slowest moisture mode, 1600-2900 s, is longer than h36 = 900 s, so a plain
       rollout from another operating point does not reach the steady state). For the grey-box model
       that fixed point is the steady state of the solver, i.e. U0' = U0; reported as the drift of
       a 900 s constant-input simulation started at the solver's steady state (control).
  U1   composite GRU of the Befund 94 pipeline (arm N), ``n_train`` 16.
  U2   the same, ``n_train`` 64.
  U3   ensemble of 5 GRU on the data of U2 (model seeds differ, same 64 runs); moisture of the
       constraint ``mean + 2 std`` (sample std, ddof 1); energy and bed temperature: mean.
  Three replicates: GRU master seeds 1, 2, 3; RL seed = master seed - 1.

Fixed-point criterion (per head, per operating point; ``twinlink_ml.surrogate_env.fixed_point``):
  scan of the residual r(x) = f_h36(x) - x at 65 points over [-5, 5] (target-standardised units) as
  multi-start of the root search; valid ("ok") iff exactly ONE sign change and it is stable (r from
  >= 0 to < 0), after a second 65-point scan inside the bracket and linear interpolation
  |r(x*)| <= 1e-3; otherwise ``no_root`` / ``multiple_roots`` / ``unstable`` / ``not_converged``.
  Counted per environment and reported, never discarded. For the controller such a point is not
  admissible: in RL it is a solver failure of the environment (masked as in the cell ``ressep``,
  counted), in RTO it gets the solver-failure value 1e6 kW. A point is valid only if all three heads
  (and, in U3, all five members) are valid.

Controllers per environment
  C1  RL as Befund 96 (cell ``ressep``, margin 5 m, all settings of Befund 93/96): SAC with the
      stable-baselines3 defaults and, because Befund 96 used SAC only, TD3 with the
      stable-baselines3 defaults plus Gaussian exploration noise sigma 0.1 (``run_surrogate_env_rl``).
      30 000 steps; the reading uses the step-30 000 checkpoint, the earlier checkpoints are the
      learning curve.
  C2  multistart Nelder-Mead (fixed point + 7 Sobol starts) on the penalised objective of the oracle
      against ``0.15 - k m``; main k = 5 (the margin of C1), k = 1 sensitivity without reading
      (``run_surrogate_env_rto``).

Metrics per (environment, controller, replicate), on the grey-box model at the 25 grid points
  energy over the true optimum (oracle of the limit 0.15 without margin, ``s05``): pooled,
  100 (sum E / sum E* - 1) over the points where the truth solved; the same over Oracle(m)
  (``bres_margin_ref_oracle_k1``) for the connection to Befund 96; share of violated points
  (any constraint, or the truth not solvable); largest moisture violation max(0, X - 0.15); the
  EXPLOITATION GAP: mean over the points of (E predicted by the training environment - E true),
  in kW and pooled in %, and the same for the moisture (predicted mean; for U3 also the moisture of
  the constraint); number of points that the environment calls feasible and the truth does not.
  Median and range over the three replicates.

Pre-registered reading (per controller SAC, TD3, RTO k = 5)
  Criterion of an environment U for controller c: (i) feasibility pooled over replicates and points
  (3 x 25 cases) >= 0.95, and (ii) energy excess over the result of the same controller on U0,
  ``sum E(U, r) / median_r' sum E(U0, r') - 1`` (points solved on both sides), median over the
  replicates <= 0.02.
  G0: U2 meets the criterion -> learned models are usable as an environment.
  G1: U2 does not (violations > 5 % or energy > 2 % worse), but U3 (ensemble, pessimistic) does ->
      a learned environment only with an uncertainty measure.
  G2: U3 does not either -> learned models are not a sufficient environment for this task.
  If the controller does not meet (i) on U0 itself, that is reported next to the reading (the
  reading is applied unchanged).
  Additionally: RL against RTO on the same learned environment -- which exploits the model errors
  more? Per U in {U1, U2, U3} and RL algorithm against RTO k = 5: paired over the replicates (same
  GRU), ``paired_resolution.paired_comparison`` (t with 2 degrees of freedom, 95 %) on (a) the mean
  energy gap in kW, (b) the mean moisture gap in kg/kg, (c) the violation share. "stronger
  exploitation" = more negative energy gap, more negative moisture gap, higher violation share; a
  claim only where resolved, otherwise "not resolved at 3 replicates" with the medians; 18 tests, no
  multiplicity correction, the number is reported.

Budget rule (of the request): run time extrapolated from a pilot; above 30 h first U1 is dropped,
then the RL seeds go to 2. Cuts are documented in the artifact and in the lab book.

PLACEHOLDERS. Quality limit 0.15 kg/kg, gas-velocity lower bound 0.80 m/s, critical moisture 0.20,
penalty weight 50 kW, margin m = 0.005961819615175404 kg/kg and the whole process config are
placeholders or earlier settings, not sourced values; every number inherits them.

    tools/run_here.sh scripts/analyze_surrogate_env.py --name surrogate_env_v1_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_rl_bandit_residual as rr  # noqa: E402
import run_surrogate_env_rl as senv  # noqa: E402

from twinlink_ml.evaluation.paired_resolution import paired_comparison  # noqa: E402
from twinlink_ml.surrogate_env.reading import (  # noqa: E402
    classify,
    criterion,
    energy_excess,
    point_metrics,
)

RUNS = rr.RUNS
STEP = 30_000
CONTROLLERS = ("SAC", "TD3", "RTO")
LEARNED = ("U1", "U2", "U3")
BEFUND96_PREFIX = "rl_bandit_residual_b_k5"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _oracle_m() -> list[float]:
    data = json.loads((RUNS / "bres_margin_ref_oracle_k1.json").read_text(encoding="utf-8"))
    return [float(r["legacy_energy_target_kw"]) for r in data["results"]]


def _with_references(rows: list[dict[str, Any]], oracle_m: list[float]) -> list[dict[str, Any]]:
    out = []
    for index, row in enumerate(rows):
        out.append(
            {
                **row,
                "energy_oracle_kw": row["energy_oracle_limit015_kw"],
                "energy_oracle_m_kw": oracle_m[index],
            }
        )
    return out


def rl_rows(env_name: str, algo: str, seed: int, step: int = STEP) -> list[dict[str, Any]]:
    path = senv.output_paths(env_name, algo, seed)["records"]
    for record in _jsonl(path):
        if int(record["params"]["n_env_steps"]) == step:
            return record["params"]["eval/points"]
    raise KeyError((env_name, algo, seed, step))


def u0_sac_rows(seed: int, step: int = STEP) -> list[dict[str, Any]]:
    """Re-evaluate the step checkpoint of the re-run SAC on U0 (the old script stores no points)."""
    from stable_baselines3 import SAC

    from twinlink_ml.rl.env import _normalise

    records = _jsonl(rr.output_paths("ressep", seed, senv.U0_SAC_PREFIX)["records"])
    record = next(r for r in records if int(r["params"]["n_env_steps"]) == step)
    model = SAC.load(rr.REPO_ROOT / record["params"]["checkpoint/path"])
    env = rr.eval_env()
    t_fix, u_fix = rr.fixed_point_from_reference()
    anchor = np.array(
        [
            _normalise(t_fix, env._inlet_bound.minimum, env._inlet_bound.maximum),
            _normalise(u_fix, env._gas_bound.minimum, env._gas_bound.maximum),
        ]
    )
    policy = rr.make_policy(model, anchor)
    points = rr.load_test_points(rr.REFERENCE_015)
    return senv.evaluate_points(policy, env, None, points)


def rto_rows(
    rto: dict[str, Any], env_name: str, seed: int | None, factor: float
) -> list[dict[str, Any]]:
    tasks = [
        t
        for t in rto["tasks"]
        if t["env"] == env_name
        and t["margin_factor"] == factor
        and (env_name == "U0" or t["replicate_master_seed"] == seed)
    ]
    tasks.sort(key=lambda t: t["grid_index"])
    rows = []
    for t in tasks:
        truth = t["truth"]
        rows.append(
            {
                "true_feasible": truth["feasible"],
                "true_solve_failed": truth["solve_failed"],
                "true_energy_kw": truth["energy_kw"],
                "true_moisture_kg_per_kg": truth["moisture_kg_per_kg"],
                "true_bed_temperature_celsius": truth["bed_temperature_celsius"],
                "energy_oracle_limit015_kw": t["energy_oracle_limit015_kw"],
                "inlet_temperature_celsius": t["result"]["inlet_temperature_celsius"],
                "gas_velocity_m_per_s": t["result"]["gas_velocity_m_per_s"],
                "predicted": t.get("predicted"),
            }
        )
    if len(rows) != 25:
        raise ValueError(f"{env_name} seed {seed} k {factor}: {len(rows)} points")
    return rows


def _energies(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [np.nan if r["true_solve_failed"] else float(r["true_energy_kw"]) for r in rows]
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "per_replicate": values,
        }
    return {
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "per_replicate": [float(v) for v in values],
    }


def collect(
    rto: dict[str, Any], plan: dict[str, Any]
) -> dict[tuple[str, str], list[list[dict[str, Any]]]]:
    """Per (environment, controller) the per-point rows of every replicate (main margin)."""
    oracle_m = _oracle_m()
    out: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    for env_name in plan["environments"]:
        for algo in ("SAC", "TD3"):
            replicates = []
            for seed in plan["rl_seeds"]:
                if env_name == "U0" and algo == "SAC":
                    rows = u0_sac_rows(seed)
                else:
                    rows = rl_rows(env_name, algo, seed)
                replicates.append(_with_references(rows, oracle_m))
            out[(env_name, algo)] = replicates
        seeds = [None] if env_name == "U0" else plan["gru_seeds"]
        out[(env_name, "RTO")] = [
            _with_references(rto_rows(rto, env_name, s, 5.0), oracle_m) for s in seeds
        ]
        out[(env_name, "RTO_k1")] = [
            _with_references(rto_rows(rto, env_name, s, 1.0), oracle_m) for s in seeds
        ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--environments",
        nargs="+",
        default=["U0", "U1", "U2", "U3"],
        help="only to apply a documented budget cut (U1 dropped)",
    )
    parser.add_argument(
        "--rl-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="only to apply a documented budget cut (RL seeds 0, 1)",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    from _common import provenance, require_clean_tree, write_artifact

    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(rr.REPO_ROOT)
    require_clean_tree()
    prov = provenance("not used")
    plan = {
        "environments": args.environments,
        "rl_seeds": args.rl_seeds,
        "gru_seeds": [s + 1 for s in args.rl_seeds] if len(args.rl_seeds) < 3 else [1, 2, 3],
    }
    rto = json.loads((RUNS / "surrogate_env_rto_v1.json").read_text(encoding="utf-8"))
    rows = collect(rto, plan)
    limit = rr.LIMIT_KG_PER_KG

    table: dict[str, Any] = {}
    for (env_name, controller), replicates in rows.items():
        metrics = [point_metrics(r, limit) for r in replicates]
        keys = sorted({k for m in metrics for k in m})
        table[f"{env_name}/{controller}"] = {
            k: _summary([m.get(k, float("nan")) for m in metrics]) for k in keys
        }

    reading: dict[str, Any] = {}
    for controller in CONTROLLERS:
        reference = [_energies(r) for r in rows[("U0", controller)]]
        crits = {}
        for env_name in [e for e in plan["environments"] if e != "U0"] + ["U0"]:
            replicates = rows[(env_name, controller)]
            counts = [(sum(bool(p["true_feasible"]) for p in r), len(r)) for r in replicates]
            excess = [energy_excess(_energies(r), reference) for r in replicates]
            crits[env_name] = criterion(counts, excess)
        reading[controller] = {
            "class": classify(crits["U2"], crits["U3"]),
            "criteria": {
                e: {
                    "feasibility_share_pooled": c.feasibility_share_pooled,
                    "energy_excess_median": c.energy_excess_median,
                    "energy_excess_per_replicate": list(c.energy_excess_per_replicate),
                    "passes": c.passes,
                }
                for e, c in crits.items()
            },
            "u0_meets_feasibility": crits["U0"].feasibility_share_pooled >= 0.95,
        }

    comparisons = []
    gru_seeds = plan["gru_seeds"]
    for env_name in [e for e in plan["environments"] if e in LEARNED]:
        rto_metrics = {
            s: point_metrics(r, limit)
            for s, r in zip(gru_seeds, rows[(env_name, "RTO")], strict=True)
        }
        for algo in ("SAC", "TD3"):
            rl_metrics = {
                s + 1: point_metrics(r, limit)
                for s, r in zip(plan["rl_seeds"], rows[(env_name, algo)], strict=True)
            }
            for metric, sign in (
                ("gap_energy_mean_kw", 1.0),
                ("gap_moisture_mean_kg_per_kg", 1.0),
                ("violation_share", -1.0),
            ):
                # paired_comparison: lower is "better"; "better" here = LESS exploitation
                candidate = {s: -sign * rl_metrics[s][metric] for s in rl_metrics}
                ref = {s: -sign * rto_metrics[s][metric] for s in rl_metrics}
                result = paired_comparison(
                    candidate, ref, candidate_label=algo, reference_label="RTO"
                )
                resolved = (not result.degenerate) and abs(
                    result.mean_difference
                ) > result.threshold
                stronger = None
                if resolved:
                    stronger = "RTO" if result.mean_difference > 0 else algo
                comparisons.append(
                    {
                        "env": env_name,
                        "rl": algo,
                        "metric": metric,
                        "rl_values": {str(s): rl_metrics[s][metric] for s in rl_metrics},
                        "rto_values": {str(s): rto_metrics[s][metric] for s in rl_metrics},
                        "mean_difference_transformed": result.mean_difference,
                        "threshold": result.threshold,
                        "p_value": result.p_value,
                        "degenerate": result.degenerate,
                        "resolved": resolved,
                        "stronger_exploitation": stronger,
                    }
                )

    curves: dict[str, Any] = {}
    for env_name in plan["environments"]:
        for algo in ("SAC", "TD3"):
            if env_name == "U0" and algo == "SAC":
                series = {}
                for seed in plan["rl_seeds"]:
                    records = _jsonl(rr.output_paths("ressep", seed, senv.U0_SAC_PREFIX)["records"])
                    series[str(seed)] = {
                        str(r["params"]["n_env_steps"]): r["metrics"]["feasibility_rate"]
                        for r in records
                    }
                curves[f"{env_name}/{algo}"] = {"feasibility_rate": series}
                continue
            series_f, series_e = {}, {}
            for seed in plan["rl_seeds"]:
                records = _jsonl(senv.output_paths(env_name, algo, seed)["records"])
                series_f[str(seed)] = {
                    str(r["params"]["n_env_steps"]): r["metrics"]["feasibility_rate"]
                    for r in records
                }
                series_e[str(seed)] = {
                    str(r["params"]["n_env_steps"]): point_metrics(
                        _with_references(r["params"]["eval/points"], _oracle_m()), limit
                    )["energy_over_optimum_pct"]
                    for r in records
                }
            curves[f"{env_name}/{algo}"] = {
                "feasibility_rate": series_f,
                "energy_over_optimum_pct": series_e,
            }

    reproduction = reproduction_u0_sac(plan["rl_seeds"])
    payload = {
        "provenance": prov,
        "plan": plan,
        "test_set": "25 Befund-52 grid points; truth = grey-box steady state against the limit 0.15",
        "placeholders": "quality limit 0.15 kg/kg; gas velocity lower bound 0.80 m/s; margin m; "
        "penalty weight 50 kW",
        "table": table,
        "reading": reading,
        "rl_vs_rto": {"n_tests": len(comparisons), "tests": comparisons},
        "learning_curves": curves,
        "reproduction_u0_sac_befund96": reproduction,
        "fixed_point_status_counts": status_counts(rto, plan),
    }
    write_artifact(args.name, payload)
    for controller, entry in reading.items():
        print(
            controller,
            entry["class"],
            {
                e: (round(c["feasibility_share_pooled"], 3), round(c["energy_excess_median"], 4))
                for e, c in entry["criteria"].items()
            },
        )
    return 0


def reproduction_u0_sac(seeds: list[int]) -> dict[str, Any]:
    """Records of the re-run SAC on U0 against Befund 96: every metric at every checkpoint."""
    out: dict[str, Any] = {}
    for seed in seeds:
        new = _jsonl(rr.output_paths("ressep", seed, senv.U0_SAC_PREFIX)["records"])
        old = _jsonl(rr.output_paths("ressep", seed, BEFUND96_PREFIX)["records"])
        old_by_step = {int(r["params"]["n_env_steps"]): r for r in old}
        worst, n, identical = 0.0, 0, True
        for record in new:
            step = int(record["params"]["n_env_steps"])
            reference = old_by_step[step]["metrics"]
            for key, value in record["metrics"].items():
                a, b = float(value), float(reference[key])
                if np.isnan(a) and np.isnan(b):
                    continue
                n += 1
                identical &= a == b
                worst = max(worst, abs(a - b))
        out[str(seed)] = {
            "n_checkpoints": len(new),
            "n_values_compared": n,
            "max_abs_difference": worst,
            "identical": bool(identical),
        }
    return out


def status_counts(rto: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Fixed-point statuses: RL training (last record of each run), RTO results, error map."""
    out: dict[str, Any] = {"rl_training": {}, "rto_result_points": {}, "error_map": {}}
    for env_name in [e for e in plan["environments"] if e in LEARNED]:
        for algo in ("SAC", "TD3"):
            for seed in plan["rl_seeds"]:
                records = _jsonl(senv.output_paths(env_name, algo, seed)["records"])
                last = max(records, key=lambda r: int(r["params"]["n_env_steps"]))
                out["rl_training"][f"{env_name}/{algo}/seed{seed}"] = {
                    "counts": last["params"]["fixed_point_status_counts_training"],
                    "n_masked": last["params"]["n_masked"],
                }
        totals: dict[str, int] = {}
        for task in rto["tasks"]:
            if task["env"] == env_name:
                for key, value in task.get("predicted_status_counts", {}).items():
                    totals[key] = totals.get(key, 0) + int(value)
        out["rto_result_points"][env_name] = totals
    path = RUNS / "surrogate_env_error_map_v1.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for model in data["models"]:
            out["error_map"][f"{model['env']}/seed{model['replicate_master_seed']}"] = model[
                "status_counts"
            ]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
