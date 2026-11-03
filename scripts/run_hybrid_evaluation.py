r"""Evaluation of the study docs/NN_hybrid_model.md: oracle, margins, one-step and rollout errors.

Pre-registration: docstring of ``scripts/analyze_hybrid_model.py`` (committed before the first run).
Settings: ``conf/hybrid_model/hybrid_model.yaml``. Every part appends records to its own JSONL
artifact; ``--resume`` skips the jobs that have one; an existing artifact is never overwritten
otherwise.

``oracle``   per draw: the optimum of the TRUTH (grid + zoom, margin 0, limit 0.15) at every case
             (the reference of the energy, and the operating point of the coverage classification);
             the classification (b) of every interior case for every (seed, ``n_train``); the
             discretisation check (addendum 1): at ``n_cases`` interior cases the oracle once more
             by the 8-start Nelder-Mead of Befund 113 (``surrogate_env.multistart_minimize``, the
             fixed point of Befund 93 and 7 Sobol starts) on the same objective; energy and slack
             difference.                                         -> ``<prefix>_oracle``
``margins``  per job (model, seed, ``n_train``), models M0 (prior knowledge), M1 (GRU), M2 (refit),
             M3 (hybrid), M4 (hybrid ensemble): per case INSIDE the training range the needed
             constraint parameter by bisection (``grid_rto.search_parameter``; M0-M3 a constant
             margin m, M4 the factor k of ``mean + k std``); the parameter of the model = the 95 %
             quantile over the inside cases of all 20 draws (raise rule of Befund 111 if the
             holding share at it is below 95 %); then EVERY case (inside, outside (b), extrapolation
             (a)) is optimised at that parameter and evaluated at the truth. Plus the box-wide
             moisture bias on the coarse grid.                   -> ``<prefix>_margins``
``errors``   per (draw, seed): one-step (h1) and rollout (h36) RMSE and bias in physical units of
             every model and ``n_train`` on the excited test set of the pool, on the constant-run
             holdout (seed 999; runs split by (b)) and on a constant-run holdout with the widened
             ambient ranges (runs outside the data box = (a)).    -> ``<prefix>_errors``

    tools/run_here.sh scripts/run_hybrid_evaluation.py --part oracle --workers 14
    tools/run_here.sh scripts/run_hybrid_evaluation.py --part margins --workers 14
    tools/run_here.sh scripts/run_hybrid_evaluation.py --part errors --workers 14
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

ROOT = SCRIPTS.parent
MODELS = ("M0", "M1", "M2", "M3", "M4")
LIMIT_TOLERANCE = 1.0e-4  # moisture tolerance of the bandit's constraint check

_SETUP: Any = None


def setup() -> Any:
    global _SETUP
    if _SETUP is None:
        from twinlink_ml.hybrid_model.study import build_setup

        _SETUP = build_setup()
    return _SETUP


# --- shared helpers -----------------------------------------------------------------------------


def truth_map(draw: int) -> Any:
    from twinlink_ml.hybrid_model.maps import GreyBoxMap

    s = setup()
    config = s.steady_truth(draw)
    return GreyBoxMap(
        config,
        s.wide(config),
        nozzle_pressure_bar=s.nozzle_pressure_bar,
        quality_limit_kg_per_kg=s.config.quality_limit_kg_per_kg,
    )


def nominal_map(theta: dict[str, float] | None = None) -> Any:
    from twinlink_ml.hybrid_model.maps import GreyBoxMap
    from twinlink_ml.hybrid_model.refit import refit_config

    s = setup()
    config = s.steady_nominal
    if theta is not None:
        config = refit_config(config, list(theta), list(theta.values()))
    return GreyBoxMap(
        config,
        s.wide(config),
        nozzle_pressure_bar=s.nozzle_pressure_bar,
        quality_limit_kg_per_kg=s.config.quality_limit_kg_per_kg,
    )


def fixed_inputs() -> Any:
    from twinlink_ml.surrogate_env.model import FixedInputs

    s = setup()
    return FixedInputs(
        nozzle_pressure_bar=s.nozzle_pressure_bar,
        spray_rate_kg_per_h=s.steady_nominal.actions.spray_rate_kg_per_h.nominal,
        suspension_solid_fraction=s.steady_nominal.disturbances.suspension_solid_fraction.nominal,
    )


def training_records(name: str) -> dict[str, dict[str, Any]]:
    path = ROOT / "artifacts" / "runs" / f"{name}.jsonl"
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "task":
                out[row["key"]] = row
    return out


def load_network(record: dict[str, Any]) -> Any:
    from twinlink_ml.surrogate_env.model import GruFixedPointModel

    path = ROOT / record["checkpoint"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint {path} does not match its record")
    return GruFixedPointModel.load(path, fixed_inputs())


def build_map(model: str, draw: int, seed: int, n: int, records: dict[str, dict[str, Any]]) -> Any:
    from twinlink_ml.hybrid_model.maps import GruMap, HybridMap

    def rec(role: str) -> dict[str, Any]:
        return records[f"d{draw:02d}_s{seed}_n{n}_{role}"]

    if model == "M0":
        return nominal_map()
    if model == "M1":
        return GruMap(load_network(rec("M1")))
    if model == "M2":
        r = rec("M2")["refit"]
        return nominal_map(dict(zip(r["names"], r["theta"], strict=True)))
    if model == "M3":
        return HybridMap(nominal_map(), [load_network(rec("M3"))])
    if model == "M4":
        members = ["M3"] + [f"M4m{j}" for j in range(1, setup().config.ensemble_members)]
        return HybridMap(nominal_map(), [load_network(rec(m)) for m in members])
    raise ValueError(model)


def append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_default) + "\n")


def _default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "job":
                keys.add(row["key"])
    return keys


def finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


# --- part oracle --------------------------------------------------------------------------------


def oracle_job(draw: int) -> dict[str, Any]:
    import run_arch_rerun_worker as rerun_worker

    from twinlink_ml.hybrid_model.coverage import chebyshev_distance
    from twinlink_ml.hybrid_model.grid_rto import CaseEvaluator, grid_zoom
    from twinlink_ml.hybrid_model.residual import run_rows
    from twinlink_ml.hybrid_model.study import case_arrays

    warnings.filterwarnings("ignore")
    started = time.perf_counter()
    s = setup()
    cfg = s.config
    cases = s.draw_cases(draw)
    contexts, wide = case_arrays(cases)
    tmap = truth_map(draw)
    ev = CaseEvaluator(tmap, contexts, wide)
    idx = np.arange(len(cases))
    limit = cfg.quality_limit_kg_per_kg
    actions, j = grid_zoom(
        ev, idx, np.full(len(cases), limit), np.zeros(len(cases)), s.bounds, cfg.grid
    )
    values = ev.values(idx, actions)
    rows = []
    for c, case in enumerate(cases):
        full = tmap.solve(actions[c : c + 1], contexts[c : c + 1], wide=bool(wide[c]))
        rows.append(
            {
                **case.as_dict(),
                "action": actions[c].tolist(),
                "objective": float(j[c]),
                "energy_kw": finite(values["energy_kw"][c]),
                "moisture_kg_per_kg": finite(values["moisture_kg_per_kg"][c]),
                "bed_temperature_celsius": finite(values["bed_temperature_celsius"][c]),
                "valid": bool(values["valid"][c]),
                "feasible_full_check": bool(full.feasible[0]),
                "normalized_violation": finite(full.normalized_violation[0]),
            }
        )
    # classification (b) of the interior cases for every seed and n_train
    lower = np.array([s.bounds[0][0], s.bounds[1][0], 5.0, 0.25])
    d = s.data_nominal.disturbances
    lower[2], lower[3] = d.ambient_temperature_celsius.minimum, d.ambient_relative_humidity.minimum
    upper = np.array(
        [
            s.bounds[0][1],
            s.bounds[1][1],
            d.ambient_temperature_celsius.maximum,
            d.ambient_relative_humidity.maximum,
        ]
    )
    points = np.column_stack([actions, contexts])
    coverage: dict[str, Any] = {}
    spec = rerun_worker.arm_spec("N")
    for seed in cfg.master_seeds:
        pool = rerun_worker._generate(s.data_truth(draw), spec, seed)
        for n in cfg.n_train:
            r = run_rows(pool, list(pool.split.subset(n).train))
            samples = np.concatenate(
                [pool.trajectory.actions[r][..., :2], pool.trajectory.disturbances[r][..., :2]],
                axis=-1,
            ).reshape(-1, 4)
            distance = chebyshev_distance(points, samples, lower, upper)
            inside = (distance <= cfg.coverage_threshold) & ~wide
            coverage[f"s{seed}_n{n}"] = {
                "distance": distance.tolist(),
                "inside": inside.tolist(),
                "excitation_fingerprint": pool.fingerprints["excitation/fingerprint"],
            }
    return {
        "key": f"oracle_d{draw:02d}",
        "draw": draw,
        "cases": rows,
        "coverage": coverage,
        "coverage_bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
        "n_predicted": ev.n_predicted,
        "elapsed_s": time.perf_counter() - started,
    }


def discretisation_job(i: int) -> dict[str, Any]:
    from twinlink_ml.hybrid_model.grid_rto import CaseEvaluator, grid_zoom, objective
    from twinlink_ml.hybrid_model.study import case_arrays
    from twinlink_ml.surrogate_env import PointEvaluation, multistart_minimize

    warnings.filterwarnings("ignore")
    started = time.perf_counter()
    s = setup()
    cfg = s.config
    draw, slot = 2 * i, i % 5
    case = s.draw_cases(draw)[slot]
    contexts, wide = case_arrays([case])
    tmap = truth_map(draw)
    limit = cfg.quality_limit_kg_per_kg
    ev = CaseEvaluator(tmap, contexts, wide)
    grid_action, grid_j = grid_zoom(
        ev, np.array([0]), np.array([limit]), np.zeros(1), s.bounds, cfg.grid
    )

    def evaluate(t_in: float, u_gas: float) -> PointEvaluation:
        p = tmap.predict(np.array([[t_in, u_gas]]), contexts, wide=False)
        values = {
            "energy_kw": p.energy_kw,
            "moisture_kg_per_kg": p.moisture_kg_per_kg,
            "moisture_std_kg_per_kg": np.zeros(1),
            "bed_temperature_celsius": p.bed_temperature_celsius,
            "valid": p.valid,
        }
        j = float(objective(values, np.array([limit]), np.zeros(1))[0])
        return PointEvaluation(
            j,
            float(p.energy_kw[0]),
            float(p.moisture_kg_per_kg[0]),
            float(p.bed_temperature_celsius[0]),
            bool(p.valid[0]),
        )

    nm = multistart_minimize(evaluate, [s.bounds[0], s.bounds[1]], s.fixed_point)
    g = ev.values(np.array([0]), grid_action)
    e_grid, x_grid = float(g["energy_kw"][0]), float(g["moisture_kg_per_kg"][0])
    return {
        "key": f"discretisation_{i}",
        "draw": draw,
        "slot": slot,
        "case": case.as_dict(),
        "grid": {
            "action": grid_action[0].tolist(),
            "objective": float(grid_j[0]),
            "energy_kw": e_grid,
            "moisture_kg_per_kg": x_grid,
            "slack_kg_per_kg": limit - x_grid,
        },
        "nelder_mead": {
            "action": [nm["inlet_temperature_celsius"], nm["gas_velocity_m_per_s"]],
            "objective": nm["objective"],
            "energy_kw": nm["energy_kw"],
            "moisture_kg_per_kg": nm["moisture_kg_per_kg"],
            "slack_kg_per_kg": limit - nm["moisture_kg_per_kg"],
            "n_evaluations": nm["n_evaluations"],
        },
        "energy_difference_relative": (e_grid - nm["energy_kw"]) / nm["energy_kw"],
        "objective_difference_relative": (float(grid_j[0]) - nm["objective"]) / nm["objective"],
        "slack_difference_kg_per_kg": (limit - x_grid) - (limit - nm["moisture_kg_per_kg"]),
        "elapsed_s": time.perf_counter() - started,
    }


# --- part margins -------------------------------------------------------------------------------


def load_oracle(prefix: str) -> dict[int, dict[str, Any]]:
    path = ROOT / "artifacts" / "runs" / f"{prefix}_oracle.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "job" and row["key"].startswith("oracle_d"):
                out[row["draw"]] = row
    return out


def margin_job(
    prefix: str, training: str, model: str, seed: int, n: int, draws: tuple[int, ...]
) -> dict[str, Any]:
    import torch

    from twinlink_ml.hybrid_model.grid_rto import (
        CaseEvaluator,
        _box,
        grid_zoom,
        search_parameter,
        truth_holds,
    )
    from twinlink_ml.hybrid_model.study import case_arrays

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    started = time.perf_counter()
    s = setup()
    cfg = s.config
    limit = cfg.quality_limit_kg_per_kg
    oracle = load_oracle(prefix)
    records = training_records(training)
    is_k = model == "M4"
    settings = cfg.margin_k if is_k else cfg.margin_constant

    def limits_ks(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if is_k:
            return np.full(p.shape, limit), p
        return limit - p, np.zeros(p.shape)

    per_draw: list[dict[str, Any]] = []
    for draw in draws:
        cases = s.draw_cases(draw)
        contexts, wide = case_arrays(cases)
        inside = np.array(oracle[draw]["coverage"][f"s{seed}_n{n}"]["inside"], dtype=bool)
        model_ev = CaseEvaluator(build_map(model, draw, seed, n, records), contexts, wide)
        truth_ev = CaseEvaluator(truth_map(draw), contexts, wide)
        inside_idx = np.flatnonzero(inside)

        def hold_at(
            rows: np.ndarray,
            p: np.ndarray,
            _m: Any = model_ev,
            _t: Any = truth_ev,
            _i: np.ndarray = inside_idx,
        ) -> np.ndarray:
            lim, ks = limits_ks(p)
            actions, _ = grid_zoom(_m, _i[rows], lim, ks, s.bounds, cfg.grid)
            return truth_holds(_t.values(_i[rows], actions), limit)

        result = search_parameter(hold_at, inside_idx.size, settings)
        per_draw.append(
            {
                "draw": draw,
                "cases": cases,
                "contexts": contexts,
                "wide": wide,
                "inside": inside,
                "model_ev": model_ev,
                "truth_ev": truth_ev,
                "needed": dict(zip(inside_idx.tolist(), result.needed.tolist(), strict=True)),
                "n_solves": dict(zip(inside_idx.tolist(), result.n_solves.tolist(), strict=True)),
            }
        )

    needed_all = np.array(
        [v for entry in per_draw for v in entry["needed"].values()], dtype=np.float64
    )
    n_inside = needed_all.size
    ordered = np.sort(needed_all)
    rank = max(1, math.ceil(cfg.holding_share * n_inside)) if n_inside else 0
    parameter = float(ordered[rank - 1]) if n_inside else math.nan
    candidates = [
        parameter,
        *sorted({float(v) for v in ordered if np.isfinite(v) and v > parameter}),
    ]

    def evaluate_all(p: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in per_draw:
            k = len(entry["cases"])
            idx = np.arange(k)
            pv = np.full(k, p)
            lim, ks = limits_ks(pv)
            actions, j = grid_zoom(entry["model_ev"], idx, lim, ks, s.bounds, cfg.grid)
            pred = entry["model_ev"].values(idx, actions)
            tru = entry["truth_ev"].values(idx, actions)
            holds = truth_holds(tru, limit)
            for c, case in enumerate(entry["cases"]):
                orc = oracle[entry["draw"]]["cases"][c]
                e_true = tru["energy_kw"][c]
                rows.append(
                    {
                        "draw": entry["draw"],
                        "slot": case.slot,
                        "kind": case.kind,
                        "inside": bool(entry["inside"][c]),
                        "action": actions[c].tolist(),
                        "objective": float(j[c]),
                        "predicted": {f: finite(pred[f][c]) for f in pred if f != "valid"}
                        | {"valid": bool(pred["valid"][c])},
                        "truth": {
                            f: finite(tru[f][c])
                            for f in ("energy_kw", "moisture_kg_per_kg", "bed_temperature_celsius")
                        }
                        | {"valid": bool(tru["valid"][c])},
                        "holds": bool(holds[c]),
                        "bed_violated": bool(
                            tru["valid"][c] and tru["bed_temperature_celsius"][c] > 80.0 + 0.05
                        ),
                        "oracle_energy_kw": orc["energy_kw"],
                        "energy_excess_relative": finite(
                            (e_true - orc["energy_kw"]) / orc["energy_kw"]
                        )
                        if orc["energy_kw"]
                        else None,
                        "moisture_bias_at_action": finite(
                            pred["moisture_kg_per_kg"][c] - tru["moisture_kg_per_kg"][c]
                        ),
                        # constant margin: m; ensemble: k * std at the chosen point
                        "effective_margin": finite(
                            (limit - lim[c])
                            + (p if is_k else 0.0) * pred["moisture_std_kg_per_kg"][c]
                        )
                        if pred["valid"][c]
                        else None,
                    }
                )
        return rows

    chosen = parameter
    raised = 0
    final_rows: list[dict[str, Any]] = []
    reached = bool(n_inside) and np.isfinite(parameter)
    if reached:
        for p in candidates:
            final_rows = evaluate_all(p)
            share = np.mean([r["holds"] for r in final_rows if r["inside"]])
            chosen = p
            if share >= cfg.holding_share:
                break
            raised += 1
    else:
        # not reached: report the cases at the upper end of the search
        final_rows = evaluate_all(settings.upper)
        chosen = settings.upper

    # box-wide moisture bias on the coarse grid (model minus truth, valid points of both)
    coarse = _box(s.bounds, cfg.grid.coarse)
    bias_rows = []
    for entry in per_draw:
        for c, case in enumerate(entry["cases"]):
            cidx = np.full(coarse.shape[0], c)
            pm = entry["model_ev"].values(cidx, coarse)
            pt = entry["truth_ev"].values(cidx, coarse)
            ok = pm["valid"] & pt["valid"]
            diff = pm["moisture_kg_per_kg"][ok] - pt["moisture_kg_per_kg"][ok]
            bias_rows.append(
                {
                    "draw": entry["draw"],
                    "slot": case.slot,
                    "kind": case.kind,
                    "inside": bool(entry["inside"][c]),
                    "n_valid": int(ok.sum()),
                    "n_model_invalid": int((~pm["valid"]).sum()),
                    "bias": finite(diff.mean()) if diff.size else None,
                    "rmse": finite(np.sqrt(np.mean(diff**2))) if diff.size else None,
                }
            )
    needed_rows = [
        {"draw": e["draw"], "slot": int(c), "needed": finite(v), "n_solves": e["n_solves"][c]}
        for e in per_draw
        for c, v in e["needed"].items()
    ]
    return {
        "key": f"margins_{model}_s{seed}_n{n}",
        "model": model,
        "master_seed": seed,
        "n_train": n,
        "draws": list(draws),
        "parameter_kind": "k" if is_k else "margin",
        "search": {"upper": settings.upper, "tolerance": settings.tolerance},
        "n_inside": int(n_inside),
        "quantile_rank": rank,
        "parameter_quantile": finite(parameter),
        "parameter": finite(chosen) if reached else None,
        "reached": reached,
        "raised": raised,
        "needed": needed_rows,
        "cases": final_rows,
        "coarse_bias": bias_rows,
        "n_predicted_model": int(sum(e["model_ev"].n_predicted for e in per_draw)),
        "n_predicted_truth": int(sum(e["truth_ev"].n_predicted for e in per_draw)),
        "elapsed_s": time.perf_counter() - started,
    }


# --- part errors --------------------------------------------------------------------------------


def _errors(
    predicted: np.ndarray,
    truth: np.ndarray,
    names: list[str],
    mask: np.ndarray,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_windows": int(mask.sum())}
    if not mask.any():
        return out
    for k, target in enumerate(names):
        for h in horizons:
            err = predicted[mask, h - 1, k] - truth[mask, h - 1, k]
            out[f"rmse/{target}/h{h}"] = float(np.sqrt(np.mean(err**2)))
            out[f"bias/{target}/h{h}"] = float(np.mean(err))
    return out


def errors_job(training: str, draw: int, seed: int) -> dict[str, Any]:
    import run_arch_rerun_worker as rerun_worker
    import torch

    from twinlink_ml.data.splits import SplitName
    from twinlink_ml.data.windows import TrajectoryWindowDataset
    from twinlink_ml.evaluation import ValidationPolicy, build_datasets
    from twinlink_ml.hybrid_model.coverage import chebyshev_distance
    from twinlink_ml.hybrid_model.refit import refit_config
    from twinlink_ml.hybrid_model.residual import replay, residual_data, run_rows
    from twinlink_ml.utils.checkpoints import load_checkpoint

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    started = time.perf_counter()
    s = setup()
    cfg = s.config
    records = training_records(training)
    spec = rerun_worker.arm_spec("N")
    hspec = rerun_worker.holdout_spec(spec)
    truth_cfg = s.data_truth(draw)
    wide_truth_cfg = s.wide(truth_cfg)
    pools = {
        "excited": (rerun_worker._generate(truth_cfg, spec, seed), s.data_nominal),
        "constant_holdout": (
            rerun_worker._generate(truth_cfg, hspec, cfg.holdout_seed),
            s.data_nominal,
        ),
        "constant_holdout_wide": (
            rerun_worker._generate(wide_truth_cfg, hspec, cfg.holdout_seed),
            s.wide(s.data_nominal),
        ),
    }
    own = pools["excited"][0]
    residual_own = residual_data(own, replay(own, s.data_nominal))
    d = s.data_nominal.disturbances
    lower = np.array(
        [
            s.bounds[0][0],
            s.bounds[1][0],
            d.ambient_temperature_celsius.minimum,
            d.ambient_relative_humidity.minimum,
        ]
    )
    upper = np.array(
        [
            s.bounds[0][1],
            s.bounds[1][1],
            d.ambient_temperature_celsius.maximum,
            d.ambient_relative_humidity.maximum,
        ]
    )
    horizons = cfg.error_horizons
    out: dict[str, Any] = {
        "key": f"errors_d{draw:02d}_s{seed}",
        "draw": draw,
        "master_seed": seed,
        "cells": [],
    }
    for n in cfg.n_train:
        if f"d{draw:02d}_s{seed}_n{n}_M1" not in records:
            continue
        train_truth, _, _ = build_datasets(own, n, policy=ValidationPolicy.FIXED_POOL)
        train_res, _, _ = build_datasets(residual_own, n, policy=ValidationPolicy.FIXED_POOL)
        rows = run_rows(own, list(own.split.subset(n).train))
        samples = np.concatenate(
            [own.trajectory.actions[rows][..., :2], own.trajectory.disturbances[rows][..., :2]],
            axis=-1,
        ).reshape(-1, 4)
        theta = records[f"d{draw:02d}_s{seed}_n{n}_M2"]["refit"]
        nets: dict[str, Any] = {}
        for role in ["M1", "M3"] + [f"M4m{j}" for j in range(1, cfg.ensemble_members)]:
            key = f"d{draw:02d}_s{seed}_n{n}_{role}"
            if key in records:
                net, _ = load_checkpoint(ROOT / records[key]["checkpoint"])
                nets[role] = net
        cell: dict[str, Any] = {"n_train": n, "test_sets": {}}
        for set_name, (pool, nominal_cfg) in pools.items():
            res_nom = residual_data(pool, replay(pool, nominal_cfg))
            refit_cfg = refit_config(nominal_cfg, theta["names"], theta["theta"])
            res_fit = residual_data(pool, replay(pool, refit_cfg))
            which = SplitName.TEST
            ds_truth = TrajectoryWindowDataset.from_trajectory(
                pool.trajectory, pool.split, which, spec.window, scalers=train_truth.scalers
            )
            ds_res = TrajectoryWindowDataset.from_trajectory(
                res_nom.trajectory, res_nom.split, which, spec.window, scalers=train_res.scalers
            )
            ds_fit = TrajectoryWindowDataset.from_trajectory(
                res_fit.trajectory, res_fit.split, which, spec.window, scalers=train_res.scalers
            )
            names = list(ds_truth.target_names)
            b_truth = ds_truth.batch(range(ds_truth.n_windows))
            b_res = ds_res.batch(range(ds_res.n_windows))
            b_fit = ds_fit.batch(range(ds_fit.n_windows))
            y_truth = ds_truth.target_scaler.inverse(b_truth.future_targets_truth)
            r_nom = ds_res.target_scaler.inverse(b_res.future_targets_truth)
            r_fit = ds_fit.target_scaler.inverse(b_fit.future_targets_truth)
            preds: dict[str, tuple[np.ndarray, np.ndarray]] = {
                "M0": (np.zeros_like(r_nom), r_nom),
                "M2": (np.zeros_like(r_fit), r_fit),
            }
            if "M1" in nets:
                preds["M1"] = (ds_truth.target_scaler.inverse(nets["M1"].predict(b_truth)), y_truth)
            res_preds = [
                ds_res.target_scaler.inverse(nets[r].predict(b_res))
                for r in ["M3"] + [f"M4m{j}" for j in range(1, cfg.ensemble_members)]
                if r in nets
            ]
            if res_preds:
                preds["M3"] = (res_preds[0], r_nom)
            if len(res_preds) == cfg.ensemble_members:
                preds["M4"] = (np.mean(res_preds, axis=0), r_nom)
            # run classification
            # window run index -> row of the trajectory batch (dataset rows follow the split order)
            test_rows = run_rows(pool, list(pool.split.as_mapping[which]))
            run_index = np.asarray(test_rows, dtype=int)[np.asarray(b_truth.run_index, dtype=int)]
            run_ids = sorted(set(run_index.tolist()))
            traj = pool.trajectory
            outside_run = {}
            for ri in run_ids:
                a = traj.actions[ri][..., :2]
                w = traj.disturbances[ri][..., :2]
                if set_name == "constant_holdout_wide":
                    outside_run[ri] = bool(
                        np.any(w[:, 0] < lower[2])
                        or np.any(w[:, 0] > upper[2])
                        or np.any(w[:, 1] < lower[3])
                        or np.any(w[:, 1] > upper[3])
                    )
                else:
                    point = np.concatenate([a.mean(axis=0), w.mean(axis=0)])[None, :]
                    outside_run[ri] = bool(
                        chebyshev_distance(point, samples, lower, upper)[0] > cfg.coverage_threshold
                    )
            outside = np.array([outside_run[ri] for ri in run_index])
            entry: dict[str, Any] = {
                "n_runs": len(run_ids),
                "n_runs_outside": int(sum(outside_run.values())),
                "models": {},
            }
            for m, (p, y) in preds.items():
                entry["models"][m] = {
                    "all": _errors(p, y, names, np.ones(len(run_index), bool), horizons),
                    "inside": _errors(p, y, names, ~outside, horizons),
                    "outside": _errors(p, y, names, outside, horizons),
                }
            cell["test_sets"][set_name] = entry
        out["cells"].append(cell)
    out["elapsed_s"] = time.perf_counter() - started
    return out


# --- driver -------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("oracle", "margins", "errors"), required=True)
    parser.add_argument("--prefix", default="hybrid_model_v1")
    parser.add_argument("--training", default="hybrid_model_training_v1")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-models", nargs="*", default=None)
    parser.add_argument("--only-n", type=int, nargs="*", default=None)
    parser.add_argument("--only-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--only-draws", type=int, nargs="*", default=None)
    args = parser.parse_args()

    from _common import RUNS_DIR, provenance, require_clean_tree

    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    output = RUNS_DIR / f"{args.prefix}_{args.part}.jsonl"
    # The input artifacts (training records, the oracle of the same prefix) may still be
    # uncommitted (size); they are left out of the clean-tree check and pinned by sha256 instead.
    inputs = [RUNS_DIR / f"{args.training}.jsonl"]
    if args.part != "oracle":
        inputs.append(RUNS_DIR / f"{args.prefix}_oracle.jsonl")
    inputs = [path for path in inputs if path.exists()]
    require_clean_tree(ignore_paths=[output, *inputs])
    if output.exists() and not args.resume:
        raise SystemExit(f"{output} exists -- use --resume or a new prefix")
    done = done_keys(output)
    s = setup()
    cfg = s.config
    seeds = args.only_seeds or list(cfg.master_seeds)
    sizes = args.only_n or list(cfg.n_train)
    draws = args.only_draws if args.only_draws is not None else list(range(len(s.draws)))
    jobs: list[tuple[str, tuple[Any, ...]]] = []
    if args.part == "oracle":
        jobs += [(f"discretisation_{i}", ("disc", i)) for i in range(cfg.discretisation_n_cases)]
        jobs += [(f"oracle_d{d:02d}", ("oracle", d)) for d in draws]
    elif args.part == "margins":
        models = args.only_models or list(MODELS)
        # most expensive first: M4, then the single networks, then the grey-box models
        order = {"M4": 0, "M3": 1, "M1": 2, "M2": 3, "M0": 4}
        for m in sorted(models, key=order.__getitem__):
            for n in sorted(sizes, reverse=True):
                for seed in seeds:
                    jobs.append((f"margins_{m}_s{seed}_n{n}", ("margins", m, seed, n)))
    else:
        jobs += [(f"errors_d{d:02d}_s{seed}", ("errors", d, seed)) for d in draws for seed in seeds]
    jobs = [j for j in jobs if j[0] not in done]
    append(
        output,
        {
            "kind": "header",
            "provenance": provenance("not used", ignore_paths=[output, *inputs]),
            "part": args.part,
            "training": args.training,
            "inputs_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
            },
            "config": cfg.raw,
            "n_jobs": len(jobs),
            "selection": {"models": args.only_models, "n": sizes, "seeds": seeds, "draws": draws},
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    print(f"{len(jobs)} jobs ({len(done)} done before)", flush=True)

    def submit(pool: ProcessPoolExecutor, spec: tuple[Any, ...]) -> Any:
        if spec[0] == "disc":
            return pool.submit(discretisation_job, spec[1])
        if spec[0] == "oracle":
            return pool.submit(oracle_job, spec[1])
        if spec[0] == "margins":
            return pool.submit(margin_job, args.prefix, args.training, *spec[1:], tuple(draws))
        return pool.submit(errors_job, args.training, *spec[1:])

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {submit(pool, spec): key for key, spec in jobs}
        finished = 0
        while pending:
            complete, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in complete:
                key = pending.pop(future)
                record = {"kind": "job", **future.result()}
                append(output, record)
                finished += 1
                print(
                    f"[{finished}/{len(jobs)}] {key} {record['elapsed_s']:.0f} s (wall {time.time() - started:.0f} s)",
                    flush=True,
                )
    append(output, {"kind": "footer", "runtime_s": time.time() - started})
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    raise SystemExit(main())
