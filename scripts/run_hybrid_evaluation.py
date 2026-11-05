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
``margins``  per cell (model, master seed, ``n_train``), models M0 (prior knowledge), M1 (GRU), M2
             (refit), M3 (hybrid), M4 (hybrid ensemble), split per draw since the failed run of
             a343840 (decision of the project lead, 2026-11-05; the computation is unchanged,
             ``hybrid_model/margin_tasks.py``): one SEARCH task per (cell, draw) -- per case INSIDE
             the training range the needed constraint parameter by bisection (M0-M3 a constant
             margin m, M4 the factor k of ``mean + k std``; a draw without any inside case has no
             search) and the box-wide moisture bias on the coarse grid; once all 20 draws of a cell
             are searched, the parameter of the cell = the 95 % quantile over its pooled inside
             cases; then one EVALUATE task per (cell, draw, round) -- EVERY case (inside, outside
             (b), extrapolation (a)) optimised at that parameter and evaluated at the truth; raise
             rule of Befund 111 over the rounds. Every task record is written when the task ends
             (``--resume`` continues), an exception is recorded per task (its cell stops, the others
             go on), the worker's pid and memory peak so far in every record and log line; the
             cell record (format of the former per-cell job) when the cell is done. Workers are
             reused (a fresh process costs about 40-70 s of imports from the network drive).
                                                                -> ``<prefix>_margins_per_draw``
``errors``   per (draw, seed): one-step (h1) and rollout (h36) RMSE and bias in physical units of
             every model and ``n_train`` on the excited test set of the pool, on the constant-run
             holdout (seed 999; runs split by (b)) and on a constant-run holdout with the widened
             ambient ranges (runs outside the data box = (a)).    -> ``<prefix>_errors``

``--dry-run minimal`` (1 bisection step, 3 x 3 grid, no zoom) runs every task at minimal cost
(edge cases such as empty case sets show before the main run); ``--dry-run coarse`` the same with
the full coarse grid (the largest prediction batch, hence the memory peak). Dry runs write to a
name starting with ``_`` (gitignored). The input artifacts training records and oracle may be
uncommitted (``clean_tree_exceptions``); their sha256 is in the header.

    tools/run_here.sh scripts/run_hybrid_evaluation.py --part oracle --workers 8
    tools/run_here.sh scripts/run_hybrid_evaluation.py --part margins --workers 8
    tools/run_here.sh scripts/run_hybrid_evaluation.py --part errors --workers 8
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
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
_SETUP_DIGEST: str | None = None
TASK_SEED = 0  # global random sources at the start of every task (none is drawn from; audit)


def setup() -> Any:
    """The study setup (configs, draws, cases), built once per process, checked every task."""
    global _SETUP, _SETUP_DIGEST
    if _SETUP is None:
        from twinlink_ml.hybrid_model.study import build_setup

        _SETUP = build_setup()
        _SETUP_DIGEST = _digest(_SETUP)
    return _SETUP


def _digest(obj: object) -> str:
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


def reset_task_state() -> None:
    """Start of every task in a (possibly reused) worker.

    The global random sources (``random``, ``numpy``, ``torch``) are set to ``TASK_SEED`` and
    torch to one thread; the cached setup must be unchanged since it was built (sha256 of its
    pickle), else the task fails. Maps, networks (from their checkpoints) and evaluators are built
    in every task. Audit of 2026-11-05: nothing in the oracle, margins or errors tasks draws from a
    global random source (data generation uses its own seeded generators), there is no
    module-level cache besides the setup, and the networks predict in eval mode.
    """
    import random

    import torch

    random.seed(TASK_SEED)
    np.random.seed(TASK_SEED)  # noqa: NPY002 - resetting the global state is the purpose
    torch.manual_seed(TASK_SEED)
    torch.set_num_threads(1)
    if _SETUP is not None and _digest(_SETUP) != _SETUP_DIGEST:
        raise RuntimeError("the cached study setup changed between tasks")


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


def load_oracle(name: str) -> dict[int, dict[str, Any]]:
    path = ROOT / "artifacts" / "runs" / f"{name}.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "job" and row["key"].startswith("oracle_d"):
                out[row["draw"]] = row
    return out


@dataclass(frozen=True)
class MarginRun:
    """What every margins task needs besides its cell and draw (settings of this run)."""

    oracle: str
    training: str
    grid: Any  # GridSettings
    constant: Any  # BisectionSettings of M0-M3
    ensemble_k: Any  # BisectionSettings of M4
    max_batch: int | None = None  # points per call of a map (memory; predictions unchanged)

    def bisection(self, model: str) -> Any:
        return self.ensemble_k if model == "M4" else self.constant


def search_key(model: str, seed: int, n: int, draw: int) -> str:
    return f"search_{model}_s{seed}_n{n}_d{draw:02d}"


def evaluate_key(model: str, seed: int, n: int, round_: int, draw: int) -> str:
    return f"evaluate_{model}_s{seed}_n{n}_r{round_}_d{draw:02d}"


def cell_key(model: str, seed: int, n: int) -> str:
    return f"margins_{model}_s{seed}_n{n}"


def _draw_context(run: MarginRun, model: str, seed: int, n: int, draw: int) -> tuple[Any, ...]:
    from twinlink_ml.hybrid_model.grid_rto import CaseEvaluator
    from twinlink_ml.hybrid_model.study import case_arrays

    s = setup()
    cases = s.draw_cases(draw)
    contexts, wide = case_arrays(cases)
    oracle = load_oracle(run.oracle)[draw]
    inside = np.array(oracle["coverage"][f"s{seed}_n{n}"]["inside"], dtype=bool)
    records = training_records(run.training)
    model_map = build_map(model, draw, seed, n, records)
    model_ev = CaseEvaluator(model_map, contexts, wide, max_batch=run.max_batch)
    truth_ev = CaseEvaluator(truth_map(draw), contexts, wide, max_batch=run.max_batch)
    return s, cases, oracle, inside, model_ev, truth_ev


def margin_search_task(run: MarginRun, model: str, seed: int, n: int, draw: int) -> dict[str, Any]:
    """Needed parameter per inside case of one draw (none without any) and the coarse-grid bias."""
    import torch

    from twinlink_ml.hybrid_model.margin_tasks import coarse_bias_draw, search_draw

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    started = time.perf_counter()
    s, cases, _, inside, model_ev, truth_ev = _draw_context(run, model, seed, n, draw)
    result = search_draw(
        model_ev,
        truth_ev,
        inside,
        limit=s.config.quality_limit_kg_per_kg,
        is_k=model == "M4",
        bounds=s.bounds,
        grid=run.grid,
        settings=run.bisection(model),
    )
    bias = coarse_bias_draw(
        model_ev, truth_ev, cases, inside, draw=draw, bounds=s.bounds, grid=run.grid
    )
    return {
        "key": search_key(model, seed, n, draw),
        "task": "search",
        "model": model,
        "master_seed": seed,
        "n_train": n,
        "draw": draw,
        "n_inside": len(result.needed),
        # needed: None = not reached within the upper end (inf)
        "needed": [
            {"draw": draw, "slot": int(c), "needed": finite(v), "n_solves": result.n_solves[c]}
            for c, v in result.needed.items()
        ],
        "coarse_bias": bias,
        "n_predicted_model": model_ev.n_predicted,
        "n_predicted_truth": truth_ev.n_predicted,
        "elapsed_s": time.perf_counter() - started,
    }


def margin_evaluate_task(
    run: MarginRun, model: str, seed: int, n: int, draw: int, round_: int, parameter: float
) -> dict[str, Any]:
    """Every case of one draw optimised at the cell's parameter of this round, at the truth."""
    import torch

    from twinlink_ml.hybrid_model.margin_tasks import evaluate_draw

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    started = time.perf_counter()
    s, cases, oracle, inside, model_ev, truth_ev = _draw_context(run, model, seed, n, draw)
    rows = evaluate_draw(
        model_ev,
        truth_ev,
        cases,
        inside,
        oracle["cases"],
        parameter,
        draw=draw,
        limit=s.config.quality_limit_kg_per_kg,
        is_k=model == "M4",
        bounds=s.bounds,
        grid=run.grid,
    )
    return {
        "key": evaluate_key(model, seed, n, round_, draw),
        "task": "evaluate",
        "model": model,
        "master_seed": seed,
        "n_train": n,
        "draw": draw,
        "round": round_,
        "parameter": parameter,
        "cases": rows,
        "n_predicted_model": model_ev.n_predicted,
        "n_predicted_truth": truth_ev.n_predicted,
        "elapsed_s": time.perf_counter() - started,
    }


@dataclass
class Cell:
    """Scheduling state of one cell (model, master seed, ``n_train``) of the margins part."""

    model: str
    seed: int
    n: int
    draws: tuple[int, ...]
    search: dict[int, dict[str, Any]] = field(default_factory=dict)
    rounds: list[dict[int, dict[str, Any]]] = field(default_factory=list)
    parameter: Any = None  # CellParameter once every draw is searched
    blocked: bool = False
    finished: bool = False

    @property
    def key(self) -> str:
        return cell_key(self.model, self.seed, self.n)

    def needed(self) -> list[float]:
        # JSON None = not reached = inf (finite() on the way out)
        return [
            math.inf if row["needed"] is None else row["needed"]
            for d in self.draws
            for row in self.search[d]["needed"]
        ]

    def shares(self) -> list[float]:
        return [
            float(np.mean([c["holds"] for d in self.draws for c in r[d]["cases"] if c["inside"]]))
            if any(c["inside"] for d in self.draws for c in r[d]["cases"])
            else math.nan
            for r in self.rounds
            if len(r) == len(self.draws)
        ]


def assemble_cell(cell: Cell, run: MarginRun, decision: Any) -> dict[str, Any]:
    """The cell record in the format of the former per-cell job (read by the analysis)."""
    p = cell.parameter
    final = cell.rounds[-1]
    settings = run.bisection(cell.model)
    tasks = [cell.search[d] for d in cell.draws] + [r[d] for r in cell.rounds for d in cell.draws]
    return {
        "key": cell.key,
        "model": cell.model,
        "master_seed": cell.seed,
        "n_train": cell.n,
        "draws": list(cell.draws),
        "parameter_kind": "k" if cell.model == "M4" else "margin",
        "search": {"upper": settings.upper, "tolerance": settings.tolerance},
        "n_inside": p.n_inside,
        "quantile_rank": p.rank,
        "parameter_quantile": finite(p.quantile),
        "parameter": finite(decision.parameter) if p.reached else None,
        "reached": p.reached,
        "raised": decision.raised,
        "rounds": [
            {"round": i, "parameter": r[cell.draws[0]]["parameter"], "share_inside": s}
            for i, (r, s) in enumerate(zip(cell.rounds, cell.shares(), strict=True))
        ],
        "needed": [row for d in cell.draws for row in cell.search[d]["needed"]],
        "cases": [c for d in cell.draws for c in final[d]["cases"]],
        "coarse_bias": [b for d in cell.draws for b in cell.search[d]["coarse_bias"]],
        # summed over the tasks (each task predicts with its own cache)
        "n_predicted_model": int(sum(t["n_predicted_model"] for t in tasks)),
        "n_predicted_truth": int(sum(t["n_predicted_truth"] for t in tasks)),
        "elapsed_s": float(sum(t["elapsed_s"] for t in tasks)),
        "peak_commit_mb": max(t["memory"].get("peak_commit_mb", math.nan) for t in tasks),
        "n_tasks": len(tasks),
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


def clean_tree_exceptions(
    runs_dir: Path, output: Path, training: str, oracle: str | None
) -> list[Path]:
    """Paths the clean-tree check leaves out: the run's own output and exactly two inputs.

    The inputs that may be uncommitted are the training records (12.6 MB, above the size rule)
    and the oracle; their sha256 is pinned in the header instead. ``status_is_dirty`` exempts
    only these exact UNTRACKED files; any other untracked or modified file still counts.
    """
    paths = [output, runs_dir / f"{training}.jsonl"]
    if oracle is not None:
        paths.append(runs_dir / f"{oracle}.jsonl")
    return paths


def peak_memory_mb() -> dict[str, float]:
    """Peak working set and peak commit (private bytes) of this process so far, and the current
    values (end of a task; a steady rise over the tasks of a worker would point to a leak), in MB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [  # noqa: RUF012
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return {}
        return {
            "peak_working_set_mb": counters.PeakWorkingSetSize / 2**20,
            "peak_commit_mb": counters.PeakPagefileUsage / 2**20,
            "working_set_mb": counters.WorkingSetSize / 2**20,
            "commit_mb": counters.PagefileUsage / 2**20,
        }
    import resource

    return {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def run_task(fn: Any, args: tuple[Any, ...]) -> dict[str, Any]:
    """Run one task in a worker; an exception becomes an error record (never lost)."""
    started = time.perf_counter()
    try:
        reset_task_state()
        record = {"status": "ok", **fn(*args)}
    except BaseException:  # recorded and reported, the run goes on
        record = {"status": "error", "traceback": traceback.format_exc()}
    record.setdefault("elapsed_s", time.perf_counter() - started)
    record["memory"] = peak_memory_mb()
    record["worker_pid"] = os.getpid()
    return record


VOLATILE = ("elapsed_s", "memory", "worker_pid")


def comparable(record: dict[str, Any]) -> str:
    """A record without its run-time fields, as exact JSON (for bit-identity checks)."""
    kept = {k: v for k, v in record.items() if k not in VOLATILE}
    return json.dumps(kept, sort_keys=True, default=_default)


def fresh_and_reused(
    tasks: list[tuple[Any, tuple[Any, ...]]], warmup: list[tuple[Any, tuple[Any, ...]]]
) -> list[dict[str, Any]]:
    """Every task once in a fresh worker process and once in one reused worker.

    The reused worker first runs ``warmup``, then the tasks in order (each after the ones before
    it). Per task: both statuses, whether the records are identical (``comparable``), the pids.
    """
    fresh = []
    for fn, args in tasks:
        with ProcessPoolExecutor(max_workers=1) as pool:
            fresh.append(pool.submit(run_task, fn, args).result())
    with ProcessPoolExecutor(max_workers=1) as pool:
        before = [pool.submit(run_task, fn, args).result() for fn, args in warmup]
        reused = [pool.submit(run_task, fn, args).result() for fn, args in tasks]
    pids = {r["worker_pid"] for r in [*before, *reused]}
    return [
        {
            "key": f.get("key"),
            "status": [f["status"], r["status"]],
            "identical": comparable(f) == comparable(r),
            "fresh_pid": f["worker_pid"],
            "reused_pid": r["worker_pid"],
            "reused_worker_ran_before": len(before) + i,
            "one_reused_worker": len(pids) == 1,
        }
        for i, (f, r) in enumerate(zip(fresh, reused, strict=True))
    ]


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


DRY_RUNS = {
    # 1 bisection step (bracket = the whole interval: holds at 0, at the upper end, or neither),
    # the smallest grid, no zoom -- every task runs once, at minimal cost
    "minimal": {"coarse": 3, "zoom": 3, "zoom_levels": 0, "max_moves": 1, "starts": 1},
    # as minimal but with the full coarse grid: the largest prediction batch of the main run
    # (coarse grid x cases), hence its memory peak
    "coarse": {"coarse": None, "zoom": 3, "zoom_levels": 0, "max_moves": 1, "starts": 1},
}


class Scheduler:
    """Tasks of one part; the margins cells add their next tasks when a stage completes."""

    def __init__(self, output: Path, run: MarginRun | None, holding_share: float) -> None:
        self.output = output
        self.run = run
        self.holding_share = holding_share
        self.ready: list[tuple[tuple[Any, ...], str, Any, tuple[Any, ...]]] = []
        self.cells: dict[str, Cell] = {}
        self.owner: dict[str, Cell] = {}
        self.n_done = 0
        self.n_errors = 0
        self.n_cells_done = 0

    def add(self, priority: tuple[Any, ...], key: str, fn: Any, args: tuple[Any, ...]) -> None:
        self.ready.append((priority, key, fn, args))

    def pop(self) -> tuple[str, Any, tuple[Any, ...]] | None:
        """The next task by priority; tasks of a cell stopped by an error are dropped."""
        self.ready.sort(key=lambda item: item[0])
        while self.ready:
            _, key, fn, args = self.ready.pop(0)
            cell = self.owner.get(key)
            if cell is None or not cell.blocked:
                return key, fn, args
        return None

    # margins -----------------------------------------------------------------------------------

    def add_cell(self, cell: Cell, done: dict[str, dict[str, Any]]) -> None:
        self.cells[cell.key] = cell
        for d in cell.draws:
            key = search_key(cell.model, cell.seed, cell.n, d)
            self.owner[key] = cell
            if key in done:
                cell.search[d] = done[key]
            else:
                self.add(
                    self._priority(cell, "search"),
                    key,
                    margin_search_task,
                    (self.run, cell.model, cell.seed, cell.n, d),
                )
        if len(cell.search) == len(cell.draws):
            self._advance(cell, done)

    def _priority(self, cell: Cell, stage: str) -> tuple[Any, ...]:
        # evaluations first (they finish cells), then the most expensive searches
        order = {"M4": 0, "M3": 1, "M1": 2, "M2": 3, "M0": 4}
        return (0 if stage == "evaluate" else 1, order[cell.model], -cell.n, cell.seed)

    def _advance(self, cell: Cell, done: dict[str, dict[str, Any]]) -> None:
        from twinlink_ml.hybrid_model.margin_tasks import cell_parameter, next_round

        assert self.run is not None
        if cell.parameter is None:
            cell.parameter = cell_parameter(cell.needed(), self.holding_share)
        upper = self.run.bisection(cell.model).upper
        while True:
            decision = next_round(cell.parameter, cell.shares(), self.holding_share, upper)
            if decision.done:
                if cell.key not in done:
                    self.write({"kind": "cell", **assemble_cell(cell, self.run, decision)})
                cell.finished = True
                self.n_cells_done += 1
                return
            r = len(cell.shares())
            if len(cell.rounds) == r:
                cell.rounds.append({})
            missing = []
            for d in cell.draws:
                key = evaluate_key(cell.model, cell.seed, cell.n, r, d)
                self.owner[key] = cell
                if key in done:
                    if done[key]["parameter"] != decision.parameter:
                        raise SystemExit(
                            f"{key}: recorded parameter differs from the recomputed one"
                        )
                    cell.rounds[r][d] = done[key]
                elif d not in cell.rounds[r]:
                    missing.append(d)
            if len(cell.rounds[r]) == len(cell.draws):
                continue  # the round is complete from the records: decide again
            for d in missing:
                self.add(
                    self._priority(cell, "evaluate"),
                    evaluate_key(cell.model, cell.seed, cell.n, r, d),
                    margin_evaluate_task,
                    (self.run, cell.model, cell.seed, cell.n, d, r, decision.parameter),
                )
            return

    def complete(self, key: str, record: dict[str, Any]) -> None:
        cell = self.owner.get(key)
        if cell is None or cell.blocked:
            return
        if record["task"] == "search":
            cell.search[record["draw"]] = record
            if len(cell.search) == len(cell.draws):
                self._advance(cell, {})
        else:
            cell.rounds[record["round"]][record["draw"]] = record
            if len(cell.rounds[record["round"]]) == len(cell.draws):
                self._advance(cell, {})

    # bookkeeping -------------------------------------------------------------------------------

    def write(self, record: dict[str, Any]) -> None:
        append(self.output, record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("oracle", "margins", "errors"), required=True)
    parser.add_argument("--prefix", default="hybrid_model_v1")
    parser.add_argument("--training", default="hybrid_model_training_v1")
    parser.add_argument(
        "--name",
        default=None,
        help="output name (default <prefix>_<part>, margins: <prefix>_margins_per_draw)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", choices=tuple(DRY_RUNS), default=None)
    parser.add_argument(
        "--max-batch",
        type=int,
        default=256,
        help="margins: points per map call (memory; 0 = one call per request)",
    )
    parser.add_argument("--only-models", nargs="*", default=None)
    parser.add_argument("--only-n", type=int, nargs="*", default=None)
    parser.add_argument("--only-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--only-draws", type=int, nargs="*", default=None)
    args = parser.parse_args()

    from _common import RUNS_DIR, provenance, require_clean_tree

    from twinlink_ml.hybrid_model.grid_rto import BisectionSettings, GridSettings
    from twinlink_ml.utils.code_location import require_code_location

    require_code_location(ROOT)
    default = (
        f"{args.prefix}_margins_per_draw"
        if args.part == "margins"
        else f"{args.prefix}_{args.part}"
    )
    name = args.name or default
    if args.dry_run and not name.startswith("_"):
        raise SystemExit("a dry run writes to a name starting with '_' (gitignored)")
    output = RUNS_DIR / f"{name}.jsonl"
    oracle = None if args.part == "oracle" else f"{args.prefix}_oracle"
    exempt = clean_tree_exceptions(RUNS_DIR, output, args.training, oracle)
    require_clean_tree(ignore_paths=exempt)
    if output.exists() and not args.resume:
        raise SystemExit(f"{output} exists -- use --resume or a new name")
    s = setup()
    cfg = s.config
    seeds = args.only_seeds or list(cfg.master_seeds)
    sizes = args.only_n or list(cfg.n_train)
    draws = tuple(args.only_draws if args.only_draws is not None else range(len(s.draws)))
    grid, constant, ensemble_k = cfg.grid, cfg.margin_constant, cfg.margin_k
    if args.dry_run:
        over = dict(DRY_RUNS[args.dry_run])
        over["coarse"] = over["coarse"] or cfg.grid.coarse
        grid = GridSettings(**over)
        constant = BisectionSettings(upper=constant.upper, tolerance=constant.upper)
        ensemble_k = BisectionSettings(upper=ensemble_k.upper, tolerance=ensemble_k.upper)
    run = MarginRun(
        f"{args.prefix}_oracle", args.training, grid, constant, ensemble_k, args.max_batch or None
    )

    previous = read_records(output)
    done = {
        r["key"]: r
        for r in previous
        if r.get("kind") in ("task", "job", "cell") and r.get("status", "ok") == "ok"
    }
    inputs = [p for p in exempt[1:] if p.exists()]
    append(
        output,
        {
            "kind": "header",
            "provenance": provenance("not used", ignore_paths=exempt),
            "part": args.part,
            "training": args.training,
            "oracle": oracle,
            "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            "config": cfg.raw,
            "dry_run": args.dry_run,
            "settings": {
                "grid": dataclasses.asdict(grid),
                "constant": dataclasses.asdict(constant),
                "ensemble_k": dataclasses.asdict(ensemble_k),
                "max_batch": run.max_batch,
            },
            "workers": args.workers,
            "selection": {
                "models": args.only_models,
                "n": sizes,
                "seeds": seeds,
                "draws": list(draws),
            },
            "resumed_with": len(done),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    sched = Scheduler(output, run, cfg.holding_share)
    if args.part == "oracle":
        for i in range(cfg.discretisation_n_cases):
            if f"discretisation_{i}" not in done:
                sched.add((0,), f"discretisation_{i}", discretisation_job, (i,))
        for d in draws:
            if f"oracle_d{d:02d}" not in done:
                sched.add((1,), f"oracle_d{d:02d}", oracle_job, (d,))
    elif args.part == "margins":
        for m in args.only_models or list(MODELS):
            for n in sizes:
                for seed in seeds:
                    sched.add_cell(Cell(m, seed, n, draws), done)
    else:
        for d in draws:
            for seed in seeds:
                if f"errors_d{d:02d}_s{seed}" not in done:
                    sched.add(
                        (0,), f"errors_d{d:02d}_s{seed}", errors_job, (args.training, d, seed)
                    )
    n_cells = len(sched.cells)
    print(
        f"{len(sched.ready)} tasks ready, {len(done)} records done before"
        + (f", {n_cells} cells ({sched.n_cells_done} done)" if n_cells else ""),
        flush=True,
    )

    started = time.time()
    in_flight: dict[Any, str] = {}
    broken = False
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while (sched.ready and not broken) or in_flight:
            while sched.ready and len(in_flight) < args.workers and not broken:
                task = sched.pop()
                if task is None:
                    break
                key, fn, task_args = task
                in_flight[pool.submit(run_task, fn, task_args)] = key
            if not in_flight:
                break
            complete, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in complete:
                key = in_flight.pop(future)
                try:
                    record = future.result()
                except BrokenProcessPool:
                    # a worker died (e.g. out of memory): every task in flight is lost; stop
                    # scheduling, record the error, resume later
                    broken = True
                    record = {
                        "status": "error",
                        "traceback": traceback.format_exc(),
                        "elapsed_s": 0.0,
                        "memory": {},
                    }
                kind = "job" if args.part != "margins" else "task"
                if record["status"] == "ok":
                    sched.write({"kind": kind, **record})
                    sched.n_done += 1
                    if args.part == "margins":
                        sched.complete(key, record)
                else:
                    sched.write({"kind": "error", "key": key, **record})
                    sched.n_errors += 1
                    cell = sched.owner.get(key)
                    if cell is not None:
                        cell.blocked = True
                mem = record.get("memory", {})
                print(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{sched.n_done} done, {sched.n_errors} errors, "
                    f"{len(sched.ready) + len(in_flight)} open"
                    + (f", cells {sched.n_cells_done}/{n_cells}" if n_cells else "")
                    + f"] {key} {record['status']} {record['elapsed_s']:.0f} s"
                    f" worker {record.get('worker_pid')}"
                    f" commit {mem.get('commit_mb', math.nan):.0f} MB"
                    f" peak commit {mem.get('peak_commit_mb', math.nan):.0f} MB"
                    f" peak ws {mem.get('peak_working_set_mb', math.nan):.0f} MB"
                    f" (wall {time.time() - started:.0f} s)",
                    flush=True,
                )
                if record["status"] != "ok":
                    print(record["traceback"], flush=True)
    blocked = [c.key for c in sched.cells.values() if c.blocked]
    append(
        output,
        {
            "kind": "footer",
            "runtime_s": time.time() - started,
            "n_done": sched.n_done,
            "n_errors": sched.n_errors,
            "cells_done": sched.n_cells_done,
            "cells_blocked": blocked,
            "pool_broken": broken,
        },
    )
    print(
        f"finished: {sched.n_done} tasks, {sched.n_errors} errors"
        + (f", cells {sched.n_cells_done}/{n_cells}, blocked {blocked}" if n_cells else ""),
        flush=True,
    )
    return 1 if sched.n_errors or broken else 0


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    raise SystemExit(main())
