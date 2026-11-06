"""Units of the run (tasks), their points, their evaluation and the assembly of the arrays.

A unit (task) is one chunk of one Sobol set in one variant, one 1D cut, or one row of a 2D cut;
each unit evaluates its points in all three modes. A failure at one point and mode is recorded at
that point (``error``) and does not discard the unit; a failing unit does not discard others
(``scripts/run_case_ranking.py``).
"""

from __future__ import annotations

import ctypes
import os
import time
import traceback
from typing import Any

import numpy as np

from .config import SET_NAMES, VARIANTS, CaseRankingConfig
from .evaluator import CASES, MODES, FlowsheetCaseEvaluator, PointResult
from .space import INPUT_NAMES, OperatingPoint, sobol_points
from .workbook import WORKBOOK_POINT

__all__ = [
    "assemble",
    "peak_memory_mb",
    "run_task",
    "task_list",
    "task_points",
]

Task = tuple[Any, ...]


def peak_memory_mb() -> float:
    """Peak resident memory of this process in MB (Windows: peak working set)."""
    if os.name == "nt":

        class _Counters(ctypes.Structure):
            _fields_ = [  # noqa: RUF012 - ctypes layout
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 1e6
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def task_list(config: CaseRankingConfig) -> list[Task]:
    """Every unit of the run (the dry run has the same units with fewer points)."""
    tasks: list[Task] = []
    for v in VARIANTS:
        for s in SET_NAMES:
            n = config.points[v][s]
            tasks += [("set", v, s, k) for k in range(-(-n // config.chunk_points))]
        tasks += [("cut1d", v, name, 0) for name in INPUT_NAMES]
    for p in range(len(config.two_d_pairs)):
        tasks += [("cut2d", "fixed", p, row) for row in range(config.two_d_points)]
    return tasks


def _cut_axis(config: CaseRankingConfig, name: str, count: int) -> np.ndarray:
    i = config.box.interval(name)
    return np.linspace(i.low, i.high, count)


def task_points(config: CaseRankingConfig, task: Task, dry_run: bool) -> np.ndarray:
    """Points of one unit, shape (m, 8)."""
    kind, variant = task[0], task[1]
    if kind == "set":
        name, chunk = task[2], int(task[3])
        n = config.points[variant][name]
        x = sobol_points(config.set_box(name), n, config.set_seed(name))
        x = x[chunk * config.chunk_points : min(n, (chunk + 1) * config.chunk_points)]
    elif kind == "cut1d":
        name = task[2]
        count = config.one_d_points if variant == "fixed" else config.one_d_points_optimised
        axis = _cut_axis(config, name, count)
        x = np.array([WORKBOOK_POINT.with_(**{name: float(v)}).as_array() for v in axis])
    elif kind == "cut2d":
        a, b = config.two_d_pairs[int(task[2])]
        row = int(task[3])
        ax_a = _cut_axis(config, a, config.two_d_points)
        ax_b = _cut_axis(config, b, config.two_d_points)
        x = np.array(
            [WORKBOOK_POINT.with_(**{a: float(ax_a[row]), b: float(v)}).as_array() for v in ax_b]
        )
    else:
        raise ValueError(kind)
    return x[: config.dry_run_points_per_chunk] if dry_run else x


def _record(r: PointResult) -> dict[str, Any]:
    return {
        "fuel": [c.fuel_kg_per_h for c in r.cases],
        "medium_max": [c.medium_hot_max_celsius for c in r.cases],
        "ndf": [c.stages_no_driving_force for c in r.cases],
        "neg": [c.stages_negative_recovery for c in r.cases],
        "mlim": [c.stages_medium_limit for c in r.cases],
        "flows": [c.htm_flow_kg_per_h for c in r.cases],
        "nfev": [c.optimiser_evaluations for c in r.cases],
        "exh_T": list(r.exhaust_celsius),
        "exh_Y": list(r.exhaust_humidity_kg_per_kg),
        "exh_rh": list(r.exhaust_relative_humidity),
    }


def run_task(config: CaseRankingConfig, task: Task, dry_run: bool) -> dict[str, Any]:
    """Evaluate one unit in all modes; per point and mode either a record or an ``error``."""
    t0 = time.perf_counter()
    x = task_points(config, task, dry_run)
    variant = task[1]
    optimiser = config.optimiser if variant == "optimised" else None
    evaluators = {m: FlowsheetCaseEvaluator(MODES[m], optimiser, cases=CASES) for m in MODES}
    out: dict[str, list[dict[str, Any]]] = {m: [] for m in MODES}
    errors = 0
    for row in x:
        p = OperatingPoint.from_array(row)
        for m, ev in evaluators.items():
            try:
                out[m].append(_record(ev.evaluate(p)))
            except Exception:
                errors += 1
                out[m].append({"error": traceback.format_exc(limit=3)})
    return {
        "points": x.tolist(),
        "modes": out,
        "point_errors": errors,
        "seconds": time.perf_counter() - t0,
        "peak_memory_mb": peak_memory_mb(),
    }


def assemble(
    config: CaseRankingConfig, results: dict[str, dict[str, Any]], dry_run: bool
) -> dict[str, np.ndarray]:
    """Arrays per (variant, set or cut, mode) from the unit results keyed by ``json`` of the task.

    Keys ``<variant>/<set>/points`` (m, 8) and ``<variant>/<set>/<mode>/<quantity>``; a point
    whose unit is missing or failed is absent from every array of that set (``.../present``
    records which positions are filled).
    """
    import json

    groups: dict[tuple[str, str], list[Task]] = {}
    for t in task_list(config):
        key = (t[1], t[2] if t[0] == "set" else f"{t[0]}_{t[2]}")
        groups.setdefault(key, []).append(t)
    arrays: dict[str, np.ndarray] = {}
    for (variant, label), tasks in groups.items():
        pts: list[np.ndarray] = []
        recs: dict[str, list[dict[str, Any]]] = {m: [] for m in MODES}
        present: list[bool] = []
        for t in tasks:
            res = results.get(json.dumps(list(t)))
            expected = task_points(config, t, dry_run)
            if res is None:
                present += [False] * len(expected)
                continue
            pts.append(np.asarray(res["points"]))
            present += [True] * len(res["points"])
            for m in MODES:
                recs[m] += res["modes"][m]
        prefix = f"{variant}/{label}"
        arrays[f"{prefix}/present"] = np.array(present, dtype=bool)
        arrays[f"{prefix}/points"] = np.concatenate(pts) if pts else np.zeros((0, len(INPUT_NAMES)))
        for m in MODES:
            rs = recs[m]
            err = np.array(["error" in r for r in rs], dtype=bool)
            ok = [r if "error" not in r else None for r in rs]

            def col(name: str, width: int, ok: list[Any] = ok) -> np.ndarray:
                return np.array(
                    [
                        [np.nan if v is None else v for v in r[name]]
                        if r is not None
                        else [np.nan] * width
                        for r in ok
                    ],
                    dtype=np.float64,
                ).reshape(len(ok), width)

            q = f"{prefix}/{m}"
            arrays[f"{q}/error"] = err
            arrays[f"{q}/fuel"] = col("fuel", 7)
            arrays[f"{q}/medium_max"] = col("medium_max", 7).astype(np.float32)
            for flag in ("ndf", "neg", "mlim"):
                arrays[f"{q}/{flag}"] = np.nan_to_num(col(flag, 7), nan=-1).astype(np.int8)
            arrays[f"{q}/exh_T"] = col("exh_T", 2)
            arrays[f"{q}/exh_Y"] = col("exh_Y", 2)
            arrays[f"{q}/exh_rh"] = col("exh_rh", 2)
            if variant == "optimised":
                flows = np.full((len(ok), 7, 2), np.nan, dtype=np.float32)
                for i, r in enumerate(ok):
                    if r is not None:
                        for k, f in enumerate(r["flows"]):
                            if f is not None:
                                flows[i, k] = f
                arrays[f"{q}/flows"] = flows
                arrays[f"{q}/nfev"] = np.nan_to_num(col("nfev", 7)).astype(np.int32)
    return arrays
