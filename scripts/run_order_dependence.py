"""Order dependence of the steady-state optimiser: measurement runs.

Study document: ``docs/NN_order_dependence.md``; reading (pre-registered): docstring of
``scripts/analyze_order_dependence.py``; settings:
``twinlink_ml.order_dependence.settings.OrderDependenceSettings``.

Background: ``headroom.controllers.SteadyStateOptimizer.solve`` starts every solve from the
solution of the previous one, and the caches in front of it (``load_following.steady.SteadyMap``,
``QuasiStationaryController``) use rounded keys. The steady optimum returned for a query therefore
depends on what the worker process solved before (control of the learned-setpoints study). This
driver measures the effect; it changes no existing result. The reference controllers are run with
the unchanged code ("warm") and with the proposed switch ``cold_starts`` ("cold": a cold
deterministic multi-start per query and unrounded cache keys).

Units and inputs: the 105 scenarios of the family "observed" of Befund 105 (R0, R1) and the 40
test days of Befund 126 (C0, C1, C2; variant E, main band, true prices; band and rule of C1 from
the stage-0 artifact). "Artifact margins" are the margins of ``dynamic_headroom_v2.json.gz`` and
``load_following_v1.json.gz``. The twin of the learned-setpoints study is C2 at the artifact
margin with the tail at the artifact margin of C0, i.e. the run C2 of part ``cold_held``.

Stages (``--stage``)
--------------------
``dryrun``
    ALL units of every part at minimal cost: scenarios and days shortened to 2 h, coarse margin
    tolerance. Surfaces failing units; reports time and peak memory. Output
    ``<name>_dryrun.json.gz``.
``probe``
    Full length and full batch size under the load of all workers: R1 and C2 once with the cold
    multi-start on all units, then the first query chunk of every source in parallel. Reports
    the time per evaluation and the peak memory per worker and extrapolates the main run. Output
    ``<name>_probe.json.gz``.
``main``
    Parts (``--parts``, default all, in this order):

    ``warm_rerun``
        R0, R1, C0, C1, C2 at the artifact margins with the unchanged code, in fresh worker
        processes with the chunks of this run: a second run of the code of the findings
        (the difference to the artifact records is the order dependence itself).
    ``cold_held``
        The same with the cold multi-start (key numbers at the held artifact margins).
    ``cold_search``
        The margin searches of the findings repeated with the cold multi-start (same bisection,
        bracket and tolerances), then all units at the margin found.
    ``cold_repeat``
        ``cold_held`` once more with other chunks, in reversed order and in worker processes
        that have solved other queries before: with the switch the records must be bit-equal.
    ``queries``
        Per controller (R1, C0, C1, C2; artifact margins) and chunk of units (the chunks of
        ``warm_rerun``, run in lockstep on a fresh optimiser as one task of the drivers of the
        findings): the queries the controllers send to the optimiser are recorded and solved
        again warm in several orders, cold, with the multi-start and with a dense reference
        (``twinlink_ml.order_dependence.queries``).
    ``stage0_check``
        Stage 0 of the load-following study on the steady-state map with the cold multi-start:
        admissible range, bands and the tuning of the rule C1, against the stage-0 artifact.

Every unit is appended to ``artifacts/runs/_<name>_<stage>_cache.jsonl`` (gitignored) as soon as
it is finished; ``--resume`` reuses it (same commit and settings required). A failed unit is
logged and recorded as an error; it does not discard other units. At most four workers. Run from
the worktree with ``tools/run_here.sh scripts/run_order_dependence.py --stage ... --name ...``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import REPO_ROOT, RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.headroom.config import (  # noqa: E402
    HeadroomStudyConfig,
    load_headroom_config,
    study_process_config,
)
from twinlink_ml.headroom.controllers import SteadyStateOptimizer  # noqa: E402
from twinlink_ml.headroom.margin import (  # noqa: E402
    MarginSearch,
    bisect_margin,
    bisect_margin_over_scenarios,
)
from twinlink_ml.headroom.scenarios import build_scenarios  # noqa: E402
from twinlink_ml.headroom.study import Solvers as HeadroomSolvers  # noqa: E402
from twinlink_ml.headroom.study import run_controller  # noqa: E402
from twinlink_ml.load_following.config import (  # noqa: E402
    LoadFollowingConfig,
    load_load_following_config,
)
from twinlink_ml.load_following.market import build_days, build_market  # noqa: E402
from twinlink_ml.load_following.memory import peak_memory_mb  # noqa: E402
from twinlink_ml.load_following.stage0 import (  # noqa: E402
    admissible_range,
    bands,
    hourly_curves,
    map_table,
    tune_rule,
)
from twinlink_ml.load_following.steady import SteadyMap  # noqa: E402
from twinlink_ml.load_following.study import RunSpec, run_days  # noqa: E402
from twinlink_ml.load_following.study import Solvers as LoadSolvers  # noqa: E402
from twinlink_ml.order_dependence.queries import (  # noqa: E402
    RecordingOptimizer,
    measure,
    summarise,
)
from twinlink_ml.order_dependence.runs import chunks_of, compare_records  # noqa: E402
from twinlink_ml.order_dependence.settings import OrderDependenceSettings  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning  # noqa: E402

STAGES = ("dryrun", "probe", "main")
PARTS = ("warm_rerun", "cold_held", "cold_search", "cold_repeat", "queries", "stage0_check")
HEADROOM_KINDS = ("R1", "R0")  # order of the searches of the headroom driver
LOAD_KINDS = ("C0", "C1", "C2")
FAMILY = "observed"
MAX_WORKERS = 4
MAX_MARGIN_HEADROOM = 0.12  # MAX_MARGIN of scripts/run_dynamic_headroom.py
DRY_HOURS = 2
DRY_MARGIN_TOLERANCE = 4.0e-3
HEADROOM_ARTIFACT = "dynamic_headroom_v2.json.gz"
LOAD_ARTIFACT = "load_following_v1.json.gz"
STAGE0_ARTIFACT = "load_following_v1_stage0.json.gz"
SETTINGS = OrderDependenceSettings()
WORST_PER_UNIT = 10


def log(message: str) -> None:
    """One progress line with a time stamp."""
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}", flush=True)


def load_configs(dry: bool) -> tuple[LoadFollowingConfig, HeadroomStudyConfig]:
    """Configs of the two studies; in the dry run days and scenarios are shortened."""
    load = load_load_following_config(day_hours=DRY_HOURS if dry else None)
    study = load_headroom_config()
    if dry:
        study = dataclasses.replace(
            study, scenarios=dataclasses.replace(study.scenarios, duration_s=DRY_HOURS * 3600.0)
        )
    return load, study


# ------------------------------------------------------------------------------ worker side

_WORKER: dict[str, Any] = {}


def _init_worker(dry: bool, stage0: dict[str, Any]) -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    load, study = load_configs(dry)
    process = study_process_config(study)
    market = build_market(load, REPO_ROOT)
    cold = SETTINGS.cold_starts
    _WORKER.update(
        load=load,
        study=study,
        process=process,
        market=market,
        stage0=stage0,
        scenarios=None,
        days={},
        headroom_solvers={
            "warm": HeadroomSolvers(study, process),
            "cold": HeadroomSolvers(study, process, steady_cold_starts=cold),
        },
        load_solvers={
            "warm": LoadSolvers(load, market.reference_tariff),
            "cold": LoadSolvers(load, market.reference_tariff, steady_cold_starts=cold),
        },
    )


def _scenarios(ids: Sequence[int]) -> list[Any]:
    if _WORKER["scenarios"] is None:
        _WORKER["scenarios"] = build_scenarios(_WORKER["study"], _WORKER["process"], FAMILY)
    return [_WORKER["scenarios"][i] for i in ids]


def _days(ids: Sequence[int]) -> list[Any]:
    cache = _WORKER["days"]
    missing = [i for i in ids if i not in cache]
    if missing:
        for day in build_days(_WORKER["load"], _WORKER["market"], missing):
            cache[day.day] = day
    return [cache[i] for i in ids]


def _run_spec(kind: str, margin: float, tail_margin: float) -> RunSpec:
    """Spec of variant E on the main band with the band and the rule of the stage-0 artifact."""
    stage0 = _WORKER["stage0"]
    rule = stage0["rule"]["E"]
    return RunSpec(
        kind=kind,  # type: ignore[arg-type]
        margin=float(margin),
        tail_margin=float(tail_margin),
        band=tuple(stage0["bands"]["main"]),  # type: ignore[arg-type]
        band_name="main",
        carrier="E",
        rule_quantile=rule["quantile"] if kind == "C1" else None,
        rule_spread_kg_per_h=rule["spread_kg_per_h"] if kind == "C1" else None,
    )


def _headroom_record(record: dict[str, Any], unit: int) -> dict[str, Any]:
    """Record of one scenario without the per-interval energies; set points kept (rounded)."""
    commands = np.array(record["commands"])
    keep = {
        k: v
        for k, v in record.items()
        if k not in ("energy_kwh_per_interval", "moisture_max_per_interval", "commands", "change")
    }
    keep["unit"] = unit
    keep["inlet_command_per_interval"] = [float(f"{v:.7g}") for v in commands[:, 0]]
    keep["gas_command_per_interval"] = [float(f"{v:.7g}") for v in commands[:, 1]]
    return keep


def _records(request: dict[str, Any], ids: list[int], solvers: Any) -> list[dict[str, Any]]:
    if request["domain"] == "headroom":
        out = run_controller(request["kind"], request["margin"], _scenarios(ids), solvers)
        return [_headroom_record(r.as_dict(), i) for r, i in zip(out, ids, strict=True)]
    spec = _run_spec(request["kind"], request["margin"], request["tail_margin"])
    records = run_days(spec, _days(ids), solvers)
    return [{**r, "unit": r["day_of_year"]} for r in records]


def _task_run(request: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    """Run one controller on some units; an exception becomes an error entry (never lost)."""
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    error = None
    try:
        solvers = _WORKER[f"{request['domain']}_solvers"][request["mode"]]
        records = _records(request, ids, solvers)
    except Exception:  # a failed unit must not stop the run
        error = traceback.format_exc()
    return {
        "records": records,
        "error": error,
        "seconds": time.perf_counter() - started,
        "peak_mb": peak_memory_mb(),
        "pid": os.getpid(),
    }


def _plain(values: np.ndarray) -> Any:
    """Nested lists with ``None`` for ``nan`` (power, inlet temperature, gas velocity)."""
    return np.where(np.isfinite(values), values, None).tolist()


def _task_queries(request: dict[str, Any], units: list[int]) -> dict[str, Any]:
    """Record the queries of one controller on one chunk and measure them in all passes.

    The chunk is run in lockstep on one fresh optimiser, as a task of the drivers of the
    findings: the natural order is the order in which that optimiser receives the queries.
    """
    started = time.perf_counter()
    result: dict[str, Any] | None = None
    error = None
    try:
        if request["domain"] == "headroom":
            study, process = _WORKER["study"], _WORKER["process"]
            recorder = RecordingOptimizer(study, process)
            solvers: Any = HeadroomSolvers(study, process, _steady=recorder)
        else:
            load = _WORKER["load"]
            study, process = load.headroom, load.process
            recorder = RecordingOptimizer(study, process)
            solvers = LoadSolvers(
                load, _WORKER["market"].reference_tariff, _map=SteadyMap(load, optimizer=recorder)
            )
        _records(request, units, solvers)
        run_s = time.perf_counter() - started
        queries = recorder.queries

        def factory(cold_starts: int, seed: int) -> SteadyStateOptimizer:
            return SteadyStateOptimizer(study, process, cold_starts=cold_starts, start_seed=seed)

        measured = measure(queries, factory, SETTINGS)
        result = summarise(measured, recorder.results, SETTINGS)
        with np.errstate(invalid="ignore"):
            deviation = np.nan_to_num(
                np.abs(measured["warm"][0, :, 0] - measured["multi"][:, 0])
                / measured["multi"][:, 0],
                nan=np.inf,
            )
        worst = np.argsort(-deviation, kind="stable")[:WORST_PER_UNIT]
        result["worst"] = [
            {
                "index": int(i),
                "disturbance": [float(v) for v in queries[int(i)].disturbance],
                "bound": queries[int(i)].bound,
                "warm": _plain(measured["warm"][:, int(i), :]),
                "cold": _plain(measured["cold"][int(i)]),
                "multi": _plain(measured["multi"][int(i)]),
                "reference": _plain(measured["reference"][int(i)]),
            }
            for i in worst
        ]
        result["run_s"] = run_s
    except Exception:  # a failed unit must not stop the run
        error = traceback.format_exc()
    return {
        "result": result,
        "error": error,
        "seconds": time.perf_counter() - started,
        "peak_mb": peak_memory_mb(),
        "pid": os.getpid(),
    }


def _task_stage0_check() -> dict[str, Any]:
    """Stage 0 of the load-following study on the map with the cold multi-start."""
    started = time.perf_counter()
    result: dict[str, Any] | None = None
    error = None
    try:
        load: LoadFollowingConfig = _WORKER["load"]
        stage0 = _WORKER["stage0"]
        steady_map = _WORKER["load_solvers"]["cold"].steady
        limit = load.headroom.line.product_moisture_limit_kg_per_kg
        table = map_table(load, steady_map)
        grid = np.array(table["spray_kg_per_h"])
        admissible = admissible_range(
            grid, np.array(table["feasible_with_margin"]), load.nominal_spray_rate
        )
        both = bands(load, admissible)
        training = build_days(load, _WORKER["market"], stage0["day_sets"]["training"])
        curves = hourly_curves(load, steady_map, training, both["main"], limit)
        rule = {}
        for carrier in ("E", "G"):
            tuned = tune_rule(load, training, curves, both["main"], carrier)  # type: ignore[arg-type]
            rule[carrier] = {k: v for k, v in tuned.items() if k != "grid"}
        result = {
            "admissible_range_kg_per_h": list(admissible),
            "bands": {name: list(band) for name, band in both.items()},
            "rule": rule,
            "map": table,
            "steady_failures": steady_map.failures,
        }
    except Exception:  # a failed unit must not stop the run
        error = traceback.format_exc()
    return {
        "result": result,
        "error": error,
        "seconds": time.perf_counter() - started,
        "peak_mb": peak_memory_mb(),
        "pid": os.getpid(),
    }


# ------------------------------------------------------------------------------ main side


def _request(
    kind: str, margin: float, mode: str, *, tail_margin: float | None = None, layout: str = "a"
) -> dict[str, Any]:
    domain = "headroom" if kind.startswith("R") else "load"
    if domain == "load" and tail_margin is None:
        raise ValueError("a load-following run needs the margin of its tail")
    return {
        "domain": domain,
        "kind": kind,
        "margin": float(margin),
        "tail_margin": None if tail_margin is None else float(tail_margin),
        "mode": mode,
        "layout": layout,
    }


def _key(request: dict[str, Any], unit: int) -> str:
    return (
        f"run|{request['kind']}|{request['mode']}|{request['layout']}|"
        f"{request['margin']!r}|{request['tail_margin']!r}|{unit}"
    )


class Evaluator:
    """Runs units in the pool and keeps every record in the resumable cache."""

    def __init__(self, pool: ProcessPoolExecutor, workers: int, cache: Path, header: dict) -> None:
        self.pool = pool
        self.workers = workers
        self.cache_path = cache
        self.records: dict[str, Any] = {}
        self.errors: list[dict[str, Any]] = []
        self.peak_mb: dict[int, float] = {}
        self.seconds: dict[str, list[float]] = {}
        self.units_run = 0
        if cache.exists():
            with cache.open(encoding="utf-8") as handle:
                first = json.loads(handle.readline())
                if first.get("header") != header:
                    raise SystemExit(
                        f"{cache} was written by another commit/settings: {first.get('header')}"
                    )
                for line in handle:
                    entry = json.loads(line)
                    self.records[entry["key"]] = entry["record"]
            log(f"resume: {len(self.records)} cached units")
        else:
            with cache.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"header": header}) + "\n")

    def store(self, key: str, record: Any) -> None:
        self.records[key] = record
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "record": record}) + "\n")

    def note(self, label: str, out: dict[str, Any], units: int) -> None:
        self.peak_mb[out["pid"]] = max(self.peak_mb.get(out["pid"], 0.0), out["peak_mb"])
        self.seconds.setdefault(label, []).append(out["seconds"])
        self.units_run += units

    def run(self, request: dict[str, Any], ids: Sequence[int]) -> list[dict[str, Any]]:
        """Records of ``request`` on the units ``ids`` (in that order), cached or computed."""
        missing = [i for i in ids if _key(request, i) not in self.records]
        label = f"{request['kind']}/{request['mode']}/{request['layout']}"
        if missing:
            chunks = chunks_of(missing, self.workers, request["layout"])
            started = time.time()
            while chunks:
                futures: dict[Future, list[int]] = {
                    self.pool.submit(_task_run, request, chunk): chunk for chunk in chunks
                }
                chunks = []
                done = 0
                for future in as_completed(futures):
                    chunk = futures[future]
                    out = future.result()
                    done += len(chunk)
                    self.note(label, out, len(chunk))
                    if out["error"] is None:
                        for record in out["records"]:
                            self.store(_key(request, record["unit"]), record)
                    elif len(chunk) > 1:
                        log(f"batch {label} {chunk} failed, retrying unit by unit")
                        chunks.extend([[unit] for unit in chunk])
                    else:
                        entry = {**request, "unit": chunk[0], "error": out["error"]}
                        self.errors.append(entry)
                        log(f"ERROR unit {label} {chunk[0]}: {out['error'][-300:]}")
                        self.store(_key(request, chunk[0]), {**entry, "max_violation": None})
                    log(
                        f"unit {label} m={request['margin']:.5f} units {chunk[0]}..{chunk[-1]} "
                        f"({len(chunk)}) {out['seconds']:.0f} s "
                        f"[{done} done of {len(missing)}, {time.time() - started:.0f} s]"
                    )
        return [self.records[_key(request, i)] for i in ids]


def _violations(records: Sequence[dict[str, Any]]) -> np.ndarray:
    """Largest violation per record; an errored unit counts as holding (it is reported)."""
    return np.array(
        [0.0 if r.get("max_violation") is None else r["max_violation"] for r in records]
    )


def _search_dict(search: MarginSearch) -> dict[str, Any]:
    return {
        "margin": search.margin,
        "bracket": list(search.bracket),
        "converged": search.converged,
        "evaluations": [[m, ok] for m, ok in search.evaluations],
    }


class Study:
    """The parts of the run (shared by ``dryrun`` and ``main``)."""

    def __init__(
        self,
        ev: Evaluator,
        load: LoadFollowingConfig,
        study: HeadroomStudyConfig,
        artifact_margins: dict[str, float],
        day_ids: list[int],
        n_scenarios: int,
        *,
        margin_tolerance: float,
    ) -> None:
        self.ev = ev
        self.load = load
        self.study = study
        self.margins = artifact_margins
        self.units = {"headroom": list(range(n_scenarios)), "load": day_ids}
        control = study.control
        self.tol_violation = control.violation_tolerance_kg_per_kg
        self.tol_margin = margin_tolerance
        self.upper = control.margin_upper_kg_per_kg

    def ids(self, kind: str) -> list[int]:
        return self.units["headroom" if kind.startswith("R") else "load"]

    def at_artifact_margin(self, kind: str, mode: str, layout: str = "a") -> dict[str, Any]:
        tail = None if kind.startswith("R") else self.margins["C0"]
        return _request(kind, self.margins[kind], mode, tail_margin=tail, layout=layout)

    def _held(self, mode: str) -> dict[str, Any]:
        out = {}
        for kind in (*HEADROOM_KINDS, *LOAD_KINDS):
            records = self.ev.run(self.at_artifact_margin(kind, mode), self.ids(kind))
            out[kind] = {"margin": self.margins[kind], "records": records}
            log(
                f"{mode} at the artifact margin, {kind}: largest violation "
                f"{float(np.max(_violations(records))):.3e}"
            )
        return out

    def part_warm_rerun(self) -> dict[str, Any]:
        return self._held("warm")

    def part_cold_held(self) -> dict[str, Any]:
        return self._held("cold")

    def part_cold_repeat(self) -> dict[str, Any]:
        """Layout ``b`` against layout ``a`` of ``cold_held``: bit equality of the records."""
        out = {}
        for kind in (*HEADROOM_KINDS, *LOAD_KINDS):
            ids = self.ids(kind)
            first = self.ev.run(self.at_artifact_margin(kind, "cold"), ids)
            second = self.ev.run(self.at_artifact_margin(kind, "cold", layout="b"), ids)
            out[kind] = compare_records(first, second)
            log(f"cold repeat {kind}: {out[kind]['equal_units']} of {len(ids)} units bit-equal")
        return out

    def part_cold_search(self) -> dict[str, Any]:
        """The margin searches of the two drivers with the cold multi-start."""
        out: dict[str, Any] = {}
        n = len(self.units["headroom"])
        for kind in HEADROOM_KINDS:

            def violations(m: float, ids: Sequence[int], kind: str = kind) -> np.ndarray:
                return _violations(self.ev.run(_request(kind, m, "cold"), list(ids)))

            search, final = bisect_margin_over_scenarios(
                violations,
                n,
                tolerance_violation=self.tol_violation,
                upper=self.upper,
                tolerance=self.tol_margin,
                max_upper=MAX_MARGIN_HEADROOM,
            )
            records = self.ev.run(_request(kind, search.margin, "cold"), self.units["headroom"])
            out[kind] = {
                "search": _search_dict(search),
                "margin": search.margin,
                "largest_violation": float(np.max(final)),
                "records": records,
            }
            log(f"cold margin {kind}: {search.margin:.6f} (converged {search.converged})")
        days = self.units["load"]
        tail: float | None = None
        for kind in LOAD_KINDS:

            def feasible(m: float, kind: str = kind, tail: float | None = tail) -> bool:
                request = _request(kind, m, "cold", tail_margin=m if tail is None else tail)
                return bool(np.all(_violations(self.ev.run(request, days)) <= self.tol_violation))

            search = bisect_margin(
                feasible,
                lower=0.0,
                upper=self.upper,
                tolerance=self.tol_margin,
                max_upper=self.load.margin_maximum,
            )
            if tail is None:
                tail = search.margin  # the tail of every run uses the margin of C0
            request = _request(kind, search.margin, "cold", tail_margin=tail)
            records = self.ev.run(request, days)
            out[kind] = {
                "search": _search_dict(search),
                "margin": search.margin,
                "tail_margin": tail,
                "largest_violation": float(np.max(_violations(records))),
                "records": records,
            }
            log(f"cold margin {kind}: {search.margin:.6f} (converged {search.converged})")
        return out

    def query_units(self) -> list[tuple[dict[str, Any], list[int]]]:
        """One unit per controller and chunk of layout ``a`` (R0 sends one query: not a unit)."""
        return [
            (self.at_artifact_margin(kind, "warm"), chunk)
            for kind in ("R1", *LOAD_KINDS)
            for chunk in chunks_of(self.ids(kind), self.ev.workers, "a")
        ]

    def part_queries(
        self, units: Sequence[tuple[dict[str, Any], list[int]]] | None = None
    ) -> dict[str, Any]:
        units = self.query_units() if units is None else list(units)
        ev = self.ev

        def key(request: dict[str, Any], chunk: list[int]) -> str:
            return f"queries|{request['kind']}|{chunk[0]}..{chunk[-1]}|{len(chunk)}"

        missing = [(r, c) for r, c in units if key(r, c) not in ev.records]
        futures = {ev.pool.submit(_task_queries, r, c): (r, c) for r, c in missing}
        started = time.time()
        for done, future in enumerate(as_completed(futures), start=1):
            request, chunk = futures[future]
            out = future.result()
            ev.note(f"queries/{request['kind']}", out, len(chunk))
            entry = {"kind": request["kind"], "units": chunk}
            if out["error"] is None:
                ev.store(key(request, chunk), {**entry, **out["result"], "seconds": out["seconds"]})
                n = out["result"]["n_queries"]
            else:
                ev.errors.append({"part": "queries", **entry, "error": out["error"]})
                ev.store(key(request, chunk), {**entry, "error": out["error"]})
                log(f"ERROR queries {request['kind']} {chunk}: {out['error'][-300:]}")
                n = 0
            log(
                f"queries {request['kind']} units {chunk[0]}..{chunk[-1]} ({len(chunk)}): "
                f"{n} queries, {out['seconds']:.0f} s "
                f"[{done} done of {len(missing)}, {time.time() - started:.0f} s]"
            )
        out_units: dict[str, dict[str, Any]] = {}
        for request, chunk in units:
            name = f"{chunk[0]}..{chunk[-1]}"
            out_units.setdefault(request["kind"], {})[name] = ev.records[key(request, chunk)]
        return {"margins": {k: self.margins[k] for k in out_units}, "units": out_units}

    def part_stage0_check(self) -> dict[str, Any]:
        key = "stage0_check"
        if key not in self.ev.records:
            out = self.ev.pool.submit(_task_stage0_check).result()
            self.ev.note(key, out, 1)
            if out["error"] is not None:
                self.ev.errors.append({"part": key, "error": out["error"]})
                log(f"ERROR stage0_check: {out['error'][-300:]}")
                self.ev.store(key, {"error": out["error"]})
            else:
                self.ev.store(key, {**out["result"], "seconds": out["seconds"]})
            log(f"stage0_check: {out['seconds']:.0f} s")
        return self.ev.records[key]  # type: ignore[no-any-return]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)  # type: ignore[no-any-return]


def read_artifacts() -> dict[str, Any]:
    """Margins, day set, band and rule of the artifacts of the findings (read only)."""
    headroom = _load_json_gz(RUNS_DIR / HEADROOM_ARTIFACT)
    load = _load_json_gz(RUNS_DIR / LOAD_ARTIFACT)
    stage0 = _load_json_gz(RUNS_DIR / STAGE0_ARTIFACT)["stage0"]
    margins = {kind: float(headroom[FAMILY]["margins"][kind]) for kind in HEADROOM_KINDS}
    margins.update({kind: float(load["margins"][kind]) for kind in LOAD_KINDS})
    return {
        "margins": margins,
        "day_ids": [int(d) for d in load["day_ids"]],
        "n_scenarios": int(headroom[FAMILY]["n_scenarios"]),
        "scenarios": list(headroom[FAMILY]["scenarios"]),
        "stage0": {
            "bands": stage0["bands"],
            "rule": {
                c: {k: v for k, v in r.items() if k != "grid"} for c, r in stage0["rule"].items()
            },
            "day_sets": stage0["day_sets"],
        },
        "fingerprints": {
            "headroom_study": headroom["study_fingerprint"],
            "headroom_process": headroom["process_config_fingerprint"],
            "load_study": load["study_fingerprint"],
            "load_headroom": load["headroom_fingerprint"],
            "load_process": load["process_config_fingerprint"],
        },
        "sha256": {
            name: _sha256(RUNS_DIR / name)
            for name in (HEADROOM_ARTIFACT, LOAD_ARTIFACT, STAGE0_ARTIFACT)
        },
    }


def check_fingerprints(
    artifacts: dict[str, Any], load: LoadFollowingConfig, study: HeadroomStudyConfig
) -> None:
    """Abort if the configs of this checkout are not those of the artifacts."""
    process = study_process_config(study)
    now = {
        "headroom_study": study.fingerprint,
        "headroom_process": process.fingerprint,
        "load_study": load.fingerprint,
        "load_headroom": load.headroom.fingerprint,
        "load_process": load.process.fingerprint,
    }
    if now != artifacts["fingerprints"]:
        raise SystemExit(f"configs differ from the artifacts: {now} != {artifacts['fingerprints']}")


def stage_probe(study: Study, workers: int) -> dict[str, Any]:
    """Time and memory at full length and full batch size, and the estimate of the main run."""
    ev = study.ev
    timing: dict[str, float] = {}
    for kind in ("R1", "C2"):
        started = time.time()
        ev.run(study.at_artifact_margin(kind, "cold"), study.ids(kind))
        timing[kind] = time.time() - started
        log(f"probe: one evaluation of {kind} (cold, all units): {timing[kind]:.0f} s")
    all_units = study.query_units()
    first = {r["kind"]: (r, chunk) for r, chunk in reversed(all_units)}
    study.part_queries(list(first.values()))
    query_s = {k.split("/")[1]: float(np.mean(v)) for k, v in ev.seconds.items() if "queries" in k}
    control = study.study.control
    steps = 2 + math.ceil(math.log2(control.margin_upper_kg_per_kg / study.tol_margin))
    evaluations = {"headroom": 2 * (3 + steps), "load": 3 * (3 + steps)}
    cpu_queries = sum(query_s[r["kind"]] for r, _ in all_units)
    wall_s = (
        evaluations["headroom"] * timing["R1"]
        + evaluations["load"] * timing["C2"]
        + cpu_queries / workers
    )
    return {
        "seconds_per_evaluation": timing,
        "seconds_per_query_chunk": query_s,
        "evaluations_assumed": evaluations,
        "bisection_evaluations_assumed": steps,
        "estimate_wall_h": wall_s / 3600.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--name", required=True, help="base name of the artifacts")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--parts", default=",".join(PARTS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit-units", type=int, help="smoke runs only: the first N scenarios and days"
    )
    parser.add_argument(
        "--out-dir",
        default=str(RUNS_DIR),
        help="output directory; anything but artifacts/runs is a smoke run (no clean-tree check)",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be between 1 and {MAX_WORKERS}")
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    dry = args.stage == "dryrun"
    load, study_config = load_configs(dry)
    artifacts = read_artifacts()
    check_fingerprints(artifacts, *load_configs(False))
    out_dir = Path(args.out_dir).resolve()
    smoke = out_dir != RUNS_DIR.resolve()
    suffix = "" if args.stage == "main" else f"_{args.stage}"
    output = out_dir / f"{args.name}{suffix}.json.gz"
    cache = out_dir / f"_{args.name}{suffix}_cache.jsonl"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if cache.exists() and not args.resume:
        raise SystemExit(f"{cache} exists -- pass --resume to continue it, or use a new name")
    if smoke:
        log(f"smoke run into {out_dir}: no clean-tree requirement, not an artifact")
    else:
        require_clean_tree([output, cache])
    parts = [p for p in args.parts.split(",") if p]
    for part in parts:
        if part not in PARTS:
            raise SystemExit(f"unknown part {part!r}")
    if "cold_repeat" in parts and "cold_held" not in parts:
        raise SystemExit("part cold_repeat needs part cold_held")
    day_ids = list(artifacts["day_ids"])
    n_scenarios = artifacts["n_scenarios"]
    if args.limit_units is not None:
        if not smoke:
            raise SystemExit("--limit-units is for smoke runs (--out-dir) only")
        day_ids, n_scenarios = day_ids[: args.limit_units], min(n_scenarios, args.limit_units)
    tolerance = DRY_MARGIN_TOLERANCE if dry else study_config.control.margin_tolerance_kg_per_kg
    payload: dict[str, Any] = {
        "provenance": provenance("not used", ignore_paths=[output, cache]),
        "study": "order dependence of the steady-state optimiser (docs/NN_order_dependence.md)",
        "stage": args.stage,
        "smoke": smoke,
        "settings": dataclasses.asdict(SETTINGS),
        "settings_fingerprint": SETTINGS.fingerprint,
        "artifacts": {k: v for k, v in artifacts.items() if k != "stage0"},
        "stage0_of_the_artifact": artifacts["stage0"],
        "hours": DRY_HOURS if dry else None,
        "margin_tolerance": tolerance,
        "workers": args.workers,
        "day_ids": day_ids,
        "n_scenarios": n_scenarios,
    }
    header = {
        "commit": git("rev-parse", "HEAD"),
        "settings": SETTINGS.fingerprint,
        "stage": args.stage,
        "fingerprints": artifacts["fingerprints"],
        "margins": artifacts["margins"],
        "limit_units": args.limit_units,
    }
    started = time.time()
    log(f"stage {args.stage} started, name {args.name}, workers {args.workers}")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(dry, artifacts["stage0"])
    ) as pool:
        ev = Evaluator(pool, args.workers, cache, header)
        study = Study(
            ev,
            load,
            study_config,
            artifacts["margins"],
            day_ids,
            n_scenarios,
            margin_tolerance=tolerance,
        )
        if args.stage == "probe":
            payload["probe"] = stage_probe(study, args.workers)
            log(f"probe: estimate of the main run {payload['probe']['estimate_wall_h']:.1f} h")
        else:
            runtime: dict[str, float] = {}
            run: dict[str, Callable[[], dict[str, Any]]] = {
                part: getattr(study, f"part_{part}") for part in PARTS
            }
            for part in PARTS:
                if part not in parts:
                    continue
                t0 = time.time()
                payload[part] = run[part]()
                runtime[part] = time.time() - t0
                log(f"part {part} done in {runtime[part] / 60:.1f} min")
            payload["parts"] = parts
            payload["part_runtime_s"] = runtime
        payload["errors"] = ev.errors
        payload["units_run"] = ev.units_run
        payload["peak_memory_mb_per_worker"] = sorted(ev.peak_mb.values())
        payload["seconds_per_task"] = {
            label: {"mean": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)}
            for label, v in ev.seconds.items()
        }
    payload["runtime_s"] = time.time() - started
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None, separators=(",", ":"))
    log(
        f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB) after "
        f"{payload['runtime_s'] / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
