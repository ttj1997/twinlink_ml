"""Load-following study: throughput as a manipulated variable under an hourly price.

Study document: ``docs/NN_load_following.md``; settings and placeholders:
``conf/load_following/load_following.yaml``; reading (pre-registered): docstring of
``scripts/analyze_load_following.py``.

Question: with a production target per day and an hourly tariff, does shifting the throughput in
time lower the cost per kg of product at a held moisture limit, against constant throughput with
a quasi-stationary optimiser? Truth = the unchanged grey-box model (4 states, RK4, 5 s), no
surrogate. Quality limit as a PATH condition at every truth sample, also in the tail of the day.

Controllers (``twinlink_ml.load_following.controllers``): C0 constant throughput (reference), C1
price rule, C2 schedule on the steady-state map, C3 economic MPC, C4 perfect preview. Each with
the smallest family-wide moisture margin that keeps the path condition on all days of the family
(bisection, ``twinlink_ml.headroom.margin``; settings of the headroom config). The tail of every
run uses the margin of C0.

Stages (``--stage``)
--------------------
``stage0``
    Steady-state map over the spray rate, admissible range, the two bands (main, box), the day
    sets (training days for tuning, both candidate sets of test days), tuning of C1 on the
    training days, static gain on the training days. Single process, minutes.
    Output ``<name>_stage0.json.gz``.
``dryrun``
    ALL units of the main run on days shortened to ``--day-hours`` (default 2 h plus the tail)
    with a coarse margin tolerance, on the larger candidate set of test days: surfaces failing
    units, reports time and peak memory per worker. Output ``<name>_dryrun.json.gz``.
``probe``
    Full-length days: C0-C4 on three TRAINING days at a fixed margin under parallel load; from the
    times the wall-clock time of the main run is extrapolated for both candidate sets of test
    days and the set is chosen by the rule fixed beforehand (:func:`extrapolate_hours`; whole
    test months if the estimate is within the budget, else the stratified days).
    Output ``<name>_probe.json.gz``.
``main``
    Parts (``--parts``, default all, in this order): ``main`` (variant E, main band: margin
    searches of C0, C1, C2 on all days, of C4 and C3 on a critical subset verified on all days),
    ``decomposition`` (C0 at the margins of the others), ``control`` (constant price: C0, C2, C4,
    C3 at their margins), ``gas`` (variant G: C1, C2), ``weekly`` (weekly target, steady-state
    map only), ``box`` (whole admissible box: C2, C4). Output ``<name>.json.gz``.

Every unit (controller x day x margin) is appended to ``artifacts/runs/_<name>_<stage>_cache.jsonl``
(gitignored) as soon as it is finished; ``--resume`` reuses it (same commit and config required).
A failed unit is logged and recorded as an error; it does not discard other units. Run from the
worktree with ``tools/run_here.sh scripts/run_load_following.py --stage ... --name ...``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
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

from twinlink_ml.headroom.margin import (  # noqa: E402
    MarginSearch,
    bisect_margin,
    bisect_margin_over_scenarios,
)
from twinlink_ml.load_following.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    LoadFollowingConfig,
    load_load_following_config,
)
from twinlink_ml.load_following.market import (  # noqa: E402
    Market,
    build_days,
    build_market,
    daily_spread,
    stratified_days,
)
from twinlink_ml.load_following.memory import peak_memory_mb  # noqa: E402
from twinlink_ml.load_following.stage0 import (  # noqa: E402
    admissible_range,
    bands,
    hourly_curves,
    map_table,
    static_costs,
    tune_rule,
    weekly_costs,
)
from twinlink_ml.load_following.steady import SteadyMap  # noqa: E402
from twinlink_ml.load_following.study import RunSpec, Solvers, run_days  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning  # noqa: E402

STAGES = ("stage0", "dryrun", "probe", "main")
PARTS = ("main", "decomposition", "control", "gas", "weekly", "box")
HEAVY = ("C3", "C4")
DRY_MARGIN_TOLERANCE = 4.0e-3
PROBE_MARGIN = 0.005
EXTRAPOLATION_SAFETY = 1.25
WEEKLY_HOURS = 0.3


def log(message: str) -> None:
    """One progress line with a time stamp."""
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}", flush=True)


# ------------------------------------------------------------------------------ worker side

_WORKER: dict[str, Any] = {}


def _init_worker(config_path: str, day_hours: int | None) -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    config = load_load_following_config(config_path, day_hours=day_hours)
    market = build_market(config, REPO_ROOT)
    _WORKER.update(
        config=config, market=market, solvers=Solvers(config, market.reference_tariff), days={}
    )


def _days(ids: Sequence[int], mode: str) -> list[Any]:
    cache = _WORKER["days"]
    missing = [i for i in ids if (i, mode) not in cache]
    if missing:
        built = build_days(_WORKER["config"], _WORKER["market"], missing, price_mode=mode)  # type: ignore[arg-type]
        for day in built:
            cache[(day.day, mode)] = day
    return [cache[(i, mode)] for i in ids]


def _task(spec: dict[str, Any], ids: list[int], mode: str) -> dict[str, Any]:
    """Run one spec on some days; an exception becomes an error entry (never lost)."""
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    error = None
    try:
        run_spec = RunSpec(**{**spec, "band": tuple(spec["band"])})
        records = run_days(run_spec, _days(ids, mode), _WORKER["solvers"])
    except Exception:  # a failed unit must not stop the run
        error = traceback.format_exc()
    return {
        "records": records,
        "error": error,
        "seconds": time.perf_counter() - started,
        "peak_mb": peak_memory_mb(),
        "pid": os.getpid(),
    }


# ------------------------------------------------------------------------------ main side


def _spec_dict(spec: RunSpec) -> dict[str, Any]:
    out = dataclasses.asdict(spec)
    out["band"] = list(spec.band)
    return out


def _key(spec: RunSpec, mode: str, day: int) -> str:
    return (
        f"{spec.kind}|{spec.carrier}|{spec.band_name}|{mode}|"
        f"{spec.margin!r}|{spec.tail_margin!r}|{day}"
    )


class Evaluator:
    """Runs units in the pool and keeps every record in the resumable cache."""

    def __init__(self, pool: ProcessPoolExecutor, workers: int, cache: Path, header: dict) -> None:
        self.pool = pool
        self.workers = workers
        self.cache_path = cache
        self.records: dict[str, dict] = {}
        self.errors: list[dict[str, Any]] = []
        self.peak_mb: dict[int, float] = {}
        self.seconds: dict[str, list[float]] = {}
        self.units_run = 0
        if cache.exists():
            with cache.open(encoding="utf-8") as handle:
                first = json.loads(handle.readline())
                if first.get("header") != header:
                    raise SystemExit(
                        f"{cache} was written by another commit/config: {first.get('header')}"
                    )
                for line in handle:
                    entry = json.loads(line)
                    self.records[entry["key"]] = entry["record"]
            log(f"resume: {len(self.records)} cached units")
        else:
            with cache.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"header": header}) + "\n")

    def _store(self, key: str, record: dict) -> None:
        self.records[key] = record
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "record": record}) + "\n")

    def _finish(self, spec: RunSpec, mode: str, ids: list[int], out: dict[str, Any]) -> list[int]:
        """Store the result of one task; returns the days to retry one by one."""
        self.peak_mb[out["pid"]] = max(self.peak_mb.get(out["pid"], 0.0), out["peak_mb"])
        if out["error"] is None:
            per_day = out["seconds"] / len(ids)
            self.seconds.setdefault(spec.kind, []).extend([per_day] * len(ids))
            for record in out["records"]:
                self._store(_key(spec, mode, record["day_of_year"]), record)
            self.units_run += len(ids)
            return []
        if len(ids) > 1:
            log(f"batch {spec.kind} {mode} {ids} failed, retrying day by day")
            return ids
        entry = {
            "controller": spec.kind,
            "carrier": spec.carrier,
            "band": spec.band_name,
            "price_mode": mode,
            "margin": spec.margin,
            "day_of_year": ids[0],
            "error": out["error"],
        }
        self.errors.append(entry)
        log(
            f"ERROR unit {spec.kind} {mode} day {ids[0]} margin {spec.margin}: {out['error'][-300:]}"
        )
        self._store(_key(spec, mode, ids[0]), {**entry, "max_violation": None})
        self.units_run += 1
        return []

    def run(self, spec: RunSpec, ids: Sequence[int], mode: str = "true") -> list[dict]:
        """Records of ``spec`` on the days ``ids`` (in that order), from the cache or computed."""
        missing = [i for i in ids if _key(spec, mode, i) not in self.records]
        if missing:
            per_task = 1 if spec.kind in HEAVY else max(1, math.ceil(len(missing) / self.workers))
            chunks = [missing[s : s + per_task] for s in range(0, len(missing), per_task)]
            started = time.time()
            while chunks:
                futures: dict[Future, list[int]] = {
                    self.pool.submit(_task, _spec_dict(spec), chunk, mode): chunk
                    for chunk in chunks
                }
                chunks = []
                done = 0
                for future in as_completed(futures):
                    chunk = futures[future]
                    out = future.result()
                    done += len(chunk)
                    retry = self._finish(spec, mode, chunk, out)
                    chunks.extend([[day] for day in retry])
                    log(
                        f"unit {spec.kind}/{spec.carrier}/{spec.band_name}/{mode} m={spec.margin:.5f} "
                        f"days {chunk[0]}..{chunk[-1]} ({len(chunk)}) {out['seconds']:.0f} s "
                        f"[{done} done of {len(missing)}, {time.time() - started:.0f} s]"
                    )
        return [self.records[_key(spec, mode, i)] for i in ids]


def _violations(records: Sequence[dict]) -> np.ndarray:
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
    """Margin searches and parts of the main run (shared by ``dryrun`` and ``main``)."""

    def __init__(
        self,
        ev: Evaluator,
        config: LoadFollowingConfig,
        stage0: dict[str, Any],
        ids: list[int],
        *,
        margin_tolerance: float,
    ) -> None:
        self.ev = ev
        self.config = config
        self.ids = ids
        control = config.headroom.control
        self.tol_violation = control.violation_tolerance_kg_per_kg
        self.tol_margin = margin_tolerance
        self.upper = control.margin_upper_kg_per_kg
        self.maximum = config.margin_maximum
        self.bands = {name: tuple(band) for name, band in stage0["bands"].items()}
        self.rule = stage0["rule"]
        self.reduced = bool(stage0["reduced_stage1"])
        self.margins: dict[str, float] = {}

    # ---------------------------------------------------------------------- specs

    def spec(self, kind: str, margin: float, *, band: str = "main", carrier: str = "E") -> RunSpec:
        tail = margin if kind == "C0" else self.margins["C0"]
        rule = self.rule[carrier]
        return RunSpec(
            kind=kind,  # type: ignore[arg-type]
            margin=float(margin),
            tail_margin=float(tail),
            band=self.bands[band],  # type: ignore[arg-type]
            band_name=band,
            carrier=carrier,  # type: ignore[arg-type]
            rule_quantile=rule["quantile"] if kind == "C1" else None,
            rule_spread_kg_per_h=rule["spread_kg_per_h"] if kind == "C1" else None,
        )

    # ---------------------------------------------------------------------- searches

    def search_all(self, make: Callable[[float], RunSpec], label: str) -> dict[str, Any]:
        """Bisection on all days (cheap controllers)."""

        def feasible(m: float) -> bool:
            return bool(np.all(_violations(self.ev.run(make(m), self.ids)) <= self.tol_violation))

        search = bisect_margin(
            feasible,
            lower=0.0,
            upper=self.upper,
            tolerance=self.tol_margin,
            max_upper=self.maximum,
        )
        records = self.ev.run(make(search.margin), self.ids)
        log(f"margin {label}: {search.margin:.5f} (converged {search.converged})")
        return {"search": _search_dict(search), "margin": search.margin, "records": records}

    def search_subset(
        self, make: Callable[[float], RunSpec], label: str, critical: list[int], upper: float
    ) -> dict[str, Any]:
        """Bisection on a critical subset, verified on all days (expensive controllers)."""

        def violations(m: float, idx: Sequence[int]) -> np.ndarray:
            return _violations(self.ev.run(make(m), [self.ids[i] for i in idx]))

        search, final = bisect_margin_over_scenarios(
            violations,
            len(self.ids),
            tolerance_violation=self.tol_violation,
            lower=0.0,
            upper=upper,
            tolerance=self.tol_margin,
            max_upper=self.maximum,
            initial_critical=critical,
        )
        records = self.ev.run(make(search.margin), self.ids)
        log(f"margin {label}: {search.margin:.5f} (converged {search.converged})")
        return {
            "search": _search_dict(search),
            "margin": search.margin,
            "critical_seed": [self.ids[i] for i in critical],
            "largest_violation": float(np.max(final)),
            "records": records,
        }

    def _critical(self, make: Callable[[float], RunSpec]) -> list[int]:
        """Indices of the days with the largest violation of a cheap controller at margin 0."""
        violations = _violations(self.ev.run(make(0.0), self.ids))
        order = np.argsort(-violations, kind="stable")
        return sorted(int(i) for i in order[: min(self.config.critical_days, len(self.ids))])

    def _heavy_upper(self, cheap_margin: float) -> float:
        return max(2.0 * cheap_margin, 4.0 * self.tol_margin)

    # ---------------------------------------------------------------------- parts

    def part_main(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["C0"] = self.search_all(lambda m: self.spec("C0", m), "C0")
        self.margins["C0"] = out["C0"]["margin"]
        if not self.reduced:
            out["C1"] = self.search_all(lambda m: self.spec("C1", m), "C1")
        out["C2"] = self.search_all(lambda m: self.spec("C2", m), "C2")
        critical = self._critical(lambda m: self.spec("C2", m))
        upper = self._heavy_upper(out["C2"]["margin"])
        out["C4"] = self.search_subset(lambda m: self.spec("C4", m), "C4", critical, upper)
        if not self.reduced:
            out["C3"] = self.search_subset(lambda m: self.spec("C3", m), "C3", critical, upper)
        for kind, block in out.items():
            self.margins[kind] = block["margin"]
        return out

    def part_decomposition(self) -> dict[str, Any]:
        """C0 at the margin of each other controller (margin share against schedule share)."""
        return {
            kind: self.ev.run(
                dataclasses.replace(self.spec("C0", margin), tail_margin=self.margins["C0"]),
                self.ids,
            )
            for kind, margin in self.margins.items()
            if kind != "C0"
        }

    def part_control(self) -> dict[str, Any]:
        """Constant price (negative control): the controllers at their margins of part main."""
        kinds = ["C0", "C2", "C4"] + ([] if self.reduced else ["C3"])
        return {
            kind: self.ev.run(
                dataclasses.replace(
                    self.spec(kind, self.margins[kind]), tail_margin=self.margins["C0"]
                ),
                self.ids,
                "flat",
            )
            for kind in kinds
        }

    def part_gas(self) -> dict[str, Any]:
        return {
            "C1": self.search_all(lambda m: self.spec("C1", m, carrier="G"), "C1/G"),
            "C2": self.search_all(lambda m: self.spec("C2", m, carrier="G"), "C2/G"),
        }

    def part_box(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["C2"] = self.search_all(lambda m: self.spec("C2", m, band="box"), "C2/box")
        critical = self._critical(lambda m: self.spec("C2", m, band="box"))
        upper = self._heavy_upper(out["C2"]["margin"])
        out["C4"] = self.search_subset(
            lambda m: self.spec("C4", m, band="box"), "C4/box", critical, upper
        )
        return out


def part_weekly(
    config: LoadFollowingConfig, market: Market, ids: list[int], band: tuple[float, float]
) -> dict[str, Any]:
    """Weekly against daily target on the steady-state map (descriptive; hindsight ambient air)."""
    limit = config.headroom.line.product_moisture_limit_kg_per_kg
    steady_map = SteadyMap(config)
    by_month: dict[int, list[int]] = {}
    for day in ids:
        by_month.setdefault(int(market.clock.month[day * 24]), []).append(day)
    weeks_ids = []
    for days in by_month.values():
        days = sorted(days)
        consecutive = bool(np.all(np.diff(days) == 1))
        if not consecutive:
            continue  # stratified days: no whole weeks
        for w in range(config.weeks_per_test_month):
            week = days[7 * w : 7 * w + 7]
            if len(week) == 7:
                weeks_ids.append(week)
    if not weeks_ids:
        return {"weeks": [], "note": "no whole weeks in this set of days"}
    weeks = [build_days(config, market, week) for week in weeks_ids]
    curves = [hourly_curves(config, steady_map, week, band, limit) for week in weeks]
    out = {}
    for carrier in ("E", "G"):
        out[carrier] = weekly_costs(config, weeks, curves, carrier)  # type: ignore[arg-type]
    return {"weeks": [list(w) for w in weeks_ids], "bound": "limit (margin 0)", **out}


# ------------------------------------------------------------------------------ stage 0


def day_sets(config: LoadFollowingConfig, market: Market) -> dict[str, list[int]]:
    """Training days for tuning and the two candidate sets of test days."""
    split = config.prices
    return {
        "training": stratified_days(
            market, split.training_months, config.day.training_per_month_by_tercile, config.day.seed
        ),
        "stratified": stratified_days(
            market, split.test_months, config.day.test_per_month_by_tercile, config.day.seed
        ),
        "whole": market.months(split.test_months),
    }


def stage0(config: LoadFollowingConfig, market: Market) -> dict[str, Any]:
    """Everything of stage 0, see the module docstring."""
    limit = config.headroom.line.product_moisture_limit_kg_per_kg
    nominal = config.nominal_spray_rate
    steady_map = SteadyMap(config)
    log("stage 0: steady-state map")
    table = map_table(config, steady_map)
    grid = np.array(table["spray_kg_per_h"])
    admissible = admissible_range(grid, np.array(table["feasible_with_margin"]), nominal)
    both = bands(config, admissible)
    sets = day_sets(config, market)
    training = build_days(config, market, sets["training"])
    out: dict[str, Any] = {
        "map": table,
        "admissible_range_kg_per_h": list(admissible),
        "bands": {name: list(band) for name, band in both.items()},
        "day_sets": sets,
        "spread_training": [daily_spread(market.tariff, d) for d in sets["training"]],
        "market": {
            "reference_tariff": market.reference_tariff,
            "floored_hours": market.floored_hours,
            "gas_cost_eur_per_kwh_fuel": market.gas_cost_eur_per_kwh_fuel,
            "raw_sha256": market.raw_sha256,
        },
        "static": {},
    }
    for name, band in both.items():
        log(f"stage 0: hourly curves of the training days, band {name} {band}")
        curves = hourly_curves(config, steady_map, training, band, limit)
        if name == "main":
            out["rule"] = {c: tune_rule(config, training, curves, band, c) for c in ("E", "G")}  # type: ignore[arg-type]
        out["static"][name] = {
            c: static_costs(config, training, curves, c)  # type: ignore[arg-type]
            for c in ("E", "G")
        }
    # Rule of the study document (section 8): a reduced stage 1 (C0, C2, C4 only) if the static
    # gain on the training days is below 2 % AND the admissible range is narrower than +-10 %.
    narrow = admissible[0] > 0.9 * nominal or admissible[1] < 1.1 * nominal
    gain = out["static"]["main"]["E"]["gain"]
    out["reduced_stage1"] = bool(gain < 0.02 and narrow)
    out["steady_failures"] = steady_map.failures
    return out


# ------------------------------------------------------------------------------ probe


def extrapolate_hours(
    seconds: dict[str, float], n_days: int, *, workers: int, config: LoadFollowingConfig
) -> dict[str, Any]:
    """Wall-clock estimate of the main run on ``n_days`` test days (rule fixed beforehand).

    ``seconds``: time per day-run of C3 and C4 and per day of a cheap run (C0-C2), measured in
    the probe under parallel load. Counted day-runs: per margin search of an expensive controller
    ``K * S`` (critical subset times bisection evaluations) plus two verification passes over all
    days; plus one pass with a constant price; cheap controllers ``S`` passes per search (C0, C1,
    C2; C1 and C2 of variant G; C2 of the box) plus four decomposition and two control passes.
    ``S = 1 + ceil(log2(upper / tolerance))`` evaluations of a bisection on ``[0, upper]``.
    Safety factor :data:`EXTRAPOLATION_SAFETY`; the weekly part is a constant.
    """
    control = config.headroom.control
    s = 1 + math.ceil(
        math.log2(control.margin_upper_kg_per_kg / control.margin_tolerance_kg_per_kg)
    )
    k = config.critical_days
    runs = {
        "C3": k * s + 3 * n_days,
        "C4": k * s + 3 * n_days,
        "C4_box": k * s + 2 * n_days,
        "cheap": n_days * (6 * s + 4 + 2),
    }
    cpu_s = (
        runs["C3"] * seconds["C3"]
        + (runs["C4"] + runs["C4_box"]) * seconds["C4"]
        + runs["cheap"] * seconds["cheap"]
    )
    factor = EXTRAPOLATION_SAFETY / (3600.0 * workers)
    box_cpu_s = runs["C4_box"] * seconds["C4"] + n_days * s * seconds["cheap"]
    return {
        "day_runs": runs,
        "bisection_evaluations": s,
        "cpu_h": cpu_s / 3600.0,
        "wall_h": factor * cpu_s + WEEKLY_HOURS,
        "wall_h_without_weekly": factor * cpu_s,
        "wall_h_without_weekly_and_box": factor * (cpu_s - box_cpu_s),
    }


def choose_days_and_parts(
    estimate: dict[str, dict[str, Any]], budget_h: float
) -> tuple[str, list[str], str]:
    """Day set and parts of the main run by the rule fixed beforehand (decisions 5 and 7).

    Whole test months if their estimate is within the budget, else the stratified days; if those
    are still above the budget the weekly part is dropped, then the box part. Parts that carry
    the reading are never cut.
    """
    parts = list(PARTS)
    if estimate["whole"]["wall_h"] <= budget_h:
        return "whole", parts, "whole test months within the budget"
    if estimate["stratified"]["wall_h"] <= budget_h:
        return "stratified", parts, "whole test months above the budget; stratified days within"
    parts.remove("weekly")
    if estimate["stratified"]["wall_h_without_weekly"] <= budget_h:
        return "stratified", parts, "stratified days above the budget; weekly part cut"
    parts.remove("box")
    return "stratified", parts, "stratified days above the budget; weekly and box parts cut"


def stage_probe(
    ev: Evaluator, config: LoadFollowingConfig, stage0_data: dict[str, Any], workers: int
) -> dict[str, Any]:
    """Time and memory of full-length day-runs on three training days under parallel load."""
    training = stage0_data["day_sets"]["training"]
    spread = np.array(stage0_data["spread_training"])
    order = np.argsort(spread, kind="stable")
    ids = sorted(
        {training[int(order[0])], training[int(order[len(order) // 2])], training[int(order[-1])]}
    )
    band = tuple(stage0_data["bands"]["main"])
    rule = stage0_data["rule"]["E"]
    specs = [
        RunSpec(
            kind=kind,  # type: ignore[arg-type]
            margin=PROBE_MARGIN,
            tail_margin=PROBE_MARGIN,
            band=band,  # type: ignore[arg-type]
            rule_quantile=rule["quantile"] if kind == "C1" else None,
            rule_spread_kg_per_h=rule["spread_kg_per_h"] if kind == "C1" else None,
        )
        for kind in ("C3", "C4", "C0", "C1", "C2")
    ]
    futures: dict[Future, tuple[RunSpec, list[int]]] = {}
    for spec in specs:
        chunks = [[i] for i in ids] if spec.kind in HEAVY else [ids]
        for chunk in chunks:
            futures[ev.pool.submit(_task, _spec_dict(spec), chunk, "true")] = (spec, chunk)
    times: dict[str, list[float]] = {}
    units = []
    for future in as_completed(futures):
        spec, chunk = futures[future]
        out = future.result()
        if out["error"] is not None:
            raise SystemExit(f"probe unit {spec.kind} {chunk} failed:\n{out['error']}")
        ev.peak_mb[out["pid"]] = max(ev.peak_mb.get(out["pid"], 0.0), out["peak_mb"])
        times.setdefault(spec.kind, []).append(out["seconds"] / len(chunk))
        log(f"probe {spec.kind} days {chunk}: {out['seconds']:.0f} s, peak {out['peak_mb']:.0f} MB")
        for record in out["records"]:
            units.append(
                {
                    "controller": spec.kind,
                    "day_of_year": record["day_of_year"],
                    "seconds_task": out["seconds"],
                    "peak_mb": out["peak_mb"],
                    "max_violation": record["max_violation"],
                    "solver": record["solver"],
                    "decide_s": record["decide_s"],
                    "reset_s": record["reset_s"],
                }
            )
    seconds = {
        "C3": float(np.mean(times["C3"])),
        "C4": float(np.mean(times["C4"])),
        "cheap": float(np.mean([np.mean(times[k]) for k in ("C0", "C1", "C2")])),
    }
    sets = stage0_data["day_sets"]
    estimate = {
        name: extrapolate_hours(seconds, len(sets[name]), workers=workers, config=config)
        for name in ("whole", "stratified")
    }
    budget = config.budget_wall_clock_h
    choice, parts, reason = choose_days_and_parts(estimate, budget)
    return {
        "probe_days": ids,
        "margin": PROBE_MARGIN,
        "seconds_per_day_run": seconds,
        "units": units,
        "estimate": estimate,
        "budget_wall_clock_h": budget,
        "choice": choice,
        "parts": parts,
        "reason": reason,
    }


# ------------------------------------------------------------------------------ entry point


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)  # type: ignore[no-any-return]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--name", required=True, help="base name of the artifacts")
    parser.add_argument("--workers", type=int, default=8, help="at most the physical cores")
    parser.add_argument("--parts", default=",".join(PARTS))
    parser.add_argument("--days", choices=("whole", "stratified"))
    parser.add_argument("--day-hours", type=int, default=2, help="length of a dry-run day")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit-days", type=int, help="smoke runs only: use the first N days of the day set"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--out-dir",
        default=str(RUNS_DIR),
        help="output directory; anything but artifacts/runs is a smoke run (no clean-tree check)",
    )
    args = parser.parse_args()
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    dry = args.stage == "dryrun"
    day_hours = args.day_hours if dry else None
    config = load_load_following_config(args.config, day_hours=day_hours)
    market = build_market(config, REPO_ROOT)
    out_dir = Path(args.out_dir).resolve()
    smoke = out_dir != RUNS_DIR.resolve()
    suffix = "" if args.stage == "main" else f"_{args.stage}"
    output = out_dir / f"{args.name}{suffix}.json.gz"
    cache = out_dir / f"_{args.name}{suffix}_cache.jsonl"
    stage0_path = out_dir / f"{args.name}_stage0.json.gz"
    probe_path = out_dir / f"{args.name}_probe.json.gz"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if cache.exists() and not args.resume:
        raise SystemExit(f"{cache} exists -- pass --resume to continue it, or use a new name")
    if smoke:
        log(f"smoke run into {out_dir}: no clean-tree requirement, not an artifact")
    else:
        require_clean_tree([output, cache])
    payload: dict[str, Any] = {
        "provenance": provenance("not used", ignore_paths=[output, cache]),
        "study": "load following (docs/NN_load_following.md)",
        "stage": args.stage,
        "smoke": smoke,
        "study_config": config.raw,
        "study_fingerprint": config.fingerprint,
        "headroom_fingerprint": config.headroom.fingerprint,
        "process_config_fingerprint": config.process.fingerprint,
        "day_hours": config.day.hours,
        "tail_hours": config.day.tail_hours,
        "workers": args.workers,
    }
    started = time.time()
    log(f"stage {args.stage} started, name {args.name}, workers {args.workers}")
    if args.stage == "stage0":
        payload["stage0"] = stage0(config, market)
    else:
        if not stage0_path.exists():
            raise SystemExit(f"{stage0_path} missing -- run --stage stage0 first")
        stage0_file = _load_json_gz(stage0_path)
        if stage0_file["headroom_fingerprint"] != config.headroom.fingerprint or (
            stage0_file["study_config"] != config.raw
        ):
            raise SystemExit("stage-0 artifact was written with another configuration")
        stage0_data = stage0_file["stage0"]
        payload["stage0_artifact"] = stage0_path.name
        header = {
            "commit": git("rev-parse", "HEAD"),
            "study_fingerprint": config.fingerprint,
            "process_fingerprint": config.process.fingerprint,
            "stage": args.stage,
            "day_hours": config.day.hours,
            "rule": {
                c: [r["quantile"], r["spread_kg_per_h"]] for c, r in stage0_data["rule"].items()
            },
            "bands": stage0_data["bands"],
        }
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(args.config, day_hours)
        ) as pool:
            ev = Evaluator(pool, args.workers, cache, header)
            if args.stage == "probe":
                payload["probe"] = stage_probe(ev, config, stage0_data, args.workers)
                log(
                    f"probe: estimate {payload['probe']['estimate']}, choice {payload['probe']['choice']}"
                )
            else:
                if dry:
                    days_name = "whole"
                    tolerance = DRY_MARGIN_TOLERANCE
                else:
                    if args.days is None:
                        raise SystemExit("--days whole|stratified is required for the main run")
                    days_name = args.days
                    tolerance = config.headroom.control.margin_tolerance_kg_per_kg
                    if probe_path.exists():
                        probe = _load_json_gz(probe_path)["probe"]
                        payload["probe_choice"] = probe["choice"]
                        if probe["choice"] != days_name:
                            log(
                                f"NOTE: --days {days_name} differs from the probe choice {probe['choice']}"
                            )
                ids = list(stage0_data["day_sets"][days_name])
                if args.limit_days is not None:
                    if not smoke:
                        raise SystemExit("--limit-days is for smoke runs (--out-dir) only")
                    ids = ids[: args.limit_days]
                parts = [p for p in args.parts.split(",") if p]
                for part in parts:
                    if part not in PARTS:
                        raise SystemExit(f"unknown part {part!r}")
                if "main" not in parts:
                    raise SystemExit("part main is required (the others use its margins)")
                payload.update(
                    {
                        "days": days_name,
                        "day_ids": ids,
                        "parts": parts,
                        "margin_tolerance": tolerance,
                        "day_price_spread": {str(d): daily_spread(market.tariff, d) for d in ids},
                        "day_month": {str(d): int(market.clock.month[d * 24]) for d in ids},
                        "reference_tariff": market.reference_tariff,
                    }
                )
                study = Study(ev, config, stage0_data, ids, margin_tolerance=tolerance)
                runtime: dict[str, float] = {}
                for part in PARTS:
                    if part not in parts:
                        continue
                    t0 = time.time()
                    if part == "weekly":
                        # whole test months whichever day set is chosen (steady-state map only)
                        payload[part] = part_weekly(
                            config,
                            market,
                            list(stage0_data["day_sets"]["whole"]),
                            study.bands["main"],  # type: ignore[arg-type]
                        )
                    else:
                        payload[part] = getattr(study, f"part_{part}")()
                    runtime[part] = time.time() - t0
                    log(f"part {part} done in {runtime[part] / 60:.1f} min")
                payload["margins"] = study.margins
                payload["part_runtime_s"] = runtime
                payload["reduced_stage1"] = study.reduced
            payload["errors"] = ev.errors
            payload["units_run"] = ev.units_run
            payload["peak_memory_mb_per_worker"] = sorted(ev.peak_mb.values())
            payload["seconds_per_day_run"] = {
                kind: {"mean": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)}
                for kind, v in ev.seconds.items()
            }
    payload["runtime_s"] = time.time() - started
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None)
    log(
        f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB) after {payload['runtime_s'] / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
