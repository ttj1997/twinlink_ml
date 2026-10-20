"""Dynamic headroom study: required margins and energies of R0-R3 on the grey-box truth model.

Study document: ``docs/24_dynamic_headroom.md``; settings and placeholders:
``conf/headroom/dynamic_headroom.yaml``; reading (pre-registered): docstring of
``scripts/analyze_dynamic_headroom.py``.

Question: in continuous operation, does a dynamic operation save energy over quasi-stationary
RTO when disturbances are ridden out? Truth = the unchanged grey-box model (4 states, RK4, 5 s),
no surrogate. Quality limit 0.15 kg/kg as a PATH condition at every truth sample.

Controllers (``twinlink_ml.headroom.controllers``): R0 fixed point, R1 quasi-stationary RTO
(perfect knowledge of the current disturbance), R2 dynamic optimum with perfect preview
(multiple shooting, 5-min piecewise-constant commands, CasADi/IPOPT), R3 MPC without disturbance
preview (2 h horizon, receding). Each with the smallest family-wide moisture margin that keeps
the path condition in all scenarios of the family (bisection; tolerance and bracket in the
config; violation tolerance 1e-4 kg/kg as ``operating_point.TOLERANCE_MOISTURE_KG_PER_KG``).

Stages (``--stages``, default all, in this order)
-------------------------------------------------
``observed`` / ``stress``
    R1 search (first evaluation at margin 0 on all scenarios), R0 search, R3 and R2 searches on a
    critical subset seeded with the three scenarios per type with the largest R1 violation at
    margin 0, each verified on all scenarios (``margin.bisect_margin_over_scenarios``); then R1 at
    the margins of R3 and R2 (decomposition into margin and trajectory share).
``product_change``
    R1, R2, R3 at the margins of the family "observed" on the changeover scenarios.
``robustness``
    20 GAB spreads of the TRUTH model (controllers keep the nominal model) on the subset
    ``robustness.scenario_indices`` of every observed type: R1 per-draw margin (bisection per
    draw), its 95 % margin (19 of 20 draws hold); R3 95 % margin by ``bisect_margin_k_of_n``
    (critical draws ranked by the R1 per-draw margin); R1 and R3 at their 95 % margins on the
    NOMINAL truth (energy cost of robustness).

Output: ``artifacts/runs/<name>.json.gz`` (written once at the end, never overwritten). Every
single evaluation is appended to ``artifacts/runs/_<name>_cache.jsonl`` (gitignored); ``--resume``
reuses it (same commit and config required). Run from the worktree with
``tools/run_here.sh scripts/run_dynamic_headroom.py --name dynamic_headroom_v1``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import REPO_ROOT, RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.headroom.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_headroom_config,
    study_process_config,
)
from twinlink_ml.headroom.margin import (  # noqa: E402
    MarginSearch,
    bisect_margin,
    bisect_margin_k_of_n,
    bisect_margin_over_scenarios,
)
from twinlink_ml.headroom.scenarios import Scenario, build_scenarios  # noqa: E402
from twinlink_ml.headroom.study import Solvers, isotherm_draws, run_controller  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning  # noqa: E402

STAGES = ("observed", "stress", "product_change", "robustness")
SEEDS_PER_TYPE = 3
MAX_MARGIN = 0.12

# ------------------------------------------------------------------------------ worker side

_WORKER: dict[str, Any] = {}


def _init_worker(config_path: str) -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    study = load_headroom_config(config_path)
    process = study_process_config(study)
    _WORKER["study"] = study
    _WORKER["process"] = process
    _WORKER["solvers"] = Solvers(study, process)
    _WORKER["scenarios"] = {}
    _WORKER["draws"] = None


def _scenarios(family: str) -> list[Scenario]:
    cache = _WORKER["scenarios"]
    if family not in cache:
        cache[family] = build_scenarios(_WORKER["study"], _WORKER["process"], family)
    return cache[family]


def _truth(variant: int | None) -> Any:
    if variant is None:
        return None
    if _WORKER["draws"] is None:
        _WORKER["draws"], _ = isotherm_draws(_WORKER["study"], _WORKER["process"])
    return _WORKER["draws"][variant].process


def _task(kind: str, margin: float, family: str, ids: list[int], variant: int | None) -> list[dict]:
    scenarios = [_scenarios(family)[i] for i in ids]
    records = run_controller(
        kind,  # type: ignore[arg-type]
        margin,
        scenarios,
        _WORKER["solvers"],
        truth=_truth(variant),
        variant=variant,
    )
    return [r.as_dict() for r in records]


def _task_r1_draw_search(
    draw: int, ids: list[int], lower: float, upper: float, tolerance: float, violation_tol: float
) -> dict[str, Any]:
    """Per-draw R1 margin search inside one worker (R1 is cheap; the draws run in parallel)."""
    collected: list[dict] = []

    def feasible(m: float) -> bool:
        records = _task("R1", m, "observed", ids, draw)
        collected.extend(records)
        return all(r["max_violation"] <= violation_tol for r in records)

    search = bisect_margin(
        feasible, lower=lower, upper=upper, tolerance=tolerance, max_upper=MAX_MARGIN
    )
    return {"draw": draw, "search": _search_dict(search), "records": collected}


# ------------------------------------------------------------------------------ main side


def _search_dict(search: MarginSearch) -> dict[str, Any]:
    return {
        "margin": search.margin,
        "bracket": list(search.bracket),
        "converged": search.converged,
        "evaluations": [[m, ok] for m, ok in search.evaluations],
    }


def _key(kind: str, family: str, variant: int | None, margin: float, scenario: str) -> str:
    return f"{kind}|{family}|{variant}|{margin!r}|{scenario}"


class Evaluator:
    """Runs evaluations in the pool and keeps every record in the resumable cache."""

    def __init__(self, pool: ProcessPoolExecutor, workers: int, cache: Path, header: dict) -> None:
        self.pool = pool
        self.workers = workers
        self.cache_path = cache
        self.records: dict[str, dict] = {}
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
            print(f"resume: {len(self.records)} cached evaluations")
        else:
            with cache.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"header": header}) + "\n")
        self.names: dict[str, list[str]] = {}

    def names_of(self, family: str) -> list[str]:
        if family not in self.names:
            self.names[family] = [s.name for s in build_scenarios(STUDY, PROCESS, family)]
        return self.names[family]

    def store(self, record: dict) -> None:
        key = _key(
            record["controller"],
            record["family"],
            record["variant"],
            record["margin"],
            record["scenario"],
        )
        self.records[key] = record
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "record": record}) + "\n")

    def run_many(
        self, requests: Sequence[tuple[str, float, str, Sequence[int], int | None]]
    ) -> list[list[dict]]:
        """Records for several (kind, margin, family, ids, variant) requests, in request order."""
        tasks = []
        for kind, margin, family, ids, variant in requests:
            names = self.names_of(family)
            missing = [
                i for i in ids if _key(kind, family, variant, margin, names[i]) not in self.records
            ]
            if not missing:
                continue
            heavy = kind in ("R2", "R3")
            total_missing = sum(len(r[3]) for r in requests)
            per_chunk = (
                max(1, math.ceil(total_missing / self.workers))
                if heavy
                else max(1, math.ceil(len(missing) / self.workers))
            )
            for start in range(0, len(missing), per_chunk):
                tasks.append((kind, margin, family, missing[start : start + per_chunk], variant))
        futures = [self.pool.submit(_task, *task) for task in tasks]
        for future in futures:
            for record in future.result():
                self.store(record)
        out = []
        for kind, margin, family, ids, variant in requests:
            names = self.names_of(family)
            out.append([self.records[_key(kind, family, variant, margin, names[i])] for i in ids])
        return out

    def run(
        self, kind: str, margin: float, family: str, ids: Sequence[int], variant: int | None = None
    ) -> list[dict]:
        return self.run_many([(kind, margin, family, ids, variant)])[0]


def _summary(record: dict, *, arrays: bool) -> dict:
    """Record for the artifact; per-interval arrays only where asked (size), rounded."""
    keep = {
        k: v
        for k, v in record.items()
        if k not in ("energy_kwh_per_interval", "moisture_max_per_interval", "commands")
    }
    commands = np.array(record["commands"])
    keep["gas_velocity_command_range"] = [float(commands[:, 1].min()), float(commands[:, 1].max())]
    keep["inlet_command_mean"] = float(commands[:, 0].mean())
    if arrays:
        keep["energy_kwh_per_interval"] = [
            float(f"{v:.6g}") for v in record["energy_kwh_per_interval"]
        ]
        keep["moisture_max_per_interval"] = [
            float(f"{v:.6g}") for v in record["moisture_max_per_interval"]
        ]
        keep["inlet_command_per_interval"] = [float(f"{v:.5g}") for v in commands[:, 0]]
    return keep


def _seeds(records: list[dict], per_type: int) -> list[int]:
    by_kind: dict[str, list[tuple[float, int]]] = {}
    for i, record in enumerate(records):
        by_kind.setdefault(record["kind"], []).append((record["max_violation"], i))
    seeds = []
    for values in by_kind.values():
        seeds += [i for _, i in sorted(values, reverse=True)[:per_type]]
    return sorted(seeds)


def stage_family(ev: Evaluator, family: str) -> dict[str, Any]:
    control = STUDY.control
    tol = control.violation_tolerance_kg_per_kg
    n = len(ev.names_of(family))
    all_ids = list(range(n))
    out: dict[str, Any] = {"n_scenarios": n, "scenarios": ev.names_of(family)}
    timing: dict[str, float] = {}

    def violations(kind: str) -> Any:
        def call(m: float, ids: Sequence[int]) -> np.ndarray:
            return np.array([r["max_violation"] for r in ev.run(kind, m, family, ids)])

        return call

    searches: dict[str, Any] = {}
    margins: dict[str, float] = {}
    for kind, seeds in (("R1", None), ("R0", None), ("R3", "r1"), ("R2", "r1")):
        start = time.time()
        initial = None
        upper = control.margin_upper_kg_per_kg
        if seeds == "r1":
            initial = _seeds(ev.run("R1", 0.0, family, all_ids), SEEDS_PER_TYPE)
            # Bracket from R1's margin (fewer expensive evaluations; doubled if infeasible, so
            # the result is the same to within the bisection tolerance).
            upper = max(2.0 * margins["R1"], 4.0 * control.margin_tolerance_kg_per_kg)
        search, final = bisect_margin_over_scenarios(
            violations(kind),
            n,
            tolerance_violation=tol,
            upper=upper,
            tolerance=control.margin_tolerance_kg_per_kg,
            max_upper=MAX_MARGIN,
            initial_critical=initial,
        )
        searches[kind] = {**_search_dict(search), "initial_critical": initial}
        margins[kind] = search.margin
        timing[f"search_{kind}_s"] = time.time() - start
        print(
            f"[{family}] {kind}: margin {search.margin:.6f} (bracket {search.bracket}), "
            f"max final violation {final.max():.2e}, {timing[f'search_{kind}_s'] / 60:.1f} min",
            flush=True,
        )
    start = time.time()
    decomposition = {
        "R1_at_R3_margin": ev.run("R1", margins["R3"], family, all_ids),
        "R1_at_R2_margin": ev.run("R1", margins["R2"], family, all_ids),
    }
    timing["decomposition_s"] = time.time() - start
    main = {kind: ev.run(kind, margins[kind], family, all_ids) for kind in ("R0", "R1", "R2", "R3")}
    out["margins"] = margins
    out["searches"] = searches
    out["records"] = {
        kind: [_summary(r, arrays=kind in ("R1", "R2", "R3")) for r in recs]
        for kind, recs in main.items()
    }
    out["records"].update(
        {k: [_summary(r, arrays=False) for r in recs] for k, recs in decomposition.items()}
    )
    out["timing_s"] = timing
    return out


def stage_product_change(ev: Evaluator, margins: dict[str, float]) -> dict[str, Any]:
    n = len(ev.names_of("product_change"))
    ids = list(range(n))
    start = time.time()
    records = {
        kind: ev.run(kind, margins[kind], "product_change", ids) for kind in ("R1", "R2", "R3")
    }
    return {
        "margins_from": "observed",
        "margins": {k: margins[k] for k in ("R1", "R2", "R3")},
        "scenarios": ev.names_of("product_change"),
        "records": {k: [_summary(r, arrays=True) for r in v] for k, v in records.items()},
        "timing_s": {"total_s": time.time() - start},
    }


def stage_robustness(ev: Evaluator, margins: dict[str, float]) -> dict[str, Any]:
    rob = STUDY.robustness
    control = STUDY.control
    tol = control.violation_tolerance_kg_per_kg
    draws, rejected = isotherm_draws(STUDY, PROCESS)
    observed = build_scenarios(STUDY, PROCESS, "observed")
    subset = [i for i, s in enumerate(observed) if s.index in rob.scenario_indices]
    n_draws = len(draws)
    k_required = math.ceil(rob.holding_share * n_draws - 1e-12)
    timing: dict[str, float] = {}

    # R1 per-draw margins (in parallel over draws)
    start = time.time()
    futures = [
        ev.pool.submit(
            _task_r1_draw_search,
            d.index,
            subset,
            0.0,
            control.margin_upper_kg_per_kg,
            control.margin_tolerance_kg_per_kg,
            tol,
        )
        for d in draws
    ]
    r1_draw = []
    for future in futures:
        result = future.result()
        for record in result["records"]:
            ev.store(record)
        r1_draw.append({"draw": result["draw"], **result["search"]})
    timing["r1_draw_searches_s"] = time.time() - start
    per_draw = np.array([d["margin"] for d in sorted(r1_draw, key=lambda d: d["draw"])])
    m1_robust = float(np.sort(per_draw)[k_required - 1])
    print(f"[robustness] R1 95 % margin {m1_robust:.6f}", flush=True)

    # R3: k-of-n search, critical draws ranked by the R1 per-draw margin
    start = time.time()

    def holds(m: float, draw_ids: Sequence[int]) -> np.ndarray:
        results = ev.run_many([("R3", m, "observed", subset, d) for d in draw_ids])
        return np.array([all(r["max_violation"] <= tol for r in recs) for recs in results])

    ranking = list(np.argsort(-per_draw))
    r3_search, r3_holds = bisect_margin_k_of_n(
        holds,
        n_draws,
        k_required,
        ranking,
        lower=margins["R3"],
        # bracket from R1's robustness shift; doubled if infeasible (see stage_family)
        upper=margins["R3"]
        + max(2.0 * (m1_robust - margins["R1"]), 4.0 * control.margin_tolerance_kg_per_kg),
        tolerance=control.margin_tolerance_kg_per_kg,
        max_upper=MAX_MARGIN,
    )
    timing["r3_search_s"] = time.time() - start
    print(f"[robustness] R3 95 % margin {r3_search.margin:.6f}", flush=True)

    # energy cost on the nominal truth
    start = time.time()
    nominal = {
        "R1_nominal_margin": ev.run("R1", margins["R1"], "observed", subset),
        "R1_robust_margin": ev.run("R1", m1_robust, "observed", subset),
        "R3_nominal_margin": ev.run("R3", margins["R3"], "observed", subset),
        "R3_robust_margin": ev.run("R3", r3_search.margin, "observed", subset),
    }
    timing["cost_runs_s"] = time.time() - start
    r1_holds_at_robust = [bool(m <= m1_robust) for m in per_draw]
    return {
        "draws": [{"index": d.index, **d.parameters} for d in draws],
        "rejected_draws": rejected,
        "k_required": k_required,
        "subset": [observed[i].name for i in subset],
        "R1": {
            "per_draw": sorted(r1_draw, key=lambda d: d["draw"]),
            "robust_margin": m1_robust,
            "holds_at_robust_margin": r1_holds_at_robust,
        },
        "R3": {
            "search": _search_dict(r3_search),
            "robust_margin": r3_search.margin,
            "holds_at_robust_margin": [bool(h) for h in r3_holds],
        },
        "nominal_margins": {"R1": margins["R1"], "R3": margins["R3"]},
        "records_nominal_truth": {
            k: [_summary(r, arrays=False) for r in v] for k, v in nominal.items()
        },
        "timing_s": timing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="study YAML (default: conf/headroom/dynamic_headroom.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RUNS_DIR),
        help="output directory; anything but artifacts/runs is a smoke run (no clean-tree check)",
    )
    args = parser.parse_args()
    global STUDY, PROCESS
    STUDY = load_headroom_config(args.config)
    PROCESS = study_process_config(STUDY)
    out_dir = Path(args.out_dir).resolve()
    smoke = out_dir != RUNS_DIR.resolve()
    stages = [s for s in args.stages.split(",") if s]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s!r}")
    output = out_dir / f"{args.name}.json.gz"
    cache = out_dir / f"_{args.name}_cache.jsonl"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if cache.exists() and not args.resume:
        raise SystemExit(f"{cache} exists -- pass --resume to continue it, or use a new name")
    if smoke:
        print(f"smoke run into {out_dir}: no clean-tree requirement, not an artifact")
    else:
        require_clean_tree([output, cache])
    prov = provenance("not used", ignore_paths=[output, cache])
    header = {
        "commit": git("rev-parse", "HEAD"),
        "study_fingerprint": STUDY.fingerprint,
        "process_fingerprint": PROCESS.fingerprint,
    }
    started = time.time()
    payload: dict[str, Any] = {
        "provenance": prov,
        "study": "dynamic headroom (docs/24_dynamic_headroom.md)",
        "study_config": STUDY.raw,
        "study_config_path": str(Path(args.config).resolve().relative_to(REPO_ROOT))
        if Path(args.config).resolve().is_relative_to(REPO_ROOT)
        else str(args.config),
        "smoke": smoke,
        "study_fingerprint": STUDY.fingerprint,
        "process_config_fingerprint": PROCESS.fingerprint,
        "workers": args.workers,
        "stages": stages,
    }
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.config,)
    ) as pool:
        ev = Evaluator(pool, args.workers, cache, header)
        for stage in stages:
            t0 = time.time()
            if stage in ("observed", "stress"):
                payload[stage] = stage_family(ev, stage)
            elif stage == "product_change":
                if "observed" not in payload:
                    raise SystemExit("product_change needs the observed margins (stage observed)")
                payload[stage] = stage_product_change(ev, payload["observed"]["margins"])
            elif stage == "robustness":
                if "observed" not in payload:
                    raise SystemExit("robustness needs the observed margins (stage observed)")
                payload[stage] = stage_robustness(ev, payload["observed"]["margins"])
            payload.setdefault("stage_runtime_s", {})[stage] = time.time() - t0
            print(f"stage {stage} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    payload["runtime_s"] = time.time() - started
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


warnings.simplefilter("ignore", PlaceholderConfigWarning)
STUDY = load_headroom_config()
PROCESS = study_process_config(STUDY)

if __name__ == "__main__":
    raise SystemExit(main())
