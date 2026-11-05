"""Campaign network, follow-up study: does a better simple rule or a rolling planning close the gap.

Study document ``docs/NN_campaign_network_2.md``; model ``twinlink_ml.campaign_network``
(``dynamic.py``: K0b, K1d; ``planning.py``: P1, P2); basis Befund 118 / ``docs/35`` and the artifact
``campaign_network_v2`` (same configuration and schedule draws, paired; main variant 10 K, 80 degC;
process water 0 / 200 kW / seasonal, the levels of ``scripts/run_campaign_network_v2.py``). Reading
(committed before the run): the docstring of ``scripts/analyze_campaign_network_2.py``. Rules and
LP only, no training.

Jobs (one process each):

* ``draw:<level>:<a>:<mode>:<draw>:<power>`` -- one source power on one draw: K0, K0b, K1, K1d
  (storage 0 / 2 / 8 / 24 h), K2b and P1 (storage > 0), P2 (storage > 0 and a < 1; at a = 1 there
  is no change of state, P2 = P1), K3 (every storage size). Every finished job is written to the
  cache at once (``--resume``); the artifact merges the powers of a draw. P1 / P2: horizon 7 d, a new plan every 24 h
  and at once when the observed state differs from the plan. With storage 0, P1 = P2 = K3 without
  storage by structure (not computed). a = 1 is one schedule (computed once).
* ``check:replan`` -- P1 with a new plan every hour against every 24 h on one draw (process water
  200 kW, a = 0.57, independent, draw 0, 800 kW, 24 h): the price of planning once a day.

Times are MEASURED UNDER PARALLEL LOAD (the workers of this run share the machine). Budget rule
(wall-clock, decisions of the project lead 2026-11-04 / 2026-11-05): if the extrapolation of the
probe (``--probe``, not an artifact) is above 30 h or the run would end after the machine is
needed again (``--deadline``), drop storage 2 h first, then the draws to 3; cuts are recorded. At
most 8 workers.

Output ``artifacts/runs/<name>.json.gz`` (never overwritten), cache
``artifacts/runs/_<name>_cache.jsonl``; run from the worktree:
``tools/run_here.sh scripts/run_campaign_network_2.py --name campaign_network_2_v1 --workers 8``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))
sys.path.insert(0, str(SCRIPTS))

from _common import RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402
from run_campaign_network import Cache, _run_jobs, _worker_init  # noqa: E402
from run_campaign_network_v2 import level_config  # noqa: E402

from twinlink_ml.campaign_network import (  # noqa: E402
    build_consumers,
    key_figures,
    plant_schedules,
    run_k0,
    run_k1,
    run_k2,
    run_k3,
)
from twinlink_ml.campaign_network.config import DEFAULT_CONFIG_PATH  # noqa: E402
from twinlink_ml.campaign_network.dynamic import run_k0b, run_k1d  # noqa: E402
from twinlink_ml.campaign_network.planning import (  # noqa: E402
    PlanForecast,
    PlanResult,
    TrueForecast,
    cut_data,
    planned_transitions,
    rolling_plan,
)

MAX_WORKERS = 8
LEVELS = ("water_0", "water_200", "water_seasonal")
HORIZON_H = 168
REPLAN_EVERY_H = 24
TIMING_NOTE = "measured under parallel load ({workers} workers of this run on one machine)"


def plan_figures(res: PlanResult, power_kw: float, cfg: Any) -> dict[str, Any]:
    """Key figures of a planning run (every consumer has the same efficiency, checked by cut_data)."""
    eta = cfg.plants[0].gas_efficiency
    use = float(res.use_kw.sum())
    source = power_kw * res.use_kw.size
    pump = cfg.pump_fraction * use
    gas = use / eta
    return {
        "source_kwh": source,
        "use_kwh": use,
        "utilisation": use / source,
        "gas_kwh": gas,
        "co2_gas_kg": cfg.gas_co2_kg_per_kwh * gas,
        "pump_electricity_kwh": pump,
        "co2_pump_kg": cfg.electricity_co2_kg_per_kwh * pump,
        "co2_net_kg": cfg.gas_co2_kg_per_kwh * gas - cfg.electricity_co2_kg_per_kwh * pump,
        "charge_kwh": float(res.charge_kw.sum()),
        "discharge_kwh": float(res.discharge_kw.sum()),
        "content_end_kwh": float(res.content_kwh[-1]),
        "plans": res.plans,
        "lp_failures": res.lp_failures,
        "solve_seconds": res.solve_seconds,
    }


def job_draw(
    config_path: str,
    level: str,
    a: float,
    mode: str,
    draw: int,
    power: float,
    sizes: tuple[float, ...],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = level_config(config_path, level)
    on = plant_schedules(cfg, a, mode, draw)
    c = build_consumers(cfg, on)
    cuts = cut_data(cfg)
    planned = planned_transitions(cfg, a, mode, draw) if a < 1.0 else None
    seconds: dict[str, float] = {"rules": 0.0, "K3": 0.0, "P1": 0.0, "P2": 0.0}
    out: dict[str, Any] = {
        "availability": a,
        "mode": mode,
        "draw": draw,
        "realised_availability": {k: float(np.mean(v)) for k, v in on.items()},
        "powers": {},
    }
    for p in (power,):
        s0 = time.perf_counter()
        k0 = key_figures(run_k0(c, p, cfg), c, p, cfg)
        k0b = key_figures(run_k0b(c, p, cfg), c, p, cfg)
        k1 = key_figures(run_k1(c, p, cfg), c, p, cfg)
        row: dict[str, Any] = {
            "K0": k0,
            "K0b": k0b,
            "K1": k1,
            "V_N": (k1["use_kwh"] - k0["use_kwh"]) / k0["source_kwh"],
            "V_N_fair": (k1["use_kwh"] - k0b["use_kwh"]) / k0b["source_kwh"],
            "K1d": {},
            "K2b": {},
            "K3": {},
            "P1": {},
            "P2": {},
        }
        for s in sizes:
            key = f"{s:g}"
            row["K1d"][key] = key_figures(run_k1d(c, p, s, cfg), c, p, cfg)
            if s > 0.0:
                row["K2b"][key] = key_figures(run_k2(c, p, s, cfg, rule="k2b"), c, p, cfg)
        seconds["rules"] += time.perf_counter() - s0
        for s in sizes:
            key = f"{s:g}"
            s0 = time.perf_counter()
            row["K3"][key] = key_figures(run_k3(c, p, s, cfg), c, p, cfg)
            seconds["K3"] += time.perf_counter() - s0
            if s <= 0.0:
                continue
            s0 = time.perf_counter()
            r1 = rolling_plan(cfg, cuts, on, TrueForecast(on), p, s, HORIZON_H, REPLAN_EVERY_H)
            row["P1"][key] = plan_figures(r1, p, cfg)
            seconds["P1"] += time.perf_counter() - s0
            if planned is not None:
                s0 = time.perf_counter()
                fc = PlanForecast(on, planned)
                r2 = rolling_plan(cfg, cuts, on, fc, p, s, HORIZON_H, REPLAN_EVERY_H)
                row["P2"][key] = plan_figures(r2, p, cfg)
                seconds["P2"] += time.perf_counter() - s0
        out["powers"][f"{p:g}"] = row
    out["key"] = f"draw:{level}:{a:g}:{mode}:{draw}:{power:g}"
    out["variant"] = level
    out["fingerprint"] = cfg.fingerprint
    out["seconds_by_part"] = seconds
    out["seconds"] = time.perf_counter() - t0
    return out


def job_replan_check(config_path: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = level_config(config_path, "water_200")
    on = plant_schedules(cfg, 0.57, "independent", 0)
    cuts = cut_data(cfg)
    out: dict[str, Any] = {"key": "check:replan", "cell": "water_200|0.57|independent|0|800|24"}
    for every in (1, REPLAN_EVERY_H):
        r = rolling_plan(cfg, cuts, on, TrueForecast(on), 800.0, 24.0, HORIZON_H, every)
        out[f"every_{every}h"] = plan_figures(r, 800.0, cfg)
    a, b = out["every_1h"]["use_kwh"], out[f"every_{REPLAN_EVERY_H}h"]["use_kwh"]
    out["relative_difference_use"] = (a - b) / a
    out["seconds"] = time.perf_counter() - t0
    return out


def _jobs(config_path: str, sizes: tuple[float, ...], draws: int) -> list[tuple[str, Any, tuple]]:
    jobs: list[tuple[str, Any, tuple]] = []
    for level in LEVELS:
        cfg = level_config(config_path, level)
        for a in cfg.campaigns.availability_levels:
            cells = (
                [("independent", 0)]
                if a >= 1.0
                else [(m, d) for m in cfg.campaigns.modes for d in range(draws)]
            )
            for mode, d in cells:
                for p in cfg.source.power_kw_levels:
                    key = f"draw:{level}:{a:g}:{mode}:{d}:{p:g}"
                    jobs.append((key, job_draw, (config_path, level, a, mode, d, p, sizes)))
    jobs.append(("check:replan", job_replan_check, (config_path,)))
    return jobs


def _merge_powers(results: list[dict[str, Any]]) -> dict[str, Any]:
    """One entry per draw (key without the power) with the rows of every power."""
    merged: dict[str, Any] = {}
    drop = ("powers", "key", "seconds", "seconds_by_part")
    for r in results:
        key = r["key"].rsplit(":", 1)[0]
        if key not in merged:
            merged[key] = {k: v for k, v in r.items() if k not in drop}
            merged[key].update({"key": key, "powers": {}, "seconds": 0.0})
        merged[key]["powers"].update(r["powers"])
        merged[key]["seconds"] += r["seconds"]
    return merged


def _estimate(results: list[dict[str, Any]], main_cfg: Any, args: Any) -> dict[str, Any]:
    """Extrapolation of the probe for the full run and the two cuts of the budget rule.

    Per power the slowest probe job stands for every job of that power (a = 1 jobs have no P2 and
    are counted like a < 1: conservative); the planning time scales with the number of storage
    sizes > 0; 10 % are added for imbalance and 200 s for the replan check.
    """
    slowest: dict[str, float] = {}
    for r in results:
        p = r["key"].rsplit(":", 1)[1]
        slowest[p] = max(slowest.get(p, 0.0), r["seconds"])
    sizes = main_cfg.storage.hours_levels
    no2 = tuple(s for s in sizes if s != 2.0)
    estimates: dict[str, Any] = {}
    for label, cut_sizes, cut_draws in (
        ("full", sizes, main_cfg.campaigns.draws),
        ("without_storage_2", no2, main_cfg.campaigns.draws),
        ("without_storage_2_draws_3", no2, 3),
    ):
        keys = [k for k, _, _ in _jobs(args.config, cut_sizes, cut_draws) if k.startswith("draw:")]
        factor = (len(cut_sizes) - 1) / (len(sizes) - 1)
        cpu = sum(slowest[k.rsplit(":", 1)[1]] * factor for k in keys) + 200.0
        wall_h = 1.1 * cpu / args.workers / 3600.0
        end = datetime.now() + timedelta(hours=wall_h)
        estimates[label] = {
            "jobs": len(keys),
            "cpu_hours": cpu / 3600.0,
            "wall_hours": wall_h,
            "expected_end": end.isoformat(timespec="minutes"),
            "within_30h": wall_h <= 30.0,
            "before_deadline": None
            if args.deadline is None
            else end <= datetime.fromisoformat(args.deadline),
        }
    return {
        "note": TIMING_NOTE.format(workers=args.workers),
        "probe_jobs": [r["key"] for r in results],
        "seconds_by_job": {r["key"]: r["seconds"] for r in results},
        "seconds_by_part": {r["key"]: r["seconds_by_part"] for r in results},
        "slowest_by_power": slowest,
        "workers": args.workers,
        "deadline": args.deadline,
        "estimates": estimates,
        "budget_hours": 30.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-dir", default=str(RUNS_DIR))
    parser.add_argument(
        "--probe", action="store_true", help="time a few jobs (budget), no artifact"
    )
    parser.add_argument("--drop-storage-2", action="store_true", help="budget rule, recorded")
    parser.add_argument("--draws", type=int, default=None, help="budget rule (3), recorded")
    parser.add_argument(
        "--deadline", default=None, help="probe: end of the free machine time, YYYY-MM-DDTHH:MM"
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")
    main_cfg = level_config(args.config, "water_200")
    sizes = tuple(
        s for s in main_cfg.storage.hours_levels if not (args.drop_storage_2 and s == 2.0)
    )
    draws = main_cfg.campaigns.draws if args.draws is None else args.draws
    out_dir = Path(args.out_dir).resolve()
    if args.probe and out_dir == RUNS_DIR.resolve():
        raise SystemExit("probe runs must not write into artifacts/runs (use --out-dir)")
    suffix = "_probe" if args.probe else ""
    output = out_dir / f"{args.name}{suffix}.json.gz"
    cache_path = out_dir / f"_{args.name}{suffix}_cache.jsonl"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if out_dir == RUNS_DIR.resolve():
        require_clean_tree([output, cache_path])
    else:
        print(f"run into {out_dir}: not an artifact")
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance("not used", ignore_paths=[output, cache_path])
    configs = {lv: level_config(args.config, lv) for lv in LEVELS}
    header = {
        "commit": git("rev-parse", "HEAD"),
        "fingerprints": {lv: c.fingerprint for lv, c in configs.items()},
        "storage_hours": list(sizes),
        "draws": draws,
    }
    cache = Cache(cache_path, header, args.resume)
    jobs = _jobs(args.config, sizes, draws)
    if args.probe:
        # every power of draw 0 (independent) per availability level < 1 of the 200 kW level
        jobs = [j for j in jobs if j[0].startswith("draw:water_200:") and ":independent:0:" in j[0]]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        _run_jobs(pool, jobs, cache)
    results = [cache.done[k] for k, _, _ in jobs]
    elapsed = time.time() - started
    if args.probe:
        estimate = _estimate(results, main_cfg, args)
        (out_dir / f"{args.name}_probe.json").write_text(json.dumps(estimate, indent=1))
        print(json.dumps(estimate, indent=1))
        return 0
    payload: dict[str, Any] = {
        "provenance": prov,
        "study": "campaign network follow-up: better simple rule or rolling planning (docs/NN_campaign_network_2.md)",
        "basis": "Befund 118, docs/35_campaign_network.md, artifact campaign_network_v2 (same draws)",
        "config": main_cfg.raw,
        "variant_configs": {lv: c.raw for lv, c in configs.items()},
        "fingerprints": header["fingerprints"],
        "planning": {"horizon_h": HORIZON_H, "replan_every_h": REPLAN_EVERY_H},
        "budget": {
            "storage_hours": list(sizes),
            "draws": draws,
            "rule": "above 30 h wall-clock: drop storage 2 h first, then draws to 3",
        },
        "timing_note": TIMING_NOTE.format(workers=args.workers),
        "workers": args.workers,
        "a1_note": "a = 1: one schedule (all on), computed once; P2 = P1 there",
        "draws": _merge_powers([r for r in results if r["key"].startswith("draw:")]),
        "checks": {r["key"]: r for r in results if r["key"].startswith("check:")},
        "runtime_s": elapsed,
    }
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
