"""Campaign network, post-hoc extension v2: process water as a factor and the storage rule K2b.

DECLARED POST HOC (after the run v1 and its reading, on request of the project lead,
2026-11-03). The run v1 (``campaign_network_v1``) and its pre-registered reading stay unchanged and
are the main statement; this run extends them. Reading of v2 (committed before this run): the
docstring of ``scripts/analyze_campaign_network_v2.py``.

Factor process water (every other factor and setting as the main variant of v1: source powers,
availabilities, modes, the same 5 schedule draws per cell (paired), storage 0/2/8/24 h, approach
10 K, supply 80 degC):

* ``water_0``        -- no process water;
* ``water_100``      -- 100 kW constant;
* ``water_200``      -- 200 kW constant: the main configuration of v1 unchanged (same fingerprint;
  the control that v2 reproduces v1);
* ``water_seasonal`` -- 200 kW from October to April, 0 kW from May to September.

Configurations: K0, K1, K2 (old rule, unchanged), K2b (new rule, docstring of
``twinlink_ml.campaign_network.cascade``), K3; plus the descriptive sweep of v1 (K1 and K3 0 h at
a = 1, 100-3000 kW) per level. The sensitivity variants approach_5 / supply_70 of v1 are not
repeated.

Output ``artifacts/runs/<name>.json.gz`` (never overwritten), cache
``artifacts/runs/_<name>_cache.jsonl``; run from the worktree:
``tools/run_here.sh scripts/run_campaign_network_v2.py --name campaign_network_v2``.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))
sys.path.insert(0, str(SCRIPTS))

from _common import RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402
from run_campaign_network import (  # noqa: E402
    MAX_WORKERS,
    SWEEP_POWERS_KW,
    Cache,
    _run_jobs,
    _worker_init,
)

from twinlink_ml.campaign_network import (  # noqa: E402
    build_consumers,
    evaluate_draw,
    key_figures,
    load_campaign_network_config,
    plant_schedules,
    run_k1,
    run_k3,
)
from twinlink_ml.campaign_network.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    CampaignNetworkConfig,
    from_raw,
)

WATER_LEVELS: dict[str, dict[str, Any] | None] = {
    "water_0": {"power_kw": 0.0},
    "water_100": {"power_kw": 100.0},
    "water_200": None,
    "water_seasonal": {"power_kw": 200.0, "months_off": [5, 6, 7, 8, 9]},
}
"""Overrides of ``low_temperature.water`` per level (``None`` = main configuration of v1)."""


def level_config(config_path: str, level: str) -> CampaignNetworkConfig:
    """Main configuration of v1 with the process-water override of ``level``."""
    main = load_campaign_network_config(config_path, "main")
    override = WATER_LEVELS[level]
    if override is None:
        return main
    raw = copy.deepcopy(main.raw)
    raw["low_temperature"]["water"].update(override)
    raw["variant"] = f"posthoc_{level}"
    return from_raw(raw, raw["variant"])


def job_draw(config_path: str, level: str, a: float, mode: str, draw: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = level_config(config_path, level)
    res = evaluate_draw(cfg, a, mode, draw, with_k2b=True)
    res["key"] = f"draw:{level}:{a:g}:{mode}:{draw}"
    res["variant"] = level
    res["fingerprint"] = cfg.fingerprint
    res["seconds"] = time.perf_counter() - t0
    return res


def job_sweep(config_path: str, level: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = level_config(config_path, level)
    c = build_consumers(cfg, plant_schedules(cfg, 1.0, "independent", 0))
    rows = []
    for p in SWEEP_POWERS_KW:
        k1 = key_figures(run_k1(c, p, cfg), c, p, cfg)
        k3 = key_figures(run_k3(c, p, 0.0, cfg), c, p, cfg)
        rows.append({"power_kw": p, "K1": k1, "K3": k3})
    return {
        "key": f"sweep:{level}",
        "variant": level,
        "fingerprint": cfg.fingerprint,
        "rows": rows,
        "demand_kwh": {s.name: float(s.demand_kw.sum()) for s in c.streams},
        "seconds": time.perf_counter() - t0,
    }


def _jobs(config_path: str) -> list[tuple[str, Any, tuple[Any, ...]]]:
    jobs: list[tuple[str, Any, tuple[Any, ...]]] = []
    for level in WATER_LEVELS:
        cfg = level_config(config_path, level)
        for a in cfg.campaigns.availability_levels:
            if a >= 1.0:
                key = f"draw:{level}:{a:g}:independent:0"
                jobs.append((key, job_draw, (config_path, level, a, "independent", 0)))
                continue
            for mode in cfg.campaigns.modes:
                for d in range(cfg.campaigns.draws):
                    key = f"draw:{level}:{a:g}:{mode}:{d}"
                    jobs.append((key, job_draw, (config_path, level, a, mode, d)))
        jobs.append((f"sweep:{level}", job_sweep, (config_path, level)))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-dir", default=str(RUNS_DIR))
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")
    main_cfg = load_campaign_network_config(args.config, "main")
    out_dir = Path(args.out_dir).resolve()
    output = out_dir / f"{args.name}.json.gz"
    cache_path = out_dir / f"_{args.name}_cache.jsonl"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if out_dir == RUNS_DIR.resolve():
        require_clean_tree([output, cache_path])
    else:
        print(f"run into {out_dir}: not an artifact")
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance("not used", ignore_paths=[output, cache_path])
    configs = {lv: level_config(args.config, lv) for lv in WATER_LEVELS}
    header = {
        "commit": git("rev-parse", "HEAD"),
        "fingerprints": {lv: c.fingerprint for lv, c in configs.items()},
        "storage_hours": list(main_cfg.storage.hours_levels),
        "draws": main_cfg.campaigns.draws,
    }
    cache = Cache(cache_path, header, args.resume)
    jobs = _jobs(args.config)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        _run_jobs(pool, jobs, cache)
    results = [cache.done[k] for k, _, _ in jobs]
    payload: dict[str, Any] = {
        "provenance": prov,
        "study": "campaign network, post-hoc extension v2 (declared post hoc; v1 is the main statement)",
        "declared_post_hoc": True,
        "config": main_cfg.raw,
        "water_levels": WATER_LEVELS,
        "variant_configs": {lv: c.raw for lv, c in configs.items()},
        "fingerprints": header["fingerprints"],
        "fingerprint_v1_main": main_cfg.fingerprint,
        "budget": {
            "storage_hours": list(main_cfg.storage.hours_levels),
            "draws": main_cfg.campaigns.draws,
            "rule": "above 30 h wall-clock: drop storage 24 h first, then draws to 3",
        },
        "a1_note": "a = 1: one schedule (all on), computed once, valid for every draw and mode",
        "workers": args.workers,
        "draws": {r["key"]: r for r in results if r["key"].startswith("draw:")},
        "sweeps": {r["variant"]: r for r in results if r["key"].startswith("sweep:")},
        "sweep_powers_kw": list(SWEEP_POWERS_KW),
        "runtime_s": time.time() - started,
    }
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
