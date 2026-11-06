"""POST-HOC sensitivity v2 of the static transfer study: L3 with a stronger fine-tuning.

Declared post hoc (decision of the project lead, 2026-11-06); v1 and its main reading R3 stay
unchanged. Settings: ``conf/static_transfer/static_transfer_v2.yaml``; reading (pre-registered
before the pilot and the rerun): docstring of ``scripts/analyze_static_transfer_v2.py``;
docs/NN_static_transfer.md section 15.

Phases
------
``--phase pilot --name static_transfer_pilot_v2``
    Pretrains one MLP per pilot seed (21-23), then evaluates every L3 setting of the v2 grid on the
    full-strength pilot plant PP-full (N = 4 and 16, its own validation set; no test set is used).
    Selection: lowest mean criterion C (as v1); a selected value at the edge of its dimension extends
    it by one geometric step (at most twice per dimension, also beyond the named grid limits; an
    edge at 0 is a natural bound). Output ``artifacts/runs/<name>.json``; the selection is then copied
    into the ``selected`` block of the v2 YAML (own commit).
``--phase rerun --name static_transfer_v2``
    Checks the v2 selection against the pilot-2 artifact, loads the v1 artifact (test sets and
    designs as stored there) and the v1 pretrained nets, and reruns every unit of the six plants x
    5 seeds under ``clean_low`` with the v1 unit code and ONLY the L3 setting replaced. Every other
    method is recomputed as a check: its errors must equal v1 (``check_against_v1``). Output
    ``artifacts/runs/<name>.json.gz``.

Same operation as v1: per-task cache ``artifacts/runs/_<name>_cache.jsonl``, ``--resume``, at most
8 workers, clean tree, provenance. Start with ``tools/run_here.sh scripts/run_static_transfer_v2.py``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))
sys.path.insert(0, str(SCRIPTS))

import run_static_transfer as v1driver  # noqa: E402
from _common import RUNS_DIR, _json_default, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.static_transfer import study  # noqa: E402
from twinlink_ml.static_transfer.balance import plant_model, source_model  # noqa: E402
from twinlink_ml.static_transfer.config import TARGET_NAMES, load_config  # noqa: E402
from twinlink_ml.static_transfer.posthoc import (  # noqa: E402
    DEFAULT_V2_PATH,
    PosthocConfig,
    check_against_v1,
    load_posthoc,
    pilot_context,
    rerun_config,
    rerun_context,
)

REPO_ROOT = SCRIPTS.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pilot_deviation(ctx: study.StudyContext) -> dict[str, Any]:
    """L0 against PP-full on the pilot validation set (RMSE and RMSE / sigma per target)."""
    import numpy as np

    config = ctx.config
    x = ctx.pilot_validation.x
    truth = plant_model(config.source, config.pilot_plant).predict(x)
    rmse = np.sqrt(np.mean((source_model(config.source).predict(x) - truth) ** 2, axis=0))
    sigma = study.criterion_scale(config, truth)
    return {
        "rmse": rmse.tolist(),
        "rmse_over_sigma": (rmse / sigma).tolist(),
        "targets": list(TARGET_NAMES),
    }


def run_pilot(args: argparse.Namespace, spec: PosthocConfig) -> None:
    out_path = RUNS_DIR / f"{args.name}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
    require_clean_tree()
    prov = provenance()
    base = load_config()
    header = {"commit": git("rev-parse", "HEAD"), "config": spec.fingerprint, "phase": "pilot_v2"}
    cache = v1driver.Cache(RUNS_DIR / f"_{args.name}_cache.jsonl", header, args.resume)
    t0 = time.perf_counter()
    ctx = pilot_context(base, spec)
    config = ctx.config
    seeds = list(spec.pilot_seeds)
    checkpoints, pre_records = v1driver._pretrain(
        config, seeds, args.name, False, cache, args.workers
    )
    grid = {k: list(v) for k, v in spec.grid.items()}
    extensions: dict[str, int] = {}
    rounds: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def key(s: dict[str, float]) -> str:
        return json.dumps(["pilot", "L3", json.dumps(s, sort_keys=True), None])

    while True:
        settings = study.grid_settings(grid)
        tasks = [("pilot", "L3", json.dumps(s, sort_keys=True), None) for s in settings]
        failures += v1driver._run_pool(tasks, cache, args.workers, ctx, checkpoints)
        results = [cache.done[key(s)]["result"] for s in settings if key(s) in cache.done]
        best = study.select_setting(results)
        grid_new, extensions, extended = study.extend_grid(
            grid, best["setting"], extensions, spec.max_extensions
        )
        rounds.append(
            {"best": best["setting"], "criterion": best["criterion"], "extended": extended}
        )
        print(f"pilot-2 round {len(rounds)}: {rounds[-1]}", flush=True)
        if not extended:
            break
        grid = grid_new
    results = sorted(
        (cache.done[key(s)]["result"] for s in study.grid_settings(grid) if key(s) in cache.done),
        key=lambda r: r["criterion"],
    )
    best = results[0]
    at_edge = {
        k: float(best["setting"][k]) in (min(v), max(v))
        and len(v) > 1
        and float(best["setting"][k]) != 0.0
        for k, v in grid.items()
    }
    payload = {
        "provenance": prov,
        "study": "static_transfer",
        "phase": "pilot_v2 (post hoc)",
        "config_path": str(DEFAULT_V2_PATH.relative_to(REPO_ROOT)),
        "config_fingerprint": spec.fingerprint,
        "config_fingerprint_without_selection": spec.fingerprint_without_selection,
        "config_v2": spec.raw,
        "config_v1_fingerprint": base.fingerprint,
        "pilot_plant": spec.pilot_plant.name,
        "pilot_deviation_on_validation_set": pilot_deviation(ctx),
        "validation_admissibility": {
            "candidates": ctx.pilot_validation.candidates,
            "admissible": ctx.pilot_validation.admissible,
        },
        "pretraining": pre_records,
        "final_grid": grid,
        "extensions": extensions,
        "rounds": rounds,
        "selection": {
            "L3": {
                "setting": best["setting"],
                "criterion": best["criterion"],
                "at_edge_after_extensions": at_edge,
            }
        },
        "results": results,
        "failures": failures,
        "wall_seconds": time.perf_counter() - t0,
    }
    out_path.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    print(f"wrote {out_path}")


def run_rerun(args: argparse.Namespace, spec: PosthocConfig) -> None:
    out_path = RUNS_DIR / f"{args.name}.json.gz"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
    require_clean_tree()
    prov = provenance()
    pilot_name = spec.raw["selected"].get("pilot_artifact")
    setting = spec.selected_l3()
    if not pilot_name or setting is None:
        raise SystemExit("selected block of the v2 YAML is empty -- run pilot 2 and fill it first")
    pilot = json.loads((RUNS_DIR / f"{pilot_name}.json").read_text(encoding="utf-8"))
    chosen = {k: float(v) for k, v in pilot["selection"]["L3"]["setting"].items()}
    if chosen != setting:
        raise SystemExit("selected.L3 differs from the pilot-2 artifact")
    if pilot["config_fingerprint_without_selection"] != spec.fingerprint_without_selection:
        raise SystemExit("the v2 YAML changed outside the selected block since pilot 2")
    base = load_config()
    with gzip.open(RUNS_DIR / f"{spec.v1_run}.json.gz", "rt", encoding="utf-8") as fh:
        v1 = json.load(fh)
    if v1["config_fingerprint"] != base.fingerprint:
        raise SystemExit("the v1 YAML changed since the v1 run")
    config = rerun_config(base, setting)
    ctx = rerun_context(config, v1)
    checkpoints = {s: str(spec.v1_checkpoints / f"pretrain_seed{s}.pt") for s in config.seeds}
    for path in checkpoints.values():
        if not Path(path).exists():
            raise SystemExit(f"v1 checkpoint missing: {path}")
    header = {
        "commit": git("rev-parse", "HEAD"),
        "config": spec.fingerprint,
        "phase": "rerun_v2",
        "l3": setting,
    }
    cache = v1driver.Cache(RUNS_DIR / f"_{args.name}_cache.jsonl", header, args.resume)
    t0 = time.perf_counter()
    plants = [p.name for p in config.plants]
    tasks = [("unit", p, spec.condition, s, None, None) for p in plants for s in config.seeds]
    failures = v1driver._run_pool(tasks, cache, args.workers, ctx, checkpoints)
    v1_units = {(u["plant"], u["condition"], int(u["seed"])): u for u in v1["units"]}
    units = []
    checks = []
    for t in tasks:
        k = json.dumps(list(t))
        if k not in cache.done:
            continue
        unit = cache.done[k]["result"]
        check = check_against_v1(
            unit, v1_units[(unit["plant"], unit["condition"], int(unit["seed"]))]
        )
        checks.append({"plant": unit["plant"], "seed": unit["seed"], **check})
        units.append(unit)
    payload = {
        "provenance": prov,
        "study": "static_transfer",
        "phase": "rerun_v2 (post hoc)",
        "config_path": str(DEFAULT_V2_PATH.relative_to(REPO_ROOT)),
        "config_fingerprint": spec.fingerprint,
        "config_v2": spec.raw,
        "config_v1_fingerprint": base.fingerprint,
        "config": config.raw,
        "l3_setting": setting,
        "pilot_artifact": pilot_name,
        "v1_run": spec.v1_run,
        "v1_run_commit": v1["provenance"]["git/commit"],
        "checkpoints": {
            str(s): {"path": str(Path(p).relative_to(REPO_ROOT)), "sha256": _sha256(Path(p))}
            for s, p in checkpoints.items()
        },
        "condition": spec.condition,
        "reported_test_sets": list(spec.reported_test_sets),
        "units": units,
        "check_against_v1": checks,
        "check_max_abs_difference": max(
            (c["max_abs_difference"] for c in checks), default=float("nan")
        ),
        "failures": failures,
        "wall_seconds": time.perf_counter() - t0,
        "timing_context": f"workers = {args.workers} on this machine; not a result",
    }
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, default=_json_default)
    print(
        f"wrote {out_path}; {len(failures)} failed units; largest difference of the other "
        f"methods against v1: {payload['check_max_abs_difference']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=["pilot", "rerun"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.workers > 8:
        raise SystemExit("at most 8 workers (physical cores of this machine)")
    spec = load_posthoc()
    if args.phase == "pilot":
        run_pilot(args, spec)
    else:
        run_rerun(args, spec)


if __name__ == "__main__":
    main()
