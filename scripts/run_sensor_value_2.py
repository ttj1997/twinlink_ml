"""Follow-up sensor-value study: margins and energies of the bias-corrected R1 with F4 and F5.

Study document: ``docs/29_sensor_value_2.md``; settings and placeholders:
``conf/sensor_value/sensor_value_2.yaml`` (reads ``conf/sensor_value/sensor_value.yaml`` of the
predecessor for everything shared); reading (pre-registered): docstring of
``scripts/analyze_sensor_value_2.py``.

Identical to the predecessor (``scripts/run_sensor_value.py``, Befund 111) and only read: the 20
parameter draws of the truth, the 20 scenarios, R1 with the bias correction, warm-up and pre-roll of
6 h, the path condition, the per-pair bisection and the margins of criteria (i) and (ii) with their
verification on all pairs (``stage_search``/``stage_verify`` of that driver, imported). New feedback:

* ``F4:<h>h`` -- periodic laboratory sample every h hours (``LabSample``), h in the config;
* ``F5:<variant>:<seed>`` -- GRU soft sensor with measured process responses (``SoftSensorFeedback``
  with a ``ResponseSensor``), checkpoints of ``scripts/train_sensor_value_2_soft_sensor.py``.

The reference variants F0, F1 and F2 are NOT recomputed: the analysis takes them from the
predecessor's artifact. Stage ``control`` recomputes 10 pairs of F0, F1 and F2 at the predecessor's
criterion-(ii) margins and compares every field of the record with the artifact (per-interval arrays
after the artifact's rounding).

Output: ``artifacts/runs/<name>.json.gz`` (once at the end, never overwritten), without per-interval
arrays (size); every evaluation is appended to ``artifacts/runs/_<name>_cache.jsonl`` (gitignored),
``--resume`` reuses it. ``--probe`` times one full-length job per variant and writes nothing into
``artifacts/runs``. Run from the worktree with ``tools/run_here.sh scripts/run_sensor_value_2.py
--name sensor_value_2_v1 --soft-sensor-record sensor_value_2_soft_sensor_v1``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_sensor_value as base  # noqa: E402
from _common import REPO_ROOT, RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.headroom.scenarios import Scenario  # noqa: E402
from twinlink_ml.sensitivity.config import load_sensitivity_config  # noqa: E402
from twinlink_ml.sensor_value import (  # noqa: E402
    Feedback,
    LabSample,
    SoftSensor,
    SoftSensorFeedback,
    run_pairs,
)
from twinlink_ml.sensor_value.config import (  # noqa: E402
    FOLLOW_UP_CONFIG_PATH,
    FollowUpConfig,
    load_follow_up_config,
)
from twinlink_ml.sensor_value.responses import ResponseSensor  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning  # noqa: E402
from twinlink_ml.utils.seeding import _derive  # noqa: E402

REFERENCE_VARIANTS = base.BASE_VARIANTS  # F0, F1, F2: taken over, only the control pairs rerun
ROUNDED_ARRAYS = ("energy_kwh_per_interval", "moisture_max_per_interval", "bias_per_interval")


def lab_variants(follow_up: FollowUpConfig) -> list[str]:
    """``F4:<h>h`` labels in the order of the config."""
    return [f"F4:{h:g}h" for h in follow_up.lab.sampling_intervals_h]


def soft_variants(follow_up: FollowUpConfig, seeds: list[int] | None = None) -> list[str]:
    """``F5:<variant>:<seed>`` labels, variants in the order of the config."""
    seeds = list(follow_up.soft.training_seeds) if seeds is None else seeds
    return [f"F5:{v.label}:{s}" for v in follow_up.soft.variants for s in seeds]


def control_pairs(n_draws: int, n_scenarios: int, k: int) -> list[tuple[int, int]]:
    """``k`` pairs (draw, scenario index) at ``round(linspace(0, N - 1, k))`` of the draw-major order."""
    flat = np.round(np.linspace(0, n_draws * n_scenarios - 1, k)).astype(int)
    return [(int(p) // n_scenarios, int(p) % n_scenarios) for p in flat]


# ------------------------------------------------------------------------------ worker side

_WORKER: dict[str, Any] = {}


def _init_worker(follow_up_path: str, smoke: dict[str, Any] | None, soft: dict[str, str]) -> None:
    follow_up = load_follow_up_config(follow_up_path)
    base._init_worker(str(follow_up.base_path), smoke, {})
    _WORKER["follow_up"] = follow_up
    _WORKER["soft_paths"] = soft
    _WORKER["soft"] = {}


def _soft_sensor(label: str) -> SoftSensor:
    cache = _WORKER["soft"]
    if label not in cache:
        import torch

        torch.set_num_threads(1)
        cache[label] = SoftSensor.load(REPO_ROOT / _WORKER["soft_paths"][label])
    return cache[label]


def make_feedback(
    setup: base.Setup, follow_up: FollowUpConfig, variant: str, draw: int, scenario: Scenario
) -> Feedback:
    """Feedback object of one pair."""
    if variant in REFERENCE_VARIANTS:
        return base.make_feedback(setup, variant, draw, scenario)
    interval = setup.config.loop.measurement_interval_s
    seed = setup.config.noise_seed
    if variant.startswith("F4:"):
        label = variant.removeprefix("F4:")
        hours = float(label.removesuffix("h"))
        lab = follow_up.lab
        return LabSample(
            sampling_interval_s=hours * 3600.0,
            delay_s=lab.delay_s,
            error_std_kg_per_kg=lab.error_std_kg_per_kg,
            gain=lab.gain,
            interval_s=interval,
            seed=_derive(seed, f"F4/{label}/lab", draw, scenario.name),
            label=label,
        )
    if variant.startswith("F5:"):
        sensor = _soft_sensor(variant)
        responses = ResponseSensor(
            setup.nominal,
            setup.measurement,
            interval,
            sensor.measured_input_names,
            bed_seed=_derive(seed, "F5/bed", draw, scenario.name),
            exhaust_seed=_derive(seed, "F5/exhaust", draw, scenario.name),
        )
        return SoftSensorFeedback(sensor, variant.removeprefix("F5:"), responses)
    raise ValueError(f"unknown variant {variant!r}")


def _task_eval(
    variant: str,
    draw: int,
    ids: list[int],
    margins: list[float],
    states: list[list[float]],
    keep_arrays: bool,
) -> dict[str, Any]:
    setup: base.Setup = base._WORKER["setup"]
    follow_up: FollowUpConfig = _WORKER["follow_up"]
    started = time.perf_counter()
    scenarios = [setup.scenarios[i] for i in ids]
    loop = setup.config.loop
    records = run_pairs(
        setup.study,
        setup.nominal,
        scenarios,
        margins,
        [make_feedback(setup, follow_up, variant, draw, s) for s in scenarios],
        truth=setup.draws[draw].process,
        truth_initial_states=np.array(states)[ids],
        pre_roll_s=loop.pre_roll_s,
        measurement_interval_s=loop.measurement_interval_s,
        bias_time_constant_s=loop.bias_filter_time_constant_s,
        variant=variant,
        draw=draw,
        keep_arrays=keep_arrays,
    )
    out = []
    for r in records:
        d = r.as_dict()
        if keep_arrays:  # control pairs: the rounding of the predecessor's artifact
            for key in ROUNDED_ARRAYS:
                d["extra"][key] = [float(f"{v:.6g}") for v in d["extra"][key]]
            d["extra"]["inlet_command_per_interval"] = [
                float(f"{a:.5g}") for a, _ in d["extra"].pop("commands")
            ]
        out.append(d)
    return {"records": out, "seconds": time.perf_counter() - started, "n_pairs": len(ids)}


# ------------------------------------------------------------------------------ main side


class Runner(base.Runner):
    """The predecessor's runner with this study's feedback; no per-interval arrays (size)."""

    def submit(
        self, stage: str, variant: str, draw: int, ids: list[int], margins: list[float]
    ) -> Future:
        return self.pool.submit(
            _task_eval, variant, draw, ids, margins, self.states[draw], stage == "control"
        )


def _compare(ours: dict[str, Any], theirs: dict[str, Any]) -> dict[str, Any]:
    """Field-by-field comparison of two pair records (numbers: largest absolute difference)."""
    differing: dict[str, float | str] = {}

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differing[f"{path}/{key}"] = "missing"
                else:
                    walk(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differing[path] = "length"
            elif a != b:
                diff = np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))
                differing[path] = float(diff)
        elif a != b:
            try:
                differing[path] = float(abs(float(a) - float(b)))
            except (TypeError, ValueError):
                differing[path] = f"{a!r} != {b!r}"

    walk(ours, theirs, "")
    return {"identical": not differing, "differing": differing}


def stage_control(runner: Runner, predecessor: dict[str, Any], k: int) -> dict[str, Any]:
    """Recompute ``k`` pairs of F0, F1, F2 at the predecessor's (ii) margins and compare."""
    setup = runner.setup
    names = [s.name for s in setup.scenarios]
    pairs = control_pairs(len(setup.draws), len(names), k)
    out: dict[str, Any] = {"pairs": [[d, names[i]] for d, i in pairs], "variants": {}}
    futures: dict[Future, tuple[str, int]] = {}
    margins = {}
    for v in REFERENCE_VARIANTS:
        margins[v] = predecessor["results"][v]["criteria"]["ii"]["margin"]
        by_draw: dict[int, list[int]] = {}
        for d, i in pairs:
            by_draw.setdefault(d, []).append(i)
        for d, ids in by_draw.items():
            futures[runner.submit("control", v, d, ids, [margins[v]] * len(ids))] = (v, d)
    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    for future, (v, _d) in futures.items():
        for r in runner.collect("control", future):
            records[(v, int(r["draw"]), r["scenario"])] = r
    for v in REFERENCE_VARIANTS:
        reference = {
            (int(r["draw"]), r["scenario"]): r
            for r in predecessor["results"][v]["criteria"]["ii"]["records"]
        }
        comparisons = []
        for d, i in pairs:
            ours = records[(v, d, names[i])]
            comparisons.append(
                {"draw": d, "scenario": names[i], **_compare(ours, reference[(d, names[i])])}
            )
        numeric = [x for c in comparisons for x in c["differing"].values() if isinstance(x, float)]
        out["variants"][v] = {
            "margin": margins[v],
            "identical_pairs": sum(c["identical"] for c in comparisons),
            "max_abs_difference": max(numeric) if numeric else 0.0,
            "comparisons": comparisons,
        }
        print(
            f"[control] {v}: {out['variants'][v]['identical_pairs']}/{len(pairs)} identical, "
            f"max |diff| {out['variants'][v]['max_abs_difference']:.3g}",
            flush=True,
        )
    return out


def load_predecessor(follow_up: FollowUpConfig, setup: base.Setup) -> tuple[dict, dict]:
    """The predecessor's artifact, checked against the shared configs."""
    path = RUNS_DIR / f"{follow_up.predecessor_run}.json.gz"
    raw = path.read_bytes()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    checks = {
        "git/dirty": payload["provenance"]["git/dirty"] is False,
        "sensor_value_fingerprint": payload["sensor_value_fingerprint"]
        == follow_up.base.fingerprint,
        "study_fingerprint": payload["study_fingerprint"] == setup.study.fingerprint,
        "process_config_fingerprint": payload["process_config_fingerprint"]
        == setup.nominal.fingerprint,
        "draws": payload["draws"] == [d.as_dict() for d in setup.draws],
        "scenarios": payload["scenarios"]
        == [{"name": s.name, "kind": s.kind, "index": s.index} for s in setup.scenarios],
    }
    if not all(checks.values()):
        raise SystemExit(f"predecessor artifact does not match the shared settings: {checks}")
    summary = {
        "artifact": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "commit": payload["provenance"]["git/commit"],
        "checks": checks,
    }
    return payload, summary


def _soft_paths(
    record_name: str | None, follow_up: FollowUpConfig
) -> tuple[dict[str, str], dict[str, Any] | None]:
    if record_name is None:
        return {}, None
    path = RUNS_DIR / f"{record_name}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["provenance"]["git/dirty"]:
        raise SystemExit(f"{path} was written from a dirty tree")
    if record["follow_up_fingerprint"] != follow_up.fingerprint:
        raise SystemExit("soft-sensor record was trained under another follow-up config")
    paths = {}
    for r in record["records"]:
        checkpoint = REPO_ROOT / r["checkpoint"]
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != r["checkpoint_sha256"]:
            raise SystemExit(f"{checkpoint}: SHA-256 differs from the training record")
        paths[f"F5:{r['variant']}:{r['seed']}"] = r["checkpoint"]
    summary = {
        "record": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoints": dict(paths),
        "seeds": record["seeds"],
    }
    return paths, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=str(FOLLOW_UP_CONFIG_PATH))
    parser.add_argument("--soft-sensor-record", default=None, help="name of the training record")
    parser.add_argument(
        "--drop", default="", help="comma list of variants dropped by the budget rule (recorded)"
    )
    parser.add_argument("--variants", default=None, help="comma list (smoke/probe only)")
    parser.add_argument("--out-dir", default=str(RUNS_DIR))
    parser.add_argument("--smoke", default=None, help="hours,draws,per_type,pre_roll_s (smoke run)")
    parser.add_argument("--probe", type=float, default=None, help="margin of the budget probe")
    parser.add_argument("--skip-control", action="store_true", help="smoke/probe only")
    args = parser.parse_args()

    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    smoke = None
    if args.smoke:
        hours, n_draws, per_type, pre = args.smoke.split(",")
        smoke = {
            "hours": float(hours),
            "draws": int(n_draws),
            "per_type": int(per_type),
            "pre_roll_s": float(pre),
        }
    follow_up = load_follow_up_config(args.config)
    setup = base.build_setup(str(follow_up.base_path), smoke)
    config = setup.config
    soft, soft_summary = _soft_paths(args.soft_sensor_record, follow_up)
    configured = lab_variants(follow_up) + soft_variants(follow_up)
    available = lab_variants(follow_up) + sorted(soft, key=configured.index)
    dropped = [v for v in args.drop.split(",") if v]
    if set(dropped) - set(configured):
        raise SystemExit(f"unknown dropped variants {sorted(set(dropped) - set(configured))}")
    variants = [v for v in available if v not in dropped]
    if args.variants:
        variants = [v for v in args.variants.split(",") if v]
        unknown = [v for v in variants if v not in available + list(REFERENCE_VARIANTS)]
        if unknown:
            raise SystemExit(f"unknown or unavailable variants {unknown}")
    out_dir = Path(args.out_dir).resolve()
    artifact_run = out_dir == RUNS_DIR.resolve() and smoke is None and args.probe is None
    if out_dir == RUNS_DIR.resolve() and not artifact_run:
        raise SystemExit("smoke and probe runs must not write into artifacts/runs")
    if artifact_run:
        if args.variants or args.skip_control:
            raise SystemExit("the artifact run takes every configured variant and the control")
        missing = [v for v in configured if v not in variants and v not in dropped]
        if missing:
            raise SystemExit(f"the artifact run lacks {missing} (not dropped by the budget rule)")
    output = out_dir / f"{args.name}.json.gz"
    cache_path = out_dir / f"_{args.name}_cache.jsonl"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if cache_path.exists() and not args.resume:
        raise SystemExit(f"{cache_path} exists -- pass --resume to continue it, or use a new name")
    if artifact_run:
        require_clean_tree([output, cache_path])
    else:
        print(
            f"{'probe' if args.probe is not None else 'smoke'} run into {out_dir}: not an artifact"
        )
    predecessor, predecessor_summary = (None, None) if smoke else load_predecessor(follow_up, setup)
    prov = provenance("not used", ignore_paths=[output, cache_path])
    header = {
        "commit": git("rev-parse", "HEAD"),
        "follow_up_fingerprint": follow_up.fingerprint,
        "sensor_value_fingerprint": config.fingerprint,
        "study_fingerprint": setup.study.fingerprint,
        "process_fingerprint": setup.nominal.fingerprint,
        "soft_sensor": soft_summary,
        "smoke": smoke,
        "variants": variants,
    }
    started = time.time()
    payload: dict[str, Any] = {
        "provenance": prov,
        "study": "sensor value, follow-up (docs/29_sensor_value_2.md)",
        "config": follow_up.raw,
        "base_config": config.raw,
        "headroom_config": setup.study.raw,
        "follow_up_fingerprint": follow_up.fingerprint,
        "sensor_value_fingerprint": config.fingerprint,
        "study_fingerprint": setup.study.fingerprint,
        "process_config_fingerprint": setup.nominal.fingerprint,
        "sensitivity_fingerprint": load_sensitivity_config(config.sensitivity_config).fingerprint,
        "outlier_config_digest": setup.outliers.digest,
        "predecessor": predecessor_summary,
        "soft_sensor": soft_summary,
        "smoke": smoke,
        "workers": args.workers,
        "variants": variants,
        "budget_dropped": dropped,
        "draws": [d.as_dict() for d in setup.draws],
        "rejected_sequence_positions": setup.rejected,
        "scenarios": [{"name": s.name, "kind": s.kind, "index": s.index} for s in setup.scenarios],
        "violation_tolerance_kg_per_kg": setup.study.control.violation_tolerance_kg_per_kg,
    }
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.config, smoke, soft)
    ) as pool:
        runner = Runner(pool, base.Cache(cache_path, header), setup)
        t0 = time.time()
        runner.warm_up()
        payload.setdefault("stage_runtime_s", {})["warm_up"] = time.time() - t0
        print(f"warm-up done in {(time.time() - t0) / 60:.1f} min", flush=True)
        if args.probe is not None:
            payload["probe"] = base.probe(runner, variants, args.probe)
            payload["runtime_s"] = time.time() - started
            (out_dir / f"{args.name}_probe.json").write_text(
                json.dumps(payload, indent=1, default=str), encoding="utf-8"
            )
            return 0
        if not args.skip_control:
            assert predecessor is not None
            t0 = time.time()
            payload["control"] = stage_control(runner, predecessor, follow_up.control_pairs)
            payload["stage_runtime_s"]["control"] = time.time() - t0
        t0 = time.time()
        searches = base.stage_search(runner, variants)
        payload["stage_runtime_s"]["search"] = time.time() - t0
        t0 = time.time()
        payload["results"] = base.stage_verify(runner, variants, searches)
        payload["stage_runtime_s"]["verify"] = time.time() - t0
        payload["seconds_per_pair_and_job"] = {
            v: {"median": float(np.median(s)), "n_jobs": len(s)}
            for v, s in runner.job_seconds.items()
        }
    payload["runtime_s"] = time.time() - started
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
