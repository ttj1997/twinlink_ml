"""Sensor-value study: required margins and energies of R1 with bias correction from F0-F3.

Study document: ``docs/28_sensor_value.md``; settings and placeholders:
``conf/sensor_value/sensor_value.yaml`` (reads the headroom, Sobol and outlier configs); reading
(pre-registered): docstring of ``scripts/analyze_sensor_value.py``.

Question: how much energy does it cost not to measure the product moisture, or to measure it
badly, when the controller's model deviates from the process? Truth = the grey-box model of
Befund 105 with 20 drawn parameter sets (critical moisture, nu_max, GAB); controller = R1 on the
NOMINAL model with the bias correction ``X_model + b <= 0.15 - margin``
(``twinlink_ml.sensor_value``); feedback F0 none, F1 ideal, F2 inline sensor with outliers, F3 GRU
soft sensor (three training seeds, checkpoints of ``scripts/train_sensor_value_soft_sensor.py``).

Stages
------
``warm_up``
    truth states after the warm-up at the nominal initial command, per (draw, scenario).
``search``
    per pair (draw, scenario) and variant the smallest margin with which the path condition holds
    (``PairBisection``, the step form of Befund 105's ``bisect_margin``: bracket [0, 0.05] doubled
    up to 0.12, tolerance 2.5e-4 kg/kg), all pairs of one draw advancing together in one job;
    jobs of different (variant, draw) chains run concurrently.
``verify``
    per variant the margins of criterion (i) all pairs and (ii) at least 95 % of the pairs
    (order statistics of the per-pair margins), each evaluated on ALL pairs; a candidate that
    fails its criterion is raised to the next per-pair margin (then by the tolerance, at most
    four times) and evaluated again. These runs give the energies of the artifact.

Output: ``artifacts/runs/<name>.json.gz`` (written once at the end, never overwritten). Every
single evaluation is appended to ``artifacts/runs/_<name>_cache.jsonl`` (gitignored); ``--resume``
reuses it (same commit and configs required). Run from the worktree with
``tools/run_here.sh scripts/run_sensor_value.py --name sensor_value_v1 --soft-sensor-record
sensor_value_soft_sensor_v1``. ``--probe`` times one job per variant at full length (budget rule)
and writes nothing into ``artifacts/runs``.
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
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import REPO_ROOT, RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.headroom.config import load_headroom_config, study_process_config  # noqa: E402
from twinlink_ml.headroom.scenarios import Scenario, build_scenarios  # noqa: E402
from twinlink_ml.sensitivity.config import load_sensitivity_config  # noqa: E402
from twinlink_ml.sensor_value import (  # noqa: E402
    Feedback,
    IdealMeasurement,
    InlineSensor,
    NoFeedback,
    PairBisection,
    SoftSensor,
    SoftSensorFeedback,
    criterion_margin,
    load_sensor_value_config,
    next_candidate,
    parameter_draws,
    required_count,
    run_pairs,
    scenario_subset,
    warmed_up_states,
)
from twinlink_ml.sensor_value.config import DEFAULT_CONFIG_PATH  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402
from twinlink_ml.utils.seeding import _derive  # noqa: E402

BASE_VARIANTS = ("F0", "F1", "F2")
MAX_EXTRA_STEPS = 4

# ------------------------------------------------------------------------------ setup


@dataclasses.dataclass
class Setup:
    """Everything a worker or the main process needs, rebuilt from the configs."""

    config: Any
    study: Any
    nominal: Any
    draws: list[Any]
    rejected: list[int]
    scenarios: list[Scenario]
    measurement: Any
    outliers: Any


def build_setup(config_path: str, smoke: dict[str, Any] | None) -> Setup:
    """Configs, draws and scenarios; ``smoke`` shortens everything for a smoke run."""
    from twinlink_ml.outliers import load_outlier_config

    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    config = load_sensor_value_config(config_path)
    if smoke:
        config = dataclasses.replace(
            config,
            draws=dataclasses.replace(config.draws, n=smoke["draws"]),
            scenarios_per_type=smoke["per_type"],
            loop=dataclasses.replace(
                config.loop, warm_up_s=smoke["pre_roll_s"], pre_roll_s=smoke["pre_roll_s"]
            ),
        )
    study = load_headroom_config(config.headroom_config)
    nominal = study_process_config(study)
    draws, rejected = parameter_draws(
        config.draws, load_sensitivity_config(config.sensitivity_config), nominal
    )
    scenarios = scenario_subset(
        build_scenarios(study, nominal, config.scenario_family), config.scenarios_per_type
    )
    if smoke:
        n = round(smoke["hours"] * 3600.0 / study.scenarios.sample_time_s)
        scenarios = [
            dataclasses.replace(
                s, disturbances=s.disturbances[:, :n], moisture_limit=s.moisture_limit[:n]
            )
            for s in scenarios
        ]
    measurement = load_process_config(study.process_config).measurement
    return Setup(
        config,
        study,
        nominal,
        draws,
        rejected,
        scenarios,
        measurement,
        load_outlier_config(config.outlier_config),
    )


# ------------------------------------------------------------------------------ worker side

_WORKER: dict[str, Any] = {}


def _init_worker(config_path: str, smoke: dict[str, Any] | None, soft: dict[str, str]) -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    _WORKER["setup"] = build_setup(config_path, smoke)
    _WORKER["soft_paths"] = soft
    _WORKER["soft"] = {}


def _soft_sensor(label: str) -> SoftSensor:
    cache = _WORKER["soft"]
    if label not in cache:
        import torch

        torch.set_num_threads(1)
        cache[label] = SoftSensor.load(REPO_ROOT / _WORKER["soft_paths"][label])
    return cache[label]


def make_feedback(setup: Setup, variant: str, draw: int, scenario: Scenario) -> Feedback:
    """Feedback object of one pair."""
    if variant == "F0":
        return NoFeedback()
    if variant == "F1":
        return IdealMeasurement()
    if variant == "F2":
        oc = setup.outliers.outliers
        return InlineSensor(
            setup.nominal,
            setup.measurement,
            setup.config.loop.measurement_interval_s,
            outlier_rate=setup.config.feedback.outlier_rate,
            degrees_of_freedom=oc.t_degrees_of_freedom,
            scale_factor=oc.scale_factor,
            noise_seed=_derive(setup.config.noise_seed, "F2/noise", draw, scenario.name),
            outlier_seed=_derive(setup.config.noise_seed, "F2/outliers", draw, scenario.name),
        )
    if variant.startswith("F3:"):
        return SoftSensorFeedback(_soft_sensor(variant), variant.removeprefix("F3:"))
    raise ValueError(f"unknown variant {variant!r}")


def _task_warm_up(draw: int) -> list[list[float]]:
    setup: Setup = _WORKER["setup"]
    states = warmed_up_states(
        setup.study,
        setup.nominal,
        setup.draws[draw].process,
        setup.scenarios,
        setup.config.loop.warm_up_s,
    )
    return np.asarray(states).tolist()


def _task_eval(
    variant: str,
    draw: int,
    ids: list[int],
    margins: list[float],
    states: list[list[float]],
    keep_arrays: bool,
) -> dict[str, Any]:
    setup: Setup = _WORKER["setup"]
    started = time.perf_counter()
    scenarios = [setup.scenarios[i] for i in ids]
    loop = setup.config.loop
    records = run_pairs(
        setup.study,
        setup.nominal,
        scenarios,
        margins,
        [make_feedback(setup, variant, draw, s) for s in scenarios],
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
        if keep_arrays:
            for key in (
                "energy_kwh_per_interval",
                "moisture_max_per_interval",
                "bias_per_interval",
            ):
                d["extra"][key] = [float(f"{v:.6g}") for v in d["extra"][key]]
            # commands: interval means of the inlet temperature only (size of the artifact)
            d["extra"]["inlet_command_per_interval"] = [
                float(f"{a:.5g}") for a, _ in d["extra"].pop("commands")
            ]
        out.append(d)
    return {"records": out, "seconds": time.perf_counter() - started, "n_pairs": len(ids)}


# ------------------------------------------------------------------------------ main side


def _key(stage: str, variant: str, draw: int, margin: float, scenario: str) -> str:
    return f"{stage}|{variant}|{draw}|{margin!r}|{scenario}"


class Cache:
    """Append-only JSONL of every evaluation, checked against a header on resume."""

    def __init__(self, path: Path, header: dict[str, Any]) -> None:
        self.path = path
        self.entries: dict[str, Any] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                first = json.loads(handle.readline())
                if first.get("header") != header:
                    raise SystemExit(f"{path} was written by another commit/config: {first}")
                for line in handle:
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry["value"]
            print(f"resume: {len(self.entries)} cached entries", flush=True)
        else:
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"header": header}) + "\n")

    def get(self, key: str) -> Any:
        return self.entries.get(key)

    def put(self, key: str, value: Any) -> None:
        self.entries[key] = value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value}) + "\n")


class Runner:
    """Submits evaluations, keeps the pool busy, stores every record in the cache."""

    def __init__(self, pool: ProcessPoolExecutor, cache: Cache, setup: Setup) -> None:
        self.pool = pool
        self.cache = cache
        self.setup = setup
        self.states: dict[int, list[list[float]]] = {}
        self.job_seconds: dict[str, list[float]] = {}
        self.tolerance = setup.study.control.violation_tolerance_kg_per_kg

    def warm_up(self) -> None:
        futures = {}
        for d in range(len(self.setup.draws)):
            cached = self.cache.get(f"warm_up|{d}")
            if cached is not None:
                self.states[d] = cached
            else:
                futures[d] = self.pool.submit(_task_warm_up, d)
        for d, future in futures.items():
            self.states[d] = future.result()
            self.cache.put(f"warm_up|{d}", self.states[d])

    def holds(self, record: dict[str, Any]) -> bool:
        return record["max_violation"] <= self.tolerance

    def submit(
        self, stage: str, variant: str, draw: int, ids: list[int], margins: list[float]
    ) -> Future:
        return self.pool.submit(
            _task_eval, variant, draw, ids, margins, self.states[draw], stage == "verify"
        )

    def collect(self, stage: str, future: Future) -> list[dict[str, Any]]:
        result = future.result()
        records = result["records"]
        for r in records:
            self.cache.put(_key(stage, r["variant"], r["draw"], r["margin"], r["scenario"]), r)
        if records:
            self.job_seconds.setdefault(records[0]["variant"], []).append(
                result["seconds"] / max(result["n_pairs"], 1)
            )
        return records


def stage_search(runner: Runner, variants: list[str]) -> dict[str, list[list[PairBisection]]]:
    """Per-pair bisections of all variants and draws, as concurrent chains."""
    setup = runner.setup
    control = setup.study.control
    names = [s.name for s in setup.scenarios]
    n_draws = len(setup.draws)
    searches = {
        v: [
            [
                PairBisection(
                    lower=setup.config.margin.lower_kg_per_kg,
                    upper=control.margin_upper_kg_per_kg,
                    tolerance=control.margin_tolerance_kg_per_kg,
                    max_upper=setup.config.margin.max_upper_kg_per_kg,
                )
                for _ in names
            ]
            for _ in range(n_draws)
        ]
        for v in variants
    }

    def advance(v: str, d: int) -> tuple[list[int], list[float]]:
        """Feed cached results until every pending pair needs a new evaluation."""
        while True:
            pending = [i for i, s in enumerate(searches[v][d]) if not s.done]
            progressed = False
            for i in pending:
                record = runner.cache.get(
                    _key("search", v, d, searches[v][d][i].next_margin(), names[i])
                )
                if record is not None:
                    searches[v][d][i].report(runner.holds(record))
                    progressed = True
            if not progressed:
                pending = [i for i, s in enumerate(searches[v][d]) if not s.done]
                return pending, [searches[v][d][i].next_margin() for i in pending]

    running: dict[Future, tuple[str, int]] = {}
    started = time.time()
    for v in variants:
        for d in range(n_draws):
            ids, margins = advance(v, d)
            if ids:
                running[runner.submit("search", v, d, ids, margins)] = (v, d)
    finished_jobs = 0
    while running:
        done, _ = wait(running, return_when=FIRST_COMPLETED)
        for future in done:
            v, d = running.pop(future)
            runner.collect("search", future)
            finished_jobs += 1
            ids, margins = advance(v, d)
            if ids:
                running[runner.submit("search", v, d, ids, margins)] = (v, d)
            else:
                print(
                    f"[search] {v} draw {d} done after "
                    f"{max(len(s.evaluations) for s in searches[v][d])} rounds "
                    f"({(time.time() - started) / 3600:.2f} h, {finished_jobs} jobs)",
                    flush=True,
                )
    return searches


def stage_verify(
    runner: Runner, variants: list[str], searches: dict[str, list[list[PairBisection]]]
) -> dict[str, Any]:
    """Margins of criteria (i) and (ii) per variant, each evaluated on all pairs."""
    setup = runner.setup
    names = [s.name for s in setup.scenarios]
    n_draws = len(setup.draws)
    n_pairs = n_draws * len(names)
    k_ii = required_count(n_pairs, setup.config.margin.holding_share)
    tolerance = setup.study.control.margin_tolerance_kg_per_kg
    cap = setup.config.margin.max_upper_kg_per_kg
    out: dict[str, Any] = {}
    for v in variants:
        per_pair = np.array(
            [
                [searches[v][d][i].smallest_holding for i in range(len(names))]
                for d in range(n_draws)
            ]
        )
        out[v] = {
            "per_pair_margin": per_pair.tolist(),
            "per_pair_converged": [
                [searches[v][d][i].converged for i in range(len(names))] for d in range(n_draws)
            ],
            "per_pair_evaluations": [
                [len(searches[v][d][i].evaluations) for i in range(len(names))]
                for d in range(n_draws)
            ],
            "criteria": {},
        }
    for criterion, k_required in (("ii", k_ii), ("i", n_pairs)):
        state = {}
        for v in variants:
            flat = np.array(out[v]["per_pair_margin"]).ravel()
            state[v] = {"candidate": criterion_margin(flat, k_required), "tried": [], "extra": 0}
        active = [v for v in variants if math.isfinite(state[v]["candidate"])]
        for v in variants:
            if v not in active:
                out[v]["criteria"][criterion] = {
                    "margin": None,
                    "converged": False,
                    "k_required": k_required,
                    "reason": f"fewer than {k_required} pairs hold at any margin up to {cap}",
                }
        while active:
            futures: dict[Future, tuple[str, int]] = {}
            for v in active:
                m = state[v]["candidate"]
                for d in range(n_draws):
                    missing = [
                        i
                        for i, n in enumerate(names)
                        if runner.cache.get(_key("verify", v, d, m, n)) is None
                    ]
                    if missing:
                        futures[runner.submit("verify", v, d, missing, [m] * len(missing))] = (v, d)
            for future in futures:
                runner.collect("verify", future)
            still = []
            for v in active:
                m = state[v]["candidate"]
                records = [
                    runner.cache.get(_key("verify", v, d, m, n))
                    for d in range(n_draws)
                    for n in names
                ]
                holding = sum(runner.holds(r) for r in records)
                state[v]["tried"].append([m, holding])
                print(
                    f"[verify {criterion}] {v}: margin {m:.6f}, {holding}/{n_pairs} hold",
                    flush=True,
                )
                if holding >= k_required:
                    if criterion != "ii":  # per-interval arrays only at the main criterion (size)
                        records = [
                            {
                                **r,
                                "extra": {
                                    k: x for k, x in r["extra"].items() if k == "outlier_share"
                                },
                            }
                            for r in records
                        ]
                    out[v]["criteria"][criterion] = {
                        "margin": m,
                        "converged": True,
                        "k_required": k_required,
                        "holding": holding,
                        "tried": state[v]["tried"],
                        "records": records,
                    }
                    continue
                flat = np.array(out[v]["per_pair_margin"]).ravel()
                nxt = next_candidate(flat, m)
                if not math.isfinite(nxt):
                    if state[v]["extra"] >= MAX_EXTRA_STEPS or m + tolerance > cap:
                        out[v]["criteria"][criterion] = {
                            "margin": None,
                            "converged": False,
                            "k_required": k_required,
                            "tried": state[v]["tried"],
                            "reason": "verification failed (non-monotone pairs), steps exhausted",
                        }
                        continue
                    state[v]["extra"] += 1
                    nxt = m + tolerance
                state[v]["candidate"] = nxt
                still.append(v)
            active = still
    return out


def probe(runner: Runner, variants: list[str], margin: float) -> dict[str, Any]:
    """One full-length job per variant (draw 0, all scenarios) for the budget estimate."""
    setup = runner.setup
    ids = list(range(len(setup.scenarios)))
    timing = {}
    futures = {v: runner.submit("probe", v, 0, ids, [margin] * len(ids)) for v in variants}
    for v, future in futures.items():
        result = future.result()
        timing[v] = {
            "seconds": result["seconds"],
            "n_pairs": result["n_pairs"],
            "max_violation": max(r["max_violation"] for r in result["records"]),
        }
        print(f"[probe] {v}: {result['seconds']:.0f} s for {result['n_pairs']} pairs", flush=True)
    return timing


def _soft_paths(
    record_name: str | None, config: Any
) -> tuple[dict[str, str], dict[str, Any] | None]:
    if record_name is None:
        return {}, None
    path = RUNS_DIR / f"{record_name}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["provenance"]["git/dirty"]:
        raise SystemExit(f"{path} was written from a dirty tree")
    if record["sensor_value_fingerprint"] != config.fingerprint:
        raise SystemExit("soft-sensor record was trained under another sensor-value config")
    paths = {}
    for seed_record in record["seeds"]:
        checkpoint = REPO_ROOT / seed_record["checkpoint"]
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != seed_record["checkpoint_sha256"]:
            raise SystemExit(f"{checkpoint}: SHA-256 differs from the training record")
        paths[f"F3:{seed_record['seed']}"] = seed_record["checkpoint"]
    expected = {f"F3:{s}" for s in config.feedback.soft_sensor_training_seeds}
    if set(paths) != expected:
        raise SystemExit(
            f"soft-sensor seeds {sorted(paths)} differ from the config {sorted(expected)}"
        )
    summary = {
        "record": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoints": dict(paths),
    }
    return paths, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--soft-sensor-record", default=None, help="name of the training record")
    parser.add_argument("--variants", default=None, help="comma list (default: all)")
    parser.add_argument("--out-dir", default=str(RUNS_DIR))
    parser.add_argument("--smoke", default=None, help="hours,draws,per_type,pre_roll_s (smoke run)")
    parser.add_argument("--probe", type=float, default=None, help="margin of the budget probe")
    args = parser.parse_args()

    smoke = None
    if args.smoke:
        hours, n_draws, per_type, pre = args.smoke.split(",")
        smoke = {
            "hours": float(hours),
            "draws": int(n_draws),
            "per_type": int(per_type),
            "pre_roll_s": float(pre),
        }
    setup = build_setup(args.config, smoke)
    config = setup.config
    soft, soft_summary = _soft_paths(args.soft_sensor_record, config)
    variants = list(BASE_VARIANTS) + sorted(soft)
    if args.variants:
        variants = [v for v in args.variants.split(",") if v]
        unknown = [v for v in variants if v not in list(BASE_VARIANTS) + sorted(soft)]
        if unknown:
            raise SystemExit(f"unknown or unavailable variants {unknown}")
    out_dir = Path(args.out_dir).resolve()
    artifact_run = out_dir == RUNS_DIR.resolve() and smoke is None and args.probe is None
    if out_dir == RUNS_DIR.resolve() and not artifact_run:
        raise SystemExit("smoke and probe runs must not write into artifacts/runs")
    if artifact_run and set(variants) != set(BASE_VARIANTS) | {
        f"F3:{s}" for s in config.feedback.soft_sensor_training_seeds
    }:
        raise SystemExit("the artifact run needs all variants (F0-F2 and every F3 seed)")
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
    prov = provenance("not used", ignore_paths=[output, cache_path])
    header = {
        "commit": git("rev-parse", "HEAD"),
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
        "study": "sensor value (docs/28_sensor_value.md)",
        "config": config.raw,
        "headroom_config": setup.study.raw,
        "sensor_value_fingerprint": config.fingerprint,
        "study_fingerprint": setup.study.fingerprint,
        "process_config_fingerprint": setup.nominal.fingerprint,
        "sensitivity_fingerprint": load_sensitivity_config(config.sensitivity_config).fingerprint,
        "outlier_config_digest": setup.outliers.digest,
        "soft_sensor": soft_summary,
        "smoke": smoke,
        "workers": args.workers,
        "variants": variants,
        "draws": [d.as_dict() for d in setup.draws],
        "rejected_sequence_positions": setup.rejected,
        "scenarios": [{"name": s.name, "kind": s.kind, "index": s.index} for s in setup.scenarios],
        "violation_tolerance_kg_per_kg": setup.study.control.violation_tolerance_kg_per_kg,
    }
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.config, smoke, soft)
    ) as pool:
        runner = Runner(pool, Cache(cache_path, header), setup)
        t0 = time.time()
        runner.warm_up()
        payload.setdefault("stage_runtime_s", {})["warm_up"] = time.time() - t0
        print(f"warm-up done in {(time.time() - t0) / 60:.1f} min", flush=True)
        if args.probe is not None:
            payload["probe"] = probe(runner, variants, args.probe)
            payload["runtime_s"] = time.time() - started
            (out_dir / f"{args.name}_probe.json").write_text(
                json.dumps(payload, indent=1, default=str), encoding="utf-8"
            )
            return 0
        t0 = time.time()
        searches = stage_search(runner, variants)
        payload["stage_runtime_s"]["search"] = time.time() - t0
        t0 = time.time()
        payload["results"] = stage_verify(runner, variants, searches)
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
