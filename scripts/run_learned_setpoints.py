"""Learned controllers on the load-following task: models, labels, margin searches, runs.

Study document: ``docs/NN_learned_setpoints.md``; settings and placeholders:
``conf/learned_setpoints/learned_setpoints.yaml``; reading (pre-registered): docstring of
``scripts/analyze_learned_setpoints.py``.

Question: what does it cost to replace the known model inside a controller by a learned one, as
a function of the data, and in which form (learned model plus optimiser, or direct prediction of
the command)? Task, truth, days, tail and margin search are those of the load-following study
(Befund 126, reading T1: the learned arms use the schedule, their twin is C2).

Stages (``--stage``)
--------------------
``stage0``
    No truth run of a test day. Units, each saved as soon as it is finished and skipped on
    ``--resume``: (a) per (master seed, ``n_train``) the data of arm N of Befund 94, the composite
    GRU and the static quadratic regression (checked against the Befund 94 records), their
    steady tables and the map error at the limit (descriptive); (b) the table of the true steady
    response (control M-tab); (c) per label day the expert C2 at the label margins (training
    months); (d) per (arm, budget, seed) a policy of family P.
    Files in ``artifacts/checkpoints/learned_setpoints/<name>/`` (gitignored), record with their
    SHA-256 in ``<name>_stage0.json.gz``.
``dryrun``
    ALL units at minimal cost: stage 0 with tiny data, two epochs and coarse tables, then every
    cell of the main run on days shortened to ``--day-hours`` with a coarse margin tolerance.
    Surfaces failing units, reports time and peak memory per worker. ``<name>_dryrun.json.gz``.
``probe``
    Full-length days: every kind of cell on training days under parallel load; from the times
    the wall-clock time of the main run is extrapolated and the budget rule of decision 8 is
    applied (:func:`apply_budget_rule`). ``<name>_probe.json.gz``.
``main``
    Parts (``--parts``, default all): ``reference`` (Cfix, C0, C2, C3-15, M-tab), ``M``, ``D``,
    ``P`` (margin search of every cell on the test days), ``twin_curve`` (C2 over a margin grid),
    ``decomposition`` (C2 at the margin of every admissible cell of the largest budget),
    ``deployable`` (margin from tuning days of the training months, then the test days at that
    margin; cells of the largest budget). ``<name>.json.gz``.

Margin of a cell: smallest margin with which the path condition holds on all days, tail included
(bisection of ``twinlink_ml.headroom.margin`` on a critical subset -- the days with the largest
violation of the cell itself at margin 0 -- verified on all days; settings of the headroom
config, upper end of the load-following config). A cell whose search does not reach a margin is
NOT ADMISSIBLE: it gets no number.

Every unit (cell x margin x day) is appended to ``artifacts/runs/_<name>_<stage>_cache.jsonl``
(gitignored) as soon as it is finished; ``--resume`` reuses it (same commit and config required).
The truth simulation costs about the same for one row as for some hundred, so the searches of all
cells run side by side and their rows are packed into common batches. A failed unit is logged and
recorded as an error; it does not discard other units. Run from the worktree with
``tools/run_here.sh scripts/run_learned_setpoints.py --stage ... --name ...``.
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
import threading
import time
import traceback
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import REPO_ROOT, RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.headroom.margin import MarginSearch, bisect_margin_over_scenarios  # noqa: E402
from twinlink_ml.learned_setpoints.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    LearnedSetpointsConfig,
    load_learned_setpoints_config,
)
from twinlink_ml.learned_setpoints.labels import LabelSet, nested_label_days  # noqa: E402
from twinlink_ml.learned_setpoints.policy import POLICY_ARMS, fit_policy, save_policy  # noqa: E402
from twinlink_ml.learned_setpoints.study import Assets, Cell, run_rows  # noqa: E402
from twinlink_ml.load_following.market import (  # noqa: E402
    Market,
    build_days,
    build_market,
    daily_spread,
    month_days,
)
from twinlink_ml.load_following.memory import peak_memory_mb  # noqa: E402
from twinlink_ml.load_following.study import Solvers  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning  # noqa: E402

STAGES = ("stage0", "dryrun", "probe", "main")
PARTS = ("reference", "M", "D", "P", "twin_curve", "decomposition", "deployable")
ASSET_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "learned_setpoints"
DRY_MARGIN_TOLERANCE = 4.0e-3
PROBE_MARGIN = 0.01
ROWS_PER_TASK = 320
MIN_ROWS_PER_TASK = 6
INADMISSIBLE_VIOLATION = 1.0  # kg/kg; stands for "no such controller at this margin"


_LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    """One progress line with a time stamp (searches log from their own threads)."""
    with _LOG_LOCK:
        sys.stdout.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
        sys.stdout.flush()


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------ cells and days


def study_cells(config: LearnedSetpointsConfig, cuts: Sequence[str] = ()) -> dict[str, list[Cell]]:
    """Cells of the main run per part, after the cuts of decision 8 (``cuts``, in their order)."""
    n_train = [n for n in config.n_train if not ("n_train_32" in cuts and n == 32)]
    seeds = list(config.seeds)
    levels = {"M-quad": ["none", "fb"], "M-gru": ["none", "fb", "lab"]}
    if "lab" in cuts:
        levels["M-gru"].remove("lab")
    d_seeds = seeds[:3] if "d_gru_seeds_3" in cuts else seeds
    p_arms = [a for a in POLICY_ARMS if not ("p_gru" in cuts and a == "P-gru")]
    return {
        "reference": [Cell("Cfix"), Cell("C3-15"), Cell("M-tab")],
        "M": [
            Cell(arm, level, n, s)
            for arm, arm_levels in levels.items()
            for level in arm_levels
            for n in n_train
            for s in seeds
        ],
        "D": [
            Cell("D-gru", "none", n, s)
            for n in config.gru_mpc_n_train
            if n in n_train
            for s in d_seeds
        ],
        "P": [
            Cell(arm, "none", k, s)
            for arm in p_arms
            for k in config.policy.budgets_days
            for s in config.policy.seeds
        ],
    }


def largest_budget(cell: Cell, config: LearnedSetpointsConfig) -> bool:
    """Whether a cell is at the largest budget of its family."""
    if cell.family == "P":
        return cell.budget == max(config.policy.budgets_days)
    if cell.family == "D":
        return cell.budget == max(config.gru_mpc_n_train)
    return cell.family == "M" and cell.budget == max(config.n_train)


def tuning_days(
    market: Market,
    months: Sequence[int],
    per_tercile: tuple[int, int, int],
    seed: int,
    exclude: Sequence[int],
) -> list[int]:
    """Days of ``months`` drawn per tercile of the price spread, none of them in ``exclude``."""
    chosen: list[int] = []
    for month in months:
        days = [d for d in month_days(market.clock, month) if d not in set(exclude)]
        order = sorted(days, key=lambda d: (daily_spread(market.tariff, d), d))
        rng = np.random.default_rng([seed, month])
        for part, count in zip(np.array_split(np.array(order), 3), per_tercile, strict=True):
            chosen.extend(int(d) for d in rng.choice(part, size=count, replace=False))
    return sorted(chosen)


def reference_task(config: LearnedSetpointsConfig, market: Market) -> dict[str, Any]:
    """What is taken from the artifacts of the load-following study (checked against its config)."""
    lf = config.load_following
    with gzip.open(RUNS_DIR / config.stage0_artifact, "rt", encoding="utf-8") as handle:
        stage0 = json.load(handle)
    with gzip.open(RUNS_DIR / config.main_artifact, "rt", encoding="utf-8") as handle:
        main = json.load(handle)
    analysis = json.loads((RUNS_DIR / config.analysis_artifact).read_text(encoding="utf-8"))
    for name, artifact in (("stage 0", stage0), ("main run", main)):
        if artifact["study_fingerprint"] != lf.fingerprint:
            raise SystemExit(f"load-following {name} artifact: study fingerprint differs")
        if artifact["headroom_fingerprint"] != lf.headroom.fingerprint:
            raise SystemExit(f"load-following {name} artifact: headroom fingerprint differs")
    reading = analysis["reading"]
    if not reading["valid"]:
        raise SystemExit("the reading of the load-following study is not valid")
    # reference rule of the study document (section 4), fixed before the reading was known
    twin = "C0" if reading["reading"] == "T0" else "C2"
    label_days = [int(d) for d in stage0["stage0"]["day_sets"]["training"]]
    tuning = tuning_days(
        market,
        lf.prices.training_months,
        config.tuning_per_month_by_tercile,
        config.tuning_seed,
        label_days,
    )
    j = {
        kind: float(
            sum(r["cost_e"] for r in block["records"])
            / sum(r["product_dry_kg"] for r in block["records"])
        )
        for kind, block in main["main"].items()
    }
    return {
        "load_following_reading": reading["reading"],
        "twin": twin,
        "band": [float(v) for v in stage0["stage0"]["bands"]["main"]],
        "margins": {k: float(v) for k, v in main["margins"].items()},
        "cost_per_kg": j,
        "c2_records": {
            str(r["day_of_year"]): {k: r[k] for k in ("cost_e", "product_dry_kg", "max_violation")}
            for r in main["main"]["C2"]["records"]
        },
        "day_sets": {
            "test": [int(d) for d in main["day_ids"]],
            "label": label_days,
            "tuning": tuning,
        },
        "label_spread": [float(v) for v in stage0["stage0"]["spread_training"]],
        "training_months": [int(m) for m in lf.prices.training_months],
        "test_months": [int(m) for m in lf.prices.test_months],
        "source_commits": {
            "stage0": stage0["provenance"]["git/commit"],
            "main": main["provenance"]["git/commit"],
        },
    }


# ------------------------------------------------------------------------------ stage 0 units


def _befund94_record(path: Path, label: str, seed: int, n_train: int, test_set: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            p = record["params"]
            if (
                p["model/label"] == label
                and p["master_seed"] == seed
                and p["n_train"] == n_train
                and p["eval/primary"]
                and p["test_set"] == test_set
            ):
                return record  # type: ignore[no-any-return]
    raise KeyError((label, seed, n_train, test_set))


def _max_difference(new: dict[str, Any], old: dict[str, Any]) -> tuple[float, int]:
    """Largest absolute difference over shared numeric entries (timings left out)."""
    worst, n = 0.0, 0
    for key, value in new.items():
        if key not in old or "inference" in key or "elapsed" in key:
            continue
        try:
            a = np.asarray(value, dtype=np.float64).ravel()
            b = np.asarray(old[key], dtype=np.float64).ravel()
        except (TypeError, ValueError):
            continue
        if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
            return math.inf, n
        mask = np.isfinite(a)
        if mask.any():
            worst = max(worst, float(np.max(np.abs(a[mask] - b[mask]))))
        n += 1
    return worst, n


def _map_error(table: Any, config: LearnedSetpointsConfig, band: Sequence[float]) -> dict[str, Any]:
    """Predicted minus true moisture and power at the points the map chooses (descriptive)."""
    from twinlink_ml.headroom.scenarios import weather_grid
    from twinlink_ml.learned_setpoints.maps import TableMap, TruthResponse

    lf = config.load_following
    limit = lf.headroom.line.product_moisture_limit_kg_per_kg
    steady = TableMap(table, config)
    rows, predicted = [], []
    none = 0
    for ambient, humidity in weather_grid(
        lf.process, lf.headroom.scenarios.ambient.grid_points_per_axis
    ):
        for spray in (band[0], lf.nominal_spray_rate, band[1]):
            d = np.array([ambient, humidity, table.solid_fraction, spray, 0.0])
            point = steady.point(d, spray, limit)
            if point is None:
                none += 1
                continue
            rows.append(
                [
                    point.command[0],
                    point.command[1],
                    spray,
                    table.nozzle_pressure_bar,
                    ambient,
                    humidity,
                    table.solid_fraction,
                ]
            )
            values, _ = steady.response(
                point.command[0], point.command[1], spray, ambient, humidity
            )
            predicted.append(values)
    out: dict[str, Any] = {"points": len(rows), "no_admissible_point": none, "bound": limit}
    if rows:
        truth, valid = TruthResponse(lf.process).predict(np.array(rows))
        diff = (np.array(predicted) - truth)[valid]
        for k, name in enumerate(("power_kw", "moisture_kg_per_kg", "bed_temperature_celsius")):
            out[name] = {
                "mean": float(diff[:, k].mean()),
                "min": float(diff[:, k].min()),
                "max": float(diff[:, k].max()),
            }
        out["truly_above_limit"] = int(np.sum(truth[valid][:, 1] > limit + 1e-4))
    return out


def model_unit(
    seed: int, n_train: int, smoke: bool, directory: str, config_path: str, band: list[float]
) -> dict[str, Any]:
    """Stage-0 unit (a): data, GRU, static quadratic regression, their tables and map errors."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    import run_arch_learning_curve_worker as old_worker
    import run_arch_rerun_worker as rerun_worker

    from twinlink_ml.evaluation import ValidationPolicy, build_datasets
    from twinlink_ml.evaluation.evaluator import RolloutEvaluator
    from twinlink_ml.learned_setpoints.maps import (
        GruFixedPointResponse,
        QuadraticResponse,
        table_axes,
        tabulate,
    )
    from twinlink_ml.models import StaticRegressionModel
    from twinlink_ml.simulation import load_process_config
    from twinlink_ml.utils.checkpoints import load_checkpoint, save_checkpoint

    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    started = time.perf_counter()
    config = load_learned_setpoints_config(config_path)
    if smoke:
        config = dataclasses.replace(
            config,
            table=dataclasses.replace(
                config.table,
                inlet_temperature_points=5,
                gas_velocity_points=4,
                spray_rate_points=3,
                ambient_temperature_points=3,
                relative_humidity_points=3,
            ),
        )
    process = load_process_config(old_worker.CONFIG_PATH)
    process = dataclasses.replace(
        process,
        numerics=dataclasses.replace(process.numerics, enforce_gas_velocity_projection=True),
    )
    spec = rerun_worker.arm_spec("N")
    if smoke:
        spec = dataclasses.replace(spec, run_duration_s=6000.0, n_train_max=4, n_val=2, n_test=2)
    own = rerun_worker._generate(process, spec, seed)
    if not smoke and own.process_config.fingerprint != config.process_fingerprint:
        raise RuntimeError(f"process/fingerprint {own.process_config.fingerprint!r}")
    size = min(n_train, 4) if smoke else n_train
    train, validation, test = build_datasets(own, size, policy=ValidationPolicy.FIXED_POOL)
    out_dir = Path(directory)
    lf = config.load_following
    axes = table_axes(config, (band[0], band[1]))
    fixed = {
        "nozzle_pressure_bar": lf.process.actions.nozzle_pressure_bar.nominal,
        "solid_fraction": lf.headroom.line.nominal_solid_fraction,
    }
    evaluator = RolloutEvaluator(horizons=(1, spec.horizon))
    records = Path(REPO_ROOT / config.befund94_records)
    result: dict[str, Any] = {
        "master_seed": seed,
        "n_train": n_train,
        "smoke": smoke,
        "process_fingerprint": own.process_config.fingerprint,
        "split_digest": own.split.digest,
        "data_spec": spec.as_parameters(),
    }

    # composite GRU of the Befund 94 pipeline
    gru = old_worker._build_model(
        old_worker._architecture_factory("gru"), 2 if smoke else config.max_epochs
    )
    fit_started = time.perf_counter()
    gru.fit(train, seed=own.seeds.model_seed, validation=validation)
    checkpoint = out_dir / Assets.checkpoint_name(n_train, seed)
    if checkpoint.exists():
        raise SystemExit(f"{checkpoint} exists -- checkpoints are never overwritten")
    save_checkpoint(
        checkpoint,
        gru,
        model_label="gru",
        run_uid=checkpoint.stem,
        target_names=train.target_names,
        exogenous_names=train.exogenous_names,
        n_exogenous=len(train.exogenous_names),
        input_scaler=train.input_scaler,
        target_scaler=train.target_scaler,
        process_fingerprint=own.process_config.fingerprint,
        data_spec_params=spec.as_parameters(),
        seed_bundle_params=own.seeds.as_parameters(),
        git_commit=git("rev-parse", "HEAD"),
    )
    gru_metrics = dict(evaluator.evaluate(gru, test).as_flat_metrics())
    result["gru"] = {
        "fit_s": time.perf_counter() - fit_started,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "training": gru.training_diagnostics(),
        "metrics_excited": gru_metrics,
    }
    # static quadratic regression of the Befund 94 pipeline (label "static" in its records)
    static = StaticRegressionModel()
    static.fit(train, seed=own.seeds.model_seed, validation=validation)
    static_metrics = dict(evaluator.evaluate(static, test).as_flat_metrics())
    result["quad"] = {"name": static.name, "metrics_excited": static_metrics}
    if not smoke:
        for key, label, metrics in (
            ("gru", "gru", gru_metrics),
            ("quad", "static", static_metrics),
        ):
            old = _befund94_record(records, label, seed, n_train, "excited")
            difference, count = _max_difference(metrics, old["metrics"])
            result[key]["reproduction_befund94"] = {
                "test_set": "excited",
                "max_abs_difference_metrics": difference,
                "n_metrics_compared": count,
            }
    # tables
    model, metadata = load_checkpoint(checkpoint)
    responses = {
        "gru": GruFixedPointResponse(model, metadata, chunk_points=config.table.chunk_points),
        "quad": QuadraticResponse(static, train.input_scaler, train.target_scaler),
    }
    for kind, response in responses.items():
        table_started = time.perf_counter()
        table = tabulate(response, axes, source=f"{kind}/n{n_train}/s{seed}", **fixed)
        arm = "M-gru" if kind == "gru" else "M-quad"
        path = out_dir / Assets.table_name(arm, n_train, seed)
        table.save(path)
        result[kind].update(
            {
                "table": path.name,
                "table_sha256": sha256(path),
                "table_digest": table.digest,
                "table_s": time.perf_counter() - table_started,
                "table_valid_share": float(table.valid.mean()),
                "table_status": response.status_counts(),
                "map_error_at_limit": _map_error(table, config, band),
            }
        )
    result["total_s"] = time.perf_counter() - started
    result["peak_mb"] = peak_memory_mb()
    return result


def truth_table_unit(
    directory: str, config_path: str, band: list[float], smoke: bool
) -> dict[str, Any]:
    """Stage-0 unit (b): table of the true steady response (control M-tab)."""
    from twinlink_ml.learned_setpoints.maps import TruthResponse, table_axes, tabulate

    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    config = load_learned_setpoints_config(config_path)
    if smoke:
        config = dataclasses.replace(
            config,
            table=dataclasses.replace(
                config.table,
                inlet_temperature_points=5,
                gas_velocity_points=4,
                spray_rate_points=3,
                ambient_temperature_points=3,
                relative_humidity_points=3,
            ),
        )
    lf = config.load_following
    started = time.perf_counter()
    table = tabulate(
        TruthResponse(lf.process),
        table_axes(config, (band[0], band[1])),
        nozzle_pressure_bar=lf.process.actions.nozzle_pressure_bar.nominal,
        solid_fraction=lf.headroom.line.nominal_solid_fraction,
        source="truth",
    )
    path = Path(directory) / Assets.table_name("M-tab", 0, 0)
    table.save(path)
    return {
        "table": path.name,
        "table_sha256": sha256(path),
        "table_digest": table.digest,
        "table_valid_share": float(table.valid.mean()),
        "table_s": time.perf_counter() - started,
        "map_error_at_limit": _map_error(table, config, band),
    }


def policy_unit(
    arm: str, budget: int, seed: int, days: list[int], directory: str, config_path: str, smoke: bool
) -> dict[str, Any]:
    """Stage-0 unit (d): one policy of family P."""
    config = load_learned_setpoints_config(config_path)
    settings = config.policy
    if smoke:
        settings = dataclasses.replace(settings, max_epochs=3)
    out_dir = Path(directory)
    labels = LabelSet.load(out_dir / "labels.npz")
    started = time.perf_counter()
    policy, info = fit_policy(arm, labels, days, settings, seed)
    path = out_dir / Assets.policy_name(arm, budget, seed)
    save_policy(policy, path)
    return {
        **info,
        "budget": budget,
        "seed": seed,
        "file": path.name,
        "sha256": sha256(path),
        "fit_s": time.perf_counter() - started,
    }


# ------------------------------------------------------------------------------ worker side

_WORKER: dict[str, Any] = {}


def _init_worker(
    config_path: str, day_hours: int | None, directory: str, band: list[float]
) -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    config = load_learned_setpoints_config(config_path, day_hours=day_hours)
    market = build_market(config.load_following, REPO_ROOT)
    _WORKER.update(
        config=config,
        market=market,
        solvers=Solvers(config.load_following, market.reference_tariff),
        assets=Assets(Path(directory)),
        band=(band[0], band[1]),
        days={},
    )


def _day(day: int) -> Any:
    cache = _WORKER["days"]
    if day not in cache:
        cache[day] = build_days(_WORKER["config"].load_following, _WORKER["market"], [day])[0]
    return cache[day]


def _task(
    rows: list[tuple[str, float, int]], tail_margin: float, log_decisions: bool
) -> dict[str, Any]:
    """Run rows ``(cell key, margin, day)`` in one batch; an exception becomes an error entry."""
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    error = None
    try:
        records = run_rows(
            [(Cell.from_key(key), margin, _day(day)) for key, margin, day in rows],
            tail_margin,
            _WORKER["solvers"],
            _WORKER["assets"],
            _WORKER["config"],
            _WORKER["band"],
            log_decisions=log_decisions,
        )
        for record in records:
            if "no fixed point at margin" in (record.get("error") or ""):
                # Cfix has no set point at this margin: not an error of the run, the margin fails
                record["inadmissible"] = "no fixed point at this margin"
                record["max_violation"] = INADMISSIBLE_VIOLATION
                del record["error"]
            if log_decisions and "decisions" in record:
                record["decisions"] = {k: v.tolist() for k, v in record["decisions"].items()}
    except Exception:  # a failed batch must not stop the run
        error = traceback.format_exc()
    return {
        "records": records,
        "error": error,
        "seconds": time.perf_counter() - started,
        "peak_mb": peak_memory_mb(),
        "pid": os.getpid(),
    }


# ------------------------------------------------------------------------------ main side


def _key(cell: str, margin: float, tail: float, day_set: str, day: int) -> str:
    return f"{cell}|{margin!r}|{tail!r}|{day_set}|{day}"


@dataclasses.dataclass
class _Request:
    cell: str
    margin: float
    days: list[int]
    day_set: str
    event: threading.Event = dataclasses.field(default_factory=threading.Event)


class Evaluator:
    """Runs units in the pool, packs the requests of concurrent searches into batches, and keeps
    every record in the resumable cache."""

    def __init__(
        self, pool: ProcessPoolExecutor, workers: int, cache: Path, header: dict, tail_margin: float
    ) -> None:
        self.pool = pool
        self.workers = workers
        self.cache_path = cache
        self.tail_margin = tail_margin
        self.records: dict[str, dict] = {}
        self.errors: list[dict[str, Any]] = []
        self.peak_mb: dict[int, float] = {}
        self.task_seconds: list[dict[str, Any]] = []
        self.units_run = 0
        self.rounds = 0
        self._lock = threading.Condition()
        self._pending: list[_Request] = []
        self._active = 0
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

    # ---------------------------------------------------------------------- called by searches

    def run(self, cell: Cell, margin: float, days: Sequence[int], day_set: str) -> list[dict]:
        """Records of ``cell`` at ``margin`` on ``days`` (blocks until they are computed)."""
        margin = float(margin)
        keys = [_key(cell.key, margin, self.tail_margin, day_set, d) for d in days]
        missing = [d for d, k in zip(days, keys, strict=True) if k not in self.records]
        if missing:
            request = _Request(cell.key, margin, missing, day_set)
            with self._lock:
                self._pending.append(request)
                self._lock.notify_all()
            request.event.wait()
        return [self.records[k] for k in keys]

    # ---------------------------------------------------------------------- dispatcher

    def _store(self, key: str, record: dict) -> None:
        self.records[key] = record
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "record": record}) + "\n")

    def _dispatch(self, requests: list[_Request]) -> None:
        """Compute the rows of ``requests`` in common batches and release the requests."""
        rows: dict[tuple[bool, str], list[tuple[str, float, int]]] = {}
        seen: set[str] = set()
        for request in requests:
            pre_roll = Cell.from_key(request.cell).level != "none"
            for day in request.days:
                key = _key(request.cell, request.margin, self.tail_margin, request.day_set, day)
                if key in seen or key in self.records:
                    continue
                seen.add(key)
                rows.setdefault((pre_roll, request.day_set), []).append(
                    (request.cell, request.margin, day)
                )
        futures: dict[Future, tuple[str, list[tuple[str, float, int]]]] = {}
        for (_, day_set), group in sorted(rows.items()):
            # every worker gets a batch as soon as there are enough rows; more than
            # ROWS_PER_TASK rows per batch only split further
            per_wave = self.workers * ROWS_PER_TASK
            if len(group) <= per_wave:
                n_tasks = min(self.workers, math.ceil(len(group) / MIN_ROWS_PER_TASK))
            else:
                n_tasks = self.workers * math.ceil(len(group) / per_wave)
            # deal the rows cell by cell, so that every batch gets a share of each kind of cell
            ordered = sorted(group)
            for t in range(n_tasks):
                chunk = ordered[t::n_tasks]
                futures[self.pool.submit(_task, chunk, self.tail_margin, False)] = (day_set, chunk)
        self.rounds += 1
        n_rows = sum(len(g) for g in rows.values())
        started = time.time()
        if n_rows:
            log(
                f"round {self.rounds}: {n_rows} rows of {len(requests)} requests in {len(futures)} batches"
            )
        for future in as_completed(futures):
            day_set, chunk = futures[future]
            out = future.result()
            self.peak_mb[out["pid"]] = max(self.peak_mb.get(out["pid"], 0.0), out["peak_mb"])
            records = out["records"]
            if out["error"] is not None:
                log(f"ERROR batch of {len(chunk)} rows: {out['error'][-400:]}")
                records = [
                    {
                        "cell": c,
                        "margin": m,
                        "day_of_year": d,
                        "error": out["error"],
                        "max_violation": None,
                    }
                    for c, m, d in chunk
                ]
            decide = 0.0
            for (cell, margin, day), record in zip(chunk, records, strict=True):
                if record.get("error"):
                    self.errors.append(
                        {
                            "cell": cell,
                            "margin": margin,
                            "day_of_year": day,
                            "day_set": day_set,
                            "error": record["error"],
                        }
                    )
                    log(f"ERROR unit {cell} m={margin} day {day}: {record['error'][-300:]}")
                else:
                    decide += row_seconds(record)
                self._store(_key(cell, margin, self.tail_margin, day_set, day), record)
            self.units_run += len(chunk)
            self.task_seconds.append(
                {
                    "rows": len(chunk),
                    "seconds": out["seconds"],
                    "decide_s": decide,
                    "pre_roll": chunk and Cell.from_key(chunk[0][0]).level != "none",
                }
            )
        if n_rows:
            log(f"round {self.rounds} done in {time.time() - started:.0f} s")
        for request in requests:
            request.event.set()

    def run_all(self, jobs: Sequence[Callable[[], Any]]) -> list[Any]:
        """Run ``jobs`` side by side (one thread each); their requests are computed in rounds.

        A round starts when every job that is still running waits for records. A job that raises
        is recorded with its traceback and returns ``None``.
        """
        results: list[Any] = [None] * len(jobs)

        def work(index: int) -> None:
            try:
                results[index] = jobs[index]()
            except Exception:
                self.errors.append({"job": index, "error": traceback.format_exc()})
                log(f"ERROR job {index}: {traceback.format_exc()[-400:]}")
            finally:
                with self._lock:
                    self._active -= 1
                    self._lock.notify_all()

        with self._lock:
            self._active = len(jobs)
        threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(jobs))]
        for thread in threads:
            thread.start()
        while True:
            with self._lock:
                while self._active > 0 and len(self._pending) < self._active:
                    self._lock.wait(timeout=1.0)
                if self._active == 0 and not self._pending:
                    break
                requests, self._pending = self._pending, []
            self._dispatch(requests)
        for thread in threads:
            thread.join()
        return results


def row_seconds(record: dict) -> float:
    """Controller time of one row: decisions, reset and the feedback it was handed."""
    decide = record.get("decide_s") or {"sum": 0.0}
    return float(decide["sum"] + record.get("reset_s", 0.0) + record.get("observe_s", 0.0))


def _violations(records: Sequence[dict]) -> np.ndarray:
    """Largest violation per record; an errored unit counts as holding (it is reported)."""
    return np.array(
        [0.0 if r.get("max_violation") is None else r["max_violation"] for r in records]
    )


def _slim(record: dict) -> dict:
    """A record without the hourly lists (records of search evaluations are not kept in full)."""
    return {k: v for k, v in record.items() if not k.endswith("_per_hour")}


class Study:
    """Margin searches and parts of the main run (shared by ``dryrun`` and ``main``)."""

    def __init__(
        self,
        ev: Evaluator,
        config: LearnedSetpointsConfig,
        task: dict[str, Any],
        *,
        margin_tolerance: float,
        limit_days: int | None = None,
    ) -> None:
        self.ev = ev
        self.config = config
        self.task = task
        control = config.load_following.headroom.control
        self.tol_violation = control.violation_tolerance_kg_per_kg
        self.tol_margin = margin_tolerance
        self.upper = control.margin_upper_kg_per_kg
        self.maximum = config.load_following.margin_maximum
        self.days = {
            name: (ids if limit_days is None else ids[:limit_days])
            for name, ids in task["day_sets"].items()
        }
        self.results: dict[str, dict[str, Any]] = {}

    def search(self, cell: Cell, day_set: str = "test") -> dict[str, Any]:
        """Margin of ``cell`` on a day set, see the module docstring."""
        ids = self.days[day_set]
        first = _violations(self.ev.run(cell, 0.0, ids, day_set))
        order = np.argsort(-first, kind="stable")
        n_critical = min(self.config.load_following.critical_days, len(ids))
        critical = sorted(int(i) for i in order[:n_critical])

        def violations(m: float, idx: Sequence[int]) -> np.ndarray:
            return _violations(self.ev.run(cell, m, [ids[i] for i in idx], day_set))

        search: MarginSearch
        search, final = bisect_margin_over_scenarios(
            violations,
            len(ids),
            tolerance_violation=self.tol_violation,
            lower=0.0,
            upper=self.upper,
            tolerance=self.tol_margin,
            max_upper=self.maximum,
            initial_critical=critical,
        )
        records = self.ev.run(cell, search.margin, ids, day_set)
        errored = sum(1 for r in records if r.get("error"))
        admissible = bool(search.converged and np.all(final <= self.tol_violation))
        log(
            f"margin {cell.key} [{day_set}]: "
            + (f"{search.margin:.5f}" if admissible else "NOT ADMISSIBLE")
            + f" ({len(search.evaluations)} evaluations)"
        )
        return {
            "cell": cell.key,
            "day_set": day_set,
            "admissible": admissible,
            "margin": search.margin if admissible else None,
            "largest_margin_tried": float(max(m for m, _ in search.evaluations)),
            "largest_violation_at_last_margin": float(np.max(final)),
            "search": {
                "bracket": list(search.bracket),
                "converged": search.converged,
                "evaluations": [[m, ok] for m, ok in search.evaluations],
                "critical_seed": [ids[i] for i in critical],
            },
            "errored_units": errored,
            "records": records if admissible else [_slim(r) for r in records],
        }

    def fixed(self, cell: Cell, margin: float, day_set: str = "test") -> list[dict]:
        """Records of ``cell`` at a given margin."""
        return self.ev.run(cell, margin, self.days[day_set], day_set)

    # ---------------------------------------------------------------------- parts

    def part_searches(self, cells: Sequence[Cell]) -> dict[str, Any]:
        """Margin searches of ``cells`` on the test days, side by side."""
        found = self.ev.run_all([lambda c=c: self.search(c) for c in cells])  # type: ignore[misc]
        out = {}
        for cell, block in zip(cells, found, strict=True):
            if block is None:
                block = {"cell": cell.key, "admissible": False, "margin": None, "job_error": True}
            out[cell.key] = block
            self.results[cell.key] = block
        return out

    def part_reference(self, cells: Sequence[Cell]) -> dict[str, Any]:
        """Ladder at the margins of the load-following study, and the searches of the reference cells."""
        margins = self.task["margins"]
        jobs: list[Callable[[], Any]] = [
            lambda: self.fixed(Cell("C0"), margins["C0"]),
            lambda: self.fixed(Cell("C2"), margins["C2"]),
        ]
        jobs += [lambda c=c: self.search(c) for c in cells]  # type: ignore[misc]
        done = self.ev.run_all(jobs)
        out: dict[str, Any] = {
            "C0": {"margin": margins["C0"], "records": done[0]},
            "C2": {"margin": margins["C2"], "records": done[1]},
        }
        for cell, block in zip(cells, done[2:], strict=True):
            out[cell.key] = block
            self.results[cell.key] = block
        # control: this runner gives the C2 records of the load-following artifact
        old = self.task["c2_records"]
        worst = 0.0
        compared = 0
        for record in done[1] or []:
            ref = old.get(str(record["day_of_year"]))
            if ref is None or record.get("error"):
                continue
            compared += 1
            for key in ("cost_e", "product_dry_kg", "max_violation"):
                worst = max(worst, abs(record[key] - ref[key]))
        out["control_c2_against_load_following"] = {
            "days_compared": compared,
            "max_abs_difference": worst,
        }
        return out

    def part_twin_curve(self) -> dict[str, Any]:
        """The twin over the margin grid (margin shares of the cells below the largest budget)."""
        twin = Cell(self.task["twin"])
        grid = list(self.config.twin_margin_grid)
        done = self.ev.run_all([lambda m=m: self.fixed(twin, m) for m in grid])  # type: ignore[misc]
        return {
            "twin": twin.key,
            "margins": grid,
            "records": [[_slim(r) for r in recs or []] for recs in done],
        }

    def part_decomposition(self) -> dict[str, Any]:
        """The twin at the margin of every admissible cell of the largest budget."""
        twin = Cell(self.task["twin"])
        cells = [
            (key, block["margin"])
            for key, block in self.results.items()
            if block.get("admissible")
            and (
                largest_budget(Cell.from_key(key), self.config)
                or Cell.from_key(key).family == "reference"
            )
        ]
        done = self.ev.run_all([lambda m=m: self.fixed(twin, m) for _, m in cells])  # type: ignore[misc]
        return {
            key: {"margin": m, "records": [_slim(r) for r in recs or []]}
            for (key, m), recs in zip(cells, done, strict=True)
        }

    def part_deployable(self, cells: Sequence[Cell]) -> dict[str, Any]:
        """Margin from the tuning days (training months), then the test days at that margin."""
        cells = [c for c in cells if largest_budget(c, self.config) and c.level != "lab"]

        def job(cell: Cell) -> dict[str, Any]:
            block = self.search(cell, "tuning")
            block["records"] = [_slim(r) for r in block["records"]]
            if block["admissible"]:
                block["test_records"] = [
                    _slim(r) for r in self.fixed(cell, block["margin"], "test")
                ]
            return block

        done = self.ev.run_all([lambda c=c: job(c) for c in cells])  # type: ignore[misc]
        return {cell.key: block for cell, block in zip(cells, done, strict=True)}


# ------------------------------------------------------------------------------ stage 0 driver


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _unit_file(directory: Path, name: str) -> Path:
    return directory / "units" / f"{name}.json"


def run_stage0(
    config: LearnedSetpointsConfig,
    task: dict[str, Any],
    directory: Path,
    config_path: str,
    *,
    workers: int,
    smoke: bool,
    day_hours: int | None,
    resume: bool,
) -> dict[str, Any]:
    """All units of stage 0, see the module docstring; returns the stage-0 record."""
    if directory.exists() and not resume and any(directory.iterdir()):
        raise SystemExit(f"{directory} exists -- pass --resume to continue it, or use a new name")
    (directory / "units").mkdir(parents=True, exist_ok=True)
    band = task["band"]
    units: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    def collect(futures: dict[Future, str]) -> None:
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except BaseException:  # a failed unit is logged, the others go on
                failures.append({"unit": name, "error": traceback.format_exc()})
                log(f"ERROR stage-0 unit {name}: {traceback.format_exc()[-400:]}")
                continue
            _unit_file(directory, name).write_text(
                json.dumps(result, default=_json_default), encoding="utf-8"
            )
            units[name] = result
            log(f"stage-0 unit {name} done")

    def cached(name: str) -> bool:
        path = _unit_file(directory, name)
        if path.exists():
            units[name] = json.loads(path.read_text(encoding="utf-8"))
            return True
        return False

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    seeds = list(config.seeds)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future, str] = {}
        if not cached("table_truth"):
            futures[pool.submit(truth_table_unit, str(directory), config_path, band, smoke)] = (
                "table_truth"
            )
        # largest models first, so that the long units do not end up at the tail of the queue
        for n_train in sorted(config.n_train, reverse=True):
            for seed in seeds:
                name = f"model_n{n_train}_s{seed}"
                if not cached(name):
                    futures[
                        pool.submit(
                            model_unit, seed, n_train, smoke, str(directory), config_path, band
                        )
                    ] = name
        collect(futures)
    # labels: the expert (twin) at the label margins on the label days
    label_path = directory / "labels.npz"
    label_days = task["day_sets"]["label"]
    if not label_path.exists():
        log("stage 0: labels")
        twin = Cell(task["twin"])
        rows = [(twin.key, m, d) for d in label_days for m in config.policy.label_margins]
        parts: list[LabelSet] = []
        label_info: list[dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(config_path, day_hours, str(directory), band),
        ) as pool:
            n_tasks = min(workers, len(rows))
            chunks = [rows[t::n_tasks] for t in range(n_tasks)]
            label_futures = {
                pool.submit(_task, chunk, task["margins"]["C0"], True): chunk for chunk in chunks
            }
            for future in as_completed(label_futures):
                out = future.result()
                if out["error"] is not None:
                    raise SystemExit(f"label batch failed:\n{out['error']}")
                for (_, margin, day), record in zip(
                    label_futures[future], out["records"], strict=True
                ):
                    if record.get("error"):
                        raise SystemExit(
                            f"label unit day {day} margin {margin} failed:\n{record['error']}"
                        )
                    decisions = record.pop("decisions")
                    n = len(decisions["features"])
                    parts.append(
                        LabelSet(
                            np.array(decisions["features"]),
                            np.array(decisions["commands"]),
                            np.full(n, day),
                            np.full(n, margin),
                            np.arange(n),
                        )
                    )
                    label_info.append(
                        {
                            "day_of_year": day,
                            "margin": margin,
                            "max_violation": record["max_violation"],
                            "map_failures": record["solver"]["failures"],
                            "cost_e": record["cost_e"],
                        }
                    )
        order = sorted(
            range(len(parts)), key=lambda i: (label_info[i]["day_of_year"], label_info[i]["margin"])
        )
        LabelSet.concatenate([parts[i] for i in order]).save(label_path)
        _unit_file(directory, "labels").write_text(
            json.dumps([label_info[i] for i in order]), encoding="utf-8"
        )
    labels_record = json.loads(_unit_file(directory, "labels").read_text(encoding="utf-8"))
    # policies
    spread = task["label_spread"]
    label_sets: dict[str, dict[str, list[int]]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for seed in config.policy.seeds:
            nested = nested_label_days(
                label_days, spread, config.policy.budgets_days, config.policy.strata, seed
            )
            label_sets[str(seed)] = {str(k): v for k, v in nested.items()}
            for budget, days in nested.items():
                for arm in POLICY_ARMS:
                    name = Path(Assets.policy_name(arm, budget, seed)).stem
                    if not cached(name):
                        futures[
                            pool.submit(
                                policy_unit,
                                arm,
                                budget,
                                seed,
                                days,
                                str(directory),
                                config_path,
                                smoke,
                            )
                        ] = name
        collect(futures)
    return {
        "directory": (
            str(directory.relative_to(REPO_ROOT)).replace("\\", "/")
            if directory.is_relative_to(REPO_ROOT)
            else str(directory)
        ),
        "smoke": smoke,
        "units": units,
        "failures": failures,
        "labels": {"file": "labels.npz", "sha256": sha256(label_path), "rows": labels_record},
        "label_days_per_seed_and_budget": label_sets,
    }


# ------------------------------------------------------------------------------ probe


def bisection_evaluations(config: LearnedSetpointsConfig) -> int:
    """Evaluations of a bisection on ``[0, upper]`` to the margin tolerance."""
    control = config.load_following.headroom.control
    return 1 + math.ceil(
        math.log2(control.margin_upper_kg_per_kg / control.margin_tolerance_kg_per_kg)
    )


def extrapolate_hours(
    config: LearnedSetpointsConfig,
    cells: dict[str, list[Cell]],
    seconds_per_row: dict[str, float],
    batch_seconds: dict[str, float],
    n_days: int,
    *,
    workers: int,
) -> dict[str, Any]:
    """Wall-clock estimate of the main run (rule fixed beforehand).

    Rows of a margin search: one pass over all days at margin 0, ``S`` evaluations on the
    critical subset, two verification passes over all days. Per row the controller time of its
    kind of cell (``seconds_per_row``, measured in the probe under parallel load); per round and
    worker one batch whose truth simulation costs ``batch_seconds`` (with and without pre-roll).
    Rounds: ``S + 3`` per group of searches plus one per fixed pass. Decomposition: one twin pass
    per cell of the largest budget; twin curve: one pass per grid margin; deployable: a search
    on the tuning days plus one test pass per cell of the largest budget (level lab excluded).
    Safety factor of the config.
    """
    s = bisection_evaluations(config)
    k = config.load_following.critical_days
    search_rows = n_days + s * k + 2 * n_days
    rounds = s + 3

    def kind(cell: Cell) -> str:
        return f"{cell.arm}|{cell.level}" if cell.level != "none" else cell.arm

    cpu = 0.0
    rows_total = 0
    per_part: dict[str, float] = {}
    for part, part_cells in cells.items():
        part_cpu = sum(search_rows * seconds_per_row[kind(c)] for c in part_cells)
        groups = {c.level != "none" for c in part_cells}
        part_cpu += sum(
            rounds * workers * batch_seconds["pre_roll" if g else "plain"] for g in groups
        )
        per_part[part] = part_cpu
        cpu += part_cpu
        rows_total += search_rows * len(part_cells)
    twin = seconds_per_row["C2"]
    large = [c for cs in cells.values() for c in cs if largest_budget(c, config)]
    extra = {
        "reference_fixed": 2 * n_days * twin + workers * batch_seconds["plain"],
        "twin_curve": len(config.twin_margin_grid) * n_days * twin
        + workers * batch_seconds["plain"],
        "decomposition": len(large) * n_days * twin + workers * batch_seconds["plain"],
        "deployable": sum(
            (search_rows + n_days) * seconds_per_row[kind(c)] for c in large if c.level != "lab"
        )
        + (rounds + 1) * workers * (batch_seconds["plain"] + batch_seconds["pre_roll"]),
    }
    cpu += sum(extra.values())
    factor = config.extrapolation_safety / (3600.0 * workers)
    return {
        "bisection_evaluations": s,
        "rows_per_search": search_rows,
        "search_rows_total": rows_total,
        "cpu_h": cpu / 3600.0,
        "cpu_h_per_part": {
            **{p: v / 3600.0 for p, v in per_part.items()},
            **{p: v / 3600.0 for p, v in extra.items()},
        },
        "wall_h": factor * cpu,
    }


def apply_budget_rule(
    config: LearnedSetpointsConfig, estimate: Callable[[Sequence[str]], float], stage0_h: float
) -> dict[str, Any]:
    """Decision 8: budget, extension and cuts from the extrapolated wall-clock time.

    At most 30 h: no change. Above 30 h and at most 48 h: uncut with the budget 48 h. Above 48 h:
    cuts in the fixed order, only as far as needed, until the estimate is at most 48 h.
    ``estimate(cuts)`` is the wall-clock estimate of the main run in hours; the time stage 0
    took is added.
    """
    base, extended = config.budget_wall_clock_h, config.extended_wall_clock_h
    uncut = estimate(()) + stage0_h
    history = [{"cuts": [], "wall_h": uncut}]
    if uncut <= base:
        return {
            "budget_h": base,
            "cuts": [],
            "wall_h": uncut,
            "reason": "within the budget",
            "history": history,
        }
    if uncut <= extended:
        return {
            "budget_h": extended,
            "cuts": [],
            "wall_h": uncut,
            "reason": "above the budget, within the extension; uncut",
            "history": history,
        }
    cuts: list[str] = []
    wall = uncut
    for cut in config.cut_order:
        cuts.append(cut)
        wall = estimate(tuple(cuts)) + stage0_h
        history.append({"cuts": list(cuts), "wall_h": wall})
        if wall <= extended:
            break
    reason = "above the extension; cut in the fixed order"
    if wall > extended:
        reason += "; still above the extension after all cuts"
    return {
        "budget_h": extended,
        "cuts": cuts,
        "wall_h": wall,
        "reason": reason,
        "history": history,
    }


def stage_probe(
    ev: Evaluator,
    config: LearnedSetpointsConfig,
    task: dict[str, Any],
    workers: int,
    stage0_h: float,
) -> dict[str, Any]:
    """Time and memory of full-length days of every kind of cell under parallel load."""
    label = task["day_sets"]["label"]
    spread = np.array(task["label_spread"])
    order = np.argsort(spread, kind="stable")
    picks = np.linspace(0, len(order) - 1, config.probe_days).round().astype(int)
    ids = sorted({label[int(order[i])] for i in picks})
    n_max, k_max = max(config.n_train), max(config.policy.budgets_days)
    kinds = [
        Cell("C2"), Cell("Cfix"), Cell("C3-15"), Cell("M-tab"),
        Cell("M-quad", "none", n_max, 1), Cell("M-gru", "none", n_max, 1),
        Cell("M-quad", "fb", n_max, 1), Cell("M-gru", "fb", n_max, 1), Cell("M-gru", "lab", n_max, 1),
        Cell("D-gru", "none", max(config.gru_mpc_n_train), 1),
        Cell("P-lin", "none", k_max, 1), Cell("P-mlp", "none", k_max, 1), Cell("P-gru", "none", k_max, 1),
    ]  # fmt: skip
    done = ev.run_all([lambda c=c: ev.run(c, PROBE_MARGIN, ids, "label") for c in kinds])  # type: ignore[misc]
    seconds_per_row: dict[str, float] = {}
    units = []
    for cell, records in zip(kinds, done, strict=True):
        name = f"{cell.arm}|{cell.level}" if cell.level != "none" else cell.arm
        good = [r for r in records or [] if not r.get("error")]
        if not good:
            raise SystemExit(f"probe: every unit of {cell.key} failed")
        per_row = float(np.mean([row_seconds(r) for r in good]))
        seconds_per_row[name] = per_row
        units.append(
            {
                "cell": cell.key,
                "seconds_per_row": per_row,
                "max_violation": [r["max_violation"] for r in good],
                "solver": good[0]["solver"],
            }
        )
        log(f"probe {cell.key}: {per_row:.1f} s controller time per day")
    seconds_per_row["C0"] = seconds_per_row["C2"]
    plain = [t["seconds"] - t["decide_s"] for t in ev.task_seconds if not t["pre_roll"]]
    pre = [t["seconds"] - t["decide_s"] for t in ev.task_seconds if t["pre_roll"]]
    batch_seconds = {"plain": float(np.max(plain)), "pre_roll": float(np.max(pre))}
    n_days = len(task["day_sets"]["test"])

    def estimate(cuts: Sequence[str]) -> float:
        cells = study_cells(config, cuts)
        return float(
            extrapolate_hours(
                config, cells, seconds_per_row, batch_seconds, n_days, workers=workers
            )["wall_h"]
        )

    rule = apply_budget_rule(config, estimate, stage0_h)
    cells = study_cells(config, rule["cuts"])
    return {
        "probe_days": ids,
        "margin": PROBE_MARGIN,
        "seconds_per_row": seconds_per_row,
        "batch_seconds": batch_seconds,
        "batches": ev.task_seconds,
        "units": units,
        "stage0_h": stage0_h,
        "estimate": extrapolate_hours(
            config, cells, seconds_per_row, batch_seconds, n_days, workers=workers
        ),
        "budget_rule": rule,
        "cells_per_part": {part: len(cs) for part, cs in cells.items()},
    }


# ------------------------------------------------------------------------------ entry point


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)  # type: ignore[no-any-return]


def check_assets(stage0: dict[str, Any]) -> None:
    """Abort if a file of stage 0 is missing or differs from the hash in its record."""
    directory = REPO_ROOT / stage0["directory"]
    expected = {stage0["labels"]["file"]: stage0["labels"]["sha256"]}
    for unit in stage0["units"].values():
        for block in (unit, unit.get("gru", {}), unit.get("quad", {})):
            for name_key, hash_key in (
                ("table", "table_sha256"),
                ("checkpoint", "checkpoint_sha256"),
                ("file", "sha256"),
            ):
                if name_key in block:
                    expected[block[name_key]] = block[hash_key]
    for name, digest in expected.items():
        path = directory / name
        if not path.exists() or sha256(path) != digest:
            raise SystemExit(f"{path} is missing or differs from the stage-0 record")
    log(f"stage-0 files checked: {len(expected)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--name", required=True, help="base name of the artifacts")
    parser.add_argument("--workers", type=int, default=8, help="at most the physical cores")
    parser.add_argument("--parts", default=",".join(PARTS))
    parser.add_argument("--day-hours", type=int, default=1, help="length of a dry-run day")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit-days", type=int, help="smoke runs only: first N days of each day set"
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
    config = load_learned_setpoints_config(args.config, day_hours=day_hours)
    lf = config.load_following
    market = build_market(lf, REPO_ROOT)
    out_dir = Path(args.out_dir).resolve()
    smoke = out_dir != RUNS_DIR.resolve()
    suffix = "" if args.stage == "main" else f"_{args.stage}"
    output = out_dir / f"{args.name}{suffix}.json.gz"
    cache = out_dir / f"_{args.name}{suffix}_cache.jsonl"
    stage0_path = out_dir / f"{args.name}_stage0.json.gz"
    probe_path = out_dir / f"{args.name}_probe.json.gz"
    asset_dir = ASSET_ROOT / (f"{args.name}_dryrun" if dry else args.name)
    if smoke:
        asset_dir = out_dir / "assets" / asset_dir.name
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if cache.exists() and not args.resume:
        raise SystemExit(f"{cache} exists -- pass --resume to continue it, or use a new name")
    if smoke:
        log(f"smoke run into {out_dir}: no clean-tree requirement, not an artifact")
    else:
        require_clean_tree([output, cache])
    task = reference_task(config, market)
    payload: dict[str, Any] = {
        "provenance": provenance("not used", ignore_paths=[output, cache]),
        "study": "learned set points (docs/NN_learned_setpoints.md)",
        "stage": args.stage,
        "smoke": smoke,
        "study_config": config.raw,
        "study_fingerprint": config.fingerprint,
        "load_following_fingerprint": lf.fingerprint,
        "headroom_fingerprint": lf.headroom.fingerprint,
        "process_config_fingerprint": lf.process.fingerprint,
        "day_hours": lf.day.hours,
        "tail_hours": lf.day.tail_hours,
        "workers": args.workers,
        "task": task,
    }
    started = time.time()
    log(f"stage {args.stage} started, name {args.name}, workers {args.workers}")
    stage0_h = 0.0
    if args.stage in ("stage0", "dryrun"):
        payload["stage0"] = run_stage0(
            config, task, asset_dir, args.config,
            workers=args.workers, smoke=dry, day_hours=day_hours, resume=args.resume,
        )  # fmt: skip
        payload["stage0_runtime_s"] = time.time() - started
        stage0 = payload["stage0"]
    else:
        if not stage0_path.exists():
            raise SystemExit(f"{stage0_path} missing -- run --stage stage0 first")
        stage0_file = _load_json_gz(stage0_path)
        if stage0_file["study_fingerprint"] != config.fingerprint:
            raise SystemExit("stage-0 artifact was written with another configuration")
        if stage0_file["stage0"]["failures"]:
            raise SystemExit("stage 0 has failed units -- see its artifact")
        stage0 = stage0_file["stage0"]
        stage0_h = stage0_file["runtime_s"] / 3600.0
        payload["stage0_artifact"] = stage0_path.name
        check_assets(stage0)
    if args.stage != "stage0":
        header = {
            "commit": git("rev-parse", "HEAD"),
            "study_fingerprint": config.fingerprint,
            "process_fingerprint": lf.process.fingerprint,
            "stage": args.stage,
            "day_hours": lf.day.hours,
            "labels_sha256": stage0["labels"]["sha256"],
        }
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.config, day_hours, str(REPO_ROOT / stage0["directory"]), task["band"]),
        ) as pool:
            ev = Evaluator(pool, args.workers, cache, header, task["margins"]["C0"])
            if args.stage == "probe":
                payload["probe"] = stage_probe(ev, config, task, args.workers, stage0_h)
                log(
                    f"probe: estimate {payload['probe']['estimate']['wall_h']:.1f} h, rule {payload['probe']['budget_rule']}"
                )
            else:
                cuts: list[str] = []
                tolerance = DRY_MARGIN_TOLERANCE
                if not dry:
                    tolerance = lf.headroom.control.margin_tolerance_kg_per_kg
                    if not probe_path.exists():
                        raise SystemExit(f"{probe_path} missing -- run --stage probe first")
                    rule = _load_json_gz(probe_path)["probe"]["budget_rule"]
                    cuts = list(rule["cuts"])
                    payload["budget_rule"] = rule
                    log(f"budget rule of the probe: {rule['reason']}; cuts {cuts}")
                if args.limit_days is not None and not smoke:
                    raise SystemExit("--limit-days is for smoke runs (--out-dir) only")
                parts = [p for p in args.parts.split(",") if p]
                for part in parts:
                    if part not in PARTS:
                        raise SystemExit(f"unknown part {part!r}")
                cells = study_cells(config, cuts)
                study = Study(
                    ev, config, task, margin_tolerance=tolerance, limit_days=args.limit_days
                )
                payload.update(
                    {
                        "parts": parts,
                        "cuts": cuts,
                        "margin_tolerance": tolerance,
                        "violation_tolerance": study.tol_violation,
                        "cells": {part: [c.key for c in cs] for part, cs in cells.items()},
                        "day_price_spread": {
                            str(d): daily_spread(market.tariff, d)
                            for ids in study.days.values()
                            for d in ids
                        },
                        "reference_tariff": market.reference_tariff,
                    }
                )
                runtime: dict[str, float] = {}
                for part in PARTS:
                    if part not in parts:
                        continue
                    t0 = time.time()
                    if part == "reference":
                        payload[part] = study.part_reference(cells["reference"])
                    elif part in ("M", "D", "P"):
                        payload[part] = study.part_searches(cells[part])
                    elif part == "twin_curve":
                        payload[part] = study.part_twin_curve()
                    elif part == "decomposition":
                        payload[part] = study.part_decomposition()
                    else:
                        payload[part] = study.part_deployable(cells["M"] + cells["D"] + cells["P"])
                    runtime[part] = time.time() - t0
                    log(f"part {part} done in {runtime[part] / 60:.1f} min")
                payload["part_runtime_s"] = runtime
            payload["errors"] = ev.errors
            payload["units_run"] = ev.units_run
            payload["rounds"] = ev.rounds
            payload["peak_memory_mb_per_worker"] = sorted(ev.peak_mb.values())
            payload["batches"] = ev.task_seconds
    payload["runtime_s"] = time.time() - started
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None, default=_json_default)
    log(
        f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB) after {payload['runtime_s'] / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
