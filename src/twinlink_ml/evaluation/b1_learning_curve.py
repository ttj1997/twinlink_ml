r"""Learning curves of study B1: one arm (excitation or selection rule) and one master seed.

Arms: ``b0``-``b4`` the plans of ``excitation_plans`` (a pool of the largest N built once and
used as nested prefixes), ``b5`` active learning (ensemble disagreement), ``b5r`` random
selection from the same candidate pool as ``b5``. All arms count N = number of training runs;
there are no validation runs (fixed epoch budget, the same for every arm and model).

B5 / B5r
--------
The candidate pool of one seed holds ``per_plan`` candidates of each of B1-B4 (equal parts, no B0),
built from inputs only before any simulation. Both arms start from the same ``sizes[0]`` runs drawn
at random from the pool (same stream), and add runs round by round up to the next size. B5 trains an
ensemble on the current pool (member 0 is the model evaluated at that N, identical to the model of
the other arms, so the extra cost of B5 is the other members) and adds the candidates with the
largest disagreement; B5r draws uniformly from the remaining candidates.

Records
-------
One record per ``(arm, seed, N, model)``: fit and evaluation time, parameters, and for each test set
the flat metrics of the evaluator (run-median RMSE in physical units, NRMSE against measurement and
truth, at the configured horizons). Per ``(arm, seed, N)`` an ``information`` record holds the
model-free measures of the pool (spectral coverage at 1 / 10 / 25 % thresholds, FIR and ARX Fisher
information). B5 records carry the chosen candidates and their disagreement.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.splits import SplitName
from ..models.base import SequenceModel
from ..models.registry import build_factories
from ..simulation.config import ProcessConfig
from ..simulation.excitation import ExcitationSchedule
from ..simulation.excitation_information import (
    arx_fisher_information,
    fir_fisher_information,
    normalise_actions,
    spectral_coverage,
)
from ..simulation.excitation_plans import make_plan
from ..utils.seeding import SeedBundle
from .active_learning import ensemble_scores, select_top_k, train_ensemble
from .b1_testsets import HoldoutSet, assert_disjoint
from .evaluator import RolloutEvaluator
from .experiments import DataSpec
from .growing_pool import (
    GrowingPool,
    prepare_process_config,
    projection_counts,
    window_dataset,
)

__all__ = [
    "ARMS",
    "CurveConfig",
    "candidate_pool",
    "run_arm",
]

ARMS: tuple[str, ...] = ("b0", "b1", "b2", "b3", "b4", "b5", "b5r")
PLAN_ARMS: tuple[str, ...] = ("b0", "b1", "b2", "b3", "b4")
POOL_PLANS: tuple[str, ...] = ("b1", "b2", "b3", "b4")
KEPT_METRIC_PREFIXES = ("rmse_run_median_physical/", "nrmse/", "nrmse_truth/")


@dataclass(frozen=True, slots=True)
class CurveConfig:
    """Settings of a learning curve."""

    sizes: tuple[int, ...] = (2, 4, 8, 16, 32)
    models: tuple[str, ...] = ("persistence", "arx", "lstm")
    network_label: str = "lstm"
    """The model that is also the ensemble member of B5."""
    n_members: int = 5
    per_plan: int = 32
    horizons: tuple[int, ...] = (1, 24)

    def __post_init__(self) -> None:
        if tuple(sorted(self.sizes)) != self.sizes or len(set(self.sizes)) != len(self.sizes):
            raise ValueError("sizes must be strictly increasing")
        if self.network_label not in self.models:
            raise ValueError("network_label must be one of the models")


def _derive(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def candidate_pool(
    process_config: ProcessConfig,
    spec: DataSpec,
    data_seed: int,
    *,
    per_plan: int,
) -> ExcitationSchedule:
    """``per_plan`` candidates of each of B1-B4 (inputs only), one plan seed each."""
    prepared = prepare_process_config(process_config, spec)
    parts: list[ExcitationSchedule] = []
    for name in POOL_PLANS:
        plan = make_plan(
            name,
            prepared,
            sample_time_s=spec.sample_time_s,
            run_duration_s=spec.run_duration_s,
            solid_fraction_process=spec.solid_fraction_process,
            gas_velocity_cap=spec.gas_velocity_cap,
        )
        parts.append(plan.build(per_plan, seed=_derive(data_seed, "pool", name)))
    return ExcitationSchedule(
        run_id=tuple(r for p in parts for r in p.run_id),
        actions=np.concatenate([p.actions for p in parts]),
        disturbances=np.concatenate([p.disturbances for p in parts]),
        initial_states=np.concatenate([p.initial_states for p in parts]),  # type: ignore[misc]
        signal_kinds=tuple(k for p in parts for k in p.signal_kinds),
        metadata={**parts[0].metadata, "design_name": "pool", "plans": list(POOL_PLANS)},
    )


def _information(
    pool: GrowingPool,
    prepared: ProcessConfig,
    spec: DataSpec,
    output_columns: Sequence[int],
    plan_timings: Mapping[str, float],
) -> dict[str, float]:
    trajectory = pool.trajectory()
    bounds = prepared.actions.bounds
    lower = np.array([b.minimum for b in bounds])
    span = np.array([b.span for b in bounds])
    normalised = normalise_actions(trajectory.actions, lower, span)
    outputs = trajectory.outputs[..., list(output_columns)]
    standardised = (outputs - outputs.mean(axis=(0, 1))) / np.maximum(
        outputs.std(axis=(0, 1)), 1e-12
    )
    result: dict[str, float] = {}
    for fraction in (0.01, 0.10, 0.25):
        report = spectral_coverage(
            normalised,
            spec.sample_time_s,
            band_low_hz=plan_timings["band_low_hz"],
            band_high_hz=plan_timings["band_high_hz"],
            threshold_fraction=fraction,
        )
        result[f"spectral_coverage_{round(100 * fraction)}pct"] = report.coverage
    result["spectral_flatness"] = report.flatness
    result["max_abs_cross_correlation"] = report.max_abs_cross_correlation
    fir = fir_fisher_information(normalised)
    arx = arx_fisher_information(normalised, standardised)
    result.update(
        fir_logdet_per_parameter=fir.logdet_per_parameter,
        fir_lambda_min=fir.lambda_min,
        fir_condition=fir.condition,
        arx_logdet_per_parameter=arx.logdet_per_parameter,
        arx_lambda_min=arx.lambda_min,
        arx_condition=arx.condition,
    )
    return result


def _plan_timings(prepared: ProcessConfig, spec: DataSpec) -> dict[str, float]:
    plan = make_plan(
        "b2",
        prepared,
        sample_time_s=spec.sample_time_s,
        run_duration_s=spec.run_duration_s,
        solid_fraction_process=spec.solid_fraction_process,
        gas_velocity_cap=spec.gas_velocity_cap,
    )
    return {
        "band_low_hz": plan.timings.band_low_hz,
        "band_high_hz": plan.timings.band_high_hz,
    }


def run_arm(
    arm: str,
    master_seed: int,
    process_config: ProcessConfig,
    spec: DataSpec,
    holdouts: Sequence[HoldoutSet],
    curve: CurveConfig,
    *,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Learning curve of one arm and one master seed; every record also goes to ``on_record``."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}, known: {ARMS}")
    seeds = SeedBundle.from_master(master_seed)
    prepared = prepare_process_config(process_config, spec)
    factories = build_factories(curve.models)
    evaluator = RolloutEvaluator(horizons=curve.horizons)
    plan_timings = _plan_timings(prepared, spec)
    output_names = None
    records: list[dict[str, Any]] = []

    def emit(record: dict[str, Any]) -> None:
        records.append(record)
        if on_record is not None:
            on_record(record)

    max_n = curve.sizes[-1]
    adaptive = arm in ("b5", "b5r")
    if adaptive:
        candidates = candidate_pool(process_config, spec, seeds.data_seed, per_plan=curve.per_plan)
        rng_start = np.random.default_rng(_derive(seeds.data_seed, "start"))
        rng_select = np.random.default_rng(_derive(seeds.data_seed, "random-select"))
        start = [int(i) for i in rng_start.choice(candidates.n_runs, curve.sizes[0], replace=False)]
    else:
        plan = make_plan(
            arm,
            prepared,
            sample_time_s=spec.sample_time_s,
            run_duration_s=spec.run_duration_s,
            solid_fraction_process=spec.solid_fraction_process,
            gas_velocity_cap=spec.gas_velocity_cap,
        )
        candidates = plan.build(max_n, seed=seeds.data_seed)
        start = list(range(curve.sizes[0]))
    pool = GrowingPool(process_config, spec, candidates, noise_seed=seeds.noise_seed)
    pool.add(start)

    for position, n_train in enumerate(curve.sizes):
        base = {
            "arm": arm,
            "master_seed": master_seed,
            "n_train": n_train,
            "data/new_study_version": spec.new_study_version,
            "data/gas_velocity_cap": spec.gas_velocity_cap,
        }
        if len(pool.selected) != n_train:
            raise RuntimeError("pool size does not match the learning-curve point")
        trajectory = pool.trajectory()
        assert_disjoint(trajectory, holdouts)
        if output_names is None:
            output_names = tuple(trajectory.metadata["output_names"])
        columns = [output_names.index(name) for name in spec.window.target_names]
        train = window_dataset(trajectory, spec.window, role=SplitName.TRAIN)
        tests = {
            h.name: window_dataset(
                h.trajectory, spec.window, role=SplitName.TEST, scalers=train.scalers
            )
            for h in holdouts
        }
        emit(
            {
                **base,
                "type": "information",
                "n_train_windows": train.n_windows,
                "selected": list(pool.selected),
                "projection": projection_counts(trajectory),
                "information": _information(pool, prepared, spec, columns, plan_timings),
            }
        )
        ensemble: list[SequenceModel] = []
        for label, factory in factories.items():
            fit_start = time.perf_counter()
            if adaptive and arm == "b5" and label == curve.network_label and n_train < max_n:
                ensemble = train_ensemble(
                    factory,  # type: ignore[arg-type]
                    train,
                    base_seed=seeds.model_seed,
                    n_members=curve.n_members,
                )
                model = ensemble[0]
            else:
                model = factory()  # type: ignore[operator]
                model.fit(train, seed=seeds.model_seed, validation=None)
            fit_s = time.perf_counter() - fit_start
            eval_start = time.perf_counter()
            metrics = {
                name: {
                    key: value
                    for key, value in evaluator.evaluate(model, dataset).as_flat_metrics().items()
                    if key.startswith(KEPT_METRIC_PREFIXES)
                }
                for name, dataset in tests.items()
            }
            emit(
                {
                    **base,
                    "type": "model",
                    "model": label,
                    "fit_s": fit_s,
                    "evaluate_s": time.perf_counter() - eval_start,
                    "n_parameters": model.n_parameters,
                    "metrics": metrics,
                }
            )
        if position + 1 == len(curve.sizes):
            break
        k = curve.sizes[position + 1] - n_train
        remaining = list(pool.remaining) if adaptive else list(range(n_train, n_train + k))
        if not adaptive:
            chosen, info = remaining, {}
        elif arm == "b5r":
            chosen = [int(i) for i in rng_select.choice(remaining, k, replace=False)]
            info = {}
        else:
            score_start = time.perf_counter()
            scores = ensemble_scores(ensemble, train, prepared, pool.candidate_inputs(remaining))
            chosen = select_top_k(scores, remaining, k)
            info = {
                "scores_chosen": [float(scores[remaining.index(i)]) for i in chosen],
                "score_median_all": float(np.median(scores)),
                "scoring_s": time.perf_counter() - score_start,
                "chosen_plans": [candidates.run_id[i].split("-")[0] for i in chosen],
            }
        if info:
            emit({**base, "type": "selection", "chosen": chosen, **info})
        pool.add(chosen)
    return records
