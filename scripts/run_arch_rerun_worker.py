r"""One job of the re-run of the architecture comparison (plan: ``docs/15_plan_rerun_architecture_comparison.md``).

Protocol of ``run_arch_learning_curve_worker.py`` (Befunde 51/58) unchanged -- same excited spec, same
constant-run holdout (seed 999), same hyperparameters, ``max_epochs`` 200, one model seed -- with three arms:

* ``A0``  legacy data spec (future inputs realised, solid fraction diurnal): same specification on the
          new machine/environment; platform and library effect against the old records.
* ``A1``  legacy trajectories, ``future_actions="planned"``: D1 alone (review D1).
* ``N``   ``DataSpec.new_study()``: planned future inputs and an own solid-fraction process (D1 + E3).

Cross-evaluation: every trained model is evaluated on both window variants (future inputs ``realised`` and
``planned``) and, for arm N, on both pools (own = new-study pool, ``legacy`` = diurnal pool of the same seed).
For A0/A1 the own pool IS the legacy pool, so only the two window variants exist. Records carry
``eval/pool``, ``eval/window`` and ``eval/primary`` (own pool, own window = the arm's specification; the
exhaust reconstruction metrics of the old worker are computed for the primary variant only).

A job = one arm, one seed, a subset of methods (``--architectures``, ``--skip-baselines``) and budgets
(``--sizes``). The old worker stays unchanged for reproduction.

    tools/run_here.sh scripts/run_arch_rerun_worker.py --arm N --seed 1 --sizes 8 \
        --output artifacts/runs/_rerun_pilot/N_seed1.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_arch_learning_curve_worker as old_worker

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import (
    NEW_STUDY_OPTIONS,
    DataSpec,
    ValidationPolicy,
    build_datasets,
    generate_data,
)
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.evaluation.exhaust_reconstruction import (
    evaluate_exhaust_reconstruction,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.checkpoints import checkpoint_path, save_checkpoint
from twinlink_ml.utils.tracking import (
    JsonlExperimentLogger,
    RunRecord,
    environment_fingerprint,
)

if TYPE_CHECKING:
    from twinlink_ml.evaluation.experiments import GeneratedData

ARMS = ("A0", "A1", "N")
WINDOWS = ("realised", "planned")
HOLDOUT_SEED = old_worker.HOLDOUT_SEED
SMOKE_SIZE = 4


def legacy_spec(*, smoke: bool = False) -> DataSpec:
    """The excited spec of the old worker (Befunde 51/58), or a tiny one for the smoke test."""
    base = old_worker._excited_spec()
    if not smoke:
        return base
    return dataclasses.replace(
        base,
        run_duration_s=6000.0,
        n_train_max=SMOKE_SIZE,
        n_val=2,
        n_test=2,
        history_length=8,
        horizon=6,
        stride=4,
    )


def arm_spec(arm: str, *, smoke: bool = False) -> DataSpec:
    """Data specification of an arm."""
    legacy = legacy_spec(smoke=smoke)
    if arm == "A0":
        return legacy
    if arm == "A1":
        return dataclasses.replace(legacy, future_actions="planned")
    if arm == "N":
        # DataSpec.new_study fixes the preset options (NEW_STUDY_OPTIONS, version 2: planned future
        # inputs, own solid-fraction process, per-run gas-velocity cap); every other field comes from
        # the legacy spec
        fields = {
            f.name: getattr(legacy, f.name)
            for f in dataclasses.fields(legacy)
            if f.name not in NEW_STUDY_OPTIONS
        }
        return DataSpec.new_study(**fields)
    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


def holdout_spec(excited: DataSpec) -> DataSpec:
    return old_worker._holdout_spec(excited)


def evaluation_variants(arm: str) -> list[tuple[str, str, bool]]:
    """``(pool, window, primary)`` triples; pool ``own`` or ``legacy``, window ``realised``/``planned``."""
    own_window = arm_spec(arm).future_actions
    pools = ["own", "legacy"] if arm == "N" else ["own"]
    return [
        (pool, window, pool == "own" and window == own_window)
        for pool in pools
        for window in WINDOWS
    ]


def expected_record_count(arm: str, sizes: list[int], n_methods: int) -> int:
    """Records of a job: methods x budgets x variants x 2 test sets."""
    return n_methods * len(sizes) * len(evaluation_variants(arm)) * 2


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=old_worker.CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(old_worker.BUDGET_POINTS))
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="*",
        choices=old_worker.ARCHITECTURE_LABELS,
        default=list(old_worker.ARCHITECTURE_LABELS),
    )
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--net-max-epochs", type=int, default=200)
    parser.add_argument("--smoke", action="store_true", help="tiny spec for the tests")
    return parser.parse_args(argv)


def _generate(config: Any, spec: DataSpec, seed: int) -> GeneratedData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return generate_data(config, spec, master_seed=seed)


def run(arguments: argparse.Namespace) -> int:
    """Run one job; returns the number of records written."""
    arm, seed = arguments.arm, arguments.seed
    if arguments.architectures:
        import torch

        torch.set_num_threads(arguments.num_threads)
    checkpoint_dir = arguments.checkpoint_dir or Path("artifacts") / "checkpoints" / "rerun" / arm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )

    own_spec = arm_spec(arm, smoke=arguments.smoke)
    own_holdout_spec = holdout_spec(own_spec)
    horizon = own_spec.horizon
    evaluator = RolloutEvaluator(horizons=(1, horizon))
    logger = JsonlExperimentLogger(arguments.output)

    own = _generate(config, own_spec, seed)
    own_holdout = _generate(config, own_holdout_spec, HOLDOUT_SEED)
    if not arguments.smoke and own.process_config.fingerprint != "bddca37f716f5cfc":
        raise RuntimeError(
            f"process/fingerprint {own.process_config.fingerprint!r}, expected 'bddca37f716f5cfc'"
        )
    pools: dict[str, tuple[GeneratedData, GeneratedData]] = {"own": (own, own_holdout)}
    if arm == "N":
        legacy = legacy_spec(smoke=arguments.smoke)
        pools["legacy"] = (
            _generate(config, legacy, seed),
            _generate(config, holdout_spec(legacy), HOLDOUT_SEED),
        )
    print(f"[{arm} seed={seed}] setup done, pools: {sorted(pools)}", flush=True)

    variants = evaluation_variants(arm)
    written = 0

    def log_model(
        n_train: int,
        n_val: int,
        label: str,
        model: Any,
        elapsed: float,
        train: TrajectoryWindowDataset,
        checkpoint_uid: str | None,
    ) -> None:
        nonlocal written
        for pool_name, window_name, primary in variants:
            pool_excited, pool_holdout = pools[pool_name]
            window = dataclasses.replace(own_spec.window, future_actions=window_name)
            test_sets = (
                (
                    "excited",
                    pool_excited,
                    TrajectoryWindowDataset.from_trajectory(
                        pool_excited.trajectory,
                        pool_excited.split.subset(n_train),
                        SplitName.TEST,
                        window,
                        scalers=train.scalers,
                    ),
                ),
                (
                    "constant_holdout",
                    pool_holdout,
                    TrajectoryWindowDataset.from_trajectory(
                        pool_holdout.trajectory,
                        pool_holdout.split,
                        SplitName.TEST,
                        window,
                        scalers=train.scalers,
                    ),
                ),
            )
            for test_name, source, dataset in test_sets:
                result = evaluator.evaluate(model, dataset)
                metrics = dict(result.as_flat_metrics())
                if primary:
                    exhaust = evaluate_exhaust_reconstruction(
                        model, source, dataset, horizons=(1, horizon)
                    )
                    metrics.update(exhaust.as_flat_metrics())
                params: dict[str, Any] = {
                    **own_spec.as_parameters(),
                    "rerun/arm": arm,
                    "eval/pool": pool_name,
                    "eval/window": window_name,
                    "eval/primary": primary,
                    "process/fingerprint": own.process_config.fingerprint,
                    "excitation/fingerprint": own.schedule.metadata.get("fingerprint", "unknown"),
                    "split/digest": own.split.digest,
                    "eval/pool_excitation_fingerprint": source.schedule.metadata.get(
                        "fingerprint", "unknown"
                    ),
                    **own.fingerprints,
                    **own.seeds.as_parameters(),
                    "test_set": test_name,
                    "model/label": label,
                    "n_train": n_train,
                    "n_val": n_val,
                    "n_budget": n_train + n_val,
                    "training/elapsed_s": elapsed,
                    **{f"model/{k}": v for k, v in model.describe().items()},
                    **{f"training/{k}": v for k, v in model.training_diagnostics().items()},
                    **environment_fingerprint(),
                    "master_seed": seed,
                    "holdout/seed": HOLDOUT_SEED,
                    "holdout/n_runs": len(own_holdout.split.test),
                    # guard of the new-study preset: how often the projection acted (expected 0)
                    **{f"pool/{k}": v for k, v in own.projection_guard.items()},
                    **{f"holdout_pool/{k}": v for k, v in own_holdout.projection_guard.items()},
                }
                if checkpoint_uid is not None:
                    params["checkpoint/path"] = str(checkpoint_path(checkpoint_dir, checkpoint_uid))
                logger.log(
                    RunRecord(
                        group="arch_rerun",
                        params=params,
                        metrics=metrics,
                        tags={
                            "model": label,
                            "group": "arch_rerun",
                            "test_set": test_name,
                            "n_train": str(n_train),
                            "arm": arm,
                        },
                    )
                )
                written += 1

    for n_train in arguments.sizes:
        train, validation, _ = build_datasets(own, n_train, policy=ValidationPolicy.FIXED_POOL)
        n_val = 0 if validation is None else len(own.split.val)

        if not arguments.skip_baselines:
            for label, factory in old_worker.BASELINE_FACTORIES.items():
                model = factory()
                start = time.perf_counter()
                model.fit(train, seed=own.seeds.model_seed, validation=validation)
                elapsed = time.perf_counter() - start
                log_model(n_train, n_val, label, model, elapsed, train, None)
                print(f"[{arm} seed={seed}] n_train={n_train} {label}: done", flush=True)

        for architecture in arguments.architectures:
            model = old_worker._build_model(
                old_worker._architecture_factory(architecture), arguments.net_max_epochs
            )
            start = time.perf_counter()
            model.fit(train, seed=own.seeds.model_seed, validation=validation)
            elapsed = time.perf_counter() - start
            uid = f"{architecture}_{arm}_seed{seed}_n{n_train}"
            save_checkpoint(
                checkpoint_path(checkpoint_dir, uid),
                model,
                model_label=architecture,
                run_uid=uid,
                target_names=train.target_names,
                exogenous_names=train.exogenous_names,
                n_exogenous=len(train.exogenous_names),
                input_scaler=train.input_scaler,
                target_scaler=train.target_scaler,
                process_fingerprint=own.process_config.fingerprint,
                data_spec_params=own_spec.as_parameters(),
                seed_bundle_params=own.seeds.as_parameters(),
                git_commit=environment_fingerprint()["git/commit"],
            )
            log_model(n_train, n_val, architecture, model, elapsed, train, uid)
            print(
                f"[{arm} seed={seed}] n_train={n_train} {architecture}: done ({elapsed:.0f}s)",
                flush=True,
            )

    logger.close()
    print(f"[{arm} seed={seed}] finished, {written} records in {arguments.output}", flush=True)
    return written


def main() -> None:
    arguments = parse_arguments()
    n_methods = len(arguments.architectures) + (0 if arguments.skip_baselines else 3)
    expected = expected_record_count(arguments.arm, arguments.sizes, n_methods)
    written = run(arguments)
    if written != expected:
        raise SystemExit(f"wrote {written} records, expected {expected}")


if __name__ == "__main__":
    main()
