r"""One job of the outlier robustness study (``docs/27_outlier_robustness.md``).

Protocol of arm N of Befund 94 (``run_arch_rerun_worker.py``): ``DataSpec.new_study()`` (preset
version 2), the same excited pool per seed and the same constant-run holdout (seed 999), run-level
splits, ``n_train`` budgets as nested prefixes, validation pool, ``max_epochs`` 200, one model seed;
primary evaluation variant only (own pool, planned windows), test sets excited and constant-run
holdout. What is new:

* the moisture channel of the measurements of BOTH pools gets the outlier process of the condition
  (``twinlink_ml.outliers``, settings ``conf/sensor/outliers_placeholder.yaml``), on top of the
  unchanged sensor model, without a new simulation; then, if the condition says so, the causal
  Hampel filter on that channel. Every model sees the processed signal in the input history and as
  training target, persistence and ARX included;
* the networks train ONLY the moisture head: the composite networks have one independent network
  per target, which sees only the exogenous inputs and its own target history and gets its seed
  from (model seed, target name) (``models/composite.py``). A moisture-only data view therefore
  trains exactly the moisture head of Befund 94. Persistence and ARX are fitted on the three-target
  data as in Befund 94 (ARX regresses on the histories of all three channels); only moisture is
  evaluated;
* metrics against three references (``twinlink_ml.outliers.evaluation``): noise-free truth (main),
  disturbed measurement, processed measurement.

A job = one condition, one seed and either one network at one budget, or the two baselines at the
given budgets.

    tools/run_here.sh scripts/run_outlier_robustness_worker.py --condition p0.25_burst_hampel8 \
        --seed 1 --models gru --sizes 16 --output artifacts/runs/_outlier_probe/x.jsonl
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
import run_arch_rerun_worker as rerun_worker

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import ValidationPolicy, build_datasets
from twinlink_ml.outliers import (
    Condition,
    apply_condition,
    evaluate_against_references,
    load_outlier_config,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.code_location import require_code_location
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

if TYPE_CHECKING:
    from twinlink_ml.evaluation.experiments import GeneratedData
    from twinlink_ml.outliers import ContaminatedData

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ("gru", "lstm", "transformer")
BASELINES = ("persistence", "arx")
MODELS = BASELINES + NETWORKS
HOLDOUT_SEED = rerun_worker.HOLDOUT_SEED
EXPECTED_PROCESS_FINGERPRINT = "bddca37f716f5cfc"
TEST_SETS = ("excited", "constant_holdout")


def expected_record_count(models: list[str], sizes: list[int]) -> int:
    """Records of a job: models x budgets x 2 test sets."""
    return len(models) * len(sizes) * len(TEST_SETS)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, help="e.g. p0.25_independent_hampel8")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=old_worker.CONFIG_PATH)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--net-max-epochs", type=int, default=200)
    parser.add_argument("--smoke", action="store_true", help="tiny spec for the tests")
    return parser.parse_args(argv)


def moisture_view(data: GeneratedData, channel: str) -> GeneratedData:
    """The same data with ``channel`` as the only target (input history: exogenous + channel)."""
    return dataclasses.replace(data, spec=dataclasses.replace(data.spec, target_names=(channel,)))


def _generate(config: Any, spec: Any, seed: int) -> GeneratedData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return rerun_worker._generate(config, spec, seed)


def run(arguments: argparse.Namespace) -> int:
    """Run one job; returns the number of records written."""
    if not arguments.smoke:
        require_code_location(ROOT)
    condition = Condition.parse(arguments.condition)
    study = load_outlier_config()
    channel = study.channel
    networks = [m for m in arguments.models if m in NETWORKS]
    if networks:
        import torch

        torch.set_num_threads(arguments.num_threads)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )
    spec = rerun_worker.arm_spec("N", smoke=arguments.smoke)
    seed = arguments.seed
    started = time.perf_counter()
    own = _generate(config, spec, seed)
    holdout = _generate(config, rerun_worker.holdout_spec(spec), HOLDOUT_SEED)
    if not arguments.smoke and own.process_config.fingerprint != EXPECTED_PROCESS_FINGERPRINT:
        raise RuntimeError(
            f"process/fingerprint {own.process_config.fingerprint!r}, "
            f"expected {EXPECTED_PROCESS_FINGERPRINT!r}"
        )
    own_c = apply_condition(own, condition, study)
    holdout_c = apply_condition(holdout, condition, study)
    setup_s = time.perf_counter() - started
    print(f"[{condition.id} seed={seed}] setup done in {setup_s:.0f} s", flush=True)

    logger = JsonlExperimentLogger(arguments.output)
    horizons = (1, spec.horizon)
    environment = environment_fingerprint()
    written = 0

    def log_model(
        n_train: int,
        n_val: int,
        label: str,
        model: Any,
        elapsed: float,
        train: TrajectoryWindowDataset,
        view: str,
    ) -> None:
        nonlocal written
        targets = (channel,) if view == "moisture" else spec.target_names
        window = dataclasses.replace(spec.window, target_names=targets)
        sources: tuple[tuple[str, ContaminatedData, Any], ...] = (
            ("excited", own_c, own.split.subset(n_train)),
            ("constant_holdout", holdout_c, holdout.split),
        )
        for test_name, contaminated, split in sources:
            datasets = [
                TrajectoryWindowDataset.from_trajectory(
                    data.trajectory, split, SplitName.TEST, window, scalers=train.scalers
                )
                for data in (contaminated.processed, contaminated.disturbed)
            ]
            metrics = evaluate_against_references(
                model, datasets[0], datasets[1], channel=channel, horizons=horizons
            )
            params: dict[str, Any] = {
                **spec.as_parameters(),
                **study.as_parameters(),
                **condition.as_parameters(),
                "study": "outlier_robustness",
                "rerun/arm": "N",
                "eval/pool": "own",
                "eval/window": spec.future_actions,
                "eval/primary": True,
                "eval/channel": channel,
                "train/target_view": view,
                "train/target_names": list(targets),
                "process/fingerprint": own.process_config.fingerprint,
                "excitation/fingerprint": own.schedule.metadata.get("fingerprint", "unknown"),
                "split/digest": own.split.digest,
                "eval/pool_excitation_fingerprint": contaminated.processed.schedule.metadata.get(
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
                "setup/elapsed_s": setup_s,
                **{f"model/{k}": v for k, v in model.describe().items()},
                **{f"training/{k}": v for k, v in model.training_diagnostics().items()},
                **environment,
                "master_seed": seed,
                "holdout/seed": HOLDOUT_SEED,
                "holdout/n_runs": len(holdout.split.test),
                **{f"pool/{k}": v for k, v in own.projection_guard.items()},
                **{f"holdout_pool/{k}": v for k, v in holdout.projection_guard.items()},
                **{f"pool/{k}": v for k, v in own_c.diagnostics.items()},
                **{f"holdout_pool/{k}": v for k, v in holdout_c.diagnostics.items()},
            }
            logger.log(
                RunRecord(
                    group="outlier_robustness",
                    params=params,
                    metrics=metrics,
                    tags={
                        "model": label,
                        "group": "outlier_robustness",
                        "test_set": test_name,
                        "n_train": str(n_train),
                        "condition": condition.id,
                    },
                )
            )
            written += 1

    for n_train in arguments.sizes:
        for label in arguments.models:
            view = "moisture" if label in NETWORKS else "all_targets"
            data = (
                moisture_view(own_c.processed, channel) if view == "moisture" else own_c.processed
            )
            train, validation, _ = build_datasets(data, n_train, policy=ValidationPolicy.FIXED_POOL)
            n_val = 0 if validation is None else len(own.split.val)
            if label in NETWORKS:
                model = old_worker._build_model(
                    old_worker._architecture_factory(label), arguments.net_max_epochs
                )
            else:
                model = old_worker.BASELINE_FACTORIES[label]()
            start = time.perf_counter()
            model.fit(train, seed=own.seeds.model_seed, validation=validation)
            elapsed = time.perf_counter() - start
            log_model(n_train, n_val, label, model, elapsed, train, view)
            print(
                f"[{condition.id} seed={seed}] n_train={n_train} {label}: done ({elapsed:.0f}s)",
                flush=True,
            )

    logger.close()
    print(f"[{condition.id} seed={seed}] finished, {written} records", flush=True)
    return written


def main() -> None:
    arguments = parse_arguments()
    expected = expected_record_count(arguments.models, arguments.sizes)
    written = run(arguments)
    if written != expected:
        raise SystemExit(f"wrote {written} records, expected {expected}")


if __name__ == "__main__":
    main()
