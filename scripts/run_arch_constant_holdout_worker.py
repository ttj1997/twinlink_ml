r"""One (architecture, seed) job: train, evaluate on two test sets, save a checkpoint.

Inner building block of ``scripts/run_arch_constant_holdout_sweep.py``, which starts several calls
of this script as independent processes in parallel -- see there for the
orchestration. Callable on its own for a single test::

    python scripts/run_arch_constant_holdout_worker.py --architecture gru --seed 1 \
        --with-baselines --output artifacts/runs/_worker_gru_1.jsonl

Two test sets, one trained model
----------------------------------------
1. The previous excited test set from the same pool as the training (reproduction
   of Befund 28 under the GAB isotherm and the 900-s horizon from Befund 35).
2. A **fixed** constant-run holdout pool (``constant_run_fraction=1.0``), with **one**
   fixed seed across all 20 jobs (``HOLDOUT_SEED``) -- not drawn anew per job.
   Otherwise the observed spread over the ten training seeds mixes two sources:
   how well the training went for this seed, and which random realisation of the
   constant operation happened to be drawn. In reality there is one plant
   that is operated constantly, not ten different constant-run worlds.

The windowing of the second test set uses **the scalers of the training dataset
of this run** (``train.scalers``), not newly estimated ones -- otherwise the comparison would no
longer be "same model, different test data". The same principle as
``TrajectoryWindowDataset.from_trajectory(..., scalers=...)`` in ``build_datasets``,
only called directly outside of it, because ``build_datasets`` only knows a single
coherent pool.

Baselines only with ``--with-baselines``: persistence/static/ARX are independent of the
architecture and would otherwise be computed twice and logged twice with two jobs per seed
(gru, lstm) -- the sweep activates the flag only for the first job of each seed.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models import LinearARXModel, PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.checkpoints import checkpoint_path, save_checkpoint
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable

    from twinlink_ml.models.base import SequenceModel
    from twinlink_ml.models.composite import CompositeSequenceModel

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)
HOLDOUT_SEED = 999
"""Fester Seed des Konstantlauf-Holdout-Pools -- **ein** Pool für alle 20 Jobs, nicht
pro Seed neu gezogen. Deutlich von den zehn Trainings-Seeds (1..10) unterschieden."""

ARCHITECTURE_LABELS = ("gru", "lstm")
BASELINE_FACTORIES = {
    "persistence": PersistenceModel,
    "static": StaticRegressionModel,
    "arx": LinearARXModel,
}


def _excited_spec(config_path: Path) -> DataSpec:
    del config_path  # the config itself determines no spec values, only for signature symmetry
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=36,  # 900s, Befund 35
        stride=8,
    )


def _holdout_spec(excited: DataSpec) -> DataSpec:
    """Constant-run pool, n_test matched to the standard test set (Befund 26/28 convention).

    ``n_train_max=2, n_val=2`` are dummy blocks, never used -- ``IndexBlockSplitter``
    requires ``n_train>=1, n_test>=1``, and ``_finalize`` additionally ``train>=2`` per split.
    Only ``.test`` is read.
    """
    import dataclasses

    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def _architecture_factory(label: str) -> Callable[[], CompositeSequenceModel]:
    from twinlink_ml.models.composite import CompositeGruSequenceModel, CompositeLstmSequenceModel

    return {"gru": CompositeGruSequenceModel, "lstm": CompositeLstmSequenceModel}[label]


def parse_arguments() -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURE_LABELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True, help="JSONL-Ausgabe dieses Jobs")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts") / "checkpoints")
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help="Persistenz/static/ARX zusaetzlich fahren (nur einmal je Seed noetig)",
    )
    parser.add_argument("--num-threads", type=int, default=1, help="torch.set_num_threads")
    return parser.parse_args()


def main() -> None:
    """One job: train, evaluate two test sets, write checkpoint + JSONL."""
    arguments = parse_arguments()

    import torch

    torch.set_num_threads(arguments.num_threads)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)

    excited_spec = _excited_spec(arguments.config)
    holdout_spec = _holdout_spec(excited_spec)
    evaluator = RolloutEvaluator(horizons=(1, excited_spec.horizon))
    logger = JsonlExperimentLogger(arguments.output)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        excited = generate_data(config, excited_spec, master_seed=arguments.seed)
    train, validation, test_excited = build_datasets(
        excited, excited_spec.n_train_max, policy=ValidationPolicy.FIXED_POOL
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        holdout = generate_data(config, holdout_spec, master_seed=HOLDOUT_SEED)
    test_constant = TrajectoryWindowDataset.from_trajectory(
        holdout.trajectory,
        holdout.split,
        SplitName.TEST,
        excited_spec.window,
        scalers=train.scalers,
    )

    # Marker for the staggered sweep start: signals that config, data generation
    # and windowing (which would most likely reveal network-drive/configuration errors)
    # ran cleanly before the long training begins. The orchestrator reads this from
    # the log file of this process, not from the process status alone (which after
    # a few seconds would only say "still running" anyway).
    print(
        f"[{arguments.architecture} seed={arguments.seed}] Setup fertig "
        f"(Daten generiert, {len(holdout.split.test)} Konstantlauf-Testlaeufe), Training startet",
        flush=True,
    )

    common_extra = {
        "master_seed": arguments.seed,
        "holdout/seed": HOLDOUT_SEED,
        "holdout/n_runs": len(holdout.split.test),
    }

    def _log_model(label: str, model: SequenceModel, *, checkpoint_run_uid: str | None) -> None:
        test_sets = (("excited", test_excited), ("constant_holdout", test_constant))
        for test_name, test_dataset in test_sets:
            result = evaluator.evaluate(model, test_dataset)
            params: dict[str, object] = {
                **excited_spec.as_parameters(),
                # data/train_constant_run_fraction repeats data/constant_run_fraction
                # (from the same excited_spec) deliberately redundantly -- do NOT remove as
                # superfluous. data/constant_run_fraction is a generic spec attribute and
                # stays identical for EVERY test_set value in this record (also for
                # test_set="constant_holdout"), so on its own it says nothing about the
                # kind of training. data/train_constant_run_fraction is the explicit,
                # unambiguously named statement "trained like this" -- without it a
                # constant-trained run would be indistinguishable from an excited-trained one
                # by its record (see the finding entry on this field in
                # docs/04_stand_ap1.md).
                "data/train_constant_run_fraction": excited_spec.constant_run_fraction,
                "process/fingerprint": excited.process_config.fingerprint,
                "excitation/fingerprint": excited.schedule.metadata.get("fingerprint", "unknown"),
                "split/digest": excited.split.digest,
                **excited.seeds.as_parameters(),
                "test_set": test_name,
                "model/label": label,
                **environment_fingerprint(),
                **common_extra,
            }
            if checkpoint_run_uid is not None:
                params["checkpoint/path"] = str(
                    checkpoint_path(arguments.checkpoint_dir, checkpoint_run_uid)
                )
            logger.log(
                RunRecord(
                    group="arch_gab_constant_holdout",
                    params=params,
                    metrics=result.as_flat_metrics(),
                    tags={
                        "model": label,
                        "group": "arch_gab_constant_holdout",
                        "test_set": test_name,
                    },
                )
            )

    if arguments.with_baselines:
        for label, factory in BASELINE_FACTORIES.items():
            model = factory()
            model.fit(train, seed=excited.seeds.model_seed, validation=validation)
            _log_model(label, model, checkpoint_run_uid=None)

    start = time.perf_counter()
    architecture_factory = _architecture_factory(arguments.architecture)
    model = architecture_factory()
    model.fit(train, seed=excited.seeds.model_seed, validation=validation)
    elapsed = time.perf_counter() - start

    training_run_uid = f"{arguments.architecture}_seed{arguments.seed}"
    ckpt_path = checkpoint_path(arguments.checkpoint_dir, training_run_uid)
    save_checkpoint(
        ckpt_path,
        model,
        model_label=arguments.architecture,
        run_uid=training_run_uid,
        target_names=train.target_names,
        exogenous_names=train.exogenous_names,
        n_exogenous=len(train.exogenous_names),
        input_scaler=train.input_scaler,
        target_scaler=train.target_scaler,
        process_fingerprint=excited.process_config.fingerprint,
        data_spec_params=excited_spec.as_parameters(),
        seed_bundle_params=excited.seeds.as_parameters(),
        git_commit=environment_fingerprint()["git/commit"],
    )
    _log_model(arguments.architecture, model, checkpoint_run_uid=training_run_uid)

    logger.close()
    print(
        f"[{arguments.architecture} seed={arguments.seed}] fertig, "
        f"Training {elapsed:.1f}s, Checkpoint {ckpt_path}, Log {arguments.output}"
    )


if __name__ == "__main__":
    main()
