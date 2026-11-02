r"""Multi-pool evaluation of already trained checkpoints -- no retraining.

Complements ``scripts/run_arch_constant_holdout_worker.py``: instead of the one
constant-run holdout pool used there (``HOLDOUT_SEED=999``), two
further, independently drawn pools are generated here (997, 998) and **all 20 already
saved checkpoints** (``load_checkpoint()``, no ``fit()``) are
evaluated against them. Goal: check whether the improvement over persistence on the
constant-run holdout depends on the random choice of the one pool used so far,
or stays stable across several independent pools.

Needs **no** training/excited data -- ``load_checkpoint()`` returns
model weights and the scalers belonging to the training directly from the
``.pt`` file (see ``utils/checkpoints.py``). For each new pool
the raw trajectory is generated **once** and then windowed 20 times, each with
the checkpoint's own scaler pair -- exactly as in the worker script for the
999 pool, only without the preceding training step.

Persistence is logged alongside for comparison: its NRMSE in physical
units is scaler-invariant (round-trip argument, Befund 5 preliminary stage),
so ``PersistenceModel.fit()`` on the test set itself suffices -- the
method only reads the structural channel assignment there, no training data.

Usage::

    python scripts/run_arch_multipool_eval.py

Writes to ``artifacts/runs/arch_gab_constant_holdout_multipool.jsonl``
(existing ``arch_gab_constant_holdout.jsonl`` stays untouched).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import SequenceDataset, TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models import PersistenceModel
from twinlink_ml.models.base import SequenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.checkpoints import load_checkpoint
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "runs" / "arch_gab_constant_holdout_multipool.jsonl"

NEW_HOLDOUT_SEEDS = (997, 998)
"""Zwei zusaetzliche, unabhaengig gezogene Konstantlauf-Pools -- verschieden vom
bisher verwendeten ``HOLDOUT_SEED=999`` in ``run_arch_constant_holdout_worker.py``."""

ARCHITECTURE_LABELS = ("gru", "lstm")
TRAIN_SEEDS = range(1, 11)


class CheckpointSpecMismatchError(RuntimeError):
    """Two checkpoints of this sweep carry different ``data_spec`` parameters.

    Can arise, among other things, when checkpoints come from different states of the
    ``DataSpec`` default (see G5, ``docs/04_stand_ap1.md`` Befund H2) -- an
    evaluation run across several such checkpoints would no longer be
    comparable; the window for one checkpoint does not fit the other.
    """


def _spec_from_checkpoint(data_spec_params: Mapping[str, Any]) -> DataSpec:
    """Reconstruct ``DataSpec`` unchanged from the parameters stored in the checkpoint.

    The module docstring of this script (and of ``run_arch_constant_holdout_worker.py``)
    demands an exact match with the sweep spec against which the loaded checkpoint
    was trained -- window geometry **and** channel count. A ``DataSpec()`` hard-coded
    here would, after a future change of the ``DataSpec`` default (like G5),
    deviate unnoticed from the actually trained checkpoint (H2). Reading it from the
    checkpoint itself makes this dependency structurally impossible, instead of
    asserting it again.
    """
    prefix = "data/"
    field_names = {f.name for f in dataclasses.fields(DataSpec)}
    kwargs: dict[str, Any] = {}
    for key, value in data_spec_params.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        if name not in field_names:
            continue
        kwargs[name] = tuple(value) if isinstance(value, list | tuple) else value
    return DataSpec(**kwargs)


def _verify_matching_spec(
    reference: DataSpec, candidate: DataSpec, *, checkpoint_path: Path
) -> None:
    """Aborts hard if a checkpoint deviates from the reference spec of the sweep.

    Hard error instead of a warning (task requirement J2): an evaluation that silently
    ran two checkpoints with different channel counts against the same window
    would compute wrongly (or with ``IndexError``) for one of the checkpoints, without
    this being noticed later.
    """
    if reference != candidate:
        raise CheckpointSpecMismatchError(
            f"{checkpoint_path}: data_spec weicht vom ersten geladenen Checkpoint ab "
            f"({candidate} != {reference}) -- dieser Sweep ist nicht homogen, eine "
            "gemeinsame Auswertung wäre nicht vergleichbar."
        )


def _holdout_spec(excited: DataSpec) -> DataSpec:
    """Constant-run pool, ``n_train_max=2, n_val=2`` are dummy blocks, never read."""
    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def _log_result(
    logger: JsonlExperimentLogger,
    evaluator: RolloutEvaluator,
    dataset: SequenceDataset,
    *,
    excited_spec: DataSpec,
    seed: int,
    holdout_seed: int,
    common_extra: dict[str, Any],
    label: str,
    model: SequenceModel,
    checkpoint_path: str | None,
) -> None:
    """Evaluate one model on one pool and log it as a ``RunRecord``."""
    result = evaluator.evaluate(model, dataset)
    test_set = f"constant_holdout_{holdout_seed}"
    params: dict[str, object] = {
        **excited_spec.as_parameters(),
        # data/train_constant_run_fraction repeats data/constant_run_fraction (from
        # the same excited_spec) deliberately redundantly -- do NOT remove as superfluous.
        # data/constant_run_fraction is a generic spec attribute and stays
        # identical for EVERY test_set value in this record (also for
        # test_set="constant_holdout_*"), so on its own it says nothing about the
        # kind of training. data/train_constant_run_fraction is the explicit, unambiguously
        # named statement "trained like this" -- without it a constant-trained run
        # would be indistinguishable from an excited-trained one by its record (see the
        # finding entry on this field in docs/04_stand_ap1.md).
        "data/train_constant_run_fraction": excited_spec.constant_run_fraction,
        "master_seed": seed,
        "test_set": test_set,
        "model/label": label,
        **environment_fingerprint(),
        **common_extra,
    }
    if checkpoint_path is not None:
        params["checkpoint/path"] = checkpoint_path
    logger.log(
        RunRecord(
            group="arch_gab_constant_holdout_multipool",
            params=params,
            metrics=result.as_flat_metrics(),
            tags={
                "model": label,
                "group": "arch_gab_constant_holdout_multipool",
                "test_set": test_set,
            },
        )
    )


def _load_all_checkpoints() -> dict[tuple[str, int], tuple[SequenceModel, dict[str, Any]]]:
    """Load all 20 checkpoints once, before the pools are generated.

    Makes it possible to derive the reference spec (target channels, window geometry) from
    the checkpoints themselves and to check their mutual consistency before any
    ``TrajectoryWindowDataset`` windowing takes place.
    """
    loaded: dict[tuple[str, int], tuple[SequenceModel, dict[str, Any]]] = {}
    for architecture in ARCHITECTURE_LABELS:
        for seed in TRAIN_SEEDS:
            ckpt_path = CHECKPOINT_DIR / f"{architecture}_seed{seed}.pt"
            loaded[(architecture, seed)] = load_checkpoint(ckpt_path)
    return loaded


def main() -> None:
    """Generate two new pools, evaluate all 20 checkpoints per pool, write JSONL."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    checkpoints = _load_all_checkpoints()
    (first_arch, first_seed), (_first_model, first_meta) = next(iter(checkpoints.items()))
    excited_spec = _spec_from_checkpoint(first_meta["data_spec"])
    reference_path = CHECKPOINT_DIR / f"{first_arch}_seed{first_seed}.pt"
    for (architecture, seed), (_model, meta) in checkpoints.items():
        ckpt_path = CHECKPOINT_DIR / f"{architecture}_seed{seed}.pt"
        candidate = _spec_from_checkpoint(meta["data_spec"])
        _verify_matching_spec(excited_spec, candidate, checkpoint_path=ckpt_path)
    print(
        f"Referenz-Spec aus {reference_path.name} abgeleitet, "
        f"target_names={excited_spec.target_names}, "
        f"gegen alle {len(checkpoints)} Checkpoints geprueft."
    )

    holdout_spec = _holdout_spec(excited_spec)
    evaluator = RolloutEvaluator(horizons=(1, excited_spec.horizon))
    logger = JsonlExperimentLogger(OUTPUT_PATH)

    for holdout_seed in NEW_HOLDOUT_SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            holdout = generate_data(config, holdout_spec, master_seed=holdout_seed)
        print(
            f"[pool {holdout_seed}] Daten generiert, "
            f"{len(holdout.split.test)} Konstantlauf-Testlaeufe",
            flush=True,
        )

        common_extra = {
            "holdout/seed": holdout_seed,
            "holdout/n_runs": len(holdout.split.test),
            "process/fingerprint": holdout.process_config.fingerprint,
        }

        for architecture in ARCHITECTURE_LABELS:
            for seed in TRAIN_SEEDS:
                ckpt_path = CHECKPOINT_DIR / f"{architecture}_seed{seed}.pt"
                model, metadata = checkpoints[(architecture, seed)]
                scalers = (metadata["input_scaler"], metadata["target_scaler"])
                test_constant = TrajectoryWindowDataset.from_trajectory(
                    holdout.trajectory,
                    holdout.split,
                    SplitName.TEST,
                    excited_spec.window,
                    scalers=scalers,
                )

                _log_result(
                    logger,
                    evaluator,
                    test_constant,
                    excited_spec=excited_spec,
                    seed=seed,
                    holdout_seed=holdout_seed,
                    common_extra=common_extra,
                    label=architecture,
                    model=model,
                    checkpoint_path=str(ckpt_path),
                )

                persistence = PersistenceModel()
                persistence.fit(test_constant, seed=0)
                _log_result(
                    logger,
                    evaluator,
                    test_constant,
                    excited_spec=excited_spec,
                    seed=seed,
                    holdout_seed=holdout_seed,
                    common_extra=common_extra,
                    label="persistence",
                    model=persistence,
                    checkpoint_path=None,
                )

        print(
            f"[pool {holdout_seed}] fertig, 20 Checkpoints + Persistenz ausgewertet",
            flush=True,
        )

    logger.close()
    print(f"Fertig. Ergebnis: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
