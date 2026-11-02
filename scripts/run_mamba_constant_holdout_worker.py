r"""One (Mamba, seed) job: train, evaluate on two test sets, save a checkpoint.

Full constant-run holdout scale (Befund 36) for Mamba, instead of the cost-saving
single-seed measurement from Befund 38 (``run_mamba_reduced_comparison.py``, ``n_train=16``,
1 seed). GRU/LSTM/persistence/static/ARX are NOT recomputed -- the 100 records in
``artifacts/runs/arch_gab_constant_holdout.jsonl`` (5 models x 10 seeds x 2 test sets,
Befund 36) remain authoritative and are reused directly for the comparison table,
no ``--with-baselines`` equivalent here.

Same basic pattern as ``run_mamba_reduced_comparison.py``: verify existing records via
fingerprint/split/seed assertion, compute only the new architecture -- here
generalised to all 10 seeds instead of hard-coded values for seed 1 (``expected``
comes per seed from the already logged file, not from a literal).

``_excited_spec``/``_holdout_spec``/``HOLDOUT_SEED``/``CONFIG_PATH`` are imported from
``run_arch_constant_holdout_worker.py`` instead of retyped -- bit-identical
DataSpec guaranteed, no drift risk with a future change there. The import
works because both scripts live in the same ``scripts/`` directory (Python prepends
the script directory to ``sys.path`` on the call ``python scripts/x.py``); the
imported file has no module side effect (``main()`` sits behind
``if __name__ == "__main__"``).

Mamba configuration: ``CompositeMambaSequenceModel()`` without config override, i.e. the
final state from Befund 38 (``hidden_size=16``, ``state_size=16``, ``num_layers=1``,
RMSNorm fix + weight-decay exclusion for ``A_log``/``D``, ``models/mamba.py`` /
``models/composite.py`` -- NOT touched here). Standard ``learning_rate``/
``weight_decay``/``max_epochs=200``/``patience=20`` (dataclass default,
``models/torch_sequence.py``), unchanged.

Callable for a single test::

    python scripts/run_mamba_constant_holdout_worker.py --seed 1 \
        --output artifacts/runs/_mamba_worker_1.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from run_arch_constant_holdout_worker import CONFIG_PATH, HOLDOUT_SEED, _excited_spec, _holdout_spec

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models.composite import CompositeMambaSequenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.checkpoints import checkpoint_path, save_checkpoint
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

if TYPE_CHECKING:
    from twinlink_ml.models.base import SequenceModel

REPO_ROOT = Path(__file__).resolve().parents[1]
EXISTING_RECORDS_PATH = REPO_ROOT / "artifacts" / "runs" / "arch_gab_constant_holdout.jsonl"
"""Befund 36: 5 Modelle (persistence/static/arx/gru/lstm) x 10 Seeds x 2 Testmengen,
n_train_max=64, HOLDOUT_SEED=999 -- Sollwert-Quelle fuer die Bitgleichheits-Pruefung
unten, unveraendert gelassen (nur gelesen, nie beschrieben)."""


def _expected_from_existing(seed: int) -> dict[str, object]:
    """Read fingerprint/split/seeds/target_names for ``seed`` from the logged records.

    Analogous to the hard-coded ``EXPECTED`` dict in ``run_mamba_reduced_comparison.py``,
    but here read per seed from the file instead of typed in once for seed 1 --
    the same reuse/verification logic for 10 seeds instead of 1.

    ``data/target_names`` is read from the log, not hard-coded -- a hard-coded
    channel list would, with a future ``DataSpec`` change, produce the same
    self-contradiction as in ``run_xlstm_reduced_comparison.py`` (hard-coded there
    and checked against a two-channel ``_spec()`` default, see the docstring there).
    """
    with EXISTING_RECORDS_PATH.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            params = record["params"]
            if params.get("master_seed") == seed and params.get("test_set") == "excited":
                return {
                    "process/fingerprint": params["process/fingerprint"],
                    "split/digest": params["split/digest"],
                    "seed/data": params["seed/data"],
                    "seed/noise": params["seed/noise"],
                    "seed/model": params["seed/model"],
                    "data/target_names": tuple(params["data/target_names"]),
                }
    raise RuntimeError(
        f"Kein bestehender Record fuer master_seed={seed} in {EXISTING_RECORDS_PATH} -- "
        "Befund-36-Datei unvollstaendig oder Seed ausserhalb 1..10."
    )


def _check_expected(seed: int, expected: dict[str, object], actual: dict[str, object]) -> None:
    """Check ``expected`` (from the log file) against ``actual``, abort hard on a deviation.

    Separate function (not inline in ``main()``) so that the schema check is testable
    without a full data-generation/training run -- analogous to ``_check_expected`` in
    ``run_xlstm_reduced_comparison.py``/``run_mamba_reduced_comparison.py``.
    """
    mismatches = {k: (v, actual[k]) for k, v in expected.items() if v != actual[k]}
    if not mismatches:
        return
    fingerprint_only = set(mismatches) == {"process/fingerprint"}
    target_names_only = set(mismatches) == {"data/target_names"}
    projection_hint = (
        "\n\nWahrscheinliche Ursache: NumericsConfig.enforce_gas_velocity_projection "
        "wurde eingefuehrt (Default AUS) -- ProcessConfig.fingerprint hasht die gesamte "
        "Config, daher aendert sich der Wert bereits durch das neue Schema-Feld, auch "
        "ohne die Projektion zu aktivieren. Das ist HIER kein blosser Bitgleichheits-"
        f"Fehlschlag: {EXISTING_RECORDS_PATH.name} (Befund 36, 5 Modelle x 10 Seeds x 2 "
        "Testmengen) muesste neu erzeugt werden (run_arch_constant_holdout_worker.py / "
        "-sweep.py, echtes Training), um mit dem aktuellen Fingerprint bitgleich zu "
        "sein -- das ist ein separater, hier bewusst NICHT ausgefuehrter Trainings-Sweep. "
        "Siehe scripts/probe_gas_velocity_projection.py und "
        "tests/test_gas_velocity_projection.py fuer die Aequivalenz bei "
        "ausgeschalteter Projektion."
        if fingerprint_only
        else ""
    )
    target_names_hint = (
        f"\n\nWahrscheinliche Ursache: {EXISTING_RECORDS_PATH.name} traegt "
        f"{len(expected['data/target_names'])} Zielkanaele {expected['data/target_names']}, "
        f"der aktuelle DataSpec-Default hat {len(actual['data/target_names'])} Zielkanaele "
        f"{actual['data/target_names']} -- die Kanalzahl der Vergleichsbasis und des "
        "aktuellen Codes laufen auseinander. Ein Log mit uneinheitlicher Kanalzahl ueber "
        "seine Records ist wertlos und faellt niemandem auf. Loesung: die Vergleichsbasis "
        f"selbst neu erzeugen ({EXISTING_RECORDS_PATH.name}, Befund 36, "
        "run_arch_constant_holdout_worker.py/-sweep.py, echtes Training) -- kein Patch an "
        "diesem Skript."
        if target_names_only
        else ""
    )
    raise RuntimeError(
        f"Seed {seed}: Daten weichen von {EXISTING_RECORDS_PATH.name} ab, "
        f"Wiederverwendung der GRU/LSTM/Baseline-Records waere UNGUELTIG: "
        f"{mismatches}{projection_hint}{target_names_hint}"
    )


def parse_arguments() -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True, help="JSONL-Ausgabe dieses Jobs")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts") / "checkpoints")
    parser.add_argument("--num-threads", type=int, default=1, help="torch.set_num_threads")
    return parser.parse_args()


def main() -> None:
    """One job: check data compatibility, train, evaluate two test sets."""
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

    expected = _expected_from_existing(arguments.seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        excited = generate_data(config, excited_spec, master_seed=arguments.seed)

    actual = {
        "process/fingerprint": excited.process_config.fingerprint,
        "split/digest": excited.split.digest,
        "seed/data": excited.seeds.data_seed,
        "seed/noise": excited.seeds.noise_seed,
        "seed/model": excited.seeds.model_seed,
        "data/target_names": tuple(excited_spec.target_names),
    }
    _check_expected(arguments.seed, expected, actual)

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

    print(
        f"[mamba seed={arguments.seed}] Fingerprint/Split/Seeds gegen "
        f"{EXISTING_RECORDS_PATH.name} bitgleich bestaetigt. Setup fertig "
        f"({len(holdout.split.test)} Konstantlauf-Testlaeufe), Training startet",
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

    start = time.perf_counter()
    model = CompositeMambaSequenceModel()
    model.fit(train, seed=excited.seeds.model_seed, validation=validation, verbose=True)
    elapsed = time.perf_counter() - start

    for target, history in model.training_history.items():
        print(
            f"[mamba seed={arguments.seed}]   Teilnetz {target!r}: "
            f"{len(history)}/{model.config.max_epochs} Epochen",
            flush=True,
        )

    training_run_uid = f"mamba_seed{arguments.seed}"
    ckpt_path = checkpoint_path(arguments.checkpoint_dir, training_run_uid)
    save_checkpoint(
        ckpt_path,
        model,
        model_label="mamba",
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
    _log_model("mamba", model, checkpoint_run_uid=training_run_uid)

    logger.close()
    print(
        f"[mamba seed={arguments.seed}] fertig, "
        f"Training {elapsed:.1f}s, Checkpoint {ckpt_path}, Log {arguments.output}"
    )


if __name__ == "__main__":
    main()
