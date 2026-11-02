r"""X2: lean prediction protocol for the exhaust air temperature channel (Befund 50).

One architecture (GRU), three seeds, constant holdout, ``n_train_max=64`` (convention from
``arch_gab_constant_holdout.jsonl``/Befund 36). Mandatory baselines: persistence AND static
regression (``StaticRegressionModel`` -- mandatory baseline after Befund 2, 98.7 % statically
explainable for ``energy_target``; for ``product_moisture_kg_per_kg``/bed temperature the
static reference is the binding hurdle, not persistence, see ``models/baselines.py``).

Exhaust air temperature as a target channel: not a simulator output
------------------------------------------------------
``outlet_temperature_celsius`` can NOT be used directly as a ``DataSpec.target_names`` entry
-- ``TrajectoryWindowDataset.from_trajectory`` requires
``target_names ⊆ output_names`` (``data/windows.py``), and ``output_names`` is the fixed
four-channel set of the simulator (``energy_target``, ``bed_temperature_celsius``,
``product_moisture_kg_per_kg``, ``d50_um``, see ``GreyBoxFluidBedGranulator.output_names``).
``outlet_temperature_celsius`` only exists as a downstream algebraic reconstruction
(``simulation.exhaust.reconstruct_exhaust_state``, ``ExhaustConfig``, X1).

Therefore: training is against ``bed_temperature_celsius`` (the channel that really exists).
For the report EVERY prediction is additionally passed through
``evaluation.exhaust_reconstruction.evaluate_exhaust_reconstruction()`` (the same
function that Befund 41/42 already use for T_gas_out/Y_gas_out) and reported there as
``outlet_temperature_celsius`` -- with ``exhaust.outlet_duct_heat_loss_k=0.0``
(current placeholder default) numerically bit-identical to ``T_gas_out_celsius`` (== the
bed temperature prediction itself). ``data/target_names`` in the log therefore carries
``bed_temperature_celsius``, not ``outlet_temperature_celsius`` -- that is the training channel
actually used; ``exhaust/outlet_duct_heat_loss_k`` and
``exhaust/reported_channel`` make machine-readable that the temperature metrics are additionally
reported under the exhaust air name, without mixing the two quantities in the log.

Environment variables BEFORE any torch/NumPy import
-----------------------------------------------------
``torch.set_num_threads(1)`` only limits torch's own intra-op threads, not the
underlying BLAS/OpenMP libraries (Befund 43g, ``docs/04_stand_ap1.md`` --
the most plausible explanation, not conclusively verified there, for the ~3.6-4.9x
contention overhead in the GRU/LSTM sweep). ``MKL_NUM_THREADS``/``OMP_NUM_THREADS`` must
therefore be set as process environment variables before the libraries are initialised
-- this is only possible BEFORE the first import, see below.

Projection: ``enforce_gas_velocity_projection`` stays OFF (default) -- X2 measures the
prediction quality of the temperature channel, not the effect of the projection (Befund 46).
Expected fingerprint after the introduction of ``ExhaustConfig`` (X1): ``6ecedc6bd993fb3d`` --
before the pseudonymisation cut 2026-09-25: ``078572b3a7a0fb45``, see ``docs/ARCHIV.md``.
"""

from __future__ import annotations

import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.evaluation.exhaust_reconstruction import (
    ExhaustReconstructionResult,
    evaluate_exhaust_reconstruction,
)
from twinlink_ml.models import PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.checkpoints import checkpoint_path, save_checkpoint
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

if TYPE_CHECKING:
    from twinlink_ml.data.windows import SequenceDataset
    from twinlink_ml.evaluation.experiments import GeneratedData
    from twinlink_ml.models.base import SequenceModel

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "runs"
    / "outlet_temperature_gru_protocol.jsonl"
)
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "checkpoints"
GROUP = "outlet_temperature_gru_protocol"

SEEDS = (1, 2, 3)
HOLDOUT_SEED = 999
"""Fester Seed des Konstantlauf-Holdout-Pools, identische Konvention wie
``run_arch_constant_holdout_worker.py`` (Befund 36) -- **ein** Pool fuer alle drei Seeds."""

TARGET_NAMES = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
EXPECTED_FINGERPRINT_PROJECTION_OFF = "6ecedc6bd993fb3d"
"""``ProcessConfig.fingerprint`` nach Einfuehrung von ``ExhaustConfig`` (X1), Projektion AUS
-- siehe ``docs/04_stand_ap1.md``, Abschnitt "Fingerprint-Historie"."""

BASELINE_FACTORIES = {
    "persistence": PersistenceModel,
    "static": StaticRegressionModel,
}


def _excited_spec() -> DataSpec:
    """``n_train_max=64``, the same convention as ``arch_gab_constant_holdout.jsonl``.

    Befund 36, ``run_arch_constant_holdout_worker.py``. ``target_names`` deviates: the
    third channel here is ``bed_temperature_celsius`` instead of ``d50_um``, see module docstring.
    """
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=36,
        stride=8,
        target_names=TARGET_NAMES,
    )


def _holdout_spec(excited: DataSpec) -> DataSpec:
    """Constant-run pool, identical pattern to ``run_arch_constant_holdout_worker.py``."""
    import dataclasses

    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def parse_arguments() -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--num-threads", type=int, default=1, help="torch.set_num_threads")
    return parser.parse_args()


def _verify_thread_environment_logged(
    record_params: dict[str, object], *, seed: int, require_device: bool
) -> None:
    """Aborts if the Y1 diagnostic fields (dcdc41d/e164460) are empty.

    First opportunity to see these fields in real operation (so far only in
    ``befund42_tbed_exhaust_sweep.jsonl``, which motivated the contention finding 43g) --
    check before ALL nine fits (3 seeds x 3 models), not only at the end.

    ``model/net_actual_device`` only required with ``require_device=True`` (GRU record): the
    baselines (``PersistenceModel``/``StaticRegressionModel``) do not run on torch; their
    ``describe()`` does not carry this field by construction -- ``super().describe()``
    without device information, see ``models/baselines.py``.
    """
    required = ["torch/num_threads", "env/mkl_num_threads", "env/omp_num_threads"]
    if require_device:
        required.append("model/net_actual_device")
    missing = [name for name in required if not record_params.get(name)]
    if missing:
        raise RuntimeError(
            f"Seed {seed}: Diagnosefelder {missing} sind leer/None im ersten fertigen Record -- "
            "abgebrochen VOR den restlichen Fits, wie angewiesen. Erwartet: "
            "torch/num_threads=1, env/mkl_num_threads='1', env/omp_num_threads='1'"
            + (", model/net_actual_device=<'cpu'|'cuda:...'>." if require_device else ".")
        )


def _verify_outlet_identity(
    exhaust_result: ExhaustReconstructionResult, *, seed: int, label: str
) -> None:
    """``T_gas_out_celsius`` and ``outlet_temperature_celsius`` must be bit-identical.

    With ``outlet_duct_heat_loss_k=0.0`` both are reconstructed from the same bed temperature
    prediction -- otherwise the reconstruction contains more than the pure loss subtraction, and
    that must be noticed BEFORE reporting under the exhaust air name.
    """
    for horizon in exhaust_result.horizons:
        direct = exhaust_result.rmse_run_median_physical["T_gas_out_celsius"][horizon]
        reported = exhaust_result.rmse_run_median_physical["outlet_temperature_celsius"][horizon]
        if direct != reported:
            raise RuntimeError(
                f"Seed {seed} [{label}], h={horizon}: T_gas_out_celsius-RMSE ({direct}) und "
                f"outlet_temperature_celsius-RMSE ({reported}) weichen voneinander ab, obwohl "
                "outlet_duct_heat_loss_k=0.0 -- Identitaet verletzt, abgebrochen vor weiterer "
                "Auswertung."
            )
        direct_nrmse = exhaust_result.nrmse_run_spread["T_gas_out_celsius"][horizon]
        reported_nrmse = exhaust_result.nrmse_run_spread["outlet_temperature_celsius"][horizon]
        if direct_nrmse != reported_nrmse:
            raise RuntimeError(
                f"Seed {seed} [{label}], h={horizon}: NRMSE-Spannen weichen ab: "
                f"{direct_nrmse} (T_gas_out_celsius) vs. {reported_nrmse} "
                "(outlet_temperature_celsius) -- Identitaet verletzt."
            )


def _log_model(
    *,
    logger: JsonlExperimentLogger,
    evaluator: RolloutEvaluator,
    label: str,
    model: SequenceModel,
    seed: int,
    excited: GeneratedData,
    holdout: GeneratedData,
    test_excited: SequenceDataset,
    test_constant: SequenceDataset,
    excited_spec: DataSpec,
    common_extra: dict[str, object],
    checkpoint_dir: Path,
    checkpoint_run_uid: str | None,
    training_elapsed_s: float | None = None,
) -> dict[str, object]:
    """Evaluate one model against both test sets and log one ``RunRecord`` each.

    Module function instead of a closure in ``main()``: a closure over the seed loop would
    capture loop variables (Ruff B023) -- here instead everything as explicit
    parameters, no ambiguity about which seed state the function sees.
    """
    test_sets: tuple[tuple[str, SequenceDataset, GeneratedData], ...] = (
        ("excited", test_excited, excited),
        ("constant_holdout", test_constant, holdout),
    )
    first_params: dict[str, object] | None = None
    for test_name, test_dataset, source_data in test_sets:
        result = evaluator.evaluate(model, test_dataset)
        exhaust_result = evaluate_exhaust_reconstruction(model, source_data, test_dataset)
        _verify_outlet_identity(exhaust_result, seed=seed, label=f"{label}/{test_name}")

        metrics = {**result.as_flat_metrics(), **exhaust_result.as_flat_metrics()}
        if training_elapsed_s is not None:
            metrics["training/elapsed_s"] = training_elapsed_s

        params: dict[str, object] = {
            **excited_spec.as_parameters(),
            "process/fingerprint": excited.process_config.fingerprint,
            "excitation/fingerprint": excited.schedule.metadata.get("fingerprint", "unknown"),
            "split/digest": excited.split.digest,
            **excited.seeds.as_parameters(),
            "test_set": test_name,
            "model/label": label,
            **{f"model/{key}": value for key, value in model.describe().items()},
            **environment_fingerprint(),
            **common_extra,
        }
        if checkpoint_run_uid is not None:
            params["checkpoint/path"] = str(checkpoint_path(checkpoint_dir, checkpoint_run_uid))
        logger.log(
            RunRecord(
                group=GROUP,
                params=params,
                metrics=metrics,
                tags={"model": label, "group": GROUP, "test_set": test_name},
            )
        )
        if first_params is None:
            first_params = params
    assert first_params is not None
    return first_params


def main() -> None:
    """Three seeds: persistence + static baseline + GRU, each against two test sets."""
    arguments = parse_arguments()

    import torch

    torch.set_num_threads(arguments.num_threads)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)

    if config.numerics.enforce_gas_velocity_projection:
        raise RuntimeError(
            "enforce_gas_velocity_projection ist AN -- X2 braucht AUS (Default), siehe "
            "Moduldocstring. Config pruefen."
        )

    excited_spec = _excited_spec()
    holdout_spec = _holdout_spec(excited_spec)
    evaluator = RolloutEvaluator(horizons=(1, excited_spec.horizon))
    logger = JsonlExperimentLogger(arguments.output)

    first_record_checked = False
    first_gru_record_checked = False

    for seed in SEEDS:
        seed_start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            excited = generate_data(config, excited_spec, master_seed=seed)

        if not fingerprints_equivalent(
            excited.process_config.fingerprint, EXPECTED_FINGERPRINT_PROJECTION_OFF
        ):
            raise RuntimeError(
                f"Seed {seed}: process/fingerprint={excited.process_config.fingerprint!r}, "
                f"erwartet {EXPECTED_FINGERPRINT_PROJECTION_OFF!r} (Projektion AUS, nach "
                "ExhaustConfig/X1) -- Config oder Code-Stand weicht ab, abgebrochen."
            )

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
            f"[seed={seed}] Setup fertig (Daten generiert, {len(holdout.split.test)} "
            "Konstantlauf-Testlaeufe), Training startet",
            flush=True,
        )

        common_extra: dict[str, object] = {
            "master_seed": seed,
            "holdout/seed": HOLDOUT_SEED,
            "holdout/n_runs": len(holdout.split.test),
            "exhaust/outlet_duct_heat_loss_k": config.exhaust.outlet_duct_heat_loss_k,
            "exhaust/reported_channel": "outlet_temperature_celsius",
        }

        for label, factory in BASELINE_FACTORIES.items():
            baseline_model = factory()
            baseline_model.fit(train, seed=excited.seeds.model_seed, validation=validation)
            first_params = _log_model(
                logger=logger,
                evaluator=evaluator,
                label=label,
                model=baseline_model,
                seed=seed,
                excited=excited,
                holdout=holdout,
                test_excited=test_excited,
                test_constant=test_constant,
                excited_spec=excited_spec,
                common_extra=common_extra,
                checkpoint_dir=arguments.checkpoint_dir,
                checkpoint_run_uid=None,
            )
            if not first_record_checked:
                _verify_thread_environment_logged(first_params, seed=seed, require_device=False)
                first_record_checked = True

        from twinlink_ml.models.composite import CompositeGruSequenceModel

        start = time.perf_counter()
        gru_model = CompositeGruSequenceModel()
        gru_model.fit(train, seed=excited.seeds.model_seed, validation=validation)
        elapsed = time.perf_counter() - start

        training_run_uid = f"gru_seed{seed}"
        ckpt_path = checkpoint_path(arguments.checkpoint_dir, training_run_uid)
        save_checkpoint(
            ckpt_path,
            gru_model,
            model_label="gru",
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
        gru_first_params = _log_model(
            logger=logger,
            evaluator=evaluator,
            label="gru",
            model=gru_model,
            seed=seed,
            excited=excited,
            holdout=holdout,
            test_excited=test_excited,
            test_constant=test_constant,
            excited_spec=excited_spec,
            common_extra=common_extra,
            checkpoint_dir=arguments.checkpoint_dir,
            checkpoint_run_uid=training_run_uid,
            training_elapsed_s=elapsed,
        )
        if not first_gru_record_checked:
            _verify_thread_environment_logged(gru_first_params, seed=seed, require_device=True)
            first_gru_record_checked = True

        seed_elapsed = time.perf_counter() - seed_start
        print(
            f"[seed={seed}] fertig, GRU-Training {elapsed:.1f}s, gesamt {seed_elapsed:.1f}s, "
            f"Checkpoint {ckpt_path}",
            flush=True,
        )

    logger.close()
    print(f"Alle drei Seeds fertig. Log: {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
