r"""X2: schlankes Vorhersageprotokoll fuer den Ablufttemperatur-Kanal (Befund 50).

Eine Architektur (GRU), drei Seeds, konstantes Holdout, ``n_train_max=64`` (Konvention aus
``arch_gab_constant_holdout.jsonl``/Befund 36). Pflichtbaselines: Persistenz UND statische
Regression (``StaticRegressionModel`` -- Pflicht-Baseline nach Befund 2, 98.7 % statisch
erklaerbar fuer ``energy_target``; bei ``product_moisture_kg_per_kg``/Betttemperatur ist die
statische Referenz die bindende Huerde, nicht Persistenz, siehe ``models/baselines.py``).

Ablufttemperatur als Zielkanal: kein Simulator-Output
------------------------------------------------------
``outlet_temperature_celsius`` kann NICHT direkt als ``DataSpec.target_names``-Eintrag
verwendet werden -- ``TrajectoryWindowDataset.from_trajectory`` verlangt
``target_names ⊆ output_names`` (``data/windows.py``), und ``output_names`` ist der feste
Vier-Kanal-Satz des Simulators (``energy_target``, ``bed_temperature_celsius``,
``product_moisture_kg_per_kg``, ``d50_um``, siehe ``GreyBoxFluidBedGranulator.output_names``).
``outlet_temperature_celsius`` existiert erst als nachgelagerte algebraische Rekonstruktion
(``simulation.exhaust.reconstruct_exhaust_state``, ``ExhaustConfig``, X1).

Deshalb: trainiert wird gegen ``bed_temperature_celsius`` (der real existierende Kanal).
Fuer den Bericht wird JEDE Vorhersage zusaetzlich durch
``evaluation.exhaust_reconstruction.evaluate_exhaust_reconstruction()`` geschickt (dieselbe
Funktion, die Befund 41/42 bereits fuer T_gas_out/Y_gas_out nutzen) und dort als
``outlet_temperature_celsius`` berichtet -- bei ``exhaust.outlet_duct_heat_loss_k=0.0``
(aktueller Platzhalter-Default) numerisch bitgleich zu ``T_gas_out_celsius`` (== der
Betttemperatur-Vorhersage selbst). ``data/target_names`` im Log traegt deshalb
``bed_temperature_celsius``, nicht ``outlet_temperature_celsius`` -- das ist der real
verwendete Trainingskanal; ``exhaust/outlet_duct_heat_loss_k`` und
``exhaust/reported_channel`` machen maschinenlesbar, dass die Temperaturmetriken zusaetzlich
unter dem Abluftnamen berichtet werden, ohne die beiden Groessen im Log zu vermischen.

Environment-Variablen VOR jedem Torch-/NumPy-Import
-----------------------------------------------------
``torch.set_num_threads(1)`` begrenzt nur Torchs eigene Intraop-Threads, nicht
darunterliegende BLAS/OpenMP-Bibliotheken (Befund 43g, ``docs/04_stand_ap1.md`` --
plausibelste, dort nicht abschliessend verifizierte Erklaerung fuer den ~3.6-4.9x
Kontentions-Aufschlag im GRU/LSTM-Sweep). ``MKL_NUM_THREADS``/``OMP_NUM_THREADS`` muessen
deshalb als Prozessumgebungsvariablen gesetzt werden, bevor die Bibliotheken initialisiert
werden -- das geht nur VOR dem ersten Import, siehe unten.

Projektion: ``enforce_gas_velocity_projection`` bleibt AUS (Default) -- X2 misst die
Vorhersageguete des Temperaturkanals, nicht die Wirkung der Projektion (Befund 46). Erwarteter
Fingerprint nach Einfuehrung von ``ExhaustConfig`` (X1): ``6ecedc6bd993fb3d`` -- vor dem
Pseudonymisierungs-Schnitt 2026-09-25: ``078572b3a7a0fb45``, siehe ``docs/ARCHIV.md``.
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
    """``n_train_max=64``, dieselbe Konvention wie ``arch_gab_constant_holdout.jsonl``.

    Befund 36, ``run_arch_constant_holdout_worker.py``. ``target_names`` abweichend: der
    dritte Kanal ist hier ``bed_temperature_celsius`` statt ``d50_um``, siehe Moduldocstring.
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
    """Konstantlauf-Pool, identisches Muster zu ``run_arch_constant_holdout_worker.py``."""
    import dataclasses

    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def parse_arguments() -> argparse.Namespace:
    """Kommandozeile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--num-threads", type=int, default=1, help="torch.set_num_threads")
    return parser.parse_args()


def _verify_thread_environment_logged(
    record_params: dict[str, object], *, seed: int, require_device: bool
) -> None:
    """Bricht ab, wenn die Y1-Diagnosefelder (dcdc41d/e164460) leer sind.

    Erste Gelegenheit, diese Felder im Echtbetrieb zu sehen (bisher nur in
    ``befund42_tbed_exhaust_sweep.jsonl``, das den Kontentions-Befund 43g motiviert hat) --
    vor ALLEN neun Fits (3 Seeds x 3 Modelle) prüfen, nicht erst am Ende.

    ``model/net_actual_device`` nur bei ``require_device=True`` (GRU-Record) verlangt: die
    Baselines (``PersistenceModel``/``StaticRegressionModel``) laufen nicht auf Torch, ihr
    ``describe()`` traegt dieses Feld konstruktionsbedingt nicht -- ``super().describe()``
    ohne Device-Information, siehe ``models/baselines.py``.
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
    """``T_gas_out_celsius`` und ``outlet_temperature_celsius`` muessen bitgleich sein.

    Bei ``outlet_duct_heat_loss_k=0.0`` sind beide aus derselben Betttemperatur-Vorhersage
    rekonstruiert -- sonst steckt in der Rekonstruktion mehr als der reine Verlustabzug, und
    das muss VOR dem Berichten unter dem Abluftnamen auffallen.
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
    """Ein Modell gegen beide Testmengen auswerten und je einen ``RunRecord`` loggen.

    Modulfunktion statt Closure in ``main()``: eine Closure ueber die Seed-Schleife wuerde
    Schleifenvariablen einfangen (Ruff B023) -- hier stattdessen alles als explizite
    Parameter, keine Mehrdeutigkeit ueber welchen Seed-Stand die Funktion sieht.
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
    """Drei Seeds: Persistenz + statische Baseline + GRU, je gegen zwei Testmengen."""
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
