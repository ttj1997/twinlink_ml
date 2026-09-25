r"""Ein Seed-Job: Lernkurve ueber n_train in {8,16,32,64}, alle fuenf Architekturen --
KONSTANT trainierter Zwilling zu ``run_arch_learning_curve_worker.py``.

Beantwortet die Frage, ob das Anregungsexperiment (Spruenge/PRBS/Rampen ueber den Lauf)
ueberhaupt noetig ist, oder ob Betriebsdaten (Stellgroessen je Lauf konstant, nur zwischen
Laeufen unterschiedlich) genuegen -- am Anwendungspartner die Entscheidung zwischen "Modell
aus vorhandenen Aufzeichnungen bauen" und "Anlagenversuch noetig".

Einzige inhaltliche Aenderung gegenueber ``run_arch_learning_curve_worker.py``:
``DataSpec.constant_run_fraction=1.0`` im TRAININGSPOOL statt des dortigen Defaults 0.10 --
Stellgroessen bleiben ueber die volle Laufdauer (28800 s) fest, nur die Stoergroessen
bewegen sich; die Sollwerte selbst variieren weiterhin ZWISCHEN den 128 Pool-Laeufen (siehe
Vorbehalt zu diesem Befund in ``docs/04_stand_ap1.md``: das ist der guenstige Fall fuer
Betriebsdaten, nicht der ungewuenstigste -- ein realer Datensatz haette moeglicherweise auch
zwischen Laeufen wenig Variation).

Alles andere unveraendert -- Budgetpunkte, Seeds, Zielkanaele, Fenstergeometrie,
Architekturen, Fingerprint-Pruefung, Exhaust-Rekonstruktion -- UND ausgewertet auf demselben
Konstantlauf-Holdout (``HOLDOUT_SEED=999``) wie der angeregte Arm: gleiche Testmenge,
verschiedene Trainingsart, damit ein Vergleich ueberhaupt aussagekraeftig ist.

``test_set``-Label der eigenen Pool-Testmenge heisst hier ``"own_pool"``, nicht ``"excited"``
wie im Zwillingsskript -- der Pool, aus dem diese Testmenge stammt, ist hier ein
Konstantlauf-Pool, ``"excited"`` waere irrefuehrend. ``data/train_constant_run_fraction``
(LA1, ``docs/04_stand_ap1.md``) traegt in jedem Record dieses Skripts 1.0 -- explizit, nicht
nur ueber das allgemeine ``data/constant_run_fraction``-Feld, das fuer beide ``test_set``-Werte
desselben Records identisch bliebe.

Checkpoints tragen den Praefix ``constant_trained_`` (``run_uid``), damit sie nicht
dieselben Dateinamen wie die Checkpoints aus ``run_arch_learning_curve_worker.py`` im selben
``--checkpoint-dir`` belegen.

Aufruf::

    python scripts/run_arch_constant_trained_learning_curve_worker.py --seed 1 \
        --output artifacts/runs/_constant_trained_learning_curve_work/_worker_seed1.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.evaluation.exhaust_reconstruction import evaluate_exhaust_reconstruction
from twinlink_ml.models import LinearARXModel, PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.checkpoints import checkpoint_path, save_checkpoint
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable

    from twinlink_ml.evaluation.experiments import GeneratedData
    from twinlink_ml.models.base import SequenceModel
    from twinlink_ml.models.composite import CompositeSequenceModel

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)
HOLDOUT_SEED = 999
"""Fester Seed des Konstantlauf-Holdout-Pools -- MUSS mit
``run_arch_learning_curve_worker.HOLDOUT_SEED`` uebereinstimmen: der Vergleich der beiden
Trainingsarten (LA3) ist nur auf derselben Testmenge aussagekraeftig."""

BUDGET_POINTS = (8, 16, 32, 64)
SEEDS_DEFAULT = (1, 2, 3, 4, 5)
HORIZON = 36  # 900 s bei sample_time_s=25.0, Befund 35

TARGET_NAMES: tuple[str, str, str] = (
    "energy_target",
    "product_moisture_kg_per_kg",
    "bed_temperature_celsius",
)
"""Identisch zu ``run_arch_learning_curve_worker.TARGET_NAMES`` -- Vergleichbarkeit."""

BASELINE_FACTORIES = {
    "persistence": PersistenceModel,
    "static": StaticRegressionModel,
    "arx": LinearARXModel,
}
ARCHITECTURE_LABELS = ("gru", "lstm")


def _train_spec() -> DataSpec:
    """Trainingspool: identisch zu ``run_arch_learning_curve_worker._excited_spec()``, bis
    auf ``constant_run_fraction=1.0`` -- der gesamte Pool besteht aus Konstantlaeufen."""
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=HORIZON,
        stride=8,
        target_names=TARGET_NAMES,
        constant_run_fraction=1.0,
    )


def _holdout_spec(train: DataSpec) -> DataSpec:
    """Konstantlauf-Pool, ``n_test`` an die Standard-Testmenge angeglichen.

    ``constant_run_fraction=1.0`` ist hier bereits der Wert aus ``train`` selbst (anders als
    im angeregten Zwillingsskript, wo dieselbe Zeile 0.10 auf 1.0 umstellt) -- die
    ``dataclasses.replace``-Zeile bleibt trotzdem explizit stehen, dieselbe Struktur wie dort,
    damit ein Diff zwischen beiden Skripten nur die tatsaechliche Abweichung zeigt
    (``_train_spec``), nicht zusaetzlich eine strukturelle.
    """
    return dataclasses.replace(
        train, n_train_max=2, n_val=2, n_test=train.n_test, constant_run_fraction=1.0
    )


def _architecture_factory(label: str) -> Callable[[], CompositeSequenceModel]:
    from twinlink_ml.models.composite import CompositeGruSequenceModel, CompositeLstmSequenceModel

    return {"gru": CompositeGruSequenceModel, "lstm": CompositeLstmSequenceModel}[label]


def parse_arguments() -> argparse.Namespace:
    """Kommandozeile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True, help="JSONL-Ausgabe dieses Jobs")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts") / "checkpoints")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(BUDGET_POINTS), help="n_train-Stuetzstellen"
    )
    parser.add_argument("--num-threads", type=int, default=1, help="torch.set_num_threads")
    return parser.parse_args()


def main() -> None:
    """Ein Seed: Pool einmal erzeugen, alle Budgetpunkte x alle Architekturen durchlaufen."""
    arguments = parse_arguments()

    import torch

    torch.set_num_threads(arguments.num_threads)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)
    config = dataclasses.replace(
        config,
        numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True),
    )

    train_spec = _train_spec()
    holdout_spec = _holdout_spec(train_spec)
    evaluator = RolloutEvaluator(horizons=(1, HORIZON))
    logger = JsonlExperimentLogger(arguments.output)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        train_data = generate_data(config, train_spec, master_seed=arguments.seed)
    if not fingerprints_equivalent(train_data.process_config.fingerprint, "bddca37f716f5cfc"):
        raise RuntimeError(
            f"process/fingerprint={train_data.process_config.fingerprint!r}, erwartet "
            "'bddca37f716f5cfc' -- Config weicht von der verifizierten BA0(c)-Grundlage ab, "
            "Abbruch vor der Simulation."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        holdout = generate_data(config, holdout_spec, master_seed=HOLDOUT_SEED)

    print(
        f"[seed={arguments.seed}] Setup fertig (Pool erzeugt, "
        f"{len(holdout.split.test)} Konstantlauf-Testlaeufe), Training startet",
        flush=True,
    )

    common_extra = {
        "master_seed": arguments.seed,
        "holdout/seed": HOLDOUT_SEED,
        "holdout/n_runs": len(holdout.split.test),
        "exhaust/outlet_duct_heat_loss_k": config.exhaust.outlet_duct_heat_loss_k,
        "numerics/enforce_gas_velocity_projection": config.numerics.enforce_gas_velocity_projection,
    }

    def _log_model(
        n_train: int,
        n_val: int,
        label: str,
        model: SequenceModel,
        training_elapsed_s: float,
        *,
        checkpoint_run_uid: str | None,
    ) -> None:
        test_sets: tuple[tuple[str, TrajectoryWindowDataset, GeneratedData], ...] = (
            ("own_pool", test_own_pool, train_data),
            ("constant_holdout", test_constant, holdout),
        )
        for test_name, test_dataset, source_data in test_sets:
            result = evaluator.evaluate(model, test_dataset)
            exhaust_result = evaluate_exhaust_reconstruction(
                model, source_data, test_dataset, horizons=(1, HORIZON)
            )
            params: dict[str, object] = {
                **train_spec.as_parameters(),
                # data/train_constant_run_fraction wiederholt data/constant_run_fraction (aus
                # derselben train_spec) bewusst redundant -- NICHT als ueberfluessig entfernen.
                # data/constant_run_fraction ist ein generisches Spec-Attribut und bleibt fuer
                # JEDEN test_set-Wert in diesem Record identisch; data/train_constant_run_fraction
                # ist die explizite, eindeutig benannte Aussage "so trainiert" (hier: 1.0) --
                # siehe Befund-Eintrag zu diesem Feld in docs/04_stand_ap1.md.
                "data/train_constant_run_fraction": train_spec.constant_run_fraction,
                "process/fingerprint": train_data.process_config.fingerprint,
                "excitation/fingerprint": train_data.schedule.metadata.get(
                    "fingerprint", "unknown"
                ),
                "split/digest": train_data.split.digest,
                **train_data.seeds.as_parameters(),
                "test_set": test_name,
                "model/label": label,
                "n_train": n_train,
                "n_val": n_val,
                "n_budget": n_train + n_val,
                "training/elapsed_s": training_elapsed_s,
                **{f"model/{key}": value for key, value in model.describe().items()},
                **{f"training/{key}": value for key, value in model.training_diagnostics().items()},
                **environment_fingerprint(),
                **common_extra,
            }
            if checkpoint_run_uid is not None:
                params["checkpoint/path"] = str(
                    checkpoint_path(arguments.checkpoint_dir, checkpoint_run_uid)
                )
            metrics = {**result.as_flat_metrics(), **exhaust_result.as_flat_metrics()}
            logger.log(
                RunRecord(
                    group="arch_learning_curve_constant_trained",
                    params=params,
                    metrics=metrics,
                    tags={
                        "model": label,
                        "group": "arch_learning_curve_constant_trained",
                        "test_set": test_name,
                        "n_train": str(n_train),
                    },
                )
            )

    for n_train in arguments.sizes:
        train, validation, test_own_pool = build_datasets(
            train_data, n_train, policy=ValidationPolicy.FIXED_POOL
        )
        n_val = 0 if validation is None else len(train_data.split.val)
        test_constant = TrajectoryWindowDataset.from_trajectory(
            holdout.trajectory,
            holdout.split,
            SplitName.TEST,
            train_spec.window,
            scalers=train.scalers,
        )

        for label, factory in BASELINE_FACTORIES.items():
            model = factory()
            start = time.perf_counter()
            model.fit(train, seed=train_data.seeds.model_seed, validation=validation)
            elapsed = time.perf_counter() - start
            _log_model(n_train, n_val, label, model, elapsed, checkpoint_run_uid=None)
            print(
                f"[seed={arguments.seed}] n_train={n_train} {label}: fertig ({elapsed:.2f}s)",
                flush=True,
            )

        for architecture in ARCHITECTURE_LABELS:
            architecture_factory = _architecture_factory(architecture)
            model = architecture_factory()
            start = time.perf_counter()
            model.fit(train, seed=train_data.seeds.model_seed, validation=validation)
            elapsed = time.perf_counter() - start

            training_run_uid = f"constant_trained_{architecture}_seed{arguments.seed}_n{n_train}"
            ckpt_path = checkpoint_path(arguments.checkpoint_dir, training_run_uid)
            save_checkpoint(
                ckpt_path,
                model,
                model_label=architecture,
                run_uid=training_run_uid,
                target_names=train.target_names,
                exogenous_names=train.exogenous_names,
                n_exogenous=len(train.exogenous_names),
                input_scaler=train.input_scaler,
                target_scaler=train.target_scaler,
                process_fingerprint=train_data.process_config.fingerprint,
                data_spec_params=train_spec.as_parameters(),
                seed_bundle_params=train_data.seeds.as_parameters(),
                git_commit=environment_fingerprint()["git/commit"],
            )
            _log_model(
                n_train, n_val, architecture, model, elapsed, checkpoint_run_uid=training_run_uid
            )
            print(
                f"[seed={arguments.seed}] n_train={n_train} {architecture}: fertig "
                f"({elapsed:.2f}s), Checkpoint {ckpt_path}",
                flush=True,
            )

    logger.close()
    print(f"[seed={arguments.seed}] alle Budgetpunkte fertig, Log {arguments.output}")


if __name__ == "__main__":
    main()
