r"""Ein Seed-Job: Lernkurve ueber n_train in {8,16,32,64}, alle fuenf Architekturen.

Ersetzt (nicht: korrigiert) ``arch_composite.jsonl`` (Befund 33: git/commit="unknown",
Config nicht rekonstruierbar) durch einen Datensatz unter dokumentierter, git-verfolgter
Konfiguration. Struktur angelehnt an ``run_arch_constant_holdout_worker.py`` (zwei
Testmengen je trainiertem Modell: angeregter Pool-Holdout und fixer Konstantlauf-Holdout),
erweitert um die Budgetstaffelung aus ``evaluation.experiments.run_learning_curve``
(verschachtelte Praefixe ``train[:n_train]`` derselben Pool-Realisierung je Seed, siehe
``RunSplit.subset``).

Ein Prozess je Seed (nicht je (Seed, n_train)): die Datenerzeugung (Pool-Simulation,
28800 s / 25 s = 1152 Schritte, 128 Laeufe) ist teuer und wird pro Seed genau einmal
gemacht, dann fuer alle vier Budgetpunkte wiederverwendet -- exakt das Verschachtelungsprinzip,
das die Zweierpotenzen-Vorgabe verlangt.

Zielkanaele EXPLIZIT gesetzt (``TARGET_NAMES`` unten): ``DataSpec.target_names`` default
reduziert auf zwei Groessen (``energy_target``, ``product_moisture_kg_per_kg``, siehe
``evaluation/experiments.py``) -- dieser Lauf braucht zusaetzlich
``bed_temperature_celsius``, sowohl als eigener Zielkanal als auch als Voraussetzung fuer
``evaluate_exhaust_reconstruction`` (braucht ``bed_temperature_celsius``/
``product_moisture_kg_per_kg`` in ``test.target_names``, siehe
``evaluation/exhaust_reconstruction.py``). ``outlet_temperature_celsius`` ist selbst KEIN
gueltiger Eintrag in ``target_names`` -- es ist die algebraische Rekonstruktion aus
vorhergesagtem ``bed_temperature_celsius``/``product_moisture_kg_per_kg`` ueber
``exhaust.reconstruct_exhaust_state`` (``exhaust/outlet_duct_heat_loss_k`` steuert den
Versatz, hier unveraendert 0.0). Temperaturmetriken werden deshalb ZUSAETZLICH unter dem
Abluftnamen geloggt (``exhaust_*``-Praefix), mit explizitem ``exhaust/outlet_duct_heat_loss_k``
im Record.

``NumericsConfig.enforce_gas_velocity_projection`` wird hier EXPLIZIT auf ``True`` gesetzt
(Default in ``NumericsConfig`` und in der YAML-Datei ist AUS) -- Fingerprint der
resultierenden, tatsaechlich angewendeten Config (nach ``sample_time_s=25.0``-Ueberschreibung
durch ``generate_data``) ist ``bddca37f716f5cfc`` (vor dem Pseudonymisierungs-Schnitt
2026-09-25: ``2d2679eeed716577``, siehe ``docs/ARCHIV.md``; siehe auch ``docs/04_stand_ap1.md``,
Fingerprint-Historie).

Aufruf::

    python scripts/run_arch_learning_curve_worker.py --seed 1 \
        --output artifacts/runs/_learning_curve_work/_worker_seed1.jsonl
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
from twinlink_ml.models.capacity_matched import CAPACITY_MATCHED_CONFIGS
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
"""Fester Seed des Konstantlauf-Holdout-Pools -- identisch zu
``run_arch_constant_holdout_worker.HOLDOUT_SEED``: ein Pool fuer alle Seed-Jobs, nicht pro
Seed neu gezogen."""

BUDGET_POINTS = (8, 16, 32, 64)
SEEDS_DEFAULT = (1, 2, 3, 4, 5)
HORIZON = 36  # 900 s bei sample_time_s=25.0, Befund 35

TARGET_NAMES: tuple[str, str, str] = (
    "energy_target",
    "product_moisture_kg_per_kg",
    "bed_temperature_celsius",
)
"""EXPLIZIT gesetzt (BA0d) -- der ``DataSpec.target_names``-Default deckt nur zwei
Zielgroessen ab. ``bed_temperature_celsius`` ist der Simulator-Output; der Abluftname
``outlet_temperature_celsius`` ist keine gueltige Zielgroesse (algebraische Rekonstruktion,
siehe Moduldocstring)."""

BASELINE_FACTORIES = {
    "persistence": PersistenceModel,
    "static": StaticRegressionModel,
    "arx": LinearARXModel,
}
ARCHITECTURE_LABELS = ("gru", "lstm", "transformer", "xlstm", "mamba")
"""Transformer/xLSTM/Mamba ergaenzt (OA1-Entscheidung): dieselbe Pool-/Fingerprint-/
Zielkanal-Logik oben gilt fuer alle acht Architekturen gemeinsam -- Wiederverwendung des
Protokolls war der Zweck der Erweiterung, statt separater Reduced-Comparison-/
Constant-Holdout-Skripte je Architektur (die kein n_budget kennen, siehe
docs/04_stand_ap1.md, Befund zur Transformer/xLSTM/Mamba-Lernkurvenerweiterung).
Kapazitaets-Defaults der drei neuen Kerne (Transformer/Mamba ``hidden_size=16``, xLSTM
``hidden_size=64`` wie GRU/LSTM) kommen unveraendert aus ``composite.py``."""


CAPACITY_MATCHED_LABELS = tuple(CAPACITY_MATCHED_CONFIGS)
"""Befund 67 (SX3): capacity-matched xLSTM/Mamba variants and the small-LSTM control. Selectable
with ``--architectures`` but NOT part of the default selection (``ARCHITECTURE_LABELS``), so the
protocol of Befund 51/58 and its tests are unchanged."""
SELECTABLE_LABELS = ARCHITECTURE_LABELS + CAPACITY_MATCHED_LABELS


def _excited_spec() -> DataSpec:
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
    )


def _holdout_spec(excited: DataSpec) -> DataSpec:
    """Konstantlauf-Pool, ``n_test`` an die Standard-Testmenge angeglichen.

    ``target_names`` wird ueber ``dataclasses.replace`` von ``excited`` uebernommen, nicht
    erneut aus dem Default konstruiert -- sonst waere genau der Fehler wieder da, den
    BA0d vermeiden soll.
    """
    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def _architecture_factory(label: str) -> Callable[[], CompositeSequenceModel]:
    from twinlink_ml.models.composite import (
        CompositeGruSequenceModel,
        CompositeLstmSequenceModel,
        CompositeMambaSequenceModel,
        CompositeTransformerSequenceModel,
        CompositeXLSTMSequenceModel,
    )

    if label in CAPACITY_MATCHED_CONFIGS:
        from twinlink_ml.models.composite import CompositeSequenceModel

        config = CAPACITY_MATCHED_CONFIGS[label]
        return lambda: CompositeSequenceModel(config)

    return {
        "gru": CompositeGruSequenceModel,
        "lstm": CompositeLstmSequenceModel,
        "transformer": CompositeTransformerSequenceModel,
        "xlstm": CompositeXLSTMSequenceModel,
        "mamba": CompositeMambaSequenceModel,
    }[label]


def _build_model(
    architecture_factory: Callable[[], CompositeSequenceModel], net_max_epochs: int
) -> CompositeSequenceModel:
    """Modell ueber die Architektur-Fabrik bauen, ``max_epochs`` bei Abweichung ersetzt.

    Ruft die Fabrik ohne Argumente auf, um die architekturspezifischen Standardwerte
    (z. B. Transformer/Mamba ``hidden_size=16``, siehe ``composite.py``) nicht hier
    zu duplizieren -- nur ``max_epochs`` wird bei Abweichung vom Standard (200) ueber
    ``dataclasses.replace`` ersetzt, alle anderen Konfigurationswerte bleiben die der
    Fabrik. Bei ``net_max_epochs == 200`` (Default von ``--net-max-epochs``) identisch
    zum bisherigen Verhalten.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel

    model = architecture_factory()
    if net_max_epochs == model.config.max_epochs:
        return model
    return CompositeSequenceModel(dataclasses.replace(model.config, max_epochs=net_max_epochs))


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
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        choices=SELECTABLE_LABELS,
        default=list(ARCHITECTURE_LABELS),
        help="Teilmenge von ARCHITECTURE_LABELS (Default: alle) -- z. B. nur die neu "
        "hinzugekommenen Kerne rechnen, ohne GRU/LSTM/Baselines erneut zu erzeugen, die "
        "unter demselben Protokoll bereits in einem anderen Datensatz vorliegen.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="persistence/static/arx NICHT rechnen (Default: rechnen) -- fuer denselben "
        "Fall wie --architectures: schon vorhandene, protokollgleiche Baseline-Records "
        "nicht erneut erzeugen.",
    )
    parser.add_argument(
        "--net-max-epochs",
        type=int,
        default=200,
        help="TorchSequenceConfig.max_epochs fuer alle Architektur-Fits dieses Jobs "
        "(Default: 200, der Dataclass-Standard -- unveraendertes Verhalten). Betrifft nur "
        "die neuronalen Architekturen, nicht persistence/static/arx (kein Epochenbegriff). "
        "Einzige zulaessige Abweichung vom bitgenauen Protokoll aus Befund 51/58, siehe "
        "Aufrufer.",
    )
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

    excited_spec = _excited_spec()
    holdout_spec = _holdout_spec(excited_spec)
    evaluator = RolloutEvaluator(horizons=(1, HORIZON))
    logger = JsonlExperimentLogger(arguments.output)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        excited = generate_data(config, excited_spec, master_seed=arguments.seed)
    if not fingerprints_equivalent(excited.process_config.fingerprint, "bddca37f716f5cfc"):
        raise RuntimeError(
            f"process/fingerprint={excited.process_config.fingerprint!r}, erwartet "
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
            ("excited", test_excited, excited),
            ("constant_holdout", test_constant, holdout),
        )
        for test_name, test_dataset, source_data in test_sets:
            result = evaluator.evaluate(model, test_dataset)
            exhaust_result = evaluate_exhaust_reconstruction(
                model, source_data, test_dataset, horizons=(1, HORIZON)
            )
            params: dict[str, object] = {
                **excited_spec.as_parameters(),
                # data/train_constant_run_fraction wiederholt data/constant_run_fraction
                # (aus derselben excited_spec) bewusst redundant -- NICHT als ueberfluessig
                # entfernen. data/constant_run_fraction ist ein generisches Spec-Attribut und
                # bleibt fuer JEDEN test_set-Wert in diesem Record identisch (auch fuer
                # test_set="constant_holdout"), sagt also fuer sich genommen nichts ueber die
                # Trainingsart aus. data/train_constant_run_fraction ist die explizite,
                # eindeutig benannte Aussage "so trainiert" -- ohne sie waere ein
                # konstant-trainierter Lauf von einem angeregt-trainierten anhand des Records
                # nicht unterscheidbar (siehe Befund-Eintrag zu diesem Feld in
                # docs/04_stand_ap1.md).
                "data/train_constant_run_fraction": excited_spec.constant_run_fraction,
                "process/fingerprint": excited.process_config.fingerprint,
                "excitation/fingerprint": excited.schedule.metadata.get("fingerprint", "unknown"),
                "split/digest": excited.split.digest,
                **excited.seeds.as_parameters(),
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
                    group="arch_learning_curve_projected",
                    params=params,
                    metrics=metrics,
                    tags={
                        "model": label,
                        "group": "arch_learning_curve_projected",
                        "test_set": test_name,
                        "n_train": str(n_train),
                    },
                )
            )

    for n_train in arguments.sizes:
        train, validation, test_excited = build_datasets(
            excited, n_train, policy=ValidationPolicy.FIXED_POOL
        )
        n_val = 0 if validation is None else len(excited.split.val)
        test_constant = TrajectoryWindowDataset.from_trajectory(
            holdout.trajectory,
            holdout.split,
            SplitName.TEST,
            excited_spec.window,
            scalers=train.scalers,
        )

        if not arguments.skip_baselines:
            for label, factory in BASELINE_FACTORIES.items():
                model = factory()
                start = time.perf_counter()
                model.fit(train, seed=excited.seeds.model_seed, validation=validation)
                elapsed = time.perf_counter() - start
                _log_model(n_train, n_val, label, model, elapsed, checkpoint_run_uid=None)
                print(
                    f"[seed={arguments.seed}] n_train={n_train} {label}: fertig ({elapsed:.2f}s)",
                    flush=True,
                )

        for architecture in arguments.architectures:
            architecture_factory = _architecture_factory(architecture)
            model = _build_model(architecture_factory, arguments.net_max_epochs)
            start = time.perf_counter()
            model.fit(train, seed=excited.seeds.model_seed, validation=validation)
            elapsed = time.perf_counter() - start

            training_run_uid = f"{architecture}_seed{arguments.seed}_n{n_train}"
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
                process_fingerprint=excited.process_config.fingerprint,
                data_spec_params=excited_spec.as_parameters(),
                seed_bundle_params=excited.seeds.as_parameters(),
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
