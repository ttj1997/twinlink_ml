r"""Ein (Architektur, Seed)-Job fuer den T_bed/Exhaust-Sweep (Befund 42).

Struktur identisch zu ``run_arch_constant_holdout_worker.py`` (Befund 36) -- Standard-
Produktionskonfiguration (``greybox_gab_placeholder.yaml``, ``n_train=64``, ``horizon=36``,
angeregte Testmenge PLUS fester Konstantlauf-Holdout-Pool ``HOLDOUT_SEED=999``)
unveraendert uebernommen, nur um die vierte Zielgroesse (``bed_temperature_celsius``,
expliziter Override, Befund 41f) und die End-zu-End-Exhaust-Metrik
(``evaluation.exhaust_reconstruction``, Befund 42) erweitert. Fuer jede der zwei
Testmengen wird ``evaluate_exhaust_reconstruction()`` mit dem PASSENDEN ``GeneratedData``-
Objekt aufgerufen (``excited`` fuer die angeregte Testmenge, ``holdout`` fuer den
Konstantlauf) -- die Selbstpruefung in ``exhaust_reconstruction.py`` (Befund 42, die dabei
gefundene stille Fehlpaarungs-Luecke) wuerde eine Verwechslung dieser beiden sofort mit
einem ``ValueError`` aufdecken.

Kein Checkpoint-Schreiben -- diese Aufgabe ist eine Bestandsaufnahme, keine Grundlage fuer
spaetere Weiterverwendung trainierter Gewichte. Keine Baselines (Persistenz/static/ARX) --
nicht Teil des bestaetigten Sweep-Umfangs.
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
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord, environment_fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable

    from twinlink_ml.data.windows import SequenceDataset
    from twinlink_ml.evaluation.experiments import GeneratedData
    from twinlink_ml.models.composite import CompositeSequenceModel

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)
HOLDOUT_SEED = 999
"""Fester Seed des Konstantlauf-Holdout-Pools -- identisch zu Befund 36/40, EIN Pool fuer
alle Jobs dieses Sweeps, nicht pro Seed neu gezogen."""

ARCHITECTURE_LABELS = ("gru", "lstm")
FOUR_TARGETS = (
    "energy_target",
    "product_moisture_kg_per_kg",
    "d50_um",
    "bed_temperature_celsius",
)


def _excited_spec() -> DataSpec:
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=36,
        stride=8,
        target_names=FOUR_TARGETS,
    )


def _holdout_spec(excited: DataSpec) -> DataSpec:
    """Konstantlauf-Pool, ``n_test`` an die Standard-Testmenge angeglichen (Befund 36)."""
    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def _architecture_factory(label: str) -> Callable[[], CompositeSequenceModel]:
    from twinlink_ml.models.composite import CompositeGruSequenceModel, CompositeLstmSequenceModel

    return {"gru": CompositeGruSequenceModel, "lstm": CompositeLstmSequenceModel}[label]


def parse_arguments() -> argparse.Namespace:
    """Kommandozeile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURE_LABELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True, help="JSONL-Ausgabe dieses Jobs")
    parser.add_argument("--num-threads", type=int, default=1, help="torch.set_num_threads")
    return parser.parse_args()


def main() -> None:
    """Ein Job: trainieren, zwei Testmengen (je mit Vier-Ziele-NRMSE + Exhaust-Metrik) auswerten."""
    arguments = parse_arguments()

    import torch

    torch.set_num_threads(arguments.num_threads)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)

    excited_spec = _excited_spec()
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

    print(
        f"[{arguments.architecture} seed={arguments.seed}] Setup fertig "
        f"(Daten generiert, {len(holdout.split.test)} Konstantlauf-Testlaeufe), "
        "Training startet",
        flush=True,
    )

    common_extra = {
        "master_seed": arguments.seed,
        "holdout/seed": HOLDOUT_SEED,
        "holdout/n_runs": len(holdout.split.test),
        "study/note": "befund42_tbed_exhaust_sweep",
    }

    start = time.perf_counter()
    architecture_factory = _architecture_factory(arguments.architecture)
    model = architecture_factory()
    model.fit(train, seed=excited.seeds.model_seed, validation=validation)
    elapsed = time.perf_counter() - start

    epoch_counts = {target: len(history) for target, history in model.training_history.items()}

    test_sets: tuple[tuple[str, SequenceDataset, GeneratedData], ...] = (
        ("excited", test_excited, excited),
        ("constant_holdout", test_constant, holdout),
    )
    for test_name, test_dataset, source_data in test_sets:
        result = evaluator.evaluate(model, test_dataset)
        exhaust_result = evaluate_exhaust_reconstruction(
            model, source_data, test_dataset, horizons=(1, excited_spec.horizon)
        )
        params: dict[str, object] = {
            **excited_spec.as_parameters(),
            "process/fingerprint": excited.process_config.fingerprint,
            "excitation/fingerprint": excited.schedule.metadata.get("fingerprint", "unknown"),
            "split/digest": excited.split.digest,
            **excited.seeds.as_parameters(),
            "test_set": test_name,
            "model/label": arguments.architecture,
            "training/elapsed_s": elapsed,
            **{f"training/epochs/{target}": count for target, count in epoch_counts.items()},
            **environment_fingerprint(),
            **common_extra,
        }
        logger.log(
            RunRecord(
                group="befund42_tbed_exhaust_sweep",
                params=params,
                metrics={**result.as_flat_metrics(), **exhaust_result.as_flat_metrics()},
                tags={
                    "model": arguments.architecture,
                    "group": "befund42_tbed_exhaust_sweep",
                    "test_set": test_name,
                },
            )
        )

    logger.close()
    print(
        f"[{arguments.architecture} seed={arguments.seed}] fertig, "
        f"Training {elapsed:.1f}s, Log {arguments.output}"
    )


if __name__ == "__main__":
    main()
