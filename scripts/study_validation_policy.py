r"""FIXED_POOL gegen NONE: was Early Stopping an den kleinen Stützstellen wert ist.

Aufruf::

    python scripts/study_validation_policy.py --log artifacts/runs/policy.jsonl

``FIXED_POOL`` ist die faire Betriebsart für den Architekturvergleich (fester
Validierungs-Pool, alle Architekturen gleich behandelt). ``NONE`` liefert die Zahl, die im
M2-Bericht als Trainingsdatenbedarf stehen darf -- mit 32 zusätzlichen Validierungsläufen
ist eine Aussage über 8 Trainingsläufe irreführend.

**Erwartung für die Baselines: die Differenz ist exakt Null.** Keine der drei benutzt den
Validierungssplit. Das ist kein Leerlauf, sondern eine Leakage-Regressionsprüfung: eine
Differenz ungleich Null wäre der Beweis, dass Validierungsläufe ins Training sickern. Die
inhaltliche Zahl entsteht mit dem ersten früh abbrechenden Modell (Schritt 6).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from twinlink_ml.evaluation import (
    DataPoolCache,
    DataSpec,
    RolloutEvaluator,
    run_validation_policy_comparison,
    summarise_policy_gap,
)
from twinlink_ml.evaluation.resolution import resolution_thresholds
from twinlink_ml.models import LinearARXModel, PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils import JsonlExperimentLogger, SeedPlan

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

FACTORIES = {
    "persistence": PersistenceModel,
    "static": StaticRegressionModel,
    "arx": LinearARXModel,
}


def parse_arguments() -> argparse.Namespace:
    """Kommandozeile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Prozess-Config. Erlaubt einen Lauf gegen eine korrigierte "
        "Parametrisierung (z. B. conf/process/greybox_lab_moisture.yaml) "
        "ohne Code-Aenderung.",
    )
    parser.add_argument("--n-train-max", type=int, default=64)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--run-duration-s", type=float, default=28_800.0)
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Beide Betriebsarten fahren und die Differenz gegen die Auflösungsschwelle halten."""
    arguments = parse_arguments()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)

    spec = DataSpec(
        run_duration_s=arguments.run_duration_s,
        n_train_max=arguments.n_train_max,
        n_val=arguments.n_train_max // 2,
        n_test=arguments.n_train_max // 2,
        horizon=arguments.horizon,
    )
    cache = DataPoolCache(config, spec)
    logger = JsonlExperimentLogger(arguments.log) if arguments.log else None

    records = run_validation_policy_comparison(
        config,
        FACTORIES,
        spec=spec,
        plan=SeedPlan(master_seeds=tuple(arguments.seeds)),
        sizes=arguments.sizes,
        evaluator=RolloutEvaluator(horizons=(1, arguments.horizon)),
        logger=logger,
        cache=cache,
    )

    for target in spec.target_names:
        print(f"\n--- {target}, h = {arguments.horizon} ---")
        try:
            thresholds = resolution_thresholds(records, target=target, horizon=arguments.horizon)
        except ValueError:
            thresholds = {}
        print(
            f"{'Modell':>14}{'n_train':>9}{'FIXED_POOL':>13}"
            f"{'NONE':>10}{'Delta':>10}{'Schwelle':>11}"
        )
        gaps = summarise_policy_gap(records, target=target, horizon=arguments.horizon)
        for (label, n_train), (fixed, none, difference) in gaps.items():
            threshold = thresholds.get(n_train)
            band = f"{threshold * fixed:>11.4f}" if threshold is not None else f"{'-':>11}"
            print(f"{label:>14}{n_train:>9}{fixed:>13.4f}{none:>10.4f}{difference:>10.4f}{band}")

    if logger is not None:
        logger.close()
        print(f"\n{len(records)} Einträge in {arguments.log}")


if __name__ == "__main__":
    main()
