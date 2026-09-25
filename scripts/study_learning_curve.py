r"""Lernkurve: Fehler über Anzahl Trainingsläufe, mehrere Seeds, alle Baselines.

Aufruf::

    python scripts/study_learning_curve.py --log artifacts/curve.jsonl --seeds 1 2 3

Primäres Ergebnisformat von AP 1 (Meilenstein M2). Ausgegeben wird der Median über die
Seeds mit Spannweite, nicht der Mittelwert: bei drei bis fünf Seeds ist die
Normalverteilungsannahme nicht prüfbar.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from twinlink_ml.evaluation import (
    DataSpec,
    RolloutEvaluator,
    ValidationPolicy,
    run_learning_curve,
    summarise_curve,
)
from twinlink_ml.models import (
    LinearARXModel,
    PersistenceModel,
    RandomFeatureNARX,
    StaticRegressionModel,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.tracking import JsonlExperimentLogger

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_gab_placeholder.yaml"
)

FACTORIES = {
    "persistence": PersistenceModel,
    "static": StaticRegressionModel,
    "arx": LinearARXModel,
    "narx": RandomFeatureNARX,
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
    parser.add_argument("--n-train-max", type=int, default=256)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument(
        "--policy",
        choices=[policy.value for policy in ValidationPolicy],
        default=ValidationPolicy.FIXED_POOL.value,
    )
    return parser.parse_args()


def main() -> None:
    """Lernkurve rechnen und ausgeben."""
    arguments = parse_arguments()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)

    spec = DataSpec(
        n_train_max=arguments.n_train_max,
        n_val=arguments.n_train_max // 2,
        n_test=arguments.n_train_max // 2,
        horizon=arguments.horizon,
    )
    logger = JsonlExperimentLogger(arguments.log) if arguments.log else None
    records = run_learning_curve(
        config,
        FACTORIES,
        spec=spec,
        sizes=arguments.sizes,
        master_seeds=arguments.seeds,
        policy=ValidationPolicy(arguments.policy),
        evaluator=RolloutEvaluator(horizons=(1, arguments.horizon)),
        logger=logger,
    )

    print(
        f"Δt = {spec.sample_time_s:.0f} s, Pool {spec.pool_size} Läufe, "
        f"{spec.n_test} Testläufe, Seeds {arguments.seeds}, "
        f"Validierung: {arguments.policy}"
    )
    for target in spec.target_names:
        for horizon in (1, arguments.horizon):
            print(f"\n--- {target}, h={horizon} — NRMSE Median [Min .. Max] über Seeds ---")
            curve = summarise_curve(records, target=target, horizon=horizon)
            labels = sorted(curve)
            header = "".join(f"{label:>26}" for label in labels)
            print(f"{'n_train':>8}{header}")
            for n_train in arguments.sizes:
                cells = ""
                for label in labels:
                    entry = curve[label].get(n_train)
                    cells += (
                        f"{entry[0]:>12.4f} [{entry[1]:.3f}..{entry[2]:.3f}]"
                        if entry
                        else f"{'-':>26}"
                    )
                print(f"{n_train:>8}{cells}")

    if logger is not None:
        logger.close()
        print(f"\n{len(records)} Einträge in {arguments.log}")


if __name__ == "__main__":
    main()
