r"""Rauschniveau als Versuchsfaktor und die Sättigungshöhe der Lernkurven (Schritt 5, 3).

Aufruf::

    python scripts/study_noise_level.py --log artifacts/runs/noise.jsonl

Zwei Zahlen, die zusammengehören:

* ``nrmse`` gegen die Messung -- das, was an der Anlage prüfbar ist.
* die **Rauschgrenze**: der Fehler, den ein perfektes Modell auf denselben Fenstern
  erreicht. Sie wird ohne Modell berechnet, indem der deterministische Anteil der Messung
  (Ausgangsgröße nach Sensordynamik, Rauschen null) als Vorhersage eingesetzt wird.

Ohne die zweite Zahl ist eine Lernkurve, die bei 0.12 abflacht, nicht als Modellgrenze
*oder* Rauschgrenze lesbar. Das Rauschniveau selbst ist ein Platzhalter (0.5 %) und offene
Frage an Anwendungspartner -- diese Studie ersetzt die Messung nicht, sie zeigt, wie dringend sie ist.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.evaluation import (
    DataPoolCache,
    DataSpec,
    RolloutEvaluator,
    run_noise_study,
)
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
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.005, 0.02])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--run-duration-s", type=float, default=28_800.0)
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Studie fahren und Tabellen drucken."""
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

    records, floors = run_noise_study(
        config,
        FACTORIES,
        spec=spec,
        plan=SeedPlan(master_seeds=tuple(arguments.seeds)),
        sizes=arguments.sizes,
        noise_levels=tuple(arguments.levels),
        evaluator=RolloutEvaluator(horizons=(1, arguments.horizon)),
        logger=logger,
        cache=cache,
    )

    print(
        f"Pool {spec.pool_size} Läufe, Δt = {spec.sample_time_s:.0f} s, "
        f"{cache.n_simulations} Simulationen für {cache.n_remeasurements} Datensätze"
    )
    for level in sorted(floors):
        print()
        print(floors[level].format())

    key_template = "nrmse_run_median/{target}/h{horizon}"
    for target in spec.target_names:
        print(f"\n--- {target}, h = {arguments.horizon}: NRMSE über Rauschniveau ---")
        labels = sorted(FACTORIES)
        print(
            f"{'σ_rel':>8}{'n_train':>9}"
            + "".join(f"{label:>12}" for label in labels)
            + f"{'Grenze':>12}"
        )
        for level in sorted(floors):
            for n_train in arguments.sizes:
                cells = ""
                for label in labels:
                    values = [
                        record.metrics[
                            key_template.format(target=target, horizon=arguments.horizon)
                        ]
                        for record in records
                        if record.params.get("model/label") == label
                        and float(record.params["noise/relative_std"]) == level
                        and int(record.params["n_train"]) == n_train
                    ]
                    cells += f"{np.median(values):>12.4f}" if values else f"{'-':>12}"
                floor = floors[level].per_run[target][arguments.horizon][0]
                print(f"{level:>8.4f}{n_train:>9}{cells}{floor:>12.4f}")

    if logger is not None:
        logger.close()
        print(f"\n{len(records)} Einträge in {arguments.log}")


if __name__ == "__main__":
    main()
