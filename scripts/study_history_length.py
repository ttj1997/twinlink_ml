r"""Wie viel Vergangenheit das Modell braucht (Schritt 6, Nebenstudie).

Aufruf::

    python scripts/study_history_length.py --log artifacts/runs/history.jsonl

Anlass: der Soft Sensor aus dem Vorprojekt (Kräuter et al.) erreicht seine Genauigkeit mit
einer Historie von **zwei Abtastschritten**, also 4 s -- die Zeitinformation erreicht ihn
nur über erste und zweite Ableitung der Inline-Messungen. Für eine Ein-Schritt-Schätzung
bei Δt = 2 s ist das plausibel. Für einen Mehrschritt-Rollout über Minuten ist die Frage
offen, und sie ist nicht durch Nachdenken zu beantworten, sondern durch Messen.

Die Studie hält **Horizont und Abtastzeit fest** und variiert allein die Fensterhistorie.
Berichtet wird in Sekunden, nicht in Schritten: nur so ist die Zahl mit der Historie des
Vorprojekts und mit den Zeitkonstanten des Prozesses vergleichbar.
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.evaluation import (
    DataSpec,
    ObservationMode,
    RolloutEvaluator,
    evaluate_point,
    generate_data,
    primary_metric,
    spec_for_mode,
)
from twinlink_ml.models.registry import build_factories
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils import JsonlExperimentLogger, SeedPlan
from twinlink_ml.utils.seeding import VarianceSource

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"


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
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--sample-time-s", type=float, default=25.0)
    parser.add_argument("--run-duration-s", type=float, default=28_800.0)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-lags", type=int, default=4)
    parser.add_argument("--mode", choices=["inline", "soft_sensor"], default="inline")
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Historienlänge variieren, Rollout-Fehler berichten."""
    arguments = parse_arguments()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)

    mode = ObservationMode(arguments.mode)
    base = spec_for_mode(
        DataSpec(
            sample_time_s=arguments.sample_time_s,
            run_duration_s=arguments.run_duration_s,
            n_train_max=arguments.n_train,
            n_val=arguments.n_train // 2,
            n_test=arguments.n_train // 2,
            horizon=arguments.horizon,
            **({} if arguments.stride is None else {"stride": arguments.stride}),
        ),
        mode,
    )
    fully_observed = mode is ObservationMode.INLINE
    labels = ("persistence", "arx", "narx") if fully_observed else ("climatology", "arx", "narx")
    factories = build_factories(labels, fully_observed=fully_observed)
    plan = SeedPlan(master_seeds=tuple(arguments.seeds))
    logger = JsonlExperimentLogger(arguments.log) if arguments.log else None
    evaluator = RolloutEvaluator(horizons=(1, arguments.horizon))
    metric = primary_metric(mode)
    records = []

    for length in arguments.lengths:
        # Die ARX-Ordnung wird mitgeführt: eine feste Ordnung 4 würde bei Historie 1 oder 2
        # abbrechen, und ein Vergleich, in dem ein Modell die kurze Historie nicht nutzen
        # *kann*, misst die Implementierung statt der Information.
        order = max(1, min(arguments.max_lags, length))
        spec = dataclasses.replace(base, history_length=length)
        local = {
            "persistence": factories.get("persistence"),
            "climatology": factories.get("climatology"),
            "arx": lambda o=order: __import__(
                "twinlink_ml.models", fromlist=["LinearARXModel"]
            ).LinearARXModel(n_lags_target=o, n_lags_exogenous=min(2, o)),
            "narx": lambda o=order: __import__(
                "twinlink_ml.models", fromlist=["RandomFeatureNARX"]
            ).RandomFeatureNARX(n_lags_target=o, n_lags_exogenous=min(2, o)),
        }
        active = {key: value for key, value in local.items() if value is not None and key in labels}
        for replicate in range(plan.n_replicates):
            bundle = plan.bundle(VarianceSource.JOINT, replicate)
            data = generate_data(config, spec, seeds=bundle)
            records += evaluate_point(
                data,
                arguments.n_train,
                active,
                evaluator=evaluator,
                logger=logger,
                group="history_length",
                extra={
                    "variance/source": VarianceSource.JOINT.value,
                    "variance/replicate": replicate,
                    "window/history_length": length,
                    "window/history_s": length * arguments.sample_time_s,
                    "observation/mode": mode.value,
                },
            )

    print(
        f"Betriebsart {mode.value}, Δt = {arguments.sample_time_s:.0f} s, "
        f"Horizont {arguments.horizon} Schritte = "
        f"{arguments.horizon * arguments.sample_time_s:.0f} s, n_train = {arguments.n_train}\n"
        f"Primärmetrik {metric}, Median über {plan.n_replicates} Seeds"
    )
    for target in base.target_names:
        print(f"\n--- {target} ---")
        print(f"{'Historie':>10}{'Schritte':>10}" + "".join(f"{name:>14}" for name in labels))
        for length in arguments.lengths:
            cells = ""
            for name in labels:
                values = [
                    record.metrics[f"{metric}/{target}/h{arguments.horizon}"]
                    for record in records
                    if record.params.get("model/label") == name
                    and int(record.params["window/history_length"]) == length
                ]
                cells += f"{np.median(values):>14.4f}" if values else f"{'-':>14}"
            print(f"{length * arguments.sample_time_s:>9.0f}s{length:>10}{cells}")

    if logger is not None:
        logger.close()
        print(f"\n{len(records)} Einträge in {arguments.log}")


if __name__ == "__main__":
    main()
