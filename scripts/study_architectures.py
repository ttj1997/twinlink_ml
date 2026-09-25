r"""Architekturvergleich (Arbeitsplan Schritt 6), beide Beobachtbarkeits-Betriebsarten.

Aufruf::

    python scripts/study_architectures.py --mode both --log artifacts/runs/architectures.jsonl

Zwei Dinge, die dieses Skript anders macht als ein üblicher Vergleich:

1. **Jede Differenz wird gegen die Auflösungsschwelle gehalten** (Schritt 5). Eine
   Verbesserung unterhalb der Schwelle wird als "kein nachweisbarer Unterschied" gemeldet
   und nicht als Prozentzahl, die dann in einen Vortrag wandert.
2. **Beide Betriebsarten**: mit und ohne Inline-Messung der Qualitätsgrößen. Ob die
   Produktfeuchte an der Zielanlage inline gemessen wird, ist offen; die Antwort ändert,
   welche Baseline überhaupt definiert ist (Persistenz braucht die Messung,
   Mittelwertvorhersage nicht).

Die Referenz, die zu schlagen ist, hängt von der Zielgröße ab und ist keine Wahl:
``energy_target`` gegen den linearen ARX, ``product_moisture_kg_per_kg`` gegen die
Persistenz (inline) bzw. die Mittelwertvorhersage (Soft Sensor). Bei der Feuchte ist der
ARX schlechter als die Persistenz -- ihn dort als Referenz zu nehmen wäre eine zu niedrige
Hürde.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from twinlink_ml.evaluation import (
    DataSpec,
    ObservationMode,
    RolloutEvaluator,
    primary_metric,
    run_architecture_comparison,
    spec_for_mode,
)
from twinlink_ml.evaluation.resolution import (
    architecture_verdicts,
    resolution_report,
    select,
)
from twinlink_ml.models.registry import (
    MODEL_SLOTS,
    available_labels,
    build_factories,
    describe_slots,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils import JsonlExperimentLogger, SeedPlan

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

REFERENCE_BY_TARGET = {
    "energy_target": "arx",
    "product_moisture_kg_per_kg": None,  # parameterfreie Referenz, siehe unten
    "d50_um": None,
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
    parser.add_argument("--mode", choices=["inline", "soft_sensor", "both"], default="both")
    parser.add_argument("--n-train-max", type=int, default=64)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--run-duration-s", type=float, default=28_800.0)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Vergleich fahren und Verdikte drucken."""
    arguments = parse_arguments()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)

    print(describe_slots())
    modes = (
        (ObservationMode.INLINE, ObservationMode.SOFT_SENSOR)
        if arguments.mode == "both"
        else (ObservationMode(arguments.mode),)
    )
    base = DataSpec(
        run_duration_s=arguments.run_duration_s,
        n_train_max=arguments.n_train_max,
        n_val=arguments.n_train_max // 2,
        n_test=arguments.n_train_max // 2,
        horizon=arguments.horizon,
    )
    plan = SeedPlan(master_seeds=tuple(arguments.seeds))
    logger = JsonlExperimentLogger(arguments.log) if arguments.log else None
    evaluator = RolloutEvaluator(horizons=(1, arguments.horizon))
    records = []

    by_label = {slot.label: slot for slot in MODEL_SLOTS}

    for mode in modes:
        fully_observed = mode is ObservationMode.INLINE
        if arguments.models:
            # Explizite Liste kann Modelle enthalten, die nur in einer Betriebsart gelten
            # (z. B. persistence nur INLINE, climatology nur sinnvoll referenziert im
            # SOFT_SENSOR-Fall). Ohne Filterung würde build_factories() bei --mode both
            # in der zweiten Betriebsart abbrechen -- laut, aber nach bereits berechneten
            # und schon geloggten Punkten der ersten Betriebsart. Die Filterung hier ist
            # bewusst NICHT still: jedes ausgelassene Modell wird genannt.
            labels, skipped = [], []
            for label in arguments.models:
                slot = by_label.get(label)
                if slot is None:
                    raise KeyError(f"unbekanntes Modell-Label {label!r}")
                if not fully_observed and slot.needs_full_observation:
                    skipped.append(label)
                else:
                    labels.append(label)
            if skipped:
                print(
                    f"  ({mode.value}: {', '.join(skipped)} übersprungen -- braucht "
                    "Inline-Messung, hier nicht gegeben)"
                )
            labels = tuple(labels)
        else:
            labels = available_labels(fully_observed=fully_observed)
        factories = build_factories(labels, fully_observed=fully_observed)
        print(f"\n=== Betriebsart {mode.value}: {', '.join(factories)} ===")

        def report_progress(
            *,
            source: str,
            replicate: int,
            n_train: int,
            mode: ObservationMode = mode,
        ) -> None:
            print(
                f"  [{mode.value}] Achse {source}  Replikat {replicate + 1}/"
                f"{plan.n_replicates}  n_train={n_train}",
                flush=True,
            )

        records += run_architecture_comparison(
            config,
            factories,
            spec=base,
            plan=plan,
            sizes=arguments.sizes,
            modes=(mode,),
            evaluator=evaluator,
            logger=logger,
            on_point=report_progress,
        )

    for mode in modes:
        subset = select(records, **{"observation/mode": mode.value})
        reference_free = "persistence" if mode is ObservationMode.INLINE else "climatology"
        spec = spec_for_mode(base, mode)
        metric = primary_metric(mode)
        print(f"\n\n############ Betriebsart {mode.value} ############")
        print(f"Primärmetrik: {metric} (siehe evaluation.protocol.primary_metric)")
        for target in spec.target_names:
            print()
            print(
                resolution_report(
                    subset,
                    target=target,
                    horizon=arguments.horizon,
                    reference_label=reference_free,
                    metric=metric,
                )
            )
            reference = REFERENCE_BY_TARGET.get(target) or reference_free
            print(f"\nVerdikte gegen {reference}:")
            for verdict in architecture_verdicts(
                subset,
                target=target,
                horizon=arguments.horizon,
                reference_label=reference,
                resolution_label=reference_free,
                metric=metric,
            ):
                print(f"  n={verdict.n_train:>3}  {verdict.statement}")

    if logger is not None:
        logger.close()
        print(f"\n{len(records)} Einträge in {arguments.log}")


if __name__ == "__main__":
    main()
