r"""Diagnose zu Befund 25 (``04_stand_ap1.md``): negativer Transfer im Soft-Sensor-Fall.

Beobachtung, die dieses Skript prüft
-------------------------------------
Im Soft-Sensor-Betrieb ist ``energy_target`` bei GRU und LSTM in *jeder* Stützstelle
nachweisbar schlechter als ARX, obwohl dieselben Kerne dieselbe Zielgröße im Inline-Fall
klar besser treffen als ARX (siehe ``artifacts/runs/arch_gru.jsonl``,
``observation/mode``). ``src/twinlink_ml/models/torch_sequence.py`` trainiert alle
Zielgrößen mit einem gemeinsamen Encoder und einem einzigen Skalarverlust
(``_weighted_mse``), gemittelt **ohne** Gewichtung pro Zielgröße -- bestätigt durch
Lektüre des Trainers, nicht durch dieses Skript. Im Soft-Sensor-Betrieb sind
``product_moisture_kg_per_kg`` und ``d50_um`` für jedes bisher geprüfte Modell praktisch
unlösbar (Climatology-NRMSE dort nahe 1.0). Die Hypothese: das Gradientenrauschen dieser
beiden schweren, kaum lernbaren Teilaufgaben verschlechtert über den geteilten Encoder die
sonst gute Lösung für ``energy_target`` -- negativer Transfer zwischen Teilaufgaben
ungleicher Schwierigkeit.

Aufbau
------
Drei GRU-Läufe auf **identischem** Datenpool und Seed je Wiederholung, Soft-Sensor-Betrieb,
``n_train = 32``:

a) wie im bestehenden Architekturvergleich: ``target_names`` = alle drei Zielgrößen, ein
   gemeinsames Netz (``torch_sequence.GruSequenceModel``).
b) ``target_names`` = nur ``("energy_target",)`` -- der Kopf sieht die beiden schweren
   Teilaufgaben gar nicht.
c) ``CompositeSequenceModel`` (Befund 26, Option 2): ein eigenes Netz je Zielgröße, alle
   drei Zielgrößen wie in (a), aber ohne geteilten Zustand.

(a) und (b) teilen Trajektorie, Split und Modell-Seed (``dataclasses.replace`` auf
demselben :class:`~twinlink_ml.evaluation.GeneratedData`); nur ``spec.target_names``
unterscheidet sich. Eine Verbesserung in (b) kann damit nicht aus einem anderen
Datenpool oder einer anderen Initialisierung kommen, nur aus der entfernten
Mitverlustfunktion. (c) läuft auf demselben Datenpool wie (a), mit derselben
``target_names``-Menge -- der einzige Unterschied ist die Architektur. Referenzwerte
(ARX, Climatology) laufen auf demselben Pool mit, ausschließlich zur Einordnung.

Aufruf::

    python scripts/diagnose_soft_sensor_interference.py --seeds 1 2 3

``--gradient-clip`` erlaubt die Fortsetzung aus Befund 26 (04_stand_ap1.md): ob das
globale ``clip_grad_norm_`` über alle Netzparameter (statt getrennt je Zielgruppe) an der
schlechten Leistung von Fall (a) mitwirkt, lässt sich prüfen, indem die Clip-Norm so hoch
gesetzt wird, dass sie in der Praxis nicht mehr greift (z.B. 100.0 statt Voreinstellung
1.0) -- bewusst nicht deaktiviert, um keine Divergenz als neues Störsignal einzuführen::

    python scripts/diagnose_soft_sensor_interference.py --seeds 1 2 3 --gradient-clip 100.0

Fall (a) und Fall (c) laufen in **derselben** Skriptausführung, damit die
Trainingszeit-Mehrkosten von (c) gegenüber (a) auf derselben Maschine, derselben Last und
demselben Zeitpunkt gemessen sind -- ein Vergleich über zwei getrennte Läufe wäre durch
Hintergrundlast auf der Maschine verzerrbar.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.evaluation import (
    DataSpec,
    GeneratedData,
    ObservationMode,
    RolloutEvaluator,
    ValidationPolicy,
    build_datasets,
    generate_data,
    spec_for_mode,
)
from twinlink_ml.models.base import SequenceModel
from twinlink_ml.models.baselines import ClimatologyModel, LinearARXModel
from twinlink_ml.models.composite import CompositeGruSequenceModel
from twinlink_ml.models.torch_sequence import GruSequenceModel, TorchSequenceConfig
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

HORIZON = 24
TARGET = "energy_target"
ALL_TARGETS = ("energy_target", "product_moisture_kg_per_kg", "d50_um")


def parse_arguments() -> argparse.Namespace:
    """Kommandozeile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--run-duration-s", type=float, default=28_800.0)
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="An beide GRU-Läufe (a und b) durchgereicht, siehe Befund 26 Fortsetzung "
        "in 04_stand_ap1.md: mit einem hohen Wert (z.B. 100.0) wirkt clip_grad_norm_ "
        "de facto nicht mehr, ohne das Clipping ganz abzuschalten.",
    )
    return parser.parse_args()


def _soft_sensor_variant(base: DataSpec, target_names: tuple[str, ...]) -> DataSpec:
    """Soft-Sensor-Spezifikation für eine gegebene Zielgrößenmenge.

    ``spec_for_mode`` beobachtet alles außer den beiden Qualitätsgrößen; für
    ``target_names = ("energy_target",)`` ist das ohnehin die einzige vorhandene
    Zielgröße, das Verhalten bleibt also konsistent zwischen (a) und (b).
    """
    variant = dataclasses.replace(base, target_names=target_names)
    return spec_for_mode(variant, ObservationMode.SOFT_SENSOR)


def _fit_and_evaluate(
    data: GeneratedData, n_train: int, model: SequenceModel, evaluator: RolloutEvaluator
) -> tuple[dict[int, float], object, float]:
    """Fit, auswerten, und die reine ``fit``-Zeit zurückgeben (für den Kostenvergleich)."""
    train, validation, test = build_datasets(data, n_train, policy=ValidationPolicy.FIXED_POOL)
    start = time.perf_counter()
    model.fit(train, seed=data.seeds.model_seed, validation=validation)
    fit_seconds = time.perf_counter() - start
    result = evaluator.evaluate(model, test)
    return result.nrmse_by_horizon[TARGET], model, fit_seconds


def main() -> None:
    """Beide Varianten über mehrere Seeds fahren und energy_target vergleichen."""
    arguments = parse_arguments()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        process_config = load_process_config(arguments.config)

    base = DataSpec(
        run_duration_s=arguments.run_duration_s,
        n_train_max=arguments.n_train,
        n_val=arguments.n_train // 2,
        n_test=arguments.n_train // 2,
        horizon=HORIZON,
    )
    evaluator = RolloutEvaluator(horizons=(1, HORIZON))

    case_a_nrmse: list[float] = []
    case_b_nrmse: list[float] = []
    case_c_nrmse: list[float] = []
    arx_nrmse: list[float] = []
    climatology_nrmse: list[float] = []
    fit_seconds_a: list[float] = []
    fit_seconds_c: list[float] = []
    epochs_a: list[int] = []
    epochs_c: list[dict[str, int]] = []
    last_history_a: tuple[dict[str, float], ...] = ()

    for seed in arguments.seeds:
        print(f"\n=== seed {seed} ===")
        pool_spec = dataclasses.replace(
            base, n_train_max=base.n_train_max, n_val=base.n_val, n_test=base.n_test
        )
        data_all = generate_data(process_config, pool_spec, master_seed=seed)

        data_a = dataclasses.replace(data_all, spec=_soft_sensor_variant(base, ALL_TARGETS))
        data_b = dataclasses.replace(data_all, spec=_soft_sensor_variant(base, (TARGET,)))

        model_a = GruSequenceModel(
            TorchSequenceConfig(core="gru", gradient_clip=arguments.gradient_clip)
        )
        nrmse_a, model_a, seconds_a = _fit_and_evaluate(
            data_a, arguments.n_train, model_a, evaluator
        )
        print(
            f"  (a) alle drei Zielgrößen:      nrmse[{TARGET}, h{HORIZON}] = "
            f"{nrmse_a[HORIZON]:.4f}  ({seconds_a:.1f}s fit, "
            f"{len(model_a.training_history)} Epochen)"
        )
        case_a_nrmse.append(nrmse_a[HORIZON])
        fit_seconds_a.append(seconds_a)
        epochs_a.append(len(model_a.training_history))
        last_history_a = model_a.training_history

        model_b = GruSequenceModel(
            TorchSequenceConfig(core="gru", gradient_clip=arguments.gradient_clip)
        )
        nrmse_b, _, _ = _fit_and_evaluate(data_b, arguments.n_train, model_b, evaluator)
        print(
            f"  (b) nur energy_target:         nrmse[{TARGET}, h{HORIZON}] = {nrmse_b[HORIZON]:.4f}"
        )
        case_b_nrmse.append(nrmse_b[HORIZON])

        model_c = CompositeGruSequenceModel(
            TorchSequenceConfig(core="gru", gradient_clip=arguments.gradient_clip)
        )
        nrmse_c, model_c, seconds_c = _fit_and_evaluate(
            data_a, arguments.n_train, model_c, evaluator
        )
        seed_epochs_c = {name: len(history) for name, history in model_c.training_history.items()}
        print(
            f"  (c) CompositeSequenceModel:    nrmse[{TARGET}, h{HORIZON}] = "
            f"{nrmse_c[HORIZON]:.4f}  ({seconds_c:.1f}s fit)"
        )
        print(f"      Epochen je Teilmodell:      {seed_epochs_c}")
        case_c_nrmse.append(nrmse_c[HORIZON])
        fit_seconds_c.append(seconds_c)
        epochs_c.append(seed_epochs_c)

        arx_model = LinearARXModel()
        nrmse_arx, _, _ = _fit_and_evaluate(data_a, arguments.n_train, arx_model, evaluator)
        print(
            f"  Referenz ARX:                  nrmse[{TARGET}, h{HORIZON}] = "
            f"{nrmse_arx[HORIZON]:.4f}"
        )
        arx_nrmse.append(nrmse_arx[HORIZON])

        climatology_model = ClimatologyModel()
        nrmse_clim, _, _ = _fit_and_evaluate(
            data_a, arguments.n_train, climatology_model, evaluator
        )
        print(
            f"  Referenz Climatology:          nrmse[{TARGET}, h{HORIZON}] = "
            f"{nrmse_clim[HORIZON]:.4f}"
        )
        climatology_nrmse.append(nrmse_clim[HORIZON])

    print("\n\n############ Zusammenfassung ############")
    print(f"n_train={arguments.n_train}, Soft-Sensor, {TARGET}, h={HORIZON}, gepoolter NRMSE\n")
    print(
        f"  (a) alle drei Zielgrößen (Median über Seeds): {np.median(case_a_nrmse):.4f}"
        f"  [{', '.join(f'{v:.4f}' for v in case_a_nrmse)}]"
    )
    print(
        f"  (b) nur energy_target    (Median über Seeds): {np.median(case_b_nrmse):.4f}"
        f"  [{', '.join(f'{v:.4f}' for v in case_b_nrmse)}]"
    )
    print(
        f"  (c) CompositeSequenceModel      (Median): {np.median(case_c_nrmse):.4f}"
        f"  [{', '.join(f'{v:.4f}' for v in case_c_nrmse)}]"
    )
    print(
        f"  ARX-Referenz             (Median über Seeds): {np.median(arx_nrmse):.4f}"
        f"  [{', '.join(f'{v:.4f}' for v in arx_nrmse)}]"
    )
    print(
        f"  Climatology-Referenz     (Median über Seeds): {np.median(climatology_nrmse):.4f}"
        f"  [{', '.join(f'{v:.4f}' for v in climatology_nrmse)}]"
    )

    print("\nTrainingszeit fit() (Median über Seeds, dieselbe Skriptausführung):")
    print(
        f"  (a) gemeinsames Netz:        {np.median(fit_seconds_a):.2f}s"
        f"  [{', '.join(f'{v:.2f}' for v in fit_seconds_a)}]"
    )
    print(
        f"  (c) CompositeSequenceModel:  {np.median(fit_seconds_c):.2f}s"
        f"  [{', '.join(f'{v:.2f}' for v in fit_seconds_c)}]"
    )
    print(
        f"  Faktor (c)/(a):              {np.median(fit_seconds_c) / np.median(fit_seconds_a):.2f}x"
    )

    print("\nTatsächlich gelaufene Epochen je Seed (max_epochs=200, patience=20):")
    print(f"  (a) gemeinsames Netz (alle drei Zielgrößen in einem Trainingslauf): {epochs_a}")
    for target in ALL_TARGETS:
        per_seed = [record[target] for record in epochs_c]
        print(f"  (c) Teilmodell {target:<28} {per_seed}  (Median {np.median(per_seed):.0f})")
    sum_c = [sum(record.values()) for record in epochs_c]
    print(f"  (c) Summe der drei Teilmodelle je Seed: {sum_c}")
    print(
        f"  Verhältnis Summe(c)/(a) je Seed: "
        f"{[round(s / a, 2) for s, a in zip(sum_c, epochs_a, strict=True)]}"
        "  (Erwartung bei gleicher Kosten je Epoche: nahe am Zeitfaktor oben)"
    )

    print("\nLetzter Seed, letzte Trainingsepoche, Verlust pro Zielgröße (Fall a):")
    if last_history_a:
        last = last_history_a[-1]
        for key, value in last.items():
            if key.startswith("train_loss"):
                print(f"  {key:<32} {value:.4f}")


if __name__ == "__main__":
    main()
