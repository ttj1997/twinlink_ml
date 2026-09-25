"""Vertikaler Durchstich: Anregung, Rollout, Split, Fensterung, Baselines, Metrik.

Aufruf::

    python scripts/report_baselines.py --n-runs 32 --sample-time 25

Der Bericht ist das eigentliche Ergebnis von Schritt 4: er beantwortet, welche Hürde eine
Architektur überhaupt überspringen muss -- getrennt je Zielgröße und je Horizont.
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.data import (
    IndexBlockSplitter,
    SplitName,
    TrajectoryWindowDataset,
    WindowConfig,
    signal_kind_strata,
    split_report,
    split_schedule,
)
from twinlink_ml.evaluation import RolloutEvaluator
from twinlink_ml.models import LinearARXModel, PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import (
    ExcitationConfig,
    GreyBoxFluidBedGranulator,
    MixedExcitationDesign,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.diagnostics import static_explained_variance
from twinlink_ml.utils.seeding import SeedBundle

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
    parser.add_argument("--n-runs", type=int, default=32, help="Pool-Größe, Zweierpotenz")
    parser.add_argument("--sample-time", type=float, default=25.0, help="Abtastzeit in s")
    parser.add_argument("--run-duration", type=float, default=28_800.0, help="Laufdauer in s")
    parser.add_argument("--history", type=int, default=32, help="Fensterhistorie L")
    parser.add_argument("--horizon", type=int, default=24, help="Rollout-Horizont H")
    parser.add_argument("--stride", type=int, default=8, help="Fensterabstand")
    parser.add_argument("--seed", type=int, default=20260812, help="Master-Seed")
    parser.add_argument(
        "--randomise-initial-conditions",
        dest="randomise_initial_conditions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Anfangszustände über den Betriebsbereich streuen (Standarddesign, "
        "Default) oder beim nominellen Startpunkt belassen "
        "(--no-randomise-initial-conditions). Ohne diesen Schalter war die Bedingung "
        "'ohne Streuung' bisher gar nicht erzeugbar (02_prozess_setup.md, "
        "R²-Tabelle statischer Varianzanteil).",
    )
    return parser.parse_args()


def main() -> None:
    """Bericht erzeugen."""
    arguments = parse_arguments()
    seeds = SeedBundle.from_master(arguments.seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(arguments.config)
        config = dataclasses.replace(
            base,
            numerics=dataclasses.replace(
                base.numerics, sample_time_s=arguments.sample_time, substeps=5
            ),
        )
        simulator = GreyBoxFluidBedGranulator(config)

    design = MixedExcitationDesign(
        ExcitationConfig(
            run_duration_s=arguments.run_duration,
            randomise_initial_conditions=arguments.randomise_initial_conditions,
        ),
        config.actions,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
    )
    schedule = design.build(arguments.n_runs, seed=seeds.data_seed)

    initial = None
    if schedule.initial_states is not None:
        columns = schedule.initial_states
        initial = simulator.state_from_physical(
            columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        )
    trajectory = simulator.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=seeds.noise_seed,
        initial_state=initial,
    )

    print(
        f"Δt = {config.numerics.sample_time_s:.0f} s, {trajectory.n_steps} Schritte pro Lauf, "
        f"randomise_initial_conditions={arguments.randomise_initial_conditions}"
    )
    print("\nStatischer Varianzanteil (quadratische Regression, ohne Historie):")
    for name, value in static_explained_variance(trajectory).items():
        print(f"  {name:<32} R² = {value:.4f}")

    n_train = arguments.n_runs // 2
    n_block = arguments.n_runs // 4
    splitter = IndexBlockSplitter(n_train=n_train, n_val=n_block, n_test=n_block)
    split = split_schedule(schedule, splitter)
    print("\n" + split_report(split, signal_kind_strata(schedule, channel=0)))

    window = WindowConfig(
        history_length=arguments.history,
        horizon=arguments.horizon,
        stride=arguments.stride,
        target_names=("energy_target", "product_moisture_kg_per_kg", "d50_um"),
    )
    train_set = TrajectoryWindowDataset.from_trajectory(trajectory, split, SplitName.TRAIN, window)
    test_set = TrajectoryWindowDataset.from_trajectory(
        trajectory, split, SplitName.TEST, window, scalers=train_set.scalers
    )
    print(
        f"\nFenster: {train_set.n_windows} train / {test_set.n_windows} test, "
        f"L={window.history_length}, H={window.horizon}, "
        f"Skalierer-Fingerprint {train_set.target_scaler.digest}"
    )

    evaluator = RolloutEvaluator(horizons=(1, 6, 12, 24))
    for model in (PersistenceModel(), StaticRegressionModel(), LinearARXModel()):
        model.fit(train_set, seed=seeds.model_seed)
        result = evaluator.evaluate(model, test_set)
        print("\n" + result.format())

    print("\nLaufweise Streuung (Median [Min .. Max]) bei h=24, ARX:")
    arx = LinearARXModel()
    arx.fit(train_set, seed=seeds.model_seed)
    spread = evaluator.evaluate(arx, test_set).nrmse_run_spread
    for target, by_horizon in spread.items():
        median, low, high = by_horizon[24]
        print(f"  {target:<32} {median:.4f} [{low:.4f} .. {high:.4f}]")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
