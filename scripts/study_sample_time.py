r"""Δt preliminary study: 5 / 15 / 25 / 60 s at a constant physical window, 900 s horizon.

Usage::

    python scripts/study_sample_time.py --config conf/process/greybox_gab_placeholder.yaml \
        --log artifacts/dt_study.jsonl

The result is a recommendation for the standard sample time. Deliberately run over the baselines:
the Δt recommendation should not depend on an architecture, and the
persistence error is the direct measure of the difficulty of the one-step task.

Horizon 900 s (instead of the earlier 600 s): primary decision horizon modelled on the
quarter-hour cycle of electricity-market/renewables signals, not on second-by-second control
of the bed. A fourth candidate, 15 s, supplements (does not replace) the historical basis
{5, 25, 60}: τ1 has risen to ~30 s under the GAB isotherm, so 25 s now lies only at
τ1/Δt ≈ 1.2 instead of comfortably below it -- 15 s resolves the fastest mode with
τ1/Δt ≈ 2.0 more clearly again (see docs/04_stand_ap1.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.evaluation import DataSpec, run_sample_time_study
from twinlink_ml.models import LinearARXModel, PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    ProcessConfig,
    load_process_config,
)
from twinlink_ml.simulation.diagnostics import time_constant_report
from twinlink_ml.utils.tracking import JsonlExperimentLogger, RunRecord

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

SAMPLE_TIMES_S: tuple[float, ...] = (5.0, 15.0, 25.0, 60.0)

FACTORIES = {
    "persistence": PersistenceModel,
    "static": StaticRegressionModel,
    "arx": LinearARXModel,
}


def parse_arguments() -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Prozess-Config. Erlaubt einen Lauf gegen eine korrigierte "
        "Parametrisierung (z. B. conf/process/greybox_gab_placeholder.yaml) "
        "ohne Code-Aenderung.",
    )
    parser.add_argument("--pool", type=int, default=32, help="Trainingsblock (Zweierpotenz)")
    parser.add_argument("--log", type=Path, default=None, help="JSONL-Ergebnis-Log")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Master-Seeds. Default 10 statt 5: bei n_train=32 unterschaetzen 5 Seeds die "
        "Streuung von Rollout-Horizont-Metriken systematisch, siehe docs/04_stand_ap1.md "
        "Befund 35.",
    )
    return parser.parse_args()


def _sequence_length_context(
    config: ProcessConfig, sample_times_s: tuple[float, ...]
) -> dict[float, dict[str, float]]:
    """τ1 and the recommended sequence length derived from it per Δt candidate.

    Calls :func:`time_constant_report` per candidate with its ``sample_time_s``, instead of
    recomputing the rule of thumb (5·τ1/Δt) by hand -- the numbers should come from the same code
    as ``diagnostics.time_constant_report``, not from a second implementation.
    """
    context: dict[float, dict[str, float]] = {}
    for sample_time in sample_times_s:
        candidate_config = dataclasses.replace(
            config, numerics=dataclasses.replace(config.numerics, sample_time_s=sample_time)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            simulator = GreyBoxFluidBedGranulator(candidate_config)
        context[sample_time] = time_constant_report(simulator)
    return context


def _spread(records: list[RunRecord], label: str, key: str) -> tuple[float, float, float]:
    """Median, minimum, maximum of a metric over all seeds of a model."""
    values = np.asarray([r.metrics[key] for r in records if r.params["model/label"] == label])
    return float(np.median(values)), float(np.min(values)), float(np.max(values))


def main() -> None:
    """Run the study and print tables."""
    arguments = parse_arguments()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)

    spec = DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=arguments.pool,
        n_val=arguments.pool // 2,
        n_test=arguments.pool // 2,
        history_length=32,
        horizon=36,  # 36*25s = 900s: primary decision horizon, see module docstring
        stride=8,
    )
    logger = JsonlExperimentLogger(arguments.log) if arguments.log else None

    tau_context = _sequence_length_context(config, SAMPLE_TIMES_S)

    records: list[RunRecord] = []
    timings: dict[float, list[float]] = {sample_time: [] for sample_time in SAMPLE_TIMES_S}
    for seed in arguments.seeds:
        for sample_time in SAMPLE_TIMES_S:
            start = time.perf_counter()
            candidate_records = run_sample_time_study(
                config,
                FACTORIES,
                sample_times_s=(sample_time,),
                base_spec=spec,
                master_seed=seed,
                logger=logger,
            )
            timings[sample_time].append(time.perf_counter() - start)
            records.extend(candidate_records)

    history_s = spec.history_length * spec.sample_time_s
    horizon_s = spec.horizon * spec.sample_time_s
    total_time_s = sum(sum(values) for values in timings.values())

    print(f"Fenster konstant: {history_s:.0f} s Historie, {horizon_s:.0f} s Horizont")
    print(f"{spec.n_train_max} Trainingsläufe, {spec.n_test} Testläufe, Seeds {arguments.seeds}")
    print(f"Gesamtzeit (alle Kandidaten, alle Seeds): {total_time_s:.1f} s\n")

    print("--- Rechenzeit je Kandidat (über Seeds, alle Modelle) ---")
    print(f"{'Δt / s':>8}{'Median s':>12}{'Min s':>10}{'Max s':>10}")
    for sample_time in SAMPLE_TIMES_S:
        values = np.asarray(timings[sample_time])
        print(
            f"{sample_time:>8.0f}{np.median(values):>12.2f}{np.min(values):>10.2f}"
            f"{np.max(values):>10.2f}"
        )
    print()

    for target in spec.target_names:
        candidates_by_dt = {
            sample_time: [r for r in records if r.params["data/sample_time_s"] == sample_time]
            for sample_time in SAMPLE_TIMES_S
        }

        print(f"--- {target} — rohe NRMSE je Kandidat, je Seed ---")
        print(f"{'Δt / s':>8}{'Seed':>6}{'Persistenz h=1':>16}{'ARX h=1':>10}{'ARX Horizont':>14}")
        for sample_time in SAMPLE_TIMES_S:
            candidate_records = candidates_by_dt[sample_time]
            horizon = int(
                next(r for r in candidate_records if r.params["model/label"] == "arx").params[
                    "data/horizon"
                ]
            )
            for seed in arguments.seeds:
                persistence = next(
                    r
                    for r in candidate_records
                    if r.params["model/label"] == "persistence" and r.params["master_seed"] == seed
                )
                arx = next(
                    r
                    for r in candidate_records
                    if r.params["model/label"] == "arx" and r.params["master_seed"] == seed
                )
                print(
                    f"{sample_time:>8.0f}{seed:>6}"
                    f"{persistence.metrics[f'nrmse_run_median/{target}/h1']:>16.4f}"
                    f"{arx.metrics[f'nrmse_run_median/{target}/h1']:>10.4f}"
                    f"{arx.metrics[f'nrmse_run_median/{target}/h{horizon}']:>14.4f}"
                )
        print()

        print(f"--- {target} — Median [Min, Max] über Seeds, mit τ1/Δt-Kontext ---")
        print(
            f"{'Δt / s':>8}{'τ1/Δt':>8}{'Seq(schnell)':>13}  "
            f"{'Persistenz h=1':<24}{'ARX h=1':<24}{'ARX Horizont':<24}"
        )
        for sample_time in SAMPLE_TIMES_S:
            candidate_records = candidates_by_dt[sample_time]
            horizon = int(
                next(r for r in candidate_records if r.params["model/label"] == "arx").params[
                    "data/horizon"
                ]
            )
            pers_med, pers_min, pers_max = _spread(
                candidate_records, "persistence", f"nrmse_run_median/{target}/h1"
            )
            arx1_med, arx1_min, arx1_max = _spread(
                candidate_records, "arx", f"nrmse_run_median/{target}/h1"
            )
            arxh_med, arxh_min, arxh_max = _spread(
                candidate_records, "arx", f"nrmse_run_median/{target}/h{horizon}"
            )
            tau1 = tau_context[sample_time]["tau_1_s"]
            seq_fast = tau_context[sample_time]["recommended_sequence_length_fast"]
            print(
                f"{sample_time:>8.0f}{tau1 / sample_time:>8.2f}{seq_fast:>13.0f}  "
                f"{pers_med:.4f} [{pers_min:.4f}, {pers_max:.4f}]".ljust(24)
                + f"{arx1_med:.4f} [{arx1_min:.4f}, {arx1_max:.4f}]".ljust(24)
                + f"{arxh_med:.4f} [{arxh_min:.4f}, {arxh_max:.4f}]".ljust(24)
            )
        print()

    if logger is not None:
        logger.close()


if __name__ == "__main__":
    main()
