#!/usr/bin/env python3
"""Bewertung eines Anregungsdesigns -- vor der Datenproduktion auszuführen.

Das Skript ist als Abstimmungsgrundlage mit TUHH-A gedacht: es beantwortet in einer
Bildschirmseite, ob ein Sampling-Design den Prozess anregt oder nur beschäftigt.

Aufruf::

    python scripts/report_excitation.py conf/process/greybox_placeholder.yaml \
        --runs 16 --figure artifacts/excitation.png

Die Verweilzeiten und der PRBS-Takt werden aus den Zeitkonstanten des Prozessmodells
abgeleitet (``ExcitationConfig.from_time_constants``), nicht geraten. Nach jeder Änderung
an der Prozess-Config ändern sie sich also mit.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.simulation import (
    ExcitationConfig,
    GreyBoxFluidBedGranulator,
    MixedExcitationDesign,
    constraint_violation_summary,
    excitation_report,
    load_process_config,
)
from twinlink_ml.simulation.diagnostics import static_explained_variance, time_constant_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--runs", type=int, default=16, help="Anzahl Läufe (Potenz von 2)")
    parser.add_argument("--duration-s", type=float, default=28_800.0, help="Laufdauer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--sampler", choices=("sobol", "lhs"), default="sobol", help="QMC-Sampler")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    process_config = load_process_config(args.config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulator = GreyBoxFluidBedGranulator(process_config)

    taus = time_constant_report(simulator)
    tau_values = sorted(v for k, v in taus.items() if k.startswith("tau_"))
    tau_fast, tau_slow = tau_values[0], tau_values[-1]

    excitation = ExcitationConfig.from_time_constants(
        tau_fast_s=tau_fast,
        tau_slow_s=tau_slow,
        sample_time_s=simulator.sample_time_s,
        run_duration_s=args.duration_s,
        sampler=args.sampler,
    )
    design = MixedExcitationDesign(
        excitation,
        process_config.actions,
        process_config.disturbances,
        sample_time_s=simulator.sample_time_s,
        geometry=process_config.geometry,
        material=process_config.material,
        operation=process_config.operation,
    )
    schedule = design.build(args.runs, seed=args.seed)

    print("=" * 96)
    print(f"Prozess-Config    {args.config}  ({process_config.fingerprint})")
    print(f"Anregungsdesign   {schedule.metadata['fingerprint']}  Sampler {args.sampler}")
    print(
        f"tau_fast {tau_fast:.1f} s   tau_slow {tau_slow:.0f} s   "
        f"Verweilzeit {excitation.hold_time.minimum_s:.0f}.."
        f"{excitation.hold_time.maximum_s:.0f} s   "
        f"PRBS {excitation.prbs_register_length} bit @ {excitation.prbs_clock_s:.0f} s "
        f"(Periode {excitation.prbs_period_s / 3600:.1f} h)"
    )
    print("=" * 96)

    report = excitation_report(
        schedule,
        process_config.actions,
        process_config.disturbances,
        sample_time_s=simulator.sample_time_s,
        tau_fast_s=tau_fast,
        tau_slow_s=tau_slow,
        excitation_config=excitation,
    )
    print("\n--- Anregungsqualität " + "-" * 74)
    print(report.format())

    print("\n--- Rollout " + "-" * 84)
    assert schedule.initial_states is not None
    initial = simulator.state_from_physical(
        schedule.initial_states[:, 0],
        schedule.initial_states[:, 1],
        schedule.initial_states[:, 2],
        schedule.initial_states[:, 3],
    )
    trajectory = simulator.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=args.seed,
        initial_state=initial,
    )
    for name, fraction in constraint_violation_summary(trajectory).items():
        print(f"{name:34s} {fraction:7.2%} der Abtastschritte")

    print("\n--- Statischer Varianzanteil der Zielgrößen " + "-" * 52)
    for name, r_squared in static_explained_variance(trajectory).items():
        verdict = "praktisch statisch" if r_squared > 0.95 else "dynamischer Anteil vorhanden"
        print(f"{name:34s} R² = {r_squared:7.4f}   {verdict}")

    if args.figure is not None:
        _plot(schedule, trajectory, simulator.sample_time_s, args.figure)
        print(f"\nGrafik gespeichert: {args.figure}")
    return 0


def _plot(schedule: object, trajectory: object, sample_time_s: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actions = schedule.actions  # type: ignore[attr-defined]
    kinds = schedule.signal_kinds  # type: ignore[attr-defined]
    names = schedule.metadata["action_names"]  # type: ignore[attr-defined]
    outputs = trajectory.outputs  # type: ignore[attr-defined]
    output_names = trajectory.metadata["output_names"]  # type: ignore[attr-defined]
    hours = np.arange(actions.shape[1]) * sample_time_s / 3600.0

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 2, figsize=(14, 10))

    # links: je ein Beispiellauf pro Signalart auf dem T_in-Kanal
    selected: list[int] = []
    for wanted in ("step", "prbs", "ramp"):
        matches = [index for index, k in enumerate(kinds) if k[0] == wanted]
        if matches:
            selected.append(matches[0])
    while len(selected) < 3:
        selected.append(min(len(selected), actions.shape[0] - 1))

    for row in range(3):
        run = selected[row]
        axis = axes[row, 0]
        axis.step(hours, actions[run, :, 0], where="post", color="tab:red", lw=0.8)
        axis.set_ylabel(f"T_in [°C]\nLauf {run}: {kinds[run][0]}", fontsize=8)
        twin = axis.twinx()
        twin.step(hours, actions[run, :, 2], where="post", color="tab:green", lw=0.8)
        twin.set_ylabel(f"m_spray\n{kinds[run][2]}", fontsize=7)
    axes[3, 0].set_xlabel("Zeit [h]")
    axes[3, 0].step(hours, actions[selected[0], :, 1], where="post", color="tab:orange", lw=0.8)
    axes[3, 0].set_ylabel(f"u_gas [m/s]\nLauf {selected[0]}: {kinds[selected[0]][1]}", fontsize=8)

    # rechts: Leistungsdichte je Kanal und Zielgrößen
    from scipy import signal as scipy_signal

    axis = axes[0, 1]
    for channel in range(actions.shape[2]):
        values = actions[:, :, channel]
        frequencies, density = scipy_signal.welch(
            values - values.mean(axis=1, keepdims=True),
            fs=1.0 / sample_time_s,
            nperseg=min(values.shape[1], 1024),
            axis=1,
        )
        mean_density = density.mean(axis=0)
        normalised = mean_density / mean_density[1:].sum()
        axis.loglog(frequencies[1:], normalised[1:], lw=0.9, label=names[channel])
    axis.set_xlabel("Frequenz [Hz]", fontsize=8)
    axis.set_ylabel("normierte Leistungsdichte", fontsize=8)
    axis.legend(fontsize=7)
    axis.grid(True, which="both", alpha=0.2)

    for row, index in enumerate(range(min(3, outputs.shape[2]))):
        axis = axes[row + 1, 1]
        for run in range(min(6, outputs.shape[0])):
            axis.plot(hours, outputs[run, :, index], lw=0.7, alpha=0.8)
        axis.set_ylabel(output_names[index], fontsize=8)
    axes[3, 1].set_xlabel("Zeit [h]")

    figure.suptitle("Anregungsdesign: Signale, Spektren, Antworten", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
