#!/usr/bin/env python3
"""Charakterisierung des Grey-Box-Modells für einen gegebenen Config-Stand.

Nach jedem Ersetzen von Platzhalterwerten laufen lassen. Der Report ist die Grundlage
für zwei Entscheidungen, die sich sonst schleichend verschieben:

* Sequenzlänge und Abtastzeit (aus den Zeitkonstanten)
* Auswahl der Zielgrößen (aus dem statischen Varianzanteil)

Aufruf::

    python scripts/report_operating_point.py conf/process/greybox_placeholder.yaml \
        --figure artifacts/operating_point.png

Das Skript enthält bewusst keine Logik, die nicht auch als Modul existiert -- alles
Fachliche liegt in ``twinlink_ml.simulation.diagnostics``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.simulation import GreyBoxFluidBedGranulator, load_process_config
from twinlink_ml.simulation.diagnostics import (
    static_explained_variance,
    steady_state_report,
    time_constant_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Pfad zur Prozess-Config (YAML)")
    parser.add_argument(
        "--figure", type=Path, default=None, help="optionaler Pfad für die Übersichtsgrafik"
    )
    parser.add_argument(
        "--hold-time-s", type=float, default=300.0, help="Haltezeit der Sprungfolge"
    )
    parser.add_argument("--duration-s", type=float, default=30_000.0, help="Simulationsdauer")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def step_sequence(
    simulator: GreyBoxFluidBedGranulator, n_steps: int, hold_steps: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Provisorische Sprungfolge zur Charakterisierung.

    **Kein** Ersatz für das Anregungsdesign aus Schritt 2 des Arbeitsplans (PRBS, Rampen,
    Latin-Hypercube über die Störgrößen). Hier nur, um überhaupt Varianz auf den Eingängen
    zu haben -- ohne Anregung ist der statische Varianzanteil nicht definiert.
    """
    rng = np.random.default_rng(seed)
    actions = np.repeat(simulator.nominal_action(1)[:, None, :], n_steps, axis=1)
    for start in range(0, n_steps, hold_steps):
        for index, bound in enumerate(simulator.config.actions.bounds):
            actions[0, start : start + hold_steps, index] = rng.uniform(
                bound.minimum, bound.maximum
            )
    disturbances = np.repeat(simulator.nominal_disturbance(1)[:, None, :], n_steps, axis=1)
    for start in range(0, n_steps, 4 * hold_steps):
        for index, span in enumerate(simulator.config.disturbances.ranges):
            disturbances[0, start : start + 4 * hold_steps, index] = rng.uniform(
                span.minimum, span.maximum
            )
    return actions, disturbances


def main() -> int:
    args = parse_args()
    config = load_process_config(args.config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulator = GreyBoxFluidBedGranulator(config)

    print("=" * 78)
    print(f"Config          {args.config}")
    print(f"Fingerprint     {config.fingerprint}")
    print(f"Status          {config.provenance.status}")
    print(f"Offene TODOs    {len(config.provenance.open_todos)}")
    print("=" * 78)

    print("\n--- Beharrungszustand bei nomineller Anregung " + "-" * 32)
    print(steady_state_report(simulator).format())

    print("\n--- Zeitkonstanten " + "-" * 59)
    taus = time_constant_report(simulator)
    for key in sorted(k for k in taus if k.startswith("tau_")):
        value = taus[key]
        print(f"{key:34s} {value:12.1f} s" if value is not None else f"{key:34s} {'--':>12s}")
    print(f"{'Steifigkeitsverhältnis':34s} {taus.get('stiffness_ratio', float('nan')):12.0f}")
    print(f"{'Abtastzeit':34s} {taus['sample_time_s']:12.1f} s")
    print(f"{'schnellste Mode in Abtastschritten':34s} {taus['fastest_mode_samples']:12.1f}")
    print(
        f"{'Sequenzlänge für schnelle Mode':34s} {taus['recommended_sequence_length_fast']:12.0f}"
        "  (Untergrenze Abtastauflösung, KEINE Fensterempfehlung)"
    )
    print(f"{'Schritte für langsamste Mode':34s} {taus['steps_for_slowest_mode']:12.0f}")
    seq_coupled_slow = taus.get("recommended_sequence_length_coupled_slow")
    if seq_coupled_slow is not None:
        print(f"{'Sequenzlänge (gekoppelte langsame Mode)':34s} {seq_coupled_slow:12.0f}")
    else:
        print(
            f"{'Sequenzlänge (gekoppelte langsame Mode)':34s} {'--':>12s}  "
            f"({taus.get('mode_identification_note')})"
        )

    print("\n--- Statischer Varianzanteil der Ausgangsgrößen " + "-" * 30)
    n_steps = int(args.duration_s / simulator.sample_time_s)
    hold_steps = max(1, int(args.hold_time_s / simulator.sample_time_s))
    actions, disturbances = step_sequence(simulator, n_steps, hold_steps, args.seed)
    trajectory = simulator.rollout(actions, disturbances, seed=args.seed)
    for name, r_squared in static_explained_variance(trajectory).items():
        verdict = "praktisch statisch" if r_squared > 0.95 else "dynamischer Anteil vorhanden"
        print(f"{name:34s} R² = {r_squared:7.4f}   {verdict}")

    if args.figure is not None:
        _plot(trajectory, args.figure)
        print(f"\nGrafik gespeichert: {args.figure}")
    return 0


def _plot(trajectory: object, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert hasattr(trajectory, "time_s")
    hours = trajectory.time_s / 3600.0  # type: ignore[attr-defined]
    actions = trajectory.actions[0]  # type: ignore[attr-defined]
    outputs = trajectory.outputs[0]  # type: ignore[attr-defined]
    measurements = trajectory.measurements[0]  # type: ignore[attr-defined]
    names = trajectory.metadata["output_names"]  # type: ignore[attr-defined]
    action_names = trajectory.metadata["action_names"]  # type: ignore[attr-defined]

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    axes[0].step(hours, actions[:, 0], where="post", color="tab:red", label=action_names[0])
    axes[0].set_ylabel("T_in [°C]", color="tab:red")
    twin = axes[0].twinx()
    twin.step(hours, actions[:, 1], where="post", color="tab:orange", label=action_names[1])
    twin.set_ylabel("u_gas [m/s]", color="tab:orange")
    axes[1].step(hours, actions[:, 2], where="post", color="tab:green")
    axes[1].set_ylabel("m_spray [kg/h]")
    for row, index in enumerate(range(len(names))):
        axis = axes[2 + row]
        axis.plot(hours, measurements[:, index], lw=0.6, color="0.7", label="gemessen")
        axis.plot(hours, outputs[:, index], lw=1.2, color="tab:blue", label="wahr")
        axis.set_ylabel(names[index], fontsize=8)
        if row == 0:
            axis.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Zeit [h]")
    figure.suptitle("Grey-Box-Modell: Sprungfolge auf den Stellgrößen", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
