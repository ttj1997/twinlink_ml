r"""Bewertung der Anregungsqualität -- der Report, den TUHH-A sehen muss.

Vier Fragen, die ein Anregungsdesign beantworten muss, bevor damit 18 Monate lang
Simulationsläufe produziert werden:

1. **Wird der Betriebsraum abgedeckt?** Amplitudenausnutzung je Kanal.
2. **Sind die Kanäle trennbar?** Kollinearität zerstört die Identifizierbarkeit der
   partiellen Ableitungen -- und damit die Überprüfbarkeit der Monotonie-Constraints.
3. **Wird das relevante Frequenzband angeregt?** Spektrale Flachheit und Bandenergie,
   bezogen auf die Zeitkonstanten des Prozesses.
4. **Ist die Anregung realisierbar?** Verletzung der Ratenbegrenzung und der
   Fluidisierungsgrenzen.

Die Zahlen aus diesem Report sind Teil des Datensatz-Metadatums. Eine Lernkurve ohne den
zugehörigen Anregungsreport ist nicht interpretierbar: dieselbe Architektur auf
schwach angeregten Daten sieht aus wie eine schlechtere Architektur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from .base import Trajectory
from .config import ActionSpaceConfig, DisturbanceSpaceConfig
from .excitation import ExcitationConfig, ExcitationSchedule

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ChannelExcitationReport:
    """Kennzahlen eines Stellgrößenkanals."""

    name: str
    range_utilisation: float
    standard_deviation_fraction: float
    switch_count_per_run: float
    spectral_flatness: float
    bandwidth_p90_hz: float
    bandwidth_over_fast_corner: float
    rate_limit_utilisation: float
    full_amplitude_switch_time_over_tau_fast: float
    fastest_amplitude_switch_time_over_tau_fast: float

    def format(self) -> str:
        return (
            f"{self.name:30s} "
            f"Spanne {self.range_utilisation:5.2f}  "
            f"σ/Spanne {self.standard_deviation_fraction:5.3f}  "
            f"Flachheit {self.spectral_flatness:6.4f}  "
            f"f90 {self.bandwidth_p90_hz:9.2e} Hz  "
            f"f90/f_fast {self.bandwidth_over_fast_corner:6.3f}  "
            f"Rate {self.rate_limit_utilisation:5.2f}  "
            f"t_switch/τ_fast {self.full_amplitude_switch_time_over_tau_fast:6.1f}"
            f" / {self.fastest_amplitude_switch_time_over_tau_fast:5.1f}"
        )


@dataclass(frozen=True, slots=True)
class ExcitationReport:
    """Gesamtbewertung eines Anregungsplans."""

    n_runs: int
    n_steps: int
    run_duration_s: float
    channels: tuple[ChannelExcitationReport, ...]
    max_absolute_action_correlation: float
    action_correlation: FloatArray
    disturbance_coverage: dict[str, float]
    initial_condition_coverage: dict[str, float]
    warnings: tuple[str, ...]

    def format(self) -> str:
        lines = [
            f"Läufe {self.n_runs}, {self.n_steps} Schritte, "
            f"{self.run_duration_s / 3600.0:.2f} h pro Lauf",
            "",
            *[channel.format() for channel in self.channels],
            "",
            f"maximale |Korrelation| zwischen Stellgrößen: "
            f"{self.max_absolute_action_correlation:.3f}",
            "Abdeckung Störgrößenraum: "
            + ", ".join(f"{k} {v:.2f}" for k, v in self.disturbance_coverage.items()),
        ]
        if self.initial_condition_coverage:
            lines.append(
                "Abdeckung Anfangsbedingungen: "
                + ", ".join(f"{k} {v:.2f}" for k, v in self.initial_condition_coverage.items())
            )
        if self.warnings:
            lines.extend(["", *[f"WARNUNG  {w}" for w in self.warnings]])
        return "\n".join(lines)


def excitation_report(
    schedule: ExcitationSchedule,
    action_space: ActionSpaceConfig,
    disturbance_space: DisturbanceSpaceConfig,
    *,
    sample_time_s: float,
    tau_fast_s: float,
    tau_slow_s: float,
    excitation_config: ExcitationConfig | None = None,
) -> ExcitationReport:
    r"""Bewertet einen Anregungsplan gegen die Zeitkonstanten des Prozesses.

    Kennzahlen je Kanal:

    ``range_utilisation``
        Verhältnis der tatsächlich durchfahrenen Spanne zur erlaubten. Werte deutlich
        unter 1 bedeuten, dass das Modell später auf Betriebspunkte extrapolieren muss,
        die es nie gesehen hat.

    ``spectral_flatness``
        Geometrisches durch arithmetisches Mittel der Leistungsdichte,
        :math:`1` = weiß, :math:`\to 0` = schmalbandig. Nach Ratenbegrenzung ist die
        Flachheit eines PRBS notwendig kleiner als 1 -- die Kennzahl macht sichtbar,
        wie viel Bandbreite die Realisierbarkeit gekostet hat.

    ``bandwidth_p90_hz``
        Frequenz, unter der 90 % der Signalenergie liegen -- die realisierte Bandbreite.
        Ins Verhältnis gesetzt zur Eckfrequenz der schnellsten Prozessmode
        :math:`f_{fast} = 1/(2\pi\tau_{fast})` ergibt sich
        ``bandwidth_over_fast_corner``. Werte deutlich unter 1 heißen: die schnelle Mode
        des Prozesses wird nicht angeregt, sie verhält sich gegenüber dieser Anregung
        quasistationär. Ein Energieanteil in einem breit definierten Band ist als Kennzahl
        untauglich -- er liegt bei fast jedem Design nahe 1, weil das Band von
        :math:`1/(2\pi\tau_{slow})` bis :math:`1/(2\pi\tau_{fast})` praktisch das
        gesamte Spektrum unterhalb Nyquist umfasst.

    ``full_amplitude_switch_time_over_tau_fast`` /
    ``fastest_amplitude_switch_time_over_tau_fast``
        :math:`t_{switch}/\tau_{fast}` mit :math:`t_{switch} = \Delta u/\dot u_{max}`,
        einmal für die volle Spanne und einmal für die kleinste Amplitudenstufe des
        Designs. Die zweite Zahl ist die entscheidende: liegt **sie** über 1, ist die
        schnellste Prozessmode mit keiner Amplitude anregbar, und die Abtastzeit ist zu
        fein gewählt. Genau dafür existieren die Amplitudenstufen.
    """
    actions = schedule.actions
    n_runs, n_steps, n_channels = actions.shape
    bounds = action_space.bounds
    names = action_space.names

    fast_corner_hz = 1.0 / (2.0 * np.pi * tau_fast_s)
    smallest_amplitude = (
        min(excitation_config.amplitude_levels) if excitation_config is not None else 1.0
    )
    _ = tau_slow_s

    channels: list[ChannelExcitationReport] = []
    messages: list[str] = []
    for channel in range(n_channels):
        values = actions[:, :, channel]
        bound = bounds[channel]
        realised = float(values.max() - values.min())
        differences = np.diff(values, axis=1)
        limit = bound.max_rate_per_s * sample_time_s
        switch_count = float(np.mean(np.sum(np.abs(differences) > 1e-9, axis=1)))
        full_switch_time = bound.span / bound.max_rate_per_s

        flatness, bandwidth = _spectral_measures(values, sample_time_s)
        report = ChannelExcitationReport(
            name=names[channel],
            range_utilisation=realised / bound.span,
            standard_deviation_fraction=float(np.std(values) / bound.span),
            switch_count_per_run=switch_count,
            spectral_flatness=flatness,
            bandwidth_p90_hz=bandwidth,
            bandwidth_over_fast_corner=bandwidth / fast_corner_hz,
            rate_limit_utilisation=float(np.max(np.abs(differences)) / limit)
            if differences.size
            else 0.0,
            full_amplitude_switch_time_over_tau_fast=full_switch_time / tau_fast_s,
            fastest_amplitude_switch_time_over_tau_fast=(
                smallest_amplitude * full_switch_time / tau_fast_s
            ),
        )
        channels.append(report)

        if report.range_utilisation < 0.7:
            messages.append(
                f"{report.name}: nur {report.range_utilisation:.0%} der Spanne durchfahren"
            )
        if report.rate_limit_utilisation > 1.0 + 1e-6:
            messages.append(
                f"{report.name}: Ratenbegrenzung verletzt "
                f"(Faktor {report.rate_limit_utilisation:.2f})"
            )
        if report.fastest_amplitude_switch_time_over_tau_fast > 1.0:
            messages.append(
                f"{report.name}: selbst die kleinste Amplitudenstufe braucht "
                f"{report.fastest_amplitude_switch_time_over_tau_fast:.1f} tau_fast -- "
                "die schnellste Prozessmode ist nicht anregbar, Abtastzeit zu fein"
            )
        if report.bandwidth_over_fast_corner < 0.3:
            messages.append(
                f"{report.name}: realisierte Bandbreite nur "
                f"{report.bandwidth_over_fast_corner:.2f} der Eckfrequenz der schnellen Mode"
            )
        if report.rate_limit_utilisation > 0.99:
            messages.append(
                f"{report.name}: Ratenbegrenzung dauerhaft aktiv -- das Signaldesign wird "
                "durch die Aktordynamik ersetzt, die Signalart ist praktisch wirkungslos"
            )

    correlation = _pooled_correlation(actions)
    off_diagonal = correlation - np.eye(n_channels)
    max_correlation = float(np.max(np.abs(off_diagonal)))
    if max_correlation > 0.5:
        messages.append(
            f"Stellgrößen korrelieren mit |r| = {max_correlation:.2f} -- "
            "partielle Ableitungen sind schwer trennbar"
        )

    disturbance_coverage = _coverage(
        schedule.disturbances.reshape(-1, schedule.disturbances.shape[-1]),
        np.array([r.minimum for r in disturbance_space.ranges]),
        np.array([r.maximum for r in disturbance_space.ranges]),
        disturbance_space.names,
    )
    initial_coverage: dict[str, float] = {}
    if schedule.initial_states is not None:
        columns = schedule.metadata.get("initial_condition_columns", ())
        assert isinstance(columns, tuple)
        values = schedule.initial_states
        initial_coverage = {
            str(columns[index]): float(values[:, index].max() / max(values[:, index].min(), 1e-12))
            for index in range(values.shape[1])
        }

    if n_runs < 8 and schedule.metadata.get("sampler") == "sobol":
        messages.append(
            "Sobol-Designs sollten Potenzen von 2 als Laufzahl verwenden, "
            "sonst leidet die Diskrepanz"
        )

    return ExcitationReport(
        n_runs=n_runs,
        n_steps=n_steps,
        run_duration_s=n_steps * sample_time_s,
        channels=tuple(channels),
        max_absolute_action_correlation=max_correlation,
        action_correlation=correlation,
        disturbance_coverage=disturbance_coverage,
        initial_condition_coverage=initial_coverage,
        warnings=tuple(messages),
    )


def constraint_violation_summary(trajectory: Trajectory) -> dict[str, float]:
    r"""Anteil der Abtastschritte, in denen Prozessgrenzen verletzt sind.

    Auswertung der Diagnosegrößen eines Rollouts. Zwei Zwecke:

    * **Datensatzqualität.** Läufe, die dauerhaft außerhalb des Fluidisierungsfensters
      liegen, beschreiben einen Zustand, den die Anlage nicht hat (Bett zusammengebrochen
      oder ausgetragen). Das Grey-Box-Modell rechnet trotzdem weiter -- es kennt keinen
      Kollaps. Solche Abschnitte sind für das Lernen der *regulären* Dynamik schädlich.
    * **RL-Vorbereitung.** Dieselbe Kennzahl ist später die Verletzungsrate der
      Safety-Constraints. Wer sie schon in AP 1 mitschreibt, hat in AP 3 eine Referenz
      dafür, was ein Regler verbessern muss.

    Anmerkung zur Interpretation: ``gas_velocity_above_terminal`` ist bei kleinen
    Partikeln normal -- Feingut wird real ausgetragen und über den Zyklon zurückgeführt.
    Das Modell bilanziert diesen Austrag nicht (siehe README). Ein hoher Wert heißt also
    nicht "Betrieb unmöglich", sondern "hier fehlt Modellphysik".
    """
    diagnostics = trajectory.diagnostics
    total = float(trajectory.n_runs * trajectory.n_steps)
    summary: dict[str, float] = {}
    if "fluidization_margin_lower_m_per_s" in diagnostics:
        summary["below_minimum_fluidization"] = (
            float(np.sum(diagnostics["fluidization_margin_lower_m_per_s"] < 0.0)) / total
        )
    if "fluidization_margin_upper_m_per_s" in diagnostics:
        summary["gas_velocity_above_terminal"] = (
            float(np.sum(diagnostics["fluidization_margin_upper_m_per_s"] < 0.0)) / total
        )
    if "saturation_degree" in diagnostics:
        summary["at_equilibrium_moisture"] = (
            float(np.sum(diagnostics["saturation_degree"] <= 0.0)) / total
        )
    if "outlet_humidity_kg_per_kg" in diagnostics:
        summary["condensation"] = (
            float(np.sum(diagnostics["evaporation_rate_kg_per_s"] < 0.0)) / total
        )
    if "particle_moisture_kg_per_kg" in diagnostics:
        summary["above_critical_moisture"] = float(
            np.mean(diagnostics["particle_moisture_kg_per_kg"] > 0.02)
        )
    return summary


def _spectral_measures(values: FloatArray, sample_time_s: float) -> tuple[float, float]:
    """Spektrale Flachheit und Frequenz, unter der 90 % der Energie liegen."""
    segment = min(values.shape[1], 1024)
    frequencies, density = scipy_signal.welch(
        values - values.mean(axis=1, keepdims=True),
        fs=1.0 / sample_time_s,
        nperseg=segment,
        axis=1,
    )
    mean_density = np.mean(density, axis=0)
    positive = mean_density[1:]
    total = float(np.sum(positive))
    if total <= 0.0:
        return 0.0, 0.0
    guarded = np.maximum(positive, total * 1e-15)
    flatness = float(np.exp(np.mean(np.log(guarded))) / np.mean(guarded))
    cumulative = np.cumsum(positive) / total
    index = int(np.searchsorted(cumulative, 0.9))
    bandwidth = float(frequencies[1:][min(index, positive.size - 1)])
    return flatness, bandwidth


def _pooled_correlation(actions: FloatArray) -> FloatArray:
    """Korrelationsmatrix der Stellgrößen über alle Läufe und Zeitpunkte."""
    flattened = actions.reshape(-1, actions.shape[-1])
    centred = flattened - flattened.mean(axis=0, keepdims=True)
    deviation = centred.std(axis=0)
    deviation[deviation == 0.0] = 1.0
    normalised = centred / deviation
    return np.asarray(normalised.T @ normalised / flattened.shape[0], dtype=np.float64)


def _coverage(
    values: FloatArray, low: FloatArray, high: FloatArray, names: tuple[str, ...]
) -> dict[str, float]:
    """Ausgenutzter Anteil des erlaubten Bereichs je Kanal."""
    span = np.maximum(high - low, 1e-12)
    realised = values.max(axis=0) - values.min(axis=0)
    return {names[index]: float(realised[index] / span[index]) for index in range(len(names))}
