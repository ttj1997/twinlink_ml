r"""Diagnose des Prozessmodells -- Voraussetzung für einen interpretierbaren Vergleich.

Die ersten drei Funktionen hier beantworten Fragen, die **vor** dem ersten
Architekturvergleich geklärt sein müssen:

1. :func:`steady_state_report` -- liegt der Betriebspunkt überhaupt im plausiblen Bereich?
2. :func:`time_constant_report` -- welche Zeitkonstanten hat das System, und welche
   Sequenzlänge folgt daraus?
3. :func:`static_explained_variance` -- **ist die Zielgröße überhaupt dynamisch?**

Eine vierte, :func:`mass_balance_residual`, beantwortet eine andere Frage -- nicht vor dem
Architekturvergleich, sondern vor jedem Trainingslauf: schließt die Feststoffbilanz eines
erzeugten Datensatzes innerhalb der numerischen Toleranz? Sie ist die Grundlage von
``tests/test_greybox_balances.py::test_solid_balance_closes_within_tolerance_in_continuous_mode``
und als Datensatz-Assertion gedacht (siehe ihr eigener Docstring), nicht als
Datensatz-Report wie die ersten drei.

Zu Punkt 3, dem unangenehmsten
------------------------------
Wenn sich eine Zielgröße bereits durch eine statische Abbildung der aktuellen Stell- und
Störgrößen erklären lässt, gibt es für ein Sequenzmodell nichts zu lernen. Der Vergleich
GRU / xLSTM / Mamba / Transformer misst dann Optimierungsrauschen, und das Ergebnis ist
beliebig -- ein Artefakt, das exakt so aussieht wie ein Resultat.

Für die Momentanleistung des Grey-Box-Modells liegt dieser statische Anteil bei
:math:`R^2 \approx 0.98`: die Heizleistung ist algebraisch in
:math:`(T_{in}, u_{gas}, T_{amb})`, und der Prozesszustand wirkt nur über die Gasdichte
und den Bettdruckverlust. Die dynamischen Größen sind die **Qualitätsgrößen**
(Partikelfeuchte, Korngröße), nicht der Energieverbrauch.

Diese Kennzahl gehört deshalb in jeden Datensatz-Report, und die Baseline-Liste des
Arbeitsplans braucht einen vierten Eintrag: eine **statische Regression** auf die
aktuellen Eingänge. Persistenz ist für diese Zielgröße die falsche Referenz -- sie
verliert jeden Stellgrößensprung und lässt damit jede Architektur gut aussehen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .backend import FloatArray
from .base import Trajectory
from .config import ProcessConfig
from .greybox import IDX_SOLID_MASS, GreyBoxFluidBedGranulator
from .steady_state import solve_steady_state

_SECONDS_PER_HOUR = 3600.0
"""Dieselbe Umrechnung wie ``greybox._SECONDS_PER_HOUR`` -- hier dupliziert statt
importiert, weil es sich um eine private Modulkonstante handelt."""

_MODE_MATCH_RTOL = 1e-6
"""Relative Toleranz für den Wertevergleich in :func:`_match_single_mode`. Die beiden
strukturellen Moden (Feststoff-Regelkreis, Feststoff-Verweilzeit) sind reine
Konfigurationskonstanten mit geschlossener Form (s. dort) -- der Jacobi-Pfad in
:meth:`GreyBoxFluidBedGranulator.time_constants` reproduziert sie numerisch praktisch
bitgleich (siehe ``tests/test_diagnostics.py``, ``tau_3_s``-Bitgleichheits-Test), 1e-6
relativ lässt trotzdem Luft für andere Konfigurationen/Backends."""


@dataclass(frozen=True, slots=True)
class SteadyStateReport:
    """Beharrungszustand bei konstanter, nomineller Anregung."""

    bed_temperature_celsius: float
    particle_moisture_kg_per_kg: float
    diameter_um: float
    evaporation_kg_per_h: float
    dry_air_flow_kg_per_h: float
    product_flow_kg_per_h: float
    saturation_degree: float
    minimum_fluidization_velocity_m_per_s: float
    terminal_velocity_m_per_s: float
    power_total_kw: float
    power_shares: dict[str, float]
    residual_inf_norm: float
    """Skaliertes Residuum des Lösers am Berichtspunkt, siehe ``steady_state.py``."""
    feasible: bool
    """Zulässigkeitsflag des Lösers (Fluidisierungsfenster, Sättigung, Feuchtebereich)."""

    def format(self) -> str:
        """Menschenlesbarer Bericht für Run-Logs und Abstimmung mit TUHH-A."""
        shares = ", ".join(f"{k} {v:5.1%}" for k, v in self.power_shares.items())
        return (
            f"Betttemperatur          {self.bed_temperature_celsius:8.2f} °C\n"
            f"Partikelfeuchte         {self.particle_moisture_kg_per_kg:8.5f} kg/kg\n"
            f"d50                     {self.diameter_um:8.1f} µm\n"
            f"Verdampfungsrate        {self.evaporation_kg_per_h:8.3f} kg/h\n"
            f"Trockenluftstrom        {self.dry_air_flow_kg_per_h:8.1f} kg/h\n"
            f"Produktmassenstrom      {self.product_flow_kg_per_h:8.3f} kg/h\n"
            f"Sättigungsgrad Abluft   {self.saturation_degree:8.4f} -\n"
            f"Fluidisierungsfenster   {self.minimum_fluidization_velocity_m_per_s:8.3f}"
            f" .. {self.terminal_velocity_m_per_s:.3f} m/s\n"
            f"Leistung                {self.power_total_kw:8.3f} kW  ({shares})\n"
            f"Zulässig                {'ja' if self.feasible else 'nein':>8s}\n"
            f"Residuum (skaliert)     {self.residual_inf_norm:8.2e}"
        )


def steady_state_report(simulator: GreyBoxFluidBedGranulator) -> SteadyStateReport:
    """Exakter Beharrungszustand bei nomineller Anregung.

    Ersetzt seit Schritt 3 die transiente Integration durch den exakten Löser aus
    :mod:`twinlink_ml.simulation.steady_state`. Eine Simulationsdauer ist damit kein
    Parameter mehr -- die frühere transiente Fassung berichtete bei
    ``duration_s = 30 000 s`` einen Zwischenzustand statt des Beharrungszustands
    (610.1 µm statt 750.0 µm bei d50, siehe ``docs/04_stand_ap1.md``, Befund A). Die
    Signatur ist damit enger als zuvor: ein ``ProcessSimulator`` im Allgemeinen kann
    keine rechte Seite liefern, gegen deren Nullstelle sich lösen ließe (Dyssol nicht),
    der exakte Bericht ist deshalb auf :class:`GreyBoxFluidBedGranulator` beschränkt.
    """
    solution = solve_steady_state(simulator)
    total = solution.diagnostics["power_total_kw"]
    shares = {
        key: solution.diagnostics[f"power_{key}_kw"] / total
        for key in ("heater", "fan", "pump", "atomiser")
    }
    return SteadyStateReport(
        bed_temperature_celsius=solution.bed_temperature_celsius,
        particle_moisture_kg_per_kg=solution.particle_moisture_kg_per_kg,
        diameter_um=solution.diameter_m * 1e6,
        evaporation_kg_per_h=solution.diagnostics["evaporation_rate_kg_per_s"] * 3600.0,
        dry_air_flow_kg_per_h=solution.dry_air_mass_flow_kg_per_s * 3600.0,
        product_flow_kg_per_h=solution.diagnostics["product_mass_flow_kg_per_s"] * 3600.0,
        saturation_degree=solution.diagnostics["saturation_degree"],
        minimum_fluidization_velocity_m_per_s=solution.diagnostics[
            "minimum_fluidization_velocity_m_per_s"
        ],
        terminal_velocity_m_per_s=solution.diagnostics["terminal_velocity_m_per_s"],
        power_total_kw=total,
        power_shares=shares,
        residual_inf_norm=solution.residual_inf_norm,
        feasible=solution.feasible,
    )


def _solid_residence_time_s(
    config: ProcessConfig, spray_rate_kg_per_h: float, solid_fraction: float
) -> float:
    """Geschlossene Form der Feststoff-Verweilzeit (Befund 47/M1, ``docs/04_stand_ap1.md``).

    Bettmasse-Sollwert (``geometry.bed_mass_solid_setpoint_kg``) geteilt durch den
    stationären Feststoff-Zulauf am Linearisierungspunkt (Sprühsuspension plus
    Saatgutstrom, dieselbe Bilanz wie ``GreyBoxFluidBedGranulator.evaluate_derivative``,
    ``solid_inflow = spray_rate * solid_fraction + seed_flow``). Eine reine
    Konfigurationskonstante am nominellen Betriebspunkt -- unabhängig von
    ``bed_temperature_celsius``, solange Sprührate/Feststoffanteil/Bettmasse-Sollwert
    unverändert bleiben (numerisch bestätigt an drei Beharrungszuständen, Befund 47/M1).
    """
    spray_solid_kg_per_s = spray_rate_kg_per_h / _SECONDS_PER_HOUR * solid_fraction
    seed_flow_kg_per_s = config.operation.seed_mass_flow_kg_per_h / _SECONDS_PER_HOUR
    solid_inflow_kg_per_s = spray_solid_kg_per_s + seed_flow_kg_per_s
    return config.geometry.bed_mass_solid_setpoint_kg / solid_inflow_kg_per_s


def _match_single_mode(taus: list[float], reference: float, *, rtol: float) -> int | None:
    """Index in ``taus``, dessen Wert um weniger als ``rtol`` relativ von ``reference`` abweicht.

    ``None`` bei keinem oder bei mehr als einem Treffer.

    Rangposition (``tau_1_s`` .. ``tau_4_s``, sortiert von schnell nach langsam) ist
    keine stabile Identität: welcher Rang der Feststoff-Regelkreis-Mode
    (``mass_control_time_constant_s``, s. :func:`time_constant_report`) einnimmt, hängt
    vom Betriebspunkt ab -- bei 60 °C Eintrittstemperatur Rang 2, bei 90/120 °C Rang 3
    (numerisch geprüft). Deshalb Identifikation über den Zahlenwert der jeweiligen
    geschlossenen Form, nicht über die Position in der sortierten Liste. Mehr als ein
    Treffer (zwei Moden zufällig innerhalb der Toleranz) wird nicht geraten, sondern als
    nicht identifizierbar behandelt.
    """
    matches = [i for i, tau in enumerate(taus) if math.isclose(tau, reference, rel_tol=rtol)]
    return matches[0] if len(matches) == 1 else None


def time_constant_report(
    simulator: GreyBoxFluidBedGranulator, *, mode_match_rtol: float = _MODE_MATCH_RTOL
) -> dict[str, float | str | None]:
    """Zeitkonstanten im exakten Beharrungszustand plus abgeleitete Kennzahlen.

    Der Linearisierungspunkt kommt seit Schritt 3b aus dem exakten Löser
    (:mod:`twinlink_ml.simulation.steady_state`) statt aus einer transienten
    Integration -- dieselbe Begründung wie bei :func:`steady_state_report` (Befund A):
    zwei verschiedene Vorstellungen vom Beharrungszustand im selben Modul wäre genau die
    Divergenz, die dort vermieden wurde. Numerisch nahezu folgenlos (siehe
    ``tests/test_diagnostics.py``): drei der vier Zeitkonstanten ändern sich relativ um
    ~1e-10 (Rauschen zweier verschiedener Fließkomma-Pfade auf demselben Punkt), die
    vierte (Feststoff-Regelkreis, Zeitkonstante = ``mass_control_time_constant_s``) ist
    pfadunabhängig und bleibt bitgleich.

    ``recommended_sequence_length_fast`` = ``ceil(5 * tau_schnellste / dt)``: Untergrenze
    für die Abtastauflösung -- so viele Schritte braucht es, damit die schnellste Mode
    überhaupt sichtbar ist. **KEINE Empfehlung für die Fensterlänge** -- dafür
    ``recommended_sequence_length_coupled_slow`` verwenden. ``steps_for_slowest_mode``
    (``ceil(3 * tau_langsamste / dt)``) rechnet dieselbe Faktor-3-Fensterlänge gegen die
    langsamste ALLER vier Moden, einschließlich der Feststoff-Verweilzeit (Befund 47/M1,
    ~16000 s) -- als Fensterlänge unbrauchbar groß, siehe ``docs/04_stand_ap1.md``,
    Befund 49.

    Zwei der vier Moden sind reine Konfigurationskonstanten mit geschlossener Form,
    unabhängig vom Betriebspunkt (Befund 47/M1): der Feststoff-Regelkreis
    (``tau = mass_control_time_constant_s``) und die Feststoff-Verweilzeit (s.
    :func:`_solid_residence_time_s`). Nur das verbleibende, gekoppelte Teilsystem
    (Wassermasse, Betttemperatur) hängt vom Betriebspunkt ab und bestimmt damit die
    tatsächlich nötige Fensterlänge -- ``tau_coupled_fast_s``/``tau_coupled_slow_s`` sind
    dessen zwei Moden, per Wertevergleich identifiziert (s. :func:`_match_single_mode`,
    Toleranz ``mode_match_rtol``), nicht per Rangposition in ``tau_1_s``..``tau_4_s``
    (die bleiben unverändert erhalten, Rückwärtskompatibilität der Logs). Lässt sich eine
    der beiden strukturellen Moden nicht eindeutig zuordnen (kein oder mehr als ein
    Treffer innerhalb der Toleranz -- z. B. bei einer Konfiguration mit zufällig
    zusammenfallenden Zeitkonstanten oder einer anderen Zustandszahl), bleiben
    ``tau_mass_control_s``/``tau_residence_s``/``tau_coupled_fast_s``/
    ``tau_coupled_slow_s``/``recommended_sequence_length_coupled_slow`` auf ``None``,
    mit einer Notiz in ``mode_identification_note``.
    """
    solution = solve_steady_state(simulator)
    backend = simulator.backend
    state = backend.asarray(solution.state[None, :])
    action = backend.asarray(simulator.nominal_action(1))
    disturbance = backend.asarray(simulator.nominal_disturbance(1))
    try:
        time_constants = dict(simulator.time_constants(state, action, disturbance))
    except NotImplementedError as error:
        raise TypeError(f"{type(simulator).__name__} liefert keine Zeitkonstanten") from error
    taus = sorted(v for k, v in time_constants.items() if k.startswith("tau_"))
    report: dict[str, float | str | None] = dict(time_constants)
    if taus:
        dt = simulator.sample_time_s
        report["sample_time_s"] = dt
        report["fastest_mode_samples"] = taus[0] / dt
        report["recommended_sequence_length_fast"] = float(np.ceil(5.0 * taus[0] / dt))
        report["steps_for_slowest_mode"] = float(np.ceil(3.0 * taus[-1] / dt))

        action_np = backend.to_numpy(action)[0]
        disturbance_np = backend.to_numpy(disturbance)[0]
        mass_control_tau = simulator.config.operation.mass_control_time_constant_s
        residence_tau = _solid_residence_time_s(
            simulator.config, float(action_np[2]), float(disturbance_np[2])
        )
        mass_control_idx = _match_single_mode(taus, mass_control_tau, rtol=mode_match_rtol)
        residence_idx = _match_single_mode(taus, residence_tau, rtol=mode_match_rtol)

        report["tau_mass_control_s"] = (
            taus[mass_control_idx] if mass_control_idx is not None else None
        )
        report["tau_residence_s"] = taus[residence_idx] if residence_idx is not None else None
        report["tau_coupled_fast_s"] = None
        report["tau_coupled_slow_s"] = None
        report["recommended_sequence_length_coupled_slow"] = None
        report["mode_identification_note"] = None

        if mass_control_idx is None or residence_idx is None:
            report["mode_identification_note"] = (
                "mass_control- oder residence-Mode nicht eindeutig identifizierbar "
                f"(rtol={mode_match_rtol:g}) -- coupled_fast/slow und "
                "recommended_sequence_length_coupled_slow nicht berechnet."
            )
        else:
            structural_indices = {mass_control_idx, residence_idx}
            coupled_indices = [i for i in range(len(taus)) if i not in structural_indices]
            if len(coupled_indices) != 2:
                report["mode_identification_note"] = (
                    f"{len(coupled_indices)} statt 2 verbleibende Moden nach Abzug von "
                    "mass_control/residence -- Zustandszahl weicht vom erwarteten Fall "
                    "(N_STATE=4, ein gekoppeltes Paar) ab, coupled_fast/slow nicht "
                    "berechnet."
                )
            else:
                coupled_fast, coupled_slow = sorted(taus[i] for i in coupled_indices)
                report["tau_coupled_fast_s"] = coupled_fast
                report["tau_coupled_slow_s"] = coupled_slow
                report["recommended_sequence_length_coupled_slow"] = float(
                    np.ceil(3.0 * coupled_slow / dt)
                )
    return report


def static_explained_variance(
    trajectory: Trajectory, *, quadratic: bool = True
) -> dict[str, float]:
    r"""Anteil der Varianz jeder Ausgangsgröße, den eine statische Abbildung erklärt.

    Regression von :math:`y[k]` auf :math:`(\mathbf u[k], \mathbf z[k])` -- ohne jede
    Historie. Optional mit quadratischen Termen, um Nichtlinearität nicht mit Dynamik zu
    verwechseln.

    Lesart:

    * :math:`R^2 > 0.95` -- die Größe ist praktisch statisch. Ein Sequenzmodell kann hier
      nichts zeigen; der Vergleich ist nicht aussagekräftig.
    * :math:`R^2 < 0.8` -- es gibt echten dynamischen Anteil. Erst hier ist die Frage
      "welche Architektur" sinnvoll gestellt.

    Diese Kennzahl ist bewusst **vor** dem ersten Modelltraining zu berechnen, für jeden
    Datensatz und jede Zielgröße. Sie kostet Sekunden und verhindert Monate an
    Interpretation von Rauschen.
    """
    n_runs, n_steps, _ = trajectory.actions.shape
    inputs = np.concatenate([trajectory.actions, trajectory.disturbances], axis=-1)
    inputs = inputs.reshape(n_runs * n_steps, -1)
    if quadratic:
        inputs = np.concatenate([inputs, inputs**2], axis=-1)
    design = np.concatenate([inputs, np.ones((inputs.shape[0], 1))], axis=-1)

    names = trajectory.metadata.get("output_names", ())
    outputs = trajectory.outputs.reshape(n_runs * n_steps, -1)
    result: dict[str, float] = {}
    for index in range(outputs.shape[1]):
        target = outputs[:, index]
        variance = float(np.var(target))
        label = names[index] if index < len(names) else f"output_{index}"
        if variance <= 0.0:
            result[label] = float("nan")
            continue
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        result[label] = 1.0 - float(np.var(residual)) / variance
    return result


def mass_balance_residual(trajectory: Trajectory, *, solid_index: int = IDX_SOLID_MASS) -> float:
    """Relatives Restglied der integralen Feststoffbilanz eines Rollouts.

    Gedacht als Zusicherung für generierte Datensätze: ein Datensatz mit
    Bilanzrestglied über :math:`10^{-6}` sollte nicht ins Training gehen.
    """
    dt = float(trajectory.time_s[1] - trajectory.time_s[0])
    product = trajectory.diagnostics["product_mass_flow_kg_per_s"]
    accumulated = trajectory.states[:, -1, solid_index] - trajectory.states[:, 0, solid_index]
    withdrawn = np.sum(product, axis=1) * dt
    scale = np.maximum(np.abs(accumulated) + np.abs(withdrawn), 1e-12)
    return float(np.max(np.abs(accumulated + withdrawn - _solid_inflow(trajectory, dt)) / scale))


def _solid_inflow(trajectory: Trajectory, dt: float) -> FloatArray:
    spray_rate = trajectory.actions[..., 2] / 3600.0
    solid_fraction = trajectory.disturbances[..., 2]
    return np.asarray(np.sum(spray_rate * solid_fraction, axis=1) * dt, dtype=np.float64)
