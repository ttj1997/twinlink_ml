r"""Problem-Table-Algorithmus: das thermodynamische Minimum an Zusatzwaerme.

Beantwortet eine andere Frage als ``balance.evaluate_cascade``: nicht "wie viel Brennstoff
braucht DIESE Verschaltung", sondern "wie viel Brennstoff braucht JEDE DENKBARE
Verschaltung mindestens" -- der Massstab, gegen den ``evaluate_cascade``-Ergebnisse (und
jeder spaetere Regler) gemessen werden. Fuer das reine Zielwertproblem (kein Netzwerk-
entwurf, nur der minimale Heiz-/Kuehlbedarf) ist der Problem-Table-Algorithmus exakt und
braucht keinen Solver -- ein LP waere erst fuer die Frage noetig, welcher Waermeuebertrager
wohin kommt (Netzwerksynthese), nicht Teil dieses Moduls.

Kondensation: keine neue Physik
--------------------------------
Die Abluftquellen haben KEINE konstante Waermekapazitaet -- unterhalb ihres Taupunkts wird
latente Waerme frei, die Enthalpie-Temperatur-Kurve knickt dort scharf. Ein Strom mit
konstantem cp waere hier falsch. Dieses Modul fuehrt dafuer KEINE zweite Kondensationsformel
ein: :func:`hot_segments_for_source` tastet die Kurve am vorhandenen, bereits getesteten
:func:`~.balance.evaluate_cascade` (``recovery_mode="condensing"``) ab -- derselbe Code, der
auch die Kaskadenrechnung selbst treibt -- und baut daraus eine stueckweise lineare
Naeherung. Oberhalb des Taupunkts ist die Kurve exakt linear (Feuchte konstant, kein
Diskretisierungsfehler moeglich, ein Segment genuegt); unterhalb wird sie an
``n_segments_below_dew_point`` Stuetzstellen abgetastet, mit dem Taupunkt selbst als exakter
Intervallgrenze (nicht dem Zufall eines Rasters ueberlassen). Wie fein diese Abtastung sein
muss, ist eine empirische Frage -- siehe ``docs/06_waermenetz.md`` fuer die
Konvergenzstudie.

Problem-Table-Algorithmus (Standardverfahren, hier nur die Kurzfassung; Lehrbuch z. B.
Kemp, "Pinch Analysis and Process Integration")
------------------------------------------------------------------------------------------
1. Verschobene Temperaturen: heisse Segmente um ``dT_min/2`` nach unten, kalte um
   ``dT_min/2`` nach oben (:meth:`StreamSegment.shifted`) -- macht die Mindestgraedigkeit
   zu einer einzigen gemeinsamen Temperaturskala, auf der sich heiss und kalt beruehren
   duerfen.
2. Alle verschobenen Segmentgrenzen (heiss und kalt) bilden die Intervallgrenzen.
3. Je Intervall: Waermebilanz = (Summe CP heiss aktiv - Summe CP kalt aktiv) * Delta T.
   Kaskade von der hoechsten Temperatur nach unten (Startwert 0). Der betragsmaessig
   groesste negative Kaskadenwert ist der minimale Heizbedarf (Hot Utility); addiert man
   ihn auf die gesamte Kaskade, wird sie ueberall nichtnegativ, ihr Endwert ist der minimale
   Kuehlbedarf (Cold Utility), und die Stelle, an der sie exakt null wird, ist die
   Pinch-Temperatur.

Die untere Schranke, keine Bauvorschrift
------------------------------------------
Der hier berechnete Heizbedarf setzt beliebig viele, ideal gegeneinander geschaltete
Waermeuebertrager voraus. Er ist eine UNTERE SCHRANKE fuer jede reale Verschaltung mit
DERSELBEN Grädigkeit, keine Aussage darueber, ob sie mit zwei Uebertragern erreichbar ist.
``tests/test_heatnetwork_pinch.py::test_pinch_target_never_exceeds_the_best_of_six_cases``
macht diese Ungleichung explizit pruefbar.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from ..simulation.backend import NumpyBackend
from ..simulation.config import GasConfig
from ..simulation.thermo import dew_point_celsius
from .balance import SinkState, SourceState, evaluate_cascade
from .config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig

__all__ = [
    "PinchResult",
    "StreamSegment",
    "cold_segment_for_sink",
    "composite_curve_kw",
    "compute_pinch_target",
    "dew_point_of_source_celsius",
    "evaluate_pinch_target",
    "hot_segments_for_source",
]

_BACKEND = NumpyBackend()
_BOUNDARY_TOLERANCE_K = 1e-6
"""Toleranz beim Vergleich verschobener Temperaturgrenzen (Float-Rundung nach der
Rundung auf 9 Nachkommastellen in :func:`compute_pinch_target`/:func:`composite_curve_kw`)."""

_DUMMY_SINK_NAME = "__pinch_probe_sink__"
"""Name der Wegwerf-Senke in :func:`_recoverable_power_to_floor_kw` -- taucht in keinem
Ergebnis auf, dient nur dazu, ``evaluate_cascade`` ueberhaupt aufrufen zu koennen (das Feld
``source_available_power_kw`` ist unabhaengig von der Senke, siehe dort)."""


@dataclass(frozen=True, slots=True)
class StreamSegment:
    r"""Ein linearer Teilabschnitt eines Stroms mit konstanter Waermekapazitaetsrate.

    :param name: Quellen-/Senkenname (fuer Diagnose, Composite-Curve-Zuordnung).
    :param is_hot: ``True`` fuer Abluftquellen (kuehlen sich ab), ``False`` fuer Senken
        (erwaermen sich).
    :param high_temperature_celsius: obere Grenze dieses Abschnitts, UNVERSCHOBEN (reale
        Temperatur). Bei einer Quelle die Eintritts-, bei einer Senke die Austrittsseite.
    :param low_temperature_celsius: untere Grenze, analog.
    :param capacity_rate_kw_per_k: :math:`\dot m\,(c_{p,L}+Y\,c_{p,D})` dieses Abschnitts in
        kW/K, stets nichtnegativ.
    """

    name: str
    is_hot: bool
    high_temperature_celsius: float
    low_temperature_celsius: float
    capacity_rate_kw_per_k: float

    def __post_init__(self) -> None:
        if self.high_temperature_celsius <= self.low_temperature_celsius:
            raise ValueError(
                "high_temperature_celsius muss über low_temperature_celsius liegen, sind "
                f"{self.high_temperature_celsius}/{self.low_temperature_celsius}"
            )
        if self.capacity_rate_kw_per_k < 0.0:
            raise ValueError(
                f"capacity_rate_kw_per_k darf nicht negativ sein, ist {self.capacity_rate_kw_per_k}"
            )

    def duty_kw(self) -> float:
        """Waermemenge dieses Abschnitts, kW."""
        return self.capacity_rate_kw_per_k * (
            self.high_temperature_celsius - self.low_temperature_celsius
        )

    def shifted(self, minimum_pinch_k: float) -> tuple[float, float]:
        """(hohe, niedrige) verschobene Grenze.

        Heiss um ``dT_min/2`` nach unten, kalt nach oben (Schritt 1 des
        Problem-Table-Algorithmus, siehe Moduldocstring).
        """
        half = minimum_pinch_k / 2.0
        shift = -half if self.is_hot else half
        return self.high_temperature_celsius + shift, self.low_temperature_celsius + shift


def dew_point_of_source_celsius(source: HeatSourceConfig, gas: GasConfig) -> float:
    """Taupunkt einer Quelle aus ihrer eigenen Abluftfeuchte.

    Dieselbe Buck-Korrelation wie ``balance.py`` (``thermo.dew_point_celsius``) -- keine
    zweite Formel.
    """
    return float(
        _BACKEND.to_numpy(
            dew_point_celsius(_BACKEND, gas, _BACKEND.asarray(source.outlet_humidity_kg_per_kg))
        )
    )


def _recoverable_power_to_floor_kw(
    source: HeatSourceConfig, gas: GasConfig, floor_temperature_celsius: float
) -> float:
    r"""EIN Abtastpunkt der Enthalpie-Temperatur-Kurve.

    :math:`\dot Q_{rec}` beim Kuehlen von ``source`` von ihrer Austrittstemperatur bis
    ``floor_temperature_celsius``. Ruft dafuer ``balance.evaluate_cascade`` mit
    ``recovery_mode="condensing"`` auf und liest ``source_available_power_kw`` aus --
    dieses Feld wird dort direkt aus
    Ausgangstemperatur und Kaltgrenze berechnet, UNABHAENGIG von der Senke und den Stufen
    (siehe ``balance.py``, Berechnung nach der Stufenschleife). Die Wegwerf-Senke unten
    begrenzt also nichts; sie existiert nur, weil ``evaluate_cascade`` eine Stufenliste
    verlangt. Keine eigene Kondensationsformel -- siehe Moduldocstring.
    """
    if floor_temperature_celsius >= source.outlet_temperature_celsius:
        return 0.0
    dummy_sink = HeatSinkConfig(
        name=_DUMMY_SINK_NAME,
        dry_air_mass_flow_kg_per_s=1.0e9,
        humidity_kg_per_kg=0.0,
        intake_temperature_celsius=-40.0,
        setpoint_temperature_celsius=source.outlet_temperature_celsius + 50.0,
    )
    network = HeatNetworkConfig(
        gas=gas,
        sources=(source,),
        sinks=(dummy_sink,),
        minimum_pinch_k=1.0e-6,
        recovery_mode="condensing",
        cold_temperature_celsius=floor_temperature_celsius,
    )
    stages = ((source.name, dummy_sink.name),)
    result = evaluate_cascade(
        network,
        stages,
        (SourceState.from_config(source),),
        (SinkState.from_config(dummy_sink),),
    )
    return float(result.source_available_power_kw[0, 0])


def hot_segments_for_source(
    source: HeatSourceConfig,
    gas: GasConfig,
    floor_temperature_celsius: float,
    *,
    n_segments_below_dew_point: int = 50,
) -> tuple[StreamSegment, ...]:
    r"""Zerlegt eine Quelle in stueckweise lineare Segmente.

    Von ``outlet_temperature_celsius`` bis ``floor_temperature_celsius`` (= ``T_kalt``
    dieses Laufs). Liegt der Taupunkt bei oder unter ``floor_temperature_celsius``,
    kondensiert die Quelle in diesem Lauf gar nicht (z. B. Quelle 2 oberhalb von ca. 25 °C, siehe
    ``docs/06_waermenetz.md``) -- dann ein einziges, exakt lineares Segment. Sonst: ein
    exaktes lineares Segment oberhalb des Taupunkts (Feuchte konstant) plus
    ``n_segments_below_dew_point`` abgetastete Segmente darunter (siehe Moduldocstring).

    :raises ValueError: wenn ``n_segments_below_dew_point < 1``.
    """
    t_out = source.outlet_temperature_celsius
    if floor_temperature_celsius >= t_out:
        return ()
    if n_segments_below_dew_point < 1:
        raise ValueError(
            f"n_segments_below_dew_point muss >= 1 sein, ist {n_segments_below_dew_point}"
        )

    dew_point = dew_point_of_source_celsius(source, gas)
    sensible_capacity_rate_kw_per_k = (
        source.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + source.outlet_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        / 1000.0
    )

    segments: list[StreamSegment] = []
    if dew_point <= floor_temperature_celsius:
        segments.append(
            StreamSegment(
                source.name, True, t_out, floor_temperature_celsius, sensible_capacity_rate_kw_per_k
            )
        )
        return tuple(segments)

    top_of_condensing_region = min(dew_point, t_out)
    if top_of_condensing_region < t_out:
        segments.append(
            StreamSegment(
                source.name, True, t_out, top_of_condensing_region, sensible_capacity_rate_kw_per_k
            )
        )

    grid = np.linspace(
        top_of_condensing_region, floor_temperature_celsius, n_segments_below_dew_point + 1
    )
    duty_at_grid_kw = np.array(
        [_recoverable_power_to_floor_kw(source, gas, float(t)) for t in grid]
    )
    for index in range(n_segments_below_dew_point):
        t_high, t_low = float(grid[index]), float(grid[index + 1])
        if t_high - t_low <= 0.0:
            continue
        capacity_rate = (duty_at_grid_kw[index + 1] - duty_at_grid_kw[index]) / (t_high - t_low)
        segments.append(StreamSegment(source.name, True, t_high, t_low, max(capacity_rate, 0.0)))
    return tuple(segments)


def cold_segment_for_sink(sink: HeatSinkConfig, gas: GasConfig) -> StreamSegment:
    """Eine Senke als einzelnes Segment.

    Rein sensible Erwaermung, konstante Feuchte, also exakt linear (kein
    Diskretisierungsbedarf, anders als bei den Quellen).
    """
    capacity_rate = (
        sink.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + sink.humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        / 1000.0
    )
    return StreamSegment(
        sink.name,
        False,
        sink.setpoint_temperature_celsius,
        sink.intake_temperature_celsius,
        capacity_rate,
    )


@dataclass(frozen=True, slots=True)
class PinchResult:
    """Ergebnis eines Problem-Table-Laufs.

    :ivar minimum_hot_utility_kw: minimaler Heizbedarf (Hot Utility) -- die untere Schranke
        aus dem Moduldocstring.
    :ivar minimum_cold_utility_kw: minimaler Kuehlbedarf (Cold Utility).
    :ivar pinch_temperature_hot_side_celsius: Pinch-Temperatur auf der heissen Skala
        (reale Quellentemperatur an der Engpassstelle).
    :ivar pinch_temperature_cold_side_celsius: Pinch-Temperatur auf der kalten Skala;
        Differenz zu ``pinch_temperature_hot_side_celsius`` ist per Konstruktion ``dT_min``.
    :ivar interval_boundaries_shifted_celsius: verschobene Intervallgrenzen, absteigend
        sortiert, Laenge ``n+1`` fuer ``n`` Intervalle.
    :ivar cascade_kw: Kaskade NACH Zugabe von ``minimum_hot_utility_kw`` (ueberall
        nichtnegativ), an denselben Stuetzstellen wie ``interval_boundaries_shifted_celsius``
        -- das ist die Grand Composite Curve (PA4).
    """

    minimum_hot_utility_kw: float
    minimum_cold_utility_kw: float
    pinch_temperature_hot_side_celsius: float
    pinch_temperature_cold_side_celsius: float
    interval_boundaries_shifted_celsius: tuple[float, ...]
    cascade_kw: tuple[float, ...]


def _segment_active(
    high_low_cp: tuple[float, float, float], t_high: float, t_low: float
) -> bool:
    high, low, _ = high_low_cp
    return high >= t_high - _BOUNDARY_TOLERANCE_K and low <= t_low + _BOUNDARY_TOLERANCE_K


def compute_pinch_target(
    hot_segments: Sequence[StreamSegment],
    cold_segments: Sequence[StreamSegment],
    *,
    minimum_pinch_k: float,
) -> PinchResult:
    """Problem-Table-Algorithmus ueber bereits gebaute Segmente.

    Siehe Moduldocstring, Schritte 1-3. Segmente duerfen aus mehreren Quellen/Senken UND
    aus mehreren Teilabschnitten je Quelle (Kondensationskurve) stammen -- der Algorithmus
    behandelt jedes Segment wie einen eigenen Teilstrom mit konstantem CP, das ist Standard
    bei nichtlinearen Stroemen in der Pinch-Analyse.

    :raises ValueError: wenn ``minimum_pinch_k`` negativ ist oder keine Segmente uebergeben
        wurden.
    """
    if minimum_pinch_k < 0.0:
        raise ValueError(f"minimum_pinch_k darf nicht negativ sein, ist {minimum_pinch_k}")
    if not hot_segments and not cold_segments:
        raise ValueError("mindestens ein Segment (heiss oder kalt) noetig")

    shifted_hot = [
        (*seg.shifted(minimum_pinch_k), seg.capacity_rate_kw_per_k) for seg in hot_segments
    ]
    shifted_cold = [
        (*seg.shifted(minimum_pinch_k), seg.capacity_rate_kw_per_k) for seg in cold_segments
    ]

    boundary_set = {round(t, 9) for high, low, _ in shifted_hot for t in (high, low)}
    boundary_set |= {round(t, 9) for high, low, _ in shifted_cold for t in (high, low)}
    boundaries = sorted(boundary_set, reverse=True)
    if len(boundaries) < 2:
        raise ValueError("weniger als zwei unterschiedliche Temperaturgrenzen -- leere Stroeme?")

    net_kw: list[float] = []
    for t_high, t_low in pairwise(boundaries):
        delta_t = t_high - t_low
        cp_hot = sum(
            entry[2] for entry in shifted_hot if _segment_active(entry, t_high, t_low)
        )
        cp_cold = sum(
            entry[2] for entry in shifted_cold if _segment_active(entry, t_high, t_low)
        )
        net_kw.append((cp_hot - cp_cold) * delta_t)

    cascade = [0.0]
    for net in net_kw:
        cascade.append(cascade[-1] + net)

    minimum_hot_utility_kw = max(0.0, -min(cascade))
    feasible_cascade = tuple(value + minimum_hot_utility_kw for value in cascade)
    minimum_cold_utility_kw = feasible_cascade[-1]

    pinch_index = min(range(len(cascade)), key=lambda i: cascade[i])
    pinch_shifted_celsius = boundaries[pinch_index]
    half = minimum_pinch_k / 2.0

    return PinchResult(
        minimum_hot_utility_kw=minimum_hot_utility_kw,
        minimum_cold_utility_kw=minimum_cold_utility_kw,
        pinch_temperature_hot_side_celsius=pinch_shifted_celsius + half,
        pinch_temperature_cold_side_celsius=pinch_shifted_celsius - half,
        interval_boundaries_shifted_celsius=tuple(boundaries),
        cascade_kw=feasible_cascade,
    )


def composite_curve_kw(segments: Sequence[StreamSegment]) -> tuple[tuple[float, float], ...]:
    r"""Verbundkurve (heiss ODER kalt, je nachdem welche ``segments`` uebergeben werden).

    ``(Temperatur, kumulierte Leistung)``-Stuetzpunkte -- UNVERSCHOBENE (reale)
    Temperaturen, das, was man dem Verbundpartner zeigt (PA4). Kumuliert von der
    niedrigsten beteiligten Temperatur aufwaerts (:math:`H=0` am kalten Ende), Segmente wie
    im Problem-Table-Algorithmus additiv ueber ihre jeweiligen Gueltigkeitsintervalle --
    dieselbe Intervallzerlegung, hier ohne Verschiebung und ohne Kaskade.
    """
    if not segments:
        return ()
    boundary_temperatures = (
        (seg.high_temperature_celsius, seg.low_temperature_celsius) for seg in segments
    )
    boundaries = sorted({round(t, 9) for pair in boundary_temperatures for t in pair})
    cumulative_kw = 0.0
    points: list[tuple[float, float]] = [(boundaries[0], 0.0)]
    for t_low, t_high in pairwise(boundaries):
        active_cp = sum(
            seg.capacity_rate_kw_per_k
            for seg in segments
            if seg.low_temperature_celsius <= t_low + _BOUNDARY_TOLERANCE_K
            and seg.high_temperature_celsius >= t_high - _BOUNDARY_TOLERANCE_K
        )
        cumulative_kw += active_cp * (t_high - t_low)
        points.append((t_high, cumulative_kw))
    return tuple(points)


def evaluate_pinch_target(
    sources: Sequence[HeatSourceConfig],
    sinks: Sequence[HeatSinkConfig],
    gas: GasConfig,
    *,
    cold_temperature_celsius: float,
    minimum_pinch_k: float,
    n_segments_below_dew_point: int = 50,
) -> tuple[PinchResult, tuple[StreamSegment, ...], tuple[StreamSegment, ...]]:
    """Baut Segmente fuer alle ``sources``/``sinks`` und wertet den Pinch-Zielwert aus.

    Quellen-Ziel: ``cold_temperature_celsius``, Senken-Ziel: ihr eigener Sollwert; ausgewertet
    ueber :func:`compute_pinch_target`.

    :returns: ``(PinchResult, hot_segments, cold_segments)`` -- die Segmente zusaetzlich
        zurueckgegeben, damit :func:`composite_curve_kw` sie ohne erneute Abtastung
        wiederverwenden kann (die Abtastung ueber ``evaluate_cascade`` ist der teure Teil).
    """
    hot_segments = tuple(
        segment
        for source in sources
        for segment in hot_segments_for_source(
            source,
            gas,
            cold_temperature_celsius,
            n_segments_below_dew_point=n_segments_below_dew_point,
        )
    )
    cold_segments = tuple(cold_segment_for_sink(sink, gas) for sink in sinks)
    result = compute_pinch_target(hot_segments, cold_segments, minimum_pinch_k=minimum_pinch_k)
    return result, hot_segments, cold_segments
