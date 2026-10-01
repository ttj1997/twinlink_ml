r"""Kaskadierte Wärmerückgewinnungsbilanz: Abluftquellen -> Prozessluft-Senken.

Dieses Modul ist die Spezifikation für das künftige Dyssol-Bilanzmodul (AP3 des
Software-Integrationspartners, siehe ``docs/06_waermenetz.md``) -- es portiert die
Rechenvorschriften einer bestehenden
Excel-Bilanz des Verbundpartners (zwei Trocknerlinien, zwei Wärmeübertrager, zwei
Lufterhitzer), korrigiert dabei aber drei Defekte der Vorlage. Alle drei sind hier benannt,
weil ein künftiges Dyssol-Modul dieselben Defekte nicht erben darf, nur weil es dieselbe
Vorlage liest.

Grundformeln (EA1, unverändert gegenüber der Vorlage)
------------------------------------------------------
Rückgewinnbare Leistung je Quelle:

.. math:: \dot Q_{rec} = \dot m_L \bigl(T_{out} - (T_{dp}(Y_{out}) + \Delta T_{approach})\bigr)
    \,(c_{p,L} + Y_{out}\, c_{p,D})

mit :math:`\Delta T_{approach} = 3\,\mathrm K` (Grenze zur Kondensation,
:data:`~.config.DEFAULT_DEW_POINT_APPROACH_K`). Brennstoffbedarf je Senke:

.. math:: \dot m_{gas} = \dot m_L\,(T_{soll} - T_{nach\,WUET})\,(c_{p,L} + Y\, c_{p,D})
    \,/\,(H_u\,\eta_H)

Taupunkt über :func:`twinlink_ml.simulation.thermo.dew_point_celsius` (Buck 1996) --
dieselbe Korrelation wie überall im Projekt, nicht neu hergeleitet oder gegen Magnus
ausgetauscht (der Verbundpartner rechnet mit Magnus; die Differenz liegt unter 0,06 K,
dokumentiert, aber irrelevant -- das Repo bleibt bei Buck, siehe ``thermo.py``,
``saturation_pressure_pa``-Docstring). Stoffwerte ausschließlich aus dem injizierten
:class:`~twinlink_ml.simulation.config.GasConfig` (``cp_dry_air_j_per_kg_k``,
``cp_vapour_j_per_kg_k``), nicht neu gesetzt.

Zweiter Rückgewinnungsmodus: Kondensation (MA1)
------------------------------------------------
:attr:`~.config.HeatNetworkConfig.recovery_mode` (Config, kein Funktionsparameter) wählt
zwischen zwei Kühlgrenzen je Quelle:

``"sensible"`` (Default, EA1 unverändert)
    Kühlung bis zur Taupunktgrenze ``T_dp(Y_out) + dT_approach`` -- keine Kondensation im
    Übertrager, jedes Ergebnis bitgleich zum Stand vor MA1.

``"condensing"``
    Kühlung bis zur Kalttemperatur :attr:`~.config.HeatNetworkConfig.
    cold_temperature_celsius` (``T_kalt``), mit zusätzlicher latenter Wärme aus dem dabei
    anfallenden Kondensat:

    .. math::
        \dot m_{Kondensat} &= \max(0,\, Y_{out} - Y_s(T_{kalt}))\, \dot m_L \\
        \dot Q_{latent} &= \dot m_{Kondensat}\, \Delta h_v(T_{kalt}) \\
        \dot Q_{sensibel} &= \dot m_L\,(T_{out} - T_{kalt})\,(c_{p,L} + Y_{out}\, c_{p,D}) \\
        \dot Q_{rec} &= \dot Q_{sensibel} + \dot Q_{latent}

    :math:`Y_s` (:func:`~twinlink_ml.simulation.thermo.saturation_humidity_kg_per_kg`) und
    :math:`\Delta h_v` (:func:`~twinlink_ml.simulation.thermo.latent_heat_j_per_kg`) kommen
    unverändert aus ``thermo.py`` -- keine zweite Sättigungs-/Verdampfungsformel. Liegt
    ``T_kalt`` über dem Taupunkt der Quelle, ist :math:`\dot m_{Kondensat}=0` und der Modus
    fällt stetig auf reine sensible Kühlung bis ``T_kalt`` zurück (kein Sprung an der
    Taupunktgrenze).

    Innerhalb der Kaskade (mehrstufig, EA2 Korrektur 2) wird die latente Wärme als fester
    Vorrat je Quelle geführt (``Q_latent`` bei ihrer Ausgangstemperatur, nicht neu bei jeder
    Stufe berechnet) und erst dann angezapft, wenn die sensible Kühlreserve einer Stufe
    bereits erschöpft ist -- Kondensation setzt physikalisch erst ein, wenn der Strom
    tatsächlich bis ``T_kalt`` abgekühlt wurde, nicht schrittweise vorher. Das Kondensat
    verlässt die Bilanz als Flüssigwasser; :attr:`HeatNetworkResult.
    source_condensate_kg_per_s` macht das für die Massenbilanz nachprüfbar (Test:
    ``tests/test_heatnetwork_balance.py``).

Drei Korrekturen gegenüber der Excel-Vorlage (EA2)
----------------------------------------------------
Die Vorlage hat drei Defekte, die :func:`evaluate_cascade` NICHT reproduziert:

1. **Taupunktbezug je Quelle.** In der Vorlage rechnet der zweite Wärmeübertrager gegen
   den Taupunkt der ERSTEN Linie -- ein Kopierfehler zwischen zwei nebeneinander
   liegenden Tabellenblättern, kein Modellierungsentscheid. Hier hat jede Quelle ihre
   eigene, aus ihrer eigenen ``outlet_humidity_kg_per_kg`` berechnete Taupunktgrenze
   (``source_floor_temperature_celsius`` in :func:`evaluate_cascade`), über alle Stufen
   hinweg konstant (die Feuchte ändert sich nicht, solange nur sensibel gekühlt wird,
   siehe Korrektur 3).

2. **Restenthalpie kaskadieren, nicht verwerfen.** Die Vorlage kappt die Lufttemperatur
   einer Senke auf ihren Sollwert und verwirft die nicht genutzte Quellenthalpie -- in
   einem dokumentierten Fall 891,8 kW bei 87,4 °C, weit oberhalb jeder Taupunktgrenze und
   damit weiter nutzbar. Hier ist jede Quelle ein Strom mit über die Kaskade ABNEHMENDER
   Temperatur: nach jeder Stufe sinkt ihre Temperatur um
   :math:`\dot Q_{genutzt} / (\dot m_L\, c_{p,eff})`, der Rest steht der nächsten Stufe
   zur Verfügung (ggf. einer anderen Senke), bis die Taupunktgrenze erreicht ist. Welcher
   Anteil je Quelle tatsächlich genutzt wurde, steht in
   :attr:`HeatNetworkResult.source_used_power_kw`/``source_discarded_power_kw`` --
   ``used + discarded == available`` bis auf Rundung (Energieerhaltung, siehe
   ``tests/test_heatnetwork_balance.py``).

3. **Mindestgrädigkeit je Stufe.** Die Vorlage prüft nirgends
   :math:`T_{Luft,aus} \le T_{Quelle} - \Delta T_{min}`; die dortige 120-°C-Kappung
   verhindert eine Verletzung nur zufällig, nicht strukturell. Hier ist die Grädigkeit
   eine harte Nebenbedingung jeder Stufe (:attr:`~.config.HeatNetworkConfig.
   minimum_pinch_k`, Platzhalter-Begründung siehe
   :data:`~.config.DEFAULT_MINIMUM_PINCH_K`). Kaskadenreihenfolge: Niedertemperaturquellen
   zuerst am kalten Ende, Hochtemperaturquellen zuletzt (sonst ist die Senkenluft bereits
   zu warm, um eine kalte Quelle noch aufzunehmen) -- Default-Regel in
   ``routing.resolve_cascade_order``, überschreibbar über
   :attr:`~.config.HeatNetworkConfig.cascade_order`.

``stage_binding_code`` in :class:`HeatNetworkResult` kennt zusätzlich zu den zwei in EA3
genannten Zuständen (``pinch``, ``dew_point``) einen dritten, ``sink_satisfied``: eine
Stufe, die die Senke ohne Anstehen einer der beiden physikalischen Grenzen auf ihren
Sollwert bringt, ist kein Grenzfall -- sie fälschlich als "Grädigkeit bindend" oder
"Taupunkt bindend" zu melden wäre selbst ein neuer Interpretationsfehler.

Anbindung an den Simulator (EA4)
----------------------------------
Der Abluftzustand kommt aus ``simulation/exhaust.py``. Wörtliche Signatur::

    def reconstruct_exhaust_state(
        backend: ArrayBackend,
        gas: GasConfig,
        material: MaterialConfig,
        isotherm: SorptionIsotherm,
        exhaust: ExhaustConfig,
        *,
        bed_temperature_celsius: Array,
        particle_moisture_kg_per_kg: Array,
        dry_air_mass_flow_kg_per_s: Array,
        inlet_absolute_humidity_kg_per_kg: Array,
    ) -> tuple[Array, Array, Array]:
        ...
        :returns: ``(T_gas_out_celsius, Y_gas_out_kg_per_kg, outlet_temperature_celsius)``

Sie LIEFERT: ``T_gas_out_celsius`` (== Betttemperatur, per Modellannahme eines ideal
durchmischten Betts), ``Y_gas_out_kg_per_kg`` und ``outlet_temperature_celsius`` (Betttemperatur
minus ``ExhaustConfig.outlet_duct_heat_loss_k``). Sie VERLANGT bereits als Eingabe (nicht als
Rückgabe), was hier als :attr:`~.config.HeatSourceConfig.dry_air_mass_flow_kg_per_s` gebraucht
wird -- dieser Massenstrom ist KEINE Rückgabegröße der Funktion, sondern eine ihrer Eingaben,
typischerweise aus ``dry_air_mass_flow_kg_per_s`` in ``diagnostics``/``SteadyStateSolution``
übernommen, nicht aus der Gasgeschwindigkeit neu berechnet (dieselbe Größe, nur schon vorher
im Aufrufer bekannt).

**Wichtig:** :class:`~.config.HeatSourceConfig.outlet_temperature_celsius` ist der AUSLASS-Kanal
(dritter Rückgabewert, ``outlet_temperature_celsius``), NICHT die Betttemperatur (erster
Rückgabewert, ``T_gas_out_celsius``/``bed_temperature_celsius``). Bei
``outlet_duct_heat_loss_k = 0.0`` (Default) sind beide numerisch gleich, der Vertrag an der
Verbundschnittstelle ist aber die Ablufttemperatur -- diese Unterscheidung ist im Repo bereits
angelegt (``exhaust.py``, Vertragskorrektur-Absatz) und wird hier nicht wieder verschmiert.

**Skalenvorbehalt.** Das Verbundmodell rechnet mit rund 8 kg/s Prozessluft
(``evaluation.recoverable_power.DISTRICT_MASS_FLOW_KG_PER_S``), der Laborsimulator mit rund
0,03 kg/s (``dry_air_mass_flow_kg_per_s`` in ``data.trajectory.diagnostics``) -- zwei
Größenordnungen. Jede kW-Zahl aus der Kopplung von Labor-Quellen mit Verbund-Senken (oder
umgekehrt) ist eine Skalierungsrechnung, keine Anlagenaussage (siehe
``docs/06_waermenetz.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..simulation.backend import NumpyBackend
from ..simulation.thermo import (
    dew_point_celsius,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)
from .config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig

__all__ = [
    "BINDING_DEW_POINT",
    "BINDING_LABELS",
    "BINDING_PINCH",
    "BINDING_SINK_SATISFIED",
    "HeatNetworkResult",
    "SinkState",
    "SourceState",
    "binding_label",
    "evaluate_cascade",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

BINDING_PINCH: Final = 0
"""Stufe durch die Mindestgrädigkeit begrenzt (EA2, Korrektur 3)."""
BINDING_DEW_POINT: Final = 1
"""Stufe durch die Taupunktgrenze der Quelle begrenzt (EA2, Korrektur 1)."""
BINDING_SINK_SATISFIED: Final = 2
"""Senke erreicht ihren Sollwert, ohne dass Grädigkeit oder Taupunkt anstehen -- kein
physikalischer Grenzfall, siehe Moduldocstring."""

BINDING_LABELS: Final[dict[int, str]] = {
    BINDING_PINCH: "pinch",
    BINDING_DEW_POINT: "dew_point",
    BINDING_SINK_SATISFIED: "sink_satisfied",
}

_POWER_TOLERANCE_W = 1e-6
"""Numerische Toleranz für den Vergleich zweier Leistungen bei der Bindungsklassifikation
-- verhindert, dass Rundung im letzten Bit eine ``sink_satisfied``-Stufe fälschlich als
``dew_point`` meldet."""

_TEMPERATURE_TOLERANCE_K = 1e-9
"""Analoge Toleranz für den Vergleich zweier Temperaturen (Pinch- gegen Sollwert-Grenze)."""


def _as_step_array(value: float, n_steps: int) -> FloatArray:
    """Skalar -> Array der Form ``(n_steps,)``, bereits ``(n_steps,)``-Arrays unverändert.

    Erlaubt, dieselbe :func:`evaluate_cascade` sowohl für eine einzelne, statische
    Auswertung (``n_steps == 1``, direkt aus einer :class:`~.config.HeatSourceConfig`
    gebaut) als auch für eine ganze Trajektorie zu verwenden (``n_steps > 1``, echte
    Zeitreihen als Eingabe) -- ein Code-Pfad für beide Fälle statt einer Sonderbehandlung
    des Skalarfalls.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(n_steps, float(array), dtype=np.float64)
    if array.shape != (n_steps,):
        raise ValueError(f"erwartete Form ({n_steps},), ist {array.shape}")
    return array


@dataclass(frozen=True, slots=True)
class SourceState:
    """Zeitreihe (oder Einzelwert, ``n_steps == 1``) einer Abluftquelle für einen Kaskadenlauf."""

    name: str
    dry_air_mass_flow_kg_per_s: FloatArray
    outlet_temperature_celsius: FloatArray
    outlet_humidity_kg_per_kg: FloatArray

    def __post_init__(self) -> None:
        shapes = {
            self.dry_air_mass_flow_kg_per_s.shape,
            self.outlet_temperature_celsius.shape,
            self.outlet_humidity_kg_per_kg.shape,
        }
        if len(shapes) != 1:
            raise ValueError(f"{self.name}: Felder haben unterschiedliche Form {shapes}")

    @classmethod
    def from_config(cls, config: HeatSourceConfig, n_steps: int = 1) -> SourceState:
        """Statische Quelle (``n_steps == 1`` per Default) aus einer :class:`HeatSourceConfig`.

        Für eine echte Trajektorie stattdessen direkt mit ``(n_steps,)``-Arrays
        konstruieren -- diese Fabrik ist der Brückenfall (Einzelstufen-Vergleich,
        Excel-Nachrechnung), nicht der RL-/Auswertungsfall.
        """
        return cls(
            name=config.name,
            dry_air_mass_flow_kg_per_s=_as_step_array(config.dry_air_mass_flow_kg_per_s, n_steps),
            outlet_temperature_celsius=_as_step_array(config.outlet_temperature_celsius, n_steps),
            outlet_humidity_kg_per_kg=_as_step_array(config.outlet_humidity_kg_per_kg, n_steps),
        )


@dataclass(frozen=True, slots=True)
class SinkState:
    """Zeitreihe (oder Einzelwert) einer Senke für einen Kaskadenlauf."""

    name: str
    dry_air_mass_flow_kg_per_s: FloatArray
    humidity_kg_per_kg: FloatArray
    intake_temperature_celsius: FloatArray
    setpoint_temperature_celsius: FloatArray

    def __post_init__(self) -> None:
        shapes = {
            self.dry_air_mass_flow_kg_per_s.shape,
            self.humidity_kg_per_kg.shape,
            self.intake_temperature_celsius.shape,
            self.setpoint_temperature_celsius.shape,
        }
        if len(shapes) != 1:
            raise ValueError(f"{self.name}: Felder haben unterschiedliche Form {shapes}")

    @classmethod
    def from_config(cls, config: HeatSinkConfig, n_steps: int = 1) -> SinkState:
        """Statische Senke (``n_steps == 1`` per Default) aus einer :class:`HeatSinkConfig`."""
        return cls(
            name=config.name,
            dry_air_mass_flow_kg_per_s=_as_step_array(config.dry_air_mass_flow_kg_per_s, n_steps),
            humidity_kg_per_kg=_as_step_array(config.humidity_kg_per_kg, n_steps),
            intake_temperature_celsius=_as_step_array(config.intake_temperature_celsius, n_steps),
            setpoint_temperature_celsius=_as_step_array(
                config.setpoint_temperature_celsius, n_steps
            ),
        )


@dataclass(frozen=True, slots=True)
class HeatNetworkResult:
    """Ergebnis eines Kaskadenlaufs.

    Alle Arrays vektorisiert über die Zeitschrittachse (letzte Achse, Länge ``n_steps``);
    die vorletzte Achse indiziert Stufen/Quellen/Senken in der Reihenfolge der jeweiligen
    ``*_names``-Tupel.

    :ivar stage_source_names: Quellenname je Stufe, Länge ``n_stages``.
    :ivar stage_sink_names: Senkenname je Stufe, Länge ``n_stages``.
    :ivar stage_recovered_power_kw: Genutzte Leistung je Stufe, Form ``(n_stages, n_steps)``.
    :ivar stage_binding_code: Bindende Nebenbedingung je Stufe (siehe :data:`BINDING_LABELS`),
        Form ``(n_stages, n_steps)``.
    :ivar stage_source_outlet_temperature_celsius: Temperatur der Quelle NACH dieser Stufe
        (vor der nächsten Stufe, die dieselbe Quelle referenziert, falls vorhanden), Form
        ``(n_stages, n_steps)`` -- Diagnosegröße, um die Grädigkeits-/Taupunkt-Invarianten
        je Stufe von außen prüfbar zu machen, ohne die Kaskadenlogik im Test zu duplizieren.
    :ivar stage_sink_outlet_temperature_celsius: Temperatur der Senke NACH dieser Stufe,
        analog, Form ``(n_stages, n_steps)``.
    :ivar source_names: Quellennamen, Länge ``n_sources``.
    :ivar source_available_power_kw: Maximal rückgewinnbare Leistung je Quelle (Kühlung bis
        zur Taupunktgrenze, EA1-Formel unverändert), Form ``(n_sources, n_steps)``.
    :ivar source_used_power_kw: Über alle ihre Stufen hinweg tatsächlich genutzte Leistung
        je Quelle, Form ``(n_sources, n_steps)``.
    :ivar source_discarded_power_kw: ``available - used`` je Quelle -- bei korrekter
        Kaskadierung nur noch das, was wegen der Taupunktgrenze nicht mehr entziehbar war,
        nicht mehr die volle Restenthalpie wie in der unkorrigierten Vorlage (Korrektur 2).
    :ivar source_condensate_kg_per_s: Tatsächlich ausgekondensierter Massenstrom je Quelle
        (MA1), Form ``(n_sources, n_steps)``. Null in ``recovery_mode="sensible"`` (Default)
        und immer dann, wenn ``cold_temperature_celsius`` über dem Taupunkt der jeweiligen
        Quelle liegt. Verlässt die Bilanz als Flüssigwasser -- für die Massenbilanz gilt
        ``outlet_humidity = outlet_humidity_kg_per_kg - source_condensate_kg_per_s /
        dry_air_mass_flow_kg_per_s`` (bis zu ``Y_s(T_kalt)``, siehe Moduldocstring).
    :ivar source_outlet_temperature_celsius: Austrittstemperatur der Quelle nach ALLEN ihren
        Stufen, Form ``(n_sources, n_steps)``.
    :ivar sink_names: Senkennamen, Länge ``n_sinks``.
    :ivar sink_outlet_temperature_celsius: ``T_nach_WUET`` je Senke nach allen ihren Stufen,
        Form ``(n_sinks, n_steps)``.
    :ivar fuel_mass_flow_kg_per_s: ``m_gas`` je Senke, Form ``(n_sinks, n_steps)``.
    """

    stage_source_names: tuple[str, ...]
    stage_sink_names: tuple[str, ...]
    stage_recovered_power_kw: FloatArray
    stage_binding_code: IntArray
    stage_source_outlet_temperature_celsius: FloatArray
    stage_sink_outlet_temperature_celsius: FloatArray
    source_names: tuple[str, ...]
    source_available_power_kw: FloatArray
    source_used_power_kw: FloatArray
    source_discarded_power_kw: FloatArray
    source_condensate_kg_per_s: FloatArray
    source_outlet_temperature_celsius: FloatArray
    sink_names: tuple[str, ...]
    sink_outlet_temperature_celsius: FloatArray
    fuel_mass_flow_kg_per_s: FloatArray

    def total_recovered_power_kw(self) -> FloatArray:
        """Summe der genutzten Leistung über alle Stufen, je Zeitschritt."""
        return np.sum(self.stage_recovered_power_kw, axis=0)

    def total_fuel_mass_flow_kg_per_s(self) -> FloatArray:
        """Summe des Brennstoffbedarfs über alle Senken, je Zeitschritt."""
        return np.sum(self.fuel_mass_flow_kg_per_s, axis=0)


def binding_label(code: int) -> str:
    """Menschenlesbares Label für einen Eintrag aus ``stage_binding_code``."""
    return BINDING_LABELS[code]


def _shared_step_count(sources: tuple[SourceState, ...], sinks: tuple[SinkState, ...]) -> int:
    shapes = {state.outlet_temperature_celsius.shape[0] for state in sources}
    shapes |= {state.intake_temperature_celsius.shape[0] for state in sinks}
    if len(shapes) != 1:
        raise ValueError(f"alle Quellen/Senken müssen dieselbe Schrittzahl haben, sind {shapes}")
    return int(next(iter(shapes)))


def _validate_stage_names(
    stages: tuple[tuple[str, str], ...],
    source_by_name: dict[str, SourceState],
    sink_by_name: dict[str, SinkState],
) -> None:
    for source_name, sink_name in stages:
        if source_name not in source_by_name:
            raise ValueError(f"stages referenziert unbekannte Quelle {source_name!r}")
        if sink_name not in sink_by_name:
            raise ValueError(f"stages referenziert unbekannte Senke {sink_name!r}")


def evaluate_cascade(
    config: HeatNetworkConfig,
    stages: tuple[tuple[str, str], ...],
    sources: tuple[SourceState, ...],
    sinks: tuple[SinkState, ...],
) -> HeatNetworkResult:
    r"""Werte die Kaskade entlang ``stages`` aus -- der eigentliche Rechenkern (EA2/EA3).

    ``stages`` ist eine geordnete Liste von ``(Quellenname, Senkenname)``-Paaren, typischerweise
    das Ergebnis von ``routing.HeatNetworkRouting.route`` -- diese Funktion kennt ``routing.py``
    nicht (Dependency Injection: die Verschaltung wird fertig aufgelöst übergeben, nicht hier
    entschieden). Jede Stufe wird in der gegebenen Reihenfolge abgearbeitet; Quellen und Senken
    führen dabei laufenden Zustand (aktuelle Temperatur), der von Stufe zu Stufe weitergereicht
    wird -- genau die Kaskadierung aus Korrektur 2.

    Je Stufe: die Senke wird so weit aufgeheizt, wie es der kleinere von zwei Werten erlaubt --
    ihr eigener Sollwert oder die Mindestgrädigkeit gegenüber der AKTUELLEN (bereits durch
    vorherige Stufen abgekühlten) Quellentemperatur. Die dafür nötige Leistung wird zusätzlich
    durch das gekappt, was die Quelle noch hergeben kann, ohne ihre Taupunktgrenze zu
    unterschreiten (Korrektur 1: die eigene Taupunktgrenze, nicht die einer anderen Quelle).
    Nicht verbrauchte Quellenenthalpie bleibt für die nächste Stufe verfügbar (Korrektur 2),
    solange ``stages`` sie einer weiteren Senke zuweist.

    Massenstrom- und Feuchte-bedingte Wärmekapazitätsraten (:math:`\dot m_L (c_{p,L}+Y c_{p,D})`)
    werden je Quelle/Senke als über die Kaskade KONSTANT behandelt, berechnet aus der
    ursprünglichen ``outlet_humidity_kg_per_kg`` -- reine sensible Wärmeübertragung, keine
    Befeuchtung/Entfeuchtung der SENKE in diesem Modul. In ``recovery_mode="sensible"``
    (Default) gilt das auch für die Quelle unverändert (die Feuchte ändert sich nicht,
    solange die Quelle oberhalb ihrer Taupunktgrenze bleibt; unterhalb wird per Konstruktion
    nicht mehr gekühlt). In ``recovery_mode="condensing"`` (MA1) entzieht die Quelle
    zusätzlich latente Wärme aus Kondensation -- das ändert ihre Feuchte (Kondensat verlässt
    die Bilanz, siehe :attr:`HeatNetworkResult.source_condensate_kg_per_s`), aber NICHT die
    hier verwendete Kapazitätsrate: die sensible Kühlreserve wird weiterhin mit der
    ursprünglichen Feuchte gerechnet, weil Kondensation modellhaft erst beim Erreichen von
    ``cold_temperature_celsius`` einsetzt (Flash-Kondensation am kalten Ende, keine
    graduelle Entfeuchtung über den Kühlverlauf) -- danach ist die Quelle für diese Stufe
    ohnehin ausgeschöpft.

    :raises ValueError: wenn ``stages`` eine Quelle/Senke referenziert, die nicht in
        ``sources``/``sinks`` vorkommt, oder wenn Quellen/Senken unterschiedliche Schrittzahlen
        mitbringen.
    """
    source_by_name = {state.name: state for state in sources}
    sink_by_name = {state.name: state for state in sinks}
    _validate_stage_names(stages, source_by_name, sink_by_name)
    n_steps = _shared_step_count(sources, sinks)

    gas = config.gas
    backend = NumpyBackend()

    initial_source_temperature_celsius = {
        name: state.outlet_temperature_celsius for name, state in source_by_name.items()
    }
    source_capacity_rate_w_per_k = {
        name: state.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + state.outlet_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        for name, state in source_by_name.items()
    }
    is_condensing = config.recovery_mode == "condensing"
    cold_temperature_celsius = backend.asarray(config.cold_temperature_celsius)
    condensate_specific_enthalpy_j_per_kg = latent_heat_j_per_kg(backend, cold_temperature_celsius)

    source_floor_temperature_celsius = {
        name: (
            np.full(n_steps, config.cold_temperature_celsius, dtype=np.float64)
            if is_condensing
            else dew_point_celsius(backend, gas, backend.asarray(state.outlet_humidity_kg_per_kg))
            + config.dew_point_approach_k
        )
        for name, state in source_by_name.items()
    }
    # MA1: latenter Leistungsvorrat je Quelle, einmalig aus der Ausgangsfeuchte berechnet
    # (keine Neuberechnung je Stufe -- Kondensation setzt erst ein, wenn der Strom
    # tatsächlich bis T_kalt abgekühlt wurde, siehe Moduldocstring). Null in
    # ``recovery_mode="sensible"`` und immer dann, wenn T_kalt über dem Taupunkt der Quelle
    # liegt -- in beiden Fällen bitgleich zum Stand vor MA1 (kein latenter Beitrag).
    source_latent_power_w = {
        name: (
            np.maximum(
                state.outlet_humidity_kg_per_kg
                - saturation_humidity_kg_per_kg(backend, gas, cold_temperature_celsius),
                0.0,
            )
            * state.dry_air_mass_flow_kg_per_s
            * condensate_specific_enthalpy_j_per_kg
            if is_condensing
            else np.zeros(n_steps, dtype=np.float64)
        )
        for name, state in source_by_name.items()
    }
    remaining_latent_power_w = {name: array.copy() for name, array in source_latent_power_w.items()}
    current_source_temperature_celsius = {
        name: array.copy() for name, array in initial_source_temperature_celsius.items()
    }
    source_used_power_w = {name: np.zeros(n_steps, dtype=np.float64) for name in source_by_name}
    source_condensate_used_kg_per_s = {
        name: np.zeros(n_steps, dtype=np.float64) for name in source_by_name
    }

    sink_capacity_rate_w_per_k = {
        name: state.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + state.humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        for name, state in sink_by_name.items()
    }
    current_sink_temperature_celsius = {
        name: state.intake_temperature_celsius.copy() for name, state in sink_by_name.items()
    }

    n_stages = len(stages)
    stage_power_kw = np.zeros((n_stages, n_steps), dtype=np.float64)
    stage_binding_code = np.zeros((n_stages, n_steps), dtype=np.int64)
    stage_source_outlet_temperature_celsius = np.zeros((n_stages, n_steps), dtype=np.float64)
    stage_sink_outlet_temperature_celsius = np.zeros((n_stages, n_steps), dtype=np.float64)

    for stage_index, (source_name, sink_name) in enumerate(stages):
        capacity_source = source_capacity_rate_w_per_k[source_name]
        capacity_sink = sink_capacity_rate_w_per_k[sink_name]
        temperature_source = current_source_temperature_celsius[source_name]
        temperature_sink = current_sink_temperature_celsius[sink_name]
        floor_temperature = source_floor_temperature_celsius[source_name]
        setpoint_temperature = sink_by_name[sink_name].setpoint_temperature_celsius

        pinch_cap_temperature = temperature_source - config.minimum_pinch_k
        target_sink_temperature = np.minimum(setpoint_temperature, pinch_cap_temperature)
        target_sink_temperature = np.maximum(target_sink_temperature, temperature_sink)

        power_for_target_w = capacity_sink * (target_sink_temperature - temperature_sink)
        sensible_available_w = np.maximum(
            capacity_source * (temperature_source - floor_temperature), 0.0
        )
        # MA1: verbleibender latenter Vorrat kommt oben auf die sensible Kühlreserve --
        # bei ``recovery_mode="sensible"`` ist er 0 und ``power_available_w`` bitgleich zu
        # ``sensible_available_w`` (IEEE-754: ``x + 0.0 == x``).
        power_available_w = sensible_available_w + remaining_latent_power_w[source_name]
        power_actual_w = np.clip(np.minimum(power_for_target_w, power_available_w), 0.0, None)
        # Sensibler Anteil zuerst (senkt die Quellentemperatur), latenter Anteil erst danach
        # (Kondensation setzt erst bei Erreichen von ``floor_temperature`` ein) -- ``np.minimum``
        # gibt hier ``power_actual_w`` unverändert zurück, solange kein latenter Vorrat
        # vorhanden ist, also ohne zusätzliche Rundung gegenüber dem Stand vor MA1.
        power_sensible_used_w = np.minimum(power_actual_w, sensible_available_w)
        power_latent_used_w = power_actual_w - power_sensible_used_w

        dew_point_binds = power_actual_w < power_for_target_w - _POWER_TOLERANCE_W
        pinch_binds = (~dew_point_binds) & (
            pinch_cap_temperature < setpoint_temperature - _TEMPERATURE_TOLERANCE_K
        )
        binding_code = np.full(n_steps, BINDING_SINK_SATISFIED, dtype=np.int64)
        binding_code = np.where(pinch_binds, BINDING_PINCH, binding_code)
        binding_code = np.where(dew_point_binds, BINDING_DEW_POINT, binding_code)

        stage_power_kw[stage_index] = power_actual_w / 1000.0
        stage_binding_code[stage_index] = binding_code

        current_source_temperature_celsius[source_name] = (
            temperature_source - power_sensible_used_w / capacity_source
        )
        remaining_latent_power_w[source_name] = (
            remaining_latent_power_w[source_name] - power_latent_used_w
        )
        current_sink_temperature_celsius[sink_name] = (
            temperature_sink + power_actual_w / capacity_sink
        )
        source_used_power_w[source_name] = source_used_power_w[source_name] + power_actual_w
        source_condensate_used_kg_per_s[source_name] = source_condensate_used_kg_per_s[
            source_name
        ] + (power_latent_used_w / condensate_specific_enthalpy_j_per_kg if is_condensing else 0.0)
        stage_source_outlet_temperature_celsius[stage_index] = current_source_temperature_celsius[
            source_name
        ]
        stage_sink_outlet_temperature_celsius[stage_index] = current_sink_temperature_celsius[
            sink_name
        ]

    source_names = tuple(source_by_name)
    source_available_power_kw = np.stack(
        [
            (
                np.maximum(
                    source_capacity_rate_w_per_k[name]
                    * (
                        initial_source_temperature_celsius[name]
                        - source_floor_temperature_celsius[name]
                    ),
                    0.0,
                )
                + source_latent_power_w[name]
            )
            / 1000.0
            for name in source_names
        ]
    )
    source_used_power_kw = np.stack([source_used_power_w[name] / 1000.0 for name in source_names])
    source_discarded_power_kw = source_available_power_kw - source_used_power_kw
    source_condensate_kg_per_s = np.stack(
        [source_condensate_used_kg_per_s[name] for name in source_names]
    )
    source_outlet_temperature_celsius = np.stack(
        [current_source_temperature_celsius[name] for name in source_names]
    )

    sink_names = tuple(sink_by_name)
    sink_outlet_temperature_celsius = np.stack(
        [current_sink_temperature_celsius[name] for name in sink_names]
    )
    fuel_mass_flow_kg_per_s = np.stack(
        [
            np.maximum(
                sink_capacity_rate_w_per_k[name]
                * (
                    sink_by_name[name].setpoint_temperature_celsius
                    - current_sink_temperature_celsius[name]
                ),
                0.0,
            )
            / (config.fuel_lower_heating_value_j_per_kg * config.heater_efficiency)
            for name in sink_names
        ]
    )

    return HeatNetworkResult(
        stage_source_names=tuple(source_name for source_name, _ in stages),
        stage_sink_names=tuple(sink_name for _, sink_name in stages),
        stage_recovered_power_kw=stage_power_kw,
        stage_binding_code=stage_binding_code,
        stage_source_outlet_temperature_celsius=stage_source_outlet_temperature_celsius,
        stage_sink_outlet_temperature_celsius=stage_sink_outlet_temperature_celsius,
        source_names=source_names,
        source_available_power_kw=source_available_power_kw,
        source_used_power_kw=source_used_power_kw,
        source_discarded_power_kw=source_discarded_power_kw,
        source_condensate_kg_per_s=source_condensate_kg_per_s,
        source_outlet_temperature_celsius=source_outlet_temperature_celsius,
        sink_names=sink_names,
        sink_outlet_temperature_celsius=sink_outlet_temperature_celsius,
        fuel_mass_flow_kg_per_s=fuel_mass_flow_kg_per_s,
    )
