r"""Konfigurationsobjekte des Wärmenetz-Moduls (Kaskade Abluft -> Prozessluft).

Siehe ``docs/06_waermenetz.md`` für Zweck, Abgrenzung und die drei Korrekturen
gegenüber der Excel-Vorlage des Verbundpartners; dieses Docstring beschränkt sich auf
die Konfigurationsobjekte selbst.

Stoffwerte kommen ausschließlich aus :class:`~twinlink_ml.simulation.config.GasConfig`
(``cp_dry_air_j_per_kg_k``, ``cp_vapour_j_per_kg_k``) -- keine zweite, hier neu gesetzte
Kopie derselben Größen. ``MaterialConfig`` ist hier nicht relevant: das Wärmenetz
rechnet mit Luftströmen (Abluft, Prozessluft), nicht mit dem Feststoffsystem der
Wirbelschicht.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

from ..simulation.config import (
    GasConfig,
    PlaceholderConfigWarning,
    Provenance,
    config_fingerprint,
    section_fingerprints_of,
    without_documentary_provenance,
)

__all__ = [
    "DEFAULT_APPROACH_CHECK",
    "DEFAULT_APPROACH_PROFILE_NODES",
    "DEFAULT_COLD_TEMPERATURE_CELSIUS",
    "DEFAULT_DEW_POINT_APPROACH_K",
    "DEFAULT_FUEL_LOWER_HEATING_VALUE_J_PER_KG",
    "DEFAULT_HEATER_EFFICIENCY",
    "DEFAULT_MINIMUM_PINCH_K",
    "DEFAULT_RECOVERY_MODE",
    "ApproachCheck",
    "HeatNetworkConfig",
    "HeatSinkConfig",
    "HeatSourceConfig",
    "RecoveryMode",
]

ApproachCheck = Literal["along_exchanger", "warm_end"]
"""Prüfung der Mindestgrädigkeit je Stufe (Befund 78/79, Prüfbefund A1).

``"along_exchanger"`` (Default seit Befund 79): ``dT_min`` als Minimum über den ganzen
Gegenstrom-Übertrager, auf dem Quellprofil mit Kondensation ab dem Taupunkt
(:mod:`~twinlink_ml.heatnetwork.approach`). ``"warm_end"``: nur am warmen Ende, Kondensation als
Flash bei ``T_kalt`` -- der Stand vor Befund 79, NUR zur Reproduktion alter Artefakte.
"""

DEFAULT_APPROACH_CHECK: ApproachCheck = "along_exchanger"
"""Default der Grädigkeitsprüfung (Befund 79). Alte Artefakte: ``approach_check="warm_end"``."""

DEFAULT_APPROACH_PROFILE_NODES = 400
"""Stützstellen des Quellprofils zwischen Taupunkt und Kühlgrenze (plus ``T_out`` und der
Taupunkt als exakter Knoten); Diskretisierungsfehler der Stufenleistung siehe
``tests/test_heatnetwork_approach.py``."""

RecoveryMode = Literal["sensible", "condensing"]
"""Rückgewinnungsmodus je :class:`HeatNetworkConfig` (MA1).

``"sensible"`` (Default, bisheriges Verhalten): Kühlung bis zur Taupunktgrenze
``T_dp(Y_out) + dew_point_approach_k`` -- keine Kondensation im Übertrager, bitgleich zum
Stand vor MA1. ``"condensing"``: Kühlung bis ``cold_temperature_celsius`` (``T_kalt``),
mit zusätzlicher latenter Wärme aus dem dabei anfallenden Kondensat. Bewusst ein
Config-Feld, kein Funktionsparameter von :func:`~.balance.evaluate_cascade` -- sonst wäre
der gewählte Modus im ``RunRecord`` nicht dokumentiert (siehe ``fingerprint``)."""

DEFAULT_DEW_POINT_APPROACH_K = 3.0
"""Sicherheitsmarge zum Taupunkt gegen Kondensation im Wärmeübertrager (EA1).

Zahlenwert identisch zu ``evaluation.recoverable_power.DEW_POINT_MARGIN_K`` (Befund 50)
-- hier als eigener Modul-Default dupliziert statt von dort importiert, weil
``evaluation`` auf ``simulation`` aufbaut und nicht umgekehrt; ``heatnetwork`` importiert
bewusst nichts aus ``evaluation``, um diese Abhängigkeitsrichtung nicht für einen
einzelnen Bezeichner umzukehren. Beide Stellen leiten den Wert aus derselben
Modellannahme her (3 K Marge zur Kondensationsgrenze) und sollten bei einer Änderung
gemeinsam angepasst werden.
"""

DEFAULT_MINIMUM_PINCH_K = 20.0
"""Mindestgrädigkeit ``dT_min`` je Kaskadenstufe (EA2, Korrektur 3) -- PLATZHALTER.

Begründung des Zahlenwerts (nicht "konservatives Ende eines Richtwertbereichs",
sondern Kettenstruktur): der Verbundpartner modelliert den Zwischenkreis zwischen
Abluft und Prozessluft als Black Box -- tatsächlich liegen darin vermutlich ZWEI
Übertragerstufen in Reihe (Abluft -> Wasser, Wasser -> Prozessluft). Die effektive
Grädigkeit über die Kette ist näherungsweise die Summe der Einzelstufen-Grädigkeiten.
10 K ist ein für Gas/Gas- bzw. Gas/Luft-Übertrager üblicher Richtwert für EINE Stufe;
20 K setzt zwei Stufen in Reihe an. Zusätzlich gilt: eine zu klein angesetzte
Grädigkeit ÜBERSCHÄTZT die rückgewinnbare Leistung -- der konservative Wert ist in
dieser Richtung der ehrlichere Fehler.

STATUS: Platzhalter, ohne Quelle beim Verbundpartner (siehe :meth:`HeatNetworkConfig.
warn_if_placeholder`, derselbe Mechanismus wie ``ProcessConfig``). Offene Frage: besteht
der Zwischenkreis tatsächlich aus zwei Stufen, oder wird direkt Abluft gegen Prozessluft
übertragen? Bei Direktübertragung wären 10-15 K angemessener. Eine
Sensitivitätsrechnung über 5/10/15/20/25 K ist als nächster Schritt vorgesehen (nicht
Teil dieses Auftrags) -- siehe ``docs/06_waermenetz.md``.
"""

DEFAULT_FUEL_LOWER_HEATING_VALUE_J_PER_KG = 50.0e6
"""Unterer Heizwert ``H_u`` des Zusatzbrennstoffs -- PLATZHALTER (Richtwert Erdgas H,
ca. 50 MJ/kg), keine Quelle beim Verbundpartner erfragt."""

DEFAULT_HEATER_EFFICIENCY = 0.90
"""Wirkungsgrad ``eta_H`` des Lufterhitzers -- PLATZHALTER, keine Quelle erfragt."""

DEFAULT_RECOVERY_MODE: RecoveryMode = "sensible"
"""Default-Rückgewinnungsmodus -- bewusst unverändert gegenüber dem Stand vor MA1, damit
bestehende Ergebnisse (Befund 52 und früher) bitgleich reproduzierbar bleiben."""

DEFAULT_COLD_TEMPERATURE_CELSIUS = 15.0
"""Kalttemperatur ``T_kalt`` des kondensierenden Übertragers -- PLATZHALTER, keine Quelle
beim Verbundpartner erfragt (derselbe Status wie :data:`DEFAULT_MINIMUM_PINCH_K`).

Begründung des Zahlenwerts: 15 °C liegt im plausiblen Bereich eines Kaltwasserkreises
(Rückkühlwerk oder leicht gekühltes Kühlwasser, kein Kältemaschinen-Kaltwassernetz mit
6-12 °C Vorlauf) und deckt sich mit dem eigenen Überschlag am Betriebspunkt des
Verbundpartners (8 kg/s, Y=0.01998, T_out=81.4 °C), der diesen Wert als Ankerpunkt
verwendet. Weder das eine noch das andere ist eine Messung oder eine Angabe des
Verbundpartners -- offene Frage wie bei ``minimum_pinch_k``: welches Temperaturniveau der
tatsächliche Kaltwasserkreis im Partnermodell (MA5, sechs Verschaltungsfälle) tatsächlich
fährt, ist unbekannt. Geht in den Fingerprint ein und löst wie jeder andere Platzhalter im
Projekt :meth:`HeatNetworkConfig.warn_if_placeholder` aus.
"""


@dataclass(frozen=True, slots=True)
class HeatSourceConfig:
    r"""Eine Abluftquelle (Trocknerlinie) mit Massenstrom, Austrittstemperatur, -feuchte.

    Felder entweder direkt gesetzt (Verbundmodell, ca. 8 kg/s) oder aus einem simulierten
    Abluftzustand übernommen (Laborsimulator, ca. 0,03 kg/s) -- siehe ``balance.py``
    Moduldocstring, Abschnitt "Anbindung an den Simulator" (EA4), für die wörtliche
    Signatur von :func:`~twinlink_ml.simulation.exhaust.reconstruct_exhaust_state` und den
    Skalenvorbehalt. Diese Klasse kennt den Simulator nicht (Dependency Injection bleibt
    beim Aufrufer, wie überall im Projekt).

    :param name: Eindeutiger Bezeichner -- Referenzpunkt für ``routing.py``.
    :param dry_air_mass_flow_kg_per_s: Trockenluftmassenstrom der Abluft. Aus
        ``dry_air_mass_flow_kg_per_s`` in ``diagnostics``/``SteadyStateSolution``
        übernehmen, nicht aus der Gasgeschwindigkeit neu rechnen (EA4).
    :param outlet_temperature_celsius: Ablufttemperatur -- der AUSLASS-Kanal
        (``outlet_temperature_celsius`` aus ``reconstruct_exhaust_state``), nicht die
        Betttemperatur (``bed_temperature_celsius``/``T_gas_out_celsius``). Bei
        ``ExhaustConfig.outlet_duct_heat_loss_k == 0.0`` (Default) numerisch identisch,
        der Vertrag an der Verbundschnittstelle ist aber der Auslass-Kanal (EA4).
    :param outlet_humidity_kg_per_kg: Absolute Abluftfeuchte ``Y_out``.
    """

    name: str
    dry_air_mass_flow_kg_per_s: float
    outlet_temperature_celsius: float
    outlet_humidity_kg_per_kg: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name darf nicht leer sein")
        if self.dry_air_mass_flow_kg_per_s <= 0.0:
            raise ValueError(
                "dry_air_mass_flow_kg_per_s muss positiv sein, ist "
                f"{self.dry_air_mass_flow_kg_per_s}"
            )
        if self.outlet_humidity_kg_per_kg < 0.0:
            raise ValueError(
                f"outlet_humidity_kg_per_kg darf nicht negativ sein, ist "
                f"{self.outlet_humidity_kg_per_kg}"
            )


@dataclass(frozen=True, slots=True)
class HeatSinkConfig:
    r"""Eine Senke (Lufterhitzer einer Trocknerlinie).

    Muss auf ``setpoint_temperature_celsius`` gebracht werden.

    :param name: Eindeutiger Bezeichner -- Referenzpunkt für ``routing.py``.
    :param dry_air_mass_flow_kg_per_s: Trockenluftmassenstrom der Prozessluft dieser Senke.
    :param humidity_kg_per_kg: Absolute Feuchte der Prozessluft (als über die Kaskade
        konstant angenommen -- reine sensible Erwärmung, keine Befeuchtung in diesem
        Modul).
    :param intake_temperature_celsius: Temperatur der Ansaugluft VOR jeder
        Rückgewinnung und vor dem Erhitzer. Bewusst nicht ``inlet_temperature_celsius``
        genannt: dieser Bezeichner ist im Prozessmodul bereits als Stellgröße belegt
        (``ActionSpaceConfig.inlet_temperature_celsius`` -- die Zulufttemperatur NACH dem
        Erhitzer, also ein Sollwert). Hier ist die Temperatur VOR dem Erhitzer gemeint;
        zwei gleichnamige Felder mit entgegengesetzter Bedeutung an derselben Anlage
        wären eine unnötige Verwechslungsgefahr. Bewusst ein Feld je Senke, nicht eine
        netzweite Umgebungsgröße auf :class:`HeatNetworkConfig`: die Ansaugbedingungen
        (Außenluft/Umluft/Vorkonditionierung) sind ortsabhängig, zwei Senken im selben
        Netz können unterschiedliche Werte haben -- eine zweite Quelle für dieselbe
        physikalische Größe (netzweit UND je Senke) wäre der Widerspruch, aus dem später
        ein stiller Fehler entsteht.
    :param setpoint_temperature_celsius: Sollwert nach dem Erhitzer (``T_soll``).
    """

    name: str
    dry_air_mass_flow_kg_per_s: float
    humidity_kg_per_kg: float
    intake_temperature_celsius: float
    setpoint_temperature_celsius: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name darf nicht leer sein")
        if self.dry_air_mass_flow_kg_per_s <= 0.0:
            raise ValueError(
                "dry_air_mass_flow_kg_per_s muss positiv sein, ist "
                f"{self.dry_air_mass_flow_kg_per_s}"
            )
        if self.humidity_kg_per_kg < 0.0:
            raise ValueError(
                f"humidity_kg_per_kg darf nicht negativ sein, ist {self.humidity_kg_per_kg}"
            )
        if self.setpoint_temperature_celsius <= self.intake_temperature_celsius:
            raise ValueError(
                "setpoint_temperature_celsius muss über intake_temperature_celsius liegen "
                f"({self.setpoint_temperature_celsius} <= {self.intake_temperature_celsius})"
            )


@dataclass(frozen=True, slots=True)
class HeatNetworkConfig:
    r"""Vollständige Konfiguration des Wärmenetz-Moduls: Quellen, Senken, Kaskadenparameter.

    Enthält bewusst KEINE Verschaltungsregel (welche Quelle speist welche Senke) --
    das ist Verhalten, kein Konfigurationswert, und lebt als injizierbare Komponente in
    ``routing.py`` (:class:`~twinlink_ml.heatnetwork.routing.HeatNetworkRouting`,
    Dependency Injection wie ``SorptionIsotherm`` in ``simulation.sorption``). Was hier
    lebt, ist ``cascade_order``: die physikalische Regel "Niedertemperaturquellen zuerst"
    (EA2, Korrektur 3) ist keine austauschbare Strategie, sondern eine Modellinvariante,
    und gehört deshalb als Konfigurationswert (mit sinnvollem Default) hierher, nicht in
    eine Strategieklasse.

    Fingerprint/``section_fingerprints`` über :func:`~twinlink_ml.simulation.config.
    config_fingerprint`/:func:`~twinlink_ml.simulation.config.section_fingerprints_of` --
    dieselbe Mechanik wie :class:`~twinlink_ml.simulation.config.ProcessConfig`,
    wiederverwendet statt neu gebaut.

    :param provenance: Herkunft der Zahlenwerte, wie ``ProcessConfig.provenance``.
    :param gas: Stoffwerte der Luft (``cp_dry_air_j_per_kg_k``, ``cp_vapour_j_per_kg_k``)
        -- aus dem vorhandenen :class:`~twinlink_ml.simulation.config.GasConfig`, nicht
        neu gesetzt (EA1).
    :param sources: Abluftquellen, siehe :class:`HeatSourceConfig`.
    :param sinks: Senken, siehe :class:`HeatSinkConfig`.
    :param minimum_pinch_k: ``dT_min``, siehe :data:`DEFAULT_MINIMUM_PINCH_K`.
    :param dew_point_approach_k: ``dT_approach``, siehe :data:`DEFAULT_DEW_POINT_APPROACH_K`.
    :param fuel_lower_heating_value_j_per_kg: ``H_u`` des Zusatzbrennstoffs.
    :param heater_efficiency: ``eta_H`` des Lufterhitzers, in (0, 1].
    :param recovery_mode: ``"sensible"`` (Default) oder ``"condensing"``, siehe
        :data:`RecoveryMode`.
    :param cold_temperature_celsius: ``T_kalt``, siehe
        :data:`DEFAULT_COLD_TEMPERATURE_CELSIUS`. Nur in ``recovery_mode="condensing"``
        wirksam; im Default-Modus mitgeführt (geht trotzdem in den Fingerprint ein), aber
        ohne Effekt auf das Ergebnis.
    :param approach_check: Prüfung der Mindestgrädigkeit, siehe :data:`ApproachCheck`.
    :param approach_profile_nodes: Stützstellen des Quellprofils, siehe
        :data:`DEFAULT_APPROACH_PROFILE_NODES`.
    :param cascade_order: Explizite Bearbeitungsreihenfolge der Quellennamen
        (Niedertemperatur zuerst). Leeres Tupel (Default) bedeutet: automatisch
        aufsteigend nach ``outlet_temperature_celsius`` sortieren (siehe
        ``routing.resolve_cascade_order``). Eine explizite Angabe muss eine Permutation
        aller Quellennamen sein -- geprüft in ``__post_init__``, nicht erst beim
        Kaskadenlauf, damit ein Tippfehler im Quellennamen sofort auffällt.
    """

    provenance: Provenance = field(default_factory=Provenance)
    gas: GasConfig = field(default_factory=GasConfig)
    sources: tuple[HeatSourceConfig, ...] = ()
    sinks: tuple[HeatSinkConfig, ...] = ()
    minimum_pinch_k: float = DEFAULT_MINIMUM_PINCH_K
    dew_point_approach_k: float = DEFAULT_DEW_POINT_APPROACH_K
    fuel_lower_heating_value_j_per_kg: float = DEFAULT_FUEL_LOWER_HEATING_VALUE_J_PER_KG
    heater_efficiency: float = DEFAULT_HEATER_EFFICIENCY
    recovery_mode: RecoveryMode = DEFAULT_RECOVERY_MODE
    cold_temperature_celsius: float = DEFAULT_COLD_TEMPERATURE_CELSIUS
    cascade_order: tuple[str, ...] = ()
    approach_check: ApproachCheck = DEFAULT_APPROACH_CHECK
    approach_profile_nodes: int = DEFAULT_APPROACH_PROFILE_NODES

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("mindestens eine Quelle (sources) nötig")
        if not self.sinks:
            raise ValueError("mindestens eine Senke (sinks) nötig")
        source_names = [source.name for source in self.sources]
        if len(set(source_names)) != len(source_names):
            raise ValueError(f"Quellennamen müssen eindeutig sein, sind {source_names}")
        sink_names = [sink.name for sink in self.sinks]
        if len(set(sink_names)) != len(sink_names):
            raise ValueError(f"Senkennamen müssen eindeutig sein, sind {sink_names}")
        if self.minimum_pinch_k <= 0.0:
            raise ValueError(f"minimum_pinch_k muss positiv sein, ist {self.minimum_pinch_k}")
        if self.dew_point_approach_k <= 0.0:
            raise ValueError(
                f"dew_point_approach_k muss positiv sein, ist {self.dew_point_approach_k}"
            )
        if self.fuel_lower_heating_value_j_per_kg <= 0.0:
            raise ValueError(
                "fuel_lower_heating_value_j_per_kg muss positiv sein, ist "
                f"{self.fuel_lower_heating_value_j_per_kg}"
            )
        if not 0.0 < self.heater_efficiency <= 1.0:
            raise ValueError(
                f"heater_efficiency muss in (0, 1] liegen, ist {self.heater_efficiency}"
            )
        if self.recovery_mode not in ("sensible", "condensing"):
            raise ValueError(
                f"recovery_mode muss 'sensible' oder 'condensing' sein, ist {self.recovery_mode!r}"
            )
        if not -40.0 <= self.cold_temperature_celsius <= 100.0:
            raise ValueError(
                "cold_temperature_celsius muss in [-40, 100] °C liegen (Gültigkeitsbereich "
                f"von thermo.saturation_pressure_pa), ist {self.cold_temperature_celsius}"
            )
        if self.approach_check not in ("along_exchanger", "warm_end"):
            raise ValueError(
                "approach_check muss 'along_exchanger' oder 'warm_end' sein, ist "
                f"{self.approach_check!r}"
            )
        if self.approach_profile_nodes < 1:
            raise ValueError(
                f"approach_profile_nodes muss >= 1 sein, ist {self.approach_profile_nodes}"
            )
        if self.cascade_order and set(self.cascade_order) != set(source_names):
            raise ValueError(
                f"cascade_order {self.cascade_order} ist keine Permutation der "
                f"Quellennamen {tuple(source_names)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Stabiler Hash der aufgelösten Config -- siehe ``ProcessConfig.fingerprint``."""
        return config_fingerprint(without_documentary_provenance(self.to_dict()))

    @property
    def section_fingerprints(self) -> dict[str, str]:
        """Teil-Fingerprints je Konfigurationsfeld.

        Siehe ``ProcessConfig.section_fingerprints``.
        """
        sections = tuple(f.name for f in fields(self))
        return section_fingerprints_of(without_documentary_provenance(self.to_dict()), sections)

    def warn_if_placeholder(self) -> None:
        """Wie ``ProcessConfig.warn_if_placeholder``, hier auf ``HeatNetworkConfig`` angewendet.

        ``minimum_pinch_k`` (siehe :data:`DEFAULT_MINIMUM_PINCH_K`) ist der Platzhalter, der
        diese Warnung im Default-Fall auslöst.
        """
        if self.provenance.status == "placeholder":
            warnings.warn(
                "HeatNetworkConfig enthält Platzhalterwerte (u.a. minimum_pinch_k) "
                f"(offene TODOs: {len(self.provenance.open_todos)}). "
                "Absolutwerte sind nicht belastbar; nur relative Vergleiche.",
                PlaceholderConfigWarning,
                stacklevel=2,
            )
