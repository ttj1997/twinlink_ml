r"""Configuration objects of the heat-network module (cascade exhaust air -> process air).

See ``docs/06_waermenetz.md`` for purpose, scope and the three corrections
to the Excel template of the consortium partner; this docstring is limited to
the configuration objects themselves.

Material properties come exclusively from :class:`~twinlink_ml.simulation.config.GasConfig`
(``cp_dry_air_j_per_kg_k``, ``cp_vapour_j_per_kg_k``) -- no second copy of the same
quantities set anew here. ``MaterialConfig`` is not relevant here: the heat network
computes with air streams (exhaust air, process air), not with the solids system of the
fluidized bed.
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
    r"""An exhaust air source (dryer line) with mass flow, outlet temperature, outlet humidity.

    Fields either set directly (site-network model, approx. 8 kg/s) or taken from a simulated
    exhaust air state (lab simulator, approx. 0.03 kg/s) -- see the ``balance.py``
    module docstring, section "Coupling to the simulator" (EA4), for the literal
    signature of :func:`~twinlink_ml.simulation.exhaust.reconstruct_exhaust_state` and the
    scale caveat. This class does not know the simulator (dependency injection stays
    with the caller, as everywhere in the project).

    :param name: Unique identifier -- reference point for ``routing.py``.
    :param dry_air_mass_flow_kg_per_s: Dry-air mass flow of the exhaust air. Take it from
        ``dry_air_mass_flow_kg_per_s`` in ``diagnostics``/``SteadyStateSolution``,
        do not recompute it from the gas velocity (EA4).
    :param outlet_temperature_celsius: Exhaust air temperature -- the OUTLET duct
        (``outlet_temperature_celsius`` from ``reconstruct_exhaust_state``), not the
        bed temperature (``bed_temperature_celsius``/``T_gas_out_celsius``). With
        ``ExhaustConfig.outlet_duct_heat_loss_k == 0.0`` (default) numerically identical,
        but the contract at the site-network interface is the outlet duct (EA4).
    :param outlet_humidity_kg_per_kg: Absolute exhaust air humidity ``Y_out``.
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
    r"""A sink (air heater of a dryer line).

    Must be brought to ``setpoint_temperature_celsius``.

    :param name: Unique identifier -- reference point for ``routing.py``.
    :param dry_air_mass_flow_kg_per_s: Dry-air mass flow of the process air of this sink.
    :param humidity_kg_per_kg: Absolute humidity of the process air (assumed constant
        over the cascade -- pure sensible heating, no humidification in this
        module).
    :param intake_temperature_celsius: Temperature of the intake air BEFORE any
        recovery and before the heater. Deliberately not called ``inlet_temperature_celsius``:
        this identifier is already taken in the process module as an action
        (``ActionSpaceConfig.inlet_temperature_celsius`` -- the inlet air temperature AFTER the
        heater, i.e. a set point). Here the temperature BEFORE the heater is meant;
        two fields of the same name with opposite meaning on the same plant
        would be an unnecessary risk of confusion. Deliberately one field per sink, not a
        network-wide ambient quantity on :class:`HeatNetworkConfig`: the intake conditions
        (outside air/recirculated air/preconditioning) depend on the location, two sinks in the
        same network can have different values -- a second source for the same
        physical quantity (network-wide AND per sink) would be the contradiction from which
        a silent error later arises.
    :param setpoint_temperature_celsius: Set point after the heater (``T_soll``).
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
    r"""Complete configuration of the heat-network module: sources, sinks, cascade parameters.

    Deliberately contains NO interconnection rule (which source feeds which sink) --
    that is behaviour, not a configuration value, and lives as an injectable component in
    ``routing.py`` (:class:`~twinlink_ml.heatnetwork.routing.HeatNetworkRouting`,
    dependency injection like ``SorptionIsotherm`` in ``simulation.sorption``). What lives
    here is ``cascade_order``: the physical rule "low-temperature sources first"
    (EA2, correction 3) is not an exchangeable strategy but a model invariant,
    and therefore belongs here as a configuration value (with a sensible default), not in
    a strategy class.

    Fingerprint/``section_fingerprints`` via :func:`~twinlink_ml.simulation.config.
    config_fingerprint`/:func:`~twinlink_ml.simulation.config.section_fingerprints_of` --
    the same mechanism as :class:`~twinlink_ml.simulation.config.ProcessConfig`,
    reused instead of rebuilt.

    :param provenance: Origin of the numerical values, like ``ProcessConfig.provenance``.
    :param gas: Material properties of the air (``cp_dry_air_j_per_kg_k``, ``cp_vapour_j_per_kg_k``)
        -- from the existing :class:`~twinlink_ml.simulation.config.GasConfig`, not
        set anew (EA1).
    :param sources: Exhaust air sources, see :class:`HeatSourceConfig`.
    :param sinks: Sinks, see :class:`HeatSinkConfig`.
    :param minimum_pinch_k: ``dT_min``, see :data:`DEFAULT_MINIMUM_PINCH_K`.
    :param dew_point_approach_k: ``dT_approach``, see :data:`DEFAULT_DEW_POINT_APPROACH_K`.
    :param fuel_lower_heating_value_j_per_kg: ``H_u`` of the supplementary fuel.
    :param heater_efficiency: ``eta_H`` of the air heater, in (0, 1].
    :param recovery_mode: ``"sensible"`` (default) or ``"condensing"``, see
        :data:`RecoveryMode`.
    :param cold_temperature_celsius: ``T_kalt``, see
        :data:`DEFAULT_COLD_TEMPERATURE_CELSIUS`. Effective only in
        ``recovery_mode="condensing"``; carried along in the default mode (still enters the
        fingerprint), but without effect on the result.
    :param approach_check: Check of the minimum approach temperature, see :data:`ApproachCheck`.
    :param approach_profile_nodes: Nodes of the source profile, see
        :data:`DEFAULT_APPROACH_PROFILE_NODES`.
    :param cascade_order: Explicit processing order of the source names
        (low temperature first). An empty tuple (default) means: sort automatically
        ascending by ``outlet_temperature_celsius`` (see
        ``routing.resolve_cascade_order``). An explicit value must be a permutation
        of all source names -- checked in ``__post_init__``, not only during the
        cascade run, so that a typo in a source name is noticed immediately.
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
        """Stable hash of the resolved config -- see ``ProcessConfig.fingerprint``."""
        return config_fingerprint(without_documentary_provenance(self.to_dict()))

    @property
    def section_fingerprints(self) -> dict[str, str]:
        """Partial fingerprints per configuration field.

        See ``ProcessConfig.section_fingerprints``.
        """
        sections = tuple(f.name for f in fields(self))
        return section_fingerprints_of(without_documentary_provenance(self.to_dict()), sections)

    def warn_if_placeholder(self) -> None:
        """Like ``ProcessConfig.warn_if_placeholder``, here applied to ``HeatNetworkConfig``.

        ``minimum_pinch_k`` (see :data:`DEFAULT_MINIMUM_PINCH_K`) is the placeholder that
        triggers this warning in the default case.
        """
        if self.provenance.status == "placeholder":
            warnings.warn(
                "HeatNetworkConfig enthält Platzhalterwerte (u.a. minimum_pinch_k) "
                f"(offene TODOs: {len(self.provenance.open_todos)}). "
                "Absolutwerte sind nicht belastbar; nur relative Vergleiche.",
                PlaceholderConfigWarning,
                stacklevel=2,
            )
