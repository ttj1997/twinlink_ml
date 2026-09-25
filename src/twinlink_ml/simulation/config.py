"""Konfigurationsobjekte des Grey-Box-Prozessmodells.

Alle Zahlenwerte des Modells leben hier bzw. in ``conf/process/*.yaml`` -- keiner im Code.
Grund: der Austausch der Platzhalter gegen echte Werte von TUHH-A / Anwendungspartner soll ein
Diff in einer YAML-Datei sein, kein Eingriff in die Physik.

Einheiten stehen im Feldnamen (Konvention aus ``03_code_konventionen.md``).
Intern rechnet das Modell in SI; die Umrechnung ingenieursüblicher Einheiten
(kg/h, bar, °C) passiert an exakt einer Stelle: :meth:`ProcessConfig.to_si`-nahe
Helfer in ``greybox.py``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

T = TypeVar("T")

ConfigStatus = Literal["placeholder", "partially_calibrated", "calibrated"]
OperationMode = Literal["continuous", "batch"]
EnergyTarget = Literal["power_kw", "specific_kwh_per_kg"]
Integrator = Literal["rk4", "euler"]
SorptionIsothermKind = Literal["constant", "gab"]


class PlaceholderConfigWarning(UserWarning):
    """Wird beim Laden einer Config mit Platzhalterwerten ausgelöst.

    ``02_prozess_setup.md`` hält fest, dass eine Datei mit veralteten Zahlen schlechter
    ist als keine, weil sie stillschweigend falsche Annahmen ins Modell trägt. Diese
    Warnung macht aus der Konvention eine Zusicherung des Codes: ein Lauf mit
    Platzhaltern ist nachträglich im Log identifizierbar.
    """


@dataclass(frozen=True, slots=True)
class Provenance:
    """Herkunft der Zahlenwerte -- wandert in jeden Run-Log."""

    status: ConfigStatus = "placeholder"
    description: str = ""
    open_todos: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """Apparategeometrie. Quelle: ``02_prozess_setup.md`` -> Abschnitt Anlagengeometrie."""

    cross_section_area_m2: float = 0.0177
    bed_mass_solid_setpoint_kg: float = 2.0
    heat_loss_ua_w_per_k: float = 2.0

    def __post_init__(self) -> None:
        _require_positive(self, "cross_section_area_m2", "bed_mass_solid_setpoint_kg")
        _require_non_negative(self, "heat_loss_ua_w_per_k")


@dataclass(frozen=True, slots=True)
class GasConfig:
    """Eigenschaften der Fluidisierungsluft (feuchte Luft, ideal)."""

    pressure_pa: float = 101_325.0
    r_specific_dry_air_j_per_kg_k: float = 287.06
    cp_dry_air_j_per_kg_k: float = 1006.0
    cp_vapour_j_per_kg_k: float = 1860.0
    viscosity_ref_pa_s: float = 1.716e-5
    viscosity_ref_temperature_k: float = 273.15
    sutherland_constant_k: float = 110.4

    def __post_init__(self) -> None:
        _require_positive(self, "pressure_pa", "cp_dry_air_j_per_kg_k", "viscosity_ref_pa_s")


@dataclass(frozen=True, slots=True)
class MaterialConfig:
    """Stoffsystem. ``02_prozess_setup.md``: Stoffsystem ist noch TODO."""

    particle_density_kg_per_m3: float = 1500.0
    sphericity: float = 0.9
    cp_solid_j_per_kg_k: float = 1200.0
    cp_water_liquid_j_per_kg_k: float = 4180.0
    cp_suspension_j_per_kg_k: float = 3500.0
    suspension_density_kg_per_m3: float = 1100.0
    # Trocknungskinetik: normierte Trocknungskurve, linear fallende Rate
    critical_moisture_kg_per_kg: float = 0.020
    # Nominalwert/Anker der Gleichgewichtsfeuchte. Doppelrolle, unabhängig von
    # sorption_isotherm_kind: (i) Eingang der Validierung gegen
    # critical_moisture_kg_per_kg unten -- eine grobe, isothermenunabhängige
    # Plausibilitätsschranke, keine Aussage über X_eq(phi,T); (ii) bei
    # sorption_isotherm_kind="gab" der Ankerwert, an den die GAB-Parameter
    # unten kalibriert sind (phi=0.5, T=gab_reference_temperature_celsius --
    # siehe sorption.GabIsotherm). Bei sorption_isotherm_kind="constant" ist es
    # zugleich der tatsächliche, zustandsunabhängige X_eq-Wert der Simulation.
    equilibrium_moisture_kg_per_kg: float = 0.003
    max_saturation_degree: float = 0.60
    max_driving_humidity_kg_per_kg: float = 0.50
    # Sorptionsisotherme der Trocknungskurve (siehe simulation.sorption).
    # "constant" reproduziert equilibrium_moisture_kg_per_kg exakt als
    # zustandsunabhängigen X_eq-Wert -- Default, rückwärtskompatibel: bestehende
    # Configs ohne diese Felder laden unverändert. "gab" aktiviert die
    # zustandsabhängige GAB-Isotherme unten; alle gab_*-Felder sind dann
    # Pflicht (siehe __post_init__). GabIsotherm markiert sich selbst
    # unabhängig von provenance.status als "placeholder" (siehe sorption.py).
    sorption_isotherm_kind: SorptionIsothermKind = "constant"
    gab_monolayer_moisture_kg_per_kg: float | None = None
    gab_c_constant_at_reference_temperature: float | None = None
    gab_k_constant_at_reference_temperature: float | None = None
    gab_c_enthalpy_j_per_mol: float | None = None
    gab_k_enthalpy_j_per_mol: float | None = None
    gab_reference_temperature_celsius: float | None = None
    # Populationsbilanz (reduziert auf zwei Momente)
    nucleus_diameter_m: float = 150e-6
    nucleation_solid_fraction: float = 0.008
    initial_diameter_m: float = 400e-6

    def __post_init__(self) -> None:
        _require_positive(
            self,
            "particle_density_kg_per_m3",
            "cp_solid_j_per_kg_k",
            "nucleus_diameter_m",
            "initial_diameter_m",
            "max_driving_humidity_kg_per_kg",
        )
        if not 0.0 < self.max_saturation_degree <= 1.0:
            raise ValueError("max_saturation_degree muss in (0, 1] liegen")
        if self.critical_moisture_kg_per_kg <= self.equilibrium_moisture_kg_per_kg:
            raise ValueError(
                "critical_moisture_kg_per_kg muss über equilibrium_moisture_kg_per_kg liegen"
            )
        if not 0.0 <= self.nucleation_solid_fraction < 1.0:
            raise ValueError("nucleation_solid_fraction muss in [0, 1) liegen")
        self._validate_gab_fields()

    def _validate_gab_fields(self) -> None:
        """Bei ``sorption_isotherm_kind == "gab"``: alle gab_*-Felder gesetzt und plausibel.

        Rein lokale Prüfung (keine Abhängigkeit von ``sorption.py`` oder anderen
        ``ProcessConfig``-Feldern) -- die weitergehende Prüfung, dass
        ``critical_moisture_kg_per_kg`` über dem Maximum der Isotherme im
        Sicherheitsraster liegt, braucht Aktions-/Störgrößengrenzen aus dem
        umgebenden ``ProcessConfig`` und lebt deshalb in
        ``sorption.validate_isotherm_against_critical_moisture`` -- siehe dort,
        warum sie nicht hierher gehört (Abhängigkeitsrichtung: ``config.py``
        bleibt frei von Imports aus ``sorption.py``/``thermo.py``).
        """
        if self.sorption_isotherm_kind != "gab":
            return
        gab_fields = {
            "gab_monolayer_moisture_kg_per_kg": self.gab_monolayer_moisture_kg_per_kg,
            "gab_c_constant_at_reference_temperature": self.gab_c_constant_at_reference_temperature,
            "gab_k_constant_at_reference_temperature": self.gab_k_constant_at_reference_temperature,
            "gab_c_enthalpy_j_per_mol": self.gab_c_enthalpy_j_per_mol,
            "gab_k_enthalpy_j_per_mol": self.gab_k_enthalpy_j_per_mol,
            "gab_reference_temperature_celsius": self.gab_reference_temperature_celsius,
        }
        missing = sorted(name for name, value in gab_fields.items() if value is None)
        if missing:
            raise ValueError(
                f"sorption_isotherm_kind='gab' verlangt alle gab_*-Felder gesetzt, "
                f"fehlend: {missing}"
            )
        monolayer = self.gab_monolayer_moisture_kg_per_kg
        c_ref = self.gab_c_constant_at_reference_temperature
        k_ref = self.gab_k_constant_at_reference_temperature
        assert monolayer is not None and c_ref is not None and k_ref is not None
        if monolayer <= 0.0:
            raise ValueError(f"gab_monolayer_moisture_kg_per_kg muss positiv sein, ist {monolayer}")
        if c_ref <= 0.0:
            raise ValueError(
                f"gab_c_constant_at_reference_temperature muss positiv sein, ist {c_ref}"
            )
        if not 0.0 < k_ref < 1.0:
            raise ValueError(
                "gab_k_constant_at_reference_temperature muss in (0, 1) liegen "
                f"(Polstelle der GAB-Gleichung bei K*phi -> 1), ist {k_ref}"
            )

    @property
    def nucleation_limited_diameter_m(self) -> float:
        r"""Beharrungsdurchmesser des Spezialfalls ohne Keimzufuhr.

        Gilt nur für ``seed_mass_flow_kg_per_h == 0``. Aus der Zahlenbilanz im
        Beharrungszustand folgt für diesen Fall

        .. math:: d_{50,\mathrm{ss}} = d_{\mathrm{nuc}}\, b_{\mathrm{nuc}}^{-1/3}

        Mit Keimzufuhr liefert der allgemeine Beharrungszustand einen anderen Wert --
        siehe :mod:`twinlink_ml.simulation.steady_state`. Der Nukleationsanteil ist
        damit **kein** freier Tuning-Parameter, sondern durch die Zielkorngröße
        festgelegt. Siehe :func:`nucleation_fraction_for_target_diameter`.
        """
        if self.nucleation_solid_fraction == 0.0:
            return float("inf")
        return float(self.nucleus_diameter_m * self.nucleation_solid_fraction ** (-1.0 / 3.0))


@dataclass(frozen=True, slots=True)
class EnergyConfig:
    """Aggregate der Energiebilanz. Bestimmt die Zielgröße ``dE``."""

    heater_efficiency: float = 0.98
    fan_efficiency: float = 0.55
    distributor_loss_coefficient: float = 350.0
    pump_pressure_rise_pa: float = 3.0e5
    pump_efficiency: float = 0.40
    atomiser_air_kg_per_s_per_bar: float = 1.5e-4
    compressor_efficiency: float = 0.50
    target: EnergyTarget = "power_kw"
    reference_value: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "heater_efficiency",
            "fan_efficiency",
            "pump_efficiency",
            "compressor_efficiency",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} muss in (0, 1] liegen, ist {value}")


@dataclass(frozen=True, slots=True)
class ExhaustConfig:
    """Nachgelagerte Abluft-Schnittstelle: Rohrleitung/Filter zwischen Bett und Wärmeverbund.

    Bewusst eigener Abschnitt, nicht Teil von :class:`GeometryConfig`: dessen
    ``heat_loss_ua_w_per_k`` ist ein Term der Bett-Enthalpiebilanz (``greybox._rhs``) und
    verändert die simulierte Zustandstrajektorie. Felder hier wirken ausschließlich
    nachgelagert in :func:`twinlink_ml.simulation.exhaust.reconstruct_exhaust_state` -- rein
    algebraische Nachbearbeitung von bereits simuliertem/vorhergesagtem Zustand, kein
    Zustand selbst und kein Eingang in ``_rhs``. Beide im selben Abschnitt zu führen würde
    dazu einladen, sie versehentlich zu addieren oder den neuen Parameter fälschlich in der
    ODE zu erwarten.

    STATUS: Platzhalter, keine Quelle -- Rohrleitungs-/Filterverlust zwischen Bett und
    Wärmeübertrager, bislang nicht vermessen und nicht bei Anwendungspartner erfragt.
    """

    outlet_duct_heat_loss_k: float = 0.0
    """Pauschaler Temperaturabfall zwischen Bett-Austritt und der gemessenen
    Ablufttemperatur am Wärmeverbund, in Kelvin. Vorzeichenkonvention:

    .. math:: T_{outlet} = T_{bed} - \\text{outlet\\_duct\\_heat\\_loss\\_k}

    Ein positiver Wert bedeutet Abkühlung entlang Rohrleitung/Filter; negative Werte sind
    nicht vorgesehen (keine Wärmequelle zwischen Bett und Übertrager) und werden in
    ``__post_init__`` zurückgewiesen. Default 0.0: Identität zu ``bed_temperature_celsius``,
    numerisch keine Änderung gegenüber dem bisherigen (fehlerhaft benannten) Verhalten.

    Bewusst ein pauschaler Abkühlbetrag, kein UA-Wert: anders als
    ``geometry.heat_loss_ua_w_per_k`` hängt dieser Platzhalter nicht von Massenstrom oder
    Umgebungstemperatur ab -- kein Wärmeübertragungsmodell, nur ein fester Versatz. Ein
    UA-basierter Ansatz wäre die spätere Verfeinerung, sobald eine Quelle vorliegt.
    """

    def __post_init__(self) -> None:
        _require_non_negative(self, "outlet_duct_heat_loss_k")


@dataclass(frozen=True, slots=True)
class OperationConfig:
    """Betriebsart. ``02_prozess_setup.md``: kontinuierlich/Batch/Semi-Batch ist TODO."""

    mode: OperationMode = "continuous"
    mass_control_time_constant_s: float = 600.0
    seed_mass_flow_kg_per_h: float = 0.0
    suspension_temperature_celsius: float = 20.0
    initial_moisture_kg_per_kg: float = 0.005
    initial_bed_temperature_celsius: float = 55.0

    def __post_init__(self) -> None:
        _require_positive(self, "mass_control_time_constant_s")
        _require_non_negative(self, "seed_mass_flow_kg_per_h", "initial_moisture_kg_per_kg")


@dataclass(frozen=True, slots=True)
class ActionBound:
    """Box- und Ratenbegrenzung einer Stellgröße."""

    minimum: float
    maximum: float
    nominal: float
    max_rate_per_s: float

    def __post_init__(self) -> None:
        if not self.minimum <= self.nominal <= self.maximum:
            raise ValueError(
                f"nominal={self.nominal} liegt außerhalb [{self.minimum}, {self.maximum}]"
            )
        if self.max_rate_per_s <= 0.0:
            raise ValueError("max_rate_per_s muss positiv sein")

    @property
    def span(self) -> float:
        return self.maximum - self.minimum


@dataclass(frozen=True, slots=True)
class ActionSpaceConfig:
    """Stellgrößenraum. Alle Werte TODO in ``02_prozess_setup.md`` -> Platzhalter.

    Die Normierung auf ``[0, 1]`` (siehe :meth:`normalise`) ist die Umgehung aus dem
    Arbeitsplan: das Sampling-Design und die RL-Action-Bounds bleiben unverändert,
    wenn die echten Bereiche eingesetzt werden.
    """

    inlet_temperature_celsius: ActionBound = field(
        default_factory=lambda: ActionBound(60.0, 120.0, 90.0, 0.5)
    )
    gas_velocity_m_per_s: ActionBound = field(
        default_factory=lambda: ActionBound(0.8, 2.5, 1.6, 0.05)
    )
    spray_rate_kg_per_h: ActionBound = field(
        default_factory=lambda: ActionBound(0.4, 3.0, 1.5, 0.02)
    )
    nozzle_pressure_bar: ActionBound = field(
        default_factory=lambda: ActionBound(1.0, 4.0, 2.5, 0.05)
    )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    @property
    def bounds(self) -> tuple[ActionBound, ...]:
        return tuple(getattr(self, name) for name in self.names)


@dataclass(frozen=True, slots=True)
class DisturbanceRange:
    """Bereich einer Störgröße für das Latin-Hypercube-Sampling (Schritt 2)."""

    minimum: float
    maximum: float
    nominal: float

    def __post_init__(self) -> None:
        if not self.minimum <= self.nominal <= self.maximum:
            raise ValueError("nominal liegt außerhalb des Bereichs")


@dataclass(frozen=True, slots=True)
class DisturbanceSpaceConfig:
    """Exogene, messbare, nicht steuerbare Größen.

    Der Tag/Nacht-Indikator aus ``02_prozess_setup.md`` ist bewusst **keine** physikalische
    Eingangsgröße des Modells: er wirkt ausschließlich über ``T_amb`` und ``phi_amb``.
    Als Feature für das Sequenzmodell wird er aus dem Zeitstempel abgeleitet
    (sin/cos-Kodierung), nicht vom Simulator geliefert. Sonst bekommt das Modell einen
    Eingang, der an der Realanlage keine kausale Wirkung hat.
    """

    ambient_temperature_celsius: DisturbanceRange = field(
        default_factory=lambda: DisturbanceRange(5.0, 35.0, 20.0)
    )
    ambient_relative_humidity: DisturbanceRange = field(
        default_factory=lambda: DisturbanceRange(0.25, 0.85, 0.50)
    )
    suspension_solid_fraction: DisturbanceRange = field(
        default_factory=lambda: DisturbanceRange(0.20, 0.40, 0.30)
    )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    @property
    def ranges(self) -> tuple[DisturbanceRange, ...]:
        return tuple(getattr(self, name) for name in self.names)


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    """Zeitliche Auflösung. ``02_prozess_setup.md``: Δt Dyssol ist TODO."""

    sample_time_s: float = 5.0
    substeps: int = 5
    integrator: Integrator = "rk4"
    state_floor: float = 1e-9
    enforce_action_bounds: bool = True
    enforce_gas_velocity_projection: bool = False
    """Projiziert ``gas_velocity_m_per_s`` je Schritt auf ``[u_mf(d50), u_terminal(d50)]``,
    berechnet aus dem wahren Zustand unmittelbar vor dem Integrationsschritt (siehe
    ``ProcessSimulator.rollout``/``GreyBoxFluidBedGranulator.project_action``). Kein
    Sicherheitsabstand (Faktor 1.0), kein anderer Kanal betroffen -- insbesondere bleibt
    ``inlet_temperature_celsius`` unbeschraenkt (Befund 49). Default AUS: Teil von
    ``NumericsConfig``, geht damit automatisch in ``ProcessConfig.fingerprint`` ein, sodass
    Datensaetze mit und ohne Projektion nie denselben Fingerprint tragen (siehe
    ``docs/04_stand_ap1.md`` Befund 46, ``scripts/probe_gas_velocity_projection.py``)."""

    def __post_init__(self) -> None:
        _require_positive(self, "sample_time_s")
        if self.substeps < 1:
            raise ValueError("substeps muss >= 1 sein")

    @property
    def internal_time_step_s(self) -> float:
        return self.sample_time_s / self.substeps


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    """Sensormodell: Rauschen und Sensordynamik.

    Warum das nicht optional ist: das Rauschniveau bestimmt die Lernkurven stärker als
    die Architektur. Auf rauschfreien Simulationsdaten sättigt jedes hinreichend große
    Modell und der Vergleich wird bedeutungslos. Die Werte hier sind Platzhalter und
    müssen aus den Anwendungspartner-Zeitreihen bzw. den Laborsensoren kalibriert werden.
    """

    enabled: bool = True
    relative_noise_std: float = 0.005
    absolute_noise_std_temperature_k: float = 0.2
    sensor_time_constant_s: float = 10.0

    def __post_init__(self) -> None:
        _require_non_negative(
            self,
            "relative_noise_std",
            "absolute_noise_std_temperature_k",
            "sensor_time_constant_s",
        )


@dataclass(frozen=True, slots=True)
class ProcessConfig:
    """Vollständige Konfiguration eines Simulationsexperiments."""

    provenance: Provenance = field(default_factory=Provenance)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    exhaust: ExhaustConfig = field(default_factory=ExhaustConfig)
    operation: OperationConfig = field(default_factory=OperationConfig)
    actions: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    disturbances: DisturbanceSpaceConfig = field(default_factory=DisturbanceSpaceConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    backend: str = "numpy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Stabiler Hash der aufgelösten Config -- gehört in jeden Run-Log.

        Gilt für die ANGEWENDETE Config (diese Instanz, so wie sie tatsächlich simuliert
        wird), nicht für die Datei wie geladen: ``sample_time_s`` wird typischerweise von
        ``DataSpec.sample_time_s`` überschrieben (siehe ``evaluation.experiments.generate_data``,
        ``dataclasses.replace`` auf ``numerics`` vor dem Simulator-Aufbau), bevor der Fingerprint
        gebildet wird. Ein Vergleich gegen den Fingerprint der unveränderten, per
        ``load_process_config`` geladenen Datei (deren ``sample_time_s`` der YAML-Default ist)
        liefert deshalb einen ANDEREN Hash als der der tatsächlich verwendeten Config.

        Berechnung über :func:`config_fingerprint` -- derselbe Algorithmus wie
        :class:`~twinlink_ml.heatnetwork.config.HeatNetworkConfig`s Fingerprint, ein Hashing-
        Verfahren für alle Config-Hierarchien im Projekt statt je Klasse neu gebaut.

        Dokumentarische Freitextfelder (``provenance.description``, ``provenance.
        open_todos``) fließen NICHT ein -- siehe :func:`without_documentary_provenance`.
        """
        return config_fingerprint(without_documentary_provenance(self.to_dict()))

    @property
    def section_fingerprints(self) -> dict[str, str]:
        """Teil-Fingerprints je Config-Abschnitt, additiv zu :attr:`fingerprint`.

        :attr:`fingerprint` ist ein monolithischer Hash über die gesamte Config -- bei
        einer Abweichung zwischen zwei Configs (wie bei ``arch_composite.jsonl``,
        Befund 33: ``process/fingerprint`` stimmt mit keiner aktuellen
        ``conf/process/*.yaml`` überein) lässt sich damit nicht feststellen, WELCHER
        Abschnitt abweicht, ohne jede Kombination durchzuprobieren. Diese Teil-Fingerprints
        erlauben den direkten, abschnittsweisen Vergleich zweier :class:`ProcessConfig`
        -Instanzen. ``fingerprint`` selbst bleibt unverändert und bitgleich zu bereits
        geloggten Records; dies ist eine rein additive Erweiterung.
        """
        sections = (
            "provenance",
            "geometry",
            "gas",
            "material",
            "energy",
            "exhaust",
            "operation",
            "actions",
            "disturbances",
            "numerics",
            "measurement",
            "backend",
        )
        return section_fingerprints_of(without_documentary_provenance(self.to_dict()), sections)

    def warn_if_placeholder(self) -> None:
        if self.provenance.status == "placeholder":
            warnings.warn(
                "ProcessConfig enthält Platzhalterwerte "
                f"(offene TODOs: {len(self.provenance.open_todos)}). "
                "Absolutwerte sind nicht belastbar; nur relative Architekturvergleiche.",
                PlaceholderConfigWarning,
                stacklevel=2,
            )


_DOCUMENTARY_PROVENANCE_FIELDS = ("description", "open_todos")
"""Freitextfelder in :class:`Provenance`, die den Fingerprint NICHT beeinflussen sollen.

Begründung (Audit 2026-09-25, siehe :func:`without_documentary_provenance`): der
Fingerprint identifiziert die NUMERISCHE Konfiguration. Eine reine Textänderung an
``provenance.description`` oder ``provenance.open_todos`` (Präzisierung, abgehaktes
TODO, künftige Übersetzung ins Englische) ändert an der simulierten Physik nichts und
darf die Vergleichbarkeit zweier Läufe nicht brechen. Alle Felder aller Config-Klassen
wurden geprüft, im Zweifel blieb ein Feld im Hash:

* ``provenance.status`` bleibt im Hash -- kategorial (placeholder/partially_calibrated/
  calibrated), semantisch relevant für die Aussagekraft eines Laufs.
* ``HeatSourceConfig.name``/``HeatSinkConfig.name`` bleiben im Hash -- Referenzpunkte für
  ``routing.py``/``cascade_order``, kein Freitext, sondern Bezeichner mit Bedeutung.
* ``ProcessConfig.backend`` bleibt im Hash -- wählt die Rechen-Implementierung (numpy/
  torch), kein Kommentar.
* Alle übrigen Felder sind Zahlen, Enums oder Bezeichner, keine Prosa.
"""


def without_documentary_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    """Entfernt :data:`_DOCUMENTARY_PROVENANCE_FIELDS` aus ``payload["provenance"]``.

    Arbeitet auf der vollen, verschachtelten ``to_dict()``-Form (mit ``"provenance"``-
    Schlüssel) -- :func:`section_fingerprints_of` liest ``payload[section]`` danach aus
    demselben, bereits bereinigten Dict, sodass ``section_fingerprints["provenance"]``
    und :attr:`ProcessConfig.fingerprint` dieselbe Bereinigung sehen. Kein ``"provenance"``-
    Schlüssel (z. B. ein einzelner, bereits extrahierter Abschnitt ohne Provenance) --
    unverändert zurückgeben.
    """
    if "provenance" not in payload or not isinstance(payload["provenance"], dict):
        return payload
    result = dict(payload)
    result["provenance"] = {
        key: value
        for key, value in payload["provenance"].items()
        if key not in _DOCUMENTARY_PROVENANCE_FIELDS
    }
    return result


def config_fingerprint(payload: dict[str, Any]) -> str:
    """Stabiler 16-stelliger Hex-Hash eines aufgelösten Config-``dict`` (``dataclasses.asdict``).

    Gemeinsamer Hashing-Algorithmus für alle Config-Hierarchien des Projekts
    (:class:`ProcessConfig`, ``heatnetwork.config.HeatNetworkConfig``) -- ein Verfahren,
    einmal geprüft (Stabilität, Sortierung der Schlüssel, ``str``-Fallback für nicht
    JSON-native Typen), nicht je Config-Klasse per Copy-Paste neu gebaut. Extrahiert aus
    :attr:`ProcessConfig.fingerprint`; identische Eingabe liefert identische Ausgabe wie
    vor dieser Extraktion (reiner Refactor, kein neues Verhalten).

    Erwartet, dass ``payload`` bereits von dokumentarischen Freitextfeldern bereinigt ist
    (:func:`without_documentary_provenance`) -- diese Funktion selbst hasht, was man ihr
    gibt, ohne eigene Kenntnis der Config-Struktur (bleibt so generisch für beliebige
    Config-Hierarchien, wie der Docstring oben verspricht).
    """
    return str(
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
    )


def section_fingerprints_of(
    payload: dict[str, Any], sections: tuple[str, ...]
) -> dict[str, str]:
    """:func:`config_fingerprint` je benanntem Abschnitt von ``payload``.

    Gemeinsame Basis für :attr:`ProcessConfig.section_fingerprints` und die entsprechende
    Eigenschaft auf ``HeatNetworkConfig`` -- siehe :func:`config_fingerprint`.
    """
    return {section: config_fingerprint(payload[section]) for section in sections}


def nucleation_fraction_for_target_diameter(
    target_diameter_m: float, nucleus_diameter_m: float
) -> float:
    r"""Nukleationsanteil, der im Beharrungszustand die Zielkorngröße liefert.

    .. math:: b_{\mathrm{nuc}} = \left(\frac{d_{\mathrm{nuc}}}{d_{50,\mathrm{soll}}}\right)^{3}
    """
    if target_diameter_m <= nucleus_diameter_m:
        raise ValueError("Zielkorngröße muss über der Keimgröße liegen")
    return (nucleus_diameter_m / target_diameter_m) ** 3


def load_process_config(path: str | Path) -> ProcessConfig:
    """Lädt eine YAML-Config in die Dataclass-Hierarchie.

    Bewusst ohne Hydra-Abhängigkeit: die Dataclasses sind gültige Hydra-Structured-Configs
    und können später per ``ConfigStore.store(node=ProcessConfig)`` registriert werden,
    ohne dass sich das Schema ändert.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: erwartet ein YAML-Mapping auf oberster Ebene")
    config = _from_mapping(ProcessConfig, raw)
    config.warn_if_placeholder()
    return config


def dump_process_config(config: ProcessConfig, path: str | Path) -> None:
    """Serialisiert die vollständig aufgelöste Config (Reproduzierbarkeitsanforderung)."""
    Path(path).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _from_mapping(cls: type[T], data: dict[str, Any]) -> T:
    """Rekursive, typgeleitete Konstruktion -- keine losen Dicts im Modellcode."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} ist keine Dataclass")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unbekannte Felder {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        annotation = known[name].type
        if isinstance(value, dict):
            nested = _resolve_nested_type(annotation)
            kwargs[name] = _from_mapping(nested, value) if nested is not None else value
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


_NESTED_TYPES: dict[str, type[Any]] = {
    "Provenance": Provenance,
    "GeometryConfig": GeometryConfig,
    "GasConfig": GasConfig,
    "MaterialConfig": MaterialConfig,
    "EnergyConfig": EnergyConfig,
    "ExhaustConfig": ExhaustConfig,
    "OperationConfig": OperationConfig,
    "ActionSpaceConfig": ActionSpaceConfig,
    "ActionBound": ActionBound,
    "DisturbanceSpaceConfig": DisturbanceSpaceConfig,
    "DisturbanceRange": DisturbanceRange,
    "NumericsConfig": NumericsConfig,
    "MeasurementConfig": MeasurementConfig,
}


def _resolve_nested_type(annotation: Any) -> type[Any] | None:
    key = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    return _NESTED_TYPES.get(key)


def _require_positive(obj: Any, *names: str) -> None:
    for name in names:
        if getattr(obj, name) <= 0.0:
            raise ValueError(f"{name} muss positiv sein, ist {getattr(obj, name)}")


def _require_non_negative(obj: Any, *names: str) -> None:
    for name in names:
        if getattr(obj, name) < 0.0:
            raise ValueError(f"{name} darf nicht negativ sein, ist {getattr(obj, name)}")
