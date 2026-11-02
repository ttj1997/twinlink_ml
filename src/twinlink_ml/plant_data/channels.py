r"""Measuring-point mapping of the application partner (plant F-5, granulator D 3204.

Source: three control-system screenshots of the application partner (granulation, heat
recovery 1/2, date 05.-08.09.2025 -- a different point in time than the two
validated exports, see below) as well as the column headers of the two 24h exports
themselves (file 1: product "prod1", 26.03.2026; file 2: product "prod2", 11.05.2026,
each 1441 rows at one-minute intervals). Product names are deliberately not written out here
(customer data of the application partner, see NA1/``docs/04_stand_ap1.md``).

Two data sources diverge; it is important here to keep them apart:

1. The **export files** (31 logged channels, see :data:`GRANULATOR_D3204_CHANNELS`
   with ``exported=True``) -- this is the validated time series.
2. The **screenshot** (snapshot on a third date) shows additional
   channels that are NOT contained in the export (``exported=False`` below) -- among
   others bed mass, suspension density, nozzle pressure and the pressure drops. These are
   only listed for orientation/geometry estimation (see NA5 script), not available as a
   time series.

Important finding when comparing screenshot <-> export column headers: the channels
``FIRAL-323805`` and ``FIRAL-323604`` carry the unit ``[°C]`` in the export file
in row 2 -- this is a labelling error of the source. The tag prefix ``FIR``
(flow indicator) and the screenshot value (FIRAL-323604: 37.2 m3/h,
FIRAL-323805: 17.98 m3/h, both close to the mean of the respective export column)
confirm: both are flows in m3/h, not temperatures. The unit is
therefore explicitly corrected below (``unit="m3/h"``), not taken from the file.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GRANULATOR_D3204_CHANNELS",
    "PlantChannel",
    "ambient_temperature_tag",
    "bed_temperature_tags",
    "channels_by_tag",
    "inlet_temperature_tags",
    "nozzle_atomising_air_tags",
    "outlet_temperature_tag",
    "process_air_flow_tag",
    "spray_rate_tag",
    "upper_position_temperature_tags",
]


@dataclass(frozen=True, slots=True)
class PlantChannel:
    """A measuring-point tag of the control system and its meaning in the model context.

    ``confirmed`` is independent of ``exported``: a channel can be present in the export
    and still be unconfirmed in its *meaning* (see the upper temperature
    triad, NA3) -- and conversely a channel from the screenshot can be unambiguous in its
    meaning although it was not exported (e.g. ``PDIRAH-320404``, whose
    screenshot value agrees with an independent control-system display).
    """

    tag: str
    unit: str
    description: str
    exported: bool
    confirmed: bool
    note: str = ""


GRANULATOR_D3204_CHANNELS: tuple[PlantChannel, ...] = (
    # -- Inlet air / ambient -------------------------------------------------------
    PlantChannel(
        "FIRCAL-322411",
        "Nm3/h",
        "Prozessluftmenge (mit Sollwertanzeige im Leitsystem)",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "TIR-322412",
        "degC",
        "Ansaugluft-/Umgebungstemperatur",
        exported=True,
        confirmed=True,
        note=(
            "Messwerte plausibel fuer die Exportdaten: product 1 (26.03.) 0.9-7.6 degC, "
            "product 2 (11.05.) 7.7-15.3 degC -- konsistent mit den Dateidaten (frueres "
            "Fruehjahr vs. Mai)."
        ),
    ),
    PlantChannel(
        "TRZSAH-320408",
        "degC",
        "Zulufttemperatur, Granulatorunterteil unter dem Anstroemboden (Sensor 1/3)",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "TRZSAH-320409",
        "degC",
        "Zulufttemperatur, Granulatorunterteil unter dem Anstroemboden (Sensor 2/3)",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "TRZSAH-320410",
        "degC",
        "Zulufttemperatur, Granulatorunterteil unter dem Anstroemboden (Sensor 3/3)",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "TIRCSAH-320411",
        "degC",
        "Zuluft, Istwert-Auswahl der Regelung (Sensor 1/3)",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar (~80.5-80.6 degC), nicht Teil der Exportdatei.",
    ),
    PlantChannel(
        "TIRCSAH-320412",
        "degC",
        "Zuluft, Istwert-Auswahl der Regelung (Sensor 2/3)",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar, nicht Teil der Exportdatei.",
    ),
    PlantChannel(
        "TIRCSAH-320413",
        "degC",
        "Zuluft, Istwert-Auswahl der Regelung (Sensor 3/3)",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar, nicht Teil der Exportdatei.",
    ),
    # -- Bed temperature -------------------------------------------------------------
    PlantChannel(
        "TIRSAH-320414",
        "degC",
        "Betttemperatur (Sensor 1/3)",
        exported=True,
        confirmed=True,
        note=(
            "Bestaetigt durch Leitsystem-Anzeige 'Betttemperatur: 43-44 degC' "
            "(Screenshot Granulation): TIRSAH-320414/415/416 lesen exakt in diesem "
            "Fenster. Als Modell-Betttemperatur verwendet."
        ),
    ),
    PlantChannel(
        "TIRSAH-320415",
        "degC",
        "Betttemperatur (Sensor 2/3)",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "TIRSAH-320416",
        "degC",
        "Betttemperatur (Sensor 3/3)",
        exported=True,
        confirmed=True,
    ),
    # -- Higher position (NA3, unconfirmed) -------------------------------------------
    PlantChannel(
        "TRZSAH-320417",
        "degC",
        "Temperatur an hoeherer Position im Apparat D 3204 (Sensor 1/3)",
        exported=True,
        confirmed=False,
        note=(
            "Position im Screenshot oberhalb der Bett-Trias bestaetigt (gleiche "
            "vertikale Zone, Zylinderteil von D 3204 oberhalb TIRSAH-320414/416). "
            "In BEIDEN Exportdateien liest diese Trias hoeher als die Bett-Trias "
            "(product 1 +0.89 K, product 2 +3.10 K im Mittel) -- kein Einzelfall. Anders als "
            "die eng zusammenliegende Bett- und Zuluft-Trias (Streuung < 0.4 K "
            "zwischen den drei Sensoren) streuen 417/418/419 selbst deutlich "
            "untereinander (product 2: 45.6/47.4/46.1 degC, Spanne 1.8 K) -- Hinweis auf "
            "eine nicht gut durchmischte, ortsabhaengige Messung (z. B. Wandeinfluss "
            "oder unterschiedliche Einbauposition je Sensor) statt dreier Redundanzen "
            "derselben Kernstroemung. Ob Gas- oder Wandmessung, laesst sich aus dem "
            "Screenshot NICHT entscheiden -- offene Rueckfrage an den "
            "Anwendungspartner. Fuer die Validierung nicht als Betttemperatur "
            "verwendet; TIRSAH-320414/415/416 bleibt die Referenz, mit diesem "
            "Vorbehalt."
        ),
    ),
    PlantChannel(
        "TRZSAH-320418",
        "degC",
        "Temperatur an hoeherer Position im Apparat D 3204 (Sensor 2/3)",
        exported=True,
        confirmed=False,
        note="Siehe TRZSAH-320417.",
    ),
    PlantChannel(
        "TRZSAH-320419",
        "degC",
        "Temperatur an hoeherer Position im Apparat D 3204 (Sensor 3/3)",
        exported=True,
        confirmed=False,
        note="Siehe TRZSAH-320417.",
    ),
    # -- Exhaust air -------------------------------------------------------------------
    PlantChannel(
        "TIRSAH-320420",
        "degC",
        "Abluft am Granulatoraustritt, vor dem Filter",
        exported=True,
        confirmed=True,
        note="Position im Screenshot am Apparatkopf, vor 'Filter'/V 3246 bestaetigt.",
    ),
    # -- Additional channels, not expected in the NA2 initial state ---------------------
    PlantChannel(
        "TIR-324010",
        "degC",
        "Filterstrecke F 3240, nicht im Grey-Box-Modell abgebildet",
        exported=True,
        confirmed=True,
        note="Im Filter-Screenshot exakt an gleicher Stelle/Wert bestaetigt (35.0 degC).",
    ),
    PlantChannel(
        "TIRAH-324405",
        "degC",
        "Filterstrecke F 3240, nicht im Grey-Box-Modell abgebildet",
        exported=True,
        confirmed=True,
        note="Im Filter-Screenshot exakt an gleicher Stelle/Wert bestaetigt (35.0 degC).",
    ),
    # -- Fans ----------------------------------------------------------------------------
    PlantChannel(
        "Leistung V-3226",
        "kW",
        "Ventilatorleistung (Prozessluft)",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "Leistung V-3246",
        "kW",
        "Ventilatorleistung (Abluft/Filter)",
        exported=True,
        confirmed=True,
    ),
    # -- Spray system ----------------------------------------------------------------------
    PlantChannel(
        "FIR-325160",
        "kg/h",
        "Spruehrate",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "DIR-325160",
        "kg/m3",
        "Suspensionsdichte",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar (1339 kg/m3), nicht Teil der Exportdatei.",
    ),
    PlantChannel(
        "FIRSAL-320515",
        "m3/h",
        "Zerstaeuberluft, Duese 1/6",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "FIRSAL-320525",
        "m3/h",
        "Zerstaeuberluft, Duese 2/6",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "FIRSAL-320535",
        "m3/h",
        "Zerstaeuberluft, Duese 3/6",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "FIRSAL-320545",
        "m3/h",
        "Zerstaeuberluft, Duese 4/6",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "FIRSAL-320555",
        "m3/h",
        "Zerstaeuberluft, Duese 5/6",
        exported=True,
        confirmed=True,
        note="Bei product 1 inaktiv (Messwert nahe 0, leicht negativ -- Nullpunktrauschen).",
    ),
    PlantChannel(
        "FIRSAL-320565",
        "m3/h",
        "Zerstaeuberluft, Duese 6/6",
        exported=True,
        confirmed=True,
        note="Bei product 1 inaktiv (Messwert nahe 0, leicht negativ -- Nullpunktrauschen).",
    ),
    PlantChannel(
        "PIRSAH-320513",
        "bar",
        "Duesendruck, Duese 1/6",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar, nicht Teil der Exportdatei.",
    ),
    PlantChannel(
        "PIRSAH-320523",
        "bar",
        "Duesendruck, Duese 2/6",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar, nicht Teil der Exportdatei.",
    ),
    PlantChannel(
        "PIRSAH-320533",
        "bar",
        "Duesendruck, Duese 3/6",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar, nicht Teil der Exportdatei.",
    ),
    PlantChannel(
        "PIRSAH-320543",
        "bar",
        "Duesendruck, Duese 4/6",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar, nicht Teil der Exportdatei.",
    ),
    PlantChannel(
        "PIRSAH-320553",
        "bar",
        "Duesendruck, Duese 5/6",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar, nicht Teil der Exportdatei.",
    ),
    PlantChannel(
        "PIRSAH-320563",
        "bar",
        "Duesendruck, Duese 6/6",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar, nicht Teil der Exportdatei.",
    ),
    # -- Pressure drops / bed mass -------------------------------------------------------
    PlantChannel(
        "PDIRAH-320402",
        "mbar",
        "Druckverlust Anstroemboden",
        exported=False,
        confirmed=True,
        note="Nur im Screenshot sichtbar (6.05 mbar), nicht Teil der Exportdatei.",
    ),
    PlantChannel(
        "PDIRAH-320404",
        "mbar",
        "Druckverlust Bett",
        exported=False,
        confirmed=True,
        note=(
            "Nur im Screenshot sichtbar (19.2 mbar, konsistent mit der Anzeige "
            "'Betthoehe: 20 mbar'), nicht Teil der Exportdatei. Verwendet in "
            "scripts/study_plant_steady_state.py zur Abschaetzung der "
            "Querschnittsflaeche ueber dp_bed = m_bed * g / A -- Momentaufnahme "
            "an einem dritten Datum, nicht aus den validierten 24h-Exporten."
        ),
    ),
    PlantChannel(
        "A-3215",
        "kg",
        "Bettmasse (Zuordnung unbestaetigt, siehe Note)",
        exported=False,
        confirmed=False,
        note=(
            "Nur im Screenshot sichtbar (600.0 kg), ein Regler ist nicht abgebildet. "
            "Anwendungspartner nannte muendlich 800 kg -- Diskrepanz ungeklaert, siehe "
            "NA5-Bericht. Die Zuordnung 'Bettmasse' selbst ist eine Vermutung, keine "
            "bestaetigte Bedeutung: das Praefix 'A-' bezeichnet in diesem Leitsystem "
            "sonst Aggregate (A-3234 Kompressorstation, A-3203 pneumatische "
            "Foerderung) -- A-3215 koennte auch ein Behaelter oder eine Waage sein. "
            "Offen, beim Anwendungspartner zu klaeren (docs/04_stand_ap1.md, Befund "
            "56 NA2-Nachtrag)."
        ),
    ),
    # -- Heat recovery ---------------------------------------------------------------------
    PlantChannel(
        "TIRCSAH-322813",
        "degC",
        "Waermerueckgewinnung, Kompressor-Zwischenkreis (W 3228A)",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "CV-322710/11",
        "%",
        "Waermerueckgewinnung, Ventilstellung",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "TIR-322302",
        "degC",
        "Waermerueckgewinnung, Vorlauf Prozessluftvorwaermung (W 3223)",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "TIR-322303",
        "degC",
        "Waermerueckgewinnung, Ruecklauf Prozessluftvorwaermung (W 3223)",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "FIRAL-323805",
        "m3/h",
        "Waermerueckgewinnung, Volumenstrom Zwischenkreis an W 3237",
        exported=True,
        confirmed=True,
        note=(
            "Einheit in der Exportdatei fehlerhaft als [°C] beschriftet -- Tag-Praefix "
            "FIR und Screenshot-Wert (17.98 m3/h) bestaetigen Durchfluss in m3/h, "
            "siehe Moduldocstring."
        ),
    ),
    PlantChannel(
        "TIR-323603",
        "degC",
        "Waermerueckgewinnung, Warmwasserspeicher B-3236 (Sensor, einer von drei)",
        exported=True,
        confirmed=True,
    ),
    PlantChannel(
        "FIRAL-323604",
        "m3/h",
        "Waermerueckgewinnung, Volumenstrom Zwischenkreis an W 3237",
        exported=True,
        confirmed=True,
        note=(
            "Einheit in der Exportdatei fehlerhaft als [°C] beschriftet -- Tag-Praefix "
            "FIR und Screenshot-Wert (37.2 m3/h) bestaetigen Durchfluss in m3/h, "
            "siehe Moduldocstring."
        ),
    ),
)


def channels_by_tag() -> dict[str, PlantChannel]:
    """Lookup table tag -> :class:`PlantChannel`."""
    return {channel.tag: channel for channel in GRANULATOR_D3204_CHANNELS}


def process_air_flow_tag() -> str:
    """Tag of the process air flow."""
    return "FIRCAL-322411"


def ambient_temperature_tag() -> str:
    """Tag of the intake air / ambient temperature."""
    return "TIR-322412"


def inlet_temperature_tags() -> tuple[str, str, str]:
    """Tags of the inlet air temperature below the distributor plate (three sensors)."""
    return ("TRZSAH-320408", "TRZSAH-320409", "TRZSAH-320410")


def bed_temperature_tags() -> tuple[str, str, str]:
    """Bed temperature -- see :data:`GRANULATOR_D3204_CHANNELS`, confirmed via screenshot."""
    return ("TIRSAH-320414", "TIRSAH-320415", "TIRSAH-320416")


def upper_position_temperature_tags() -> tuple[str, str, str]:
    """Higher position, NOT confirmed as bed temperature -- see NA3 note."""
    return ("TRZSAH-320417", "TRZSAH-320418", "TRZSAH-320419")


def outlet_temperature_tag() -> str:
    """Tag of the exhaust air temperature at the granulator outlet, before the filter."""
    return "TIRSAH-320420"


def spray_rate_tag() -> str:
    """Tag of the spray rate."""
    return "FIR-325160"


def nozzle_atomising_air_tags() -> tuple[str, str, str, str, str, str]:
    """Tags of the atomising air per nozzle (six nozzles)."""
    return (
        "FIRSAL-320515",
        "FIRSAL-320525",
        "FIRSAL-320535",
        "FIRSAL-320545",
        "FIRSAL-320555",
        "FIRSAL-320565",
    )
