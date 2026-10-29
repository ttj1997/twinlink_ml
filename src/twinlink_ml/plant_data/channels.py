r"""Messstellenzuordnung des Anwendungspartners (Anlage F-5, Granulator D 3204.

Quelle: drei Leitsystem-Screenshots des Anwendungspartners (Granulation, Waerme-
rueckgewinnung 1/2, Datum 05.-08.09.2025 -- ein anderer Zeitpunkt als die beiden
validierten Exporte, siehe unten) sowie die Spaltenkoepfe der beiden 24h-Exporte
selbst (Datei 1: Produkt "P1", 26.03.2026; Datei 2: Produkt "P2", 11.05.2026,
je 1441 Zeilen im Minutentakt). Produktnamen werden hier bewusst nicht ausgeschrieben
(Kundendaten des Anwendungspartners, siehe NA1/``docs/04_stand_ap1.md``).

Zwei Datenquellen faellt auseinander, das ist hier wichtig auseinanderzuhalten:

1. Die **Exportdateien** (31 geloggte Kanaele, siehe :data:`GRANULATOR_D3204_CHANNELS`
   mit ``exported=True``) -- das ist die validierte Zeitreihe.
2. Der **Screenshot** (Momentaufnahme an einem dritten Datum) zeigt zusaetzliche
   Kanaele, die NICHT im Export enthalten sind (``exported=False`` unten) -- unter
   anderem Bettmasse, Suspensionsdichte, Duesendruck und die Druckverluste. Diese sind
   nur zur Einordnung/Geometrieabschaetzung aufgefuehrt (siehe NA5-Skript), nicht als
   Zeitreihe verfuegbar.

Wichtiger Befund beim Abgleich Screenshot <-> Exportspaltenkoepfe: die Kanaele
``FIRAL-323805`` und ``FIRAL-323604`` tragen in der Exportdatei die Einheit ``[°C]``
in Zeile 2 -- das ist ein Beschriftungsfehler der Quelle. Der Tag-Praefix ``FIR``
(Durchflussanzeige) und der Screenshot-Wert (FIRAL-323604: 37.2 m3/h,
FIRAL-323805: 17.98 m3/h, beide nahe am Mittelwert der jeweiligen Exportspalte)
bestaetigen: beides sind Durchfluesse in m3/h, keine Temperaturen. Die Einheit ist
deshalb unten explizit korrigiert (``unit="m3/h"``), nicht aus der Datei uebernommen.
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
    """Ein Messstellenkennzeichen des Leitsystems und seine Bedeutung im Modellkontext.

    ``confirmed`` ist unabhaengig von ``exported``: ein Kanal kann im Export vorhanden
    und trotzdem in seiner *Bedeutung* unbestaetigt sein (siehe die obere Temperatur-
    Trias, NA3) -- und umgekehrt kann ein Kanal aus dem Screenshot in seiner Bedeutung
    eindeutig sein, obwohl er nicht exportiert wurde (z. B. ``PDIRAH-320404``, dessen
    Screenshot-Wert mit einer unabhaengigen Leitsystem-Anzeige uebereinstimmt).
    """

    tag: str
    unit: str
    description: str
    exported: bool
    confirmed: bool
    note: str = ""


GRANULATOR_D3204_CHANNELS: tuple[PlantChannel, ...] = (
    # -- Zuluft / Umgebung ---------------------------------------------------------
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
            "Messwerte plausibel fuer die Exportdaten: P1 (26.03.) 0.9-7.6 degC, "
            "P2 (11.05.) 7.7-15.3 degC -- konsistent mit den Dateidaten (frueres "
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
    # -- Betttemperatur --------------------------------------------------------------
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
    # -- Hoehere Position (NA3, unbestaetigt) -----------------------------------------
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
            "(P1 +0.89 K, P2 +3.10 K im Mittel) -- kein Einzelfall. Anders als "
            "die eng zusammenliegende Bett- und Zuluft-Trias (Streuung < 0.4 K "
            "zwischen den drei Sensoren) streuen 417/418/419 selbst deutlich "
            "untereinander (P2: 45.6/47.4/46.1 degC, Spanne 1.8 K) -- Hinweis auf "
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
    # -- Abluft ------------------------------------------------------------------------
    PlantChannel(
        "TIRSAH-320420",
        "degC",
        "Abluft am Granulatoraustritt, vor dem Filter",
        exported=True,
        confirmed=True,
        note="Position im Screenshot am Apparatkopf, vor 'Filter'/V 3246 bestaetigt.",
    ),
    # -- Zusaetzliche, nicht im NA2-Ausgangsstand erwartete Kanaele ---------------------
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
    # -- Ventilatoren --------------------------------------------------------------------
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
    # -- Spruehsystem ----------------------------------------------------------------------
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
        note="Bei P1 inaktiv (Messwert nahe 0, leicht negativ -- Nullpunktrauschen).",
    ),
    PlantChannel(
        "FIRSAL-320565",
        "m3/h",
        "Zerstaeuberluft, Duese 6/6",
        exported=True,
        confirmed=True,
        note="Bei P1 inaktiv (Messwert nahe 0, leicht negativ -- Nullpunktrauschen).",
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
    # -- Druckverluste / Bettmasse -------------------------------------------------------
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
    # -- Waermerueckgewinnung --------------------------------------------------------------
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
    """Nachschlagetabelle Tag -> :class:`PlantChannel`."""
    return {channel.tag: channel for channel in GRANULATOR_D3204_CHANNELS}


def process_air_flow_tag() -> str:
    """Tag der Prozessluftmenge."""
    return "FIRCAL-322411"


def ambient_temperature_tag() -> str:
    """Tag der Ansaugluft-/Umgebungstemperatur."""
    return "TIR-322412"


def inlet_temperature_tags() -> tuple[str, str, str]:
    """Tags der Zulufttemperatur unter dem Anstroemboden (drei Sensoren)."""
    return ("TRZSAH-320408", "TRZSAH-320409", "TRZSAH-320410")


def bed_temperature_tags() -> tuple[str, str, str]:
    """Betttemperatur -- siehe :data:`GRANULATOR_D3204_CHANNELS`, bestaetigt via Screenshot."""
    return ("TIRSAH-320414", "TIRSAH-320415", "TIRSAH-320416")


def upper_position_temperature_tags() -> tuple[str, str, str]:
    """Hoehere Position, NICHT als Betttemperatur bestaetigt -- siehe NA3-Notiz."""
    return ("TRZSAH-320417", "TRZSAH-320418", "TRZSAH-320419")


def outlet_temperature_tag() -> str:
    """Tag der Ablufttemperatur am Granulatoraustritt, vor dem Filter."""
    return "TIRSAH-320420"


def spray_rate_tag() -> str:
    """Tag der Spruehrate."""
    return "FIR-325160"


def nozzle_atomising_air_tags() -> tuple[str, str, str, str, str, str]:
    """Tags der Zerstaeuberluft je Duese (sechs Duesen)."""
    return (
        "FIRSAL-320515",
        "FIRSAL-320525",
        "FIRSAL-320535",
        "FIRSAL-320545",
        "FIRSAL-320555",
        "FIRSAL-320565",
    )
