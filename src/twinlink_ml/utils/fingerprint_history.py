r"""Aequivalenz der ``greybox_gab_placeholder.yaml``-Fingerprints über den Namens-Schnitt.

Grund: die Trennung des Repos von seiner Historie (siehe ``docs/ARCHIV.md``, 2026-09-25) ersetzte
Partnernamen in ``provenance.open_todos`` durch Pseudonyme. Diese Freitextliste war Teil
der gehashten Config, verschob also ``ProcessConfig.fingerprint`` -- obwohl an keiner
einzigen Zahl etwas geaendert wurde. Seit diesem Schnitt schliesst
:func:`twinlink_ml.simulation.config.without_documentary_provenance` ``open_todos`` (und
``description``) aus dem Hash aus; die drei Paare unten sind deshalb die LETZTEN, die
dieser Schnitt erzeugt -- kuenftige Aenderungen an dokumentarischen Feldern bewegen den
Fingerprint nicht mehr.

Aequivalenznachweis (einmalig, siehe Audit-Konversation 2026-09-25): fuer jedes Paar wurde
``config_fingerprint(without_documentary_provenance(...))`` sowohl ueber die unveraenderte
Archiv-Config (mit Partnernamen, aus dem Vorgaengerrepo) als auch ueber die bereinigte
Config berechnet -- beide Seiten lieferten fuer jede der drei Varianten (Projektion aus/an,
und ohne Ueberschreibung "wie geladen") denselben Wert. Das belegt: die Config-Paare
bezeichnen dieselbe numerische Konfiguration, nur die Dokumentation unterscheidet sich.

Diese Tabelle ist GESCHLOSSEN. Sie enthaelt ausschliesslich die drei durch den obigen
Nachweis gedeckten Paare -- eine kuenftige Aenderung, die den Fingerprint bewegt, bekommt
KEINEN neuen Eintrag hier. Eine solche Aenderung ist echt (eine Zahl hat sich geaendert)
und soll von jedem Guard, der :func:`fingerprints_equivalent` benutzt, gemeldet werden wie
jede andere unerwartete Drift auch. Siehe ``tests/test_fingerprint_history.py``, der genau
diese drei Paare erzwingt -- nicht mehr, nicht weniger.
"""

from __future__ import annotations

PRE_CUT_FINGERPRINT_EQUIVALENTS: dict[str, str] = {
    # alt (vor dem Schnitt) -> neu (ab without_documentary_provenance)
    "abbe8325f1326d80": "2acd06f38963c3af",  # greybox_gab_placeholder.yaml, wie geladen
    "078572b3a7a0fb45": "6ecedc6bd993fb3d",  # ... sample_time_s=25.0, projection AUS
    "2d2679eeed716577": "bddca37f716f5cfc",  # ... sample_time_s=25.0, projection AN
}
"""Nur lesen, nie erweitern -- siehe Moduldocstring."""


def fingerprints_equivalent(a: str, b: str) -> bool:
    """``True``, wenn ``a``/``b`` identisch sind oder laut der Tabelle aequivalent sind.

    "Aequivalent" heisst: dieselbe numerische Konfiguration, eine Seite vor, die andere
    nach dem Pseudonymisierungs-Schnitt (siehe Moduldocstring). Neue Laeufe erzeugen immer
    den neuen Fingerprint -- die Tabelle wird nur zum ANERKENNEN alter, eingefrorener
    Artefaktwerte benutzt, nie zum Schreiben. Symmetrisch: die Reihenfolge von ``a``/``b``
    spielt keine Rolle.
    """
    if a == b:
        return True
    return (
        PRE_CUT_FINGERPRINT_EQUIVALENTS.get(a) == b or PRE_CUT_FINGERPRINT_EQUIVALENTS.get(b) == a
    )
