r"""Einlesen der Anlagenexporte des Anwendungspartners.

Liest ausschliesslich -- schreibt nichts. Der Dateipfad ist ein Funktionsparameter,
keine Konstante in diesem Modul: die Rohdaten (Kundendaten, siehe NA1/
``docs/04_stand_ap1.md``) bleiben ausserhalb des Repos, Aufrufer uebergeben den Pfad
(z. B. aus einer Umgebungsvariable im Treiberskript).

``openpyxl`` ist hierfuer eine echte Laufzeitabhaengigkeit (``pyproject.toml``,
Extra ``plant-data``), nicht optional wie ``torch``/``sklearn``: ohne sie laesst sich
kein Anlagenexport lesen. Trotzdem lazy importiert, damit der Rest des Pakets (Grey-Box-
Modell, Training) ohne dieses Extra installierbar bleibt.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PlantTimeSeries", "load_plant_export"]

# Beschriftungsfehler der Quelle: diese Tags tragen in Zeile 2 der Exportdatei die
# Einheit "°C", sind aber Durchfluesse (m3/h) -- siehe channels.py-Moduldocstring.
_UNIT_OVERRIDES: dict[str, str] = {
    "FIRAL-323805": "m3/h",
    "FIRAL-323604": "m3/h",
}


@dataclass(frozen=True, slots=True)
class PlantTimeSeries:
    """Eine eingelesene 24h-Exportdatei, Spalten nach Messstellenkennzeichen."""

    timestamps: tuple[dt.datetime, ...]
    units: dict[str, str]
    columns: dict[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        lengths = {len(values) for values in self.columns.values()}
        if lengths and lengths != {len(self.timestamps)}:
            raise ValueError(
                f"Spaltenlaengen {lengths} passen nicht zu {len(self.timestamps)} Zeitstempeln"
            )


def load_plant_export(path: str | Path) -> PlantTimeSeries:
    """Liest einen 24h-Anlagenexport (Kopfzeile Tag, zweite Zeile Einheit, ab Zeile 3 Daten).

    :param path: Pfad zur ``.xlsx``-Datei, ausserhalb des Repos (siehe Moduldocstring).
    :raises ValueError: wenn die Kopfzeile keine Spalte "Time" enthaelt.
    """
    import openpyxl

    workbook = openpyxl.load_workbook(Path(path), read_only=True, data_only=True)
    worksheet = workbook[workbook.sheetnames[0]]
    rows = worksheet.iter_rows(values_only=True)
    header = [name for name in next(rows) if name is not None]
    if not header or header[0] != "Time":
        raise ValueError(f"Erwarte Spalte 'Time' als erste Kopfzeile, gefunden: {header[:1]!r}")
    unit_row = next(rows)
    units = {
        name: _UNIT_OVERRIDES.get(name, str(unit).strip("[]") if unit else "")
        for name, unit in zip(header[1:], unit_row[1 : len(header)], strict=True)
    }

    timestamps: list[dt.datetime] = []
    columns: dict[str, list[float]] = {name: [] for name in header[1:]}
    for row in rows:
        if row[0] is None:
            continue
        timestamps.append(row[0])
        for name, value in zip(header[1:], row[1 : len(header)], strict=True):
            columns[name].append(float(value))

    workbook.close()
    return PlantTimeSeries(
        timestamps=tuple(timestamps),
        units=units,
        columns={name: tuple(values) for name, values in columns.items()},
    )
