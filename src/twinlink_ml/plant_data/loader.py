r"""Reading of the application partner's plant exports.

Reads only -- writes nothing. The file path is a function parameter,
not a constant in this module: the raw data (customer data, see NA1/
``docs/04_stand_ap1.md``) stay outside the repo, callers pass the path
(e.g. from an environment variable in the driver script).

``openpyxl`` is a real runtime dependency for this (``pyproject.toml``,
extra ``plant-data``), not optional like ``torch``/``sklearn``: without it no
plant export can be read. Nevertheless imported lazily, so that the rest of the package
(grey-box model, training) remains installable without this extra.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PlantTimeSeries", "load_plant_export"]

# Labelling error of the source: these tags carry the unit "°C" in row 2 of the export
# file, but are flows (m3/h) -- see the channels.py module docstring.
_UNIT_OVERRIDES: dict[str, str] = {
    "FIRAL-323805": "m3/h",
    "FIRAL-323604": "m3/h",
}


@dataclass(frozen=True, slots=True)
class PlantTimeSeries:
    """One read-in 24h export file, columns by measuring-point tag."""

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
    """Reads a 24h plant export (header row tag, second row unit, data from row 3).

    :param path: path to the ``.xlsx`` file, outside the repo (see module docstring).
    :raises ValueError: if the header row contains no column "Time".
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
