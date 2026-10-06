"""Read formulas and cached values of single cells of ``Model_v4.xlsx`` (needs ``openpyxl``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["CellInfo", "read_cells"]


@dataclass(frozen=True)
class CellInfo:
    """One cell: formula text (``None`` for a constant) and the cached value Excel stored."""

    address: str
    formula: str | None
    value: object


def read_cells(path: str | Path, addresses: list[str]) -> dict[str, CellInfo]:
    """Read cells given as ``"Sheet!A1"``; empty cells have ``value None``."""
    import openpyxl

    formulas = openpyxl.load_workbook(Path(path))
    cached = openpyxl.load_workbook(Path(path), data_only=True)
    out: dict[str, CellInfo] = {}
    for address in addresses:
        sheet, cell = address.split("!")
        raw = formulas[sheet][cell].value
        is_formula = isinstance(raw, str) and raw.startswith("=")
        out[address] = CellInfo(
            address=address,
            formula=raw if is_formula else None,
            value=cached[sheet][cell].value if is_formula else raw,
        )
    return out
