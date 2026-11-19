"""Materials database of Dyssol: the one added constant and the properties the rebuild uses.

The stationary dryer unit reads the normal boiling point of liquid water from the materials
database; the database shipped with Dyssol 1.6.0 has no entry for it, and the unit refuses to
start. The unit's author set 373.15 K in the editor and changed nothing else (statement of
2026-11-19, checked: three of seven delivered flowsheets are reproduced to the last stored
digit with exactly this change). :func:`patch_materials_database` applies the same change to a
copy; the installed file is never written.

:func:`read_properties` parses the few entries the rebuild (:mod:`.reference`) needs. A property
without an entry takes the default of Dyssol (``DefinesMDB.h`` of the SDK): molar mass
0.0180154 kg/mol, heat capacity 1000 J/(kg K).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "COMPOUND_NAMES",
    "DEFAULT_HEAT_CAPACITY",
    "DEFAULT_MOLAR_MASS",
    "WATER_BOILING_POINT_K",
    "DyssolProperties",
    "compound_names",
    "patch_materials_database",
    "read_properties",
    "sha256_of",
]

WATER_BOILING_POINT_K = 373.15
DEFAULT_MOLAR_MASS = 0.0180154
"""Default of a missing molar mass [kg/mol] (``DefinesMDB.h``)."""
DEFAULT_HEAT_CAPACITY = 1000.0
"""Default of a missing heat capacity [J/(kg K)] (``DefinesMDB.h``)."""

COMPOUND_NAMES = {"gas": "Air", "vapour": "H2O_gas", "liquid": "H2O", "solid": "Al2O3"}
"""Compound names the units ask for (hard-coded in their ``Initialize``)."""

_KEY, _NAME, _CONST, _DEPENDENT = "21", "22", "26", "27"
_HEAT_OF_VAPORIZATION, _MOLAR_MASS, _BOILING_POINT = 115, 121, 122
_HEAT_CAPACITY, _ENTHALPY = 207, 208
_CONSTANT, _LINEAR, _SHOMATE_ENTHALPY = 2, 3, 8


def sha256_of(path: Path | str) -> str:
    """Hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _blocks(text: str) -> dict[str, list[list[str]]]:
    """Compound name -> token lists of its lines (comments after ``$`` removed)."""
    blocks: dict[str, list[list[str]]] = {}
    current: list[list[str]] | None = None
    pending_key: list[str] | None = None
    for raw in text.splitlines():
        tokens = raw.split("$", 1)[0].split()
        if not tokens:
            continue
        if tokens[0] == _KEY and len(tokens) == 2:
            pending_key = tokens
            current = None
        elif tokens[0] == _NAME and pending_key is not None:
            current = [pending_key, tokens]
            blocks[" ".join(tokens[1:])] = current
            pending_key = None
        elif current is not None:
            current.append(tokens)
    return blocks


def compound_names(path: Path | str) -> dict[str, str]:
    """Compound key -> name of a materials database."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {lines[0][1]: name for name, lines in _blocks(text).items()}


def patch_materials_database(
    source: Path | str, target: Path | str, boiling_point_k: float = WATER_BOILING_POINT_K
) -> dict[str, str]:
    """Write a copy of ``source`` with the normal boiling point of liquid water set.

    Returns the digests of both files. Fails if the compound is missing or already has a boiling
    point (then the copy would not be the documented change).
    """
    source, target = Path(source), Path(target)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    liquid = COMPOUND_NAMES["liquid"]
    out: list[str] = []
    inside = done = False
    for line in lines:
        tokens = line.split("$", 1)[0].split()
        if tokens[:1] == [_NAME]:
            inside = " ".join(tokens[1:]) == liquid
            out.append(line)
            if inside:
                out.append(
                    f"{_CONST} {_BOILING_POINT} {boiling_point_k:g} \t$ Normal boiling point [K]"
                )
                done = True
            continue
        if inside and tokens[:2] == [_CONST, str(_BOILING_POINT)]:
            raise ValueError(f"{source} already has a boiling point for {liquid!r}")
        if tokens[:1] == [_KEY]:
            inside = False
        out.append(line)
    if not done:
        raise ValueError(f"compound {liquid!r} not found in {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {"source_sha256": sha256_of(source), "patched_sha256": sha256_of(target)}


@dataclass(frozen=True)
class DyssolProperties:
    """Properties of the four compounds as the units read them (SI)."""

    cp_gas: float
    cp_vapour: float
    cp_liquid: float
    cp_solid: float
    heat_of_vaporization_j_per_kg: float
    boiling_point_k: float
    gas_enthalpy_linear: tuple[float, float]
    """``h = a T + c`` of dry air [J/kg] (correlation *Linear*, the pressure term is unused)."""
    vapour_enthalpy_shomate: tuple[float, ...]
    """Seven constants of *Shomate standard enthalpy* of the vapour [J/kg]."""
    gas_enthalpy_range_k: tuple[float, float]
    vapour_enthalpy_range_k: tuple[float, float]
    """Temperature range of each correlation. Outside it Dyssol evaluates the correlation at the
    nearest bound (observed 2026-11-19: the heater duty of a feed below the lower bound of the
    vapour correlation is reproduced only with this clamp)."""

    def as_dict(self) -> dict[str, object]:
        """Plain mapping for records."""
        return {
            "cp_gas": self.cp_gas,
            "cp_vapour": self.cp_vapour,
            "cp_liquid": self.cp_liquid,
            "cp_solid": self.cp_solid,
            "heat_of_vaporization_j_per_kg": self.heat_of_vaporization_j_per_kg,
            "boiling_point_k": self.boiling_point_k,
            "gas_enthalpy_linear": list(self.gas_enthalpy_linear),
            "vapour_enthalpy_shomate": list(self.vapour_enthalpy_shomate),
            "gas_enthalpy_range_k": list(self.gas_enthalpy_range_k),
            "vapour_enthalpy_range_k": list(self.vapour_enthalpy_range_k),
        }


def _const(lines: list[list[str]], prop: int) -> float | None:
    for t in lines:
        if t[0] == _CONST and int(t[1]) == prop:
            return float(t[2])
    return None


def _dependent(
    lines: list[list[str]], prop: int
) -> tuple[int, tuple[float, ...], tuple[float, float]] | None:
    """Type, constants and temperature range of the FIRST correlation of a property.

    Line format: ``27 id type T1 T2 P1 P2 constants...``.
    """
    for t in lines:
        if t[0] == _DEPENDENT and int(t[1]) == prop:
            return int(t[2]), tuple(float(v) for v in t[7:]), (float(t[3]), float(t[4]))
    return None


def _heat_capacity(lines: list[list[str]], name: str) -> float:
    entry = _dependent(lines, _HEAT_CAPACITY)
    if entry is None:
        return DEFAULT_HEAT_CAPACITY
    kind, constants, _ = entry
    if kind != _CONSTANT:
        raise ValueError(f"heat capacity of {name!r} is not a constant (correlation type {kind})")
    return constants[0]


def read_properties(path: Path | str) -> DyssolProperties:
    """Read the properties of the four compounds from a (patched) materials database."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = _blocks(text)
    missing = [n for n in COMPOUND_NAMES.values() if n not in blocks]
    if missing:
        raise ValueError(f"compounds {missing} not in {path}")
    by_role: Mapping[str, list[list[str]]] = {r: blocks[n] for r, n in COMPOUND_NAMES.items()}
    liquid = by_role["liquid"]
    boiling = _const(liquid, _BOILING_POINT)
    h_vap = _const(liquid, _HEAT_OF_VAPORIZATION)
    if boiling is None or h_vap is None:
        raise ValueError("liquid water needs a boiling point and a heat of vaporization")
    molar = _const(liquid, _MOLAR_MASS) or DEFAULT_MOLAR_MASS
    gas_h = _dependent(by_role["gas"], _ENTHALPY)
    vap_h = _dependent(by_role["vapour"], _ENTHALPY)
    if gas_h is None or gas_h[0] != _LINEAR:
        raise ValueError("enthalpy of the dry gas is expected as a linear correlation")
    if vap_h is None or vap_h[0] != _SHOMATE_ENTHALPY:
        raise ValueError("enthalpy of the vapour is expected as Shomate standard enthalpy")
    return DyssolProperties(
        cp_gas=_heat_capacity(by_role["gas"], "gas"),
        cp_vapour=_heat_capacity(by_role["vapour"], "vapour"),
        cp_liquid=_heat_capacity(liquid, "liquid"),
        cp_solid=_heat_capacity(by_role["solid"], "solid"),
        heat_of_vaporization_j_per_kg=h_vap / molar,
        boiling_point_k=boiling,
        gas_enthalpy_linear=(gas_h[1][0], gas_h[1][2]),
        vapour_enthalpy_shomate=vap_h[1][:7],
        gas_enthalpy_range_k=gas_h[2],
        vapour_enthalpy_range_k=vap_h[2],
    )
