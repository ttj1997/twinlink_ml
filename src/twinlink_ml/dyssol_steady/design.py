"""Box over the varied inputs, Sobol design, nested training subsets.

Five inputs are varied per line (decision of the project lead, 2026-11-19): dry-gas flow, dryer
inlet temperature, spray flow, solid fraction of the spray, medium flow of the heat recovery that
cools the line's exhaust. Everything else stays at the value stored in the source flowsheet.

The box is NOT in the repository: it is read from a local YAML (template
``conf/dyssol_steady/box_local.yaml.example``). An entry is either ``[lower, upper]`` in the unit
of the input or ``{relative: r}``, i.e. ``nominal x (1 -+ r)`` around the value of the source
file. Records carry the SHA-256 of the resolved box only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import qmc

from .reference import LineInputs

__all__ = [
    "VARIED",
    "Box",
    "apply_point",
    "load_box",
    "resolve_box",
    "sobol_points",
    "training_subset",
]

VARIED: tuple[str, ...] = (
    "dry_gas_kg_per_h",
    "dryer_inlet_celsius",
    "spray_kg_per_h",
    "solid_fraction",
    "medium_kg_per_h",
)
"""Varied inputs of one line, in the column order of every design array."""


@dataclass(frozen=True)
class Box:
    """Axis-aligned box; columns are ``line<k>/<input>`` for every line and input of ``VARIED``."""

    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if not len(self.names) == len(self.lower) == len(self.upper):
            raise ValueError("a box needs one interval per input")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise ValueError("box with lower >= upper")

    @property
    def dimension(self) -> int:
        """Number of varied inputs."""
        return len(self.names)

    def scale(self, unit: np.ndarray) -> np.ndarray:
        """Map points of the unit cube into the box."""
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        return np.asarray(lo + np.asarray(unit, dtype=np.float64) * (hi - lo))

    def to_unit_symmetric(self, x: np.ndarray) -> np.ndarray:
        """Map the box to ``[-1, 1]`` per input (the fixed a-priori input scaling)."""
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        return np.asarray(2.0 * (np.asarray(x, dtype=np.float64) - lo) / (hi - lo) - 1.0)

    def contains(self, x: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """Row-wise membership."""
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        span = hi - lo
        x = np.asarray(x, dtype=np.float64)
        return np.asarray(np.all((x >= lo - tol * span) & (x <= hi + tol * span), axis=1))

    def sha256(self) -> str:
        """Digest of the resolved box (the only trace of it in committed records)."""
        payload = json.dumps([self.names, self.lower, self.upper], separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_box(entries: Sequence[Mapping[str, Any]], nominal: Sequence[LineInputs]) -> Box:
    """Box from one mapping per line (``input -> [lower, upper]`` or ``{relative: r}``)."""
    if len(entries) != len(nominal):
        raise ValueError("the box needs one entry per line of the flowsheet")
    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    for k, (entry, line) in enumerate(zip(entries, nominal, strict=True), start=1):
        if set(entry) != set(VARIED):
            raise ValueError(f"line {k}: the box must name exactly {VARIED}")
        for name in VARIED:
            spec = entry[name]
            if isinstance(spec, Mapping):
                r = float(spec["relative"])
                centre = float(getattr(line, name))
                lo, hi = centre * (1.0 - r), centre * (1.0 + r)
            else:
                lo, hi = float(spec[0]), float(spec[1])
            names.append(f"line{k}/{name}")
            lower.append(lo)
            upper.append(hi)
    return Box(tuple(names), tuple(lower), tuple(upper))


def load_box(path: Path | str, case: str, nominal: Sequence[LineInputs]) -> Box:
    """Read the box of ``case`` from the local YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if case not in raw.get("cases", {}):
        raise KeyError(f"case {case!r} not in {path}")
    return resolve_box(raw["cases"][case]["lines"], nominal)


def sobol_points(box: Box, n: int, seed: int) -> np.ndarray:
    """``n`` scrambled Sobol points in the box (``n`` a power of two)."""
    if n <= 0 or n & (n - 1):
        raise ValueError("n must be a power of two")
    sampler = qmc.Sobol(d=box.dimension, scramble=True, seed=seed)
    return box.scale(sampler.random_base2(int(np.log2(n))))


def training_subset(pool_size: int, n: int, seed: int) -> np.ndarray:
    """Indices of the training set of size ``n`` for ``seed``: a prefix of one permutation.

    The subsets of one seed are nested over ``n``; different seeds draw different permutations.
    """
    if n > pool_size:
        raise ValueError("training set larger than the pool")
    return np.asarray(np.random.default_rng(seed).permutation(pool_size)[:n])


def apply_point(nominal: Sequence[LineInputs], row: np.ndarray) -> tuple[LineInputs, ...]:
    """Line inputs for one design row (columns as in :class:`Box`)."""
    width = len(VARIED)
    if len(row) != width * len(nominal):
        raise ValueError("design row does not match the number of lines")
    return tuple(
        replace(
            line, **dict(zip(VARIED, map(float, row[k * width : (k + 1) * width]), strict=True))
        )
        for k, line in enumerate(nominal)
    )
