"""Settings of the case-ranking study, loaded from ``conf/case_ranking/case_ranking.yaml``.

Every value is a study setting or a placeholder (see the YAML header). The dataclasses are frozen;
``raw`` keeps the parsed YAML for the fingerprint and the record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .evaluator import FlowOptimiser
from .space import INPUT_NAMES, Box
from .workbook import WORKBOOK_POINT

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "SET_NAMES",
    "CaseRankingConfig",
    "load_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "case_ranking" / "case_ranking.yaml"
SET_NAMES: tuple[str, ...] = ("full", "workbook_neighbourhood", "plant_like", "rule_test")
"""The Sobol sets; ``rule_test`` is an own stream over the full box (test set of the rule)."""
VARIANTS: tuple[str, ...] = ("fixed", "optimised")


@dataclass(frozen=True)
class CaseRankingConfig:
    """The parsed study settings (see the YAML for the meaning and source of every value)."""

    box: Box
    regions: dict[str, Box]
    seed: int
    points: dict[str, dict[str, int]]
    """``variant -> set -> number of points`` (prefixes of the set's stream)."""
    convergence_prefixes: tuple[int, ...]
    one_d_points: int
    one_d_points_optimised: int
    two_d_points: int
    two_d_pairs: tuple[tuple[str, str], ...]
    max_exhaust_relative_humidity: float
    partner_medium_boiling_celsius: float
    tie_exact_kg_per_h: float
    tie_practical_fraction: float
    tie_practical_sensitivity: tuple[float, ...]
    optimiser: FlowOptimiser
    rule_max_depth: int
    rule_min_samples_leaf: int
    rule_accuracy_reading: float
    chunk_points: int
    max_workers: int
    dry_run_points_per_chunk: int
    raw: dict[str, Any] = field(repr=False, compare=False)

    def set_box(self, name: str) -> Box:
        return self.box if name in ("full", "rule_test") else self.regions[name]

    def set_seed(self, name: str) -> int:
        return self.seed + {"full": 0, "workbook_neighbourhood": 1, "plant_like": 2}.get(name, 100)

    @property
    def fingerprint(self) -> str:
        text = json.dumps(self.raw, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> CaseRankingConfig:
    """Read and check the study YAML."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    box = Box.from_mapping(raw["box"])
    if not box.contains(WORKBOOK_POINT):
        raise ValueError("the workbook point must lie in the box")
    reg = raw["regions"]
    regions = {
        "workbook_neighbourhood": box.neighbourhood(
            WORKBOOK_POINT, float(reg["workbook_neighbourhood"]["fraction_of_width"])
        ),
        "plant_like": box.replace(**reg["plant_like"]),
    }
    for name, r in regions.items():
        if not all(
            b.low <= i.low and i.high <= b.high
            for b, i in zip(box.intervals, r.intervals, strict=True)
        ):
            raise ValueError(f"region {name} leaves the box")
    d = raw["designs"]
    points = {v: {s: int(d[v][s]) for s in SET_NAMES} for v in VARIANTS}
    for s in SET_NAMES:
        if points["optimised"][s] > points["fixed"][s]:
            raise ValueError("the optimised variant uses prefixes of the fixed sets")
    for n in (n for v in points.values() for n in v.values()):
        if n & (n - 1):
            raise ValueError(f"{n} points: use powers of two (Sobol balance)")
    pairs = tuple((str(a), str(b)) for a, b in raw["cuts"]["two_d_pairs"])
    for a, b in pairs:
        if a not in INPUT_NAMES or b not in INPUT_NAMES:
            raise KeyError((a, b))
    o = raw["optimiser"]
    ties = raw["ties"]
    return CaseRankingConfig(
        box=box,
        regions=regions,
        seed=int(d["seed"]),
        points=points,
        convergence_prefixes=tuple(int(n) for n in d["convergence_prefixes"]),
        one_d_points=int(raw["cuts"]["one_d_points"]),
        one_d_points_optimised=int(raw["cuts"]["one_d_points_optimised"]),
        two_d_points=int(raw["cuts"]["two_d_points"]),
        two_d_pairs=pairs,
        max_exhaust_relative_humidity=float(raw["admissibility"]["max_exhaust_relative_humidity"]),
        partner_medium_boiling_celsius=float(
            raw["admissibility"]["partner_medium_boiling_celsius"]
        ),
        tie_exact_kg_per_h=float(ties["exact_kg_per_h"]),
        tie_practical_fraction=float(ties["practical_fraction_of_saving"]),
        tie_practical_sensitivity=tuple(float(v) for v in ties["practical_sensitivity"]),
        optimiser=FlowOptimiser(
            bounds_kg_per_h=(float(o["bounds_kg_per_h"][0]), float(o["bounds_kg_per_h"][1])),
            grid=int(o["grid"]),
            starts=int(o["starts"]),
            max_evaluations_per_start=int(o["max_evaluations_per_start"]),
        ),
        rule_max_depth=int(raw["rule"]["max_depth"]),
        rule_min_samples_leaf=int(raw["rule"]["min_samples_leaf"]),
        rule_accuracy_reading=float(raw["rule"]["accuracy_reading"]),
        chunk_points=int(raw["run"]["chunk_points"]),
        max_workers=int(raw["run"]["max_workers"]),
        dry_run_points_per_chunk=int(raw["run"]["dry_run_points_per_chunk"]),
        raw=raw,
    )
