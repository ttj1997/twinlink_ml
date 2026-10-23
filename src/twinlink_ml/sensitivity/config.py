"""Settings of the Sobol study, loaded from ``conf/sensitivity/sobol_steady.yaml``.

Every range is a placeholder with its reason in the YAML; the dataclasses only give the values
names and types. Inputs are sampled on the unit cube and mapped by :meth:`InputSpec.value`
(uniform on ``[low, high]``, or uniform in the logarithm for ``scale: log``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

__all__ = [
    "Criteria",
    "DesignSpec",
    "InputSpec",
    "SensitivityStudyConfig",
    "load_sensitivity_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "sensitivity" / "sobol_steady.yaml"

Group = Literal["epistemic", "operative"]
Scale = Literal["linear", "log"]


@dataclass(frozen=True, slots=True)
class InputSpec:
    """One uncertain input: group, nominal value and placeholder range."""

    name: str
    group: Group
    nominal: float
    low: float
    high: float
    unit: str
    rationale: str
    scale: Scale = "linear"
    question: str | None = None
    hard: tuple[float, float] | None = None
    """Physical limits of the input (used when a range is widened, design R)."""

    def __post_init__(self) -> None:
        if self.group not in ("epistemic", "operative"):
            raise ValueError(f"{self.name}: unknown group {self.group!r}")
        if not self.low < self.high:
            raise ValueError(f"{self.name}: low must be below high")
        if self.scale == "log" and self.low <= 0.0:
            raise ValueError(f"{self.name}: a log scale needs a positive range")

    def value(self, unit: np.ndarray) -> np.ndarray:
        """Map unit-cube coordinates to values (uniform, or log-uniform)."""
        u = np.asarray(unit, dtype=np.float64)
        if self.scale == "log":
            lo, hi = math.log(self.low), math.log(self.high)
            return np.exp(lo + u * (hi - lo))
        return self.low + u * (self.high - self.low)

    def widened(self, factor: float) -> InputSpec:
        """The range widened by ``factor`` about its centre.

        In the logarithm for ``scale: log``; clipped to the hard bounds :attr:`hard` if given.
        """
        if self.scale == "log":
            lo, hi = math.log(self.low), math.log(self.high)
        else:
            lo, hi = self.low, self.high
        centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
        low, high = centre - half, centre + half
        if self.scale == "log":
            low, high = math.exp(low), math.exp(high)
        if self.hard is not None:
            low, high = max(low, self.hard[0]), min(high, self.hard[1])
        return replace(self, low=low, high=high)


@dataclass(frozen=True, slots=True)
class Criteria:
    """Pre-registered verdict thresholds (docs/26 section 4)."""

    dominant_total_index: float
    max_ci_width: float
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    robustness_top_k: int


@dataclass(frozen=True, slots=True)
class DesignSpec:
    """One Saltelli design: base sample ``2**base_log2`` (per model) and seed."""

    name: str
    base_log2: int
    seed: int
    base_log2_p2: int | None = None

    def base_log2_for(self, model: str) -> int:
        if model == "p2" and self.base_log2_p2 is not None:
            return self.base_log2_p2
        return self.base_log2


@dataclass(frozen=True, slots=True)
class SensitivityStudyConfig:
    """Criteria, designs, the two models' settings and inputs."""

    version: int
    criteria: Criteria
    designs: dict[str, DesignSpec]
    inputs: dict[str, tuple[InputSpec, ...]]
    settings: dict[str, dict[str, Any]]
    questions: dict[str, str]
    raw: dict[str, Any] = field(compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Hash of the YAML content (sorted JSON), recorded in every artifact."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    def group(self, model: str, group: Group) -> tuple[InputSpec, ...]:
        return tuple(s for s in self.inputs[model] if s.group == group)


def _inputs(raw: dict[str, Any]) -> tuple[InputSpec, ...]:
    out = []
    for name, spec in raw.items():
        nominal = spec.get("nominal")
        out.append(
            InputSpec(
                name=name,
                group=spec["group"],
                nominal=math.nan if nominal is None else float(nominal),
                low=float(spec["low"]),
                high=float(spec["high"]),
                unit=str(spec["unit"]),
                rationale=" ".join(str(spec["rationale"]).split()),
                scale=spec.get("scale", "linear"),
                question=spec.get("question"),
                hard=None
                if spec.get("hard") is None
                else (float(spec["hard"][0]), float(spec["hard"][1])),
            )
        )
    return tuple(out)


def load_sensitivity_config(path: Path | str = DEFAULT_CONFIG_PATH) -> SensitivityStudyConfig:
    """Read the study YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    crit = raw["criteria"]
    designs = {
        name: DesignSpec(
            name=name,
            base_log2=int(d["base_log2"]),
            seed=int(d["seed"]),
            base_log2_p2=None if d.get("base_log2_p2") is None else int(d["base_log2_p2"]),
        )
        for name, d in raw["designs"].items()
    }
    models = ("p1", "p2")
    return SensitivityStudyConfig(
        version=int(raw["version"]),
        criteria=Criteria(
            dominant_total_index=float(crit["dominant_total_index"]),
            max_ci_width=float(crit["max_ci_width"]),
            confidence_level=float(crit["confidence_level"]),
            bootstrap_resamples=int(crit["bootstrap_resamples"]),
            bootstrap_seed=int(crit["bootstrap_seed"]),
            robustness_top_k=int(crit["robustness_top_k"]),
        ),
        designs=designs,
        inputs={m: _inputs(raw[m]["inputs"]) for m in models},
        settings={m: {k: v for k, v in raw[m].items() if k != "inputs"} for m in models},
        questions={k: " ".join(str(v).split()) for k, v in raw["questions"].items()},
        raw=raw,
    )
