"""Settings of the surrogate-RTO study, loaded from ``conf/surrogate_opt/surrogate_rto.yaml``.

Every value is a study setting or a placeholder (see the YAML header). :func:`build_problems`
turns the two problem blocks into :class:`~.problem.ProblemFamily` objects. The truth of the
hidden P2 parameter is read from the headroom config named in the YAML (``truth_from``), so the
shift is the one of the docs/24 robustness band and cannot drift from it.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from twinlink_ml.partner_v4.inputs import SiMaterial

from .problem import Box, ParameterSpec, ProblemFamily

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "SurrogateStudyConfig",
    "build_problems",
    "load_study_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "surrogate_opt" / "surrogate_rto.yaml"


@dataclass(frozen=True)
class SurrogateStudyConfig:
    """The YAML, typed where the study code reads it; ``raw`` for the problem blocks."""

    version: int
    seeds: tuple[int, ...]
    budgets: tuple[int, ...]
    kappas: tuple[float, ...]
    test_count: int
    test_seed: int
    adaptive_initial: int
    adaptive_max_batch: int
    adaptive_pool_seed_offset: int
    raw: dict[str, Any] = field(compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Hash of the YAML content (sorted JSON), recorded in every artifact."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    def optimizer(self, kind: str) -> dict[str, Any]:
        return dict(self.raw["optimizer"][kind])

    @property
    def feasibility_tolerance(self) -> float:
        return float(self.raw["optimizer"]["feasibility_tolerance"])

    def model(self, key: str) -> dict[str, Any]:
        return dict(self.raw["models"][key])


def load_study_config(path: Path | str = DEFAULT_CONFIG_PATH) -> SurrogateStudyConfig:
    """Read the study YAML (budgets must be powers of two)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    budgets = tuple(int(n) for n in raw["budgets"])
    if any(n & (n - 1) for n in budgets):
        raise ValueError("budgets must be powers of two (Sobol)")
    ad = raw["adaptive"]
    if int(ad["initial"]) > min(budgets):
        raise ValueError("the adaptive start design must not exceed the smallest budget")
    return SurrogateStudyConfig(
        version=int(raw["version"]),
        seeds=tuple(int(s) for s in raw["seeds"]),
        budgets=budgets,
        kappas=tuple(float(k) for k in raw["kappas"]),
        test_count=int(raw["test_states"]["count"]),
        test_seed=int(raw["test_states"]["seed"]),
        adaptive_initial=int(ad["initial"]),
        adaptive_max_batch=int(ad["max_batch"]),
        adaptive_pool_seed_offset=int(ad["pool_seed_offset"]),
        raw=raw,
    )


def _box(block: dict[str, list[float]]) -> Box:
    names = tuple(block)
    return Box(
        names, tuple(float(block[n][0]) for n in names), tuple(float(block[n][1]) for n in names)
    )


def _lookup(raw: dict[str, Any], path: str) -> float:
    """``a.b[1]`` in a nested mapping."""
    value: Any = raw
    for part in path.split("."):
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
        if m is None:
            raise ValueError(f"bad path {path!r}")
        value = value[m.group(1)]
        if m.group(2) is not None:
            value = value[int(m.group(2))]
    return float(value)


def _parameters(
    block: list[dict[str, Any]], source: dict[str, Any] | None = None
) -> tuple[ParameterSpec, ...]:
    out = []
    for p in block:
        if "truth_from" in p:
            if source is None:
                raise ValueError("truth_from needs a source config")
            truth = _lookup(source, p["truth_from"])
        else:
            truth = float(p["truth"])
        out.append(
            ParameterSpec(
                p["name"], float(p["nominal"]), truth, float(p["lower"]), float(p["upper"]),
                bool(p["fitted"]), str(p.get("source", "")),
            )
        )  # fmt: skip
    return tuple(out)


def build_problems(config: SurrogateStudyConfig) -> dict[str, ProblemFamily]:
    """``{"P1": LineProblemFamily, "P2": GreyBoxProblemFamily}``."""
    from twinlink_ml.headroom.config import load_headroom_config, study_process_config
    from twinlink_ml.simulation import PlaceholderConfigWarning

    from .greybox_problem import GreyBoxProblemFamily
    from .line_problem import LineProblemFamily, LineSettings

    p1 = config.raw["p1"]
    m, cons = p1["material"], p1["constraints"]
    material = SiMaterial(
        cp_gas_j_per_kg_k=float(m["cp_gas_j_per_kg_k"]),
        lower_heating_value_j_per_kg=float(m["lower_heating_value_j_per_kg"]),
        heater_efficiency=1.0,  # set by the parameter heater_efficiency
        cp_liquid_j_per_kg_k=float(m["cp_liquid_j_per_kg_k"]),
        cp_solid_j_per_kg_k=float(m["cp_solid_j_per_kg_k"]),
        cp_vapour_j_per_kg_k=float(m["cp_vapour_j_per_kg_k"]),
        equilibrium_moisture_kg_per_kg=float(m["equilibrium_moisture_kg_per_kg"]),
        enthalpy_of_evaporation_j_per_kg=float(m["enthalpy_of_evaporation_j_per_kg"]),
    )
    settings = LineSettings(
        decisions=_box(p1["decisions"]),
        disturbances=_box(p1["disturbances"]),
        nominal_disturbance=tuple(float(v) for v in p1["nominal_disturbance"]),
        material=material,
        spray_temperature_celsius=float(p1["spray_temperature_celsius"]),
        medium_cp_j_per_kg_k=float(p1["medium_cp_j_per_kg_k"]),
        medium_max_celsius=float(cons["medium_max_celsius"]),
        dew_point_margin_k=float(cons["dew_point_margin_k"]),
        fluidisation_min_m3_per_h=float(cons["fluidisation_min_m3_per_h"]),
        product_max_celsius=float(cons["product_max_celsius"]),
        tolerances={k: float(v) for k, v in p1["tolerances"].items()},
    )
    line = LineProblemFamily(settings, _parameters(p1["parameters"]))

    p2 = config.raw["p2"]
    headroom_path = REPO_ROOT / p2["headroom_config"]
    headroom_raw = yaml.safe_load(headroom_path.read_text(encoding="utf-8"))
    study = load_headroom_config(headroom_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        process = study_process_config(study)
    greybox = GreyBoxProblemFamily(
        study,
        process,
        _box(p2["disturbances"]),
        tuple(float(v) for v in p2["nominal_disturbance"]),
        _parameters(p2["parameters"], headroom_raw),
        tolerances={k: float(v) for k, v in p2["tolerances"].items()},
    )
    return {"P1": line, "P2": greybox}
