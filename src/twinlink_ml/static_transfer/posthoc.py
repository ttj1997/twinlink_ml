"""Post-hoc sensitivity v2 of the static transfer study: L3 with a stronger fine-tuning.

Declared post hoc (decision of the project lead, 2026-11-06); the main reading R3 of v1 stays.
Settings: ``conf/static_transfer/static_transfer_v2.yaml``. Two contexts:

* :func:`pilot_context` -- the pilot on the full-strength pilot plant ``PP-full`` with its own
  seeds, designs and validation set. Its admissibility uses the source, the six main plants and
  ``PP-full``; the v1 test sets are not touched.
* :func:`rerun_context` -- the v1 test sets and training designs exactly as stored in the v1
  artifact, and the v1 configuration with ONLY the L3 setting replaced. The unit code of v1
  (:func:`.study.run_unit`) is reused unchanged; every method except L3 is recomputed only as a
  check against v1 (:func:`check_against_v1`).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .balance import plant_model, source_model
from .config import REPO_ROOT, TARGET_NAMES, PlantSpec, StaticTransferConfig
from .design import TEST_SET_NAMES, AdmissibleSample, admissible_points, training_design
from .study import StudyContext

__all__ = [
    "DEFAULT_V2_PATH",
    "PosthocConfig",
    "check_against_v1",
    "load_posthoc",
    "pilot_config",
    "pilot_context",
    "rerun_config",
    "rerun_context",
]

DEFAULT_V2_PATH = REPO_ROOT / "conf" / "static_transfer" / "static_transfer_v2.yaml"


@dataclass(frozen=True)
class PosthocConfig:
    """The v2 YAML, typed where the code reads it."""

    v1_run: str
    v1_checkpoints: Path
    pilot_plant: PlantSpec
    pilot_seeds: tuple[int, ...]
    pilot_budgets: tuple[int, ...]
    validation_seed_offset: int
    max_extensions: int
    grid: dict[str, tuple[float, ...]]
    condition: str
    reported_test_sets: tuple[str, ...]
    raw: dict[str, Any] = field(compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def fingerprint_without_selection(self) -> str:
        raw = {k: v for k, v in self.raw.items() if k != "selected"}
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]

    def selected_l3(self) -> dict[str, float] | None:
        value = self.raw["selected"].get("L3")
        return None if value is None else {k: float(v) for k, v in value.items()}


def load_posthoc(path: Path | str = DEFAULT_V2_PATH) -> PosthocConfig:
    """Read the v2 YAML."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ((name, plant_raw),) = raw["pilot_plant"].items()
    pilot = raw["pilot"]
    rerun = raw["rerun"]
    sets = tuple(str(s) for s in rerun["test_sets"])
    if set(sets) - set(TEST_SET_NAMES):
        raise ValueError(f"unknown test sets {sets}")
    return PosthocConfig(
        v1_run=str(raw["v1_run"]),
        v1_checkpoints=REPO_ROOT / str(raw["v1_checkpoints"]),
        pilot_plant=PlantSpec.from_mapping(name, plant_raw),
        pilot_seeds=tuple(int(s) for s in pilot["seeds"]),
        pilot_budgets=tuple(int(n) for n in pilot["budgets"]),
        validation_seed_offset=int(pilot["validation_seed_offset"]),
        max_extensions=int(pilot["max_extensions"]),
        grid={k: tuple(float(v) for v in vals) for k, vals in pilot["grid"].items()},
        condition=str(rerun["condition"]),
        reported_test_sets=sets,
        raw=raw,
    )


def pilot_config(base: StaticTransferConfig, spec: PosthocConfig) -> StaticTransferConfig:
    """v1 configuration with the full-strength pilot plant and the v2 pilot block."""
    raw = copy.deepcopy(base.raw)
    raw["pilot"] = {
        "seeds": list(spec.pilot_seeds),
        "budgets": list(spec.pilot_budgets),
        "max_extensions": spec.max_extensions,
        "grids": {"L3": {k: list(v) for k, v in spec.grid.items()}},
    }
    return dataclasses.replace(base, pilot_plant=spec.pilot_plant, raw=raw)


def pilot_context(base: StaticTransferConfig, spec: PosthocConfig) -> StudyContext:
    """Designs of the pilot seeds and the pilot validation set (no test set is built)."""
    config = pilot_config(base, spec)
    models = [source_model(config.source)]
    models += [plant_model(config.source, p) for p in (*config.plants, spec.pilot_plant)]
    validation = admissible_points(
        config.boxes["interpolation"],
        config.test_count,
        config.test_seed + spec.validation_seed_offset,
        models,
        config.max_relative_humidity,
    )
    return StudyContext(
        config=config,
        tests={},
        pilot_validation=validation,
        designs={s: training_design(config, s, models) for s in spec.pilot_seeds},
    )


def rerun_config(
    base: StaticTransferConfig, l3_setting: Mapping[str, float]
) -> StaticTransferConfig:
    """v1 configuration with ONLY the selected L3 setting replaced."""
    raw = copy.deepcopy(base.raw)
    raw["selected"]["L3"] = {k: float(v) for k, v in l3_setting.items()}
    return dataclasses.replace(base, raw=raw)


def rerun_context(config: StaticTransferConfig, v1_run: Mapping[str, Any]) -> StudyContext:
    """Test sets and designs exactly as stored in the v1 artifact."""
    adm = v1_run["coverage"]["admissibility"]
    tests = {
        k: AdmissibleSample(
            np.asarray(v1_run["test_sets"][k], dtype=np.float64),
            int(adm[k]["candidates"]),
            int(adm[k]["admissible"]),
        )
        for k in TEST_SET_NAMES
    }
    designs = {
        int(s): AdmissibleSample(np.asarray(x, dtype=np.float64), -1, -1)
        for s, x in v1_run["designs"].items()
    }
    return StudyContext(
        config=config, tests=tests, pilot_validation=tests["interpolation"], designs=designs
    )


def check_against_v1(
    unit: Mapping[str, Any], v1_unit: Mapping[str, Any], skip: tuple[str, ...] = ("L3",)
) -> dict[str, Any]:
    """Largest absolute difference of every recorded error of every method except ``skip``."""
    worst = 0.0
    compared = 0
    for method, sets in unit["fixed"].items():
        for ts, err in sets.items():
            for metric, values in err.items():
                diff = np.abs(np.asarray(values) - np.asarray(v1_unit["fixed"][method][ts][metric]))
                worst = max(worst, float(diff.max()))
                compared += len(values)
    for n, methods in unit["fitted"].items():
        for method, payload in methods.items():
            if method in skip:
                continue
            for ts, err in payload["test"].items():
                for metric, values in err.items():
                    ref = v1_unit["fitted"][n][method]["test"][ts][metric]
                    diff = np.abs(np.asarray(values) - np.asarray(ref))
                    worst = max(worst, float(diff.max()))
                    compared += len(values)
    return {"max_abs_difference": worst, "values_compared": compared, "targets": list(TARGET_NAMES)}
