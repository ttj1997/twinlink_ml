r"""Set-up of docs/42_hybrid_structural_2.md.

Settings: ``conf/hybrid_structural_2/hybrid_structural_2.yaml``.

The study of Befund 124 (docs/41_hybrid_structural.md) with a structural error of the OTHER sign:
the truth is WETTER than the prior model predicts. Everything of that study -- code, task, cases,
models, seeds and the 20 parameter draws of Befund 111 -- is reused; the one difference is the
structural term of the truth:

* Befund 124: a share of the spray water evaporates above the bed (bed drier and warmer).
* Here: **exhaust recirculation**, an existing switch of the grey-box model
  (``OperationConfig.exhaust_recirculation_fraction``, default ``None`` = off). A fraction ``r``
  of the dry air through the bed is taken from the bed exhaust (``T_bed``, ``Y_out``) and mixed
  with fresh ambient air ahead of the heater, which brings the mixture to the commanded inlet
  temperature. The bed sees the same inlet temperature and gas velocity but a higher inlet
  humidity, dries less and is wetter. The switch is part of the right-hand side, of the steady
  solvers and of the energy target (the heater intake is the mixture), so the truth needs no
  emulation: it is the grey-box model of the config with the switch set
  (:func:`with_recirculation`), for every data pool and every steady evaluation.

Prior knowledge (the nominal configs, basis of M0, M2, M2b and M3) stays the model with the switch
off. The drivers are those of Befund 124; ``hybrid_structural.study.build_structural_setup``
returns :class:`RecirculationSetup` for ``structure.mode = recirculation``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twinlink_ml.hybrid_model.study import (
    REPO_ROOT,
    HybridStudyConfig,
    StudySetup,
    build_setup,
    load_hybrid_config,
)
from twinlink_ml.simulation import ProcessConfig

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "RecirculationSetup",
    "build_recirculation_setup",
    "load_recirculation_config",
    "recirculation_fraction",
    "with_recirculation",
]

DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "hybrid_structural_2" / "hybrid_structural_2.yaml"


def _check_fraction(fraction: float) -> float:
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("the recirculation fraction of the truth must lie in (0, 1)")
    return fraction


def with_recirculation(config: ProcessConfig, fraction: float) -> ProcessConfig:
    """``config`` with the exhaust recirculation switched on at ``fraction``; nothing else."""
    return dataclasses.replace(
        config,
        operation=dataclasses.replace(
            config.operation, exhaust_recirculation_fraction=_check_fraction(fraction)
        ),
    )


def load_recirculation_config(path: Path | str = DEFAULT_CONFIG_PATH) -> HybridStudyConfig:
    """The study config (schema of ``conf/hybrid_model/hybrid_model.yaml`` plus ``structure``)."""
    config = load_hybrid_config(path)
    structure = config.raw.get("structure")
    if not isinstance(structure, dict) or structure.get("mode") != "recirculation":
        raise ValueError("this study needs structure.mode = recirculation")
    return config


def recirculation_fraction(config: HybridStudyConfig) -> float:
    """The fixed recirculation fraction of the truth (set before the runs)."""
    fraction = config.raw["structure"].get("fraction")
    if fraction is None:
        raise ValueError(
            "structure.fraction is not set -- run scripts/probe_hybrid_structural_2_strength.py"
        )
    return _check_fraction(fraction)


@dataclass(frozen=True)
class RecirculationSetup(StudySetup):
    """The study set-up of Befund 122 with a truth that recirculates a fraction of its exhaust."""

    fraction: float

    study_label = (
        "hybrid against GRU against refit, structural error to the unsafe side "
        "(docs/42_hybrid_structural_2.md)"
    )

    def structure_record(self) -> dict[str, Any]:
        """The structural term of the truth, as written into every record."""
        return {"mode": "recirculation", "fraction": self.fraction}

    def data_truth(self, draw: int) -> ProcessConfig:
        return with_recirculation(super().data_truth(draw), self.fraction)

    def steady_truth(self, draw: int) -> ProcessConfig:
        return with_recirculation(super().steady_truth(draw), self.fraction)


def build_recirculation_setup(config: HybridStudyConfig | None = None) -> RecirculationSetup:
    """Configs, draws and cases of the study, with the recirculation fraction of the config."""
    cfg = config if config is not None else load_recirculation_config()
    base = build_setup(cfg)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    return RecirculationSetup(**fields, fraction=recirculation_fraction(cfg))
