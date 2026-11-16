r"""Set-up of docs/41_hybrid_structural.md (``conf/hybrid_structural/hybrid_structural.yaml``).

Everything of the hybrid-model study of Befund 122 (``hybrid_model.study``: configs, the 20 draws of
Befund 111, cases, action box) is read by the same code from a config of the same schema. The one
difference is the truth: :class:`StructuralSetup` simulates every data pool with
:class:`~twinlink_ml.hybrid_structural.truth.FreeboardEvaporationGranulator` and evaluates the
steady truth on :func:`~twinlink_ml.hybrid_structural.truth.steady_config`. Prior knowledge (the
nominal configs, basis of M0, M2, M2b and M3) stays the model WITHOUT the freeboard term.

The drivers of the study (``scripts/train_hybrid_structural.py``,
``scripts/run_hybrid_structural_evaluation.py``) are also the drivers of the follow-up study
docs/NN_hybrid_structural_2.md, whose truth errs to the other side: :func:`build_structural_setup`
returns the set-up that ``structure.mode`` of the config names (``evaporation``: this module;
``recirculation``: :mod:`twinlink_ml.hybrid_structural_2.study`). Every set-up describes its
structural term for the records (``structure_record``) and names its study (``study_label``).
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

from .truth import FreeboardEvaporationGranulator, steady_config

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "MODELS",
    "StructuralSetup",
    "build_structural_setup",
    "freeboard_share",
    "load_structural_config",
]

DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "hybrid_structural" / "hybrid_structural.yaml"
MODELS = ("M0", "M1", "M2", "M2b", "M3")
STRUCTURE_MODES = ("evaporation", "recirculation")


def load_structural_config(path: Path | str = DEFAULT_CONFIG_PATH) -> HybridStudyConfig:
    """The study config (schema of ``conf/hybrid_model/hybrid_model.yaml`` plus ``structure``)."""
    config = load_hybrid_config(path)
    structure = config.raw.get("structure")
    if not isinstance(structure, dict) or structure.get("mode") not in STRUCTURE_MODES:
        raise ValueError(f"the structural study needs structure.mode in {STRUCTURE_MODES}")
    return config


def freeboard_share(config: HybridStudyConfig) -> float:
    """The fixed share of the spray water evaporating above the bed (set before the runs)."""
    if config.raw["structure"].get("mode") != "evaporation":
        raise ValueError("the freeboard share belongs to structure.mode = evaporation")
    share = config.raw["structure"].get("share")
    if share is None:
        raise ValueError(
            "structure.share is not set -- run scripts/probe_hybrid_structural_share.py first"
        )
    share = float(share)
    if not 0.0 < share < 1.0:
        raise ValueError("structure.share must lie in (0, 1)")
    return share


@dataclass(frozen=True)
class StructuralSetup(StudySetup):
    """The study set-up of Befund 122 with the structurally deviating truth."""

    share: float

    study_label = (
        "hybrid against GRU against refit, structurally deviating truth "
        "(docs/41_hybrid_structural.md)"
    )

    def structure_record(self) -> dict[str, Any]:
        """The structural term of the truth, as written into every record."""
        return {"mode": "evaporation", "share": self.share}

    def generate(self, config: ProcessConfig, spec: Any, seed: int) -> Any:
        """Data pool of ``spec`` simulated with the freeboard truth of ``config``."""
        import warnings

        from twinlink_ml.evaluation.experiments import generate_data
        from twinlink_ml.simulation import PlaceholderConfigWarning

        share = self.share
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            return generate_data(
                config,
                spec,
                master_seed=seed,
                simulator_factory=lambda c: FreeboardEvaporationGranulator(c, share),
            )

    def steady_truth(self, draw: int) -> ProcessConfig:
        return steady_config(super().steady_truth(draw), self.share)


def build_structural_setup(config: HybridStudyConfig | None = None) -> Any:
    """Configs, draws and cases of the study, with the structural term of the config.

    ``structure.mode = evaporation`` (default config): :class:`StructuralSetup` with the freeboard
    share. ``recirculation``: the set-up of the follow-up study
    (:func:`twinlink_ml.hybrid_structural_2.study.build_recirculation_setup`).
    """
    cfg = config if config is not None else load_structural_config()
    if cfg.raw["structure"]["mode"] == "recirculation":
        from twinlink_ml.hybrid_structural_2.study import build_recirculation_setup

        return build_recirculation_setup(cfg)
    base = build_setup(cfg)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    return StructuralSetup(**fields, share=freeboard_share(cfg))
