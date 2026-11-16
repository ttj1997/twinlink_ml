r"""Set-up of docs/41_hybrid_structural.md (``conf/hybrid_structural/hybrid_structural.yaml``).

Everything of the hybrid-model study of Befund 122 (``hybrid_model.study``: configs, the 20 draws of
Befund 111, cases, action box) is read by the same code from a config of the same schema. The one
difference is the truth: :class:`StructuralSetup` simulates every data pool with
:class:`~twinlink_ml.hybrid_structural.truth.FreeboardEvaporationGranulator` and evaluates the
steady truth on :func:`~twinlink_ml.hybrid_structural.truth.steady_config`. Prior knowledge (the
nominal configs, basis of M0, M2, M2b and M3) stays the model WITHOUT the freeboard term.
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


def load_structural_config(path: Path | str = DEFAULT_CONFIG_PATH) -> HybridStudyConfig:
    """The study config (schema of ``conf/hybrid_model/hybrid_model.yaml`` plus ``structure``)."""
    config = load_hybrid_config(path)
    structure = config.raw.get("structure")
    if not isinstance(structure, dict) or structure.get("mode") != "evaporation":
        raise ValueError("the structural study needs structure.mode = evaporation")
    return config


def freeboard_share(config: HybridStudyConfig) -> float:
    """The fixed share of the spray water evaporating above the bed (set before the runs)."""
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


def build_structural_setup(config: HybridStudyConfig | None = None) -> StructuralSetup:
    """Configs, draws and cases of the study, with the freeboard share of the config."""
    cfg = config if config is not None else load_structural_config()
    base = build_setup(cfg)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    return StructuralSetup(**fields, share=freeboard_share(cfg))
