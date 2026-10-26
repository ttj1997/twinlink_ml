"""Parameter draws of the truth model and the scenario subset.

Draws: a scrambled Sobol' sequence (:class:`scipy.stats.qmc.Sobol`, fixed seed) of
``sequence_length`` points over the unit cube of the named parameters; each coordinate is mapped
with the :class:`~twinlink_ml.sensitivity.config.InputSpec` of the Sobol study (band and scale read
from ``conf/sensitivity/sobol_steady.yaml``, model ``p2``). The first ``n`` points whose material
passes :func:`~twinlink_ml.simulation.sorption.validate_isotherm_against_critical_moisture` are
used, in sequence order; rejected points are counted, not dropped silently. Only these parameters
of the truth differ from the nominal model (the controller's).

Scenarios: of every type of the family, the indices ``round(linspace(0, n_type - 1, per_type))``
(deterministic, documented in docs/28).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from twinlink_ml.headroom.scenarios import Scenario
from twinlink_ml.sensitivity.config import SensitivityStudyConfig
from twinlink_ml.simulation import ProcessConfig
from twinlink_ml.simulation.sorption import validate_isotherm_against_critical_moisture

from .config import DrawSettings

__all__ = ["ParameterDraw", "parameter_draws", "scenario_subset", "truth_config"]


@dataclass(frozen=True)
class ParameterDraw:
    """One draw: index among the valid draws, position in the sequence, unit and values."""

    index: int
    sequence_position: int
    unit: dict[str, float]
    values: dict[str, float]
    process: ProcessConfig

    def as_dict(self) -> dict[str, object]:
        """Plain dict for the record (without the config)."""
        return {
            "index": self.index,
            "sequence_position": self.sequence_position,
            "unit": dict(self.unit),
            "values": dict(self.values),
        }


def truth_config(nominal: ProcessConfig, values: dict[str, float]) -> ProcessConfig:
    """Nominal process config with the named material parameters replaced."""
    material_fields = {f.name for f in dataclasses.fields(nominal.material)}
    unknown = set(values) - material_fields
    if unknown:
        raise ValueError(f"not material parameters: {sorted(unknown)}")
    return dataclasses.replace(nominal, material=dataclasses.replace(nominal.material, **values))


def parameter_draws(
    settings: DrawSettings, sensitivity: SensitivityStudyConfig, nominal: ProcessConfig
) -> tuple[list[ParameterDraw], list[int]]:
    """The valid draws and the sequence positions of the rejected ones."""
    specs = {s.name: s for s in sensitivity.inputs["p2"]}
    missing = [p for p in settings.parameters if p not in specs]
    if missing:
        raise ValueError(f"parameters not in the p2 inputs of the Sobol config: {missing}")
    if settings.sequence_length & (settings.sequence_length - 1):
        raise ValueError("sequence_length must be a power of two (balance of the Sobol' points)")
    sampler = qmc.Sobol(d=len(settings.parameters), scramble=True, seed=settings.seed)
    points = sampler.random(settings.sequence_length)
    draws: list[ParameterDraw] = []
    rejected: list[int] = []
    for position, row in enumerate(points):
        if len(draws) == settings.n:
            break
        unit = {p: float(u) for p, u in zip(settings.parameters, row, strict=True)}
        values = {p: float(specs[p].value(np.array(unit[p]))) for p in settings.parameters}
        config = truth_config(nominal, values)
        try:
            validate_isotherm_against_critical_moisture(config)
        except ValueError:
            rejected.append(position)
            continue
        draws.append(ParameterDraw(len(draws), position, unit, values, config))
    if len(draws) < settings.n:
        raise RuntimeError(
            f"only {len(draws)} valid draws in {settings.sequence_length} points "
            f"({len(rejected)} rejected)"
        )
    return draws, rejected


def scenario_subset(scenarios: list[Scenario], per_type: int) -> list[Scenario]:
    """``per_type`` scenarios of every type at evenly spread indices, in family order."""
    kinds: dict[str, list[Scenario]] = {}
    for scenario in scenarios:
        kinds.setdefault(scenario.kind, []).append(scenario)
    chosen: set[str] = set()
    for members in kinds.values():
        ordered = sorted(members, key=lambda s: s.index)
        picks = np.round(np.linspace(0, len(ordered) - 1, per_type)).astype(int)
        if len(set(picks.tolist())) != per_type:
            raise ValueError("per_type exceeds the number of scenarios of a type")
        chosen.update(ordered[i].name for i in picks)
    return [s for s in scenarios if s.name in chosen]
