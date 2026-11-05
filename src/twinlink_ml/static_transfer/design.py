"""Designs, test sets, input errors and measurement noise of the static transfer study.

* Every point set is drawn from a scrambled Sobol stream in a box and filtered by admissibility
  (the bed exhaust of the source AND of every plant stays below a relative humidity); the first
  ``count`` admissible points are kept, so training designs are nested prefixes.
* Test sets are fixed (own scramble per set, never a training seed) and shared by every plant,
  condition and seed (paired). The extrapolation set is the outer box minus the interpolation box.
* Input errors: the methods see the MEASURED inputs; the plant runs at the TRUE ones
  (``true = measured + offset``), offsets fixed per seed (sign drawn per seed and quantity).
* Noise acts on the targets only; one standard-normal draw per seed is shared by the noise levels
  (the high level scales it) so the conditions differ only in what they are meant to vary.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .balance import DryerPlantModel
from .config import Box, InputErrorSpec, NoiseSpec, StaticTransferConfig

__all__ = [
    "TEST_SET_NAMES",
    "AdmissibleSample",
    "InputOffsets",
    "admissible",
    "admissible_points",
    "input_offsets",
    "noise_draws",
    "noisy",
    "test_sets",
    "training_design",
]

TEST_SET_NAMES: tuple[str, ...] = ("interpolation", "extrapolation", "plant_like", "product_change")


def admissible(x: np.ndarray, models: Sequence[DryerPlantModel], max_rh: float) -> np.ndarray:
    """True where the bed exhaust of every model stays at or below ``max_rh`` (and above 0 degC)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    ok = np.ones(len(x), dtype=bool)
    for model in models:
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            break
        _, bed, _ = model.solve(x[idx])
        good = (bed.relative_humidity <= max_rh) & (bed.temperature_celsius > 0.0)
        ok[idx[~good]] = False
    return ok


@dataclass(frozen=True)
class AdmissibleSample:
    """The kept points and the bookkeeping of the filter (reported)."""

    x: np.ndarray
    candidates: int
    admissible: int


def admissible_points(
    box: Box,
    count: int,
    seed: int,
    models: Sequence[DryerPlantModel],
    max_rh: float,
    *,
    exclude: Box | None = None,
    batch: int = 1024,
    max_candidates: int = 1 << 18,
) -> AdmissibleSample:
    """First ``count`` admissible points of one Sobol stream (optionally outside ``exclude``)."""
    sampler = qmc.Sobol(5, scramble=True, seed=seed)
    kept: list[np.ndarray] = []
    total = n_ok = 0
    while sum(len(k) for k in kept) < count:
        if total >= max_candidates:
            raise RuntimeError(f"fewer than {count} admissible points in {max_candidates}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unit = sampler.random(batch)
        total += batch
        x = box.scale(unit)
        if exclude is not None:
            x = x[~exclude.contains(x)]
        ok = admissible(x, models, max_rh)
        n_ok += int(ok.sum())
        kept.append(x[ok])
    return AdmissibleSample(np.concatenate(kept)[:count], total, n_ok)


def training_design(
    config: StaticTransferConfig, seed: int, models: Sequence[DryerPlantModel]
) -> AdmissibleSample:
    """Nested design of one seed in the interpolation box (``max(budgets)`` points)."""
    return admissible_points(
        config.boxes["interpolation"],
        max(config.budgets),
        config.design_seed_offset + seed,
        models,
        config.max_relative_humidity,
        batch=256,
    )


def test_sets(
    config: StaticTransferConfig, models: Sequence[DryerPlantModel], count: int | None = None
) -> dict[str, AdmissibleSample]:
    """The four named test sets (same for every plant, condition and seed)."""
    n = config.test_count if count is None else count
    b = config.boxes
    rh = config.max_relative_humidity
    seed = config.test_seed
    return {
        "interpolation": admissible_points(b["interpolation"], n, seed + 1, models, rh),
        "extrapolation": admissible_points(
            b["extrapolation_outer"], n, seed + 2, models, rh, exclude=b["interpolation"]
        ),
        "plant_like": admissible_points(b["plant_like"], n, seed + 3, models, rh),
        "product_change": admissible_points(b["product_change"], n, seed + 4, models, rh),
    }


@dataclass(frozen=True)
class InputOffsets:
    """``true = measured + offset`` per input (zero for the clean conditions)."""

    inlet_k: float = 0.0
    spray_relative: float = 0.0
    solid_fraction_relative: float = 0.0
    inlet_humidity_g_per_kg: float = 0.0

    def true_inputs(self, measured: np.ndarray) -> np.ndarray:
        x = np.array(measured, dtype=np.float64, copy=True)
        x[:, 0] += self.inlet_k
        x[:, 1] *= 1.0 + self.spray_relative
        x[:, 3] = np.maximum(x[:, 3] + self.inlet_humidity_g_per_kg, 0.0)
        x[:, 4] *= 1.0 + self.solid_fraction_relative
        return x

    def as_dict(self) -> dict[str, float]:
        return {
            "inlet_k": self.inlet_k,
            "spray_relative": self.spray_relative,
            "solid_fraction_relative": self.solid_fraction_relative,
            "inlet_humidity_g_per_kg": self.inlet_humidity_g_per_kg,
        }


def input_offsets(spec: InputErrorSpec, seed: int, active: bool) -> InputOffsets:
    """Offsets of one seed: full magnitude, sign drawn per quantity (zero if not active)."""
    if not active:
        return InputOffsets()
    signs = np.random.default_rng(spec.sign_seed_offset + seed).choice([-1.0, 1.0], size=4)
    return InputOffsets(
        inlet_k=float(signs[0] * spec.inlet_k),
        spray_relative=float(signs[1] * spec.spray_relative),
        solid_fraction_relative=float(signs[2] * spec.solid_fraction_relative),
        inlet_humidity_g_per_kg=float(signs[3] * spec.inlet_humidity_g_per_kg),
    )


def noise_draws(seed: int, n: int) -> np.ndarray:
    """Standard-normal draws ``(n, 3)`` of one seed (shared by the noise levels)."""
    return np.random.default_rng(1000 * seed + 17).standard_normal((n, 3))


def noisy(truth: np.ndarray, noise: NoiseSpec, draws: np.ndarray) -> np.ndarray:
    """Measured targets: truth plus scaled draws (relative for the power)."""
    return np.asarray(truth + noise.sigma(truth) * draws[: len(truth)])
