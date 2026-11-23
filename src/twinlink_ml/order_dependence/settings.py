"""Settings of the order-dependence measurement (study settings, no plant values)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from twinlink_ml.headroom.controllers import COLD_START_SEED

__all__ = ["OrderDependenceSettings"]


@dataclass(frozen=True, slots=True)
class OrderDependenceSettings:
    """What is measured and with which seeds.

    ``cold_starts``: starts of the cold multi-start under test (the proposed switch).
    ``reference_starts``: starts of the dense reference that judges whether ``cold_starts``
    suffice (plain minimum over the strictly converged starts).
    ``permutation_seeds``: seeds of the random orders of the warm passes (next to the natural and
    the reversed order). ``thresholds``: relative power differences at which shares are counted;
    the first is the "not equal" threshold of the control of the learned-setpoints study.
    ``same_optimum_*``: two solutions count as the same local optimum if their set points agree
    within these steps (clusters of the reference starts). ``reading_threshold``: relative change
    of a key number that separates O0 from O1.
    """

    cold_starts: int = 8
    reference_starts: int = 64
    start_seed: int = COLD_START_SEED
    reference_seed: int = COLD_START_SEED + 1
    permutation_seeds: tuple[int, ...] = (1, 2, 3)
    thresholds: tuple[float, ...] = (1.0e-9, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2)
    same_optimum_inlet_k: float = 0.05
    same_optimum_gas_m_per_s: float = 5.0e-4
    reading_threshold: float = 0.10

    @property
    def orders(self) -> tuple[str, ...]:
        """Names of the warm passes, in the order of the result columns."""
        return ("natural", "reversed", *(f"permutation_{s}" for s in self.permutation_seeds))

    @property
    def fingerprint(self) -> str:
        """Short hash of the settings."""
        text = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
