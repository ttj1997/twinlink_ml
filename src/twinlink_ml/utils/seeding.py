"""Central seed handling -- one place for Python, NumPy, PyTorch, CUDA.

Why this is in place from step 1 and not later: as soon as ``np.random.seed`` or
``torch.manual_seed`` appear in several places in the code, the question "was run X
reproducible" can no longer be answered. For learning curves with 3-5 seeds per point the
answer to this question is the basis of the statement.

Important for the evaluation: in the process model the seed affects **only** the sensor model
and the excitation sampling. The grey-box model itself is deterministic. Spread in
the learning curves thus comes from three separable sources -- dataset realisation
(``data_seed``), sensor noise (``noise_seed``) and initialisation/batch order
of the network (``model_seed``). Whoever couples all three to one seed cannot later
say whether the spread comes from the data or from the training.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "SeedBundle",
    "SeedPlan",
    "VarianceSource",
    "seed_everything",
]


def _derive(*parts: str | int) -> int:
    """32-bit seed from arbitrary components.

    SHA-256 instead of a linear transformation, so that neighbouring master seeds do not
    produce correlated sub-seeds.
    """
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class VarianceSource(str, Enum):
    """Random source whose spread is varied in a study.

    ``JOINT`` varies all three together via one master seed. This is the
    conservative, hence most honest figure and the default of the learning curves. The
    individual sources answer the question that ``JOINT`` leaves open: does a difference
    between two architectures come from the training or from the dataset realisation?
    """

    DATA = "data"
    NOISE = "noise"
    MODEL = "model"
    JOINT = "joint"

    @property
    def is_parameter_free_relevant(self) -> bool:
        """Whether a parameter-free model can spread at all on this axis.

        Persistence has no parameters: its spread over ``MODEL`` is exactly zero, while over
        ``DATA`` and ``NOISE`` it is pure realisation of the dataset. Exactly for this reason
        it is the measure of the resolving power -- see
        :mod:`twinlink_ml.evaluation.resolution`.
        """
        return self in {VarianceSource.DATA, VarianceSource.NOISE, VarianceSource.JOINT}


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """Separate seeds for the three independent random sources."""

    data_seed: int
    noise_seed: int
    model_seed: int

    @classmethod
    def from_master(cls, master_seed: int) -> SeedBundle:
        """Deterministic derivation of three seeds from one master seed."""
        return cls(
            data_seed=_derive(master_seed, "data"),
            noise_seed=_derive(master_seed, "noise"),
            model_seed=_derive(master_seed, "model"),
        )

    def generator(self, role: str) -> np.random.Generator:
        """NumPy generator for a role. Preferred over global state."""
        seeds = {"data": self.data_seed, "noise": self.noise_seed, "model": self.model_seed}
        if role not in seeds:
            raise ValueError(f"Unbekannte Rolle {role!r}, erlaubt: {sorted(seeds)}")
        return np.random.default_rng(seeds[role])

    def replace(self, source: VarianceSource, value: int) -> SeedBundle:
        """Bundle with one sub-seed exchanged.

        ``JOINT`` is not permitted here -- a joint master seed is not a
        sub-seed but produces all three at once via :meth:`from_master`.
        """
        field = {
            VarianceSource.DATA: "data_seed",
            VarianceSource.NOISE: "noise_seed",
            VarianceSource.MODEL: "model_seed",
        }.get(source)
        if field is None:
            raise ValueError("VarianceSource.JOINT ersetzt keinen einzelnen Teilseed")
        return dataclasses.replace(self, **{field: value})

    def as_parameters(self) -> dict[str, int]:
        """For the run log."""
        return {
            "seed/data": self.data_seed,
            "seed/noise": self.noise_seed,
            "seed/model": self.model_seed,
        }


@dataclass(frozen=True, slots=True)
class SeedPlan:
    """Which seeds a study runs, broken down by random source.

    Construction **one source at a time** (one-factor-at-a-time): replicate :math:`k`
    of an axis uses the sub-seed of master seed :math:`m_k` and, for the two
    other axes, those of the reference master seed :math:`m_0`.

    Two properties that arise from this and are both needed:

    * The sub-seeds of the axis ``DATA`` are **the same** as those of ``JOINT``. This
      allows the simulated run pool to be reused between the axes -- the
      grey-box model is deterministic, a pool depends only on the ``data_seed``.
    * Replicate :math:`k = 0` is identical on all axes (the reference bundle). It
      is computed only once and hooked into every axis.

    The decomposition is thus **not** an analysis of variance with interactions: what is
    measured are main effects. Whether this is admissible is checked by
    :meth:`~twinlink_ml.evaluation.resolution.VarianceDecomposition.additivity_ratio`
    against the joint spread -- which is why ``JOINT`` is run as well and not claimed as
    the sum of the individual axes.
    """

    master_seeds: tuple[int, ...] = (1, 2, 3)

    def __post_init__(self) -> None:
        if len(self.master_seeds) < 2:
            raise ValueError("mindestens zwei Master-Seeds, sonst gibt es keine Streuung")
        if len(set(self.master_seeds)) != len(self.master_seeds):
            raise ValueError("Master-Seeds müssen paarweise verschieden sein")

    @property
    def n_replicates(self) -> int:
        """Number of replicates per axis."""
        return len(self.master_seeds)

    @property
    def reference(self) -> SeedBundle:
        """Reference bundle, replicate 0 of all axes."""
        return SeedBundle.from_master(self.master_seeds[0])

    def bundle(self, source: VarianceSource, replicate: int) -> SeedBundle:
        """Seed bundle for axis and replicate index."""
        if not 0 <= replicate < self.n_replicates:
            raise ValueError(f"replicate {replicate} außerhalb 0..{self.n_replicates - 1}")
        variant = SeedBundle.from_master(self.master_seeds[replicate])
        if source is VarianceSource.JOINT:
            return variant
        field = {
            VarianceSource.DATA: variant.data_seed,
            VarianceSource.NOISE: variant.noise_seed,
            VarianceSource.MODEL: variant.model_seed,
        }[source]
        return self.reference.replace(source, field)

    def bundles(self, source: VarianceSource) -> tuple[SeedBundle, ...]:
        """All bundles of an axis, replicate 0 first."""
        return tuple(self.bundle(source, k) for k in range(self.n_replicates))

    def data_seeds(self, sources: tuple[VarianceSource, ...]) -> tuple[int, ...]:
        """All occurring ``data_seed`` values -- the number of expensive simulations."""
        seen = {bundle.data_seed for source in sources for bundle in self.bundles(source)}
        return tuple(sorted(seen))

    def as_parameters(self) -> dict[str, object]:
        """For the run log."""
        return {"seedplan/master_seeds": list(self.master_seeds)}


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> SeedBundle:
    """Sets all global random sources and returns the derived seed bundle.

    :param deterministic_torch: enables deterministic CUDA kernels. Costs
        compute time and makes some operations fail -- for the learning curves in
        AP 1 the price is well invested, for long RL runs later to be weighed up.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 - global state is the purpose here
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False

    return SeedBundle.from_master(seed)
