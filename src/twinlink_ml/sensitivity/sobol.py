r"""Sobol indices from a Saltelli design, computed on valid samples only.

Design
------
:func:`saltelli_unit` draws the design of SALib (``SALib.sample.sobol``, scrambled Sobol'
sequence, no second-order indices) on the unit cube: for a base sample ``N`` and ``D`` inputs,
``N`` blocks of ``D + 2`` rows ``(A, AB_1, ..., AB_D, B)``. The first ``N / 2`` blocks of a
design with base ``N`` are the design with base ``N / 2`` (same seed; checked in the tests), so
convergence is read from nested prefixes (doubling) of one design.

Invalid samples
---------------
A model run can be invalid (a solver fails, the steady state lies outside the model's validity,
the input combination is rejected by a model check). The Saltelli estimators need complete
blocks, so a block counts only if all its ``D + 2`` rows are valid; :func:`sobol_indices` drops
the other blocks and reports how many it dropped (rows and blocks). The indices then describe
the input distribution restricted to valid blocks; with a small invalid share the difference is
small, with a large one it is a conditional analysis (reported with the share).

Estimators
----------
SALib's ``sobol.analyze``: first order after Saltelli et al. (2010), total after Jansen (1999),
confidence intervals by bootstrap over blocks (normal approximation, SALib's ``*_conf`` is the
half width). "CI width" in this package is the FULL width ``2 * conf``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "SobolIndices",
    "block_valid",
    "rank_change",
    "saltelli_unit",
    "sobol_indices",
    "verdict",
]


def _problem(names: Sequence[str]) -> dict[str, Any]:
    return {"num_vars": len(names), "names": list(names), "bounds": [[0.0, 1.0]] * len(names)}


def saltelli_unit(n_inputs: int, base_log2: int, seed: int) -> np.ndarray:
    """Saltelli design on the unit cube, shape ``(2**base_log2 * (n_inputs + 2), n_inputs)``."""
    from SALib.sample import sobol as salib_sobol

    names = [f"x{i}" for i in range(n_inputs)]
    return np.asarray(
        salib_sobol.sample(
            _problem(names), 2**base_log2, calc_second_order=False, scramble=True, seed=seed
        ),
        dtype=np.float64,
    )


def block_valid(valid: np.ndarray, n_inputs: int) -> np.ndarray:
    """Per block, whether all ``n_inputs + 2`` rows are valid."""
    v = np.asarray(valid, dtype=bool)
    rows = n_inputs + 2
    if v.size % rows:
        raise ValueError("the row count is not a multiple of n_inputs + 2")
    return v.reshape(-1, rows).all(axis=1)


@dataclass(frozen=True)
class SobolIndices:
    """First-order and total indices with half widths of the confidence interval."""

    names: tuple[str, ...]
    s1: np.ndarray
    s1_conf: np.ndarray
    st: np.ndarray
    st_conf: np.ndarray
    n_blocks: int
    n_blocks_valid: int
    n_rows_invalid: int
    mean: float
    std: float

    @property
    def valid_share(self) -> float:
        return self.n_blocks_valid / self.n_blocks if self.n_blocks else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_blocks": self.n_blocks,
            "n_blocks_valid": self.n_blocks_valid,
            "valid_block_share": self.valid_share,
            "n_rows_invalid": self.n_rows_invalid,
            "mean": self.mean,
            "std": self.std,
            "indices": {
                name: {
                    "S1": float(self.s1[i]),
                    "S1_ci_width": float(2.0 * self.s1_conf[i]),
                    "ST": float(self.st[i]),
                    "ST_ci_width": float(2.0 * self.st_conf[i]),
                }
                for i, name in enumerate(self.names)
            },
        }

    def ranking(self, subset: Sequence[str] | None = None) -> list[str]:
        """Names by descending total index (ties by name)."""
        names = list(self.names) if subset is None else [n for n in self.names if n in subset]
        st = {n: float(self.st[self.names.index(n)]) for n in names}
        return sorted(names, key=lambda n: (-st[n], n))


def sobol_indices(
    y: np.ndarray,
    valid: np.ndarray,
    names: Sequence[str],
    *,
    n_blocks: int | None = None,
    resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> SobolIndices:
    """Indices of output ``y`` (one value per design row) on the valid blocks.

    ``n_blocks``: use only the first ``n_blocks`` blocks (prefix = smaller base sample).
    Invalid rows may hold any value (NaN included); they are never used.
    """
    from SALib.analyze import sobol as salib_sobol

    d = len(names)
    rows = d + 2
    y = np.asarray(y, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(y)
    total_blocks = y.size // rows
    n = total_blocks if n_blocks is None else n_blocks
    if n > total_blocks:
        raise ValueError("n_blocks exceeds the design")
    y, valid = y[: n * rows], valid[: n * rows]
    keep = block_valid(valid, d)
    yk = y.reshape(n, rows)[keep]

    def indices(
        s1: np.ndarray, s1c: np.ndarray, st: np.ndarray, stc: np.ndarray, mean: float, std: float
    ) -> SobolIndices:
        return SobolIndices(
            names=tuple(names),
            s1=s1,
            s1_conf=s1c,
            st=st,
            st_conf=stc,
            n_blocks=n,
            n_blocks_valid=int(keep.sum()),
            n_rows_invalid=int((~valid).sum()),
            mean=mean,
            std=std,
        )

    if keep.sum() < 2:
        nan = np.full(d, np.nan)
        return indices(nan, nan, nan, nan, math.nan, math.nan)
    ab = np.concatenate([yk[:, 0], yk[:, -1]])
    mean, std = float(ab.mean()), float(ab.std())
    if std == 0.0:
        zero = np.zeros(d)
        return indices(zero, zero, zero, zero, mean, 0.0)
    result = salib_sobol.analyze(
        _problem(names),
        yk.ravel(),
        calc_second_order=False,
        num_resamples=resamples,
        conf_level=confidence_level,
        seed=seed,
    )
    return indices(
        np.asarray(result["S1"], dtype=np.float64),
        np.asarray(result["S1_conf"], dtype=np.float64),
        np.asarray(result["ST"], dtype=np.float64),
        np.asarray(result["ST_conf"], dtype=np.float64),
        mean,
        std,
    )


def verdict(
    indices: SobolIndices, *, dominant_total_index: float, max_ci_width: float
) -> dict[str, Any]:
    """Dominant inputs (``ST >= dominant_total_index``) and the convergence verdict.

    Converged: the full CI width of S1 AND of ST is below ``max_ci_width`` for every dominant
    input (pre-registered, docs/26 section 4). No dominant input: converged is ``None``.
    """
    dominant = [
        n
        for i, n in enumerate(indices.names)
        if np.isfinite(indices.st[i]) and indices.st[i] >= dominant_total_index
    ]
    widths = {
        n: {
            "S1": float(2.0 * indices.s1_conf[indices.names.index(n)]),
            "ST": float(2.0 * indices.st_conf[indices.names.index(n)]),
        }
        for n in dominant
    }
    converged = (
        None
        if not dominant
        else all(w["S1"] < max_ci_width and w["ST"] < max_ci_width for w in widths.values())
    )
    return {
        "dominant": sorted(dominant, key=lambda n: -indices.st[indices.names.index(n)]),
        "ci_widths_of_dominant": widths,
        "converged": converged,
    }


def rank_change(
    reference: SobolIndices,
    other: SobolIndices,
    subset: Sequence[str],
    *,
    dominant_total_index: float,
    top_k: int,
) -> dict[str, Any]:
    """Whether the ranking by total index of ``subset`` changes from ``reference`` to ``other``.

    Rule (pre-registered, docs/26 section 4): compare the first ``K`` places, ``K`` = the number
    of inputs of ``subset`` dominant in ``reference``, at least ``top_k``. Changed = the ordered
    list of the first ``K`` places differs. Affected = every input whose place within the first
    ``K`` differs or that enters or leaves them. A swap is "resolved" if the CIs of the two
    total indices do not overlap in either design, else "within CI".
    """
    ref_rank = reference.ranking(subset)
    oth_rank = other.ranking(subset)
    n_dom = sum(1 for n in subset if reference.st[reference.names.index(n)] >= dominant_total_index)
    k = min(max(n_dom, top_k), len(ref_rank))
    ref_top, oth_top = ref_rank[:k], oth_rank[:k]
    affected = sorted(
        {n for i, n in enumerate(ref_top) if i >= len(oth_top) or oth_top[i] != n}
        | {n for n in oth_top if n not in ref_top}
    )

    def separated(ix: SobolIndices, a: str, b: str) -> bool:
        ia, ib = ix.names.index(a), ix.names.index(b)
        lo_a, hi_a = ix.st[ia] - ix.st_conf[ia], ix.st[ia] + ix.st_conf[ia]
        lo_b, hi_b = ix.st[ib] - ix.st_conf[ib], ix.st[ib] + ix.st_conf[ib]
        return bool(hi_a < lo_b or hi_b < lo_a)

    swaps = []
    for i, a in enumerate(ref_top):
        for b in ref_top[i + 1 :]:
            if b in oth_rank and oth_rank.index(b) < oth_rank.index(a):
                resolved = separated(reference, a, b) and separated(other, a, b)
                swaps.append({"pair": [a, b], "resolved": resolved})
    return {
        "k": k,
        "reference_top": ref_top,
        "other_top": oth_top,
        "changed": ref_top != oth_top,
        "affected": affected,
        "swaps": swaps,
        "any_resolved_swap": any(s["resolved"] for s in swaps),
        "reference_ranking": ref_rank,
        "other_ranking": oth_rank,
    }
