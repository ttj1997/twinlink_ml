r"""Steady state of a learned sequence model as a fixed point of its horizon map.

A composite network of the Befund 94 pipeline (one network per target) maps a window -- history of
``L`` blocks of exogenous inputs and of its own target, future of ``H`` blocks of exogenous inputs
-- to the predicted target over the horizon. With constant exogenous inputs ``u`` over history and
future and a constant target history ``x``, the predicted value at the last step of the horizon is a
scalar function ``f_u(x)``. The steady state of the learned model at ``u`` is the fixed point

.. math:: r_u(x^*) = f_u(x^*) - x^* = 0 .

Why not the plain rollout: the slowest moisture mode of the grey-box model has a time constant of
1600-2900 s, the horizon H = 36 covers 900 s; a constant-input rollout from another operating point
ends far from the steady state (scratch probe of this study, docs/30 section 3). The fixed point
needs no history from the plant; for the grey-box model it is the steady state of the solver.

Criterion (fixed before any run of the study; the docstring of ``scripts/analyze_surrogate_env.py``
repeats it)
------------------------------------------------------------------------------------------------
Per head and operating point, in standardised units of the head's target:

1. **Scan (multi-start of the root search).** ``r`` on ``n_grid`` = 65 equidistant candidates over
   ``[-5, 5]`` (the training data of the head standardised; the range covers it with room).
   A sign change is a pair of neighbours with ``r >= 0`` then ``r < 0`` (stable: the model relaxes
   towards the root) or ``r < 0`` then ``r >= 0`` (unstable).
2. **Unique and stable.** Exactly one sign change on the scan, and it is stable. Otherwise the
   status is ``no_root`` (no sign change), ``multiple_roots`` (more than one) or ``unstable`` (one,
   unstable).
3. **Refinement.** A second scan of 65 candidates inside the bracket, then linear interpolation
   inside the bracket of the second scan.
4. **Converged.** ``|r(x*)| <= residual_tolerance`` = 1e-3 (standardised) at the interpolated root,
   evaluated by one more model call; otherwise ``not_converged``.

Only status ``ok`` gives a value. A point without it is counted and, for the controllers, is not
admissible (the environment treats it as a solver failure would be treated on the grey-box model).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "FIXED_POINT_STATUSES",
    "FixedPointCriterion",
    "FixedPointResult",
    "solve_fixed_points",
]

FIXED_POINT_STATUSES: tuple[str, ...] = (
    "ok",
    "no_root",
    "multiple_roots",
    "unstable",
    "not_converged",
)


@dataclass(frozen=True, slots=True)
class FixedPointCriterion:
    """Settings of the fixed-point search (standardised units); fixed before the study runs."""

    lower: float = -5.0
    upper: float = 5.0
    n_grid: int = 65
    residual_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError("lower must be below upper")
        if self.n_grid < 3:
            raise ValueError("n_grid must be at least 3")
        if self.residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be positive")


@dataclass(frozen=True, slots=True)
class FixedPointResult:
    """Per point and head: root (standardised, NaN unless ``ok``), status, residual."""

    root: np.ndarray
    status: np.ndarray
    residual: np.ndarray
    n_sign_changes: np.ndarray

    @property
    def ok(self) -> np.ndarray:
        return self.status == "ok"


ResidualFunction = Callable[[np.ndarray], np.ndarray]
"""``candidates (n_points, n_candidates, n_heads) -> residuals`` of the same shape."""


def _sign_changes(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable (``+ -> -``) and unstable (``- -> +``) sign changes between neighbours (last axis)."""
    nonneg = residual >= 0.0
    stable = nonneg[..., :-1] & ~nonneg[..., 1:]
    unstable = ~nonneg[..., :-1] & nonneg[..., 1:]
    return stable, unstable


def solve_fixed_points(
    residual_fn: ResidualFunction,
    n_points: int,
    n_heads: int,
    criterion: FixedPointCriterion | None = None,
) -> FixedPointResult:
    """Fixed point of every head at every point, by the criterion of the module docstring."""
    crit = criterion if criterion is not None else FixedPointCriterion()
    grid = np.linspace(crit.lower, crit.upper, crit.n_grid)
    candidates = np.broadcast_to(grid[None, :, None], (n_points, crit.n_grid, n_heads)).copy()
    r = residual_fn(candidates)  # (p, g, h)
    r_t = np.moveaxis(r, 1, -1)  # (p, h, g)
    stable, unstable = _sign_changes(r_t)
    n_stable = stable.sum(axis=-1)
    n_unstable = unstable.sum(axis=-1)
    n_changes = n_stable + n_unstable

    status = np.full((n_points, n_heads), "ok", dtype=object)
    status[n_changes == 0] = "no_root"
    status[n_changes > 1] = "multiple_roots"
    status[(n_changes == 1) & (n_unstable == 1)] = "unstable"
    candidate_ok = status == "ok"

    # bracket of the single stable change (index of its left neighbour); 0 where not ok
    left_index = np.where(candidate_ok, np.argmax(stable, axis=-1), 0)
    step = grid[1] - grid[0]
    lo = grid[left_index]
    fine = lo[:, None, :] + np.linspace(0.0, step, crit.n_grid)[None, :, None]  # (p, g, h)
    r_fine = np.moveaxis(residual_fn(fine), 1, -1)  # (p, h, g)
    fine_t = np.moveaxis(fine, 1, -1)
    stable_f, _ = _sign_changes(r_fine)
    # the coarse bracket has r(lo) >= 0 > r(lo + step); the fine scan holds at least one change
    has_fine = stable_f.any(axis=-1)
    j = np.argmax(stable_f, axis=-1)
    take = np.take_along_axis
    x0 = take(fine_t, j[..., None], -1)[..., 0]
    x1 = take(fine_t, (j + 1)[..., None], -1)[..., 0]
    r0 = take(r_fine, j[..., None], -1)[..., 0]
    r1 = take(r_fine, (j + 1)[..., None], -1)[..., 0]
    denominator = np.where(r0 - r1 == 0.0, 1.0, r0 - r1)
    root = x0 + r0 * (x1 - x0) / denominator
    root = np.where(candidate_ok & has_fine, root, np.nan)

    check_input = np.where(np.isfinite(root), root, 0.0)[:, None, :]
    residual = residual_fn(check_input)[:, 0, :]
    residual = np.where(np.isfinite(root), residual, np.nan)
    converged = np.abs(residual) <= crit.residual_tolerance
    status[candidate_ok & ~(has_fine & converged)] = "not_converged"
    root = np.where(status == "ok", root, np.nan)
    return FixedPointResult(
        root=root,
        status=status,
        residual=residual,
        n_sign_changes=n_changes,
    )
