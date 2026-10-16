r"""Ideal storage (linear programme over the horizon) and the greedy rule without foresight.

Both act on :class:`~twinlink_ml.storage_timeslice.table.LevelGroups`: the hot utility of a step is
a maximum of linear pieces in the charge and discharge duties, so the storage problem needs no
problem table inside the optimiser.

Storage model (the same for the programme and the rule)
    ``s_{t+1} = (1 - lambda dt) s_t + (q_c,t - q_d,t) dt`` with the stored energy ``s`` in kWh,
    ``lambda`` the loss rate per hour, ``0 <= s <= C`` (the capacities of all levels together),
    no limit on the charge or discharge power (ideal), the horizon cyclic (``s`` at the end equals
    ``s`` at the start). Charging and discharging may coincide; with losses that only wastes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import linprog

from .table import LevelGroups

__all__ = ["RuleResult", "StorageResult", "simulate_rule", "solve_ideal_storage"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StorageResult:
    """Optimal (or simulated) storage operation over one horizon.

    :ivar energy_kwh: sum of the hot utility over the horizon (kWh).
    :ivar utility_kw: hot utility per step.
    :ivar charge_kw, discharge_kw, stored_kwh: ``(steps, levels)``; ``stored_kwh`` is the content at
        the START of each step.
    :ivar loss_kwh: energy lost over the horizon.
    """

    energy_kwh: float
    utility_kw: FloatArray
    charge_kw: FloatArray
    discharge_kw: FloatArray
    stored_kwh: FloatArray
    loss_kwh: float

    def balance_residual_kwh(self, dt_h: float) -> float:
        """Closure of the cyclic energy balance, charged - discharged - lost (should be ~0)."""
        charged = float(np.sum(self.charge_kw)) * dt_h
        discharged = float(np.sum(self.discharge_kw)) * dt_h
        return charged - discharged - self.loss_kwh


def _direct(groups: LevelGroups, dt_h: float) -> StorageResult:
    n_steps, n_levels = groups.offset_kw.shape[0], groups.charge.shape[1]
    zero = np.zeros((n_steps, n_levels))
    utility = groups.utility_kw(zero, zero)
    return StorageResult(
        float(np.sum(utility)) * dt_h, utility, zero, zero.copy(), zero.copy(), 0.0
    )


def solve_ideal_storage(
    groups: LevelGroups, *, dt_h: float, capacity_kwh: float, loss_per_h: float
) -> StorageResult:
    """Minimum of ``sum_t U_t dt`` over the horizon for an ideal storage of capacity ``C``.

    Capacity 0 returns the direct hot utility exactly (no solver call).
    """
    if capacity_kwh < 0.0 or loss_per_h < 0.0 or dt_h <= 0.0:
        raise ValueError("capacity, loss rate must be >= 0 and dt > 0")
    if capacity_kwh == 0.0:
        return _direct(groups, dt_h)
    offset = groups.offset_kw
    n, n_groups = offset.shape
    k = groups.charge.shape[1]
    keep = 1.0 - loss_per_h * dt_h
    # variable layout: qc (n*k) | qd (n*k) | U (n) | s (n*k), index = t*k + j
    n_qc, n_qd, n_u = n * k, n * k, n
    i_qd, i_u, i_s = n_qc, n_qc + n_qd, n_qc + n_qd + n_u
    n_var = i_s + n * k

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    row = 0
    t_idx = np.arange(n)
    # pieces without a storage term (and the zero floor) only raise the lower bound of U
    constant = ~(groups.charge.any(axis=1) | groups.discharge.any(axis=1))
    floor = np.maximum(0.0, np.max(offset[:, constant], axis=1)) if constant.any() else np.zeros(n)
    for g in range(n_groups):
        active = np.isfinite(offset[:, g])
        a, b = groups.charge[g], groups.discharge[g]
        if not np.any(active) or (not np.any(a) and not np.any(b)):
            continue
        a_cols, b_cols = np.nonzero(a)[0], np.nonzero(b)[0]
        for t in t_idx[active]:
            # M_g + a.qc - b.qd - U <= 0
            rows.append(row)
            cols.append(i_u + int(t))
            vals.append(-1.0)
            for j in a_cols:
                rows.append(row)
                cols.append(int(t) * k + int(j))
                vals.append(1.0)
            for j in b_cols:
                rows.append(row)
                cols.append(i_qd + int(t) * k + int(j))
                vals.append(-1.0)
            rhs.append(-float(offset[t, g]))
            row += 1
    # capacity: sum_j s_tj <= C
    for t in range(n):
        for j in range(k):
            rows.append(row)
            cols.append(i_s + t * k + j)
            vals.append(1.0)
        rhs.append(capacity_kwh)
        row += 1
    a_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(row, n_var))

    e_rows: list[int] = []
    e_cols: list[int] = []
    e_vals: list[float] = []
    erow = 0
    for t in range(n):
        nxt = (t + 1) % n
        for j in range(k):
            # s_{t+1} - keep s_t - dt (qc - qd) = 0
            e_rows += [erow, erow, erow, erow]
            e_cols += [i_s + nxt * k + j, i_s + t * k + j, t * k + j, i_qd + t * k + j]
            e_vals += [1.0, -keep, -dt_h, dt_h]
            erow += 1
    a_eq = sparse.csr_matrix((e_vals, (e_rows, e_cols)), shape=(erow, n_var))

    cost = np.zeros(n_var)
    cost[i_u : i_u + n_u] = dt_h
    lower = np.zeros(n_var)
    upper = np.full(n_var, np.inf)
    lower[i_u : i_u + n_u] = floor
    upper[i_s:] = capacity_kwh
    result = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.asarray(rhs),
        A_eq=a_eq,
        b_eq=np.zeros(erow),
        bounds=np.column_stack([lower, upper]),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"storage programme failed: {result.message}")
    x = result.x
    qc = x[:n_qc].reshape(n, k)
    qd = x[i_qd:i_u].reshape(n, k)
    utility = x[i_u:i_s]
    stored = x[i_s:].reshape(n, k)
    loss = float(np.sum(stored)) * loss_per_h * dt_h
    return StorageResult(float(np.sum(utility)) * dt_h, utility, qc, qd, stored, loss)


@dataclass(frozen=True, slots=True)
class RuleResult:
    """Greedy rule over the capacities in one call (single level).

    :ivar energy_kwh: ``(capacities,)`` energy of the last cycle.
    :ivar converged: ``(capacities,)`` whether the content at the start of the last cycle equals
        the content at its end (periodic steady state reached).
    :ivar charge_kw, discharge_kw, stored_kwh: ``(capacities, steps)`` of the last cycle.
    :ivar loss_kwh: ``(capacities,)``.
    """

    energy_kwh: FloatArray
    converged: NDArray[np.bool_]
    charge_kw: FloatArray
    discharge_kw: FloatArray
    stored_kwh: FloatArray
    loss_kwh: FloatArray


def simulate_rule(
    groups: LevelGroups,
    *,
    dt_h: float,
    capacities_kwh: FloatArray,
    loss_per_h: float,
    max_cycles: int = 60,
) -> RuleResult:
    """Greedy "charge surplus, discharge on demand" for ONE level, cyclic steady state.

    Per step, with the pieces ``M_0`` (constant), ``M_1`` (+q_c) and ``M_2`` (+q_c - q_d) of the
    level (:func:`~twinlink_ml.storage_timeslice.table.level_groups` with one level):

    1. discharge ``q_d = clip(M_2 - max(0, M_0, M_1), 0, content/dt)``: exactly as much as lowers
       the hot utility (not more);
    2. charge the surplus that does not raise the utility,
       ``q_c = clip(max(0, M_0) - max(M_1, M_2 - q_d), 0, room/dt)``.

    The horizon is repeated until the content at the start of a cycle repeats (at most
    ``max_cycles``); the last cycle is returned.
    """
    if groups.charge.shape[1] != 1:
        raise ValueError("the rule is defined for a single storage level")
    caps = np.asarray(capacities_kwh, dtype=np.float64)
    n = groups.offset_kw.shape[0]
    m = groups.offset_kw
    keep = 1.0 - loss_per_h * dt_h
    # groups of one level (two sorted points): g = 0 constant, g = 1 (+q_c), g = 2 (+q_c - q_d);
    # g = 1 holds the boundaries between the discharge and the charge temperature (at least the
    # boundary of the charge stream itself) and is -inf only if there is none.
    m0 = np.maximum(0.0, m[:, 0])
    m1 = m[:, 1]
    m2 = m[:, 2]
    s = np.zeros(len(caps))
    converged = np.zeros(len(caps), dtype=bool)
    qc_log = np.zeros((len(caps), n))
    qd_log = np.zeros((len(caps), n))
    s_log = np.zeros((len(caps), n))
    energy = np.zeros(len(caps))
    loss = np.zeros(len(caps))
    for _ in range(max_cycles):
        start = s.copy()
        energy[:] = 0.0
        loss[:] = 0.0
        for t in range(n):
            content = keep * s
            loss += s * (1.0 - keep)
            s_log[:, t] = s
            q_d = np.clip(m2[t] - max(m0[t], m1[t]), 0.0, content / dt_h)
            after = np.maximum(m1[t], m2[t] - q_d)
            q_c = np.clip(m0[t] - after, 0.0, np.maximum(caps - content + q_d * dt_h, 0.0) / dt_h)
            utility = np.maximum(m0[t], np.maximum(m1[t] + q_c, m2[t] + q_c - q_d))
            energy += utility * dt_h
            qc_log[:, t] = q_c
            qd_log[:, t] = q_d
            s = content + (q_c - q_d) * dt_h
        converged = np.abs(s - start) <= 1e-6 * np.maximum(1.0, caps)
        if bool(np.all(converged)):
            break
    return RuleResult(energy, converged, qc_log, qd_log, s_log, loss)
