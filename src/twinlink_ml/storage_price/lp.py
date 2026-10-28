r"""Linear programme of pump and storage over a window, with power limits (MPC and hindsight).

Same pieces and objective as :func:`twinlink_ml.heatpump.storage.solve_foresight` (Befund 101),
plus the limits of this study and an open end instead of the cyclic horizon:

.. math::
   \min \sum_k \Delta t\,(c_U U_k + c_{el,k} Q_k / \mathrm{COP}) - v_T\, s_H

   U_k \ge M_{k,g} + \alpha_g Q_k + \beta_g q_{c,k} - \gamma_g q_{d,k},\quad U_k \ge 0,

   s_{k+1} = (1 - \lambda\Delta t) s_k + (q_{c,k} - q_{d,k}) \Delta t,\quad s_0 \text{ given},

   0 \le Q \le Q_{max},\ 0 \le q_c \le P_c,\ 0 \le q_d \le P_d,\ 0 \le s \le C.

``cyclic=True`` (``s_H = s_0`` free, no terminal value) reproduces ``solve_foresight`` when the
power limits are not binding; the tests check that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import linprog

__all__ = ["WindowSolution", "solve_window"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WindowSolution:
    """Plan over the window: per step duties and the content at the START of each step."""

    objective: float
    heat_kw: FloatArray
    charge_kw: FloatArray
    discharge_kw: FloatArray
    utility_kw: FloatArray
    stored_kwh: FloatArray
    final_kwh: float


def solve_window(
    offset_kw: FloatArray,
    alpha: FloatArray,
    beta: FloatArray,
    gamma: FloatArray,
    *,
    electricity: FloatArray,
    heat_value: float,
    cop: float,
    q_max_kw: float,
    charge_max_kw: float,
    discharge_max_kw: float,
    capacity_kwh: float,
    keep: float,
    dt_h: float,
    initial_kwh: float = 0.0,
    terminal_value: float = 0.0,
    cyclic: bool = False,
) -> WindowSolution:
    """Optimal plan over the rows of ``offset_kw`` (``(steps, groups)``, ``-inf`` = no piece)."""
    n = int(offset_kw.shape[0])
    elec = np.asarray(electricity, dtype=np.float64)
    if elec.shape != (n,):
        raise ValueError("electricity must have one value per step")
    i_q, i_c, i_d, i_u, i_s = 0, n, 2 * n, 3 * n, 4 * n
    n_var = 5 * n  # s block: s_1..s_n (open end) or s_0..s_{n-1} (cyclic)
    coeff = np.stack([alpha, beta, gamma], axis=1)
    has = np.any(np.abs(coeff) > 0.0, axis=1)
    finite = np.isfinite(offset_kw)
    floor = np.zeros(n)
    const = ~has
    if np.any(const):
        floor = np.maximum(
            0.0, np.max(np.where(finite[:, const], offset_kw[:, const], -np.inf), axis=1)
        )
    rows_l, cols_l, vals_l, rhs_l = [], [], [], []
    row0 = 0
    t_all = np.arange(n)
    for g in np.nonzero(has)[0]:
        ts = t_all[finite[:, g]]
        if len(ts) == 0:
            continue
        r = row0 + np.arange(len(ts))
        a, b, c = coeff[g]
        for col0, val in ((i_q, a), (i_c, b), (i_d, -c), (i_u, -1.0)):
            if val != 0.0:
                rows_l.append(r)
                cols_l.append(col0 + ts)
                vals_l.append(np.full(len(ts), val))
        rhs_l.append(-offset_kw[ts, g])
        row0 += len(ts)
    a_ub = sparse.csr_matrix(
        (np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(row0, n_var),
    )
    b_ub = np.concatenate(rhs_l)
    b_eq = np.zeros(n)
    if cyclic:
        # s_{(t+1) mod n} - keep s_t - dt qc_t + dt qd_t = 0
        nxt = (t_all + 1) % n
        e_cols = np.stack([i_s + nxt, i_s + t_all, i_c + t_all, i_d + t_all], axis=1).ravel()
        e_vals = np.tile([1.0, -keep, -dt_h, dt_h], n)
        e_rows = np.repeat(t_all, 4)
    else:
        # s_{t+1} - keep s_t - dt qc_t + dt qd_t = 0, s_0 given (moved to the right-hand side)
        rows, cols, vals = [], [], []
        for t in range(n):
            rows += [t, t, t]
            cols += [i_s + t, i_c + t, i_d + t]
            vals += [1.0, -dt_h, dt_h]
            if t > 0:
                rows.append(t)
                cols.append(i_s + t - 1)
                vals.append(-keep)
        e_rows, e_cols, e_vals = np.array(rows), np.array(cols), np.array(vals)
        b_eq[0] = keep * initial_kwh
    a_eq = sparse.csr_matrix((e_vals, (e_rows, e_cols)), shape=(n, n_var))
    cost = np.zeros(n_var)
    cost[i_q:i_c] = elec / cop * dt_h
    cost[i_u:i_s] = heat_value * dt_h
    if not cyclic:
        cost[i_s + n - 1] -= terminal_value
    lower = np.zeros(n_var)
    upper = np.full(n_var, np.inf)
    upper[i_q:i_c] = q_max_kw
    upper[i_c:i_d] = charge_max_kw
    upper[i_d:i_u] = discharge_max_kw
    lower[i_u:i_s] = floor
    upper[i_s:] = capacity_kwh
    res = linprog(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=np.column_stack([lower, upper]),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"window programme failed: {res.message}")
    x = res.x
    s_block = x[i_s:]
    if cyclic:
        stored, final = s_block.copy(), float(s_block[0])
    else:
        stored = np.concatenate([[initial_kwh], s_block[:-1]])
        final = float(s_block[-1])
    return WindowSolution(
        objective=float(res.fun),
        heat_kw=x[i_q:i_c],
        charge_kw=x[i_c:i_d],
        discharge_kw=x[i_d:i_u],
        utility_kw=x[i_u:i_s],
        stored_kwh=stored,
        final_kwh=final,
    )
