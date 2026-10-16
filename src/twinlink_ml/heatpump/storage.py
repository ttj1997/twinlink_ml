r"""Stage B: heat pump plus storage under a time-varying electricity value.

Both pseudo-stream pairs sit in one problem table (:func:`~twinlink_ml.storage_timeslice.table.
level_groups` with two "levels": 0 = heat pump, 1 = storage). With ``Q`` the condenser heat,
``q_c``/``q_d`` the storage charge/discharge duty and ``r = 1 - 1/COP``, the hot utility of a step
is

.. math:: U = \max\bigl(0,\ \max_g (M_g + \alpha_g Q + \beta_g q_c - \gamma_g q_d)\bigr),\quad
          \alpha_g = a_{g0} r - b_{g0},\ \beta_g = a_{g1},\ \gamma_g = b_{g1}.

Storage as in storage test 1: ``s_{t+1} = (1 - lambda dt) s_t + (q_c - q_d) dt``, ``0 <= s <= C``,
no power limit, cyclic over the horizon. The heat pump: ``0 <= Q <= Q_max``. Objective of a
criterion: ``J = sum_t dt (c_U U_t + c_el,t Q_t / COP)``.

* :func:`solve_foresight` -- linear programme with PERFECT FORESIGHT over the whole horizon
  (HiGHS); optionally "daily state-neutral" (content at every midnight equal: a 24 h foresight
  that cannot shift energy between days).
* :func:`simulate_rule` -- rules WITHOUT FORESIGHT: per step the small LP
  ``min c_U U + p_t Q - v_t (q_c - q_d)`` (``p_t = c_el,t / COP``) with a value ``v_t`` of stored
  heat that uses only the present and the past, solved EXACTLY by vertex enumeration
  (:func:`step_optimum`); run cyclically to the periodic steady state. Rules:

  ``myopic``    ``v = eps * c_U``: stores only heat that is free at the moment (as storage test 1);
  ``gas_value`` ``v = (1 - eps) c_U``: fills the storage whenever heat-pump heat is cheaper than
                gas heat, discharges whenever gas would be burnt ("pump first, storage, gas last");
  ``trailing``  ``v_t = min(c_U, cbar_t / COP)`` with ``cbar_t`` the mean electricity value of the
                past 24 h: charges with the pump when power is cheaper than its recent mean,
                discharges when it is dearer (merit order with a backward-looking threshold).

Charging and discharging in the same step is never better (it only moves heat down the
temperature scale, which the table does for free), so the step LP splits into a charge case and a
discharge case of two variables each.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import linprog

from ..storage_timeslice.table import Cascade, level_groups
from .signal import trailing_mean

__all__ = [
    "RULES",
    "CoupledPieces",
    "DispatchResult",
    "RuleName",
    "coupled_pieces",
    "simulate_rule",
    "solve_foresight",
    "step_optimum",
]

FloatArray = NDArray[np.float64]
RuleName = Literal["myopic", "gas_value", "trailing"]
RULES: tuple[RuleName, ...] = ("myopic", "gas_value", "trailing")
_BIG_NEGATIVE = -1.0e12
_EPS_VALUE = 1e-6
_TOL = 1e-9


@dataclass(frozen=True, slots=True)
class CoupledPieces:
    """Pieces of ``U`` for one heat-pump pair and one storage level.

    :ivar offset_kw: ``(steps, groups)``, ``-inf`` where a group has no boundary.
    :ivar alpha, beta, gamma: ``(groups,)`` coefficients of ``Q``, ``q_c``, ``-q_d``.
    :ivar cop: heating COP.
    """

    offset_kw: FloatArray
    alpha: FloatArray
    beta: FloatArray
    gamma: FloatArray
    cop: float

    @property
    def n_steps(self) -> int:
        return int(self.offset_kw.shape[0])

    def utility_kw(self, q: FloatArray, q_c: FloatArray, q_d: FloatArray) -> FloatArray:
        """``U`` per step for arrays of shape ``(..., steps)``."""
        value = (
            self.offset_kw
            + np.asarray(q)[..., None] * self.alpha
            + np.asarray(q_c)[..., None] * self.beta
            - np.asarray(q_d)[..., None] * self.gamma
        )
        return np.maximum(0.0, np.max(value, axis=-1))


def coupled_pieces(
    cascade: Cascade,
    *,
    evaporator_shifted: float,
    condenser_shifted: float,
    cop: float,
    charge_shifted: float,
    discharge_shifted: float,
) -> CoupledPieces:
    """Pieces for the pair and the storage level (all four temperatures must be boundaries)."""
    if cop <= 1.0:
        raise ValueError("cop must exceed 1")
    groups = level_groups(
        cascade,
        np.array([evaporator_shifted, charge_shifted]),
        np.array([condenser_shifted, discharge_shifted]),
    )
    r = 1.0 - 1.0 / cop
    return CoupledPieces(
        offset_kw=groups.offset_kw,
        alpha=groups.charge[:, 0] * r - groups.discharge[:, 0],
        beta=groups.charge[:, 1].copy(),
        gamma=groups.discharge[:, 1].copy(),
        cop=float(cop),
    )


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Operation over one horizon (one capacity).

    :ivar objective: ``J`` over the horizon (kg CO2 or EUR).
    :ivar heat_kw, charge_kw, discharge_kw, utility_kw, stored_kwh: per step (content at the START).
    :ivar loss_kwh: storage loss over the horizon.
    :ivar converged: periodic steady state reached (rules; always True for the programme).
    """

    objective: float
    heat_kw: FloatArray
    charge_kw: FloatArray
    discharge_kw: FloatArray
    utility_kw: FloatArray
    stored_kwh: FloatArray
    loss_kwh: float
    converged: bool = True

    def balance_residual_kwh(self, dt_h: float) -> float:
        """Charged - discharged - lost (cyclic: should be ~0)."""
        return float((np.sum(self.charge_kw) - np.sum(self.discharge_kw)) * dt_h - self.loss_kwh)

    def work_kwh(self, dt_h: float, cop: float) -> float:
        return float(np.sum(self.heat_kw) * dt_h / cop)


def _objective(
    pieces: CoupledPieces,
    heat_value: float,
    electricity: FloatArray,
    dt_h: float,
    q: FloatArray,
    u: FloatArray,
) -> float:
    return float(np.sum(heat_value * u + electricity * q / pieces.cop) * dt_h)


def solve_foresight(
    pieces: CoupledPieces,
    *,
    q_max_kw: float,
    capacity_kwh: float,
    loss_per_h: float,
    dt_h: float,
    heat_value: float,
    electricity: FloatArray,
    steps_per_day: int | None = None,
) -> DispatchResult:
    """Linear programme with perfect foresight over the horizon (cyclic).

    :param electricity: electricity value per step (EUR or kg per kWh electric).
    :param steps_per_day: if given, the content at every multiple of it (midnight) must equal the
        content at the start ("daily state-neutral", 24 h foresight inside a longer horizon).
    """
    if capacity_kwh < 0.0 or q_max_kw < 0.0 or loss_per_h < 0.0 or dt_h <= 0.0:
        raise ValueError("capacity, size, loss must be >= 0 and dt > 0")
    n = pieces.n_steps
    elec = np.asarray(electricity, dtype=np.float64)
    if elec.shape != (n,):
        raise ValueError("electricity must have one value per step")
    keep = 1.0 - loss_per_h * dt_h
    i_q, i_c, i_d, i_u, i_s = 0, n, 2 * n, 3 * n, 4 * n
    n_var = 5 * n
    offset = pieces.offset_kw
    coeff = np.stack([pieces.alpha, pieces.beta, pieces.gamma], axis=1)
    has = np.any(np.abs(coeff) > 0.0, axis=1)
    finite = np.isfinite(offset)
    const_rows = ~has
    floor = np.zeros(n)
    if np.any(const_rows):
        floor = np.maximum(
            0.0, np.max(np.where(finite[:, const_rows], offset[:, const_rows], -np.inf), axis=1)
        )
    rows_l: list[NDArray[np.int64]] = []
    cols_l: list[NDArray[np.int64]] = []
    vals_l: list[FloatArray] = []
    rhs_l: list[FloatArray] = []
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
        rhs_l.append(-offset[ts, g])
        row0 += len(ts)
    if row0:
        a_ub = sparse.csr_matrix(
            (np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
            shape=(row0, n_var),
        )
        b_ub = np.concatenate(rhs_l)
    else:
        a_ub, b_ub = None, None
    # dynamics s_{t+1} - keep s_t - dt qc_t + dt qd_t = 0 (cyclic)
    nxt = (t_all + 1) % n
    e_rows = np.repeat(t_all, 4)
    e_cols = np.stack([i_s + nxt, i_s + t_all, i_c + t_all, i_d + t_all], axis=1).ravel()
    e_vals = np.tile([1.0, -keep, -dt_h, dt_h], n)
    eq_rows = [e_rows]
    eq_cols = [e_cols]
    eq_vals = [e_vals]
    n_eq = n
    if steps_per_day is not None and n > steps_per_day:
        mids = np.arange(steps_per_day, n, steps_per_day)
        k = len(mids)
        eq_rows.append(np.repeat(n + np.arange(k), 2))
        eq_cols.append(np.stack([i_s + mids, np.full(k, i_s)], axis=1).ravel())
        eq_vals.append(np.tile([1.0, -1.0], k))
        n_eq += k
    a_eq = sparse.csr_matrix(
        (np.concatenate(eq_vals), (np.concatenate(eq_rows), np.concatenate(eq_cols))),
        shape=(n_eq, n_var),
    )
    cost = np.zeros(n_var)
    cost[i_q:i_c] = elec / pieces.cop * dt_h
    cost[i_u:i_s] = heat_value * dt_h
    lower = np.zeros(n_var)
    upper = np.full(n_var, np.inf)
    upper[i_q:i_c] = q_max_kw
    lower[i_u:i_s] = floor
    upper[i_s:] = capacity_kwh
    res = linprog(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.zeros(n_eq),
        bounds=np.column_stack([lower, upper]),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"foresight programme failed: {res.message}")
    x = res.x
    q, qc, qd, s = x[i_q:i_c], x[i_c:i_d], x[i_d:i_u], x[i_s:]
    # recompute U from the pieces (the LP's U is only bounded from below by them)
    u = pieces.utility_kw(q, qc, qd)
    loss = float(np.sum(s)) * loss_per_h * dt_h
    return DispatchResult(
        objective=_objective(pieces, heat_value, elec, dt_h, q, u),
        heat_kw=q,
        charge_kw=qc,
        discharge_kw=qd,
        utility_kw=u,
        stored_kwh=s,
        loss_kwh=loss,
    )


# ------------------------------------------------------------------------- exact step optimum


@dataclass(frozen=True, slots=True)
class _CaseTable:
    """State-independent data of one case (charge or discharge) for all steps."""

    m: FloatArray  # (steps, L) offsets incl. the zero floor
    a: FloatArray  # (L,) coefficient of Q
    b: FloatArray  # (L,) coefficient of y
    w: FloatArray  # (steps,) linear objective coefficient of y (per kW)
    static_q: FloatArray  # (steps, S)
    static_y: FloatArray  # (steps, S)
    static_f: FloatArray  # (steps, S), +inf where statically infeasible
    pairs: tuple[tuple[int, int], ...]  # breaklines (i, j)


def _lines(
    pieces: CoupledPieces, case: Literal["charge", "discharge"]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    m = np.where(np.isfinite(pieces.offset_kw), pieces.offset_kw, _BIG_NEGATIVE)
    m = np.concatenate([np.zeros((pieces.n_steps, 1)), m], axis=1)
    a = np.concatenate([[0.0], pieces.alpha])
    b = np.concatenate([[0.0], pieces.beta if case == "charge" else -pieces.gamma])
    return m, a, b


def _evaluate(
    m: FloatArray,
    a: FloatArray,
    b: FloatArray,
    c_u: float | FloatArray,
    p: FloatArray,
    w: FloatArray,
    q: FloatArray,
    y: FloatArray,
) -> FloatArray:
    """``F = c_U max(0, max_i(m_i + a_i Q + b_i y)) + p Q + w y`` (``m`` broadcasts)."""
    val = m + q[..., None] * a + y[..., None] * b
    return np.asarray(c_u * np.max(val, axis=-1) + p * q + w * y)  # line 0 is the floor


def _case_table(
    pieces: CoupledPieces,
    case: Literal["charge", "discharge"],
    q_max: float,
    c_u: float,
    p: FloatArray,
    w: FloatArray,
) -> _CaseTable:
    m, a, b = _lines(pieces, case)
    n = m.shape[0]
    n_lines = len(a)
    pairs = tuple(
        (i, j)
        for i, j in combinations(range(n_lines), 2)
        if abs(a[i] - a[j]) > _TOL or abs(b[i] - b[j]) > _TOL
    )
    qs: list[FloatArray] = []
    ys: list[FloatArray] = []
    # breakline (i,j): da Q + db y = dm, with da = a_i - a_j, db = b_i - b_j, dm = m_j - m_i
    for (i, j), (k, h) in combinations(pairs, 2):
        da1, db1, dm1 = a[i] - a[j], b[i] - b[j], m[:, j] - m[:, i]
        da2, db2, dm2 = a[k] - a[h], b[k] - b[h], m[:, h] - m[:, k]
        det = da1 * db2 - da2 * db1
        if abs(det) <= _TOL:
            continue
        qs.append((dm1 * db2 - dm2 * db1) / det)
        ys.append((da1 * dm2 - da2 * dm1) / det)
    for i, j in pairs:
        da, db, dm = a[i] - a[j], b[i] - b[j], m[:, j] - m[:, i]
        if abs(db) > _TOL:  # with Q = 0 and Q = Q_max
            for qv in (0.0, q_max):
                qs.append(np.full(n, qv))
                ys.append((dm - da * qv) / db)
        if abs(da) > _TOL:  # with y = 0
            qs.append(dm / da)
            ys.append(np.zeros(n))
    for qv in (0.0, q_max):
        qs.append(np.full(n, qv))
        ys.append(np.zeros(n))
    sq = np.stack(qs, axis=1)
    sy = np.stack(ys, axis=1)
    scale = max(1.0, q_max)
    ok = (sq >= -1e-9 * scale) & (sq <= q_max + 1e-9 * scale) & (sy >= -1e-9 * scale)
    sq = np.clip(sq, 0.0, q_max)
    sy = np.maximum(sy, 0.0)
    f = _evaluate(m[:, None, :], a, b, c_u, p[:, None], w[:, None], sq, sy)
    f = np.where(ok, f, np.inf)
    return _CaseTable(m=m, a=a, b=b, w=w, static_q=sq, static_y=sy, static_f=f, pairs=pairs)


def _case_step(
    table: _CaseTable,
    t: int,
    q_max: float,
    y_max: FloatArray,
    c_u: float,
    p_t: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Best (F, Q, y) of one case at step ``t`` for every capacity (``y_max`` per capacity)."""
    ym = np.maximum(y_max, 0.0)
    feas = table.static_y[t][None, :] <= ym[:, None] + 1e-9 * np.maximum(1.0, ym[:, None])
    f_static = np.where(feas, table.static_f[t][None, :], np.inf)
    k = np.argmin(f_static, axis=1)
    rows = np.arange(len(ym))
    best_f = f_static[rows, k]
    best_q = table.static_q[t][k]
    best_y = np.minimum(table.static_y[t][k], ym)
    # dynamic candidates on the face y = y_max
    m_t = table.m[t]
    dq: list[FloatArray] = [np.zeros_like(ym), np.full_like(ym, q_max)]
    for i, j in table.pairs:
        da = table.a[i] - table.a[j]
        if abs(da) <= _TOL:
            continue
        db, dm = table.b[i] - table.b[j], m_t[j] - m_t[i]
        dq.append((dm - db * ym) / da)
    cq = np.clip(np.stack(dq, axis=1), 0.0, q_max)
    cy = np.broadcast_to(ym[:, None], cq.shape)
    f_dyn = _evaluate(m_t, table.a, table.b, c_u, np.asarray(p_t), table.w[t], cq, cy)
    kd = np.argmin(f_dyn, axis=1)
    fd = f_dyn[rows, kd]
    better = fd < best_f
    best_f = np.where(better, fd, best_f)
    best_q = np.where(better, cq[rows, kd], best_q)
    best_y = np.where(better, ym, best_y)
    return best_f, best_q, best_y


def step_optimum(
    pieces: CoupledPieces,
    *,
    t: int,
    q_max_kw: float,
    charge_max_kw: float,
    discharge_max_kw: float,
    heat_value: float,
    electricity_per_heat: float,
    stored_value: float,
) -> tuple[float, float, float, float]:
    """Exact optimum ``(F, Q, q_c, q_d)`` of the step LP (one capacity; for tests and checks)."""
    p = np.full(pieces.n_steps, electricity_per_heat)
    out = []
    cases: tuple[tuple[Literal["charge", "discharge"], float], ...] = (
        ("charge", -stored_value),
        ("discharge", stored_value),
    )
    for case, w in cases:
        table = _case_table(pieces, case, q_max_kw, heat_value, p, np.full(pieces.n_steps, w))
        y_max = np.array([charge_max_kw if case == "charge" else discharge_max_kw])
        f, q, y = _case_step(table, t, q_max_kw, y_max, heat_value, electricity_per_heat)
        out.append((float(f[0]), float(q[0]), float(y[0])))
    (fc, qc_, yc), (fd, qd_, yd) = out
    if fc <= fd:
        return fc, qc_, yc, 0.0
    return fd, qd_, 0.0, yd


def rule_values(
    rule: RuleName, heat_value: float, electricity: FloatArray, cop: float, window: int
) -> FloatArray:
    """Value ``v_t`` of stored heat of a rule (past and present only)."""
    n = len(electricity)
    if rule == "myopic":
        return np.full(n, _EPS_VALUE * heat_value)
    if rule == "gas_value":
        return np.full(n, (1.0 - _EPS_VALUE) * heat_value)
    if rule == "trailing":
        return np.minimum(heat_value, trailing_mean(electricity, window) / cop)
    raise ValueError(f"unknown rule {rule!r}")


@dataclass(frozen=True, slots=True)
class RuleRun:
    """One rule over several capacities (last cycle).

    :ivar objective: ``(capacities,)``; :ivar converged: ``(capacities,)``;
    :ivar heat_kw, charge_kw, discharge_kw, utility_kw, stored_kwh: ``(capacities, steps)``;
    :ivar loss_kwh: ``(capacities,)``; :ivar cycles: cycles run.
    """

    objective: FloatArray
    converged: NDArray[np.bool_]
    heat_kw: FloatArray
    charge_kw: FloatArray
    discharge_kw: FloatArray
    utility_kw: FloatArray
    stored_kwh: FloatArray
    loss_kwh: FloatArray
    cycles: int


def simulate_rule(
    pieces: CoupledPieces,
    rule: RuleName,
    *,
    q_max_kw: float,
    capacities_kwh: FloatArray,
    loss_per_h: float,
    dt_h: float,
    heat_value: float,
    electricity: FloatArray,
    window_steps: int = 96,
    max_cycles: int = 30,
    stored_value: FloatArray | None = None,
) -> RuleRun:
    """Rule without foresight, all capacities at once, cyclic steady state (see module).

    :param stored_value: if given, the value ``v_t`` per step overrides the named ``rule``
        (e.g. a constant ``kappa * c_U`` for the post-hoc rule-tuning analysis).
    """
    caps = np.asarray(capacities_kwh, dtype=np.float64)
    n = pieces.n_steps
    elec = np.asarray(electricity, dtype=np.float64)
    p = elec / pieces.cop
    if stored_value is None:
        v = rule_values(rule, heat_value, elec, pieces.cop, window_steps)
    else:
        v = np.broadcast_to(np.asarray(stored_value, dtype=np.float64), (n,)).copy()
    t_charge = _case_table(pieces, "charge", q_max_kw, heat_value, p, -v)
    t_dis = _case_table(pieces, "discharge", q_max_kw, heat_value, p, v)
    keep = 1.0 - loss_per_h * dt_h
    k = len(caps)
    s = np.zeros(k)
    q_log = np.zeros((k, n))
    c_log = np.zeros((k, n))
    d_log = np.zeros((k, n))
    s_log = np.zeros((k, n))
    loss = np.zeros(k)
    converged = np.zeros(k, dtype=bool)
    cycles = 0
    for _ in range(max_cycles):
        cycles += 1
        start = s.copy()
        loss[:] = 0.0
        for t in range(n):
            s_log[:, t] = s
            content = keep * s
            loss += s * (1.0 - keep)
            fc, qc_q, qc_y = _case_step(
                t_charge, t, q_max_kw, (caps - content) / dt_h, heat_value, float(p[t])
            )
            fd, qd_q, qd_y = _case_step(t_dis, t, q_max_kw, content / dt_h, heat_value, float(p[t]))
            use_c = fc <= fd
            q_t = np.where(use_c, qc_q, qd_q)
            c_t = np.where(use_c, qc_y, 0.0)
            d_t = np.where(use_c, 0.0, qd_y)
            q_log[:, t] = q_t
            c_log[:, t] = c_t
            d_log[:, t] = d_t
            s = np.clip(content + (c_t - d_t) * dt_h, 0.0, caps)
        converged = np.abs(s - start) <= 1e-6 * np.maximum(1.0, caps)
        if bool(np.all(converged)):
            break
    u = pieces.utility_kw(q_log, c_log, d_log)
    objective = np.sum(heat_value * u + elec * q_log / pieces.cop, axis=1) * dt_h
    return RuleRun(
        objective=objective,
        converged=converged,
        heat_kw=q_log,
        charge_kw=c_log,
        discharge_kw=d_log,
        utility_kw=u,
        stored_kwh=s_log,
        loss_kwh=loss,
        cycles=cycles,
    )
