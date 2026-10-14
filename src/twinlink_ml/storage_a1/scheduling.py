r"""Day-ahead scheduling of the preheater-loop flow: replay, rule, perfect-foresight optimum.

The question behind it (S2): how much steam could better operation of storage A1 save at all? Three
policies run on the SAME discrete model (explicit Euler step of ``dt_s`` seconds, extraction chain
of :mod:`twinlink_ml.storage_a1.model`):

* **R0** -- :func:`replay`: the logged preheater-loop flow, no extra compressor heat taken.
* **R1** -- :func:`rule_policy`: simple rule, maximum flow while ``T_S > T_S,min`` (minimum flow
  otherwise); optionally takes all discarded compressor heat that fits below ``T_S,max``.
* **R*** -- :func:`dp_policy`: optimum with perfect foresight over the whole day by backward
  dynamic programming. DP rather than LP because the state is one-dimensional (``T_S``) and the
  extraction duty is a non-linear function of state and flow (the effectiveness depends on the
  flow); a grid on ``T_S`` makes the optimum exact up to the grid, an LP would need a linearisation
  whose error is as large as the effects looked for.

Decisions per step: preheater-loop flow ``V_2`` (levels between ``v2_min`` and ``v2_max``) and the
fraction of the otherwise discarded compressor heat ``Q_dump(t)`` that is taken into the storage
loop (``Q_in = Q_in_base + u``, ``0 <= u <= Q_dump``). Cost: steam energy ``(Q_air - Q_out) dt``.
Constraints: ``T_S`` in ``[T_min, T_max]``, ``Q_out <= Q_air`` (the supply-air set point is kept:
the steam heater can only add heat), approach of both exchangers above its floor, and a terminal
condition ``T_S(end) >= T_ref`` (no gain by emptying the storage). Differences in the stored energy
at the end are booked in :func:`corrected_steam_kwh` at a credit factor of one (optimistic; see
there).

``Q_in_base(t)`` is the compressor heat the storage loop takes today, reconstructed from the
measured day for a REFERENCE volume (it does not depend on the volume of a hypothetical storage);
everything that depends on placeholders is a scenario, not a result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..evaluation.net_energy import GAS_CO2_KG_PER_KWH
from .balance import air_heating_kw
from .calibration import oracle_q_in_kw
from .config import StorageConfig, StorageSystemConfig
from .model import ExtractionChain
from .roles import PlantDay

__all__ = [
    "Policy",
    "ShiftProblem",
    "ShiftResult",
    "build_problem",
    "co2_kg",
    "corrected_steam_kwh",
    "dp_policy",
    "replay",
    "rule_policy",
    "simulate",
]

FloatArray = NDArray[np.float64]
Policy = Callable[[int, float], tuple[float, float]]
"""``policy(k, t_s) -> (v2_m3_per_h, dump_kw)`` for step ``k`` at storage temperature ``t_s``."""

INFEASIBLE_COST_KWH = 1.0e7


@dataclass(frozen=True, slots=True)
class ShiftProblem:
    """One day on the control grid: exogenous inputs per step and the decision limits."""

    cfg: StorageSystemConfig
    dt_s: float
    t_ambient: FloatArray
    v_storage_loop: FloatArray
    air_flow: FloatArray
    q_air_kw: FloatArray
    q_in_base_kw: FloatArray
    q_dump_kw: FloatArray
    v2_logged: FloatArray
    t_s0: float
    t_min: float
    t_max: float
    v2_min: float
    v2_max: float

    @property
    def n_steps(self) -> int:
        return len(self.t_ambient)

    @property
    def chain(self) -> ExtractionChain:
        return ExtractionChain.from_config(self.cfg)

    def step(
        self, k: int, t_s: FloatArray | float, v2: FloatArray | float, dump_kw: FloatArray | float
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """One model step (arrays broadcast): ``(t_next, q_out, q_in_effective, feasible)``.

        ``q_in_effective`` is the heat actually taken up: ``Q_in_base + dump`` curtailed so that the
        storage does not exceed ``T_max`` (the rest is rejected over the cooler).
        """
        chain = self.chain
        cap = chain.storage.heat_capacity_kj_per_k()
        t = np.asarray(t_s, dtype=np.float64)
        q_in = self.q_in_base_kw[k] + np.asarray(dump_kw, dtype=np.float64)
        args = (self.v_storage_loop[k], np.asarray(v2), self.air_flow[k], self.t_ambient[k])
        res = chain.solve_given_q_in(t, q_in, *args)
        # the storage cannot exceed T_max: heat that would push it over goes to the cooler
        room = res.q_out_kw + chain.storage.loss_kw(t) + cap * (self.t_max - t) / self.dt_s
        q_eff = np.minimum(q_in, np.maximum(room, 0.0))
        res = chain.solve_given_q_in(t, q_eff, *args)
        deriv = chain.storage.derivative_k_per_s(t, q_eff, res.q_out_kw)
        feasible = res.feasible & (res.q_out_kw <= self.q_air_kw[k])
        return t + self.dt_s * deriv, res.q_out_kw, q_eff, feasible


@dataclass(frozen=True, slots=True)
class ShiftResult:
    """Trajectory of one policy over the day."""

    t_s: FloatArray
    q_out_kw: FloatArray
    q_in_kw: FloatArray
    v2: FloatArray
    dump_kw: FloatArray
    q_air_kw: FloatArray
    dt_s: float
    capacity_kj_per_k: float
    feasible: bool

    @property
    def steam_kwh(self) -> float:
        return float(np.sum(self.q_air_kw - self.q_out_kw) * self.dt_s / 3600.0)

    @property
    def waste_heat_kwh(self) -> float:
        return float(np.sum(self.q_out_kw) * self.dt_s / 3600.0)

    @property
    def dump_used_kwh(self) -> float:
        return float(np.sum(self.dump_kw) * self.dt_s / 3600.0)

    @property
    def t_end(self) -> float:
        return float(self.t_s[-1])


def corrected_steam_kwh(result: ShiftResult, t_end_reference: float) -> float:
    """Steam energy corrected to the same stored energy at the end of the day.

    ``E_corr = E_steam - C_S (T_end - T_ref) / 3600``: heat left in the storage is credited at a
    factor of ONE (as if it replaced steam later in full; optimistic for the policy that ends
    hotter), heat taken out of it is charged the same way.
    """
    return result.steam_kwh - result.capacity_kj_per_k * (result.t_end - t_end_reference) / 3600.0


def co2_kg(steam_kwh: float, cfg: StorageSystemConfig) -> float:
    """CO2 of the gas needed for ``steam_kwh`` of steam (boiler efficiency, project gas factor)."""
    return steam_kwh / cfg.steam.boiler_efficiency * GAS_CO2_KG_PER_KWH


def build_problem(
    day: PlantDay,
    cfg: StorageSystemConfig,
    *,
    reference_volume_m3: float,
    q_dump_kw: FloatArray,
    ambient_shift_k: float = 0.0,
    block: int = 5,
    t_min: float = 40.0,
    v2_min: float = 5.0,
    v2_max_factor: float = 1.5,
) -> ShiftProblem:
    """Put a measured day on the control grid (block means of ``block`` samples).

    ``cfg.storage.volume_m3`` is the volume of the storage under study;
    ``reference_volume_m3`` fixes ``Q_in_base`` (compressor behaviour, reconstructed from the
    day's own storage temperature). ``q_dump_kw`` is the scenario (one value per block).
    ``ambient_shift_k`` shifts the ambient temperature (weather variant).
    """
    n = (len(day.time_s) // block) * block
    shifted = dict(day.roles)
    shifted["ambient_temperature"] = day["ambient_temperature"] + ambient_shift_k
    day_s = PlantDay(day.label, day.time_s, shifted)

    def blocks(values: FloatArray) -> FloatArray:
        return np.asarray(values[:n].reshape(-1, block).mean(axis=1), dtype=np.float64)

    ref_cfg = _with_volume(cfg, reference_volume_m3)
    n_steps = n // block
    if len(q_dump_kw) != n_steps:
        raise ValueError(f"q_dump_kw needs {n_steps} values, has {len(q_dump_kw)}")
    v2_logged = blocks(day["preheater_loop_flow"])
    return ShiftProblem(
        cfg=cfg,
        dt_s=day.step_s * block,
        t_ambient=blocks(day_s["ambient_temperature"]),
        v_storage_loop=blocks(day["storage_loop_flow"]),
        air_flow=blocks(day["process_air_flow"]),
        q_air_kw=blocks(air_heating_kw(day_s, cfg)),
        q_in_base_kw=np.maximum(blocks(oracle_q_in_kw(ref_cfg, day)), 0.0),
        q_dump_kw=np.asarray(q_dump_kw, dtype=np.float64),
        v2_logged=v2_logged,
        t_s0=float(day["storage_temperature"][0]),
        t_min=t_min,
        t_max=cfg.storage.max_temperature_celsius,
        v2_min=v2_min,
        v2_max=v2_max_factor * float(np.median(v2_logged)),
    )


def _with_volume(cfg: StorageSystemConfig, volume_m3: float) -> StorageSystemConfig:
    import dataclasses

    s = cfg.storage
    return dataclasses.replace(
        cfg,
        storage=StorageConfig(
            volume_m3=volume_m3,
            ua_kw_per_k=s.ua_kw_per_k,
            room_temperature_celsius=s.room_temperature_celsius,
            max_temperature_celsius=s.max_temperature_celsius,
        ),
    )


def simulate(problem: ShiftProblem, policy: Policy) -> ShiftResult:
    """Run a policy over the day with the model step; ``feasible`` = no constraint violated."""
    n = problem.n_steps
    t_s = np.empty(n + 1)
    q_out, q_in, v2s, dumps = (np.empty(n) for _ in range(4))
    t_s[0] = problem.t_s0
    ok = True
    for k in range(n):
        v2, dump = policy(k, float(t_s[k]))
        t_next, qo, qi, feas = problem.step(k, t_s[k], v2, dump)
        t_s[k + 1], q_out[k], q_in[k], v2s[k] = float(t_next), float(qo), float(qi), v2
        dumps[k] = max(0.0, float(qi) - float(problem.q_in_base_kw[k]))
        ok = ok and bool(feas) and t_s[k + 1] >= problem.t_min - 1e-6
    return ShiftResult(
        t_s=t_s,
        q_out_kw=q_out,
        q_in_kw=q_in,
        v2=v2s,
        dump_kw=dumps,
        q_air_kw=problem.q_air_kw,
        dt_s=problem.dt_s,
        capacity_kj_per_k=problem.chain.storage.heat_capacity_kj_per_k(),
        feasible=ok,
    )


def replay(problem: ShiftProblem) -> ShiftResult:
    """R0: logged preheater-loop flow, no extra compressor heat."""
    return simulate(problem, lambda k, _t: (float(problem.v2_logged[k]), 0.0))


def rule_policy(problem: ShiftProblem, use_dump: bool) -> Policy:
    """R1: maximum flow while ``T_S > T_min`` (else minimum); request all discarded heat."""

    def policy(k: int, t_s: float) -> tuple[float, float]:
        v2 = problem.v2_max if t_s > problem.t_min else problem.v2_min
        if not use_dump:
            return v2, 0.0
        return v2, float(problem.q_dump_kw[k])  # the step curtails what does not fit below T_max

    return policy


def dp_policy(
    problem: ShiftProblem,
    *,
    allow_dump: bool,
    t_end_reference: float,
    grid_k: float = 0.25,
    v2_levels: int = 8,
    dump_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Policy:
    """R*: backward dynamic programming on a ``T_S`` grid, greedy forward policy.

    The terminal value penalises ending below ``t_end_reference`` (``P`` kWh per K, five times the
    value of the stored heat); the forward pass re-minimises the Bellman equation at the actual
    (continuous) state, so constraints are checked exactly, not on the grid.
    """
    grid = np.arange(problem.t_min, problem.t_max + 0.5 * grid_k, grid_k)
    v2_set = np.linspace(problem.v2_min, problem.v2_max, v2_levels)
    fractions = np.asarray(dump_fractions if allow_dump else (0.0,), dtype=np.float64)
    n = problem.n_steps
    cap = problem.chain.storage.heat_capacity_kj_per_k()
    value = np.empty((n + 1, len(grid)))
    value[n] = 5.0 * cap / 3600.0 * np.maximum(0.0, t_end_reference - grid)

    def q_function(
        k: int, t: FloatArray, v_next: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        v2 = np.broadcast_to(v2_set[:, None], (len(v2_set), len(fractions))).reshape(1, -1)
        dump = (fractions[None, :] * problem.q_dump_kw[k]) + np.zeros((len(v2_set), 1))
        dump = dump.reshape(1, -1)
        t_next, q_out, _q_in, feas = problem.step(k, t[:, None], v2, dump)
        ok = feas & (t_next >= problem.t_min)
        t_next = np.minimum(t_next, problem.t_max)
        steam = (problem.q_air_kw[k] - q_out) * problem.dt_s / 3600.0
        cost = np.where(ok, steam + np.interp(t_next, grid, v_next), INFEASIBLE_COST_KWH)
        return cost, v2, dump

    for k in range(n - 1, -1, -1):
        cost, _, _ = q_function(k, grid, value[k + 1])
        value[k] = cost.min(axis=1)

    def policy(k: int, t_s: float) -> tuple[float, float]:
        cost, v2, dump = q_function(k, np.array([t_s]), value[k + 1])
        best = int(np.argmin(cost[0]))
        return float(v2[0, best]), float(dump[0, best])

    return policy
