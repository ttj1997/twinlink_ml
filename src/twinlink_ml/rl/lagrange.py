r"""S0.3: Lagrangian treatment of the product-moisture constraint in the bandit.

The penalty treatment of Befund 62 can respect a limit only with a safety distance: the
penalty agent keeps a median margin of 0.1395 kg/kg (GA1) and pays for it with a negative
pooled closed-regret fraction. Here the moisture term of the penalty is replaced by a
Lagrange term with a SIGNED constraint value

.. math:: \tilde g = \frac{X_{prod} - (X_{lim} - m)}{X_{lim}}

so that for :math:`\tilde g < 0` (slack) the multiplier decreases and the agent is pushed
towards the tightened limit, and for :math:`\tilde g > 0` it increases. The reward is

.. math:: r = -\bigl(E + P\,v_{rest} + \lambda_b\,\tilde g\bigr)

with :math:`E` the ``energy_target`` in kW, :math:`P v_{rest}` the unchanged Befund 52
penalty of all OTHER constraints (bed temperature, fluidisation window, solver validity;
the moisture part of ``normalized_violation`` is removed), and :math:`\lambda_b \ge 0` in kW
the multiplier of the context bin :math:`b`. Only the moisture handling differs from the
penalty arm, so a difference between the arms is attributable to it.

Why one multiplier per context bin
-----------------------------------
A single global multiplier would enforce the constraint only in expectation over the
disturbance distribution (slack in some contexts can offset violations in others). The gate
asks for feasibility at (nearly) every test point, which is a pointwise requirement, so the
multiplier is a function of the context: a table over a grid of ``context_bins`` equal-width
cells of the two disturbance ranges, each updated by dual ascent from the samples that fall
into it. ``context_bins=(1, 1)`` is the global multiplier.

Failure value
-------------
A solver failure gets the worst Lagrangian cost that a SOLVABLE action reaches over the
13 x 9 reference grid at the nominal disturbance (rule of Befund 62 FA1), evaluated with the
multiplier of the current bin. It is not the Befund 52 sentinel. The grid is evaluated once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .env import BanditEnvConfig, OperatingPointBanditEnv

__all__ = [
    "REFERENCE_GRID_T_IN",
    "REFERENCE_GRID_U_GAS",
    "LagrangeBanditEnv",
    "LagrangeConfig",
    "ReferenceTable",
    "dual_ascent",
    "failure_bound_kw",
    "relabel_rewards",
]

REFERENCE_GRID_T_IN = np.linspace(60.0, 120.0, 13)
REFERENCE_GRID_U_GAS = np.linspace(0.8, 2.5, 9)
"""Grid of ``scripts/probe_bandit_env_solve_failure_bound.py`` (Befund 62 EA2/FA1)."""


@dataclass(frozen=True, slots=True)
class LagrangeConfig:
    """Settings of the Lagrangian treatment. Margin and dual settings are NOT defaults."""

    margin_kg_per_kg: float
    initial_multiplier_kw: float
    step_size_kw: float
    update_interval_steps: int
    context_bins: tuple[int, int] = (5, 5)

    def __post_init__(self) -> None:
        if self.margin_kg_per_kg < 0.0:
            raise ValueError("margin_kg_per_kg must be non-negative")
        if self.initial_multiplier_kw < 0.0:
            raise ValueError("initial_multiplier_kw must be non-negative")
        if self.step_size_kw <= 0.0:
            raise ValueError("step_size_kw must be positive")
        if self.update_interval_steps < 1:
            raise ValueError("update_interval_steps must be at least 1")
        if min(self.context_bins) < 1:
            raise ValueError("context_bins must be at least (1, 1)")


@dataclass(frozen=True, slots=True)
class ReferenceTable:
    """Solvable reference-grid points: per point ``(E, rest penalty, g_tilde)``."""

    energy_kw: np.ndarray
    rest_penalty_kw: np.ndarray
    g_tilde: np.ndarray


def dual_ascent(multiplier: float, mean_g_tilde: float, step_size_kw: float) -> float:
    """``max(0, lambda + eta * mean(g))`` -- projected dual ascent step."""
    return max(0.0, multiplier + step_size_kw * mean_g_tilde)


def failure_bound_kw(table: ReferenceTable, multiplier: float) -> float:
    """Worst Lagrangian cost over the solvable reference points at ``multiplier``."""
    cost = table.energy_kw + table.rest_penalty_kw + multiplier * table.g_tilde
    return float(np.max(cost))


def relabel_rewards(
    *,
    energy_kw: np.ndarray,
    rest_penalty_kw: np.ndarray,
    g_tilde: np.ndarray,
    solve_failed: np.ndarray,
    bin_index: np.ndarray,
    multipliers: np.ndarray,
    table: ReferenceTable,
) -> np.ndarray:
    """Rewards of stored transitions under the CURRENT multipliers.

    The bandit has no bootstrapping (Befund 62 CB1/CB2): the critic regresses
    ``Q(s, a) = r(s, a)``. A reward stored with an outdated multiplier would be a wrong
    regression target, so the replay buffer is relabelled after every dual update.
    """
    lam = multipliers.reshape(-1)[bin_index]
    reward = -(energy_kw + rest_penalty_kw + lam * np.where(solve_failed, 0.0, g_tilde))
    if np.any(solve_failed):
        bound = np.array(
            [failure_bound_kw(table, float(value)) for value in lam[solve_failed]],
            dtype=np.float64,
        )
        reward[solve_failed] = -bound
    return reward


class LagrangeBanditEnv(OperatingPointBanditEnv):
    """Bandit with the moisture constraint as a Lagrange term (see module docstring)."""

    def __init__(
        self,
        lagrange: LagrangeConfig,
        config: BanditEnvConfig | None = None,
        *,
        reference_table: ReferenceTable | None = None,
    ) -> None:
        env_config = config if config is not None else BanditEnvConfig()
        if not env_config.include_penalty:
            raise ValueError("LagrangeBanditEnv needs include_penalty=True (it removes the term)")
        super().__init__(env_config)
        self._lagrange = lagrange
        self._multipliers = np.full(
            lagrange.context_bins, lagrange.initial_multiplier_kw, dtype=np.float64
        )
        self._quality_limit = self._base_config.material.quality_moisture_limit_kg_per_kg
        self._reference_table = reference_table

    # ------------------------------------------------------------------ multipliers

    @property
    def lagrange_config(self) -> LagrangeConfig:
        return self._lagrange

    @property
    def multipliers(self) -> np.ndarray:
        return self._multipliers.copy()

    def set_multipliers(self, multipliers: np.ndarray) -> None:
        array = np.asarray(multipliers, dtype=np.float64)
        if array.shape != self._multipliers.shape:
            raise ValueError(f"shape {array.shape} != {self._multipliers.shape}")
        if np.any(array < 0.0):
            raise ValueError("multipliers must be non-negative")
        self._multipliers = array.copy()

    def bin_of(self, ambient_temperature_celsius: float, ambient_relative_humidity: float) -> int:
        """Flat index of the context cell (row-major over temperature, humidity)."""
        n_t, n_phi = self._lagrange.context_bins
        indices = []
        for value, bound, n in (
            (ambient_temperature_celsius, self._t_amb_range, n_t),
            (ambient_relative_humidity, self._phi_range, n_phi),
        ):
            fraction = (value - bound.minimum) / (bound.maximum - bound.minimum)
            indices.append(int(min(max(fraction * n, 0.0), n - 1e-9)))
        return indices[0] * n_phi + indices[1]

    @property
    def tightened_limit_kg_per_kg(self) -> float:
        return self._quality_limit - self._lagrange.margin_kg_per_kg

    # ------------------------------------------------------------------ reference table

    @property
    def reference_table(self) -> ReferenceTable:
        if self._reference_table is None:
            self._reference_table = self._build_reference_table()
        return self._reference_table

    def _build_reference_table(self) -> ReferenceTable:
        t_amb = self._t_amb_range.nominal
        phi = self._phi_range.nominal
        energies: list[float] = []
        rests: list[float] = []
        gs: list[float] = []
        for t_in in REFERENCE_GRID_T_IN:
            for u_gas in REFERENCE_GRID_U_GAS:
                _, info = super().evaluate_operating_point(
                    inlet_temperature_celsius=float(t_in),
                    gas_velocity_m_per_s=float(u_gas),
                    ambient_temperature_celsius=t_amb,
                    ambient_relative_humidity=phi,
                )
                if info.get("solve_failed"):
                    continue
                energy, rest, g_tilde = self._components(info)
                energies.append(energy)
                rests.append(rest)
                gs.append(g_tilde)
        return ReferenceTable(np.array(energies), np.array(rests), np.array(gs))

    # ------------------------------------------------------------------ reward

    def _components(self, info: dict[str, Any]) -> tuple[float, float, float]:
        """``(E, rest penalty in kW, signed g_tilde)`` of a solved point."""
        check = info["constraint_check"]
        limit = self._quality_limit
        rest_violation = float(check["normalized_violation"]) - float(
            check["moisture_violation_kg_per_kg"]
        ) / max(limit, 1e-6)
        energy = float(info["energy_target"])
        g_tilde = (float(check["moisture_kg_per_kg"]) - self.tightened_limit_kg_per_kg) / limit
        return energy, self._penalty_weight_kw * rest_violation, g_tilde

    def evaluate_operating_point(
        self,
        *,
        inlet_temperature_celsius: float,
        gas_velocity_m_per_s: float,
        ambient_temperature_celsius: float,
        ambient_relative_humidity: float,
    ) -> tuple[float, dict[str, Any]]:
        """Lagrangian reward and info; ``info["feasible"]`` is judged against the TRUE limit."""
        _, info = super().evaluate_operating_point(
            inlet_temperature_celsius=inlet_temperature_celsius,
            gas_velocity_m_per_s=gas_velocity_m_per_s,
            ambient_temperature_celsius=ambient_temperature_celsius,
            ambient_relative_humidity=ambient_relative_humidity,
        )
        bin_index = self.bin_of(ambient_temperature_celsius, ambient_relative_humidity)
        multiplier = float(self._multipliers.reshape(-1)[bin_index])
        info["lagrange_bin"] = bin_index
        info["lagrange_multiplier_kw"] = multiplier
        if info.get("solve_failed"):
            bound = failure_bound_kw(self.reference_table, multiplier)
            info["lagrange_failure_bound_kw"] = bound
            info["lagrange_g_tilde"] = float("nan")
            return -bound, info
        energy, rest_penalty, g_tilde = self._components(info)
        info["lagrange_energy_kw"] = energy
        info["lagrange_rest_penalty_kw"] = rest_penalty
        info["lagrange_g_tilde"] = g_tilde
        return -(energy + rest_penalty + multiplier * g_tilde), info
