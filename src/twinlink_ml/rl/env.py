r"""One-step bandit environment -- Befund 52 in RL form.

Reproduces ``scripts/study_operating_point_regret.py`` (Befund 52) as a contextual
bandit, no new physics and no new objective function.

Observation
    ``ambient_temperature_celsius``, ``ambient_relative_humidity`` -- drawn from
    ``ProcessConfig.disturbances`` (``DisturbanceRange.minimum``/``maximum``, NOT newly
    set).
Action
    ``inlet_temperature_celsius``, ``gas_velocity_m_per_s`` -- normalised to ``[-1, 1]``,
    bounds from ``ProcessConfig.actions`` (``ActionBound.minimum``/``maximum``, NOT newly
    set).
Fixed
    ``nozzle_pressure_bar`` = :data:`FIXED_NOZZLE_PRESSURE_BAR` (Befund 52, fixed
    operating point), ``spray_rate_kg_per_h`` = ``ActionBound.nominal``,
    ``suspension_solid_fraction`` = ``DisturbanceRange.nominal`` -- both justifications from
    Befund 52 unchanged: the spray rate fixes the throughput (otherwise "minimise
    energy" is degenerate), the nozzle pressure varies over all grid optima by only 0.4 %.
Transition
    ``solve_steady_state`` WITH THE SEARCH PRECISION from Befund 52
    (``DEFAULT_SEARCH_MAX_ITERATIONS``/``DEFAULT_SEARCH_TOLERANCE``, 30 bisection steps,
    tolerance 1e-7), NOT the solver defaults (100/1e-13). Not a compromise for this
    task, but numerically necessary: with the solver defaults, 6 of the 25
    Befund 52 grid points fail due to a bracket sign error in the outer
    temperature bisection (checked empirically) -- exactly the phenomenon that ``evaluate_point``'s
    docstring in ``study_operating_point_regret.py`` describes: at the lower bracket bound
    (``T_amb + 10``) the true root lies just outside, and whether the residual value evaluated
    there shows numerical noise with a positive or negative sign depends on
    the bisection depth of the INNER moisture bisection at exactly this point. Befund 52 solves
    this by having the search AND report calls use the same (search) precision -- this environment
    adopts that unchanged, for EVERY step (not only for the VA2 oracle test).
Reward
    ``-(energy_target + penalty_weight_kw * normalized_violation)``, as in Befund 52's
    ``_penalized_energy`` (there the objective function of a minimisation, here its negative as
    reward).

Constraint handling (VA1 decision, see the task and the answer to it)
---------------------------------------------------------------------
Penalty function for ALL three constraints from Befund 52 (product moisture,
fluidisation window, bed temperature <= 80 C), imported unchanged via ``check_constraints``
-- NO projection.

Finding, not just an implementation decision: in this environment the
product moisture is the only practically effective constraint. The
fluidisation window structurally never binds, because the steady-state diameter d50 (see
``steady_state.py``, step 2) here depends only on ``material.nucleus_diameter_m`` and
``material.nucleation_solid_fraction`` -- spray rate and suspension solid fraction
cancel out in the derivation, and both are fixed in this environment anyway.
d50 is thus a constant OVER THE ENTIRE ACTION AND DISTURBANCE SPACE (750 um with
the placeholder values). A sweep calculation over bed temperatures 20-140 C (far beyond the
plausible steady-state window) and the full disturbance range yields worst-case
u_mf ~= 0.236 m/s, worst-case u_terminal ~= 3.42 m/s -- the ``ActionBound`` for
``gas_velocity_m_per_s`` ([0.8, 2.5] m/s) lies completely within it with >= 0.56 m/s margin
below and >= 0.92 m/s margin above. A projection would have nothing to do in this environment;
``check_constraints`` nevertheless remains the sole, unchanged source
of the feasibility check (also for fluidisation/bed temperature), so that a later
change of the nucleation parameters (which would cancel the result above) would not
go silently unnoticed.

IMPORTANT: this is a property of THIS FOUR-STATE MODEL, not a statement about the
real plant or a Dyssol model with agglomeration/breakage/growth coupling to the
drying conditions -- there the particle size does depend on the mode of operation,
and the state-dependent action projection
(:meth:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator.project_action`) only
becomes effective in TRANSIENTS, when d50 deviates from the steady-state value (start-ups,
product changes) -- in the steady-state case here it is a fixed box. That at
plant scale (separate investigation, RA3) with the same d50 placeholder 15 grid points failed at
the upper fluidisation bound is consistent with this, not contradictory:
presumably higher gas velocities in the plant grid there or different
gas properties at a different bed temperature -- not recalculated in this worktree,
mentioned here only for orientation, so that both statements can stand side by side.

``_PENALTY_WEIGHT_KW`` (Befund 52, 50.0 kW) is adopted UNCHANGED -- for the
VA2 oracle test the reward at the Befund 52 optimum must be bit-identical to ``-energy_target``,
and the weight was never recalibrated. It is, however, a hyperparameter with a real
effect on an RL agent (which, with binding product moisture, sits directly at the bound,
unlike the optimiser, which only passes through it as a search heuristic) and not tuned
for RL -- see the draft finding (VA5) and the reported feasibility rate/mean
moisture exceedance (VA3).

Deliberately NOT in this environment (see the task)
---------------------------------------------------
Storage, several lines, net objective function, plant scale -- later extensions.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    ProcessConfig,
    load_process_config,
)
from twinlink_ml.simulation.operating_point import (
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
    PENALTY_WEIGHT_KW,
    SOLVE_FAILURE_PENALTY_KW,
    build_variant_config,
    check_constraints,
    specific_energy_kwh_per_kg,
)
from twinlink_ml.simulation.steady_state import solve_steady_state

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
ANCHOR_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "runs" / "bb1_bb3_operating_point_regret.json"

EXPECTED_CONFIG_FINGERPRINT = "2acd06f38963c3af"
"""Muss mit dem Fingerprint von :data:`CONFIG_PATH`, so wie geladen, uebereinstimmen

(siehe ``probe_critical_moisture_sensitivity.py``, gleiche Konstante dort). Sonst ist der
VA2-Orakeltest nicht am selben Ausgangspunkt gerechnet wie Befund 52.

Der in :data:`ANCHOR_ARTIFACT_PATH` gespeicherte ``config_fingerprint`` traegt noch den
Wert von VOR dem Pseudonymisierungs-Schnitt 2026-09-25 (``abbe8325f1326d80``) -- das
Artefakt ist ein eingefrorener Provenienz-Record und wird nicht angefasst. Der Vergleich
in ``tests/test_operating_point_bandit_env.py`` nutzt deshalb
:func:`twinlink_ml.utils.fingerprint_history.fingerprints_equivalent` statt ``==``, siehe
``docs/ARCHIV.md`` fuer den Aequivalenznachweis."""

FIXED_NOZZLE_PRESSURE_BAR = 1.002634656115166
"""Befund 52, ``fixed_point.nozzle_pressure_bar`` in :data:`ANCHOR_ARTIFACT_PATH` (der
fester, ueberall zulaessige Betriebspunkt) -- ueber alle 25 Gitteroptima variiert der
Duesendruck nur um 0.4 % der Stellgroessenspanne (``setpoint_spread.nozzle_pressure_bar.
span_fraction_of_bound``), deshalb hier als dritte, in diesem Auftrag nicht gelernte Aktion
fixiert (Auftrag VA1). Voller Gleitkommawert, nicht die im Auftragstext gerundete
Kurzform "1.003 bar" -- fuer den VA2-Bitgleichheitstest zaehlt die Praezision."""

FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS = 90.4128503528912
"""Befund 52, ``fixed_point.inlet_temperature_celsius`` in :data:`ANCHOR_ARTIFACT_PATH` --
zusammen mit :data:`FIXED_OPERATING_POINT_GAS_VELOCITY_M_PER_S` der feste, ueberall
zulaessige Betriebspunkt, der auch :data:`FIXED_NOZZLE_PRESSURE_BAR` traegt. Fuer BB2s
zentrierte Belohnung (:class:`RewardCentering`) -- NICHT fuer die Umgebung selbst gelernt
oder als Aktionsgrenze verwendet."""

FIXED_OPERATING_POINT_GAS_VELOCITY_M_PER_S = 0.8001296812578764
"""Siehe :data:`FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS`."""

DEFAULT_REWARD_CENTERING_SCALE_KW = 0.2296627007886406
"""Vorschlag fuer :attr:`RewardCentering.scale_kw`: das 95. Perzentil des Befund-52-Regrets
(``regret.absolute_kw.p95`` in :data:`ANCHOR_ARTIFACT_PATH`, Median 0.174 kW, Maximum 0.241
kW -- p95 als Kompromiss zwischen "typisch" und "fast das Maximum"). Begruendung: eine
Politik, die den GESAMTEN Regret an einem typischen Punkt schliesst, sieht damit eine
Belohnungsdifferenz der Groessenordnung 1 -- dieselbe Grossenordnung wie VA3s "Anteil
geschlossenen Regrets" (0 = fest, 1 = Orakel). Kein Bezug zur Strafskala (siehe
:class:`RewardCentering`, Punkt 3)."""

ACTION_NAMES: tuple[str, str] = ("inlet_temperature_celsius", "gas_velocity_m_per_s")
DISTURBANCE_NAMES: tuple[str, str] = (
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
)


def _normalise(value: float, minimum: float, maximum: float) -> float:
    return 2.0 * (value - minimum) / (maximum - minimum) - 1.0


def _denormalise(value: float, minimum: float, maximum: float) -> float:
    return minimum + 0.5 * (value + 1.0) * (maximum - minimum)


@dataclass(frozen=True, slots=True)
class RewardCentering:
    r"""BB2: affine reward transformation ``r' = (r - r_ref) / scale_kw``.

    ``r_ref`` is the RAW reward (no penalty part, since Befund 52's fixed operating point
    is feasible at all 25 grid points) of the FIXED Befund 52 operating point
    (:data:`FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS`/``..._GAS_VELOCITY_M_PER_S``),
    evaluated anew for EVERY context -- one additional ``solve_steady_state`` call per
    step, roughly double the cost compared with the raw mode.

    IMPORTANT, three points (task):

    1. The affine mapping DOES NOT CHANGE THE OPTIMUM: for ``scale_kw > 0``,
       ``a -> (a - c) / scale_kw`` is strictly monotonically increasing, the argmax of a policy
       stays identical. Befund 52 remains the yardstick -- this transformation changes ONLY the
       reward SCALE the agent sees, not the underlying optimisation task.
    2. The environment keeps a reproduction mode with RAW reward
       (``BanditEnvConfig.reward_centering=None``, default) -- the VA2 oracle check runs
       exclusively in this mode and stays bit-identical (see
       ``tests/test_operating_point_bandit_env.py``, unchanged).
    3. The dominance of the penalty measured in YA2 (factor ~516 relative to the regret to be
       closed) DOES NOT CHANGE BECAUSE OF THIS: penalty and energy are scaled EQUALLY (both
       divided by the same ``scale_kw``), their RATIO to each other is preserved
       exactly. That is a separate problem (see YA2's four options) and is
       explicitly NOT addressed here -- BB2 tests exclusively whether removing the
       CONTEXT-DEPENDENT raw-energy base scale (YA2 (b): ~2-4 kW spread of feasible
       actions, ~18 times the regret signal) helps the critic, independently of the
       penalty dominance.
    """

    scale_kw: float = DEFAULT_REWARD_CENTERING_SCALE_KW


@dataclass(frozen=True, slots=True)
class BanditEnvConfig:
    """The environment's own configuration.

    Action/disturbance bounds are already in ``ProcessConfig``
    (``config.actions``/``config.disturbances``) and are NOT duplicated here.
    """

    process_config_path: Path = CONFIG_PATH
    fixed_nozzle_pressure_bar: float = FIXED_NOZZLE_PRESSURE_BAR
    penalty_weight_kw: float | None = None
    """``None`` (Default) uebernimmt ``operating_point.PENALTY_WEIGHT_KW`` (Befund 52, in S0.1
    aus dem Skript verschoben) unveraendert -- Bitgleichheit fuer den VA2-Orakeltest.
    Explizit gesetzt nur fuer spaeteres RL-Tuning ausserhalb dieses Auftrags (siehe
    Moduldocstring)."""
    reward_centering: RewardCentering | None = None
    """``None`` (Default): ROHE Belohnung, bitgleich zu VA2 -- siehe :class:`RewardCentering`
    fuer den BB2-Modus mit affiner Transformation."""
    include_penalty: bool = True
    """DA1: ``True`` (Default, unveraendert) -- Belohnung ist ``-(energy_target +
    penalty_weight_kw * normalized_violation)`` wie in Befund 52 und der VA2-Orakelpruefung.
    ``False``: Belohnung ist NUR ``-energy_target``, die Nebenbedingungen werden weiterhin
    ueber ``check_constraints`` geprueft und vollstaendig in ``info`` protokolliert (nur
    NICHT in die Belohnung eingerechnet) -- trennt die Frage "kollabiert SAC wegen der
    Strafklippe?" von "kollabiert SAC grundsaetzlich, mit oder ohne Strafe?" (DA1). Ein
    Loeserfehlschlag bleibt UNABHAENGIG von diesem Schalter bestraft, siehe
    :attr:`solve_failure_penalty_kw` -- das ist keine Nebenbedingungsstrafe, sondern die
    numerische Notwendigkeit, dass ohne geloeste Energie kein ``-energy_target`` existiert,
    das zurueckgegeben werden koennte."""
    product_moisture_limit_kg_per_kg: float | None = None
    """S0.6: quality limit of the product moisture, overriding the loaded config.

    ``None`` (Default): the config decides (its own ``product_moisture_limit_kg_per_kg`` or,
    if unset, the critical moisture). Set: ``MaterialConfig.product_moisture_limit_kg_per_kg``
    of the loaded config is replaced; the drying kinetics keep ``critical_moisture_kg_per_kg``."""
    solve_failure_penalty_kw: float | None = None
    """EA2: ``None`` (Default, unveraendert) uebernimmt Befund 52s Sentinel
    (``operating_point.SOLVE_FAILURE_PENALTY_KW`` = 1.000.000) -- Bitgleichheit fuer VA2, das
    nur loesbare Orakelaktionen einspeist und den Sentinel nie beruehrt. Explizit gesetzt: ein
    Loeserfehlschlag liefert ``reward = -solve_failure_penalty_kw`` statt des Sentinels.

    KEIN fester, willkuerlich gewaehlter Ersatzwert -- die vorgesehene Berechnung (siehe
    ``scripts/probe_bandit_env_solve_failure_bound.py``) ist MODUSABHAENGIG: die
    schlechteste Belohnung, die eine LOESBARE Aktion im jeweils aktiven Modus
    (``include_penalty``) tatsaechlich erreicht (ueber ein Referenzgitter) -- ein
    Loeserfehlschlag wird damit "so schlecht wie der schlechteste loesbare Betriebspunkt",
    nicht ein Ausreisser mehrere Groessenordnungen darueber. Anlass: EA1 zeigt, dass der
    Sentinel vermutlich die Kritiker-Regression vergiftet (Q-Werte an der numerisch
    fragilen kalten Ecke liegen bei allen beobachteten Checkpoints weit jenseits jeder
    physikalisch moeglichen Belohnung)."""


class OperatingPointBanditEnv(gym.Env[np.ndarray, np.ndarray]):
    """Contextual bandit: one episode = one step. See module docstring."""

    # dict instead of ClassVar: gym.Env itself declares ``metadata`` as an instance attribute
    # (mypy rejects a ClassVar override of it) -- standard pattern throughout the
    # Gymnasium ecosystem, no actual mutable-default danger.
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(self, config: BanditEnvConfig | None = None) -> None:
        self._env_config = config if config is not None else BanditEnvConfig()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            self._base_config: ProcessConfig = load_process_config(
                self._env_config.process_config_path
            )
        if self._env_config.product_moisture_limit_kg_per_kg is not None:
            self._base_config = dataclasses.replace(
                self._base_config,
                material=dataclasses.replace(
                    self._base_config.material,
                    product_moisture_limit_kg_per_kg=self._env_config.product_moisture_limit_kg_per_kg,
                ),
            )

        self._penalty_weight_kw = (
            self._env_config.penalty_weight_kw
            if self._env_config.penalty_weight_kw is not None
            else PENALTY_WEIGHT_KW
        )

        actions = self._base_config.actions
        self._inlet_bound = actions.inlet_temperature_celsius
        self._gas_bound = actions.gas_velocity_m_per_s

        disturbances = self._base_config.disturbances
        self._t_amb_range = disturbances.ambient_temperature_celsius
        self._phi_range = disturbances.ambient_relative_humidity

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self._t_amb_range.minimum, self._phi_range.minimum], dtype=np.float32),
            high=np.array([self._t_amb_range.maximum, self._phi_range.maximum], dtype=np.float32),
            dtype=np.float32,
        )

        self._disturbance: tuple[float, float] | None = None

    # ---------------------------------------------------------------- Gymnasium interface

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Draws a new disturbance point (or adopts a given one).

        ``options`` with ``ambient_temperature_celsius``/``ambient_relative_humidity``
        forces a specific point instead of a draw -- for the VA2 oracle test
        (the exact 25 Befund 52 grid points) and the VA3 metric (fixed test point set),
        which tolerate no randomness.
        """
        super().reset(seed=seed)
        options = options or {}
        if "ambient_temperature_celsius" in options or "ambient_relative_humidity" in options:
            t_amb = float(options.get("ambient_temperature_celsius", self._t_amb_range.nominal))
            phi_amb = float(options.get("ambient_relative_humidity", self._phi_range.nominal))
        else:
            t_amb = float(
                self.np_random.uniform(self._t_amb_range.minimum, self._t_amb_range.maximum)
            )
            phi_amb = float(
                self.np_random.uniform(self._phi_range.minimum, self._phi_range.maximum)
            )
        self._disturbance = (t_amb, phi_amb)
        observation = np.array([t_amb, phi_amb], dtype=np.float32)
        info: dict[str, Any] = {
            "ambient_temperature_celsius": t_amb,
            "ambient_relative_humidity": phi_amb,
        }
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._disturbance is None:
            raise RuntimeError("step() vor reset() aufgerufen")
        t_amb, phi_amb = self._disturbance

        requested_normalised = np.asarray(action, dtype=np.float64)
        applied_normalised = np.clip(requested_normalised, -1.0, 1.0)
        t_in = _denormalise(
            float(applied_normalised[0]), self._inlet_bound.minimum, self._inlet_bound.maximum
        )
        u_gas = _denormalise(
            float(applied_normalised[1]), self._gas_bound.minimum, self._gas_bound.maximum
        )

        reward, info = self.evaluate_operating_point(
            inlet_temperature_celsius=t_in,
            gas_velocity_m_per_s=u_gas,
            ambient_temperature_celsius=t_amb,
            ambient_relative_humidity=phi_amb,
        )
        info["action_normalised_requested"] = requested_normalised.tolist()
        info["action_normalised_applied"] = applied_normalised.tolist()

        observation = np.array([t_amb, phi_amb], dtype=np.float32)
        terminated = True
        truncated = False
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------ Core calculation

    def evaluate_operating_point(
        self,
        *,
        inlet_temperature_celsius: float,
        gas_velocity_m_per_s: float,
        ambient_temperature_celsius: float,
        ambient_relative_humidity: float,
    ) -> tuple[float, dict[str, Any]]:
        """Reward (raw or centred, see :class:`RewardCentering`) and info dict.

        Core method without ``reset()``/``step()`` and without a normalisation round trip --
        used by :meth:`step` AND by the VA2 oracle test as well as the VA3 metric
        (``evaluation.py``), so that reward/feasibility are computed in exactly one place.
        VA2 demands bit-identity in the RAW mode (``reward_centering=None``, default) --
        this path internally calls exactly the same :meth:`_evaluate_operating_point_raw`
        as before, unchanged.
        """
        raw_reward, info = self._evaluate_operating_point_raw(
            inlet_temperature_celsius=inlet_temperature_celsius,
            gas_velocity_m_per_s=gas_velocity_m_per_s,
            ambient_temperature_celsius=ambient_temperature_celsius,
            ambient_relative_humidity=ambient_relative_humidity,
        )
        centering = self._env_config.reward_centering
        if centering is None:
            return raw_reward, info

        # BB2: r' = (r - r_ref) / scale_kw -- r_ref is Befund 52's fixed operating point in the
        # SAME context, evaluated raw (no recursion risk: this call goes directly
        # to the raw method, not to evaluate_operating_point itself).
        fixed_reward, fixed_info = self._evaluate_operating_point_raw(
            inlet_temperature_celsius=FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS,
            gas_velocity_m_per_s=FIXED_OPERATING_POINT_GAS_VELOCITY_M_PER_S,
            ambient_temperature_celsius=ambient_temperature_celsius,
            ambient_relative_humidity=ambient_relative_humidity,
        )
        centered_reward = (raw_reward - fixed_reward) / centering.scale_kw
        info["reward_raw"] = raw_reward
        info["reward_centered"] = centered_reward
        info["reward_centering_scale_kw"] = centering.scale_kw
        info["fixed_point_reward_reference"] = fixed_reward
        info["fixed_point_solve_failed_for_reference"] = bool(fixed_info.get("solve_failed"))
        return centered_reward, info

    def _evaluate_operating_point_raw(
        self,
        *,
        inlet_temperature_celsius: float,
        gas_velocity_m_per_s: float,
        ambient_temperature_celsius: float,
        ambient_relative_humidity: float,
    ) -> tuple[float, dict[str, Any]]:
        """Raw reward, as before BB2 -- unchanged, so that VA2 stays bit-identical."""
        variant_config = build_variant_config(
            self._base_config,
            t_in=inlet_temperature_celsius,
            u_gas=gas_velocity_m_per_s,
            p_nozzle=self._env_config.fixed_nozzle_pressure_bar,
            t_amb=ambient_temperature_celsius,
            phi_amb=ambient_relative_humidity,
        )
        simulator = GreyBoxFluidBedGranulator(variant_config)

        requested_physical = {
            "inlet_temperature_celsius": inlet_temperature_celsius,
            "gas_velocity_m_per_s": gas_velocity_m_per_s,
            "nozzle_pressure_bar": self._env_config.fixed_nozzle_pressure_bar,
            "spray_rate_kg_per_h": self._base_config.actions.spray_rate_kg_per_h.nominal,
        }

        try:
            solution = solve_steady_state(
                simulator,
                max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
                tolerance=DEFAULT_SEARCH_TOLERANCE,
            )
        except (RuntimeError, ValueError) as exc:
            solve_failure_penalty_kw = (
                self._env_config.solve_failure_penalty_kw
                if self._env_config.solve_failure_penalty_kw is not None
                else SOLVE_FAILURE_PENALTY_KW
            )
            reward = -solve_failure_penalty_kw
            info: dict[str, Any] = {
                "feasible": False,
                "solve_failed": True,
                "solve_failure_reason": str(exc),
                "solve_failure_penalty_kw": solve_failure_penalty_kw,
                "action_physical_requested": requested_physical,
                "action_physical_applied": requested_physical,
                "ambient_temperature_celsius": ambient_temperature_celsius,
                "ambient_relative_humidity": ambient_relative_humidity,
            }
            return reward, info

        check = check_constraints(
            solution,
            critical_moisture_kg_per_kg=(
                self._base_config.material.quality_moisture_limit_kg_per_kg
            ),
            gas_velocity_m_per_s=gas_velocity_m_per_s,
        )
        energy_target = float(solution.diagnostics["energy_target"])
        if self._env_config.include_penalty:
            reward = -(energy_target + self._penalty_weight_kw * check.normalized_violation)
        else:
            # DA1: constraints remain checked/logged (below in ``info``),
            # but do NOT enter the reward.
            reward = -energy_target

        info = {
            "feasible": check.feasible,
            "solve_failed": False,
            "energy_target": energy_target,
            "energy_target_unit": self._base_config.energy.target,
            "specific_energy_kwh_per_kg": specific_energy_kwh_per_kg(solution),
            "constraint_check": check.as_dict(),
            "infeasibility_reasons": list(solution.infeasibility_reasons),
            # No projection mechanism in this environment (see module docstring) --
            # "requested"/"applied" are always identical here; the fields remain nevertheless,
            # so that a caller does not have to distinguish between "no projection mechanism" and
            # "projection did not take effect" (analogy to
            # ``Trajectory.requested_actions``, see ``simulation/base.py``).
            "action_physical_requested": requested_physical,
            "action_physical_applied": requested_physical,
            "ambient_temperature_celsius": ambient_temperature_celsius,
            "ambient_relative_humidity": ambient_relative_humidity,
        }
        return float(reward), info
