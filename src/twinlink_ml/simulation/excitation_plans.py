r"""Equal-cost excitation plans B0-B4 for the study of information per run (B1.1).

Why a module of its own
-----------------------
``MixedExcitationDesign`` draws, per run and per channel, one of step / PRBS / ramp with
random amplitude. The question of study B1 is which *planned* excitation yields the most
information per run. That needs designs that differ in the excitation only: everything else
(run duration, amplitude limits, rate limits, disturbance process, initial conditions) is
shared by construction. ``excitation.py`` is not touched; the plans use its public parts.

Equal cost, by construction
---------------------------
All plans share one base class that draws the disturbances and the initial states exactly
as ``MixedExcitationDesign.build`` does (same random-number order). For the same ``seed`` the
disturbances and initial states are therefore bit-identical across B0-B4, which also pairs
the designs. The actions are generated in the normalised box ``[0, 1]``, de-normalised with
the action bounds and rate-limited with ``apply_rate_limit``. :func:`check_equal_cost` states
the invariants and is used by the tests.

Time constants: measured, not rule-of-thumb
-------------------------------------------
The plans are parameterised by :class:`MeasuredTimeConstants`, the values measured in this
project (docs/04_stand_ap1.md, Befund 3 and its addendum M1, N1/N2), at the three linearisation
points computed there (bed temperature 37.4 / 62.6 / 87.8 degC):

* fast mode of the coupled (water mass, bed temperature) subsystem: 30.2 - 38.8 s
* slow mode of the same subsystem: 133.5 - 1194.9 s (depends on the operating point; the
  structure of the pair changes at the clipping threshold of Befund 46)
* solids control loop: 600 s (``OperationConfig.mass_control_time_constant_s``)
* solids residence time: 16000 s (configuration constant, 1.8 times the run duration; cannot
  be excited by set points within a run and is covered by the spread of the initial states)

The task text gave 15-80 s for moisture and temperature and about 600 s for the bed mass.
The measured fast mode starts at 30 s (not 15 s) and the coupled slow mode reaches 1195 s at
the cold end (not 80 s). The derived parameters below follow the measured values:

* B2 fast clock = ``2 tau_fast_min`` rounded up to a multiple of the sample time (75 s at
  ``dt`` = 25 s), slow clock = ``0.75 tau_slow_max`` rounded to a multiple of ``dt`` (900 s).
* B3 band = ``[1 / (2 pi tau_slow_max), 1 / (2 pi tau_fast_min)]`` = 1.3e-4 .. 5.3e-3 Hz
  (corner frequencies of the slowest and the fastest measured mode).
* B4 hold time: the longest hold that leaves an integer number of holds per run with at least
  ``2 tau_slow_max`` (86 % settling at the cold end; 4 tau and more elsewhere): 2400 s,
  12 holds per run. Three time constants would leave only 8 holds per run.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ActionSpaceConfig, DisturbanceSpaceConfig, ProcessConfig
from .excitation import (
    DisturbanceDesign,
    DiurnalDisturbanceDesign,
    ExcitationConfig,
    ExcitationDesign,
    ExcitationSchedule,
    InitialConditionDesign,
    MixedExcitationDesign,
    PrbsGenerator,
    apply_rate_limit,
    quasi_random_points,
)

FloatArray = NDArray[np.float64]

__all__ = [
    "PLAN_NAMES",
    "ConstantPlan",
    "MeasuredTimeConstants",
    "MultiLevelPrbsPlan",
    "MultisinePlan",
    "PlanTimings",
    "PlannedExcitation",
    "RandomStepPlan",
    "SpaceFillingHoldPlan",
    "check_equal_cost",
    "make_plan",
]

PLAN_NAMES: tuple[str, ...] = ("b0", "b1", "b2", "b3", "b4")


@dataclass(frozen=True, slots=True)
class MeasuredTimeConstants:
    """Time constants measured in this project, in seconds. Source: Befund 3, M1 addendum."""

    fast_s: tuple[float, float] = (30.23, 38.81)
    coupled_slow_s: tuple[float, float] = (133.49, 1194.88)
    mass_control_s: float = 600.0
    residence_s: float = 16000.0
    source: str = (
        "docs/04_stand_ap1.md: Befund 3 (62.55 degC), addendum M1 (37.4 / 87.8 degC), "
        "N1/N2; fast and slow mode of the coupled subsystem"
    )


def _round_up_to(value: float, step: float) -> float:
    return math.ceil(value / step - 1e-9) * step


def _round_to(value: float, step: float) -> float:
    return max(step, round(value / step) * step)


@dataclass(frozen=True, slots=True)
class PlanTimings:
    """Plan parameters derived from the measured time constants (see module docstring)."""

    sample_time_s: float
    run_duration_s: float
    fast_clock_s: float
    slow_clock_s: float
    band_low_hz: float
    band_high_hz: float
    hold_s: float
    n_holds: int

    @classmethod
    def from_time_constants(
        cls,
        constants: MeasuredTimeConstants,
        *,
        sample_time_s: float,
        run_duration_s: float,
    ) -> PlanTimings:
        tau_fast = constants.fast_s[0]
        tau_slow = constants.coupled_slow_s[1]
        fast_clock = _round_up_to(2.0 * tau_fast, sample_time_s)
        slow_clock = _round_to(0.75 * tau_slow, sample_time_s)
        n_holds = max(1, math.floor(run_duration_s / (2.0 * tau_slow)))
        hold = _round_to(run_duration_s / n_holds, sample_time_s)
        return cls(
            sample_time_s=sample_time_s,
            run_duration_s=run_duration_s,
            fast_clock_s=fast_clock,
            slow_clock_s=slow_clock,
            band_low_hz=1.0 / (2.0 * math.pi * tau_slow),
            band_high_hz=1.0 / (2.0 * math.pi * tau_fast),
            hold_s=hold,
            n_holds=round(run_duration_s / hold),
        )


# ------------------------------------------------------------------------------ base


class PlannedExcitation(ExcitationDesign):
    """Common machinery: disturbances and initial states as in ``MixedExcitationDesign``."""

    name: str

    def __init__(
        self,
        action_space: ActionSpaceConfig,
        disturbance_space: DisturbanceSpaceConfig,
        *,
        process_config: ProcessConfig,
        sample_time_s: float,
        run_duration_s: float,
        timings: PlanTimings,
        disturbance_design: DisturbanceDesign | None = None,
        initial_condition_design: InitialConditionDesign | None = None,
    ) -> None:
        self._actions = action_space
        self._disturbances = disturbance_space
        self._process = process_config
        self._dt = sample_time_s
        self._duration = run_duration_s
        self._timings = timings
        self._disturbance_design = (
            disturbance_design if disturbance_design is not None else DiurnalDisturbanceDesign()
        )
        self._initial_design = (
            initial_condition_design
            if initial_condition_design is not None
            else InitialConditionDesign()
        )

    @property
    def n_steps(self) -> int:
        return round(self._duration / self._dt)

    @property
    def timings(self) -> PlanTimings:
        return self._timings

    def _build_normalised_actions(
        self, n_runs: int, n_steps: int, rng: np.random.Generator, seed: int
    ) -> tuple[FloatArray, tuple[tuple[str, ...], ...]]:
        """Actions in ``[0, 1]``, shape ``(runs, steps, channels)``, and the kind per channel."""
        raise NotImplementedError

    def build(self, n_runs: int, *, seed: int) -> ExcitationSchedule:
        """Plan for ``n_runs`` runs; deterministic in ``(plan, n_runs, seed)``."""
        if n_runs < 1:
            raise ValueError("n_runs must be at least 1")
        n_steps = self.n_steps
        rng = np.random.default_rng(seed)
        n_disturbance = len(self._disturbances.names)
        n_initial = self._initial_design.n_dimensions
        unit_points = quasi_random_points(
            n_runs, n_disturbance + n_initial, sampler="sobol", seed=seed
        )
        disturbances = self._disturbance_design.build(
            self._disturbances, unit_points[:, :n_disturbance], n_steps, self._dt, rng
        )
        initial_states = self._initial_design.build(
            unit_points[:, n_disturbance:],
            geometry=self._process.geometry,
            material=self._process.material,
            operation=self._process.operation,
        )
        normalised, kinds = self._build_normalised_actions(n_runs, n_steps, rng, seed)
        actions = np.empty_like(normalised)
        for channel, bound in enumerate(self._actions.bounds):
            actions[:, :, channel] = bound.minimum + normalised[:, :, channel] * bound.span
            actions[:, :, channel] = apply_rate_limit(
                actions[:, :, channel], bound.max_rate_per_s, self._dt
            )
        fingerprint = self._fingerprint(n_runs, seed)
        return ExcitationSchedule(
            run_id=tuple(f"{self.name}-{fingerprint}-{index:05d}" for index in range(n_runs)),
            actions=actions,
            disturbances=disturbances,
            initial_states=initial_states,
            signal_kinds=kinds,
            metadata={
                "design": type(self).__name__,
                "design_name": self.name,
                "fingerprint": fingerprint,
                "seed": seed,
                "sampler": "sobol",
                "sample_time_s": self._dt,
                "run_duration_s": self._duration,
                "rate_limited": True,
                "action_names": self._actions.names,
                "disturbance_names": self._disturbances.names,
                "initial_condition_columns": (
                    "solid_mass_kg",
                    "moisture_kg_per_kg",
                    "diameter_m",
                    "bed_temperature_celsius",
                ),
                "plan_timings": asdict(self._timings),
            },
        )

    def _fingerprint(self, n_runs: int, seed: int) -> str:
        payload = {
            "plan": type(self).__name__,
            "timings": asdict(self._timings),
            "actions": [(b.minimum, b.maximum, b.max_rate_per_s) for b in self._actions.bounds],
            "disturbances": [(r.minimum, r.maximum) for r in self._disturbances.ranges],
            "disturbance_design": type(self._disturbance_design).__name__,
            "disturbance_options": self._disturbance_design.fingerprint_options(),
            "n_runs": n_runs,
            "seed": seed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]

    def _operating_points(self, n_points: int, n_channels: int, seed: int) -> FloatArray:
        """Nested Sobol points in the action box (first ``n`` points do not depend on ``n``)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # balance warning for n != 2**m
            return quasi_random_points(n_points, n_channels, sampler="sobol", seed=seed + 7919)


# ------------------------------------------------------------------------------ plans


class ConstantPlan(PlannedExcitation):
    """B0: constant operation, one low-discrepancy operating point per run."""

    name = "b0"

    def _build_normalised_actions(
        self, n_runs: int, n_steps: int, rng: np.random.Generator, seed: int
    ) -> tuple[FloatArray, tuple[tuple[str, ...], ...]]:
        del rng
        n_channels = len(self._actions.bounds)
        points = self._operating_points(n_runs, n_channels, seed)
        normalised = np.repeat(points[:, None, :], n_steps, axis=1)
        return normalised, tuple(("constant",) * n_channels for _ in range(n_runs))


class RandomStepPlan(PlannedExcitation):
    """B1: the previous excitation (random steps / PRBS / ramps), without constant runs.

    Delegates to ``MixedExcitationDesign`` with ``constant_run_fraction = 0``: with the
    default 10 % constant runs B1 would contain B0 runs and the comparison would not be clean.
    """

    name = "b1"

    def build(self, n_runs: int, *, seed: int) -> ExcitationSchedule:
        config = ExcitationConfig(run_duration_s=self._duration, constant_run_fraction=0.0)
        design = MixedExcitationDesign(
            config,
            self._actions,
            self._disturbances,
            sample_time_s=self._dt,
            geometry=self._process.geometry,
            material=self._process.material,
            operation=self._process.operation,
            disturbance_design=self._disturbance_design,
            initial_condition_design=self._initial_design,
            name=self.name,
        )
        return design.build(n_runs, seed=seed)


class MultiLevelPrbsPlan(PlannedExcitation):
    r"""B2: two-scale multi-level PRBS, independent registers per channel.

    :math:`u = 0.5 + 0.5\,(a_s\,b_s + a_f\,b_f)` with :math:`b \in \{-1, +1\}` m-sequences:
    a fast one (clock ``fast_clock_s``, covers the fast mode) and a slow one (clock
    ``slow_clock_s``, covers the slow coupled mode up to 1195 s). Four levels per channel,
    ``a_s + a_f = 1`` so that the full box is used. A single PRBS clock cannot do both: with
    clock 75 s the power below 1e-3 Hz is the same PSD level but the slow mode would see
    switching only every 75 s on average; the slow component puts the energy where the slow
    mode lives. The fast step ``a_f * span`` in ``fast_clock_s`` stays below the rate limits of
    all four channels (checked in the tests).
    """

    name = "b2"
    slow_weight = 0.6
    fast_weight = 0.4

    @staticmethod
    def _register_length(n_steps: int, clock_steps: int) -> int:
        register = 7
        while register < 11 and (2**register - 1) * clock_steps < n_steps:
            register += 1
        return register

    def _build_normalised_actions(
        self, n_runs: int, n_steps: int, rng: np.random.Generator, seed: int
    ) -> tuple[FloatArray, tuple[tuple[str, ...], ...]]:
        del seed
        timings = self._timings
        n_channels = len(self._actions.bounds)
        generators = {}
        for label, clock_s in (("fast", timings.fast_clock_s), ("slow", timings.slow_clock_s)):
            clock_steps = max(1, round(clock_s / self._dt))
            generators[label] = PrbsGenerator(self._register_length(n_steps, clock_steps), clock_s)
        out = np.empty((n_runs, n_steps, n_channels), dtype=np.float64)
        for run in range(n_runs):
            for channel in range(n_channels):
                fast = generators["fast"].generate(n_steps, self._dt, rng, 1.0)
                slow = generators["slow"].generate(n_steps, self._dt, rng, 1.0)
                out[run, :, channel] = (
                    0.5
                    + 0.5 * self.slow_weight * (2.0 * slow - 1.0)
                    + 0.5 * self.fast_weight * (2.0 * fast - 1.0)
                )
        return np.clip(out, 0.0, 1.0), tuple(("prbs2",) * n_channels for _ in range(n_runs))


class MultisinePlan(PlannedExcitation):
    r"""B3: multisine with Schroeder phases over the measured frequency band.

    ``K`` log-spaced frequencies between ``band_low_hz`` and ``band_high_hz`` per channel; the
    grids of the channels are interleaved (channel ``c`` is shifted by ``c / n_channels`` of a
    grid step) so that no two channels share a frequency. Amplitudes :math:`a_k \propto
    f_k^{-1}` (equal slope contribution per tone, so that the rate limit does not cap the
    low-frequency tones); the Schroeder phases :math:`\varphi_k = -\pi k (k-1) / K` keep the
    crest factor low; a random time shift per run and channel (a pure shift, which keeps the
    relative phases) makes the runs differ. The overall scale is the largest one for which the
    peak stays inside ``[0, 1]`` and the peak slope stays below 0.8 times the rate limit of the
    channel.
    """

    name = "b3"
    n_components = 16
    amplitude_exponent = -1.0

    def _unit_signal(self, channel: int, n_channels: int) -> tuple[FloatArray, FloatArray]:
        timings = self._timings
        k = np.arange(self.n_components, dtype=np.float64)
        grid = (k + channel / n_channels) / self.n_components
        low, high = math.log(timings.band_low_hz), math.log(timings.band_high_hz)
        frequencies = np.exp(low + grid * (high - low))
        amplitudes = frequencies**self.amplitude_exponent
        amplitudes = amplitudes / amplitudes.sum()
        phases = -math.pi * k * (k - 1.0) / self.n_components
        return frequencies, np.stack([amplitudes, phases])

    def _build_normalised_actions(
        self, n_runs: int, n_steps: int, rng: np.random.Generator, seed: int
    ) -> tuple[FloatArray, tuple[tuple[str, ...], ...]]:
        del seed
        n_channels = len(self._actions.bounds)
        time_s = np.arange(n_steps, dtype=np.float64) * self._dt
        longest_period = 1.0 / self._timings.band_low_hz
        probe_time = np.arange(0.0, self._duration + longest_period, 5.0)
        out = np.empty((n_runs, n_steps, n_channels), dtype=np.float64)
        for channel, bound in enumerate(self._actions.bounds):
            frequencies, (amplitudes, phases) = self._unit_signal(channel, n_channels)
            omega = 2.0 * math.pi * frequencies
            probe_phase = omega[None, :] * probe_time[:, None] + phases[None, :]
            peak = float(np.abs((amplitudes[None, :] * np.cos(probe_phase)).sum(axis=1)).max())
            slope = float(
                np.abs(
                    (amplitudes[None, :] * omega[None, :] * np.sin(probe_phase)).sum(axis=1)
                ).max()
            )
            scale = min(0.5 / peak, 0.8 * bound.max_rate_per_s / bound.span / slope)
            for run in range(n_runs):
                shift_s = float(rng.uniform(0.0, longest_period))
                phase = omega[None, :] * (time_s[:, None] + shift_s) + phases[None, :]
                out[run, :, channel] = 0.5 + scale * (amplitudes[None, :] * np.cos(phase)).sum(
                    axis=1
                )
        return np.clip(out, 0.0, 1.0), tuple(("multisine",) * n_channels for _ in range(n_runs))


class SpaceFillingHoldPlan(PlannedExcitation):
    """B4: space-filling set points with hold phases.

    ``n_holds`` set points per run (hold ``hold_s``) from one continuing low-discrepancy
    sequence in the action box: run ``r`` takes points ``r * n_holds .. (r + 1) * n_holds - 1``,
    in random order within the run. Every prefix of ``N`` runs is a low-discrepancy design of
    ``N * n_holds`` operating points (nested, like the Sobol designs of the repository).
    Transitions follow the rate limits.
    """

    name = "b4"

    def _build_normalised_actions(
        self, n_runs: int, n_steps: int, rng: np.random.Generator, seed: int
    ) -> tuple[FloatArray, tuple[tuple[str, ...], ...]]:
        n_channels = len(self._actions.bounds)
        n_holds = self._timings.n_holds
        hold_steps = max(1, round(self._timings.hold_s / self._dt))
        points = self._operating_points(n_runs * n_holds, n_channels, seed)
        out = np.empty((n_runs, n_steps, n_channels), dtype=np.float64)
        for run in range(n_runs):
            order = rng.permutation(n_holds)
            run_points = points[run * n_holds : (run + 1) * n_holds][order]
            levels = np.repeat(run_points, hold_steps, axis=0)
            if levels.shape[0] < n_steps:
                levels = np.concatenate(
                    [levels, np.repeat(levels[-1:], n_steps - levels.shape[0], axis=0)]
                )
            out[run] = levels[:n_steps]
        return out, tuple(("holds",) * n_channels for _ in range(n_runs))


_PLAN_CLASSES: dict[str, type[PlannedExcitation]] = {
    "b0": ConstantPlan,
    "b1": RandomStepPlan,
    "b2": MultiLevelPrbsPlan,
    "b3": MultisinePlan,
    "b4": SpaceFillingHoldPlan,
}


def make_plan(
    name: str,
    process_config: ProcessConfig,
    *,
    sample_time_s: float,
    run_duration_s: float,
    solid_fraction_process: str = "independent",
    time_constants: MeasuredTimeConstants | None = None,
) -> PlannedExcitation:
    """Plan ``name`` (``"b0"`` .. ``"b4"``) for a process config.

    ``process_config`` must carry ``sample_time_s`` as the simulator will run it (as
    ``generate_data`` sets it). The disturbance process is the one of ``DataSpec``
    (``"independent"`` is the new-study preset).
    """
    try:
        cls = _PLAN_CLASSES[name]
    except KeyError:
        raise KeyError(f"unknown plan {name!r}, known: {sorted(_PLAN_CLASSES)}") from None
    timings = PlanTimings.from_time_constants(
        time_constants if time_constants is not None else MeasuredTimeConstants(),
        sample_time_s=sample_time_s,
        run_duration_s=run_duration_s,
    )
    disturbance_design = (
        None
        if solid_fraction_process == "diurnal"
        else DiurnalDisturbanceDesign(solid_fraction_process=solid_fraction_process)  # type: ignore[arg-type]
    )
    return cls(
        process_config.actions,
        process_config.disturbances,
        process_config=process_config,
        sample_time_s=sample_time_s,
        run_duration_s=run_duration_s,
        timings=timings,
        disturbance_design=disturbance_design,
    )


# --------------------------------------------------------------------- equal cost


def check_equal_cost(
    schedules: Mapping[str, ExcitationSchedule],
    process_config: ProcessConfig,
    *,
    reference: str | None = None,
    tolerance: float = 1e-9,
) -> list[str]:
    """Violations of the equal-cost invariants of B1.1 (empty list = all hold).

    Same number of runs, steps and sample time; actions inside the bounds; rate limits kept;
    disturbances and initial states bit-identical to the reference plan's.
    """
    violations: list[str] = []
    names = list(schedules)
    if not names:
        return ["no schedules"]
    reference = reference if reference is not None else names[0]
    ref = schedules[reference]
    dt = float(ref.metadata["sample_time_s"])
    for name, schedule in schedules.items():
        if schedule.actions.shape != ref.actions.shape:
            violations.append(f"{name}: shape {schedule.actions.shape} != {ref.actions.shape}")
            continue
        if float(schedule.metadata["sample_time_s"]) != dt:
            violations.append(f"{name}: different sample time")
        for channel, bound in enumerate(process_config.actions.bounds):
            values = schedule.actions[:, :, channel]
            if values.min() < bound.minimum - tolerance or values.max() > bound.maximum + tolerance:
                violations.append(f"{name}: channel {channel} leaves the bounds")
            rate = np.abs(np.diff(values, axis=1)).max() / dt
            if rate > bound.max_rate_per_s * (1.0 + 1e-6) + tolerance:
                violations.append(
                    f"{name}: channel {channel} rate {rate:.4g} > {bound.max_rate_per_s}"
                )
        if not np.array_equal(schedule.disturbances, ref.disturbances):
            violations.append(f"{name}: disturbances differ from {reference}")
        if (schedule.initial_states is None) != (ref.initial_states is None) or (
            schedule.initial_states is not None
            and not np.array_equal(schedule.initial_states, ref.initial_states)  # type: ignore[arg-type]
        ):
            violations.append(f"{name}: initial states differ from {reference}")
    return violations
