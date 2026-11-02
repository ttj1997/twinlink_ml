r"""Excitation design for data generation.

Why this is a module of its own with abstractions and not a script
------------------------------------------------------------------
The excitation design decides **which task** the sequence models will later
learn. It is thus an experimental parameter on the same level as the architecture, and it
must be versioned, reproducible and documentable towards TUHH-A --
``02_prozess_setup.md``
records that the sampling design must be agreed *before* runs are produced there for
18 months. A script that contains a few ``rng.uniform`` calls is
no basis for this conversation.

Three design decisions that deviate from the work plan
------------------------------------------------------
**1. Sobol instead of Latin hypercube as the default.** For learning curves, datasets with
5, 10, 25, 50, 100 runs are needed. With LHS a 5-run design is not part of the
100-run design: the points must be drawn anew, and the space coverage changes
between the support points of the learning curve. The curve then mixes two effects --
data amount and coverage quality. Sobol sequences are **nested**: the
first :math:`n` points of a Sobol sequence are themselves a low-discrepancy sequence.
The learning-curve datasets are thus true subsets, and the curve measures only the
data amount. ``test_sobol_design_is_nested`` checks exactly that,
``test_latin_hypercube_is_not_nested`` the opposite. LHS remains available as an option.

**2. Rate limiting (slew rate limiting) is part of the design, not optional.**
A PRBS jumps in one sampling step across half the action span. No real
inlet air control does that. Without rate limiting the dataset contains transitions that the
plant cannot execute -- and because the heater dynamics are missing in the grey-box model
(see step 1), rate limiting is currently the **only** place where the
actuator dynamics are represented at all.

From this follows a hard design limit: the minimal realisable switching time of a
channel is

.. math:: t_{min} = \frac{\Delta u}{\dot u_{max}}

For the inlet air temperature with the placeholder values (span 60 K, 0.5 K/s) this is 120 s
-- **more** than the thermal time constant of 84 s. Over the full span the
step response of the bed is thus fundamentally not measurable. Whoever wants to excite the
fast dynamics must reduce the amplitude. That is why the amplitude here is a
design parameter with several levels and not a constant.

**3. Initial states are varied as well.** If every run starts at
:math:`d_{50} = 400\,\mu m` and grows monotonically, then :math:`d_{50}` within a
run is practically a function of the time index. A model with position information
(Transformer with positional encoding, but also an RNN over the sequence position) can then
"predict" the slow mode from the step count without learning the physics. That is
not leakage in the sense of overlapping windows, but acts the same way: the comparison comes
out in favour of the architecture with the strongest position signal. The countermeasure is the
spread of the initial conditions over the operating range.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Final, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.stats import qmc

from .backend import NumpyBackend
from .config import (
    ActionSpaceConfig,
    DisturbanceSpaceConfig,
    GasConfig,
    GeometryConfig,
    MaterialConfig,
    OperationConfig,
)
from .fluidization import terminal_velocity_m_per_s
from .thermo import absolute_humidity_kg_per_kg

FloatArray = NDArray[np.float64]
SamplerName = Literal["sobol", "lhs"]
SignalKind = Literal["step", "prbs", "ramp", "constant"]
SolidFractionProcess = Literal["diurnal", "independent"]
GasVelocityCap = Literal["none", "initial_diameter"]

_SECONDS_PER_DAY: Final = 86_400.0
_DEFAULT_RUN_DURATION_S: Final = 28_800.0

# Feedback polynomials of maximum length (m-sequences), register length -> taps
_LFSR_TAPS: Final[dict[int, tuple[int, ...]]] = {
    7: (7, 6),
    8: (8, 6, 5, 4),
    9: (9, 5),
    10: (10, 7),
    11: (11, 9),
}


# --------------------------------------------------------------------------- Configs


@dataclass(frozen=True, slots=True)
class HoldTimeRange:
    """Hold time range of a step sequence in seconds."""

    minimum_s: float
    maximum_s: float

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_s <= self.maximum_s:
            raise ValueError("Verweilzeitbereich muss positiv und aufsteigend sein")


@dataclass(frozen=True, slots=True)
class ExcitationConfig:
    """Parameters of the excitation design.

    :param amplitude_levels: amplitudes as a fraction of the action span. Several levels
        are necessary, not cosmetic: small amplitudes allow fast
        switching within the rate limit and thus excite the fast modes,
        large amplitudes cover the operating space. A single amplitude value cannot
        do both.
    :param signal_weights: relative frequency of the signal kinds. Drawn independently per run
        **and per channel** -- this decorrelates the channels without artificially
        orthogonalising them.
    :param constant_run_fraction: fraction of runs with constant actions. These runs
        are uninformative for the architecture comparison, but the reference for
        steady states and for checking whether a model can hold a stationary point.
        Without them the dataset lacks any information about what the process
        looks like when nothing happens. At ``1.0`` the entire dataset consists of
        constant runs -- that is the evaluation dataset that reflects the real mode of operation
        of the reference plant (constant actions in the running process). It is an
        evaluation dataset, not a training dataset, and not suitable on its own for the
        architecture comparison.
    """

    run_duration_s: float = _DEFAULT_RUN_DURATION_S
    hold_time: HoldTimeRange = field(default_factory=lambda: HoldTimeRange(60.0, 1200.0))
    prbs_register_length: int = 8
    prbs_clock_s: float = 120.0
    ramp_segment_time: HoldTimeRange = field(default_factory=lambda: HoldTimeRange(300.0, 2400.0))
    amplitude_levels: tuple[float, ...] = (0.15, 0.35, 0.70, 1.00)
    signal_weights: dict[str, float] = field(
        default_factory=lambda: {"step": 0.40, "prbs": 0.30, "ramp": 0.30}
    )
    constant_run_fraction: float = 0.10
    apply_rate_limits: bool = True
    sampler: SamplerName = "sobol"
    randomise_initial_conditions: bool = True

    def __post_init__(self) -> None:
        if self.run_duration_s <= 0.0:
            raise ValueError("run_duration_s muss positiv sein")
        if self.prbs_register_length not in _LFSR_TAPS:
            raise ValueError(
                f"prbs_register_length muss in {sorted(_LFSR_TAPS)} liegen, "
                f"ist {self.prbs_register_length}"
            )
        if not all(0.0 < level <= 1.0 for level in self.amplitude_levels):
            raise ValueError("amplitude_levels müssen in (0, 1] liegen")
        if not self.signal_weights or min(self.signal_weights.values()) < 0.0:
            raise ValueError("signal_weights müssen nichtnegativ und nichtleer sein")
        if not 0.0 <= self.constant_run_fraction <= 1.0:
            raise ValueError("constant_run_fraction muss in [0, 1] liegen")

    @property
    def prbs_period_s(self) -> float:
        """Period of the m-sequence.

        Must lie above the settling time of the process, otherwise the excitation repeats
        before the process has responded to it -- the low frequencies
        then stay unexcited although the signal looks broadband.
        """
        # int(...) before the multiplication: typeshed types int.__pow__ with a
        # general exponent as Any (negative exponents give float in Python);
        # without the cast the Any edge would propagate into the return value.
        period_steps = int(2**self.prbs_register_length) - 1
        return period_steps * self.prbs_clock_s

    @classmethod
    def from_time_constants(
        cls,
        *,
        tau_fast_s: float,
        tau_slow_s: float,
        sample_time_s: float,
        **overrides: object,
    ) -> ExcitationConfig:
        r"""Constructs hold times and clock times from the time constants of the process.

        Rules:

        * shortest hold time :math:`\approx 2\,\tau_{fast}` -- shorter, one does not see the
          settling, and the excitation acts like noise for the fast mode
        * longest hold time :math:`\approx 3\sqrt{\tau_{fast}\tau_{slow}}`, so that
          steady states also occur in the dataset
        * PRBS clock :math:`\approx 2\,\tau_{fast}`, register length minimal such that the
          period exceeds the **run duration**

        Why the period criterion depends on the run duration and not on
        :math:`\tau_{slow}`: with :math:`\tau_{slow}/\tau_{fast} \approx 1700` a
        period of :math:`3\tau_{slow}` at a clock of :math:`2\tau_{fast}` would only be
        reachable with a 12-bit register, and the slowest mode would thus still be excited
        only once per run. That is the wrong calculation: **the slow mode cannot be
        excited by the actions within a run at all**, because the
        particle growth is monotonic and has no feedback. It is covered by the
        spread of the initial conditions between runs (see
        :class:`InitialConditionDesign`). For the action excitation it is therefore only
        important that the sequence does not repeat within a run --
        repetition wastes sample points without delivering new information.
        """
        hold_minimum = max(2.0 * tau_fast_s, 4.0 * sample_time_s)
        geometric_mean = float(np.sqrt(tau_fast_s * tau_slow_s))
        hold_maximum = max(3.0 * geometric_mean, 4.0 * hold_minimum)
        clock = max(2.0 * tau_fast_s, 2.0 * sample_time_s)
        duration = float(
            overrides.pop("run_duration_s", _DEFAULT_RUN_DURATION_S)  # type: ignore[arg-type]
        )
        register = 7
        while register < max(_LFSR_TAPS) and (2**register - 1) * clock < duration:
            register += 1
        defaults: dict[str, object] = {
            "run_duration_s": duration,
            "hold_time": HoldTimeRange(hold_minimum, hold_maximum),
            "prbs_clock_s": clock,
            "prbs_register_length": register,
            "ramp_segment_time": HoldTimeRange(5.0 * hold_minimum, 4.0 * hold_maximum),
        }
        defaults.update(overrides)
        return cls(**defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------------- Schedule


@dataclass(frozen=True, slots=True)
class ExcitationSchedule:
    """Finished excitation schedule, directly passable to :meth:`ProcessSimulator.rollout`."""

    run_id: tuple[str, ...]
    actions: FloatArray
    disturbances: FloatArray
    initial_states: FloatArray | None
    signal_kinds: tuple[tuple[str, ...], ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.actions.ndim != 3 or self.disturbances.ndim != 3:
            raise ValueError("actions und disturbances brauchen die Form (batch, time, channel)")
        if self.actions.shape[:2] != self.disturbances.shape[:2]:
            raise ValueError("actions und disturbances haben unterschiedliche Batch-/Zeitform")
        if len(self.run_id) != self.actions.shape[0]:
            raise ValueError("run_id passt nicht zur Batch-Dimension")
        if len(self.signal_kinds) != self.actions.shape[0]:
            raise ValueError("signal_kinds passt nicht zur Batch-Dimension")
        if (
            self.initial_states is not None
            and self.initial_states.shape[0] != self.actions.shape[0]
        ):
            raise ValueError("initial_states passt nicht zur Batch-Dimension")

    @property
    def n_runs(self) -> int:
        return int(self.actions.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.actions.shape[1])

    def subset(self, n_runs: int) -> ExcitationSchedule:
        """Nested subset for the learning-curve support points.

        Only with a Sobol design is the subset also a low-discrepancy sequence --
        with LHS it is merely truncated here, which worsens the coverage. The
        sampler is therefore in ``metadata`` and should be named when reporting the
        learning curve.
        """
        if not 0 < n_runs <= self.n_runs:
            raise ValueError(f"n_runs muss in (0, {self.n_runs}] liegen")
        return ExcitationSchedule(
            run_id=self.run_id[:n_runs],
            actions=self.actions[:n_runs],
            disturbances=self.disturbances[:n_runs],
            initial_states=None if self.initial_states is None else self.initial_states[:n_runs],
            signal_kinds=self.signal_kinds[:n_runs],
            metadata={**self.metadata, "subset_of": self.n_runs},
        )


# ----------------------------------------------------------------- Signal generators


class SignalGenerator(ABC):
    """Generates normalised excitation signals in the range :math:`[0, 1]`.

    The normalisation is the workaround from the work plan for the still missing
    action ranges: the design stays unchanged when the real limits come from
    TUHH-A -- only the denormalisation is swapped.
    """

    kind: SignalKind

    @abstractmethod
    def generate(
        self, n_steps: int, sample_time_s: float, rng: np.random.Generator, amplitude: float
    ) -> FloatArray:
        """A signal of length ``n_steps`` with values in :math:`[0, 1]`."""

    @staticmethod
    def _band(rng: np.random.Generator, amplitude: float) -> tuple[float, float]:
        """Random amplitude band of width ``amplitude`` within [0, 1]."""
        half = 0.5 * amplitude
        centre = rng.uniform(half, 1.0 - half) if amplitude < 1.0 else 0.5
        return centre - half, centre + half


class StepSequenceGenerator(SignalGenerator):
    """Step sequence with random amplitudes and hold times."""

    kind: SignalKind = "step"

    def __init__(self, hold_time: HoldTimeRange) -> None:
        self._hold = hold_time

    def generate(
        self, n_steps: int, sample_time_s: float, rng: np.random.Generator, amplitude: float
    ) -> FloatArray:
        lower, upper = self._band(rng, amplitude)
        signal = np.empty(n_steps, dtype=np.float64)
        minimum = max(1, round(self._hold.minimum_s / sample_time_s))
        maximum = max(minimum, round(self._hold.maximum_s / sample_time_s))
        index = 0
        while index < n_steps:
            hold = int(rng.integers(minimum, maximum + 1))
            signal[index : index + hold] = rng.uniform(lower, upper)
            index += hold
        return signal


class PrbsGenerator(SignalGenerator):
    r"""Pseudo-random binary signal from an m-sequence (maximum length sequence).

    A real shift register and not a Bernoulli coin: the m-sequence has an
    approximately flat power spectral density up to half the clock frequency and a
    known period :math:`(2^n - 1)\,T_{clock}`. A coin gives the same spectrum on average,
    but per realisation an uncontrolled distribution of the switching intervals
    -- with five runs in the dataset "on average" is no useful guarantee.
    """

    kind: SignalKind = "prbs"

    def __init__(self, register_length: int, clock_s: float) -> None:
        if register_length not in _LFSR_TAPS:
            raise ValueError(f"Registerlänge {register_length} nicht unterstützt")
        self._length = register_length
        self._clock_s = clock_s

    def generate(
        self, n_steps: int, sample_time_s: float, rng: np.random.Generator, amplitude: float
    ) -> FloatArray:
        lower, upper = self._band(rng, amplitude)
        clock_steps = max(1, round(self._clock_s / sample_time_s))
        n_bits = int(np.ceil(n_steps / clock_steps))
        bits = self._maximum_length_sequence(n_bits, rng)
        signal = np.repeat(np.where(bits, upper, lower), clock_steps)[:n_steps]
        return np.asarray(signal, dtype=np.float64)

    def _maximum_length_sequence(self, n_bits: int, rng: np.random.Generator) -> NDArray[np.bool_]:
        taps = _LFSR_TAPS[self._length]
        register = np.ones(self._length, dtype=bool)
        # random but non-degenerate initial state
        while True:
            candidate = rng.integers(0, 2, self._length).astype(bool)
            if candidate.any():
                register = candidate
                break
        output = np.empty(n_bits, dtype=bool)
        for index in range(n_bits):
            feedback = register[taps[0] - 1]
            for tap in taps[1:]:
                feedback ^= register[tap - 1]
            output[index] = register[-1]
            register = np.roll(register, 1)
            register[0] = feedback
        return output


class RampGenerator(SignalGenerator):
    """Piecewise linear ramps with random slopes.

    Ramps excite the process differently from steps: with a ramp the output follows with
    a constant lag error that is directly proportional to the time constant. For
    a model that is later to deliver multi-step rollouts, this is the more informative
    form of excitation -- steps are easier to hit in one-step prediction than
    they yield for the rollout.
    """

    kind: SignalKind = "ramp"

    def __init__(self, segment_time: HoldTimeRange) -> None:
        self._segment = segment_time

    def generate(
        self, n_steps: int, sample_time_s: float, rng: np.random.Generator, amplitude: float
    ) -> FloatArray:
        lower, upper = self._band(rng, amplitude)
        minimum = max(2, round(self._segment.minimum_s / sample_time_s))
        maximum = max(minimum, round(self._segment.maximum_s / sample_time_s))
        waypoints: list[float] = [float(rng.uniform(lower, upper))]
        positions: list[int] = [0]
        while positions[-1] < n_steps - 1:
            positions.append(positions[-1] + int(rng.integers(minimum, maximum + 1)))
            waypoints.append(float(rng.uniform(lower, upper)))
        grid = np.arange(n_steps, dtype=np.float64)
        interpolated = np.interp(
            grid, np.asarray(positions, dtype=np.float64), np.asarray(waypoints)
        )
        return np.asarray(interpolated, dtype=np.float64)


class ConstantGenerator(SignalGenerator):
    """Constant signal -- reference runs for steady states."""

    kind: SignalKind = "constant"

    def generate(
        self,
        n_steps: int,
        sample_time_s: float,  # noqa: ARG002 - constant signal is time-independent
        rng: np.random.Generator,
        amplitude: float,
    ) -> FloatArray:
        lower, upper = self._band(rng, amplitude)
        return np.full(n_steps, rng.uniform(lower, upper), dtype=np.float64)


# ------------------------------------------------------------------- Rate limiting


def apply_rate_limit(signal: FloatArray, max_rate_per_s: float, sample_time_s: float) -> FloatArray:
    r"""Limits the rate of change of a signal, shape ``(runs, steps)``.

    .. math:: |u[k] - u[k-1]| \le \dot u_{max}\,\Delta t

    Acts like a nonlinear low-pass filter: the spectrum of a rate-limited PRBS is
    no longer flat, because the steps become ramps. The loss of bandwidth is the
    price for realisability and must be visible in the report -- ``excitation_report``
    therefore reports amplitude, realised bandwidth and minimal switching time together.
    """
    if signal.ndim != 2:
        raise ValueError("signal muss die Form (runs, steps) haben")
    limit = max_rate_per_s * sample_time_s
    limited = np.empty_like(signal)
    limited[:, 0] = signal[:, 0]
    for step in range(1, signal.shape[1]):
        delta = np.clip(signal[:, step] - limited[:, step - 1], -limit, limit)
        limited[:, step] = limited[:, step - 1] + delta
    return limited


# -------------------------------------------------------------------- QMC-Sampling


def quasi_random_points(
    n_points: int, n_dimensions: int, *, sampler: SamplerName, seed: int
) -> FloatArray:
    """Low-discrepancy points in the unit cube, shape ``(n_points, n_dimensions)``.

    Sobol is nested and thus suitable for learning curves; LHS is not.
    """
    if sampler == "sobol":
        engine: qmc.QMCEngine = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
    elif sampler == "lhs":
        engine = qmc.LatinHypercube(d=n_dimensions, seed=seed)
    else:  # pragma: no cover - covered by the Literal type
        raise ValueError(f"Unbekannter Sampler {sampler!r}")
    return np.asarray(engine.random(n_points), dtype=np.float64)


# --------------------------------------------------------------- Disturbance designs


# ----------------------------------------------------------- Gas velocity limit


def gas_velocity_cap_from_initial_diameter(
    initial_diameters_m: FloatArray,
    *,
    gas: GasConfig,
    material: MaterialConfig,
    actions: ActionSpaceConfig,
    disturbances: DisturbanceSpaceConfig,
    grid_points: int = 9,
) -> FloatArray:
    r"""Per-run upper limit of the gas velocity: ``u_t`` at the run's INITIAL diameter.

    Rationale (Befund 86): the operator knows the starting material of a run, and the particles
    only grow during the run, so ``u_t`` only rises -- a request of at most ``u_t(d_0)`` never
    leaves the fluidisation window at the upper side. ``u_t`` also depends on the bed temperature
    and the inlet humidity; the cap takes the minimum over the whole a-priori range of both:
    bed temperature between the smallest ambient temperature and the largest inlet temperature of
    the action space, absolute humidity between the corners of the ambient temperature/relative
    humidity ranges, on a ``grid_points`` x ``grid_points`` grid. The lower side needs no cap
    (``u_mf`` stays far below the box minimum for every particle size of the design, Befund 85).
    The result is not smaller than the box minimum of the gas velocity.
    """
    backend = NumpyBackend()
    ambient_t, ambient_rh = (
        disturbances.ambient_temperature_celsius,
        disturbances.ambient_relative_humidity,
    )
    humidity_corners = [
        float(absolute_humidity_kg_per_kg(backend, gas, np.float64(t), np.float64(rh)))
        for t in (ambient_t.minimum, ambient_t.maximum)
        for rh in (ambient_rh.minimum, ambient_rh.maximum)
    ]
    bed_low, bed_high = ambient_t.minimum, actions.inlet_temperature_celsius.maximum
    diameters = np.asarray(initial_diameters_m, dtype=np.float64)
    worst = np.full_like(diameters, np.inf)
    for temperature in np.linspace(bed_low, bed_high, grid_points):
        for humidity in np.linspace(min(humidity_corners), max(humidity_corners), grid_points):
            values = terminal_velocity_m_per_s(
                backend,
                gas,
                material,
                diameters,
                np.full_like(diameters, temperature),
                np.full_like(diameters, humidity),
            )
            worst = np.minimum(worst, np.asarray(values))
    return np.maximum(worst, actions.gas_velocity_m_per_s.minimum)


class DisturbanceDesign(ABC):
    """Generates disturbance trajectories, shape ``(runs, steps, channels)``."""

    @abstractmethod
    def build(
        self,
        space: DisturbanceSpaceConfig,
        unit_points: FloatArray,
        n_steps: int,
        sample_time_s: float,
        rng: np.random.Generator,
    ) -> FloatArray: ...

    def fingerprint_options(self) -> dict[str, object]:
        """Options that enter the excitation fingerprint, **only when set**.

        Empty for the behaviour that existed before an option was added, so that every
        existing schedule keeps its fingerprint (``MixedExcitationDesign._fingerprint``).
        """
        return {}


class PiecewiseConstantDisturbanceDesign(DisturbanceDesign):
    """One disturbance operating point per run, constant within the run.

    Clean for identification: the effect of the disturbances is identifiable between runs
    without being entangled with the actuator dynamics. Unsuitable as the
    only design, because in operation the disturbances drift within a run -- a
    model that has only seen constant disturbances extrapolates in their rate of change.
    """

    def build(
        self,
        space: DisturbanceSpaceConfig,
        unit_points: FloatArray,
        n_steps: int,
        sample_time_s: float,  # noqa: ARG002 - part of the common design interface
        rng: np.random.Generator,  # noqa: ARG002 - deterministic from unit_points
    ) -> FloatArray:
        low = np.array([r.minimum for r in space.ranges])
        high = np.array([r.maximum for r in space.ranges])
        values = low + unit_points * (high - low)
        return np.repeat(values[:, None, :], n_steps, axis=1)


class DiurnalDisturbanceDesign(DisturbanceDesign):
    r"""Diurnal cycle plus Ornstein-Uhlenbeck drift.

    .. math::
        T_{amb}(t) = \bar T + A_T \sin\!\left(\frac{2\pi t}{86400} + \varphi\right) + \xi_T(t)

    The relative humidity gets the same course with a phase shift
    :math:`\varphi_{rel} = \pi` -- warm daytime air is relatively drier. That is
    physically correct and methodologically unpleasant: a deterministic
    anticorrelation between two inputs makes their effects individually not
    identifiable. The OU part breaks up the correlation; ``excitation_report``
    reports the realised correlation so that the decision stays visible.

    The day/night indicator from ``02_prozess_setup.md`` thus arises implicitly. As a
    feature for the model it should be formed from the timestamp as a sin/cos pair,
    not as a binary variable -- a jump at midnight does not exist in the process.

    Solid fraction of the suspension (review E3, 2026-10-09)
    --------------------------------------------------------
    ``solid_fraction_process="diurnal"`` (default, behaviour up to Befund 81) also puts
    ``suspension_solid_fraction`` on the diurnal cycle in phase with ``T_amb``: within a
    run a median correlation of 0.71 (max. 0.94, ``broadband_derived``, 64 runs). This builds
    a confusion of heater power and water load into the data. The solid fraction is
    not a weather channel but a property of the prepared suspension.
    ``"independent"``: the channel gets **no** diurnal cycle but its own
    Ornstein-Uhlenbeck process (standard deviation ``solid_fraction_std_fraction`` times
    span, correlation time ``drift_correlation_time_s``), independent of ``T_amb`` and
    ``RH``. The random numbers are drawn at the same place as before: ``T_amb`` and
    ``RH`` are bit-identical for the same seed in both modes. The default value
    ``0.09`` is a PLACEHOLDER without a source; it is chosen so that the mean
    standard deviation of the channel **within a run of 8 h** matches the previous one
    (64 runs, ``broadband_derived``, seeds 11/12/13: 0.0140-0.0146 against
    0.0135-0.0145), so that only the correlation with ``T_amb`` changes and not the
    variability. Measured at seed 11: correlation within the run with median 0.03
    instead of 0.59 (mean 0.03 instead of 0.56; |correlation| median 0.24 instead of 0.59),
    because two slow independent processes can correlate by chance over 8 h.
    """

    def __init__(
        self,
        *,
        diurnal_amplitude_fraction: float = 0.35,
        drift_amplitude_fraction: float = 0.10,
        drift_correlation_time_s: float = 3600.0,
        humidity_phase_shift_rad: float = float(np.pi),
        solid_fraction_process: SolidFractionProcess = "diurnal",
        solid_fraction_std_fraction: float = 0.09,
    ) -> None:
        if solid_fraction_process not in ("diurnal", "independent"):
            raise ValueError(
                "solid_fraction_process muss 'diurnal' oder 'independent' sein, "
                f"ist {solid_fraction_process!r}"
            )
        if solid_fraction_std_fraction < 0.0:
            raise ValueError("solid_fraction_std_fraction muss >= 0 sein")
        self._diurnal = diurnal_amplitude_fraction
        self._drift = drift_amplitude_fraction
        self._correlation_time_s = drift_correlation_time_s
        self._phase_shift = humidity_phase_shift_rad
        self._solid_fraction_process = solid_fraction_process
        self._solid_fraction_std_fraction = solid_fraction_std_fraction

    def fingerprint_options(self) -> dict[str, object]:
        if self._solid_fraction_process == "diurnal":
            return {}
        return {
            "solid_fraction_process": self._solid_fraction_process,
            "solid_fraction_std_fraction": self._solid_fraction_std_fraction,
        }

    def build(
        self,
        space: DisturbanceSpaceConfig,
        unit_points: FloatArray,
        n_steps: int,
        sample_time_s: float,
        rng: np.random.Generator,
    ) -> FloatArray:
        n_runs, n_channels = unit_points.shape
        low = np.array([r.minimum for r in space.ranges])
        high = np.array([r.maximum for r in space.ranges])
        span = high - low
        centre = low + unit_points * span

        time_s = np.arange(n_steps, dtype=np.float64) * sample_time_s
        start_phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_runs, 1))
        base_phase = 2.0 * np.pi * time_s[None, :] / _SECONDS_PER_DAY + start_phase

        names = space.names
        signal = np.empty((n_runs, n_steps, n_channels), dtype=np.float64)
        for channel, name in enumerate(names):
            if self._solid_fraction_process == "independent" and "solid_fraction" in name:
                # Own process, no diurnal term (review E3). Same draw pattern as the drift of
                # the other channels, so T_amb and RH do not change.
                process = self._ornstein_uhlenbeck(
                    n_runs,
                    n_steps,
                    sample_time_s,
                    self._solid_fraction_std_fraction * span[channel],
                    rng,
                )
                signal[:, :, channel] = centre[:, channel, None] + process
                continue
            phase = base_phase + (self._phase_shift if "humidity" in name else 0.0)
            diurnal = 0.5 * self._diurnal * span[channel] * np.sin(phase)
            drift = self._ornstein_uhlenbeck(
                n_runs, n_steps, sample_time_s, 0.5 * self._drift * span[channel], rng
            )
            signal[:, :, channel] = centre[:, channel, None] + diurnal + drift
        return np.clip(signal, low, high)

    def _ornstein_uhlenbeck(
        self,
        n_runs: int,
        n_steps: int,
        sample_time_s: float,
        standard_deviation: float,
        rng: np.random.Generator,
    ) -> FloatArray:
        decay = float(np.exp(-sample_time_s / self._correlation_time_s))
        noise_scale = standard_deviation * float(np.sqrt(1.0 - decay**2))
        path = np.empty((n_runs, n_steps), dtype=np.float64)
        path[:, 0] = rng.normal(0.0, standard_deviation, n_runs)
        for step in range(1, n_steps):
            path[:, step] = decay * path[:, step - 1] + rng.normal(0.0, noise_scale, n_runs)
        return path


# --------------------------------------------------------- Initial condition design


@dataclass(frozen=True, slots=True)
class InitialConditionRanges:
    """Spread range of the initial states.

    :param diameter_log_uniform: spread of :math:`d_{50}` logarithmic instead of linear.
        The growth rate is proportional to :math:`d_{50}`, so equal
        *relative* distances are the natural metric of the slow mode.
    """

    diameter_min_m: float = 250e-6
    diameter_max_m: float = 900e-6
    moisture_min_kg_per_kg: float | None = None
    moisture_max_kg_per_kg: float | None = None
    bed_temperature_min_celsius: float = 35.0
    bed_temperature_max_celsius: float = 80.0
    solid_mass_fraction_min: float = 0.8
    solid_mass_fraction_max: float = 1.2
    diameter_log_uniform: bool = True


class InitialConditionDesign:
    """Spreads the initial states over the operating range."""

    def __init__(self, ranges: InitialConditionRanges | None = None) -> None:
        self._ranges = ranges if ranges is not None else InitialConditionRanges()

    @property
    def n_dimensions(self) -> int:
        return 4

    def build(
        self,
        unit_points: FloatArray,
        *,
        geometry: GeometryConfig,
        material: MaterialConfig,
        operation: OperationConfig,
    ) -> FloatArray:
        """Physical initial quantities, shape ``(runs, 4)``.

        Column order: solid mass [kg], particle moisture [kg/kg],
        diameter [m], bed temperature [°C]. The conversion into the state vector
        is done by ``GreyBoxFluidBedGranulator.state_from_physical``, so that knowledge
        of the state encoding stays in the model.
        """
        ranges = self._ranges
        moisture_low = (
            ranges.moisture_min_kg_per_kg
            if ranges.moisture_min_kg_per_kg is not None
            else material.equilibrium_moisture_kg_per_kg
        )
        moisture_high = (
            ranges.moisture_max_kg_per_kg
            if ranges.moisture_max_kg_per_kg is not None
            else material.critical_moisture_kg_per_kg
        )

        solid_mass = geometry.bed_mass_solid_setpoint_kg * (
            ranges.solid_mass_fraction_min
            + unit_points[:, 0] * (ranges.solid_mass_fraction_max - ranges.solid_mass_fraction_min)
        )
        moisture = moisture_low + unit_points[:, 1] * (moisture_high - moisture_low)
        if ranges.diameter_log_uniform:
            log_low, log_high = np.log(ranges.diameter_min_m), np.log(ranges.diameter_max_m)
            diameter = np.exp(log_low + unit_points[:, 2] * (log_high - log_low))
        else:
            diameter = ranges.diameter_min_m + unit_points[:, 2] * (
                ranges.diameter_max_m - ranges.diameter_min_m
            )
        temperature = ranges.bed_temperature_min_celsius + unit_points[:, 3] * (
            ranges.bed_temperature_max_celsius - ranges.bed_temperature_min_celsius
        )
        _ = operation
        return np.stack([solid_mass, moisture, diameter, temperature], axis=-1)


# --------------------------------------------------------------------- Overall design


class ExcitationDesign(ABC):
    """Generates a complete excitation schedule."""

    @abstractmethod
    def build(self, n_runs: int, *, seed: int) -> ExcitationSchedule: ...


class MixedExcitationDesign(ExcitationDesign):
    r"""Step sequences, PRBS and ramps on the actions, QMC on disturbances and start.

    Composition per run: for every action channel, signal kind and
    amplitude level are drawn independently. Why independently and not one common design for all
    channels: if all channels ran the same step sequence simultaneously, their
    effects on the target would be collinear and the partial derivatives
    :math:`\\partial \\dot E/\\partial T_{in}` and
    :math:`\\partial \\dot E/\\partial u_{gas}` no longer separable. Exactly these
    derivatives are what the monotonicity constraints are to enforce later -- on a
    collinear dataset that cannot be checked.

    :param sample_time_s: sample time; must match that of the simulator
    :param initial_condition_design: ``None`` switches off the spread of the initial states
    """

    def __init__(
        self,
        config: ExcitationConfig,
        action_space: ActionSpaceConfig,
        disturbance_space: DisturbanceSpaceConfig,
        *,
        sample_time_s: float,
        geometry: GeometryConfig,
        material: MaterialConfig,
        operation: OperationConfig,
        disturbance_design: DisturbanceDesign | None = None,
        initial_condition_design: InitialConditionDesign | None = None,
        name: str = "mixed",
        gas: GasConfig | None = None,
        gas_velocity_cap: GasVelocityCap = "none",
    ) -> None:
        if gas_velocity_cap not in ("none", "initial_diameter"):
            raise ValueError(f"unknown gas_velocity_cap {gas_velocity_cap!r}")
        if gas_velocity_cap != "none" and gas is None:
            raise ValueError("gas_velocity_cap needs the GasConfig (gas=...)")
        self._gas = gas
        self._gas_velocity_cap = gas_velocity_cap
        self._config = config
        self._actions = action_space
        self._disturbances = disturbance_space
        self._sample_time_s = sample_time_s
        self._geometry = geometry
        self._material = material
        self._operation = operation
        self._disturbance_design = (
            disturbance_design if disturbance_design is not None else DiurnalDisturbanceDesign()
        )
        self._initial_design = (
            initial_condition_design
            if initial_condition_design is not None
            else (InitialConditionDesign() if config.randomise_initial_conditions else None)
        )
        self._name = name
        self._generators: dict[str, SignalGenerator] = {
            "step": StepSequenceGenerator(config.hold_time),
            "prbs": PrbsGenerator(config.prbs_register_length, config.prbs_clock_s),
            "ramp": RampGenerator(config.ramp_segment_time),
            "constant": ConstantGenerator(),
        }

    @property
    def config(self) -> ExcitationConfig:
        return self._config

    @property
    def n_steps(self) -> int:
        return round(self._config.run_duration_s / self._sample_time_s)

    def build(self, n_runs: int, *, seed: int) -> ExcitationSchedule:
        """Builds the schedule. Deterministic in ``(config, n_runs, seed)``."""
        if n_runs < 1:
            raise ValueError("n_runs muss mindestens 1 sein")
        config = self._config
        n_steps = self.n_steps
        rng = np.random.default_rng(seed)

        n_disturbance = len(self._disturbances.names)
        n_initial = 0 if self._initial_design is None else self._initial_design.n_dimensions
        unit_points = quasi_random_points(
            n_runs, n_disturbance + n_initial, sampler=config.sampler, seed=seed
        )

        disturbances = self._disturbance_design.build(
            self._disturbances,
            unit_points[:, :n_disturbance],
            n_steps,
            self._sample_time_s,
            rng,
        )

        initial_states: FloatArray | None = None
        if self._initial_design is not None:
            initial_states = self._initial_design.build(
                unit_points[:, n_disturbance:],
                geometry=self._geometry,
                material=self._material,
                operation=self._operation,
            )

        actions, kinds = self._build_actions(n_runs, n_steps, rng)
        caps: FloatArray | None = None
        if self._gas_velocity_cap == "initial_diameter":
            assert self._gas is not None
            diameters = (
                initial_states[:, 2]
                if initial_states is not None
                else np.full(n_runs, self._material.initial_diameter_m)
            )
            caps = gas_velocity_cap_from_initial_diameter(
                diameters,
                gas=self._gas,
                material=self._material,
                actions=self._actions,
                disturbances=self._disturbances,
            )
            gas_index = self._actions.names.index("gas_velocity_m_per_s")
            # min with a constant keeps the rate limit of the signal
            actions[:, :, gas_index] = np.minimum(actions[:, :, gas_index], caps[:, None])
        fingerprint = self._fingerprint(n_runs, seed)
        return ExcitationSchedule(
            run_id=tuple(f"{self._name}-{fingerprint}-{index:05d}" for index in range(n_runs)),
            actions=actions,
            disturbances=disturbances,
            initial_states=initial_states,
            signal_kinds=kinds,
            metadata={
                "design": type(self).__name__,
                "design_name": self._name,
                "fingerprint": fingerprint,
                "seed": seed,
                "sampler": config.sampler,
                "sample_time_s": self._sample_time_s,
                "run_duration_s": config.run_duration_s,
                "prbs_period_s": config.prbs_period_s,
                "rate_limited": config.apply_rate_limits,
                "gas_velocity_cap": self._gas_velocity_cap,
                "gas_velocity_cap_m_per_s": None if caps is None else caps.tolist(),
                "action_names": self._actions.names,
                "disturbance_names": self._disturbances.names,
                "initial_condition_columns": (
                    "solid_mass_kg",
                    "moisture_kg_per_kg",
                    "diameter_m",
                    "bed_temperature_celsius",
                ),
            },
        )

    def _build_actions(
        self, n_runs: int, n_steps: int, rng: np.random.Generator
    ) -> tuple[FloatArray, tuple[tuple[str, ...], ...]]:
        config = self._config
        bounds = self._actions.bounds
        n_channels = len(bounds)

        weight_names = sorted(config.signal_weights)
        weights = np.array([config.signal_weights[k] for k in weight_names], dtype=np.float64)
        weights = weights / weights.sum()

        constant_runs = _constant_run_flags(n_runs, config.constant_run_fraction)
        normalised = np.empty((n_runs, n_steps, n_channels), dtype=np.float64)
        kinds: list[tuple[str, ...]] = []
        for run in range(n_runs):
            run_kinds: list[str] = []
            for channel in range(n_channels):
                if constant_runs[run]:
                    kind = "constant"
                    amplitude = 1.0
                else:
                    kind = str(rng.choice(weight_names, p=weights))
                    amplitude = float(rng.choice(config.amplitude_levels))
                normalised[run, :, channel] = self._generators[kind].generate(
                    n_steps, self._sample_time_s, rng, amplitude
                )
                run_kinds.append(kind)
            kinds.append(tuple(run_kinds))

        actions = np.empty_like(normalised)
        for channel, bound in enumerate(bounds):
            actions[:, :, channel] = bound.minimum + normalised[:, :, channel] * bound.span
            if config.apply_rate_limits:
                actions[:, :, channel] = apply_rate_limit(
                    actions[:, :, channel], bound.max_rate_per_s, self._sample_time_s
                )
        return actions, tuple(kinds)

    def _fingerprint(self, n_runs: int, seed: int) -> str:
        import hashlib
        import json

        payload = {
            "config": {
                f.name: _serialise(getattr(self._config, f.name)) for f in fields(self._config)
            },
            "actions": [(b.minimum, b.maximum, b.max_rate_per_s) for b in self._actions.bounds],
            "disturbances": [(r.minimum, r.maximum) for r in self._disturbances.ranges],
            "sample_time_s": self._sample_time_s,
            "disturbance_design": type(self._disturbance_design).__name__,
            "initial_design": type(self._initial_design).__name__
            if self._initial_design is not None
            else None,
            "n_runs": n_runs,
            "seed": seed,
        }
        if self._gas_velocity_cap != "none":
            # Only when set: schedules built before the option keep their fingerprint.
            payload["gas_velocity_cap"] = self._gas_velocity_cap
        options = self._disturbance_design.fingerprint_options()
        if options:
            # Only when set: schedules built before the option keep their fingerprint.
            payload["disturbance_design_options"] = options
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return digest[:8]


def _constant_run_flags(n_runs: int, fraction: float) -> NDArray[np.bool_]:
    r"""Distributes the constant runs evenly over the design order.

    Allocation by the fractional part: run ``i`` is constant if
    :math:`(i f) \bmod 1 < f`. Thus **every prefix** of length :math:`n` has approximately
    the fraction ``fraction`` of constant runs, and the first lies at index 0 -- an
    allocation at the end of the block would have delivered none at all in the prefix of length 8.

    Why this is not arbitrary: the earlier rule ``run < n_constant`` put all
    constant runs as a block at the beginning. With ``fraction = 0.1`` and a 32-run design
    the smallest learning-curve support point ``subset(8)`` thereby contained 3 of 8 runs
    without any excitation -- 37 % instead of 10 %. In addition, with a
    prefix-block split they all ended up in ``train``, so that validation and test did not see
    a single steady-state reference run. Both distort exactly the left end of the
    learning curve, i.e. the statement on the training data requirement.

    Caution: this change shifts the signal kinds compared with datasets generated with
    an earlier version. The excitation fingerprint stays the same, because
    it hashes only the configuration -- existing runs must therefore be **regenerated**.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("constant_run_fraction muss in [0, 1] liegen")
    if fraction == 0.0:
        return np.zeros(n_runs, dtype=np.bool_)
    index = np.arange(n_runs, dtype=np.float64)
    return np.asarray(np.mod(index * fraction, 1.0) < fraction - 1e-12, dtype=np.bool_)


def _serialise(value: object) -> object:
    if isinstance(value, HoldTimeRange):
        return [value.minimum_s, value.maximum_s]
    if isinstance(value, dict | tuple | list):
        return str(value)
    return value


# ------------------------------------------------------------- Serialisation


def load_excitation_config(path: str | Path) -> ExcitationConfig:
    """Loads an excitation config from YAML.

    Even if the hold times are usually derived via
    :meth:`ExcitationConfig.from_time_constants`: the **resolved**
    state must be recorded with every dataset. Otherwise a learning curve is later
    not reproducible, because the derived values shift with every change of the
    process config.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: erwartet ein YAML-Mapping")
    known = {f.name for f in fields(ExcitationConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"ExcitationConfig: unbekannte Felder {sorted(unknown)}")
    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key in ("hold_time", "ramp_segment_time"):
            kwargs[key] = HoldTimeRange(**value)
        elif key == "amplitude_levels":
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return ExcitationConfig(**kwargs)  # type: ignore[arg-type]


def dump_excitation_config(config: ExcitationConfig, path: str | Path) -> None:
    """Serialises the fully resolved excitation config."""
    payload = {
        field.name: (
            {"minimum_s": value.minimum_s, "maximum_s": value.maximum_s}
            if isinstance(value := getattr(config, field.name), HoldTimeRange)
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for field in fields(config)
    }
    Path(path).write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
