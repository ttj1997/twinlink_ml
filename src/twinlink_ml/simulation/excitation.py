r"""Anregungsdesign (excitation design) für die Datengenerierung.

Warum das ein eigenes Modul mit Abstraktionen ist und nicht ein Skript
---------------------------------------------------------------------
Das Anregungsdesign entscheidet darüber, **welche Aufgabe** die Sequenzmodelle später
lernen. Es ist damit ein Versuchsparameter auf derselben Ebene wie die Architektur, und es
muss versioniert, reproduzierbar und gegenüber TUHH-A dokumentierbar sein --
``02_prozess_setup.md``
hält fest, dass das Sampling-Design abgestimmt sein muss, *bevor* dort 18 Monate lang
Läufe produziert werden. Ein Skript, das ein paar ``rng.uniform``-Aufrufe enthält, ist
keine Grundlage für dieses Gespräch.

Drei Entwurfsentscheidungen, die vom Arbeitsplan abweichen
----------------------------------------------------------
**1. Sobol statt Latin-Hypercube als Standard.** Für Lernkurven werden Datensätze mit
5, 10, 25, 50, 100 Läufen gebraucht. Bei LHS ist ein 5er-Design kein Teil des
100er-Designs: die Punkte müssen neu gezogen werden, und die Raumabdeckung ändert sich
zwischen den Stützstellen der Lernkurve. Dann vermischt die Kurve zwei Effekte --
Datenmenge und Abdeckungsqualität. Sobol-Folgen sind **verschachtelt** (nested): die
ersten :math:`n` Punkte einer Sobol-Folge sind selbst eine Folge niedriger Diskrepanz.
Die Lernkurven-Datensätze sind damit echte Teilmengen, und die Kurve misst nur noch die
Datenmenge. ``test_sobol_design_is_nested`` prüft genau das, ``test_latin_hypercube_is_not_nested``
den Gegensatz. LHS bleibt als Option erhalten.

**2. Ratenbegrenzung (slew rate limiting) ist Teil des Designs, nicht optional.**
Ein PRBS springt in einem Abtastschritt über die halbe Stellgrößenspanne. Keine reale
Zuluftregelung tut das. Ohne Ratenbegrenzung enthält der Datensatz Übergänge, die die
Anlage nicht ausführen kann -- und weil im Grey-Box-Modell die Heizerdynamik fehlt
(siehe Schritt 1), ist die Ratenbegrenzung derzeit der **einzige** Ort, an dem die
Aktordynamik überhaupt repräsentiert ist.

Daraus folgt eine harte Entwurfsgrenze: die minimal realisierbare Umschaltzeit eines
Kanals ist

.. math:: t_{min} = \frac{\Delta u}{\dot u_{max}}

Für die Zulufttemperatur mit den Platzhalterwerten (Spanne 60 K, 0.5 K/s) sind das 120 s
-- **mehr** als die thermische Zeitkonstante von 84 s. Über die volle Spanne ist die
Sprungantwort des Betts also grundsätzlich nicht messbar. Wer die schnelle Dynamik anregen
will, muss die Amplitude reduzieren. Deshalb ist die Amplitude hier ein
Designparameter mit mehreren Stufen und keine Konstante.

**3. Anfangszustände werden mitvariiert.** Startet jeder Lauf bei
:math:`d_{50} = 400\,\mu m` und wächst monoton, dann ist :math:`d_{50}` innerhalb eines
Laufs praktisch eine Funktion des Zeitindex. Ein Modell mit Positionsinformation
(Transformer mit Positional Encoding, aber auch ein RNN über die Sequenzposition) kann die
langsame Mode dann aus der Schrittzahl "vorhersagen", ohne die Physik zu lernen. Das ist
kein Leakage im Sinne überlappender Fenster, wirkt aber genauso: der Vergleich fällt zu
Gunsten der Architektur mit dem stärksten Positionssignal aus. Gegenmittel ist die
Streuung der Anfangsbedingungen über den Betriebsbereich.
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

# Rückkopplungspolynome maximaler Länge (m-Sequenzen), Registerlänge -> Abgriffe
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
    """Verweilzeitbereich einer Sprungfolge in Sekunden."""

    minimum_s: float
    maximum_s: float

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_s <= self.maximum_s:
            raise ValueError("Verweilzeitbereich muss positiv und aufsteigend sein")


@dataclass(frozen=True, slots=True)
class ExcitationConfig:
    """Parameter des Anregungsdesigns.

    :param amplitude_levels: Amplituden als Anteil der Stellgrößenspanne. Mehrere Stufen
        sind notwendig, nicht kosmetisch: kleine Amplituden erlauben schnelle
        Umschaltungen innerhalb der Ratenbegrenzung und regen damit die schnellen Moden an,
        große Amplituden decken den Betriebsraum ab. Ein einziger Amplitudenwert kann
        nicht beides.
    :param signal_weights: relative Häufigkeit der Signalarten. Wird pro Lauf **und pro
        Kanal** unabhängig gezogen -- das dekorreliert die Kanäle, ohne sie künstlich zu
        orthogonalisieren.
    :param constant_run_fraction: Anteil Läufe mit konstanten Stellgrößen. Diese Läufe
        sind für den Architekturvergleich uninformativ, aber die Referenz für
        Beharrungszustände und für die Prüfung, ob ein Modell einen stationären Punkt
        halten kann. Ohne sie fehlt im Datensatz jede Information darüber, wie der Prozess
        aussieht, wenn nichts passiert. Bei ``1.0`` besteht der gesamte Datensatz aus
        Konstantläufen -- das ist der Auswertungsdatensatz, der die reale Betriebsweise
        der Referenzanlage abbildet (konstante Stellgrößen im laufenden Prozess). Er ist
        Auswertungs-, nicht Trainingsdatensatz und für den Architekturvergleich allein
        nicht geeignet.
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
        """Periodendauer der m-Sequenz.

        Muss über der Einschwingzeit des Prozesses liegen, sonst wiederholt sich die
        Anregung, bevor der Prozess auf sie geantwortet hat -- die tiefen Frequenzen
        bleiben dann unangeregt, obwohl das Signal breitbandig aussieht.
        """
        # int(...) vor der Multiplikation: typeshed typisiert int.__pow__ mit
        # allgemeinem Exponenten als Any (negative Exponenten liefern in Python float),
        # ohne den Cast würde die Any-Kante bis in den Rückgabewert durchschlagen.
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
        r"""Konstruiert Verweilzeiten und Taktzeiten aus den Zeitkonstanten des Prozesses.

        Regeln:

        * kürzeste Verweilzeit :math:`\approx 2\,\tau_{fast}` -- kürzer sieht man das
          Einschwingen nicht, und die Anregung wirkt für die schnelle Mode wie Rauschen
        * längste Verweilzeit :math:`\approx 3\sqrt{\tau_{fast}\tau_{slow}}`, damit auch
          Beharrungszustände im Datensatz vorkommen
        * PRBS-Takt :math:`\approx 2\,\tau_{fast}`, Registerlänge minimal so, dass die
          Periode die **Laufdauer** übersteigt

        Warum das Periodenkriterium an der Laufdauer hängt und nicht an
        :math:`\tau_{slow}`: mit :math:`\tau_{slow}/\tau_{fast} \approx 1700` wäre eine
        Periode von :math:`3\tau_{slow}` bei einem Takt von :math:`2\tau_{fast}` nur mit
        einem 12-Bit-Register erreichbar, und die langsamste Mode wäre damit pro Lauf
        immer noch einmal angeregt. Das ist die falsche Rechnung: **die langsame Mode ist
        innerhalb eines Laufs überhaupt nicht durch die Stellgrößen anregbar**, weil das
        Partikelwachstum monoton ist und keine Rückführung besitzt. Sie wird über die
        Streuung der Anfangsbedingungen zwischen den Läufen abgedeckt (siehe
        :class:`InitialConditionDesign`). Für die Stellgrößenanregung ist deshalb nur
        wichtig, dass sich die Sequenz innerhalb eines Laufs nicht wiederholt --
        Wiederholung verschwendet Abtastpunkte, ohne neue Information zu liefern.
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
    """Fertiger Anregungsplan, direkt an :meth:`ProcessSimulator.rollout` übergebbar."""

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
        """Verschachtelte Teilmenge für die Lernkurven-Stützstellen.

        Nur bei Sobol-Design ist die Teilmenge auch eine Folge niedriger Diskrepanz --
        bei LHS wird hier lediglich abgeschnitten, was die Abdeckung verschlechtert. Der
        Sampler steht deshalb in ``metadata`` und sollte beim Berichten der Lernkurve
        mitgenannt werden.
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


# ----------------------------------------------------------------- Signalgeneratoren


class SignalGenerator(ABC):
    """Erzeugt normierte Anregungssignale im Berech :math:`[0, 1]`.

    Die Normierung ist die Umgehung aus dem Arbeitsplan für die noch fehlenden
    Stellgrößenbereiche: das Design bleibt unverändert, wenn die echten Grenzen von
    TUHH-A kommen -- getauscht wird nur die Denormierung.
    """

    kind: SignalKind

    @abstractmethod
    def generate(
        self, n_steps: int, sample_time_s: float, rng: np.random.Generator, amplitude: float
    ) -> FloatArray:
        """Ein Signal der Länge ``n_steps`` mit Werten in :math:`[0, 1]`."""

    @staticmethod
    def _band(rng: np.random.Generator, amplitude: float) -> tuple[float, float]:
        """Zufälliges Amplitudenband der Breite ``amplitude`` innerhalb von [0, 1]."""
        half = 0.5 * amplitude
        centre = rng.uniform(half, 1.0 - half) if amplitude < 1.0 else 0.5
        return centre - half, centre + half


class StepSequenceGenerator(SignalGenerator):
    """Sprungfolge mit zufälligen Amplituden und Verweilzeiten."""

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
    r"""Pseudo-Rausch-Binärsignal aus einer m-Sequenz (maximum length sequence).

    Ein echtes Schieberegister und keine Bernoulli-Münze: die m-Sequenz hat ein
    näherungsweise flaches Leistungsdichtespektrum bis zur halben Taktfrequenz und eine
    bekannte Periode :math:`(2^n - 1)\,T_{clock}`. Eine Münze liefert im Mittel dasselbe
    Spektrum, aber pro Realisierung eine unkontrollierte Verteilung der Umschaltabstände
    -- bei fünf Läufen im Datensatz ist "im Mittel" keine nützliche Zusicherung.
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
        # zufälliger, aber nicht entarteter Startzustand
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
    """Stückweise lineare Rampen mit zufälligen Steigungen.

    Rampen regen den Prozess anders an als Sprünge: bei einer Rampe folgt der Ausgang mit
    einer konstanten Schleppabweichung, die direkt proportional zur Zeitkonstante ist. Für
    ein Modell, das später Mehrschritt-Rollouts liefern soll, ist das die informativere
    Anregungsform -- Sprünge sind bei der Ein-Schritt-Vorhersage leichter zu treffen, als
    sie für den Rollout hergeben.
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
    """Konstantes Signal -- Referenzläufe für Beharrungszustände."""

    kind: SignalKind = "constant"

    def generate(
        self,
        n_steps: int,
        sample_time_s: float,  # noqa: ARG002 - Konstantsignal ist zeitunabhängig
        rng: np.random.Generator,
        amplitude: float,
    ) -> FloatArray:
        lower, upper = self._band(rng, amplitude)
        return np.full(n_steps, rng.uniform(lower, upper), dtype=np.float64)


# ------------------------------------------------------------------- Ratenbegrenzung


def apply_rate_limit(signal: FloatArray, max_rate_per_s: float, sample_time_s: float) -> FloatArray:
    r"""Begrenzt die Änderungsrate eines Signals, Form ``(runs, steps)``.

    .. math:: |u[k] - u[k-1]| \le \dot u_{max}\,\Delta t

    Wirkt wie ein nichtlinearer Tiefpass: das Spektrum eines ratenbegrenzten PRBS ist
    nicht mehr flach, weil die Sprünge zu Rampen werden. Der Verlust an Bandbreite ist der
    Preis für Realisierbarkeit und muss im Report sichtbar sein -- ``excitation_report``
    weist Amplitude, realisierte Bandbreite und minimale Umschaltzeit deshalb gemeinsam aus.
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
    """Punkte niedriger Diskrepanz im Einheitswürfel, Form ``(n_points, n_dimensions)``.

    Sobol ist verschachtelt und damit für Lernkurven geeignet; LHS ist es nicht.
    """
    if sampler == "sobol":
        engine: qmc.QMCEngine = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
    elif sampler == "lhs":
        engine = qmc.LatinHypercube(d=n_dimensions, seed=seed)
    else:  # pragma: no cover - durch Literal-Typ abgedeckt
        raise ValueError(f"Unbekannter Sampler {sampler!r}")
    return np.asarray(engine.random(n_points), dtype=np.float64)


# --------------------------------------------------------------- Störgrößen-Designs


# ----------------------------------------------------------- Gasgeschwindigkeits-Grenze


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
    """Erzeugt Störgrößen-Verläufe, Form ``(runs, steps, channels)``."""

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
    """Ein Störgrößen-Betriebspunkt pro Lauf, innerhalb des Laufs konstant.

    Sauber für die Identifikation: die Wirkung der Störgrößen ist zwischen den Läufen
    identifizierbar, ohne dass sie mit der Aktordynamik verschränkt ist. Ungeeignet als
    einziges Design, weil die Störgrößen im Betrieb innerhalb eines Laufs driften -- ein
    Modell, das nur konstante Störgrößen gesehen hat, extrapoliert auf ihre Änderungsrate.
    """

    def build(
        self,
        space: DisturbanceSpaceConfig,
        unit_points: FloatArray,
        n_steps: int,
        sample_time_s: float,  # noqa: ARG002 - Teil des gemeinsamen Design-Interfaces
        rng: np.random.Generator,  # noqa: ARG002 - deterministisch aus unit_points
    ) -> FloatArray:
        low = np.array([r.minimum for r in space.ranges])
        high = np.array([r.maximum for r in space.ranges])
        values = low + unit_points * (high - low)
        return np.repeat(values[:, None, :], n_steps, axis=1)


class DiurnalDisturbanceDesign(DisturbanceDesign):
    r"""Tagesgang plus Ornstein-Uhlenbeck-Drift.

    .. math::
        T_{amb}(t) = \bar T + A_T \sin\!\left(\frac{2\pi t}{86400} + \varphi\right) + \xi_T(t)

    Die relative Feuchte erhält denselben Verlauf mit Phasenverschiebung
    :math:`\varphi_{rel} = \pi` -- warme Tagesluft ist relativ trockener. Das ist
    physikalisch richtig und methodisch unangenehm: eine deterministische
    Antikorrelation zwischen zwei Eingängen macht deren Wirkungen einzeln nicht
    identifizierbar. Der OU-Anteil bricht die Korrelation auf; ``excitation_report``
    weist die realisierte Korrelation aus, damit die Entscheidung sichtbar bleibt.

    Der Tag/Nacht-Indikator aus ``02_prozess_setup.md`` entsteht damit implizit. Als
    Feature für das Modell sollte er aus dem Zeitstempel als sin/cos-Paar gebildet werden,
    nicht als Binärgröße -- ein Sprung um Mitternacht existiert im Prozess nicht.

    Feststoffanteil der Suspension (Review E3, 2026-10-09)
    ------------------------------------------------------
    ``solid_fraction_process="diurnal"`` (Voreinstellung, Verhalten bis Befund 81) legt auch
    ``suspension_solid_fraction`` auf den Tagesgang in Phase mit ``T_amb``: innerhalb eines
    Laufs Korrelation im Median 0.71 (max. 0.94, ``broadband_derived``, 64 Läufe). Das baut
    eine Verwechslung von Heizleistung und Wasserlast in die Daten. Der Feststoffanteil ist
    kein Wetterkanal, sondern eine Eigenschaft der angesetzten Suspension.
    ``"independent"``: der Kanal bekommt **keinen** Tagesgang, sondern einen eigenen
    Ornstein-Uhlenbeck-Prozess (Standardabweichung ``solid_fraction_std_fraction`` mal
    Spanne, Korrelationszeit ``drift_correlation_time_s``), unabhängig von ``T_amb`` und
    ``RH``. Die Zufallszahlen werden an derselben Stelle gezogen wie bisher: ``T_amb`` und
    ``RH`` sind für denselben Seed in beiden Betriebsarten bitgleich. Der Standardwert
    ``0.09`` ist PLATZHALTER ohne Quelle; er ist so gewählt, dass die mittlere
    Standardabweichung des Kanals **innerhalb eines Laufs von 8 h** der bisherigen
    entspricht (64 Läufe, ``broadband_derived``, Seeds 11/12/13: 0.0140-0.0146 gegen
    0.0135-0.0145), damit sich nur die Korrelation mit ``T_amb`` ändert und nicht die
    Variabilität. Gemessen bei Seed 11: Korrelation innerhalb des Laufs im Median 0.03
    statt 0.59 (Mittel 0.03 statt 0.56; |Korrelation| im Median 0.24 statt 0.59), weil zwei
    langsame unabhängige Prozesse über 8 h zufällig korrelieren können.
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


# --------------------------------------------------------- Anfangsbedingungs-Design


@dataclass(frozen=True, slots=True)
class InitialConditionRanges:
    """Streubereich der Anfangszustände.

    :param diameter_log_uniform: Streuung von :math:`d_{50}` logarithmisch statt linear.
        Die Wachstumsrate ist proportional zu :math:`d_{50}`, also sind gleiche
        *relative* Abstände die natürliche Metrik der langsamen Mode.
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
    """Streut die Anfangszustände über den Betriebsbereich."""

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
        """Physikalische Anfangsgrößen, Form ``(runs, 4)``.

        Spaltenreihenfolge: Feststoffmasse [kg], Partikelfeuchte [kg/kg],
        Durchmesser [m], Betttemperatur [°C]. Die Umrechnung in den Zustandsvektor
        übernimmt ``GreyBoxFluidBedGranulator.state_from_physical``, damit die Kenntnis
        der Zustandskodierung im Modell bleibt.
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


# --------------------------------------------------------------------- Gesamtdesign


class ExcitationDesign(ABC):
    """Erzeugt einen vollständigen Anregungsplan."""

    @abstractmethod
    def build(self, n_runs: int, *, seed: int) -> ExcitationSchedule: ...


class MixedExcitationDesign(ExcitationDesign):
    r"""Sprungfolgen, PRBS und Rampen auf den Stellgrößen, QMC auf Störgrößen und Start.

    Zusammensetzung pro Lauf: für jeden Stellgrößenkanal wird unabhängig Signalart und
    Amplitudenstufe gezogen. Warum unabhängig und nicht ein gemeinsames Design für alle
    Kanäle: würden alle Kanäle gleichzeitig dieselbe Sprungfolge fahren, wären ihre
    Wirkungen auf die Zielgröße kollinear und die partiellen Ableitungen
    :math:`\\partial \\dot E/\\partial T_{in}` und
    :math:`\\partial \\dot E/\\partial u_{gas}` nicht mehr trennbar. Genau diese
    Ableitungen sollen die Monotonie-Constraints später erzwingen -- auf einem
    kollinearen Datensatz ist das nicht überprüfbar.

    :param sample_time_s: Abtastzeit; muss mit der des Simulators übereinstimmen
    :param initial_condition_design: ``None`` schaltet die Streuung der Anfangszustände ab
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
        """Baut den Plan. Deterministisch in ``(config, n_runs, seed)``."""
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
    r"""Verteilt die Konstantläufe gleichmäßig über die Design-Reihenfolge.

    Zuteilung nach dem gebrochenen Anteil: Lauf ``i`` ist konstant, wenn
    :math:`(i f) \bmod 1 < f`. Damit hat **jedes Präfix** der Länge :math:`n` ungefähr
    den Anteil ``fraction`` an Konstantläufen, und der erste liegt bei Index 0 -- eine
    Zuteilung am Blockende hätte im Präfix der Länge 8 gar keinen geliefert.

    Warum das nicht beliebig ist: die frühere Regel ``run < n_constant`` legte alle
    Konstantläufe als Block an den Anfang. Bei ``fraction = 0.1`` und einem 32er-Design
    enthielt die kleinste Lernkurven-Stützstelle ``subset(8)`` dadurch 3 von 8 Läufen
    ohne jede Anregung -- 37 % statt 10 %. Zusätzlich landeten sie bei einem
    Präfix-Block-Split vollständig in ``train``, sodass Validierung und Test keinen
    einzigen Beharrungsreferenzlauf sahen. Beides verzerrt genau das linke Ende der
    Lernkurve, also die Aussage zum Trainingsdatenbedarf.

    Achtung: diese Änderung verschiebt die Signalarten gegenüber Datensätzen, die mit
    einer früheren Version erzeugt wurden. Der Anregungs-Fingerprint bleibt gleich, weil
    er nur die Konfiguration hasht -- vorhandene Läufe sind daher **neu zu erzeugen**.
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


# ------------------------------------------------------------- Serialisierung


def load_excitation_config(path: str | Path) -> ExcitationConfig:
    """Lädt eine Anregungs-Config aus YAML.

    Auch wenn die Verweilzeiten üblicherweise über
    :meth:`ExcitationConfig.from_time_constants` abgeleitet werden: der **aufgelöste**
    Stand muss zu jedem Datensatz mitgeschrieben werden. Sonst ist eine Lernkurve später
    nicht reproduzierbar, weil sich die abgeleiteten Werte mit jeder Änderung der
    Prozess-Config mitverschieben.
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
    """Serialisiert die vollständig aufgelöste Anregungs-Config."""
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
