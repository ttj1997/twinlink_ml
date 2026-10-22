"""Disturbance scenarios of the dynamic headroom study, deterministic from the configured seed.

Three families (docs/24 section 3), each type with its own child seed so that adding a type
never changes another:

``observed`` (the reading L0-L2 applies here)
    ``ambient`` -- day profiles around each point of the 5 x 5 bandit weather grid
    (:class:`~twinlink_ml.simulation.excitation.DiurnalDisturbanceDesign`, its defaults; solid
    fraction constant at the nominal); ``throughput`` -- piecewise-constant levels within +-1.5 %;
    ``solid_fraction`` -- slow Ornstein-Uhlenbeck drift in the band 0.30 +- 7.7 % (the relative
    half width of the plant band 0.60-0.70); ``inlet_temperature_slow`` / ``_fast`` -- additive
    OU offset of the realised inlet temperature (0.35 K / 3 h; 1.5 K / 4 min).
``stress`` (upper estimate)
    ``throughput_steps`` (+-10 / +-20 %), ``solid_fraction_steps`` (whole band 0.20-0.40),
    ``inlet_temperature_slow`` / ``_fast`` with doubled standard deviation.
``product_change``
    one changeover after 2 h of a 6 h scenario: moisture target 0.15 -> 0.12 / 0.12 -> 0.15, solid
    fraction 0.30 -> 0.25 / 0.25 -> 0.30.

Outside ``ambient`` the ambient air is constant at one bandit grid point (scenario ``i`` takes the
point ``perm[i % 25]`` of a seeded permutation). Steps happen on the control grid. The throughput
is rate limited with the spray-rate limit of the action box (the excitation design's rule); the
solid fraction steps (a property of the prepared suspension) are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twinlink_ml.simulation import DiurnalDisturbanceDesign, ProcessConfig
from twinlink_ml.simulation.excitation import apply_rate_limit

from .config import HeadroomStudyConfig
from .symbolic import DISTURBANCE_CHANNELS

__all__ = ["FAMILIES", "Scenario", "build_scenarios", "weather_grid"]

FAMILIES: tuple[str, ...] = ("observed", "stress", "product_change")

_TYPE_IDS: dict[str, int] = {
    "observed/ambient": 1,
    "observed/throughput": 2,
    "observed/solid_fraction": 3,
    "observed/inlet_temperature_slow": 4,
    "observed/inlet_temperature_fast": 5,
    "stress/throughput_steps": 11,
    "stress/solid_fraction_steps": 12,
    "stress/inlet_temperature_slow": 13,
    "stress/inlet_temperature_fast": 14,
    "product_change/moisture_target_tighten": 21,
    "product_change/moisture_target_relax": 22,
    "product_change/solid_fraction_down": 23,
    "product_change/solid_fraction_up": 24,
}


@dataclass(frozen=True, eq=False)
class Scenario:
    """One disturbance scenario on the truth sample grid.

    ``disturbances`` has shape ``(5, n_samples)`` in the order of
    :data:`~twinlink_ml.headroom.symbolic.DISTURBANCE_CHANNELS`; sample ``k`` holds over
    ``[k dt, (k+1) dt)``. ``moisture_limit`` is the quality limit per sample (constant except for
    the moisture-target changeovers).
    """

    name: str
    family: str
    kind: str
    index: int
    sample_time_s: float
    disturbances: np.ndarray
    moisture_limit: np.ndarray
    change_time_s: float | None = None

    @property
    def n_samples(self) -> int:
        """Number of truth samples."""
        return int(self.disturbances.shape[1])

    @property
    def duration_s(self) -> float:
        """Length in s."""
        return self.n_samples * self.sample_time_s


def weather_grid(process: ProcessConfig, points_per_axis: int) -> list[tuple[float, float]]:
    """Bandit weather grid: ambient temperature x relative humidity between the range limits."""
    t_range = process.disturbances.ambient_temperature_celsius
    rh_range = process.disturbances.ambient_relative_humidity
    temps = np.linspace(t_range.minimum, t_range.maximum, points_per_axis)
    rhs = np.linspace(rh_range.minimum, rh_range.maximum, points_per_axis)
    return [(float(t), float(r)) for t in temps for r in rhs]


def _ou(rng: np.random.Generator, n: int, dt: float, std: float, tau: float) -> np.ndarray:
    decay = float(np.exp(-dt / tau))
    scale = std * float(np.sqrt(1.0 - decay**2))
    path = np.empty(n)
    path[0] = rng.normal(0.0, std)
    noise = rng.normal(0.0, scale, n)
    for k in range(1, n):
        path[k] = decay * path[k - 1] + noise[k]
    return path


def _steps(
    rng: np.random.Generator,
    n: int,
    samples_per_interval: int,
    hold_s: tuple[float, float],
    interval_s: float,
    draw: object,
) -> np.ndarray:
    """Piecewise-constant levels, switching on the control grid after uniform hold times."""
    values = np.empty(n)
    start = 0
    while start < n:
        hold_intervals = max(1, round(rng.uniform(*hold_s) / interval_s))
        stop = min(n, start + hold_intervals * samples_per_interval)
        values[start:stop] = draw(rng)  # type: ignore[operator]
        start = stop
    return values


class _Builder:
    def __init__(self, study: HeadroomStudyConfig, process: ProcessConfig) -> None:
        self.study = study
        self.process = process
        sc = study.scenarios
        self.dt = sc.sample_time_s
        self.interval_s = study.control.interval_s
        self.spi = round(self.interval_s / self.dt)
        self.grid = weather_grid(process, sc.ambient.grid_points_per_axis)
        self.perm = np.random.default_rng([sc.seed, 0]).permutation(len(self.grid))
        self.spray = study.line.nominal_spray_rate_kg_per_h
        self.solid = study.line.nominal_solid_fraction
        self.limit = study.line.product_moisture_limit_kg_per_kg

    def rng(self, key: str, index: int) -> np.random.Generator:
        return np.random.default_rng([self.study.scenarios.seed, _TYPE_IDS[key], index])

    def constant(self, index: int, n: int) -> np.ndarray:
        t_amb, rh = self.grid[int(self.perm[index % len(self.grid)])]
        base = np.array([t_amb, rh, self.solid, self.spray, 0.0])
        return np.repeat(base[:, None], n, axis=1)

    def scenario(
        self,
        key: str,
        index: int,
        disturbances: np.ndarray,
        limit: np.ndarray | None = None,
        change_time_s: float | None = None,
    ) -> Scenario:
        family, kind = key.split("/")
        n = disturbances.shape[1]
        return Scenario(
            name=f"{family}/{kind}/{index:02d}",
            family=family,
            kind=kind,
            index=index,
            sample_time_s=self.dt,
            disturbances=disturbances,
            moisture_limit=np.full(n, self.limit) if limit is None else limit,
            change_time_s=change_time_s,
        )

    def rate_limited_spray(self, values: np.ndarray) -> np.ndarray:
        rate = self.process.actions.spray_rate_kg_per_h.max_rate_per_s
        return apply_rate_limit(values[None, :], rate, self.dt)[0]


def _ambient(b: _Builder, n: int) -> list[Scenario]:
    amb = b.study.scenarios.ambient
    design = DiurnalDisturbanceDesign(
        diurnal_amplitude_fraction=amb.diurnal_amplitude_fraction,
        drift_amplitude_fraction=amb.drift_amplitude_fraction,
        drift_correlation_time_s=amb.drift_correlation_time_s,
        solid_fraction_process="independent",
        solid_fraction_std_fraction=0.0,
    )
    space = b.process.disturbances
    t_r, rh_r, sf_r = (
        space.ambient_temperature_celsius,
        space.ambient_relative_humidity,
        space.suspension_solid_fraction,
    )
    scenarios = []
    for index, (t_amb, rh) in enumerate(b.grid):
        unit = np.array(
            [
                [
                    (t_amb - t_r.minimum) / (t_r.maximum - t_r.minimum),
                    (rh - rh_r.minimum) / (rh_r.maximum - rh_r.minimum),
                    (b.solid - sf_r.minimum) / (sf_r.maximum - sf_r.minimum),
                ]
            ]
        )
        signal = design.build(space, unit, n, b.dt, b.rng("observed/ambient", index))[0]
        d = b.constant(index, n)
        d[0:3] = signal.T
        scenarios.append(b.scenario("observed/ambient", index, d))
    return scenarios


def _observed(b: _Builder, n: int) -> list[Scenario]:
    obs = b.study.scenarios.observed
    per_type = b.study.scenarios.per_type
    out = _ambient(b, n)
    for i in range(per_type):
        key = "observed/throughput"
        rng = b.rng(key, i)
        amp = obs.throughput_relative_amplitude
        levels = _steps(
            rng,
            n,
            b.spi,
            obs.throughput_hold_time_s,
            b.interval_s,
            lambda r, a=amp: r.uniform(-a, a),
        )
        d = b.constant(i, n)
        d[3] = b.rate_limited_spray(b.spray * (1.0 + levels))
        out.append(b.scenario(key, i, d))
    for i in range(per_type):
        key = "observed/solid_fraction"
        rng = b.rng(key, i)
        half = obs.solid_fraction_relative_half_band * b.solid
        path = _ou(
            rng,
            n,
            b.dt,
            obs.solid_fraction_std_fraction_of_half_band * half,
            obs.solid_fraction_correlation_time_s,
        )
        d = b.constant(i, n)
        d[2] = np.clip(b.solid + path, b.solid - half, b.solid + half)
        out.append(b.scenario(key, i, d))
    for kind, (std, tau) in (
        ("inlet_temperature_slow", obs.inlet_temperature_slow),
        ("inlet_temperature_fast", obs.inlet_temperature_fast),
    ):
        key = f"observed/{kind}"
        for i in range(per_type):
            d = b.constant(i, n)
            d[4] = _ou(b.rng(key, i), n, b.dt, std, tau)
            out.append(b.scenario(key, i, d))
    return out


def _stress(b: _Builder, n: int) -> list[Scenario]:
    st = b.study.scenarios.stress
    obs = b.study.scenarios.observed
    per_type = b.study.scenarios.per_type
    out = []
    levels_set = np.asarray(st.throughput_levels)
    for i in range(per_type):
        key = "stress/throughput_steps"
        rng = b.rng(key, i)
        levels = _steps(
            rng, n, b.spi, st.throughput_hold_time_s, b.interval_s, lambda r: r.choice(levels_set)
        )
        d = b.constant(i, n)
        d[3] = b.rate_limited_spray(b.spray * (1.0 + levels))
        out.append(b.scenario(key, i, d))
    low, high = st.solid_fraction_band
    for i in range(per_type):
        key = "stress/solid_fraction_steps"
        rng = b.rng(key, i)
        d = b.constant(i, n)
        d[2] = _steps(
            rng,
            n,
            b.spi,
            st.solid_fraction_hold_time_s,
            b.interval_s,
            lambda r: r.uniform(low, high),
        )
        out.append(b.scenario(key, i, d))
    for kind, (std, tau) in (
        ("inlet_temperature_slow", obs.inlet_temperature_slow),
        ("inlet_temperature_fast", obs.inlet_temperature_fast),
    ):
        key = f"stress/{kind}"
        for i in range(per_type):
            d = b.constant(i, n)
            d[4] = _ou(b.rng(key, i), n, b.dt, st.inlet_temperature_std_factor * std, tau)
            out.append(b.scenario(key, i, d))
    return out


def _product_change(b: _Builder) -> list[Scenario]:
    pc = b.study.scenarios.product_change
    n = round(pc.duration_s / b.dt)
    change = round(pc.change_time_s / b.dt)
    limit_a, limit_b = b.limit, pc.moisture_limit_b_kg_per_kg
    solid_a, solid_b = b.solid, pc.solid_fraction_b
    out = []
    directions = (
        ("moisture_target_tighten", (limit_a, limit_b), None),
        ("moisture_target_relax", (limit_b, limit_a), None),
        ("solid_fraction_down", None, (solid_a, solid_b)),
        ("solid_fraction_up", None, (solid_b, solid_a)),
    )
    offset = 0
    for kind, limits, solids in directions:
        for i in range(pc.per_direction):
            d = b.constant(offset + i, n)
            limit = np.full(n, b.limit)
            if limits is not None:
                limit[:change], limit[change:] = limits
            if solids is not None:
                d[2, :change], d[2, change:] = solids
            out.append(
                b.scenario(f"product_change/{kind}", i, d, limit, change_time_s=pc.change_time_s)
            )
        offset += pc.per_direction
    return out


def build_scenarios(
    study: HeadroomStudyConfig, process: ProcessConfig, family: str
) -> list[Scenario]:
    """All scenarios of one family, deterministic from ``study.scenarios.seed``."""
    if len(DISTURBANCE_CHANNELS) != 5:  # pragma: no cover - guard for the channel order
        raise RuntimeError("channel order changed")
    builder = _Builder(study, process)
    n = round(study.scenarios.duration_s / study.scenarios.sample_time_s)
    if family == "observed":
        return _observed(builder, n)
    if family == "stress":
        return _stress(builder, n)
    if family == "product_change":
        return _product_change(builder)
    raise ValueError(f"unknown family {family!r}, allowed {FAMILIES}")
