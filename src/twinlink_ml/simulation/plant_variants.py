r"""Plant variants for the transfer study (A1): the grey-box model with a different "plant".

The question behind the variants: how much does a simulator help a data-poor model when the
plant is structurally different from it? The simulator stays the unchanged
:class:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator` (P0); a *plant* is a
:class:`PlantVariantSimulator` that deviates from it in a controlled, separable way.

======  =====================================================================================
Name    Deviation from P0
======  =====================================================================================
P0      none (control, bit-identical to the grey-box model)
P1      parameters only: heat loss UA x1.30, gas-solid transfer x0.75, critical moisture x1.20
P2a     kinetics only: drying curve ``f(eta) = eta ** 2``
P2b     measurement only: sensor lag and dead time, offset between bed and exhaust
P3      P2a + P2b + fouling within a run (slow drift, restarts with every run)
PP      pilot only: a mixture of P1 and P2 with its own values
======  =====================================================================================

Separating P2a from P2b is what lets a later study tell whether a method helps under wrong
*physics* or under wrong *measurement*.

Proxies and conventions (named wherever they occur)
---------------------------------------------------
* The model has no separate heat-transfer coefficient (ideal mixing, exhaust at bed
  temperature). ``max_saturation_degree`` -- the degree to which the exhaust approaches
  saturation -- is the only gas-solid transfer parameter and stands in for "heat transfer" in
  P1 and PP. ``saturation_degree_factor`` is a PROXY for the heat/mass transfer.
* Shifting the critical moisture changes the drying kinetics only. The quality limit of the
  product moisture (``product_moisture_limit_kg_per_kg``) is pinned to its value in the base
  config, so a constraint check against it means the same on every plant.
* The drying-curve shape is an exact option of the kinetics (``MaterialConfig.
  drying_curve_exponent``), not a correction on top of it.
* Drift is **fouling within a run**: heat loss and the temperature-sensor offset grow linearly
  from the start of every run and start again at zero with the next run. Nothing carries over
  between runs.
* Sensor path of P2b, per sample: offset -> first-order lag -> dead time -> noise. The channels
  are ``output_names`` = (energy target, bed temperature, product moisture, d50).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from .backend import Array, ArrayBackend, FloatArray
from .base import StepResult, Trajectory
from .config import ProcessConfig, config_fingerprint
from .greybox import IDX_SOLID_MASS, IDX_WATER_MASS, GreyBoxFluidBedGranulator

_N_CHANNELS: Final = 4
_IDX_TEMPERATURE: Final = 1


@dataclass(frozen=True, slots=True)
class SensorSpec:
    """Measurement path of a plant. The defaults leave the base sensor model unchanged."""

    time_constants_s: tuple[float | None, ...] = (None,) * _N_CHANNELS
    """Lag per output channel; ``None``: ``MeasurementConfig.sensor_time_constant_s``."""
    dead_times_s: tuple[float, ...] = (0.0,) * _N_CHANNELS
    """Dead time per output channel; must be a multiple of the sample time."""
    temperature_offset_k: float = 0.0
    """Constant amount by which the bed-temperature reading is below the bed (exhaust)."""
    temperature_offset_gain: float = 0.0
    """Offset proportional to ``T_bed - reference_temperature_celsius`` (duct cooling)."""
    reference_temperature_celsius: float = 20.0

    def __post_init__(self) -> None:
        if len(self.time_constants_s) != _N_CHANNELS or len(self.dead_times_s) != _N_CHANNELS:
            raise ValueError(f"time_constants_s and dead_times_s need {_N_CHANNELS} entries")
        if any(t is not None and t <= 0.0 for t in self.time_constants_s):
            raise ValueError("time constants must be positive")
        if any(t < 0.0 for t in self.dead_times_s):
            raise ValueError("dead times must not be negative")


@dataclass(frozen=True, slots=True)
class FoulingSpec:
    """Slow drift within a run (build-up of fouling), linear in the time since the run start."""

    heat_loss_growth: float = 0.0
    """Relative growth of the heat loss UA over ``horizon_s`` (0.4: +40 %)."""
    sensor_offset_growth_k: float = 0.0
    """Growth of the temperature-sensor offset in K over ``horizon_s``."""
    horizon_s: float = 28_800.0
    """Time in which the growth above is reached (the 8 h run of ``DataSpec``)."""

    def __post_init__(self) -> None:
        if self.horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")


@dataclass(frozen=True, slots=True)
class PlantVariantSpec:
    """Frozen description of one plant. All factors 1.0 and no options = P0."""

    name: str = "P0"
    heat_loss_factor: float = 1.0
    saturation_degree_factor: float = 1.0
    """PROXY for the gas-solid heat/mass transfer, see the module docstring."""
    critical_moisture_factor: float = 1.0
    drying_curve_exponent: float | None = None
    sensor: SensorSpec | None = None
    fouling: FoulingSpec | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "heat_loss_factor",
            "saturation_degree_factor",
            "critical_moisture_factor",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")

    @property
    def content(self) -> dict[str, Any]:
        """Everything that defines the plant, without the name (for the fingerprint)."""
        payload = dataclasses.asdict(self)
        del payload["name"]
        return payload


_P2B_SENSOR: Final = SensorSpec(
    time_constants_s=(None, 30.0, 60.0, None),
    dead_times_s=(0.0, 50.0, 75.0, 0.0),
    temperature_offset_k=1.0,
    temperature_offset_gain=0.08,
)

PLANT_VARIANTS: Final[dict[str, PlantVariantSpec]] = {
    "P0": PlantVariantSpec(name="P0"),
    "P1": PlantVariantSpec(
        name="P1",
        heat_loss_factor=1.30,
        saturation_degree_factor=0.75,
        critical_moisture_factor=1.20,
    ),
    "P2a": PlantVariantSpec(name="P2a", drying_curve_exponent=2.0),
    "P2b": PlantVariantSpec(name="P2b", sensor=_P2B_SENSOR),
    "P3": PlantVariantSpec(
        name="P3",
        drying_curve_exponent=2.0,
        sensor=_P2B_SENSOR,
        fouling=FoulingSpec(heat_loss_growth=0.40, sensor_offset_growth_k=1.5),
    ),
    "PP": PlantVariantSpec(
        name="PP",
        heat_loss_factor=1.40,
        saturation_degree_factor=0.85,
        critical_moisture_factor=0.80,
        drying_curve_exponent=1.5,
        sensor=SensorSpec(
            time_constants_s=(None, None, 40.0, None), dead_times_s=(0.0, 0.0, 75.0, 0.0)
        ),
    ),
}
"""The plants of study A1. ``PP`` is for the pilot only and is not used in the main run."""


def shifted_config(base: ProcessConfig, spec: PlantVariantSpec) -> ProcessConfig:
    """Process config of the plant: the parameter shifts and the drying-curve exponent applied."""
    material = base.material
    changes: dict[str, Any] = {
        "max_saturation_degree": material.max_saturation_degree * spec.saturation_degree_factor,
        "critical_moisture_kg_per_kg": (
            material.critical_moisture_kg_per_kg * spec.critical_moisture_factor
        ),
    }
    if spec.critical_moisture_factor != 1.0:
        # Kinetics only: the quality limit keeps its meaning (the base config's limit).
        changes["product_moisture_limit_kg_per_kg"] = material.quality_moisture_limit_kg_per_kg
    if spec.drying_curve_exponent is not None:
        changes["drying_curve_exponent"] = spec.drying_curve_exponent
    return dataclasses.replace(
        base,
        geometry=dataclasses.replace(
            base.geometry,
            heat_loss_ua_w_per_k=base.geometry.heat_loss_ua_w_per_k * spec.heat_loss_factor,
        ),
        material=dataclasses.replace(material, **changes),
    )


class PlantVariantSimulator(GreyBoxFluidBedGranulator):
    """The grey-box model as a different plant, see the module docstring.

    :param base_config: the config of the simulator (P0); the variant derives its own
        process config from it with :func:`shifted_config`
    :param spec: the deviation, see :data:`PLANT_VARIANTS`

    With ``PlantVariantSpec()`` (P0) every method is bit-identical to the base class.
    Fouling counts the steps since the start of a run (reset by :meth:`rollout`) and the
    measurements since :meth:`reset_sensor`; both restart with every run.
    """

    def __init__(
        self,
        base_config: ProcessConfig,
        spec: PlantVariantSpec,
        backend: ArrayBackend | None = None,
    ) -> None:
        self._base_config = base_config
        self._spec = spec
        super().__init__(shifted_config(base_config, spec), backend)
        self._step_index = 0
        self._measure_index = 0
        self._delay_buffer: list[FloatArray] = []
        self._dead_steps = self._dead_time_steps()

    # ------------------------------------------------------------------ description

    @property
    def spec(self) -> PlantVariantSpec:
        return self._spec

    @property
    def base_config(self) -> ProcessConfig:
        """The config of the simulator the plant deviates from (before any shift)."""
        return self._base_config

    @property
    def variant_fingerprint(self) -> str:
        """Hash of the shifted process config and everything else that defines the plant.

        Needed because the process fingerprint alone does not separate P0 from P2b/P3's
        measurement path or from the fouling.
        """
        return config_fingerprint(
            {"process": self._config.fingerprint, "plant": self._spec.content}
        )

    def run_metadata(self, *, seed: int) -> dict[str, Any]:
        return {
            **super().run_metadata(seed=seed),
            "plant_variant": self._spec.name,
            "plant_variant_fingerprint": self.variant_fingerprint,
        }

    # --------------------------------------------------------------------- process

    def rollout(
        self,
        actions: FloatArray,
        disturbances: FloatArray,
        *,
        run_id: tuple[str, ...] | None = None,
        seed: int = 0,
        initial_state: Array | None = None,
    ) -> Trajectory:
        self._step_index = 0  # fouling starts at zero with every run
        return super().rollout(
            actions, disturbances, run_id=run_id, seed=seed, initial_state=initial_state
        )

    def step(self, state: Array, action: Array, disturbance: Array) -> StepResult:
        result = super().step(state, action, disturbance)
        self._step_index += 1
        return result

    def _rhs(
        self, state: Array, action: Array, disturbance: Array
    ) -> tuple[Array, dict[str, Array]]:
        derivative, diagnostics = super()._rhs(state, action, disturbance)
        fouling = self._spec.fouling
        if fouling is None or fouling.heat_loss_growth == 0.0:
            return derivative, diagnostics
        # Heat loss UA grows by a factor (1 + g * t / horizon): the extra loss enters the
        # temperature balance exactly like the base loss term, with the heat capacity of the
        # base class (so that this equals a config with the larger UA at that time).
        backend = self._backend
        cfg = self._config
        growth = fouling.heat_loss_growth * self._elapsed_s() / fouling.horizon_s
        solid_mass = backend.maximum(state[..., IDX_SOLID_MASS], cfg.numerics.state_floor)
        water_mass = backend.maximum(state[..., IDX_WATER_MASS], 0.0)
        heat_capacity = (
            solid_mass * cfg.material.cp_solid_j_per_kg_k
            + water_mass * cfg.material.cp_water_liquid_j_per_kg_k
        )
        extra_loss = (
            growth * cfg.geometry.heat_loss_ua_w_per_k * (state[..., 3] - disturbance[..., 0])
        )
        correction = backend.stack(
            [
                backend.zeros_like(extra_loss),
                backend.zeros_like(extra_loss),
                backend.zeros_like(extra_loss),
                -extra_loss / heat_capacity,
            ],
            axis=-1,
        )
        return derivative + correction, diagnostics

    def _elapsed_s(self) -> float:
        return self._step_index * self.sample_time_s

    # ----------------------------------------------------------------- measurement

    def reset_sensor(self) -> None:
        super().reset_sensor()
        self._measure_index = 0
        self._delay_buffer = []

    def measure(self, outputs: Array, rng: np.random.Generator) -> Array:
        """Sensor path of the plant; the base path when the plant has no sensor deviation."""
        fouling = self._spec.fouling
        drifts = fouling is not None and fouling.sensor_offset_growth_k != 0.0
        if (self._spec.sensor is None and not drifts) or not self._config.measurement.enabled:
            return super().measure(outputs, rng)
        return self._structured_measure(outputs, rng)

    def _dead_time_steps(self) -> tuple[int, ...]:
        sensor = self._spec.sensor
        if sensor is None:
            return (0,) * _N_CHANNELS
        steps = []
        for dead_time in sensor.dead_times_s:
            n = round(dead_time / self.sample_time_s)
            if abs(n * self.sample_time_s - dead_time) > 1e-9:
                raise ValueError(
                    f"dead time {dead_time} s is not a multiple of the sample time "
                    f"{self.sample_time_s} s"
                )
            steps.append(n)
        return tuple(steps)

    def _structured_measure(self, outputs: Array, rng: np.random.Generator) -> Array:
        config = self._config.measurement
        sensor = self._spec.sensor if self._spec.sensor is not None else SensorSpec()
        fouling = self._spec.fouling
        values = np.array(self._backend.to_numpy(outputs), dtype=np.float64)

        # 1. offset between bed and exhaust (+ its drift by fouling) on the temperature reading
        temperature = values[..., _IDX_TEMPERATURE]
        offset = sensor.temperature_offset_k + sensor.temperature_offset_gain * (
            temperature - sensor.reference_temperature_celsius
        )
        if fouling is not None:
            offset = offset + fouling.sensor_offset_growth_k * (
                self._measure_index * self.sample_time_s / fouling.horizon_s
            )
        values[..., _IDX_TEMPERATURE] = temperature - offset

        # 2. first-order lag, per channel (same arithmetic as the base class when unchanged)
        tau = np.array(
            [config.sensor_time_constant_s if t is None else t for t in sensor.time_constants_s],
            dtype=np.float64,
        )
        alpha = np.exp(-self.sample_time_s / np.maximum(tau, 1e-12))
        if self._filter_state is None:
            self._filter_state = values.copy()
        else:
            self._filter_state = alpha * self._filter_state + (1.0 - alpha) * values
        filtered = np.asarray(self._filter_state, dtype=np.float64).copy()

        # 3. dead time per channel; before the first delayed sample the first value is held
        self._delay_buffer.append(filtered)
        keep = max(self._dead_steps) + 1
        del self._delay_buffer[:-keep]
        delayed = np.empty_like(filtered)
        for channel, steps in enumerate(self._dead_steps):
            index = max(len(self._delay_buffer) - 1 - steps, 0)
            delayed[..., channel] = self._delay_buffer[index][..., channel]
        self._measure_index += 1

        # 4. noise, same draws and order as the base class
        noisy = delayed
        noisy += rng.normal(0.0, config.relative_noise_std * np.abs(noisy))
        noisy[..., _IDX_TEMPERATURE] += rng.normal(
            0.0, config.absolute_noise_std_temperature_k, size=noisy[..., _IDX_TEMPERATURE].shape
        )
        return self._backend.asarray(noisy)


def make_plant(
    name: str, base_config: ProcessConfig, backend: ArrayBackend | None = None
) -> PlantVariantSimulator:
    """The plant ``name`` of :data:`PLANT_VARIANTS` on top of ``base_config``."""
    try:
        spec = PLANT_VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown plant variant {name!r}; known: {sorted(PLANT_VARIANTS)}"
        ) from None
    return PlantVariantSimulator(base_config, spec, backend)


__all__ = [
    "PLANT_VARIANTS",
    "FoulingSpec",
    "PlantVariantSimulator",
    "PlantVariantSpec",
    "SensorSpec",
    "make_plant",
    "shifted_config",
]
