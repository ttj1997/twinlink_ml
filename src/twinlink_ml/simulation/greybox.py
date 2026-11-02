r"""Reduced grey-box model of fluidized-bed spray granulation.

Model scope
-----------
Four states, ideally mixed bed (CSTR assumption), gas phase quasi-stationary:

.. math::
    \mathbf{x} = \bigl[m_s,\; m_w,\; N,\; T_{bed}\bigr]^{\mathsf T}

============ ================================================ ===========
State        Meaning                                          Unit
============ ================================================ ===========
:math:`m_s`  Solid mass in the bed (dry)                      kg
:math:`m_w`  Water mass in the bed                            kg
:math:`N`    Particle count                                   -
:math:`T_bed` Bed temperature                                 °C
============ ================================================ ===========

Why count and mass instead of :math:`d_{50}` as state
------------------------------------------------------
:math:`d_{50}` follows algebraically from the two moments:

.. math:: d_{50} = \left(\frac{6\,m_s}{\pi\,\rho_p\,N}\right)^{1/3}

Mass and number balance are thus **linear** and exactly conserved in the integrator -- the
unit tests on balance closure become trivial and sharp. If one instead formulates a
growth equation for :math:`d_{50}` directly, mass conservation is only
approximate and balance errors migrate unnoticed into the training data.

The layering growth from ``02_prozess_setup.md``,
:math:`G = 2\dot V_{fest}/A_{Partikel}`, is not postulated in this formulation
but a consequence: with :math:`A = 6 m_s/(\rho_p d)` the pure
layering case gives

.. math:: \frac{\mathrm d d_{50}}{\mathrm d t}
    = \frac{\dot m_{spray}\, w_{solid}\, d_{50}}{3\, m_s}
    = \frac{2\,\dot V_{fest}}{A}

The test ``test_layering_growth_matches_analytic_rate`` checks exactly this identity.

Gas phase quasi-stationary
--------------------------
The residence time of the gas in the bed is about :math:`H/u \approx 0.3\,\mathrm{m} /
1.6\,\mathrm{m/s} \approx 0.2\,\mathrm s`, i.e. three orders of magnitude below the
thermal time constant and far below any sensible sample time. The exhaust air humidity
is therefore determined algebraically instead of being carried as a further state. This removes the
stiffest mode from the system and allows explicit RK4 with an acceptable step size.

What the model deliberately does **not** contain
------------------------------------------------
* Particle size *distribution* -- only two moments. The project target accuracy of 10 %
  for the PSD is **not** addressable with this model; that remains reserved for Dyssol
  (TUHH-A-AP 2) or the lab experiments (AP 2).
* Agglomeration, attrition, breakage. The nucleation term is a placeholder for
  overspray/fines.
* Elutriation. With the placeholder values :math:`u_{gas}` lies above the
  terminal velocity of the seeds -- in reality these would be elutriated into the cyclone and
  returned. The elutriation is not modelled, but ``u_t`` is carried along as a
  diagnostic.

  This concerns not only the case above (trivially always violated against the seed diameter)
  but also the reference to the actual bed d50: under the
  default excitation design (V0, GAB isotherm) 26.6 % of the runs and 6.0 % of the
  sample points violate :math:`u_{gas} > u_t(d_{50})` -- 59.4 % of these violations in the first
  quarter of the run, median of the first violation time 25 s (``docs/04_stand_ap1.md``,
  Befund 46; ``scripts/probe_f1_reference_diameter.py``,
  ``scripts/probe_sweep_contamination.py``). ``u_mf``/``u_terminal`` are computed in
  :meth:`GreyBoxFluidBedGranulator._rhs` **after** the state derivative
  (pure diagnostics, see there) -- above ``u_terminal`` no different physics is thus activated
  for either reference diameter; the model keeps computing unchanged.

  ``numerics.enforce_gas_velocity_projection`` (default OFF) prevents exciting
  ``u_gas`` above this window, in that :meth:`project_action` clips the action BEFORE the
  step to ``[u_mf, u_terminal]`` -- that is an excitation projection, not an
  elutriation term: elutriation still remains unmodelled; only commanding
  states outside the validated window is prevented. See
  ``scripts/probe_gas_velocity_projection.py`` for the measured effect.
* Droplet size distribution and thus any real effect of ``p_nozzle`` on the process
  (see note in ``energy.py``).
* Spatial gradients (spray zone vs. drying zone).

This list is not a shortcoming but the purpose: for the architecture comparison in AP 1
what counts is a plausible **dynamic structure**, not absolute accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from .backend import Array, ArrayBackend, FloatArray, get_backend
from .base import ProcessSimulator, StepResult
from .config import ProcessConfig
from .energy import energy_target, power_breakdown_kw
from .fluidization import (
    bed_pressure_drop_pa,
    minimum_fluidization_velocity_m_per_s,
    terminal_velocity_m_per_s,
)
from .sorption import isotherm_from_material, validate_isotherm_against_critical_moisture
from .thermo import (
    absolute_humidity_kg_per_kg,
    evaporation_rate_kg_per_s,
    gas_density_kg_per_m3,
    latent_heat_j_per_kg,
    mixing_temperature_celsius,
    recirculated_inlet_humidity,
    relative_humidity_from_absolute_humidity,
    saturation_humidity_kg_per_kg,
)

IDX_SOLID_MASS: Final = 0
IDX_WATER_MASS: Final = 1
IDX_PARTICLE_COUNT: Final = 2
IDX_BED_TEMPERATURE: Final = 3
N_STATE: Final = 4

_SECONDS_PER_HOUR: Final = 3600.0


@dataclass(frozen=True, slots=True)
class BedState:
    """Readable view of the state vector. Only for diagnostics and tests.

    The integration runs on a contiguous array of shape
    ``(batch, 4)`` -- a dataclass per time step and batch element would be
    orders of magnitude slower.
    """

    solid_mass_kg: Array
    water_mass_kg: Array
    particle_count: Array
    bed_temperature_celsius: Array

    @classmethod
    def from_array(cls, state: Array) -> BedState:
        return cls(
            solid_mass_kg=state[..., IDX_SOLID_MASS],
            water_mass_kg=state[..., IDX_WATER_MASS],
            particle_count=state[..., IDX_PARTICLE_COUNT],
            bed_temperature_celsius=state[..., IDX_BED_TEMPERATURE],
        )


class GreyBoxFluidBedGranulator(ProcessSimulator):
    """Physically motivated data source until Dyssol is available (~M21).

    :param config: complete process configuration
    :param backend: array backend; passed via dependency injection so that
        the same physics code serves NumPy data generation and differentiable
        Torch rollouts. If not given, ``config.backend`` is used.
    """

    def __init__(self, config: ProcessConfig, backend: ArrayBackend | None = None) -> None:
        config.warn_if_placeholder()
        # Once per construction, not per step()/rollout() call: rasterises the
        # safety grid (200x100 points) exactly once, see
        # sorption.validate_isotherm_against_critical_moisture.
        validate_isotherm_against_critical_moisture(config)
        self._config = config
        self._backend = backend if backend is not None else get_backend(config.backend)
        self._filter_state: Array | None = None

    # ------------------------------------------------------------------ Interface

    @property
    def config(self) -> ProcessConfig:
        return self._config

    @property
    def backend(self) -> ArrayBackend:
        return self._backend

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._config.actions.names

    @property
    def disturbance_names(self) -> tuple[str, ...]:
        return self._config.disturbances.names

    @property
    def state_names(self) -> tuple[str, ...]:
        return (
            "solid_mass_kg",
            "water_mass_kg",
            "particle_count",
            "bed_temperature_celsius",
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return (
            "energy_target",
            "bed_temperature_celsius",
            "product_moisture_kg_per_kg",
            "d50_um",
        )

    @property
    def sample_time_s(self) -> float:
        return self._config.numerics.sample_time_s

    # ------------------------------------------------------------------- State

    def initial_state(self, n_batch: int) -> Array:
        """Initial state at the nominal operating point.

        Also resets the sensor model -- a rollout always starts with a
        settled measurement path; otherwise the sensor dynamics produce a transient
        that the sequence model learns as apparent process dynamics.
        """
        material = self._config.material
        geometry = self._config.geometry
        operation = self._config.operation

        solid_mass = geometry.bed_mass_solid_setpoint_kg
        particle_mass = (
            material.particle_density_kg_per_m3 * math.pi * material.initial_diameter_m**3 / 6.0
        )
        state = np.tile(
            np.array(
                [
                    solid_mass,
                    solid_mass * operation.initial_moisture_kg_per_kg,
                    solid_mass / particle_mass,
                    operation.initial_bed_temperature_celsius,
                ],
                dtype=np.float64,
            ),
            (n_batch, 1),
        )
        self._filter_state = None
        return self._backend.asarray(state)

    def state_from_physical(
        self,
        solid_mass_kg: FloatArray,
        moisture_kg_per_kg: FloatArray,
        diameter_m: FloatArray,
        bed_temperature_celsius: FloatArray,
    ) -> Array:
        r"""State vector from physically interpretable quantities.

        Inversion of :math:`d_{50} = (6 m_s / (\pi \rho_p N))^{1/3}`:

        .. math:: N = \frac{6\, m_s}{\pi\, \rho_p\, d_{50}^3}

        The excitation design can thus spread initial conditions in physical quantities
        without knowing the state encoding. The knowledge that
        particle count and not diameter is the state stays in this class.

        Also resets the sensor model -- like :meth:`initial_state`.
        """
        material = self._config.material
        particle_mass = material.particle_density_kg_per_m3 * math.pi * diameter_m**3 / 6.0
        state = np.stack(
            [
                np.asarray(solid_mass_kg, dtype=np.float64),
                np.asarray(solid_mass_kg * moisture_kg_per_kg, dtype=np.float64),
                np.asarray(solid_mass_kg / particle_mass, dtype=np.float64),
                np.asarray(bed_temperature_celsius, dtype=np.float64),
            ],
            axis=-1,
        )
        self._filter_state = None
        return self._backend.asarray(state)

    # ------------------------------------------------------------------- Step

    def step(self, state: Array, action: Array, disturbance: Array) -> StepResult:
        """One sampling step with zero-order hold on actions and disturbances."""
        backend = self._backend
        numerics = self._config.numerics
        action = self._clip_action(action) if numerics.enforce_action_bounds else action

        dt = numerics.internal_time_step_s
        current = state
        for _ in range(numerics.substeps):
            current = self._integrate(current, action, disturbance, dt)
        _, diagnostics = self._rhs(current, action, disturbance)

        outputs = backend.stack(
            [
                diagnostics["energy_target"],
                current[..., IDX_BED_TEMPERATURE],
                diagnostics["particle_moisture_kg_per_kg"],
                diagnostics["d50_m"] * 1e6,
            ],
            axis=-1,
        )
        return StepResult(state=current, outputs=outputs, diagnostics=diagnostics)

    def measure(self, outputs: Array, rng: np.random.Generator) -> Array:
        r"""Sensor model: first-order lag element plus noise.

        .. math:: y_f[k] = a\, y_f[k-1] + (1-a)\, y[k], \qquad
                  a = \exp\!\left(-\frac{\Delta t}{\tau_{sens}}\right)

        Here -- and **only** here -- is where the first-order lag element belongs that
        ``02_prozess_setup.md`` names as a model of the bed inertia. The inertia of the bed
        itself already follows from the balances (see :meth:`time_constants`); an
        additional filter on top would count the time constant twice.

        Why the sensor model should not be optional: on noise-free data the
        one-step prediction is almost trivial (persistence is nearly exact), and the
        architecture comparison degenerates. The noise level is thus an
        **experimental parameter** of the learning curves, not a detail.
        """
        if not self._config.measurement.enabled:
            return outputs
        return self._filtered_noisy(outputs, rng)

    def reset_sensor(self) -> None:
        """Discard the filter state of the sensor model, see base class."""
        self._filter_state = None

    def _filtered_noisy(self, outputs: Array, rng: np.random.Generator) -> Array:
        config = self._config.measurement
        backend = self._backend

        values = backend.to_numpy(outputs)
        alpha = math.exp(-self.sample_time_s / max(config.sensor_time_constant_s, 1e-12))
        if self._filter_state is None:
            self._filter_state = values.copy()
        else:
            self._filter_state = alpha * self._filter_state + (1.0 - alpha) * values

        noisy = np.asarray(self._filter_state, dtype=np.float64).copy()
        noisy += rng.normal(0.0, config.relative_noise_std * np.abs(noisy))
        noisy[..., 1] += rng.normal(
            0.0, config.absolute_noise_std_temperature_k, size=noisy[..., 1].shape
        )
        return backend.asarray(noisy)

    # --------------------------------------------------------------- Right-hand side

    def evaluate_derivative(
        self, state: Array, action: Array, disturbance: Array
    ) -> tuple[Array, dict[str, Array]]:
        """Public version of :meth:`_rhs`: time derivatives at a given state.

        For diagnostics that must access the right-hand side of the balances without
        integrating -- namely the exact steady-state solver in
        :mod:`twinlink_ml.simulation.steady_state`, which searches for roots of ``derivative``.
        Dyssol cannot provide this access; the method is therefore deliberately
        not an ABC requirement (see ``03_code_konventionen.md``, section on
        ``steady_state.py``).
        """
        return self._rhs(state, action, disturbance)

    def _rhs(
        self, state: Array, action: Array, disturbance: Array
    ) -> tuple[Array, dict[str, Array]]:
        r"""Time derivatives and diagnostic variables.

        Balances
        --------
        Solid
            .. math:: \dot m_s = \dot m_{spray} w_{solid} + \dot m_{seed} - \dot m_{prod}
        Water
            .. math:: \dot m_w = \dot m_{spray}(1 - w_{solid})
                      - \dot m_{evap} - \dot m_{prod} X_p
        Particle count
            .. math:: \dot N = \frac{b_{nuc}\dot m_{spray} w_{solid} + \dot m_{seed}}
                      {\rho_p \pi d_{nuc}^3/6}
                      - \frac{\dot m_{prod}}{\rho_p \pi d_{50}^3/6}
        Enthalpy
            .. math:: \bigl(m_s c_{p,s} + m_w c_{p,w}\bigr)\dot T_{bed}
                      = \dot m_L (c_{p,L} + Y_{in} c_{p,D})(T_{in} - T_{bed})
                      + \dot m_{spray} c_{p,susp}(T_{susp} - T_{bed})
                      - \dot m_{evap}\,\Delta h_v(T_{bed})
                      - UA\,(T_{bed} - T_{amb})

        All outgoing streams leave the bed at :math:`T_{bed}` and therefore do not
        contribute to the temperature change -- exact for the ideally mixed
        vessel, and the reason why the balance may be written directly in :math:`T_{bed}`.
        """
        backend = self._backend
        cfg = self._config
        material, gas, geometry, operation = cfg.material, cfg.gas, cfg.geometry, cfg.operation

        solid_mass = backend.maximum(state[..., IDX_SOLID_MASS], cfg.numerics.state_floor)
        water_mass = backend.maximum(state[..., IDX_WATER_MASS], 0.0)
        particle_count = backend.maximum(state[..., IDX_PARTICLE_COUNT], cfg.numerics.state_floor)
        bed_temperature = state[..., IDX_BED_TEMPERATURE]

        inlet_temperature = action[..., 0]
        gas_velocity = action[..., 1]
        spray_rate = action[..., 2] / _SECONDS_PER_HOUR
        nozzle_pressure = action[..., 3]

        ambient_temperature = disturbance[..., 0]
        ambient_humidity = disturbance[..., 1]
        solid_fraction = disturbance[..., 2]

        moisture = water_mass / solid_mass
        diameter = (
            6.0 * solid_mass / (math.pi * material.particle_density_kg_per_m3 * particle_count)
        ) ** (1.0 / 3.0)

        recirculation = operation.exhaust_recirculation_fraction
        # Cheap enough to construct afresh per _rhs call (reads only fields from
        # material, no iteration) -- no need for caching, see sorption.py.
        isotherm = isotherm_from_material(material)
        if recirculation is None:
            inlet_absolute_humidity = absolute_humidity_kg_per_kg(
                backend, gas, ambient_temperature, ambient_humidity
            )
            bed_gas_density = gas_density_kg_per_m3(
                backend, gas, bed_temperature, inlet_absolute_humidity
            )
            moist_air_flow = bed_gas_density * gas_velocity * geometry.cross_section_area_m2
            dry_air_flow = moist_air_flow / (1.0 + inlet_absolute_humidity)

            evaporation, outlet_humidity, saturation_degree = evaporation_rate_kg_per_s(
                backend,
                gas,
                material,
                isotherm,
                dry_air_flow,
                bed_temperature,
                inlet_absolute_humidity,
                moisture,
            )
        else:
            # Exhaust recirculation (OperationConfig.exhaust_recirculation_fraction): the inlet
            # air is the mixture of fresh air and bed exhaust, see
            # thermo.recirculated_inlet_humidity; the bed itself sees only Y_in and T_in.
            fresh_air_humidity = absolute_humidity_kg_per_kg(
                backend, gas, ambient_temperature, ambient_humidity
            )
            inlet_absolute_humidity, outlet_humidity, saturation_degree = (
                recirculated_inlet_humidity(
                    backend,
                    gas,
                    material,
                    isotherm,
                    fresh_air_humidity,
                    recirculation,
                    bed_temperature,
                    moisture,
                )
            )
            bed_gas_density = gas_density_kg_per_m3(
                backend, gas, bed_temperature, inlet_absolute_humidity
            )
            moist_air_flow = bed_gas_density * gas_velocity * geometry.cross_section_area_m2
            dry_air_flow = moist_air_flow / (1.0 + inlet_absolute_humidity)
            evaporation = dry_air_flow * (outlet_humidity - inlet_absolute_humidity)
            mixing_temperature = mixing_temperature_celsius(
                gas,
                recirculation,
                ambient_temperature,
                fresh_air_humidity,
                bed_temperature,
                outlet_humidity,
            )

        spray_solid = spray_rate * solid_fraction
        seed_flow = operation.seed_mass_flow_kg_per_h / _SECONDS_PER_HOUR
        solid_inflow = spray_solid + seed_flow
        product_flow = self._product_mass_flow(solid_inflow, solid_mass)

        nucleus_mass = (
            material.particle_density_kg_per_m3 * math.pi * material.nucleus_diameter_m**3 / 6.0
        )
        bed_particle_mass = material.particle_density_kg_per_m3 * math.pi * diameter**3 / 6.0

        d_solid = solid_inflow - product_flow
        d_water = spray_rate * (1.0 - solid_fraction) - evaporation - product_flow * moisture
        d_count = (
            material.nucleation_solid_fraction * spray_solid + seed_flow
        ) / nucleus_mass - product_flow / bed_particle_mass

        heat_capacity = (
            solid_mass * material.cp_solid_j_per_kg_k
            + water_mass * material.cp_water_liquid_j_per_kg_k
        )
        heat_from_air = (
            dry_air_flow
            * (gas.cp_dry_air_j_per_kg_k + inlet_absolute_humidity * gas.cp_vapour_j_per_kg_k)
            * (inlet_temperature - bed_temperature)
        )
        heat_from_suspension = (
            spray_rate
            * material.cp_suspension_j_per_kg_k
            * (operation.suspension_temperature_celsius - bed_temperature)
        )
        heat_of_evaporation = evaporation * latent_heat_j_per_kg(backend, bed_temperature)
        heat_loss = geometry.heat_loss_ua_w_per_k * (bed_temperature - ambient_temperature)
        d_temperature = (
            heat_from_air + heat_from_suspension - heat_of_evaporation - heat_loss
        ) / heat_capacity

        derivative = backend.stack([d_solid, d_water, d_count, d_temperature], axis=-1)

        power = power_breakdown_kw(
            backend,
            gas,
            material,
            cfg.energy,
            geometry.cross_section_area_m2,
            dry_air_flow,
            inlet_absolute_humidity,
            bed_gas_density,
            inlet_temperature,
            gas_velocity,
            spray_rate,
            nozzle_pressure,
            ambient_temperature,
            solid_mass + water_mass,
            heater_intake_temperature_celsius=None if recirculation is None else mixing_temperature,
            recirculation_fraction=recirculation,
        )
        reference_flow = product_flow if cfg.operation.mode == "continuous" else solid_inflow
        target = energy_target(backend, cfg.energy, power.total_kw, reference_flow)

        u_mf = minimum_fluidization_velocity_m_per_s(
            backend, gas, material, diameter, bed_temperature, inlet_absolute_humidity
        )
        u_terminal = terminal_velocity_m_per_s(
            backend, gas, material, diameter, bed_temperature, inlet_absolute_humidity
        )

        diagnostics: dict[str, Array] = {
            "energy_target": target,
            "d50_m": diameter,
            "particle_moisture_kg_per_kg": moisture,
            "evaporation_rate_kg_per_s": evaporation,
            "outlet_humidity_kg_per_kg": outlet_humidity,
            "saturation_humidity_kg_per_kg": saturation_humidity_kg_per_kg(
                backend, gas, bed_temperature
            ),
            "inlet_humidity_kg_per_kg": inlet_absolute_humidity,
            "saturation_degree": saturation_degree,
            "dry_air_mass_flow_kg_per_s": dry_air_flow,
            "product_mass_flow_kg_per_s": product_flow,
            "bed_pressure_drop_pa": bed_pressure_drop_pa(
                solid_mass + water_mass, geometry.cross_section_area_m2
            ),
            "minimum_fluidization_velocity_m_per_s": u_mf,
            "terminal_velocity_m_per_s": u_terminal,
            "fluidization_margin_lower_m_per_s": gas_velocity - u_mf,
            "fluidization_margin_upper_m_per_s": u_terminal - gas_velocity,
            "power_heater_kw": power.heater_kw,
            "power_fan_kw": power.fan_kw,
            "power_pump_kw": power.pump_kw,
            "power_atomiser_kw": power.atomiser_kw,
            "power_total_kw": power.total_kw,
        }
        if recirculation is not None:
            # Added only with recirculation, so the diagnostics of every existing config keep
            # their keys. ``exhaust_*``: the purge leaving the line, (1 - r) of the bed flow.
            assert power.recirculation_fan_kw is not None
            diagnostics.update(
                {
                    "exhaust_recirculation_fraction": backend.zeros_like(bed_temperature)
                    + recirculation,
                    "fresh_air_humidity_kg_per_kg": fresh_air_humidity,
                    "mixing_temperature_celsius": mixing_temperature,
                    "mixing_relative_humidity": relative_humidity_from_absolute_humidity(
                        backend, gas, inlet_absolute_humidity, mixing_temperature
                    ),
                    "exhaust_dry_air_mass_flow_kg_per_s": (1.0 - recirculation) * dry_air_flow,
                    "power_recirculation_fan_kw": power.recirculation_fan_kw,
                }
            )
        return derivative, diagnostics

    def _product_mass_flow(self, solid_inflow: Array, solid_mass: Array) -> Array:
        r"""Product withdrawal.

        Continuous: P controller on the bed mass, which mimics the real level
        control.

        .. math:: \dot m_{prod} = \max\!\left(0,\;
            \dot m_{s,in} + \frac{m_s - m_{s,soll}}{\tau_{m}}\right)

        The feedforward part :math:`\dot m_{s,in}` prevents a steady-state control error.
        :math:`\tau_m` is a further, slow time constant of the system and must be calibrated
        against the real discharge intervals of the plant -- at the application partner
        probably a discrete discharge, so rather a clocked than a
        continuous variable. That is a deliberate simplification.
        """
        backend = self._backend
        if self._config.operation.mode == "batch":
            return backend.zeros_like(solid_mass)
        setpoint = self._config.geometry.bed_mass_solid_setpoint_kg
        correction = (solid_mass - setpoint) / self._config.operation.mass_control_time_constant_s
        return backend.maximum(solid_inflow + correction, 0.0)

    # ------------------------------------------------------------------ Numerics

    def _integrate(self, state: Array, action: Array, disturbance: Array, dt: float) -> Array:
        """One internal integration step.

        RK4 is no luxury here: with the placeholder values the moisture mode has a
        time constant of a few seconds, i.e. of the order of the sample time.
        Explicit Euler then requires a clearly smaller step size for the same
        accuracy and is more expensive on balance.

        The lower bound for :meth:`_floor` (second bound, see there) is obtained once per
        internal step from the already computed right-hand side at the start of the step
        (``k1`` or ``derivative``), not iterated anew -- see
        :meth:`_local_equilibrium_moisture`.
        """
        if self._config.numerics.integrator == "euler":
            derivative, diagnostics = self._rhs(state, action, disturbance)
            equilibrium_moisture_local = self._local_equilibrium_moisture(diagnostics, state)
            return self._floor(state + dt * derivative, equilibrium_moisture_local)

        k1, diagnostics = self._rhs(state, action, disturbance)
        equilibrium_moisture_local = self._local_equilibrium_moisture(diagnostics, state)
        k2, _ = self._rhs(
            self._floor(state + 0.5 * dt * k1, equilibrium_moisture_local), action, disturbance
        )
        k3, _ = self._rhs(
            self._floor(state + 0.5 * dt * k2, equilibrium_moisture_local), action, disturbance
        )
        k4, _ = self._rhs(
            self._floor(state + dt * k3, equilibrium_moisture_local), action, disturbance
        )
        return self._floor(
            state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), equilibrium_moisture_local
        )

    def _local_equilibrium_moisture(self, diagnostics: dict[str, Array], state: Array) -> Array:
        r"""Local :math:`X_{eq}(\varphi, T)` at the step start -- lower bound for :meth:`_floor`.

        Lightweight variant of option (i) from the step-0 analysis: instead of passing the
        full disturbance through ``_floor`` (that would have touched ``ProcessSimulator``'s
        interface), the one number already present in ``diagnostics``
        (``outlet_humidity_kg_per_kg``) is reused, together with the bed temperature,
        which is part of ``state`` anyway. No additional computational cost: the
        Picard iteration in :func:`~.thermo.evaporation_rate_kg_per_s` has already converged
        ``outlet_humidity_kg_per_kg`` for exactly this state.

        Approximation, worth documenting: the same value is reused for all four RK4 stages,
        not recomputed per stage -- exact would be the values at
        k2/k3/k4, but ``_floor`` is a safety net, not part of the model (see
        there), and the intermediate states lie close to k1 at normal step size.
        """
        backend = self._backend
        material = self._config.material
        isotherm = isotherm_from_material(material)
        bed_temperature = state[..., IDX_BED_TEMPERATURE]
        relative_humidity = relative_humidity_from_absolute_humidity(
            backend, self._config.gas, diagnostics["outlet_humidity_kg_per_kg"], bed_temperature
        )
        return isotherm.equilibrium_moisture(backend, relative_humidity, bed_temperature)

    def _floor(self, state: Array, equilibrium_moisture_local: Array) -> Array:
        r"""State projection onto the physically feasible range.

        Two bounds:

        1. Masses and particle count positive. An explicit method can produce negative
           values for large steps, and the model equations (cube root,
           division) are then undefined. That this bound does not take effect in normal
           operation is checked by ``test_state_floor_is_inactive_at_nominal_operation``.
        2. :math:`m_w \ge X_{eq}(\varphi, T)\, m_s`, i.e. particle moisture above the
           local equilibrium moisture -- see :meth:`_local_equilibrium_moisture`.

        The second bound is a **numerical safety-net case, not part of the
        model**, and the reason for it is worth documenting: the
        evaporation rate vanishes at :math:`X_p = X_{eq}` with a kink. With
        hot inlet air and minimal spray rate the time constant of the moisture mode falls to
        a few seconds -- to the order of the internal step size. RK4 then overshoots
        the kink, and because the rate stays zero below it, the
        state does not return. The constraint would thus be violated by the integrator,
        not by the physics.

        Whoever omits the projection gets, at exactly this operating point,
        training data with negative supersaturation -- and a sequence model that learns
        to undershoot the equilibrium moisture. A monotonic network or an
        output projection in the ML part cannot heal that, because the reference data themselves
        are wrong. Diagnostically the region is recognisable by ``saturation_degree`` = 0.

        With a state-dependent isotherm the only *constant* lower bound that is safe over the
        entire reachable state space is close to zero (an isotherm
        monotonically increasing in :math:`\varphi` itself goes to zero for :math:`\varphi\to 0`,
        and dry/hot operating points reach very small
        :math:`\varphi_{Abluft}`, see ``docs/04_stand_ap1.md`` Befund 34) --
        ``equilibrium_moisture_local`` is therefore no longer a constant value but is
        obtained per integration step from the actual state
        (:meth:`_local_equilibrium_moisture`), and thereby stays effective in this corner
        as well.
        """
        backend = self._backend
        floor = self._config.numerics.state_floor
        solid_mass = backend.maximum(state[..., IDX_SOLID_MASS], floor)
        minimum_water = solid_mass * equilibrium_moisture_local
        return backend.stack(
            [
                solid_mass,
                backend.maximum(state[..., IDX_WATER_MASS], minimum_water),
                backend.maximum(state[..., IDX_PARTICLE_COUNT], floor),
                state[..., IDX_BED_TEMPERATURE],
            ],
            axis=-1,
        )

    def _clip_action(self, action: Array) -> Array:
        """Box constraints of the actions. For RL the hard safety limit."""
        backend = self._backend
        bounds = self._config.actions.bounds
        return backend.stack(
            [
                backend.clip(action[..., i], bound.minimum, bound.maximum)
                for i, bound in enumerate(bounds)
            ],
            axis=-1,
        )

    # ------------------------------------------------------ Fluidization projection

    @property
    def projection_active(self) -> bool:
        """See ``ProcessSimulator.projection_active``. Reads the config switch."""
        return self._config.numerics.enforce_gas_velocity_projection

    def project_action(self, state: Array, action: Array, disturbance: Array) -> Array:
        r"""Projects ``gas_velocity_m_per_s`` onto ``[u_mf(d50), u_terminal(d50)]``.

        Fixes the physical incompleteness described in Befund 46 (``docs/04_stand_ap1.md``,
        ``scripts/probe_gas_velocity_projection.py``):
        above ``u_terminal`` an elutriation term is missing; the model keeps computing unchanged
        although the real plant would elutriate solid into the cyclone. Affected
        are 26.6 % of the runs / 6.0 % of the sample points of the default excitation design.

        No safety margin (factor 1.0) and no other channel -- in particular
        ``inlet_temperature_celsius`` stays unrestricted, because its operating point variation
        is the subject of Befund 49 and must not be capped along with it.

        The margins are determined from ``state``/``disturbance`` as they arrive here --
        that is the caller's contractual responsibility (``ProcessSimulator.rollout``): ``state``
        must be the true, noise-free state BEFORE the integration step, never a measurement.
        The same material-specific calculation as in :meth:`_rhs` (there only AFTER the
        derivative, purely diagnostic, see class docstring), but called here BEFORE the step
        -- hence a separate, small method instead of reusing the
        _rhs-internal intermediate variables (which are reused there for other purposes, and
        extracting them would be a regression risk for the existing, tested path).
        """
        if not self.projection_active:
            return action
        backend = self._backend
        u_mf, u_terminal = self._fluidization_margins(state, disturbance)
        gas_velocity_index = self._config.actions.names.index("gas_velocity_m_per_s")
        return backend.stack(
            [
                backend.clip(action[..., i], u_mf, u_terminal)
                if i == gas_velocity_index
                else action[..., i]
                for i in range(action.shape[-1])
            ],
            axis=-1,
        )

    def _fluidization_margins(self, state: Array, disturbance: Array) -> tuple[Array, Array]:
        """u_mf, u_terminal at the given state -- for :meth:`project_action`.

        Calculation identical to the diagnostic variables of the same name in :meth:`_rhs`, but
        callable here directly from raw ``state``/``disturbance`` (local variables already
        decomposed in :meth:`_rhs`, such as ``diameter``/``bed_temperature``, cannot be
        reused there in isolation without touching the existing path).
        """
        backend = self._backend
        cfg = self._config
        material, gas = cfg.material, cfg.gas

        solid_mass = backend.maximum(state[..., IDX_SOLID_MASS], cfg.numerics.state_floor)
        particle_count = backend.maximum(state[..., IDX_PARTICLE_COUNT], cfg.numerics.state_floor)
        diameter = (
            6.0 * solid_mass / (math.pi * material.particle_density_kg_per_m3 * particle_count)
        ) ** (1.0 / 3.0)
        bed_temperature = state[..., IDX_BED_TEMPERATURE]

        ambient_temperature = disturbance[..., 0]
        ambient_humidity = disturbance[..., 1]
        inlet_absolute_humidity = absolute_humidity_kg_per_kg(
            backend, gas, ambient_temperature, ambient_humidity
        )
        recirculation = cfg.operation.exhaust_recirculation_fraction
        if recirculation is not None:
            # Same inlet humidity as in _rhs (mixture of fresh air and exhaust).
            water_mass = backend.maximum(state[..., IDX_WATER_MASS], 0.0)
            inlet_absolute_humidity, _, _ = recirculated_inlet_humidity(
                backend,
                gas,
                material,
                isotherm_from_material(material),
                inlet_absolute_humidity,
                recirculation,
                bed_temperature,
                water_mass / solid_mass,
            )
        u_mf = minimum_fluidization_velocity_m_per_s(
            backend, gas, material, diameter, bed_temperature, inlet_absolute_humidity
        )
        u_terminal = terminal_velocity_m_per_s(
            backend, gas, material, diameter, bed_temperature, inlet_absolute_humidity
        )
        return u_mf, u_terminal

    # ---------------------------------------------------------------- Diagnostics

    def nominal_action(self, n_batch: int = 1) -> FloatArray:
        """Nominal actions as ``(batch, n_action)``."""
        return np.tile(
            np.array([b.nominal for b in self._config.actions.bounds], dtype=np.float64),
            (n_batch, 1),
        )

    def nominal_disturbance(self, n_batch: int = 1) -> FloatArray:
        """Nominal disturbances as ``(batch, n_disturbance)``."""
        return np.tile(
            np.array([r.nominal for r in self._config.disturbances.ranges], dtype=np.float64),
            (n_batch, 1),
        )

    def time_constants(self, state: Array, action: Array, disturbance: Array) -> dict[str, float]:
        r"""Time constants from the eigenvalues of the linearised dynamics.

        .. math:: \tau_i = -\frac{1}{\mathrm{Re}(\lambda_i)},
                  \qquad \lambda_i \in \mathrm{spec}\!\left(
                  \frac{\partial \mathbf f}{\partial \mathbf x}\right)

        The grey-box model thus provides a justified placeholder for the "dominant time
        constant" still open in ``02_prozess_setup.md`` -- and later a
        quantitative criterion for the calibration against Dyssol: if the
        time constants match, the dynamics relevant for sequence models match.

        The Jacobian is formed by central differences (relative step size
        :math:`10^{-6}`). With the Torch backend it could be determined exactly via autograd;
        for a diagnostic variable the difference is irrelevant.

        :returns: time constants in s, sorted from fast to slow, plus
            ``stiffness_ratio`` as the ratio slowest/fastest mode.
        """
        backend = self._backend
        x0 = backend.to_numpy(state)[0]
        action_np = backend.to_numpy(action)[0]
        disturbance_np = backend.to_numpy(disturbance)[0]

        def rhs_at(x: FloatArray) -> FloatArray:
            derivative, _ = self.evaluate_derivative(
                backend.asarray(x[None, :]),
                backend.asarray(action_np[None, :]),
                backend.asarray(disturbance_np[None, :]),
            )
            return np.asarray(backend.to_numpy(derivative)[0], dtype=np.float64)

        jacobian = np.zeros((N_STATE, N_STATE), dtype=np.float64)
        for j in range(N_STATE):
            delta = 1e-6 * max(abs(x0[j]), 1.0)
            plus, minus = x0.copy(), x0.copy()
            plus[j] += delta
            minus[j] -= delta
            jacobian[:, j] = (rhs_at(plus) - rhs_at(minus)) / (2.0 * delta)

        eigenvalues = np.linalg.eigvals(jacobian)
        real_parts = np.real(eigenvalues)
        stable = real_parts[real_parts < -1e-12]
        taus = np.sort(-1.0 / stable) if stable.size else np.array([])
        result = {f"tau_{i + 1}_s": float(t) for i, t in enumerate(taus)}
        result["n_unstable_modes"] = float(np.sum(real_parts >= -1e-12))
        if taus.size >= 2:
            result["stiffness_ratio"] = float(taus[-1] / taus[0])
        return result
