r"""Reduziertes Grey-Box-Modell der Wirbelschichtsprühgranulation.

Modellumfang
------------
Vier Zustände, ideal durchmischtes Bett (CSTR-Annahme), Gasphase quasistationär:

.. math::
    \mathbf{x} = \bigl[m_s,\; m_w,\; N,\; T_{bed}\bigr]^{\mathsf T}

============ ================================================ ===========
Zustand      Bedeutung                                        Einheit
============ ================================================ ===========
:math:`m_s`  Feststoffmasse im Bett (trocken)                 kg
:math:`m_w`  Wassermasse im Bett                              kg
:math:`N`    Partikelanzahl                                   -
:math:`T_bed` Betttemperatur                                  °C
============ ================================================ ===========

Warum Anzahl und Masse statt :math:`d_{50}` als Zustand
--------------------------------------------------------
:math:`d_{50}` folgt algebraisch aus den beiden Momenten:

.. math:: d_{50} = \left(\frac{6\,m_s}{\pi\,\rho_p\,N}\right)^{1/3}

Damit sind Massen- und Zahlenbilanz **linear** und im Integrator exakt erhalten -- die
Unit-Tests auf Bilanzschluss werden trivial und scharf. Formuliert man stattdessen eine
Wachstumsgleichung für :math:`d_{50}` direkt, ist die Massenerhaltung nur noch
approximativ und Bilanzfehler wandern unbemerkt in die Trainingsdaten.

Das Layering-Wachstum aus ``02_prozess_setup.md``,
:math:`G = 2\dot V_{fest}/A_{Partikel}`, ist in dieser Formulierung nicht postuliert,
sondern eine Folgerung: mit :math:`A = 6 m_s/(\rho_p d)` ergibt sich für den reinen
Layering-Fall

.. math:: \frac{\mathrm d d_{50}}{\mathrm d t}
    = \frac{\dot m_{spray}\, w_{solid}\, d_{50}}{3\, m_s}
    = \frac{2\,\dot V_{fest}}{A}

Der Test ``test_layering_growth_matches_analytic_rate`` prüft genau diese Identität.

Gasphase quasistationär
-----------------------
Die Verweilzeit des Gases im Bett liegt bei :math:`H/u \approx 0.3\,\mathrm{m} /
1.6\,\mathrm{m/s} \approx 0.2\,\mathrm s`, also drei Größenordnungen unter der
thermischen Zeitkonstante und weit unter jeder sinnvollen Abtastzeit. Die Abluftfeuchte
wird deshalb algebraisch bestimmt statt als weiterer Zustand geführt. Das entfernt die
steifste Mode aus dem System und erlaubt explizites RK4 mit vertretbarer Schrittweite.

Was das Modell bewusst **nicht** enthält
----------------------------------------
* Partikelgrößen*verteilung* -- nur zwei Momente. Die Projektzielgenauigkeit von 10 %
  für die PSD ist mit diesem Modell **nicht** adressierbar; das bleibt Dyssol
  (TUHH-A-AP 2) bzw. den Laborversuchen (AP 2) vorbehalten.
* Agglomeration, Abrieb, Bruch. Der Nukleationsterm ist ein Platzhalter für
  Überspray/Feingut.
* Elutriation. Bei den Platzhalterwerten liegt :math:`u_{gas}` über der
  Sinkgeschwindigkeit der Keime -- real würden diese in den Zyklon ausgetragen und
  rückgeführt. Der Austrag ist nicht modelliert, ``u_t`` wird aber als Diagnose
  mitgeführt.

  Das betrifft nicht nur den (gegen den Keimdurchmesser trivial immer verletzten) Fall
  oben, sondern auch den Bezug auf den tatsächlichen Bett-d50: unter dem
  Standard-Anregungsdesign (V0, GAB-Isotherme) verletzen 26.6 % der Läufe und 6.0 % der
  Abtastpunkte :math:`u_{gas} > u_t(d_{50})` -- 59.4 % dieser Verletzungen im ersten
  Laufviertel, Median des ersten Verletzungszeitpunkts 25 s (``docs/04_stand_ap1.md``,
  Befund 46; ``scripts/probe_f1_reference_diameter.py``,
  ``scripts/probe_sweep_contamination.py``). ``u_mf``/``u_terminal`` werden in
  :meth:`GreyBoxFluidBedGranulator._rhs` **nach** der Zustandsableitung berechnet
  (reine Diagnose, siehe dort) -- oberhalb ``u_terminal`` wird also für keinen der beiden
  Bezugsdurchmesser eine andere Physik aktiviert, das Modell rechnet unverändert weiter.

  ``numerics.enforce_gas_velocity_projection`` (Default AUS) verhindert das Anregen von
  ``u_gas`` oberhalb dieses Fensters, indem :meth:`project_action` die Aktion VOR dem
  Schritt auf ``[u_mf, u_terminal]`` clippt -- das ist eine Anregungs-Projektion, kein
  Austragsterm: Elutriation bleibt weiterhin nicht modelliert, die Kommandierung von
  Zuständen ausserhalb des validierten Fensters wird nur unterbunden. Siehe
  ``scripts/probe_gas_velocity_projection.py`` für die gemessene Wirkung.
* Tropfengrößenverteilung und damit jede echte Wirkung von ``p_nozzle`` auf den Prozess
  (siehe Hinweis in ``energy.py``).
* Räumliche Gradienten (Sprühzone vs. Trocknungszone).

Diese Liste ist kein Mangel, sondern der Zweck: für den Architekturvergleich in AP 1
zählt eine plausible **Dynamikstruktur**, nicht absolute Genauigkeit.
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
    """Lesbare Sicht auf den Zustandsvektor. Nur für Diagnose und Tests.

    Die Integration läuft auf einem zusammenhängenden Array der Form
    ``(batch, 4)`` -- eine Dataclass pro Zeitschritt und Batch-Element wäre um
    Größenordnungen langsamer.
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
    """Physikalisch motivierte Datenquelle bis Dyssol verfügbar ist (~M21).

    :param config: vollständige Prozesskonfiguration
    :param backend: Array-Backend; wird per Dependency Injection übergeben, damit
        derselbe Physikcode für NumPy-Datengenerierung und differenzierbare
        Torch-Rollouts dient. Ohne Angabe wird ``config.backend`` verwendet.
    """

    def __init__(self, config: ProcessConfig, backend: ArrayBackend | None = None) -> None:
        config.warn_if_placeholder()
        # Einmal pro Konstruktion, nicht pro step()/rollout()-Aufruf: rastert das
        # Sicherheitsraster (200x100 Punkte) genau einmal, siehe
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

    # ------------------------------------------------------------------- Zustand

    def initial_state(self, n_batch: int) -> Array:
        """Anfangszustand im nominellen Betriebspunkt.

        Setzt außerdem das Sensormodell zurück -- ein Rollout beginnt immer mit einem
        eingeschwungenen Messpfad, sonst erzeugt die Sensordynamik einen Einschwingvorgang,
        den das Sequenzmodell als scheinbare Prozessdynamik lernt.
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
        r"""Zustandsvektor aus physikalisch interpretierbaren Größen.

        Umkehrung von :math:`d_{50} = (6 m_s / (\pi \rho_p N))^{1/3}`:

        .. math:: N = \frac{6\, m_s}{\pi\, \rho_p\, d_{50}^3}

        Damit kann das Anregungsdesign Anfangsbedingungen in physikalischen Größen
        streuen, ohne die Zustandskodierung zu kennen. Die Kenntnis darüber, dass
        Partikelanzahl und nicht Durchmesser der Zustand ist, bleibt in dieser Klasse.

        Setzt außerdem das Sensormodell zurück -- wie :meth:`initial_state`.
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

    # ------------------------------------------------------------------- Schritt

    def step(self, state: Array, action: Array, disturbance: Array) -> StepResult:
        """Ein Abtastschritt mit Zero-Order-Hold auf Stell- und Störgrößen."""
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
        r"""Sensormodell: Verzögerungsglied erster Ordnung plus Rauschen.

        .. math:: y_f[k] = a\, y_f[k-1] + (1-a)\, y[k], \qquad
                  a = \exp\!\left(-\frac{\Delta t}{\tau_{sens}}\right)

        Hier -- und **nur** hier -- gehört das Verzögerungsglied erster Ordnung hin, das
        ``02_prozess_setup.md`` als Modell der Bettträgheit nennt. Die Trägheit des Betts
        selbst folgt bereits aus den Bilanzen (siehe :meth:`time_constants`); ein
        zusätzlicher Filter darauf würde die Zeitkonstante doppelt zählen.

        Warum das Sensormodell nicht optional sein sollte: auf rauschfreien Daten ist die
        Ein-Schritt-Vorhersage nahezu trivial (Persistenz ist fast exakt), und der
        Architekturvergleich degeneriert. Das Rauschniveau ist damit ein
        **Versuchsparameter** der Lernkurven, kein Detail.
        """
        if not self._config.measurement.enabled:
            return outputs
        return self._filtered_noisy(outputs, rng)

    def reset_sensor(self) -> None:
        """Filterzustand des Sensormodells verwerfen, siehe Basisklasse."""
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

    # --------------------------------------------------------------- Rechte Seite

    def evaluate_derivative(
        self, state: Array, action: Array, disturbance: Array
    ) -> tuple[Array, dict[str, Array]]:
        """Öffentliche Fassung von :meth:`_rhs`: Zeitableitungen an einem gegebenen Zustand.

        Für Diagnosen, die auf die rechte Seite der Bilanzen zugreifen müssen, ohne zu
        integrieren -- namentlich den exakten Beharrungszustands-Löser in
        :mod:`twinlink_ml.simulation.steady_state`, der Nullstellen von ``derivative``
        sucht. Dyssol kann diesen Zugriff nicht liefern; die Methode ist deshalb bewusst
        keine ABC-Vorgabe (siehe ``03_code_konventionen.md``, Abschnitt zu
        ``steady_state.py``).
        """
        return self._rhs(state, action, disturbance)

    def _rhs(
        self, state: Array, action: Array, disturbance: Array
    ) -> tuple[Array, dict[str, Array]]:
        r"""Zeitableitungen und Diagnosegrößen.

        Bilanzen
        --------
        Feststoff
            .. math:: \dot m_s = \dot m_{spray} w_{solid} + \dot m_{seed} - \dot m_{prod}
        Wasser
            .. math:: \dot m_w = \dot m_{spray}(1 - w_{solid})
                      - \dot m_{evap} - \dot m_{prod} X_p
        Partikelanzahl
            .. math:: \dot N = \frac{b_{nuc}\dot m_{spray} w_{solid} + \dot m_{seed}}
                      {\rho_p \pi d_{nuc}^3/6}
                      - \frac{\dot m_{prod}}{\rho_p \pi d_{50}^3/6}
        Enthalpie
            .. math:: \bigl(m_s c_{p,s} + m_w c_{p,w}\bigr)\dot T_{bed}
                      = \dot m_L (c_{p,L} + Y_{in} c_{p,D})(T_{in} - T_{bed})
                      + \dot m_{spray} c_{p,susp}(T_{susp} - T_{bed})
                      - \dot m_{evap}\,\Delta h_v(T_{bed})
                      - UA\,(T_{bed} - T_{amb})

        Alle abfließenden Ströme verlassen das Bett auf :math:`T_{bed}` und tragen
        deshalb nicht zur Temperaturänderung bei -- exakt für den ideal durchmischten
        Behälter, und der Grund, warum die Bilanz direkt in :math:`T_{bed}` geschrieben
        werden darf.
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
        # Guenstig genug, um pro _rhs-Aufruf frisch zu konstruieren (liest nur Felder aus
        # material, keine Iteration) -- keine Caching-Notwendigkeit, siehe sorption.py.
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
        r"""Produktabzug.

        Kontinuierlich: P-Regler auf die Bettmasse, der die reale Füllstandsregelung
        nachbildet.

        .. math:: \dot m_{prod} = \max\!\left(0,\;
            \dot m_{s,in} + \frac{m_s - m_{s,soll}}{\tau_{m}}\right)

        Der Vorsteueranteil :math:`\dot m_{s,in}` verhindert eine bleibende Regelabweichung.
        :math:`\tau_m` ist eine weitere, langsame Zeitkonstante des Systems und muss gegen
        die realen Austragsintervalle der Anlage kalibriert werden -- bei Anwendungspartner
        wahrscheinlich ein diskreter Austrag, also eher eine getaktete als eine
        kontinuierliche Größe. Das ist eine bewusste Vereinfachung.
        """
        backend = self._backend
        if self._config.operation.mode == "batch":
            return backend.zeros_like(solid_mass)
        setpoint = self._config.geometry.bed_mass_solid_setpoint_kg
        correction = (solid_mass - setpoint) / self._config.operation.mass_control_time_constant_s
        return backend.maximum(solid_inflow + correction, 0.0)

    # ------------------------------------------------------------------ Numerik

    def _integrate(self, state: Array, action: Array, disturbance: Array, dt: float) -> Array:
        """Ein interner Integrationsschritt.

        RK4 ist hier nicht Luxus: die Feuchtemode hat bei den Platzhalterwerten eine
        Zeitkonstante von wenigen Sekunden, also in der Größenordnung der Abtastzeit.
        Explizites Euler verlangt dann eine deutlich kleinere Schrittweite für dieselbe
        Genauigkeit und ist unter dem Strich teurer.

        Die Untergrenze für :meth:`_floor` (zweite Schranke, siehe dort) wird einmal pro
        internem Schritt aus der bereits berechneten rechten Seite am Schrittanfang
        (``k1`` bzw. ``derivative``) gewonnen, nicht neu iteriert -- siehe
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
        r"""Lokales :math:`X_{eq}(\varphi, T)` am Schrittanfang -- Untergrenze für :meth:`_floor`.

        Leichtgewichtige Variante von Option (i) aus der Schritt-0-Analyse: statt die
        volle Störgröße durch ``_floor`` zu reichen (das hätte ``ProcessSimulator``s
        Schnittstelle angefasst), wird die eine bereits in ``diagnostics`` vorhandene Zahl
        (``outlet_humidity_kg_per_kg``) weiterverwendet, zusammen mit der Betttemperatur,
        die ohnehin Teil von ``state`` ist. Kein zusätzlicher Berechnungsaufwand: die
        Picard-Iteration in :func:`~.thermo.evaporation_rate_kg_per_s` hat
        ``outlet_humidity_kg_per_kg`` für genau diesen Zustand bereits konvergiert.

        Näherung, dokumentationswürdig: derselbe Wert wird für alle vier RK4-Stufen
        wiederverwendet, nicht je Stufe neu berechnet -- exakt wären die Werte an
        k2/k3/k4, aber ``_floor`` ist Sicherheitsnetz, kein Modellbestandteil (siehe
        dort), und die Zwischenzustände liegen bei normaler Schrittweite nah an k1.
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
        r"""Zustandsprojektion auf den physikalisch zulässigen Bereich.

        Zwei Schranken:

        1. Massen und Partikelanzahl positiv. Ein explizites Verfahren kann bei großen
           Schritten negative Werte erzeugen, und die Modellgleichungen (dritte Wurzel,
           Division) sind dann undefiniert. Dass diese Schranke im Normalbetrieb nicht
           greift, prüft ``test_state_floor_is_inactive_at_nominal_operation``.
        2. :math:`m_w \ge X_{eq}(\varphi, T)\, m_s`, also Partikelfeuchte über der
           lokalen Gleichgewichtsfeuchte -- siehe :meth:`_local_equilibrium_moisture`.

        Die zweite Schranke ist ein **numerischer Sicherheitsnetz-Fall, kein
        Modellbestandteil**, und der Grund dafür ist dokumentationswürdig: die
        Verdampfungsrate verschwindet bei :math:`X_p = X_{eq}` mit einem Knick. Bei
        heißer Zuluft und minimaler Sprührate fällt die Zeitkonstante der Feuchtemode auf
        wenige Sekunden -- in die Größenordnung der internen Schrittweite. RK4 schießt
        dann über den Knick hinaus, und weil die Rate darunter Null bleibt, kehrt der
        Zustand nicht zurück. Der Constraint wäre also durch den Integrator verletzt,
        nicht durch die Physik.

        Wer die Projektion weglässt, bekommt in genau diesem Betriebspunkt
        Trainingsdaten mit negativer Übersättigung -- und ein Sequenzmodell, das lernt,
        die Gleichgewichtsfeuchte zu unterschreiten. Ein monotones Netz oder eine
        Output-Projektion im ML-Teil kann das nicht heilen, weil die Referenzdaten selbst
        falsch sind. Diagnostisch erkennbar ist der Bereich am ``saturation_degree`` = 0.

        Bei einer zustandsabhängigen Isotherme ist die einzige über den gesamten
        erreichbaren Zustandsraum sichere *konstante* Untergrenze nahe null (eine
        monoton in :math:`\varphi` steigende Isotherme geht bei :math:`\varphi\to 0`
        selbst gegen null, und trockene/heiße Betriebspunkte erreichen sehr kleine
        :math:`\varphi_{Abluft}`, siehe ``docs/04_stand_ap1.md`` Befund 34) --
        ``equilibrium_moisture_local`` ist deshalb kein konstanter Wert mehr, sondern wird
        pro Integrationsschritt aus dem tatsächlichen Zustand gewonnen
        (:meth:`_local_equilibrium_moisture`), und bleibt dadurch auch in dieser Ecke
        wirksam.
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
        """Box-Constraints der Stellgrößen. Für RL die harte Sicherheitsgrenze."""
        backend = self._backend
        bounds = self._config.actions.bounds
        return backend.stack(
            [
                backend.clip(action[..., i], bound.minimum, bound.maximum)
                for i, bound in enumerate(bounds)
            ],
            axis=-1,
        )

    # ------------------------------------------------------ Fluidisierungs-Projektion

    @property
    def projection_active(self) -> bool:
        """Siehe ``ProcessSimulator.projection_active``. Liest den Config-Schalter."""
        return self._config.numerics.enforce_gas_velocity_projection

    def project_action(self, state: Array, action: Array, disturbance: Array) -> Array:
        r"""Projiziert ``gas_velocity_m_per_s`` auf ``[u_mf(d50), u_terminal(d50)]``.

        Behebt die in Befund 46 (``docs/04_stand_ap1.md``, ``scripts/
        probe_gas_velocity_projection.py``) beschriebene physikalische Unvollständigkeit:
        oberhalb ``u_terminal`` fehlt ein Austragsterm, das Modell rechnet unverändert
        weiter, obwohl die reale Anlage Feststoff in den Zyklon austragen würde. Betroffen
        sind 26.6 % der Läufe / 6.0 % der Abtastpunkte des Standard-Anregungsdesigns.

        Kein Sicherheitsabstand (Faktor 1.0) und kein anderer Kanal -- insbesondere bleibt
        ``inlet_temperature_celsius`` unbeschränkt, weil dessen Betriebspunktvariation
        Gegenstand von Befund 49 ist und nicht mitgekappt werden darf.

        Die Margen werden aus ``state``/``disturbance`` bestimmt, wie sie hier ankommen --
        das ist Vertragssache des Aufrufers (``ProcessSimulator.rollout``): ``state`` muss
        der wahre, unverrauschte Zustand VOR dem Integrationsschritt sein, nie eine Messung.
        Dieselbe stoffspezifische Rechnung wie in :meth:`_rhs` (dort erst NACH der
        Ableitung, rein diagnostisch, siehe Klassendocstring), hier aber VOR dem Schritt
        aufgerufen -- deshalb eine eigene, kleine Methode statt Wiederverwendung der
        _rhs-internen Zwischengrößen (die dort für andere Zwecke weiterverwendet werden und
        eine Extraktion mit Regressionsrisiko für den bestehenden, getesteten Pfad wäre).
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
        """u_mf, u_terminal am gegebenen Zustand -- fuer :meth:`project_action`.

        Rechnung identisch zu den gleichnamigen Diagnosegrößen in :meth:`_rhs`, hier aber
        direkt aus rohem ``state``/``disturbance`` aufrufbar (in :meth:`_rhs` bereits
        zerlegte lokale Größen wie ``diameter``/``bed_temperature`` sind dort nicht
        isoliert wiederverwendbar, ohne den bestehenden Pfad anzufassen).
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

    # ---------------------------------------------------------------- Diagnostik

    def nominal_action(self, n_batch: int = 1) -> FloatArray:
        """Nominelle Stellgrößen als ``(batch, n_action)``."""
        return np.tile(
            np.array([b.nominal for b in self._config.actions.bounds], dtype=np.float64),
            (n_batch, 1),
        )

    def nominal_disturbance(self, n_batch: int = 1) -> FloatArray:
        """Nominelle Störgrößen als ``(batch, n_disturbance)``."""
        return np.tile(
            np.array([r.nominal for r in self._config.disturbances.ranges], dtype=np.float64),
            (n_batch, 1),
        )

    def time_constants(self, state: Array, action: Array, disturbance: Array) -> dict[str, float]:
        r"""Zeitkonstanten aus den Eigenwerten der linearisierten Dynamik.

        .. math:: \tau_i = -\frac{1}{\mathrm{Re}(\lambda_i)},
                  \qquad \lambda_i \in \mathrm{spec}\!\left(
                  \frac{\partial \mathbf f}{\partial \mathbf x}\right)

        Damit liefert das Grey-Box-Modell einen begründeten Platzhalter für die in
        ``02_prozess_setup.md`` noch offene "dominante Zeitkonstante" -- und später ein
        quantitatives Kriterium für die Kalibrierung gegen Dyssol: stimmen die
        Zeitkonstanten, stimmt die für Sequenzmodelle relevante Dynamik.

        Die Jacobi-Matrix wird per zentraler Differenz gebildet (relative Schrittweite
        :math:`10^{-6}`). Mit dem Torch-Backend ließe sie sich exakt per Autograd
        bestimmen; für eine Diagnosegröße ist der Unterschied irrelevant.

        :returns: Zeitkonstanten in s, sortiert von schnell nach langsam, plus
            ``stiffness_ratio`` als Verhältnis langsamste/schnellste Mode.
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
