"""Configuration objects of the grey-box process model.

All numerical values of the model live here or in ``conf/process/*.yaml`` -- none in the code.
Reason: replacing the placeholders with real values from TUHH-A / application partner should be
a diff in a YAML file, not an intervention in the physics.

Units are in the field name (convention from ``03_code_konventionen.md``).
Internally the model computes in SI; the conversion of customary engineering units
(kg/h, bar, °C) happens in exactly one place: :meth:`ProcessConfig.to_si`-like
helpers in ``greybox.py``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

T = TypeVar("T")

ConfigStatus = Literal["placeholder", "partially_calibrated", "calibrated"]
OperationMode = Literal["continuous", "batch"]
EnergyTarget = Literal["power_kw", "specific_kwh_per_kg"]
Integrator = Literal["rk4", "euler"]
SorptionIsothermKind = Literal["constant", "gab"]


MAX_EXHAUST_RECIRCULATION_FRACTION = 0.95
"""Upper bound of ``OperationConfig.exhaust_recirculation_fraction``: at ``r -> 1`` the purge
flow vanishes and the inlet humidity diverges (the water leaves only with the purge)."""


class PlaceholderConfigWarning(UserWarning):
    """Raised when loading a config with placeholder values.

    ``02_prozess_setup.md`` records that a file with outdated numbers is worse
    than none, because it silently carries wrong assumptions into the model. This
    warning turns the convention into a guarantee of the code: a run with
    placeholders is identifiable afterwards in the log.
    """


@dataclass(frozen=True, slots=True)
class Provenance:
    """Origin of the numerical values -- goes into every run log."""

    status: ConfigStatus = "placeholder"
    description: str = ""
    open_todos: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """Apparatus geometry. Source: ``02_prozess_setup.md`` -> section Plant geometry."""

    cross_section_area_m2: float = 0.0177
    bed_mass_solid_setpoint_kg: float = 2.0
    heat_loss_ua_w_per_k: float = 2.0

    def __post_init__(self) -> None:
        _require_positive(self, "cross_section_area_m2", "bed_mass_solid_setpoint_kg")
        _require_non_negative(self, "heat_loss_ua_w_per_k")


@dataclass(frozen=True, slots=True)
class GasConfig:
    """Properties of the fluidization air (moist air, ideal)."""

    pressure_pa: float = 101_325.0
    r_specific_dry_air_j_per_kg_k: float = 287.06
    cp_dry_air_j_per_kg_k: float = 1006.0
    cp_vapour_j_per_kg_k: float = 1860.0
    viscosity_ref_pa_s: float = 1.716e-5
    viscosity_ref_temperature_k: float = 273.15
    sutherland_constant_k: float = 110.4

    def __post_init__(self) -> None:
        _require_positive(self, "pressure_pa", "cp_dry_air_j_per_kg_k", "viscosity_ref_pa_s")


@dataclass(frozen=True, slots=True)
class MaterialConfig:
    """Material system. ``02_prozess_setup.md``: material system is still TODO."""

    particle_density_kg_per_m3: float = 1500.0
    sphericity: float = 0.9
    cp_solid_j_per_kg_k: float = 1200.0
    cp_water_liquid_j_per_kg_k: float = 4180.0
    cp_suspension_j_per_kg_k: float = 3500.0
    suspension_density_kg_per_m3: float = 1100.0
    # Drying kinetics: normalised drying curve, linearly falling rate
    critical_moisture_kg_per_kg: float = 0.020
    # Nominal value/anchor of the equilibrium moisture. Dual role, independent of
    # sorption_isotherm_kind: (i) input of the validation against
    # critical_moisture_kg_per_kg below -- a coarse, isotherm-independent
    # plausibility bound, no statement about X_eq(phi,T); (ii) with
    # sorption_isotherm_kind="gab" the anchor value to which the GAB parameters
    # below are calibrated (phi=0.5, T=gab_reference_temperature_celsius --
    # see sorption.GabIsotherm). With sorption_isotherm_kind="constant" it is
    # at the same time the actual, state-independent X_eq value of the simulation.
    equilibrium_moisture_kg_per_kg: float = 0.003
    max_saturation_degree: float = 0.60
    max_driving_humidity_kg_per_kg: float = 0.50
    # Sorption isotherm of the drying curve (see simulation.sorption).
    # "constant" reproduces equilibrium_moisture_kg_per_kg exactly as a
    # state-independent X_eq value -- default, backward compatible: existing
    # configs without these fields load unchanged. "gab" activates the
    # state-dependent GAB isotherm below; all gab_* fields are then
    # mandatory (see __post_init__). GabIsotherm marks itself as "placeholder"
    # independently of provenance.status (see sorption.py).
    sorption_isotherm_kind: SorptionIsothermKind = "constant"
    gab_monolayer_moisture_kg_per_kg: float | None = None
    gab_c_constant_at_reference_temperature: float | None = None
    gab_k_constant_at_reference_temperature: float | None = None
    gab_c_enthalpy_j_per_mol: float | None = None
    gab_k_enthalpy_j_per_mol: float | None = None
    gab_reference_temperature_celsius: float | None = None
    # Population balance (reduced to two moments)
    nucleus_diameter_m: float = 150e-6
    nucleation_solid_fraction: float = 0.008
    initial_diameter_m: float = 400e-6
    # Quality limit of the product moisture (a specification), optional. ``None``: the
    # limit equals ``critical_moisture_kg_per_kg`` as before. The drying kinetics always
    # use ``critical_moisture_kg_per_kg``; only constraints read the quality limit (see
    # ``quality_moisture_limit_kg_per_kg``). Enters the fingerprint ONLY when set
    # (``without_unset_optional_fields``), so every existing config keeps its fingerprint.
    product_moisture_limit_kg_per_kg: float | None = None
    # Shape of the drying curve in the falling-rate period: f(eta) = eta ** q with the
    # normalised moisture eta. ``None`` (default): linear, f(eta) = eta, exactly the behaviour
    # of every existing config. Set: the exponent q > 0 (q = 1.0 is numerically the linear
    # curve but a different config). Read in ``thermo.evaporation_rate_kg_per_s``, so every
    # consumer of the kinetics (rollout, steady-state solver, exhaust reconstruction) sees it.
    # Enters the fingerprint ONLY when set (``without_unset_optional_fields``).
    drying_curve_exponent: float | None = None

    @property
    def quality_moisture_limit_kg_per_kg(self) -> float:
        """Moisture threshold of the quality constraint (not of the drying kinetics)."""
        if self.product_moisture_limit_kg_per_kg is None:
            return self.critical_moisture_kg_per_kg
        return self.product_moisture_limit_kg_per_kg

    def __post_init__(self) -> None:
        _require_positive(
            self,
            "particle_density_kg_per_m3",
            "cp_solid_j_per_kg_k",
            "nucleus_diameter_m",
            "initial_diameter_m",
            "max_driving_humidity_kg_per_kg",
        )
        if not 0.0 < self.max_saturation_degree <= 1.0:
            raise ValueError("max_saturation_degree muss in (0, 1] liegen")
        if self.critical_moisture_kg_per_kg <= self.equilibrium_moisture_kg_per_kg:
            raise ValueError(
                "critical_moisture_kg_per_kg muss über equilibrium_moisture_kg_per_kg liegen"
            )
        if not 0.0 <= self.nucleation_solid_fraction < 1.0:
            raise ValueError("nucleation_solid_fraction muss in [0, 1) liegen")
        if (
            self.product_moisture_limit_kg_per_kg is not None
            and self.product_moisture_limit_kg_per_kg <= self.equilibrium_moisture_kg_per_kg
        ):
            raise ValueError(
                "product_moisture_limit_kg_per_kg muss über equilibrium_moisture_kg_per_kg liegen"
            )
        if self.drying_curve_exponent is not None and self.drying_curve_exponent <= 0.0:
            raise ValueError("drying_curve_exponent must be positive when set")
        self._validate_gab_fields()

    def _validate_gab_fields(self) -> None:
        """With ``sorption_isotherm_kind == "gab"``: all gab_* fields set and plausible.

        Purely local check (no dependency on ``sorption.py`` or other
        ``ProcessConfig`` fields) -- the further check that
        ``critical_moisture_kg_per_kg`` lies above the maximum of the isotherm on the
        safety grid needs action/disturbance bounds from the
        enclosing ``ProcessConfig`` and therefore lives in
        ``sorption.validate_isotherm_against_critical_moisture`` -- see there
        why it does not belong here (dependency direction: ``config.py``
        stays free of imports from ``sorption.py``/``thermo.py``).
        """
        if self.sorption_isotherm_kind != "gab":
            return
        gab_fields = {
            "gab_monolayer_moisture_kg_per_kg": self.gab_monolayer_moisture_kg_per_kg,
            "gab_c_constant_at_reference_temperature": self.gab_c_constant_at_reference_temperature,
            "gab_k_constant_at_reference_temperature": self.gab_k_constant_at_reference_temperature,
            "gab_c_enthalpy_j_per_mol": self.gab_c_enthalpy_j_per_mol,
            "gab_k_enthalpy_j_per_mol": self.gab_k_enthalpy_j_per_mol,
            "gab_reference_temperature_celsius": self.gab_reference_temperature_celsius,
        }
        missing = sorted(name for name, value in gab_fields.items() if value is None)
        if missing:
            raise ValueError(
                f"sorption_isotherm_kind='gab' verlangt alle gab_*-Felder gesetzt, "
                f"fehlend: {missing}"
            )
        monolayer = self.gab_monolayer_moisture_kg_per_kg
        c_ref = self.gab_c_constant_at_reference_temperature
        k_ref = self.gab_k_constant_at_reference_temperature
        assert monolayer is not None and c_ref is not None and k_ref is not None
        if monolayer <= 0.0:
            raise ValueError(f"gab_monolayer_moisture_kg_per_kg muss positiv sein, ist {monolayer}")
        if c_ref <= 0.0:
            raise ValueError(
                f"gab_c_constant_at_reference_temperature muss positiv sein, ist {c_ref}"
            )
        if not 0.0 < k_ref < 1.0:
            raise ValueError(
                "gab_k_constant_at_reference_temperature muss in (0, 1) liegen "
                f"(Polstelle der GAB-Gleichung bei K*phi -> 1), ist {k_ref}"
            )

    @property
    def nucleation_limited_diameter_m(self) -> float:
        r"""Steady-state diameter of the special case without seed feed.

        Holds only for ``seed_mass_flow_kg_per_h == 0``. From the number balance at
        steady state it follows for this case that

        .. math:: d_{50,\mathrm{ss}} = d_{\mathrm{nuc}}\, b_{\mathrm{nuc}}^{-1/3}

        With seed feed the general steady state gives a different value --
        see :mod:`twinlink_ml.simulation.steady_state`. The nucleation fraction is
        thus **not** a free tuning parameter but fixed by the target particle size.
        See :func:`nucleation_fraction_for_target_diameter`.
        """
        if self.nucleation_solid_fraction == 0.0:
            return float("inf")
        return float(self.nucleus_diameter_m * self.nucleation_solid_fraction ** (-1.0 / 3.0))


@dataclass(frozen=True, slots=True)
class EnergyConfig:
    """Units of the energy balance. Determine the target ``dE``."""

    heater_efficiency: float = 0.98
    fan_efficiency: float = 0.55
    distributor_loss_coefficient: float = 350.0
    pump_pressure_rise_pa: float = 3.0e5
    pump_efficiency: float = 0.40
    atomiser_air_kg_per_s_per_bar: float = 1.5e-4
    compressor_efficiency: float = 0.50
    target: EnergyTarget = "power_kw"
    reference_value: float | None = None
    # Pressure-loss coefficient of the exhaust recirculation duct (damper, duct, filter), referred
    # to the superficial velocity of the RECIRCULATED flow over the bed cross-section,
    # ``dp_rec = zeta_rec * rho_g * (r * u)**2 / 2``; the extra fan power is
    # ``r * u * A * dp_rec / fan_efficiency``, i.e. proportional to the recirculated volume flow
    # cubed. ``None`` (default): no extra fan power. Only read when
    # ``OperationConfig.exhaust_recirculation_fraction`` is set. PLACEHOLDER without a source
    # (study docs/22). Enters the fingerprint ONLY when set (``without_unset_optional_fields``).
    recirculation_duct_loss_coefficient: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "heater_efficiency",
            "fan_efficiency",
            "pump_efficiency",
            "compressor_efficiency",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} muss in (0, 1] liegen, ist {value}")
        if (
            self.recirculation_duct_loss_coefficient is not None
            and self.recirculation_duct_loss_coefficient < 0.0
        ):
            raise ValueError("recirculation_duct_loss_coefficient must not be negative when set")


@dataclass(frozen=True, slots=True)
class ExhaustConfig:
    """Downstream exhaust air interface: duct/filter between bed and heat integration system.

    Deliberately a section of its own, not part of :class:`GeometryConfig`: its
    ``heat_loss_ua_w_per_k`` is a term of the bed enthalpy balance (``greybox._rhs``) and
    changes the simulated state trajectory. Fields here act exclusively
    downstream in :func:`twinlink_ml.simulation.exhaust.reconstruct_exhaust_state` -- purely
    algebraic post-processing of an already simulated/predicted state, not
    state itself and no input to ``_rhs``. Keeping both in the same section would
    invite adding them by mistake or wrongly expecting the new parameter in the
    ODE.

    STATUS: placeholder, no source -- duct/filter loss between bed and
    heat exchanger, so far not measured and not requested from the application partner.
    """

    outlet_duct_heat_loss_k: float = 0.0
    """Pauschaler Temperaturabfall zwischen Bett-Austritt und der gemessenen
    Ablufttemperatur am Wärmeverbund, in Kelvin. Vorzeichenkonvention:

    .. math:: T_{outlet} = T_{bed} - \\text{outlet\\_duct\\_heat\\_loss\\_k}

    Ein positiver Wert bedeutet Abkühlung entlang Rohrleitung/Filter; negative Werte sind
    nicht vorgesehen (keine Wärmequelle zwischen Bett und Übertrager) und werden in
    ``__post_init__`` zurückgewiesen. Default 0.0: Identität zu ``bed_temperature_celsius``,
    numerisch keine Änderung gegenüber dem bisherigen (fehlerhaft benannten) Verhalten.

    Bewusst ein pauschaler Abkühlbetrag, kein UA-Wert: anders als
    ``geometry.heat_loss_ua_w_per_k`` hängt dieser Platzhalter nicht von Massenstrom oder
    Umgebungstemperatur ab -- kein Wärmeübertragungsmodell, nur ein fester Versatz. Ein
    UA-basierter Ansatz wäre die spätere Verfeinerung, sobald eine Quelle vorliegt.
    """

    def __post_init__(self) -> None:
        _require_non_negative(self, "outlet_duct_heat_loss_k")


@dataclass(frozen=True, slots=True)
class OperationConfig:
    """Operating mode. ``02_prozess_setup.md``: continuous/batch/semi-batch is TODO."""

    mode: OperationMode = "continuous"
    mass_control_time_constant_s: float = 600.0
    seed_mass_flow_kg_per_h: float = 0.0
    suspension_temperature_celsius: float = 20.0
    initial_moisture_kg_per_kg: float = 0.005
    initial_bed_temperature_celsius: float = 55.0
    # Exhaust recirculation: fraction ``r`` of the dry air through the bed that is taken from the
    # exhaust (bed outlet state, T_bed and Y_out) and mixed with fresh ambient air ahead of the air
    # heater; the heater then brings the mixture to the inlet temperature. ``None`` (default):
    # fresh air only, exactly the behaviour of every existing config (a different config from
    # ``0.0``, which is numerically the same). Allowed range [0, 0.95]. Read in
    # ``greybox._rhs`` (inlet humidity, heater intake temperature) through
    # ``thermo.recirculated_inlet_humidity``. Enters the fingerprint ONLY when set
    # (``without_unset_optional_fields``), so every existing config keeps its fingerprint.
    exhaust_recirculation_fraction: float | None = None

    def __post_init__(self) -> None:
        _require_positive(self, "mass_control_time_constant_s")
        _require_non_negative(self, "seed_mass_flow_kg_per_h", "initial_moisture_kg_per_kg")
        if self.exhaust_recirculation_fraction is not None and not (
            0.0 <= self.exhaust_recirculation_fraction <= MAX_EXHAUST_RECIRCULATION_FRACTION
        ):
            raise ValueError(
                "exhaust_recirculation_fraction must lie in "
                f"[0, {MAX_EXHAUST_RECIRCULATION_FRACTION}] when set, "
                f"is {self.exhaust_recirculation_fraction}"
            )


@dataclass(frozen=True, slots=True)
class ActionBound:
    """Box and rate limit of an action."""

    minimum: float
    maximum: float
    nominal: float
    max_rate_per_s: float

    def __post_init__(self) -> None:
        if not self.minimum <= self.nominal <= self.maximum:
            raise ValueError(
                f"nominal={self.nominal} liegt außerhalb [{self.minimum}, {self.maximum}]"
            )
        if self.max_rate_per_s <= 0.0:
            raise ValueError("max_rate_per_s muss positiv sein")

    @property
    def span(self) -> float:
        return self.maximum - self.minimum


@dataclass(frozen=True, slots=True)
class ActionSpaceConfig:
    """Action space. All values TODO in ``02_prozess_setup.md`` -> placeholders.

    The normalisation to ``[0, 1]`` (see :meth:`normalise`) is the workaround from the
    work plan: the sampling design and the RL action bounds stay unchanged
    when the real ranges are inserted.
    """

    inlet_temperature_celsius: ActionBound = field(
        default_factory=lambda: ActionBound(60.0, 120.0, 90.0, 0.5)
    )
    gas_velocity_m_per_s: ActionBound = field(
        default_factory=lambda: ActionBound(0.8, 2.5, 1.6, 0.05)
    )
    spray_rate_kg_per_h: ActionBound = field(
        default_factory=lambda: ActionBound(0.4, 3.0, 1.5, 0.02)
    )
    nozzle_pressure_bar: ActionBound = field(
        default_factory=lambda: ActionBound(1.0, 4.0, 2.5, 0.05)
    )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    @property
    def bounds(self) -> tuple[ActionBound, ...]:
        return tuple(getattr(self, name) for name in self.names)


@dataclass(frozen=True, slots=True)
class DisturbanceRange:
    """Range of a disturbance for the Latin hypercube sampling (step 2)."""

    minimum: float
    maximum: float
    nominal: float

    def __post_init__(self) -> None:
        if not self.minimum <= self.nominal <= self.maximum:
            raise ValueError("nominal liegt außerhalb des Bereichs")


@dataclass(frozen=True, slots=True)
class DisturbanceSpaceConfig:
    """Exogenous, measurable, non-controllable variables.

    The day/night indicator from ``02_prozess_setup.md`` is deliberately **not** a physical
    input of the model: it acts exclusively through ``T_amb`` and ``phi_amb``.
    As a feature for the sequence model it is derived from the timestamp
    (sin/cos encoding), not supplied by the simulator. Otherwise the model gets an
    input that has no causal effect on the real plant.
    """

    ambient_temperature_celsius: DisturbanceRange = field(
        default_factory=lambda: DisturbanceRange(5.0, 35.0, 20.0)
    )
    ambient_relative_humidity: DisturbanceRange = field(
        default_factory=lambda: DisturbanceRange(0.25, 0.85, 0.50)
    )
    suspension_solid_fraction: DisturbanceRange = field(
        default_factory=lambda: DisturbanceRange(0.20, 0.40, 0.30)
    )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    @property
    def ranges(self) -> tuple[DisturbanceRange, ...]:
        return tuple(getattr(self, name) for name in self.names)


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    """Temporal resolution. ``02_prozess_setup.md``: Δt Dyssol is TODO."""

    sample_time_s: float = 5.0
    substeps: int = 5
    integrator: Integrator = "rk4"
    state_floor: float = 1e-9
    enforce_action_bounds: bool = True
    enforce_gas_velocity_projection: bool = False
    """Projiziert ``gas_velocity_m_per_s`` je Schritt auf ``[u_mf(d50), u_terminal(d50)]``,
    berechnet aus dem wahren Zustand unmittelbar vor dem Integrationsschritt (siehe
    ``ProcessSimulator.rollout``/``GreyBoxFluidBedGranulator.project_action``). Kein
    Sicherheitsabstand (Faktor 1.0), kein anderer Kanal betroffen -- insbesondere bleibt
    ``inlet_temperature_celsius`` unbeschraenkt (Befund 49). Default AUS: Teil von
    ``NumericsConfig``, geht damit automatisch in ``ProcessConfig.fingerprint`` ein, sodass
    Datensaetze mit und ohne Projektion nie denselben Fingerprint tragen (siehe
    ``docs/04_stand_ap1.md`` Befund 46, ``scripts/probe_gas_velocity_projection.py``)."""

    def __post_init__(self) -> None:
        _require_positive(self, "sample_time_s")
        if self.substeps < 1:
            raise ValueError("substeps muss >= 1 sein")

    @property
    def internal_time_step_s(self) -> float:
        return self.sample_time_s / self.substeps


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    """Sensor model: noise and sensor dynamics.

    Why this is not optional: the noise level determines the learning curves more strongly than
    the architecture. On noise-free simulation data every sufficiently large
    model saturates and the comparison becomes meaningless. The values here are placeholders and
    must be calibrated from the application-partner time series or the lab sensors.
    """

    enabled: bool = True
    relative_noise_std: float = 0.005
    absolute_noise_std_temperature_k: float = 0.2
    sensor_time_constant_s: float = 10.0

    def __post_init__(self) -> None:
        _require_non_negative(
            self,
            "relative_noise_std",
            "absolute_noise_std_temperature_k",
            "sensor_time_constant_s",
        )


@dataclass(frozen=True, slots=True)
class ProcessConfig:
    """Complete configuration of a simulation experiment."""

    provenance: Provenance = field(default_factory=Provenance)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    exhaust: ExhaustConfig = field(default_factory=ExhaustConfig)
    operation: OperationConfig = field(default_factory=OperationConfig)
    actions: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    disturbances: DisturbanceSpaceConfig = field(default_factory=DisturbanceSpaceConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    backend: str = "numpy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Stable hash of the resolved config -- belongs in every run log.

        Applies to the APPLIED config (this instance, as it is actually simulated),
        not to the file as loaded: ``sample_time_s`` is typically overridden by
        ``DataSpec.sample_time_s`` (see ``evaluation.experiments.generate_data``,
        ``dataclasses.replace`` on ``numerics`` before the simulator is built), before the
        fingerprint is formed. A comparison against the fingerprint of the unchanged file
        loaded via ``load_process_config`` (whose ``sample_time_s`` is the YAML default)
        therefore gives a DIFFERENT hash than that of the config actually used.

        Computed via :func:`config_fingerprint` -- the same algorithm as
        :class:`~twinlink_ml.heatnetwork.config.HeatNetworkConfig`'s fingerprint, one hashing
        procedure for all config hierarchies in the project instead of one built anew per class.

        Documentary free-text fields (``provenance.description``, ``provenance.
        open_todos``) do NOT enter -- see :func:`without_documentary_provenance`.
        """
        return config_fingerprint(self._fingerprint_payload())

    def _fingerprint_payload(self) -> dict[str, Any]:
        return without_unset_optional_fields(without_documentary_provenance(self.to_dict()))

    @property
    def section_fingerprints(self) -> dict[str, str]:
        """Partial fingerprints per config section, in addition to :attr:`fingerprint`.

        :attr:`fingerprint` is a monolithic hash over the entire config -- with
        a deviation between two configs (as with ``arch_composite.jsonl``,
        Befund 33: ``process/fingerprint`` matches none of the current
        ``conf/process/*.yaml``) it cannot tell WHICH
        section deviates without trying every combination. These partial fingerprints
        allow the direct, section-wise comparison of two :class:`ProcessConfig`
        instances. ``fingerprint`` itself stays unchanged and bit-identical to already
        logged records; this is a purely additive extension.
        """
        sections = (
            "provenance",
            "geometry",
            "gas",
            "material",
            "energy",
            "exhaust",
            "operation",
            "actions",
            "disturbances",
            "numerics",
            "measurement",
            "backend",
        )
        return section_fingerprints_of(self._fingerprint_payload(), sections)

    def warn_if_placeholder(self) -> None:
        if self.provenance.status == "placeholder":
            warnings.warn(
                "ProcessConfig enthält Platzhalterwerte "
                f"(offene TODOs: {len(self.provenance.open_todos)}). "
                "Absolutwerte sind nicht belastbar; nur relative Architekturvergleiche.",
                PlaceholderConfigWarning,
                stacklevel=2,
            )


_DOCUMENTARY_PROVENANCE_FIELDS = ("description", "open_todos")
"""Freitextfelder in :class:`Provenance`, die den Fingerprint NICHT beeinflussen sollen.

Begründung (Audit 2026-09-25, siehe :func:`without_documentary_provenance`): der
Fingerprint identifiziert die NUMERISCHE Konfiguration. Eine reine Textänderung an
``provenance.description`` oder ``provenance.open_todos`` (Präzisierung, abgehaktes
TODO, künftige Übersetzung ins Englische) ändert an der simulierten Physik nichts und
darf die Vergleichbarkeit zweier Läufe nicht brechen. Alle Felder aller Config-Klassen
wurden geprüft, im Zweifel blieb ein Feld im Hash:

* ``provenance.status`` bleibt im Hash -- kategorial (placeholder/partially_calibrated/
  calibrated), semantisch relevant für die Aussagekraft eines Laufs.
* ``HeatSourceConfig.name``/``HeatSinkConfig.name`` bleiben im Hash -- Referenzpunkte für
  ``routing.py``/``cascade_order``, kein Freitext, sondern Bezeichner mit Bedeutung.
* ``ProcessConfig.backend`` bleibt im Hash -- wählt die Rechen-Implementierung (numpy/
  torch), kein Kommentar.
* Alle übrigen Felder sind Zahlen, Enums oder Bezeichner, keine Prosa.
"""


def without_documentary_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    """Removes :data:`_DOCUMENTARY_PROVENANCE_FIELDS` from ``payload["provenance"]``.

    Works on the full, nested ``to_dict()`` form (with ``"provenance"``
    key) -- :func:`section_fingerprints_of` then reads ``payload[section]`` from
    the same, already cleaned dict, so that ``section_fingerprints["provenance"]``
    and :attr:`ProcessConfig.fingerprint` see the same cleaning. No ``"provenance"``
    key (e.g. a single, already extracted section without provenance) --
    return unchanged.
    """
    if "provenance" not in payload or not isinstance(payload["provenance"], dict):
        return payload
    result = dict(payload)
    result["provenance"] = {
        key: value
        for key, value in payload["provenance"].items()
        if key not in _DOCUMENTARY_PROVENANCE_FIELDS
    }
    return result


_OPTIONAL_UNSET_OMITTED_FIELDS: dict[str, tuple[str, ...]] = {
    "material": ("product_moisture_limit_kg_per_kg", "drying_curve_exponent"),
    "operation": ("exhaust_recirculation_fraction",),
    "energy": ("recirculation_duct_loss_coefficient",),
}
"""Optional fields that are left OUT of the fingerprint payload while they are ``None``.

A field added later with default ``None`` would otherwise add a key to every payload and
move the fingerprint of every existing config and every logged record. Once set, the field
is part of the hash like any other. Per section, the field names."""


def without_unset_optional_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop :data:`_OPTIONAL_UNSET_OMITTED_FIELDS` entries whose value is ``None``."""
    result = dict(payload)
    for section, names in _OPTIONAL_UNSET_OMITTED_FIELDS.items():
        if isinstance(result.get(section), dict):
            result[section] = {
                key: value
                for key, value in result[section].items()
                if not (key in names and value is None)
            }
    return result


def config_fingerprint(payload: dict[str, Any]) -> str:
    """Stable 16-digit hex hash of a resolved config ``dict`` (``dataclasses.asdict``).

    Common hashing algorithm for all config hierarchies of the project
    (:class:`ProcessConfig`, ``heatnetwork.config.HeatNetworkConfig``) -- one procedure,
    checked once (stability, sorting of keys, ``str`` fallback for non
    JSON-native types), not rebuilt per config class by copy-paste. Extracted from
    :attr:`ProcessConfig.fingerprint`; identical input gives identical output as
    before this extraction (pure refactor, no new behaviour).

    Expects ``payload`` to be already cleaned of documentary free-text fields
    (:func:`without_documentary_provenance`) -- this function itself hashes what it is
    given, without its own knowledge of the config structure (thus stays generic for arbitrary
    config hierarchies, as the docstring above promises).
    """
    return str(
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
    )


def section_fingerprints_of(payload: dict[str, Any], sections: tuple[str, ...]) -> dict[str, str]:
    """:func:`config_fingerprint` per named section of ``payload``.

    Common basis for :attr:`ProcessConfig.section_fingerprints` and the corresponding
    property on ``HeatNetworkConfig`` -- see :func:`config_fingerprint`.
    """
    return {section: config_fingerprint(payload[section]) for section in sections}


def nucleation_fraction_for_target_diameter(
    target_diameter_m: float, nucleus_diameter_m: float
) -> float:
    r"""Nucleation fraction that gives the target particle size at steady state.

    .. math:: b_{\mathrm{nuc}} = \left(\frac{d_{\mathrm{nuc}}}{d_{50,\mathrm{soll}}}\right)^{3}
    """
    if target_diameter_m <= nucleus_diameter_m:
        raise ValueError("Zielkorngröße muss über der Keimgröße liegen")
    return (nucleus_diameter_m / target_diameter_m) ** 3


def load_process_config(path: str | Path) -> ProcessConfig:
    """Loads a YAML config into the dataclass hierarchy.

    Deliberately without a Hydra dependency: the dataclasses are valid Hydra structured configs
    and can later be registered via ``ConfigStore.store(node=ProcessConfig)``
    without the schema changing.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: erwartet ein YAML-Mapping auf oberster Ebene")
    config = _from_mapping(ProcessConfig, raw)
    config.warn_if_placeholder()
    return config


def dump_process_config(config: ProcessConfig, path: str | Path) -> None:
    """Serialises the fully resolved config (reproducibility requirement)."""
    Path(path).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _from_mapping(cls: type[T], data: dict[str, Any]) -> T:
    """Recursive, type-driven construction -- no loose dicts in the model code."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} ist keine Dataclass")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unbekannte Felder {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        annotation = known[name].type
        if isinstance(value, dict):
            nested = _resolve_nested_type(annotation)
            kwargs[name] = _from_mapping(nested, value) if nested is not None else value
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


_NESTED_TYPES: dict[str, type[Any]] = {
    "Provenance": Provenance,
    "GeometryConfig": GeometryConfig,
    "GasConfig": GasConfig,
    "MaterialConfig": MaterialConfig,
    "EnergyConfig": EnergyConfig,
    "ExhaustConfig": ExhaustConfig,
    "OperationConfig": OperationConfig,
    "ActionSpaceConfig": ActionSpaceConfig,
    "ActionBound": ActionBound,
    "DisturbanceSpaceConfig": DisturbanceSpaceConfig,
    "DisturbanceRange": DisturbanceRange,
    "NumericsConfig": NumericsConfig,
    "MeasurementConfig": MeasurementConfig,
}


def _resolve_nested_type(annotation: Any) -> type[Any] | None:
    key = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    return _NESTED_TYPES.get(key)


def _require_positive(obj: Any, *names: str) -> None:
    for name in names:
        if getattr(obj, name) <= 0.0:
            raise ValueError(f"{name} muss positiv sein, ist {getattr(obj, name)}")


def _require_non_negative(obj: Any, *names: str) -> None:
    for name in names:
        if getattr(obj, name) < 0.0:
            raise ValueError(f"{name} darf nicht negativ sein, ist {getattr(obj, name)}")
