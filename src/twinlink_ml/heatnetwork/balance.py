r"""Cascaded heat-recovery balance: exhaust air sources -> process air sinks.

This module is the specification for the future Dyssol balance module (AP3 of the
software integration partner, see ``docs/06_waermenetz.md``) -- it ports the
calculation rules of an existing
Excel balance of the consortium partner (two dryer lines, two heat exchangers, two
air heaters), but corrects three defects of the template in doing so. All three are named here,
because a future Dyssol module must not inherit the same defects just because it reads the same
template.

Basic formulas (EA1, unchanged from the template)
------------------------------------------------------
Recoverable power per source:

.. math:: \dot Q_{rec} = \dot m_L \bigl(T_{out} - (T_{dp}(Y_{out}) + \Delta T_{approach})\bigr)
    \,(c_{p,L} + Y_{out}\, c_{p,D})

with :math:`\Delta T_{approach} = 3\,\mathrm K` (limit to condensation,
:data:`~.config.DEFAULT_DEW_POINT_APPROACH_K`). Fuel demand per sink:

.. math:: \dot m_{gas} = \dot m_L\,(T_{soll} - T_{nach\,WUET})\,(c_{p,L} + Y\, c_{p,D})
    \,/\,(H_u\,\eta_H)

Dew point via :func:`twinlink_ml.simulation.thermo.dew_point_celsius` (Buck 1996) --
the same correlation as everywhere in the project, not re-derived or swapped for Magnus
(the consortium partner computes with Magnus; the difference is below 0.06 K,
documented, but irrelevant -- the repo stays with Buck, see ``thermo.py``,
``saturation_pressure_pa`` docstring). Material properties exclusively from the injected
:class:`~twinlink_ml.simulation.config.GasConfig` (``cp_dry_air_j_per_kg_k``,
``cp_vapour_j_per_kg_k``), not set anew.

Second recovery mode: condensation (MA1)
------------------------------------------------
:attr:`~.config.HeatNetworkConfig.recovery_mode` (config, not a function parameter) chooses
between two cooling limits per source:

``"sensible"`` (default, EA1 unchanged)
    Cooling down to the dew-point limit ``T_dp(Y_out) + dT_approach`` -- no condensation in the
    exchanger, every result bit-identical to the state before MA1.

``"condensing"``
    Cooling down to the cold temperature :attr:`~.config.HeatNetworkConfig.
    cold_temperature_celsius` (``T_kalt``), with additional latent heat from the
    condensate formed in the process:

    .. math::
        \dot m_{Kondensat} &= \max(0,\, Y_{out} - Y_s(T_{kalt}))\, \dot m_L \\
        \dot Q_{latent} &= \dot m_{Kondensat}\, \Delta h_v(T_{kalt}) \\
        \dot Q_{sensibel} &= \dot m_L\,(T_{out} - T_{kalt})\,(c_{p,L} + Y_{out}\, c_{p,D}) \\
        \dot Q_{rec} &= \dot Q_{sensibel} + \dot Q_{latent}

    :math:`Y_s` (:func:`~twinlink_ml.simulation.thermo.saturation_humidity_kg_per_kg`) and
    :math:`\Delta h_v` (:func:`~twinlink_ml.simulation.thermo.latent_heat_j_per_kg`) come
    unchanged from ``thermo.py`` -- no second saturation/evaporation formula. If
    ``T_kalt`` lies above the dew point of the source, :math:`\dot m_{Kondensat}=0` and the mode
    falls back continuously to purely sensible cooling down to ``T_kalt`` (no jump at the
    dew-point limit).

    Within the cascade (multi-stage, EA2 correction 2) the latent heat is carried as a fixed
    reserve per source (``Q_latent`` at its outlet temperature, not recomputed at every
    stage) and only tapped once the sensible cooling reserve of a stage
    is already exhausted -- physically, condensation only starts once the stream
    has actually been cooled down to ``T_kalt``, not gradually before. The condensate
    leaves the balance as liquid water; :attr:`HeatNetworkResult.
    source_condensate_kg_per_s` makes this checkable for the mass balance (test:
    ``tests/test_heatnetwork_balance.py``).

Three corrections to the Excel template (EA2)
----------------------------------------------------
The template has three defects that :func:`evaluate_cascade` does NOT reproduce:

1. **Dew-point reference per source.** In the template the second heat exchanger computes
   against the dew point of the FIRST line -- a copy error between two adjacent
   worksheets, not a modelling decision. Here every source has its
   own dew-point limit, computed from its own ``outlet_humidity_kg_per_kg``
   (``source_floor_temperature_celsius`` in :func:`evaluate_cascade`), constant over all
   stages (the humidity does not change as long as only sensible cooling takes place,
   see correction 3).

2. **Cascade the residual enthalpy, do not discard it.** The template caps the air temperature
   of a sink at its set point and discards the unused source enthalpy -- in
   one documented case 891.8 kW at 87.4 °C, far above any dew-point limit and
   thus still usable. Here every source is a stream with a temperature DECREASING over
   the cascade: after each stage its temperature drops by
   :math:`\dot Q_{genutzt} / (\dot m_L\, c_{p,eff})`, the rest is available to the next stage
   (possibly a different sink) until the dew-point limit is reached. Which
   share per source was actually used is given in
   :attr:`HeatNetworkResult.source_used_power_kw`/``source_discarded_power_kw`` --
   ``used + discarded == available`` up to rounding (energy conservation, see
   ``tests/test_heatnetwork_balance.py``).

3. **Minimum approach temperature per stage.** The template nowhere checks
   :math:`T_{Luft,aus} \le T_{Quelle} - \Delta T_{min}`; its 120 °C cap
   prevents a violation only by chance, not structurally. Here the approach temperature is
   a hard constraint of every stage (:attr:`~.config.HeatNetworkConfig.
   minimum_pinch_k`, placeholder rationale see
   :data:`~.config.DEFAULT_MINIMUM_PINCH_K`). Cascade order: low-temperature sources
   first at the cold end, high-temperature sources last (otherwise the sink air is already
   too warm to still take up a cold source) -- default rule in
   ``routing.resolve_cascade_order``, overridable via
   :attr:`~.config.HeatNetworkConfig.cascade_order`.

``stage_binding_code`` in :class:`HeatNetworkResult` knows, in addition to the two states
named in EA3 (``pinch``, ``dew_point``), a third one, ``sink_satisfied``: a
stage that brings the sink to its set point without either of the two physical limits
being reached is not a limiting case -- reporting it wrongly as "approach temperature
binding" or "dew point binding" would itself be a new interpretation error.

Coupling to the simulator (EA4)
----------------------------------
The exhaust air state comes from ``simulation/exhaust.py``. Literal signature::

    def reconstruct_exhaust_state(
        backend: ArrayBackend,
        gas: GasConfig,
        material: MaterialConfig,
        isotherm: SorptionIsotherm,
        exhaust: ExhaustConfig,
        *,
        bed_temperature_celsius: Array,
        particle_moisture_kg_per_kg: Array,
        dry_air_mass_flow_kg_per_s: Array,
        inlet_absolute_humidity_kg_per_kg: Array,
    ) -> tuple[Array, Array, Array]:
        ...
        :returns: ``(T_gas_out_celsius, Y_gas_out_kg_per_kg, outlet_temperature_celsius)``

It RETURNS: ``T_gas_out_celsius`` (== bed temperature, by the model assumption of an ideally
mixed bed), ``Y_gas_out_kg_per_kg`` and ``outlet_temperature_celsius`` (bed temperature
minus ``ExhaustConfig.outlet_duct_heat_loss_k``). It already REQUIRES as an input (not as a
return value) what is needed here as :attr:`~.config.HeatSourceConfig.dry_air_mass_flow_kg_per_s`
-- this mass flow is NOT a return quantity of the function but one of its inputs,
typically taken from ``dry_air_mass_flow_kg_per_s`` in ``diagnostics``/``SteadyStateSolution``,
not recomputed from the gas velocity (the same quantity, just already known beforehand
in the caller).

**Important:** :class:`~.config.HeatSourceConfig.outlet_temperature_celsius` is the OUTLET duct
(third return value, ``outlet_temperature_celsius``), NOT the bed temperature (first
return value, ``T_gas_out_celsius``/``bed_temperature_celsius``). With
``outlet_duct_heat_loss_k = 0.0`` (default) both are numerically equal, but the contract at the
consortium interface is the exhaust air temperature -- this distinction is already established
in the repo (``exhaust.py``, contract-correction paragraph) and is not blurred again here.

**Scale caveat.** The consortium model computes with about 8 kg/s of process air
(``evaluation.recoverable_power.DISTRICT_MASS_FLOW_KG_PER_S``), the lab simulator with about
0.03 kg/s (``dry_air_mass_flow_kg_per_s`` in ``data.trajectory.diagnostics``) -- two
orders of magnitude. Every kW figure from coupling lab sources with consortium sinks (or
vice versa) is a scaling calculation, not a plant statement (see
``docs/06_waermenetz.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..simulation.backend import NumpyBackend
from ..simulation.thermo import (
    dew_point_celsius,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)
from .approach import (
    SourceProfile,
    build_source_profile,
    max_duty_along,
    profile_temperature,
    stage_approach_diagnostics,
)
from .config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig

__all__ = [
    "BINDING_DEW_POINT",
    "BINDING_EXCHANGER_CAP",
    "BINDING_LABELS",
    "BINDING_PINCH",
    "BINDING_SINK_SATISFIED",
    "HeatNetworkResult",
    "SharedCap",
    "SinkState",
    "SourceState",
    "binding_label",
    "build_source_profiles",
    "evaluate_cascade",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

BINDING_PINCH: Final = 0
"""Stufe durch die Mindestgrädigkeit begrenzt (EA2, Korrektur 3)."""
BINDING_DEW_POINT: Final = 1
"""Stufe durch die Taupunktgrenze der Quelle begrenzt (EA2, Korrektur 1)."""
BINDING_SINK_SATISFIED: Final = 2
"""Senke erreicht ihren Sollwert, ohne dass Grädigkeit oder Taupunkt anstehen -- kein
physikalischer Grenzfall, siehe Moduldocstring."""

BINDING_EXCHANGER_CAP: Final = 3
"""Stufe durch die Leistungsgrenze eines geteilten Übertragers begrenzt (E1, :class:`SharedCap`);
tritt nur auf, wenn ``evaluate_cascade`` mit ``shared_caps`` aufgerufen wird."""

BINDING_LABELS: Final[dict[int, str]] = {
    BINDING_PINCH: "pinch",
    BINDING_DEW_POINT: "dew_point",
    BINDING_SINK_SATISFIED: "sink_satisfied",
    BINDING_EXCHANGER_CAP: "exchanger_cap",
}

_POWER_TOLERANCE_W = 1e-6
"""Numerische Toleranz für den Vergleich zweier Leistungen bei der Bindungsklassifikation
-- verhindert, dass Rundung im letzten Bit eine ``sink_satisfied``-Stufe fälschlich als
``dew_point`` meldet."""

_TEMPERATURE_TOLERANCE_K = 1e-9
"""Analoge Toleranz für den Vergleich zweier Temperaturen (Pinch- gegen Sollwert-Grenze)."""


def _as_step_array(value: float, n_steps: int) -> FloatArray:
    """Scalar -> array of shape ``(n_steps,)``, existing ``(n_steps,)`` arrays unchanged.

    Allows the same :func:`evaluate_cascade` to be used both for a single, static
    evaluation (``n_steps == 1``, built directly from a :class:`~.config.HeatSourceConfig`)
    and for a whole trajectory (``n_steps > 1``, real
    time series as input) -- one code path for both cases instead of special handling
    of the scalar case.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(n_steps, float(array), dtype=np.float64)
    if array.shape != (n_steps,):
        raise ValueError(f"erwartete Form ({n_steps},), ist {array.shape}")
    return array


@dataclass(frozen=True, slots=True)
class SourceState:
    """Time series (or single value, ``n_steps == 1``) of an exhaust source for a cascade run."""

    name: str
    dry_air_mass_flow_kg_per_s: FloatArray
    outlet_temperature_celsius: FloatArray
    outlet_humidity_kg_per_kg: FloatArray

    def __post_init__(self) -> None:
        shapes = {
            self.dry_air_mass_flow_kg_per_s.shape,
            self.outlet_temperature_celsius.shape,
            self.outlet_humidity_kg_per_kg.shape,
        }
        if len(shapes) != 1:
            raise ValueError(f"{self.name}: Felder haben unterschiedliche Form {shapes}")

    @classmethod
    def from_config(cls, config: HeatSourceConfig, n_steps: int = 1) -> SourceState:
        """Static source (``n_steps == 1`` by default) from a :class:`HeatSourceConfig`.

        For a real trajectory, construct directly with ``(n_steps,)`` arrays
        instead -- this factory is the bridging case (single-stage comparison,
        Excel recalculation), not the RL/evaluation case.
        """
        return cls(
            name=config.name,
            dry_air_mass_flow_kg_per_s=_as_step_array(config.dry_air_mass_flow_kg_per_s, n_steps),
            outlet_temperature_celsius=_as_step_array(config.outlet_temperature_celsius, n_steps),
            outlet_humidity_kg_per_kg=_as_step_array(config.outlet_humidity_kg_per_kg, n_steps),
        )


@dataclass(frozen=True, slots=True)
class SinkState:
    """Time series (or single value) of a sink for a cascade run."""

    name: str
    dry_air_mass_flow_kg_per_s: FloatArray
    humidity_kg_per_kg: FloatArray
    intake_temperature_celsius: FloatArray
    setpoint_temperature_celsius: FloatArray

    def __post_init__(self) -> None:
        shapes = {
            self.dry_air_mass_flow_kg_per_s.shape,
            self.humidity_kg_per_kg.shape,
            self.intake_temperature_celsius.shape,
            self.setpoint_temperature_celsius.shape,
        }
        if len(shapes) != 1:
            raise ValueError(f"{self.name}: Felder haben unterschiedliche Form {shapes}")

    @classmethod
    def from_config(cls, config: HeatSinkConfig, n_steps: int = 1) -> SinkState:
        """Static sink (``n_steps == 1`` by default) from a :class:`HeatSinkConfig`."""
        return cls(
            name=config.name,
            dry_air_mass_flow_kg_per_s=_as_step_array(config.dry_air_mass_flow_kg_per_s, n_steps),
            humidity_kg_per_kg=_as_step_array(config.humidity_kg_per_kg, n_steps),
            intake_temperature_celsius=_as_step_array(config.intake_temperature_celsius, n_steps),
            setpoint_temperature_celsius=_as_step_array(
                config.setpoint_temperature_celsius, n_steps
            ),
        )


@dataclass(frozen=True, slots=True)
class HeatNetworkResult:
    """Result of a cascade run.

    All arrays vectorised over the time-step axis (last axis, length ``n_steps``);
    the second-to-last axis indexes stages/sources/sinks in the order of the respective
    ``*_names`` tuple.

    :ivar stage_source_names: Source name per stage, length ``n_stages``.
    :ivar stage_sink_names: Sink name per stage, length ``n_stages``.
    :ivar stage_recovered_power_kw: Used power per stage, shape ``(n_stages, n_steps)``.
    :ivar stage_binding_code: Binding constraint per stage (see :data:`BINDING_LABELS`),
        shape ``(n_stages, n_steps)``.
    :ivar stage_source_outlet_temperature_celsius: Temperature of the source AFTER this stage
        (before the next stage that references the same source, if any), shape
        ``(n_stages, n_steps)`` -- diagnostic quantity to make the approach/dew-point invariants
        per stage checkable from outside without duplicating the cascade logic in the test.
    :ivar stage_sink_outlet_temperature_celsius: Temperature of the sink AFTER this stage,
        analogously, shape ``(n_stages, n_steps)``.
    :ivar source_names: Source names, length ``n_sources``.
    :ivar source_available_power_kw: Maximum recoverable power per source (cooling down to
        the dew-point limit, EA1 formula unchanged), shape ``(n_sources, n_steps)``.
    :ivar source_used_power_kw: Power actually used per source across all its stages,
        shape ``(n_sources, n_steps)``.
    :ivar source_discarded_power_kw: ``available - used`` per source -- with correct
        cascading only what could no longer be extracted because of the dew-point limit,
        no longer the full residual enthalpy as in the uncorrected template (correction 2).
    :ivar source_condensate_kg_per_s: Mass flow actually condensed out per source
        (MA1), shape ``(n_sources, n_steps)``. Zero in ``recovery_mode="sensible"`` (default)
        and whenever ``cold_temperature_celsius`` lies above the dew point of the respective
        source. Leaves the balance as liquid water -- for the mass balance,
        ``outlet_humidity = outlet_humidity_kg_per_kg - source_condensate_kg_per_s /
        dry_air_mass_flow_kg_per_s`` holds (down to ``Y_s(T_kalt)``, see module docstring).
    :ivar source_outlet_temperature_celsius: Outlet temperature of the source after ALL its
        stages, shape ``(n_sources, n_steps)``.
    :ivar sink_names: Sink names, length ``n_sinks``.
    :ivar sink_outlet_temperature_celsius: ``T_nach_WUET`` per sink after all its stages,
        shape ``(n_sinks, n_steps)``.
    :ivar fuel_mass_flow_kg_per_s: ``m_gas`` per sink, shape ``(n_sinks, n_steps)``.
    :ivar stage_min_approach_k: Smallest approach temperature along the stage (mode
        ``"along_exchanger"``; ``NaN`` in mode ``"warm_end"``), shape ``(n_stages, n_steps)``.
    :ivar stage_pinch_position: Position of this bottleneck, 0 = warm end, 1 = cold end
        (``NaN`` in mode ``"warm_end"``), shape ``(n_stages, n_steps)``.
    """

    stage_source_names: tuple[str, ...]
    stage_sink_names: tuple[str, ...]
    stage_recovered_power_kw: FloatArray
    stage_binding_code: IntArray
    stage_source_outlet_temperature_celsius: FloatArray
    stage_sink_outlet_temperature_celsius: FloatArray
    source_names: tuple[str, ...]
    source_available_power_kw: FloatArray
    source_used_power_kw: FloatArray
    source_discarded_power_kw: FloatArray
    source_condensate_kg_per_s: FloatArray
    source_outlet_temperature_celsius: FloatArray
    sink_names: tuple[str, ...]
    sink_outlet_temperature_celsius: FloatArray
    fuel_mass_flow_kg_per_s: FloatArray
    stage_min_approach_k: FloatArray
    stage_pinch_position: FloatArray

    def total_recovered_power_kw(self) -> FloatArray:
        """Sum of the used power over all stages, per time step."""
        return np.sum(self.stage_recovered_power_kw, axis=0)

    def total_fuel_mass_flow_kg_per_s(self) -> FloatArray:
        """Sum of the fuel demand over all sinks, per time step."""
        return np.sum(self.fuel_mass_flow_kg_per_s, axis=0)


def binding_label(code: int) -> str:
    """Human-readable label for an entry of ``stage_binding_code``."""
    return BINDING_LABELS[code]


@dataclass(frozen=True, slots=True)
class SharedCap:
    """E1: power limit of ONE exchanger that several stages share.

    ``stages`` are ``(Quellenname, Senkenname)`` pairs; the sum of the transferred power of all
    stages of the group that appear in the stage list must not exceed ``power_kw``. The
    stages serve the limit in the order of the cascade (the first takes what it would get after
    pinch and dew point, at most the limit; the next one the rest). Pairs that are missing from the
    stage list do nothing -- a wiring without these stages is not affected by the
    limit. Without ``shared_caps`` the cascade is bit-identical to the state before E1.
    """

    stages: tuple[tuple[str, str], ...]
    power_kw: float

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("SharedCap braucht mindestens eine Stufe")
        if not self.power_kw >= 0.0:
            raise ValueError(f"power_kw muss >= 0 sein, ist {self.power_kw}")


def _shared_step_count(sources: tuple[SourceState, ...], sinks: tuple[SinkState, ...]) -> int:
    shapes = {state.outlet_temperature_celsius.shape[0] for state in sources}
    shapes |= {state.intake_temperature_celsius.shape[0] for state in sinks}
    if len(shapes) != 1:
        raise ValueError(f"alle Quellen/Senken müssen dieselbe Schrittzahl haben, sind {shapes}")
    return int(next(iter(shapes)))


def _validate_stage_names(
    stages: tuple[tuple[str, str], ...],
    source_by_name: dict[str, SourceState],
    sink_by_name: dict[str, SinkState],
) -> None:
    for source_name, sink_name in stages:
        if source_name not in source_by_name:
            raise ValueError(f"stages referenziert unbekannte Quelle {source_name!r}")
        if sink_name not in sink_by_name:
            raise ValueError(f"stages referenziert unbekannte Senke {sink_name!r}")


def evaluate_cascade(
    config: HeatNetworkConfig,
    stages: tuple[tuple[str, str], ...],
    sources: tuple[SourceState, ...],
    sinks: tuple[SinkState, ...],
    shared_caps: tuple[SharedCap, ...] = (),
    profiles: dict[str, SourceProfile] | None = None,
) -> HeatNetworkResult:
    r"""Evaluate the cascade along ``stages`` -- the actual computational core (EA2/EA3).

    ``stages`` is an ordered list of ``(Quellenname, Senkenname)`` pairs, typically
    the result of ``routing.HeatNetworkRouting.route`` -- this function does not know
    ``routing.py`` (dependency injection: the wiring is passed in fully resolved, not decided
    here). Every stage is processed in the given order; sources and sinks
    carry a running state (current temperature) that is passed on from stage to stage
    -- exactly the cascading of correction 2.

    Per stage: the sink is heated as far as the smaller of two values allows --
    its own set point or the minimum approach temperature to the CURRENT source temperature
    (already cooled by previous stages). The power needed for this is additionally
    capped by what the source can still give without falling below its dew-point
    limit (correction 1: its own dew-point limit, not that of another source).
    Unused source enthalpy remains available for the next stage (correction 2),
    as long as ``stages`` assigns it to a further sink.

    Heat capacity rates determined by mass flow and humidity (:math:`\dot m_L (c_{p,L}+Y c_{p,D})`)
    are treated per source/sink as CONSTANT over the cascade, computed from the
    original ``outlet_humidity_kg_per_kg`` -- purely sensible heat transfer, no
    humidification/dehumidification of the SINK in this module. In ``recovery_mode="sensible"``
    (default) this also holds unchanged for the source (the humidity does not change
    as long as the source stays above its dew-point limit; below it, by construction,
    no further cooling takes place). In ``recovery_mode="condensing"`` (MA1) the source
    additionally gives off latent heat from condensation -- this changes its humidity (condensate
    leaves the balance, see :attr:`HeatNetworkResult.source_condensate_kg_per_s`), but NOT the
    capacity rate used here: the sensible cooling reserve is still computed with the
    original humidity, because in the model condensation only sets in on reaching
    ``cold_temperature_celsius`` (flash condensation at the cold end, no
    gradual dehumidification along the cooling path) -- after that the source is
    exhausted for this stage anyway.

    ``shared_caps`` (E1, optional): power limits of shared exchangers, see
    :class:`SharedCap`. Empty (default) means: calculation path and result bit-identical to the
    state before E1.

    **Approach check (Befund 79, review finding A1).** The text above describes the mode
    ``config.approach_check="warm_end"`` (state before Befund 79, only for reproducing old
    artifacts). In the default ``"along_exchanger"``, ``dT_min`` applies as the minimum over the
    whole counter-current exchanger, on the source profile with condensation from the dew point,
    see :mod:`~twinlink_ml.heatnetwork.approach`; ``profiles`` (optional) are pre-built
    profiles per source (:func:`build_source_profiles`), so that many wirings of the same
    sources build them only once.

    :raises ValueError: if ``stages`` references a source/sink that does not appear in
        ``sources``/``sinks``, or if sources/sinks come with different numbers of
        steps.
    """
    source_by_name = {state.name: state for state in sources}
    sink_by_name = {state.name: state for state in sinks}
    _validate_stage_names(stages, source_by_name, sink_by_name)
    for cap in shared_caps:
        _validate_stage_names(cap.stages, source_by_name, sink_by_name)
    n_steps = _shared_step_count(sources, sinks)
    cap_group_of_stage: dict[int, int] = {}
    for group_index, cap in enumerate(shared_caps):
        for stage_index, pair in enumerate(stages):
            if pair in cap.stages:
                cap_group_of_stage[stage_index] = group_index
    remaining_cap_w = [np.full(n_steps, cap.power_kw * 1000.0) for cap in shared_caps]
    if config.approach_check == "along_exchanger":
        return _evaluate_cascade_along(
            config,
            stages,
            source_by_name,
            sink_by_name,
            cap_group_of_stage,
            remaining_cap_w,
            n_steps,
            profiles,
        )

    gas = config.gas
    backend = NumpyBackend()

    initial_source_temperature_celsius = {
        name: state.outlet_temperature_celsius for name, state in source_by_name.items()
    }
    source_capacity_rate_w_per_k = {
        name: state.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + state.outlet_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        for name, state in source_by_name.items()
    }
    is_condensing = config.recovery_mode == "condensing"
    cold_temperature_celsius = backend.asarray(config.cold_temperature_celsius)
    condensate_specific_enthalpy_j_per_kg = latent_heat_j_per_kg(backend, cold_temperature_celsius)

    source_floor_temperature_celsius = {
        name: (
            np.full(n_steps, config.cold_temperature_celsius, dtype=np.float64)
            if is_condensing
            else dew_point_celsius(backend, gas, backend.asarray(state.outlet_humidity_kg_per_kg))
            + config.dew_point_approach_k
        )
        for name, state in source_by_name.items()
    }
    # MA1: latent power reserve per source, computed once from the outlet humidity
    # (no recomputation per stage -- condensation only sets in once the stream has
    # actually been cooled down to T_kalt, see module docstring). Zero in
    # ``recovery_mode="sensible"`` and whenever T_kalt lies above the dew point of the source
    # -- in both cases bit-identical to the state before MA1 (no latent contribution).
    source_latent_power_w = {
        name: (
            np.maximum(
                state.outlet_humidity_kg_per_kg
                - saturation_humidity_kg_per_kg(backend, gas, cold_temperature_celsius),
                0.0,
            )
            * state.dry_air_mass_flow_kg_per_s
            * condensate_specific_enthalpy_j_per_kg
            if is_condensing
            else np.zeros(n_steps, dtype=np.float64)
        )
        for name, state in source_by_name.items()
    }
    remaining_latent_power_w = {name: array.copy() for name, array in source_latent_power_w.items()}
    current_source_temperature_celsius = {
        name: array.copy() for name, array in initial_source_temperature_celsius.items()
    }
    source_used_power_w = {name: np.zeros(n_steps, dtype=np.float64) for name in source_by_name}
    source_condensate_used_kg_per_s = {
        name: np.zeros(n_steps, dtype=np.float64) for name in source_by_name
    }

    sink_capacity_rate_w_per_k = {
        name: state.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + state.humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        for name, state in sink_by_name.items()
    }
    current_sink_temperature_celsius = {
        name: state.intake_temperature_celsius.copy() for name, state in sink_by_name.items()
    }

    n_stages = len(stages)
    stage_power_kw = np.zeros((n_stages, n_steps), dtype=np.float64)
    stage_binding_code = np.zeros((n_stages, n_steps), dtype=np.int64)
    stage_source_outlet_temperature_celsius = np.zeros((n_stages, n_steps), dtype=np.float64)
    stage_sink_outlet_temperature_celsius = np.zeros((n_stages, n_steps), dtype=np.float64)

    for stage_index, (source_name, sink_name) in enumerate(stages):
        capacity_source = source_capacity_rate_w_per_k[source_name]
        capacity_sink = sink_capacity_rate_w_per_k[sink_name]
        temperature_source = current_source_temperature_celsius[source_name]
        temperature_sink = current_sink_temperature_celsius[sink_name]
        floor_temperature = source_floor_temperature_celsius[source_name]
        setpoint_temperature = sink_by_name[sink_name].setpoint_temperature_celsius

        pinch_cap_temperature = temperature_source - config.minimum_pinch_k
        target_sink_temperature = np.minimum(setpoint_temperature, pinch_cap_temperature)
        target_sink_temperature = np.maximum(target_sink_temperature, temperature_sink)

        power_for_target_w = capacity_sink * (target_sink_temperature - temperature_sink)
        sensible_available_w = np.maximum(
            capacity_source * (temperature_source - floor_temperature), 0.0
        )
        # MA1: the remaining latent reserve is added on top of the sensible cooling reserve --
        # with ``recovery_mode="sensible"`` it is 0 and ``power_available_w`` bit-identical to
        # ``sensible_available_w`` (IEEE-754: ``x + 0.0 == x``).
        power_available_w = sensible_available_w + remaining_latent_power_w[source_name]
        power_actual_w = np.clip(np.minimum(power_for_target_w, power_available_w), 0.0, None)
        power_uncapped_w = power_actual_w
        group_index = cap_group_of_stage.get(stage_index)
        if group_index is not None:
            power_actual_w = np.minimum(power_actual_w, remaining_cap_w[group_index])
            remaining_cap_w[group_index] = remaining_cap_w[group_index] - power_actual_w
        # Sensible share first (lowers the source temperature), latent share only afterwards
        # (condensation only sets in on reaching ``floor_temperature``) -- ``np.minimum``
        # returns ``power_actual_w`` unchanged here as long as there is no latent reserve,
        # i.e. without additional rounding compared with the state before MA1.
        power_sensible_used_w = np.minimum(power_actual_w, sensible_available_w)
        power_latent_used_w = power_actual_w - power_sensible_used_w

        dew_point_binds = power_uncapped_w < power_for_target_w - _POWER_TOLERANCE_W
        pinch_binds = (~dew_point_binds) & (
            pinch_cap_temperature < setpoint_temperature - _TEMPERATURE_TOLERANCE_K
        )
        binding_code = np.full(n_steps, BINDING_SINK_SATISFIED, dtype=np.int64)
        binding_code = np.where(pinch_binds, BINDING_PINCH, binding_code)
        binding_code = np.where(dew_point_binds, BINDING_DEW_POINT, binding_code)
        if group_index is not None:
            binding_code = np.where(
                power_actual_w < power_uncapped_w - _POWER_TOLERANCE_W,
                BINDING_EXCHANGER_CAP,
                binding_code,
            )

        stage_power_kw[stage_index] = power_actual_w / 1000.0
        stage_binding_code[stage_index] = binding_code

        current_source_temperature_celsius[source_name] = (
            temperature_source - power_sensible_used_w / capacity_source
        )
        remaining_latent_power_w[source_name] = (
            remaining_latent_power_w[source_name] - power_latent_used_w
        )
        current_sink_temperature_celsius[sink_name] = (
            temperature_sink + power_actual_w / capacity_sink
        )
        source_used_power_w[source_name] = source_used_power_w[source_name] + power_actual_w
        source_condensate_used_kg_per_s[source_name] = source_condensate_used_kg_per_s[
            source_name
        ] + (power_latent_used_w / condensate_specific_enthalpy_j_per_kg if is_condensing else 0.0)
        stage_source_outlet_temperature_celsius[stage_index] = current_source_temperature_celsius[
            source_name
        ]
        stage_sink_outlet_temperature_celsius[stage_index] = current_sink_temperature_celsius[
            sink_name
        ]

    source_names = tuple(source_by_name)
    source_available_power_kw = np.stack(
        [
            (
                np.maximum(
                    source_capacity_rate_w_per_k[name]
                    * (
                        initial_source_temperature_celsius[name]
                        - source_floor_temperature_celsius[name]
                    ),
                    0.0,
                )
                + source_latent_power_w[name]
            )
            / 1000.0
            for name in source_names
        ]
    )
    source_used_power_kw = np.stack([source_used_power_w[name] / 1000.0 for name in source_names])
    source_discarded_power_kw = source_available_power_kw - source_used_power_kw
    source_condensate_kg_per_s = np.stack(
        [source_condensate_used_kg_per_s[name] for name in source_names]
    )
    source_outlet_temperature_celsius = np.stack(
        [current_source_temperature_celsius[name] for name in source_names]
    )

    sink_names = tuple(sink_by_name)
    sink_outlet_temperature_celsius = np.stack(
        [current_sink_temperature_celsius[name] for name in sink_names]
    )
    fuel_mass_flow_kg_per_s = np.stack(
        [
            np.maximum(
                sink_capacity_rate_w_per_k[name]
                * (
                    sink_by_name[name].setpoint_temperature_celsius
                    - current_sink_temperature_celsius[name]
                ),
                0.0,
            )
            / (config.fuel_lower_heating_value_j_per_kg * config.heater_efficiency)
            for name in sink_names
        ]
    )

    return HeatNetworkResult(
        stage_source_names=tuple(source_name for source_name, _ in stages),
        stage_sink_names=tuple(sink_name for _, sink_name in stages),
        stage_recovered_power_kw=stage_power_kw,
        stage_binding_code=stage_binding_code,
        stage_source_outlet_temperature_celsius=stage_source_outlet_temperature_celsius,
        stage_sink_outlet_temperature_celsius=stage_sink_outlet_temperature_celsius,
        source_names=source_names,
        source_available_power_kw=source_available_power_kw,
        source_used_power_kw=source_used_power_kw,
        source_discarded_power_kw=source_discarded_power_kw,
        source_condensate_kg_per_s=source_condensate_kg_per_s,
        source_outlet_temperature_celsius=source_outlet_temperature_celsius,
        sink_names=sink_names,
        sink_outlet_temperature_celsius=sink_outlet_temperature_celsius,
        fuel_mass_flow_kg_per_s=fuel_mass_flow_kg_per_s,
        stage_min_approach_k=np.full((n_stages, n_steps), np.nan),
        stage_pinch_position=np.full((n_stages, n_steps), np.nan),
    )


def build_source_profiles(
    config: HeatNetworkConfig, sources: tuple[SourceState, ...]
) -> dict[str, SourceProfile]:
    """Profiles per source for the mode ``"along_exchanger"`` (see :mod:`.approach`)."""
    return {
        state.name: build_source_profile(
            config,
            dry_air_mass_flow_kg_per_s=state.dry_air_mass_flow_kg_per_s,
            outlet_temperature_celsius=state.outlet_temperature_celsius,
            outlet_humidity_kg_per_kg=state.outlet_humidity_kg_per_kg,
        )
        for state in sources
    }


def _evaluate_cascade_along(
    config: HeatNetworkConfig,
    stages: tuple[tuple[str, str], ...],
    source_by_name: dict[str, SourceState],
    sink_by_name: dict[str, SinkState],
    cap_group_of_stage: dict[int, int],
    remaining_cap_w: list[FloatArray],
    n_steps: int,
    profiles: dict[str, SourceProfile] | None,
) -> HeatNetworkResult:
    """Cascade with ``dT_min`` as the minimum along each stage (Befund 79, A1).

    The same mechanism as the mode ``"warm_end"`` (order of the stages, set-point limit of the
    sink, residual heat of the source down to its cooling limit, shared exchangers, fuel), but
    the stage power is the largest at which the approach temperature at EVERY point of the
    counter-current exchanger is at least ``dT_min``, and the latent heat is released along
    the profile from the dew point onwards (no flash at ``T_kalt``).
    """
    gas = config.gas
    if profiles is None:
        profiles = build_source_profiles(config, tuple(source_by_name.values()))
    sink_capacity_rate_w_per_k = {
        name: state.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + state.humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)
        for name, state in sink_by_name.items()
    }
    current_sink_temperature_celsius = {
        name: state.intake_temperature_celsius.copy() for name, state in sink_by_name.items()
    }
    released_w = {name: np.zeros(n_steps, dtype=np.float64) for name in source_by_name}

    n_stages = len(stages)
    stage_power_kw = np.zeros((n_stages, n_steps), dtype=np.float64)
    stage_binding_code = np.zeros((n_stages, n_steps), dtype=np.int64)
    stage_source_outlet_temperature_celsius = np.zeros((n_stages, n_steps), dtype=np.float64)
    stage_sink_outlet_temperature_celsius = np.zeros((n_stages, n_steps), dtype=np.float64)
    stage_min_approach_k = np.zeros((n_stages, n_steps), dtype=np.float64)
    stage_pinch_position = np.zeros((n_stages, n_steps), dtype=np.float64)

    for stage_index, (source_name, sink_name) in enumerate(stages):
        profile = profiles[source_name]
        capacity_sink = sink_capacity_rate_w_per_k[sink_name]
        temperature_sink = current_sink_temperature_celsius[sink_name]
        setpoint_temperature = sink_by_name[sink_name].setpoint_temperature_celsius
        q_before = released_w[source_name]

        power_for_setpoint_w = np.maximum(
            capacity_sink * (setpoint_temperature - temperature_sink), 0.0
        )
        remaining_source_w = np.maximum(profile.total_w - q_before, 0.0)
        upper_w = np.minimum(power_for_setpoint_w, remaining_source_w)
        power_uncapped_w = max_duty_along(
            profile, q_before, temperature_sink, capacity_sink, config.minimum_pinch_k, upper_w
        )
        power_actual_w = power_uncapped_w
        group_index = cap_group_of_stage.get(stage_index)
        if group_index is not None:
            power_actual_w = np.minimum(power_actual_w, remaining_cap_w[group_index])
            remaining_cap_w[group_index] = remaining_cap_w[group_index] - power_actual_w

        pinch_binds = power_uncapped_w < upper_w - _POWER_TOLERANCE_W
        dew_point_binds = (~pinch_binds) & (
            remaining_source_w < power_for_setpoint_w - _POWER_TOLERANCE_W
        )
        binding_code = np.full(n_steps, BINDING_SINK_SATISFIED, dtype=np.int64)
        binding_code = np.where(dew_point_binds, BINDING_DEW_POINT, binding_code)
        binding_code = np.where(pinch_binds, BINDING_PINCH, binding_code)
        if group_index is not None:
            binding_code = np.where(
                power_actual_w < power_uncapped_w - _POWER_TOLERANCE_W,
                BINDING_EXCHANGER_CAP,
                binding_code,
            )

        minimum, position = stage_approach_diagnostics(
            profile, q_before, power_actual_w, temperature_sink, capacity_sink
        )
        stage_power_kw[stage_index] = power_actual_w / 1000.0
        stage_binding_code[stage_index] = binding_code
        stage_min_approach_k[stage_index] = minimum
        stage_pinch_position[stage_index] = position

        released_w[source_name] = q_before + power_actual_w
        current_sink_temperature_celsius[sink_name] = (
            temperature_sink + power_actual_w / capacity_sink
        )
        stage_source_outlet_temperature_celsius[stage_index] = profile_temperature(
            profile, released_w[source_name]
        )
        stage_sink_outlet_temperature_celsius[stage_index] = current_sink_temperature_celsius[
            sink_name
        ]

    source_names = tuple(source_by_name)
    source_available_power_kw = np.stack([profiles[name].total_w / 1000.0 for name in source_names])
    source_used_power_kw = np.stack([released_w[name] / 1000.0 for name in source_names])
    source_condensate_kg_per_s = np.stack(
        [
            (
                profiles[name].latent_released_w(released_w[name]) / profiles[name].latent_j_per_kg
                if profiles[name].latent_j_per_kg > 0.0
                else np.zeros(n_steps, dtype=np.float64)
            )
            for name in source_names
        ]
    )
    source_outlet_temperature_celsius = np.stack(
        [profile_temperature(profiles[name], released_w[name]) for name in source_names]
    )
    sink_names = tuple(sink_by_name)
    sink_outlet_temperature_celsius = np.stack(
        [current_sink_temperature_celsius[name] for name in sink_names]
    )
    fuel_mass_flow_kg_per_s = np.stack(
        [
            np.maximum(
                sink_capacity_rate_w_per_k[name]
                * (
                    sink_by_name[name].setpoint_temperature_celsius
                    - current_sink_temperature_celsius[name]
                ),
                0.0,
            )
            / (config.fuel_lower_heating_value_j_per_kg * config.heater_efficiency)
            for name in sink_names
        ]
    )
    return HeatNetworkResult(
        stage_source_names=tuple(source_name for source_name, _ in stages),
        stage_sink_names=tuple(sink_name for _, sink_name in stages),
        stage_recovered_power_kw=stage_power_kw,
        stage_binding_code=stage_binding_code,
        stage_source_outlet_temperature_celsius=stage_source_outlet_temperature_celsius,
        stage_sink_outlet_temperature_celsius=stage_sink_outlet_temperature_celsius,
        source_names=source_names,
        source_available_power_kw=source_available_power_kw,
        source_used_power_kw=source_used_power_kw,
        source_discarded_power_kw=source_available_power_kw - source_used_power_kw,
        source_condensate_kg_per_s=source_condensate_kg_per_s,
        source_outlet_temperature_celsius=source_outlet_temperature_celsius,
        sink_names=sink_names,
        sink_outlet_temperature_celsius=sink_outlet_temperature_celsius,
        fuel_mass_flow_kg_per_s=fuel_mass_flow_kg_per_s,
        stage_min_approach_k=stage_min_approach_k,
        stage_pinch_position=stage_pinch_position,
    )
