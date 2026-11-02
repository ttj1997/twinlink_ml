r"""Exact steady-state solver for the grey-box model.

Why a module function and not a third ABC implementation
---------------------------------------------------------
``03_code_konventionen.md`` requires ABCs for the four core abstractions, among them
``ProcessSimulator``. A steady-state solver, however, is not a data source -- it
does not belong behind the same interface as ``rollout``. An abstraction of its own for it
would still be conceivable (``SteadyStateSolver``), but with exactly one implementation it is
speculative building: Dyssol can deliver a second case in steady-state flowsheet mode (M7),
without it being clear whether its interface has the same form (Dyssol delivers
no RHS function whose root could be bisected for). The abstraction will be
introduced as soon as the second case comes up -- not before.

Why the nominally four-dimensional problem reduces
--------------------------------------------------
Instead of searching blindly in 4D, the solver uses the structure of the balances from
``greybox.py``:

1. **Solid mass** -- ``_product_mass_flow`` has a feedforward part
   (``solid_inflow``) that makes the control error vanish exactly at steady state:
   :math:`m_s = m_{s,\mathrm{soll}}` (``bed_mass_solid_setpoint_kg``).
2. **Diameter/particle count** -- follow in closed form from the number balance. With
   :math:`\dot m_{prod} = \dot m_{s,in}` (from point 1), :math:`\dot N = 0` reduces
   to

   .. math:: d_{50} = d_{\mathrm{nuc}}\,
       \left(\frac{\dot m_{spray}\,w_{solid} + \dot m_{seed}}
       {b_{\mathrm{nuc}}\,\dot m_{spray}\,w_{solid} + \dot m_{seed}}\right)^{1/3}

   For :math:`\dot m_{seed} = 0` this reduces exactly to
   :meth:`~twinlink_ml.simulation.config.MaterialConfig.nucleation_limited_diameter_m`
   (:math:`d_{50} = d_{\mathrm{nuc}}\,b_{\mathrm{nuc}}^{-1/3}`). At the nominal point of
   ``greybox_placeholder.yaml`` the nucleation from the spray solid is
   :math:`b_{\mathrm{nuc}}\,\dot m_{spray}\,w_{solid} \approx 3.6\,\mathrm{g/h}`
   -- a seed feed of this order of magnitude or above shifts :math:`d_{50}`
   noticeably (at 50 g/h, 14 times that: 750 µm -> 316 µm; at 2 kg/h: 160 µm, below
   the fluidization window). That is not an error source in the solver but the
   model structure -- the feasibility flag catches the case, see below.
3. **Water mass and bed temperature** -- both remaining balances are monotonic at a fixed
   value of the respective other state (:math:`\dot m_w` falls monotonically in :math:`m_w`,
   :math:`\dot T_{bed}` falls monotonically in :math:`T_{bed}`). Two nested
   bisections suffice: outer over :math:`T_{bed}`, inner over the particle moisture
   :math:`X_p = m_w/m_s` at fixed :math:`T_{bed}`. No Newton, no Jacobian -- the
   right-hand side contains ``clip``/``max`` operations (saturation degree, state floor)
   and is not differentiable everywhere.

Both bisections call exclusively
:meth:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator.evaluate_derivative`
-- the same right-hand side as the integrator and ``time_constants``, no second,
deviating physics.

Feasibility is a flag, not an exception
---------------------------------------
Checked are :math:`u_{mf}(d_{50}) < u_{gas} < u_t(d_{50})`, saturation degree
:math:`\le 1` and whether the particle moisture lies in the plausible range
:math:`[X_{eq}, X_{cr}]` (above :math:`X_{cr}` the model signals that
the drying capacity is insufficient for the given spray flow). Hard errors occur
in only two places: batch operation (no steady state defined) and a
violated bracket sign condition (the monotonicity assumption itself is violated -- a
bisection would otherwise silently return the bracket edge, and that would end up unnoticed as a
wrong value in the result).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeVar

import numpy as np

from .backend import ArrayBackend, FloatArray
from .greybox import (
    IDX_BED_TEMPERATURE,
    IDX_PARTICLE_COUNT,
    IDX_SOLID_MASS,
    IDX_WATER_MASS,
    GreyBoxFluidBedGranulator,
)
from .sorption import isotherm_from_material, sorption_safety_grid
from .thermo import (
    latent_heat_j_per_kg,
    relative_humidity_from_absolute_humidity,
    saturation_humidity_kg_per_kg,
    saturation_pressure_pa,
)

_DEFAULT_MAX_ITERATIONS = 100
_DEFAULT_TOLERANCE = 1e-13
_DRY_AIR_GAS_CONSTANT_RATIO = 0.622

BED_TEMPERATURE_SEARCH_MARGIN_K: Final = 10.0
"""Untere Grenze der äußeren Bisektion im ersten Versuch: ``T_amb + 10 K`` (Stand vor Befund 78)."""

BED_TEMPERATURE_FALLBACK_MARGIN_K: Final = 0.5
"""Untere Grenze des Rückfallintervalls ``[T_amb + 0.5, T_amb + 10]`` (Befund 78, Prüfbefund K1).

Das Rückfallintervall wird NUR durchsucht, wenn das Energiebilanz-Residuum an
``T_amb + 10`` und am oberen Rand negativ ist (die Nullstelle liegt unterhalb von
``T_amb + 10``; genau dann scheiterte der Löser vorher mit "kein Vorzeichenwechsel"). Überall,
wo der Löser vorher eine Lösung fand, ist der Rechenweg unverändert und das Ergebnis bitgleich.
Physikalische Begründung der neuen Grenze: die Zuluft trägt die absolute Feuchte der
Umgebung, ``Y_in <= Y_s(T_amb) < Y_s(T_bed)`` für jedes ``T_bed > T_amb`` -- oberhalb der
Umgebungstemperatur gibt es die Kondensationsbranche von ``evaporation_rate_kg_per_s``
(``Y_in > Y_s(T_bed)``) nicht. 0.5 K hält Abstand zu gesättigter Zuluft (``phi -> 1``)."""

BED_TEMPERATURE_BRACKET_RULE: Final = (
    "T_amb+10 K; fallback [T_amb+0.5 K, T_amb+10 K] iff the residual at T_amb+10 K is negative"
)
"""Kennung der Suchgrenze für Provenienzblöcke (Befund 78: in jedes neue Artefakt)."""

LOWER_SEARCH_BOUND_TOLERANCE_K: Final = 1.0e-6
"""Abstand, unterhalb dessen eine Lösung als "auf der unteren Suchgrenze" gemeldet wird."""

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class SteadyStateSolution:
    """Exact steady state of the grey-box model under constant, nominal excitation.

    ``diagnostics`` contains all diagnostic variables of
    :meth:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator.evaluate_derivative`
    at the solution point, taken over unchanged (keys as in ``greybox._rhs``).
    ``dry_air_mass_flow_kg_per_s``, ``outlet_humidity_kg_per_kg`` and
    ``bed_temperature_celsius`` are additionally kept as explicit fields, as are the
    newly derived ``dew_point_celsius``, ``exhaust_sensible_kw`` and
    ``exhaust_latent_kw`` -- together they form the interface to the
    site flowsheet (mass flow / moisture load / temperature per air stream).

    Reference state of ``exhaust_sensible_kw`` and ``exhaust_latent_kw``: **ambient**
    (``T_amb``, liquid water) -- not bed temperature. Both quantities answer "how
    much enthalpy does the exhaust air carry beyond the ambient state", the question of the
    condensing heat exchanger for heat recovery. That is a different
    reference state from ``heat_of_evaporation`` in ``greybox._rhs`` (there: bed temperature,
    the question "how much heat does the bed need for evaporation") -- an
    enthalpy quantity without a stated reference state is not a physical quantity, hence
    named explicitly here instead of being visible only in the code.
    """

    state: FloatArray
    """Zustandsvektor ``[m_s, m_w, N, T_bed]``, Reihenfolge wie ``IDX_*`` in ``greybox.py``."""

    diameter_m: float
    residual_inf_norm: float
    r"""Skaliertes Residuum :math:`\|\dot{\mathbf x}/\mathrm{scale}\|_\infty` am Lösungspunkt."""

    feasible: bool
    infeasibility_reasons: tuple[str, ...]
    diagnostics: dict[str, float]

    dry_air_mass_flow_kg_per_s: float
    outlet_humidity_kg_per_kg: float
    bed_temperature_celsius: float
    dew_point_celsius: float

    exhaust_sensible_kw: float
    """Enthalpiestrom der Abluft oberhalb des Umgebungszustands (T_amb), sensibler Anteil."""

    exhaust_latent_kw: float
    """Enthalpiestrom der Abluft oberhalb des Umgebungszustands (T_amb), latenter Anteil
    -- Verdampfungsenthalpie bei ``T_amb``, nicht bei ``T_bed`` (siehe Klassendocstring)."""

    lower_search_bound_celsius: float = math.nan
    """Unterste Betttemperatur, die die äußere Bisektion durchsuchen konnte (Voreinstellung:
    ``T_amb + 0.5``, siehe :data:`BED_TEMPERATURE_FALLBACK_MARGIN_K`; sonst die untere Grenze
    des übergebenen Brackets)."""

    bed_temperature_fallback_used: bool = False
    """``True``, wenn im Rückfallintervall ``[T_amb + 0.5, T_amb + 10]`` gesucht wurde."""

    drying_regime: bool = True
    """``Y_in < Y_out < Y_s(T_bed)`` am Lösungspunkt (die Luft nimmt Wasser auf und ist
    ungesättigt); ``False`` ist ein Unzulässigkeitsgrund."""

    @property
    def solid_mass_kg(self) -> float:
        return float(self.state[IDX_SOLID_MASS])

    @property
    def water_mass_kg(self) -> float:
        return float(self.state[IDX_WATER_MASS])

    @property
    def particle_count(self) -> float:
        return float(self.state[IDX_PARTICLE_COUNT])

    @property
    def particle_moisture_kg_per_kg(self) -> float:
        return self.water_mass_kg / self.solid_mass_kg

    @property
    def exhaust_dry_air_mass_flow_kg_per_s(self) -> float:
        """Dry-air flow of the exhaust that LEAVES the line.

        With exhaust recirculation the purge, ``(1 - r)`` of the bed flow; without, the bed flow
        ``dry_air_mass_flow_kg_per_s``. ``exhaust_sensible_kw``/``exhaust_latent_kw`` refer to
        this flow.
        """
        return self.diagnostics.get(
            "exhaust_dry_air_mass_flow_kg_per_s", self.dry_air_mass_flow_kg_per_s
        )


def solve_steady_state(
    simulator: GreyBoxFluidBedGranulator,
    *,
    moisture_bracket_kg_per_kg: tuple[float, float] | None = None,
    temperature_bracket_celsius: tuple[float, float] | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> SteadyStateSolution:
    r"""Exact steady state under nominal excitation, see module docstring.

    :param moisture_bracket_kg_per_kg: bracket for the inner bisection (particle moisture).
        Default ``[X_eq, max(10*X_cr, 1.0)]`` -- wide enough, because
        :math:`\dot m_w \to -\infty` for :math:`X_p \to \infty` (the
        product withdrawal term dominates), so a sign change exists in the bracket,
        provided the monotonicity assumption itself holds.
    :param temperature_bracket_celsius: bracket for the outer bisection (bed temperature).
        Default derived from the nominal actions/disturbances (see code).
    :param max_iterations: bisection steps per level (outer temperature, inner
        moisture bisection). WARNING to future callers: the default here
        (:data:`_DEFAULT_MAX_ITERATIONS`, 100) does NOT reproduce Befund 52
        (``scripts/study_operating_point_regret.py``,
        ``artifacts/runs/bb1_bb3_operating_point_regret.json``) bit for bit -- with
        this precision 6 of the 25 grid points reported there fail with a
        bracket sign error in the outer temperature bisection (empirically checked,
        see the ``twinlink_ml.rl.env`` module docstring). Cause: at the lower
        bracket bound (``T_amb + 10``) the true root lies just outside, and whether
        the residual value evaluated there shows numerical noise with a positive or
        negative sign depends on the bisection depth of the INNER
        moisture bisection at exactly this point -- different precisions can give
        different signs at the same bracket edge. Befund 52 itself reports
        its grid points NOT with this default but throughout with the
        SEARCH precision (``DEFAULT_SEARCH_MAX_ITERATIONS``/``DEFAULT_SEARCH_TOLERANCE``, 30
        steps/1e-7 in ``study_operating_point_regret.py``) -- whoever wants to reproduce
        Befund 52 at an arbitrary operating point must pass the same precision explicitly,
        not the default of this function.
    :param tolerance: termination tolerance per bisection level, see ``max_iterations`` for the
        bit-identity warning -- the same restriction applies to :data:`_DEFAULT_TOLERANCE`.
    :raises ValueError: in batch operation -- without product discharge the solid mass grows
        without bound; a steady state does not exist.
    :raises RuntimeError: if one of the two bisections finds no sign change at its
        bracket edges -- the monotonicity assumption is violated for this config,
        not a numerical detail.
    """
    config = simulator.config
    if config.operation.mode == "batch":
        raise ValueError(
            "Kein Beharrungszustand im Batch-Betrieb definiert: ohne Produktaustrag "
            "wächst die Feststoffmasse unbeschränkt (siehe GreyBoxFluidBedGranulator."
            "_product_mass_flow)."
        )

    backend = simulator.backend
    material, geometry, operation = config.material, config.geometry, config.operation

    action = simulator.nominal_action(1)
    disturbance = simulator.nominal_disturbance(1)
    action_arr = backend.asarray(action)
    disturbance_arr = backend.asarray(disturbance)

    inlet_temperature = float(action[0, 0])
    gas_velocity = float(action[0, 1])
    spray_rate = float(action[0, 2]) / 3600.0
    solid_fraction = float(disturbance[0, 2])
    ambient_temperature = float(disturbance[0, 0])

    # -- Step 1: solid mass --------------------------------------------------------
    solid_mass = geometry.bed_mass_solid_setpoint_kg

    # -- Step 2: diameter/particle count from the number balance --------------------
    spray_solid = spray_rate * solid_fraction
    seed_flow = operation.seed_mass_flow_kg_per_h / 3600.0
    solid_inflow = spray_solid + seed_flow
    nucleation_denominator = material.nucleation_solid_fraction * spray_solid + seed_flow
    diameter_m = material.nucleus_diameter_m * (solid_inflow / nucleation_denominator) ** (
        1.0 / 3.0
    )
    bed_particle_mass = material.particle_density_kg_per_m3 * math.pi * diameter_m**3 / 6.0
    particle_count = solid_mass / bed_particle_mass

    # -- Step 3: water mass and bed temperature, nested bisection --------------------
    def evaluate(moisture: float, temperature: float) -> tuple[FloatArray, dict[str, float]]:
        water_mass = moisture * solid_mass
        state_np = np.array(
            [[solid_mass, water_mass, particle_count, temperature]], dtype=np.float64
        )
        derivative, diagnostics = simulator.evaluate_derivative(
            backend.asarray(state_np), action_arr, disturbance_arr
        )
        derivative_np = np.asarray(backend.to_numpy(derivative)[0], dtype=np.float64)
        diagnostics_scalar = {
            key: float(backend.to_numpy(value)[0]) for key, value in diagnostics.items()
        }
        return derivative_np, diagnostics_scalar

    # Lower bracket bound: conservative, state-independent bound (the same
    # logic as greybox._floor's safety-grid fallback), not the exact local
    # X_eq -- before the bisection phi_exhaust at the solution point is not yet known. For
    # ConstantIsotherm this is exactly equilibrium_moisture_kg_per_kg (the isotherm
    # ignores phi/T), no change in behaviour compared with before.
    isotherm = isotherm_from_material(material)
    phi_min, _phi_max, _t_min, t_max = sorption_safety_grid(
        backend, config.gas, config.actions, config.disturbances
    )
    moisture_safety_lower_bound = float(
        backend.to_numpy(
            isotherm.equilibrium_moisture(backend, backend.asarray(phi_min), backend.asarray(t_max))
        )
    )
    moisture_low, moisture_high = moisture_bracket_kg_per_kg or (
        moisture_safety_lower_bound,
        max(10.0 * material.critical_moisture_kg_per_kg, 1.0),
    )

    def solve_moisture_for_temperature(
        temperature: float,
    ) -> tuple[float, FloatArray, dict[str, float]]:
        def water_residual(moisture: float) -> tuple[float, tuple[FloatArray, dict[str, float]]]:
            derivative_np, diagnostics_scalar = evaluate(moisture, temperature)
            return float(derivative_np[IDX_WATER_MASS]), (derivative_np, diagnostics_scalar)

        moisture, (derivative_np, diagnostics_scalar) = _bisect_with_payload(
            water_residual,
            moisture_low,
            moisture_high,
            max_iterations=max_iterations,
            tolerance=tolerance,
            description=(
                f"Wasserbilanz bei T_bed={temperature:.4f} °C (dm_w/dt fällt monoton in m_w)"
            ),
        )
        return moisture, derivative_np, diagnostics_scalar

    # Lower bound (Befund 78, review finding K1). First attempt as before from ``T_amb + 10``.
    # If the root lies below (residual at ``T_amb + 10`` and at the upper edge negative --
    # the case in which the solver previously failed with "no sign change"), the
    # fallback interval ``[T_amb + 0.5, T_amb + 10]`` is searched. The earlier reasoning for the
    # 10 K margin (condensation branch near ``T_amb``) holds only BELOW the ambient
    # temperature, see :data:`BED_TEMPERATURE_FALLBACK_MARGIN_K`. A passed
    # ``temperature_bracket_celsius`` is used unchanged, without fallback.
    default_bracket = temperature_bracket_celsius is None
    temperature_low, temperature_high = temperature_bracket_celsius or (
        ambient_temperature + BED_TEMPERATURE_SEARCH_MARGIN_K,
        max(inlet_temperature, operation.suspension_temperature_celsius) + 5.0,
    )
    lower_search_bound = (
        ambient_temperature + BED_TEMPERATURE_FALLBACK_MARGIN_K
        if default_bracket
        else temperature_low
    )

    def temperature_residual(
        temperature: float,
    ) -> tuple[float, tuple[float, FloatArray, dict[str, float]]]:
        moisture, derivative_np, diagnostics_scalar = solve_moisture_for_temperature(temperature)
        return float(derivative_np[IDX_BED_TEMPERATURE]), (
            moisture,
            derivative_np,
            diagnostics_scalar,
        )

    low_value = high_value = None
    fallback_used = False
    if default_bracket:
        # The same evaluations in the same order as in ``_bisect_with_payload`` --
        # they are passed through to it; the computation path without fallback stays bit-identical.
        low_value = temperature_residual(temperature_low)
        high_value = temperature_residual(temperature_high)
        if low_value[0] < 0.0 and high_value[0] < 0.0:
            fallback_used = True
            temperature_high, high_value = temperature_low, low_value
            temperature_low, low_value = lower_search_bound, None
    temperature, (moisture, derivative_np, diagnostics_scalar) = _bisect_with_payload(
        temperature_residual,
        temperature_low,
        temperature_high,
        max_iterations=max_iterations,
        tolerance=tolerance,
        description=(
            "Energiebilanz (dT_bed/dt fällt monoton in T_bed)"
            + (", Rückfallintervall ab T_amb + 0.5 K" if fallback_used else "")
        ),
        low_value=low_value,
        high_value=high_value,
    )

    water_mass = moisture * solid_mass
    state = np.array([solid_mass, water_mass, particle_count, temperature], dtype=np.float64)

    # -- Scaled residual -------------------------------------------------------------
    scale = np.array(
        [
            max(abs(solid_mass), 1e-9),
            max(abs(water_mass), 1e-9),
            max(abs(particle_count), 1e-9),
            max(abs(temperature), 1.0),
        ]
    )
    residual_inf_norm = float(np.max(np.abs(derivative_np / scale)))

    # -- Derived fields --------------------------------------------------------------
    # Reference state: ambient (T_amb, liquid water) -- see class docstring of
    # SteadyStateSolution. Each term uses the humidity of its own stream: the
    # exhaust air carries Y_out, hence weighted there -- not Y_in as for the
    # heater power in energy.py, whose air stream has not yet evaporated anything.
    gas = config.gas
    dry_air_flow = diagnostics_scalar["dry_air_mass_flow_kg_per_s"]
    outlet_humidity = diagnostics_scalar["outlet_humidity_kg_per_kg"]
    inlet_humidity = diagnostics_scalar["inlet_humidity_kg_per_kg"]
    # Exhaust recirculation: only the purge (1 - r) of the bed flow leaves the line, and its
    # latent part is counted against the FRESH air humidity (the recirculated water stays in the
    # loop). Without recirculation both names are the old quantities (same floats).
    exhaust_flow = diagnostics_scalar.get("exhaust_dry_air_mass_flow_kg_per_s", dry_air_flow)
    fresh_air_humidity = diagnostics_scalar.get("fresh_air_humidity_kg_per_kg", inlet_humidity)

    exhaust_sensible_kw = (
        exhaust_flow
        * (gas.cp_dry_air_j_per_kg_k + outlet_humidity * gas.cp_vapour_j_per_kg_k)
        * (temperature - ambient_temperature)
        / 1000.0
    )
    latent_heat_at_ambient = float(
        backend.to_numpy(
            latent_heat_j_per_kg(backend, backend.asarray(np.array([ambient_temperature])))
        )[0]
    )
    exhaust_latent_kw = (
        exhaust_flow * (outlet_humidity - fresh_air_humidity) * latent_heat_at_ambient / 1000.0
    )

    dew_point_celsius = _solve_dew_point(
        backend=backend,
        outlet_humidity_kg_per_kg=outlet_humidity,
        pressure_pa=gas.pressure_pa,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    reasons: list[str] = []
    u_mf = diagnostics_scalar["minimum_fluidization_velocity_m_per_s"]
    u_terminal = diagnostics_scalar["terminal_velocity_m_per_s"]
    if not gas_velocity > u_mf:
        reasons.append(f"u_gas ({gas_velocity:.4f} m/s) <= u_mf(d50) ({u_mf:.4f} m/s)")
    if not gas_velocity < u_terminal:
        reasons.append(f"u_gas ({gas_velocity:.4f} m/s) >= u_t(d50) ({u_terminal:.4f} m/s)")
    saturation_degree = diagnostics_scalar["saturation_degree"]
    saturation_humidity_at_bed = float(
        backend.to_numpy(
            saturation_humidity_kg_per_kg(backend, gas, backend.asarray(np.array([temperature])))
        )[0]
    )
    drying_regime = inlet_humidity < outlet_humidity < saturation_humidity_at_bed
    if not drying_regime:
        reasons.append(
            f"kein Trocknungsregime: Y_in={inlet_humidity:.6f}, Y_out={outlet_humidity:.6f}, "
            f"Y_s(T_bed)={saturation_humidity_at_bed:.6f} kg/kg"
        )
    if temperature - lower_search_bound <= LOWER_SEARCH_BOUND_TOLERANCE_K:
        reasons.append(
            f"Betttemperatur ({temperature:.6f} °C) auf der unteren Suchgrenze "
            f"({lower_search_bound:.6f} °C)"
        )
    if not saturation_degree <= 1.0 + 1e-9:
        reasons.append(f"Sättigungsgrad ({saturation_degree:.4f}) > 1")
    # At the steady-state point phi_exhaust is fixed -- unlike the bracket bound above
    # (before the solution), here the exact, state-dependent isotherm can be used at no
    # extra cost; no conservative constant needed.
    relative_humidity_at_solution = relative_humidity_from_absolute_humidity(
        backend, gas, backend.asarray(outlet_humidity), backend.asarray(temperature)
    )
    local_equilibrium_moisture = float(
        backend.to_numpy(
            isotherm.equilibrium_moisture(
                backend, relative_humidity_at_solution, backend.asarray(temperature)
            )
        )
    )
    if not (
        local_equilibrium_moisture - 1e-9 <= moisture <= material.critical_moisture_kg_per_kg + 1e-9
    ):
        reasons.append(
            f"Partikelfeuchte ({moisture:.5f} kg/kg) außerhalb des plausiblen Bereichs "
            f"[{local_equilibrium_moisture:.5f}, "
            f"{material.critical_moisture_kg_per_kg}]"
        )

    return SteadyStateSolution(
        state=state,
        diameter_m=diameter_m,
        residual_inf_norm=residual_inf_norm,
        feasible=not reasons,
        infeasibility_reasons=tuple(reasons),
        diagnostics=diagnostics_scalar,
        dry_air_mass_flow_kg_per_s=dry_air_flow,
        outlet_humidity_kg_per_kg=outlet_humidity,
        bed_temperature_celsius=temperature,
        dew_point_celsius=dew_point_celsius,
        exhaust_sensible_kw=exhaust_sensible_kw,
        exhaust_latent_kw=exhaust_latent_kw,
        lower_search_bound_celsius=lower_search_bound,
        bed_temperature_fallback_used=fallback_used,
        drying_regime=drying_regime,
    )


def _bisect_with_payload(
    residual_and_payload: Callable[[float], tuple[float, _T]],
    low: float,
    high: float,
    *,
    max_iterations: int,
    tolerance: float,
    description: str,
    low_value: tuple[float, _T] | None = None,
    high_value: tuple[float, _T] | None = None,
) -> tuple[float, _T]:
    """Bisection with payload (e.g. derived quantities at the respective point).

    Checks the sign change at the bracket edges **before** the iteration --
    different signs are a precondition, not a side effect. Without this check
    a bisection with equal signs silently converges to one of the two
    bracket edges without this becoming visible as an error.
    """
    f_low, payload_low = low_value if low_value is not None else residual_and_payload(low)
    f_high, payload_high = high_value if high_value is not None else residual_and_payload(high)
    if f_low == 0.0:
        return low, payload_low
    if f_high == 0.0:
        return high, payload_high
    if f_low * f_high > 0.0:
        raise RuntimeError(
            f"{description}: kein Vorzeichenwechsel im Bracket [{low}, {high}] "
            f"(f({low})={f_low:.6e}, f({high})={f_high:.6e}) -- die Monotonieannahme "
            "ist für diese Konfiguration verletzt."
        )
    a, b, fa = low, high, f_low
    mid, payload = low, payload_low
    for _ in range(max_iterations):
        mid = 0.5 * (a + b)
        f_mid, payload = residual_and_payload(mid)
        if f_mid == 0.0 or (b - a) < tolerance * max(abs(a), abs(b), 1.0):
            break
        if (fa < 0.0) == (f_mid < 0.0):
            a, fa = mid, f_mid
        else:
            b = mid
    return mid, payload


def _solve_dew_point(
    *,
    backend: ArrayBackend,
    outlet_humidity_kg_per_kg: float,
    pressure_pa: float,
    max_iterations: int,
    tolerance: float,
) -> float:
    r"""Dew point of the exhaust air via the same saturation relation as ``thermo.py``.

    :math:`p_{partial} = Y\,p / (0.622 + Y)` (inversion of
    :func:`~twinlink_ml.simulation.thermo.absolute_humidity_kg_per_kg`), then
    :math:`T_{dp}` as the root of :math:`p_s(T) - p_{partial}` via
    :func:`~twinlink_ml.simulation.thermo.saturation_pressure_pa` -- the same
    Buck correlation that ``_rhs`` uses for ``saturation_humidity_kg_per_kg``. The
    Buck formula cannot be inverted for ``T`` in closed form (transcendental), hence
    numerically via bisection; ``saturation_pressure_pa`` is strictly monotonic in ``T`` in the
    valid range, so solvable with the same bisection logic as the main solution.
    """
    p_partial = (
        outlet_humidity_kg_per_kg
        * pressure_pa
        / (_DRY_AIR_GAS_CONSTANT_RATIO + outlet_humidity_kg_per_kg)
    )

    def residual(temperature: float) -> tuple[float, None]:
        value = float(
            backend.to_numpy(
                saturation_pressure_pa(backend, backend.asarray(np.array([temperature])))
            )[0]
        )
        return value - p_partial, None

    root, _ = _bisect_with_payload(
        residual,
        -40.0,
        100.0,
        max_iterations=max_iterations,
        tolerance=tolerance,
        description="Taupunktbestimmung",
    )
    return root
