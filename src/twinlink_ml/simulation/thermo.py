r"""Psychrometrics and drying kinetics.

Pure functions, backend-agnostic. Every quantity carries its unit in its name.

The saturation humidity is the place where the hard limit from
``02_prozess_setup.md`` (exhaust air humidity :math:`\le` saturation humidity) arises. It is
**not** enforced here by a downstream projection but by the
structure of the drying kinetics: the saturation degree :math:`\nu \le \nu_{max} \le 1`
limits the evaporation such that the exhaust air can never exceed saturation.
A constraint that follows from the model structure cannot be violated by numerical
inaccuracy -- a projection can.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backend import Array, ArrayBackend
from .config import GasConfig, MaterialConfig

if TYPE_CHECKING:
    # Only for type annotations: sorption.py already imports from thermo.py
    # (absolute_humidity_kg_per_kg, relative_humidity_from_absolute_humidity). A
    # module top-level import here would create a circular reference. The isotherm is
    # therefore injected (dependency injection, like ``backend``), not constructed internally
    # from ``material`` -- no runtime import needed.
    from .sorption import SorptionIsotherm

_KELVIN_OFFSET = 273.15
_PICARD_ITERATIONS = 5
_PICARD_DAMPING = 0.7


def kelvin(temperature_celsius: Array) -> Array:
    """Celsius to Kelvin."""
    return temperature_celsius + _KELVIN_OFFSET


def saturation_pressure_pa(backend: ArrayBackend, temperature_celsius: Array) -> Array:
    r"""Saturation vapour pressure according to Buck (1996).

    .. math:: p_s(T) = 611.21\,
        \exp\!\left[\left(18.678 - \frac{T}{234.5}\right)\frac{T}{257.14 + T}\right]

    Accuracy better than 0.2 % in the range 0-100 °C.

    Deliberately **not** the simpler Magnus formula: it deviates at 100 °C by 2.6 kPa
    (2.6 %). Because inlet air temperatures up to 120 °C lie in the action range and the
    bed temperature runs into this region at a small spray rate, this would be a
    systematic error exactly in the energetically interesting edge region -- i.e. where
    an RL agent will later optimise towards.

    :param temperature_celsius: temperature in °C
    :returns: saturation vapour pressure in Pa
    """
    t = backend.clip(temperature_celsius, -40.0, 200.0)
    return 611.21 * backend.exp((18.678 - t / 234.5) * t / (257.14 + t))


def absolute_humidity_kg_per_kg(
    backend: ArrayBackend,
    gas: GasConfig,
    temperature_celsius: Array,
    relative_humidity: Array,
) -> Array:
    r"""Absolute humidity from relative humidity.

    .. math:: Y = 0.622\,\frac{\varphi\, p_s(T)}{p - \varphi\, p_s(T)}

    Unit: kg water per kg dry air.
    """
    p_partial = relative_humidity * saturation_pressure_pa(backend, temperature_celsius)
    p_partial = backend.minimum(p_partial, 0.95 * gas.pressure_pa)
    return 0.622 * p_partial / (gas.pressure_pa - p_partial)


def relative_humidity_from_absolute_humidity(
    backend: ArrayBackend,
    gas: GasConfig,
    humidity_kg_per_kg: Array,
    temperature_celsius: Array,
    *,
    saturation_pressure: Array | None = None,
) -> Array:
    r"""Relative humidity from absolute humidity -- inverse of :func:`absolute_humidity_kg_per_kg`.

    .. math:: \varphi = \mathrm{clip}\!\left(\frac{Y\,p}{(0.622+Y)\,p_s(T)},\, 0,\, 1\right)

    The same Buck saturation pressure formula as :func:`saturation_pressure_pa` -- no second
    psychrometric formula. ``temperature_celsius`` here is not necessarily the
    temperature at which ``humidity_kg_per_kg`` was measured: the function answers
    "which relative humidity would this absolute humidity have at this temperature", e.g.
    for inlet air that is heated from :math:`T_{amb}` to :math:`T_{bed}` without its
    absolute humidity changing.

    Exact algebraic inversion of :func:`absolute_humidity_kg_per_kg`: numerically
    confirmed up to float64 rounding (< 1e-15). One exception: at very high
    ``temperature_celsius`` combined with high ``humidity_kg_per_kg``,
    :func:`absolute_humidity_kg_per_kg`'s own clip already present there takes effect
    (``p_partial = min(p_partial, 0.95*p)``, saturation vapour pressure exceeds the
    total pressure from about 95-100 °C) -- the ``Y`` already clipped at that point is then
    no longer the exact inversion of the original relative humidity, regardless of
    how exactly this function itself computes. Whoever uses this function at temperatures clearly
    above the safety grid (:func:`~.sorption.sorption_safety_grid`, evaluated here only
    up to 130 °C) should take this effect into account.

    :returns: relative humidity in [0, 1]
    """
    p_partial = humidity_kg_per_kg * gas.pressure_pa / (0.622 + humidity_kg_per_kg)
    # ``saturation_pressure``: p_s(temperature_celsius) precomputed by a caller that evaluates
    # many humidities at one temperature (recirculated_inlet_humidity); same value, same formula.
    p_saturation = (
        saturation_pressure_pa(backend, temperature_celsius)
        if saturation_pressure is None
        else saturation_pressure
    )
    return backend.clip(p_partial / p_saturation, 0.0, 1.0)


def dew_point_celsius(
    backend: ArrayBackend,
    gas: GasConfig,
    absolute_humidity_kg_per_kg: Array,
    *,
    iterations: int = 60,
) -> Array:
    r"""Dew point from absolute humidity -- inversion of :func:`saturation_pressure_pa`.

    :math:`p_s(T_{dp}) = p_{partial}`, with :math:`p_{partial} = Y\,p / (0.622 + Y)`
    (inversion of :func:`absolute_humidity_kg_per_kg`, the same formula as in
    :func:`relative_humidity_from_absolute_humidity`). The same Buck correlation
    (:func:`saturation_pressure_pa`) as everywhere in this module -- no second
    psychrometric formula. Numerically via bisection, not Newton: ``saturation_pressure_pa``
    clips at the bracket edges, not an everywhere differentiable function (the same
    reasoning as for ``simulation.steady_state._solve_dew_point``, the scalar, private
    variant for the steady-state solver). This function is the array-capable,
    public counterpart -- needed because an evaluation runs on whole
    prediction arrays (many windows/runs/horizons), not at a single
    steady-state point.

    60 fixed iterations, no termination check: with the start bracket ``[-40, 100]`` °C the
    interval halves to :math:`140 / 2^{60}` °C -- far below any
    float64 resolution, without conditional control flow across array elements (no early
    termination at convergence of individual elements).

    :param absolute_humidity_kg_per_kg: :math:`Y`, kg water per kg dry air.
    :returns: dew point in °C.
    """
    p_partial = (
        absolute_humidity_kg_per_kg * gas.pressure_pa / (0.622 + absolute_humidity_kg_per_kg)
    )
    low = backend.zeros_like(p_partial) - 40.0
    high = backend.zeros_like(p_partial) + 100.0
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        residual = saturation_pressure_pa(backend, mid) - p_partial
        too_low = residual < 0.0
        low = backend.where(too_low, mid, low)
        high = backend.where(too_low, high, mid)
    return 0.5 * (low + high)


def saturation_humidity_kg_per_kg(
    backend: ArrayBackend, gas: GasConfig, temperature_celsius: Array
) -> Array:
    """Saturation humidity :math:`Y_s(T)` in kg/kg dry air."""
    one = backend.zeros_like(temperature_celsius) + 1.0
    return absolute_humidity_kg_per_kg(backend, gas, temperature_celsius, one)


def latent_heat_j_per_kg(backend: ArrayBackend, temperature_celsius: Array) -> Array:
    r"""Enthalpy of evaporation of water.

    .. math:: \Delta h_v(T) \approx 2.5008\cdot 10^{6} - 2358.6\,T \quad [\mathrm{J/kg}]
    """
    t = backend.clip(temperature_celsius, 0.0, 150.0)
    return 2.5008e6 - 2358.6 * t


def gas_density_kg_per_m3(
    backend: ArrayBackend,  # noqa: ARG001 - common signature of all correlations
    gas: GasConfig,
    temperature_celsius: Array,
    humidity_kg_per_kg: Array,
) -> Array:
    r"""Density of moist air, ideal gas mixture.

    .. math:: \rho = \frac{p}{R_{L} T}\cdot\frac{1 + Y}{1 + Y/0.622}
    """
    t_kelvin = kelvin(temperature_celsius)
    dry = gas.pressure_pa / (gas.r_specific_dry_air_j_per_kg_k * t_kelvin)
    return dry * (1.0 + humidity_kg_per_kg) / (1.0 + humidity_kg_per_kg / 0.622)


def gas_viscosity_pa_s(backend: ArrayBackend, gas: GasConfig, temperature_celsius: Array) -> Array:
    r"""Dynamic viscosity according to Sutherland.

    .. math:: \mu(T) = \mu_0
        \left(\frac{T}{T_0}\right)^{3/2}\frac{T_0 + S}{T + S}
    """
    t_kelvin = kelvin(temperature_celsius)
    ratio = t_kelvin / gas.viscosity_ref_temperature_k
    return (
        gas.viscosity_ref_pa_s
        * ratio
        * backend.sqrt(ratio)
        * (gas.viscosity_ref_temperature_k + gas.sutherland_constant_k)
        / (t_kelvin + gas.sutherland_constant_k)
    )


def normalised_moisture(
    backend: ArrayBackend,
    material: MaterialConfig,
    isotherm: SorptionIsotherm,
    moisture_kg_per_kg: Array,
    relative_humidity: Array,
    temperature_celsius: Array,
) -> Array:
    r"""Normalised particle moisture :math:`\eta` of the drying curve.

    .. math:: \eta = \mathrm{clip}\!\left(
        \frac{X_p - X_{eq}(\varphi, T)}{X_{cr} - X_{eq}(\varphi, T)},\, 0,\, 1\right)

    :math:`\eta = 1` corresponds to the first drying period (constant rate),
    :math:`\eta < 1` to the second (falling rate). The zero crossing at
    :math:`X_p = X_{eq}` structurally guarantees the second hard limit from
    ``02_prozess_setup.md``: the particle moisture cannot fall below the equilibrium moisture of the
    sorption isotherm, because the evaporation rate vanishes there.

    :math:`X_{eq}(\varphi, T)` comes from the injected sorption isotherm
    (``simulation.sorption``) -- ``ConstantIsotherm`` reproduces the historical
    scalar case exactly, ``GabIsotherm`` makes :math:`X_{eq}` dependent on the relative
    exhaust air humidity and the bed temperature. Which isotherm is active is
    up to the caller (dependency injection, like ``backend``) -- this function
    knows only the isotherm interface, not ``MaterialConfig.sorption_isotherm_kind``.

    :param relative_humidity: relative humidity of the exhaust air in [0, 1] -- in
        ``evaporation_rate_kg_per_s`` the result of the Picard iteration over
        :math:`Y_{out}`, not given directly (see there).
    """
    equilibrium_moisture = isotherm.equilibrium_moisture(
        backend, relative_humidity, temperature_celsius
    )
    span = material.critical_moisture_kg_per_kg - equilibrium_moisture
    excess = (moisture_kg_per_kg - equilibrium_moisture) / span
    return backend.clip(excess, 0.0, 1.0)


def evaporation_rate_kg_per_s(
    backend: ArrayBackend,
    gas: GasConfig,
    material: MaterialConfig,
    isotherm: SorptionIsotherm,
    dry_air_mass_flow_kg_per_s: Array,
    bed_temperature_celsius: Array,
    inlet_humidity_kg_per_kg: Array,
    particle_moisture_kg_per_kg: Array,
) -> tuple[Array, Array, Array]:
    r"""Evaporation rate via the normalised drying curve.

    .. math::
        \dot m_{evap} = \underbrace{\nu_{max}\,f(\eta)}_{\nu}\;
        \dot m_{L}\,\bigl(Y_s(T_{bed}) - Y_{in}\bigr),
        \qquad f(\eta) = \eta

    Assumption: ideally mixed bed, exhaust air at bed temperature. The saturation degree
    :math:`\nu` describes how close the exhaust air comes to saturation. For
    fluidized beds it is typically 0.4-0.8 in the first drying period.

    Pitfall: with :math:`f(\eta) = \eta` the moisture dynamics are a
    first-order lag element with a very small time constant, because the
    drying capacity of the air exceeds the sprayed-in water amount under
    granulation conditions by an order of magnitude. That is physically
    correct (the bed is dry), but makes the moisture the fastest mode of the
    system -- relevant for the choice of :math:`\Delta t`.

    Condensation
    ------------
    If the inlet air is moister than saturation at bed temperature
    (:math:`Y_{in} > Y_s(T_{bed})`, e.g. at a cold start or with very humid
    ambient air), the mass transfer reverses: water condenses onto the particles
    until the exhaust air is saturated. This branch is not kinetically limited.

    It is not mere completeness: **without** it the hard limit
    "exhaust air humidity :math:`\le` saturation humidity" holds only in the drying regime. If the
    branch is missing, the model silently delivers supersaturated exhaust air for a cold bed -- i.e.
    exactly at the operating point an untrained RL agent runs into first.

    Damped Picard iteration for :math:`Y_{out}`
    -------------------------------------------
    With a state-dependent isotherm an algebraic loop arises:
    :math:`\eta` (and thus the evaporation) depends via :math:`X_{eq}(\varphi_{Abluft}, T)`
    on :math:`\varphi_{Abluft}`, which in turn needs :math:`Y_{out}` -- the
    return value of this function. :math:`Y_{in}` is the natural zeroth iterate
    (no circular reference, immediately available). Undamped fixed-point iteration demonstrably
    oscillates near the kink :math:`X_p \approx X_{eq}` (there :math:`\eta` is
    extremely sensitive to small changes of :math:`X_{eq}`, see
    ``docs/04_stand_ap1.md`` Befund 34) -- under-relaxation with :math:`\alpha = 0{,}7` over
    :math:`n = 5` iterations keeps the error between the second-to-last and last
    iterate below 1 % relative at all checked operating points (nominal point, two Befund 31
    edge points, kink scenario), for both the NumPy and the Torch backend
    (bit-identical, both float64).

    **Computational cost, named explicitly instead of concealed:** :math:`n=5` iterations of
    saturation humidity and isotherm evaluation each mean 5 instead of 1 additional
    backend evaluation of this function per RHS call -- with RK4 (4 RHS calls per
    integration step) thus 20 instead of 4, i.e. a fivefold increase of the cost of this
    one sub-calculation. For ``ConstantIsotherm`` this is pure waste (the result
    is exact after the first iteration), but a case distinction by isotherm type
    would branch the function and was deliberately not introduced -- the extra cost is
    small compared with the other RHS terms (fluidization, energy).

    :returns: (evaporation rate kg/s, exhaust air humidity kg/kg, saturation degree -)
    """
    y_saturation = saturation_humidity_kg_per_kg(backend, gas, bed_temperature_celsius)
    surplus = y_saturation - inlet_humidity_kg_per_kg

    # Upper bound of the driving force: above ~80 °C bed temperature Y_s diverges and the
    # drying is in reality no longer saturation-limited but transport-limited. The
    # normalised drying curve loses its validity there -- the limit keeps the
    # model in a plausible range and marks the model boundary explicitly.
    y_effective = backend.minimum(y_saturation, material.max_driving_humidity_kg_per_kg)

    outlet_humidity = inlet_humidity_kg_per_kg  # zeroth iterate
    evaporation = backend.zeros_like(bed_temperature_celsius)
    saturation_degree = backend.zeros_like(bed_temperature_celsius)
    for _ in range(_PICARD_ITERATIONS):
        relative_humidity = relative_humidity_from_absolute_humidity(
            backend, gas, outlet_humidity, bed_temperature_celsius
        )
        eta = normalised_moisture(
            backend,
            material,
            isotherm,
            particle_moisture_kg_per_kg,
            relative_humidity,
            bed_temperature_celsius,
        )
        if material.drying_curve_exponent is not None:
            # Optional curve shape f(eta) = eta ** q; unset leaves the linear curve (and the
            # arithmetic) exactly as before.
            eta = eta**material.drying_curve_exponent
        saturation_degree = material.max_saturation_degree * eta
        drying = (
            saturation_degree
            * dry_air_mass_flow_kg_per_s
            * backend.maximum(y_effective - inlet_humidity_kg_per_kg, 0.0)
        )
        condensing = dry_air_mass_flow_kg_per_s * surplus
        evaporation = backend.where(surplus > 0.0, drying, condensing)
        outlet_humidity_proposed = (
            inlet_humidity_kg_per_kg + evaporation / dry_air_mass_flow_kg_per_s
        )
        # Damping only in the drying branch: there the fixed point depends via eta(phi(Y_out))
        # on the previous Y_out (the kink that causes the oscillation).
        # In the condensing branch outlet_humidity_proposed = Y_s(T_bed) exactly, for
        # every iteration, independent of the state -- a constant, not an iterative
        # fixed point. Damping it would satisfy the hard limit "exhaust air humidity <= Y_s"
        # (02_prozess_setup.md) only approximately instead of exactly (demonstrated:
        # test_outlet_humidity_never_exceeds_saturation failed with residuals up to 4e-5
        # before this branch was exempted from the damping) -- undamped
        # it stays exact, because the condensing branch has no risk of oscillation at all.
        outlet_humidity_damped = (
            _PICARD_DAMPING * outlet_humidity_proposed + (1.0 - _PICARD_DAMPING) * outlet_humidity
        )
        outlet_humidity = backend.where(
            surplus > 0.0, outlet_humidity_damped, outlet_humidity_proposed
        )

    return evaporation, outlet_humidity, saturation_degree


_RECIRCULATION_ITERATIONS = 15
"""Fixed step count of the bracketed root search in :func:`recirculated_inlet_humidity`.
Checked in ``tests/test_exhaust_recirculation.py`` against 200 steps at the corners of the study
box (r up to 0.9, GAB isotherm)."""


def recirculated_inlet_humidity(
    backend: ArrayBackend,
    gas: GasConfig,
    material: MaterialConfig,
    isotherm: SorptionIsotherm,
    ambient_humidity_kg_per_kg: Array,
    recirculation_fraction: float,
    bed_temperature_celsius: Array,
    particle_moisture_kg_per_kg: Array,
) -> tuple[Array, Array, Array]:
    r"""Inlet and outlet humidity of a bed whose inlet air is partly recirculated exhaust.

    A fraction :math:`r` of the dry air through the bed is taken from the exhaust (state
    :math:`T_{bed}, Y_{out}`, well-mixed bed) and mixed with fresh air (:math:`Y_a`) ahead of the
    heater. Gas phase quasi-stationary as everywhere in the model, so the loop is algebraic:

    .. math::
        Y_{in} = (1 - r)\,Y_a + r\,Y_{out}, \qquad
        Y_{out} = Y_{in} + \nu\,(Y_{eff} - Y_{in})

    with the saturation degree :math:`\nu = \nu_{max} f(\eta)` of
    :func:`evaporation_rate_kg_per_s` (same drying curve, same isotherm, same driving-force
    cap :math:`Y_{eff} = \min(Y_s(T_{bed}), Y_{max})`). For a given :math:`\nu` the pair is linear
    and solved in closed form,

    .. math:: Y_{in} = \frac{(1 - r)\,Y_a + r\,\nu\,Y_{eff}}{1 - r + r\,\nu},

    so the only fixed point left is the dependence of :math:`\nu` on :math:`\varphi_{out}` through
    the isotherm (as in the fresh-air case). A damped Picard iteration as in
    :func:`evaporation_rate_kg_per_s` does NOT converge here near the kink of the drying curve
    (measured: relative errors above 1 at r = 0.8 after 60 steps; the recirculation multiplies the
    loop gain); the fixed point is therefore found by a bracketed root search on the monotone
    residual (see the code), with a fixed step count. The returned pair is evaluated once more at
    the final :math:`\varphi_{out}`, so the two relations above hold exactly; with them the water
    balance of the whole loop closes exactly,
    :math:`(1 - r)\,(Y_{out} - Y_a) = \nu\,(Y_{eff} - Y_{in}) = \dot m_{evap} / \dot m_L`
    (the water leaves only with the purge, :math:`(1 - r)\,\dot m_L`). The dry-air flow does not
    enter: the humidity rise per kg of dry air is independent of it.

    Condensing branch (:math:`Y_a \ge Y_s(T_{bed})`, fresh air wetter than saturation at bed
    temperature, as in :func:`evaporation_rate_kg_per_s`): :math:`Y_{out} = Y_s(T_{bed})`,
    :math:`Y_{in} = (1 - r)\,Y_a + r\,Y_s`. :math:`r = 0` reproduces :math:`Y_{in} = Y_a`
    exactly in both branches.

    :param recirculation_fraction: :math:`r \in [0, 0.95]` (``OperationConfig``).
    :returns: (inlet humidity :math:`Y_{in}`, outlet humidity :math:`Y_{out}`, saturation degree
        :math:`\nu`), kg/kg dry air and -.
    """
    r = recirculation_fraction
    y_saturation = saturation_humidity_kg_per_kg(backend, gas, bed_temperature_celsius)
    y_effective = backend.minimum(y_saturation, material.max_driving_humidity_kg_per_kg)
    drying_branch = y_saturation > ambient_humidity_kg_per_kg
    driving = y_effective > ambient_humidity_kg_per_kg
    p_saturation = saturation_pressure_pa(backend, bed_temperature_celsius)

    def closed_form(outlet: Array) -> tuple[Array, Array, Array]:
        relative_humidity = relative_humidity_from_absolute_humidity(
            backend, gas, outlet, bed_temperature_celsius, saturation_pressure=p_saturation
        )
        eta = normalised_moisture(
            backend,
            material,
            isotherm,
            particle_moisture_kg_per_kg,
            relative_humidity,
            bed_temperature_celsius,
        )
        if material.drying_curve_exponent is not None:
            eta = eta**material.drying_curve_exponent
        nu = backend.where(driving, material.max_saturation_degree * eta, backend.zeros_like(eta))
        inlet = ((1.0 - r) * ambient_humidity_kg_per_kg + r * nu * y_effective) / (1.0 - r + r * nu)
        return inlet, inlet + nu * (y_effective - inlet), nu

    # g(y) = closed_form(y).outlet is non-increasing in y (phi up -> X_eq up -> eta down -> nu down
    # -> Y_out = Y_a + nu (Y_eff - Y_a) / (1 - r + r nu) down), so h(y) = y - g(y) is increasing
    # with h(Y_a) <= 0 <= h(Y_eff): one root, bracketed. False position with every third step a
    # bisection (the bracket shrinks at least by 2**-(n/3) even where eta has its kink).
    low = ambient_humidity_kg_per_kg
    high = backend.maximum(y_effective, low)
    h_low = low - closed_form(low)[1]
    h_high = high - closed_form(high)[1]

    def false_position(low: Array, high: Array, h_low: Array, h_high: Array) -> Array:
        span = h_high - h_low
        safe_span = backend.where(span > 0.0, span, backend.zeros_like(span) + 1.0)
        return backend.where(
            span > 0.0, (low * h_high - high * h_low) / safe_span, 0.5 * (low + high)
        )

    for step in range(_RECIRCULATION_ITERATIONS):
        trial = 0.5 * (low + high) if step % 3 == 2 else false_position(low, high, h_low, h_high)
        h_trial = trial - closed_form(trial)[1]
        below = h_trial < 0.0
        low = backend.where(below, trial, low)
        h_low = backend.where(below, h_trial, h_low)
        high = backend.where(below, high, trial)
        h_high = backend.where(below, h_high, h_trial)
    inlet, outlet, nu = closed_form(false_position(low, high, h_low, h_high))

    inlet_condensing = (1.0 - r) * ambient_humidity_kg_per_kg + r * y_saturation
    return (
        backend.where(drying_branch, inlet, inlet_condensing),
        backend.where(drying_branch, outlet, y_saturation),
        nu,
    )


def mixing_temperature_celsius(
    gas: GasConfig,
    recirculation_fraction: float,
    ambient_temperature_celsius: Array,
    ambient_humidity_kg_per_kg: Array,
    exhaust_temperature_celsius: Array,
    exhaust_humidity_kg_per_kg: Array,
) -> Array:
    r"""Temperature after adiabatic mixing of fresh air and recirculated exhaust.

    Enthalpy per kg dry air :math:`h = c_{p,L} T + Y (\Delta h_0 + c_{p,D} T)` is linear in the
    mixing ratio, and :math:`Y` mixes linearly, so the latent parts cancel and

    .. math:: T_{mix} = T_a + \frac{r\,c_o\,(T_o - T_a)}{(1 - r)\,c_a + r\,c_o},
        \qquad c = c_{p,L} + Y c_{p,D}

    (``r = 0`` gives :math:`T_a` exactly). Fog in the mixing chamber is not modelled; the caller
    can check :math:`\varphi(Y_{in}, T_{mix}) \le 1`.
    """
    c_ambient = gas.cp_dry_air_j_per_kg_k + ambient_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k
    c_exhaust = gas.cp_dry_air_j_per_kg_k + exhaust_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k
    r = recirculation_fraction
    return ambient_temperature_celsius + r * c_exhaust * (
        exhaust_temperature_celsius - ambient_temperature_celsius
    ) / ((1.0 - r) * c_ambient + r * c_exhaust)
