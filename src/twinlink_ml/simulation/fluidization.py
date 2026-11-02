r"""Fluidization window: minimum fluidization velocity and terminal velocity.

Why this is in the process model and not in a notebook
------------------------------------------------------
The fluidization window :math:`u_{mf}(d_{50}, T) < u_{gas} < u_t(d_{50}, T)` is
**state-dependent**: it shifts while the particles grow and the
bed temperature changes. For the later RL part this means that the
safety constraints are not constant box limits but a function of the
state. Whoever notices this only at the RL design stage has defined the action space wrongly.

For a bed with :math:`d_{50}` growing from 400 to 750 µm, :math:`u_{mf}` grows by more than
a factor of 3. A fixed lower limit for ``u_gas`` that is valid at the start
leads to the collapse of the fluidization at the end of the batch.
"""

from __future__ import annotations

from .backend import Array, ArrayBackend
from .config import GasConfig, MaterialConfig
from .thermo import gas_density_kg_per_m3, gas_viscosity_pa_s

GRAVITY_M_PER_S2 = 9.80665


def archimedes_number(
    backend: ArrayBackend,  # noqa: ARG001 - common signature of all correlations
    material: MaterialConfig,
    diameter_m: Array,
    gas_density: Array,
    gas_viscosity: Array,
) -> Array:
    r"""Archimedes number.

    .. math:: Ar = \frac{\rho_g\, d^3 (\rho_p - \rho_g)\, g}{\mu^2}

    ``backend`` stays in the signature so that all correlations have the same one and are
    interchangeable in the caller -- it is not needed here.
    """
    delta_rho = material.particle_density_kg_per_m3 - gas_density
    return gas_density * diameter_m**3 * delta_rho * GRAVITY_M_PER_S2 / gas_viscosity**2


def minimum_fluidization_velocity_m_per_s(
    backend: ArrayBackend,
    gas: GasConfig,
    material: MaterialConfig,
    diameter_m: Array,
    temperature_celsius: Array,
    humidity_kg_per_kg: Array,
) -> Array:
    r"""Minimum fluidization velocity according to Wen & Yu.

    .. math::
        Re_{mf} = \sqrt{33.7^2 + 0.0408\,Ar} - 33.7,
        \qquad u_{mf} = \frac{Re_{mf}\,\mu}{\rho_g\, d}

    The correlation assumes a bed porosity of about 0.4 and spherical particles.
    For granules with a rough surface the real value is typically 10-30 % higher --
    one of the points to be calibrated against lab experiments in AP 2.
    """
    rho_g = gas_density_kg_per_m3(backend, gas, temperature_celsius, humidity_kg_per_kg)
    mu = gas_viscosity_pa_s(backend, gas, temperature_celsius)
    ar = archimedes_number(backend, material, diameter_m, rho_g, mu)
    reynolds = backend.sqrt(33.7**2 + 0.0408 * ar) - 33.7
    return reynolds * mu / (rho_g * diameter_m)


def terminal_velocity_m_per_s(
    backend: ArrayBackend,
    gas: GasConfig,
    material: MaterialConfig,
    diameter_m: Array,
    temperature_celsius: Array,
    humidity_kg_per_kg: Array,
) -> Array:
    r"""Terminal velocity according to Haider & Levenspiel (1989), shape-factor dependent.

    .. math::
        d^{*} = d\left[\frac{g\,\rho_g(\rho_p-\rho_g)}{\mu^2}\right]^{1/3},
        \qquad
        u^{*}_t = \left[\frac{18}{(d^{*})^2}
        + \frac{2.335 - 1.744\,\phi}{\sqrt{d^{*}}}\right]^{-1}

    with :math:`u_t = u^{*}_t \left[\dfrac{g\,\mu\,(\rho_p-\rho_g)}{\rho_g^2}\right]^{1/3}`,
    valid for :math:`0.5 < \phi \le 1`.
    """
    rho_g = gas_density_kg_per_m3(backend, gas, temperature_celsius, humidity_kg_per_kg)
    mu = gas_viscosity_pa_s(backend, gas, temperature_celsius)
    delta_rho = material.particle_density_kg_per_m3 - rho_g
    scale = (GRAVITY_M_PER_S2 * rho_g * delta_rho / mu**2) ** (1.0 / 3.0)
    d_star = diameter_m * scale
    u_star = 1.0 / (18.0 / d_star**2 + (2.335 - 1.744 * material.sphericity) / backend.sqrt(d_star))
    velocity_scale = (GRAVITY_M_PER_S2 * mu * delta_rho / rho_g**2) ** (1.0 / 3.0)
    return u_star * velocity_scale


def bed_pressure_drop_pa(bed_mass_kg: Array, cross_section_area_m2: float) -> Array:
    r"""Pressure drop of the fluidized layer.

    .. math:: \Delta p_{bed} = \frac{m_{bed}\, g}{A}

    Remarkable and relevant for the target: above :math:`u_{mf}` the
    bed pressure drop is **independent** of the gas velocity -- it depends only on the
    bed mass. The dependence of the fan power on ``u_gas`` therefore comes from the
    volume flow and the distributor plate/filter, not from the bed. Whoever models this wrongly
    gets a qualitatively wrong sensitivity
    :math:`\partial \dot E / \partial u_{gas}`.
    """
    return bed_mass_kg * GRAVITY_M_PER_S2 / cross_section_area_m2
