r"""Fluidisierungsfenster: Minimalfluidisierungs- und Austragsgeschwindigkeit.

Warum das im Prozessmodell steht und nicht in einem Notebook
------------------------------------------------------------
Das Fluidisierungsfenster :math:`u_{mf}(d_{50}, T) < u_{gas} < u_t(d_{50}, T)` ist
**zustandsabhängig**: es verschiebt sich, während die Partikel wachsen und die
Betttemperatur sich ändert. Für den späteren RL-Teil heißt das, dass die
Safety-Constraints keine konstanten Box-Grenzen sind, sondern eine Funktion des
Zustands. Wer das erst beim RL-Design merkt, hat den Action-Space falsch definiert.

Für ein Bett mit :math:`d_{50}` von 400 auf 750 µm wächst :math:`u_{mf}` um mehr als
den Faktor 3. Ein fester unterer Grenzwert für ``u_gas``, der beim Start gültig ist,
führt am Ende der Charge zum Zusammenbruch der Fluidisierung.
"""

from __future__ import annotations

from .backend import Array, ArrayBackend
from .config import GasConfig, MaterialConfig
from .thermo import gas_density_kg_per_m3, gas_viscosity_pa_s

GRAVITY_M_PER_S2 = 9.80665


def archimedes_number(
    backend: ArrayBackend,  # noqa: ARG001 - gemeinsame Signatur aller Korrelationen
    material: MaterialConfig,
    diameter_m: Array,
    gas_density: Array,
    gas_viscosity: Array,
) -> Array:
    r"""Archimedes-Zahl.

    .. math:: Ar = \frac{\rho_g\, d^3 (\rho_p - \rho_g)\, g}{\mu^2}

    ``backend`` bleibt in der Signatur, damit alle Korrelationen dieselbe haben und im
    Aufrufer austauschbar sind -- hier wird es nicht gebraucht.
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
    r"""Minimalfluidisierungsgeschwindigkeit nach Wen & Yu.

    .. math::
        Re_{mf} = \sqrt{33.7^2 + 0.0408\,Ar} - 33.7,
        \qquad u_{mf} = \frac{Re_{mf}\,\mu}{\rho_g\, d}

    Die Korrelation setzt eine Bettporosität um 0.4 und kugelige Partikel voraus.
    Für Granulate mit rauer Oberfläche liegt der reale Wert typisch 10-30 % höher --
    einer der Punkte, die gegen Laborversuche in AP 2 zu kalibrieren sind.
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
    r"""Sinkgeschwindigkeit nach Haider & Levenspiel (1989), formfaktorabhängig.

    .. math::
        d^{*} = d\left[\frac{g\,\rho_g(\rho_p-\rho_g)}{\mu^2}\right]^{1/3},
        \qquad
        u^{*}_t = \left[\frac{18}{(d^{*})^2}
        + \frac{2.335 - 1.744\,\phi}{\sqrt{d^{*}}}\right]^{-1}

    mit :math:`u_t = u^{*}_t \left[\dfrac{g\,\mu\,(\rho_p-\rho_g)}{\rho_g^2}\right]^{1/3}`,
    gültig für :math:`0.5 < \phi \le 1`.
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
    r"""Druckverlust der fluidisierten Schicht.

    .. math:: \Delta p_{bed} = \frac{m_{bed}\, g}{A}

    Bemerkenswert und für die Zielgröße relevant: oberhalb von :math:`u_{mf}` ist der
    Bettdruckverlust **unabhängig** von der Gasgeschwindigkeit -- er hängt nur an der
    Bettmasse. Die Abhängigkeit der Ventilatorleistung von ``u_gas`` kommt daher aus dem
    Volumenstrom und dem Anströmboden/Filter, nicht aus dem Bett. Wer das falsch
    modelliert, bekommt eine qualitativ falsche Sensitivität
    :math:`\partial \dot E / \partial u_{gas}`.
    """
    return bed_mass_kg * GRAVITY_M_PER_S2 / cross_section_area_m2
