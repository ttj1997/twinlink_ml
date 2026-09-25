r"""Sorptionsisotherme der normierten Trocknungskurve.

Hintergrund
-----------
``thermo.normalised_moisture`` nutzte bisher eine Gleichgewichtsfeuchte
:math:`X_{eq}` als Konstante (``MaterialConfig.equilibrium_moisture_kg_per_kg``).
Physikalisch richtig ist :math:`X_{eq}(\varphi_{Abluft}, T_{bed})` aus der
Sorptionsisotherme des Stoffsystems -- die des tatsächlichen Anwendungspartner-Materials ist
offen (``02_prozess_setup.md``, Prioritätenliste Platz 3; ``04_stand_ap1.md``, Befund
34). Dieses Modul repariert die Struktur (Konstante -> Funktion von :math:`\varphi`
und :math:`T`), mit einer Platzhalter-Parametrisierung, die ausdrücklich als
``ConfigStatus == "placeholder"`` markiert bleibt.

Zwei Implementierungen, deshalb eine ABC
-----------------------------------------
Anders als beim exakten Beharrungszustands-Löser (siehe ``03_code_konventionen.md``,
Abschnitt zur bewussten Abweichung bei ``steady_state.py``) liegen hier von Anfang an
**zwei** konkrete Fälle vor: :class:`ConstantIsotherm` (Rückwärtskompatibilität) und
:class:`GabIsotherm` (parametrisierte Isotherme). Das ist dieselbe Schwelle, die
``excitation.DisturbanceDesign`` bereits über zwei Implementierungen zieht -- eine ABC
ist hier keine Vorratsabstraktion, sondern folgt der bestehenden Konvention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .backend import Array, ArrayBackend, get_backend
from .config import (
    ActionSpaceConfig,
    DisturbanceSpaceConfig,
    GasConfig,
    MaterialConfig,
    ProcessConfig,
)
from .thermo import absolute_humidity_kg_per_kg, relative_humidity_from_absolute_humidity

_KELVIN_OFFSET = 273.15
_GAS_CONSTANT_J_PER_MOL_K = 8.314462618

_SAFETY_GRID_PHI_MAX = 0.99
_SAFETY_GRID_TEMPERATURE_MARGIN_CELSIUS = 10.0
_SAFETY_GRID_PHI_POINTS = 200
_SAFETY_GRID_TEMPERATURE_POINTS = 100


class SorptionIsotherm(ABC):
    r"""Gleichgewichtsfeuchte :math:`X_{eq}(\varphi, T)` der Trocknungskurve."""

    @abstractmethod
    def equilibrium_moisture(
        self, backend: ArrayBackend, relative_humidity: Array, temperature_celsius: Array
    ) -> Array:
        """:math:`X_{eq}` in kg Wasser pro kg Trockenmasse.

        :param relative_humidity: relative Feuchte der Abluft in [0, 1]
        :param temperature_celsius: Betttemperatur in °C
        """
        ...


@dataclass(frozen=True, slots=True)
class ConstantIsotherm(SorptionIsotherm):
    r"""Bildet das bisherige Verhalten exakt nach: :math:`X_{eq}` unabhängig von :math:`\varphi, T`.

    Default für Configs, die nichts anderes angeben -- kein bestehender Test ändert
    dadurch sein Ergebnis.
    """

    equilibrium_moisture_kg_per_kg: float

    def equilibrium_moisture(
        self,
        backend: ArrayBackend,
        relative_humidity: Array,  # noqa: ARG002 - Teil des gemeinsamen Isothermen-Interfaces
        temperature_celsius: Array,
    ) -> Array:
        return backend.zeros_like(temperature_celsius) + self.equilibrium_moisture_kg_per_kg


@dataclass(frozen=True, slots=True)
class GabIsotherm(SorptionIsotherm):
    r"""GAB-Isotherme (Guggenheim-Anderson-de Boer) mit Arrhenius-Temperaturabhängigkeit.

    Standardform der Feststoffverfahrenstechnik für granulare Feststoffe:

    .. math:: X_{eq}(\varphi, T) = \frac{X_m\, C(T)\, K(T)\, \varphi}
        {\bigl(1 - K(T)\varphi\bigr)\,\bigl(1 - K(T)\varphi + C(T)K(T)\varphi\bigr)}

    .. math:: C(T) = C_{ref}\,\exp\!\left[\frac{\Delta H_C}{R}
        \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right], \qquad
        K(T) = K_{ref}\,\exp\!\left[\frac{\Delta H_K}{R}
        \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right]

    :math:`T, T_{ref}` in Kelvin. Mit :math:`\Delta H_C, \Delta H_K > 0` fallen
    :math:`C` und :math:`K` mit steigendem :math:`T` -- Desorption: :math:`X_{eq}` fällt
    mit steigender Temperatur bei fester relativer Feuchte.

    **Status: placeholder, unabhängig von ``provenance.status`` der umgebenden Config.**
    Die Parameter sind an einem einzigen Ankerpunkt (:math:`\varphi=0{,}5`,
    :math:`T=62\,°C \to X_{eq}=0{,}030\,\mathrm{kg/kg}`, dem Laborwert aus
    ``greybox_lab_moisture.yaml``) verankert, nicht an der tatsächlichen
    Sorptionsisotherme des Anwendungspartner-Materials -- die ist offen (``04_stand_ap1.md``,
    Befund 34). Ein Lauf mit ``GabIsotherm`` darf sich im Log nie als kalibriert
    ausweisen; deshalb :attr:`status` fest auf ``"placeholder"``, unabhängig davon, was
    der Rest der Config sagt.

    :math:`K_{ref} \in (0, 1)` ist eine harte Voraussetzung (siehe ``__post_init__``):
    die GAB-Gleichung hat bei :math:`K\varphi \to 1` eine Polstelle. Ein
    :math:`K_{ref}` nahe 1 kann bei kalten Randbedingungen (K steigt mit fallendem T)
    und hoher relativer Feuchte in den numerisch erreichbaren Bereich laufen --
    :func:`sorption_safety_grid` deckt genau diesen Rand ab.
    """

    monolayer_moisture_kg_per_kg: float
    c_constant_at_reference_temperature: float
    k_constant_at_reference_temperature: float
    c_enthalpy_j_per_mol: float
    k_enthalpy_j_per_mol: float
    reference_temperature_celsius: float = 62.0

    def __post_init__(self) -> None:
        if self.monolayer_moisture_kg_per_kg <= 0.0:
            raise ValueError(
                f"monolayer_moisture_kg_per_kg muss positiv sein, ist "
                f"{self.monolayer_moisture_kg_per_kg}"
            )
        if self.c_constant_at_reference_temperature <= 0.0:
            raise ValueError(
                f"c_constant_at_reference_temperature muss positiv sein, ist "
                f"{self.c_constant_at_reference_temperature}"
            )
        if not 0.0 < self.k_constant_at_reference_temperature < 1.0:
            raise ValueError(
                "k_constant_at_reference_temperature muss in (0, 1) liegen "
                f"(Polstelle der GAB-Gleichung bei K*phi -> 1), ist "
                f"{self.k_constant_at_reference_temperature}"
            )

    @property
    def status(self) -> str:
        """Immer ``"placeholder"`` -- siehe Klassendocstring."""
        return "placeholder"

    def equilibrium_moisture(
        self, backend: ArrayBackend, relative_humidity: Array, temperature_celsius: Array
    ) -> Array:
        t_kelvin = temperature_celsius + _KELVIN_OFFSET
        t_reference_kelvin = self.reference_temperature_celsius + _KELVIN_OFFSET
        inverse_temperature_difference = 1.0 / t_kelvin - 1.0 / t_reference_kelvin
        c = self.c_constant_at_reference_temperature * backend.exp(
            (self.c_enthalpy_j_per_mol / _GAS_CONSTANT_J_PER_MOL_K) * inverse_temperature_difference
        )
        k = self.k_constant_at_reference_temperature * backend.exp(
            (self.k_enthalpy_j_per_mol / _GAS_CONSTANT_J_PER_MOL_K) * inverse_temperature_difference
        )
        k_phi = k * relative_humidity
        denominator = (1.0 - k_phi) * (1.0 - k_phi + c * k_phi)
        return self.monolayer_moisture_kg_per_kg * c * k_phi / denominator


def isotherm_from_material(material: MaterialConfig) -> SorptionIsotherm:
    """Baut die konfigurierte Isotherme aus den flachen ``MaterialConfig``-Feldern.

    Polymorphie lebt ausschließlich im Code (hier), nicht in der YAML-Serialisierung --
    ``config.py``s ``_from_mapping``/``_NESTED_TYPES`` kennt nur eindeutige
    Feld->Dataclass-Zuordnungen, kein Tagged-Union-Schema. Deshalb ``constant``/``gab``
    als flache, optionale Felder auf ``MaterialConfig`` (wie ``EnergyConfig.target`` +
    ``reference_value``), nicht als eigenes verschachteltes Isothermen-Objekt in der YAML.
    """
    if material.sorption_isotherm_kind == "constant":
        return ConstantIsotherm(material.equilibrium_moisture_kg_per_kg)
    if material.sorption_isotherm_kind == "gab":
        monolayer = material.gab_monolayer_moisture_kg_per_kg
        c_ref = material.gab_c_constant_at_reference_temperature
        k_ref = material.gab_k_constant_at_reference_temperature
        c_enthalpy = material.gab_c_enthalpy_j_per_mol
        k_enthalpy = material.gab_k_enthalpy_j_per_mol
        t_ref = material.gab_reference_temperature_celsius
        assert monolayer is not None and c_ref is not None and k_ref is not None
        assert c_enthalpy is not None and k_enthalpy is not None and t_ref is not None
        return GabIsotherm(
            monolayer_moisture_kg_per_kg=monolayer,
            c_constant_at_reference_temperature=c_ref,
            k_constant_at_reference_temperature=k_ref,
            c_enthalpy_j_per_mol=c_enthalpy,
            k_enthalpy_j_per_mol=k_enthalpy,
            reference_temperature_celsius=t_ref,
        )
    raise ValueError(
        f"Unbekannter sorption_isotherm_kind {material.sorption_isotherm_kind!r}"
    )  # pragma: no cover


def sorption_safety_grid(
    backend: ArrayBackend,
    gas: GasConfig,
    actions: ActionSpaceConfig,
    disturbances: DisturbanceSpaceConfig,
    *,
    temperature_margin_celsius: float = _SAFETY_GRID_TEMPERATURE_MARGIN_CELSIUS,
) -> tuple[float, float, float, float]:
    r"""Grenzen ``(phi_min, phi_max, t_min, t_max)`` des Sicherheitsrasters.

    Zwei getrennte, nicht zu verwechselnde Raster in diesem Projekt: dies ist das
    **Sicherheitsraster** (bestimmt die konservative obere Schranke für die
    ``critical_moisture_kg_per_kg``-Validierung, siehe
    :func:`validate_isotherm_against_critical_moisture`, und diente ursprünglich auch
    als Referenzwert für ``greybox._floor``, bevor die k1-basierte lokale Variante das
    überflüssig machte). Das **Plausibilitätsraster** aus der GAB-Parametrisierungs-
    Kontrolltabelle (phi=0.2/0.5/0.8, T=40/62/90 °C) ist unabhängig davon und dient nur
    dem Gegenlesen der gewählten Parameter -- keine Sicherheitsschranke.

    ``t_min``/``t_max``: an der tatsächlichen Aktions-/Störgrößenhüllkurve verankert,
    mit Marge -- **nicht** an den Buck-Clip-Grenzen (-40..200 °C): das ist eine
    numerische Stabilitätsgrenze der Sättigungsdruckformel, keine physikalisch
    erreichbare Grenze von :math:`T_{bed}` (``04_stand_ap1.md``, Befund 34).

    ``phi_min``: envelope-transformiert, **nicht** :math:`\varphi \to 0`. Die
    trockenste im Modell erreichbare Zuluft (``ambient_temperature_celsius.minimum``,
    ``ambient_relative_humidity.minimum``), erwärmt auf ``t_max`` -- die tatsächlich
    erreichbare Untergrenze von :math:`\varphi_{Abluft}`, nicht die mathematische
    Grenze der Isotherme selbst. Eine relative Feuchte ist keine transportable Größe
    wie eine Temperatur: dieselbe absolute Feuchte ergibt bei höherer Temperatur eine
    dramatisch niedrigere relative Feuchte (Sättigungsdampfdruck steigt stark mit T) --
    ``ambient_relative_humidity.minimum`` direkt als ``phi_min`` zu verwenden würde
    diesen Effekt ignorieren und wäre nicht konservativ genug (nachgerechnet an einem
    realen, bereits dokumentierten Betriebspunkt, ``04_stand_ap1.md`` Befund 34).
    """
    t_min = disturbances.ambient_temperature_celsius.minimum - temperature_margin_celsius
    t_max = actions.inlet_temperature_celsius.maximum + temperature_margin_celsius
    y_in_min = absolute_humidity_kg_per_kg(
        backend,
        gas,
        backend.asarray(disturbances.ambient_temperature_celsius.minimum),
        backend.asarray(disturbances.ambient_relative_humidity.minimum),
    )
    phi_min_array = relative_humidity_from_absolute_humidity(
        backend, gas, y_in_min, backend.asarray(t_max)
    )
    phi_min = float(backend.to_numpy(phi_min_array))
    return phi_min, _SAFETY_GRID_PHI_MAX, t_min, t_max


def validate_isotherm_against_critical_moisture(config: ProcessConfig) -> None:
    """Stellt sicher, dass ``critical_moisture_kg_per_kg`` über dem Isothermen-Maximum liegt.

    Geprüft wird gegen das Maximum der aktiven Isotherme im Sicherheitsraster. Nur bei
    ``sorption_isotherm_kind == "gab"`` -- ``ConstantIsotherm`` kann dagegen per
    Konstruktion nicht verstoßen (``MaterialConfig.__post_init__`` erzwingt bereits
    ``critical_moisture_kg_per_kg > equilibrium_moisture_kg_per_kg`` für den konstanten
    Fall). Lebt hier statt in ``MaterialConfig.__post_init__``, aus zwei Gründen:
    (1) die Prüfung braucht Aktions-/Störgrößengrenzen aus dem gesamten
    ``ProcessConfig``, nicht nur ``MaterialConfig`` -- passt strukturell nicht in ein
    lokales Dataclass-``__post_init__``; (2) ``config.py`` importiert bewusst nichts aus
    ``sorption.py``/``thermo.py`` (Abhängigkeitsrichtung: Simulationsmodule hängen von
    ``config.py`` ab, nicht umgekehrt -- ``thermo.py`` folgt demselben Muster).

    :raises ValueError: wenn das Isothermen-Maximum ``critical_moisture_kg_per_kg``
        erreicht oder überschreitet, mit beiden Zahlen in der Meldung.
    """
    material = config.material
    if material.sorption_isotherm_kind != "gab":
        return
    backend = get_backend("numpy")
    phi_min, phi_max, t_min, t_max = sorption_safety_grid(
        backend, config.gas, config.actions, config.disturbances
    )
    isotherm = isotherm_from_material(material)
    phi_grid, t_grid = np.meshgrid(
        np.linspace(phi_min, phi_max, _SAFETY_GRID_PHI_POINTS),
        np.linspace(t_min, t_max, _SAFETY_GRID_TEMPERATURE_POINTS),
    )
    values = backend.to_numpy(
        isotherm.equilibrium_moisture(backend, backend.asarray(phi_grid), backend.asarray(t_grid))
    )
    isotherm_max = float(np.max(values))
    if isotherm_max >= material.critical_moisture_kg_per_kg:
        raise ValueError(
            f"critical_moisture_kg_per_kg ({material.critical_moisture_kg_per_kg}) liegt "
            f"nicht über dem Maximum der GAB-Isotherme im Sicherheitsraster "
            f"({isotherm_max:.5f} kg/kg, "
            f"phi in [{phi_min:.4f}, {phi_max}], T in [{t_min:.1f}, {t_max:.1f}] °C) -- "
            "normalised_moisture würde eine negative Spanne (X_cr - X_eq) sehen."
        )
