r"""Sorption isotherm of the normalised drying curve.

Background
----------
``thermo.normalised_moisture`` previously used an equilibrium moisture
:math:`X_{eq}` as a constant (``MaterialConfig.equilibrium_moisture_kg_per_kg``).
Physically correct is :math:`X_{eq}(\varphi_{Abluft}, T_{bed})` from the
sorption isotherm of the material system -- that of the actual application-partner material is
open (``02_prozess_setup.md``, priority list rank 3; ``04_stand_ap1.md``, Befund
34). This module repairs the structure (constant -> function of :math:`\varphi`
and :math:`T`), with a placeholder parametrisation that explicitly stays marked as
``ConfigStatus == "placeholder"``.

Two implementations, hence an ABC
---------------------------------
Unlike the exact steady-state solver (see ``03_code_konventionen.md``,
section on the deliberate deviation for ``steady_state.py``), here there are from the start
**two** concrete cases: :class:`ConstantIsotherm` (backward compatibility) and
:class:`GabIsotherm` (parametrised isotherm). That is the same threshold that
``excitation.DisturbanceDesign`` already draws across two implementations -- an ABC
is not a speculative abstraction here but follows the existing convention.
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
    r"""Equilibrium moisture :math:`X_{eq}(\varphi, T)` of the drying curve."""

    @abstractmethod
    def equilibrium_moisture(
        self, backend: ArrayBackend, relative_humidity: Array, temperature_celsius: Array
    ) -> Array:
        """:math:`X_{eq}` in kg water per kg dry mass.

        :param relative_humidity: relative humidity of the exhaust air in [0, 1]
        :param temperature_celsius: bed temperature in °C
        """
        ...


@dataclass(frozen=True, slots=True)
class ConstantIsotherm(SorptionIsotherm):
    r"""Reproduces the previous behaviour exactly: :math:`X_{eq}` independent of :math:`\varphi, T`.

    Default for configs that specify nothing else -- no existing test changes
    its result because of it.
    """

    equilibrium_moisture_kg_per_kg: float

    def equilibrium_moisture(
        self,
        backend: ArrayBackend,
        relative_humidity: Array,  # noqa: ARG002 - part of the common isotherm interface
        temperature_celsius: Array,
    ) -> Array:
        return backend.zeros_like(temperature_celsius) + self.equilibrium_moisture_kg_per_kg


@dataclass(frozen=True, slots=True)
class GabIsotherm(SorptionIsotherm):
    r"""GAB isotherm (Guggenheim-Anderson-de Boer) with Arrhenius temperature dependence.

    Standard form of particle process engineering for granular solids:

    .. math:: X_{eq}(\varphi, T) = \frac{X_m\, C(T)\, K(T)\, \varphi}
        {\bigl(1 - K(T)\varphi\bigr)\,\bigl(1 - K(T)\varphi + C(T)K(T)\varphi\bigr)}

    .. math:: C(T) = C_{ref}\,\exp\!\left[\frac{\Delta H_C}{R}
        \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right], \qquad
        K(T) = K_{ref}\,\exp\!\left[\frac{\Delta H_K}{R}
        \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right]

    :math:`T, T_{ref}` in Kelvin. With :math:`\Delta H_C, \Delta H_K > 0`,
    :math:`C` and :math:`K` fall with rising :math:`T` -- desorption: :math:`X_{eq}` falls
    with rising temperature at fixed relative humidity.

    **Status: placeholder, independent of ``provenance.status`` of the enclosing config.**
    The parameters are anchored at a single anchor point (:math:`\varphi=0{,}5`,
    :math:`T=62\,°C \to X_{eq}=0{,}030\,\mathrm{kg/kg}`, the lab value from
    ``greybox_lab_moisture.yaml``), not at the actual
    sorption isotherm of the application-partner material -- that is open (``04_stand_ap1.md``,
    Befund 34). A run with ``GabIsotherm`` must never present itself as calibrated in the log;
    hence :attr:`status` fixed to ``"placeholder"``, regardless of what
    the rest of the config says.

    :math:`K_{ref} \in (0, 1)` is a hard requirement (see ``__post_init__``):
    the GAB equation has a pole at :math:`K\varphi \to 1`. A
    :math:`K_{ref}` close to 1 can run into the numerically reachable range under cold
    boundary conditions (K rises with falling T) and high relative humidity --
    :func:`sorption_safety_grid` covers exactly this edge.
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
        """Always ``"placeholder"`` -- see class docstring."""
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
    """Builds the configured isotherm from the flat ``MaterialConfig`` fields.

    Polymorphism lives exclusively in the code (here), not in the YAML serialisation --
    ``config.py``'s ``_from_mapping``/``_NESTED_TYPES`` knows only unique
    field->dataclass mappings, no tagged-union schema. Hence ``constant``/``gab``
    as flat, optional fields on ``MaterialConfig`` (like ``EnergyConfig.target`` +
    ``reference_value``), not as a separate nested isotherm object in the YAML.
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
    r"""Bounds ``(phi_min, phi_max, t_min, t_max)`` of the safety grid.

    Two separate grids in this project that must not be confused: this is the
    **safety grid** (determines the conservative upper bound for the
    ``critical_moisture_kg_per_kg`` validation, see
    :func:`validate_isotherm_against_critical_moisture`, and originally also served
    as reference value for ``greybox._floor``, before the k1-based local variant made that
    superfluous). The **plausibility grid** from the GAB parametrisation
    check table (phi=0.2/0.5/0.8, T=40/62/90 °C) is independent of it and serves only
    to cross-read the chosen parameters -- not a safety bound.

    ``t_min``/``t_max``: anchored at the actual action/disturbance envelope,
    with margin -- **not** at the Buck clip limits (-40..200 °C): that is a
    numerical stability limit of the saturation pressure formula, not a physically
    reachable limit of :math:`T_{bed}` (``04_stand_ap1.md``, Befund 34).

    ``phi_min``: envelope-transformed, **not** :math:`\varphi \to 0`. The
    driest inlet air reachable in the model (``ambient_temperature_celsius.minimum``,
    ``ambient_relative_humidity.minimum``), heated to ``t_max`` -- the actually
    reachable lower bound of :math:`\varphi_{Abluft}`, not the mathematical
    limit of the isotherm itself. A relative humidity is not a transportable quantity
    like a temperature: the same absolute humidity gives at a higher temperature a
    dramatically lower relative humidity (saturation vapour pressure rises strongly with T) --
    using ``ambient_relative_humidity.minimum`` directly as ``phi_min`` would
    ignore this effect and would not be conservative enough (recalculated at a
    real, already documented operating point, ``04_stand_ap1.md`` Befund 34).
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
    """Ensures that ``critical_moisture_kg_per_kg`` lies above the isotherm maximum.

    Checked against the maximum of the active isotherm on the safety grid. Only with
    ``sorption_isotherm_kind == "gab"`` -- ``ConstantIsotherm``, by contrast, cannot violate it
    by construction (``MaterialConfig.__post_init__`` already enforces
    ``critical_moisture_kg_per_kg > equilibrium_moisture_kg_per_kg`` for the constant
    case). Lives here instead of in ``MaterialConfig.__post_init__``, for two reasons:
    (1) the check needs action/disturbance bounds from the entire
    ``ProcessConfig``, not only ``MaterialConfig`` -- does not fit structurally into a
    local dataclass ``__post_init__``; (2) ``config.py`` deliberately imports nothing from
    ``sorption.py``/``thermo.py`` (dependency direction: simulation modules depend on
    ``config.py``, not the other way round -- ``thermo.py`` follows the same pattern).

    :raises ValueError: if the isotherm maximum reaches or exceeds
        ``critical_moisture_kg_per_kg``, with both numbers in the message.
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
