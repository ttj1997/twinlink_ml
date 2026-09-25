r"""Abluft-Austrittsgrößen aus (vorhergesagter) Betttemperatur und Partikelfeuchte.

Physikalische Herkunft: Befund 31 (``docs/04_stand_ap1.md``, "Abluft als
Schnittstellengröße") berechnet Massenstrom, Feuchtebeladung und Temperatur der Abluft
bereits einmal -- dort als **einmalige diagnostische Rechnung** direkt aus dem
Beharrungszustands-Löser (``steady_state.py``), mit dem exakten, simulierten Zustand als
Eingang. Diese Funktion zieht dieselbe Formel auf eine andere Eintrittsseite: Betttemperatur
und Partikelfeuchte als **Argumente** statt als Nebenprodukt einer vollen
ODE-Integration/Bisektion -- geeignet für Werte, die ein Sequenzmodell vorhergesagt hat
(``bed_temperature_celsius``, ``product_moisture_kg_per_kg``), nicht nur für Simulator-
internen Zustand.

Zwei algebraische Tatsachen aus dem Modell, hier genutzt, nicht neu hergeleitet:

1. **Gut durchmischtes Bett**: :math:`T_{gas,out} = T_{bed}` exakt (siehe
   ``thermo.evaporation_rate_kg_per_s``-Docstring: "Annahme: ideal durchmischtes Bett,
   Abluft auf Betttemperatur") -- dieselbe Gleichsetzung, die auch
   ``steady_state.SteadyStateSolution.bed_temperature_celsius`` für
   ``exhaust_sensible_kw`` verwendet, hier nur explizit benannt statt implizit in einer
   lokalen Variablen.
2. **Massenbilanz der Abluftfeuchte**: :math:`Y_{gas,out} = Y_{in} +
   \dot m_{evap}/\dot m_{L,tr}`, mit :math:`\dot m_{evap}` aus derselben
   gedämpften Picard-Iteration wie im Rollout selbst (``thermo.evaporation_rate_kg_per_s``,
   Befund 34).

**Keine neue Physik.** Diese Funktion ruft ``evaporation_rate_kg_per_s`` unverändert auf --
sie berechnet nichts, was der Simulator nicht ohnehin für jeden Zeitschritt berechnet.
Gegen ``solve_steady_state()`` am Nominalpunkt aus Befund 31 geprüft: bitgleiche
Übereinstimmung (``tests/test_exhaust.py``).

**Was NICHT abgedeckt ist.** ``dry_air_mass_flow_kg_per_s`` und
``inlet_absolute_humidity_kg_per_kg`` sind hier Argumente, keine Rückgabewerte -- sie
hängen von Stellgrößen (Gasgeschwindigkeit) und Störgrößen (Umgebungstemperatur/-feuchte)
ab, die einem Sequenzmodell-Aufrufer ohnehin bekannt sind (Teil der Aktion/Störung, nicht
der Vorhersage), nicht von Größen, die erst vorhergesagt werden müssten. Anders als
``bed_temperature_celsius``: die ist zum Zeitpunkt dieses Moduls (``docs/04_stand_ap1.md``,
Befund 26-40) **keine** gelernte Zielgröße der Sequenzmodelle -- diese Funktion setzt sie
nicht voraus, macht sie aber erst nützlich, sobald sie es wird.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backend import Array, ArrayBackend
from .config import ExhaustConfig, GasConfig, MaterialConfig
from .thermo import evaporation_rate_kg_per_s

if TYPE_CHECKING:
    # Dieselbe Dependency-Injection-Begründung wie in thermo.py: die Isotherme wird
    # injiziert statt intern aus ``material`` konstruiert, kein Laufzeit-Import nötig.
    from .sorption import SorptionIsotherm

__all__ = ["reconstruct_exhaust_state"]


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
    r"""Abluft-Temperatur und -Feuchte aus Betttemperatur und Partikelfeuchte.

    Vertragskorrektur (vormals ``reconstruct_exhaust_state`` gab unter dem Namen
    "Abluft-/Exhaust-Zustand" tatsächlich ``bed_temperature_celsius`` zurück -- unter der
    Annahme eines ideal durchmischten Betts numerisch korrekt, aber ein Vertragsfehler an
    der Verbundschnittstelle: ein Partner, der "Ablufttemperatur" verdrahtet, bekäme die
    Betttemperatur). Diese Funktion liefert jetzt zusätzlich ``outlet_temperature_celsius``
    als eigenes, explizit benanntes Feld -- die um den Rohrleitungs-/Filterverlust
    (``exhaust.outlet_duct_heat_loss_k``, Platzhalter, Default 0.0) reduzierte
    Bett-Austrittstemperatur. ``T_gas_out_celsius`` (erster Rückgabewert, unverändert) bleibt
    die Gastemperatur unmittelbar am Bett-Austritt, vor jedem Leitungsverlust -- per
    Modellannahme identisch zu ``bed_temperature_celsius`` (siehe Moduldocstring Punkt 1).

    :param backend: Array-Backend (NumPy für Datengenerierung, Torch für differenzierbare
        Rollouts) -- wie überall in ``simulation``, per Dependency Injection.
    :param gas: Gaskonfiguration des Prozesses (spezifische Wärmekapazitäten, Druck).
    :param material: Materialkonfiguration (u. a. für die Sättigungsgrad-Obergrenze in
        ``evaporation_rate_kg_per_s``).
    :param isotherm: Sorptionsisotherme, siehe ``sorption.isotherm_from_material``.
    :param exhaust: Rohrleitungs-/Filterverlust zwischen Bett und Wärmeverbund
        (``ExhaustConfig``, Platzhalter, siehe ``config.py``).
    :param bed_temperature_celsius: Betttemperatur -- Simulator-Zustand oder
        Sequenzmodell-Vorhersage, diese Funktion unterscheidet nicht.
    :param particle_moisture_kg_per_kg: Partikelfeuchte im Bett (``X_p``) -- entspricht
        ``product_moisture_kg_per_kg`` in ``output_names``/``target_names``.
    :param dry_air_mass_flow_kg_per_s: Trockenluftmassenstrom -- aus Stellgröße
        (Gasgeschwindigkeit) und Störgröße/Betttemperatur ableitbar, siehe
        ``greybox._rhs``; hier als bekannte Eingabe erwartet, nicht Teil der Vorhersage.
    :param inlet_absolute_humidity_kg_per_kg: Absolute Feuchte der Zuluft (``Y_in``) --
        aus Umgebungsbedingungen (Störgröße), ebenfalls bekannte Eingabe.
    :returns: ``(T_gas_out_celsius, Y_gas_out_kg_per_kg, outlet_temperature_celsius)``, alle
        in denselben Einheiten wie ``bed_temperature_celsius``/``particle_moisture_kg_per_kg``.
    """
    _evaporation, outlet_humidity, _saturation_degree = evaporation_rate_kg_per_s(
        backend,
        gas,
        material,
        isotherm,
        dry_air_mass_flow_kg_per_s,
        bed_temperature_celsius,
        inlet_absolute_humidity_kg_per_kg,
        particle_moisture_kg_per_kg,
    )
    # Gut durchmischtes Bett (Modellannahme, siehe Moduldocstring Punkt 1): keine eigene
    # Rechnung noetig, T_gas_out ist per Konstruktion die uebergebene Betttemperatur.
    outlet_temperature_celsius = bed_temperature_celsius - exhaust.outlet_duct_heat_loss_k
    return bed_temperature_celsius, outlet_humidity, outlet_temperature_celsius
