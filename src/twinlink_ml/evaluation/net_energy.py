r"""MA3/MA4: net objective -- gross consumption minus recovery credit.

Befund 52 (``docs/04_stand_ap1.md``) has so far optimised exclusively the gross
consumption (``simulation.energy.energy_target``, a pure kW sum of air heater, blower, pump
and atomising air -- see there) -- the caveat already noted there, "one process, not a
network", remains unquantified as long as no recovery credit is offset against it. This
module closes the gap, with two user specifications from MA3 (not measurements --
explicitly labelled as such):

1. **Credit as fuel equivalent, not thermal** (MA3b). The recovered heat ``Q_rec``
   (``heatnetwork``) replaces natural gas in the air heater, which itself only works with
   ``eta_H = HeatNetworkConfig.heater_efficiency`` -- the fuel power saved is
   ``Q_rec / eta_H``, not ``Q_rec`` itself (:func:`fuel_equivalent_credit_kw`). ``eta_H``
   is taken over from ``HeatNetworkConfig``, not set anew (EA1 principle, as everywhere in
   the project).

2. **Two objective variants, neither replaces the other** (MA3a). The gross consumption
   is a kW sum of electrical terms (blower, pump, atomising air) AND the air-heater term
   from ``simulation.energy`` -- **electrically heated in the single process**
   (``EnergyConfig.heater_efficiency=0.98``, no ``H_u``/fuel demand there, unlike the
   GAS-heated ``Lufterhitzer`` in the heat network/``heatnetwork``). Whether more air
   throughput pays off depends entirely on the weighting between this electrical gross
   power and the GAS power saved through recovery:

   - ``"kw_sum"`` (:func:`net_objective_kw`): unweighted kW difference, continuing the
     status quo of ``energy_target``. Directly comparable in kW with the gross case (V1).
   - ``"co2_weighted"`` (:func:`net_objective_co2_kg_per_h`): gross power weighted with
     :data:`ELECTRICITY_CO2_KG_PER_KWH`, credit with :data:`GAS_CO2_KG_PER_KWH` --
     electricity is 1.82 times worse than gas per kWh (overall project description,
     emission factors according to the German Environment Agency). Result in kg CO2e/h,
     NOT in kW -- a separate system of units; regret percentages remain comparable within
     this variant, but not kW for kW with V1/V2.

   The user explicitly requested both variants (not only the CO2-weighted one), because
   otherwise it cannot be separated what follows from the net formulation itself and what
   from the weighting (MA3a answer, verbatim: "Wird nur gewichtet gerechnet, ist hinterher
   nicht trennbar, was aus der Netto-Formulierung folgt und was aus der Gewichtung").

**One-stage chain, no double ``eta_H``.** The chain heat -> fuel -> CO2 passes through
``eta_H`` exactly ONCE in :func:`net_objective_co2_kg_per_h`
(``fuel_equivalent_credit_kw``, there and nowhere else) -- the gross term contains no
``eta_H`` reference (it is electrical, see above), the CO2 weighting itself only
multiplies the finished fuel power value. See
``tests/test_net_energy.py::test_co2_weighting_divides_by_heater_efficiency_exactly_once``.
"""

from __future__ import annotations

__all__ = [
    "ELECTRICITY_CO2_KG_PER_KWH",
    "GAS_CO2_KG_PER_KWH",
    "fuel_equivalent_credit_kw",
    "net_objective_co2_kg_per_h",
    "net_objective_kw",
]

ELECTRICITY_CO2_KG_PER_KWH = 0.366
"""Emissionsfaktor Strom, kg CO2e/kWh -- Gesamtvorhabenbeschreibung (dort auf das
Umweltbundesamt referenziert). Keine Messung, keine projekteigene Herleitung -- eine
Festlegung aus MA3a, in den beiden Konstanten hier benannt statt als Literal in der
Zielfunktion verstreut (MA3b, Punkt 1: der Strommix ist zeitlich veränderlich und wird im
Projekt später ggf. dynamisch -- ein benannter Parameter ist dann anpassbar, ein Literal
nicht)."""

GAS_CO2_KG_PER_KWH = 0.201
"""Emissionsfaktor Erdgas, kg CO2e/kWh -- dieselbe Quelle wie
:data:`ELECTRICITY_CO2_KG_PER_KWH`. Strom ist je kWh ``0.366/0.201 = 1.82``-mal schlechter
gewichtet als Gas -- dieser Faktor allein reicht, um das Vorzeichen des Tauschs (Strom
ausgeben, um Gas zu sparen) zu bestimmen, siehe Moduldocstring."""


def fuel_equivalent_credit_kw(q_rec_kw: float, heater_efficiency: float) -> float:
    r"""Fuel equivalent of the recovery, MA3b: ``Q_rec / eta_H``, not ``Q_rec``.

    ``Q_rec`` replaces natural gas in the air heater of the heat network
    (``heatnetwork.config.HeatNetworkConfig``), which itself only works with ``eta_H`` --
    the same formula structure as its fuel demand
    (``heatnetwork.balance`` module docstring, EA1: :math:`\dot m_{gas} = \dots /
    (H_u\,\eta_H)`, here without the heating-value factor, because the result remains a
    power, not a mass flow quantity).

    :param q_rec_kw: Recoverable power (``heatnetwork``/MA1), any mode.
    :param heater_efficiency: ``HeatNetworkConfig.heater_efficiency`` (``eta_H``), in
        (0, 1] -- validation lies with ``HeatNetworkConfig`` itself, here deliberately no
        second check of the same invariant.
    :returns: Avoided fuel power in kW.
    """
    return q_rec_kw / heater_efficiency


def net_objective_kw(gross_power_kw: float, q_rec_kw: float, heater_efficiency: float) -> float:
    """Net objective, variant ``"kw_sum"`` (MA3a): unweighted kW difference.

    :param gross_power_kw: Gross consumption (``simulation.energy.energy_target`` with
        ``target="power_kw"``), electrical.
    :param q_rec_kw: Recoverable power (``heatnetwork``), any mode.
    :param heater_efficiency: ``eta_H``, see :func:`fuel_equivalent_credit_kw`.
    :returns: Net power in kW -- smaller is better, as for the gross consumption.
    """
    return gross_power_kw - fuel_equivalent_credit_kw(q_rec_kw, heater_efficiency)


def net_objective_co2_kg_per_h(
    gross_power_kw: float,
    q_rec_kw: float,
    heater_efficiency: float,
    *,
    electricity_co2_kg_per_kwh: float = ELECTRICITY_CO2_KG_PER_KWH,
    gas_co2_kg_per_kwh: float = GAS_CO2_KG_PER_KWH,
) -> float:
    """Net objective, variant ``"co2_weighted"`` (MA3a): CO2-weighted difference.

    Gross power (electrical) weighted with ``electricity_co2_kg_per_kwh``, fuel equivalent
    of the credit (natural gas) with ``gas_co2_kg_per_kwh`` -- the chain
    heat -> fuel -> CO2 passes through ``heater_efficiency`` exactly ONCE, see the
    module docstring.

    :returns: Net emission rate in kg CO2e/h -- a DIFFERENT unit than
        :func:`net_objective_kw`, not comparable kW for kW with the gross case.
    """
    credit_kw = fuel_equivalent_credit_kw(q_rec_kw, heater_efficiency)
    return gross_power_kw * electricity_co2_kg_per_kwh - credit_kw * gas_co2_kg_per_kwh
