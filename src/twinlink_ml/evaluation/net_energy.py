r"""MA3/MA4: Netto-Zielfunktion -- Bruttoverbrauch minus Rückgewinnungsgutschrift.

Befund 52 (``docs/04_stand_ap1.md``) optimiert bislang ausschließlich den
Bruttoverbrauch (``simulation.energy.energy_target``, eine reine kW-Summe aus Lufterhitzer,
Gebläse, Pumpe und Zerstäuberluft -- siehe dort) -- der bereits dort vermerkte Vorbehalt
"Ein Prozess, kein Verbund" bleibt unquantifiziert, solange keine Rückgewinnungsgutschrift
gegengerechnet wird. Dieses Modul schließt die Lücke, mit zwei User-Festlegungen aus MA3
(nicht Messungen -- explizit als solche gekennzeichnet):

1. **Gutschrift als Brennstoffäquivalent, nicht thermisch** (MA3b). Die zurückgewonnene
   Wärme ``Q_rec`` (``heatnetwork``) ersetzt Erdgas im Lufterhitzer, der selbst nur mit
   ``eta_H = HeatNetworkConfig.heater_efficiency`` arbeitet -- die eingesparte
   Brennstoffleistung ist ``Q_rec / eta_H``, nicht ``Q_rec`` selbst
   (:func:`fuel_equivalent_credit_kw`). ``eta_H`` wird aus ``HeatNetworkConfig``
   übernommen, nicht neu gesetzt (EA1-Prinzip, wie überall im Projekt).

2. **Zwei Zielfunktionsvarianten, keine ersetzt die andere** (MA3a). Der Bruttoverbrauch
   ist eine kW-Summe aus elektrischen (Gebläse, Pumpe, Zerstäuberluft) UND dem
   Lufterhitzer-Term aus ``simulation.energy`` -- **am Einzelprozess elektrisch beheizt**
   (``EnergyConfig.heater_efficiency=0.98``, kein ``H_u``/Brennstoffbedarf dort, anders als
   der GAS-beheizte ``Lufterhitzer`` im Wärmeverbund/``heatnetwork``). Ob sich mehr
   Luftdurchsatz lohnt, hängt vollständig an der Gewichtung zwischen dieser elektrischen
   Bruttoleistung und der durch Rückgewinnung eingesparten GAS-Leistung:

   - ``"kw_sum"`` (:func:`net_objective_kw`): ungewichtete kW-Differenz, Status quo von
     ``energy_target`` fortgeführt. Direkt in kW vergleichbar mit dem Bruttofall (V1).
   - ``"co2_weighted"`` (:func:`net_objective_co2_kg_per_h`): Bruttoleistung mit
     :data:`ELECTRICITY_CO2_KG_PER_KWH`, Gutschrift mit :data:`GAS_CO2_KG_PER_KWH`
     gewichtet -- Strom ist je kWh 1,82-mal schlechter als Gas (Gesamtvorhabenbeschreibung,
     Emissionsfaktoren nach Umweltbundesamt). Ergebnis in kg CO2e/h, NICHT in kW -- ein
     eigenes Maßeinheitensystem, Regret-Prozentsätze bleiben innerhalb dieser Variante
     vergleichbar, aber nicht kW-für-kW mit V1/V2.

   Der Nutzer hat beide Varianten explizit angefordert (nicht nur die CO2-gewichtete), weil
   sonst nicht trennbar ist, was aus der Netto-Formulierung selbst folgt und was aus der
   Gewichtung (MA3a-Antwort, wörtlich: "Wird nur gewichtet gerechnet, ist hinterher nicht
   trennbar, was aus der Netto-Formulierung folgt und was aus der Gewichtung").

**Eine-Stufe-Kette, kein doppeltes ``eta_H``.** Die Kette Wärme -> Brennstoff -> CO2 geht
in :func:`net_objective_co2_kg_per_h` genau EINMAL durch ``eta_H``
(``fuel_equivalent_credit_kw``, dort und nirgends sonst) -- der Bruttoterm enthält keinen
``eta_H``-Bezug (er ist elektrisch, siehe oben), die CO2-Gewichtung selbst multipliziert nur
noch mit dem fertigen Brennstoffleistungswert. Siehe
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
    r"""Brennstoffäquivalent der Rückgewinnung, MA3b: ``Q_rec / eta_H``, nicht ``Q_rec``.

    ``Q_rec`` ersetzt Erdgas im Lufterhitzer des Wärmeverbunds
    (``heatnetwork.config.HeatNetworkConfig``), der selbst nur mit ``eta_H`` arbeitet --
    dieselbe Formelstruktur wie dessen Brennstoffbedarf
    (``heatnetwork.balance``-Moduldocstring, EA1: :math:`\dot m_{gas} = \dots /
    (H_u\,\eta_H)`, hier ohne den Heizwert-Faktor, weil das Ergebnis eine Leistung bleibt,
    keine Massenstromgröße).

    :param q_rec_kw: Rückgewinnbare Leistung (``heatnetwork``/MA1), beliebiger Modus.
    :param heater_efficiency: ``HeatNetworkConfig.heater_efficiency`` (``eta_H``), in
        (0, 1] -- Validierung liegt bei ``HeatNetworkConfig`` selbst, hier bewusst keine
        zweite Prüfung derselben Invariante.
    :returns: Vermiedene Brennstoffleistung in kW.
    """
    return q_rec_kw / heater_efficiency


def net_objective_kw(gross_power_kw: float, q_rec_kw: float, heater_efficiency: float) -> float:
    """Netto-Zielfunktion, Variante ``"kw_sum"`` (MA3a): ungewichtete kW-Differenz.

    :param gross_power_kw: Bruttoverbrauch (``simulation.energy.energy_target`` bei
        ``target="power_kw"``), elektrisch.
    :param q_rec_kw: Rückgewinnbare Leistung (``heatnetwork``), beliebiger Modus.
    :param heater_efficiency: ``eta_H``, siehe :func:`fuel_equivalent_credit_kw`.
    :returns: Netto-Leistung in kW -- kleiner ist besser, wie beim Bruttoverbrauch.
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
    """Netto-Zielfunktion, Variante ``"co2_weighted"`` (MA3a): CO2-gewichtete Differenz.

    Bruttoleistung (elektrisch) mit ``electricity_co2_kg_per_kwh``, Brennstoffäquivalent
    der Gutschrift (Erdgas) mit ``gas_co2_kg_per_kwh`` gewichtet -- die Kette
    Wärme -> Brennstoff -> CO2 geht dabei genau EINMAL durch ``heater_efficiency``, siehe
    Moduldocstring.

    :returns: Netto-Emissionsrate in kg CO2e/h -- ANDERE Einheit als
        :func:`net_objective_kw`, nicht kW-für-kW vergleichbar mit dem Bruttofall.
    """
    credit_kw = fuel_equivalent_credit_kw(q_rec_kw, heater_efficiency)
    return gross_power_kw * electricity_co2_kg_per_kwh - credit_kw * gas_co2_kg_per_kwh
