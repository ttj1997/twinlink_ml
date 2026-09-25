"""Tests für ``evaluation/net_energy.py`` -- MA3-Netto-Zielfunktion."""

from __future__ import annotations

import pytest

from twinlink_ml.evaluation.net_energy import (
    ELECTRICITY_CO2_KG_PER_KWH,
    GAS_CO2_KG_PER_KWH,
    fuel_equivalent_credit_kw,
    net_objective_co2_kg_per_h,
    net_objective_kw,
)


def test_fuel_equivalent_credit_divides_by_heater_efficiency() -> None:
    assert fuel_equivalent_credit_kw(90.0, 0.9) == pytest.approx(100.0)


def test_fuel_equivalent_credit_grows_as_efficiency_drops() -> None:
    """Ein schlechterer Erhitzer braucht MEHR Brennstoff für dieselbe Nutzwärme."""
    better = fuel_equivalent_credit_kw(100.0, 0.9)
    worse = fuel_equivalent_credit_kw(100.0, 0.5)
    assert worse > better


def test_net_objective_kw_is_gross_minus_fuel_equivalent_credit() -> None:
    gross = 300.0
    q_rec = 90.0
    eta_h = 0.9
    net = net_objective_kw(gross, q_rec, eta_h)
    assert net == pytest.approx(gross - 100.0)


def test_net_objective_kw_with_zero_recovery_equals_gross() -> None:
    assert net_objective_kw(300.0, 0.0, 0.9) == pytest.approx(300.0)


def test_net_objective_co2_weighting_matches_documented_factors() -> None:
    assert ELECTRICITY_CO2_KG_PER_KWH == 0.366
    assert GAS_CO2_KG_PER_KWH == 0.201


def test_net_objective_co2_electricity_is_weighted_worse_than_gas() -> None:
    """Kernaussage aus MA3a: Strom ist je kWh ~1.82x schlechter gewichtet als Gas."""
    assert pytest.approx(1.8209, abs=1e-3) == ELECTRICITY_CO2_KG_PER_KWH / GAS_CO2_KG_PER_KWH


def test_co2_weighting_divides_by_heater_efficiency_exactly_once() -> None:
    """Die Kette Wärme -> Brennstoff -> CO2 darf ``eta_H`` nicht doppelt anwenden.

    Konstruktion: die CO2-gewichtete Netto-Funktion mit einem EXPLIZIT über
    ``fuel_equivalent_credit_kw`` vorberechneten Kredit muss dieselbe Gutschrift liefern
    wie die interne Berechnung -- würde ``net_objective_co2_kg_per_h`` intern ein zweites
    Mal durch ``eta_H`` teilen, wichen beide Rechnungen voneinander ab.
    """
    gross = 300.0
    q_rec = 90.0
    eta_h = 0.9

    credit_kw = fuel_equivalent_credit_kw(q_rec, eta_h)
    expected = gross * ELECTRICITY_CO2_KG_PER_KWH - credit_kw * GAS_CO2_KG_PER_KWH

    assert net_objective_co2_kg_per_h(gross, q_rec, eta_h) == pytest.approx(expected, rel=1e-12)


def test_co2_weighting_accepts_explicit_factors() -> None:
    result = net_objective_co2_kg_per_h(
        100.0, 0.0, 0.9, electricity_co2_kg_per_kwh=1.0, gas_co2_kg_per_kwh=1.0
    )
    assert result == pytest.approx(100.0)


def test_co2_weighting_can_flip_sign_of_a_recovery_dominated_swap() -> None:
    """MA3-Motivation.

    Ob mehr Luftdurchsatz (mehr Strom) sich lohnt, weil er auch mehr Rückgewinnung
    (weniger Gas) bringt, hängt an der Gewichtung -- bei kW-Summe kann ein Tausch
    neutral/positiv sein, den die CO2-Gewichtung als Verschlechterung einstuft.
    """
    # Ein Delta, das kW-summiert neutral ist: +5 kW brutto, kompensiert durch +4.5 kW
    # Gutschrift-Basis (4.5 = 5 * eta_h, damit die Brennstoffaequivalent-Gutschrift genau
    # um 5 kW steigt -- exakt so viel wie die Bruttoleistung).
    gross_before, q_rec_before, eta_h = 300.0, 90.0, 0.9
    gross_after, q_rec_after = 305.0, 94.5
    net_before_kw = net_objective_kw(gross_before, q_rec_before, eta_h)
    net_after_kw = net_objective_kw(gross_after, q_rec_after, eta_h)
    assert net_after_kw == pytest.approx(net_before_kw)

    # ... ist unter CO2-Gewichtung eine Verschlechterung, weil die zusaetzlichen 5 kW
    # Strom mit 0.366 staerker wiegen als die durch sie ermoeglichte Gasersparnis mit 0.201.
    net_before_co2 = net_objective_co2_kg_per_h(gross_before, q_rec_before, eta_h)
    net_after_co2 = net_objective_co2_kg_per_h(gross_after, q_rec_after, eta_h)
    assert net_after_co2 > net_before_co2
