"""Tests for ``evaluation/net_energy.py`` -- MA3 net objective function."""

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
    """A worse heater needs MORE fuel for the same useful heat."""
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
    """Core statement from MA3a: per kWh, electricity is weighted ~1.82x worse than gas."""
    assert pytest.approx(1.8209, abs=1e-3) == ELECTRICITY_CO2_KG_PER_KWH / GAS_CO2_KG_PER_KWH


def test_co2_weighting_divides_by_heater_efficiency_exactly_once() -> None:
    """The chain heat -> fuel -> CO2 must not apply ``eta_H`` twice.

    Construction: the CO2-weighted net function with a credit precomputed EXPLICITLY via
    ``fuel_equivalent_credit_kw`` must yield the same credit
    as the internal calculation -- if ``net_objective_co2_kg_per_h`` internally divided a second
    time by ``eta_H``, the two calculations would differ from each other.
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
    """MA3 motivation.

    Whether more air throughput (more electricity) pays off because it also brings more recovery
    (less gas) depends on the weighting -- with a kW sum a trade can be
    neutral/positive that the CO2 weighting classifies as a deterioration.
    """
    # A delta that is neutral when summed in kW: +5 kW gross, compensated by +4.5 kW
    # credit basis (4.5 = 5 * eta_h, so that the fuel-equivalent credit rises by exactly
    # 5 kW -- exactly as much as the gross power).
    gross_before, q_rec_before, eta_h = 300.0, 90.0, 0.9
    gross_after, q_rec_after = 305.0, 94.5
    net_before_kw = net_objective_kw(gross_before, q_rec_before, eta_h)
    net_after_kw = net_objective_kw(gross_after, q_rec_after, eta_h)
    assert net_after_kw == pytest.approx(net_before_kw)

    # ... is a deterioration under CO2 weighting, because the additional 5 kW of
    # electricity at 0.366 weigh more heavily than the gas saving they enable at 0.201.
    net_before_co2 = net_objective_co2_kg_per_h(gross_before, q_rec_before, eta_h)
    net_after_co2 = net_objective_co2_kg_per_h(gross_after, q_rec_after, eta_h)
    assert net_after_co2 > net_before_co2
